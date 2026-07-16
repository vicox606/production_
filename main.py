
import asyncio
import io
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from enum import Enum
from typing import Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import ollama

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K_CHUNKS = 5
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
THREAD_POOL_SIZE = int(os.environ.get("THREAD_POOL_SIZE", "8"))

ollama_client = ollama.Client(host=OLLAMA_HOST)


executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE, thread_name_prefix="pdf-worker")



class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class Job:

    def __init__(self, job_id: str, kind: str, payload: dict):
        self.id = job_id
        self.kind = kind          # "upload" or "chat"
        self.payload = payload
        self.status = JobStatus.QUEUED
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.worker_id: Optional[str] = None   # which worker picked this job up
        self.event = asyncio.Event()     # set() when job reaches DONE or ERROR
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    def mark_running(self, worker_id: str):
        self.status = JobStatus.RUNNING
        self.worker_id = worker_id
        self.started_at = time.time()

    def mark_done(self, result: dict):
        self.result = result
        self.status = JobStatus.DONE
        self.finished_at = time.time()
        self.event.set()

    def mark_error(self, error: str):
        self.error = error
        self.status = JobStatus.ERROR
        self.finished_at = time.time()
        self.event.set()

    def to_dict(self):
        return {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "worker_id": self.worker_id,
            "result": self.result,
            "error": self.error,
            "queued_for": round((self.started_at or time.time()) - self.created_at, 3),
            "duration": round((self.finished_at - self.started_at), 3)
            if self.started_at and self.finished_at else None,
        }


JOBS: Dict[str, Job] = {}
JOB_QUEUE: "asyncio.Queue[Job]" = asyncio.Queue()
SESSIONS: Dict[str, Dict] = {}  


WORKERS: Dict[str, Dict] = {}


async def worker_loop(worker_id: str):
    loop = asyncio.get_running_loop()
    WORKERS[worker_id] = {"state": "idle", "current_job": None, "jobs_processed": 0}

    while True:
        job = await JOB_QUEUE.get()
        job.mark_running(worker_id)
        WORKERS[worker_id]["state"] = "busy"
        WORKERS[worker_id]["current_job"] = job.id
        try:
            if job.kind == "upload":
                result = await loop.run_in_executor(executor, process_upload_job, job.payload)
            elif job.kind == "chat":
                result = await loop.run_in_executor(executor, process_chat_job, job.payload)
            else:
                raise ValueError(f"Unknown job kind: {job.kind}")
            job.mark_done(result)
        except Exception as exc:
            job.mark_error(str(exc))
        finally:
            WORKERS[worker_id]["state"] = "idle"
            WORKERS[worker_id]["current_job"] = None
            WORKERS[worker_id]["jobs_processed"] += 1
            JOB_QUEUE.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    workers = [
        asyncio.create_task(worker_loop(f"worker-{i}"))
        for i in range(NUM_WORKERS)
    ]
    yield
    for w in workers:
        w.cancel()
    executor.shutdown(wait=False)


app = FastAPI(title="PDF Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("PDF is password protected.")

    pages_text = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages_text).strip()
    if not text:
        raise ValueError("No extractable text found in PDF (it may be scanned/image-only).")
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = end - overlap
    return chunks


def get_relevant_chunks(chunks: List[str], question: str, top_k: int = TOP_K_CHUNKS) -> List[str]:
    if len(chunks) <= top_k:
        return chunks
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks + [question])
    doc_vectors, query_vector = matrix[:-1], matrix[-1]
    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    top_indices = sorted(similarities.argsort()[::-1][:top_k])
    return [chunks[i] for i in top_indices]


def build_prompt(context_chunks: List[str], question: str) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return (
        "You are a helpful assistant answering questions about a PDF document. "
        "Use ONLY the context below to answer. If the answer isn't in the context, "
        "say you don't have enough information from the document.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}"
    )


def process_upload_job(payload: dict) -> dict:
    file_bytes = payload["file_bytes"]
    filename = payload["filename"]

    text = extract_text_from_pdf(file_bytes)
    chunks = chunk_text(text)

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {"filename": filename, "chunks": chunks, "history": []}

    return {
        "session_id": session_id,
        "filename": filename,
        "num_chunks": len(chunks),
        "num_characters": len(text),
    }


def process_chat_job(payload: dict) -> dict:
    session_id = payload["session_id"]
    question = payload["question"]

    session = SESSIONS.get(session_id)
    if not session:
        raise ValueError("Session not found. Upload a PDF first.")

    relevant_chunks = get_relevant_chunks(session["chunks"], question)
    prompt = build_prompt(relevant_chunks, question)

    try:
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response["message"]["content"]
    except Exception as exc:
        raise RuntimeError(
            f"Error contacting Ollama at {OLLAMA_HOST} (model '{OLLAMA_MODEL}'): {exc}. "
            "Make sure 'ollama serve' is running and the model has been pulled."
        )

    session["history"].append({"question": question, "answer": answer})
    return {"answer": answer, "session_id": session_id}


class ChatRequest(BaseModel):
    session_id: str
    question: str


class JobAccepted(BaseModel):
    job_id: str
    status: JobStatus




@app.post("/upload", response_model=JobAccepted, status_code=202)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    file_bytes = await file.read()  # async I/O, doesn't block the loop

    job = Job(job_id=str(uuid.uuid4()), kind="upload",
              payload={"file_bytes": file_bytes, "filename": file.filename})
    JOBS[job.id] = job
    await JOB_QUEUE.put(job)

    return JobAccepted(job_id=job.id, status=job.status)


@app.post("/chat", response_model=JobAccepted, status_code=202)
async def chat(payload: ChatRequest):
    if payload.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found. Upload a PDF first.")

    job = Job(job_id=str(uuid.uuid4()), kind="chat",
              payload={"session_id": payload.session_id, "question": payload.question})
    JOBS[job.id] = job
    await JOB_QUEUE.put(job)

    return JobAccepted(job_id=job.id, status=job.status)


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job.to_dict()


@app.get("/jobs/{job_id}/wait")
async def wait_for_job(job_id: str, timeout: float = 30.0):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    try:
        await asyncio.wait_for(job.event.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Job still running; poll /jobs/{job_id}.")
    return job.to_dict()


@app.get("/events/{job_id}")
async def stream_job_events(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    async def event_generator():
        last_snapshot = None
        while True:
            snapshot = (job.status, job.worker_id)
            if snapshot != last_snapshot:
                last_snapshot = snapshot
                payload = json.dumps({"status": job.status.value, "worker_id": job.worker_id})
                yield f"event: status\ndata: {payload}\n\n"
            if job.event.is_set():
                yield f"event: result\ndata: {json.dumps(job.to_dict())}\n\n"
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/sessions")
async def list_sessions():
    return {
        sid: {"filename": s["filename"], "num_chunks": len(s["chunks"]), "turns": len(s["history"])}
        for sid, s in SESSIONS.items()
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    del SESSIONS[session_id]
    return {"deleted": session_id}


@app.get("/workers")
async def list_workers():
    """Live view of every background worker: idle/busy, current job, jobs processed."""
    return WORKERS


@app.get("/health")
async def health():
    return {
        "queue_size": JOB_QUEUE.qsize(),
        "active_jobs": sum(1 for j in JOBS.values() if j.status == JobStatus.RUNNING),
        "num_workers": NUM_WORKERS,
        "thread_pool_size": THREAD_POOL_SIZE,
        "workers": WORKERS,
    }


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")