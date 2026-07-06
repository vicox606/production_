import os
import uuid
import io
from typing import List, Dict
 
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
 
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
import ollama

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
CHUNK_SIZE = 1200          # characters per chunk
CHUNK_OVERLAP = 200        # overlap between chunks to avoid cutting context
TOP_K_CHUNKS = 5           # how many chunks to feed the model per question
 
ollama_client = ollama.Client(host=OLLAMA_HOST)
 
app = FastAPI(title="PDF Chatbot API")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: Dict[str, Dict] = {}

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise HTTPException(status_code=400, detail="PDF is password protected.")
 
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
 
    text = "\n".join(pages_text).strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in PDF (it may be a scanned/image-only PDF).",
        )
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
    top_indices = similarities.argsort()[::-1][:top_k]
    # Preserve original document order for more coherent context
    top_indices = sorted(top_indices)
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
 
 
def get_session_or_404(session_id: str) -> Dict:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Upload a PDF first.")
    return session
 
class ChatRequest(BaseModel):
    session_id: str
    question: str
 
 
class ChatResponse(BaseModel):
    answer: str
    session_id: str
 
 
class UploadResponse(BaseModel):
    session_id: str
    filename: str
    num_chunks: int
    num_characters: int
