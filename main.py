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
