"""Shared configuration for the clinical research assistant."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = PROJECT_ROOT / "pdf"
UPLOAD_DIR = PROJECT_ROOT / "uploaded_pdfs"
VECTOR_DB_DIR = PROJECT_ROOT / "research_db"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 5
