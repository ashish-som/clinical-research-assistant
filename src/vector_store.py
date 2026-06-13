"""Vector database utilities.
"""

from pathlib import Path
import shutil
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import EMBEDDING_MODEL


def create_embedding_model() -> OpenAIEmbeddings:
    """Create the embedding model used to convert text into vectors."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def build_vector_store(chunks: List[Document], persist_dir: Path) -> Chroma:
    """Create or replace a Chroma vector database from document chunks."""
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma.from_documents(
        documents=chunks,
        embedding=create_embedding_model(),
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_vector_store(persist_dir: Path) -> Chroma:
    """Load an existing Chroma vector database from disk."""
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=create_embedding_model(),
    )


def vector_store_exists(persist_dir: Path) -> bool:
    """Check whether a persisted Chroma database already exists."""
    return (persist_dir / "chroma.sqlite3").exists()
