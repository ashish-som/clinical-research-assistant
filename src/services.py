
"""Application services for the clinical research assistant.
"""

from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI

from src.agents import (
    build_agentic_rag_graph,
    build_paper_summary_graph,
)
from src.chunking import chunk_documents
from src.config import (
    CHAT_MODEL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    VECTOR_DB_DIR,
)
from src.interfaces import ChunkingStrategy, DocumentLoader
from src.pdf_loader import load_pdfs
from src.retrieval import (
    get_all_vectorstore_documents,
    hybrid_retrieve_with_query_variations,
)
from src.vector_store import build_vector_store, load_vector_store, vector_store_exists


@dataclass
class IndexingResult:
    """Summary of what happened during PDF indexing."""

    pages: int
    chunks: int
    vectorstore: object


class PdfDocumentLoader:
    """Loads research PDFs into page-level Documents."""

    def load(self, paths: list[Path]):
        """Load PDFs from disk."""
        return load_pdfs(paths)


class SectionAwareChunker:
    """Chunks pages and labels each chunk with a likely research section."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents):
        """Split page Documents into section-aware chunks."""
        return chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )


class PaperIndexingService:
    """Runs the complete ingestion pipeline from PDFs to vector database."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        chunker: ChunkingStrategy | None = None,
        persist_dir: Path = VECTOR_DB_DIR,
    ):
        self.loader = loader or PdfDocumentLoader()
        self.chunker = chunker or SectionAwareChunker()
        self.persist_dir = persist_dir

    def index(self, pdf_paths: list[Path]) -> IndexingResult:
        """Load PDFs, create chunks, embed chunks, and persist the vector store."""
        pages = self.loader.load(pdf_paths)
        chunks = self.chunker.split(pages)
        vectorstore = build_vector_store(chunks, self.persist_dir)

        return IndexingResult(
            pages=len(pages),
            chunks=len(chunks),
            vectorstore=vectorstore,
        )


class HybridResearchRetriever:
    """Retrieves evidence using query variation, vector search, and keyword search."""

    def __init__(self, vectorstore, top_k: int = DEFAULT_TOP_K, llm=None):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.llm = llm or ChatOpenAI(model=CHAT_MODEL, temperature=0)

    def retrieve(self, query: str):
        """Return relevant chunks for a research query."""
        return hybrid_retrieve_with_query_variations(
            self.vectorstore,
            query,
            self.llm,
            top_k=self.top_k,
        )


class ClinicalResearchAssistant:
    """High-level assistant used by the Streamlit app and future API."""

    def __init__(self, vectorstore, top_k: int = DEFAULT_TOP_K):
        self.vectorstore = vectorstore
        self.top_k = top_k

    def ask(self, question: str) -> dict:
        """Answer a research question with cited evidence."""
        graph = build_agentic_rag_graph(self.vectorstore, top_k=self.top_k)
        return graph.invoke({"query": question})

    def summarize(self, focus: str = "Summarize the indexed research papers.") -> dict:
        """Generate a structured summary across the indexed papers."""
        graph = build_paper_summary_graph(self.vectorstore, top_k=self.top_k)
        return graph.invoke({"query": focus})

    def list_papers(self) -> list[str]:
        """Return unique paper names available in the vector database."""
        docs = get_all_vectorstore_documents(self.vectorstore)
        papers = {doc.metadata.get("source_file", "Unknown") for doc in docs}
        return sorted(papers)


def load_existing_vectorstore(persist_dir: Path = VECTOR_DB_DIR):
    """Load the persisted vector store if it already exists."""
    if vector_store_exists(persist_dir):
        return load_vector_store(persist_dir)

    return None
