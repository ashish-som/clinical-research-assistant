"""Interfaces for the architecture.
"""

from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document


class DocumentLoader(Protocol):
    """A component that converts files into LangChain Documents."""

    def load(self, paths: list[Path]) -> list[Document]:
        """Load source files and return page-level Documents."""
        ...


class ChunkingStrategy(Protocol):
    """A component that splits Documents into searchable chunks."""

    def split(self, documents: list[Document]) -> list[Document]:
        """Split Documents and attach retrieval metadata."""
        ...


class RetrievalStrategy(Protocol):
    """A component that retrieves evidence for a user query."""

    def retrieve(self, query: str) -> list[Document]:
        """Return relevant chunks for a query."""
        ...
