"""Document chunking utilities.
"""

import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:
    """Split page Documents into smaller chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)
    chunks = add_section_labels(chunks)
    return add_chunk_ids(chunks)


def detect_section(text: str) -> str:
    """Guess the research paper section from headings inside a chunk."""
    first_lines = "\n".join(text.strip().splitlines()[:8]).lower()

    section_patterns = {
        "abstract": r"\babstract\b",
        "introduction": r"\bintroduction\b|\bbackground\b",
        "methods": r"\bmethods?\b|\bmaterials and methods\b|\bstudy design\b",
        "results": r"\bresults?\b|\bfindings\b",
        "discussion": r"\bdiscussion\b",
        "limitations": r"\blimitations?\b",
        "conclusion": r"\bconclusions?\b",
        "references": r"\breferences\b|\bbibliography\b",
    }

    for section, pattern in section_patterns.items():
        if re.search(pattern, first_lines):
            return section

    return "unknown"


def add_section_labels(chunks: List[Document]) -> List[Document]:
    """Attach a likely section label to each chunk for section-aware retrieval."""
    current_section_by_file = {}

    for chunk in chunks:
        source_file = chunk.metadata.get("source_file", "unknown")
        detected_section = detect_section(chunk.page_content)

        if detected_section != "unknown":
            current_section_by_file[source_file] = detected_section

        chunk.metadata["section"] = current_section_by_file.get(source_file, "unknown")

    return chunks


def add_chunk_ids(chunks: List[Document]) -> List[Document]:
    """Attach a unique chunk id to each chunk for tracing and debugging."""
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks
