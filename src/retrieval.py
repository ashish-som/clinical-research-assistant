"""Retrieval techniques for the research assistant.
"""

import re
from collections import Counter
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.config import DEFAULT_TOP_K


def format_context(docs: List[Document]) -> str:
    """Format retrieved chunks into a citation-friendly prompt context."""
    context_blocks = []

    for index, doc in enumerate(docs, start=1):
        context_blocks.append(
            f"""[Source {index}]
Study: {doc.metadata.get("author_year", "Unknown")}
File: {doc.metadata.get("source_file", "Unknown")}
Page: {doc.metadata.get("page_number", "N/A")}
Chunk ID: {doc.metadata.get("chunk_id", "N/A")}

Content:
{doc.page_content}
"""
        )

    return "\n------------------------\n".join(context_blocks)


def create_query_variations(query: str, llm: ChatOpenAI, count: int = 3) -> List[str]:
    """Generate alternate phrasings of the user's question."""
    prompt = f"""
Create {count} search queries for retrieving evidence from clinical research papers.

Original question:
{query}

Rules:
- Keep the meaning same.
- Use clinical/research terminology where useful.
- Return one query per line.
"""
    response = llm.invoke(prompt).content
    variations = [line.strip("- ").strip() for line in response.splitlines()]
    variations = [line for line in variations if line]

    return [query] + variations[:count]


def infer_section_filter(query: str) -> str | None:
    """Infer which paper section is most useful for a question."""
    query_lower = query.lower()

    section_keywords = {
        "methods": ["method", "sample size", "participants", "cohort", "design", "criteria"],
        "results": ["result", "finding", "outcome", "mortality", "rate", "effect"],
        "limitations": ["limitation", "bias", "weakness"],
        "discussion": ["discussion", "interpretation", "implication"],
        "conclusion": ["conclusion", "conclude", "summary"],
        "abstract": ["abstract", "overview"],
    }

    for section, keywords in section_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return section

    return None


def tokenize(text: str) -> List[str]:
    """Convert text into lowercase keyword tokens for lexical matching."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def get_all_vectorstore_documents(vectorstore) -> List[Document]:
    """Load stored chunks from Chroma so keyword retrieval can search them."""
    raw = vectorstore.get(include=["documents", "metadatas"])
    documents = raw.get("documents", [])
    metadatas = raw.get("metadatas", [])

    return [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(documents, metadatas)
    ]


def keyword_search(docs: List[Document], query: str, top_k: int = DEFAULT_TOP_K) -> List[Document]:
    """Retrieve chunks by exact keyword overlap with the user query."""
    query_terms = Counter(tokenize(query))
    scored_docs = []

    for doc in docs:
        doc_terms = Counter(tokenize(doc.page_content))
        score = sum(min(count, doc_terms.get(term, 0)) for term, count in query_terms.items())

        if score > 0:
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]


def filter_by_section(docs: List[Document], section: str | None) -> List[Document]:
    """Prefer chunks from the inferred paper section when available."""
    if section is None:
        return docs

    filtered_docs = [doc for doc in docs if doc.metadata.get("section") == section]
    return filtered_docs or docs


def detect_requested_sources(query: str, docs: List[Document]) -> set[str]:
    """Detect whether the query names a specific paper or study."""
    query_lower = query.lower()
    requested_sources = set()

    for doc in docs:
        source_file = doc.metadata.get("source_file", "")
        author_year = doc.metadata.get("author_year", "")
        source_stem = source_file.lower().replace(".pdf", "")

        candidates = [
            source_file.lower(),
            source_stem,
            author_year.lower(),
        ]

        if any(candidate and candidate in query_lower for candidate in candidates):
            requested_sources.add(source_file)

    return requested_sources



def filter_by_requested_sources(docs: List[Document], requested_sources: set[str]) -> List[Document]:
    """Keep only chunks from explicitly requested papers."""
    if not requested_sources:
        return docs

    return [doc for doc in docs if doc.metadata.get("source_file") in requested_sources]


def deduplicate_docs(docs: List[Document]) -> List[Document]:
    """Remove repeated chunks returned by multiple retrieval methods."""
    seen = set()
    unique_docs = []

    for doc in docs:
        key = (
            doc.metadata.get("source_file"),
            doc.metadata.get("page_number"),
            doc.metadata.get("chunk_id"),
        )

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs


def diversify_by_source(docs: List[Document], max_per_source: int = 3) -> List[Document]:
    """Keep evidence from multiple papers instead of one dominant paper."""
    source_counts = Counter()
    diverse_docs = []

    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown")

        if source_counts[source] >= max_per_source:
            continue

        source_counts[source] += 1
        diverse_docs.append(doc)

    return diverse_docs


def retrieve_with_query_variations(
    retriever,
    query: str,
    llm: ChatOpenAI,
    variation_count: int = 3,
) -> List[Document]:
    """Retrieve documents using the original query plus generated variations."""
    seen = set()
    results: List[Document] = []

    for search_query in create_query_variations(query, llm, variation_count):
        docs = retriever.invoke(search_query)

        for doc in docs:
            key = (
                doc.metadata.get("source_file"),
                doc.metadata.get("page_number"),
                doc.metadata.get("chunk_id"),
            )

            if key not in seen:
                seen.add(key)
                results.append(doc)

    return results


def hybrid_retrieve_with_query_variations(
    vectorstore,
    query: str,
    llm: ChatOpenAI,
    top_k: int = DEFAULT_TOP_K,
    variation_count: int = 3,
    apply_section_filter: bool = True,
    diversify_sources: bool = False,
) -> List[Document]:
    """Combine vector search, keyword search, query variation, and section hints."""
    retriever = create_retriever(vectorstore, top_k=top_k)
    all_chunks = get_all_vectorstore_documents(vectorstore)
    section_filter = infer_section_filter(query)
    requested_sources = detect_requested_sources(query, all_chunks)
    retrieved_docs = []

    for search_query in create_query_variations(query, llm, variation_count):
        vector_docs = retriever.invoke(search_query)
        keyword_docs = keyword_search(all_chunks, search_query, top_k=top_k)
        retrieved_docs.extend(vector_docs + keyword_docs)

    unique_docs = deduplicate_docs(retrieved_docs)
    unique_docs = filter_by_requested_sources(unique_docs, requested_sources)
    final_docs = filter_by_section(unique_docs, section_filter) if apply_section_filter else unique_docs

    if diversify_sources and not requested_sources:
        final_docs = diversify_by_source(final_docs)

    return final_docs[: top_k * 2]


def create_retriever(vectorstore, top_k: int = DEFAULT_TOP_K):
    """Create a semantic retriever from the vector database."""
    return vectorstore.as_retriever(search_kwargs={"k": top_k})
