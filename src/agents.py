"""Agentic RAG workflow.
Checks whether retrieved evidence is enough before generating.
"""

from typing import Dict, List, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.config import CHAT_MODEL
from src.retrieval import format_context, hybrid_retrieve_with_query_variations


class AgentState(TypedDict, total=False):
    """Temporary memory passed between graph nodes."""

    query: str
    retrieved_docs: list
    refined_query: str
    answer: str
    sources: List[Dict]
    needs_retry: bool
    summary: str


def create_llm(temperature: float = 0) -> ChatOpenAI:
    """Create the chat model used for grading, rewriting, and answering."""
    return ChatOpenAI(model=CHAT_MODEL, temperature=temperature)


def grade_retrieval_quality(query: str, context: str, llm: ChatOpenAI) -> bool:
    """Return True if the retrieved context is enough to answer the query."""
    prompt = f"""
You are checking whether retrieved research context is sufficient.

Question:
{query}

Retrieved context:
{context}

Answer only YES or NO.
Is the context enough to answer the question with citations?
"""
    response = llm.invoke(prompt).content.strip().upper()
    return response.startswith("YES")


def refine_query(query: str, llm: ChatOpenAI) -> str:
    """Rewrite a weak query into a more retrieval-friendly clinical query."""
    prompt = f"""
Rewrite this question as a stronger search query for clinical research papers.
Keep the original meaning and add useful research terms if needed.

Question:
{query}
"""
    return llm.invoke(prompt).content.strip()


def generate_answer(query: str, docs: list, llm: ChatOpenAI) -> str:
    """Generate a grounded answer using only retrieved research context."""
    context = format_context(docs)
    prompt = f"""
You are an AI Clinical Research Assistant.

Answer the question using ONLY the provided research context.

Important rules:
- Cite sources like [Source 1], [Source 2].
- Do not invent facts.
- If the answer is not present, say: "Not found in the provided research papers."
- Use a precise academic style.

Context:
{context}

Question:
{query}
"""
    return llm.invoke(prompt).content


def generate_paper_summary(query: str, docs: list, llm: ChatOpenAI) -> str:
    """Generate a structured research summary from retrieved paper chunks."""
    context = format_context(docs)
    prompt = f"""
You are an AI Clinical Research Assistant.

Create a structured summary using ONLY the provided research context.

Use this format:
1. Research focus
2. Papers/studies covered
3. Methods or population details
4. Main findings
5. Limitations
6. Practical clinical interpretation

Rules:
- Cite sources like [Source 1], [Source 2].
- Do not invent missing details.
- If a field is not available, write "Not found in the provided research papers."

Context:
{context}

Summary focus:
{query}
"""
    return llm.invoke(prompt).content


def extract_sources(docs: list) -> List[Dict]:
    """Create clean source metadata for the final UI and reports."""
    sources = []

    for doc in docs:
        source = {
            "study": doc.metadata.get("author_year", "Unknown"),
            "file": doc.metadata.get("source_file", "Unknown"),
            "page": doc.metadata.get("page_number", "N/A"),
            "chunk_id": doc.metadata.get("chunk_id", "N/A"),
        }

        if source not in sources:
            sources.append(source)

    return sources


def build_agentic_rag_graph(vectorstore, top_k: int = 5):
    """Build a LangGraph workflow with retrieve, grade, retry, and answer nodes."""
    llm = create_llm()

    def retrieve_node(state: AgentState) -> AgentState:
        docs = hybrid_retrieve_with_query_variations(vectorstore, state["query"], llm, top_k=top_k)
        return {"retrieved_docs": docs}

    def grade_node(state: AgentState) -> AgentState:
        context = format_context(state["retrieved_docs"])
        is_enough = grade_retrieval_quality(state["query"], context, llm)

        return {
            "needs_retry": not is_enough,
            "refined_query": state["query"] if is_enough else refine_query(state["query"], llm),
        }

    def retrieve_again_node(state: AgentState) -> AgentState:
        docs = hybrid_retrieve_with_query_variations(
            vectorstore,
            state["refined_query"],
            llm,
            top_k=top_k,
        )
        return {"retrieved_docs": docs}

    def generate_node(state: AgentState) -> AgentState:
        docs = state["retrieved_docs"]
        return {
            "answer": generate_answer(state["query"], docs, llm),
            "sources": extract_sources(docs),
        }

    def decide_next_step(state: AgentState) -> str:
        return "retrieve_again" if state.get("needs_retry") else "generate"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("retrieve_again", retrieve_again_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        decide_next_step,
        {
            "retrieve_again": "retrieve_again",
            "generate": "generate",
        },
    )
    graph.add_edge("retrieve_again", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def build_paper_summary_graph(vectorstore, top_k: int = 8):
    """Build a LangGraph workflow for summarising indexed papers."""
    llm = create_llm()

    def retrieve_for_summary_node(state: AgentState) -> AgentState:
        docs = hybrid_retrieve_with_query_variations(
            vectorstore,
            state["query"],
            llm,
            top_k=top_k,
            apply_section_filter=False,
            diversify_sources=True,
        )
        return {"retrieved_docs": docs}

    def summarize_node(state: AgentState) -> AgentState:
        docs = state["retrieved_docs"]
        return {
            "summary": generate_paper_summary(state["query"], docs, llm),
            "sources": extract_sources(docs),
        }

    graph = StateGraph(AgentState)
    graph.add_node("retrieve_for_summary", retrieve_for_summary_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("retrieve_for_summary")
    graph.add_edge("retrieve_for_summary", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()

