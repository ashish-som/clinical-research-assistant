"""Command-line app for teaching the Clinical Research Assistant.
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K, PDF_DIR
from src.pdf_loader import list_pdf_files
from src.services import (
    ClinicalResearchAssistant,
    PaperIndexingService,
    SectionAwareChunker,
    load_existing_vectorstore,
)


def print_sources(sources: list[dict]) -> None:
    """Print citation metadata in a readable format."""
    print("\nSources:")

    if not sources:
        print("- No sources returned.")
        return

    for source in sources:
        print(
            f"- {source['study']} | {source['file']} | "
            f"Page {source['page']} | Chunk {source['chunk_id']}"
        )


def get_assistant(top_k: int) -> ClinicalResearchAssistant:
    """Load the indexed vector database and create the assistant service."""
    vectorstore = load_existing_vectorstore()

    if vectorstore is None:
        raise RuntimeError("No vector database found. Run: python cli_app.py index")

    return ClinicalResearchAssistant(vectorstore, top_k=top_k)


def index_command(args) -> None:
    """Index PDFs into ChromaDB."""
    pdf_dir = Path(args.pdf_dir)
    pdf_paths = list_pdf_files(pdf_dir)

    if not pdf_paths:
        raise RuntimeError(f"No PDFs found in: {pdf_dir}")

    service = PaperIndexingService(
        chunker=SectionAwareChunker(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    )
    result = service.index(pdf_paths)

    print("Index created successfully.")
    print(f"PDF folder: {pdf_dir}")
    print(f"Pages indexed: {result.pages}")
    print(f"Chunks created: {result.chunks}")


def ask_command(args) -> None:
    """Ask a question over the indexed papers."""
    assistant = get_assistant(args.top_k)
    response = assistant.ask(args.question)

    print("\nAnswer:\n")
    print(response["answer"])
    print_sources(response.get("sources", []))


def summarize_command(args) -> None:
    """Generate a structured summary of the indexed papers."""
    assistant = get_assistant(args.top_k)
    response = assistant.summarize(args.focus)

    print("\nSummary:\n")
    print(response["summary"])
    print_sources(response.get("sources", []))


def list_papers_command(args) -> None:
    """Show which papers are available in the vector database."""
    assistant = get_assistant(args.top_k)
    papers = assistant.list_papers()

    print("\nIndexed papers:")
    for paper in papers:
        print(f"- {paper}")


def build_parser() -> argparse.ArgumentParser:
    """Create CLI commands for indexing, Q&A, and summarization."""
    parser = argparse.ArgumentParser(
        description="Clinical Research Assistant CLI",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve per search query.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build/rebuild the vector index.")
    index_parser.add_argument(
        "--pdf-dir",
        default=str(PDF_DIR),
        help="Folder containing research PDFs.",
    )
    index_parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum characters per chunk.",
    )
    index_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Characters shared between neighbouring chunks.",
    )
    index_parser.set_defaults(func=index_command)

    ask_parser = subparsers.add_parser("ask", help="Ask a cited research question.")
    ask_parser.add_argument("question", help="Question to ask over indexed papers.")
    ask_parser.set_defaults(func=ask_command)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize indexed papers.",
    )
    summarize_parser.add_argument(
        "--focus",
        default="Summarize the clinical objective, population, methods, findings, and limitations.",
        help="Summary focus.",
    )
    summarize_parser.set_defaults(func=summarize_command)

    list_parser = subparsers.add_parser("list-papers", help="List indexed papers.")
    list_parser.set_defaults(func=list_papers_command)

    return parser


def main() -> None:
    """Run the command-line app."""
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
