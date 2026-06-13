# Clinical Research Assistant

An Agentic RAG (Retrieval-Augmented Generation) system for clinical research papers built using LangGraph, OpenAI, ChromaDB, and Streamlit.

The assistant can ingest clinical research PDFs, build a searchable knowledge base, answer research questions with citations, and generate structured summaries grounded in retrieved evidence.

---

## Key Features

* Agentic RAG workflow using LangGraph
* Hybrid retrieval (semantic + keyword search)
* Query expansion for improved recall
* Section-aware retrieval
* Paper-specific retrieval
* Source-diverse summarization
* Evidence quality grading
* Citation-aware answer generation
* Streamlit UI and CLI support
* Persistent ChromaDB vector storage

---

## Tech Stack

* Python
* LangChain
* LangGraph
* OpenAI
* ChromaDB
* Streamlit
* PyMuPDF
* Hybrid Retrieval
* Agentic RAG

---

## Capabilities

### Document Processing

* Ingest clinical research PDFs
* Extract page-level text and metadata
* Create section-aware chunks
* Store embeddings in ChromaDB

### Retrieval

* Semantic vector search
* Keyword search
* Hybrid retrieval
* Query variation generation
* Section-aware filtering
* Paper-specific retrieval
* Source-diverse retrieval

### Generation

* Evidence quality checking
* Citation-aware question answering
* Research paper summarization

### Interfaces

* Streamlit web application
* Command-line interface (CLI)

---

## Architecture

```text
User
  |
  v
Streamlit UI / CLI
  |
  v
ClinicalResearchAssistant Service
  |
  v
LangGraph Workflow
  |
  +---- Hybrid Retrieval
  |
  +---- ChromaDB
  |
  +---- OpenAI
```

---

## Agent Workflow

### Question Answering

```text
retrieve
    |
    v
grade_evidence
    |
    +---- sufficient ----------> generate
    |
    +---- insufficient --------> retrieve_again
                                       |
                                       v
                                   generate
```

### Summarization

```text
retrieve_for_summary
          |
          v
      summarize
          |
          v
         END
```

---

## Project Flow

```text
PDFs
  |
  v
Extract page text
  |
  v
Convert pages to Documents
  |
  v
Split Documents into chunks
  |
  v
Add metadata and section labels
  |
  v
Generate embeddings
  |
  v
Store in ChromaDB
  |
  v
User Question / Summary Request
  |
  v
Generate Query Variations
  |
  v
Hybrid Retrieval
  |
  v
Evidence Quality Check
  |
  +---- Weak Context ----> Retrieve Again
  |
  +---- Strong Context --> Generate Answer
```

---

## Folder Structure

```text
clinical_assistant/
│
├── app.py
├── cli_app.py
│
├── src/
│   ├── agents.py
│   ├── chunking.py
│   ├── config.py
│   ├── interfaces.py
│   ├── pdf_loader.py
│   ├── retrieval.py
│   ├── services.py
│   └── vector_store.py
│
├── pdf/
├── uploaded_pdfs/
├── research_db/
│
├── requirements.txt
├── requirements-core.txt
├── .env.example
└── README.md
```

---

## Main Components

### app.py

Streamlit web interface for:

* Uploading PDFs
* Building the vector index
* Asking research questions
* Generating summaries

### cli_app.py

Command-line interface for running the assistant without a UI.

Supported commands:

```bash
index
ask
summarize
list-papers
```

### src/pdf_loader.py

Responsible for:

* Loading PDFs using PyMuPDF
* Extracting page text
* Creating LangChain Documents
* Attaching metadata

Metadata includes:

* Source file
* Study ID
* Author/year
* Page number

### src/chunking.py

Responsible for:

* Splitting documents into chunks
* Assigning chunk IDs
* Detecting likely paper sections

Supported section labels:

```text
abstract
methods
results
discussion
limitations
conclusion
```

### src/vector_store.py

Responsible for:

* Creating embeddings
* Storing vectors in ChromaDB
* Persisting vector storage

### src/retrieval.py

Implements:

* Semantic retrieval
* Query expansion
* Keyword retrieval
* Hybrid retrieval
* Source filtering
* Section filtering
* Deduplication
* Source diversification

### src/agents.py

Defines LangGraph workflows for:

* Question answering
* Evidence grading
* Retrieval retry logic
* Summarization

### src/services.py

Contains application services:

```text
PaperIndexingService
HybridResearchRetriever
ClinicalResearchAssistant
```

These services keep the business logic separate from the user interface.

### src/interfaces.py

Defines lightweight interfaces using Python Protocols:

```python
DocumentLoader
ChunkingStrategy
RetrievalStrategy
```

---

## Retrieval Design

### Semantic Vector Search

The query is embedded and matched against document chunks stored in ChromaDB.

```python
create_retriever(vectorstore, top_k)
```

---

### Query Expansion

The LLM generates alternate search queries to improve retrieval recall.

Example:

```text
Original:
What are the cardiovascular outcomes?

Variations:
cardiovascular endpoints reported in the study
clinical outcomes related to heart disease
mortality and morbidity findings for cardiovascular disease
```

```python
create_query_variations(query, llm)
```

---

### Hybrid Retrieval

Combines:

1. Semantic vector search
2. Keyword search

This improves retrieval of:

* Drug names
* Biomarkers
* Clinical abbreviations
* Numeric outcomes
* Domain-specific terminology

```python
hybrid_retrieve_with_query_variations(vectorstore, query, llm, top_k)
```

---

### Section-Aware Retrieval

Chunks are tagged with likely paper sections.

Examples:

```text
"What was the sample size?"
→ methods

"What were the outcomes?"
→ results

"What are the limitations?"
→ limitations
```

```python
detect_section(text)
add_section_labels(chunks)
infer_section_filter(query)
```

---

### Paper-Specific Retrieval

If a user references a specific paper, retrieval is restricted to that paper.

Example:

```text
Summarize Honghao 2025
```

```python
detect_requested_sources(query, docs)
filter_by_requested_sources(docs, requested_sources)
```

---

### Source-Diverse Retrieval

For broad summaries, the retriever avoids selecting all context from a single paper.

This helps generate balanced summaries across multiple studies.

```python
diversify_by_source(docs)
```

---

### Evidence Quality Check

Before generation, the workflow evaluates whether the retrieved evidence is sufficient.

If evidence quality is weak:

```text
Retrieve Again
     ↓
Re-evaluate
     ↓
Generate
```

---

### Citation-Aware Generation

Answers are grounded in retrieved chunks and include source citations such as:

```text
[Source 1]
[Source 2]
```

---

## Setup

### Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux / Mac:

```bash
source .venv/bin/activate
```

---

### Install Dependencies

CLI Version:

```bash
pip install -r requirements-core.txt
```

Full Version (including Streamlit):

```bash
pip install -r requirements.txt
```

---

### Configure Environment Variables

Create a `.env` file:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

Never hardcode API keys into source code.

---

## Running the Assistant

### Build the Index

```bash
python cli_app.py index
```

### Ask a Research Question

```bash
python cli_app.py ask "What are the cardiovascular outcomes reported across the studies?"
```

### Summarize Indexed Papers

```bash
python cli_app.py summarize
```

### Summarize a Specific Paper

```bash
python cli_app.py summarize --focus "Summarize Honghao 2025"
```

### List Indexed Papers

```bash
python cli_app.py list-papers
```

---

## Streamlit Interface

Launch the web application:

```bash
streamlit run app.py
```

The Streamlit interface supports:

1. Uploading PDFs
2. Building or rebuilding the index
3. Asking research questions
4. Generating summaries

---

## Persistence

ChromaDB persists embeddings and document chunks inside:

```text
research_db/
```

Uploaded PDFs are stored in:

```text
uploaded_pdfs/
```

The index only needs to be rebuilt when source PDFs change.

---

## Deployment

The application can be deployed using:

* AWS EC2
* Streamlit Cloud
* Hugging Face Spaces
* Azure Virtual Machines

Typical deployment architecture:

```text
User
  |
  v
Streamlit
  |
  v
Clinical Research Assistant
  |
  +---- OpenAI API
  |
  +---- ChromaDB
  |
  +---- PDF Repository
```

---

## Future Improvements

* BM25 lexical retrieval
* Cross-encoder reranking
* DeepEval evaluation metrics
* Pinecone integration
* AWS production deployment
* Authentication and user management
* Multi-agent workflows
* Research paper comparison mode

---

## License

This project is intended for educational and research purposes.
