# Financial-Docs RAG Chatbot

A multi-agent RAG pipeline over financial documents (PDFs, CSVs) and live market data, powered by local Ollama models and CrewAI agent orchestration.

## Features

- **Multi-Agent Architecture**: Router automatically selects the right CrewAI agent (Summarizer, Q&A, or MCQ Generator)
- **Live API Integration**: Fetch real-time stock data and company profiles via Yahoo Finance
- **Interactive CLI Menu**: Document and API data management with scoped chat
- **Streamlit Web UI**: Optional visual interface with document upload and chat history
- **Multi-Document Support**: Chat across all documents or scope to a specific file
- **Idempotent Ingestion**: Re-ingesting the same file replaces old chunks (no duplicates)
- **Local-First**: All LLM processing via Ollama (no cloud APIs, no data leaves your machine)

## Agents

| Role       | Model (Ollama)            | Purpose                                              |
|------------|---------------------------|------------------------------------------------------|
| Router     | `mistral:7b-instruct`     | Classifies intent -> `SUMMARIZE` / `QA` / `MCQ`      |
| Summarizer | `mistral:7b-instruct`     | Summarizes documents via CrewAI agent                |
| Q&A        | `llama3.1:8b`             | Grounded answers from retrieved context              |
| MCQ        | `qwen2.5:7b-instruct`     | Generates validated multiple-choice questions (JSON) |

3 models, 4 roles — the lightest model (`mistral:7b-instruct`) serves both router and summarizer.

## Stack

- **Python 3.11+**
- **LangChain** — document loaders, text splitter, retriever, embeddings
- **CrewAI** — multi-agent orchestration (researcher, summarizer, insight generator)
- **ChromaDB** — persistent local vector store
- **Ollama** — local LLM runtime + embeddings (`nomic-embed-text`)
- **PyMuPDF4LLM** — PDF -> markdown (preserves tables, critical for financial docs)
- **pandas** — CSV ingestion (one row per Document)
- **yfinance** — Yahoo Finance API (stock prices, company profiles)
- **Pydantic** — structured MCQ output validation
- **Streamlit** — optional chat UI

## Prerequisites

### 1. Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux/Windows:**
Visit https://ollama.com/download and follow installation instructions.

### 2. Start Ollama Server

```bash
ollama serve
```

Keep this running in a separate terminal.

### 3. Pull Required Models

```bash
ollama pull mistral:7b-instruct
ollama pull llama3.1:8b
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

One-time download (~15GB total).

## Installation

```bash
# Clone the repository
git clone <repo-url> PoC
cd PoC

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

Edit `.env` if Ollama is not running on `localhost:11434`.

## Usage

Run from the project root so `from src...` imports resolve.

### Interactive CLI (Primary)

```bash
python main.py
```

```
======================================================================
    FINANCIAL DOCS RAG CHATBOT
======================================================================

MAIN MENU
----------------------------------------------------------------------
1. Load Documents (PDF/CSV)
2. Load API Data (Stock/Company Info)
3. Chat (RAG + Agentic Workflow)
4. View Loaded Documents
5. Exit
```

**1. Load Documents** — Ingest PDF or CSV files:
```
File path(s): data/apple_10k.pdf
[SUCCESS] Ingested apple_10k.pdf -> 7 chunks
```

**2. Load API Data** — Fetch from Yahoo Finance:
- Stock Price Data: Historical OHLCV (e.g., AAPL, 30 days)
- Company Information: Profile, sector, industry, market cap

**3. Chat** — Select a document scope, then ask anything:
```
> summarize the file

======================================================================
AGENT: Summarizer (mistral:7b-instruct)
OPERATION: Map-reduce summarization
======================================================================

Apple Inc. reported Q1 FY2026 revenue of $124.3B...
======================================================================

> what was the revenue growth?

======================================================================
AGENT: Q&A (llama3.1:8b)
OPERATION: Retrieval-based question answering
======================================================================

Revenue increased by 12% year-over-year to $124.3B in Q1 FY2026.
======================================================================

> give me a quiz

======================================================================
AGENT: MCQ Generator (qwen2.5:7b-instruct)
OPERATION: Multiple-choice question generation
======================================================================

Question 1: What was Apple's total revenue in Q1 FY2026?
   A. $110.5B
   B. $124.3B
   C. $135.2B
   D. $98.7B

   Answer: B. $124.3B
   Explanation: According to the financial statements...
======================================================================
```

### Streamlit Web UI

```bash
streamlit run app.py
```

Open http://localhost:8501. Features: drag-and-drop file upload, document scope selector, chat history, interactive MCQ radio buttons.

## How It Works

```
user input -> router.classify() [mistral, temp=0]
  |
  |-- SUMMARIZE -> CrewAI Summarizer Agent [mistral, temp=0.2]
  |     all chunks for doc_id -> 7-12 bullet point summary
  |
  |-- QA -> CrewAI Researcher Agent [llama3.1, temp=0.1]
  |     top-k=4 similarity search -> context stuffing -> answer
  |
  +-- MCQ -> CrewAI Insight Generator Agent [qwen2.5, temp=0.4]
        top-k=6 similarity search -> JSON MCQs -> Pydantic validation
```

- Ambiguous input defaults to QA
- Session is stateless (no chat memory across turns)
- All chunks persist in ChromaDB for fast retrieval

## Project Structure

```
PoC/
├── main.py                  # CLI entry point (interactive menu)
├── app.py                   # Streamlit UI entry point
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── config.py            # models, paths, chunk sizes (single source of truth)
│   ├── ingestion.py         # PDF + CSV loaders, chunking, ingest()
│   ├── vectorstore.py       # Chroma store, retriever, doc management
│   ├── router.py            # intent classification, dispatch, MCQ model
│   ├── crew_agents.py       # CrewAI agents and workflow orchestration
│   └── api_sources.py       # Yahoo Finance integration (stock data, company info)
├── prompts/                 # LLM prompt templates
│   ├── router.txt
│   ├── summarize.txt
│   ├── qa.txt
│   └── mcq.txt
├── data/                    # input documents (gitignored)
└── chroma_db/               # persisted vector embeddings (gitignored)
```

## Supported Document Types

### PDF
- Parsed with PyMuPDF4LLM (preserves table structure as markdown)
- One Document per page, chunked at 1000 chars with 150-char overlap
- Metadata: `{source, page, doc_id}`

### CSV
- One Document per row, formatted as `col1: val1 | col2: val2 | ...`
- NOT chunked (each row is a complete document)
- Metadata: `{source, row, doc_id}`

### API Data (Yahoo Finance)
- **Stock prices**: OHLCV per day, one Document per trading day
- **Company profiles**: sector, industry, employees, business summary
- Metadata: `{source, symbol, doc_id}`

## Configuration

All settings in `src/config.py`:

```python
ROUTER_MODEL = "mistral:7b-instruct"
SUMMARIZER_MODEL = "mistral:7b-instruct"
QA_MODEL = "llama3.1:8b"
MCQ_MODEL = "qwen2.5:7b-instruct"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
QA_K = 4      # top-k chunks for Q&A
MCQ_K = 6     # top-k chunks for MCQ generation
```

Environment variables (`.env`):
```bash
OLLAMA_HOST=http://localhost:11434
ANONYMIZED_TELEMETRY=False
```
