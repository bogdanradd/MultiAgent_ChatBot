from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

ROOT = Path(__file__).parent.parent
CHROMA_DIR = str(ROOT / "chroma_db")
PROMPTS_DIR = ROOT / "prompts"
COLLECTION_NAME = "financial_docs"

EMBED_MODEL = "nomic-embed-text"
ROUTER_MODEL = "mistral:7b-instruct"
SUMMARIZER_MODEL = "mistral:7b-instruct"
QA_MODEL = "llama3.1:8b"
MCQ_MODEL = "qwen2.5:7b-instruct"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
QA_K = 4
MCQ_K = 6
