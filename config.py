"""
CareerPath AI — Configuration
Loads environment variables and exposes project-wide settings.
"""

import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Suppress noisy transformers warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")

# ── Groq LLM ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# ── Embeddings ────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ──────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# ── Retrieval ─────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", 4))

# ── Paths ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
