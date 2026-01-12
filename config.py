"""
Centralized configuration for PolicyMindAI.

All configurable settings in one place for easy management.
"""

import os
from typing import Dict, Any
from enum import Enum


class Environment(Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# Determine current environment
ENVIRONMENT = Environment(os.environ.get("POLICYMIND_ENV", "development"))


# =============================================================================
# Model Configuration
# =============================================================================

DEFAULT_MODEL_PROVIDER = "groq"
DEFAULT_MODEL_NAME = "llama-3.3-70b-versatile"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4000


# =============================================================================
# Document Processing Configuration
# =============================================================================

# Semantic chunking settings (in tokens, not characters!)
CHUNK_SIZE_TOKENS = 400  # Target chunk size in tokens
CHUNK_OVERLAP_TOKENS = 50  # Overlap between chunks

# PDF processing
MAX_PDF_PAGES = 20
PDF_TIMEOUT_SECONDS = 60
PAGE_EXTRACTION_TIMEOUT = 5
TABLE_EXTRACTION_TIMEOUT = 10

# Supported file formats
SUPPORTED_FILE_FORMATS = {'.pdf', '.docx', '.doc', '.txt', '.md', '.csv', '.xlsx', '.xls'}


# =============================================================================
# Advanced RAG Configuration
# =============================================================================

# Retrieval settings
DEFAULT_K_RETRIEVAL = 5  # Number of chunks to retrieve
FETCH_K_MULTIPLIER = 3  # Fetch this many times k for reranking

# Hybrid search
USE_HYBRID_SEARCH = True
USE_BM25 = True
BM25_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5
RRF_K = 60  # Reciprocal Rank Fusion constant

# Cross-encoder reranking
USE_RERANKING = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 15  # Number of candidates to rerank

# Context packing
MAX_CONTEXT_TOKENS = 4000  # Max tokens for LLM context


# =============================================================================
# Query Relevance Configuration
# =============================================================================

QUERY_RELEVANCE_THRESHOLD = 0.15
USE_LLM_FOR_CLASSIFICATION = False


# =============================================================================
# Vector Store Settings
# =============================================================================

INDEX_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "indices"
)


# =============================================================================
# Logging Configuration
# =============================================================================

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
STRUCTURED_LOGGING = ENVIRONMENT == Environment.PRODUCTION


# =============================================================================
# Security Configuration
# =============================================================================

API_KEY_NAMES = {
    "groq": "GROQ_API_KEY",
    "google": "GOOGLE_API_KEY",
    "ollama": "OLLAMA_API_BASE",
}


# =============================================================================
# Streamlit Configuration
# =============================================================================

PAGE_TITLE = "PolicyMind RAG"
PAGE_ICON = "📄"
LAYOUT = "wide"


# =============================================================================
# Helper Functions
# =============================================================================

def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary."""
    return {
        "environment": ENVIRONMENT.value,
        "model": {
            "provider": DEFAULT_MODEL_PROVIDER,
            "name": DEFAULT_MODEL_NAME,
            "embedding": DEFAULT_EMBEDDING_MODEL,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        },
        "chunking": {
            "chunk_size_tokens": CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
            "max_pdf_pages": MAX_PDF_PAGES,
        },
        "retrieval": {
            "k": DEFAULT_K_RETRIEVAL,
            "use_hybrid": USE_HYBRID_SEARCH,
            "use_reranking": USE_RERANKING,
            "reranker_model": RERANKER_MODEL,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
        },
        "query": {
            "relevance_threshold": QUERY_RELEVANCE_THRESHOLD,
        },
        "logging": {
            "level": LOG_LEVEL,
            "structured": STRUCTURED_LOGGING,
        },
    }


def is_production() -> bool:
    """Check if running in production."""
    return ENVIRONMENT == Environment.PRODUCTION


def is_development() -> bool:
    """Check if running in development."""
    return ENVIRONMENT == Environment.DEVELOPMENT
