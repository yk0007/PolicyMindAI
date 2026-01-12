"""
Model Utilities for PolicyMindAI.

Simplified to use Groq only with best embedding model.
"""

import os
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Best models (no user selection needed)
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Available Groq models
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768"
]

# For backward compatibility
from enum import Enum

class ModelProvider(str, Enum):
    """Supported model providers."""
    GROQ = "groq"
    OLLAMA = "ollama"
    GOOGLE = "google"


MODEL_DISPLAY_NAMES = {
    ModelProvider.GROQ: "Groq (Cloud)",
}

AVAILABLE_EMBEDDING_MODELS = [DEFAULT_EMBEDDING_MODEL]

DEFAULT_API_KEYS = {}


# =============================================================================
# API Key Management
# =============================================================================

def get_api_key() -> str:
    """Get Groq API key from environment or Streamlit secrets."""
    # Try environment variable first
    api_key = os.environ.get("GROQ_API_KEY")
    
    # Try Streamlit secrets as fallback
    if not api_key:
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                api_key = st.secrets.get("GROQ_API_KEY") or st.secrets.get("secrets", {}).get("GROQ_API_KEY")
        except ImportError:
            pass
    
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please set it as an environment variable "
            "or in your .env file."
        )
    
    return api_key


# =============================================================================
# LLM Initialization
# =============================================================================

def get_llm_instance(
    provider: ModelProvider = ModelProvider.GROQ,
    model_name: str = DEFAULT_LLM_MODEL,
    api_keys: Optional[Dict[str, str]] = None,
    temperature: float = 0.1,
    max_tokens: int = 4000,
    **kwargs
) -> Any:
    """
    Get a Groq language model instance.
    
    Args:
        provider: Ignored (always uses Groq)
        model_name: Name of the model
        api_keys: Ignored (uses environment)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Language model instance
    """
    try:
        from langchain_groq import ChatGroq
        api_key = get_api_key()
        
        return ChatGroq(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=api_key,
            **kwargs
        )
    except ImportError:
        raise ImportError("Install langchain-groq: pip install langchain-groq")


def get_available_models(provider: ModelProvider = ModelProvider.GROQ) -> List[str]:
    """Get available models."""
    return GROQ_MODELS


# Convenience alias
def get_llm(
    model_name: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4000,
    **kwargs
) -> Any:
    """Get LLM instance with sensible defaults."""
    return get_llm_instance(
        ModelProvider.GROQ, 
        model_name, 
        temperature=temperature,
        max_tokens=max_tokens, 
        **kwargs
    )


# =============================================================================
# Embedding Models
# =============================================================================

def get_embedding_instance(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    **kwargs
) -> Any:
    """
    Get the best embeddings model instance.
    
    Args:
        model_name: Embedding model name (defaults to best model)
        
    Returns:
        Embeddings model instance
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError:
        logger.error("HuggingFace Embeddings not installed")
        raise ImportError(
            "Install sentence-transformers: pip install sentence-transformers langchain-community"
        )


# Alias for backward compatibility
get_embeddings = get_embedding_instance
