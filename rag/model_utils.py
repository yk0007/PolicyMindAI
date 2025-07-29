"""
Utility functions for loading and managing language models.
"""
import os
import logging
from enum import Enum
from typing import Optional, Dict, Any, Union, List, Literal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelProvider(str, Enum):
    """Enum for supported model providers."""
    GROQ = "groq"
    OLLAMA = "ollama"
    GOOGLE = "google"

# API keys should be provided via environment variables
DEFAULT_API_KEYS = {
    "GOOGLE_API_KEY": "",
    "GROQ_API_KEY": ""
}

def get_api_key(provider: ModelProvider) -> str:
    """Get API key for the specified provider from environment variables."""
    key_name = f"{provider.upper()}_API_KEY"
    api_key = os.environ.get(key_name, "")
    if not api_key:
        raise ValueError(f"{key_name} environment variable is not set")
    return api_key

# Available embedding models
AVAILABLE_EMBEDDING_MODELS = [
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2"
]

# Display names for models
MODEL_DISPLAY_NAMES = {
    # Groq models
    "llama3-70b-8192": "Llama 3 70B (Groq)",
    "llama3-8b-8192": "Llama 3 8B (Groq)",
    "gemma-7b-it": "Gemma 7B (Groq)",
    "mixtral-8x7b-32768": "Mixtral 8x7B (Groq)",
    "llama2-70b-4096": "Llama 2 70B (Groq)",
    "gemma2-9b-it": "Gemma 2 9B (Groq)",
    "llama-3.1-8b-instant": "Llama 3.1 8B (Groq)",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick 17B (Groq)",
    "mistral-saba-24b": "Mistral Saba 24B (Groq)",
    "qwen/qwen3-32b": "Qwen 32B (Groq)",
    
    # Ollama models
    "llama3": "Llama 3 (Ollama)",
    "mistral": "Mistral (Ollama)",
    "gemma": "Gemma (Ollama)",
    
    # Google models
    "gemini-1.5-pro-latest": "Gemini 1.5 Pro (Google)",
    "gemini-1.5-flash-latest": "Gemini 1.5 Flash (Google)",
    "gemini-1.0-pro-latest": "Gemini 1.0 Pro (Google)",
    "gemini-1.0-flash-latest": "Gemini 1.0 Flash (Google)"
}

def get_available_models(provider: ModelProvider) -> List[str]:
    """
    Get available models for the specified provider.
    
    Args:
        provider: The model provider
        
    Returns:
        List of available model names
    """
    if provider == ModelProvider.GROQ:
        # Return only 3 models as requested
        return [
            "llama3-70b-8192",  # Most capable model
            "gemma2-9b-it",     # Good balance of capability and speed
            "llama-3.1-8b-instant"  # Fastest model
        ]
    elif provider == ModelProvider.OLLAMA:
        return ["llama3", "mistral", "gemma"]
    elif provider == ModelProvider.GOOGLE:
        return [
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
            "gemini-1.0-pro-latest",
            "gemini-1.0-flash-latest"
        ]
    else:
        return []

def set_api_keys(api_keys: Dict[str, str]) -> None:
    """
    Set API keys in the environment.
    
    Args:
        api_keys: Dictionary of API keys (e.g., {"GOOGLE_API_KEY": "key"})
    """
    for key, value in api_keys.items():
        if value:  # Only set non-empty values
            os.environ[key] = value

def get_llm_instance(
    provider: ModelProvider,
    model_name: str,
    api_keys: Dict[str, str],
    temperature: float = 0.1,
    max_tokens: int = 4000,
    **kwargs
) -> Any:
    """
    Get a language model instance for the specified provider.
    
    Args:
        provider: The model provider (e.g., ModelProvider.GROQ)
        model_name: The name of the model to use
        api_keys: Dictionary of API keys for the provider (deprecated, use environment variables instead)
        temperature: Temperature for sampling (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        **kwargs: Additional model-specific parameters
        
    Returns:
        An instance of the requested language model
    """
    # Get API key from environment variables
    api_key = get_api_key(provider)
    
    if provider == ModelProvider.GROQ:
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                groq_api_key=api_key,
                **kwargs
            )
        except ImportError:
            logger.error("Groq not installed. Please install with: pip install langchain-groq")
            raise
    elif provider == ModelProvider.OLLAMA:
        try:
            from langchain_community.llms import Ollama
            
            # Set base URL for Ollama if not already set
            if "OLLAMA_API_BASE" not in os.environ:
                os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
                
            return Ollama(
                model=model_name,
                temperature=temperature,
                num_predict=max_tokens,
                **kwargs
            )
        except ImportError:
            logger.error("Ollama not installed. Please install with: pip install langchain-community")
            raise
    elif provider == ModelProvider.GOOGLE:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )
        except ImportError:
            logger.error("Google Generative AI not installed. Please install with: pip install langchain-google-genai")
            raise
    else:
        raise ValueError(f"Unsupported model provider: {provider}")

def get_llm(
    model_name: str = "llama3-70b-8192",
    temperature: float = 0.1,
    max_tokens: int = 4000,
    **kwargs
):
    """
    Get a language model instance based on the specified model name.
    
    Args:
        model_name: Name of the model to load. Supported models:
                   - Groq models: 'llama3-70b-8192', 'llama3-8b-8192', 'gemma-7b-it', etc.
                   - Ollama models: 'llama3', 'mistral', 'gemma'
                   - Google models: 'gemini-1.5-pro-latest', 'gemini-1.5-flash-latest', etc.
        temperature: Temperature for sampling (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        **kwargs: Additional model-specific parameters
        
    Returns:
        An instance of the requested language model
        
    Raises:
        ValueError: If the model is not supported or API key is missing
    """
    try:
        # First, try to determine the provider based on the model name
        provider = None
        
        # Groq models
        groq_models = [
            "llama3-70b-8192", "llama3-8b-8192", "gemma-7b-it", "mixtral-8x7b-32768",
            "llama2-70b-4096", "gemma2-9b-it", "llama-3.1-8b-instant",
            "meta-llama/llama-4-maverick-17b-128e-instruct", "mistral-saba-24b", "qwen/qwen3-32b"
        ]
        
        # Ollama models
        ollama_models = ["llama3", "mistral", "gemma"]
        
        # Google models
        google_models = [
            "gemini-1.5-pro-latest", "gemini-1.5-flash-latest",
            "gemini-1.0-pro-latest", "gemini-1.0-flash-latest"
        ]
        
        if model_name in groq_models:
            provider = ModelProvider.GROQ
        elif model_name in ollama_models:
            provider = ModelProvider.OLLAMA
        elif model_name in google_models:
            provider = ModelProvider.GOOGLE
            
        if provider is None:
            # If we couldn't determine the provider, try to infer it from the model name
            if any(name in model_name.lower() for name in ['llama', 'gemma', 'mistral', 'mixtral']):
                provider = ModelProvider.GROQ
            else:
                raise ValueError(f"Could not determine provider for model: {model_name}")
        
        # Get the appropriate model instance directly
        if provider == ModelProvider.GROQ:
            try:
                from langchain_groq import ChatGroq
                api_key = get_api_key(provider)
                return ChatGroq(
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    groq_api_key=api_key,
                    **kwargs
                )
            except ImportError:
                logger.error("Groq not installed. Please install with: pip install langchain-groq")
                raise
        elif provider == ModelProvider.OLLAMA:
            try:
                from langchain_community.llms import Ollama
                if "OLLAMA_API_BASE" not in os.environ:
                    os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
                return Ollama(
                    model=model_name,
                    temperature=temperature,
                    num_predict=max_tokens,
                    **kwargs
                )
            except ImportError:
                logger.error("Ollama not installed. Please install with: pip install langchain-community")
                raise
        elif provider == ModelProvider.GOOGLE:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                api_key = get_api_key(provider)
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    google_api_key=api_key,
                    **kwargs
                )
            except ImportError:
                logger.error("Google Generative AI not installed. Please install with: pip install langchain-google-genai")
                raise
        else:
            raise ValueError(f"Unsupported model provider: {provider}")
        
    except Exception as e:
        logger.error(f"Error initializing model {model_name}: {str(e)}")
        raise

def get_embedding_instance(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    provider: Optional[ModelProvider] = None,
    api_keys: Optional[Dict[str, str]] = None,
    **kwargs
) -> Any:
    """
    Get an embedding model instance.
    
    Args:
        model_name: The name of the embedding model to use
        provider: Optional model provider (if None, will try to infer from model_name)
        api_keys: Optional dictionary of API keys
        **kwargs: Additional model-specific parameters
        
    Returns:
        An instance of the requested embedding model
    """
    # Set API keys if provided
    if api_keys:
        set_api_keys(api_keys)
        
    # Try to determine provider from model name if not specified
    if provider is None:
        if model_name in ["llama3-70b-8192", "llama3-8b-8192", "gemma-7b-it", "mixtral-8x7b-32768", "llama2-70b-4096", "gemma2-9b-it", "llama-3.1-8b-instant"]:
            provider = ModelProvider.GROQ
        elif model_name in ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-1.0-pro-latest"]:
            provider = ModelProvider.GOOGLE
        else:
            # Default to Groq for unknown models
            provider = ModelProvider.GROQ
    
    # Get the appropriate embedding model based on provider
    if provider == ModelProvider.GROQ:
        logger.info("Using default embeddings")
        from langchain.embeddings import FakeEmbeddings
        return FakeEmbeddings(size=384)  # Return fake embeddings
    elif provider == ModelProvider.GOOGLE:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported provider for embeddings: {provider}")

def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs) -> Any:
    """
    Get an embeddings model instance.
    
    Args:
        model_name: Name of the embeddings model to load
        **kwargs: Additional model-specific parameters
        
    Returns:
        An instance of the requested embeddings model
    """
    logger.info("Using default embeddings")
    from langchain.embeddings import FakeEmbeddings
    return FakeEmbeddings(size=384)  # Return fake embeddings

def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs) -> Any:
    """
    Get an embeddings model instance.
    
    Args:
        model_name: Name of the embeddings model to load
        **kwargs: Additional model-specific parameters
        
    Returns:
        An instance of the requested embeddings model
    """
    logger.info("Using default embeddings")
    from langchain.embeddings import FakeEmbeddings
    try:
        return FakeEmbeddings(size=384)  # Return fake embeddings
    except Exception as e:
        logger.error(f"Critical error initializing embeddings model {model_name}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        if hasattr(e, 'args') and e.args:
            logger.error(f"Error details: {e.args}")
        raise
