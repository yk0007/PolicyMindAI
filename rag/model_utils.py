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
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"

# Default API keys (can be overridden by user input)
DEFAULT_API_KEYS = {
    "GOOGLE_API_KEY": "***REMOVED***",
    "GROQ_API_KEY": "***REMOVED***"
}

# Available embedding models
AVAILABLE_EMBEDDING_MODELS = [
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-base-v2",
    "thenlper/gte-base"
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
        api_keys: Dictionary of API keys for the provider
        temperature: Temperature for sampling (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        **kwargs: Additional model-specific parameters
        
    Returns:
        An instance of the requested language model
    """
    # Set the API keys first
    set_api_keys(api_keys)
    
    if provider == ModelProvider.GROQ:
        return get_llm(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
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
) -> Any:
    """
    Get a language model instance based on the specified model name.
    
    Args:
        model_name: Name of the model to load. Supported models:
                   - Groq models: 'llama3-70b-8192', 'llama3-8b-8192', 'gemma-7b-it', etc.
        temperature: Temperature for sampling (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        **kwargs: Additional model-specific parameters
        
    Returns:
        An instance of the requested language model
    """
    try:
        # Try to import Groq first
        from groq import Groq
        from langchain_groq import ChatGroq
        
        # Default Groq model mapping
        groq_models = {
            'llama3-70b-8192': 'llama3-70b-8192',
            'llama3-8b-8192': 'llama3-8b-8192',
            'gemma-7b-it': 'gemma-7b-it',
            'mixtral-8x7b-32768': 'mixtral-8x7b-32768',
            'llama2-70b-4096': 'llama2-70b-4096',
            'gemma2-9b-it': 'gemma2-9b-it',
            'llama-3.1-8b-instant': 'llama-3.1-8b-instant',
            'meta-llama/llama-4-maverick-17b-128e-instruct': 'meta-llama/llama-4-maverick-17b-128e-instruct',
            'mistral-saba-24b': 'mistral-saba-24b',
            'qwen/qwen3-32b': 'qwen/qwen3-32b'
        }
        
        # Get the model name, defaulting to the input if not found in the mapping
        model_id = groq_models.get(model_name, model_name)
        
        # Get the API key from environment variables
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        logger.info(f"Initializing Groq model: {model_id}")
        
        # Initialize the Groq chat model
        llm = ChatGroq(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=api_key,
            **kwargs
        )
        
        return llm
        
    except ImportError as e:
        logger.error(f"Failed to import required packages: {str(e)}")
        raise
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
    
    # Try to infer provider from model_name if not provided
    if provider is None:
        if any(name in model_name.lower() for name in ["ada", "babbage", "curie", "davinci", "text-embedding"]):
            provider = ModelProvider.OPENAI
        elif any(name in model_name.lower() for name in ["llama", "gemma", "mixtral", "mistral", "qwen"]):
            provider = ModelProvider.GROQ
        else:
            # Default to HuggingFace
            provider = ModelProvider.HUGGINGFACE
    
    try:
        if provider in [ModelProvider.GROQ, ModelProvider.HUGGINGFACE]:
            return get_embeddings(model_name=model_name, **kwargs)
        elif provider == ModelProvider.OPENAI:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model_name, **kwargs)
        elif provider == ModelProvider.GOOGLE:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(model=model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    except ImportError as e:
        logger.error(f"Failed to import required packages for {provider}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error initializing embedding model {model_name}: {str(e)}")
        raise

def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs) -> Any:
    """
    Get an embeddings model instance.
    
    Args:
        model_name: Name of the embeddings model to load
        **kwargs: Additional model-specific parameters
        
    Returns:
        An instance of the requested embeddings model
    """
    try:
        logger.info(f"Attempting to initialize embeddings model: {model_name}")
        
        # First try to use HuggingFace embeddings
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            # Initialize the embeddings model with CPU fallback
            model_kwargs = {'device': 'cpu'}
            if 'device' in kwargs:
                model_kwargs['device'] = kwargs.pop('device')
                
            logger.info(f"Initializing HuggingFace embeddings with model: {model_name}")
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                **kwargs
            )
            
            # Test the embeddings with a small input
            test_text = "Test embedding"
            test_embedding = embeddings.embed_query(test_text)
            if not test_embedding or len(test_embedding) == 0:
                raise ValueError("Embedding generation failed - empty result")
                
            logger.info(f"Successfully initialized embeddings model: {model_name}")
            return embeddings
            
        except Exception as hf_error:
            logger.warning(f"Failed to initialize HuggingFace embeddings: {str(hf_error)}")
            logger.info("Falling back to FakeEmbeddings for testing")
            
            # Fallback to FakeEmbeddings if available
            try:
                from langchain_community.embeddings import FakeEmbeddings
                logger.warning("Using FakeEmbeddings as fallback. This is only suitable for testing!")
                return FakeEmbeddings(size=384)  # Standard size for all-MiniLM-L6-v2
                
            except ImportError:
                logger.error("FakeEmbeddings not available. Please install langchain-community")
                raise
            
    except Exception as e:
        logger.error(f"Critical error initializing embeddings model {model_name}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        if hasattr(e, 'args') and e.args:
            logger.error(f"Error details: {e.args}")
        raise
