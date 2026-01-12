"""
RAG (Retrieval-Augmented Generation) module for PolicyMindAI.

Advanced features:
- Semantic chunking with table preservation
- Hybrid search (BM25 + Vector)
- Cross-encoder reranking
- Query relevance filtering
"""

from .document_loader import (
    process_document,
    extract_text_from_file,
    is_insurance_document,
    clean_text,
    DocumentLoader,
)
from .chunker import (
    SemanticChunker,
    chunk_document,
    estimate_tokens,
    SectionType,
)
from .rag_index import (
    create_vector_store,
    load_vector_store,
    save_vector_store,
    get_or_create_vector_store,
    RAGIndex,
)
from .retriever import (
    HybridRetriever,
    create_hybrid_retriever,
    retrieve_with_reranking,
    RetrievedDocument,
)
from .query_engine import (
    get_rag_response,
    get_suggested_questions,
)
from .query_filter import (
    is_query_relevant,
    classify_query,
    QueryRelevance,
    get_suggested_policy_questions,
    EmbeddingQueryClassifier,
    get_embedding_classifier,
)
from .model_utils import (
    get_llm_instance,
    get_embedding_instance,
    get_available_models,
    ModelProvider,
    AVAILABLE_EMBEDDING_MODELS,
    MODEL_DISPLAY_NAMES,
)

__all__ = [
    # Document processing
    'process_document',
    'extract_text_from_file',
    'is_insurance_document',
    'clean_text',
    'DocumentLoader',
    
    # Semantic chunking
    'SemanticChunker',
    'chunk_document',
    'estimate_tokens',
    'SectionType',
    
    # Vector store
    'create_vector_store',
    'load_vector_store',
    'save_vector_store',
    'get_or_create_vector_store',
    'RAGIndex',
    
    # Hybrid retrieval
    'HybridRetriever',
    'create_hybrid_retriever',
    'retrieve_with_reranking',
    'RetrievedDocument',
    
    # Query engine
    'get_rag_response',
    'get_suggested_questions',
    
    # Query filter
    'is_query_relevant',
    'classify_query',
    'QueryRelevance',
    'get_suggested_policy_questions',
    'EmbeddingQueryClassifier',
    'get_embedding_classifier',
    
    # Models
    'get_llm_instance',
    'get_embedding_instance',
    'get_available_models',
    'ModelProvider',
    'AVAILABLE_EMBEDDING_MODELS',
    'MODEL_DISPLAY_NAMES',
]
