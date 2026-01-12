"""
RAG Index Management for PolicyMindAI.

Handles vector store creation, loading, and saving using FAISS.
"""

import os
import hashlib
import pickle
import logging
from typing import List, Optional, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Default index directory
INDEX_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "indices"
)
os.makedirs(INDEX_DIR, exist_ok=True)


class RAGIndex:
    """
    Manages RAG vector store operations including creation, loading, and saving.
    
    Uses FAISS for efficient vector storage and similarity search.
    """
    
    def __init__(self, index_dir: Optional[str] = None):
        """
        Initialize the RAG index manager.
        
        Args:
            index_dir: Directory to store indices. Defaults to 'indices' folder.
        """
        self.index_dir = index_dir or INDEX_DIR
        os.makedirs(self.index_dir, exist_ok=True)
        self.vector_store = None
        self._last_index_path = None
    
    def _get_index_id(self, doc_paths: List[str]) -> str:
        """Generate a unique ID based on document paths."""
        sorted_paths = sorted(doc_paths)
        paths_str = "".join(sorted_paths)
        return hashlib.md5(paths_str.encode()).hexdigest()
    
    def _get_index_path(self, index_id: str) -> str:
        """Get the file path for an index."""
        return os.path.join(self.index_dir, index_id)
    
    def create_vector_store(
        self,
        documents: List[Document],
        embeddings: Embeddings
    ) -> FAISS:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of documents to index
            embeddings: Embedding model instance
            
        Returns:
            New FAISS vector store
        """
        if not documents:
            raise ValueError("No documents provided")
        
        logger.info(f"Creating vector store with {len(documents)} documents")
        return FAISS.from_documents(documents, embeddings)
    
    def save_vector_store(
        self,
        vector_store: FAISS,
        index_path: str
    ) -> None:
        """
        Save vector store to disk.
        
        Args:
            vector_store: The vector store to save
            index_path: Path to save the index
        """
        try:
            os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else '.', exist_ok=True)
            vector_store.save_local(index_path)
            self._last_index_path = index_path
            logger.info(f"Vector store saved to {index_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_vector_store(
        self,
        index_path: str,
        embeddings: Embeddings
    ) -> Optional[FAISS]:
        """
        Load vector store from disk.
        
        Args:
            index_path: Path to the index directory
            embeddings: Embedding model instance
            
        Returns:
            Loaded vector store or None if not found
        """
        try:
            faiss_index = os.path.join(index_path, "index.faiss")
            if not os.path.exists(faiss_index):
                logger.debug(f"No index found at {index_path}")
                return None
            
            logger.info(f"Loading vector store from {index_path}")
            return FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None
    
    def get_or_create_vector_store(
        self,
        documents: List[Document],
        embeddings: Embeddings,
        doc_paths: Optional[List[str]] = None
    ) -> FAISS:
        """
        Get existing vector store or create a new one.
        
        Args:
            documents: List of documents to index
            embeddings: Embedding model instance
            doc_paths: List of document paths for cache lookup
            
        Returns:
            Vector store (loaded or newly created)
        """
        # If no paths provided, create new index
        if not doc_paths:
            logger.info("No document paths provided, creating new vector store")
            self.vector_store = self.create_vector_store(documents, embeddings)
            return self.vector_store
        
        # Try to load existing index
        index_id = self._get_index_id(doc_paths)
        index_path = self._get_index_path(index_id)
        
        self.vector_store = self.load_vector_store(index_path, embeddings)
        
        if self.vector_store is None:
            logger.info("Creating new vector store")
            self.vector_store = self.create_vector_store(documents, embeddings)
            self.save_vector_store(self.vector_store, index_path)
        else:
            logger.info("Loaded existing vector store")
        
        return self.vector_store
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        if self.vector_store is None:
            raise ValueError("No vector store loaded")
        
        return self.vector_store.similarity_search(query, k=k, filter=filter)
    
    def save(self, index_path: Optional[str] = None) -> None:
        """Save current vector store."""
        if self.vector_store is None:
            logger.warning("No vector store to save")
            return
        
        save_path = index_path or self._last_index_path
        if save_path is None:
            raise ValueError("No index path provided")
        
        self.save_vector_store(self.vector_store, save_path)


# =============================================================================
# Module-level Functions (for backward compatibility)
# =============================================================================

_default_index = RAGIndex()


def get_or_create_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
    doc_paths: Optional[List[str]] = None
) -> FAISS:
    """
    Get or create a vector store (module-level function).
    
    Args:
        documents: List of documents to index
        embeddings: Embedding model instance
        doc_paths: List of document paths for cache lookup
        
    Returns:
        Vector store
    """
    return _default_index.get_or_create_vector_store(documents, embeddings, doc_paths)


def create_vector_store(
    documents: List[Document],
    embeddings: Embeddings
) -> FAISS:
    """Create a new vector store."""
    return _default_index.create_vector_store(documents, embeddings)


def load_vector_store(
    index_path: str,
    embeddings: Embeddings
) -> Optional[FAISS]:
    """Load a vector store from disk."""
    return _default_index.load_vector_store(index_path, embeddings)


def save_vector_store(
    vector_store: FAISS,
    index_path: str
) -> None:
    """Save a vector store to disk."""
    _default_index.save_vector_store(vector_store, index_path)
