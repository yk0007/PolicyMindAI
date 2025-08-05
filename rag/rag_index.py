import os
from typing import List, Optional, Dict, Any
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

# Define the base directory for storing vector indices
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "indices")
os.makedirs(INDEX_DIR, exist_ok=True)
import pickle
import hashlib
import logging

logger = logging.getLogger(__name__)

class RAGIndex:
    """
    A class to manage RAG vector store operations including creation, loading, and saving.
    """
    
    def __init__(self, index_dir: Optional[str] = None):
        """
        Initialize the RAGIndex.
        
        Args:
            index_dir: Directory to store the FAISS indices. If None, uses a default directory.
        """
        self.index_dir = index_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "indices"
        )
        os.makedirs(self.index_dir, exist_ok=True)
        self.vector_store = None
    
    def _get_index_id(self, doc_paths: List[str]) -> str:
        """
        Generate a unique ID for the index based on document paths.
        
        Args:
            doc_paths: List of document file paths
            
        Returns:
            A unique string ID
        """
        # Sort paths for consistent hashing
        sorted_paths = sorted(doc_paths)
        # Create a unique string from the paths
        paths_str = "".join(sorted_paths)
        # Generate a hash of the paths
        return hashlib.md5(paths_str.encode()).hexdigest()
    
    def _get_index_path(self, index_id: str) -> str:
        """
        Get the file path for an index with the given ID.
        
        Args:
            index_id: The unique ID of the index
            
        Returns:
            Path to the index file
        """
        return os.path.join(self.index_dir, f"{index_id}")
    
    def _save_vector_store(self, vector_store: VectorStore, index_path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            vector_store: The vector store to save
            index_path: Path to save the index to
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            # Save the FAISS index
            if hasattr(vector_store, 'save_local'):
                vector_store.save_local(index_path)
            else:
                # Fallback for other vector stores
                with open(index_path, 'wb') as f:
                    pickle.dump(vector_store, f)
            
            logger.info(f"Vector store saved to {index_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def _load_vector_store(self, index_path: str, embeddings: Embeddings) -> Optional[VectorStore]:
        """
        Load a vector store from disk.
        
        Args:
            index_path: Path to the index directory
            embeddings: The embeddings model to use
            
        Returns:
            The loaded vector store or None if loading failed
        """
        try:
            # Check if the index exists
            if not os.path.exists(index_path):
                logger.warning(f"Index not found at {index_path}")
                return None
                
            # Try to load as FAISS first
            if os.path.isdir(index_path) and os.path.exists(os.path.join(index_path, "index.faiss")):
                logger.info(f"Loading FAISS vector store from {index_path}")
                return FAISS.load_local(index_path, embeddings)
                
            # Fallback to pickle for other vector stores
            with open(index_path, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            logger.error(f"Error loading vector store from {index_path}: {str(e)}")
            return None
    
    def create_vector_store(self, documents: List[Document], embeddings: Embeddings) -> VectorStore:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of documents to index
            embeddings: The embeddings model to use
            
        Returns:
            A new FAISS vector store
        """
        logger.info(f"Creating new vector store with {len(documents)} documents")
        return FAISS.from_documents(documents, embeddings)
    
    def get_or_create_vector_store(
        self,
        documents: List[Document],
        embeddings: Embeddings,
        doc_paths: Optional[List[str]] = None
    ) -> VectorStore:
        """
        Get an existing vector store or create a new one if it doesn't exist.
        
        Args:
            documents: List of documents to index
            embeddings: The embeddings model to use
            doc_paths: List of document paths (for checking existing indices)
            
        Returns:
            A vector store (either loaded or newly created)
        """
        # If no doc_paths provided, always create a new index
        if not doc_paths:
            logger.info("No document paths provided, creating new vector store")
            self.vector_store = self.create_vector_store(documents, embeddings)
            return self.vector_store
        
        # Generate a unique ID for the documents
        index_id = self._get_index_id(doc_paths)
        index_path = self._get_index_path(index_id)
        
        # Try to load existing index
        self.vector_store = self._load_vector_store(index_path, embeddings)
        
        if self.vector_store is None:
            # Create a new index if loading failed
            logger.info("Creating new vector store")
            self.vector_store = self.create_vector_store(documents, embeddings)
            
            # Save the new index
            self._save_vector_store(self.vector_store, index_path)
        else:
            logger.info("Loaded existing vector store from disk")
        
        return self.vector_store
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform a similarity search on the vector store.
        
        Args:
            query: The query string
            k: Number of results to return
            filter: Optional filter to apply to the search
            
        Returns:
            List of documents most similar to the query
        """
        if self.vector_store is None:
            raise ValueError("No vector store loaded. Call get_or_create_vector_store first.")
            
        return self.vector_store.similarity_search(query, k=k, filter=filter)
    
    def save(self, index_path: Optional[str] = None) -> None:
        """
        Save the current vector store to disk.
        
        Args:
            index_path: Optional path to save the index to. If None, uses the last used path.
        """
        if self.vector_store is None:
            logger.warning("No vector store to save")
            return
            
        if index_path is None and not hasattr(self, '_last_index_path'):
            raise ValueError("No index path provided and no previous path found")
            
        save_path = index_path or getattr(self, '_last_index_path')
        self._save_vector_store(self.vector_store, save_path)
        self._last_index_path = save_path

def get_index_id(doc_paths: List[str]) -> str:
    """
    Generate a unique ID for the index based on document paths.
    
    Args:
        doc_paths: List of document file paths
        
    Returns:
        A unique string ID
    """
    # Sort paths for consistent hashing
    sorted_paths = sorted(doc_paths)
    # Create a unique string from the paths
    paths_str = "".join(sorted_paths)
    # Generate a hash of the paths
    return hashlib.md5(paths_str.encode()).hexdigest()

def get_index_path(index_id: str) -> str:
    """
    Get the file path for an index with the given ID.
    
    Args:
        index_id: The unique ID of the index
        
    Returns:
        Path to the index file
    """
    return os.path.join(INDEX_DIR, f"{index_id}")

def save_vector_store(vector_store, index_path: str) -> None:
    """
    Save the vector store to disk.
    
    Args:
        vector_store: The vector store to save
        index_path: Path to save the index to
    """
    # Save the FAISS index
    vector_store.save_local(index_path)
    
    # Save additional metadata
    metadata = {
        "document_paths": [doc.metadata.get("source") for doc in vector_store.docstore._dict.values() 
                          if hasattr(doc, "metadata") and "source" in doc.metadata],
        "embedding_model": vector_store.embedding_function.__class__.__name__
    }
    
    metadata_path = f"{index_path}.meta"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

def load_vector_store(index_path: str, embeddings: Embeddings):
    """
    Load a vector store from disk.
    
    Args:
        index_path: Path to the index directory
        embeddings: The embeddings model to use
        
    Returns:
        The loaded vector store or None if loading failed
    """
    try:
        # Check if the index exists
        if not os.path.exists(f"{index_path}/index.faiss"):
            return None
            
        # Load the FAISS index
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        return None

def create_vector_store(documents: List[Document], embeddings: Embeddings):
    """
    Create a new vector store from documents.
    
    Args:
        documents: List of documents to index
        embeddings: The embeddings model to use
        
    Returns:
        A new FAISS vector store
    """
    if not documents:
        raise ValueError("No documents provided to create vector store")
    
    # Create a new FAISS index
    return FAISS.from_documents(documents, embeddings)

def get_or_create_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
    doc_paths: Optional[List[str]] = None
):
    """
    Get an existing vector store or create a new one if it doesn't exist.
    Uses LangChain's FAISS integration for efficient vector storage and retrieval.
    
    Args:
        documents: List of documents to index
        embeddings: The embeddings model to use
        doc_paths: List of document paths (for checking existing indices)
        
    Returns:
        A vector store (either loaded or newly created)
    """
    if not doc_paths and documents:
        # Extract source paths from documents if not provided
        doc_paths = [doc.metadata.get("source") for doc in documents 
                    if hasattr(doc, "metadata") and "source" in doc.metadata]
    
    if not doc_paths:
        raise ValueError("No document paths provided and none could be extracted from documents")
    
    # Generate a unique ID for these documents
    index_id = get_index_id(doc_paths)
    index_path = get_index_path(index_id)
    
    # Try to load existing index
    vector_store = load_vector_store(index_path, embeddings)
    
    if vector_store is not None:
        return vector_store
    
    # Create a new index if none exists
    from langchain_community.vectorstores import FAISS
    
    # Create a new FAISS index with the documents
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save the new index
    vector_store.save_local(index_path)
    
    return vector_store
