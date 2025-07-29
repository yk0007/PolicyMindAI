"""
RAG (Retrieval-Augmented Generation) module for document processing and question answering.

This module provides functionality for:
- Document loading and processing
- Text chunking and embedding
- Vector store management
- Question answering with source attribution
"""

from .document_loader import process_document, extract_text_from_file, is_insurance_document
from .rag_index import create_vector_store, load_vector_store, save_vector_store
from .query_engine import get_rag_response, get_suggested_questions

__all__ = [
    'process_document',
    'extract_text_from_file',
    'is_insurance_document',
    'create_vector_store',
    'load_vector_store',
    'save_vector_store',
    'get_rag_response',
    'get_suggested_questions',
]
