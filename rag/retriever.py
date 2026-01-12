"""
Hybrid Retriever for PolicyMindAI.

Advanced retrieval combining:
- Dense vector search (FAISS)
- Sparse keyword search (BM25)
- Cross-encoder reranking
- Reciprocal Rank Fusion
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 not available, hybrid search disabled")

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers CrossEncoder not available, reranking disabled")


@dataclass
class RetrievedDocument:
    """Represents a retrieved document with scores."""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


class HybridRetriever:
    """
    Hybrid retrieval system combining dense and sparse search.
    
    Features:
    - Vector similarity search (semantic understanding)
    - BM25 keyword search (exact term matching)
    - Reciprocal Rank Fusion (combines rankings)
    - Cross-encoder reranking (refines top candidates)
    """
    
    # Default cross-encoder model
    DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(
        self,
        vector_store,
        documents: Optional[List[Dict[str, Any]]] = None,
        use_bm25: bool = True,
        use_reranker: bool = True,
        reranker_model: Optional[str] = None,
        rrf_k: int = 60,  # RRF constant
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_store: FAISS vector store instance
            documents: List of documents for BM25 index (optional)
            use_bm25: Whether to use BM25 search
            use_reranker: Whether to use cross-encoder reranking
            reranker_model: CrossEncoder model name
            rrf_k: Reciprocal Rank Fusion constant (default 60)
            vector_weight: Weight for vector search in fusion
            bm25_weight: Weight for BM25 in fusion
        """
        self.vector_store = vector_store
        self.use_bm25 = use_bm25 and BM25_AVAILABLE
        self.use_reranker = use_reranker and CROSS_ENCODER_AVAILABLE
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # BM25 index
        self.bm25 = None
        self.bm25_documents = []
        self.bm25_doc_ids = []
        
        # Cross-encoder for reranking
        self.reranker = None
        self.reranker_model = reranker_model or self.DEFAULT_RERANKER
        
        # Initialize BM25 if documents provided
        if documents and self.use_bm25:
            self._build_bm25_index(documents)
        
        # Lazy-load reranker (heavy model)
        self._reranker_loaded = False
    
    def _build_bm25_index(self, documents: List[Dict[str, Any]]):
        """Build BM25 index from documents."""
        if not BM25_AVAILABLE:
            return
        
        logger.info(f"Building BM25 index for {len(documents)} documents")
        
        self.bm25_documents = []
        self.bm25_doc_ids = []
        tokenized_docs = []
        
        for i, doc in enumerate(documents):
            text = doc.get('text') or doc.get('page_content', '')
            if text:
                self.bm25_documents.append(text)
                self.bm25_doc_ids.append(doc.get('metadata', {}).get('chunk_id', i))
                # Simple tokenization
                tokens = self._tokenize(text)
                tokenized_docs.append(tokens)
        
        if tokenized_docs:
            self.bm25 = BM25Okapi(tokenized_docs)
            logger.info(f"BM25 index built with {len(tokenized_docs)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        import re
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Remove very short tokens
        return [t for t in tokens if len(t) > 2]
    
    def _load_reranker(self):
        """Lazy-load the cross-encoder model."""
        if self._reranker_loaded or not CROSS_ENCODER_AVAILABLE:
            return
        
        try:
            logger.info(f"Loading reranker model: {self.reranker_model}")
            self.reranker = CrossEncoder(self.reranker_model)
            self._reranker_loaded = True
            logger.info("Reranker loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            self.use_reranker = False
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 20,
        rerank_candidates: int = 15
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            fetch_k: Number of candidates to fetch from each source
            rerank_candidates: Number of candidates to rerank
            
        Returns:
            List of RetrievedDocument objects, sorted by final score
        """
        # Step 1: Get vector search results
        vector_results = self._vector_search(query, fetch_k)
        logger.debug(f"Vector search returned {len(vector_results)} results")
        
        # Step 2: Get BM25 results (if enabled)
        bm25_results = []
        if self.use_bm25 and self.bm25:
            bm25_results = self._bm25_search(query, fetch_k)
            logger.debug(f"BM25 search returned {len(bm25_results)} results")
        
        # Step 3: Fuse results using Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            vector_results,
            bm25_results
        )
        logger.debug(f"Fusion produced {len(fused_results)} candidates")
        
        # Step 4: Rerank top candidates
        if self.use_reranker and len(fused_results) > 0:
            self._load_reranker()
            if self.reranker:
                candidates = fused_results[:rerank_candidates]
                reranked = self._rerank(query, candidates)
                # Merge reranked with remaining
                remaining = fused_results[rerank_candidates:]
                fused_results = reranked + remaining
        
        # Return top_k results
        return fused_results[:top_k]
    
    def _vector_search(
        self,
        query: str,
        k: int
    ) -> List[RetrievedDocument]:
        """Perform vector similarity search."""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            documents = []
            for doc, score in results:
                documents.append(RetrievedDocument(
                    doc_id=str(doc.metadata.get('chunk_id', id(doc))),
                    content=doc.page_content,
                    metadata=doc.metadata,
                    vector_score=float(score)
                ))
            
            return documents
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _bm25_search(
        self,
        query: str,
        k: int
    ) -> List[RetrievedDocument]:
        """Perform BM25 keyword search."""
        if not self.bm25:
            return []
        
        try:
            query_tokens = self._tokenize(query)
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top k indices
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:k]
            
            documents = []
            for idx in top_indices:
                if scores[idx] > 0:
                    documents.append(RetrievedDocument(
                        doc_id=str(self.bm25_doc_ids[idx]),
                        content=self.bm25_documents[idx],
                        metadata={'chunk_id': self.bm25_doc_ids[idx]},
                        bm25_score=float(scores[idx])
                    ))
            
            return documents
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievedDocument],
        bm25_results: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each result list
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            
        Returns:
            Fused and sorted list of documents
        """
        scores: Dict[str, float] = {}
        documents: Dict[str, RetrievedDocument] = {}
        
        # Score vector results
        for rank, doc in enumerate(vector_results, 1):
            doc_id = doc.doc_id
            rrf_score = self.vector_weight * (1 / (self.rrf_k + rank))
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            
            if doc_id not in documents:
                documents[doc_id] = doc
            else:
                documents[doc_id].vector_score = doc.vector_score
        
        # Score BM25 results
        for rank, doc in enumerate(bm25_results, 1):
            doc_id = doc.doc_id
            rrf_score = self.bm25_weight * (1 / (self.rrf_k + rank))
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            
            if doc_id not in documents:
                documents[doc_id] = doc
            documents[doc_id].bm25_score = doc.bm25_score
        
        # Update fusion scores and sort
        result_docs = []
        for doc_id, score in scores.items():
            doc = documents[doc_id]
            doc.fusion_score = score
            doc.final_score = score
            result_docs.append(doc)
        
        # Sort by fusion score
        result_docs.sort(key=lambda x: x.fusion_score, reverse=True)
        
        return result_docs
    
    def _rerank(
        self,
        query: str,
        candidates: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Search query
            candidates: List of candidate documents
            
        Returns:
            Reranked documents
        """
        if not self.reranker or not candidates:
            return candidates
        
        try:
            # Create query-document pairs
            pairs = [[query, doc.content] for doc in candidates]
            
            # Get reranker scores
            scores = self.reranker.predict(pairs)
            
            # Update scores
            for doc, score in zip(candidates, scores):
                doc.rerank_score = float(score)
                # Combine with fusion score
                doc.final_score = 0.7 * doc.rerank_score + 0.3 * doc.fusion_score
            
            # Sort by final score
            candidates.sort(key=lambda x: x.final_score, reverse=True)
            
            return candidates
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return candidates


# =============================================================================
# Factory Functions
# =============================================================================

def create_hybrid_retriever(
    vector_store,
    documents: Optional[List[Dict[str, Any]]] = None,
    use_bm25: bool = True,
    use_reranker: bool = True
) -> HybridRetriever:
    """
    Create a hybrid retriever instance.
    
    Args:
        vector_store: FAISS vector store
        documents: Documents for BM25 index
        use_bm25: Enable BM25 search
        use_reranker: Enable cross-encoder reranking
        
    Returns:
        HybridRetriever instance
    """
    return HybridRetriever(
        vector_store=vector_store,
        documents=documents,
        use_bm25=use_bm25,
        use_reranker=use_reranker
    )


def retrieve_with_reranking(
    query: str,
    vector_store,
    documents: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Convenience function for one-off retrieval with reranking.
    
    Args:
        query: Search query
        vector_store: FAISS vector store
        documents: Documents for BM25 (optional)
        top_k: Number of results
        
    Returns:
        List of result dictionaries
    """
    retriever = create_hybrid_retriever(
        vector_store,
        documents,
        use_bm25=BM25_AVAILABLE,
        use_reranker=CROSS_ENCODER_AVAILABLE
    )
    
    results = retriever.retrieve(query, top_k=top_k)
    
    return [
        {
            'content': r.content,
            'metadata': r.metadata,
            'score': r.final_score,
            'vector_score': r.vector_score,
            'bm25_score': r.bm25_score,
            'rerank_score': r.rerank_score
        }
        for r in results
    ]
