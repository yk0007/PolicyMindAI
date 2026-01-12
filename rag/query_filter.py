"""
Query Relevance Filter for PolicyMindAI.

Hybrid approach combining:
1. Fast keyword matching (for obvious cases)
2. Embedding-based similarity (for semantic understanding)
3. Off-topic pattern detection

This ensures the system only responds to policy-related questions.
"""

import re
import logging
import numpy as np
from typing import Tuple, Optional, List, Any
from enum import Enum

logger = logging.getLogger(__name__)


class QueryRelevance(Enum):
    """Classification of query relevance."""
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    OFF_TOPIC = "off_topic"


# =============================================================================
# Reference Examples for Embedding-Based Classification
# =============================================================================

# Policy-related question examples (for semantic similarity)
POLICY_QUESTION_EXAMPLES = [
    # Coverage questions
    "What is covered under this policy?",
    "Is surgery covered in my plan?",
    "Does this policy cover hospitalization?",
    "Are pre-existing conditions covered?",
    "Is maternity covered?",
    "What medical expenses are included?",
    
    # Exclusion questions
    "What is not covered by this policy?",
    "What are the exclusions?",
    "Is cosmetic surgery excluded?",
    "Are dental treatments excluded?",
    
    # Benefit questions
    "What is the sum insured?",
    "What are the policy benefits?",
    "How much is the coverage limit?",
    "What is the maximum claim amount?",
    
    # Terms questions
    "What is the waiting period?",
    "When does coverage start?",
    "What is the policy term?",
    "What happens if I miss a premium?",
    
    # Claims questions
    "How do I file a claim?",
    "What documents are needed for claims?",
    "How long does claim processing take?",
    "Is cashless treatment available?",
    
    # General policy questions
    "Can you explain this policy?",
    "What does this insurance cover?",
    "Tell me about my coverage",
    "What are the terms and conditions?",
]

# Off-topic question examples (things we should reject)
OFF_TOPIC_EXAMPLES = [
    # General knowledge
    "What is the capital of France?",
    "Who is the president of USA?",
    "What is the weather today?",
    "How far is the moon?",
    
    # Entertainment
    "Tell me a joke",
    "Recommend a good movie",
    "What's the latest news?",
    "Who won the game yesterday?",
    
    # Technical
    "Write me Python code",
    "How do I fix my computer?",
    "Explain blockchain",
    "What is machine learning?",
    
    # Personal
    "How are you?",
    "What's your name?",
    "What can you do?",
    "Hello, how's it going?",
    
    # Food/Lifestyle
    "Best recipe for pasta",
    "Good restaurants nearby",
    "How to lose weight",
    "What's a healthy diet?",
]


# =============================================================================
# Keyword-Based Matching (Fast Path)
# =============================================================================

POLICY_KEYWORDS = frozenset({
    'insurance', 'policy', 'premium', 'coverage', 'benefit', 'claim', 'claims',
    'deductible', 'copay', 'copayment', 'coinsurance', 'exclusion', 'exclusions',
    'rider', 'endorsement', 'underwriting', 'sum insured', 'insured', 'insurer',
    'policyholder', 'beneficiary', 'nominee', 'hospitalization', 'outpatient',
    'inpatient', 'surgical', 'medical', 'maternity', 'waiting period', 'covered',
    'not covered', 'included', 'excluded', 'eligible', 'reimbursement', 'cashless',
    'pre-existing', 'preexisting', 'hospital', 'treatment', 'surgery', 'procedure',
})

OFF_TOPIC_PATTERNS = [
    r'\b(weather|temperature|forecast)\b',
    r'\b(recipe|cook|food|restaurant)\b',
    r'\b(movie|film|actor|actress|celebrity)\b',
    r'\b(sports?|game|match|score|team|player)\b',
    r'\b(music|song|album|artist|singer)\b',
    r'\b(joke|funny|laugh|humor)\b',
    r'\b(capital of|president of|who is the)\b(?!.*insur)',
    r'\b(translate|translation)\b',
    r'\b(write me a|compose|create a story)\b',
    r'\b(code|programming|python|javascript)\b',
    r'^(hello|hi|hey|good morning|good evening)\s*[.!?]?$',
    r'^(thanks?|thank you|ok|okay|bye|goodbye)\s*[.!?]?$',
]


# =============================================================================
# Embedding-Based Classifier
# =============================================================================

class EmbeddingQueryClassifier:
    """
    Classifies queries using embedding similarity.
    
    Compares query embedding to reference policy and off-topic examples.
    """
    
    def __init__(self, embeddings_model: Any = None):
        """
        Initialize the classifier.
        
        Args:
            embeddings_model: HuggingFace embeddings instance
        """
        self.embeddings = embeddings_model
        self._policy_embeddings = None
        self._off_topic_embeddings = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of reference embeddings."""
        if self._initialized or self.embeddings is None:
            return
        
        try:
            logger.info("Initializing embedding-based query classifier...")
            
            # Embed policy examples
            self._policy_embeddings = self.embeddings.embed_documents(
                POLICY_QUESTION_EXAMPLES
            )
            
            # Embed off-topic examples
            self._off_topic_embeddings = self.embeddings.embed_documents(
                OFF_TOPIC_EXAMPLES
            )
            
            # Convert to numpy for efficient computation
            self._policy_embeddings = np.array(self._policy_embeddings)
            self._off_topic_embeddings = np.array(self._off_topic_embeddings)
            
            self._initialized = True
            logger.info("Query classifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize query classifier: {e}")
            self._initialized = False
    
    def classify(self, query: str) -> Tuple[QueryRelevance, float, str]:
        """
        Classify a query using embedding similarity.
        
        Args:
            query: The user's question
            
        Returns:
            Tuple of (relevance, confidence, explanation)
        """
        if not self._initialized:
            self._initialize()
        
        if not self._initialized or self.embeddings is None:
            # Fall back to keyword-based
            return _keyword_based_classify(query)
        
        try:
            # Embed the query
            query_embedding = np.array(self.embeddings.embed_query(query))
            
            # Calculate similarity to policy examples
            policy_similarities = self._cosine_similarity(
                query_embedding, 
                self._policy_embeddings
            )
            max_policy_sim = float(np.max(policy_similarities))
            avg_policy_sim = float(np.mean(policy_similarities))
            
            # Calculate similarity to off-topic examples
            off_topic_similarities = self._cosine_similarity(
                query_embedding,
                self._off_topic_embeddings
            )
            max_off_topic_sim = float(np.max(off_topic_similarities))
            avg_off_topic_sim = float(np.mean(off_topic_similarities))
            
            # Decision logic
            policy_score = (max_policy_sim * 0.7 + avg_policy_sim * 0.3)
            off_topic_score = (max_off_topic_sim * 0.7 + avg_off_topic_sim * 0.3)
            
            # Calculate relative confidence
            score_diff = policy_score - off_topic_score
            
            if score_diff > 0.1:
                # Clearly policy-related
                return (
                    QueryRelevance.RELEVANT,
                    min(policy_score, 1.0),
                    "Query is relevant to policy documents."
                )
            elif score_diff > 0.0:
                # Somewhat policy-related
                return (
                    QueryRelevance.PARTIALLY_RELEVANT,
                    policy_score,
                    "Query may be related to policy documents."
                )
            else:
                # Off-topic
                return (
                    QueryRelevance.OFF_TOPIC,
                    1.0 - policy_score,
                    get_rejection_message(query)
                )
                
        except Exception as e:
            logger.warning(f"Embedding classification failed: {e}")
            return _keyword_based_classify(query)
    
    def _cosine_similarity(
        self, 
        query_vec: np.ndarray, 
        ref_vecs: np.ndarray
    ) -> np.ndarray:
        """Calculate cosine similarity between query and reference vectors."""
        # Normalize
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        ref_norms = ref_vecs / (np.linalg.norm(ref_vecs, axis=1, keepdims=True) + 1e-9)
        
        # Dot product for cosine similarity
        return np.dot(ref_norms, query_norm)


# Global classifier instance (lazy initialization)
_embedding_classifier: Optional[EmbeddingQueryClassifier] = None


def get_embedding_classifier(embeddings_model: Any = None) -> EmbeddingQueryClassifier:
    """Get or create the embedding classifier."""
    global _embedding_classifier
    
    if _embedding_classifier is None:
        _embedding_classifier = EmbeddingQueryClassifier(embeddings_model)
    elif embeddings_model is not None and _embedding_classifier.embeddings is None:
        _embedding_classifier.embeddings = embeddings_model
    
    return _embedding_classifier


# =============================================================================
# Keyword-Based Classification (Fallback)
# =============================================================================

def _keyword_based_classify(query: str) -> Tuple[QueryRelevance, float, str]:
    """Fallback keyword-based classification."""
    if not query or not query.strip():
        return (
            QueryRelevance.OFF_TOPIC,
            1.0,
            "Please enter a question about your policy document."
        )
    
    query_lower = query.lower().strip()
    
    # Check off-topic patterns
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return (
                QueryRelevance.OFF_TOPIC,
                0.9,
                get_rejection_message(query)
            )
    
    # Calculate keyword score
    words = set(re.findall(r'\b\w+\b', query_lower))
    
    matching_keywords = 0
    for keyword in POLICY_KEYWORDS:
        if ' ' in keyword:
            if keyword in query_lower:
                matching_keywords += 2
        elif keyword in words:
            matching_keywords += 1
    
    word_count = max(len(words), 1)
    score = min(matching_keywords / word_count, 1.0)
    
    if score >= 0.3:
        return (
            QueryRelevance.RELEVANT,
            score,
            "Query is relevant to policy documents."
        )
    elif score >= 0.15:
        return (
            QueryRelevance.PARTIALLY_RELEVANT,
            score,
            "Query may be related to policy documents."
        )
    else:
        return (
            QueryRelevance.OFF_TOPIC,
            1.0 - score,
            get_rejection_message(query)
        )


# =============================================================================
# Main Classification Function
# =============================================================================

def classify_query(
    query: str,
    relevance_threshold: float = 0.15,
    embeddings_model: Any = None,
    use_embeddings: bool = True
) -> Tuple[QueryRelevance, float, str]:
    """
    Classify whether a query is relevant to policy documents.
    
    Uses a hybrid approach:
    1. Fast keyword check for obvious cases
    2. Embedding similarity for semantic understanding
    
    Args:
        query: The user's question
        relevance_threshold: Minimum score for relevance
        embeddings_model: Optional embeddings model for semantic classification
        use_embeddings: Whether to use embedding-based classification
        
    Returns:
        Tuple of (relevance classification, confidence score, explanation)
    """
    if not query or not query.strip():
        return (
            QueryRelevance.OFF_TOPIC,
            1.0,
            "Please enter a question about your policy document."
        )
    
    query = query.strip()
    
    # Quick check for obvious off-topic patterns
    query_lower = query.lower()
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return (
                QueryRelevance.OFF_TOPIC,
                0.95,
                get_rejection_message(query)
            )
    
    # Quick check for obvious policy keywords
    words = set(re.findall(r'\b\w+\b', query_lower))
    strong_policy_keywords = {'coverage', 'covered', 'policy', 'claim', 'insurance', 
                              'premium', 'benefit', 'exclusion', 'insured'}
    if words & strong_policy_keywords:
        return (
            QueryRelevance.RELEVANT,
            0.9,
            "Query is relevant to policy documents."
        )
    
    # Use embedding-based classification for ambiguous cases
    if use_embeddings and embeddings_model is not None:
        classifier = get_embedding_classifier(embeddings_model)
        return classifier.classify(query)
    
    # Fall back to keyword-based
    return _keyword_based_classify(query)


# =============================================================================
# Helper Functions
# =============================================================================

def get_rejection_message(query: str) -> str:
    """Generate a helpful rejection message for off-topic queries."""
    return (
        "I'm designed to help you understand your insurance policy documents. "
        "I can answer questions about:\n\n"
        "• **Coverage** - What's covered and what's not\n"
        "• **Benefits** - Policy benefits and limits\n"
        "• **Claims** - How to file claims and procedures\n"
        "• **Terms** - Policy terms, conditions, and definitions\n"
        "• **Exclusions** - What's excluded from coverage\n"
        "• **Costs** - Premiums, deductibles, and co-payments\n\n"
        "Please ask a question related to your uploaded policy documents."
    )


def get_suggested_policy_questions() -> List[str]:
    """Get example policy-related questions."""
    return [
        "What is covered under this policy?",
        "What are the exclusions in my policy?",
        "How do I file a claim?",
        "What is the waiting period for pre-existing conditions?",
        "What is my sum insured?",
        "Is hospitalization covered?",
        "What are the policy terms and conditions?",
        "What happens if I miss a premium payment?",
    ]


def is_query_relevant(
    query: str,
    threshold: float = 0.15,
    embeddings_model: Any = None
) -> Tuple[bool, str]:
    """
    Simple interface to check if a query is relevant.
    
    Args:
        query: The user's question
        threshold: Relevance threshold
        embeddings_model: Optional embeddings for semantic classification
        
    Returns:
        Tuple of (is_relevant, message)
    """
    relevance, score, message = classify_query(
        query, 
        threshold,
        embeddings_model=embeddings_model,
        use_embeddings=embeddings_model is not None
    )
    
    is_relevant = relevance in (QueryRelevance.RELEVANT, QueryRelevance.PARTIALLY_RELEVANT)
    
    return is_relevant, message
