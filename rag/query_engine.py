"""
Query Engine for PolicyMindAI.

Handles RAG-based question answering with:
- Query relevance filtering
- Hybrid retrieval (BM25 + Vector)
- Cross-encoder reranking
- Human-friendly response formatting
"""

import os
import logging
import re
from typing import Dict, Any, List, Optional

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chat_models.base import BaseChatModel
from langchain.vectorstores.base import VectorStore

from .query_filter import is_query_relevant, get_suggested_policy_questions
from .retriever import HybridRetriever, create_hybrid_retriever, RetrievedDocument
from .chunker import estimate_tokens

logger = logging.getLogger(__name__)


# =============================================================================
# Human-Friendly Prompt Templates
# =============================================================================

POLICY_QA_TEMPLATE = """You are a friendly insurance policy assistant helping someone understand their policy in simple terms.

Based on the policy documents below, answer the user's question in a clear, easy-to-understand way.

POLICY CONTEXT:
{context}

USER'S QUESTION: {question}

YOUR RESPONSE SHOULD:
1. Start with a direct, clear answer in simple language
2. Explain any technical terms in plain English (like you're explaining to a friend)
3. Use bullet points for lists of items
4. If something is covered, say "Yes, this is covered" clearly
5. If something is NOT covered, say "No, this is not covered" clearly  
6. If there are conditions or limits, explain them simply
7. End with any important things they should know

AVOID:
- Using policy jargon without explaining it
- Just quoting the policy word-for-word
- Being overly formal or legal-sounding
- Saying "based on the context" or "according to the sources"

Write like you're a helpful friend who happens to be an insurance expert:

"""

COVERAGE_CHECK_TEMPLATE = """You are a friendly insurance advisor helping someone check if something is covered by their policy.

POLICY CONTEXT:
{context}

USER'S QUESTION: {question}

Give a clear, friendly answer:

1. **Start with YES or NO** - Is it covered?
2. **Explain simply** - What does the policy say about this?
3. **Important conditions** - Any waiting periods, limits, or requirements?
4. **What to do** - If they need to make a claim, what should they know?

Keep it conversational and easy to understand. Avoid policy jargon - explain like you're talking to a friend:

"""

GENERAL_INFO_TEMPLATE = """You are a friendly insurance policy assistant. Help the user understand their policy in simple terms.

POLICY CONTEXT:
{context}

USER'S QUESTION: {question}

Give a helpful, easy-to-understand answer:

- Use simple language, not legal jargon
- Organize information clearly with bullet points
- Explain technical terms if you use them
- Be conversational and friendly

Your response:

"""


# =============================================================================
# Query Classification
# =============================================================================

def _is_coverage_question(query: str) -> bool:
    """Check if query is asking about coverage."""
    patterns = [
        r'\b(is|are)\s+\w+\s+(covered|included|excluded)\b',
        r'\b(does|do)\s+.*\s+(cover|include)\b',
        r'\bcoverage\s+for\b',
        r'\b(covered|not covered)\b',
        r'\bcan i (get|claim)\b',
        r'\bwill (you|the policy|insurance) (pay|cover)\b',
    ]
    query_lower = query.lower()
    return any(re.search(p, query_lower) for p in patterns)


def _is_what_covered_question(query: str) -> bool:
    """Check if asking what is covered/not covered."""
    patterns = [
        r'\bwhat (is|are) (covered|not covered|excluded)\b',
        r'\bwhat does (this|the) policy cover\b',
        r'\blist .*(covered|benefits|exclusions)\b',
    ]
    query_lower = query.lower()
    return any(re.search(p, query_lower) for p in patterns)


# =============================================================================
# Context Packing
# =============================================================================

def _pack_context(
    documents: List[RetrievedDocument],
    max_tokens: int = 4000
) -> str:
    """Pack documents into context string with token limit."""
    if not documents:
        return ""
    
    context_parts = []
    current_tokens = 0
    
    for i, doc in enumerate(documents, 1):
        doc_text = doc.content
        doc_tokens = estimate_tokens(doc_text)
        
        if current_tokens + doc_tokens > max_tokens:
            if context_parts:
                break
            target_tokens = max(int(max_tokens * 0.75), 1)
            doc_text = _truncate_to_tokens(doc_text, target_tokens)
            doc_tokens = estimate_tokens(doc_text)
        
        context_parts.append(doc_text)
        current_tokens += doc_tokens
    
    return "\n\n---\n\n".join(context_parts)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to a token budget."""
    if not text or max_tokens <= 0:
        return ""
    
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoder.decode(tokens[:max_tokens]).strip()
    except Exception:
        words = text.split()
        if not words:
            return ""
        max_words = max(int(max_tokens / 1.3), 1)
        return " ".join(words[:max_words]).strip()


# =============================================================================
# Main Query Function
# =============================================================================

def get_rag_response(
    query: str,
    vector_store: VectorStore,
    llm: BaseChatModel,
    documents: Optional[List[Dict[str, Any]]] = None,
    k: int = 5,
    return_sources: bool = True,
    check_relevance: bool = True,
    use_hybrid: bool = True,
    use_reranking: bool = True,
    max_context_tokens: int = 4000
) -> Dict[str, Any]:
    """
    Get a response using advanced RAG with hybrid search and reranking.
    """
    # Check query relevance first
    if check_relevance:
        is_relevant, rejection_message = is_query_relevant(query)
        if not is_relevant:
            return {
                "answer": rejection_message,
                "sources": [],
                "is_relevant": False
            }
    
    try:
        # Step 1: Retrieve documents
        if use_hybrid and documents:
            retriever = create_hybrid_retriever(
                vector_store=vector_store,
                documents=documents,
                use_bm25=True,
                use_reranker=use_reranking
            )
            retrieved_docs = retriever.retrieve(query, top_k=k, fetch_k=k*3)
        else:
            try:
                results = vector_store.similarity_search_with_score(query, k=k)
                retrieved_docs = [
                    RetrievedDocument(
                        doc_id=str(doc.metadata.get('chunk_id', i)),
                        content=doc.page_content,
                        metadata=doc.metadata,
                        vector_score=float(score),
                        final_score=float(score)
                    )
                    for i, (doc, score) in enumerate(results)
                ]
            except Exception:
                results = vector_store.similarity_search(query, k=k)
                retrieved_docs = [
                    RetrievedDocument(
                        doc_id=str(doc.metadata.get('chunk_id', i)),
                        content=doc.page_content,
                        metadata=doc.metadata,
                        final_score=1.0 - (i * 0.1)
                    )
                    for i, doc in enumerate(results)
                ]
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find relevant information in your policy to answer this question. Could you try rephrasing it?",
                "sources": [],
                "is_relevant": True
            }
        
        # Step 2: Pack context
        context = _pack_context(retrieved_docs, max_context_tokens)
        
        # Step 3: Select appropriate template
        if _is_coverage_question(query):
            template = COVERAGE_CHECK_TEMPLATE
        elif _is_what_covered_question(query):
            template = POLICY_QA_TEMPLATE
        else:
            template = GENERAL_INFO_TEMPLATE
        
        # Step 4: Generate response
        prompt = template.format(context=context, question=query)
        
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up the answer
        answer = _clean_response(answer)
        
        # Step 5: Format sources (simplified)
        sources = _format_sources(retrieved_docs) if return_sources else []
        
        return {
            "answer": answer.strip(),
            "sources": sources,
            "is_relevant": True,
            "retrieval_info": {
                "num_retrieved": len(retrieved_docs),
                "used_hybrid": use_hybrid and documents is not None,
                "used_reranking": use_reranking,
                "context_tokens": estimate_tokens(context)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in get_rag_response: {e}", exc_info=True)
        return {
            "answer": "I encountered an issue while looking up your policy. Please try again.",
            "sources": [],
            "error": str(e)
        }


def _clean_response(answer: str) -> str:
    """Clean up LLM response for better readability."""
    # Remove common LLM artifacts
    answer = re.sub(r'\[Source \d+\]', '', answer)
    answer = re.sub(r'Based on the (provided |given )?context,?\s*', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'According to the (provided |given )?(sources?|context|documents?),?\s*', '', answer, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    
    return answer.strip()


def _format_sources(docs: List[RetrievedDocument]) -> List[Dict[str, Any]]:
    """Format sources for display with scores."""
    sources = []
    seen = set()
    
    for doc in docs[:5]:  # Top 5 sources
        source_path = doc.metadata.get("source", "Your policy document")
        source_name = os.path.basename(source_path) if source_path != "Your policy document" else source_path
        page = doc.metadata.get("page", "")
        
        source_id = f"{source_path}:{page}"
        if source_id in seen:
            continue
        seen.add(source_id)
        
        preview = doc.content[:300]
        if len(doc.content) > 300:
            preview += "..."
        
        sources.append({
            "name": source_name,
            "page": page,
            "preview": preview,
            "score": doc.final_score,
            "vector_score": doc.vector_score,
            "bm25_score": doc.bm25_score,
            "rerank_score": doc.rerank_score,
        })
    
    return sources


# =============================================================================
# Suggested Questions
# =============================================================================

def get_suggested_questions(
    context: str,
    llm: BaseChatModel,
    num_questions: int = 4
) -> List[str]:
    """Generate relevant suggested questions."""
    default_questions = get_suggested_policy_questions()[:num_questions]
    
    if not context or len(context) < 100:
        return default_questions
    
    try:
        max_context = 4000
        if len(context) > max_context:
            context = context[:max_context] + "..."
        
        prompt = f"""Based on this insurance policy, suggest {num_questions} simple questions a person might want to ask.

Focus on practical questions like:
- What is covered and not covered
- Claim process
- Important limits or waiting periods
- Benefits they can use

Policy excerpt:
{context}

Write {num_questions} short, simple questions (one per line):"""
        
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        questions = []
        for line in response_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'^[\d•\-\*]+[.)\s]+', '', line).strip()
            if line and len(line) > 10:
                if not line.endswith('?'):
                    line = line.rstrip('.') + '?'
                questions.append(line)
        
        while len(questions) < num_questions:
            for q in default_questions:
                if q not in questions:
                    questions.append(q)
                    if len(questions) >= num_questions:
                        break
        
        return questions[:num_questions]
        
    except Exception as e:
        logger.warning(f"Error generating questions: {e}")
        return default_questions
