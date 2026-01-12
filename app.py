"""
PolicyMind AI - Insurance Policy Assistant

AI-powered insurance policy analysis that helps you understand your policy
in simple, easy-to-understand language.
"""

import os
import logging
import tempfile
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="PolicyMind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import RAG components
from rag.rag_index import get_or_create_vector_store
from rag.query_engine import get_rag_response, get_suggested_questions
from rag.document_loader import process_document, is_insurance_document
from rag.query_filter import is_query_relevant, get_suggested_policy_questions
from rag.model_utils import get_llm_instance, get_embedding_instance, ModelProvider

# Fixed configuration
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


# =============================================================================
# Session State
# =============================================================================

def init_session_state():
    """Initialize session state."""
    defaults = {
        'conversation': [],
        'uploaded_files': [],
        'processed_documents': [],
        'vector_store': None,
        'suggested_questions': [],
        'last_retrieval_info': None,
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


init_session_state()


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.title("🛡️ PolicyMind AI")
        st.caption("Your friendly insurance policy assistant")
        
        st.markdown("---")
        
        # Document upload section
        st.subheader("📄 Upload Your Policy")
        
        uploaded_files = st.file_uploader(
            "Upload insurance policy PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload your insurance policy PDF to get started",
            key="file_uploader"
        )
        
        if uploaded_files:
            _process_uploaded_files(uploaded_files)
        
        # Retrieval Metrics (only if we have data)
        if st.session_state.last_retrieval_info:
            st.markdown("---")
            st.subheader("📊 Last Query Metrics")
            
            info = st.session_state.last_retrieval_info
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chunks", info.get('num_retrieved', 0))
            with col2:
                st.metric("Tokens", info.get('context_tokens', 0))
            
            features = []
            if info.get('used_hybrid'):
                features.append("✅ Hybrid")
            if info.get('used_reranking'):
                features.append("✅ Rerank")
            if features:
                st.caption(" | ".join(features))
            
            if info.get('sources'):
                with st.expander("📄 Retrieved Context", expanded=False):
                    for i, source in enumerate(info['sources'][:3], 1):
                        score = source.get('score', 0)
                        st.markdown(f"**Chunk {i}** (Score: {score:.3f})")
                        st.code(source.get('preview', '')[:200], language=None)


def _process_uploaded_files(uploaded_files):
    """Process newly uploaded files with validation."""
    for uploaded_file in uploaded_files:
        # Skip if already processed
        if uploaded_file.name in [f['name'] for f in st.session_state.uploaded_files]:
            continue
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        try:
            # Check if it looks like an insurance document
            if not _is_policy_document(file_path):
                st.sidebar.error(
                    f"❌ **'{uploaded_file.name}'** doesn't appear to be an insurance policy.\n\n"
                    "Please upload an insurance policy document (health, life, auto, home insurance, etc.)"
                )
                os.unlink(file_path)  # Delete temp file
                continue
            
            st.session_state.uploaded_files.append({
                'name': uploaded_file.name,
                'path': file_path,
                'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            _update_vector_store()
            st.sidebar.success(f"✅ {uploaded_file.name} loaded!")
            
        except Exception as e:
            st.sidebar.error(f"Error with {uploaded_file.name}: {str(e)}")
            logger.error(f"File processing error: {str(e)}", exc_info=True)


def _is_policy_document(file_path: str) -> bool:
    """Check if the document appears to be an insurance policy."""
    try:
        # Quick check using existing function
        if is_insurance_document(file_path):
            return True
        
        # Additional check - read first page and look for keywords
        import fitz
        doc = fitz.open(file_path)
        if len(doc) == 0:
            return False
        
        # Get text from first few pages
        text = ""
        for page_num in range(min(3, len(doc))):
            text += doc[page_num].get_text().lower()
        doc.close()
        
        # Insurance document keywords
        insurance_keywords = [
            'insurance', 'policy', 'premium', 'coverage', 'insured', 'insurer',
            'claim', 'benefits', 'exclusion', 'sum assured', 'sum insured',
            'hospitalization', 'medical', 'health', 'life insurance', 'term plan',
            'policyholder', 'nominee', 'rider', 'indemnity', 'underwriting'
        ]
        
        # Count keyword matches
        matches = sum(1 for kw in insurance_keywords if kw in text)
        
        # Need at least 3 insurance-related keywords
        return matches >= 3
        
    except Exception as e:
        logger.warning(f"Could not validate document: {e}")
        return True  # Allow if we can't check


def _update_vector_store():
    """Update the vector store with documents."""
    if not st.session_state.uploaded_files:
        st.session_state.vector_store = None
        st.session_state.processed_documents = []
        st.session_state.suggested_questions = []
        return
    
    try:
        with st.spinner("📖 Reading your policy..."):
            embedding = get_embedding_instance(model_name=EMBEDDING_MODEL)
            
            all_documents = []
            all_docs_for_bm25 = []
            
            for file_info in st.session_state.uploaded_files:
                try:
                    docs = process_document(
                        file_path=file_info['path'],
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                        metadata={"source": file_info['name']},
                        use_semantic_chunking=True
                    )
                    
                    all_documents.extend(docs)
                    
                    for doc in docs:
                        all_docs_for_bm25.append({
                            'text': doc.page_content,
                            'page_content': doc.page_content,
                            'metadata': doc.metadata
                        })
                        
                except Exception as e:
                    st.sidebar.error(f"Error with {file_info['name']}: {str(e)}")
                    logger.error(f"Document error: {str(e)}", exc_info=True)
            
            if all_documents:
                st.session_state.processed_documents = all_docs_for_bm25
                _generate_suggested_questions(all_documents)
                
                doc_paths = [f['path'] for f in st.session_state.uploaded_files]
                st.session_state.vector_store = get_or_create_vector_store(
                    documents=all_documents,
                    embeddings=embedding,
                    doc_paths=doc_paths
                )
                
    except Exception as e:
        st.sidebar.error(f"Error processing: {str(e)}")
        logger.error(f"Vector store error: {str(e)}", exc_info=True)


def _generate_suggested_questions(documents):
    """Generate suggested questions."""
    try:
        combined_text = "\n\n".join(doc.page_content for doc in documents[:8])
        llm = get_llm_instance(ModelProvider.GROQ, LLM_MODEL)
        st.session_state.suggested_questions = get_suggested_questions(combined_text, llm)
    except Exception as e:
        logger.warning(f"Could not generate questions: {str(e)}")
        st.session_state.suggested_questions = get_suggested_policy_questions()[:4]


# =============================================================================
# Chat Interface
# =============================================================================

def render_chat():
    """Render the main chat interface."""
    # Welcome screen if no documents
    if not st.session_state.vector_store:
        st.markdown("""
        # 🛡️ Welcome to PolicyMind AI
        
        I'm your friendly insurance policy assistant. I help you understand your policy in **simple, plain language**.
        
        ### Getting Started
        1. **Upload your insurance policy PDF** using the sidebar
        2. **Ask me anything** about your coverage
        
        I'll explain everything in easy-to-understand terms - no insurance jargon! 🙂
        """)
        return
    
    st.markdown("## 🛡️ PolicyMind AI")
    st.caption("Ask me anything about your policy - I'll explain it simply!")
    
    # Display conversation
    for message in st.session_state.conversation:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🛡️"):
            st.markdown(message["content"])
    
    # Show suggested questions if no conversation
    if not st.session_state.conversation:
        _display_suggested_questions()
    
    # Chat input
    if prompt := st.chat_input("Ask about your policy..."):
        _process_question(prompt)


def _display_suggested_questions():
    """Display suggested questions."""
    if not st.session_state.suggested_questions:
        return
    
    st.markdown("### 💡 Try asking:")
    cols = st.columns(2)
    
    for i, question in enumerate(st.session_state.suggested_questions[:4]):
        with cols[i % 2]:
            if st.button(f"❓ {question}", key=f"q_{i}", use_container_width=True):
                _process_question(question)
                st.rerun()


def _is_greeting(text: str) -> bool:
    """Check if the text is a greeting."""
    greetings = {'hi', 'hello', 'hey', 'hii', 'hiii', 'helo', 'hallo', 
                 'good morning', 'good afternoon', 'good evening', 'howdy'}
    text_lower = text.lower().strip().rstrip('!.,?')
    return text_lower in greetings or text_lower.startswith(('hi ', 'hello ', 'hey '))


def _process_question(question: str):
    """Process a user question."""
    if not st.session_state.vector_store:
        st.error("Please upload a policy document first.")
        return
    
    # Add user message
    st.session_state.conversation.append({"role": "user", "content": question})
    
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)
    
    # Handle greetings specially
    if _is_greeting(question):
        greeting_response = (
            "Hello! 👋 I'm ready to help you understand your insurance policy.\n\n"
            "You can ask me questions like:\n"
            "- \"What's covered in my policy?\"\n"
            "- \"Is hospitalization covered?\"\n"
            "- \"What are the exclusions?\"\n"
            "- \"How do I file a claim?\"\n\n"
            "What would you like to know about your policy?"
        )
        st.session_state.conversation.append({
            "role": "assistant",
            "content": greeting_response
        })
        with st.chat_message("assistant", avatar="🛡️"):
            st.markdown(greeting_response)
        return
    
    # Check relevance for non-greetings
    try:
        embedding = get_embedding_instance(model_name=EMBEDDING_MODEL)
        is_relevant, rejection_message = is_query_relevant(question, embeddings_model=embedding)
    except Exception:
        is_relevant, rejection_message = is_query_relevant(question)
    
    if not is_relevant:
        friendly_rejection = (
            "I'm here to help with your insurance policy! 🛡️\n\n"
            "Try asking questions like:\n"
            "- \"What's covered?\"\n"
            "- \"Is surgery covered?\"\n"
            "- \"What are the claim limits?\"\n\n"
            "What would you like to know about your policy?"
        )
        st.session_state.conversation.append({
            "role": "assistant",
            "content": friendly_rejection
        })
        with st.chat_message("assistant", avatar="🛡️"):
            st.markdown(friendly_rejection)
        return
    
    # Get response
    try:
        llm = get_llm_instance(ModelProvider.GROQ, LLM_MODEL)
        
        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner("Let me check your policy..."):
                response = get_rag_response(
                    question,
                    st.session_state.vector_store,
                    llm,
                    documents=st.session_state.processed_documents,
                    k=5,
                    check_relevance=False,
                    use_hybrid=True,
                    use_reranking=True
                )
                
                answer = response["answer"]
                
                # Store retrieval metrics
                retrieval_info = response.get("retrieval_info", {})
                retrieval_info['sources'] = response.get("sources", [])
                st.session_state.last_retrieval_info = retrieval_info
                
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": answer
                })
                
                st.markdown(answer)
                
    except Exception as e:
        error_msg = "I had trouble looking that up. Could you try asking in a different way?"
        st.error(error_msg)
        logger.error(f"Query error: {str(e)}", exc_info=True)
        
        st.session_state.conversation.append({
            "role": "assistant",
            "content": error_msg
        })


# =============================================================================
# Main
# =============================================================================

def main():
    """Main application."""
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
