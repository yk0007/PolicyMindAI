import os
import re
import time
import streamlit as st
from dotenv import load_dotenv
import tempfile
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
from enum import Enum

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="PolicyMind RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import necessary components after setting page config to avoid Streamlit errors
from rag.rag_index import get_or_create_vector_store
from rag.query_engine import get_rag_response, get_suggested_questions
from rag.document_loader import process_document, is_insurance_document
from rag.model_utils import (
    get_available_models,
    set_api_keys,
    get_llm_instance,
    get_embedding_instance,
    ModelProvider,
    AVAILABLE_EMBEDDING_MODELS,
    DEFAULT_API_KEYS,
    MODEL_DISPLAY_NAMES
)

# Initialize session state variables
def init_session_state():
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'model_provider' not in st.session_state:
        st.session_state.model_provider = ModelProvider.GROQ  # Default to Groq
    if 'model_provider_str' not in st.session_state:
        st.session_state.model_provider_str = ModelProvider.GROQ.value  # String representation
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {**DEFAULT_API_KEYS}  # Start with default keys
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "llama3-70b-8192"  # Default model
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = "BAAI/bge-base-en-v1.5"  # Default embedding model
    if 'suggested_questions' not in st.session_state:
        st.session_state.suggested_questions = []

# Initialize session state
init_session_state()

# Sidebar for model selection and API keys
def sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Model selection
        st.subheader("Model Configuration")
        
        # Provider selection with display names
        provider_options = [
            (provider, MODEL_DISPLAY_NAMES.get(provider, provider.value)) 
            for provider in [ModelProvider.GROQ, ModelProvider.OLLAMA, ModelProvider.GOOGLE]
        ]
        
        # Create a simple dropdown for provider selection
        provider_options_display = [d for _, d in provider_options]
        
        # Get current provider string and convert to display name
        current_provider_str = st.session_state.model_provider_str
        current_provider = ModelProvider(current_provider_str)
        current_provider_display = MODEL_DISPLAY_NAMES.get(current_provider, current_provider_str)
        
        # Find the index of the current provider in the display options
        try:
            current_index = provider_options_display.index(current_provider_display)
        except ValueError:
            current_index = 0
        
        # Create the selectbox with display names
        selected_provider_display = st.selectbox(
            "Select Model Provider",
            provider_options_display,
            index=current_index,
            key="model_provider_selector"
        )
        
        # Get the provider enum from the display name
        selected_provider = next((p for p, d in provider_options if d == selected_provider_display), provider_options[0][0])
        
        # Update session state if provider changed
        if st.session_state.model_provider_str != selected_provider.value:
            st.session_state.model_provider = selected_provider
            st.session_state.model_provider_str = selected_provider.value
            # Reset the selected model when provider changes
            models = get_available_models(selected_provider)
            st.session_state.selected_model = models[0] if models else ""
        
        # Get available models for the selected provider
        available_models = get_available_models(selected_provider)
        
        if available_models:
            # Find the current model in the available models, default to first
            current_model = st.session_state.selected_model
            if current_model not in available_models and available_models:
                current_model = available_models[0]
            
            selected_model = st.selectbox(
                "Select Model",
                available_models,
                index=available_models.index(current_model) if current_model in available_models else 0,
                key=f"model_selector_{selected_provider.value}"
            )
            st.session_state.selected_model = selected_model
        
        # Embedding model selection
        st.subheader("Embedding Model")
        embedding_model = st.selectbox(
            "Select Embedding Model",
            AVAILABLE_EMBEDDING_MODELS,
            index=AVAILABLE_EMBEDDING_MODELS.index(st.session_state.embedding_model) 
            if st.session_state.embedding_model in AVAILABLE_EMBEDDING_MODELS else 0,
            key="embedding_model_selector"
        )
        st.session_state.embedding_model = embedding_model
        
        # API keys are loaded from environment variables or Streamlit secrets
        # No API key input in the frontend for security
        if selected_provider in [ModelProvider.GROQ, ModelProvider.GOOGLE]:
            # Check if API keys are available in environment variables or secrets
            if not os.environ.get("GROQ_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
                if hasattr(st, 'secrets') and hasattr(st.secrets, 'secrets'):
                    if not (st.secrets.secrets.get("GROQ_API_KEY") or st.secrets.secrets.get("GOOGLE_API_KEY")):
                        st.warning("ℹ️ API keys must be set via environment variables or Streamlit secrets.")
        
        # Ollama settings (local, no API key needed)
        elif selected_provider == ModelProvider.OLLAMA:
            st.info("Ollama runs locally. Make sure the Ollama server is running.")
            ollama_model = st.text_input(
                "Ollama Model Name",
                value=st.session_state.selected_model or "llama3"
            )
            if ollama_model:
                st.session_state.selected_model = ollama_model
        
        # Document upload
        st.subheader("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            process_uploaded_files(uploaded_files)
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.subheader("Uploaded Files")
            for i, file in enumerate(st.session_state.uploaded_files):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"📄 {file['name']}")
                with col2:
                    if st.button("❌", key=f"remove_{i}"):
                        remove_file(i)

# Process uploaded files
def process_uploaded_files(uploaded_files):
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in [f['name'] for f in st.session_state.uploaded_files]:
            # Save the file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
            
            # Process the file
            try:
                # Add to uploaded files list
                st.session_state.uploaded_files.append({
                    'name': uploaded_file.name,
                    'path': file_path,
                    'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Update the vector store with the new document
                update_vector_store()
                
                st.sidebar.success(f"Processed {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")

# Remove a file
def remove_file(file_index):
    if 0 <= file_index < len(st.session_state.uploaded_files):
        removed_file = st.session_state.uploaded_files.pop(file_index)
        # Update the vector store
        update_vector_store()
        st.sidebar.success(f"Removed {removed_file['name']}")
        # Rerun to update the UI
        st.rerun()

# Update the vector store with all uploaded files
def update_vector_store():
    if not st.session_state.uploaded_files:
        st.session_state.vector_store = None
        st.session_state.suggested_questions = []
        return
    
    try:
        with st.spinner("Processing documents..."):
            # Get embedding instance based on selected provider and model
            embedding = get_embedding_instance(
                model_name=st.session_state.embedding_model,
                provider=st.session_state.model_provider,
                api_keys=st.session_state.api_keys
            )
            
            # Process all uploaded files
            documents = []
            for file_info in st.session_state.uploaded_files:
                try:
                    # Process the document using rag.document_loader
                    docs = process_document(
                        file_path=file_info['path'],
                        metadata={"source": file_info['name']}
                    )
                    
                    # Check if the document is insurance-related using the first chunk
                    if docs and not is_insurance_document(docs[0].page_content):
                        st.sidebar.warning(f"Document {file_info['name']} may not be insurance-related. Some features may not work as expected.")
                    
                    documents.extend(docs)
                except Exception as e:
                    st.sidebar.error(f"Error processing {file_info['name']}: {str(e)}")
            
            if documents:
                # Combine all document text for suggested questions
                combined_text = "\n\n".join(doc.page_content for doc in documents[:10])  # Limit to first 10 chunks
                
                # Generate suggested questions
                try:
                    llm = get_llm_instance(
                        st.session_state.model_provider,
                        st.session_state.selected_model,
                        st.session_state.api_keys
                    )
                    st.session_state.suggested_questions = get_suggested_questions(combined_text, llm)
                except Exception as e:
                    st.sidebar.warning(f"Could not generate suggested questions: {str(e)}")
                    st.session_state.suggested_questions = []
                
                # Create or update the vector store
                doc_paths = [f['path'] for f in st.session_state.uploaded_files]
                st.session_state.vector_store = get_or_create_vector_store(
                    documents=documents,
                    embeddings=embedding,
                    doc_paths=doc_paths
                )
                
                st.sidebar.success("Document index updated successfully!")
    except Exception as e:
        st.sidebar.error(f"Error updating document index: {str(e)}")
        st.session_state.vector_store = None

# Display suggested questions
def display_suggested_questions():
    if st.session_state.get('suggested_questions') and not st.session_state.get('processing_suggested_question', False):
        st.subheader("Suggested Questions")
        cols = st.columns(2)  # Create 2 columns for better layout
        
        for i, question in enumerate(st.session_state.suggested_questions):
            with cols[i % 2]:  # Alternate between columns
                if st.button(
                    question,
                    key=f"suggested_q_{i}",
                    use_container_width=True,
                    help="Click to ask this question"
                ):
                    # Set flag to prevent recursion
                    st.session_state.processing_suggested_question = True
                    # Set the question as the current prompt and process it
                    process_user_question(question)
                    # Clear the flag after processing
                    st.session_state.processing_suggested_question = False
                    # Force a rerun to show the processing
                    st.rerun()

def display_source(source, index):
    """Display a source with preview, metadata, and page numbers."""
    import time
    import hashlib
    from pathlib import Path
    
    # Get source attributes with fallbacks
    source_path = source.get('path', '')
    source_name = Path(source_path).name if source_path else 'unknown'
    page_num = source.get('page', 0)
    
    # Create a unique key for the source
    content_hash = hashlib.md5(str(source).encode()).hexdigest()[:8]
    source_key = f"src_{index}_{source_name[:15]}_pg{page_num}_{content_hash}"
    source_key = ''.join(c if c.isalnum() or c in '_-' else '_' for c in source_key)
    
    # Format the expander title with page number if available
    expander_title = f"🔍 Source {index + 1}: {source_name}"
    if page_num and page_num != 'N/A':
        expander_title += f" (Page {page_num})"
    
    with st.expander(expander_title, expanded=False):
        col1, col2 = st.columns([1, 4])
        
        with col1:
            # Show relevance score if available
            if 'score' in source:
                st.metric("Relevance", f"{source['score']:.2f}")
            
            # Show page number prominently
            if page_num and page_num != 'N/A':
                st.metric("Page", page_num)
            
            # View in document button
            if source_path and os.path.exists(source_path):
                if st.button("View in Document", key=f"view_doc_{source_key}"):
                    st.session_state.current_document = source_path
                    st.session_state.current_page = page_num if page_num != 'N/A' else 1
                    st.rerun()
            
            # Show document metadata if available
            if 'metadata' in source and source['metadata']:
                st.caption("Metadata")
                for k, v in source['metadata'].items():
                    if k not in ['page', 'source', 'score'] and v:
                        st.text(f"{k}: {v}")
        
        with col2:
            # Show content preview with better formatting
            st.caption("Content Preview")
            
            # Get preview text with fallback
            preview = source.get('preview', '')
            if not preview and 'content' in source:
                preview = source['content']
            
            if not preview:
                st.warning("No preview available")
                return
            
            # Display the preview with syntax highlighting for code/structured content
            st.markdown("```\n" + preview[:500] + ("..." if len(preview) > 500 else "") + "\n```")
            
            # Show document path if available
            if source_path:
                st.caption(f"Source: {source_path}")

def process_user_question(question):
    """Process a user question and generate a response."""
    # Add user message to chat
    st.session_state.conversation.append({"role": "user", "content": question})
    
    with st.chat_message("user"):
        st.markdown(question)
    
    # Get LLM instance
    try:
        llm = get_llm_instance(
            st.session_state.model_provider,
            st.session_state.selected_model,
            st.session_state.api_keys
        )
        
        # Get response from RAG
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                response = get_rag_response(
                    question,
                    st.session_state.vector_store,
                    llm
                )
                
                # Display response
                response_placeholder = st.empty()
                
                # Get the raw answer text
                clean_answer = response["answer"]
                
                # Add to conversation with sources and used snippets
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": clean_answer,
                    "sources": response.get("sources", []),
                    "used_snippets": response.get("used_snippets", [])
                })
                
                # Display the response with preserved newlines
                st.markdown(clean_answer)
                
                # Show sources if available
                if "sources" in response and response["sources"]:
                    st.markdown("---")
                    st.markdown("### 📚 Sources")
                    st.caption("These are the sources used to generate the answer:")
                    
                    # Display each source with preview
                    for i, source in enumerate(response["sources"]):
                        display_source(source, i)
                    
                    # Add a note about LangChain RAG
                    st.info("This answer was generated using LangChain's Retrieval-Augmented Generation (RAG) system, which enhances responses with information from your documents.")
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        st.exception(e)  # Show full error for debugging

# Main chat interface
def chat_interface():
    st.title("📄 PolicyMind RAG")
    
    # Display document viewer if a document is selected
    if 'current_document' in st.session_state and st.session_state.current_document:
        st.sidebar.title("📑 Document Viewer")
        with st.sidebar.expander("Viewing Document", expanded=True):
            st.write(f"**File:** {os.path.basename(st.session_state.current_document)}")
            st.write(f"**Page:** {st.session_state.current_page if 'current_page' in st.session_state else 'N/A'}")
            
            # Display document content (simplified - could be enhanced with actual document viewer)
            try:
                with open(st.session_state.current_document, 'rb') as f:
                    # This is a placeholder - in a real app, you'd use a proper PDF viewer
                    st.warning("Document viewer would display here. For now, please refer to the source previews.")
            except Exception as e:
                st.error(f"Could not display document: {str(e)}")
            
            if st.button("Close Document"):
                del st.session_state.current_document
                if 'current_page' in st.session_state:
                    del st.session_state.current_page
                st.rerun()
    
    # Display conversation
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and message.get("sources"):
                st.markdown("---")
                st.markdown("### 📚 Sources")
                st.caption("These are the sources used to generate the answer:")
                
                # Display each source with preview
                for i, source in enumerate(message["sources"]):
                    display_source(source, i)
    
    # Display suggested questions if no conversation yet
    if not st.session_state.conversation and st.session_state.vector_store:
        display_suggested_questions()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents...", key="chat_input"):
        # Check if we have a vector store and a model selected
        if st.session_state.vector_store is None:
            st.error("Please upload and process at least one document first.")
            return
            
        if not st.session_state.selected_model or not st.session_state.model_provider:
            st.error("Please select a model and provide the necessary API key.")
            return
        
        # Process the user's question
        process_user_question(prompt)
        # Force a rerun to show the processing
        st.rerun()

# Main function
def main():
    # Sidebar
    sidebar()
    
    # Main content
    chat_interface()

if __name__ == "__main__":
    main()
