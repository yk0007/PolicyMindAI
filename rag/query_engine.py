import os
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import BaseRetriever, Document
from langchain.chat_models.base import BaseChatModel
from langchain.vectorstores import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryEngine:
    """
    A class to handle querying the RAG system with a vector store and language model.
    """
    
    def __init__(self, rag_index, llm):
        """
        Initialize the QueryEngine.
        
        Args:
            rag_index: The RAGIndex instance containing the vector store
            llm: The language model instance to use for generating responses
        """
        self.rag_index = rag_index
        self.llm = llm
    
    def query(
        self, 
        query: str, 
        k: int = 4, 
        return_sources: bool = True,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ) -> Tuple[str, List[Document]]:
        """
        Query the RAG system with a question.
        
        Args:
            query: The question to ask
            k: Number of document chunks to retrieve
            return_sources: Whether to return source documents
            temperature: Temperature for the LLM generation
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            A tuple of (response, sources) where sources is a list of Document objects
        """
        try:
            # Get relevant documents using similarity search
            docs = self.rag_index.similarity_search(query, k=k)
            
            if not docs:
                return "I couldn't find any relevant information to answer your question.", []
            
            # Create a prompt template
            prompt_template = """You are an AI assistant analyzing an insurance policy document. 
            Your task is to provide accurate information based on the document content, especially focusing on tables and lists that contain coverage details.

            Document Context:
            {context}

            When answering questions about coverage, benefits, or exclusions, pay special attention to any tables or lists in the context, 
            as they often contain important details about what is covered or excluded.

            If the question is about whether something is covered, first check for explicit mentions in the text, then check any tables or lists for details.
            If you find the information in a table, format your response to clearly indicate this.

            Question: {question}
            
            Guidelines for your response:
            1. Be precise and quote relevant policy details when possible
            2. If information comes from a table, mention that it's from a table
            3. If you're unsure, say so rather than guessing
            4. For coverage questions, clearly state whether something is covered, excluded, or if the information isn't clear
            
            Answer:
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Format the context to better preserve table structure
            formatted_docs = []
            for i, doc in enumerate(docs):
                source = f"Document {i+1}"
                if hasattr(doc, 'metadata'):
                    if 'source' in doc.metadata:
                        source = os.path.basename(str(doc.metadata['source']))
                    if 'page' in doc.metadata:
                        source += f" (Page {doc.metadata['page']})"
                
                # Add markers for table content
                content = doc.page_content
                if '```table' in content or '|' in content:  # Likely contains a table
                    content = "\n[TABLE CONTENT]\n" + content + "\n[END TABLE]\n"
                
                formatted_docs.append(f"[{source}]\n{content}")
            
            context = "\n\n".join(formatted_docs)
            
            # Create a QA chain
            qa_chain = load_qa_chain(
                self.llm,
                chain_type="stuff",
                prompt=prompt,
                verbose=True
            )
            
            # Get the response
            response = qa_chain(
                {"input_documents": docs, "question": query},
                return_only_outputs=True
            )
            
            # Extract the answer
            answer = response.get("output_text", "I'm sorry, I couldn't find an answer to your question.")
            
            # Clean up the response
            answer = answer.strip()
            
            # Return the response and sources if requested
            if return_sources:
                return answer, docs
            return answer, []
            
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"An error occurred while processing your request: {str(e)}", []

def get_rag_response(
    query: str,
    vector_store: VectorStore,
    llm: BaseChatModel,
    k: int = 4,
    return_sources: bool = True,
    temperature: float = 0.1,
    max_tokens: int = 4000
) -> Dict[str, Any]:
    """
    Get a response from the RAG system with enhanced error handling and source tracking.
    
    Args:
        query: The user's question
        vector_store: The vector store with document embeddings
        llm: The language model instance
        k: Number of document chunks to retrieve
        return_sources: Whether to include source documents in the response
        temperature: Temperature for the LLM generation
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Dict containing the answer and optionally sources
    """
    try:
        # Create a retriever with similarity search
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Create a RetrievalQA chain with the retriever and LLM
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        
        # Define a custom prompt template for better responses
        template = """
        You are an AI assistant specialized in insurance policies and legal documents.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always maintain a professional, helpful tone and provide accurate information based only on the context provided.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # Get the response
        chain_response = qa_chain({"query": query})
        
        # Extract the answer and source documents
        answer = chain_response.get("result", "")
        docs = chain_response.get("source_documents", [])
        
        if not docs:
            return {
                "answer": "I couldn't find any relevant information in the provided documents to answer your question.",
                "sources": []
            }
        
        # Format context from documents with source tracking
        context_parts = []
        for i, doc in enumerate(docs):
            source = f"Document {i+1}"
            if hasattr(doc, 'metadata'):
                if 'source' in doc.metadata:
                    source = os.path.basename(doc.metadata['source'])
                if 'page' in doc.metadata:
                    source += f" (Page {doc.metadata['page']})"
            context_parts.append(f"[{source}]\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Normalize the query for easier matching
        normalized_query = query.lower().strip('?.,! ')
        
        # Check for specific item coverage questions (e.g., "is knee surgery covered?")
        item_coverage_match = re.search(
            r'(?i)(?:is|are|does this policy cover|is covered|coverage for|what about|what\'?s the coverage for|tell me about the coverage for|show me the details for|details about|is .* covered|does this plan cover|does the policy cover|coverage of|coverage on)\s+([^?.,!]+)[?.,!]?$',
            normalized_query
        )
        specific_item = None
        if item_coverage_match:
            specific_item = item_coverage_match.group(1).strip()
            # Clean up the specific item (remove trailing punctuation, etc.)
            specific_item = re.sub(r'[.,!?]$', '', specific_item).strip()
            
            # Common medical procedure variations
            procedure_variations = {
                'knee surgery': ['knee operation', 'knee replacement', 'knee procedure', 'arthroscopy', 'ACL surgery'],
                'hip surgery': ['hip operation', 'hip replacement', 'hip procedure', 'hip arthroplasty'],
                'heart surgery': ['cardiac surgery', 'heart operation', 'bypass surgery', 'angioplasty'],
                'cataract surgery': ['cataract operation', 'cataract procedure', 'lens replacement'],
                'lasik': ['lasik surgery', 'laser eye surgery', 'refractive surgery'],
                'mri': ['magnetic resonance imaging', 'mri scan', 'mri test'],
                'ct scan': ['cat scan', 'computed tomography', 'ct scan'],
                'x-ray': ['xray', 'x ray', 'radiograph']
            }
            
            # Add variations to the context
            for proc, variations in procedure_variations.items():
                if proc in specific_item.lower():
                    specific_item = f"{specific_item} (or {', '.join(variations)})"
        
        # Check for policy detail questions (policy number, name, etc.)
        policy_detail_queries = {
            'policy number': 'policy number',
            'policy name': 'policy name',
            'sum insured': 'sum insured',
            'premium': 'premium',
            'coverage amount': 'coverage amount',
            'what is the policy number': 'policy number',
            'what is the policy name': 'policy name',
            'what is the sum insured': 'sum insured',
            'how much is the premium': 'premium',
            'coverage limit': 'coverage limit',
            'deductible': 'deductible',
            'policy period': 'policy period',
            'effective date': 'effective date',
            'expiration date': 'expiration date',
            'benefit amount': 'benefit amount',
            'waiting period': 'waiting period',
            'excess': 'excess',
            'sub-limit': 'sub-limit',
            'co-payment': 'co-payment'
        }
        
        policy_detail = None
        for term, detail_type in policy_detail_queries.items():
            if term in normalized_query:
                policy_detail = detail_type
                break
        
        # Check if the question is about exclusions, tables, or items not covered
        is_table_query = any(term in normalized_query for term in [
            'table', 'tabular', 'chart', 'grid', 'matrix', 'schedule',
            'show me the table', 'list of', 'details in tabular form',
            'tabular format', 'in a table', 'as a table', 'display table',
            'show table', 'view table', 'see table', 'list all', 'show all',
            'display all', 'list of all', 'show me all', 'view all', 'see all'
        ])
        
        is_exclusion_question = any(term in normalized_query for term in [
            'excluded', 'not covered', 'exclusion', 'not include', 'exclude',
            'what is not covered', 'what are the exclusions', 'list of exclusions',
            'what does the policy exclude', 'what is excluded', 'what items are excluded',
            'what is not included', 'what are the limitations', 'what are the restrictions',
            'list of items not covered', 'what does this not cover', 'exclusion list',
            'what is excluded from coverage', 'what is not covered under this policy',
            'limitations', 'restrictions', 'not eligible', 'not payable', 'not covered',
            'coverage details', 'coverage information', 'what does it cover', 'what are the benefits',
            'what is included', 'what does this cover', 'coverage includes', 'coverage excludes'
        ]) or bool(specific_item) or bool(policy_detail) or is_table_query

        # Enhanced prompt template with specialized handling for different question types
        if is_exclusion_question:
            if is_table_query or 'list of' in normalized_query or 'show me all' in normalized_query:
                # Specialized prompt for table/list queries
                prompt_template = """You are an AI assistant that extracts and presents tabular information from policy documents.
                
                INSTRUCTIONS:
                1. Extract and present the requested information in a clear, well-formatted markdown table.
                2. If the context contains tables or lists, present them as markdown tables.
                3. For long lists, consider using bullet points for better readability.
                4. Include all relevant details from the context.
                5. Preserve the original formatting and structure as much as possible.
                
                CONTEXT:
                {context}
                
                QUESTION: {question}
                
                FORMAT YOUR RESPONSE IN MARKDOWN:
                ```markdown
                ## {question}
                
                ### Extracted Information
                [Present the information in a clear, organized format using markdown tables, bullet points, or numbered lists as appropriate.]
                
                ### Source
                [Reference to document section/table/page if available]
                
                ### Notes
                [Any additional context or explanations]
                ```
                
                ANSWER:
                """.format(question=query, context=context)
            elif specific_item:
                # Specialized prompt for medical procedure coverage questions
                prompt_template = """You are an insurance policy expert analyzing coverage for medical procedures.
                
                INSTRUCTIONS:
                1. Carefully analyze the context to determine if "{item}" is covered.
                2. Look for these coverage indicators:
                   - Exact matches for "{item}" and its variations
                   - Related terms or procedures (e.g., 'knee surgery' includes 'arthroscopy')
                   - General coverage categories that might include this procedure
                   - Exclusion lists that specifically mention this procedure
                
                3. In your response:
                   - Start with a clear, direct answer (Covered/Not Covered/Unclear)
                   - Quote relevant policy language with page/section references
                   - Note any conditions, limitations, or requirements
                   - If uncertain, explain why and suggest next steps
                
                CONTEXT:
                {context}
                
                QUESTION: {question}
                
                FORMAT YOUR RESPONSE EXACTLY AS SHOWN BELOW. USE MARKDOWN FOR BOLD TEXT IN HEADINGS.
                
                **{item}**
                
                **Coverage status:**
                [✅ Covered / ❌ Not covered / ❓ Unclear]
                
                **Details:**
                [Provide specific policy language, conditions, and limitations]
                
                **Source:**
                [Reference specific page/section/table numbers]
                
                **Additional notes:**
                [Any other relevant information or recommendations]
                
                **Next steps:**
                [Recommended actions if coverage is unclear]
                
                IMPORTANT RULES:
                - Make section headers bold using ** **
                - Use title case for section headers (e.g., 'Coverage Status')
                - Start each sentence with a capital letter
                - Use newlines between sections
                - Be precise and evidence-based
                - Only state coverage status if explicitly mentioned in the policy
                - If uncertain, explain what information is missing
                - Do not use all caps or excessive punctuation
                
                ANSWER (use newlines and markdown for bold text only):
                """.format(item=specific_item, context=context, question=query)
            elif policy_detail:
                # Specialized prompt for policy detail questions
                prompt_template = """You are an insurance policy expert. Analyze the following document to find the {detail_type}.
                
                INSTRUCTIONS:
                1. Search for the {detail_type} in the provided context.
                2. Look for exact matches and related terms.
                3. If the information is not explicitly stated, check for:
                   - Tables with policy details
                   - Policy schedule sections
                   - Coverage summaries
                
                CONTEXT:
                {context}
                
                FORMAT YOUR RESPONSE EXACTLY AS SHOWN BELOW. DO NOT USE MARKDOWN CODE BLOCKS.
                
                {detail_type}
                
                Value:
                [The exact value from the document, or "Not specified" if not found]
                
                Location:
                [Page/Section/Table reference if available]
                
                Additional Information:
                [Any relevant details, conditions, or related information]
                
                Next Steps:
                [Recommended actions if information is unclear or missing]
                
                IMPORTANT RULES:
                - ALWAYS include a newline after each section header
                - DO NOT use markdown formatting (no **bold**, no headers, no code blocks)
                - DO use plain text with newlines
                - BE PRECISE and evidence-based
                - If information is not found, clearly state: 
                  "The {detail_type} is not specified in the main policy document. This information might be in a policy schedule or certificate that wasn't provided."
                
                ANSWER (with proper newlines and formatting):
                """.format(detail_type=policy_detail, context=context, question=query)
            else:
                # General exclusions prompt
                prompt_template = """You are an AI assistant that helps users understand their insurance policy documents, 
                with special attention to exclusions, limitations, and tabular data.
                
                INSTRUCTIONS:
                1. Analyze the context to answer the question accurately.
                2. If the context contains tables or lists, present them in a clean, readable format.
                3. For exclusion/limitation questions, be specific about what is and isn't covered.
                
                FORMAT YOUR RESPONSE EXACTLY AS SHOWN BELOW. DO NOT USE MARKDOWN CODE BLOCKS.
                
                [Brief Answer]
                
                Details:
                [Provide specific policy language, conditions, and limitations]
                
                Source:
                [Reference specific page/section/table numbers]
                
                Additional Notes:
                [Any other relevant information or recommendations]
                
                Next Steps:
                [Recommended actions if information is unclear or missing]
                
                IMPORTANT RULES:
                - ALWAYS include a newline after each section header
                - DO NOT use markdown formatting (no **bold**, no headers, no code blocks)
                - DO use plain text with newlines
                - BE PRECISE and evidence-based
                - If information is not found, clearly state that
                - For lists, use bullet points with dashes (-)
                
                CONTEXT:
                {context}
                
                QUESTION: {question}
                
                FORMAT YOUR RESPONSE USING MARKDOWN:
                - Use **bold** for important terms
                - Use bullet points (•) for lists
                - Use numbered lists for steps or sequences
                - Use tables for tabular data
                - Use headers (##, ###) for sections
                - Keep paragraphs short and focused
                - Include source references when possible
                
                If you don't know the answer, say so and explain why.
                
                ANSWER:
                """
        else:
            prompt_template = """You are an AI assistant that helps users understand their insurance policy documents. 
            Use the following pieces of context to answer the question at the end. 
            
            If the context doesn't contain enough information to answer the question, 
            just say that you don't know. Don't try to make up an answer.
            
            Context:
            {context}
            
            Question: {question}
            
            Please provide a clear and concise answer. If the answer includes specific numbers, 
            benefits, coverage details, or limitations, be sure to include them in your response.
            
            If the question is about comparing plans or understanding specific terms, 
            make sure to explain in simple, easy-to-understand language.
            
            Format your response in markdown with appropriate headers, bold text for important 
            terms, and bullet points for lists.
            
            Answer:
            """
        
        # Clean and prepare the context
        clean_context = context.strip()
        
        # Check if we have valid context
        if not clean_context or clean_context == "undefined" or "Policy Schedule/Policy Certificate/Endorsement" in clean_context:
            return {
                "answer": "I couldn't find specific information about this in the policy document. The details might be in a policy schedule or certificate that wasn't provided.",
                "sources": []
            }
            
        # Create prompt with context and question
        try:
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            ).format(context=clean_context, question=query)
            
            # Get response from LLM with error handling
            response = llm.invoke(prompt)
            
            # Ensure we have a valid response
            if not response:
                raise ValueError("Empty response from language model")
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return {
                "answer": "I had trouble processing this request. Please try rephrasing your question or ask about a different aspect of the policy.",
                "sources": []
            }
        
        # Extract the answer (handling different response formats)
        if hasattr(response, 'content'):
            answer = response.content
        elif isinstance(response, str):
            answer = response
        elif hasattr(response, 'text'):
            answer = response.text
        else:
            answer = str(response)
        
        # Extract and format detailed source information
        sources = []
        seen_sources = set()
        
        for doc in docs:
            source_path = doc.metadata.get("source", "Unknown")
            source_name = os.path.basename(source_path) if source_path != "Unknown" else "Unknown"
            page = doc.metadata.get("page", "")
            
            # Create a unique identifier for the source
            source_id = f"{source_path}:{page}" if page else source_path
            
            if source_id not in seen_sources:
                seen_sources.add(source_id)
                
                # Extract a preview of the content (first 200 chars)
                preview = doc.page_content[:200]
                if len(doc.page_content) > 200:
                    preview += "..."
                
                sources.append({
                    "name": source_name,
                    "path": source_path,
                    "page": page,
                    "preview": preview,
                    "relevance": doc.metadata.get("relevance_score", 0.0)
                })
        
        # Sort sources by relevance (highest first)
        sources.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Format the response
        result = {
            "answer": answer.strip(),
            "sources": sources if return_sources else [],
            "context": context if return_sources else "",
            "used_snippets": [{
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in docs]
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in get_rag_response: {str(e)}", exc_info=True)
        return {
            "answer": f"I encountered an error while processing your request: {str(e)}\n\nPlease try again or rephrase your question.",
            "sources": []
        }

def get_suggested_questions(
    context: str,
    llm: BaseChatModel,
    num_questions: int = 4,
    max_context_length: int = 8000
) -> List[str]:
    """
    Generate relevant suggested questions based on the document context.
    
    This is a simplified version that avoids recursion entirely and has robust error handling.
    
    Args:
        context: The document context to analyze
        llm: The language model instance to use
        num_questions: Number of questions to generate (default: 4)
        max_context_length: Maximum length of context to use (in characters)
        
    Returns:
        List of suggested questions (strings)
    """
    # Default fallback questions
    default_questions = [
        "What are the key benefits of this policy?",
        "What is not covered by this insurance?",
        "How do I file a claim?",
        "What is the policy's coverage limit?"
    ]
    
    try:
        # Ensure we have a valid context
        if not context or not isinstance(context, str):
            logger.warning("Invalid or empty context provided for question generation")
            return default_questions[:num_questions]
            
        # Truncate context if too long
        if len(context) > max_context_length:
            context = context[:max_context_length] + "... [truncated]"
        
        prompt = f"""You are an AI assistant helping users explore their insurance policy documents.
        
        Based on the following document context, generate exactly {num_questions} specific and useful questions 
        that a policyholder might ask about their coverage, benefits, or policy details.
        
        Guidelines:
        1. Focus on important details mentioned in the context
        2. Make questions specific and answerable from the document
        3. Vary the types of questions (coverage, benefits, exclusions, etc.)
        4. Keep questions concise and clear
        5. Avoid yes/no questions
        6. Generate EXACTLY {num_questions} questions
        
        Document Context:
        {context}
        
        Generate exactly {num_questions} questions, one per line, without numbering or bullet points.
        """
        
        # Get response from LLM with a timeout to prevent hanging
        try:
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error getting response from LLM: {str(e)}")
            return default_questions[:num_questions]
        
        # Process the response to extract questions
        questions = []
        for line in response_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Remove any numbering or bullet points
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '- ', '* ')):
                line = line[2:].strip()
            if line and line[0] in '1234567890-*•':
                line = line[1:].strip()
            if line:
                # Ensure the line ends with a question mark
                if not line.endswith('?'):
                    line = line.rstrip('.') + '?'
                questions.append(line)
        
        # If we got some questions but not enough, pad with default questions
        if 0 < len(questions) < num_questions:
            # Only add default questions that aren't already in our list
            for q in default_questions:
                if q not in questions:
                    questions.append(q)
                    if len(questions) >= num_questions:
                        break
        # If we got no questions at all, use default questions
        elif not questions:
            return default_questions[:num_questions]
        
        # Return up to the requested number of questions
        return questions[:num_questions]
        
    except Exception as e:
        logger.error(f"Unexpected error in get_suggested_questions: {str(e)}", exc_info=True)
        return default_questions[:num_questions]
