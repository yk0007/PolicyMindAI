"""
Document Loader for PolicyMindAI.

Handles loading and processing of various document formats with
advanced semantic chunking for insurance policy documents.
"""

import os
import re
import logging
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Import LangChain document types
try:
    from langchain_core.documents import Document as LangchainDocument
except ImportError:
    try:
        from langchain.schema import Document as LangchainDocument
    except ImportError:
        raise ImportError("langchain-core or langchain is required")

# Import third-party libraries with error handling
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    
try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None
    
try:
    import pandas as pd
except ImportError:
    pd = None

# Import our semantic chunker
from .chunker import SemanticChunker, chunk_document, estimate_tokens

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

INSURANCE_KEYWORDS = frozenset({
    'insurance', 'policy', 'premium', 'coverage', 'benefit', 'claim', 'deductible',
    'sum insured', 'policyholder', 'insured', 'insurer', 'exclusion', 'endorsement',
    'rider', 'annuity', 'maturity', 'surrender', 'nominee', 'proposer', 'underwriting',
    'hospitalization', 'cashless', 'reimbursement', 'waiting period', 'pre-existing'
})

DEFAULT_MAX_PAGES = 20
DEFAULT_TIMEOUT = 60


# =============================================================================
# Document Loader Class
# =============================================================================

class DocumentLoader:
    """
    Document loader with support for multiple formats.
    
    Supports: PDF, DOCX, TXT, MD, CSV, XLSX, XLS
    """
    
    def __init__(self):
        """Initialize with format handlers."""
        self.supported_formats = {
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.txt': self._load_text,
            '.md': self._load_text,
            '.csv': self._load_csv,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
        }
    
    def load_document(self, file_path: str) -> List[LangchainDocument]:
        """Load a document from file path."""
        file_path = str(file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return self.supported_formats[file_ext](file_path)
    
    def _load_pdf(self, file_path: str) -> List[LangchainDocument]:
        """Load PDF with enhanced table extraction."""
        if not pdfplumber:
            raise ImportError("pdfplumber is required for PDF processing")
        
        try:
            logger.info(f"Processing PDF: {file_path}")
            pages = []
            
            with pdfplumber.open(file_path) as pdf:
                total_pages = min(len(pdf.pages), DEFAULT_MAX_PAGES)
                
                for i, page in enumerate(pdf.pages[:total_pages]):
                    page_num = i + 1
                    try:
                        # Extract text
                        page_text = page.extract_text() or ""
                        
                        # Extract tables
                        tables_md = self._extract_tables_from_page(page)
                        
                        # Combine content
                        content_parts = []
                        if page_text.strip():
                            content_parts.append(page_text)
                        if tables_md:
                            content_parts.append(tables_md)
                        
                        if content_parts:
                            pages.append(LangchainDocument(
                                page_content='\n\n'.join(content_parts),
                                metadata={
                                    'source': file_path,
                                    'file_name': os.path.basename(file_path),
                                    'file_type': 'pdf',
                                    'page': page_num,
                                    'total_pages': total_pages
                                }
                            ))
                    except Exception as e:
                        logger.warning(f"Error on page {page_num}: {e}")
                        continue
            
            if not pages:
                # Fallback to PyMuPDF
                pages = self._fallback_pdf_extraction(file_path)
            
            return pages
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
    
    def _extract_tables_from_page(self, page) -> str:
        """Extract tables from PDF page as markdown."""
        tables = []
        
        try:
            extracted = page.extract_tables()
            for table in extracted:
                if not table or len(table) < 2:
                    continue
                
                # Clean and format table
                cleaned = []
                for row in table:
                    cleaned_row = [
                        str(cell).strip().replace('\n', ' ') if cell else ""
                        for cell in row
                    ]
                    if any(cleaned_row):
                        cleaned.append(cleaned_row)
                
                if cleaned:
                    md = self._format_table_markdown(cleaned)
                    if md:
                        tables.append(f"```table\n{md}\n```")
        except Exception as e:
            logger.debug(f"Table extraction failed: {e}")
        
        return '\n\n'.join(tables) if tables else ""
    
    def _format_table_markdown(self, rows: List[List[str]]) -> str:
        """Format rows as markdown table."""
        if not rows:
            return ""
        
        headers = rows[0]
        separator = ['---'] * len(headers)
        
        lines = [
            '| ' + ' | '.join(headers) + ' |',
            '|' + '|'.join(separator) + '|'
        ]
        
        for row in rows[1:]:
            # Pad row to header length
            while len(row) < len(headers):
                row.append("")
            lines.append('| ' + ' | '.join(row[:len(headers)]) + ' |')
        
        return '\n'.join(lines)
    
    def _fallback_pdf_extraction(self, file_path: str) -> List[LangchainDocument]:
        """Fallback PDF extraction using PyMuPDF."""
        if not fitz:
            return []
        
        try:
            docs = []
            with fitz.open(file_path) as pdf:
                for i, page in enumerate(pdf):
                    text = page.get_text()
                    if text.strip():
                        docs.append(LangchainDocument(
                            page_content=text,
                            metadata={
                                'source': file_path,
                                'file_name': os.path.basename(file_path),
                                'file_type': 'pdf',
                                'page': i + 1
                            }
                        ))
            return docs
        except Exception as e:
            logger.warning(f"Fallback extraction failed: {e}")
            return []
    
    def _load_docx(self, file_path: str) -> List[LangchainDocument]:
        """Load DOCX file."""
        if not DocxDocument:
            raise ImportError("python-docx is required for DOCX processing")
        
        doc = DocxDocument(file_path)
        text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        
        return [LangchainDocument(
            page_content=text,
            metadata={"source": file_path, "file_type": "docx"}
        )]
    
    def _load_text(self, file_path: str) -> List[LangchainDocument]:
        """Load plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        return [LangchainDocument(
            page_content=text,
            metadata={"source": file_path, "file_type": "text"}
        )]
    
    def _load_csv(self, file_path: str) -> List[LangchainDocument]:
        """Load CSV as markdown table."""
        if not pd:
            raise ImportError("pandas is required for CSV processing")
        
        df = pd.read_csv(file_path)
        md = df.to_markdown(index=False)
        
        return [LangchainDocument(
            page_content=f"```table\n{md}\n```",
            metadata={"source": file_path, "file_type": "csv"}
        )]
    
    def _load_excel(self, file_path: str) -> List[LangchainDocument]:
        """Load Excel file with all sheets."""
        if not pd:
            raise ImportError("pandas is required for Excel processing")
        
        xl = pd.ExcelFile(file_path)
        docs = []
        
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            if not df.empty:
                md = df.to_markdown(index=False)
                docs.append(LangchainDocument(
                    page_content=f"# {sheet_name}\n\n```table\n{md}\n```",
                    metadata={
                        "source": file_path,
                        "file_type": "excel",
                        "sheet_name": sheet_name
                    }
                ))
        
        return docs


# =============================================================================
# Text Processing
# =============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove control characters
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
    
    return text.strip()


def is_insurance_document(text: str, threshold: int = 3) -> bool:
    """Check if text is insurance-related."""
    if not text:
        return False
    
    text_lower = text.lower()
    matches = sum(1 for kw in INSURANCE_KEYWORDS if kw in text_lower)
    
    return matches >= threshold


def extract_text_from_file(file_path: str) -> str:
    """Extract text from a file."""
    loader = DocumentLoader()
    docs = loader.load_document(file_path)
    return '\n\n'.join(doc.page_content for doc in docs)


# =============================================================================
# Main Processing Function
# =============================================================================

def process_document(
    file_path: Union[str, os.PathLike],
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    metadata: Optional[Dict[str, Any]] = None,
    use_semantic_chunking: bool = True,
    **kwargs
) -> List[LangchainDocument]:
    """
    Process a document into semantically chunked pieces.
    
    Args:
        file_path: Path to document
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        metadata: Additional metadata
        use_semantic_chunking: Use advanced semantic chunking
        **kwargs: Additional arguments
        
    Returns:
        List of document chunks
    """
    try:
        file_path = str(file_path)
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Processing document: {path_obj.name}")
        
        # Initialize metadata
        if metadata is None:
            metadata = {}
        
        # Load document
        loader = DocumentLoader()
        docs = loader.load_document(file_path)
        
        # Combine all text
        full_text = '\n\n'.join([doc.page_content for doc in docs])
        
        # Clean text
        text = clean_text(full_text)
        
        # Update base metadata
        file_ext = path_obj.suffix.lower()
        base_metadata = {
            **metadata,
            'source': path_obj.name,
            'file_path': str(path_obj.absolute()),
            'file_name': path_obj.name,
            'file_extension': file_ext.lstrip('.'),
            'file_size': path_obj.stat().st_size,
            'is_insurance_document': is_insurance_document(text),
            'total_tokens': estimate_tokens(text)
        }
        
        # Use semantic chunking
        if use_semantic_chunking:
            chunker = SemanticChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = chunker.chunk_document(text, base_metadata)
        else:
            # Fallback to simple chunking
            chunks = _simple_chunk(text, chunk_size, chunk_overlap, base_metadata)
        
        # Convert to LangchainDocument
        langchain_docs = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = chunk.get('metadata', {})
            chunk_metadata['chunk_id'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            
            langchain_docs.append(LangchainDocument(
                page_content=chunk['text'],
                metadata=chunk_metadata
            ))
        
        logger.info(f"Created {len(langchain_docs)} chunks from {path_obj.name}")
        return langchain_docs
        
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}")
        raise


def _simple_chunk(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Simple fallback chunking."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Convert tokens to approximate characters
    char_size = chunk_size * 4
    char_overlap = chunk_overlap * 4
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_size,
        chunk_overlap=char_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    
    return [
        {'text': chunk, 'metadata': metadata.copy()}
        for chunk in chunks
    ]
