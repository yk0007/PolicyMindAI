"""
Semantic Chunker for PolicyMindAI.

Advanced document chunking that:
- Uses token counting (not characters) for accurate LLM context management
- Preserves table structure intact
- Respects section boundaries
- Maintains semantic coherence within chunks
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, falling back to word-based estimation")


class SectionType(Enum):
    """Types of document sections."""
    TABLE = "table"
    LIST = "list"
    HEADER = "header"
    PARAGRAPH = "paragraph"
    COVERAGE = "coverage"
    EXCLUSION = "exclusion"
    BENEFIT = "benefit"
    TERMS = "terms"
    CLAIMS = "claims"


@dataclass
class DocumentSection:
    """Represents a semantic section of a document."""
    content: str
    section_type: SectionType
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    is_atomic: bool = False  # If True, never split this section
    
    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = estimate_tokens(self.content)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    Uses tiktoken if available, otherwise falls back to word count estimation.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    if TIKTOKEN_AVAILABLE:
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
            return len(encoder.encode(text))
        except Exception:
            pass
    
    # Fallback: roughly 1.3 tokens per word
    words = len(text.split())
    return int(words * 1.3)


class SemanticChunker:
    """
    Semantic-aware document chunker for insurance policies.
    
    Features:
    - Token-based chunking (accurate for LLM context limits)
    - Table preservation (never splits tables)
    - Section-aware splitting (respects headers, coverage sections, etc.)
    - Insurance document structure awareness
    """
    
    def __init__(
        self,
        chunk_size: int = 400,  # Target tokens per chunk
        chunk_overlap: int = 50,  # Overlap tokens
        max_table_tokens: int = 800,  # Max tokens for a single table
        preserve_tables: bool = True,
        preserve_lists: bool = True
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            max_table_tokens: Maximum tokens for a table before splitting
            preserve_tables: Whether to keep tables as atomic units
            preserve_lists: Whether to keep lists as atomic units
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_table_tokens = max_table_tokens
        self.preserve_tables = preserve_tables
        self.preserve_lists = preserve_lists
        
        # Patterns for detecting document structure
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup regex patterns for structure detection."""
        # Table patterns
        self.table_patterns = [
            r'```table\n[\s\S]*?\n```',  # Markdown code block tables
            r'\|[^\n]+\|(?:\n\|[^\n]+\|)+',  # Markdown tables
            r'(?:(?:[^\n]+\t[^\n]+)\n?)+',  # Tab-separated tables
        ]
        
        # Header patterns
        self.header_patterns = [
            r'^#{1,6}\s+.+$',  # Markdown headers
            r'^[A-Z][A-Z\s]+:?\s*$',  # ALL CAPS headers
            r'^(?:Section|Article|Part|Chapter)\s+\d+',  # Numbered sections
            r'^(?:SCHEDULE|ANNEXURE|APPENDIX)\s+[A-Z0-9]+',  # Schedule sections
        ]
        
        # Insurance-specific section headers
        self.insurance_headers = {
            'coverage': [
                r'(?:coverage|covered|what is covered|scope of cover)',
                r'(?:benefits?|sum\s*insured|policy\s*benefits)',
            ],
            'exclusion': [
                r'(?:exclusions?|not\s*covered|what is not covered)',
                r'(?:exceptions?|limitations?)',
            ],
            'terms': [
                r'(?:terms?\s*(and|&)\s*conditions?|general\s*conditions?)',
                r'(?:definitions?|glossary)',
            ],
            'claims': [
                r'(?:claims?\s*process|how to claim|claim\s*procedure)',
                r'(?:intimation|notification)',
            ],
        }
        
        # List patterns
        self.list_patterns = [
            r'(?:^[\s]*[-•*]\s+.+\n?)+',  # Bullet lists
            r'(?:^[\s]*\d+[.)]\s+.+\n?)+',  # Numbered lists
            r'(?:^[\s]*[a-z][.)]\s+.+\n?)+',  # Lettered lists
        ]
    
    def chunk_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document into semantic units.
        
        Args:
            text: Full document text
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        
        # Step 1: Detect and extract sections
        sections = self._detect_sections(text)
        
        # Step 2: Create chunks respecting section boundaries
        chunks = self._create_chunks(sections, metadata)
        
        # Step 3: Add overlap context
        chunks = self._add_overlap(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _detect_sections(self, text: str) -> List[DocumentSection]:
        """
        Detect and extract semantic sections from text.
        
        Args:
            text: Document text
            
        Returns:
            List of DocumentSection objects
        """
        sections = []
        remaining_text = text
        position = 0
        
        # First, extract tables (highest priority - never split)
        table_positions = self._extract_tables(text)
        
        # Build sections from text, preserving tables
        current_pos = 0
        for start, end, table_text in table_positions:
            # Add text before table
            if start > current_pos:
                pre_text = text[current_pos:start].strip()
                if pre_text:
                    sections.extend(self._split_into_sections(pre_text))
            
            # Add table as atomic section(s)
            table_chunks = self._split_large_table(table_text)
            for idx, table_chunk in enumerate(table_chunks):
                table_metadata = {'type': 'table'}
                if len(table_chunks) > 1:
                    table_metadata.update({
                        'table_part': idx + 1,
                        'table_parts': len(table_chunks)
                    })
                sections.append(DocumentSection(
                    content=table_chunk,
                    section_type=SectionType.TABLE,
                    is_atomic=True,
                    metadata=table_metadata
                ))
            current_pos = end
        
        # Add remaining text after last table
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining:
                sections.extend(self._split_into_sections(remaining))
        
        # If no tables found, process entire text
        if not table_positions:
            sections = self._split_into_sections(text)
        
        return sections
    
    def _extract_tables(self, text: str) -> List[Tuple[int, int, str]]:
        """Extract table content with positions from text."""
        tables = []
        
        for pattern in self.table_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                table_text = match.group(0)
                if not table_text.strip() or len(table_text) <= 20:
                    continue
                start, end = match.span()
                if self._overlaps_existing_table(tables, start, end):
                    continue
                tables.append((start, end, table_text))
        
        tables.sort(key=lambda item: item[0])
        return tables

    def _overlaps_existing_table(
        self,
        tables: List[Tuple[int, int, str]],
        start: int,
        end: int
    ) -> bool:
        """Check if a table span overlaps an existing table."""
        for existing_start, existing_end, _ in tables:
            if start < existing_end and end > existing_start:
                return True
        return False

    def _split_large_table(self, table_text: str) -> List[str]:
        """Split oversized tables into multiple chunks."""
        stripped = table_text.strip()
        if not stripped:
            return []
        
        tokens = estimate_tokens(stripped)
        if tokens <= self.max_table_tokens:
            return [table_text]
        
        lines = stripped.splitlines()
        if len(lines) < 3:
            return [table_text]
        
        if lines[0].startswith("```table") and lines[-1].strip() == "```":
            header_lines = lines[1:3]
            row_lines = lines[3:-1]
            return self._split_table_lines(
                lines[0],
                header_lines,
                row_lines,
                lines[-1]
            )
        
        if lines[0].lstrip().startswith("|") and lines[1].lstrip().startswith("|"):
            header_lines = lines[:2]
            row_lines = lines[2:]
            return self._split_table_lines("", header_lines, row_lines, "")
        
        if "\t" in stripped:
            header_lines = [lines[0]]
            row_lines = lines[1:]
            return self._split_table_lines("", header_lines, row_lines, "")
        
        return [table_text]

    def _split_table_lines(
        self,
        fence_start: str,
        header_lines: List[str],
        row_lines: List[str],
        fence_end: str
    ) -> List[str]:
        """Split table rows into chunks while preserving headers."""
        chunks = []
        if not row_lines:
            return ["\n".join([l for l in [fence_start, *header_lines, fence_end] if l])]
        
        header_block = "\n".join([l for l in [fence_start, *header_lines] if l])
        footer_block = fence_end if fence_end else ""
        header_tokens = estimate_tokens("\n".join([header_block, footer_block]).strip())
        
        current_rows = []
        current_tokens = header_tokens
        
        for row in row_lines:
            row_tokens = estimate_tokens(row)
            if current_rows and current_tokens + row_tokens > self.max_table_tokens:
                chunks.append(self._render_table_chunk(
                    fence_start,
                    header_lines,
                    current_rows,
                    fence_end
                ))
                current_rows = [row]
                current_tokens = header_tokens + row_tokens
            else:
                current_rows.append(row)
                current_tokens += row_tokens
        
        if current_rows:
            chunks.append(self._render_table_chunk(
                fence_start,
                header_lines,
                current_rows,
                fence_end
            ))
        
        return chunks

    def _render_table_chunk(
        self,
        fence_start: str,
        header_lines: List[str],
        row_lines: List[str],
        fence_end: str
    ) -> str:
        """Render a table chunk with preserved headers."""
        parts = [line for line in [fence_start, *header_lines] if line]
        parts.extend(row_lines)
        if fence_end:
            parts.append(fence_end)
        return "\n".join(parts)
    
    def _split_into_sections(self, text: str) -> List[DocumentSection]:
        """
        Split non-table text into semantic sections.
        
        Args:
            text: Text to split
            
        Returns:
            List of DocumentSection objects
        """
        sections = []
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', text)
        
        current_section = []
        current_type = SectionType.PARAGRAPH
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Detect section type
            section_type = self._classify_section(para)
            
            # Check if this is a header (start new section)
            if self._is_header(para):
                # Save current section
                if current_section:
                    sections.append(DocumentSection(
                        content='\n\n'.join(current_section),
                        section_type=current_type
                    ))
                    current_section = []
                
                current_type = section_type
                current_section.append(para)
            
            # Check if this is a list
            elif self._is_list(para):
                # Add list to current section as atomic
                if current_section:
                    sections.append(DocumentSection(
                        content='\n\n'.join(current_section),
                        section_type=current_type
                    ))
                    current_section = []
                
                sections.append(DocumentSection(
                    content=para,
                    section_type=SectionType.LIST,
                    is_atomic=self.preserve_lists
                ))
            else:
                current_section.append(para)
        
        # Don't forget the last section
        if current_section:
            sections.append(DocumentSection(
                content='\n\n'.join(current_section),
                section_type=current_type
            ))
        
        return sections
    
    def _classify_section(self, text: str) -> SectionType:
        """Classify section type based on content."""
        text_lower = text.lower()
        
        for section_name, patterns in self.insurance_headers.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    if section_name == 'coverage':
                        return SectionType.COVERAGE
                    elif section_name == 'exclusion':
                        return SectionType.EXCLUSION
                    elif section_name == 'terms':
                        return SectionType.TERMS
                    elif section_name == 'claims':
                        return SectionType.CLAIMS
        
        return SectionType.PARAGRAPH
    
    def _is_header(self, text: str) -> bool:
        """Check if text is a header."""
        # Short text in all caps or starting with #
        if len(text) < 100:
            if text.isupper():
                return True
            if text.startswith('#'):
                return True
            for pattern in self.header_patterns:
                if re.match(pattern, text, re.MULTILINE | re.IGNORECASE):
                    return True
        return False
    
    def _is_list(self, text: str) -> bool:
        """Check if text is a list."""
        lines = text.split('\n')
        if len(lines) < 2:
            return False
        
        list_indicators = 0
        for line in lines:
            line = line.strip()
            if re.match(r'^[-•*]\s+', line):
                list_indicators += 1
            elif re.match(r'^\d+[.)]\s+', line):
                list_indicators += 1
            elif re.match(r'^[a-z][.)]\s+', line):
                list_indicators += 1
        
        return list_indicators >= len(lines) * 0.5
    
    def _create_chunks(
        self,
        sections: List[DocumentSection],
        base_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from sections, respecting token limits.
        
        Args:
            sections: List of document sections
            base_metadata: Base metadata to include
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        current_chunk_tokens = 0
        current_chunk_parts = []
        
        for section in sections:
            section_tokens = section.token_count
            
            # Atomic sections (tables, preserved lists) go as-is
            if section.is_atomic:
                # Flush current chunk first
                if current_chunk_parts:
                    chunks.append(self._finalize_chunk(
                        current_chunk_parts,
                        base_metadata,
                        len(chunks)
                    ))
                    current_chunk_parts = []
                    current_chunk_tokens = 0
                
                # Add atomic section as its own chunk
                chunks.append({
                    'text': section.content,
                    'metadata': {
                        **base_metadata,
                        'chunk_id': len(chunks),
                        'section_type': section.section_type.value,
                        'is_table': section.section_type == SectionType.TABLE,
                        'token_count': section_tokens
                    }
                })
                continue
            
            # Check if section fits in current chunk
            if current_chunk_tokens + section_tokens <= self.chunk_size:
                current_chunk_parts.append(section)
                current_chunk_tokens += section_tokens
            
            # Section too large - need to split it
            elif section_tokens > self.chunk_size:
                # Flush current chunk
                if current_chunk_parts:
                    chunks.append(self._finalize_chunk(
                        current_chunk_parts,
                        base_metadata,
                        len(chunks)
                    ))
                    current_chunk_parts = []
                    current_chunk_tokens = 0
                
                # Split large section
                sub_chunks = self._split_large_section(section, base_metadata, len(chunks))
                chunks.extend(sub_chunks)
            
            # Start new chunk
            else:
                if current_chunk_parts:
                    chunks.append(self._finalize_chunk(
                        current_chunk_parts,
                        base_metadata,
                        len(chunks)
                    ))
                
                current_chunk_parts = [section]
                current_chunk_tokens = section_tokens
        
        # Finalize last chunk
        if current_chunk_parts:
            chunks.append(self._finalize_chunk(
                current_chunk_parts,
                base_metadata,
                len(chunks)
            ))
        
        return chunks
    
    def _finalize_chunk(
        self,
        parts: List[DocumentSection],
        base_metadata: Dict[str, Any],
        chunk_id: int
    ) -> Dict[str, Any]:
        """Finalize a chunk from parts."""
        text = '\n\n'.join(p.content for p in parts)
        section_types = list(set(p.section_type.value for p in parts))
        
        return {
            'text': text,
            'metadata': {
                **base_metadata,
                'chunk_id': chunk_id,
                'section_types': section_types,
                'token_count': estimate_tokens(text)
            }
        }
    
    def _split_large_section(
        self,
        section: DocumentSection,
        base_metadata: Dict[str, Any],
        start_id: int
    ) -> List[Dict[str, Any]]:
        """Split a large section into smaller chunks."""
        chunks = []
        text = section.content
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)
            
            if current_tokens + sentence_tokens <= self.chunk_size:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                if current_chunk:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'metadata': {
                            **base_metadata,
                            'chunk_id': start_id + len(chunks),
                            'section_type': section.section_type.value,
                            'token_count': current_tokens
                        }
                    })
                current_chunk = [sentence]
                current_tokens = sentence_tokens
        
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'metadata': {
                    **base_metadata,
                    'chunk_id': start_id + len(chunks),
                    'section_type': section.section_type.value,
                    'token_count': current_tokens
                }
            })
        
        return chunks
    
    def _add_overlap(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add context overlap between chunks."""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks
        
        original_texts = [chunk.get('text', '') for chunk in chunks]
        overlap_prefix = "[...] "
        overlap_separator = "\n\n"
        overhead_tokens = estimate_tokens(f"{overlap_prefix}{overlap_separator}")
        overlap_budget = max(self.chunk_overlap - overhead_tokens, 0)
        
        for i in range(1, len(chunks)):
            # Skip adding overlap to tables
            if chunks[i].get('metadata', {}).get('is_table'):
                continue
            
            prev_text = original_texts[i - 1]
            overlap_text = self._get_overlap_text(prev_text, overlap_budget)
            
            if overlap_text:
                new_text = f"{overlap_prefix}{overlap_text}{overlap_separator}{chunks[i]['text']}"
                chunks[i]['text'] = new_text
                chunks[i]['metadata']['has_overlap'] = True
                chunks[i]['metadata']['token_count'] = estimate_tokens(new_text)
        
        return chunks

    def _get_overlap_text(self, text: str, max_tokens: int) -> str:
        """Extract trailing text up to a token budget."""
        if max_tokens <= 0 or not text:
            return ""
        
        if TIKTOKEN_AVAILABLE:
            try:
                encoder = tiktoken.get_encoding("cl100k_base")
                tokens = encoder.encode(text)
                if not tokens:
                    return ""
                overlap_tokens = tokens[-max_tokens:]
                return encoder.decode(overlap_tokens).strip()
            except Exception:
                pass
        
        words = text.split()
        if not words:
            return ""
        max_words = max(int(max_tokens / 1.3), 1)
        overlap_words = words[-min(max_words, len(words)):]
        return " ".join(overlap_words).strip()


# =============================================================================
# Convenience Functions
# =============================================================================

def chunk_document(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk a document.
    
    Args:
        text: Document text
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap in tokens
        metadata: Optional metadata
        
    Returns:
        List of chunk dictionaries
    """
    chunker = SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return chunker.chunk_document(text, metadata)
