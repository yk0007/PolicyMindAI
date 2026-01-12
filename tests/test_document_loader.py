"""
Tests for the document loader module.
"""

import pytest
from pathlib import Path
from rag.document_loader import (
    DocumentLoader,
    process_document,
    is_insurance_document,
    clean_text,
    extract_text_from_file,
)
from rag.chunker import estimate_tokens


class TestIsInsuranceDocument:
    """Tests for insurance document classification."""
    
    def test_insurance_text_detected(self, sample_policy_text):
        """Insurance policy text should be detected."""
        assert is_insurance_document(sample_policy_text)
    
    def test_non_insurance_text_rejected(self, sample_non_policy_text):
        """Non-insurance text should be rejected."""
        assert not is_insurance_document(sample_non_policy_text)
    
    def test_empty_text_returns_false(self):
        """Empty text should return False."""
        assert not is_insurance_document("")
        assert not is_insurance_document(None)
    
    def test_threshold_parameter(self, sample_policy_text):
        """Threshold parameter should affect classification."""
        # Very high threshold - should fail
        assert not is_insurance_document(sample_policy_text, threshold=100)
        
        # Very low threshold - should pass even with minimal content
        assert is_insurance_document("This is a policy document", threshold=1)


class TestCleanText:
    """Tests for text cleaning function."""
    
    def test_removes_extra_whitespace(self):
        """Should remove extra whitespace."""
        text = "This   has    extra   spaces"
        cleaned = clean_text(text)
        assert "   " not in cleaned
    
    def test_preserves_paragraph_breaks(self):
        """Should preserve paragraph breaks."""
        text = "Paragraph 1.\n\nParagraph 2."
        cleaned = clean_text(text)
        assert "\n\n" in cleaned
    
    def test_removes_control_characters(self):
        """Should remove control characters."""
        text = "Text with\x00control\x01characters"
        cleaned = clean_text(text)
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned
    
    def test_empty_text_returns_empty(self):
        """Empty text should return empty string."""
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_strips_leading_trailing_whitespace(self):
        """Should strip leading and trailing whitespace."""
        text = "   text with spaces   "
        cleaned = clean_text(text)
        assert cleaned == "text with spaces"


class TestDocumentLoader:
    """Tests for DocumentLoader class."""
    
    def test_loader_initialization(self):
        """Loader should initialize with supported formats."""
        loader = DocumentLoader()
        assert '.pdf' in loader.supported_formats
        assert '.txt' in loader.supported_formats
        assert '.docx' in loader.supported_formats
    
    def test_load_text_file(self, temp_txt):
        """Should load text files successfully."""
        loader = DocumentLoader()
        docs = loader.load_document(temp_txt)
        
        assert len(docs) > 0
        assert docs[0].page_content
        assert docs[0].metadata.get('source') == temp_txt
    
    def test_unsupported_format_raises_error(self, tmp_path):
        """Unsupported format should raise ValueError."""
        loader = DocumentLoader()
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("test content")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load_document(str(unsupported_file))
    
    def test_nonexistent_file_raises_error(self):
        """Non-existent file should raise FileNotFoundError."""
        loader = DocumentLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_document("/nonexistent/path/file.txt")


class TestProcessDocument:
    """Tests for process_document function."""
    
    def test_process_text_file(self, temp_txt):
        """Should process text files into chunks."""
        docs = process_document(temp_txt, chunk_size=500, chunk_overlap=50)
        
        assert len(docs) > 0
        assert all(hasattr(doc, 'page_content') for doc in docs)
        assert all(hasattr(doc, 'metadata') for doc in docs)
    
    def test_metadata_includes_file_info(self, temp_txt):
        """Processed documents should include file metadata."""
        docs = process_document(temp_txt)
        
        assert docs[0].metadata.get('file_name')
        assert docs[0].metadata.get('file_extension')
        assert 'chunk_id' in docs[0].metadata
        assert 'total_chunks' in docs[0].metadata
    
    def test_custom_metadata_preserved(self, temp_txt):
        """Custom metadata should be preserved."""
        custom_metadata = {"custom_key": "custom_value"}
        docs = process_document(temp_txt, metadata=custom_metadata)
        
        assert docs[0].metadata.get('custom_key') == "custom_value"
    
    def test_chunk_size_respected(self, temp_txt):
        """Chunks should respect size limits (approximately)."""
        chunk_size = 200
        chunk_overlap = 50
        docs = process_document(
            temp_txt,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        for doc in docs:
            # Allow some flexibility due to splitting by sentences
            assert estimate_tokens(doc.page_content) <= chunk_size + chunk_overlap
    
    def test_nonexistent_file_raises_error(self):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            process_document("/nonexistent/path/file.txt")


class TestExtractTextFromFile:
    """Tests for extract_text_from_file function."""
    
    def test_extract_from_text_file(self, temp_txt, sample_policy_text):
        """Should extract text from text files."""
        text = extract_text_from_file(temp_txt)
        
        assert text
        assert "HEALTH INSURANCE POLICY" in text
    
    def test_unsupported_format_raises_error(self, tmp_path):
        """Unsupported format should raise ValueError."""
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("test content")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            extract_text_from_file(str(unsupported_file))
