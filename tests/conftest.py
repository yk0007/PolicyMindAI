"""
Pytest configuration and fixtures for PolicyMindAI tests.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_policy_text():
    """Sample insurance policy text for testing."""
    return """
    HEALTH INSURANCE POLICY
    
    Policy Number: HI-2024-001234
    Sum Insured: ₹5,00,000
    Premium: ₹12,000 per annum
    Policy Period: 1 year
    
    COVERAGE:
    This policy covers the following:
    - Hospitalization expenses
    - Pre and post hospitalization
    - Day care procedures
    - Ambulance charges up to ₹2,000
    
    EXCLUSIONS:
    The following are NOT covered:
    - Cosmetic surgery
    - Dental treatment (unless due to accident)
    - Pre-existing diseases (waiting period of 2 years)
    - Self-inflicted injuries
    
    WAITING PERIOD:
    - 30 days initial waiting period
    - 2 years for pre-existing conditions
    - 4 years for specific diseases
    
    CLAIM PROCESS:
    1. Inform TPA within 24 hours
    2. Submit claim form with documents
    3. Claim settlement within 30 days
    """


@pytest.fixture
def sample_non_policy_text():
    """Sample non-insurance text for testing."""
    return """
    CHOCOLATE CAKE RECIPE
    
    Ingredients:
    - 2 cups flour
    - 1 cup sugar
    - 1/2 cup cocoa powder
    - 2 eggs
    - 1 cup milk
    
    Instructions:
    1. Preheat oven to 350°F
    2. Mix dry ingredients
    3. Add wet ingredients
    4. Bake for 30 minutes
    """


@pytest.fixture
def mock_llm():
    """Mock LLM for testing without API calls."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content="This is a test response from the mock LLM."
    )
    return mock


@pytest.fixture
def mock_embeddings():
    """Mock embeddings model for testing."""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1] * 384]  # Fake embeddings
    mock.embed_query.return_value = [0.1] * 384
    return mock


@pytest.fixture
def temp_pdf(tmp_path, sample_policy_text):
    """Create a temporary PDF-like file for testing."""
    # Note: This creates a text file with .pdf extension
    # Real PDF testing would require creating actual PDF files
    pdf_path = tmp_path / "test_policy.pdf"
    pdf_path.write_text(sample_policy_text)
    return str(pdf_path)


@pytest.fixture
def temp_txt(tmp_path, sample_policy_text):
    """Create a temporary text file for testing."""
    txt_path = tmp_path / "test_policy.txt"
    txt_path.write_text(sample_policy_text)
    return str(txt_path)
