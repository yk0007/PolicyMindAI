"""
Tests for the query filter module.
"""

import pytest
from rag.query_filter import (
    is_query_relevant,
    classify_query,
    QueryRelevance,
    calculate_keyword_score,
    check_off_topic_patterns,
    get_suggested_policy_questions,
)


class TestQueryRelevance:
    """Tests for query relevance classification."""
    
    def test_policy_question_is_relevant(self):
        """Policy-related questions should be classified as relevant."""
        policy_questions = [
            "What is covered under this policy?",
            "Is knee surgery covered?",
            "What are the exclusions?",
            "How do I file a claim?",
            "What is the waiting period for pre-existing conditions?",
            "What is the sum insured?",
            "Is hospitalization covered?",
            "What is the premium amount?",
            "Are dental treatments covered?",
            "What is the deductible?",
        ]
        
        for question in policy_questions:
            is_relevant, message = is_query_relevant(question)
            assert is_relevant, f"Question should be relevant: {question}"
    
    def test_off_topic_question_is_rejected(self):
        """Off-topic questions should be rejected."""
        off_topic_questions = [
            "What is the capital of France?",
            "How do I cook pasta?",
            "Who won the world cup?",
            "Tell me a joke",
            "What's the weather like?",
            "Write me a poem",
            "Translate this to Spanish",
        ]
        
        for question in off_topic_questions:
            is_relevant, message = is_query_relevant(question)
            assert not is_relevant, f"Question should be off-topic: {question}"
    
    def test_empty_query_is_rejected(self):
        """Empty queries should be rejected."""
        is_relevant, message = is_query_relevant("")
        assert not is_relevant
        
        is_relevant, message = is_query_relevant("   ")
        assert not is_relevant
    
    def test_classify_query_returns_correct_types(self):
        """classify_query should return correct tuple structure."""
        relevance, score, message = classify_query("What is covered?")
        
        assert isinstance(relevance, QueryRelevance)
        assert isinstance(score, float)
        assert isinstance(message, str)
        assert 0.0 <= score <= 1.0


class TestKeywordScoring:
    """Tests for keyword-based scoring."""
    
    def test_high_score_for_policy_terms(self):
        """Queries with many policy terms should score high."""
        query = "What is the sum insured for hospitalization coverage in this policy?"
        score = calculate_keyword_score(query)
        assert score >= 0.3, f"Score should be high: {score}"
    
    def test_low_score_for_general_text(self):
        """General text without policy terms should score low."""
        query = "Hello, how are you doing today?"
        score = calculate_keyword_score(query)
        assert score < 0.2, f"Score should be low: {score}"
    
    def test_medium_score_for_partial_match(self):
        """Queries with some policy terms should score medium."""
        query = "What benefits does this offer?"
        score = calculate_keyword_score(query)
        assert 0.1 <= score <= 0.5, f"Score should be medium: {score}"


class TestOffTopicPatterns:
    """Tests for off-topic pattern detection."""
    
    def test_detects_weather_queries(self):
        """Weather-related queries should be detected as off-topic."""
        assert check_off_topic_patterns("What's the weather forecast?")
    
    def test_detects_recipe_queries(self):
        """Recipe queries should be detected as off-topic."""
        assert check_off_topic_patterns("How do I cook chicken?")
    
    def test_does_not_flag_policy_queries(self):
        """Policy queries should not be flagged as off-topic."""
        assert not check_off_topic_patterns("What is covered under this policy?")
        assert not check_off_topic_patterns("How do I file a claim?")


class TestSuggestedQuestions:
    """Tests for suggested questions generation."""
    
    def test_suggested_questions_returns_list(self):
        """get_suggested_policy_questions should return a list."""
        questions = get_suggested_policy_questions()
        assert isinstance(questions, list)
        assert len(questions) > 0
    
    def test_suggested_questions_are_strings(self):
        """All suggested questions should be strings."""
        questions = get_suggested_policy_questions()
        for q in questions:
            assert isinstance(q, str)
            assert len(q) > 0
    
    def test_suggested_questions_end_with_question_mark(self):
        """All suggested questions should end with a question mark."""
        questions = get_suggested_policy_questions()
        for q in questions:
            assert q.endswith("?"), f"Question should end with ?: {q}"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_word_query(self):
        """Single word queries should be handled."""
        is_relevant, _ = is_query_relevant("coverage")
        assert is_relevant  # "coverage" is a policy keyword
        
        is_relevant, _ = is_query_relevant("hello")
        assert not is_relevant
    
    def test_query_with_special_characters(self):
        """Queries with special characters should be handled."""
        is_relevant, _ = is_query_relevant("What's covered? (including surgery)")
        assert is_relevant
    
    def test_very_long_query(self):
        """Very long queries should be handled."""
        long_query = "What " * 100 + "is covered?"
        is_relevant, _ = is_query_relevant(long_query)
        # Should not crash
        assert isinstance(is_relevant, bool)
    
    def test_unicode_query(self):
        """Unicode queries should be handled."""
        is_relevant, _ = is_query_relevant("What is the sum insured (₹)?")
        assert is_relevant
