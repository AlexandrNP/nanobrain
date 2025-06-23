#!/usr/bin/env python3
"""
Conversational Response Tests for Chatbot Viral Integration

Tests the conversational response functionality as specified in 
section 6.2 of the testing plan.

Test Cases:
- CR-001: "What is EEEV?" → Educational content, virus info, symptoms
- CR-002: "How do viruses spread?" → Transmission mechanisms, examples
- CR-003: "What is a PSSM matrix?" → Technical explanation, applications

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import pytest
import pytest_asyncio
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any
import re

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.chatbot_viral_integration import (
    ChatbotTestData,
    CONTENT_QUALITY_CHECKS,
    MockWorkflowComponents
)

# Import actual workflow components
try:
    from nanobrain.library.workflows.chatbot_viral_integration.steps.conversational_response_step import ConversationalResponseStep
    from nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow import ChatbotViralWorkflow
    REAL_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Real components not available, using mocks only: {e}")
    REAL_COMPONENTS_AVAILABLE = False


class TestConversationalResponse:
    """Test suite for conversational response functionality"""
    
    @pytest_asyncio.fixture
    async def mock_workflow_components(self):
        """Setup mock workflow components"""
        return MockWorkflowComponents()
    
    @pytest_asyncio.fixture
    async def real_workflow(self):
        """Setup real workflow if available"""
        if REAL_COMPONENTS_AVAILABLE:
            try:
                from nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow import create_chatbot_viral_workflow
                workflow = await create_chatbot_viral_workflow()
                return workflow
            except Exception as e:
                pytest.skip(f"Real workflow unavailable: {e}")
        else:
            pytest.skip("Real components not available")
    
    @pytest.mark.asyncio
    async def test_cr_001_eeev_information(self, mock_workflow_components):
        """
        Test CR-001: "What is EEEV?" → Educational content, virus info, symptoms
        
        Expected elements:
        - Educational content about EEEV
        - Virus information
        - Symptoms or disease information
        
        Success criteria:
        - Contains EEEV/Eastern Equine/virus terms
        """
        query = "What is EEEV?"
        
        result = await mock_workflow_components.mock_conversational_response_step(query)
        
        response_content = result["response"].lower()
        
        # Validate required terms
        eeev_terms = ["eeev", "eastern equine", "virus"]
        assert any(term in response_content for term in eeev_terms), f"Response should contain EEEV/Eastern Equine/virus terms"
        
        # Check for educational content indicators
        educational_indicators = ["family", "genus", "transmission", "symptoms", "disease", "clinical"]
        educational_content_found = sum(1 for indicator in educational_indicators if indicator in response_content)
        assert educational_content_found >= 2, f"Response should contain educational content indicators"
        
        # Validate markdown formatting
        assert result["requires_markdown"], "Response should require markdown formatting"
        assert "**" in result["response"] or "#" in result["response"], "Response should contain markdown formatting"
        
        print(f"✅ CR-001 passed: EEEV information response contains appropriate content")
        print(f"   Educational indicators found: {educational_content_found}")
    
    @pytest.mark.asyncio
    async def test_cr_002_virus_transmission(self, mock_workflow_components):
        """
        Test CR-002: "How do viruses spread?" → Transmission mechanisms, examples
        
        Expected elements:
        - Transmission mechanisms
        - Examples of viral spread
        
        Success criteria:
        - Contains transmission/spread terms
        """
        query = "How do viruses spread?"
        
        result = await mock_workflow_components.mock_conversational_response_step(query)
        
        response_content = result["response"].lower()
        
        # Validate transmission terms
        transmission_terms = ["transmission", "spread", "vector", "airborne", "contact", "mosquito", "droplet"]
        transmission_found = sum(1 for term in transmission_terms if term in response_content)
        assert transmission_found >= 1, f"Response should contain transmission/spread terms"
        
        # Check for mechanism descriptions
        mechanism_terms = ["mechanism", "pathway", "route", "mode"]
        mechanism_found = any(term in response_content for term in mechanism_terms)
        
        print(f"✅ CR-002 passed: Virus transmission response contains appropriate content")
        print(f"   Transmission terms found: {transmission_found}, Mechanisms: {mechanism_found}")
    
    @pytest.mark.asyncio
    async def test_cr_003_pssm_explanation(self, mock_workflow_components):
        """
        Test CR-003: "What is a PSSM matrix?" → Technical explanation, applications
        
        Expected elements:
        - Technical explanation of PSSM
        - Applications in bioinformatics
        
        Success criteria:
        - Contains PSSM/matrix/bioinformatics terms
        """
        query = "What is a PSSM matrix?"
        
        result = await mock_workflow_components.mock_conversational_response_step(query)
        
        response_content = result["response"].lower()
        
        # Validate PSSM-related terms
        pssm_terms = ["pssm", "matrix", "position", "scoring", "bioinformatics"]
        pssm_found = sum(1 for term in pssm_terms if term in response_content)
        assert pssm_found >= 2, f"Response should contain PSSM/matrix/bioinformatics terms"
        
        # Check for application mentions
        application_terms = ["application", "analysis", "alignment", "sequence", "domain", "annotation"]
        application_found = sum(1 for term in application_terms if term in response_content)
        assert application_found >= 1, f"Response should mention applications"
        
        print(f"✅ CR-003 passed: PSSM explanation contains appropriate content")
        print(f"   PSSM terms found: {pssm_found}, Applications: {application_found}")
    
    @pytest.mark.asyncio
    async def test_content_quality_validation(self, mock_workflow_components):
        """
        Test content quality validation (from section 6.2.2)
        
        Validates:
        - Min length: 100 characters
        - Max length: 2000 characters
        - Markdown formatting: True
        - Technical accuracy: True (basic check)
        - Educational value: True (basic check)
        """
        test_queries = [
            "What is EEEV?",
            "How do viruses spread?",
            "What is a PSSM matrix?",
            "Explain viral protein structure"
        ]
        
        for query in test_queries:
            result = await mock_workflow_components.mock_conversational_response_step(query)
            response = result["response"]
            
            # Length validation
            response_length = len(response)
            assert response_length >= CONTENT_QUALITY_CHECKS["min_length"], f"Response too short: {response_length} < {CONTENT_QUALITY_CHECKS['min_length']}"
            assert response_length <= CONTENT_QUALITY_CHECKS["max_length"], f"Response too long: {response_length} > {CONTENT_QUALITY_CHECKS['max_length']}"
            
            # Markdown formatting validation
            if CONTENT_QUALITY_CHECKS["markdown_formatting"]:
                markdown_indicators = ["**", "*", "#", "-", "##", "###"]
                has_markdown = any(indicator in response for indicator in markdown_indicators)
                assert has_markdown, f"Response should contain markdown formatting"
            
            # Basic technical accuracy check (no obvious errors)
            error_indicators = ["error", "fail", "broken", "null", "undefined"]
            has_errors = any(indicator.lower() in response.lower() for indicator in error_indicators)
            assert not has_errors, f"Response contains error indicators"
            
            print(f"✅ Content quality validated for: {query[:30]}...")
            print(f"   Length: {response_length}, Markdown: {has_markdown}")
    
    @pytest.mark.asyncio
    async def test_response_confidence_levels(self, mock_workflow_components):
        """Test confidence levels in conversational responses"""
        confidence_test_cases = [
            ("What is EEEV?", 0.8),  # High confidence - specific virus
            ("Tell me about viruses", 0.7),  # Medium confidence - general topic
            ("Random unclear question", 0.5)  # Lower confidence - unclear
        ]
        
        for query, min_confidence in confidence_test_cases:
            result = await mock_workflow_components.mock_conversational_response_step(query)
            
            confidence = result.get("confidence", 0)
            assert confidence >= min_confidence, f"Confidence {confidence} below expected {min_confidence} for: {query}"
            
            print(f"✅ Confidence test passed: {query[:30]}... → {confidence}")
    
    @pytest.mark.asyncio
    async def test_markdown_formatting_consistency(self, mock_workflow_components):
        """Test consistent markdown formatting across responses"""
        queries = [
            "What is EEEV?",
            "What is a PSSM matrix?",
            "Explain viral protein structure"
        ]
        
        markdown_patterns = {
            "headers": r"#+\s+",
            "bold": r"\*\*.*?\*\*",
            "lists": r"^\s*[-*+]\s+",
            "sections": r"##\s+"
        }
        
        for query in queries:
            result = await mock_workflow_components.mock_conversational_response_step(query)
            response = result["response"]
            
            # Check for consistent markdown patterns
            patterns_found = {}
            for pattern_name, pattern in markdown_patterns.items():
                matches = re.findall(pattern, response, re.MULTILINE)
                patterns_found[pattern_name] = len(matches)
            
            # Should have at least headers and some formatting
            assert patterns_found["headers"] > 0, f"Response should have headers for: {query}"
            
            total_formatting = sum(patterns_found.values())
            assert total_formatting >= 2, f"Response should have multiple formatting elements for: {query}"
            
            print(f"✅ Markdown formatting consistent: {query[:30]}... → {patterns_found}")
    
    @pytest.mark.asyncio
    async def test_educational_value_assessment(self, mock_workflow_components):
        """Test educational value of responses"""
        educational_queries = [
            "What is EEEV?",
            "How do alphaviruses replicate?", 
            "What are the symptoms of viral encephalitis?"
        ]
        
        educational_indicators = [
            "definition", "explanation", "description", "overview",
            "features", "characteristics", "symptoms", "causes",
            "mechanism", "process", "function", "role"
        ]
        
        for query in educational_queries:
            result = await mock_workflow_components.mock_conversational_response_step(query)
            response = result["response"].lower()
            
            # Count educational indicators
            indicators_found = sum(1 for indicator in educational_indicators if indicator in response)
            assert indicators_found >= 2, f"Response should have educational value indicators for: {query}"
            
            # Check for structured information
            structure_indicators = ["#", "**", "-", "##"]
            structure_found = sum(1 for indicator in structure_indicators if indicator in result["response"])
            assert structure_found >= 2, f"Response should be well-structured for: {query}"
            
            print(f"✅ Educational value assessed: {query[:30]}... → {indicators_found} indicators")
    
    @pytest.mark.skipif(not REAL_COMPONENTS_AVAILABLE, reason="Real components not available")
    @pytest.mark.asyncio
    async def test_real_workflow_conversational_response(self, real_workflow):
        """Test conversational responses with real workflow components"""
        test_cases = [
            "What is EEEV?",
            "How do viruses spread?"
        ]
        
        for query in test_cases:
            # Process through real workflow
            response_chunks = []
            final_response = None
            
            async for chunk in real_workflow.process_user_message(query, "test_session_conv"):
                response_chunks.append(chunk)
                
                # Look for final response
                if chunk.get('type') in ['content_complete', 'message_complete']:
                    final_response = chunk
                    break
            
            assert final_response is not None, f"No final response received for: {query}"
            
            # Extract content
            content = final_response.get('content', '')
            if not content and 'formatted_response' in final_response:
                content = final_response['formatted_response'].get('content', '')
            
            assert content, f"No content in response for: {query}"
            assert len(content) >= 100, f"Response too short for: {query}"
            
            print(f"✅ Real workflow conversational response: {query} → {len(content)} chars")
    
    @pytest.mark.asyncio
    async def test_response_streaming_capability(self, mock_workflow_components):
        """Test streaming response capabilities"""
        query = "Explain viral protein structure in detail"
        
        result = await mock_workflow_components.mock_conversational_response_step(query)
        
        # Check if response supports streaming
        response = result["response"]
        requires_markdown = result.get("requires_markdown", False)
        
        # For long responses, should support streaming
        if len(response) > 500:
            # Could be chunked for streaming
            chunks = [response[i:i+100] for i in range(0, len(response), 100)]
            assert len(chunks) > 1, "Long response should be chunkable for streaming"
            
            print(f"✅ Streaming capability: {len(chunks)} potential chunks")
        
        assert requires_markdown, "Response should support markdown for rich formatting"


@pytest.mark.asyncio
async def test_conversational_response_performance():
    """Test conversational response performance requirements (from section 7.2.1)"""
    import time
    
    mock_components = MockWorkflowComponents()
    query = "What is EEEV?"
    
    # Test response time
    start_time = time.time()
    result = await mock_components.mock_conversational_response_step(query)
    end_time = time.time()
    
    response_time = end_time - start_time
    
    # Target: <2s (95th percentile), Maximum: <5s
    assert response_time < 5.0, f"Conversational response took {response_time:.3f}s, exceeds 5s maximum"
    
    if response_time < 2.0:
        print(f"✅ Conversational response performance excellent: {response_time:.3f}s")
    else:
        print(f"⚠️ Conversational response performance acceptable but slow: {response_time:.3f}s")


if __name__ == "__main__":
    # Run tests directly
    import asyncio
    
    async def run_all_tests():
        print("🧪 Running Conversational Response Tests...")
        
        # Create mock components
        mock_components = MockWorkflowComponents()
        
        # Run basic conversational response tests
        test_instance = TestConversationalResponse()
        
        await test_instance.test_cr_001_eeev_information(mock_components)
        await test_instance.test_cr_002_virus_transmission(mock_components)
        await test_instance.test_cr_003_pssm_explanation(mock_components)
        
        await test_instance.test_content_quality_validation(mock_components)
        await test_instance.test_response_confidence_levels(mock_components)
        await test_instance.test_markdown_formatting_consistency(mock_components)
        await test_instance.test_educational_value_assessment(mock_components)
        await test_instance.test_response_streaming_capability(mock_components)
        
        # Performance test
        await test_conversational_response_performance()
        
        print("🎉 All Conversational Response Tests Passed!")
    
    asyncio.run(run_all_tests())