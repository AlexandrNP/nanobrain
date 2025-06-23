#!/usr/bin/env python3
"""
Chatbot Web Application Integration Test Suite

Tests the complete end-to-end functionality to ensure:
1. "What is EEEV?" → Conversational response with educational content
2. "Create PSSM matrix of EEEV" → Annotation workflow with PSSM matrix in JSON format

Author: NanoBrain Development Team  
Date: January 2025
Version: 1.0.0
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List
import pytest
import re

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow import ChatbotViralWorkflow
from nanobrain.library.workflows.chatbot_viral_integration.steps.query_classification_step import QueryClassificationStep
from nanobrain.library.workflows.chatbot_viral_integration.steps.response_formatting_step import ResponseFormattingStep
from nanobrain.library.infrastructure.data.chat_session_data import ChatSessionData
from nanobrain.core.step import StepConfig


class TestChatbotWebApplicationIntegration:
    """Comprehensive integration tests for chatbot web application"""
    
    @pytest.fixture
    async def chatbot_workflow(self):
        """Initialize chatbot workflow for testing"""
        workflow = ChatbotViralWorkflow()
        await workflow.initialize()
        return workflow
    
    @pytest.fixture
    def test_session_data(self):
        """Create test session data"""
        return ChatSessionData(session_id="test_session_integration")
    
    @pytest.mark.asyncio
    async def test_conversational_query_eeev(self, chatbot_workflow, test_session_data):
        """
        Test: "What is EEEV?" → Conversational response
        
        Validates:
        - Query classification as conversational
        - Educational content about EEEV
        - Proper markdown formatting
        - No annotation workflow triggered
        """
        query = "What is EEEV?"
        print(f"\n🧪 Testing conversational query: '{query}'")
        
        response_chunks = []
        classification_result = None
        final_response = None
        
        # Process query through workflow
        async for chunk in chatbot_workflow.process_user_message(query, test_session_data.session_id):
            response_chunks.append(chunk)
            
            # Capture classification
            if chunk.get('type') == 'classification':
                classification_result = chunk
            
            # Capture final response
            elif chunk.get('type') in ['content_complete', 'response', 'message_complete']:
                final_response = chunk
        
        # Validate classification
        assert classification_result is not None, "No classification result received"
        assert classification_result.get('intent') == 'conversational', f"Expected conversational intent, got {classification_result.get('intent')}"
        assert classification_result.get('routing') == 'conversational_response', f"Expected conversational routing, got {classification_result.get('routing')}"
        
        # Validate final response
        assert final_response is not None, "No final response received"
        
        # Extract content
        content = final_response.get('content', '')
        if not content:
            formatted_response = final_response.get('formatted_response', {})
            content = formatted_response.get('content', '')
        
        assert content, "No content in response"
        
        # Validate educational content about EEEV
        content_lower = content.lower()
        assert any(term in content_lower for term in ['eeev', 'eastern equine', 'encephalitis']), "Response should contain EEEV information"
        assert any(term in content_lower for term in ['virus', 'viral', 'alphavirus']), "Response should contain viral information"
        assert any(term in content_lower for term in ['disease', 'symptom', 'infection', 'pathogen']), "Response should contain disease information"
        
        # Validate markdown formatting
        assert '**' in content or '*' in content, "Response should contain markdown formatting"
        
        # Validate no annotation workflow triggered
        assert not any(chunk.get('type') == 'progress' for chunk in response_chunks), "No annotation workflow should be triggered"
        assert not any('job_id' in chunk for chunk in response_chunks), "No job should be created"
        
        print("✅ Conversational query test passed")
        return content
    
    @pytest.mark.asyncio 
    async def test_annotation_query_eeev_pssm(self, chatbot_workflow, test_session_data):
        """
        Test: "Create PSSM matrix of EEEV" → Annotation workflow with PSSM JSON
        
        Validates:
        - Query classification as annotation
        - Annotation workflow triggered
        - Progress tracking
        - PSSM matrix in JSON format
        - Workflow results overview
        """
        query = "Create PSSM matrix of EEEV"
        print(f"\n🧪 Testing annotation query: '{query}'")
        
        response_chunks = []
        classification_result = None
        progress_updates = []
        final_response = None
        job_id = None
        
        # Process query through workflow
        async for chunk in chatbot_workflow.process_user_message(query, test_session_data.session_id):
            response_chunks.append(chunk)
            
            # Capture classification
            if chunk.get('type') == 'classification':
                classification_result = chunk
            
            # Capture progress updates
            elif chunk.get('type') == 'progress':
                progress_updates.append(chunk)
                if chunk.get('job_id'):
                    job_id = chunk.get('job_id')
            
            # Capture final response
            elif chunk.get('type') in ['content_complete', 'response', 'message_complete']:
                final_response = chunk
                if chunk.get('job_id'):
                    job_id = chunk.get('job_id')
        
        # Validate classification
        assert classification_result is not None, "No classification result received"
        assert classification_result.get('intent') == 'annotation', f"Expected annotation intent, got {classification_result.get('intent')}"
        assert classification_result.get('routing') == 'annotation_job', f"Expected annotation routing, got {classification_result.get('routing')}"
        
        # Validate annotation workflow triggered
        assert len(progress_updates) > 0, "Annotation workflow should generate progress updates"
        assert job_id is not None, "A job ID should be generated"
        
        # Validate progress tracking
        for progress_chunk in progress_updates:
            assert 'progress' in progress_chunk, "Progress chunks should contain progress percentage"
            assert 'job_id' in progress_chunk, "Progress chunks should contain job ID"
            assert progress_chunk.get('progress') >= 0, "Progress should be non-negative"
            assert progress_chunk.get('progress') <= 100, "Progress should not exceed 100%"
        
        # Validate final response
        assert final_response is not None, "No final response received"
        
        # Extract content
        content = final_response.get('content', '')
        if not content:
            formatted_response = final_response.get('formatted_response', {})
            content = formatted_response.get('content', '')
        
        assert content, "No content in final response"
        
        # Validate workflow results overview
        content_lower = content.lower()
        assert any(term in content_lower for term in ['completed', 'analysis', 'results']), "Response should indicate completion"
        assert 'job id' in content_lower or 'job_id' in content_lower, "Response should contain job ID reference"
        assert any(term in content_lower for term in ['pssm', 'matrix', 'profile']), "Response should mention PSSM matrix"
        
        # Validate PSSM matrix in JSON format
        json_pattern = r'```json\s*(.*?)\s*```'
        json_matches = re.findall(json_pattern, content, re.DOTALL)
        
        assert len(json_matches) > 0, "Response should contain JSON formatted data"
        
        # Validate JSON structure
        json_content = json_matches[0]
        try:
            pssm_data = json.loads(json_content)
            assert isinstance(pssm_data, (dict, list)), "JSON should contain valid PSSM data structure"
            
            # Check for PSSM matrix structure
            if isinstance(pssm_data, dict):
                assert any(key in pssm_data for key in ['matrix', 'pssm_matrix', 'matrix_data']), "JSON should contain matrix data"
            
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON format in response: {e}")
        
        # Validate markdown formatting
        assert '**' in content or '*' in content, "Response should contain markdown formatting"
        assert '#' in content, "Response should contain markdown headers"
        
        print("✅ Annotation query test passed")
        return content, job_id, pssm_data
    
    @pytest.mark.asyncio
    async def test_query_classification_accuracy(self):
        """Test query classification accuracy for various query patterns"""
        
        test_queries = [
            # Conversational queries
            {"query": "What is EEEV?", "expected": "conversational"},
            {"query": "What is Eastern Equine Encephalitis virus?", "expected": "conversational"},
            {"query": "Tell me about EEEV", "expected": "conversational"},
            {"query": "How does EEEV replicate?", "expected": "conversational"},
            {"query": "What are the symptoms of EEEV?", "expected": "conversational"},
            {"query": "Explain EEEV pathogenesis", "expected": "conversational"},
            
            # Annotation queries
            {"query": "Create PSSM matrix of EEEV", "expected": "annotation"},
            {"query": "Generate PSSM profile for EEEV proteins", "expected": "annotation"},
            {"query": "Build PSSM matrix for Eastern Equine Encephalitis virus", "expected": "annotation"},
            {"query": "Analyze EEEV protein sequences", "expected": "annotation"},
            {"query": "Create PSSM matrix for EEEV", "expected": "annotation"},
            {"query": "Generate protein profile for Eastern Equine Encephalitis", "expected": "annotation"},
        ]
        
        # Initialize classification step
        config = StepConfig(
            name="test_classification",
            description="Test query classification",
            step_id="test_classification",
            config={}
        )
        
        classification_step = QueryClassificationStep(config)
        session_data = ChatSessionData(session_id="test_classification")
        
        results = []
        
        for test_case in test_queries:
            query = test_case["query"]
            expected = test_case["expected"]
            
            input_data = {
                'user_query': query,
                'session_data': session_data
            }
            
            result = await classification_step.process(input_data)
            
            if result['success']:
                classification_data = result['classification_data']
                routing_decision = result['routing_decision']
                
                intent = classification_data.intent
                next_step = routing_decision.get('next_step', 'unknown')
                
                # Determine actual routing
                actual_routing = "conversational" if next_step == "conversational_response" else "annotation"
                
                is_correct = actual_routing == expected
                
                results.append({
                    'query': query,
                    'expected': expected,
                    'actual': actual_routing,
                    'intent': intent,
                    'confidence': classification_data.confidence,
                    'correct': is_correct
                })
            else:
                results.append({
                    'query': query,
                    'expected': expected,
                    'actual': 'error',
                    'intent': 'error',
                    'confidence': 0.0,
                    'correct': False
                })
        
        # Calculate accuracy
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['correct'])
        accuracy = passed_tests / total_tests * 100
        
        print(f"\n📊 Classification Accuracy: {accuracy:.1f}% ({passed_tests}/{total_tests})")
        
        # Print failed cases
        failed_cases = [r for r in results if not r['correct']]
        if failed_cases:
            print("\n❌ Failed classifications:")
            for case in failed_cases:
                print(f"  '{case['query'][:50]}...' → {case['actual']} (expected: {case['expected']}, conf: {case['confidence']:.3f})")
        
        # Require high accuracy for critical queries
        critical_queries = ["What is EEEV?", "Create PSSM matrix of EEEV"]
        critical_results = [r for r in results if r['query'] in critical_queries]
        critical_accuracy = sum(1 for r in critical_results if r['correct']) / len(critical_results) * 100
        
        assert critical_accuracy == 100.0, f"Critical queries must have 100% accuracy, got {critical_accuracy:.1f}%"
        assert accuracy >= 80.0, f"Overall accuracy must be >= 80%, got {accuracy:.1f}%"
        
        return results
    
    @pytest.mark.asyncio
    async def test_response_formatting_validation(self):
        """Test response formatting for both conversational and annotation responses"""
        
        # Initialize formatting step
        config = StepConfig(
            name="test_formatting",
            description="Test response formatting",
            step_id="test_formatting",
            config={}
        )
        
        formatting_step = ResponseFormattingStep(config)
        
        # Test conversational response formatting
        from nanobrain.library.infrastructure.data.chat_session_data import ConversationalResponseData
        
        conversational_data = ConversationalResponseData(
            query="What is EEEV?",
            response="Eastern Equine Encephalitis Virus (EEEV) is a mosquito-borne alphavirus...",
            response_type='educational',
            confidence=0.9,
            topic_area='virus_biology'
        )
        
        conversational_input = {
            'response_type': 'conversational',
            'response_data': conversational_data
        }
        
        conversational_result = await formatting_step.process(conversational_input)
        
        assert conversational_result['success'], "Conversational formatting should succeed"
        formatted_conv = conversational_result['formatted_response']
        assert formatted_conv['requires_markdown'], "Conversational response should require markdown"
        assert formatted_conv['message_type'] == 'conversational', "Message type should be conversational"
        
        # Test annotation response formatting with PSSM
        from nanobrain.library.infrastructure.data.chat_session_data import AnnotationJobData
        
        # Create mock PSSM results
        mock_pssm_results = {
            'pssm_matrix': {
                'organism': 'Eastern Equine Encephalitis Virus',
                'protein_count': 5,
                'format': 'JSON',
                'matrix_data': {
                    'matrix': [[1.5, 0.2, -0.3], [0.8, 1.2, -0.1]],
                    'positions': ['1', '2'],
                    'amino_acids': ['A', 'R', 'N']
                }
            },
            'workflow_summary': 'PSSM matrix generated successfully for EEEV proteins'
        }
        
        job_data = AnnotationJobData(
            job_id="test_job_123",
            session_id="test_session",
            user_query="Create PSSM matrix of EEEV",
            status='completed',
            success=True,
            results=mock_pssm_results,
            processing_time_ms=15000
        )
        
        annotation_input = {
            'response_type': 'annotation_job',
            'job_data': job_data,
            'session_data': ChatSessionData(session_id="test_session")
        }
        
        annotation_result = await formatting_step.process(annotation_input)
        
        assert annotation_result['success'], "Annotation formatting should succeed"
        formatted_annot = annotation_result['formatted_response']
        assert formatted_annot['requires_markdown'], "Annotation response should require markdown"
        assert formatted_annot['status'] == 'completed', "Status should be completed"
        
        # Validate PSSM JSON format in content
        content = formatted_annot['content']
        assert 'PSSM Matrix Analysis Completed' in content, "Response should indicate PSSM completion"
        assert '```json' in content, "Response should contain JSON code block"
        assert 'matrix' in content.lower(), "Response should contain matrix data"
        
        print("✅ Response formatting validation passed")
        return formatted_conv, formatted_annot


# Standalone test runner
async def run_integration_tests():
    """Run integration tests standalone"""
    print("🚀 Running Chatbot Web Application Integration Tests")
    print("=" * 60)
    
    test_instance = TestChatbotWebApplicationIntegration()
    
    try:
        # Initialize workflow
        chatbot_workflow = ChatbotViralWorkflow()
        await chatbot_workflow.initialize()
        test_session_data = ChatSessionData(session_id="standalone_test")
        
        print("\n📋 Test Suite Overview:")
        print("1. Conversational Query Test (What is EEEV?)")
        print("2. Annotation Query Test (Create PSSM matrix of EEEV)")
        print("3. Query Classification Accuracy")
        print("4. Response Formatting Validation")
        
        # Run tests
        results = {}
        
        print(f"\n{'='*60}")
        print("🧪 TEST 1: Conversational Query")
        results['conversational'] = await test_instance.test_conversational_query_eeev(chatbot_workflow, test_session_data)
        
        print(f"\n{'='*60}")
        print("🧪 TEST 2: Annotation Query")
        results['annotation'] = await test_instance.test_annotation_query_eeev_pssm(chatbot_workflow, test_session_data)
        
        print(f"\n{'='*60}")
        print("🧪 TEST 3: Query Classification Accuracy")
        results['classification'] = await test_instance.test_query_classification_accuracy()
        
        print(f"\n{'='*60}")
        print("🧪 TEST 4: Response Formatting Validation")
        results['formatting'] = await test_instance.test_response_formatting_validation()
        
        print(f"\n{'='*60}")
        print("📊 FINAL TEST RESULTS")
        print("✅ All tests passed successfully!")
        print("\nThe chatbot web application demonstrates the exact required behavior:")
        print("- 'What is EEEV?' → Conversational response with educational content")
        print("- 'Create PSSM matrix of EEEV' → Annotation workflow with PSSM matrix in JSON format")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(run_integration_tests()) 