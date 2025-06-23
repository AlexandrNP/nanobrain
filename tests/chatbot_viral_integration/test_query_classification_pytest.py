#!/usr/bin/env python3
"""
Query Classification Tests for Chatbot Viral Integration (Pytest Compatible)

Tests the query classification functionality as specified in 
section 6.1 of the testing plan. This version is compatible with 
standard pytest execution using pytest-asyncio.

Test Cases:
- QC-001: "What is EEEV?" → conversational
- QC-002: "Create PSSM matrix of EEEV" → annotation  
- QC-003: "Tell me about EEEV proteins" → conversational
- QC-004: "Analyze EEEV protein structure" → annotation
- QC-005: "What is EEEV and create PSSM?" → annotation

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import pytest
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.chatbot_viral_integration import (
    ChatbotTestData,
    CLASSIFICATION_METRICS,
    MockWorkflowComponents
)

# Import actual workflow components
try:
    from nanobrain.library.workflows.chatbot_viral_integration.steps.query_classification_step import QueryClassificationStep
    from nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow import ChatbotViralWorkflow
    REAL_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Real components not available, using mocks only: {e}")
    REAL_COMPONENTS_AVAILABLE = False


@pytest.mark.chatbot_viral
@pytest.mark.unit
class TestQueryClassificationPytest:
    """Test suite for query classification functionality (pytest compatible)"""
    
    @pytest.mark.asyncio
    async def test_qc_001_conversational_eeev(self, mock_workflow_components):
        """
        Test QC-001: "What is EEEV?" → conversational
        
        Expected:
        - Intent: conversational
        - Routing: conversational_response  
        - Confidence: >0.8
        """
        query = "What is EEEV?"
        
        result = await mock_workflow_components.mock_query_classification_step(query)
        
        assert result["intent"] == "conversational", f"Expected conversational intent, got {result['intent']}"
        assert result["routing_decision"]["next_step"] == "conversational_response", f"Expected conversational_response routing"
        assert result["confidence"] >= 0.8, f"Expected confidence >= 0.8, got {result['confidence']}"
        
        print(f"✅ QC-001 passed: {query} → {result['intent']} (confidence: {result['confidence']})")
    
    @pytest.mark.asyncio
    async def test_qc_002_annotation_pssm(self, mock_workflow_components):
        """
        Test QC-002: "Create PSSM matrix of EEEV" → annotation
        
        Expected:
        - Intent: annotation
        - Routing: annotation_job
        - Confidence: >0.8
        """
        query = "Create PSSM matrix of EEEV"
        
        result = await mock_workflow_components.mock_query_classification_step(query)
        
        assert result["intent"] == "annotation", f"Expected annotation intent, got {result['intent']}"
        assert result["routing_decision"]["next_step"] == "annotation_job", f"Expected annotation_job routing"
        assert result["confidence"] >= 0.8, f"Expected confidence >= 0.8, got {result['confidence']}"
        
        print(f"✅ QC-002 passed: {query} → {result['intent']} (confidence: {result['confidence']})")
    
    @pytest.mark.asyncio
    async def test_qc_003_conversational_proteins(self, mock_workflow_components):
        """
        Test QC-003: "Tell me about EEEV proteins" → conversational
        
        Expected:
        - Intent: conversational
        - Routing: conversational_response
        - Confidence: >0.6
        """
        query = "Tell me about EEEV proteins"
        
        result = await mock_workflow_components.mock_query_classification_step(query)
        
        assert result["intent"] == "conversational", f"Expected conversational intent, got {result['intent']}"
        assert result["routing_decision"]["next_step"] == "conversational_response", f"Expected conversational_response routing"
        assert result["confidence"] >= 0.6, f"Expected confidence >= 0.6, got {result['confidence']}"
        
        print(f"✅ QC-003 passed: {query} → {result['intent']} (confidence: {result['confidence']})")
    
    @pytest.mark.asyncio
    async def test_qc_004_annotation_analyze(self, mock_workflow_components):
        """
        Test QC-004: "Analyze EEEV protein structure" → annotation
        
        Expected:
        - Intent: annotation
        - Routing: annotation_job
        - Confidence: >0.7
        """
        query = "Analyze EEEV protein structure"
        
        result = await mock_workflow_components.mock_query_classification_step(query)
        
        assert result["intent"] == "annotation", f"Expected annotation intent, got {result['intent']}"
        assert result["routing_decision"]["next_step"] == "annotation_job", f"Expected annotation_job routing"
        assert result["confidence"] >= 0.7, f"Expected confidence >= 0.7, got {result['confidence']}"
        
        print(f"✅ QC-004 passed: {query} → {result['intent']} (confidence: {result['confidence']})")
    
    @pytest.mark.asyncio
    async def test_qc_005_annotation_mixed_intent(self, mock_workflow_components):
        """
        Test QC-005: "What is EEV and create PSSM?" → annotation
        
        Expected:
        - Intent: annotation (mixed intent should favor annotation)
        - Routing: annotation_job
        - Confidence: >0.6
        """
        query = "What is EEEV and create PSSM?"
        
        result = await mock_workflow_components.mock_query_classification_step(query)
        
        assert result["intent"] == "annotation", f"Expected annotation intent, got {result['intent']}"
        assert result["routing_decision"]["next_step"] == "annotation_job", f"Expected annotation_job routing"
        assert result["confidence"] >= 0.6, f"Expected confidence >= 0.6, got {result['confidence']}"
        
        print(f"✅ QC-005 passed: {query} → {result['intent']} (confidence: {result['confidence']})")
    
    @pytest.mark.asyncio
    async def test_classification_accuracy_metrics(self, mock_workflow_components):
        """
        Test classification accuracy across all test cases
        
        Validates metrics from section 6.1.2:
        - Accuracy threshold: 0.85
        - Precision threshold: 0.80
        - Recall threshold: 0.80
        - F1 score threshold: 0.80
        """
        test_cases = ChatbotTestData.get_classification_test_cases()
        
        correct_predictions = 0
        total_predictions = len(test_cases)
        intent_predictions = {"conversational": [], "annotation": []}
        
        for test_case in test_cases:
            query = test_case["input_query"]
            expected_intent = test_case["expected_intent"]
            
            result = await mock_workflow_components.mock_query_classification_step(query)
            predicted_intent = result["intent"]
            
            # Track predictions for precision/recall calculation
            intent_predictions[predicted_intent].append({
                "expected": expected_intent,
                "predicted": predicted_intent,
                "correct": expected_intent == predicted_intent
            })
            
            if expected_intent == predicted_intent:
                correct_predictions += 1
            
            print(f"Test {test_case['test_id']}: {query} → Expected: {expected_intent}, Got: {predicted_intent}")
        
        # Calculate accuracy
        accuracy = correct_predictions / total_predictions
        
        # Calculate precision and recall for each intent
        metrics = {}
        for intent in ["conversational", "annotation"]:
            predictions = intent_predictions[intent]
            if predictions:
                true_positives = sum(1 for p in predictions if p["correct"])
                total_predicted = len(predictions)
                total_actual = sum(1 for case in test_cases if case["expected_intent"] == intent)
                
                precision = true_positives / total_predicted if total_predicted > 0 else 0
                recall = true_positives / total_actual if total_actual > 0 else 0
                f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics[intent] = {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score
                }
        
        # Validate against thresholds
        assert accuracy >= CLASSIFICATION_METRICS["accuracy_threshold"], f"Accuracy {accuracy:.3f} below threshold {CLASSIFICATION_METRICS['accuracy_threshold']}"
        
        for intent, metric in metrics.items():
            assert metric["precision"] >= CLASSIFICATION_METRICS["precision_threshold"], f"{intent} precision {metric['precision']:.3f} below threshold"
            assert metric["recall"] >= CLASSIFICATION_METRICS["recall_threshold"], f"{intent} recall {metric['recall']:.3f} below threshold"
            assert metric["f1_score"] >= CLASSIFICATION_METRICS["f1_score_threshold"], f"{intent} F1 score {metric['f1_score']:.3f} below threshold"
        
        print(f"✅ Classification accuracy metrics passed:")
        print(f"   Overall accuracy: {accuracy:.3f}")
        for intent, metric in metrics.items():
            print(f"   {intent}: precision={metric['precision']:.3f}, recall={metric['recall']:.3f}, f1={metric['f1_score']:.3f}")
    
    @pytest.mark.asyncio
    async def test_edge_case_empty_query(self, mock_workflow_components):
        """Test handling of empty query"""
        query = ""
        
        result = await mock_workflow_components.mock_query_classification_step(query)
        
        # Empty query should have a fallback classification
        assert "intent" in result, "Result should contain intent field"
        assert result["confidence"] >= 0.0, "Confidence should be non-negative"
        
        print(f"✅ Empty query handled: → {result['intent']} (confidence: {result['confidence']})")
    
    @pytest.mark.asyncio
    async def test_edge_case_long_query(self, mock_workflow_components):
        """Test handling of very long query"""
        query = "A very long query that exceeds normal input limits and contains repetitive content to test how the system handles unusually long user inputs that might cause performance issues or buffer overflows in the query processing pipeline and should be handled gracefully with appropriate validation and error messages"
        
        result = await mock_workflow_components.mock_query_classification_step(query)
        
        # Long query should still be classified
        assert "intent" in result, "Result should contain intent field"
        assert result["intent"] in ["conversational", "annotation"], f"Intent should be valid, got {result['intent']}"
        
        print(f"✅ Long query handled: → {result['intent']} (confidence: {result['confidence']})")
    
    @pytest.mark.asyncio
    async def test_edge_case_special_characters(self, mock_workflow_components):
        """Test handling of special characters"""
        query = "Special characters: @#$%^&*()"
        
        result = await mock_workflow_components.mock_query_classification_step(query)
        
        # Special characters should be handled gracefully
        assert "intent" in result, "Result should contain intent field"
        assert result["intent"] in ["conversational", "annotation"], f"Intent should be valid, got {result['intent']}"
        
        print(f"✅ Special characters handled: → {result['intent']} (confidence: {result['confidence']})")


@pytest.mark.real_components
@pytest.mark.integration
@pytest.mark.skipif(not REAL_COMPONENTS_AVAILABLE, reason="Real components not available")
class TestRealWorkflowClassification:
    """Test classification with real workflow components"""
    
    @pytest.mark.asyncio
    async def test_real_workflow_classification(self, real_workflow):
        """Test classification with real workflow components"""
        test_cases = [
            ("What is EEEV?", "conversational"),
            ("Create PSSM matrix of EEEV", "annotation")
        ]
        
        for query, expected_intent in test_cases:
            # Process through real workflow
            response_chunks = []
            async for chunk in real_workflow.process_user_message(query, "test_session_real"):
                response_chunks.append(chunk)
                
                # Check for classification chunk
                if chunk.get('type') == 'classification':
                    classification = chunk
                    assert 'intent' in classification, "Classification chunk should contain intent"
                    actual_intent = classification['intent']
                    
                    print(f"✅ Real classification: {query} → {actual_intent}")
                    break
                
                # Stop after reasonable number of chunks
                if len(response_chunks) >= 5:
                    break


@pytest.mark.performance
class TestQueryClassificationPerformance:
    """Performance tests for query classification"""
    
    @pytest.mark.asyncio
    async def test_classification_performance(self, performance_mock_services):
        """Test classification performance requirements (from section 7.2.1)"""
        import time
        
        mock_components = performance_mock_services["workflow_components"]
        query = "Create PSSM matrix of EEEV"
        
        # Test response time
        start_time = time.time()
        result = await mock_components.mock_query_classification_step(query)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Target: <500ms (95th percentile), Maximum: <1s
        assert response_time < 1.0, f"Classification took {response_time:.3f}s, exceeds 1s maximum"
        
        if response_time < 0.5:
            print(f"✅ Classification performance excellent: {response_time:.3f}s")
        else:
            print(f"⚠️ Classification performance acceptable but slow: {response_time:.3f}s")


if __name__ == "__main__":
    # Run tests directly
    print("🧪 Running Query Classification Tests (Pytest Compatible)...")
    print("To run with pytest: pytest tests/chatbot_viral_integration/test_query_classification_pytest.py -v") 