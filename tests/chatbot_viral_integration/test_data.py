#!/usr/bin/env python3
"""
Test Data for Chatbot Viral Integration Testing

This module contains all test data sets as specified in the
CHATBOT_VIRAL_INTEGRATION_TESTING_PLAN.md document.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class QueryTestCase:
    """Test query with expected results"""
    query: str
    expected_intent: str
    expected_routing: str
    confidence_threshold: float
    expected_elements: List[str] = None
    success_criteria: str = ""


class ChatbotTestData:
    """Centralized test data for chatbot viral integration testing"""
    
    # Conversational Test Queries (from section 5.2.1)
    CONVERSATIONAL_QUERIES = [
        QueryTestCase(
            query="What is EEEV?",
            expected_intent="conversational",
            expected_routing="conversational_response",
            confidence_threshold=0.8,
            expected_elements=["EEEV", "Eastern Equine", "virus", "educational"],
            success_criteria="Contains EEEV/Eastern Equine/virus terms"
        ),
        QueryTestCase(
            query="Tell me about Eastern Equine Encephalitis virus",
            expected_intent="conversational",
            expected_routing="conversational_response",
            confidence_threshold=0.8,
            expected_elements=["Eastern Equine", "encephalitis", "virus", "symptoms"],
            success_criteria="Contains virus information and symptoms"
        ),
        QueryTestCase(
            query="How does Alphavirus spread?",
            expected_intent="conversational",
            expected_routing="conversational_response",
            confidence_threshold=0.7,
            expected_elements=["transmission", "spread", "vector", "mosquito"],
            success_criteria="Contains transmission/spread terms"
        ),
        QueryTestCase(
            query="What are the symptoms of viral encephalitis?",
            expected_intent="conversational",
            expected_routing="conversational_response",
            confidence_threshold=0.8,
            expected_elements=["symptoms", "fever", "headache", "neurological"],
            success_criteria="Contains symptom information"
        ),
        QueryTestCase(
            query="Explain viral protein structure",
            expected_intent="conversational",
            expected_routing="conversational_response",
            confidence_threshold=0.7,
            expected_elements=["protein", "structure", "viral", "capsid"],
            success_criteria="Contains protein structure information"
        ),
        QueryTestCase(
            query="What is a PSSM matrix?",
            expected_intent="conversational",
            expected_routing="conversational_response",
            confidence_threshold=0.8,
            expected_elements=["PSSM", "matrix", "bioinformatics", "sequence"],
            success_criteria="Contains PSSM/matrix/bioinformatics terms"
        ),
        QueryTestCase(
            query="How do viruses mutate?",
            expected_intent="conversational",
            expected_routing="conversational_response",
            confidence_threshold=0.8,
            expected_elements=["mutation", "genetic", "evolution", "RNA"],
            success_criteria="Contains mutation and genetic information"
        ),
        QueryTestCase(
            query="What are the different types of alphaviruses?",
            expected_intent="conversational",
            expected_routing="conversational_response",
            confidence_threshold=0.8,
            expected_elements=["alphavirus", "types", "species", "classification"],
            success_criteria="Contains alphavirus classification information"
        )
    ]
    
    # Annotation Test Queries (from section 5.2.2)
    ANNOTATION_QUERIES = [
        QueryTestCase(
            query="Create PSSM matrix of EEEV",
            expected_intent="annotation",
            expected_routing="annotation_job",
            confidence_threshold=0.8,
            expected_elements=["job_id", "progress", "PSSM", "JSON"],
            success_criteria="Job completion, PSSM JSON output"
        ),
        QueryTestCase(
            query="Analyze Eastern Equine Encephalitis virus proteins",
            expected_intent="annotation",
            expected_routing="annotation_job",
            confidence_threshold=0.8,
            expected_elements=["job_id", "progress", "protein", "analysis"],
            success_criteria="Job completion, analysis report"
        ),
        QueryTestCase(
            query="Generate protein annotation for Alphavirus",
            expected_intent="annotation",
            expected_routing="annotation_job",
            confidence_threshold=0.8,
            expected_elements=["job_id", "annotation", "protein"],
            success_criteria="Job completion, annotation results"
        ),
        QueryTestCase(
            query="Perform viral protein clustering analysis",
            expected_intent="annotation",
            expected_routing="annotation_job",
            confidence_threshold=0.7,
            expected_elements=["job_id", "clustering", "analysis"],
            success_criteria="Clustering results, quality metrics"
        ),
        QueryTestCase(
            query="Create multiple sequence alignment for EEEV",
            expected_intent="annotation",
            expected_routing="annotation_job",
            confidence_threshold=0.8,
            expected_elements=["job_id", "alignment", "sequences"],
            success_criteria="Alignment results, quality metrics"
        ),
        QueryTestCase(
            query="Analyze protein domains in Chikungunya virus",
            expected_intent="annotation",
            expected_routing="annotation_job",
            confidence_threshold=0.8,
            expected_elements=["job_id", "domains", "protein"],
            success_criteria="Domain analysis results"
        ),
        QueryTestCase(
            query="Generate comprehensive viral analysis report",
            expected_intent="annotation",
            expected_routing="annotation_job",
            confidence_threshold=0.8,
            expected_elements=["job_id", "comprehensive", "report"],
            success_criteria="Complete analysis report"
        )
    ]
    
    # Edge Case Queries (from section 5.2.3)
    EDGE_CASE_QUERIES = [
        QueryTestCase(
            query="What is EEEV PSSM matrix analysis?",
            expected_intent="annotation",  # Ambiguous but should lean annotation
            expected_routing="annotation_job",
            confidence_threshold=0.6,
            expected_elements=["ambiguous", "intent"],
            success_criteria="Handles ambiguous intent gracefully"
        ),
        QueryTestCase(
            query="Create analysis for unknown virus XYZ",
            expected_intent="annotation",
            expected_routing="annotation_job",
            confidence_threshold=0.6,
            expected_elements=["error", "unknown", "organism"],
            success_criteria="Handles invalid organism gracefully"
        ),
        QueryTestCase(
            query="Tell me about EEEV and also create PSSM",
            expected_intent="annotation",  # Mixed intent should favor annotation
            expected_routing="annotation_job",
            confidence_threshold=0.6,
            expected_elements=["mixed", "intent"],
            success_criteria="Handles mixed intent appropriately"
        ),
        QueryTestCase(
            query="",
            expected_intent="error",
            expected_routing="error_handler",
            confidence_threshold=0.0,
            expected_elements=["validation", "error"],
            success_criteria="Handles empty input gracefully"
        ),
        QueryTestCase(
            query="A very long query that exceeds normal input limits and contains repetitive content to test how the system handles unusually long user inputs that might cause performance issues or buffer overflows in the query processing pipeline and should be handled gracefully with appropriate validation and error messages",
            expected_intent="conversational",
            expected_routing="conversational_response",
            confidence_threshold=0.5,
            expected_elements=["length", "validation"],
            success_criteria="Handles long input with validation"
        ),
        QueryTestCase(
            query="Special characters: @#$%^&*()",
            expected_intent="conversational",
            expected_routing="conversational_response",
            confidence_threshold=0.5,
            expected_elements=["special", "characters"],
            success_criteria="Handles special characters safely"
        )
    ]
    
    @classmethod
    def get_all_queries(cls) -> List[QueryTestCase]:
        """Get all test queries combined"""
        return cls.CONVERSATIONAL_QUERIES + cls.ANNOTATION_QUERIES + cls.EDGE_CASE_QUERIES
    
    @classmethod
    def get_queries_by_intent(cls, intent: str) -> List[QueryTestCase]:
        """Get queries filtered by expected intent"""
        return [q for q in cls.get_all_queries() if q.expected_intent == intent]
    
    @classmethod
    def get_classification_test_cases(cls) -> List[Dict[str, Any]]:
        """Get test cases formatted for classification testing (from section 6.1.1)"""
        return [
            {
                "test_id": "QC-001",
                "input_query": "What is EEEV?",
                "expected_intent": "conversational",
                "expected_routing": "conversational_response",
                "success_criteria": "Intent=conversational, confidence>0.8"
            },
            {
                "test_id": "QC-002",
                "input_query": "Create PSSM matrix of EEEV",
                "expected_intent": "annotation",
                "expected_routing": "annotation_job",
                "success_criteria": "Intent=annotation, confidence>0.8"
            },
            {
                "test_id": "QC-003",
                "input_query": "Tell me about EEEV proteins",
                "expected_intent": "conversational",
                "expected_routing": "conversational_response",
                "success_criteria": "Intent=conversational, confidence>0.6"
            },
            {
                "test_id": "QC-004",
                "input_query": "Analyze EEEV protein structure",
                "expected_intent": "annotation",
                "expected_routing": "annotation_job",
                "success_criteria": "Intent=annotation, confidence>0.7"
            },
            {
                "test_id": "QC-005",
                "input_query": "What is EEEV and create PSSM?",
                "expected_intent": "annotation",
                "expected_routing": "annotation_job",
                "success_criteria": "Intent=annotation, confidence>0.6"
            }
        ]
    
    @classmethod
    def get_conversational_test_cases(cls) -> List[Dict[str, Any]]:
        """Get test cases for conversational response testing (from section 6.2.1)"""
        return [
            {
                "test_id": "CR-001",
                "input_query": "What is EEEV?",
                "expected_elements": "Educational content, virus info, symptoms",
                "success_criteria": "Contains EEEV/Eastern Equine/virus terms"
            },
            {
                "test_id": "CR-002",
                "input_query": "How do viruses spread?",
                "expected_elements": "Transmission mechanisms, examples",
                "success_criteria": "Contains transmission/spread terms"
            },
            {
                "test_id": "CR-003",
                "input_query": "What is a PSSM matrix?",
                "expected_elements": "Technical explanation, applications",
                "success_criteria": "Contains PSSM/matrix/bioinformatics terms"
            }
        ]
    
    @classmethod
    def get_annotation_workflow_test_cases(cls) -> List[Dict[str, Any]]:
        """Get test cases for annotation workflow testing (from section 6.3.1)"""
        return [
            {
                "test_id": "AW-001",
                "input_query": "Create PSSM matrix of EEEV",
                "expected_steps": "All 14 workflow steps",
                "success_criteria": "Job completion, PSSM JSON output"
            },
            {
                "test_id": "AW-002",
                "input_query": "Analyze Chikungunya virus",
                "expected_steps": "All 14 workflow steps",
                "success_criteria": "Job completion, analysis report"
            },
            {
                "test_id": "AW-003",
                "input_query": "Protein clustering for Alphavirus",
                "expected_steps": "Steps 1-12",
                "success_criteria": "Clustering results, quality metrics"
            }
        ]


class MockTestData:
    """Mock data for testing external services"""
    
    MOCK_BVBRC_RESPONSE = {
        "genomes": [
            {
                "genome_id": "1234567.3",
                "genome_name": "Eastern equine encephalitis virus",
                "organism_name": "Eastern equine encephalitis virus",
                "taxon_id": 11036,
                "genome_status": "complete",
                "genome_length": 11703
            }
        ],
        "proteins": [
            {
                "feature_id": "fig|1234567.3.peg.1",
                "patric_id": "fig|1234567.3.peg.1",
                "annotation": "non-structural polyprotein",
                "product": "nsP1",
                "aa_sequence": "MTKPPSSSSKSKQR..."
            }
        ]
    }
    
    MOCK_PSSM_MATRIX = {
        "matrix_id": "test_pssm_001",
        "organism": "Eastern equine encephalitis virus",
        "protein": "nsP1",
        "matrix_data": [
            [4, -2, -1, -2],  # Position 1
            [-1, 5, -2, -1],  # Position 2
            [-2, -1, 4, -1]   # Position 3
        ],
        "position_weights": [0.8, 0.9, 0.7],
        "consensus_sequence": "MST"
    }
    
    MOCK_CLUSTERING_RESULT = {
        "cluster_id": "test_cluster_001",
        "num_clusters": 5,
        "clusters": [
            {
                "cluster_num": 1,
                "sequences": ["seq1", "seq2", "seq3"],
                "representative": "seq1",
                "size": 3
            }
        ],
        "silhouette_score": 0.85
    }


# Content quality validation criteria (from section 6.2.2)
CONTENT_QUALITY_CHECKS = {
    "min_length": 100,  # Minimum response length
    "max_length": 2000,  # Maximum response length
    "markdown_formatting": True,  # Should contain markdown
    "technical_accuracy": True,  # Should be scientifically accurate
    "educational_value": True   # Should provide learning value
}

# Classification accuracy metrics (from section 6.1.2)
CLASSIFICATION_METRICS = {
    "accuracy_threshold": 0.85,
    "precision_threshold": 0.80,
    "recall_threshold": 0.80,
    "f1_score_threshold": 0.80
}

# Workflow validation criteria (from section 6.3.2)
WORKFLOW_VALIDATION = {
    "step_completion": "All steps must complete successfully",
    "progress_tracking": "Progress updates for each step",
    "data_persistence": "Session data maintained throughout",
    "error_recovery": "Graceful handling of step failures",
    "timeout_handling": "Proper timeout management"
} 