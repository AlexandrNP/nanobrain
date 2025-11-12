#!/usr/bin/env python3
"""
Nanobrain RAG Template - General-purpose Retrieval-Augmented Generation workflow.

This module provides a complete, configuration-driven RAG implementation that demonstrates
the Nanobrain framework's "configuration over code" philosophy.

Key Components:
- QueryEnhancementAgent: LLM-based query optimization (single LLM component)
- DocumentProcessorStep: Deterministic document parsing and chunking
- EmbeddingGeneratorStep: API-based embedding generation
- VectorStorageStep: Database operations for vector storage
- SemanticRetrievalStep: Similarity search and ranking
- ResponseEnhancementStep: Template-based response formatting

Architecture Principles:
- Single LLM Component: Only QueryEnhancementAgent uses ConversationalAgent
- Deterministic Processing: All other components extend BaseStep
- Configuration-Driven: All functionality achieved through YAML configuration
- Framework Compliance: Proper inheritance patterns throughout
"""

from .rag_workflow import RAGWorkflow

# Import agents
from .agents.query_enhancement_agent import QueryEnhancementAgent

# Import steps
from .steps.query_enhancement_step import QueryEnhancementStep
from .steps.document_processor_step import DocumentProcessorStep
from .steps.embedding_generator_step import EmbeddingGeneratorStep
from .steps.vector_storage_step import VectorStorageStep
from .steps.semantic_retrieval_step import SemanticRetrievalStep
from .steps.response_enhancement_step import ResponseEnhancementStep

__all__ = [
    'RAGWorkflow',
    'QueryEnhancementAgent',
    'QueryEnhancementStep',
    'DocumentProcessorStep',
    'EmbeddingGeneratorStep',
    'VectorStorageStep',
    'SemanticRetrievalStep',
    'ResponseEnhancementStep'
]

__version__ = "1.0.0"
