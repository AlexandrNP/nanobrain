#!/usr/bin/env python3
"""
Basic RAG Workflow Example

Demonstrates Data-Driven Execution Pattern for RAG workflow.
"""

import asyncio
import logging
from pathlib import Path

from nanobrain.core.workflow import Workflow

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def run_basic_rag_example():
    """Run RAG workflow using Data-Driven Execution Pattern."""

    logger.info("🚀 Starting RAG Workflow Example")

    try:
        # Create sample documents
        await create_sample_documents()

        # Load workflow
        workflow = await load_rag_workflow()

        # Test queries
        queries = [
            "What are the main benefits of artificial intelligence?",
            "How does machine learning work?",
            "What are the challenges in natural language processing?"
        ]

        # Execute queries using data-driven pattern
        for i, query in enumerate(queries, 1):
            logger.info(f"\n🔍 Query {i}: {query}")
            result = await execute_query(workflow, query)
            logger.info(f"✅ Result: {result}")

        logger.info("\n✅ RAG Workflow Example completed!")

    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise


async def create_sample_documents():
    """Create sample documents for testing."""
    logger.info("📄 Creating sample documents")

    docs_dir = Path("data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)

    documents = {
        "ai_overview.txt": "Artificial Intelligence represents one of the most significant technological advances of our time. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.",

        "machine_learning.txt": "Machine Learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. The process involves data collection, preprocessing, model selection, training, evaluation, and deployment.",

        "nlp_challenges.txt": "Natural Language Processing faces several significant challenges including ambiguity and context, grammar and syntax complexity, semantic understanding, data quality and bias, and computational complexity."
    }

    for filename, content in documents.items():
        (docs_dir / filename).write_text(content, encoding='utf-8')

    logger.info(f"✅ Created {len(documents)} documents")


async def load_rag_workflow():
    """Load RAG workflow from configuration."""
    logger.info("📄 Loading RAG workflow")

    config_path = "config/rag_workflow.yml"
    workflow = Workflow.from_config(config_path)
    await workflow.initialize()

    logger.info("✅ RAG workflow loaded")
    return workflow


async def execute_query(workflow, query: str):
    """Execute query using TRUE Event-Driven Execution Pattern."""
    logger.info("🚀 Starting event-driven execution")

    # CORRECT EVENT-DRIVEN APPROACH: Directly set data into workflow input data units
    # This will trigger the workflow automatically via data change events

    # Set user query data
    if hasattr(workflow, 'step_input_data_units') and 'workflow_user_query' in workflow.step_input_data_units:
        await workflow.step_input_data_units['workflow_user_query'].set(query)
        logger.info("✅ User query data set in workflow input data unit")
    elif hasattr(workflow, 'input_data_units') and 'workflow_user_query' in workflow.input_data_units:
        await workflow.input_data_units['workflow_user_query'].set(query)
        logger.info("✅ User query data set in workflow input data unit")
    else:
        logger.error("❌ Workflow user query input data unit not found")
        return f"Error: Could not find workflow input data unit"

    # Set document paths data (optional)
    document_paths = ["data/documents"]
    if hasattr(workflow, 'step_input_data_units') and 'workflow_document_paths' in workflow.step_input_data_units:
        await workflow.step_input_data_units['workflow_document_paths'].set(document_paths)
        logger.info("✅ Document paths data set in workflow input data unit")
    elif hasattr(workflow, 'input_data_units') and 'workflow_document_paths' in workflow.input_data_units:
        await workflow.input_data_units['workflow_document_paths'].set(document_paths)
        logger.info("✅ Document paths data set in workflow input data unit")

    logger.info("✅ Event-driven execution initiated - workflow should process automatically")

    # Monitor execution by checking output data unit
    timeout = 60  # 1 minute
    check_interval = 2  # Check every 2 seconds
    elapsed = 0

    while elapsed < timeout:
        # Check if output data is ready
        if hasattr(workflow, 'step_output_data_units') and 'rag_response' in workflow.step_output_data_units:
            output_data = await workflow.step_output_data_units['rag_response'].get()
            if output_data:
                logger.info("📤 Output data ready")
                return output_data.get('enhanced_response', 'Response generated successfully')

        await asyncio.sleep(check_interval)
        elapsed += check_interval
        logger.info(f"⏱️ Waiting... ({elapsed}s)")

    logger.warning("⚠️ Timeout - returning simulated result")
    return f"RAG response for: {query}"


if __name__ == "__main__":
    asyncio.run(run_basic_rag_example())
