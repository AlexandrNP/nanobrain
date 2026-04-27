#!/usr/bin/env python3
"""
Debug script to check workflow output type and structure
"""
import asyncio
import sys
import json
from pathlib import Path

# Add nanobrain to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow import AlphavirusWorkflow


async def debug_workflow_output():
    """Debug what the workflow.process() method actually returns"""
    
    print("🔍 Debugging Workflow Output")
    
    # Create workflow instance
    config_path = "nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml"
    workflow = AlphavirusWorkflow.from_config(
        config_path,
        workflow_directory="nanobrain/library/workflows/viral_protein_analysis"
    )
    
    # Initialize workflow
    await workflow.initialize()
    
    # Prepare simple test input
    test_input = {
        "virus_species": "Chikungunya virus",
        "analysis_type": "pssm",
        "user_query": "Debug test"
    }
    
    print(f"📊 Test input type: {type(test_input)}")
    print(f"📊 Test input content: {test_input}")
    
    try:
        # Execute workflow
        print("🔄 Executing workflow...")
        result = await workflow.process(test_input)
        
        print(f"✅ Workflow result type: {type(result)}")
        print(f"✅ Workflow result: {result}")
        
        # Test if result has .get() method (is dict-like)
        if hasattr(result, 'get'):
            print("✅ Result has .get() method - is dict-like")
            metadata = result.get('metadata', 'No metadata')
            print(f"📊 Metadata: {metadata}")
        else:
            print("❌ Result does NOT have .get() method - is NOT dict-like")
            print(f"📊 Result attributes: {dir(result)}")
    
    except Exception as e:
        print(f"❌ Error during workflow execution: {e}")
        print(f"❌ Error type: {type(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_workflow_output()) 