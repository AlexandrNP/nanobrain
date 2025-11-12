#!/usr/bin/env python3
"""
🔥 BRUTAL TRUTH: DEBUG WORKFLOW ORCHESTRATION
This script tests the workflow step by step to isolate the hanging issue.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add nanobrain to path
sys.path.insert(0, str(Path.cwd().parent.parent))

from nanobrain.core.workflow import Workflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DebugWorkflow")

async def debug_workflow():
    """Debug the workflow step by step"""
    
    logger.info("🔥 BRUTAL TRUTH: DEBUGGING WORKFLOW ORCHESTRATION")
    logger.info("=" * 70)
    
    try:
        # Step 1: Initialize workflow
        logger.info("📋 Step 1: Loading workflow...")
        workflow = Workflow.from_config('config/mixed_execution_workflow_aurora.yml')
        await workflow.initialize()
        logger.info("✅ Workflow initialized successfully")
        
        # Step 2: Set minimal input data
        logger.info("📊 Step 2: Setting minimal test data...")
        test_data = {
            "sequences": [
                {"id": "seq1", "sequence": "ATCG", "length": 4},
                {"id": "seq2", "sequence": "GCTA", "length": 4}
            ],  # Proper sequence format
            "metadata": {"test": True}
        }
        
        # Get the raw_input data unit using correct API
        data_prep_step = workflow.child_steps["data_preparation"]
        raw_input = data_prep_step.step_input_data_units["raw_input"]
        await raw_input.set(test_data)
        logger.info("✅ Test data set successfully")

        # Step 3: Wait and monitor each step
        logger.info("🔄 Step 3: Monitoring workflow execution...")

        # Get all data units we need to monitor
        prepared_data = data_prep_step.step_output_data_units["prepared_data"]

        aurora_step = workflow.child_steps["aurora_computation"]
        aurora_input = aurora_step.step_input_data_units["aurora_input"]
        aurora_results = aurora_step.step_output_data_units["aurora_results"]

        result_step = workflow.child_steps["result_aggregation"]
        computation_data = result_step.step_input_data_units["computation_data"]
        final_results = result_step.step_output_data_units["final_results"]

        # Monitor for 60 seconds max
        for i in range(60):
            await asyncio.sleep(1)

            # Check data preparation step (has_data() is NOT async)
            prep_has_data = prepared_data.has_data()

            # Check aurora computation step
            aurora_input_has_data = aurora_input.has_data()
            aurora_results_has_data = aurora_results.has_data()

            # Check result aggregation step
            comp_data_has_data = computation_data.has_data()
            final_has_data = final_results.has_data()
            
            logger.info(f"⏳ {i+1}s: prep={prep_has_data}, aurora_in={aurora_input_has_data}, aurora_out={aurora_results_has_data}, comp_data={comp_data_has_data}, final={final_has_data}")
            
            # Check if workflow completed
            if final_has_data:
                logger.info("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
                final_data = await final_results.get()
                logger.info(f"📊 Final results: {final_data}")
                return True
                
        logger.error("❌ WORKFLOW TIMEOUT: Did not complete in 60 seconds")
        return False
        
    except Exception as e:
        logger.error(f"💥 WORKFLOW ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_workflow())
    if success:
        print("🎉 DEBUG TEST: SUCCESS!")
        sys.exit(0)
    else:
        print("💥 DEBUG TEST: FAILED!")
        sys.exit(1)
