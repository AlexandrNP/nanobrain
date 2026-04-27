"""
BRUTAL TRUTH: Simple LLM Integration

My previous integration was over-engineered. This is ACTUALLY simple:
- Minimal instrumentation
- Dead simple log format
- Easy LLM processing
- No complex abstractions
"""

import functools
import time
from typing import Dict, Any
from nanobrain.core.logging.simple_llm_logger import get_simple_logger


def instrument_workflow_simple(workflow):
    """
    BRUTAL TRUTH: Simple workflow instrumentation for LLM processing.
    
    No complex abstractions, no over-engineering. Just simple logging.
    """
    workflow_id = getattr(workflow, 'name', 'unknown')
    logger = get_simple_logger(workflow_id)
    
    # Log workflow start
    steps = len(getattr(workflow, 'child_steps', {}))
    links = len(getattr(workflow, 'step_links', {}))
    logger.workflow_start(steps, links)
    
    # Instrument steps
    if hasattr(workflow, 'child_steps'):
        for step_id, step in workflow.child_steps.items():
            _instrument_step_simple(step, step_id, logger)
    
    return workflow


def _instrument_step_simple(step, step_id: str, logger):
    """Simple step instrumentation"""
    if hasattr(step, '_simple_instrumented'):
        return
    
    step._simple_instrumented = True
    
    if hasattr(step, 'process'):
        original_process = step.process
        
        @functools.wraps(original_process)
        async def simple_traced_process(*args, **kwargs):
            start_time = time.time()
            
            # Get input keys (simple)
            input_keys = []
            if hasattr(step, 'step_input_data_units'):
                input_keys = list(step.step_input_data_units.keys())
            
            # Log step start
            logger.step_start(step_id, input_keys)
            
            try:
                # Execute
                result = await original_process(*args, **kwargs)
                
                # Get output keys (simple)
                output_keys = []
                if hasattr(step, 'step_output_data_units'):
                    output_keys = list(step.step_output_data_units.keys())
                
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Log completion
                logger.step_complete(step_id, duration_ms, output_keys)
                
                # Check for data loss (simple)
                _check_data_loss_simple(step_id, input_keys, output_keys, logger)
                
                # Check performance (simple)
                if duration_ms > 5000:
                    logger.performance_issue(step_id, "slow_execution", f"{duration_ms:.0f}ms")
                
                return result
                
            except Exception as e:
                # Log error (simple)
                logger.step_error(step_id, type(e).__name__, str(e)[:100])
                raise
        
        step.process = simple_traced_process


def _check_data_loss_simple(step_id: str, input_keys: list, output_keys: list, logger):
    """Simple data loss detection"""
    # Check for important keys that might be lost
    important_patterns = ['query', 'input', 'output', 'response']
    
    lost_keys = []
    for pattern in important_patterns:
        input_has = any(pattern in key.lower() for key in input_keys)
        output_has = any(pattern in key.lower() for key in output_keys)
        
        if input_has and not output_has:
            lost_keys.append(pattern)
    
    if lost_keys:
        logger.data_lost(step_id, lost_keys)


# Simple usage functions
def enable_simple_llm_logging():
    """Enable simple LLM logging"""
    import logging
    logging.getLogger("LLM_SIMPLE").setLevel(logging.INFO)


def get_simple_logs() -> str:
    """Get all simple logs as string for LLM processing"""
    # This is a hack - in real implementation, you'd capture logs properly
    return "Use logging capture to get logs for LLM processing"


# Example usage patterns for LLMs
def demo_llm_processing():
    """
    BRUTAL TRUTH: Example of how LLMs should process the simple logs.
    
    Log format: TIMESTAMP|EVENT|COMPONENT|STATUS|MESSAGE
    
    LLM processing patterns:
    1. Extract errors: grep lines with "|ERROR|"
    2. Find slow steps: grep lines with "slow_execution"
    3. Track data flow: grep lines with "|DATA_FLOW|"
    4. Find data loss: grep lines with "|DATA_LOST|"
    """
    
    sample_logs = """
0.000|WORKFLOW_START|test_workflow|INFO|Started with 3 steps and 2 links
0.001|STEP_START|query_input|INFO|Executing with inputs: raw_query
0.005|STEP_COMPLETE|query_input|SUCCESS|Completed in 4ms with outputs: processed_query
0.006|DATA_LOST|query_input|WARNING|Lost keys: input
0.010|STEP_START|enhancement|INFO|Executing with inputs: processed_query
2.500|STEP_COMPLETE|enhancement|SUCCESS|Completed in 2490ms with outputs: enhanced_query
2.501|PERFORMANCE|enhancement|WARNING|slow_execution: 2490ms
2.505|STEP_START|search|INFO|Executing with inputs: enhanced_query
5.000|STEP_ERROR|search|ERROR|TimeoutError: Request timed out
    """
    
    print("SAMPLE LOGS:")
    print(sample_logs)
    print()
    
    print("LLM PROCESSING EXAMPLES:")
    print("1. Extract errors:")
    for line in sample_logs.strip().split('\n'):
        if '|ERROR|' in line:
            parts = line.split('|')
            print(f"   Error in {parts[2]}: {parts[4]}")
    
    print("2. Find performance issues:")
    for line in sample_logs.strip().split('\n'):
        if '|PERFORMANCE|' in line or 'slow_execution' in line:
            parts = line.split('|')
            print(f"   Performance issue in {parts[2]}: {parts[4]}")
    
    print("3. Find data loss:")
    for line in sample_logs.strip().split('\n'):
        if '|DATA_LOST|' in line:
            parts = line.split('|')
            print(f"   Data lost in {parts[2]}: {parts[4]}")


if __name__ == "__main__":
    demo_llm_processing()
