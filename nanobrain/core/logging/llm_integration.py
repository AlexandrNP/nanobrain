"""
BRUTAL TRUTH: LLM-Optimized Workflow Integration

This module provides LLM-optimized instrumentation that focuses on:
1. CONSISTENT FORMAT - Every log entry follows the same pattern
2. MINIMAL NOISE - Only essential information for debugging
3. CLEAR PATTERNS - LLMs can easily extract specific information
4. ACTIONABLE INSIGHTS - Direct mapping from logs to debugging actions
"""

import functools
import time
from typing import Dict, Any, Optional
from nanobrain.core.logging.llm_optimized_tracer import LLMOptimizedTracer


class LLMOptimizedIntegration:
    """
    BRUTAL TRUTH: LLM-optimized workflow instrumentation.
    
    This integration provides CLEAN, CONSISTENT logging that LLMs can actually process:
    - No verbose implementation details
    - Clear semantic markers for each event type
    - Predictable format for all log entries
    - Essential context only, no noise
    """
    
    def __init__(self):
        self.tracers: Dict[str, LLMOptimizedTracer] = {}
        self.enabled = True
    
    def get_tracer(self, workflow_id: str) -> LLMOptimizedTracer:
        """Get or create LLM-optimized tracer for workflow"""
        if workflow_id not in self.tracers:
            self.tracers[workflow_id] = LLMOptimizedTracer(workflow_id)
        return self.tracers[workflow_id]
    
    def instrument_workflow(self, workflow):
        """Instrument workflow with LLM-optimized logging"""
        if not self.enabled:
            return
            
        workflow_id = getattr(workflow, 'name', 'unknown_workflow')
        tracer = self.get_tracer(workflow_id)
        
        # Store tracer reference
        workflow._llm_tracer = tracer
        
        # Log workflow start with essential context only
        context = {
            "steps": len(getattr(workflow, 'child_steps', {})),
            "links": len(getattr(workflow, 'step_links', {}))
        }
        tracer.log_workflow_start(context)
        
        # Instrument child steps
        if hasattr(workflow, 'child_steps'):
            for step_id, step in workflow.child_steps.items():
                self.instrument_step(step, step_id, tracer)
    
    def instrument_step(self, step, step_id: str, tracer: LLMOptimizedTracer):
        """Instrument step with LLM-optimized logging"""
        if hasattr(step, '_llm_traced'):
            return
            
        step._llm_traced = True
        step._llm_tracer = tracer
        step._step_id = step_id
        
        # Wrap the process method with LLM-optimized logging
        if hasattr(step, 'process'):
            original_process = step.process
            
            @functools.wraps(original_process)
            async def llm_traced_process(*args, **kwargs):
                start_time = time.time()
                
                # Extract input data for LLM logging
                input_data = self._extract_input_data(step)
                
                # Log step execution start
                correlation_id = tracer.log_step_execution(step_id, input_data)
                
                try:
                    # Execute original process
                    result = await original_process(*args, **kwargs)
                    
                    # Extract output data
                    output_data = self._extract_output_data(step)
                    
                    # Calculate duration
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Log successful completion
                    tracer.log_step_completion(step_id, output_data, duration_ms, correlation_id)
                    
                    # Check for data preservation issues
                    self._check_data_preservation(step_id, input_data, output_data, tracer)
                    
                    # Check for performance issues
                    if duration_ms > 5000:  # More than 5 seconds
                        tracer.log_performance_issue(
                            step_id, 
                            "slow_execution", 
                            {"duration_ms": duration_ms, "threshold_ms": 5000}
                        )
                    
                    return result
                    
                except Exception as e:
                    # Log critical error with essential context
                    tracer.log_critical_error(
                        step_id,
                        str(e),
                        {
                            "error_type": type(e).__name__,
                            "input_keys": list(input_data.keys()) if input_data else [],
                            "duration_ms": (time.time() - start_time) * 1000
                        }
                    )
                    raise
            
            step.process = llm_traced_process
    
    def _extract_input_data(self, step) -> Dict[str, Any]:
        """Extract input data with LLM-friendly format"""
        input_data = {}
        
        if hasattr(step, 'step_input_data_units'):
            for name, data_unit in step.step_input_data_units.items():
                try:
                    # Get data synchronously if possible, otherwise mark as async
                    if hasattr(data_unit, '_internal_data'):
                        input_data[name] = data_unit._internal_data
                    else:
                        input_data[name] = f"<async_data_unit:{name}>"
                except:
                    input_data[name] = f"<error_reading:{name}>"
        
        return input_data
    
    def _extract_output_data(self, step) -> Dict[str, Any]:
        """Extract output data with LLM-friendly format"""
        output_data = {}
        
        if hasattr(step, 'step_output_data_units'):
            for name, data_unit in step.step_output_data_units.items():
                try:
                    if hasattr(data_unit, '_internal_data'):
                        output_data[name] = data_unit._internal_data
                    else:
                        output_data[name] = f"<async_data_unit:{name}>"
                except:
                    output_data[name] = f"<error_reading:{name}>"
        
        return output_data
    
    def _check_data_preservation(self, step_id: str, input_data: Dict[str, Any], 
                               output_data: Dict[str, Any], tracer: LLMOptimizedTracer):
        """Check for data preservation issues and log them for LLM processing"""
        if not isinstance(input_data, dict) or not isinstance(output_data, dict):
            return
        
        # Check for important keys that might be lost
        important_keys = ["query", "original_query", "enhanced_query", "response", "input", "output"]
        lost_important_keys = []
        
        for key in important_keys:
            # Check if key exists in input but not in output
            input_has_key = any(key in str(k).lower() for k in input_data.keys())
            output_has_key = any(key in str(k).lower() for k in output_data.keys())
            
            if input_has_key and not output_has_key:
                lost_important_keys.append(key)
        
        if lost_important_keys:
            tracer.log_data_loss(
                step_id,
                lost_important_keys,
                {
                    "input_keys": list(input_data.keys()),
                    "output_keys": list(output_data.keys()),
                    "preservation_ratio": len(set(input_data.keys()) & set(output_data.keys())) / len(input_data) if input_data else 0
                }
            )


# Global LLM-optimized integration instance
llm_integration = LLMOptimizedIntegration()


def enable_llm_optimized_tracing():
    """Enable LLM-optimized workflow tracing"""
    llm_integration.enabled = True


def disable_llm_optimized_tracing():
    """Disable LLM-optimized workflow tracing"""
    llm_integration.enabled = False


def trace_workflow_for_llm(workflow):
    """Instrument workflow with LLM-optimized tracing"""
    llm_integration.instrument_workflow(workflow)
    return workflow


def get_llm_tracer(workflow_id: str) -> Optional[LLMOptimizedTracer]:
    """Get LLM-optimized tracer for workflow"""
    return llm_integration.tracers.get(workflow_id)


def get_llm_summary(workflow_id: str) -> str:
    """Get LLM-friendly execution summary"""
    tracer = get_llm_tracer(workflow_id)
    if tracer:
        return tracer.get_llm_summary()
    return f"WORKFLOW_SUMMARY: No tracer found for {workflow_id}"
