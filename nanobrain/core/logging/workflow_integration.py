"""
BRUTAL TRUTH: Workflow Tracing Integration

This module integrates the enhanced tracing system with the existing workflow framework.
It provides monkey-patching and instrumentation to automatically trace workflow execution
without requiring massive code changes.
"""

import functools
import inspect
from typing import Any, Dict, Optional, Callable
from nanobrain.core.logging.workflow_tracer import WorkflowTracer, EventType


class TracingIntegration:
    """
    BRUTAL TRUTH: Automatic workflow tracing integration.
    
    This class instruments the workflow framework to automatically trace:
    - Step executions
    - Trigger firings
    - Data transfers
    - Data transformations
    """
    
    def __init__(self):
        self.tracers: Dict[str, WorkflowTracer] = {}
        self.enabled = True
    
    def get_tracer(self, workflow_id: str) -> WorkflowTracer:
        """Get or create tracer for workflow"""
        if workflow_id not in self.tracers:
            self.tracers[workflow_id] = WorkflowTracer(workflow_id)
        return self.tracers[workflow_id]
    
    def instrument_workflow(self, workflow):
        """Instrument a workflow instance for tracing"""
        if not self.enabled:
            return
            
        workflow_id = getattr(workflow, 'name', 'unknown_workflow')
        tracer = self.get_tracer(workflow_id)
        
        # Store tracer reference on workflow
        workflow._tracer = tracer
        
        # Trace workflow start
        tracer.trace_event(
            EventType.WORKFLOW_START,
            workflow_id,
            "workflow",
            {
                'num_steps': len(getattr(workflow, 'child_steps', {})),
                'num_links': len(getattr(workflow, 'step_links', {})),
                'workflow_type': type(workflow).__name__
            }
        )
        
        # Instrument child steps
        if hasattr(workflow, 'child_steps'):
            for step_id, step in workflow.child_steps.items():
                self.instrument_step(step, step_id, tracer)
        
        # Instrument links
        if hasattr(workflow, 'step_links'):
            for link_id, link in workflow.step_links.items():
                self.instrument_link(link, link_id, tracer)
    
    def instrument_step(self, step, step_id: str, tracer: WorkflowTracer):
        """Instrument a step for tracing"""
        if hasattr(step, '_traced'):
            return  # Already instrumented
            
        step._traced = True
        step._tracer = tracer
        step._step_id = step_id
        
        # Wrap the process method
        if hasattr(step, 'process'):
            original_process = step.process
            
            @functools.wraps(original_process)
            async def traced_process(*args, **kwargs):
                # Extract input data for tracing
                input_data = self._extract_step_input_data(step)
                
                with tracer.trace_context(
                    EventType.STEP_START,
                    step_id,
                    "step",
                    {
                        'step_type': type(step).__name__,
                        'input_summary': tracer._summarize_data(input_data),
                        'method': 'process'
                    }
                ) as event_id:
                    # Execute original process method
                    result = await original_process(*args, **kwargs)
                    
                    # Extract output data for tracing
                    output_data = self._extract_step_output_data(step)
                    
                    # Trace data transformation
                    tracer.trace_data_transformation(
                        step_id,
                        input_data,
                        output_data,
                        'step_processing'
                    )
                    
                    return result
            
            step.process = traced_process
        
        # Instrument triggers if present
        if hasattr(step, 'step_triggers'):
            for trigger_id, trigger in step.step_triggers.items():
                self.instrument_trigger(trigger, trigger_id, step_id, tracer)
    
    def instrument_trigger(self, trigger, trigger_id: str, step_id: str, tracer: WorkflowTracer):
        """Instrument a trigger for tracing"""
        if hasattr(trigger, '_traced'):
            return
            
        trigger._traced = True
        
        # Wrap trigger activation
        if hasattr(trigger, '_execute_action'):
            original_execute = trigger._execute_action
            
            @functools.wraps(original_execute)
            async def traced_execute(*args, **kwargs):
                # Trace trigger firing
                trigger_event_id = tracer.trace_trigger_execution(
                    trigger_id,
                    type(trigger).__name__,
                    {
                        'conditions': getattr(trigger, 'conditions', {}),
                        'data_unit': getattr(trigger, 'data_unit', 'unknown'),
                        'operation': 'trigger_fired'
                    },
                    step_id
                )
                
                # Execute original action
                result = await original_execute(*args, **kwargs)
                
                return result
            
            trigger._execute_action = traced_execute
    
    def instrument_link(self, link, link_id: str, tracer: WorkflowTracer):
        """Instrument a link for tracing"""
        if hasattr(link, '_traced'):
            return
            
        link._traced = True
        
        # Wrap data transfer method
        if hasattr(link, 'transfer_data'):
            original_transfer = link.transfer_data
            
            @functools.wraps(original_transfer)
            async def traced_transfer(data, *args, **kwargs):
                # Trace data flow
                tracer.trace_data_flow(
                    getattr(link, 'source', 'unknown'),
                    getattr(link, 'target', 'unknown'),
                    tracer._summarize_data(data),
                    link_id
                )
                
                # Execute original transfer
                result = await original_transfer(data, *args, **kwargs)
                
                return result
            
            link.transfer_data = traced_transfer
    
    def _extract_step_input_data(self, step) -> Dict[str, Any]:
        """Extract input data from step for tracing"""
        input_data = {}
        
        if hasattr(step, 'step_input_data_units'):
            for name, data_unit in step.step_input_data_units.items():
                if hasattr(data_unit, 'get'):
                    try:
                        input_data[name] = data_unit.get()
                    except:
                        input_data[name] = 'error_reading_data'
        
        return input_data
    
    def _extract_step_output_data(self, step) -> Dict[str, Any]:
        """Extract output data from step for tracing"""
        output_data = {}
        
        if hasattr(step, 'step_output_data_units'):
            for name, data_unit in step.step_output_data_units.items():
                if hasattr(data_unit, 'get'):
                    try:
                        output_data[name] = data_unit.get()
                    except:
                        output_data[name] = 'error_reading_data'
        
        return output_data


# Global tracing integration instance
tracing_integration = TracingIntegration()


def enable_workflow_tracing():
    """Enable automatic workflow tracing"""
    tracing_integration.enabled = True


def disable_workflow_tracing():
    """Disable automatic workflow tracing"""
    tracing_integration.enabled = False


def trace_workflow(workflow):
    """Manually instrument a workflow for tracing"""
    tracing_integration.instrument_workflow(workflow)
    return workflow


def get_workflow_tracer(workflow_id: str) -> Optional[WorkflowTracer]:
    """Get tracer for a specific workflow"""
    return tracing_integration.tracers.get(workflow_id)
