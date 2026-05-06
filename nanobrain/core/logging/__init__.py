"""
BRUTAL TRUTH: Enhanced Workflow Logging System

This package provides comprehensive workflow execution tracing and analysis
that actually helps with debugging complex workflow issues.

Key Features:
- Complete execution timeline tracking
- Data flow visualization
- Trigger execution correlation
- Step execution analysis
- Automatic issue identification
- Human-readable execution reports

Usage:
    from nanobrain.core.logging import enable_workflow_tracing, trace_workflow
    
    # Enable automatic tracing
    enable_workflow_tracing()
    
    # Create and run workflow
    workflow = Workflow.from_config('config.yml')
    await workflow.initialize()  # Automatic instrumentation happens here
    
    # Or manually instrument
    trace_workflow(workflow)
"""

from .workflow_tracer import WorkflowTracer, EventType, TraceEvent
from .workflow_integration import (
    TracingIntegration,
    enable_workflow_tracing,
    disable_workflow_tracing,
    trace_workflow,
    get_workflow_tracer,
    tracing_integration
)
from .trace_analyzer import TraceAnalyzer

__all__ = [
    'WorkflowTracer',
    'EventType', 
    'TraceEvent',
    'TracingIntegration',
    'enable_workflow_tracing',
    'disable_workflow_tracing', 
    'trace_workflow',
    'get_workflow_tracer',
    'tracing_integration',
    'TraceAnalyzer'
]
