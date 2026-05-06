"""
BRUTAL TRUTH: Enhanced Workflow Execution Tracer

The current logging system is completely inadequate for debugging workflows.
This module provides REAL visibility into workflow execution with:
- Data flow tracing
- Trigger execution tracking  
- Step execution correlation
- Timeline visualization
- Data transformation tracking
"""

import logging
import time
import json
import threading
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of workflow events to trace"""
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    STEP_START = "step_start"
    STEP_END = "step_end"
    TRIGGER_FIRED = "trigger_fired"
    DATA_FLOW = "data_flow"
    DATA_TRANSFORM = "data_transform"
    LINK_TRANSFER = "link_transfer"
    ERROR = "error"


@dataclass
class TraceEvent:
    """A single traced event in workflow execution"""
    event_id: str
    event_type: EventType
    timestamp: float
    workflow_id: str
    component_id: str
    component_type: str  # 'workflow', 'step', 'trigger', 'link', 'data_unit'
    event_data: Dict[str, Any]
    parent_event_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            **asdict(self),
            'event_type': self.event_type.value,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp).isoformat()
        }


class WorkflowTracer:
    """
    BRUTAL TRUTH: Real workflow execution tracer that actually helps with debugging.
    
    This tracer provides:
    - Complete execution timeline
    - Data flow visualization
    - Trigger-to-execution correlation
    - Step execution details
    - Error tracking and correlation
    """
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.events: List[TraceEvent] = []
        self.active_contexts: Dict[str, str] = {}  # context_id -> parent_event_id
        self.event_counter = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        with self.lock:
            self.event_counter += 1
            return f"{self.workflow_id}_event_{self.event_counter:06d}"
    
    def trace_event(self, 
                   event_type: EventType,
                   component_id: str,
                   component_type: str,
                   event_data: Dict[str, Any],
                   parent_event_id: Optional[str] = None,
                   correlation_id: Optional[str] = None) -> str:
        """
        Trace a workflow event
        
        Returns:
            event_id for correlation with child events
        """
        event_id = self._generate_event_id()
        
        event = TraceEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=time.time(),
            workflow_id=self.workflow_id,
            component_id=component_id,
            component_type=component_type,
            event_data=event_data,
            parent_event_id=parent_event_id,
            correlation_id=correlation_id
        )
        
        with self.lock:
            self.events.append(event)
            
        # Log the event immediately for real-time debugging
        self._log_event(event)
        
        return event_id
    
    def _log_event(self, event: TraceEvent):
        """Log event with structured format for easy parsing"""
        duration = event.timestamp - self.start_time
        
        # Create structured log message
        log_data = {
            'workflow_trace': True,
            'workflow_id': self.workflow_id,
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'component': f"{event.component_type}.{event.component_id}",
            'duration_s': round(duration, 3),
            'parent_event': event.parent_event_id,
            'correlation_id': event.correlation_id,
            **event.event_data
        }
        
        # Choose appropriate log level and icon
        icon = self._get_event_icon(event.event_type)
        level = logging.ERROR if event.event_type == EventType.ERROR else logging.INFO
        
        logger.log(level, f"{icon} WORKFLOW_TRACE: {json.dumps(log_data, default=str)}")
    
    def _get_event_icon(self, event_type: EventType) -> str:
        """Get icon for event type"""
        icons = {
            EventType.WORKFLOW_START: "🚀",
            EventType.WORKFLOW_END: "✅",
            EventType.STEP_START: "🔧",
            EventType.STEP_END: "✅",
            EventType.TRIGGER_FIRED: "⚡",
            EventType.DATA_FLOW: "📊",
            EventType.DATA_TRANSFORM: "🔄",
            EventType.LINK_TRANSFER: "🔗",
            EventType.ERROR: "❌"
        }
        return icons.get(event_type, "📋")
    
    @contextmanager
    def trace_context(self, 
                     event_type: EventType,
                     component_id: str,
                     component_type: str,
                     context_data: Dict[str, Any]):
        """
        Context manager for tracing start/end events
        
        Usage:
            with tracer.trace_context(EventType.STEP_START, "query_input", "step", {...}):
                # step execution
                pass
        """
        start_event_id = self.trace_event(
            event_type, component_id, component_type, context_data
        )
        
        try:
            yield start_event_id
        except Exception as e:
            # Trace error
            self.trace_event(
                EventType.ERROR,
                component_id,
                component_type,
                {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'context': 'execution_error'
                },
                parent_event_id=start_event_id
            )
            raise
        finally:
            # Trace end event
            end_event_type = EventType.STEP_END if event_type == EventType.STEP_START else event_type
            self.trace_event(
                end_event_type,
                component_id,
                component_type,
                {'context': 'execution_complete'},
                parent_event_id=start_event_id
            )

    def trace_data_flow(self,
                       source_component: str,
                       target_component: str,
                       data_summary: Dict[str, Any],
                       link_id: Optional[str] = None):
        """Trace data flowing between components"""
        self.trace_event(
            EventType.DATA_FLOW,
            link_id or f"{source_component}->{target_component}",
            "link",
            {
                'source': source_component,
                'target': target_component,
                'data_type': data_summary.get('type', 'unknown'),
                'data_size': data_summary.get('size', 0),
                'data_keys': data_summary.get('keys', []),
                'data_preview': data_summary.get('preview', 'N/A')
            }
        )

    def trace_trigger_execution(self,
                              trigger_id: str,
                              trigger_type: str,
                              trigger_data: Dict[str, Any],
                              target_step: str):
        """Trace trigger firing and target step"""
        return self.trace_event(
            EventType.TRIGGER_FIRED,
            trigger_id,
            "trigger",
            {
                'trigger_type': trigger_type,
                'target_step': target_step,
                'trigger_conditions': trigger_data.get('conditions', {}),
                'data_unit': trigger_data.get('data_unit', 'unknown'),
                'operation': trigger_data.get('operation', 'unknown')
            }
        )

    def trace_data_transformation(self,
                                step_id: str,
                                input_data: Dict[str, Any],
                                output_data: Dict[str, Any],
                                transformation_type: str):
        """Trace how data is transformed by a step"""
        self.trace_event(
            EventType.DATA_TRANSFORM,
            step_id,
            "step",
            {
                'transformation_type': transformation_type,
                'input_summary': self._summarize_data(input_data),
                'output_summary': self._summarize_data(output_data),
                'data_preserved': self._check_data_preservation(input_data, output_data)
            }
        )

    def _summarize_data(self, data: Any) -> Dict[str, Any]:
        """Create a summary of data for logging"""
        if data is None:
            return {'type': 'None', 'size': 0}

        if isinstance(data, dict):
            return {
                'type': 'dict',
                'size': len(data),
                'keys': list(data.keys())[:10],  # Limit to first 10 keys
                'preview': {k: str(v)[:100] for k, v in list(data.items())[:3]}
            }
        elif isinstance(data, list):
            return {
                'type': 'list',
                'size': len(data),
                'preview': [str(item)[:100] for item in data[:3]]
            }
        elif isinstance(data, str):
            return {
                'type': 'str',
                'size': len(data),
                'preview': data[:200]
            }
        else:
            return {
                'type': type(data).__name__,
                'size': 1,
                'preview': str(data)[:200]
            }

    def _check_data_preservation(self, input_data: Any, output_data: Any) -> Dict[str, Any]:
        """Check what data is preserved between input and output"""
        if not isinstance(input_data, dict) or not isinstance(output_data, dict):
            return {'analysis': 'non_dict_data'}

        preserved_keys = set(input_data.keys()) & set(output_data.keys())
        lost_keys = set(input_data.keys()) - set(output_data.keys())
        new_keys = set(output_data.keys()) - set(input_data.keys())

        return {
            'preserved_keys': list(preserved_keys),
            'lost_keys': list(lost_keys),
            'new_keys': list(new_keys),
            'preservation_ratio': len(preserved_keys) / len(input_data) if input_data else 0
        }

    def get_execution_timeline(self) -> List[Dict[str, Any]]:
        """Get chronological timeline of all events"""
        with self.lock:
            return [event.to_dict() for event in sorted(self.events, key=lambda e: e.timestamp)]

    def get_data_flow_graph(self) -> Dict[str, Any]:
        """Generate data flow graph from traced events"""
        nodes = set()
        edges = []

        with self.lock:
            for event in self.events:
                if event.event_type == EventType.DATA_FLOW:
                    source = event.event_data.get('source')
                    target = event.event_data.get('target')
                    if source and target:
                        nodes.add(source)
                        nodes.add(target)
                        edges.append({
                            'source': source,
                            'target': target,
                            'data_type': event.event_data.get('data_type'),
                            'timestamp': event.timestamp
                        })

        return {
            'nodes': list(nodes),
            'edges': edges
        }

    def export_trace(self, filepath: str):
        """Export complete trace to JSON file"""
        trace_data = {
            'workflow_id': self.workflow_id,
            'start_time': self.start_time,
            'events': self.get_execution_timeline(),
            'data_flow_graph': self.get_data_flow_graph(),
            'summary': self._generate_summary()
        }

        with open(filepath, 'w') as f:
            json.dump(trace_data, f, indent=2, default=str)

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate execution summary"""
        with self.lock:
            event_counts = {}
            for event in self.events:
                event_counts[event.event_type.value] = event_counts.get(event.event_type.value, 0) + 1

            return {
                'total_events': len(self.events),
                'event_counts': event_counts,
                'duration_seconds': time.time() - self.start_time,
                'components_involved': len(set(e.component_id for e in self.events))
            }
