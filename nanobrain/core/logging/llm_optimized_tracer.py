"""
BRUTAL TRUTH: LLM-Optimized Workflow Logging System

The previous enhanced logging system was TERRIBLE for LLM processing because:
1. Inconsistent formatting made parsing difficult
2. Too much noise buried critical information
3. No clear patterns for LLMs to learn from
4. Human-centric design instead of machine-centric

This module provides REAL LLM-optimized logging with:
- Consistent structured format for ALL log entries
- Clear semantic markers for easy extraction
- Minimal noise with maximum signal
- Predictable patterns for LLM processing
- Contextual grouping of related events
"""

import json
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LLMEventType(Enum):
    """LLM-friendly event types with clear semantic meaning"""
    WORKFLOW_START = "WORKFLOW_START"
    WORKFLOW_END = "WORKFLOW_END"
    STEP_EXECUTE = "STEP_EXECUTE"
    STEP_COMPLETE = "STEP_COMPLETE"
    STEP_ERROR = "STEP_ERROR"
    DATA_FLOW = "DATA_FLOW"
    DATA_LOST = "DATA_LOST"
    TRIGGER_FIRE = "TRIGGER_FIRE"
    PERFORMANCE_ISSUE = "PERFORMANCE_ISSUE"
    CRITICAL_ERROR = "CRITICAL_ERROR"


@dataclass
class LLMLogEntry:
    """
    BRUTAL TRUTH: Single, consistent log entry format for LLM processing.
    
    Every log entry follows EXACTLY this structure - no exceptions.
    LLMs can reliably parse this format and extract specific information.
    """
    timestamp: float
    event_type: LLMEventType
    component: str
    status: str  # SUCCESS, ERROR, WARNING, INFO
    message: str
    context: Dict[str, Any]
    duration_ms: Optional[float] = None
    parent_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_llm_format(self) -> str:
        """
        Convert to LLM-optimized format with clear markers and consistent structure.
        
        Format: [TIMESTAMP] EVENT_TYPE:STATUS COMPONENT | MESSAGE | CONTEXT
        """
        timestamp_str = f"{self.timestamp:.3f}"
        context_str = json.dumps(self.context, separators=(',', ':'))
        duration_str = f" ({self.duration_ms:.1f}ms)" if self.duration_ms else ""
        
        return f"[{timestamp_str}] {self.event_type.value}:{self.status} {self.component}{duration_str} | {self.message} | {context_str}"


class LLMOptimizedTracer:
    """
    BRUTAL TRUTH: Workflow tracer optimized for LLM processing convenience.
    
    This tracer provides:
    - Consistent format for ALL log entries
    - Clear semantic markers for easy extraction
    - Minimal noise with maximum signal
    - Predictable patterns for LLM learning
    - Contextual grouping of related events
    """
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.entries: List[LLMLogEntry] = []
        self.start_time = time.time()
        self.event_counter = 0
        
    def _generate_correlation_id(self) -> str:
        """Generate correlation ID for related events"""
        self.event_counter += 1
        return f"{self.workflow_id}_{self.event_counter:04d}"
    
    def log_workflow_start(self, context: Dict[str, Any] = None):
        """Log workflow start with LLM-friendly format"""
        entry = LLMLogEntry(
            timestamp=time.time(),
            event_type=LLMEventType.WORKFLOW_START,
            component=self.workflow_id,
            status="INFO",
            message=f"Workflow {self.workflow_id} started",
            context=context or {},
            correlation_id=self._generate_correlation_id()
        )
        self._emit_log(entry)
    
    def log_step_execution(self, step_id: str, input_data: Dict[str, Any], correlation_id: str = None):
        """Log step execution start with essential context only"""
        # Extract only LLM-relevant information from input data
        llm_context = {
            "input_keys": list(input_data.keys()),
            "input_types": {k: type(v).__name__ for k, v in input_data.items()},
            "data_summary": self._summarize_for_llm(input_data)
        }
        
        entry = LLMLogEntry(
            timestamp=time.time(),
            event_type=LLMEventType.STEP_EXECUTE,
            component=step_id,
            status="INFO",
            message=f"Step {step_id} executing",
            context=llm_context,
            correlation_id=correlation_id or self._generate_correlation_id()
        )
        self._emit_log(entry)
        return entry.correlation_id
    
    def log_step_completion(self, step_id: str, output_data: Dict[str, Any], 
                          duration_ms: float, correlation_id: str = None):
        """Log step completion with output summary"""
        llm_context = {
            "output_keys": list(output_data.keys()),
            "output_types": {k: type(v).__name__ for k, v in output_data.items()},
            "data_summary": self._summarize_for_llm(output_data)
        }
        
        entry = LLMLogEntry(
            timestamp=time.time(),
            event_type=LLMEventType.STEP_COMPLETE,
            component=step_id,
            status="SUCCESS",
            message=f"Step {step_id} completed successfully",
            context=llm_context,
            duration_ms=duration_ms,
            correlation_id=correlation_id or self._generate_correlation_id()
        )
        self._emit_log(entry)
    
    def log_data_flow(self, source: str, target: str, data_summary: Dict[str, Any]):
        """Log data flow between components with LLM-friendly format"""
        entry = LLMLogEntry(
            timestamp=time.time(),
            event_type=LLMEventType.DATA_FLOW,
            component=f"{source}->{target}",
            status="INFO",
            message=f"Data flowing from {source} to {target}",
            context=data_summary,
            correlation_id=self._generate_correlation_id()
        )
        self._emit_log(entry)
    
    def log_data_loss(self, component: str, lost_keys: List[str], context: Dict[str, Any] = None):
        """Log data loss with clear LLM-parseable format"""
        entry = LLMLogEntry(
            timestamp=time.time(),
            event_type=LLMEventType.DATA_LOST,
            component=component,
            status="WARNING",
            message=f"Data loss detected in {component}: {', '.join(lost_keys)}",
            context={"lost_keys": lost_keys, **(context or {})},
            correlation_id=self._generate_correlation_id()
        )
        self._emit_log(entry)
    
    def log_performance_issue(self, component: str, issue_type: str, details: Dict[str, Any]):
        """Log performance issues with actionable context"""
        entry = LLMLogEntry(
            timestamp=time.time(),
            event_type=LLMEventType.PERFORMANCE_ISSUE,
            component=component,
            status="WARNING",
            message=f"Performance issue in {component}: {issue_type}",
            context=details,
            correlation_id=self._generate_correlation_id()
        )
        self._emit_log(entry)
    
    def log_critical_error(self, component: str, error_message: str, context: Dict[str, Any] = None):
        """Log critical errors with full context for debugging"""
        entry = LLMLogEntry(
            timestamp=time.time(),
            event_type=LLMEventType.CRITICAL_ERROR,
            component=component,
            status="ERROR",
            message=f"Critical error in {component}: {error_message}",
            context=context or {},
            correlation_id=self._generate_correlation_id()
        )
        self._emit_log(entry)
    
    def _summarize_for_llm(self, data: Any) -> Dict[str, Any]:
        """
        Create LLM-friendly data summary with essential information only.
        
        BRUTAL TRUTH: LLMs don't need full data dumps - they need semantic summaries.
        """
        if data is None:
            return {"type": "None", "content": "empty"}
        
        if isinstance(data, dict):
            # For dicts, provide key structure and sample values
            summary = {
                "type": "dict",
                "size": len(data),
                "keys": list(data.keys())[:5],  # Limit to first 5 keys
            }
            
            # Add sample values for important keys
            important_keys = ["query", "response", "original_query", "enhanced_query", "error"]
            for key in important_keys:
                if key in data:
                    value = str(data[key])[:100]  # Truncate to 100 chars
                    summary[f"sample_{key}"] = value
            
            return summary
        
        elif isinstance(data, str):
            return {
                "type": "string",
                "length": len(data),
                "preview": data[:100],  # First 100 characters
                "contains_query": "query" in data.lower(),
                "contains_error": "error" in data.lower()
            }
        
        elif isinstance(data, list):
            return {
                "type": "list",
                "size": len(data),
                "item_types": [type(item).__name__ for item in data[:3]]
            }
        
        else:
            return {
                "type": type(data).__name__,
                "preview": str(data)[:100]
            }
    
    def _emit_log(self, entry: LLMLogEntry):
        """Emit log entry in LLM-optimized format"""
        self.entries.append(entry)
        
        # Log to standard logger with LLM-optimized format
        log_line = entry.to_llm_format()
        
        if entry.status == "ERROR":
            logger.error(log_line)
        elif entry.status == "WARNING":
            logger.warning(log_line)
        else:
            logger.info(log_line)
    
    def get_llm_summary(self) -> str:
        """
        Generate LLM-friendly execution summary with clear patterns.
        
        BRUTAL TRUTH: This is what LLMs actually need - structured, predictable summaries.
        """
        if not self.entries:
            return "WORKFLOW_SUMMARY: No events recorded"
        
        # Group events by type for easy LLM processing
        events_by_type = {}
        for entry in self.entries:
            event_type = entry.event_type.value
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            events_by_type[event_type].append(entry)
        
        summary_lines = [
            f"WORKFLOW_SUMMARY: {self.workflow_id}",
            f"DURATION: {time.time() - self.start_time:.2f}s",
            f"TOTAL_EVENTS: {len(self.entries)}",
            ""
        ]
        
        # Add event type summaries
        for event_type, entries in events_by_type.items():
            summary_lines.append(f"{event_type}: {len(entries)} events")
            
            # Add specific details for critical event types
            if event_type == "STEP_ERROR" or event_type == "CRITICAL_ERROR":
                for entry in entries:
                    summary_lines.append(f"  ERROR: {entry.component} - {entry.message}")
            
            elif event_type == "DATA_LOST":
                for entry in entries:
                    lost_keys = entry.context.get("lost_keys", [])
                    summary_lines.append(f"  DATA_LOSS: {entry.component} lost {lost_keys}")
            
            elif event_type == "PERFORMANCE_ISSUE":
                for entry in entries:
                    summary_lines.append(f"  PERFORMANCE: {entry.component} - {entry.message}")
        
        return "\n".join(summary_lines)
