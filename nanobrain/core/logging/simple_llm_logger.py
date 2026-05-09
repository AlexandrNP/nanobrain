"""
BRUTAL TRUTH: Ultra-Simple LLM Logging

My previous "LLM-optimized" system was STILL TOO COMPLEX. 
LLMs need DEAD SIMPLE patterns, not more structured complexity.

This provides ACTUALLY simple logging that LLMs can process:
- Single line per event
- No JSON, no complex structures
- Predictable field order
- Easy regex patterns
- Maximum signal, zero noise
"""

import time
import logging

logger = logging.getLogger("LLM_SIMPLE")


class SimpleLLMLogger:
    """
    BRUTAL TRUTH: Dead simple logging for LLM processing.
    
    Format: TIMESTAMP|EVENT|COMPONENT|STATUS|MESSAGE
    
    That's it. No JSON, no complex structures, no over-engineering.
    LLMs can easily parse this with simple string operations.
    """
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.start_time = time.time()
    
    def _log(self, event: str, component: str, status: str, message: str):
        """Log in ultra-simple format"""
        timestamp = f"{time.time() - self.start_time:.3f}"
        log_line = f"{timestamp}|{event}|{component}|{status}|{message}"
        logger.info(log_line)
    
    def workflow_start(self, steps: int, links: int):
        """Log workflow start"""
        self._log("WORKFLOW_START", self.workflow_id, "INFO", f"Started with {steps} steps and {links} links")
    
    def step_start(self, step_id: str, input_keys: list):
        """Log step execution start"""
        keys_str = ",".join(input_keys) if input_keys else "none"
        self._log("STEP_START", step_id, "INFO", f"Executing with inputs: {keys_str}")
    
    def step_complete(self, step_id: str, duration_ms: float, output_keys: list):
        """Log step completion"""
        keys_str = ",".join(output_keys) if output_keys else "none"
        self._log("STEP_COMPLETE", step_id, "SUCCESS", f"Completed in {duration_ms:.0f}ms with outputs: {keys_str}")
    
    def step_error(self, step_id: str, error_type: str, error_msg: str):
        """Log step error"""
        self._log("STEP_ERROR", step_id, "ERROR", f"{error_type}: {error_msg}")
    
    def data_flow(self, source: str, target: str, data_type: str):
        """Log data flow"""
        self._log("DATA_FLOW", f"{source}->{target}", "INFO", f"Transferred {data_type}")
    
    def data_lost(self, component: str, lost_keys: list):
        """Log data loss"""
        keys_str = ",".join(lost_keys)
        self._log("DATA_LOST", component, "WARNING", f"Lost keys: {keys_str}")
    
    def performance_issue(self, component: str, issue: str, value: str):
        """Log performance issue"""
        self._log("PERFORMANCE", component, "WARNING", f"{issue}: {value}")
    
    def query_info(self, component: str, query_preview: str):
        """Log query information for debugging"""
        preview = query_preview[:50] + "..." if len(query_preview) > 50 else query_preview
        self._log("QUERY_INFO", component, "INFO", f"Query: {preview}")


# Global simple logger instance
_simple_loggers = {}


def get_simple_logger(workflow_id: str) -> SimpleLLMLogger:
    """Get or create simple logger for workflow"""
    if workflow_id not in _simple_loggers:
        _simple_loggers[workflow_id] = SimpleLLMLogger(workflow_id)
    return _simple_loggers[workflow_id]


def log_for_llm(workflow_id: str, event: str, component: str, status: str, message: str):
    """Direct logging function for LLM processing"""
    logger = get_simple_logger(workflow_id)
    logger._log(event, component, status, message)


# LLM Processing Helper Functions
def extract_errors_for_llm(log_text: str) -> list:
    """Extract error events in LLM-friendly format"""
    errors = []
    for line in log_text.split('\n'):
        if '|STEP_ERROR|' in line or '|ERROR|' in line:
            parts = line.split('|')
            if len(parts) >= 5:
                errors.append({
                    'timestamp': parts[0],
                    'component': parts[2], 
                    'message': parts[4]
                })
    return errors


def extract_performance_issues_for_llm(log_text: str) -> list:
    """Extract performance issues in LLM-friendly format"""
    issues = []
    for line in log_text.split('\n'):
        if '|PERFORMANCE|' in line or 'WARNING|' in line:
            parts = line.split('|')
            if len(parts) >= 5:
                issues.append({
                    'timestamp': parts[0],
                    'component': parts[2],
                    'issue': parts[4]
                })
    return issues


def extract_data_flow_for_llm(log_text: str) -> list:
    """Extract data flow events in LLM-friendly format"""
    flows = []
    for line in log_text.split('\n'):
        if '|DATA_FLOW|' in line:
            parts = line.split('|')
            if len(parts) >= 5:
                component = parts[2]
                if '->' in component:
                    source, target = component.split('->', 1)
                    flows.append({
                        'timestamp': parts[0],
                        'source': source,
                        'target': target,
                        'data_type': parts[4].replace('Transferred ', '')
                    })
    return flows


def generate_llm_summary(log_text: str) -> str:
    """Generate ultra-simple summary for LLM processing"""
    lines = log_text.strip().split('\n')
    if not lines:
        return "NO_EVENTS"
    
    events = {}
    errors = []
    performance_issues = []
    
    for line in lines:
        parts = line.split('|')
        if len(parts) >= 3:
            event_type = parts[1]
            events[event_type] = events.get(event_type, 0) + 1
            
            if 'ERROR' in parts[3]:
                errors.append(f"{parts[2]}: {parts[4]}")
            elif 'PERFORMANCE' in parts[1]:
                performance_issues.append(f"{parts[2]}: {parts[4]}")
    
    summary = []
    summary.append(f"TOTAL_EVENTS: {len(lines)}")
    
    for event_type, count in events.items():
        summary.append(f"{event_type}: {count}")
    
    if errors:
        summary.append("ERRORS:")
        for error in errors:
            summary.append(f"  {error}")
    
    if performance_issues:
        summary.append("PERFORMANCE_ISSUES:")
        for issue in performance_issues:
            summary.append(f"  {issue}")
    
    return "\n".join(summary)
