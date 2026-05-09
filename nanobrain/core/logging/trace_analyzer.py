"""
BRUTAL TRUTH: Workflow Trace Analysis and Visualization

This module provides tools to analyze workflow execution traces and generate
human-readable reports that actually help with debugging.
"""

from typing import Dict, List, Any
from datetime import datetime


class TraceAnalyzer:
    """
    BRUTAL TRUTH: Workflow trace analyzer that generates useful debugging reports.
    
    This analyzer can:
    - Generate execution timelines
    - Identify bottlenecks
    - Trace data flow issues
    - Find trigger execution problems
    - Highlight step execution failures
    """
    
    def __init__(self, trace_data: Dict[str, Any]):
        self.trace_data = trace_data
        self.events = trace_data.get('events', [])
        self.workflow_id = trace_data.get('workflow_id', 'unknown')
    
    def generate_execution_report(self) -> str:
        """Generate comprehensive execution report"""
        report = []
        report.append("=" * 80)
        report.append("🔍 WORKFLOW EXECUTION TRACE REPORT")
        report.append(f"Workflow: {self.workflow_id}")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 80)
        
        # Summary
        summary = self._analyze_summary()
        report.append("\n📊 EXECUTION SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Events: {summary['total_events']}")
        report.append(f"Duration: {summary['duration']:.2f}s")
        report.append(f"Components: {summary['components']}")
        report.append(f"Success Rate: {summary['success_rate']:.1%}")
        
        # Timeline
        timeline = self._analyze_timeline()
        report.append("\n⏱️ EXECUTION TIMELINE")
        report.append("-" * 40)
        for event in timeline[:20]:  # Show first 20 events
            report.append(f"{event['time']:8.3f}s | {event['icon']} {event['component']:25} | {event['description']}")
        
        if len(timeline) > 20:
            report.append(f"... and {len(timeline) - 20} more events")
        
        # Data Flow Analysis
        data_flow = self._analyze_data_flow()
        report.append("\n📊 DATA FLOW ANALYSIS")
        report.append("-" * 40)
        for flow in data_flow:
            report.append(f"🔗 {flow['source']} → {flow['target']}")
            report.append(f"   Type: {flow['data_type']}, Size: {flow['data_size']}")
            if flow['issues']:
                report.append(f"   ⚠️ Issues: {', '.join(flow['issues'])}")
        
        # Trigger Analysis
        triggers = self._analyze_triggers()
        report.append("\n⚡ TRIGGER ANALYSIS")
        report.append("-" * 40)
        for trigger in triggers:
            status = "✅" if trigger['executed'] else "❌"
            report.append(f"{status} {trigger['trigger_id']} → {trigger['target_step']}")
            if not trigger['executed']:
                report.append("   ⚠️ Trigger fired but step did not execute")
        
        # Step Execution Analysis
        steps = self._analyze_step_execution()
        report.append("\n🔧 STEP EXECUTION ANALYSIS")
        report.append("-" * 40)
        for step in steps:
            status = "✅" if step['completed'] else "❌"
            duration = f"{step['duration']:.3f}s" if step['duration'] else "N/A"
            report.append(f"{status} {step['step_id']:20} | {duration:8} | {step['status']}")
            
            if step['data_issues']:
                report.append(f"   ⚠️ Data Issues: {', '.join(step['data_issues'])}")
        
        # Issues and Recommendations
        issues = self._identify_issues()
        if issues:
            report.append("\n🚨 IDENTIFIED ISSUES")
            report.append("-" * 40)
            for issue in issues:
                report.append(f"❌ {issue['severity'].upper()}: {issue['description']}")
                report.append(f"   Component: {issue['component']}")
                report.append(f"   Recommendation: {issue['recommendation']}")
                report.append("")
        
        return "\n".join(report)
    
    def _analyze_summary(self) -> Dict[str, Any]:
        """Analyze overall execution summary"""
        start_time = min(e['timestamp'] for e in self.events) if self.events else 0
        end_time = max(e['timestamp'] for e in self.events) if self.events else 0
        
        error_count = len([e for e in self.events if e['event_type'] == 'error'])
        success_rate = 1.0 - (error_count / len(self.events)) if self.events else 0
        
        components = set(e['component_id'] for e in self.events)
        
        return {
            'total_events': len(self.events),
            'duration': end_time - start_time,
            'components': len(components),
            'success_rate': success_rate,
            'error_count': error_count
        }
    
    def _analyze_timeline(self) -> List[Dict[str, Any]]:
        """Analyze execution timeline"""
        if not self.events:
            return []
            
        start_time = min(e['timestamp'] for e in self.events)
        timeline = []
        
        for event in self.events:
            relative_time = event['timestamp'] - start_time
            
            # Get event icon and description
            icons = {
                'workflow_start': '🚀',
                'workflow_end': '✅',
                'step_start': '🔧',
                'step_end': '✅',
                'trigger_fired': '⚡',
                'data_flow': '📊',
                'data_transform': '🔄',
                'link_transfer': '🔗',
                'error': '❌'
            }
            
            icon = icons.get(event['event_type'], '📋')
            component = f"{event['component_type']}.{event['component_id']}"
            description = self._get_event_description(event)
            
            timeline.append({
                'time': relative_time,
                'icon': icon,
                'component': component,
                'description': description,
                'event': event
            })
        
        return sorted(timeline, key=lambda x: x['time'])
    
    def _get_event_description(self, event: Dict[str, Any]) -> str:
        """Get human-readable description for event"""
        event_type = event['event_type']
        event_data = event.get('event_data', {})
        
        if event_type == 'step_start':
            return f"Started processing ({event_data.get('step_type', 'unknown')})"
        elif event_type == 'step_end':
            return "Completed processing"
        elif event_type == 'trigger_fired':
            return f"Triggered by {event_data.get('data_unit', 'unknown')} change"
        elif event_type == 'data_flow':
            return f"Data transfer: {event_data.get('data_type', 'unknown')} ({event_data.get('data_size', 0)} items)"
        elif event_type == 'data_transform':
            return f"Data transformation: {event_data.get('transformation_type', 'unknown')}"
        elif event_type == 'error':
            return f"Error: {event_data.get('error_message', 'unknown error')}"
        else:
            return event_type.replace('_', ' ').title()
    
    def _analyze_data_flow(self) -> List[Dict[str, Any]]:
        """Analyze data flow between components"""
        data_flows = []
        
        for event in self.events:
            if event['event_type'] == 'data_flow':
                event_data = event.get('event_data', {})
                
                # Identify potential issues
                issues = []
                if event_data.get('data_size', 0) == 0:
                    issues.append("Empty data transfer")
                if event_data.get('data_type') == 'unknown':
                    issues.append("Unknown data type")
                
                data_flows.append({
                    'source': event_data.get('source', 'unknown'),
                    'target': event_data.get('target', 'unknown'),
                    'data_type': event_data.get('data_type', 'unknown'),
                    'data_size': event_data.get('data_size', 0),
                    'timestamp': event['timestamp'],
                    'issues': issues
                })
        
        return data_flows
    
    def _analyze_triggers(self) -> List[Dict[str, Any]]:
        """Analyze trigger execution"""
        triggers = []
        trigger_events = [e for e in self.events if e['event_type'] == 'trigger_fired']
        
        for trigger_event in trigger_events:
            event_data = trigger_event.get('event_data', {})
            trigger_id = trigger_event['component_id']
            target_step = event_data.get('target_step', 'unknown')
            
            # Check if target step actually executed after trigger
            step_executed = any(
                e['event_type'] == 'step_start' and 
                e['component_id'] == target_step and
                e['timestamp'] > trigger_event['timestamp']
                for e in self.events
            )
            
            triggers.append({
                'trigger_id': trigger_id,
                'target_step': target_step,
                'timestamp': trigger_event['timestamp'],
                'executed': step_executed,
                'trigger_type': event_data.get('trigger_type', 'unknown')
            })
        
        return triggers
    
    def _analyze_step_execution(self) -> List[Dict[str, Any]]:
        """Analyze step execution patterns"""
        steps = {}
        
        for event in self.events:
            if event['event_type'] in ['step_start', 'step_end']:
                step_id = event['component_id']
                
                if step_id not in steps:
                    steps[step_id] = {
                        'step_id': step_id,
                        'start_time': None,
                        'end_time': None,
                        'completed': False,
                        'duration': None,
                        'status': 'Not started',
                        'data_issues': []
                    }
                
                if event['event_type'] == 'step_start':
                    steps[step_id]['start_time'] = event['timestamp']
                    steps[step_id]['status'] = 'Running'
                elif event['event_type'] == 'step_end':
                    steps[step_id]['end_time'] = event['timestamp']
                    steps[step_id]['completed'] = True
                    steps[step_id]['status'] = 'Completed'
                    
                    if steps[step_id]['start_time']:
                        steps[step_id]['duration'] = event['timestamp'] - steps[step_id]['start_time']
        
        # Check for data transformation issues
        for event in self.events:
            if event['event_type'] == 'data_transform':
                step_id = event['component_id']
                if step_id in steps:
                    event_data = event.get('event_data', {})
                    preservation = event_data.get('data_preserved', {})
                    
                    if preservation.get('preservation_ratio', 1.0) < 0.5:
                        steps[step_id]['data_issues'].append('Low data preservation')
                    
                    if preservation.get('lost_keys'):
                        steps[step_id]['data_issues'].append(f"Lost keys: {preservation['lost_keys']}")
        
        return list(steps.values())
    
    def _identify_issues(self) -> List[Dict[str, Any]]:
        """Identify potential issues in workflow execution"""
        issues = []
        
        # Check for triggers that fired but didn't execute steps
        triggers = self._analyze_triggers()
        for trigger in triggers:
            if not trigger['executed']:
                issues.append({
                    'severity': 'high',
                    'component': trigger['trigger_id'],
                    'description': f"Trigger fired but target step '{trigger['target_step']}' did not execute",
                    'recommendation': "Check step initialization and trigger binding"
                })
        
        # Check for steps that never completed
        steps = self._analyze_step_execution()
        for step in steps:
            if step['start_time'] and not step['completed']:
                issues.append({
                    'severity': 'high',
                    'component': step['step_id'],
                    'description': "Step started but never completed",
                    'recommendation': "Check for exceptions or infinite loops in step execution"
                })
        
        # Check for data preservation issues
        for step in steps:
            if step['data_issues']:
                issues.append({
                    'severity': 'medium',
                    'component': step['step_id'],
                    'description': f"Data issues: {', '.join(step['data_issues'])}",
                    'recommendation': "Review step implementation for proper data preservation"
                })
        
        return issues
