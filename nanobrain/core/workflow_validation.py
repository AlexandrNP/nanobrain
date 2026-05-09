"""
BRUTAL TRUTH: Real workflow validation that actually works.

The framework's built-in validation is completely broken and lies about success.
This module implements independent validation that catches systemic failures.
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from nanobrain.core.workflow import Workflow

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    CRITICAL = "CRITICAL"  # Workflow will not function
    ERROR = "ERROR"       # Major functionality broken
    WARNING = "WARNING"   # Potential issues
    INFO = "INFO"         # Informational


@dataclass
class ValidationIssue:
    """A validation issue found in the workflow"""
    severity: ValidationSeverity
    component: str
    issue: str
    details: str
    fix_suggestion: Optional[str] = None


class WorkflowValidator:
    """
    BRUTAL TRUTH: Independent workflow validation that actually works.
    
    This validator doesn't trust the framework's lying validation system.
    It performs real checks to ensure workflows are actually functional.
    """
    
    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.issues: List[ValidationIssue] = []
        
    def validate(self) -> Tuple[bool, List[ValidationIssue]]:
        """
        Perform comprehensive workflow validation.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
            
        BRUTAL TRUTH: If this returns False, the workflow WILL NOT WORK.
        Don't trust the framework's validation - trust this.
        """
        self.issues = []
        
        logger.info("🔍 BRUTAL VALIDATION: Starting independent workflow validation")
        
        # Core structural validation
        self._validate_workflow_structure()
        self._validate_step_registration()
        self._validate_data_unit_mapping()
        self._validate_link_connectivity()
        self._validate_trigger_system()
        self._validate_execution_graph()
        
        # Determine if workflow is functional
        critical_issues = [i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
        
        is_valid = len(critical_issues) == 0 and len(error_issues) == 0
        
        if not is_valid:
            logger.error(f"❌ WORKFLOW VALIDATION FAILED: {len(critical_issues)} critical, {len(error_issues)} error issues")
        else:
            logger.info("✅ WORKFLOW VALIDATION PASSED: Workflow is functional")
            
        return is_valid, self.issues
    
    def _validate_workflow_structure(self):
        """Validate basic workflow structure"""
        if not hasattr(self.workflow, 'child_steps'):
            self._add_issue(ValidationSeverity.CRITICAL, "workflow", 
                          "No child_steps attribute", 
                          "Workflow object is malformed")
            return
            
        if not self.workflow.child_steps:
            self._add_issue(ValidationSeverity.CRITICAL, "workflow",
                          "No steps defined", 
                          "Workflow has zero steps - cannot execute")
            
        if not hasattr(self.workflow, 'links'):
            self._add_issue(ValidationSeverity.ERROR, "workflow",
                          "No links attribute",
                          "Workflow cannot connect steps")
    
    def _validate_step_registration(self):
        """Validate that all steps are properly registered"""
        if not hasattr(self.workflow, 'child_steps'):
            return
            
        for step_name, step in self.workflow.child_steps.items():
            # Check if step has basic step interface (more flexible than isinstance)
            if not hasattr(step, 'process'):
                self._add_issue(ValidationSeverity.CRITICAL, f"step.{step_name}",
                              f"Invalid step type: {type(step)}",
                              "Step does not have process method")
                              
            if not hasattr(step, 'step_input_data_units'):
                self._add_issue(ValidationSeverity.CRITICAL, f"step.{step_name}",
                              "No input data units",
                              "Step cannot receive data")
                              
            if not hasattr(step, 'step_output_data_units'):
                self._add_issue(ValidationSeverity.CRITICAL, f"step.{step_name}",
                              "No output data units",
                              "Step cannot produce data")

    def _validate_data_unit_mapping(self):
        """
        BRUTAL TRUTH: Validate that data units can actually be mapped to steps.

        This is where the framework fails catastrophically - it cannot map
        data unit names to their parent steps, breaking all links.
        """
        if not hasattr(self.workflow, 'child_steps'):
            return

        # Build mapping of data unit names to steps
        data_unit_to_step = {}

        # Map workflow-level data units
        if hasattr(self.workflow, 'workflow_input_data_units'):
            for du_name, du in self.workflow.workflow_input_data_units.items():
                data_unit_to_step[du_name] = "workflow"

        if hasattr(self.workflow, 'workflow_output_data_units'):
            for du_name, du in self.workflow.workflow_output_data_units.items():
                data_unit_to_step[du_name] = "workflow"

        # Map step-level data units
        for step_name, step in self.workflow.child_steps.items():
            if hasattr(step, 'step_input_data_units'):
                for du_name, du in step.step_input_data_units.items():
                    full_name = f"{step_name}.{du_name}"
                    data_unit_to_step[full_name] = step_name
                    data_unit_to_step[du_name] = step_name  # Also map short name

            if hasattr(step, 'step_output_data_units'):
                for du_name, du in step.step_output_data_units.items():
                    full_name = f"{step_name}.{du_name}"
                    data_unit_to_step[full_name] = step_name
                    data_unit_to_step[du_name] = step_name  # Also map short name

        # Store for link validation
        self._data_unit_mapping = data_unit_to_step

        logger.info(f"🔍 MAPPED DATA UNITS: {list(data_unit_to_step.keys())}")

    def _validate_link_connectivity(self):
        """
        BRUTAL TRUTH: Validate that links can actually connect data units.

        The framework silently fails here, creating broken workflows.
        """
        if not hasattr(self.workflow, 'workflow_config') or not self.workflow.workflow_config:
            self._add_issue(ValidationSeverity.ERROR, "links",
                          "No workflow config available",
                          "Cannot validate link configuration")
            return

        config = self.workflow.workflow_config
        if not hasattr(config, 'links') or not config.links:
            self._add_issue(ValidationSeverity.WARNING, "links",
                          "No links defined in configuration",
                          "Steps will not be connected")
            return

        functional_links = 0

        for link_name, link_config in config.links.items():
            if not hasattr(link_config, 'config'):
                self._add_issue(ValidationSeverity.ERROR, f"link.{link_name}",
                              "No link config",
                              "Link configuration is malformed")
                continue

            source = getattr(link_config.config, 'source', None)
            target = getattr(link_config.config, 'target', None)

            if not source or not target:
                self._add_issue(ValidationSeverity.CRITICAL, f"link.{link_name}",
                              f"Missing source ({source}) or target ({target})",
                              "Link cannot connect anything")
                continue

            # Check if source and target can be mapped to steps
            source_step = self._resolve_data_unit_to_step(source)
            target_step = self._resolve_data_unit_to_step(target)

            if not source_step:
                self._add_issue(ValidationSeverity.CRITICAL, f"link.{link_name}",
                              f"Cannot resolve source data unit: {source}",
                              f"Available data units: {list(self._data_unit_mapping.keys())}")

            if not target_step:
                self._add_issue(ValidationSeverity.CRITICAL, f"link.{link_name}",
                              f"Cannot resolve target data unit: {target}",
                              f"Available data units: {list(self._data_unit_mapping.keys())}")

            if source_step and target_step:
                functional_links += 1
                logger.info(f"✅ FUNCTIONAL LINK: {link_name} ({source} -> {target})")
            else:
                logger.error(f"❌ BROKEN LINK: {link_name} ({source} -> {target})")

        if functional_links == 0:
            self._add_issue(ValidationSeverity.CRITICAL, "links",
                          "ZERO functional links",
                          "Workflow cannot transfer data between steps - COMPLETELY BROKEN")
        else:
            logger.info(f"✅ FUNCTIONAL LINKS: {functional_links}/{len(config.links)}")

    def _resolve_data_unit_to_step(self, data_unit_name: str) -> Optional[str]:
        """Resolve a data unit name to its parent step"""
        if not hasattr(self, '_data_unit_mapping'):
            return None

        # Try exact match first
        if data_unit_name in self._data_unit_mapping:
            return self._data_unit_mapping[data_unit_name]

        # Try to extract step name from qualified name
        if '.' in data_unit_name:
            step_name = data_unit_name.split('.')[0]
            if step_name in self.workflow.child_steps:
                return step_name

        return None

    def _validate_trigger_system(self):
        """
        BRUTAL TRUTH: Validate that triggers are actually registered and functional.

        The framework often silently fails to register triggers.
        """
        if not hasattr(self.workflow, 'child_steps'):
            return

        steps_with_triggers = 0

        for step_name, step in self.workflow.child_steps.items():
            # Check if step has trigger configuration
            has_trigger_config = False

            if hasattr(step, 'step_trigger_configs') and step.step_trigger_configs:
                has_trigger_config = True

            # Check if step has actual triggers registered
            has_registered_triggers = False

            if hasattr(step, 'step_triggers') and step.step_triggers:
                has_registered_triggers = True
                steps_with_triggers += 1

            # Check input data units for change listeners
            has_change_listeners = False

            if hasattr(step, 'step_input_data_units'):
                for du_name, du in step.step_input_data_units.items():
                    if hasattr(du, '_change_listeners') and du._change_listeners:
                        has_change_listeners = True
                        break

            # Validate trigger functionality
            if not has_trigger_config and not has_registered_triggers and not has_change_listeners:
                self._add_issue(ValidationSeverity.ERROR, f"step.{step_name}",
                              "No trigger mechanism",
                              "Step will never execute - no triggers, no listeners")
            elif has_trigger_config and not has_registered_triggers:
                self._add_issue(ValidationSeverity.CRITICAL, f"step.{step_name}",
                              "Trigger config exists but triggers not registered",
                              "Framework failed to register triggers")

        if steps_with_triggers == 0:
            self._add_issue(ValidationSeverity.CRITICAL, "triggers",
                          "NO STEPS HAVE TRIGGERS",
                          "Workflow will never execute - no trigger mechanism")

    def _validate_execution_graph(self):
        """Validate that the workflow has a valid execution graph"""
        if not hasattr(self.workflow, 'child_steps'):
            return

        # Check for isolated steps (no incoming or outgoing connections)
        connected_steps = set()

        if hasattr(self.workflow, 'workflow_config') and self.workflow.workflow_config:
            config = self.workflow.workflow_config
            if hasattr(config, 'links') and config.links:
                for link_name, link_config in config.links.items():
                    if hasattr(link_config, 'config'):
                        source = getattr(link_config.config, 'source', None)
                        target = getattr(link_config.config, 'target', None)

                        source_step = self._resolve_data_unit_to_step(source) if source else None
                        target_step = self._resolve_data_unit_to_step(target) if target else None

                        if source_step:
                            connected_steps.add(source_step)
                        if target_step:
                            connected_steps.add(target_step)

        # Check for isolated steps
        for step_name in self.workflow.child_steps.keys():
            if step_name not in connected_steps:
                self._add_issue(ValidationSeverity.WARNING, f"step.{step_name}",
                              "Isolated step",
                              "Step has no connections - may not receive or send data")

    def _add_issue(self, severity: ValidationSeverity, component: str, issue: str, details: str, fix_suggestion: str = None):
        """Add a validation issue"""
        self.issues.append(ValidationIssue(
            severity=severity,
            component=component,
            issue=issue,
            details=details,
            fix_suggestion=fix_suggestion
        ))

        # Log the issue immediately
        level = logging.ERROR if severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR] else logging.WARNING
        logger.log(level, f"🔍 VALIDATION {severity.value}: {component} - {issue}: {details}")


def validate_workflow(workflow: Workflow) -> Tuple[bool, List[ValidationIssue]]:
    """
    BRUTAL TRUTH: Validate a workflow and return whether it's actually functional.

    Args:
        workflow: The workflow to validate

    Returns:
        Tuple of (is_functional, list_of_issues)

    This is the function you should call instead of trusting the framework's
    lying validation system.
    """
    validator = WorkflowValidator(workflow)
    return validator.validate()


def print_validation_report(issues: List[ValidationIssue]):
    """Print a human-readable validation report"""
    if not issues:
        print("✅ WORKFLOW VALIDATION: No issues found")
        return

    print(f"\n🔍 WORKFLOW VALIDATION REPORT: {len(issues)} issues found")
    print("=" * 80)

    for issue in issues:
        icon = "🔥" if issue.severity == ValidationSeverity.CRITICAL else "❌" if issue.severity == ValidationSeverity.ERROR else "⚠️"
        print(f"{icon} {issue.severity.value}: {issue.component}")
        print(f"   Issue: {issue.issue}")
        print(f"   Details: {issue.details}")
        if issue.fix_suggestion:
            print(f"   Fix: {issue.fix_suggestion}")
        print()
