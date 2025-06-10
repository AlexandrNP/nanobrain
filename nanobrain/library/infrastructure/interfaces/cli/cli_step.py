"""
CLI Step

A step-based CLI interface that integrates with NanoBrain workflows.

This module provides:
- Step-based CLI implementation
- Integration with workflow data units
- Configurable CLI behavior for workflows
- Event handling and lifecycle management
"""

import sys
import os
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

# Add src to path for core imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'src'))

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.data_unit import DataUnitBase
from nanobrain.core.executor import ExecutorBase
from .base_cli import BaseCLI, CLIConfig


@dataclass
class CLIStepConfig(CLIConfig):
    """Configuration for CLI steps."""
    
    # Step-specific settings
    step_name: str = "cli_step"
    step_description: str = "CLI interface step"
    
    # Data handling
    input_key: str = "user_input"
    output_key: str = "cli_output"
    
    # Workflow integration
    auto_start: bool = True
    auto_stop: bool = True
    pass_through_data: bool = True
    
    # CLI behavior in workflow context
    interactive_in_workflow: bool = True
    handle_workflow_events: bool = True
    
    # Additional step settings
    step_settings: Dict[str, Any] = field(default_factory=dict)


class CLIStep(BaseCLI, Step):
    """
    CLI Step that integrates CLI functionality with NanoBrain workflows.
    
    This class combines:
    - BaseCLI functionality for command-line interaction
    - Step interface for workflow integration
    - Data unit management for input/output
    - Event handling for workflow coordination
    
    Features:
    - Configurable CLI behavior within workflows
    - Automatic data unit integration
    - Event-driven workflow coordination
    - Extensible command system
    - Progress tracking and status reporting
    """
    
    def __init__(self, 
                 input_data_unit: Optional[DataUnitBase] = None,
                 output_data_unit: Optional[DataUnitBase] = None,
                 config: Optional[CLIStepConfig] = None,
                 executor: Optional[ExecutorBase] = None,
                 **kwargs):
        """
        Initialize the CLI Step.
        
        Args:
            input_data_unit: Data unit for user input
            output_data_unit: Data unit for CLI output
            config: CLI step configuration
            executor: Optional executor for async operations
            **kwargs: Additional configuration
        """
        # Initialize configuration
        self.cli_config = config or CLIStepConfig()
        
        # Create step configuration
        step_config = StepConfig(
            name=self.cli_config.step_name,
            description=self.cli_config.step_description
        )
        
        # Initialize both parent classes
        BaseCLI.__init__(self, self.cli_config, **kwargs)
        Step.__init__(self, step_config, executor, **kwargs)
        
        # Data units
        self.input_data_unit = input_data_unit
        self.output_data_unit = output_data_unit
        
        # Workflow state
        self.workflow_active = False
        self.current_workflow_data = None
        
        # Override logger to use step logger
        self.logger = self.step_logger
        
        self.logger.info(f"CLI Step {self.cli_config.step_name} initialized")
    
    async def initialize(self) -> None:
        """Initialize the CLI step."""
        # Initialize both parent classes
        await BaseCLI.initialize(self)
        await Step.initialize(self)
        
        # Auto-start if configured
        if self.cli_config.auto_start and self.cli_config.interactive_in_workflow:
            await self.start()
    
    async def shutdown(self) -> None:
        """Shutdown the CLI step."""
        # Auto-stop if configured
        if self.cli_config.auto_stop:
            await self.stop()
        
        # Shutdown both parent classes
        await BaseCLI.shutdown(self)
        await Step.shutdown(self)
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process workflow inputs through the CLI.
        
        Args:
            inputs: Input data from workflow
            
        Returns:
            Dict[str, Any]: Processed output for workflow
        """
        try:
            self.workflow_active = True
            self.current_workflow_data = inputs
            
            # Extract input data
            input_data = inputs.get(self.cli_config.input_key, inputs)
            
            # Handle different input types
            if isinstance(input_data, dict):
                message = input_data.get('message', str(input_data))
                command = input_data.get('command')
                
                if command and self.config.enable_commands:
                    result = await self._handle_workflow_command(command, input_data)
                else:
                    result = await self._handle_workflow_input(message, input_data)
            else:
                result = await self._handle_workflow_input(str(input_data), inputs)
            
            # Store in output data unit if available
            if self.output_data_unit:
                await self.output_data_unit.set(result)
            
            # Prepare output
            output = {self.cli_config.output_key: result}
            
            # Pass through original data if configured
            if self.cli_config.pass_through_data:
                output.update(inputs)
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error in CLI step processing: {e}")
            return {
                'error': str(e),
                'message': 'CLI step processing failed',
                self.cli_config.output_key: None
            }
        finally:
            self.workflow_active = False
            self.current_workflow_data = None
    
    async def _handle_workflow_input(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle user input in workflow context.
        
        Args:
            user_input: User's input message
            context: Workflow context data
            
        Returns:
            Dict[str, Any]: Processed input data
        """
        # Store user input in data unit
        if self.input_data_unit:
            await self.input_data_unit.set({
                'message': user_input,
                'timestamp': self._get_timestamp(),
                'source': 'cli_step',
                'context': context
            })
        
        # Process through base CLI
        result = await self._handle_user_input(user_input)
        
        # Format for workflow
        return {
            'message': user_input,
            'processed_result': result,
            'timestamp': self._get_timestamp(),
            'source': 'cli_step',
            'workflow_context': context
        }
    
    async def _handle_workflow_command(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle command in workflow context.
        
        Args:
            command: Command to execute
            context: Workflow context data
            
        Returns:
            Dict[str, Any]: Command result
        """
        try:
            # Execute command
            result = await self.execute_command(command)
            
            return {
                'command': command,
                'result': result,
                'timestamp': self._get_timestamp(),
                'success': True,
                'workflow_context': context
            }
            
        except Exception as e:
            return {
                'command': command,
                'error': str(e),
                'timestamp': self._get_timestamp(),
                'success': False,
                'workflow_context': context
            }
    
    async def handle_output(self, output_data: Any) -> None:
        """
        Handle output display in workflow context.
        
        Args:
            output_data: Data to display
        """
        try:
            if isinstance(output_data, dict):
                message = output_data.get('response', str(output_data))
            else:
                message = str(output_data)
            
            # Display through CLI
            await self.print_output(message)
            
            # Store in output data unit
            if self.output_data_unit:
                await self.output_data_unit.set({
                    'message': message,
                    'timestamp': self._get_timestamp(),
                    'source': 'cli_step_output'
                })
            
        except Exception as e:
            self.logger.error(f"Error handling output: {e}")
            await self.print_error(f"Error displaying output: {e}")
    
    # CLI-specific implementations
    
    async def _setup_commands(self) -> None:
        """Setup CLI step specific commands."""
        # Add workflow-specific commands
        self.register_command("workflow", self._cmd_workflow, "Show workflow information")
        self.register_command("data", self._cmd_data, "Show current data units")
        self.register_command("step", self._cmd_step, "Show step information")
    
    async def _handle_user_input(self, user_input: str) -> Any:
        """
        Handle user input processing.
        
        Args:
            user_input: User's input
            
        Returns:
            Processed result
        """
        # Default implementation - can be overridden
        return {
            'input': user_input,
            'processed': True,
            'timestamp': self._get_timestamp()
        }
    
    async def _format_output(self, data: Any) -> str:
        """
        Format output for display.
        
        Args:
            data: Data to format
            
        Returns:
            Formatted string
        """
        if isinstance(data, dict):
            if 'message' in data:
                return str(data['message'])
            elif 'result' in data:
                return str(data['result'])
        
        return str(data)
    
    async def _cleanup(self) -> None:
        """Cleanup CLI step resources."""
        # Close data units if needed
        if self.input_data_unit and hasattr(self.input_data_unit, 'close'):
            await self.input_data_unit.close()
        
        if self.output_data_unit and hasattr(self.output_data_unit, 'close'):
            await self.output_data_unit.close()
    
    # Workflow-specific commands
    
    async def _cmd_workflow(self) -> str:
        """Workflow information command."""
        if not self.workflow_active:
            return "No active workflow"
        
        info_lines = [
            "Workflow Information:",
            f"  Step: {self.cli_config.step_name}",
            f"  Active: {self.workflow_active}",
            f"  Input data unit: {type(self.input_data_unit).__name__ if self.input_data_unit else 'None'}",
            f"  Output data unit: {type(self.output_data_unit).__name__ if self.output_data_unit else 'None'}"
        ]
        
        if self.current_workflow_data:
            info_lines.append(f"  Current data keys: {list(self.current_workflow_data.keys())}")
        
        return "\n".join(info_lines)
    
    async def _cmd_data(self) -> str:
        """Data units information command."""
        info_lines = ["Data Units:"]
        
        if self.input_data_unit:
            try:
                input_data = await self.input_data_unit.get()
                info_lines.append(f"  Input: {type(input_data).__name__ if input_data else 'Empty'}")
            except Exception as e:
                info_lines.append(f"  Input: Error - {e}")
        else:
            info_lines.append("  Input: Not configured")
        
        if self.output_data_unit:
            try:
                output_data = await self.output_data_unit.get()
                info_lines.append(f"  Output: {type(output_data).__name__ if output_data else 'Empty'}")
            except Exception as e:
                info_lines.append(f"  Output: Error - {e}")
        else:
            info_lines.append("  Output: Not configured")
        
        return "\n".join(info_lines)
    
    async def _cmd_step(self) -> str:
        """Step information command."""
        stats = self.get_performance_stats()
        
        info_lines = [
            f"Step Information:",
            f"  Name: {self.config.name}",
            f"  Description: {self.config.description}",
            f"  Execution count: {stats.get('execution_count', 0)}",
            f"  Error count: {stats.get('error_count', 0)}",
            f"  Is initialized: {self.is_initialized}",
            f"  CLI running: {self.is_running}"
        ]
        
        return "\n".join(info_lines)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


# Convenience function for creating CLI steps
def create_cli_step(input_data_unit: Optional[DataUnitBase] = None,
                   output_data_unit: Optional[DataUnitBase] = None,
                   config: Optional[CLIStepConfig] = None,
                   **kwargs) -> CLIStep:
    """
    Create a CLI step with optional configuration.
    
    Args:
        input_data_unit: Input data unit
        output_data_unit: Output data unit
        config: CLI step configuration
        **kwargs: Additional configuration options
        
    Returns:
        Configured CLI step
    """
    return CLIStep(
        input_data_unit=input_data_unit,
        output_data_unit=output_data_unit,
        config=config,
        **kwargs
    ) 