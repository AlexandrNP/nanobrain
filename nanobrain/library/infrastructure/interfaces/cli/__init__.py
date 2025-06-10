"""
NanoBrain Library - CLI Interfaces

Configurable command-line interface components for the NanoBrain framework.

This module provides:
- Base CLI interface classes that can be extended
- Interactive CLI step for workflows
- Command processing and response formatting
- Configuration management for CLI behavior
- Progress indicators and user experience enhancements

Components:
- BaseCLI: Abstract base class for CLI implementations
- InteractiveCLI: Full-featured interactive CLI
- CLIStep: Step-based CLI for workflow integration
- CommandProcessor: Command parsing and execution
- ResponseFormatter: Output formatting and display
- CLIConfig: Configuration management
"""

from .base_cli import BaseCLI, CLIConfig
from .interactive_cli import InteractiveCLI, InteractiveCLIConfig
from .cli_step import CLIStep, CLIStepConfig
from .command_processor import CommandProcessor, CLICommand, CLIContext
from .response_formatter import ResponseFormatter, FormatterConfig
from .progress_indicator import ProgressIndicator, ProgressConfig

__all__ = [
    # Base classes
    'BaseCLI',
    'CLIConfig',
    
    # Interactive CLI
    'InteractiveCLI',
    'InteractiveCLIConfig',
    
    # Step-based CLI
    'CLIStep',
    'CLIStepConfig',
    
    # Command processing
    'CommandProcessor',
    'CLICommand',
    'CLIContext',
    
    # Response formatting
    'ResponseFormatter',
    'FormatterConfig',
    
    # Progress indication
    'ProgressIndicator',
    'ProgressConfig',
] 