"""
Response Formatting Step - Pure Response Formatting and Presentation
===================================================================

Provides focused response formatting functionality as a proper NanoBrain workflow step,
enabling response presentation within event-driven workflow orchestration with complete
framework compliance.

**SINGLE RESPONSIBILITY**: Response Formatting Only
- Response data formatting and presentation
- Markdown and template-based formatting
- Multi-format output support (JSON, HTML, text)
- Error and status response formatting

**DOES NOT INCLUDE**:
- Business logic processing (handled by workflow steps)
- Data analysis or computation (delegated to processing steps)
- Communication protocols (handled by WebInterfaceStep)

This component follows NanoBrain framework patterns:
- Inherits from BaseStep for framework compliance
- Uses from_config pattern for component creation
- Provides comprehensive configuration validation
- Supports event-driven workflow orchestration

Usage:
    from nanobrain.library.infrastructure.steps import ResponseFormattingStep
    
    # Create via from_config (framework pattern)
    step = ResponseFormattingStep.from_config('config/response_formatting_step.yml')
    
    # Execute within workflow context
    await step.execute()
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.core.component_base import ComponentConfigurationError, ComponentDependencyError
from nanobrain.core.logging_system import get_logger
from nanobrain.core.data_unit import DataUnitBase, DataUnitMemory
from nanobrain.core.trigger import TriggerBase
from nanobrain.library.infrastructure.data.progress_update_data_unit import ProgressUpdateDataUnit
from nanobrain.library.infrastructure.data.session_context_data_unit import SessionContextDataUnit

from pydantic import BaseModel, Field, ConfigDict


class ResponseFormat(Enum):
    """Supported response output formats."""
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    PLAIN_TEXT = "plain_text"
    AUTO = "auto"


class ResponseFormattingStepConfig(StepConfig):
    """
    Configuration schema for ResponseFormattingStep
    
    Provides comprehensive response formatting configuration including output
    formats, templates, and presentation settings.
    """
    
    # Output format settings
    default_output_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Default response output format"
    )
    
    enable_markdown_formatting: bool = Field(
        default=True,
        description="Enable markdown formatting for text responses"
    )
    
    enable_json_formatting: bool = Field(
        default=True,
        description="Enable JSON formatting for structured data"
    )
    
    # Content processing settings
    max_response_length: int = Field(
        default=100000,
        description="Maximum response length in characters"
    )
    
    enable_content_truncation: bool = Field(
        default=False,
        description="Enable automatic content truncation for long responses"
    )
    
    truncation_message: str = Field(
        default="[Response truncated for length...]",
        description="Message to append when content is truncated"
    )
    
    # Template and styling settings
    enable_rich_formatting: bool = Field(
        default=True,
        description="Enable rich formatting with emojis and styling"
    )
    
    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Date format for timestamps in responses"
    )
    
    include_metadata: bool = Field(
        default=True,
        description="Include metadata like timestamps and processing time"
    )
    
    # Error handling settings
    enable_error_formatting: bool = Field(
        default=True,
        description="Enable specialized error response formatting"
    )
    
    include_troubleshooting_tips: bool = Field(
        default=True,
        description="Include troubleshooting tips in error responses"
    )
    
    # Performance settings
    enable_streaming_chunks: bool = Field(
        default=False,
        description="Enable response streaming chunk generation"
    )
    
    streaming_chunk_size: int = Field(
        default=200,
        description="Size of streaming chunks in characters"
    )
    
    # MANDATORY PYDANTIC V2 CONFIGURATION
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        use_enum_values=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "name": "response_formatter",
                    "description": "General response formatting for web interfaces",
                    "default_output_format": "markdown",
                    "enable_markdown_formatting": True,
                    "max_response_length": 8000,
                    "enable_rich_formatting": True,
                    "include_metadata": True
                }
            ],
            "nanobrain_metadata": {
                "framework_version": "2.0.0",
                "component_type": "response_formatting_step",
                "config_loading_method": "from_config_only",
                "supports_recursive_references": True
            }
        }
    )


class ResponseFormattingStep(BaseStep):
    """
    Response Formatting Step - Pure Response Formatting and Presentation
    ===================================================================
    
    Provides focused response formatting functionality as a proper NanoBrain workflow step,
    enabling response presentation within event-driven workflow orchestration with complete
    framework compliance.
    
    **SINGLE RESPONSIBILITY**: Response Formatting Only
    - Response data formatting and presentation
    - Markdown and template-based formatting
    - Multi-format output support (JSON, HTML, text)
    - Error and status response formatting
    
    **DOES NOT INCLUDE**:
    - Business logic processing (handled by workflow steps)
    - Data analysis or computation (delegated to processing steps)
    - Communication protocols (handled by WebInterfaceStep)
    
    **Core Architecture:**
        This step provides clean response formatting that integrates with NanoBrain's
        event-driven workflow orchestration by converting workflow results into
        user-friendly formatted responses, enabling professional presentation systems.
        
        * **Multi-Format Support**: Markdown, JSON, HTML, and plain text formatting
        * **Template System**: Configurable response templates and styling
        * **Content Management**: Length limits, truncation, and content validation
        * **Error Handling**: Specialized error response formatting and troubleshooting
    
    **Configuration Architecture:**
        ```yaml
        # Basic response formatting step configuration
        name: "response_formatter"
        description: "General response formatting for web interfaces"
        auto_initialize: true
        enable_logging: true
        
        # Formatting settings
        default_output_format: "markdown"
        enable_markdown_formatting: true
        enable_json_formatting: true
        max_response_length: 8000
        enable_content_truncation: true
        
        # Presentation settings
        enable_rich_formatting: true
        include_metadata: true
        date_format: "%Y-%m-%d %H:%M:%S"
        
        # Error handling
        enable_error_formatting: true
        include_troubleshooting_tips: true
        ```
    
    **Usage Patterns:**
        ```python
        from nanobrain.library.infrastructure.steps import ResponseFormattingStep
        
        # Create step from configuration
        formatter_step = ResponseFormattingStep.from_config('config/response_formatting_step.yml')
        
        # Execute within workflow context
        await formatter_step.execute()
        
        # Step automatically handles:
        # - Response data formatting and presentation
        # - Multi-format output generation
        # - Content truncation and validation
        # - Error response formatting
        ```
    
    Attributes:
        name (str): Step identifier for logging and debugging
        description (str): Human-readable step description
        logger (logging.Logger): Step-specific logger instance
        config (ResponseFormattingStepConfig): Step configuration instance
        
    Note:
        This step follows the mandatory from_config pattern and cannot be
        instantiated directly. All configurations must be loaded from YAML files
        using the from_config method.
    
    See Also:
        * :class:`BaseStep`: Base framework step interface
        * :class:`ResponseFormattingStepConfig`: Configuration schema
        * :class:`DataUnitMemory`: In-memory data storage for responses
        * :class:`ProgressUpdateDataUnit`: Progress data integration
    """
    
    # MANDATORY COMPONENT METADATA
    COMPONENT_TYPE: str = "response_formatting_step"
    REQUIRED_CONFIG_FIELDS: List[str] = ['name']
    
    # Define data unit interfaces
    input_data_units = {
        'progress_data': ProgressUpdateDataUnit,  # Progress data to format
        'session_context': SessionContextDataUnit,  # Session context for formatting
        'workflow_results': DataUnitMemory,  # Any workflow results to format
    }
    
    output_data_units = {
        'formatted_responses': DataUnitMemory,  # Formatted response data
        'session_context': SessionContextDataUnit,  # Updated session context
    }
    
    # Define triggers
    triggers = {
        'formatting_complete': TriggerBase,  # Triggered when formatting completes
        'error_formatted': TriggerBase,  # Triggered when error is formatted
    }
    
    def __init__(self):
        """Initialize Response Formatting Step - use from_config for creation"""
        super().__init__()
        # Prevent direct instantiation
        if not hasattr(self, '_from_config_called'):
            raise RuntimeError(
                "Direct instantiation of ResponseFormattingStep is prohibited. "
                "Use: ResponseFormattingStep.from_config(config_file_or_object)"
            )
    
    @classmethod
    def _get_config_class(cls):
        """Return the configuration class for this component"""
        return ResponseFormattingStepConfig
    
    def _init_from_config(self, config: ResponseFormattingStepConfig, component_config: Dict[str, Any], dependencies: Dict[str, Any]) -> None:
        """
        Initialize step from validated configuration.
        
        Args:
            config: Validated ResponseFormattingStepConfig instance
            component_config: Component-specific configuration data
            dependencies: Resolved component dependencies
        """
        # Call parent initialization
        super()._init_from_config(config, component_config, dependencies)
        
        # Store configuration
        self.config = config
        self.name = config.name
        self.description = config.description
        
        # Initialize logging
        self.logger = get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
            debug_mode=config.debug_mode
        )
        
        # Configuration-driven settings
        # Handle enum properly - convert string to enum if needed
        if isinstance(config.default_output_format, str):
            self.default_format = ResponseFormat(config.default_output_format)
        else:
            self.default_format = config.default_output_format
            
        self.enable_markdown = config.enable_markdown_formatting
        self.enable_json = config.enable_json_formatting
        self.max_length = config.max_response_length
        self.enable_truncation = config.enable_content_truncation
        self.truncation_message = config.truncation_message
        self.enable_rich = config.enable_rich_formatting
        self.date_format = config.date_format
        self.include_metadata = config.include_metadata
        self.enable_error_formatting = config.enable_error_formatting
        self.include_troubleshooting = config.include_troubleshooting_tips
        self.enable_streaming = config.enable_streaming_chunks
        self.chunk_size = config.streaming_chunk_size
        
        self.logger.info(
            f"ResponseFormattingStep initialized successfully",
            extra={
                "component_name": self.name,
                "default_format": self.default_format.value,
                "max_length": self.max_length,
                "enable_rich": self.enable_rich
            }
        )
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """
        Process input data and format responses for presentation.
        
        For ResponseFormattingStep, this method handles response formatting,
        template application, and multi-format output generation based on
        progress data and workflow results.
        
        Args:
            input_data: Dictionary containing progress_data, session_context, and workflow_results
            **kwargs: Additional parameters
            
        Returns:
            Formatted response results with presentation data
        """
        try:
            self.logger.debug("Processing ResponseFormattingStep data flow")
            
            # Extract input data
            progress_data = input_data.get('progress_data', {})
            session_context = input_data.get('session_context', {})
            workflow_results = input_data.get('workflow_results', {})
            
            # Determine response type and content
            response_type = await self._determine_response_type(progress_data, workflow_results)
            
            # Format response based on type
            formatted_response = await self._format_response(
                response_type, progress_data, workflow_results, session_context
            )
            
            # Apply output format transformations
            formatted_response = await self._apply_output_format(formatted_response)
            
            # Add metadata if enabled
            if self.include_metadata:
                formatted_response = await self._add_metadata(formatted_response)
            
            # Create formatted response data unit
            response_data = await self._create_response_data(formatted_response, session_context)
            
            # Update session context
            updated_session = await self._update_session_context(session_context, formatted_response)
            
            results = {
                'formatted_responses': response_data,
                'session_context': updated_session,
                'response_type': response_type,
                'output_format': self.default_format.value,
                'processing_time': time.time()
            }
            
            self.logger.debug(
                f"ResponseFormattingStep processing completed",
                extra={
                    "response_type": response_type,
                    "output_format": self.default_format.value,
                    "content_length": len(str(formatted_response.get('content', '')))
                }
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"ResponseFormattingStep processing failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'formatted_responses': await self._create_error_response(str(e)),
                'session_context': input_data.get('session_context', {})
            }
    
    async def _determine_response_type(self, progress_data: Dict[str, Any], 
                                     workflow_results: Dict[str, Any]) -> str:
        """
        Determine the type of response based on input data.
        
        Args:
            progress_data: Progress tracking data
            workflow_results: Workflow execution results
            
        Returns:
            Response type string
        """
        # Check progress status
        if isinstance(progress_data, dict):
            progress_status = progress_data.get('status', '')
            is_complete = progress_data.get('is_complete', False)
            
            if progress_status == 'error':
                return 'error'
            elif is_complete:
                return 'completion'
            elif progress_status == 'in_progress':
                return 'progress'
        
        # Check workflow results
        if isinstance(workflow_results, dict):
            if workflow_results.get('error'):
                return 'error'
            elif workflow_results.get('success', True):
                return 'success'
        
        # Default to generic response
        return 'generic'
    
    async def _format_response(self, response_type: str, progress_data: Dict[str, Any],
                              workflow_results: Dict[str, Any], session_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format response based on type and data.
        
        Args:
            response_type: Type of response to format
            progress_data: Progress tracking data
            workflow_results: Workflow execution results
            session_context: Session context information
            
        Returns:
            Formatted response dictionary
        """
        if response_type == 'error':
            return await self._format_error_response(progress_data, workflow_results)
        elif response_type == 'progress':
            return await self._format_progress_response(progress_data)
        elif response_type == 'completion':
            return await self._format_completion_response(progress_data, workflow_results)
        elif response_type == 'success':
            return await self._format_success_response(workflow_results)
        else:
            return await self._format_generic_response(progress_data, workflow_results)
    
    async def _format_error_response(self, progress_data: Dict[str, Any], 
                                   workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format error response with troubleshooting information."""
        
        # Extract error information
        error_msg = (progress_data.get('error') or 
                    workflow_results.get('error') or 
                    "An unknown error occurred")
        
        if self.enable_rich:
            content = f"❌ **Error**\n\n{error_msg}"
        else:
            content = f"Error: {error_msg}"
        
        if self.include_troubleshooting and self.enable_error_formatting:
            troubleshooting = [
                "Please check your input and try again",
                "Verify that all required parameters are provided",
                "Contact support if the problem persists"
            ]
            
            if self.enable_rich:
                content += "\n\n🔧 **Troubleshooting Tips:**\n"
                for tip in troubleshooting:
                    content += f"- {tip}\n"
            else:
                content += "\n\nTroubleshooting:\n"
                for tip in troubleshooting:
                    content += f"• {tip}\n"
        
        return {
            'content': content,
            'response_type': 'error',
            'requires_markdown': self.enable_markdown,
            'status': 'error',
            'error_details': error_msg
        }
    
    async def _format_progress_response(self, progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format progress update response."""
        
        current_progress = progress_data.get('current_progress', 0)
        current_step = progress_data.get('current_step', 'Processing')
        eta_seconds = progress_data.get('eta_seconds')
        
        if self.enable_rich:
            # Create visual progress bar
            progress_bar = self._create_progress_bar(current_progress)
            content = f"🔄 **Processing in Progress**\n\n"
            content += f"{progress_bar} **{current_progress:.1f}%**\n\n"
            content += f"**Current Step:** {current_step}\n"
            
            if eta_seconds:
                eta_minutes = int(eta_seconds // 60)
                eta_secs = int(eta_seconds % 60)
                if eta_minutes > 0:
                    content += f"**Estimated Time:** {eta_minutes}m {eta_secs}s remaining\n"
                else:
                    content += f"**Estimated Time:** {eta_secs}s remaining\n"
        else:
            content = f"Processing: {current_progress:.1f}% complete\n"
            content += f"Current step: {current_step}\n"
            if eta_seconds:
                content += f"Estimated time remaining: {int(eta_seconds)}s\n"
        
        return {
            'content': content,
            'response_type': 'progress',
            'requires_markdown': self.enable_markdown,
            'status': 'in_progress',
            'progress': current_progress,
            'is_streaming': self.enable_streaming
        }
    
    async def _format_completion_response(self, progress_data: Dict[str, Any], 
                                        workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format completion response with results."""
        
        elapsed_time = progress_data.get('elapsed_time', 0)
        
        if self.enable_rich:
            content = "✅ **Processing Complete!**\n\n"
            content += f"**Total Time:** {elapsed_time:.1f}s\n\n"
        else:
            content = f"Processing completed in {elapsed_time:.1f}s\n\n"
        
        # Add workflow results if available
        if isinstance(workflow_results, dict) and workflow_results:
            if self.enable_json:
                content += "**Results:**\n\n"
                content += "```json\n"
                json_str = json.dumps(workflow_results, indent=2)
                if len(json_str) > 1000:
                    # Create a summary instead of breaking JSON
                    summary = {
                        "result_size": len(json_str),
                        "data_type": type(workflow_results).__name__,
                        "note": "Full results available - truncated for display"
                    }
                    if isinstance(workflow_results, dict):
                        summary["keys"] = list(workflow_results.keys())[:10]
                    content += json.dumps(summary, indent=2)
                else:
                    content += json_str
                content += "\n```"
        
        return {
            'content': content,
            'response_type': 'completion',
            'requires_markdown': self.enable_markdown,
            'status': 'completed',
            'results': workflow_results,
            'elapsed_time': elapsed_time
        }
    
    async def _format_success_response(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format successful workflow response."""
        
        if self.enable_rich:
            content = "✅ **Operation Successful**\n\n"
        else:
            content = "Operation completed successfully\n\n"
        
        # Add results summary
        if isinstance(workflow_results, dict):
            summary = workflow_results.get('summary', '')
            if summary:
                content += f"{summary}\n\n"
            
            # Add detailed results if requested
            if self.enable_json and len(workflow_results) > 1:
                content += "**Detailed Results:**\n\n"
                content += "```json\n"
                content += json.dumps(workflow_results, indent=2)
                content += "\n```"
        
        return {
            'content': content,
            'response_type': 'success',
            'requires_markdown': self.enable_markdown,
            'status': 'success',
            'results': workflow_results
        }
    
    async def _format_generic_response(self, progress_data: Dict[str, Any], 
                                     workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format generic response for unspecified data."""
        
        content = "Response generated successfully"
        
        # Try to extract meaningful content
        if isinstance(workflow_results, dict):
            content_field = workflow_results.get('content') or workflow_results.get('message', '')
            if content_field:
                content = str(content_field)
        
        return {
            'content': content,
            'response_type': 'generic',
            'requires_markdown': self.enable_markdown,
            'status': 'generated'
        }
    
    def _create_progress_bar(self, progress: float, length: int = 20) -> str:
        """Create a visual progress bar."""
        if not self.enable_rich:
            return f"Progress: {progress:.1f}%"
        
        filled_length = int(length * progress / 100)
        empty_length = length - filled_length
        
        filled_char = "█"
        empty_char = "░"
        
        return f"**Progress:** [{filled_char * filled_length}{empty_char * empty_length}]"
    
    async def _apply_output_format(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Apply output format transformations."""
        
        content = response.get('content', '')
        
        # Apply length limits
        if self.enable_truncation and len(content) > self.max_length:
            content = content[:self.max_length - len(self.truncation_message)]
            content += f"\n\n{self.truncation_message}"
            response['content'] = content
            response['was_truncated'] = True
        
        # Add format-specific metadata
        response['output_format'] = self.default_format.value
        response['content_length'] = len(content)
        
        return response
    
    async def _add_metadata(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata to response."""
        
        response['metadata'] = {
            'generated_at': datetime.now(timezone.utc).strftime(self.date_format),
            'formatter': self.__class__.__name__,
            'format': self.default_format.value,
            'content_length': len(str(response.get('content', '')))
        }
        
        return response
    
    async def _create_response_data(self, response: Dict[str, Any], 
                                   session_context: Dict[str, Any]) -> DataUnitBase:
        """Create response data unit."""
        
        data_unit = DataUnitMemory.from_config({
            'class': 'nanobrain.core.data_unit.DataUnitMemory',
            'name': 'formatted_response',
            'description': 'Formatted response data',
            'cache_size': 100,
            'persistent': False
        })
        
        await data_unit.set(response)
        return data_unit
    
    async def _update_session_context(self, session_context: Dict[str, Any], 
                                     response: Dict[str, Any]) -> DataUnitBase:
        """Update session context with response information."""
        
        updated_context = session_context.copy()
        updated_context.update({
            'last_response_generated': time.time(),
            'last_response_type': response.get('response_type', 'unknown'),
            'last_response_status': response.get('status', 'generated'),
            'response_formatting_active': True
        })
        
        session_data_unit = DataUnitMemory.from_config({
            'class': 'nanobrain.core.data_unit.DataUnitMemory',
            'name': 'session_context',
            'description': 'Session context data',
            'cache_size': 100,
            'persistent': True
        })
        
        await session_data_unit.set(updated_context)
        return session_data_unit
    
    async def _create_error_response(self, error_message: str) -> DataUnitBase:
        """Create error response data unit."""
        
        error_response = {
            'content': f"❌ **Formatting Error**\n\n{error_message}",
            'response_type': 'error',
            'requires_markdown': True,
            'status': 'error',
            'error_details': error_message
        }
        
        data_unit = DataUnitMemory.from_config({
            'class': 'nanobrain.core.data_unit.DataUnitMemory',
            'name': 'error_response',
            'description': 'Error response data',
            'cache_size': 100,
            'persistent': False
        })
        
        await data_unit.set(error_response)
        return data_unit
    
    def get_status(self) -> Dict[str, Any]:
        """Get current response formatting status and metrics."""
        
        return {
            "name": self.name,
            "type": self.COMPONENT_TYPE,
            "status": "operational",
            "configuration": {
                "default_format": self.default_format.value,
                "max_length": self.max_length,
                "enable_markdown": self.enable_markdown,
                "enable_rich": self.enable_rich,
                "enable_streaming": self.enable_streaming,
                "include_metadata": self.include_metadata
            }
        }
    
    def cleanup(self) -> None:
        """Clean up response formatting resources."""
        self.logger.info(f"Cleaning up ResponseFormattingStep")
        # No specific cleanup needed for this step 