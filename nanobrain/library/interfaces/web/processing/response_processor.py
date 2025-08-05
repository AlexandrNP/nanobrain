#!/usr/bin/env python3
"""
Universal Response Processor for NanoBrain Framework
Standardizes diverse workflow responses into universal format for frontend consumption.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, AsyncIterator
from datetime import datetime
import uuid
import json
from pydantic import Field

from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.library.interfaces.web.models.universal_models import (
    UniversalResponse, StandardizedResponse, AggregatedResponse, StreamingResponse
)

# Response processor logger
logger = logging.getLogger(__name__)


class ResponseProcessorConfig(ConfigBase):
    """Configuration for universal response processor"""
    
    # Frontend optimization configuration (component expects this field)
    frontend_optimization: Dict[str, Any] = Field(
        default_factory=lambda: {
            'max_response_items': 1000,
            'optimize_for_rendering': True,
            'paginate_large_datasets': True,
            'compress_large_responses': True,
            'include_display_hints': True
        },
        description="Frontend optimization settings"
    )
    
    # Content processing configuration (component expects this field)
    content_processing: Dict[str, Any] = Field(
        default_factory=lambda: {
            'max_content_size': 10485760,  # 10MB default
            'enable_content_validation': True,
            'auto_detect_format': True,
            'preserve_original_structure': True,
            'content_encoding': 'utf-8'
        },
        description="Content processing settings"
    )
    
    # Response standardization
    standardization: bool = Field(
        default=True,
        description="Enable response standardization"
    )
    
    # Format converter configuration
    format_converter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Format converter configuration"
    )
    
    # Streaming handler configuration
    streaming_handler: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Streaming handler configuration"
    )
    
    # Error handling configuration
    error_handling: Dict[str, Any] = Field(
        default_factory=lambda: {
            'detailed_error_messages': True,
            'include_stack_traces': False,
            'sanitize_errors': True,
            'error_format': 'structured'
        },
        description="Error handling settings"
    )
    
    # Response validation configuration
    response_validation: Dict[str, Any] = Field(
        default_factory=lambda: {
            'validate_structure': True,
            'validate_content': False,
            'strict_validation': False
        },
        description="Response validation settings"
    )
    
    # Output formatting configuration
    output_formatting: Dict[str, Any] = Field(
        default_factory=lambda: {
            'default_format': 'json',
            'supported_formats': ['json', 'text', 'html', 'markdown'],
            'format_specific_config': {}
        },
        description="Output formatting configuration"
    )
    
    # Performance settings
    performance_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'enable_response_caching': True,
            'cache_ttl_seconds': 1800,
            'compression_enabled': False,
            'parallel_processing': False
        },
        description="Performance optimization settings"
    )


class UniversalResponseProcessor(FromConfigBase):
    """
    Universal response processing for any workflow type.
    Standardizes diverse workflow responses into universal format.
    """
    
    def __init__(self):
        """Initialize response processor - use from_config for creation"""
        super().__init__()
        # Instance variables moved to _init_from_config since framework uses __new__ and bypasses __init__
        
    @classmethod
    def _get_config_class(cls):
        """Return the configuration class for this component"""
        return ResponseProcessorConfig
    
    def _init_from_config(self, config, component_config, dependencies):
        """Initialize processor from configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # ✅ FRAMEWORK COMPLIANCE: Initialize instance variables here since __init__ is bypassed
        self.config: Optional[ResponseProcessorConfig] = None
        self.response_cache: Dict[str, StandardizedResponse] = {}
        self.format_processors: Dict[str, Any] = {}
        
        logger.info("🔄 Initializing Universal Response Processor")
        self.config = config
        
        # Setup processor configuration
        self.setup_response_processing()
        
        logger.info("✅ Universal Response Processor initialized successfully")
    
    def setup_response_processing(self) -> None:
        """Setup response processing configuration and components"""
        # Initialize format processors
        self.setup_format_processors()
        
        # Validate configuration
        self.validate_processing_configuration()
        
        logger.debug("✅ Response processing setup complete")
    
    def setup_format_processors(self) -> None:
        """Setup format-specific processors"""
        self.format_processors = {
            'json': self.process_json_content,
            'text': self.process_text_content,
            'html': self.process_html_content,
            'markdown': self.process_markdown_content,
            'csv': self.process_csv_content,
            'xml': self.process_xml_content,
            'binary': self.process_binary_content
        }
        
        logger.debug(f"✅ Initialized {len(self.format_processors)} format processors")
    
    def validate_processing_configuration(self) -> None:
        """Validate response processing configuration"""
        # Validate max content size
        max_size = self.config.content_processing.get('max_content_size', 10485760)
        if max_size < 1024:  # 1KB minimum
            logger.warning("⚠️ Max content size very small, adjusting to 1KB minimum")
            self.config.content_processing['max_content_size'] = 1024
        
        # Validate max response items
        max_items = self.config.frontend_optimization.get('max_response_items', 1000)
        if max_items < 10:
            logger.warning("⚠️ Max response items very small, adjusting to 10 minimum")
            self.config.frontend_optimization['max_response_items'] = 10
    
    async def process_workflow_response(self, response: UniversalResponse, 
                                      workflow_type: str) -> StandardizedResponse:
        """
        Process response from any workflow type.
        
        Args:
            response: UniversalResponse from workflow execution
            workflow_type: Type of workflow that generated the response
            
        Returns:
            StandardizedResponse ready for frontend consumption
        """
        try:
            processing_start = datetime.now()
            
            logger.debug(f"🔄 Processing response from {workflow_type}: {response.response_id}")
            
            # Step 1: Validate response
            validation_result = await self.validate_response(response)
            if not validation_result['valid']:
                return await self.create_error_response(
                    response.response_id, 
                    f"Response validation failed: {validation_result['errors']}"
                )
            
            # Step 2: Extract and process content
            processed_content = await self.process_response_content(response.content)
            
            # Step 3: Generate standardized message
            standardized_message = await self.generate_standardized_message(response, workflow_type)
            
            # Step 4: Detect response format
            response_format = await self.detect_response_format(processed_content)
            
            # Step 5: Apply frontend optimizations
            optimized_content = await self.apply_frontend_optimizations(processed_content, response_format)
            
            # Step 6: Generate frontend hints
            frontend_hints = await self.generate_frontend_hints(optimized_content, response_format, workflow_type)
            
            # Step 7: Collect warnings
            warnings = self.collect_processing_warnings(response, validation_result)
            
            # Step 8: Create standardized response
            standardized_response = StandardizedResponse(
                response_id=f"std_{response.response_id}",
                message=standardized_message,
                data=optimized_content,
                response_format=response_format,
                success=response.success,
                error_message=response.error_details.get('error_message') if response.error_details else None,
                warnings=warnings,
                metadata={
                    'original_response_id': response.response_id,
                    'workflow_id': response.workflow_id,
                    'workflow_type': workflow_type,
                    'response_type': response.response_type,
                    'processing_time_ms': (datetime.now() - processing_start).total_seconds() * 1000,
                    'original_metadata': response.metadata,
                    'standardization_version': '1.0'
                },
                frontend_hints=frontend_hints
            )
            
            # Cache response if configured
            if self.should_cache_response(standardized_response):
                await self.cache_response(standardized_response)
            
            logger.debug(f"✅ Response processed successfully: {standardized_response.response_id}")
            return standardized_response
            
        except Exception as e:
            logger.error(f"❌ Response processing failed: {e}")
            return await self.create_error_response(
                response.response_id if response else "unknown",
                f"Response processing error: {str(e)}"
            )
    
    async def validate_response(self, response: UniversalResponse) -> Dict[str, Any]:
        """Validate response structure and content"""
        try:
            validation_result = {'valid': True, 'errors': [], 'warnings': []}
            
            # Check required fields
            if not response.response_id:
                validation_result['errors'].append('Missing response_id')
            
            if not response.workflow_id:
                validation_result['errors'].append('Missing workflow_id')
            
            if response.content is None:
                validation_result['errors'].append('Missing content')
            
            # Check content size if configured
            if self.config.content_processing.get('enable_content_validation', True):
                content_size = self.estimate_content_size(response.content)
                max_size = self.config.content_processing.get('max_content_size', 10485760)
                
                if content_size > max_size:
                    validation_result['warnings'].append(f'Content size ({content_size}) exceeds recommended maximum ({max_size})')
            
            # Mark as invalid if errors found
            if validation_result['errors']:
                validation_result['valid'] = False
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Response validation failed: {e}")
            return {'valid': False, 'errors': [f'Validation error: {str(e)}'], 'warnings': []}
    
    def estimate_content_size(self, content: Any) -> int:
        """Estimate content size in bytes"""
        try:
            if isinstance(content, str):
                return len(content.encode('utf-8'))
            elif isinstance(content, (dict, list)):
                return len(json.dumps(content, default=str).encode('utf-8'))
            elif isinstance(content, bytes):
                return len(content)
            else:
                return len(str(content).encode('utf-8'))
        except Exception:
            return 0
    
    async def process_response_content(self, content: Any) -> Dict[str, Any]:
        """Process and standardize response content"""
        try:
            if content is None:
                return {'processed_content': None, 'content_type': 'null'}
            
            # Auto-detect content format if configured
            if self.config.content_processing.get('auto_detect_format', True):
                detected_format = await self.detect_content_format(content)
            else:
                detected_format = 'unknown'
            
            # Apply format-specific processing
            if detected_format in self.format_processors:
                processor = self.format_processors[detected_format]
                processed_content = await processor(content)
            else:
                # Default processing
                processed_content = await self.process_generic_content(content)
            
            return {
                'processed_content': processed_content,
                'content_type': detected_format,
                'original_content': content if self.config.content_processing.get('preserve_original_structure', True) else None
            }
            
        except Exception as e:
            logger.error(f"❌ Content processing failed: {e}")
            return {
                'processed_content': {'error': 'Content processing failed'},
                'content_type': 'error',
                'processing_error': str(e)
            }
    
    async def detect_content_format(self, content: Any) -> str:
        """Detect content format for appropriate processing"""
        try:
            if isinstance(content, dict):
                return 'json'
            elif isinstance(content, list):
                return 'json'
            elif isinstance(content, str):
                # Check for specific string formats
                content_lower = content.lower().strip()
                if content_lower.startswith('<!doctype html') or content_lower.startswith('<html'):
                    return 'html'
                elif content_lower.startswith('#') or '##' in content:
                    return 'markdown'
                elif content.count(',') > content.count(' ') and '\n' in content:
                    return 'csv'
                elif content_lower.startswith('<?xml') or content_lower.startswith('<'):
                    return 'xml'
                else:
                    return 'text'
            elif isinstance(content, bytes):
                return 'binary'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.debug(f"⚠️ Content format detection failed: {e}")
            return 'unknown'
    
    async def process_json_content(self, content: Any) -> Dict[str, Any]:
        """Process JSON-like content"""
        try:
            if isinstance(content, (dict, list)):
                return {'data': content, 'format': 'structured'}
            else:
                # Try to parse as JSON string
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        return {'data': parsed, 'format': 'structured'}
                    except json.JSONDecodeError:
                        return {'data': content, 'format': 'text'}
                else:
                    return {'data': content, 'format': 'mixed'}
                    
        except Exception as e:
            logger.error(f"❌ JSON content processing failed: {e}")
            return {'data': str(content), 'format': 'fallback'}
    
    async def process_text_content(self, content: str) -> Dict[str, Any]:
        """Process plain text content"""
        try:
            # Basic text processing
            lines = content.split('\n') if isinstance(content, str) else [str(content)]
            
            return {
                'text': content,
                'line_count': len(lines),
                'character_count': len(content) if isinstance(content, str) else 0,
                'format': 'plain_text'
            }
            
        except Exception as e:
            logger.error(f"❌ Text content processing failed: {e}")
            return {'text': str(content), 'format': 'fallback'}
    
    async def process_html_content(self, content: str) -> Dict[str, Any]:
        """Process HTML content"""
        try:
            # Basic HTML processing (without full parsing for security)
            return {
                'html': content,
                'format': 'html',
                'sanitized': True,  # Would implement sanitization in production
                'preview': content[:200] + '...' if len(content) > 200 else content
            }
            
        except Exception as e:
            logger.error(f"❌ HTML content processing failed: {e}")
            return {'html': str(content), 'format': 'fallback'}
    
    async def process_markdown_content(self, content: str) -> Dict[str, Any]:
        """Process Markdown content"""
        try:
            lines = content.split('\n') if isinstance(content, str) else [str(content)]
            headers = [line for line in lines if line.strip().startswith('#')]
            
            return {
                'markdown': content,
                'format': 'markdown',
                'headers': headers,
                'line_count': len(lines)
            }
            
        except Exception as e:
            logger.error(f"❌ Markdown content processing failed: {e}")
            return {'markdown': str(content), 'format': 'fallback'}
    
    async def process_csv_content(self, content: str) -> Dict[str, Any]:
        """Process CSV content"""
        try:
            lines = content.split('\n') if isinstance(content, str) else [str(content)]
            headers = lines[0].split(',') if lines else []
            row_count = len(lines) - 1 if len(lines) > 1 else 0
            
            return {
                'csv': content,
                'format': 'csv',
                'headers': headers,
                'row_count': row_count,
                'preview_rows': lines[:5]  # First 5 rows for preview
            }
            
        except Exception as e:
            logger.error(f"❌ CSV content processing failed: {e}")
            return {'csv': str(content), 'format': 'fallback'}
    
    async def process_xml_content(self, content: str) -> Dict[str, Any]:
        """Process XML content"""
        try:
            return {
                'xml': content,
                'format': 'xml',
                'preview': content[:200] + '...' if len(content) > 200 else content
            }
            
        except Exception as e:
            logger.error(f"❌ XML content processing failed: {e}")
            return {'xml': str(content), 'format': 'fallback'}
    
    async def process_binary_content(self, content: bytes) -> Dict[str, Any]:
        """Process binary content"""
        try:
            return {
                'binary': True,
                'format': 'binary',
                'size': len(content),
                'preview': f"Binary data ({len(content)} bytes)"
            }
            
        except Exception as e:
            logger.error(f"❌ Binary content processing failed: {e}")
            return {'binary': True, 'format': 'fallback', 'size': 0}
    
    async def process_generic_content(self, content: Any) -> Dict[str, Any]:
        """Process generic/unknown content"""
        try:
            return {
                'content': content,
                'format': 'generic',
                'type': type(content).__name__,
                'string_representation': str(content)
            }
            
        except Exception as e:
            logger.error(f"❌ Generic content processing failed: {e}")
            return {'content': None, 'format': 'error', 'error': str(e)}
    
    async def generate_standardized_message(self, response: UniversalResponse, 
                                          workflow_type: str) -> str:
        """Generate standardized human-readable message"""
        try:
            if not response.success:
                if self.config.error_handling.get('detailed_error_messages', True):
                    error_details = response.error_details or {}
                    error_message = error_details.get('error_message', 'Unknown error occurred')
                    return f"I encountered an issue while processing your request: {error_message}"
                else:
                    return self.config.error_handling.get('fallback_message', 
                                                        "Processing completed with some issues.")
            
            # Generate success message based on workflow type and content
            if workflow_type in ['chatbot_viral_integration', 'conversational']:
                return "I've processed your request and here are the results:"
            elif workflow_type in ['viral_protein_analysis', 'analysis']:
                return "Analysis completed successfully. Here are the findings:"
            elif workflow_type in ['information_request']:
                return "Here's the information you requested:"
            else:
                return "Processing completed successfully."
                
        except Exception as e:
            logger.error(f"❌ Message generation failed: {e}")
            return "Processing completed."
    
    async def detect_response_format(self, processed_content: Dict[str, Any]) -> str:
        """Detect appropriate response format for frontend"""
        try:
            content_type = processed_content.get('content_type', 'unknown')
            
            if content_type == 'json':
                return 'structured_data'
            elif content_type in ['text', 'markdown']:
                return 'text'
            elif content_type == 'html':
                return 'html'
            elif content_type == 'csv':
                return 'table'
            elif content_type == 'binary':
                return 'file'
            else:
                return 'mixed'
                
        except Exception as e:
            logger.error(f"❌ Response format detection failed: {e}")
            return 'text'
    
    async def apply_frontend_optimizations(self, content: Dict[str, Any], 
                                         response_format: str) -> Dict[str, Any]:
        """Apply optimizations for frontend rendering"""
        try:
            if not self.config.frontend_optimization.get('optimize_for_rendering', True):
                return content
            
            optimized_content = content.copy()
            
            # Apply pagination for large datasets
            if self.config.frontend_optimization.get('paginate_large_datasets', True):
                optimized_content = await self.apply_pagination(optimized_content, response_format)
            
            # Compress large responses if configured
            if self.config.frontend_optimization.get('compress_large_responses', True):
                optimized_content = await self.apply_compression(optimized_content)
            
            return optimized_content
            
        except Exception as e:
            logger.error(f"❌ Frontend optimization failed: {e}")
            return content
    
    async def apply_pagination(self, content: Dict[str, Any], response_format: str) -> Dict[str, Any]:
        """Apply pagination to large datasets"""
        try:
            max_items = self.config.frontend_optimization.get('max_response_items', 1000)
            
            # Check if content needs pagination
            if response_format == 'structured_data' and 'data' in content:
                data = content['data']
                if isinstance(data, list) and len(data) > max_items:
                    # Apply pagination
                    content['data'] = data[:max_items]
                    content['pagination'] = {
                        'total_items': len(data),
                        'displayed_items': max_items,
                        'has_more': True,
                        'page_size': max_items
                    }
                    
                    logger.debug(f"✅ Applied pagination: {max_items}/{len(data)} items")
            
            return content
            
        except Exception as e:
            logger.error(f"❌ Pagination failed: {e}")
            return content
    
    async def apply_compression(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Apply compression for large content"""
        try:
            # Simple compression by removing redundant data for large responses
            # In production, this might use actual compression algorithms
            
            content_size = self.estimate_content_size(content)
            large_threshold = 1048576  # 1MB
            
            if content_size > large_threshold:
                # Apply content summarization/compression
                if 'processed_content' in content and isinstance(content['processed_content'], dict):
                    # Remove original content if it's very large
                    if 'original_content' in content['processed_content']:
                        del content['processed_content']['original_content']
                        content['compression_applied'] = True
                        
                        logger.debug(f"✅ Applied compression to large response ({content_size} bytes)")
            
            return content
            
        except Exception as e:
            logger.error(f"❌ Compression failed: {e}")
            return content
    
    async def generate_frontend_hints(self, content: Dict[str, Any], response_format: str, 
                                    workflow_type: str) -> Dict[str, Any]:
        """Generate hints for frontend rendering"""
        try:
            hints = {}
            
            if not self.config.frontend_optimization.get('include_display_hints', True):
                return hints
            
            # Response format hints
            hints['response_format'] = response_format
            hints['workflow_type'] = workflow_type
            
            # Content-specific hints
            if response_format == 'structured_data':
                hints['render_suggestion'] = 'json_tree'
                hints['expandable'] = True
            elif response_format == 'table':
                hints['render_suggestion'] = 'data_table'
                hints['sortable'] = True
            elif response_format == 'text':
                hints['render_suggestion'] = 'formatted_text'
                hints['preserveFormatting'] = True
            
            # Size hints
            content_size = self.estimate_content_size(content)
            if content_size > 10240:  # 10KB
                hints['large_content'] = True
                hints['lazy_loading'] = True
            
            # Interaction hints
            if 'pagination' in content:
                hints['paginated'] = True
                hints['enable_pagination_controls'] = True
            
            return hints
            
        except Exception as e:
            logger.error(f"❌ Frontend hints generation failed: {e}")
            return {}
    
    def collect_processing_warnings(self, response: UniversalResponse, 
                                  validation_result: Dict[str, Any]) -> List[str]:
        """Collect processing warnings"""
        warnings = []
        
        # Add validation warnings
        warnings.extend(validation_result.get('warnings', []))
        
        # Add response-specific warnings
        if response.error_details and not response.success:
            warnings.append("Response contains error details")
        
        # Add processing warnings
        if hasattr(self, '_processing_warnings'):
            warnings.extend(self._processing_warnings)
            self._processing_warnings = []  # Clear after collection
        
        return warnings
    
    async def create_error_response(self, response_id: str, error_message: str) -> StandardizedResponse:
        """Create standardized error response"""
        return StandardizedResponse(
            response_id=f"error_{response_id}",
            message="I apologize, but I encountered an error processing your request.",
            data={'error': True, 'details': error_message},
            response_format='error',
            success=False,
            error_message=error_message,
            warnings=[],
            metadata={
                'error_response': True,
                'original_response_id': response_id,
                'processing_timestamp': datetime.now().isoformat()
            },
            frontend_hints={
                'render_suggestion': 'error_display',
                'show_retry_option': True
            }
        )
    
    def should_cache_response(self, response: StandardizedResponse) -> bool:
        """Determine if response should be cached"""
        # Cache successful responses that are not too large
        if not response.success:
            return False
        
        content_size = self.estimate_content_size(response.data)
        max_cache_size = 102400  # 100KB
        
        return content_size <= max_cache_size
    
    async def cache_response(self, response: StandardizedResponse) -> None:
        """Cache standardized response"""
        try:
            cache_key = response.response_id
            self.response_cache[cache_key] = response
            
            # Simple cache management
            if len(self.response_cache) > 100:
                # Remove oldest entries
                oldest_keys = list(self.response_cache.keys())[:20]
                for key in oldest_keys:
                    del self.response_cache[key]
            
            logger.debug(f"✅ Cached response: {cache_key}")
            
        except Exception as e:
            logger.error(f"❌ Response caching failed: {e}")
    
    async def standardize_response(self, response: UniversalResponse) -> StandardizedResponse:
        """
        Standardize response format for frontend consumption.
        
        Args:
            response: UniversalResponse to standardize
            
        Returns:
            StandardizedResponse ready for frontend
        """
        # Determine workflow type from response metadata
        workflow_type = response.metadata.get('workflow_type', 'unknown')
        
        return await self.process_workflow_response(response, workflow_type)
    
    async def get_health_status(self) -> str:
        """Get processor health status"""
        try:
            # Basic health check
            if self.config and len(self.format_processors) > 0:
                return "healthy"
            else:
                return "unhealthy"
        except Exception as e:
            logger.error(f"❌ Processor health check failed: {e}")
            return "unhealthy" 