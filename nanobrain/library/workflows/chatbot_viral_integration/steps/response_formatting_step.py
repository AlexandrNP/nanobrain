"""
Response Formatting Step

Formats responses for presentation with markdown support, progress bars, and streaming.
Provides consistent formatting across all response types.

Author: NanoBrain Development Team
Date: December 2024
Version: 4.1.0
"""

from nanobrain.core.step import Step, StepConfig
from nanobrain.library.infrastructure.data.chat_session_data import (
    AnnotationJobData, ConversationalResponseData, MessageType
)
from typing import Dict, Any, List, Optional, AsyncGenerator
import time
from datetime import datetime


class ResponseFormattingStep(Step):
    """
    Step for formatting responses for user presentation.
    
    Handles markdown formatting, progress bars, streaming chunks,
    and consistent presentation across all response types.
    """
    
    def __init__(self, config: StepConfig):
        super().__init__(config)
        
        # Get nested config dict
        step_config = getattr(config, 'config', {})
        
        # Formatting configuration
        self.enable_markdown = step_config.get('enable_markdown', True)
        self.enable_progress_bars = step_config.get('enable_progress_bars', True)
        self.progress_bar_length = step_config.get('progress_bar_length', 20)
        self.max_response_length = step_config.get('max_response_length', 10000)
        self.streaming_chunk_size = step_config.get('streaming_chunk_size', 50)
        self.max_streaming_chunks = step_config.get('max_streaming_chunks', 100)
        
        # Error formatting
        self.include_troubleshooting = step_config.get('include_troubleshooting_tips', True)
        self.include_fallback_suggestions = step_config.get('include_fallback_suggestions', True)
        
        self.nb_logger.info("🎨 Response Formatting Step initialized")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format response for user presentation.
        
        Args:
            input_data: Contains response_type and relevant data
            
        Returns:
            Formatted response with presentation metadata
        """
        start_time = time.time()
        
        try:
            response_type = input_data.get('response_type')
            
            self.nb_logger.info(f"🎨 Formatting {response_type} response")
            
            # Route to specific formatter based on type
            if response_type == 'annotation_job':
                formatted_response = await self._format_annotation_job_response(input_data)
            elif response_type == 'annotation_progress':
                formatted_response = await self._format_progress_response(input_data)
            elif response_type == 'conversational':
                formatted_response = await self._format_conversational_response(input_data)
            elif response_type == 'error':
                formatted_response = await self._format_error_response(input_data)
            else:
                formatted_response = await self._format_generic_response(input_data)
            
            # Add formatting metadata
            formatted_response['processing_time_ms'] = (time.time() - start_time) * 1000
            formatted_response['formatted_at'] = datetime.now().isoformat()
            
            self.nb_logger.info(f"✅ Response formatted successfully")
            
            return {
                'success': True,
                'formatted_response': formatted_response
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Response formatting failed: {e}")
            
            # Return basic error formatting
            return {
                'success': False,
                'error': str(e),
                'formatted_response': {
                    'content': f"❌ **Formatting Error**: {str(e)}",
                    'requires_markdown': True,
                    'message_type': 'error'
                }
            }
    
    async def _format_annotation_job_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format annotation job submission response"""
        
        job_data: AnnotationJobData = input_data.get('job_data')
        success = input_data.get('success', False)
        error = input_data.get('error')
        backend_available = input_data.get('backend_available', True)
        
        if not success:
            # Format error response
            return await self._format_job_error(error, backend_available, job_data)
        
        # Format successful job submission
        content = f"""🔬 **Viral Annotation Analysis Started**

**Job ID:** `{job_data.job_id}`
**Status:** {job_data.status.title()}
**Analysis Type:** Viral Protein Annotation

**📋 Analysis Details:**
"""
        
        # Add extracted parameters
        params = job_data.extracted_parameters
        if 'sequences' in params:
            sequence_count = len(params['sequences'])
            content += f"- **Sequences:** {sequence_count} protein sequence{'s' if sequence_count != 1 else ''}\n"
        
        if 'genome_ids' in params:
            content += f"- **Genome IDs:** {', '.join(params['genome_ids'])}\n"
        
        if 'target_organisms' in params:
            content += f"- **Target Organisms:** {', '.join(params['target_organisms'])}\n"
        
        content += f"""
**⏱️ Estimated Duration:** 5-15 minutes
**🔄 Progress:** I'll provide real-time updates as the analysis progresses.

Your analysis is now in the queue. You can continue chatting while it runs!"""
        
        return {
            'content': content,
            'message_type': 'annotation_start',
            'requires_markdown': True,
            'job_id': job_data.job_id,
            'status': job_data.status,
            'is_streaming': False,
            'metadata': {
                'backend_job_id': job_data.backend_job_id,
                'analysis_scope': params.get('analysis_scope', 'unknown')
            }
        }
    
    async def _format_job_error(self, error: str, backend_available: bool, job_data: Optional[AnnotationJobData]) -> Dict[str, Any]:
        """Format job submission error response"""
        
        if not backend_available:
            content = f"""❌ **Service Unavailable**

The viral annotation service is currently unavailable. This could be due to:

- **Maintenance:** The service may be undergoing scheduled maintenance
- **Connectivity:** Network connectivity issues
- **Load:** High service demand

**🔄 What you can do:**
- Try again in a few minutes
- Ask me questions about alphaviruses while you wait
- Check back later for service restoration

**Error details:** {error}"""
        
        elif "Maximum concurrent jobs" in error:
            content = f"""⏳ **Job Queue Full**

You've reached the maximum number of concurrent annotation jobs ({error.split('(')[1].split(')')[0]}).

**🔄 Next steps:**
- Wait for existing jobs to complete
- Check job status with your session
- Try again once a job finishes

**💡 Tip:** You can ask me questions about alphaviruses while you wait!"""
        
        elif "Rate limit" in error or "Too many requests" in error:
            content = f"""⚠️ **Rate Limit Reached**

You're submitting requests too quickly. Please wait a moment before trying again.

**🔄 What to do:**
- Wait 30-60 seconds before resubmitting
- Use this time to refine your analysis parameters
- Ask me about optimizing your annotation requests

**Error details:** {error}"""
        
        else:
            content = f"""❌ **Annotation Job Failed**

There was an error starting your annotation job:

**Error:** {error}

**🔄 Troubleshooting:**
- Check your sequence format
- Verify genome IDs are valid
- Try with fewer sequences
- Contact support if the issue persists

**💡 Alternative:** I can help answer questions about viral proteins and annotation while you troubleshoot."""
        
        return {
            'content': content,
            'message_type': 'error',
            'requires_markdown': True,
            'is_streaming': False,
            'job_id': job_data.job_id if job_data else None,
            'backend_available': backend_available
        }
    
    async def _format_progress_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format job progress update response"""
        
        progress_data = input_data.get('progress_data', {})
        job_id = progress_data.get('job_id')
        progress = progress_data.get('progress', 0)
        status = progress_data.get('status', 'running')
        message = progress_data.get('message', '')
        elapsed_time = progress_data.get('elapsed_time', 0)
        
        # Create progress bar
        progress_bar = self._create_progress_bar(progress)
        
        # Format elapsed time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
        
        if status == 'completed':
            content = f"""✅ **Analysis Complete!**

**Job ID:** `{job_id}`
{progress_bar} **{progress}%**

**⏱️ Total Time:** {time_str}
**📊 Status:** Analysis finished successfully

Your viral protein annotation results are ready! The analysis has identified functional domains, structural features, and potential annotations."""
        
        elif status == 'failed':
            error_msg = progress_data.get('error', 'Unknown error occurred')
            content = f"""❌ **Analysis Failed**

**Job ID:** `{job_id}`
{progress_bar} **{progress}%**

**⏱️ Time Elapsed:** {time_str}
**📊 Status:** Analysis failed

**Error:** {error_msg}

**🔄 Next Steps:**
- Review your input parameters
- Try with different sequences
- Contact support if needed"""
        
        else:
            # Running or pending
            content = f"""🔄 **Analysis in Progress**

**Job ID:** `{job_id}`
{progress_bar} **{progress}%**

**⏱️ Elapsed:** {time_str}
**📊 Status:** {status.title()}"""
            
            if message:
                content += f"\n**Current Step:** {message}"
            
            content += "\n\n*I'll update you as the analysis progresses...*"
        
        return {
            'content': content,
            'message_type': 'progress',
            'requires_markdown': True,
            'is_streaming': False,
            'job_id': job_id,
            'progress': progress,
            'status': status
        }
    
    async def _format_conversational_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format conversational response with literature references"""
        
        response_data: ConversationalResponseData = input_data.get('response_data')
        enable_streaming = input_data.get('enable_streaming', False)
        
        if not response_data:
            raise ValueError("No response_data provided for conversational formatting")
        
        # Format response with references
        content = response_data.format_with_references()
        
        # Truncate if too long
        if len(content) > self.max_response_length:
            content = content[:self.max_response_length] + "\n\n*[Response truncated for length]*"
        
        formatted_response = {
            'content': content,
            'message_type': 'conversational',
            'requires_markdown': True,
            'is_streaming': enable_streaming,
            'topic_area': response_data.topic_area,
            'confidence': response_data.confidence,
            'response_type': response_data.response_type
        }
        
        # Add streaming chunks if enabled
        if enable_streaming:
            formatted_response['streaming_chunks'] = await self._create_streaming_chunks(content)
        
        return formatted_response
    
    async def _format_error_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format error response with helpful guidance"""
        
        error_message = input_data.get('error', 'An unknown error occurred')
        error_type = input_data.get('error_type', 'general')
        suggestions = input_data.get('suggestions', [])
        
        content = f"""❌ **Error**

{error_message}

**🔄 What you can try:**"""
        
        if error_type == 'classification_error':
            content += """
- Rephrase your question more clearly
- Be more specific about what you want to know
- Use keywords like "analyze", "annotate", or "explain"
- Try asking about specific alphavirus topics"""
        
        elif error_type == 'backend_error':
            content += """
- Wait a moment and try again
- Check your internet connection
- Verify your input parameters
- Try with simpler requests"""
        
        else:
            content += """
- Try rephrasing your question
- Be more specific about what you need
- Ask for help with alphavirus topics
- Check your input format"""
        
        if suggestions:
            content += "\n\n**💡 Suggestions:**\n"
            for suggestion in suggestions:
                content += f"- {suggestion}\n"
        
        content += "\n\n**Need help?** Ask me about alphavirus structure, replication, diseases, or protein annotation!"
        
        return {
            'content': content,
            'message_type': 'error',
            'requires_markdown': True,
            'is_streaming': False,
            'error_type': error_type
        }
    
    async def _format_generic_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format generic response"""
        
        content = input_data.get('content', 'Response generated')
        message_type = input_data.get('message_type', 'general')
        
        return {
            'content': content,
            'message_type': message_type,
            'requires_markdown': self.enable_markdown,
            'is_streaming': False
        }
    
    def _create_progress_bar(self, progress: int) -> str:
        """Create visual progress bar"""
        
        if not self.enable_progress_bars:
            return f"Progress: {progress}%"
        
        # Ensure progress is between 0 and 100
        progress = max(0, min(100, progress))
        
        # Calculate filled and empty segments
        filled_length = int(self.progress_bar_length * progress / 100)
        empty_length = self.progress_bar_length - filled_length
        
        # Create progress bar with Unicode characters
        filled_char = "█"
        empty_char = "░"
        
        progress_bar = f"[{filled_char * filled_length}{empty_char * empty_length}]"
        
        return f"**Progress:** {progress_bar}"
    
    async def _create_streaming_chunks(self, content: str) -> List[str]:
        """Create chunks for streaming response simulation"""
        
        words = content.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > self.streaming_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks 