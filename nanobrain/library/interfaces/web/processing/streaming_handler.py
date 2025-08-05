#!/usr/bin/env python3
"""
Streaming Handler for NanoBrain Framework
Real-time streaming response handling for progressive workflow execution and data delivery.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, AsyncIterator, Callable
from datetime import datetime
import uuid
import json
from enum import Enum

from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.library.interfaces.web.models.universal_models import StreamingResponse

# Streaming handler logger
logger = logging.getLogger(__name__)


class StreamingType(str, Enum):
    """Types of streaming supported"""
    DATA_CHUNKS = "data_chunks"
    PROGRESS_UPDATES = "progress_updates"
    STATUS_MESSAGES = "status_messages"
    REAL_TIME_RESULTS = "real_time_results"
    ERROR_NOTIFICATIONS = "error_notifications"


class StreamingHandlerConfig(ConfigBase):
    """Configuration for streaming handler"""
    
    def __init__(self, config_data: Dict[str, Any]):
        super().__init__(config_data)
        
        # Streaming configuration
        self.enable_streaming: bool = config_data.get('enable_streaming', True)
        self.max_concurrent_streams: int = config_data.get('max_concurrent_streams', 100)
        self.stream_timeout: float = config_data.get('stream_timeout', 300.0)  # 5 minutes
        
        # Streaming types configuration
        self.supported_stream_types: List[str] = config_data.get('supported_stream_types', [
            'data_chunks', 'progress_updates', 'status_messages', 'real_time_results'
        ])
        
        # Buffer configuration
        self.buffer_config: Dict[str, Any] = config_data.get('buffer_config', {
            'max_buffer_size': 10485760,  # 10MB
            'chunk_size': 8192,  # 8KB
            'buffer_timeout': 1.0,  # 1 second
            'auto_flush': True
        })
        
        # Progress tracking configuration
        self.progress_config: Dict[str, Any] = config_data.get('progress_config', {
            'enable_progress_tracking': True,
            'progress_update_interval': 1.0,  # 1 second
            'detailed_progress': True,
            'include_eta': True
        })
        
        # Error handling configuration
        self.error_handling: Dict[str, Any] = config_data.get('error_handling', {
            'stream_on_error': True,
            'include_error_details': True,
            'graceful_termination': True,
            'retry_failed_chunks': False
        })
        
        # Client connection configuration
        self.connection_config: Dict[str, Any] = config_data.get('connection_config', {
            'heartbeat_interval': 30.0,  # 30 seconds
            'connection_keepalive': True,
            'compression': False,
            'encoding': 'utf-8'
        })


class StreamSession:
    """Represents an active streaming session"""
    
    def __init__(self, stream_id: str, stream_type: StreamingType, config: StreamingHandlerConfig):
        self.stream_id = stream_id
        self.stream_type = stream_type
        self.config = config
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.is_active = True
        self.chunk_count = 0
        self.total_bytes_sent = 0
        self.progress = 0.0
        self.status = "active"
        self.error_count = 0
        self.buffer: List[StreamingResponse] = []
        self.subscribers: List[Callable] = []
        
    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        timeout = self.config.stream_timeout
        elapsed = (datetime.now() - self.last_activity).total_seconds()
        return elapsed > timeout
    
    def add_to_buffer(self, response: StreamingResponse) -> None:
        """Add response to buffer"""
        self.buffer.append(response)
        max_buffer_size = self.config.buffer_config.get('max_buffer_size', 10485760)
        
        # Simple buffer management - remove oldest if too large
        while len(self.buffer) > 1000:  # Max 1000 items in buffer
            self.buffer.pop(0)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information"""
        return {
            'stream_id': self.stream_id,
            'stream_type': self.stream_type.value,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'is_active': self.is_active,
            'chunk_count': self.chunk_count,
            'total_bytes_sent': self.total_bytes_sent,
            'progress': self.progress,
            'status': self.status,
            'error_count': self.error_count,
            'buffer_size': len(self.buffer)
        }


class StreamingHandler(FromConfigBase):
    """
    Universal streaming handler for real-time workflow response delivery.
    Supports various streaming patterns and progressive data delivery.
    """
    
    def __init__(self):
        """Initialize streaming handler - use from_config for creation"""
        super().__init__()
        self.config: Optional[StreamingHandlerConfig] = None
        self.active_streams: Dict[str, StreamSession] = {}
        self.stream_locks: Dict[str, asyncio.Lock] = {}
        
    @classmethod
    def _get_config_class(cls):
        """Return the configuration class for this component"""
        return StreamingHandlerConfig
    
    def _init_from_config(self, config, component_config, dependencies):
        """Initialize handler from configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        logger.info("🌊 Initializing Streaming Handler")
        self.config = config
        
        # Setup streaming components
        self.setup_streaming_configuration()
        
        # Start background tasks
        if self.config.enable_streaming:
            asyncio.create_task(self.cleanup_expired_streams())
        
        logger.info("✅ Streaming Handler initialized successfully")
    
    def setup_streaming_configuration(self) -> None:
        """Setup streaming configuration and validation"""
        # Validate configuration
        if self.config.max_concurrent_streams < 1:
            logger.warning("⚠️ Max concurrent streams must be at least 1, adjusting")
            self.config.max_concurrent_streams = 1
        
        if self.config.stream_timeout < 10.0:
            logger.warning("⚠️ Stream timeout very short, adjusting to 10 seconds minimum")
            self.config.stream_timeout = 10.0
        
        logger.debug("✅ Streaming configuration setup complete")
    
    async def create_stream(self, stream_type: StreamingType, 
                          initial_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new streaming session.
        
        Args:
            stream_type: Type of streaming to create
            initial_data: Optional initial data for the stream
            
        Returns:
            Stream ID for the created session
        """
        try:
            if not self.config.enable_streaming:
                raise RuntimeError("Streaming is not enabled")
            
            # Check concurrent stream limit
            if len(self.active_streams) >= self.config.max_concurrent_streams:
                await self.cleanup_expired_streams()
                if len(self.active_streams) >= self.config.max_concurrent_streams:
                    raise RuntimeError("Maximum concurrent streams limit reached")
            
            # Validate stream type
            if stream_type.value not in self.config.supported_stream_types:
                raise ValueError(f"Stream type {stream_type.value} not supported")
            
            # Create new stream session
            stream_id = f"stream_{uuid.uuid4().hex[:12]}"
            session = StreamSession(stream_id, stream_type, self.config)
            
            # Add to active streams
            self.active_streams[stream_id] = session
            self.stream_locks[stream_id] = asyncio.Lock()
            
            # Send initial chunk if data provided
            if initial_data:
                await self.send_chunk(stream_id, initial_data, chunk_type="initialization")
            
            logger.debug(f"✅ Created stream: {stream_id} (type: {stream_type.value})")
            return stream_id
            
        except Exception as e:
            logger.error(f"❌ Stream creation failed: {e}")
            raise
    
    async def send_chunk(self, stream_id: str, content: Dict[str, Any], 
                        chunk_type: str = "data", is_final: bool = False,
                        progress: Optional[float] = None) -> None:
        """
        Send a chunk of data to the stream.
        
        Args:
            stream_id: Stream session ID
            content: Content to send
            chunk_type: Type of chunk (data, status, error, etc.)
            is_final: Whether this is the final chunk
            progress: Optional progress percentage (0.0 to 1.0)
        """
        try:
            session = self.get_stream_session(stream_id)
            if not session or not session.is_active:
                raise ValueError(f"Stream {stream_id} not found or inactive")
            
            async with self.stream_locks[stream_id]:
                # Create streaming response
                chunk_id = f"{stream_id}_chunk_{session.chunk_count + 1}"
                
                streaming_response = StreamingResponse(
                    stream_id=stream_id,
                    chunk_id=chunk_id,
                    content=content,
                    is_final=is_final,
                    progress=progress if progress is not None else session.progress,
                    chunk_type=chunk_type,
                    metadata={
                        'stream_type': session.stream_type.value,
                        'chunk_sequence': session.chunk_count + 1,
                        'session_created': session.created_at.isoformat()
                    },
                    timestamp=datetime.now()
                )
                
                # Update session stats
                session.chunk_count += 1
                session.update_activity()
                if progress is not None:
                    session.progress = progress
                if is_final:
                    session.is_active = False
                    session.status = "completed"
                
                # Add to buffer
                session.add_to_buffer(streaming_response)
                
                # Estimate bytes sent
                try:
                    content_size = len(json.dumps(content, default=str).encode('utf-8'))
                    session.total_bytes_sent += content_size
                except Exception:
                    pass  # Size estimation failed, continue without it
                
                # Notify subscribers (in real implementation, this would send to clients)
                await self.notify_subscribers(session, streaming_response)
                
                logger.debug(f"✅ Sent chunk to stream {stream_id}: {chunk_id}")
                
                # Auto-flush buffer if configured
                if self.config.buffer_config.get('auto_flush', True):
                    await self.flush_stream_buffer(stream_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to send chunk to stream {stream_id}: {e}")
            await self.handle_stream_error(stream_id, str(e))
            raise
    
    async def send_progress_update(self, stream_id: str, progress: float, 
                                 status_message: Optional[str] = None,
                                 eta_seconds: Optional[float] = None) -> None:
        """
        Send a progress update to the stream.
        
        Args:
            stream_id: Stream session ID
            progress: Progress percentage (0.0 to 1.0)
            status_message: Optional status message
            eta_seconds: Optional estimated time to completion
        """
        try:
            if not self.config.progress_config.get('enable_progress_tracking', True):
                return
            
            progress_content = {
                'progress': progress,
                'progress_percentage': f"{progress * 100:.1f}%"
            }
            
            if status_message:
                progress_content['status'] = status_message
            
            if eta_seconds and self.config.progress_config.get('include_eta', True):
                progress_content['eta_seconds'] = eta_seconds
                if eta_seconds > 0:
                    eta_minutes = eta_seconds / 60
                    if eta_minutes > 60:
                        progress_content['eta_display'] = f"{eta_minutes/60:.1f} hours"
                    elif eta_minutes > 1:
                        progress_content['eta_display'] = f"{eta_minutes:.1f} minutes"
                    else:
                        progress_content['eta_display'] = f"{eta_seconds:.0f} seconds"
            
            await self.send_chunk(
                stream_id, 
                progress_content, 
                chunk_type="progress", 
                progress=progress
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to send progress update: {e}")
    
    async def send_status_message(self, stream_id: str, message: str, 
                                level: str = "info") -> None:
        """
        Send a status message to the stream.
        
        Args:
            stream_id: Stream session ID
            message: Status message
            level: Message level (info, warning, error)
        """
        try:
            status_content = {
                'message': message,
                'level': level,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.send_chunk(stream_id, status_content, chunk_type="status")
            
        except Exception as e:
            logger.error(f"❌ Failed to send status message: {e}")
    
    async def send_error(self, stream_id: str, error_message: str, 
                        error_code: Optional[str] = None,
                        terminate_stream: bool = True) -> None:
        """
        Send an error notification to the stream.
        
        Args:
            stream_id: Stream session ID
            error_message: Error message
            error_code: Optional error code
            terminate_stream: Whether to terminate stream after error
        """
        try:
            error_content = {
                'error': True,
                'error_message': error_message,
                'timestamp': datetime.now().isoformat()
            }
            
            if error_code:
                error_content['error_code'] = error_code
            
            # Include error details if configured
            if self.config.error_handling.get('include_error_details', True):
                session = self.get_stream_session(stream_id)
                if session:
                    error_content['session_info'] = {
                        'chunk_count': session.chunk_count,
                        'bytes_sent': session.total_bytes_sent,
                        'progress': session.progress
                    }
            
            await self.send_chunk(
                stream_id, 
                error_content, 
                chunk_type="error",
                is_final=terminate_stream
            )
            
            # Update session error count
            session = self.get_stream_session(stream_id)
            if session:
                session.error_count += 1
                if terminate_stream:
                    session.status = "error"
            
        except Exception as e:
            logger.error(f"❌ Failed to send error to stream: {e}")
    
    async def finalize_stream(self, stream_id: str, 
                            final_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Finalize and close a stream.
        
        Args:
            stream_id: Stream session ID
            final_data: Optional final data to send
        """
        try:
            session = self.get_stream_session(stream_id)
            if not session:
                logger.warning(f"⚠️ Attempted to finalize non-existent stream: {stream_id}")
                return
            
            # Send final chunk if data provided
            if final_data:
                await self.send_chunk(stream_id, final_data, chunk_type="finalization", is_final=True)
            else:
                # Send completion notification
                completion_data = {
                    'completed': True,
                    'total_chunks': session.chunk_count,
                    'total_bytes': session.total_bytes_sent,
                    'final_progress': session.progress,
                    'duration_seconds': (datetime.now() - session.created_at).total_seconds()
                }
                await self.send_chunk(stream_id, completion_data, chunk_type="completion", is_final=True)
            
            # Mark session as completed
            session.is_active = False
            session.status = "completed"
            
            logger.debug(f"✅ Finalized stream: {stream_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to finalize stream {stream_id}: {e}")
            await self.handle_stream_error(stream_id, str(e))
    
    async def terminate_stream(self, stream_id: str, reason: str = "terminated") -> None:
        """
        Forcefully terminate a stream.
        
        Args:
            stream_id: Stream session ID
            reason: Termination reason
        """
        try:
            session = self.get_stream_session(stream_id)
            if session:
                session.is_active = False
                session.status = "terminated"
                
                # Send termination notification if graceful termination enabled
                if self.config.error_handling.get('graceful_termination', True):
                    termination_data = {
                        'terminated': True,
                        'reason': reason,
                        'final_stats': session.get_session_info()
                    }
                    await self.send_chunk(stream_id, termination_data, chunk_type="termination", is_final=True)
            
            # Remove from active streams
            await self.cleanup_stream(stream_id)
            
            logger.debug(f"✅ Terminated stream: {stream_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to terminate stream {stream_id}: {e}")
    
    def get_stream_session(self, stream_id: str) -> Optional[StreamSession]:
        """Get stream session by ID"""
        return self.active_streams.get(stream_id)
    
    async def get_stream_chunks(self, stream_id: str, since_chunk: Optional[int] = None) -> List[StreamingResponse]:
        """
        Get stream chunks for a session.
        
        Args:
            stream_id: Stream session ID
            since_chunk: Optional chunk number to get chunks since
            
        Returns:
            List of streaming responses
        """
        try:
            session = self.get_stream_session(stream_id)
            if not session:
                return []
            
            if since_chunk is None:
                return session.buffer.copy()
            else:
                # Return chunks after the specified chunk number
                return [chunk for chunk in session.buffer 
                       if int(chunk.chunk_id.split('_')[-1]) > since_chunk]
                
        except Exception as e:
            logger.error(f"❌ Failed to get stream chunks: {e}")
            return []
    
    async def get_stream_iterator(self, stream_id: str) -> AsyncIterator[StreamingResponse]:
        """
        Get an async iterator for stream chunks.
        
        Args:
            stream_id: Stream session ID
            
        Yields:
            StreamingResponse objects as they become available
        """
        try:
            session = self.get_stream_session(stream_id)
            if not session:
                return
            
            last_chunk_count = 0
            
            while session.is_active or len(session.buffer) > last_chunk_count:
                # Yield new chunks
                if len(session.buffer) > last_chunk_count:
                    for chunk in session.buffer[last_chunk_count:]:
                        yield chunk
                    last_chunk_count = len(session.buffer)
                
                # Wait a bit before checking for new chunks
                await asyncio.sleep(0.1)
                
                # Update session reference in case it changed
                session = self.get_stream_session(stream_id)
                if not session:
                    break
                    
        except Exception as e:
            logger.error(f"❌ Stream iterator error: {e}")
    
    async def notify_subscribers(self, session: StreamSession, response: StreamingResponse) -> None:
        """Notify subscribers of new chunk (placeholder for real implementation)"""
        # In real implementation, this would notify WebSocket connections, SSE streams, etc.
        for subscriber in session.subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(response)
                else:
                    subscriber(response)
            except Exception as e:
                logger.error(f"❌ Subscriber notification failed: {e}")
    
    async def flush_stream_buffer(self, stream_id: str) -> None:
        """Flush stream buffer (placeholder for real implementation)"""
        # In real implementation, this would ensure all buffered data is sent to clients
        session = self.get_stream_session(stream_id)
        if session:
            logger.debug(f"✅ Flushed buffer for stream {stream_id}: {len(session.buffer)} chunks")
    
    async def handle_stream_error(self, stream_id: str, error_message: str) -> None:
        """Handle stream error"""
        try:
            if self.config.error_handling.get('stream_on_error', True):
                await self.send_error(stream_id, error_message, terminate_stream=False)
            
            session = self.get_stream_session(stream_id)
            if session:
                session.error_count += 1
                if session.error_count > 10:  # Too many errors, terminate
                    await self.terminate_stream(stream_id, "too_many_errors")
            
        except Exception as e:
            logger.error(f"❌ Error handling failed for stream {stream_id}: {e}")
    
    async def cleanup_stream(self, stream_id: str) -> None:
        """Clean up stream resources"""
        try:
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
            
            if stream_id in self.stream_locks:
                del self.stream_locks[stream_id]
            
            logger.debug(f"✅ Cleaned up stream: {stream_id}")
            
        except Exception as e:
            logger.error(f"❌ Stream cleanup failed: {e}")
    
    async def cleanup_expired_streams(self) -> None:
        """Background task to cleanup expired streams"""
        while True:
            try:
                current_time = datetime.now()
                expired_streams = []
                
                for stream_id, session in self.active_streams.items():
                    if session.is_expired():
                        expired_streams.append(stream_id)
                
                for stream_id in expired_streams:
                    logger.debug(f"🗑️ Cleaning up expired stream: {stream_id}")
                    await self.terminate_stream(stream_id, "expired")
                
                # Sleep before next cleanup cycle
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Stream cleanup task error: {e}")
                await asyncio.sleep(60)  # Continue despite errors
    
    def get_stream_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        try:
            active_count = len(self.active_streams)
            total_chunks = sum(session.chunk_count for session in self.active_streams.values())
            total_bytes = sum(session.total_bytes_sent for session in self.active_streams.values())
            
            stream_types = {}
            for session in self.active_streams.values():
                stream_type = session.stream_type.value
                stream_types[stream_type] = stream_types.get(stream_type, 0) + 1
            
            return {
                'active_streams': active_count,
                'max_concurrent': self.config.max_concurrent_streams,
                'total_chunks_sent': total_chunks,
                'total_bytes_sent': total_bytes,
                'stream_types': stream_types,
                'streaming_enabled': self.config.enable_streaming
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get stream statistics: {e}")
            return {'error': str(e)}
    
    async def get_health_status(self) -> str:
        """Get streaming handler health status"""
        try:
            if not self.config.enable_streaming:
                return "disabled"
            
            # Check for too many active streams
            if len(self.active_streams) >= self.config.max_concurrent_streams:
                return "degraded"
            
            # Check for any streams with excessive errors
            error_streams = sum(1 for session in self.active_streams.values() if session.error_count > 5)
            if error_streams > len(self.active_streams) * 0.5:  # More than 50% have errors
                return "degraded"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"❌ Streaming health check failed: {e}")
            return "unhealthy" 