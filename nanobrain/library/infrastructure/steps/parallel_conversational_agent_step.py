"""
Parallel Conversational Agent Step Implementation

Specialized parallel processing for conversational agents with chat-specific features.

This module provides conversational agent-specific parallel processing:
- Chat request/response handling
- Conversation context management
- Token usage tracking
- Chat-specific performance metrics
"""

import time
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from nanobrain.core.agent import ConversationalAgent
from .parallel_agent_step import (
    ParallelAgentStep,
    ParallelAgentConfig,
    AgentRequest,
    AgentResponse
)


@dataclass
class ChatRequest(AgentRequest):
    """Request for conversational agent processing."""
    message: str = ""
    user_id: str = "default_user"
    conversation_id: Optional[str] = None
    
    def __post_init__(self):
        # Set input_data to the message for compatibility with base class
        if not self.input_data and self.message:
            self.input_data = self.message
        elif not self.message and self.input_data:
            self.message = str(self.input_data)
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'message': self.message,
            'user_id': self.user_id,
            'conversation_id': self.conversation_id
        })
        return base_dict


@dataclass
class ChatResponse(AgentResponse):
    """Response from conversational agent processing."""
    response: str = ""
    tokens_used: int = 0
    conversation_id: Optional[str] = None
    
    def __post_init__(self):
        # Set output_data to the response for compatibility with base class
        if not self.output_data and self.response:
            self.output_data = self.response
        elif not self.response and self.output_data:
            self.response = str(self.output_data)
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'response': self.response,
            'tokens_used': self.tokens_used,
            'conversation_id': self.conversation_id
        })
        return base_dict


class ParallelConversationalAgentConfig(ParallelAgentConfig):
    """Configuration for parallel conversational agent processing."""
    track_token_usage: bool = True
    max_message_length: int = 4000
    enable_conversation_context: bool = True
    context_window_size: int = 10
    response_timeout: float = 30.0
    enable_response_caching: bool = False
    cache_ttl_seconds: int = 300


class ParallelConversationalAgentStep(ParallelAgentStep):
    """
    Parallel processing step specialized for conversational agents.
    
    This class provides:
    - Chat-specific request/response handling
    - Token usage tracking and optimization
    - Conversation context management
    - Response caching and optimization
    - Chat-specific performance metrics
    """
    
    def __init__(self, config: ParallelConversationalAgentConfig, 
                 agents: List[ConversationalAgent], **kwargs):
        super().__init__(config, agents, **kwargs)
        self.chat_config = config
        
        # Chat-specific metrics
        self.total_tokens_used = 0
        self.total_messages_processed = 0
        self.conversation_contexts = {}  # conversation_id -> context
        self.response_cache = {}  # message_hash -> (response, timestamp)
        
        # Validate that all agents are conversational agents
        for i, agent in enumerate(agents):
            if not isinstance(agent, ConversationalAgent):
                self.logger.warning(f"Agent {i} is not a ConversationalAgent: {type(agent).__name__}")
        
        self.logger.info(f"Initialized ParallelConversationalAgentStep with {len(agents)} conversational agents",
                        agent_count=len(agents),
                        track_tokens=config.track_token_usage,
                        enable_context=config.enable_conversation_context)
    
    async def _extract_requests(self, inputs: Dict[str, Any]) -> List[ChatRequest]:
        """Extract chat requests from input data."""
        requests = []
        
        # Handle different input formats
        if 'requests' in inputs:
            # Batch of requests
            request_data = inputs['requests']
            if not isinstance(request_data, list):
                request_data = [request_data]
            
            for data in request_data:
                request = self._create_chat_request(data)
                if request:
                    requests.append(request)
        
        elif 'user_input' in inputs or 'message' in inputs:
            # Single request
            request = self._create_chat_request(inputs)
            if request:
                requests.append(request)
        
        else:
            # Try to create request from entire input
            request = self._create_chat_request(inputs)
            if request:
                requests.append(request)
        
        return requests
    
    def _create_chat_request(self, data: Any) -> Optional[ChatRequest]:
        """Create a ChatRequest from input data."""
        try:
            if isinstance(data, dict):
                # Extract message from various possible keys
                message = ""
                if 'user_input' in data:
                    user_input = data['user_input']
                    if isinstance(user_input, dict):
                        message = user_input.get('user_input', str(user_input))
                    else:
                        message = str(user_input)
                elif 'message' in data:
                    message = str(data['message'])
                else:
                    message = str(data)
                
                # Validate message length
                if len(message) > self.chat_config.max_message_length:
                    self.logger.warning(f"Message length {len(message)} exceeds maximum {self.chat_config.max_message_length}")
                    message = message[:self.chat_config.max_message_length]
                
                return ChatRequest(
                    message=message,
                    user_id=data.get('user_id', 'default_user'),
                    conversation_id=data.get('conversation_id'),
                    priority=data.get('priority', 1),
                    timeout=data.get('timeout', self.chat_config.response_timeout),
                    context=data.get('context', {})
                )
            else:
                # Simple string message
                message = str(data)
                if len(message) > self.chat_config.max_message_length:
                    message = message[:self.chat_config.max_message_length]
                
                return ChatRequest(message=message)
                
        except Exception as e:
            self.logger.error(f"Failed to create chat request from data: {e}")
            return None
    
    async def _execute_processor(self, processor: ConversationalAgent, 
                               request: ChatRequest, processor_index: int) -> ChatResponse:
        """Execute a conversational agent with a chat request."""
        start_time = time.time()
        
        try:
            # Check if agent is healthy
            if not self.agent_health_status[processor_index]['healthy']:
                raise RuntimeError(f"Agent {processor_index} is marked as unhealthy")
            
            # Check response cache if enabled
            if self.chat_config.enable_response_caching:
                cached_response = self._check_response_cache(request.message)
                if cached_response:
                    self.logger.debug(f"Using cached response for message: {request.message[:50]}...")
                    processing_time = time.time() - start_time
                    
                    return ChatResponse(
                        request_id=request.id,
                        response=cached_response,
                        processing_time=processing_time,
                        processor_id=f"agent_{processor_index}_cached",
                        success=True,
                        conversation_id=request.conversation_id,
                        agent_metadata={
                            'agent_type': type(processor).__name__,
                            'agent_index': processor_index,
                            'cached_response': True
                        }
                    )
            
            # Add conversation context if enabled
            enhanced_message = request.message
            if (self.chat_config.enable_conversation_context and 
                request.conversation_id and 
                request.conversation_id in self.conversation_contexts):
                
                context = self.conversation_contexts[request.conversation_id]
                if context:
                    # Add recent conversation history to the message
                    context_str = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" 
                                           for msg in context[-self.chat_config.context_window_size:]])
                    enhanced_message = f"Previous conversation:\n{context_str}\n\nCurrent message: {request.message}"
            
            # Process with the conversational agent
            result = await processor.process(enhanced_message)
            
            processing_time = time.time() - start_time
            
            # Extract response text
            if isinstance(result, dict):
                response_text = result.get('response', str(result))
                tokens_used = result.get('tokens_used', 0)
            else:
                response_text = str(result)
                tokens_used = 0  # Would need to implement token counting
            
            # Update conversation context if enabled
            if (self.chat_config.enable_conversation_context and request.conversation_id):
                self._update_conversation_context(request.conversation_id, request.message, response_text)
            
            # Cache response if enabled
            if self.chat_config.enable_response_caching:
                self._cache_response(request.message, response_text)
            
            # Update metrics
            self.total_messages_processed += 1
            if self.chat_config.track_token_usage:
                self.total_tokens_used += tokens_used
            
            # Update agent performance metrics
            self._update_agent_performance(processor_index, processing_time, True)
            
            return ChatResponse(
                request_id=request.id,
                response=response_text,
                processing_time=processing_time,
                processor_id=f"agent_{processor_index}",
                success=True,
                tokens_used=tokens_used,
                conversation_id=request.conversation_id,
                agent_metadata={
                    'agent_type': type(processor).__name__,
                    'agent_index': processor_index,
                    'message_length': len(request.message),
                    'response_length': len(response_text)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Update agent performance metrics for failure
            self._update_agent_performance(processor_index, processing_time, False)
            
            # Mark agent as potentially unhealthy
            self.agent_health_status[processor_index]['consecutive_failures'] += 1
            
            raise e
    
    async def _create_error_response(self, request: ChatRequest, error_message: str, 
                                   processing_time: float = 0.0, processor_id: str = "") -> ChatResponse:
        """Create an error response for a failed chat request."""
        return ChatResponse(
            request_id=request.id,
            response=f"I apologize, but I encountered an error: {error_message}",
            processing_time=processing_time,
            processor_id=processor_id,
            success=False,
            error=error_message,
            conversation_id=request.conversation_id,
            agent_metadata={'error_type': 'chat_processing_failure'}
        )
    
    def _check_response_cache(self, message: str) -> Optional[str]:
        """Check if a response is cached for the given message."""
        import hashlib
        message_hash = hashlib.md5(message.encode()).hexdigest()
        
        if message_hash in self.response_cache:
            cached_response, timestamp = self.response_cache[message_hash]
            
            # Check if cache entry is still valid
            if (datetime.now() - timestamp).total_seconds() < self.chat_config.cache_ttl_seconds:
                return cached_response
            else:
                # Remove expired cache entry
                del self.response_cache[message_hash]
        
        return None
    
    def _cache_response(self, message: str, response: str):
        """Cache a response for the given message."""
        import hashlib
        message_hash = hashlib.md5(message.encode()).hexdigest()
        self.response_cache[message_hash] = (response, datetime.now())
        
        # Clean up old cache entries if cache is getting too large
        if len(self.response_cache) > 1000:  # Configurable limit
            # Remove oldest entries
            sorted_cache = sorted(self.response_cache.items(), 
                                key=lambda x: x[1][1])  # Sort by timestamp
            for key, _ in sorted_cache[:100]:  # Remove oldest 100 entries
                del self.response_cache[key]
    
    def _update_conversation_context(self, conversation_id: str, user_message: str, assistant_response: str):
        """Update conversation context for a given conversation."""
        if conversation_id not in self.conversation_contexts:
            self.conversation_contexts[conversation_id] = []
        
        context = self.conversation_contexts[conversation_id]
        context.append({
            'user': user_message,
            'assistant': assistant_response,
            'timestamp': datetime.now()
        })
        
        # Keep only recent messages within the context window
        if len(context) > self.chat_config.context_window_size * 2:  # *2 for user+assistant pairs
            context = context[-self.chat_config.context_window_size * 2:]
            self.conversation_contexts[conversation_id] = context
    
    def get_chat_stats(self) -> Dict[str, Any]:
        """Get chat-specific statistics."""
        return {
            'total_messages_processed': self.total_messages_processed,
            'total_tokens_used': self.total_tokens_used,
            'avg_tokens_per_message': (self.total_tokens_used / max(self.total_messages_processed, 1)),
            'active_conversations': len(self.conversation_contexts),
            'cached_responses': len(self.response_cache),
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified implementation)."""
        # This would need more sophisticated tracking in a real implementation
        return 0.0  # Placeholder
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics including chat-specific metrics."""
        base_stats = super().get_performance_stats()
        chat_stats = self.get_chat_stats()
        
        return {**base_stats, 'chat_stats': chat_stats}
    
    def clear_conversation_context(self, conversation_id: str = None):
        """Clear conversation context for a specific conversation or all conversations."""
        if conversation_id:
            if conversation_id in self.conversation_contexts:
                del self.conversation_contexts[conversation_id]
                self.logger.info(f"Cleared conversation context for {conversation_id}")
        else:
            self.conversation_contexts.clear()
            self.logger.info("Cleared all conversation contexts")
    
    def clear_response_cache(self):
        """Clear the response cache."""
        self.response_cache.clear()
        self.logger.info("Cleared response cache") 