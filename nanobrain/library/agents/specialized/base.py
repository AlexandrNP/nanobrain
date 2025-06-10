"""
Base Agent Classes

Base classes for specialized agents in the NanoBrain framework.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from abc import ABC, abstractmethod

# Updated imports for nanobrain package structure
from nanobrain.core.agent import Agent, SimpleAgent, ConversationalAgent, AgentConfig
from nanobrain.core.executor import LocalExecutor, ExecutorConfig
from nanobrain.core.logging_system import NanoBrainLogger, get_logger


class SpecializedAgentBase(ABC):
    """
    Base mixin for specialized agents that provides common specialized functionality.
    
    This class provides:
    - Specialized processing patterns
    - Domain-specific error handling
    - Performance tracking for specialized operations
    - Integration with core agent capabilities
    """
    
    def __init__(self, **kwargs):
        """Initialize specialized agent base."""
        super().__init__(**kwargs)
        
        # Specialized agent tracking
        self._specialized_operations_count = 0
        self._specialized_errors_count = 0
        self._domain_specific_metrics = {}
        
        # Get specialized logger
        if hasattr(self, 'name'):
            self.specialized_logger = get_logger(f"specialized.{self.name}")
        else:
            self.specialized_logger = get_logger("specialized.agent")
    
    async def initialize(self) -> None:
        """Initialize the specialized agent."""
        await super().initialize()
        await self._initialize_specialized_features()
        
        self.specialized_logger.info(f"Specialized agent {getattr(self, 'name', 'unknown')} initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the specialized agent."""
        await self._shutdown_specialized_features()
        await super().shutdown()
        
        self.specialized_logger.info(
            f"Specialized agent {getattr(self, 'name', 'unknown')} shutdown",
            specialized_operations=self._specialized_operations_count,
            specialized_errors=self._specialized_errors_count
        )
    
    async def _initialize_specialized_features(self) -> None:
        """Initialize specialized features. Override in subclasses."""
        pass
    
    async def _shutdown_specialized_features(self) -> None:
        """Shutdown specialized features. Override in subclasses."""
        pass
    
    def _track_specialized_operation(self, operation_name: str, success: bool = True) -> None:
        """Track specialized operation metrics."""
        self._specialized_operations_count += 1
        
        if not success:
            self._specialized_errors_count += 1
        
        # Track domain-specific metrics
        if operation_name not in self._domain_specific_metrics:
            self._domain_specific_metrics[operation_name] = {'count': 0, 'errors': 0}
        
        self._domain_specific_metrics[operation_name]['count'] += 1
        if not success:
            self._domain_specific_metrics[operation_name]['errors'] += 1
    
    def get_specialized_performance_stats(self) -> Dict[str, Any]:
        """Get specialized performance statistics."""
        base_stats = {}
        if hasattr(super(), 'get_performance_stats'):
            base_stats = super().get_performance_stats()
        
        specialized_stats = {
            'specialized_operations_count': self._specialized_operations_count,
            'specialized_errors_count': self._specialized_errors_count,
            'specialized_error_rate': (
                self._specialized_errors_count / max(1, self._specialized_operations_count)
            ),
            'domain_specific_metrics': self._domain_specific_metrics.copy()
        }
        
        return {**base_stats, **specialized_stats}
    
    @abstractmethod
    async def _process_specialized_request(self, input_text: str, **kwargs) -> Optional[str]:
        """
        Process specialized requests that don't require LLM.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional parameters
            
        Returns:
            Processed result if handled, None if should fall back to LLM
        """
        pass
    
    def _should_handle_specialized(self, input_text: str, **kwargs) -> bool:
        """
        Determine if this request should be handled by specialized logic.
        
        Args:
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            True if should be handled by specialized logic
        """
        return False


class SimpleSpecializedAgent(SpecializedAgentBase, SimpleAgent):
    """
    Simple specialized agent that processes input without conversation history.
    
    Combines SimpleAgent capabilities with specialized functionality.
    """
    
    def __init__(self, config: AgentConfig, **kwargs):
        super().__init__(config=config, **kwargs)
    
    async def process(self, input_text: str, **kwargs) -> str:
        """
        Process input with specialized logic first, then fall back to LLM.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional parameters
            
        Returns:
            Processed response
        """
        # Try specialized processing first
        if self._should_handle_specialized(input_text, **kwargs):
            try:
                specialized_result = await self._process_specialized_request(input_text, **kwargs)
                if specialized_result is not None:
                    self._track_specialized_operation("direct_processing", success=True)
                    return specialized_result
            except Exception as e:
                self._track_specialized_operation("direct_processing", success=False)
                self.specialized_logger.error(f"Specialized processing failed: {e}")
        
        # Fall back to parent LLM processing
        return await super().process(input_text, **kwargs)


class ConversationalSpecializedAgent(SpecializedAgentBase, ConversationalAgent):
    """
    Conversational specialized agent that maintains conversation history.
    
    Combines ConversationalAgent capabilities with specialized functionality.
    """
    
    def __init__(self, config: AgentConfig, **kwargs):
        super().__init__(config=config, **kwargs)
    
    async def process(self, input_text: str, **kwargs) -> str:
        """
        Process input with specialized logic first, then fall back to conversational LLM.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional parameters
            
        Returns:
            Processed response
        """
        # Try specialized processing first
        if self._should_handle_specialized(input_text, **kwargs):
            try:
                specialized_result = await self._process_specialized_request(input_text, **kwargs)
                if specialized_result is not None:
                    self._track_specialized_operation("direct_processing", success=True)
                    # Add to conversation history for context
                    self.add_to_conversation("user", input_text)
                    self.add_to_conversation("assistant", specialized_result)
                    return specialized_result
            except Exception as e:
                self._track_specialized_operation("direct_processing", success=False)
                self.specialized_logger.error(f"Specialized processing failed: {e}")
        
        # Fall back to parent conversational processing
        return await super().process(input_text, **kwargs)


def create_specialized_agent(
    agent_type: str,
    specialized_class: type,
    config: AgentConfig,
    **kwargs
) -> Agent:
    """
    Factory function to create specialized agents.
    
    Args:
        agent_type: Type of agent ('simple' or 'conversational')
        specialized_class: The specialized agent class
        config: Agent configuration
        **kwargs: Additional arguments
        
    Returns:
        Specialized agent instance
    """
    logger = get_logger("specialized.factory")
    logger.info(
        f"Creating specialized agent: {config.name}",
        agent_type=agent_type,
        specialized_class=specialized_class.__name__
    )
    
    if agent_type.lower() == "simple":
        # Create a simple specialized agent class dynamically
        class SimpleSpecialized(specialized_class, SimpleSpecializedAgent):
            pass
        return SimpleSpecialized(config, **kwargs)
    
    elif agent_type.lower() == "conversational":
        # Create a conversational specialized agent class dynamically
        class ConversationalSpecialized(specialized_class, ConversationalSpecializedAgent):
            pass
        return ConversationalSpecialized(config, **kwargs)
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}") 