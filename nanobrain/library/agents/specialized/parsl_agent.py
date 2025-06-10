"""
Parsl Agent

A distributed agent implementation using Parsl for parallel execution.
Inherits from ConversationalAgent and adds distributed processing capabilities.
"""

import asyncio
import time
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

# NanoBrain core imports - use absolute imports
try:
    from nanobrain.core.agent import ConversationalAgent, AgentConfig
    from nanobrain.core.executor import ParslExecutor
except ImportError:
    # Fallback for direct import scenarios
    import sys
    from pathlib import Path
    
    # Add project root to path
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from nanobrain.core.agent import ConversationalAgent, AgentConfig
    from nanobrain.core.executor import ParslExecutor

# Parsl imports with fallback
try:
    from parsl import python_app
    PARSL_AVAILABLE = True
except ImportError:
    PARSL_AVAILABLE = False
    
    # Mock decorator for when Parsl is not available
    def python_app(func):
        return func


class ParslAgent(ConversationalAgent):
    """
    Distributed agent using Parsl for parallel execution.
    
    Inherits all standard agent capabilities from ConversationalAgent
    and adds distributed processing using Parsl for high-performance
    computing environments.
    
    Key Features:
    - Distributed message processing via Parsl
    - Maintains full compatibility with ConversationalAgent interface
    - Supports all existing agent features (system prompts, temperature, etc.)
    - Automatic agent reconstruction in worker processes
    - Proper serialization handling for distributed execution
    """
    
    def __init__(self, config: AgentConfig, parsl_executor: Optional[ParslExecutor] = None):
        """
        Initialize ParslAgent with distributed execution capabilities.
        
        Args:
            config: Agent configuration
            parsl_executor: Optional ParslExecutor for distributed processing
        """
        super().__init__(config)
        self.parsl_executor = parsl_executor
        self.parsl_apps_registered = False
        self._serializable_config = None
        
        # Performance tracking
        self.distributed_calls = 0
        self.total_distributed_time = 0.0
        
    async def initialize(self):
        """Initialize both parent agent and Parsl components."""
        # Initialize parent agent first
        await super().initialize()
        
        # Setup Parsl applications if executor is available
        if self.parsl_executor and PARSL_AVAILABLE:
            await self._setup_parsl_apps()
            
        # Prepare serializable configuration for workers
        self._prepare_serializable_config()
        
        if hasattr(self, 'nb_logger'):
            self.nb_logger.info(
                f"ParslAgent initialized",
                agent_name=self.config.name,
                parsl_enabled=bool(self.parsl_executor and PARSL_AVAILABLE),
                model=self.config.model
            )
    
    async def process(self, message: str, context: Optional[Dict] = None) -> str:
        """
        Process message using distributed Parsl execution.
        
        This is the core agent functionality - overrides parent method
        to use Parsl for distributed processing while maintaining
        the same interface and behavior.
        
        Args:
            message: User message to process
            context: Optional conversation context
            
        Returns:
            Processed response from the agent
        """
        start_time = time.time()
        
        try:
            # If Parsl is available and configured, use distributed processing
            if (self.parsl_executor and PARSL_AVAILABLE and 
                self.parsl_apps_registered and self._serializable_config):
                
                if hasattr(self, 'nb_logger'):
                    self.nb_logger.debug(
                        f"Processing message via Parsl distributed execution",
                        agent_name=self.config.name,
                        message_length=len(message)
                    )
                
                # Use Parsl for distributed processing
                response = await self._process_distributed(message, context)
                
                # Update performance metrics
                processing_time = time.time() - start_time
                self.distributed_calls += 1
                self.total_distributed_time += processing_time
                
                if hasattr(self, 'nb_logger'):
                    self.nb_logger.info(
                        f"Distributed processing completed",
                        agent_name=self.config.name,
                        processing_time=processing_time,
                        distributed_calls=self.distributed_calls
                    )
                
                return response
                
            else:
                # Fallback to parent class processing
                if hasattr(self, 'nb_logger'):
                    self.nb_logger.debug(
                        f"Using fallback local processing",
                        agent_name=self.config.name,
                        reason="parsl_not_available" if not PARSL_AVAILABLE else "parsl_not_configured"
                    )
                
                return await super().process(message, context)
                
        except Exception as e:
            processing_time = time.time() - start_time
            
            if hasattr(self, 'nb_logger'):
                self.nb_logger.error(
                    f"Error in ParslAgent processing: {str(e)}",
                    agent_name=self.config.name,
                    error_type=type(e).__name__,
                    processing_time=processing_time
                )
            
            # Fallback to parent class on error
            return await super().process(message, context)
    
    async def _setup_parsl_apps(self):
        """Setup Parsl applications for distributed processing."""
        if not PARSL_AVAILABLE:
            return
            
        try:
            # Register the distributed processing app
            # The app function is defined as a module-level function for proper serialization
            self.parsl_apps_registered = True
            
            if hasattr(self, 'nb_logger'):
                self.nb_logger.info(
                    f"Parsl applications registered successfully",
                    agent_name=self.config.name
                )
                
        except Exception as e:
            if hasattr(self, 'nb_logger'):
                self.nb_logger.error(
                    f"Failed to setup Parsl apps: {str(e)}",
                    agent_name=self.config.name,
                    error_type=type(e).__name__
                )
            self.parsl_apps_registered = False
    
    def _prepare_serializable_config(self):
        """Prepare agent configuration that can be serialized for Parsl workers."""
        try:
            self._serializable_config = {
                'name': self.config.name,
                'description': self.config.description,
                'model': self.config.model,
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens,
                'system_prompt': self.config.system_prompt,
                'auto_initialize': False,  # Don't auto-initialize in workers
                'debug_mode': self.config.debug_mode,
                'enable_logging': False,  # Disable logging in workers to avoid conflicts
                'log_conversations': False,
                'timeout': getattr(self.config, 'timeout', 30.0),
                'retry_attempts': getattr(self.config, 'retry_attempts', 3),
                'api_key_env_var': 'OPENAI_API_KEY',  # Workers will use env var
            }
            
            if hasattr(self, 'nb_logger'):
                self.nb_logger.debug(
                    f"Serializable config prepared",
                    agent_name=self.config.name,
                    config_keys=list(self._serializable_config.keys())
                )
                
        except Exception as e:
            if hasattr(self, 'nb_logger'):
                self.nb_logger.error(
                    f"Failed to prepare serializable config: {str(e)}",
                    agent_name=self.config.name,
                    error_type=type(e).__name__
                )
            self._serializable_config = None
    
    async def _process_distributed(self, message: str, context: Optional[Dict] = None) -> str:
        """Process message using Parsl distributed execution."""
        try:
            # Submit task to Parsl
            future = process_agent_message_distributed(
                self._serializable_config,
                message,
                context or {}
            )
            
            # Wait for result
            result = future.result()
            
            # Extract response from result
            if isinstance(result, dict):
                response = result.get('response', str(result))
                
                # Log any errors from worker
                if result.get('error'):
                    if hasattr(self, 'nb_logger'):
                        self.nb_logger.warning(
                            f"Worker reported error: {result['error']}",
                            agent_name=self.config.name
                        )
            else:
                response = str(result)
            
            return response
            
        except Exception as e:
            if hasattr(self, 'nb_logger'):
                self.nb_logger.error(
                    f"Distributed processing failed: {str(e)}",
                    agent_name=self.config.name,
                    error_type=type(e).__name__
                )
            
            # Fallback to local processing
            return await super().process(message, context)
    
    async def shutdown(self):
        """Shutdown the agent and cleanup resources."""
        if hasattr(self, 'nb_logger'):
            self.nb_logger.info(
                f"Shutting down ParslAgent",
                agent_name=self.config.name,
                distributed_calls=self.distributed_calls,
                total_distributed_time=self.total_distributed_time
            )
        
        # Call parent shutdown
        await super().shutdown()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent."""
        base_metrics = {}
        if hasattr(super(), 'get_performance_metrics'):
            base_metrics = super().get_performance_metrics()
        
        parsl_metrics = {
            'distributed_calls': self.distributed_calls,
            'total_distributed_time': self.total_distributed_time,
            'avg_distributed_time': (
                self.total_distributed_time / max(self.distributed_calls, 1)
            ),
            'parsl_enabled': bool(self.parsl_executor and PARSL_AVAILABLE),
            'parsl_apps_registered': self.parsl_apps_registered,
        }
        
        return {**base_metrics, **parsl_metrics}


# Parsl app function for distributed agent processing
# This must be defined at module level for proper serialization
@python_app
def process_agent_message_distributed(agent_config: Dict, message: str, context: Dict = None):
    """
    Parsl app that processes a message using a reconstructed agent.
    
    This runs in Parsl workers and:
    1. Reconstructs the agent from serializable config
    2. Calls the agent's parent class process() method
    3. Returns the actual LLM response
    
    Args:
        agent_config: Serializable agent configuration
        message: Message to process
        context: Optional conversation context
        
    Returns:
        Dict containing response and metadata
    """
    import os
    import asyncio
    from datetime import datetime
    
    try:
        # Import NanoBrain components in worker
        from nanobrain.core.agent import ConversationalAgent, AgentConfig
        
        # Reconstruct agent configuration
        config = AgentConfig(**agent_config)
        
        # Create agent instance in worker
        agent = ConversationalAgent(config)
        
        # Initialize agent
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(agent.initialize())
            
            # Process message using real agent
            response = loop.run_until_complete(agent.process(message, context))
            
            # Cleanup
            loop.run_until_complete(agent.shutdown())
            
            return {
                'response': response,
                'agent_name': config.name,
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'error': None
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        return {
            'response': f"Error processing message: {str(e)}",
            'agent_name': agent_config.get('name', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': str(e)
        }
