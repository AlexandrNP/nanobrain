"""
Parsl Chat Workflow

A comprehensive chat workflow implementation using Parsl for distributed execution.
Integrates with the existing NanoBrain ParslExecutor and follows the library's
architectural patterns.
"""

import sys
import os
import asyncio
import yaml
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# Ensure proper module paths for Parsl serialization
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Core framework imports with proper nanobrain package structure
from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig, DataUnitType
from nanobrain.core.executor import ParslExecutor, ExecutorConfig, ExecutorType
from nanobrain.core.logging_system import get_logger, set_debug_mode
from nanobrain.core.agent import AgentConfig
from nanobrain.library.agents.conversational import EnhancedCollaborativeAgent
from nanobrain.config import get_config_manager

# Import Parsl apps for distributed processing
try:
    from .apps import process_message_with_agent, aggregate_responses, create_agent_config_dict
    PARSL_APPS_AVAILABLE = True
except ImportError:
    PARSL_APPS_AVAILABLE = False

# Parsl imports with fallback
try:
    import parsl
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    PARSL_AVAILABLE = True
except ImportError:
    PARSL_AVAILABLE = False


class ConversationHistoryUnit:
    """Simple conversation history implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.history = []
        self.max_messages = config.get('max_messages_per_conversation', 1000)
        
    async def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation history."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.history.append(message)
        
        # Trim history if needed
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]
    
    async def get_recent_messages(self, count: int = 10) -> List[Dict]:
        """Get recent messages from history."""
        return self.history[-count:] if self.history else []
    
    async def clear(self):
        """Clear conversation history."""
        self.history.clear()


class ParslChatWorkflow:
    """
    Parsl-based chat workflow using proper NanoBrain architecture.
    
    This workflow demonstrates:
    - Proper use of ParslAgent for distributed processing
    - Integration with NanoBrain package structure
    - Parsl-based parallel execution
    - Performance monitoring and logging
    """
    
    def __init__(self):
        self.config = None
        self.agents = []
        self.data_units = {}
        self.conversation_history = None
        self.executor = None
        self.is_initialized = False
        
        # Setup logging
        set_debug_mode(True)
        self.logger = get_logger("parsl_chat_workflow")
    

        
    async def initialize(self, config_path: str):
        """Initialize the workflow from configuration."""
        try:
            self.logger.info("Initializing Parsl chat workflow")
            
            # Load configuration
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Setup executor
            await self._setup_parsl_executor()
            
            # Setup data units
            await self._setup_data_units()
            
            # Setup agents
            await self._setup_agents()
            
            # Setup conversation history
            await self._setup_conversation_history()
            
            self.is_initialized = True
            self.logger.info("Parsl chat workflow initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize workflow: {e}")
            raise
    
    async def _setup_parsl_executor(self):
        """Setup the Parsl executor."""
        self.logger.info("Setting up Parsl executor")
        
        executor_config_data = self.config['config']['executor']
        
        # Create ExecutorConfig for ParslExecutor
        executor_config = ExecutorConfig(
            executor_type=executor_config_data['type'],
            max_workers=executor_config_data['max_workers'],
            timeout=executor_config_data.get('timeout'),
            parsl_config=None  # Let ParslExecutor handle Parsl config internally
        )
        
        self.executor = ParslExecutor(config=executor_config)
        await self.executor.initialize()
        
        self.logger.info("Parsl executor initialized successfully")
    
    async def _setup_data_units(self):
        """Setup data units for the workflow."""
        self.logger.info("Setting up data units")
        
        data_units_config = self.config['config']['data_units']
        
        for name, config in data_units_config.items():
            if config['type'] == 'memory':
                du_config = DataUnitConfig(
                    name=name,
                    data_type=config['type'],
                    persistent=False,
                    cache_size=config.get('cache_size', 100)
                )
                self.data_units[name] = DataUnitMemory(du_config)
                await self.data_units[name].initialize()
        
        self.logger.info(f"Created {len(self.data_units)} data units")
    
    async def _setup_agents(self):
        """Setup EnhancedCollaborativeAgent instances."""
        self.logger.info("Setting up Enhanced Collaborative agents")
        
        # Load agent configurations from the workflow config
        agents_config = self.config['config'].get('agents', [])
        
        if not agents_config:
            # Create default agents if none configured
            agents_config = [
                {'name': 'creative_agent', 'model': 'gpt-3.5-turbo', 'temperature': 0.9},
                {'name': 'analytical_agent', 'model': 'gpt-3.5-turbo', 'temperature': 0.5},
                {'name': 'balanced_agent', 'model': 'gpt-3.5-turbo', 'temperature': 0.7}
            ]
        
        # Create EnhancedCollaborativeAgent instances
        for i, agent_config_data in enumerate(agents_config):
            agent_config = AgentConfig(
                name=agent_config_data.get('name', f"agent_{i+1}"),
                model=agent_config_data.get('model', 'gpt-3.5-turbo'),
                temperature=agent_config_data.get('temperature', 0.7),
                max_tokens=agent_config_data.get('max_tokens', 1000),
                system_prompt=agent_config_data.get('system_prompt', 'You are a helpful AI assistant.'),
                auto_initialize=False,
                debug_mode=True,
                enable_logging=True,
                log_conversations=True
            )
            
            # Create EnhancedCollaborativeAgent
            agent = EnhancedCollaborativeAgent(agent_config)
            await agent.initialize()
            
            self.agents.append(agent)
            self.logger.info(f"Created EnhancedCollaborativeAgent: {agent_config.name}")
        
        self.logger.info(f"Created {len(self.agents)} Enhanced Collaborative agents")
    
    async def _setup_conversation_history(self):
        """Setup conversation history."""
        history_config = self.config['config']['data_units'].get('conversation_history', {})
        self.conversation_history = ConversationHistoryUnit(history_config)
        self.logger.info("Conversation history initialized")
    
    async def process_user_input(self, message: str) -> str:
        """
        Process user input and return the best response.
        
        Args:
            message: User message to process
            
        Returns:
            Best response from the agents
        """
        result = await self.process_message(message)
        return result.get('response', 'Sorry, I could not process your request.')
    
    async def submit_message(self, message: str):
        """
        Submit a message for asynchronous processing.
        
        Args:
            message: User message to process
            
        Returns:
            Future that will contain the response
        """
        return asyncio.create_task(self.process_user_input(message))
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the workflow."""
        return {
            'total_requests': getattr(self, '_total_requests', 0),
            'successful_requests': getattr(self, '_successful_requests', 0),
            'failed_requests': getattr(self, '_failed_requests', 0),
            'avg_response_time': getattr(self, '_avg_response_time', 0.0),
            'min_response_time': getattr(self, '_min_response_time', 0.0),
            'max_response_time': getattr(self, '_max_response_time', 0.0),
            'throughput': getattr(self, '_throughput', 0.0)
        }
    
    async def get_parsl_stats(self) -> Dict[str, Any]:
        """Get Parsl-specific statistics."""
        if self.executor and hasattr(self.executor, 'get_stats'):
            return await self.executor.get_stats()
        return {
            'active_workers': 0,
            'pending_tasks': 0,
            'completed_tasks': 0
        }
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            'name': self.config.get('name', 'parsl_chat_workflow') if self.config else 'parsl_chat_workflow',
            'initialized': self.is_initialized,
            'agent_count': len(self.agents),
            'parsl_executor': self.executor is not None and self.executor.is_initialized,
            'parsl_apps_available': PARSL_APPS_AVAILABLE,
            'data_units': list(self.data_units.keys()),
            'parsl_apps': ['process_message_with_agent', 'aggregate_responses'] if PARSL_APPS_AVAILABLE else []
        }

    async def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a message using distributed Parsl agents.
        
        Args:
            message: User message to process
            
        Returns:
            Dict containing responses from all agents
        """
        if not self.is_initialized:
            raise RuntimeError("Workflow not initialized")
        
        self.logger.info(f"Processing message with {len(self.agents)} agents via Parsl")
        
        try:
            # Track performance metrics
            start_time = asyncio.get_event_loop().time()
            self._total_requests = getattr(self, '_total_requests', 0) + 1
            
            # Add user message to history
            await self.conversation_history.add_message("user", message)
            
            # Choose processing method based on availability
            if PARSL_APPS_AVAILABLE and self.executor and self.executor.is_initialized:
                # Use Parsl for distributed processing
                self.logger.info("Using Parsl distributed processing")
                
                # Convert agent configs to serializable dictionaries
                agent_configs = []
                for agent in self.agents:
                    agent_config_dict = create_agent_config_dict(agent.config)
                    agent_configs.append(agent_config_dict)
                
                # Submit tasks to Parsl
                futures = []
                for agent_config in agent_configs:
                    future = process_message_with_agent(message, agent_config)
                    futures.append(future)
                
                # Wait for all tasks to complete
                responses = []
                for future in futures:
                    try:
                        # Get result from Parsl future
                        response = future.result()
                        responses.append(response)
                    except Exception as e:
                        responses.append(f"Error: {str(e)}")
                        
            else:
                # Fallback to local execution
                self.logger.info("Using local execution (Parsl not available or not initialized)")
                tasks = []
                for i, agent in enumerate(self.agents):
                    task = agent.process(message)
                    tasks.append(task)
                
                # Wait for all agents to complete
                responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            agent_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    self.logger.error(f"Agent {i+1} failed: {response}")
                    agent_responses.append({
                        'agent_id': f"parsl_agent_{i+1}",
                        'response': f"Error: {str(response)}",
                        'success': False
                    })
                else:
                    agent_responses.append({
                        'agent_id': f"parsl_agent_{i+1}",
                        'response': response,
                        'success': True
                    })
                    
                    # Add agent response to history
                    await self.conversation_history.add_message(
                        "assistant", 
                        response, 
                        {'agent_id': f"parsl_agent_{i+1}"}
                    )
            
            # Select best response (for now, just use the first successful one)
            best_response = None
            for resp in agent_responses:
                if resp['success']:
                    best_response = resp['response']
                    break
            
            if not best_response:
                best_response = "I apologize, but I encountered errors processing your request."
            
            # Update performance metrics
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time
            
            if any(resp['success'] for resp in agent_responses):
                self._successful_requests = getattr(self, '_successful_requests', 0) + 1
            else:
                self._failed_requests = getattr(self, '_failed_requests', 0) + 1
            
            # Update timing statistics
            self._min_response_time = min(getattr(self, '_min_response_time', float('inf')), response_time)
            self._max_response_time = max(getattr(self, '_max_response_time', 0), response_time)
            
            total_time = getattr(self, '_total_response_time', 0) + response_time
            self._total_response_time = total_time
            self._avg_response_time = total_time / self._total_requests
            self._throughput = self._total_requests / total_time if total_time > 0 else 0
            
            return {
                'response': best_response,
                'agent_responses': agent_responses,
                'agents_used': len(self.agents),
                'response_time': response_time,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {
                'response': f"Error processing message: {str(e)}",
                'success': False
            }
    
    async def shutdown(self):
        """Shutdown the workflow and cleanup resources."""
        if not self.is_initialized:
            return
            
        self.logger.info("Shutting down Parsl chat workflow")
        
        try:
            # Shutdown agents
            for agent in self.agents:
                if hasattr(agent, 'shutdown'):
                    await agent.shutdown()
            
            # Shutdown executor
            if self.executor:
                await self.executor.shutdown()
            
            # Shutdown data units
            for du in self.data_units.values():
                if hasattr(du, 'shutdown'):
                    await du.shutdown()
            
            self.is_initialized = False
            self.logger.info("Parsl chat workflow shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


async def create_parsl_chat_workflow(config_path: str) -> ParslChatWorkflow:
    """
    Create and initialize a Parsl chat workflow.
    
    Args:
        config_path: Path to the workflow configuration file
        
    Returns:
        Initialized ParslChatWorkflow instance
    """
    workflow = ParslChatWorkflow()
    await workflow.initialize(config_path)
    return workflow 