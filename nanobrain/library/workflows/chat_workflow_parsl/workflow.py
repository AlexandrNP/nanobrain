"""
Parsl Chat Workflow

A simple chat workflow implementation using existing NanoBrain ParslAgent
for distributed execution. Follows NanoBrain architecture and reuses
existing components.
"""

import sys
import os
import asyncio
import yaml
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# Core framework imports
from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig
from nanobrain.core.executor import ParslExecutor, ExecutorConfig, ExecutorType
from nanobrain.core.logging_system import get_logger
from nanobrain.core.agent import AgentConfig
from nanobrain.library.agents.specialized.parsl_agent import ParslAgent

# Step imports
from .steps.distributed_processing_step.distributed_processing_step import DistributedProcessingStep
from nanobrain.core.config import get_config_manager

# Parsl imports with fallback
try:
    import parsl
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


class ParslChatWorkflow(FromConfigBase):
    """
    Simple chat workflow using existing NanoBrain ParslAgent for distributed processing.
    Now follows unified from_config pattern.
    """
    
    COMPONENT_TYPE = "parsl_chat_workflow"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': 'Parsl-based distributed chat workflow',
        'parsl_config': {},
        'max_workers': 4
    }
    
    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return generic Dict - ParslChatWorkflow uses dictionary configuration"""
        return dict
    
    def _init_from_config(self, config: Dict[str, Any], component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ParslChatWorkflow with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        
        self.config = None
        self.data_units = {}
        self.agents = []
        self.parsl_executor = None
        self.conversation_history = None
        self.logger = get_logger("parsl_chat_workflow", category="workflows")
        self._parsl_globally_configured = False
        
    async def initialize(self, config_path: str):
        """Initialize the workflow with configuration."""
        self.logger.info(f"Initializing ParslChatWorkflow from {config_path}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup components
        await self._setup_parsl_executor()
        await self._setup_data_units()
        await self._setup_agents()
        await self._setup_conversation_history()
        
        self.logger.info("ParslChatWorkflow initialization complete")
    
    async def _setup_parsl_executor(self):
        """Setup Parsl executor using existing NanoBrain ParslExecutor."""
        try:
            if not PARSL_AVAILABLE:
                self.logger.warning("Parsl not available, workflow will use local processing")
                return
            
            # Create ParslExecutor using existing NanoBrain component
            executor_config = ExecutorConfig(
                type=ExecutorType.PARSL,
                max_workers=self.config.get('config', {}).get('executor', {}).get('max_workers', 4),
                timeout=self.config.get('config', {}).get('executor', {}).get('timeout', 15.0)
            )
            
            self.parsl_executor = ParslExecutor(executor_config)
            await self.parsl_executor.initialize()
            
            # Configure Parsl logging to use NanoBrain semantic directory structure
            from nanobrain.core.logging_system import _configure_comprehensive_parsl_logging
            _configure_comprehensive_parsl_logging()
            
            self._parsl_globally_configured = True
            self.logger.info("ParslExecutor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Parsl executor: {e}")
            self.parsl_executor = None
    
    async def _setup_data_units(self):
        """Setup data units for the workflow."""
        data_unit_configs = self.config.get('config', {}).get('data_units', {})
        
        for name, config in data_unit_configs.items():
            if config.get('type') == 'memory':
                data_unit_config = DataUnitConfig(
                    name=name,
                    **{"class": "nanobrain.core.data_unit.DataUnitMemory"},
                    description=config.get('description', ''),
                    cache_size=config.get('cache_size', 100)
                )
                self.data_units[name] = DataUnitMemory.from_config(data_unit_config)
                await self.data_units[name].initialize()
        
        self.logger.info(f"Initialized {len(self.data_units)} data units")
    
    async def _setup_agents(self):
        """Setup ParslAgent instances using existing NanoBrain component."""
        agent_configs = self.config.get('config', {}).get('agents', [])
        
        for agent_config in agent_configs:
            # Create agent configuration
            config = AgentConfig(
                name=agent_config['name'],
                model='gpt-3.5-turbo',  # Default model
                temperature=0.7,
                max_tokens=1000,
                system_prompt='You are a helpful AI assistant.',
                auto_initialize=True
            )
            
            # Create ParslAgent using existing component
            agent = ParslAgent(config, parsl_executor=self.parsl_executor)
            
            # Set up NanoBrain logging for the agent
            agent.nb_logger = get_logger(f"parsl_agent_{config.name}", category="agents")
            
            await agent.initialize()
            
            self.agents.append(agent)
        
        self.logger.info(f"Initialized {len(self.agents)} ParslAgent instances")
    
    async def _setup_distributed_step(self):
        """Setup the distributed processing step."""
        self.logger.info("Setting up distributed processing step")
        
        step_config = {
            'name': 'distributed_processing_step',
            'description': 'Handles distributed processing of multiple messages'
        }
        
        self.distributed_step = DistributedProcessingStep(step_config)
        await self.distributed_step.initialize(self.agents, self.parsl_executor)
        
        self.logger.info("Distributed processing step initialized")
    
    async def _setup_conversation_history(self):
        """Setup conversation history."""
        self.conversation_history = ConversationHistoryUnit({
            'max_messages_per_conversation': 1000
        })
        self.logger.info("Conversation history initialized")
    
    async def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a single message using the first available agent.
        For distributed processing of multiple messages, use process_messages_distributed.
        
        Args:
            message: The message to process
            
        Returns:
            Dict containing the response and metadata
        """
        self.logger.info(f"Processing single message with first available agent")
        
        if not self.agents:
            return {
                'best_response': "No agents available for processing.",
                'success': False,
                'agent_count': 0
            }
        
        # Use the first agent for single message processing
        agent = self.agents[0]
        
        try:
            # Log agent processing start
            if hasattr(agent, 'nb_logger'):
                agent.nb_logger.info(f"Agent {agent.config.name} starting message processing", 
                                   input_message=message[:100] + "..." if len(message) > 100 else message)
            
            response = await agent.process(message)
            
            # Log agent success
            if hasattr(agent, 'nb_logger'):
                agent.nb_logger.info(f"Agent {agent.config.name} completed message processing", 
                                   response_length=len(response),
                                   success=True)
            
            # Add to conversation history
            await self.conversation_history.add_message("user", message)
            await self.conversation_history.add_message("assistant", response)
            
            return {
                'best_response': response,
                'response': response,
                'success': True,
                'agent_count': 1,
                'total_agents': len(self.agents),
                'method': 'single_agent_processing'
            }
            
        except Exception as e:
            self.logger.error(f"Error in message processing: {e}")
            if hasattr(agent, 'nb_logger'):
                agent.nb_logger.error(f"Agent {agent.config.name} failed processing: {e}")
            
            return {
                'best_response': f"I encountered an error processing your request: {str(e)}",
                'success': False,
                'agent_count': 0,
                'error': str(e)
            }
    
    async def process_messages_distributed(self, messages: List[str]) -> Dict[str, Any]:
        """
        Process multiple messages using the DistributedProcessingStep.
        
        Args:
            messages: List of messages to process
            
        Returns:
            Dict containing all responses and metadata
        """
        self.logger.info(f"Processing {len(messages)} messages using distributed processing step")
        
        # Use the distributed processing step
        if not hasattr(self, 'distributed_step'):
            await self._setup_distributed_step()
            
        return await self.distributed_step.execute(messages)
    
    async def process_user_input(self, message: str) -> str:
        """Process user input and return response."""
        result = await self.process_message(message)
        return result.get('best_response', 'No response available')
    
    async def submit_message(self, message: str):
        """Submit a message for asynchronous processing."""
        return asyncio.create_task(self.process_user_input(message))
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            'name': 'ParslChatWorkflow',
            'initialized': True,
            'agent_count': len(self.agents),
            'parsl_executor': self.parsl_executor is not None,
            'parsl_available': PARSL_AVAILABLE,
            'data_units': list(self.data_units.keys())
        }
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the workflow."""
        stats = {
            'workflow_name': 'ParslChatWorkflow',
            'agent_count': len(self.agents),
            'data_units_count': len(self.data_units),
            'parsl_executor_available': self.parsl_executor is not None,
            'parsl_available': PARSL_AVAILABLE
        }
        
        # Get agent performance stats if available
        agent_stats = []
        for agent in self.agents:
            if hasattr(agent, 'get_performance_stats'):
                try:
                    agent_stat = agent.get_performance_stats()
                    agent_stats.append({
                        'agent_name': agent.config.name,
                        'stats': agent_stat
                    })
                except Exception as e:
                    agent_stats.append({
                        'agent_name': agent.config.name,
                        'error': str(e)
                    })
        
        stats['agent_stats'] = agent_stats
        
        # Get distributed step stats if available
        if hasattr(self, 'distributed_step') and hasattr(self.distributed_step, 'get_performance_stats'):
            try:
                stats['distributed_step_stats'] = self.distributed_step.get_performance_stats()
            except Exception as e:
                stats['distributed_step_error'] = str(e)
        
        return stats
    
    async def get_parsl_stats(self) -> Dict[str, Any]:
        """Get Parsl-specific statistics."""
        stats = {
            'parsl_available': PARSL_AVAILABLE,
            'parsl_executor_initialized': self.parsl_executor is not None
        }
        
        if self.parsl_executor:
            try:
                # Get basic executor info
                stats['executor_type'] = type(self.parsl_executor).__name__
                
                # Try to get Parsl-specific stats if available
                if hasattr(self.parsl_executor, 'get_stats'):
                    stats['executor_stats'] = self.parsl_executor.get_stats()
                
                # Check if we can get DFK stats
                try:
                    import parsl
                    if hasattr(parsl, 'dfk') and parsl.dfk:
                        dfk = parsl.dfk
                        stats['dfk_stats'] = {
                            'task_count': len(dfk.tasks) if hasattr(dfk, 'tasks') else 0,
                            'executors': list(dfk.executors.keys()) if hasattr(dfk, 'executors') else []
                        }
                except Exception as e:
                    stats['dfk_error'] = str(e)
                    
            except Exception as e:
                stats['executor_error'] = str(e)
        else:
            stats['executor_error'] = 'Parsl executor not initialized'
        
        return stats
    
    async def shutdown(self):
        """Shutdown the workflow and cleanup resources."""
        self.logger.info("Shutting down ParslChatWorkflow")
        
        # Shutdown agents
        for agent in self.agents:
            if hasattr(agent, 'shutdown'):
                await agent.shutdown()
        
        # Shutdown executor
        if self.parsl_executor:
            await self.parsl_executor.shutdown()
        
        self.logger.info("ParslChatWorkflow shutdown complete")


async def create_parsl_chat_workflow(config_path: str) -> ParslChatWorkflow:
    """
    Factory function to create and initialize a ParslChatWorkflow.
    
    Args:
        config_path: Path to the workflow configuration file
        
    Returns:
        Initialized ParslChatWorkflow instance
    """
    workflow = ParslChatWorkflow()
    await workflow.initialize(config_path)
    return workflow 