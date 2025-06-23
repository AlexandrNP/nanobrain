"""
Enhanced Chat Workflow

A comprehensive chat workflow implementation using the NanoBrain framework
with proper step interconnections via data units, links, and triggers.

This workflow demonstrates:
- Modular step architecture with individual step directories
- Proper data flow through data units and links
- Event-driven processing with triggers
- Hierarchical step composition with substeps
- Performance monitoring and conversation history management

Architecture:
User Input → CLI Interface Step → Conversation Manager Step → Agent Processing Step → Output
                                        ↓
                                   History Persistence (substep)
                                   Performance Tracking (substep)
"""

import sys
import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

# Core framework imports with proper nanobrain package structure
from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig, DataUnitType
from nanobrain.core.trigger import DataUpdatedTrigger, TriggerConfig, TriggerType
from nanobrain.core.link import DirectLink, LinkConfig, LinkType
from nanobrain.core.executor import LocalExecutor, ExecutorConfig
from nanobrain.core.logging_system import get_logger, OperationType
from nanobrain.core.agent import AgentConfig

# Library imports with updated paths
from nanobrain.library.agents.conversational import EnhancedCollaborativeAgent
from nanobrain.library.infrastructure.data import ConversationHistoryUnit


class ChatWorkflow(FromConfigBase):
    """
    Enhanced chat workflow with modular step architecture.
    
    This workflow provides:
    - Modular step-based architecture
    - Proper data flow through data units and links
    - Event-driven processing with triggers
    - Conversation history management
    - Performance monitoring and metrics
    """
    
    # Component configuration
    COMPONENT_TYPE = "chat_workflow"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'model': 'gpt-3.5-turbo',
        'temperature': 0.7,
        'max_tokens': 2000,
        'system_prompt': 'You are a helpful and friendly AI assistant.',
        'enable_metrics': True
    }
    
    @classmethod
    def extract_component_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ChatWorkflow configuration"""
        return {
            'name': config.get('name', 'chat_workflow'),
            'model': config.get('model', 'gpt-3.5-turbo'),
            'temperature': config.get('temperature', 0.7),
            'max_tokens': config.get('max_tokens', 2000),
            'system_prompt': config.get('system_prompt', 'You are a helpful and friendly AI assistant.'),
            'enable_metrics': config.get('enable_metrics', True),
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve ChatWorkflow dependencies"""
        # Create executor via from_config to avoid direct instantiation
        executor_config = ExecutorConfig(executor_type="local", max_workers=2)
        executor = LocalExecutor.from_config(executor_config)
        
        return {
            'executor': executor,
            'agent_config': component_config,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> 'ChatWorkflow':
        """Mandatory from_config implementation for ChatWorkflow"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Extract component-specific configuration  
        component_config = cls.extract_component_config(config)
        
        # Step 2: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 3: Create instance
        instance = cls.create_instance(config, component_config, dependencies)
        
        # Step 4: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
        
    def _init_from_config(self, config: Dict[str, Any], component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ChatWorkflow with resolved dependencies"""
        self.logger = get_logger("chat_workflow", "workflows")
        
        # Workflow state
        self.is_initialized = False
        self.is_running = False
        
        # Core components from dependencies
        self.executor = dependencies.get('executor')
        self.agent = None
        self.agent_config = dependencies.get('agent_config')
        
        # Data units
        self.data_units = {}
        
        # Conversation management
        self.conversation_history = None
        self.current_conversation_id = None
        
    async def initialize(self) -> None:
        """Initialize all workflow components with proper interconnections."""
        if self.is_initialized:
            return
            
        self.logger.info("Initializing chat workflow")
        
        try:
            # Initialize executor (already created via from_config)
            if self.executor:
                await self.executor.initialize()
            
            # Initialize data units
            await self._setup_data_units()
            
            # Initialize agent
            await self._setup_agent()
            
            self.is_initialized = True
            self.logger.info("Chat workflow initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize chat workflow: {e}")
            raise
    
    async def _setup_executor(self) -> None:
        """Setup the workflow executor (deprecated - now handled via from_config)."""
        self.logger.info("Executor setup handled via from_config dependencies")
        
    async def _setup_data_units(self) -> None:
        """Setup data units for the workflow."""
        self.logger.info("Setting up data units")
        
        # User input data unit
        self.data_units['user_input'] = DataUnitMemory(
            DataUnitConfig(
                name="user_input",
                data_type="memory",
                description="User input messages"
            )
        )
        
        # Agent output data unit
        self.data_units['agent_output'] = DataUnitMemory(
            DataUnitConfig(
                name="agent_output",
                data_type="memory",
                description="Agent response output"
            )
        )
        
        # Conversation history data unit
        self.conversation_history = ConversationHistoryUnit(
            config={'db_path': 'chat_workflow_history.db'}
        )
        self.data_units['conversation_history'] = self.conversation_history
        
        # Initialize all data units
        for name, data_unit in self.data_units.items():
            await data_unit.initialize()
            self.logger.info(f"Initialized data unit: {name}")
    
    async def _setup_agent(self) -> None:
        """Setup the enhanced collaborative agent."""
        self.logger.info("Setting up enhanced collaborative agent")
        
        agent_config = AgentConfig(
            name=self.agent_config.get('name', 'chat_assistant'),
            model=self.agent_config.get('model', 'gpt-3.5-turbo'),
            temperature=self.agent_config.get('temperature', 0.7),
            max_tokens=self.agent_config.get('max_tokens', 2000),
            system_prompt=self.agent_config.get('system_prompt', 'You are a helpful and friendly AI assistant.'),
            auto_initialize=False,
            debug_mode=True,
            enable_logging=True,
            log_conversations=True
        )
        
        # Use mandatory from_config pattern for agent creation
        self.agent = EnhancedCollaborativeAgent.from_config(
            agent_config,
            enable_metrics=self.agent_config.get('enable_metrics', True)
        )
        
        await self.agent.initialize()
    
    async def process_user_input(self, user_input: str) -> str:
        """
        Process user input through the workflow.
        
        Args:
            user_input: User's input message
            
        Returns:
            str: Agent's response
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Process through agent
            response = await self.agent.process(user_input)
            
            # Store in data units
            await self.data_units['user_input'].set(user_input)
            await self.data_units['agent_output'].set(response)
            
            return response
                
        except Exception as e:
            self.logger.error(f"Error processing user input: {e}")
            return f"Error: {e}"
    
    async def shutdown(self) -> None:
        """Shutdown the workflow and cleanup resources."""
        if not self.is_initialized:
            return
        
        self.logger.info("Shutting down chat workflow")
        
        try:
            # Shutdown agent
            if self.agent:
                await self.agent.shutdown()
            
            # Shutdown data units
            for data_unit in self.data_units.values():
                if hasattr(data_unit, 'shutdown'):
                    await data_unit.shutdown()
            
            # Shutdown executor
            if self.executor:
                await self.executor.shutdown()
            
            self.is_initialized = False
            self.logger.info("Chat workflow shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get comprehensive workflow status.
        
        Returns:
            Dict[str, Any]: Workflow status information
        """
        return {
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'current_conversation_id': self.current_conversation_id,
            'components': {
                'data_units': len(self.data_units),
                'agent': self.agent is not None,
                'executor': self.executor is not None
            },
            'agent_status': self.agent.get_enhanced_status() if self.agent else None,
            'conversation_stats': "Available (call get_conversation_stats() for details)"
        }
    
    async def get_conversation_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get conversation statistics asynchronously.
        
        Returns:
            Optional[Dict[str, Any]]: Conversation statistics or None
        """
        if self.conversation_history:
            return await self.conversation_history.get_statistics()
        return None


# Factory function for easy workflow creation
async def create_chat_workflow(config: Optional[Dict[str, Any]] = None) -> ChatWorkflow:
    """Create and initialize a chat workflow using mandatory from_config pattern."""
    default_config = {
        'name': 'chat_workflow',
        'model': 'gpt-3.5-turbo',
        'temperature': 0.7,
        'max_tokens': 2000,
        'system_prompt': 'You are a helpful and friendly AI assistant.',
        'enable_metrics': True
    }
    
    # Merge with provided config
    if config:
        default_config.update(config)
    
    # Use mandatory from_config pattern
    workflow = ChatWorkflow.from_config(default_config)
    await workflow.initialize()
    return workflow


# Main execution for testing
async def main():
    """Main function for testing the workflow."""
    workflow = ChatWorkflow()
    
    try:
        await workflow.initialize()
        
        # Test basic functionality
        response = await workflow.process_user_input("Hello, how are you?")
        print(f"Response: {response}")
        
        # Show status
        status = workflow.get_workflow_status()
        print(f"Workflow status: {status}")
        
    finally:
        await workflow.shutdown()


if __name__ == "__main__":
    asyncio.run(main()) 