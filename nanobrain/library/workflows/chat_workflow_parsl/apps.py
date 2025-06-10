"""
Parsl Apps for Chat Workflow

This module contains Parsl applications for distributed agent processing.
These functions are designed to be serializable and executable on remote workers.
"""

import os
import sys
from typing import Dict, Any, Optional

# Ensure proper module paths for Parsl serialization
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import parsl
    from parsl.app.app import python_app
    PARSL_AVAILABLE = True
except ImportError:
    PARSL_AVAILABLE = False
    # Create dummy decorator if Parsl not available
    def python_app(func):
        return func


@python_app
def process_message_with_agent(message: str, agent_config: Dict[str, Any]) -> str:
    """
    Process a message using an agent configuration.
    
    This function is designed to be serializable and executable on remote workers.
    It creates an agent instance from the configuration and processes the message.
    
    Args:
        message: The message to process
        agent_config: Agent configuration dictionary
        
    Returns:
        The agent's response to the message
    """
    try:
        # Import required modules inside the function to ensure they're available on workers
        from nanobrain.core.agent import AgentConfig
        from nanobrain.library.agents.conversational import EnhancedCollaborativeAgent
        
        # Create agent configuration
        config = AgentConfig(
            name=agent_config.get('name', 'remote_agent'),
            model=agent_config.get('model', 'gpt-3.5-turbo'),
            temperature=agent_config.get('temperature', 0.7),
            max_tokens=agent_config.get('max_tokens', 1000),
            system_prompt=agent_config.get('system_prompt', 'You are a helpful AI assistant.'),
            auto_initialize=False,
            debug_mode=False,  # Disable debug mode for remote workers
            enable_logging=False,  # Disable logging for remote workers
            log_conversations=False
        )
        
        # Create and initialize agent
        agent = EnhancedCollaborativeAgent(config)
        
        # Process message synchronously (no async in Parsl apps)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Initialize agent
            loop.run_until_complete(agent.initialize())
            
            # Process message
            response = loop.run_until_complete(agent.process(message))
            
            return response
            
        finally:
            loop.close()
            
    except Exception as e:
        return f"Error processing message: {str(e)}"


@python_app
def aggregate_responses(responses: list, message: str) -> Dict[str, Any]:
    """
    Aggregate responses from multiple agents.
    
    Args:
        responses: List of agent responses
        message: Original message
        
    Returns:
        Aggregated response dictionary
    """
    try:
        # Filter successful responses
        successful_responses = [r for r in responses if not r.startswith("Error")]
        
        if not successful_responses:
            return {
                'best_response': "I apologize, but I encountered errors processing your request.",
                'agent_count': len(responses),
                'success_count': 0,
                'success': False
            }
        
        # For now, just return the first successful response
        # In a more sophisticated implementation, you could use voting, ranking, etc.
        best_response = successful_responses[0]
        
        return {
            'best_response': best_response,
            'agent_count': len(responses),
            'success_count': len(successful_responses),
            'all_responses': responses,
            'success': True
        }
        
    except Exception as e:
        return {
            'best_response': f"Error aggregating responses: {str(e)}",
            'success': False
        }


def create_agent_config_dict(agent_config) -> Dict[str, Any]:
    """
    Convert an AgentConfig object to a serializable dictionary.
    
    Args:
        agent_config: AgentConfig instance
        
    Returns:
        Dictionary representation of the agent config
    """
    return {
        'name': agent_config.name,
        'model': agent_config.model,
        'temperature': agent_config.temperature,
        'max_tokens': agent_config.max_tokens,
        'system_prompt': agent_config.system_prompt
    } 