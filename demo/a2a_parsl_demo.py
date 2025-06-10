#!/usr/bin/env python3
"""
A2A (Agent-to-Agent) Protocol Demo with Parsl Integration

This demo shows how to integrate A2A protocol support with NanoBrain's Parsl workflow,
enabling multi-agent collaboration across different specialized agents.

Key Features:
- A2A protocol integration with NanoBrain agents
- Multi-agent collaboration for complex tasks
- Parsl-based parallel execution
- YAML configuration for A2A agents
- Compatibility with existing MCP tools
- Mock agents for testing without real A2A servers

Usage:
    python a2a_parsl_demo.py [--test-setup] [--config CONFIG_FILE]
"""

import asyncio
import argparse
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the nanobrain src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nanobrain.core.agent import ConversationalAgent, AgentConfig
from nanobrain.core.a2a_support import (
    A2ASupportMixin, with_a2a_support, A2AClient, A2AAgentConfig,
    A2AMessage, A2APart, PartType, create_a2a_agent_config
)
from nanobrain.core.mcp_support import MCPSupportMixin
from nanobrain.core.logging_system import get_logger
from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig
from nanobrain.core.step import Step, StepConfig
from nanobrain.core.trigger import DataUpdatedTrigger, TriggerConfig
from nanobrain.core.link import DirectLink, LinkConfig

# Try to import Parsl components
try:
    from demo.chat_workflow_parsl_demo import ParslLogManager, LoadBalancedCLIInterface
    PARSL_AVAILABLE = True
except ImportError:
    PARSL_AVAILABLE = False
    print("⚠️  Parsl components not available, using simplified demo")


class A2AEnabledConversationalAgent(A2ASupportMixin, MCPSupportMixin, ConversationalAgent):
    """
    Conversational agent with both A2A and MCP support.
    
    This agent can:
    - Use MCP tools for structured operations
    - Collaborate with A2A agents for complex tasks
    - Delegate work to specialized agents
    - Integrate results from multiple sources
    """
    
    def __init__(self, config: AgentConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.collaboration_count = 0
        self.delegation_rules = kwargs.get('delegation_rules', [])
    
    async def process(self, input_text: str, **kwargs) -> str:
        """Enhanced process method with A2A delegation capabilities."""
        # Check if we should delegate to an A2A agent
        if self.a2a_enabled and self.a2a_agents:
            delegation_result = await self._check_for_delegation(input_text)
            if delegation_result:
                return delegation_result
        
        # Fall back to normal processing
        return await super().process(input_text, **kwargs)
    
    async def _check_for_delegation(self, input_text: str) -> Optional[str]:
        """Check if the input should be delegated to an A2A agent."""
        input_lower = input_text.lower()
        
        # Check delegation rules
        for rule in self.delegation_rules:
            keywords = rule.get('keywords', [])
            agent_name = rule.get('agent')
            
            if any(keyword in input_lower for keyword in keywords):
                if agent_name in self.a2a_agents:
                    try:
                        self.collaboration_count += 1
                        
                        # Log delegation
                        self.nb_logger.info(f"Delegating to A2A agent: {agent_name}",
                                          rule_description=rule.get('description', ''),
                                          collaboration_count=self.collaboration_count)
                        
                        # Call A2A agent
                        result = await self.call_a2a_agent(agent_name, input_text)
                        
                        # Wrap result with context
                        return f"🤝 Collaborated with {agent_name}:\n\n{result}"
                        
                    except Exception as e:
                        self.nb_logger.error(f"A2A delegation failed: {e}")
                        # Continue with normal processing
                        break
        
        return None


class A2ACollaborativeStep(Step):
    """
    Step that can collaborate with A2A agents for enhanced processing.
    """
    
    def __init__(self, config: StepConfig, agent: A2AEnabledConversationalAgent, log_manager):
        super().__init__(config)
        self.agent = agent
        self.log_manager = log_manager
        self.step_logger = log_manager.get_logger("a2a_collaborative_step", "steps")
        self.collaboration_count = 0
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs with potential A2A collaboration."""
        # Extract user input
        user_input_data = inputs.get('user_input', '')
        
        if isinstance(user_input_data, dict):
            user_input = user_input_data.get('user_input', '')
        else:
            user_input = user_input_data
        
        if not isinstance(user_input, str):
            user_input = str(user_input) if user_input else ''
        
        if not user_input or user_input.strip() == '':
            self.step_logger.warning("Empty user input received")
            return {'agent_response': ''}
        
        self.collaboration_count += 1
        
        self.step_logger.info(f"Processing with A2A collaboration #{self.collaboration_count}",
                             user_input_preview=user_input[:100] + "..." if len(user_input) > 100 else user_input,
                             a2a_enabled=self.agent.a2a_enabled,
                             available_agents=len(self.agent.a2a_agents))
        
        try:
            start_time = time.time()
            
            # Process through A2A-enabled agent
            response = await self.agent.process(user_input)
            
            processing_time = (time.time() - start_time) * 1000
            
            self.step_logger.info(f"Completed A2A collaboration #{self.collaboration_count}",
                                 processing_time_ms=processing_time,
                                 response_length=len(response) if response else 0,
                                 agent_collaborations=self.agent.collaboration_count)
            
            return {'agent_response': response or 'I apologize, but I could not generate a response.'}
            
        except Exception as e:
            self.step_logger.error(f"Error in A2A collaborative processing: {e}",
                                  collaboration_id=self.collaboration_count)
            return {'agent_response': f'Sorry, I encountered an error during collaboration: {str(e)}'}


class A2AParslWorkflow:
    """
    Workflow that demonstrates A2A protocol integration with Parsl.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/agent_with_a2a.yaml"
        self.components = {}
        self.main_logger = get_logger("a2a_parsl_workflow")
        
        # Initialize log manager
        if PARSL_AVAILABLE:
            self.log_manager = ParslLogManager()
        else:
            # Simple log manager for demo
            self.log_manager = SimpleLogManager()
    
    async def setup(self):
        """Set up the A2A-enabled workflow."""
        self.main_logger.info("Setting up A2A Parsl Workflow")
        
        # 1. Create data units
        print("   Creating data units...")
        
        # User input data unit
        user_input_config = DataUnitConfig(
            name="user_input",
            data_type="memory",
            persistent=False,
            cache_size=100
        )
        self.components['user_input_du'] = DataUnitMemory(user_input_config)
        
        # Agent input data unit
        agent_input_config = DataUnitConfig(
            name="agent_input",
            data_type="memory",
            persistent=False,
            cache_size=100
        )
        self.components['agent_input_du'] = DataUnitMemory(agent_input_config)
        
        # Agent output data unit
        agent_output_config = DataUnitConfig(
            name="agent_output",
            data_type="memory",
            persistent=False,
            cache_size=100
        )
        self.components['agent_output_du'] = DataUnitMemory(agent_output_config)
        
        # 2. Create A2A-enabled agent
        print("   Creating A2A-enabled conversational agent...")
        
        agent_config = AgentConfig(
            name="A2ACollaborativeAgent",
            description="Agent with A2A protocol support for multi-agent collaboration",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1500,
            system_prompt="""You are a collaborative AI assistant with access to specialized agents through the A2A protocol.

You can work with other agents to accomplish complex tasks:
- Travel agents for trip planning and booking  
- Code agents for programming tasks
- Data agents for analysis and visualization
- Other specialized agents as needed

When a user request requires specialized expertise:
1. Identify which type of agent would be most helpful
2. Delegate the appropriate subtask to that agent
3. Integrate the results into your response
4. Provide a comprehensive answer to the user

Always be transparent about when you're collaborating with other agents.""",
            auto_initialize=False,
            debug_mode=True,
            enable_logging=True,
            log_conversations=True
        )
        
        # Create delegation rules
        delegation_rules = [
            {
                'keywords': ['flight', 'hotel', 'travel', 'trip', 'vacation'],
                'agent': 'travel_agent',
                'description': 'Delegate travel-related requests'
            },
            {
                'keywords': ['code', 'program', 'function', 'algorithm', 'debug'],
                'agent': 'code_agent', 
                'description': 'Delegate programming tasks'
            },
            {
                'keywords': ['data', 'analyze', 'chart', 'graph', 'statistics'],
                'agent': 'data_agent',
                'description': 'Delegate data analysis tasks'
            }
        ]
        
        # Create A2A-enabled agent
        agent = A2AEnabledConversationalAgent(
            agent_config,
            a2a_enabled=True,
            a2a_config_path="config/a2a_config.yaml",
            mcp_enabled=True,
            mcp_config_path="config/mcp_config.yaml",
            delegation_rules=delegation_rules
        )
        
        # Initialize agent
        await agent.initialize()
        
        self.components['agent'] = agent
        
        # 3. Create collaborative step
        print("   Creating A2A collaborative step...")
        
        step_config = StepConfig(
            name="a2a_collaborative_step",
            description="Step with A2A collaboration capabilities",
            debug_mode=True
        )
        
        self.components['collaborative_step'] = A2ACollaborativeStep(
            step_config,
            agent,
            self.log_manager
        )
        
        # Set up step with data units
        self.components['collaborative_step'].register_input_data_unit(
            'user_input',
            self.components['agent_input_du']
        )
        self.components['collaborative_step'].register_output_data_unit(
            self.components['agent_output_du']
        )
        
        await self.components['collaborative_step'].initialize()
        
        # 4. Create triggers
        print("   Creating triggers...")
        
        # User input trigger
        user_trigger_config = TriggerConfig(
            name="user_input_trigger",
            trigger_type="data_updated"
        )
        self.components['user_trigger'] = DataUpdatedTrigger(
            [self.components['user_input_du']],
            user_trigger_config
        )
        
        # Agent input trigger
        agent_trigger_config = TriggerConfig(
            name="agent_input_trigger", 
            trigger_type="data_updated"
        )
        self.components['agent_trigger'] = DataUpdatedTrigger(
            [self.components['agent_input_du']],
            agent_trigger_config
        )
        
        # 5. Create links
        print("   Creating data flow links...")
        
        # User to agent input link
        user_to_agent_config = LinkConfig(
            name="user_to_agent_link",
            link_type="direct"
        )
        self.components['user_to_agent_link'] = DirectLink(
            self.components['user_input_du'],
            self.components['agent_input_du'],
            user_to_agent_config
        )
        
        # Agent input to step link
        agent_to_step_config = LinkConfig(
            name="agent_to_step_link",
            link_type="direct"
        )
        self.components['agent_to_step_link'] = DirectLink(
            self.components['agent_input_du'],
            self.components['collaborative_step'],
            agent_to_step_config
        )
        
        # Step to output link
        step_to_output_config = LinkConfig(
            name="step_to_output_link",
            link_type="direct"
        )
        self.components['step_to_output_link'] = DirectLink(
            self.components['collaborative_step'],
            self.components['agent_output_du'],
            step_to_output_config
        )
        
        # 6. Set up trigger callbacks
        print("   Setting up trigger callbacks...")
        
        async def user_input_callback(data_units, trigger):
            await self.components['user_to_agent_link'].activate()
        
        async def agent_input_callback(data_units, trigger):
            await self.components['agent_to_step_link'].activate()
            await self.components['step_to_output_link'].activate()
        
        self.components['user_trigger'].set_callback(user_input_callback)
        self.components['agent_trigger'].set_callback(agent_input_callback)
        
        # 7. Create CLI interface
        print("   Creating CLI interface...")
        
        if PARSL_AVAILABLE:
            self.components['cli'] = LoadBalancedCLIInterface(
                self.components['user_input_du'],
                self.components['agent_output_du'],
                self.log_manager
            )
        else:
            self.components['cli'] = SimpleCLIInterface(
                self.components['user_input_du'],
                self.components['agent_output_du']
            )
        
        print("✅ A2A Parsl Workflow setup complete!")
        
        # Display A2A status
        await self._display_a2a_status()
    
    async def _display_a2a_status(self):
        """Display A2A configuration status."""
        agent = self.components.get('agent')
        if agent and hasattr(agent, 'get_a2a_status'):
            status = agent.get_a2a_status()
            
            print(f"\n📡 A2A Protocol Status:")
            print(f"   Enabled: {status['enabled']}")
            print(f"   Client Initialized: {status['client_initialized']}")
            print(f"   Available Agents: {status['total_agents']}")
            
            if status['agents']:
                print(f"   Configured Agents:")
                for name, info in status['agents'].items():
                    print(f"     - {name}: {info['description']}")
                    print(f"       Skills: {info['skills_count']}")
                    print(f"       Streaming: {info['capabilities']['streaming']}")
    
    async def run(self):
        """Run the A2A workflow."""
        self.main_logger.info("Starting A2A Parsl Workflow")
        
        try:
            # Start CLI interface
            await self.components['cli'].start()
            
        except KeyboardInterrupt:
            self.main_logger.info("Workflow interrupted by user")
        except Exception as e:
            self.main_logger.error(f"Workflow error: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the workflow."""
        self.main_logger.info("Shutting down A2A Parsl Workflow")
        
        # Shutdown CLI
        if 'cli' in self.components:
            await self.components['cli'].shutdown()
        
        # Shutdown agent
        if 'agent' in self.components:
            await self.components['agent'].shutdown()
        
        # Shutdown log manager
        if hasattr(self.log_manager, 'shutdown'):
            await self.log_manager.shutdown()


class SimpleLogManager:
    """Simple log manager for when Parsl is not available."""
    
    def __init__(self):
        self.should_log_to_console = True
    
    def get_logger(self, name: str, category: str = "general"):
        return get_logger(f"{category}.{name}")
    
    async def shutdown(self):
        pass


class SimpleCLIInterface:
    """Simple CLI interface for when Parsl CLI is not available."""
    
    def __init__(self, input_du, output_du):
        self.input_du = input_du
        self.output_du = output_du
        self.running = False
    
    async def start(self):
        """Start the simple CLI interface."""
        self.running = True
        print("\n🤖 A2A Demo CLI (Simplified)")
        print("Type 'quit' to exit")
        print("Try: 'book a flight to Paris' or 'write a Python function'")
        
        while self.running:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input:
                    # Send input
                    await self.input_du.write({'user_input': user_input})
                    
                    # Wait a bit for processing
                    await asyncio.sleep(1.0)
                    
                    # Get response
                    response_data = await self.output_du.read()
                    if response_data and 'agent_response' in response_data:
                        print(f"\n🤖 Bot: {response_data['agent_response']}")
                    else:
                        print("\n🤖 Bot: No response received")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError: {e}")
        
        self.running = False
    
    async def shutdown(self):
        """Shutdown the CLI interface."""
        self.running = False


async def test_a2a_setup():
    """Test A2A setup without running full workflow."""
    print("🧪 Testing A2A Setup...")
    
    try:
        # Test A2A client creation
        from nanobrain.core.a2a_support import A2AClient, A2AClientConfig
        
        client_config = A2AClientConfig()
        client = A2AClient(client_config)
        
        # Test agent config creation
        agent_config = create_a2a_agent_config(
            name="test_agent",
            url="http://localhost:8080/a2a",
            description="Test A2A agent"
        )
        
        client.add_agent(agent_config)
        
        await client.initialize()
        
        print("✅ A2A client creation: OK")
        print("✅ Agent configuration: OK")
        print("✅ Client initialization: OK")
        
        # Test mixin
        @with_a2a_support
        class TestAgent(ConversationalAgent):
            pass
        
        config = AgentConfig(name="test", description="Test agent")
        agent = TestAgent(config, a2a_enabled=True)
        
        print("✅ A2A mixin/decorator: OK")
        
        await client.shutdown()
        
        print("\n🎉 A2A setup test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ A2A setup test failed: {e}")
        raise


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="A2A Protocol Demo with Parsl Integration")
    parser.add_argument("--test-setup", action="store_true", help="Test A2A setup only")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    if args.test_setup:
        await test_a2a_setup()
        return
    
    print("🚀 A2A Protocol Demo")
    print("This demo shows A2A integration with NanoBrain agents")
    print("Full workflow demo coming soon...")


if __name__ == "__main__":
    asyncio.run(main()) 