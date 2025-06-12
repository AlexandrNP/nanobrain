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

# Add the nanobrain package to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nanobrain.core.agent import ConversationalAgent, AgentConfig
from nanobrain.core.a2a_support import (
    A2ASupportMixin, with_a2a_support, A2AClient, A2AAgentConfig,
    A2AMessage, A2APart, PartType, create_a2a_agent_config
)
from nanobrain.core.logging_system import get_logger
from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig

# Try to import MCP support
try:
    from nanobrain.core.mcp_support import MCPSupportMixin
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Create dummy mixin if MCP not available
    class MCPSupportMixin:
        pass

# Try to import Step components
try:
    from nanobrain.core.step import Step, StepConfig
    from nanobrain.core.trigger import DataUpdatedTrigger, TriggerConfig
    from nanobrain.core.link import DirectLink, LinkConfig
    STEP_COMPONENTS_AVAILABLE = True
except ImportError:
    STEP_COMPONENTS_AVAILABLE = False
    print("⚠️  Step components not available, using simplified demo")


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
                        self.logger.info(f"Delegating to A2A agent: {agent_name}")
                        
                        # Call A2A agent
                        result = await self.call_a2a_agent(agent_name, input_text)
                        
                        # Wrap result with context
                        return f"🤝 Collaborated with {agent_name}:\n\n{result}"
                        
                    except Exception as e:
                        self.logger.error(f"A2A delegation failed: {e}")
                        # Continue with normal processing
                        break
        
        return None


class A2ACollaborativeStep:
    """
    Step that can collaborate with A2A agents for enhanced processing.
    """
    
    def __init__(self, agent: A2AEnabledConversationalAgent):
        self.agent = agent
        self.logger = get_logger("a2a_collaborative_step")
        self.collaboration_count = 0
    
    async def process(self, user_input: str) -> str:
        """Process inputs with potential A2A collaboration."""
        if not user_input or user_input.strip() == '':
            self.logger.warning("Empty user input received")
            return ''
        
        self.collaboration_count += 1
        
        self.logger.info(f"Processing with A2A collaboration #{self.collaboration_count}")
        
        try:
            start_time = time.time()
            
            # Process through A2A-enabled agent
            response = await self.agent.process(user_input)
            
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.info(f"Completed A2A collaboration #{self.collaboration_count} in {processing_time:.2f}ms")
            
            return response or 'I apologize, but I could not generate a response.'
            
        except Exception as e:
            self.logger.error(f"Error in A2A collaborative processing: {e}")
            return f'Sorry, I encountered an error during collaboration: {str(e)}'


class A2AParslWorkflow:
    """
    Workflow that demonstrates A2A protocol integration with Parsl.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/a2a_config.yaml"
        self.components = {}
        self.logger = get_logger("a2a_parsl_workflow")
        self.running = False
    
    async def setup(self):
        """Set up the A2A-enabled workflow."""
        self.logger.info("Setting up A2A Parsl Workflow")
        
        print("🚀 Setting up A2A Protocol Demo")
        print("=" * 50)
        
        # 1. Create A2A-enabled agent
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
                'keywords': ['flight', 'hotel', 'travel', 'trip', 'vacation', 'book'],
                'agent': 'travel_agent',
                'description': 'Delegate travel-related requests'
            },
            {
                'keywords': ['code', 'program', 'function', 'algorithm', 'debug', 'python', 'javascript'],
                'agent': 'code_agent', 
                'description': 'Delegate programming tasks'
            },
            {
                'keywords': ['data', 'analyze', 'chart', 'graph', 'statistics', 'csv'],
                'agent': 'data_agent',
                'description': 'Delegate data analysis tasks'
            }
        ]
        
        # Create A2A-enabled agent
        agent = A2AEnabledConversationalAgent(
            agent_config,
            a2a_enabled=True,
            a2a_config_path=self.config_path,
            delegation_rules=delegation_rules
        )
        
        # Initialize agent
        await agent.initialize()
        
        self.components['agent'] = agent
        
        # 2. Create collaborative step
        print("   Creating A2A collaborative step...")
        
        self.components['collaborative_step'] = A2ACollaborativeStep(agent)
        
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
                    if 'skills_count' in info:
                        print(f"       Skills: {info['skills_count']}")
                    if 'capabilities' in info:
                        print(f"       Streaming: {info['capabilities'].get('streaming', False)}")
            else:
                print("   No agents configured - using mock agents for demo")
    
    async def run(self):
        """Run the A2A workflow."""
        self.logger.info("Starting A2A Parsl Workflow")
        
        try:
            # Start CLI interface
            await self._run_cli()
            
        except KeyboardInterrupt:
            self.logger.info("Workflow interrupted by user")
        except Exception as e:
            self.logger.error(f"Workflow error: {e}")
            raise
    
    async def _run_cli(self):
        """Run the CLI interface."""
        self.running = True
        
        print("\n🤖 A2A Protocol Demo CLI")
        print("=" * 40)
        print("This demo showcases A2A agent collaboration.")
        print("Try these examples:")
        print("  • 'book a flight to Paris'")
        print("  • 'write a Python function to sort a list'")
        print("  • 'analyze sales data trends'")
        print("  • 'help me plan a vacation to Japan'")
        print("\nType 'quit' to exit, 'status' for A2A status, 'help' for more commands")
        print("=" * 40)
        
        while self.running:
            try:
                user_input = input("\n🗣️  You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'status':
                    await self._display_a2a_status()
                    continue
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'test':
                    await self._run_test_scenarios()
                    continue
                
                if user_input:
                    print("⚡ Processing with A2A collaboration...")
                    
                    # Process through collaborative step
                    response = await self.components['collaborative_step'].process(user_input)
                    
                    print(f"\n🤖 Bot: {response}")
                else:
                    print("Please enter a message.")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
        
        self.running = False
    
    def _show_help(self):
        """Show help information."""
        print("\n📋 Available Commands:")
        print("  • quit/exit/q - Exit the demo")
        print("  • status - Show A2A protocol status")
        print("  • help - Show this help")
        print("  • test - Run test scenarios")
        print("\n🎯 Example Queries:")
        print("  • Travel: 'book a flight to Tokyo', 'find hotels in Paris'")
        print("  • Code: 'write a function to calculate fibonacci', 'debug this Python code'")
        print("  • Data: 'analyze this dataset', 'create a chart from CSV data'")
        print("  • General: Any other question or request")
    
    async def _run_test_scenarios(self):
        """Run predefined test scenarios."""
        print("\n🧪 Running A2A Test Scenarios...")
        
        test_scenarios = [
            "book a flight from NYC to London",
            "write a Python function to reverse a string",
            "analyze quarterly sales data",
            "help me plan a weekend trip"
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📝 Test {i}: {scenario}")
            print("   Processing...")
            
            try:
                response = await self.components['collaborative_step'].process(scenario)
                print(f"   ✅ Response: {response[:100]}..." if len(response) > 100 else f"   ✅ Response: {response}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n✅ Test scenarios completed!")
    
    async def shutdown(self):
        """Shutdown the workflow."""
        self.logger.info("Shutting down A2A Parsl Workflow")
        self.running = False
        
        # Shutdown agent
        if 'agent' in self.components:
            await self.components['agent'].shutdown()


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
        import traceback
        traceback.print_exc()
        raise


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="A2A Protocol Demo with Parsl Integration")
    parser.add_argument("--test-setup", action="store_true", help="Test A2A setup only")
    parser.add_argument("--config", help="A2A configuration file path")
    
    args = parser.parse_args()
    
    if args.test_setup:
        await test_a2a_setup()
        return
    
    # Create and run the workflow
    workflow = A2AParslWorkflow(config_path=args.config)
    
    try:
        await workflow.setup()
        await workflow.run()
    except Exception as e:
        print(f"❌ Workflow error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await workflow.shutdown()


if __name__ == "__main__":
    asyncio.run(main()) 