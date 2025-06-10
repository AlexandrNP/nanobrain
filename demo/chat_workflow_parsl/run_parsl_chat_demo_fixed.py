#!/usr/bin/env python3
"""
Fixed Parsl Chat Workflow Demo

Demonstrates the Parsl-based distributed chat workflow using the proper NanoBrain package structure.
This demo showcases:
- Proper nanobrain package imports
- Parsl distributed execution with ParslAgent
- Correct logging configuration that respects global settings
- Clean shutdown and error handling

Usage:
    python run_parsl_chat_demo_fixed.py
"""

import asyncio
import signal
import sys
import os
from pathlib import Path

# Setup paths for nanobrain package
demo_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd() / 'demo' / 'chat_workflow_parsl'
project_root = demo_dir.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging early to respect global settings
try:
    import nanobrain as nb
    from nanobrain.config import get_config_manager
    
    config_manager = get_config_manager()
    global_config = config_manager.get_config_dict()
    logging_mode = global_config.get('logging', {}).get('mode', 'both')
    
    if logging_mode == 'file':
        # Suppress console output for third-party libraries while keeping demo output
        import logging
        
        # Configure specific loggers to suppress console output
        loggers_to_suppress = [
            'parsl',
            'parsl.executors',
            'parsl.providers', 
            'parsl.channels',
            'parsl.launchers',
            'parsl.monitoring',
            'parsl.dataflow',
            'zmq',
            'asyncio',
            'concurrent.futures',
            'data_units.DataUnitMemory',
            'workflows.parsl_chat_workflow',
            'agents',
            'triggers',
            'links',
            'steps'
        ]
        
        for logger_name in loggers_to_suppress:
            logger = logging.getLogger(logger_name)
            logger.handlers = []  # Remove all handlers
            logger.addHandler(logging.NullHandler())  # Add null handler
            logger.propagate = False  # Don't propagate to root logger
        
        # Also suppress root logger console output
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and handler.stream.name in ['<stderr>', '<stdout>']:
                root_logger.removeHandler(handler)
    
    NANOBRAIN_AVAILABLE = True
    
except ImportError as e:
    print(f"⚠️  Warning: Could not import nanobrain package: {e}")
    print("   Please install nanobrain package: pip install -e .")
    NANOBRAIN_AVAILABLE = False
    sys.exit(1)

# Import workflow components
try:
    from nanobrain.library.workflows.chat_workflow_parsl.workflow import create_parsl_chat_workflow
    WORKFLOW_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error: Could not import Parsl chat workflow: {e}")
    WORKFLOW_AVAILABLE = False
    sys.exit(1)


class ParslChatDemo:
    """
    Fixed Parsl Chat Workflow Demo.
    
    Demonstrates proper usage of the nanobrain package structure
    with ParslAgent and distributed execution.
    """
    
    def __init__(self):
        self.workflow = None
        self.running = False
        
    async def initialize(self):
        """Initialize the demo and workflow."""
        print("🚀 NanoBrain Parsl Chat Workflow Demo (Fixed)")
        print("=" * 60)
        
        # Find configuration file
        config_path = project_root / 'library' / 'workflows' / 'chat_workflow_parsl' / 'ParslChatWorkflow.yml'
        
        if not config_path.exists():
            print(f"❌ Error: Configuration file not found: {config_path}")
            return False
        
        try:
            # Create and initialize workflow
            print("🔧 Initializing Parsl chat workflow...")
            self.workflow = await create_parsl_chat_workflow(str(config_path))
            
            print("✅ Workflow initialized: ParslChatWorkflow")
            print(f"   - Agents: {len(self.workflow.agents)}")
            print(f"   - Parsl Executor: ✅")
            print(f"   - Data Units: {len(self.workflow.data_units)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing workflow: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_interactive_demo(self):
        """Run the interactive chat demo."""
        if not self.workflow:
            print("❌ Error: Workflow not initialized")
            return
        
        self.running = True
        
        print("\n🎯 Parsl Chat Workflow Features:")
        print("  • Distributed execution with ParslAgent")
        print("  • Proper nanobrain package structure")
        print("  • Parsl-based parallel processing")
        print("  • Performance monitoring and metrics")
        print("  • Conversation history with metadata")
        
        print("\n📋 Available Commands:")
        print("  /help     - Show this help")
        print("  /stats    - Show performance statistics")
        print("  /status   - Show workflow status")
        print("  /quit     - Exit the demo")
        print("=" * 60)
        
        while self.running:
            try:
                # Get user input
                user_input = input("\n🗣️  You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() in ['/help', 'help']:
                    self._show_help()
                    continue
                
                if user_input.lower() in ['/stats', 'stats']:
                    self._show_stats()
                    continue
                
                if user_input.lower() in ['/status', 'status']:
                    self._show_status()
                    continue
                
                # Process message with workflow
                print("⚡ Processing with Parsl distributed agents...")
                
                try:
                    result = await self.workflow.process_message(user_input)
                    
                    if result['success']:
                        response = result['response']
                        print(f"\n🤖 Response: {response}")
                        
                        # Show additional info if multiple agents were used
                        if result.get('agents_used', 0) > 1:
                            print(f"     (Processed by {result['agents_used']} distributed agents)")
                    else:
                        print(f"\n❌ Error: {result['response']}")
                        
                except Exception as e:
                    print(f"\n❌ Error processing message: {e}")
                    
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        self.running = False
    
    def _show_help(self):
        """Show help information."""
        print("\n📖 Parsl Chat Workflow Demo Help")
        print("=" * 40)
        print("Commands:")
        print("  /help     - Show this help")
        print("  /stats    - Show performance statistics")
        print("  /status   - Show workflow status")
        print("  /quit     - Exit the demo")
        print("\nFeatures:")
        print("  • Messages are processed using ParslAgent")
        print("  • Distributed execution via Parsl")
        print("  • Proper nanobrain package structure")
        print("  • Performance metrics are tracked")
        print("=" * 40)
    
    def _show_stats(self):
        """Show performance statistics."""
        print("\n📊 Performance Statistics")
        print("=" * 30)
        
        if self.workflow and hasattr(self.workflow, 'agents'):
            for i, agent in enumerate(self.workflow.agents):
                if hasattr(agent, 'get_performance_metrics'):
                    metrics = agent.get_performance_metrics()
                    print(f"Agent {i+1} ({agent.config.name}):")
                    print(f"  Distributed calls: {metrics.get('distributed_calls', 0)}")
                    print(f"  Total time: {metrics.get('total_distributed_time', 0):.2f}s")
                    print(f"  Avg time: {metrics.get('avg_distributed_time', 0):.2f}s")
                    print(f"  Parsl enabled: {metrics.get('parsl_enabled', False)}")
        else:
            print("No performance data available")
        
        print("=" * 30)
    
    def _show_status(self):
        """Show workflow status."""
        print("\n📋 Workflow Status")
        print("=" * 25)
        
        if self.workflow:
            print(f"Initialized: {self.workflow.is_initialized}")
            print(f"Agents: {len(self.workflow.agents)}")
            print(f"Data Units: {len(self.workflow.data_units)}")
            print(f"Executor: {'✅' if self.workflow.executor else '❌'}")
            print(f"History: {'✅' if self.workflow.conversation_history else '❌'}")
        else:
            print("Workflow not initialized")
        
        print("=" * 25)
    
    async def shutdown(self):
        """Shutdown the demo and cleanup resources."""
        print("\n🔄 Shutting down workflow...")
        
        self.running = False
        
        if self.workflow:
            try:
                await self.workflow.shutdown()
                print("✅ Shutdown complete")
            except Exception as e:
                print(f"⚠️  Error during shutdown: {e}")
        
        print("👋 Goodbye!")


async def main():
    """Main entry point for the fixed Parsl chat workflow demo."""
    demo = ParslChatDemo()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down...")
        demo.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize demo
        if not await demo.initialize():
            return
        
        # Run interactive demo
        await demo.run_interactive_demo()
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        await demo.shutdown()


if __name__ == "__main__":
    # Run the fixed demo
    asyncio.run(main()) 