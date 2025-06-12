#!/usr/bin/env python3
"""
Comprehensive Demo for NanoBrain Parsl Chat Workflow

This demo showcases the full capabilities of the Parsl-based distributed chat workflow,
including performance monitoring, distributed processing, and interactive features.
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Setup paths
demo_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd() / 'demo' / 'chat_workflow_parsl'
project_root = demo_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import logging functions
from nanobrain.core.logging_system import get_logger, reconfigure_global_logging, get_system_log_manager

# Ensure proper logging system initialization
try:
    # Force recreation of the system log manager to ensure fresh session
    import nanobrain.core.logging_system
    nanobrain.core.logging_system._system_log_manager = None
    
    # Reconfigure global logging
    reconfigure_global_logging()
    
    # Initialize the system log manager explicitly
    log_manager = get_system_log_manager()
    print(f"📋 Logging system initialized - Session: {log_manager.session_dir}")
    
except Exception as e:
    print(f"⚠️  Warning: Logging initialization issue: {e}")

# Configure third-party loggers
try:
    from nanobrain.core.logging_system import configure_third_party_loggers
    configure_third_party_loggers()
except ImportError:
    pass

# Import the workflow
from nanobrain.library.workflows.chat_workflow_parsl.workflow import ParslChatWorkflow, create_parsl_chat_workflow


class ParslChatDemo:
    """Comprehensive demo for Parsl Chat Workflow."""
    
    def __init__(self):
        self.workflow = None
        self.config_path = project_root / 'nanobrain' / 'library' / 'workflows' / 'chat_workflow_parsl' / 'ParslChatWorkflow.yml'
        self.logger = get_logger("parsl_chat_demo", category="workflows")
        
    async def initialize(self):
        """Initialize the workflow."""
        print("🚀 Initializing NanoBrain Parsl Chat Workflow")
        print("=" * 60)
        
        self.logger.info("Starting workflow initialization")
        
        print(f"📁 Config path: {self.config_path}")
        self.workflow = await create_parsl_chat_workflow(str(self.config_path))
        
        # Display initialization status
        status = self.workflow.get_workflow_status()
        print(f"✅ Workflow initialized successfully!")
        print(f"   📊 Status: {status}")
        print()
        
        self.logger.info(f"Workflow initialized with status: {status}")
        
    async def run_performance_benchmark(self):
        """Run a performance benchmark with multiple messages."""
        print("🏃‍♂️ Running Performance Benchmark")
        print("-" * 40)
        
        test_messages = [
            "What is distributed computing?",
            "Explain the benefits of parallel processing",
            "How does Parsl help with scientific computing?",
            "What are the challenges in distributed systems?",
            "Compare local vs distributed execution"
        ]
        
        print(f"📝 Testing {len(test_messages)} messages...")
        
        start_time = time.time()
        
        # Process messages sequentially to see performance
        for i, message in enumerate(test_messages, 1):
            print(f"\n🔄 Processing message {i}/{len(test_messages)}: {message[:50]}...")
            
            msg_start = time.time()
            response = await self.workflow.process_user_input(message)
            msg_end = time.time()
            
            print(f"   ⏱️  Response time: {msg_end - msg_start:.2f}s")
            print(f"   🤖 Response: {response[:100]}...")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Get performance statistics
        stats = await self.workflow.get_performance_stats()
        parsl_stats = await self.workflow.get_parsl_stats()
        
        print(f"\n📈 Benchmark Results:")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   📊 Performance stats: {stats}")
        print(f"   🔧 Parsl stats: {parsl_stats}")
        print()
        
    async def run_interactive_session(self):
        """Run an interactive chat session."""
        print("💬 Interactive Chat Session")
        print("-" * 40)
        print("Type 'quit', 'exit', or 'bye' to end the session")
        print("Type 'stats' to see performance statistics")
        print("Type 'status' to see workflow status")
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                    
                # Check for special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'stats':
                    stats = await self.workflow.get_performance_stats()
                    parsl_stats = await self.workflow.get_parsl_stats()
                    print(f"📈 Performance: {stats}")
                    print(f"🔧 Parsl: {parsl_stats}")
                    continue
                elif user_input.lower() == 'status':
                    status = self.workflow.get_workflow_status()
                    print(f"📊 Status: {status}")
                    continue
                
                # Process the message
                print("🤖 Processing...", end="", flush=True)
                start_time = time.time()
                
                response = await self.workflow.process_user_input(user_input)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                print(f"\r🤖 Assistant ({response_time:.2f}s): {response}")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    async def demonstrate_distributed_features(self):
        """Demonstrate distributed processing features."""
        print("🌐 Distributed Processing Demonstration")
        print("-" * 40)
        
        self.logger.info("Starting distributed processing demonstration")
        
        # Test concurrent processing
        print("🔄 Testing concurrent message processing...")
        
        messages = [
            "What is machine learning?",
            "Explain quantum computing",
            "Describe blockchain technology"
        ]
        
        # Submit all messages concurrently
        print(f"📤 Submitting {len(messages)} messages concurrently...")
        start_time = time.time()
        
        self.logger.info(f"Submitting {len(messages)} messages for concurrent processing")
        
        # Use distributed processing for multiple messages
        result = await self.workflow.process_messages_distributed(messages)
        
        # Extract responses from the result
        if result.get('success', False):
            responses = [r['response'] for r in result.get('successful_results', [])]
        else:
            responses = [f"Error: {result.get('error', 'Unknown error')}" for _ in messages]
        end_time = time.time()
        
        print(f"✅ All messages processed in {end_time - start_time:.2f}s")
        
        # Show detailed results
        if result.get('success', False):
            successful_count = result.get('successful_count', 0)
            failed_count = result.get('failed_count', 0)
            print(f"   📊 Results: {successful_count} successful, {failed_count} failed")
            
            # Show successful results
            for i, result_item in enumerate(result.get('successful_results', []), 1):
                msg = result_item['message']
                resp = result_item['response']
                agent = result_item['agent']
                duration = result_item.get('duration_seconds', 0)
                print(f"   {i}. [{agent}] {msg[:30]}... → {resp[:50]}... ({duration:.2f}s)")
                self.logger.info(f"Message {i}: {msg} -> {resp[:100]} (Agent: {agent})")
                
            # Show failed results if any
            for i, failed_item in enumerate(result.get('failed_results', []), 1):
                msg = failed_item['message']
                error = failed_item['error']
                agent = failed_item['agent']
                print(f"   ❌ [{agent}] {msg[:30]}... → Error: {error}")
        else:
            print(f"   ❌ Processing failed: {result.get('error', 'Unknown error')}")
        
        self.logger.info(f"Distributed processing completed in {end_time - start_time:.2f}s with result: {result}")
        
        print()
        
    async def cleanup(self):
        """Clean up resources."""
        if self.workflow:
            print("🧹 Cleaning up workflow...")
            await self.workflow.shutdown()
            print("✅ Cleanup complete")


async def main():
    """Main demo function."""
    demo = ParslChatDemo()
    
    try:
        # Initialize
        await demo.initialize()
        
        # Show menu
        while True:
            print("🎯 NanoBrain Parsl Chat Workflow Demo")
            print("=" * 40)
            print("1. 🏃‍♂️ Performance Benchmark")
            print("2. 🌐 Distributed Processing Demo")
            print("3. 💬 Interactive Chat Session")
            print("4. 📊 Show Current Status")
            print("5. 🚪 Exit")
            print()
            
            choice = input("Select an option (1-5): ").strip()
            
            if choice == '1':
                await demo.run_performance_benchmark()
            elif choice == '2':
                await demo.demonstrate_distributed_features()
            elif choice == '3':
                await demo.run_interactive_session()
            elif choice == '4':
                status = demo.workflow.get_workflow_status()
                stats = await demo.workflow.get_performance_stats()
                parsl_stats = await demo.workflow.get_parsl_stats()
                print(f"📊 Workflow Status: {status}")
                print(f"📈 Performance Stats: {stats}")
                print(f"🔧 Parsl Stats: {parsl_stats}")
                print()
            elif choice == '5':
                break
            else:
                print("❌ Invalid choice. Please select 1-5.")
                print()
                
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Exiting...")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    print("🧠 NanoBrain Parsl Chat Workflow - Comprehensive Demo")
    print("=" * 60)
    print()
    
    asyncio.run(main()) 