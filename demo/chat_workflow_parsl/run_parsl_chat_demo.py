#!/usr/bin/env python3
"""
NanoBrain Parsl Chat Workflow Demo

A demonstration of the NanoBrain Parsl chat workflow using proper components.
This demo showcases distributed chat processing with parallel agents.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the workflow
from library.workflows.chat_workflow_parsl.workflow import ParslChatWorkflow


async def main():
    """Main demo function."""
    print("🚀 NanoBrain Parsl Chat Workflow Demo")
    print("=" * 50)
    print("This demo showcases:")
    print("  • Distributed chat processing with Parsl")
    print("  • Multiple parallel conversational agents")
    print("  • Performance monitoring and metrics")
    print("  • Proper NanoBrain component integration")
    print("=" * 50)
    print()
    
    workflow = None
    
    try:
        # Create and initialize workflow
        print("🔧 Initializing Parsl Chat Workflow...")
        workflow = ParslChatWorkflow()
        await workflow.initialize()
        
        print("✅ Workflow initialized successfully!")
        print(f"📊 Status: {workflow.get_workflow_status()}")
        print()
        
        # Test messages
        test_messages = [
            "Hello, how does Parsl distributed execution work?",
            "What are the benefits of parallel processing?",
            "Can you explain the NanoBrain framework architecture?"
        ]
        
        # Process each test message
        for i, message in enumerate(test_messages, 1):
            print(f"📝 Test {i}: {message}")
            print("   Processing with parallel agents...")
            
            try:
                response = await workflow.process_user_input(message)
                print(f"🤖 Response: {response}")
                print()
            except Exception as e:
                print(f"❌ Error processing message: {e}")
                print()
        
        # Show performance statistics
        print("📊 Performance Statistics:")
        stats = await workflow.get_performance_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        print()
        
        # Interactive mode (optional)
        print("🎯 Interactive Mode (type 'quit' to exit):")
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("   Processing...")
                response = await workflow.process_user_input(user_input)
                print(f"🤖 Bot: {response}")
                print()
                
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print()
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if workflow:
            print("\n🧹 Shutting down workflow...")
            try:
                await workflow.shutdown()
                print("✅ Shutdown complete!")
            except Exception as e:
                print(f"⚠️  Shutdown error: {e}")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: No OPENAI_API_KEY found in environment variables.")
        print("   The demo will use mock responses for demonstration.")
        print("   Set OPENAI_API_KEY to use actual LLM responses.")
        print()
    
    # Run the demo
    asyncio.run(main()) 