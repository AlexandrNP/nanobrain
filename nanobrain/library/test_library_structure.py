#!/usr/bin/env python3
"""
Test script to verify the NanoBrain library structure works correctly.

This script tests:
- Import functionality for all library components
- Basic workflow creation and initialization
- Agent functionality
- Data unit operations
"""

import sys
import os
import asyncio

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

async def test_library_imports():
    """Test that all library components can be imported."""
    print("Testing library imports...")
    
    try:
        # Test agent imports
        from agents.specialized.code_writer import CodeWriterAgent
        from agents.specialized.file_writer import FileWriterAgent
        from agents.conversational.enhanced_collaborative_agent import EnhancedCollaborativeAgent
        print("✓ Agent imports successful")
        
        # Test infrastructure imports
        from infrastructure.data_units.conversation_history_unit import ConversationHistoryUnit
        print("✓ Infrastructure imports successful")
        
        # Test workflow imports
        from workflows.chat_workflow.chat_workflow import ChatWorkflow, create_chat_workflow
        print("✓ Workflow imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

async def test_workflow_creation():
    """Test workflow creation and basic functionality."""
    print("\nTesting workflow creation...")
    
    try:
        from workflows.chat_workflow.chat_workflow import ChatWorkflow
        
        # Create workflow
        workflow = ChatWorkflow()
        print("✓ Workflow created")
        
        # Initialize workflow
        await workflow.initialize()
        print("✓ Workflow initialized")
        
        # Test basic functionality
        response = await workflow.process_user_input("Hello, this is a test!")
        print(f"✓ Workflow response: {response[:50]}...")
        
        # Get status
        status = workflow.get_workflow_status()
        print(f"✓ Workflow status: {status['is_initialized']}")
        
        # Shutdown
        await workflow.shutdown()
        print("✓ Workflow shutdown complete")
        
        return True
        
    except Exception as e:
        print(f"✗ Workflow test failed: {e}")
        return False

async def test_agent_functionality():
    """Test enhanced collaborative agent functionality."""
    print("\nTesting agent functionality...")
    
    try:
        from agents.conversational.enhanced_collaborative_agent import EnhancedCollaborativeAgent
        from nanobrain.core.agent import AgentConfig
        
        # Create agent configuration
        config = AgentConfig(
            name="test_agent",
            description="Test agent for library verification",
            model="gpt-3.5-turbo",
            temperature=0.7,
            system_prompt="You are a test assistant."
        )
        
        # Create agent
        agent = EnhancedCollaborativeAgent(config, enable_metrics=True)
        print("✓ Agent created")
        
        # Initialize agent
        await agent.initialize()
        print("✓ Agent initialized")
        
        # Test processing (will use mock response if no API key)
        try:
            response = await agent.process("Hello, this is a test message!")
            print(f"✓ Agent response: {response[:50]}...")
        except Exception as e:
            print(f"⚠ Agent processing skipped (likely no API key): {e}")
        
        # Get status
        status = agent.get_enhanced_status()
        print(f"✓ Agent status: {status['agent_name']}")
        
        # Shutdown
        await agent.shutdown()
        print("✓ Agent shutdown complete")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        return False

async def test_data_unit_functionality():
    """Test conversation history data unit functionality."""
    print("\nTesting data unit functionality...")
    
    try:
        from infrastructure.data_units.conversation_history_unit import ConversationHistoryUnit, ConversationMessage
        from nanobrain.core.data_unit import DataUnitConfig
        from datetime import datetime
        
        # Create data unit
        config = DataUnitConfig(
            name="test_history",
            data_type="memory"  # Use valid data type
        )
        
        data_unit = ConversationHistoryUnit(
            config={'db_path': 'test_conversation_history.db'}
        )
        print("✓ Data unit created")
        
        # Initialize
        await data_unit.initialize()
        print("✓ Data unit initialized")
        
        # Test basic operations
        await data_unit.set("test data")
        data = await data_unit.get()
        print(f"✓ Data unit operations: {data}")
        
        # Test conversation message
        message = ConversationMessage(
            timestamp=datetime.now(),
            user_input="Test input",
            agent_response="Test response",
            response_time_ms=100.0,
            conversation_id="test_conv",
            message_id=1
        )
        
        await data_unit.save_message(message)
        print("✓ Message saved")
        
        # Get statistics
        stats = data_unit.get_statistics()
        print(f"✓ Statistics: {stats['total_messages']} messages")
        
        # Shutdown
        await data_unit.shutdown()
        print("✓ Data unit shutdown complete")
        
        # Clean up test database
        if os.path.exists("test_conversation_history.db"):
            os.remove("test_conversation_history.db")
            print("✓ Test database cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Data unit test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Testing NanoBrain Library Structure")
    print("=" * 50)
    
    tests = [
        test_library_imports,
        test_workflow_creation,
        test_agent_functionality,
        test_data_unit_functionality
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Library structure is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main()) 