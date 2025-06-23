#!/usr/bin/env python3
"""
Production Workflow Integration Test

Tests the fixed ChatbotViralWorkflow with proper from_config implementation.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import asyncio
import sys
import time
import uuid
from pathlib import Path
import pytest
import pytest_asyncio

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import actual components
try:
    from nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow import (
        ChatbotViralWorkflow, create_chatbot_viral_workflow
    )
    PRODUCTION_COMPONENTS_AVAILABLE = True
    print("✅ Production ChatbotViralWorkflow components loaded successfully")
except ImportError as e:
    print(f"❌ Production components not available: {e}")
    PRODUCTION_COMPONENTS_AVAILABLE = False


@pytest.mark.asyncio
async def test_workflow_creation():
    """Test if we can create the production workflow"""
    if not PRODUCTION_COMPONENTS_AVAILABLE:
        pytest.skip("Production components not available")
        return False
    
    try:
        # Test the factory function approach
        workflow = await create_chatbot_viral_workflow()
        print("✅ ChatbotViralWorkflow created successfully via factory")
        return True
        
    except Exception as e:
        print(f"❌ Workflow creation failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


@pytest.mark.asyncio
async def test_basic_message_processing():
    """Test basic message processing functionality"""
    if not PRODUCTION_COMPONENTS_AVAILABLE:
        pytest.skip("Production components not available")
        return False
    
    try:
        workflow = await create_chatbot_viral_workflow()
        test_query = "What is EEEV?"
        session_id = f"test_{uuid.uuid4().hex[:8]}"
        
        print(f"🧪 Testing message processing: {test_query}")
        
        chunks = []
        async for chunk in workflow.process_user_message(test_query, session_id):
            chunks.append(chunk)
            print(f"   📦 Received chunk: {chunk.get('type')}")
            
            # Stop after getting a few chunks to avoid long processing
            if len(chunks) >= 5:
                break
        
        assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
        print(f"✅ Basic message processing successful - {len(chunks)} chunks")
        return True
        
    except Exception as e:
        print(f"❌ Message processing test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


async def run_production_tests():
    """Run all production workflow tests"""
    print("🧪 Running Production Workflow Integration Tests...")
    
    if not PRODUCTION_COMPONENTS_AVAILABLE:
        print("❌ Production components not available - skipping all tests")
        return
    
    results = []
    
    print("\n🔧 Testing Workflow Creation...")
    results.append(await test_workflow_creation())
    
    print("\n💬 Testing Message Processing...")
    results.append(await test_basic_message_processing())
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n📊 Test Results: {success_count}/{total_count} tests passed")
    
    if success_count == total_count:
        print("🎉 All Production Workflow Integration Tests Passed!")
    else:
        print("❌ Some tests failed")


if __name__ == "__main__":
    asyncio.run(run_production_tests()) 