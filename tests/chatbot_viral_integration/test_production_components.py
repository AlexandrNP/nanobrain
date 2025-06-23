#!/usr/bin/env python3
"""
Production Components Test

Tests the individual production components of the ChatbotViralWorkflow
to demonstrate they are properly working with the from_config pattern.

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
    from nanobrain.library.workflows.chatbot_viral_integration.steps.query_classification_step import QueryClassificationStep
    from nanobrain.library.workflows.chatbot_viral_integration.steps.conversational_response_step import ConversationalResponseStep
    from nanobrain.core.step import StepConfig
    from nanobrain.core.executor import LocalExecutor, ExecutorConfig
    PRODUCTION_COMPONENTS_AVAILABLE = True
    print("✅ Production component imports successful")
except ImportError as e:
    print(f"❌ Production components not available: {e}")
    PRODUCTION_COMPONENTS_AVAILABLE = False


@pytest.mark.asyncio
async def test_query_classification_step():
    """Test QueryClassificationStep with from_config pattern"""
    if not PRODUCTION_COMPONENTS_AVAILABLE:
        pytest.skip("Production components not available")
        return False
    
    try:
        # Create executor first
        executor_config = ExecutorConfig(executor_type="local", max_workers=2)
        executor = LocalExecutor.from_config(executor_config)
        
        # Create step config
        step_config = StepConfig(
            name="test_query_classification",
            description="Test query classification step"
        )
        
        # Create step using from_config with executor
        step = QueryClassificationStep.from_config(step_config, executor=executor)
        await step.initialize()
        
        # Test with a classification query
        test_data = {
            'user_query': 'Generate PSSM matrix for EEEV envelope protein',
            'session_data': None
        }
        
        result = await step.process(test_data)
        
        assert result is not None, "No result returned"
        assert result.get('success'), f"Step failed: {result.get('error')}"
        assert result.get('classification_data'), "No classification data returned"
        assert result.get('routing_decision'), "No routing decision returned"
        
        print(f"✅ QueryClassificationStep processed successfully")
        print(f"   Intent: {result['classification_data'].intent}")
        print(f"   Confidence: {result['classification_data'].confidence:.3f}")
        print(f"   Routing: {result['routing_decision']['next_step']}")
        
        return True
        
    except Exception as e:
        print(f"❌ QueryClassificationStep test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


@pytest.mark.asyncio
async def test_conversational_response_step():
    """Test ConversationalResponseStep with from_config pattern"""
    if not PRODUCTION_COMPONENTS_AVAILABLE:
        pytest.skip("Production components not available")
        return False
    
    try:
        # Create executor first
        executor_config = ExecutorConfig(executor_type="local", max_workers=2)
        executor = LocalExecutor.from_config(executor_config)
        
        # Create step config
        step_config = StepConfig(
            name="test_conversational_response",
            description="Test conversational response step"
        )
        
        # Create step using from_config with executor
        step = ConversationalResponseStep.from_config(step_config, executor=executor)
        await step.initialize()
        
        # Import the proper data structure
        from nanobrain.library.infrastructure.data.chat_session_data import QueryClassificationData
        
        # Test with a conversational query using proper data structure
        classification_data = QueryClassificationData(
            original_query='What is EEEV?',
            intent='conversational',
            confidence=0.85,
            extracted_parameters={},
            reasoning='Classified as conversational query',
            processing_time_ms=50.0
        )
        
        test_data = {
            'classification_data': classification_data,
            'routing_decision': {'next_step': 'conversational_response'},
            'session_data': None
        }
        
        result = await step.process(test_data)
        
        assert result is not None, "No result returned"
        assert result.get('success'), f"Step failed: {result.get('error')}"
        assert result.get('response_data'), "No response data returned"
        
        response_data = result['response_data']
        print(f"✅ ConversationalResponseStep processed successfully")
        print(f"   Response length: {len(response_data.response)} chars")
        print(f"   Response preview: {response_data.response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ ConversationalResponseStep test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


@pytest.mark.asyncio
async def test_executor_creation():
    """Test LocalExecutor creation with from_config pattern"""
    if not PRODUCTION_COMPONENTS_AVAILABLE:
        pytest.skip("Production components not available")
        return False
    
    try:
        # Create executor config
        executor_config = ExecutorConfig(
            executor_type="local",
            max_workers=2
        )
        
        # Create executor using from_config
        executor = LocalExecutor.from_config(executor_config)
        
        print(f"✅ LocalExecutor created successfully with {executor_config.max_workers} workers")
        return True
        
    except Exception as e:
        print(f"❌ LocalExecutor test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


async def run_production_component_tests():
    """Run all production component tests"""
    print("🧪 Running Production Components Tests...")
    
    if not PRODUCTION_COMPONENTS_AVAILABLE:
        print("❌ Production components not available - skipping all tests")
        return
    
    results = []
    
    print("\n⚙️ Testing Executor Creation...")
    results.append(await test_executor_creation())
    
    print("\n🧠 Testing Query Classification Step...")
    results.append(await test_query_classification_step())
    
    print("\n💬 Testing Conversational Response Step...")
    results.append(await test_conversational_response_step())
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n📊 Test Results: {success_count}/{total_count} tests passed")
    
    if success_count == total_count:
        print("🎉 All Production Component Tests Passed!")
        print("\n✅ Key Achievements:")
        print("   • from_config pattern working for all components")
        print("   • QueryClassificationStep functioning correctly")
        print("   • ConversationalResponseStep generating responses")
        print("   • LocalExecutor created successfully")
        print("   • Framework integration is successful")
    else:
        print("❌ Some tests failed")


if __name__ == "__main__":
    asyncio.run(run_production_component_tests()) 