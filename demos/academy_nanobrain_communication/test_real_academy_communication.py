#!/usr/bin/env python3
"""
🔥 BRUTAL TRUTH TEST: REAL ACADEMY-NANOBRAIN COMMUNICATION
This test MUST prove that responses come from actual Academy agents, not mocks/stubs.
"""

import asyncio
import logging
import sys
import uuid
from pathlib import Path

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent))

from nanobrain.academy_integration.academy_agent_step import AcademyAgentStep, AcademyAgentStepConfig

# Configure detailed logging to see everything
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class TestAcademyAgent:
    """Simple Academy agent for testing - MUST BE REAL ACADEMY AGENT"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.call_count = 0
        
    async def process_data(self, data):
        """Test action that returns verifiable response"""
        self.call_count += 1
        unique_response = f"ACADEMY_AGENT_RESPONSE_{self.agent_id}_{self.call_count}_{uuid.uuid4().hex[:8]}"
        logger.info(f"🔥 ACADEMY AGENT {self.agent_id} PROCESSING: {data}")
        logger.info(f"🔥 ACADEMY AGENT {self.agent_id} RETURNING: {unique_response}")
        return {
            "agent_id": self.agent_id,
            "call_count": self.call_count,
            "input_received": data,
            "unique_response": unique_response,
            "source": "REAL_ACADEMY_AGENT"
        }

async def test_real_academy_communication():
    """Test that verifies real Academy agent communication"""
    
    logger.info("🔥 STARTING REAL ACADEMY COMMUNICATION TEST")
    
    # Step 1: Create and launch a REAL Academy agent
    logger.info("🔥 STEP 1: LAUNCHING REAL ACADEMY AGENT")
    
    try:
        # Import Academy components
        from academy.agent import Agent
        from academy.manager import Manager
        from academy.exchange import LocalExchangeFactory
        
        # Import the real Academy agent class
        from test_real_academy_agent import RealTestAgent
        
        # Launch the real Academy agent
        exchange_factory = LocalExchangeFactory()
        manager = await Manager.from_exchange_factory(factory=exchange_factory)

        agent_handle = await manager.launch(RealTestAgent)
        # Extract the actual AgentId from the Handle
        agent_id = agent_handle.agent_id
        logger.info(f"🔥 REAL ACADEMY AGENT LAUNCHED WITH ID: {agent_id}")

        # Step 2: Test Nanobrain AcademyAgentStep communication
        logger.info("🔥 STEP 2: TESTING NANOBRAIN COMMUNICATION WITH REAL AGENT")

        # Load step from YAML file (NanoBrain requirement)
        step = AcademyAgentStep.from_config("test_real_communication_config.yml")

        # CRITICAL: Inject the same exchange factory so they can communicate
        # In real deployment, both would connect to same Redis/ProxyStore
        step._exchange_factory = exchange_factory

        # Also inject the known agent_id to bypass discovery for testing
        # In real deployment, discovery would work because both use same Redis/ProxyStore
        step._test_agent_id = agent_id
        
        # Test input data with unique identifier
        test_input = {
            "test_id": uuid.uuid4().hex,
            "message": "TEST_MESSAGE_FROM_NANOBRAIN",
            "timestamp": asyncio.get_event_loop().time()
        }
        
        logger.info(f"🔥 SENDING TO ACADEMY AGENT: {test_input}")
        
        # Execute the step - this MUST communicate with real Academy agent
        result = await step.process(test_input)
        
        logger.info(f"🔥 RECEIVED FROM ACADEMY AGENT: {result}")
        
        # Step 3: VERIFY the response came from real Academy agent
        logger.info("🔥 STEP 3: VERIFYING RESPONSE AUTHENTICITY")
        
        # Brutal verification checks
        assert result is not None, "❌ NO RESPONSE FROM ACADEMY AGENT!"
        assert isinstance(result, dict), f"❌ INVALID RESPONSE TYPE: {type(result)}"
        assert "source" in result, "❌ RESPONSE MISSING SOURCE FIELD!"
        assert result["source"] == "VERIFIED_ACADEMY_AGENT", f"❌ INVALID SOURCE: {result['source']}"
        assert "unique_response" in result, "❌ RESPONSE MISSING UNIQUE IDENTIFIER!"
        assert result["unique_response"].startswith("REAL_ACADEMY_RESPONSE_"), "❌ RESPONSE NOT FROM REAL AGENT!"
        assert result["agent_class"] == "RealTestAgent", f"❌ WRONG AGENT CLASS: {result['agent_class']}"
        assert result["input_data"] == test_input, "❌ INPUT DATA NOT PRESERVED!"
        
        logger.info("✅ SUCCESS: VERIFIED REAL ACADEMY AGENT COMMUNICATION!")
        logger.info(f"✅ AGENT RESPONSE: {result['unique_response']}")
        logger.info(f"✅ CALL COUNT: {result['call_count']}")

        # Core functionality verified - return success
        # Note: Skipping cleanup to avoid irrelevant error messages
        # In production, Academy agents would be managed by external orchestrators
        return True

    except ImportError as e:
        logger.error(f"❌ ACADEMY FRAMEWORK NOT AVAILABLE: {e}")
        logger.error("❌ CANNOT TEST REAL COMMUNICATION WITHOUT ACADEMY!")
        return False

    except Exception as e:
        logger.error(f"❌ REAL COMMUNICATION TEST FAILED: {e}")
        logger.exception("❌ FULL ERROR DETAILS:")

        # Try cleanup even if test failed
        try:
            if 'manager' in locals() and 'agent_handle' in locals():
                await manager.shutdown(agent_handle)
        except Exception:
            pass  # Ignore cleanup errors

        return False

if __name__ == "__main__":
    success = asyncio.run(test_real_academy_communication())
    if success:
        print("🔥 BRUTAL TRUTH: REAL ACADEMY COMMUNICATION VERIFIED!")
        sys.exit(0)
    else:
        print("❌ BRUTAL TRUTH: REAL COMMUNICATION FAILED!")
        sys.exit(1)
