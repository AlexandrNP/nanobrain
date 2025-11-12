#!/usr/bin/env python3
"""
Real Aurora Computation Academy Agent

This agent wraps the AuroraComputationStep and provides it as an Academy agent
for distributed execution via AcademyLink.
"""

from __future__ import annotations
import asyncio
import sys
import os
import json
import time
import socket
from pathlib import Path
from typing import Dict, Any

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from academy.agent import Agent, action
    ACADEMY_AVAILABLE = True
except ImportError:
    # Use mock Academy if real Academy not available
    from nanobrain.academy_integration.mock_academy import Agent, action
    ACADEMY_AVAILABLE = False

# Import the actual step
from demos.academylink_aurora_demo.steps.aurora_computation_step import AuroraComputationStep


class AuroraComputationAgent(Agent):
    """
    Real Academy Agent that wraps AuroraComputationStep
    
    This agent provides the Aurora computation functionality as an Academy agent
    that can be called via AcademyLink for distributed execution.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_name = "aurora_computation_agent"
        self.step_instance = None
        
        # Node information
        self.node_info = self._get_node_info()
        
    def _get_node_info(self) -> Dict[str, Any]:
        """Get current node information"""
        return {
            "hostname": socket.gethostname(),
            "pbs_jobid": os.environ.get('PBS_JOBID', 'no_pbs_job'),
            "agent_id": self.agent_name,
            "academy_available": ACADEMY_AVAILABLE
        }
    
    async def initialize_step(self):
        """Initialize the Aurora computation step"""
        if self.step_instance is None:
            # Create step instance with Aurora configuration
            step_config = {
                "name": "aurora_computation_step",
                "description": "Aurora HPC computation step",
                "computation_type": "sequence_analysis",
                "aurora_nodes": 1,
                "intensity": "low"
            }
            self.step_instance = AuroraComputationStep(step_config)
            
    @action
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data using the Aurora computation step
        
        This is the main action called by AcademyLink to execute Aurora computation.
        """
        print(f"🔥 Aurora Computation Agent Processing")
        print(f"   Agent: {self.agent_name}")
        print(f"   Node: {self.node_info['hostname']}")
        print(f"   PBS Job: {self.node_info['pbs_jobid']}")
        print(f"   Academy Available: {ACADEMY_AVAILABLE}")
        print(f"   Input Data Type: {type(input_data)}")
        
        # Initialize step if needed
        await self.initialize_step()
        
        # Process the data using the actual Aurora step
        try:
            # Call the step's process method
            result = await self.step_instance.process(input_data)
            
            # Add agent metadata
            result.update({
                "agent_info": self.node_info,
                "processing_timestamp": time.time(),
                "academy_agent": self.agent_name,
                "distributed_execution": True
            })
            
            print(f"   ✅ Aurora computation completed successfully")
            return result
            
        except Exception as e:
            print(f"   ❌ Aurora computation failed: {e}")
            raise


async def main():
    """Main function for standalone testing"""
    print("🔥 AURORA COMPUTATION ACADEMY AGENT STARTING")
    print("=" * 50)
    print(f"Academy Available: {ACADEMY_AVAILABLE}")
    
    # Create agent
    agent = AuroraComputationAgent()
    
    print(f"Agent created: {agent.agent_name}")
    print(f"Node info: {agent.node_info}")
    
    # Test processing
    test_data = {
        "sequences": [
            {"id": "test_seq_1", "data": "ATCGATCG"},
            {"id": "test_seq_2", "data": "GCTAGCTA"}
        ],
        "metadata": {
            "source": "test_data_preparation",
            "timestamp": time.time()
        }
    }
    
    print(f"\nTesting Aurora computation...")
    result = await agent.process(test_data)
    
    print(f"\n🎯 AURORA COMPUTATION AGENT TEST COMPLETED")
    print(f"Result keys: {list(result.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
