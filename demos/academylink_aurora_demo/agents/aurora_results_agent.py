#!/usr/bin/env python3
"""
Real Aurora Results Academy Agent

This agent handles the transfer of Aurora computation results
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


class AuroraResultsAgent(Agent):
    """
    Real Academy Agent that handles Aurora computation results
    
    This agent transfers Aurora computation results to the next step
    in the distributed workflow via AcademyLink.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_name = "aurora_results_agent"
        
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
    
    @action
    async def process(self, aurora_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Aurora computation results for transfer
        
        This action receives Aurora computation results and prepares them
        for transfer to the result aggregation step.
        """
        print(f"🔥 Aurora Results Agent Processing")
        print(f"   Agent: {self.agent_name}")
        print(f"   Node: {self.node_info['hostname']}")
        print(f"   PBS Job: {self.node_info['pbs_jobid']}")
        print(f"   Academy Available: {ACADEMY_AVAILABLE}")
        print(f"   Results Data Type: {type(aurora_results)}")
        
        try:
            # Process and validate the Aurora results
            processed_results = {
                "aurora_computation_results": aurora_results,
                "transfer_metadata": {
                    "agent_info": self.node_info,
                    "transfer_timestamp": time.time(),
                    "academy_agent": self.agent_name,
                    "distributed_transfer": True
                }
            }
            
            # Extract key information for logging
            if isinstance(aurora_results, dict):
                if "computed_sequences" in aurora_results:
                    seq_count = len(aurora_results["computed_sequences"])
                    print(f"   ✅ Transferring {seq_count} computed sequences")
                
                if "computation_metadata" in aurora_results:
                    print(f"   ✅ Transferring computation metadata")
            
            print(f"   ✅ Aurora results transfer completed successfully")
            return processed_results
            
        except Exception as e:
            print(f"   ❌ Aurora results transfer failed: {e}")
            raise


async def main():
    """Main function for standalone testing"""
    print("🔥 AURORA RESULTS ACADEMY AGENT STARTING")
    print("=" * 50)
    print(f"Academy Available: {ACADEMY_AVAILABLE}")
    
    # Create agent
    agent = AuroraResultsAgent()
    
    print(f"Agent created: {agent.agent_name}")
    print(f"Node info: {agent.node_info}")
    
    # Test processing with mock Aurora results
    test_results = {
        "computed_sequences": [
            {"id": "seq_1", "result": 42.5, "complexity": 0.8},
            {"id": "seq_2", "result": 38.2, "complexity": 0.7}
        ],
        "computation_metadata": {
            "total_sequences": 2,
            "computation_time": 1.5,
            "aurora_node": "x4311c0s1b0n0",
            "success_rate": 1.0
        }
    }
    
    print(f"\nTesting Aurora results transfer...")
    result = await agent.process(test_results)
    
    print(f"\n🎯 AURORA RESULTS AGENT TEST COMPLETED")
    print(f"Result keys: {list(result.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
