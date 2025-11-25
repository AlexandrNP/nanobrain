#!/usr/bin/env python3
"""
Real Academy Agent for Testing Communication
"""

import uuid
from academy.agent import Agent, action


class RealTestAgent(Agent):
    """Real Academy agent for testing communication with Nanobrain"""
    
    def __init__(self):
        super().__init__()
        self.call_count = 0
        
    @action
    async def process_data(self, data):
        """Real Academy action that returns verifiable response"""
        self.call_count += 1
        unique_id = uuid.uuid4().hex[:8]
        response = {
            "agent_type": "RealTestAgent",
            "call_count": self.call_count,
            "input_data": data,
            "unique_response": f"REAL_ACADEMY_RESPONSE_{unique_id}",
            "source": "VERIFIED_ACADEMY_AGENT",
            "agent_class": self.__class__.__name__
        }
        print(f"🔥 REAL ACADEMY AGENT PROCESSING: {data}")
        print(f"🔥 REAL ACADEMY AGENT RETURNING: {response}")
        return response
