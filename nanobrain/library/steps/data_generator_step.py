"""
Data Generator Step for NanoBrain UI Builder Test

Simple step that generates test data for demonstration purposes.
"""

import asyncio
import time
from typing import Any, Dict, Optional
from nanobrain.core.step import BaseStep


class DataGeneratorStep(BaseStep):
    """Simple data generator step for testing the UI builder."""
    
    def __init__(self, step_id: str, name: str, description: str = "", config: Optional[Dict[str, Any]] = None):
        super().__init__(step_id=step_id, name=name, description=description, config=config or {})
        
    async def process(self) -> Dict[str, Any]:
        """Generate test data based on input."""
        self.logger.info(f"🔄 Generating data in step: {self.name}")
        
        # Simulate some processing time
        await asyncio.sleep(1)
        
        # Get input data if available
        input_data = "default"
        if self.has_input_data('seed_input'):
            input_data = await self.get_input_data('seed_input')
            
        # Generate some test data
        generated_data = f"Generated data based on: {input_data} (timestamp: {time.time()})"
        
        # Set output
        await self.set_output_data('generated_data', generated_data)
        
        self.logger.info(f"✅ Data generation complete: {generated_data}")
        
        return {"status": "completed", "generated_data": generated_data}