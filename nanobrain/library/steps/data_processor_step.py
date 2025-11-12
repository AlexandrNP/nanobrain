"""
Data Processor Step for NanoBrain UI Builder Test

Simple step that processes data for demonstration purposes.
"""

import asyncio
import time
from typing import Any, Dict, Optional
from nanobrain.core.step import BaseStep
from nanobrain.core.data_unit import DataUnitString


class DataProcessorStep(BaseStep):
    """Simple data processor step for testing the UI builder."""
    
    def __init__(self, step_id: str, name: str, description: str = "", config: Optional[Dict[str, Any]] = None):
        super().__init__(step_id=step_id, name=name, description=description, config=config or {})
        
    async def process(self) -> Dict[str, Any]:
        """Process the input data."""
        self.logger.info(f"⚙️ Processing data in step: {self.name}")
        
        # Simulate some processing time
        await asyncio.sleep(1)
        
        # Get input data
        input_data = "no data"
        if self.has_input_data('raw_data'):
            input_data = await self.get_input_data('raw_data')
            
        # Process the data
        processed_data = f"Processed: {input_data} (processed at: {time.time()})"
        
        # Set output
        await self.set_output_data('processed_data', processed_data)
        
        self.logger.info(f"✅ Data processing complete: {processed_data}")
        
        return {"status": "completed", "processed_data": processed_data}