"""
Sub Workflow for NanoBrain UI Builder Test

Simple sub-workflow implementation for demonstration purposes.
"""

import asyncio
import time
from typing import Any, Dict, Optional
from nanobrain.core.step import BaseStep


class SubWorkflow(BaseStep):
    """Simple sub-workflow step for testing the UI builder."""
    
    def __init__(self, step_id: str, name: str, description: str = "", config: Optional[Dict[str, Any]] = None):
        super().__init__(step_id=step_id, name=name, description=description, config=config or {})
        
    async def process(self) -> Dict[str, Any]:
        """Process workflow input and generate output."""
        self.logger.info(f"🔄 Processing sub-workflow: {self.name}")
        
        # Simulate sub-workflow processing time
        await asyncio.sleep(1)
        
        # Get input data
        input_data = "no input"
        if self.has_input_data('workflow_input'):
            input_data = await self.get_input_data('workflow_input')
            
        # Process through sub-workflow
        output_data = f"Sub-workflow result: {input_data} (completed at: {time.time()})"
        
        # Set output
        await self.set_output_data('workflow_output', output_data)
        
        self.logger.info(f"✅ Sub-workflow complete: {output_data}")
        
        return {"status": "completed", "workflow_output": output_data}