#!/usr/bin/env python3
"""
Test script for creating a step directly using NanoBrainBuilder.
"""

import os
import asyncio
from builder.NanoBrainBuilder import NanoBrainBuilder

async def main():
    # Set testing mode to use mock models
    os.environ["NANOBRAIN_TESTING"] = "1"
    
    builder = NanoBrainBuilder()
    
    # Get the workflow path
    workflow_path = builder.get_workflow_path('test_workflow_sync')
    if not workflow_path:
        print('Workflow not found')
        return
    
    # Set as current workflow
    builder.push_workflow(workflow_path)
    print(f'Set test_workflow_sync as the current workflow')
    
    # Create step
    result = await builder.create_step('TestDirect', 'Step', 'A test step')
    print(f'Result: {result}')

if __name__ == "__main__":
    asyncio.run(main()) 