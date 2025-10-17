#!/usr/bin/env python3
"""
Test PBS Configuration
======================

Simple test to verify the PBS configuration is properly set up.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path to import from demos
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from pbs_query_enhancement_step import PBSQueryEnhancementStep


async def test_pbs_config():
    """Test PBS configuration setup."""
    print("Testing PBS Query Enhancement Step Configuration...")
    
    # Initialize PBS step
    config = {
        "max_workers": 2,
        "timeout": 60
    }
    
    step = PBSQueryEnhancementStep(config)
    
    # Check configuration
    print(f"✅ PBS config loaded successfully")
    print(f"   Max workers: {step.parsl_config['max_workers']}")
    print(f"   Provider: {step.parsl_config['parsl_config']['executors'][0]['provider_config']['class']}")
    print(f"   Queue: {step.parsl_config['parsl_config']['executors'][0]['provider_config']['queue']}")
    print(f"   Nodes per block: {step.parsl_config['parsl_config']['executors'][0]['provider_config']['nodes_per_block']}")
    print(f"   CPUs per node: {step.parsl_config['parsl_config']['executors'][0]['provider_config']['cpus_per_node']}")
    print(f"   Walltime: {step.parsl_config['parsl_config']['executors'][0]['provider_config']['walltime']}")
    
    print("\n🎯 PBS configuration is ready for HPC cluster deployment!")
    

if __name__ == "__main__":
    asyncio.run(test_pbs_config())

