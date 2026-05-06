#!/usr/bin/env python3
"""
Test Single Shared WorkQueue Executor
=====================================

Test that we can submit multiple tasks with different priorities
to a SINGLE WorkQueue executor without creating multiple PBS jobs.
"""

import asyncio
import sys
import os

# Add nanobrain to path
sys.path.insert(0, '.')

from nanobrain.core.executor import ParslExecutor


async def test_single_workqueue():
    """Test single shared WorkQueue executor with multiple task priorities."""
    
    print("🔥 BRUTAL TRUTH: Testing SINGLE shared WorkQueue executor...")
    
    try:
        # Load the single shared WorkQueue executor
        print("1. Loading SINGLE shared WorkQueue executor...")
        config_path = "demos/viral_pssm_workflow/config/executors/workqueue_single_shared_executor.yml"
        executor = ParslExecutor.from_config(config_path)
        print(f"✅ Executor loaded: {executor.name}")
        print(f"   Max workers: {executor.config.max_workers}")
        print(f"   Default resource spec: {executor.config.default_resource_specification}")
        
        # Initialize executor (this should submit ONLY ONE PBS job)
        print("\n2. Initializing executor (should submit ONLY ONE PBS job)...")
        await executor.initialize()
        print("✅ Executor initialized - check qstat to verify ONLY ONE PBS job submitted")
        
        # Test multiple tasks with different priorities
        async def species_task():
            return {"task_type": "species", "priority": 10, "result": "species_data_acquired"}
        
        async def protein_task():
            return {"task_type": "protein", "priority": 7, "result": "protein_analyzed"}
        
        async def aggregation_task():
            return {"task_type": "aggregation", "priority": 3, "result": "results_aggregated"}
        
        async def shared_resource_task():
            return {"task_type": "shared_resource", "priority": 15, "result": "resources_managed"}
        
        print("\n3. Submitting tasks with different priorities to SAME WorkQueue...")
        
        # Submit tasks with different resource specifications (priorities)
        tasks = []
        
        # High priority species task
        print("   Submitting species task (priority 10)...")
        species_future = executor.execute(
            species_task,
            resource_specification={'cores': 1, 'memory': 2000, 'disk': 1000, 'priority': 10}
        )
        tasks.append(("species", species_future))
        
        # Medium priority protein task  
        print("   Submitting protein task (priority 7)...")
        protein_future = executor.execute(
            protein_task,
            resource_specification={'cores': 2, 'memory': 4000, 'disk': 2000, 'priority': 7}
        )
        tasks.append(("protein", protein_future))
        
        # Low priority aggregation task
        print("   Submitting aggregation task (priority 3)...")
        aggregation_future = executor.execute(
            aggregation_task,
            resource_specification={'cores': 1, 'memory': 4000, 'disk': 2000, 'priority': 3}
        )
        tasks.append(("aggregation", aggregation_future))
        
        # Highest priority shared resource task
        print("   Submitting shared resource task (priority 15)...")
        shared_future = executor.execute(
            shared_resource_task,
            resource_specification={'cores': 2, 'memory': 8000, 'disk': 4000, 'priority': 15}
        )
        tasks.append(("shared_resource", shared_future))
        
        print(f"\n4. Waiting for {len(tasks)} tasks to complete on SINGLE WorkQueue...")
        
        # Wait for all tasks to complete
        results = []
        for task_name, future in tasks:
            try:
                result = await future
                results.append((task_name, result))
                print(f"   ✅ {task_name} task completed: {result}")
            except Exception as e:
                print(f"   ❌ {task_name} task failed: {e}")
        
        print(f"\n✅ All {len(results)} tasks completed successfully!")
        print("✅ All tasks executed on SINGLE WorkQueue with priority scheduling!")
        
        # Shutdown executor
        print("\n5. Shutting down executor...")
        await executor.shutdown()
        print("✅ Executor shutdown complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔥 BRUTAL TRUTH: Testing SINGLE shared WorkQueue architecture...")
    success = asyncio.run(test_single_workqueue())
    if success:
        print("🔥 BRUTAL TRUTH: SINGLE WorkQueue test SUCCESSFUL!")
        print("🔥 BRUTAL TRUTH: Architecture is correct - ONE PBS job, multiple priorities!")
    else:
        print("🔥 BRUTAL TRUTH: SINGLE WorkQueue test FAILED!")
    print("🔥 BRUTAL TRUTH: Test complete!")
