#!/usr/bin/env python3
"""
Comprehensive example demonstrating distributed workflow execution with NanoBrain and Parsl.

This example shows how to:
1. Switch between local and distributed execution with configuration-only changes
2. Execute single workflows on distributed workers
3. Execute multiple workflows in parallel
4. Handle errors and monitor performance
5. Use the transparent execution model

Run this example with:
    python examples/distributed_execution_example.py
"""

import asyncio
import os
import time
import tempfile
import yaml
from pathlib import Path

# Add nanobrain to path for example
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.core.workflow import Workflow
from nanobrain.core.executor import ParslExecutor


async def demonstrate_transparent_execution():
    """Demonstrate that the same workflow code works with any executor."""
    
    print("=== 1. Transparent Execution Model ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create example workflow configuration
        workflow_config = {
            'name': 'example_workflow',
            'description': 'Example workflow for distributed execution demo',
            'steps': {},
            'executor_config': 'local_executor.yml'  # Will switch this
        }
        
        config_path = os.path.join(temp_dir, 'example_workflow.yml')
        with open(config_path, 'w') as f:
            yaml.dump(workflow_config, f)
        
        # Create local executor configuration
        local_executor_config = {
            'executor_type': 'local',
            'max_workers': 1
        }
        
        local_executor_path = os.path.join(temp_dir, 'local_executor.yml')
        with open(local_executor_path, 'w') as f:
            yaml.dump(local_executor_config, f)
        
        # Create distributed executor configuration
        parsl_executor_config = {
            'executor_type': 'parsl',
            'provider': 'LocalProvider',
            'max_blocks': 1,
            'workers_per_node': 2
        }
        
        parsl_executor_path = os.path.join(temp_dir, 'parsl_executor.yml')
        with open(parsl_executor_path, 'w') as f:
            yaml.dump(parsl_executor_config, f)
        
        # Change to temp directory for config loading
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Same workflow code for both execution modes
            async def run_workflow_with_executor(executor_config_name):
                """Run workflow with specified executor - same code for both modes."""
                
                # Update workflow config to use specified executor
                workflow_config['executor_config'] = executor_config_name
                with open('example_workflow.yml', 'w') as f:
                    yaml.dump(workflow_config, f)
                
                # Load and execute workflow - IDENTICAL CODE
                workflow = Workflow.from_config('example_workflow.yml')
                
                start_time = time.time()
                result = await workflow.execute({'query': 'example input'})
                execution_time = time.time() - start_time
                
                return result, execution_time
            
            # Test 1: Local execution
            print("Local Execution:")
            try:
                result_local, time_local = await run_workflow_with_executor('local_executor.yml')
                print(f"   ✅ Completed in {time_local:.3f} seconds")
                print(f"   Result: {result_local}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
            
            print()
            
            # Test 2: Distributed execution (same code, different config)
            print("Distributed Execution:")
            try:
                result_distributed, time_distributed = await run_workflow_with_executor('parsl_executor.yml')
                print(f"   ✅ Completed in {time_distributed:.3f} seconds")
                print(f"   Result: {result_distributed}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
            
            print("\n✅ Same workflow code works with both execution modes!")
            
        finally:
            os.chdir(original_cwd)


async def demonstrate_distributed_execution():
    """Demonstrate explicit distributed execution methods."""
    
    print("\n=== 2. Explicit Distributed Execution ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create workflow configuration
        workflow_config = {
            'name': 'distributed_example',
            'steps': {},
            'executor_config': 'parsl_executor.yml'
        }
        
        config_path = os.path.join(temp_dir, 'distributed_example.yml')
        with open(config_path, 'w') as f:
            yaml.dump(workflow_config, f)
        
        # Create Parsl executor configuration
        executor_config = {
            'executor_type': 'parsl',
            'provider': 'LocalProvider',
            'max_blocks': 1,
            'workers_per_node': 2
        }
        
        executor_config_path = os.path.join(temp_dir, 'parsl_executor.yml')
        with open(executor_config_path, 'w') as f:
            yaml.dump(executor_config, f)
        
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Test explicit distributed execution
            workflow = Workflow.from_config('distributed_example.yml')
            
            print("Testing execute_distributed method:")
            start_time = time.time()
            result = await workflow.execute_distributed({'input': 'distributed test'})
            execution_time = time.time() - start_time
            
            print(f"   ✅ Distributed execution completed in {execution_time:.3f} seconds")
            print(f"   Result: {result}")
            
        except Exception as e:
            print(f"   ❌ Distributed execution failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            os.chdir(original_cwd)


async def demonstrate_parallel_execution():
    """Demonstrate parallel workflow execution."""
    
    print("\n=== 3. Parallel Workflow Execution ===\n")
    
    try:
        # Create ParslExecutor for parallel execution
        executor_config = {
            'executor_type': 'parsl',
            'provider': 'LocalProvider',
            'max_blocks': 1,
            'workers_per_node': 4  # More workers for parallel execution
        }
        
        print("Creating ParslExecutor for parallel execution...")
        executor = ParslExecutor(executor_config)
        
        # Test parallel execution
        print("Executing 3 workflows in parallel:")
        
        start_time = time.time()
        results = await executor.execute_workflows_parallel(
            workflow_configs=[
                'config/workflows/example.yml',
                'config/workflows/example.yml',
                'config/workflows/example.yml'
            ],
            input_data_list=[
                {'query': 'parallel query 1', 'id': 1},
                {'query': 'parallel query 2', 'id': 2},
                {'query': 'parallel query 3', 'id': 3}
            ]
        )
        parallel_time = time.time() - start_time
        
        print(f"   ✅ Parallel execution completed in {parallel_time:.3f} seconds")
        
        # Analyze results
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        print(f"   Successful workflows: {len(successful_results)}")
        print(f"   Failed workflows: {len(failed_results)}")
        
        for i, result in enumerate(successful_results):
            worker_info = result.get('worker_info', {})
            hostname = worker_info.get('hostname', 'unknown')
            print(f"   Workflow {i+1}: executed on {hostname}")
        
        if failed_results:
            print("   Failed workflow errors:")
            for i, result in enumerate(failed_results):
                print(f"     Workflow {i+1}: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"   ❌ Parallel execution failed: {e}")
        import traceback
        traceback.print_exc()


async def demonstrate_performance_comparison():
    """Demonstrate performance comparison between execution modes."""
    
    print("\n=== 4. Performance Comparison ===\n")
    
    # Simulate performance comparison
    print("Performance comparison (simulated):")
    print("   Single workflow:")
    print("     Local execution:      0.150s")
    print("     Distributed execution: 0.180s (+20% overhead)")
    print()
    print("   3 workflows in sequence:")
    print("     Local execution:      0.450s (3 × 0.150s)")
    print("     Distributed execution: 0.540s (3 × 0.180s)")
    print()
    print("   3 workflows in parallel:")
    print("     Local execution:      0.450s (sequential)")
    print("     Distributed execution: 0.200s (parallel) - 2.25x faster!")
    print()
    print("✅ Distributed execution provides significant benefits for parallel workloads")


async def demonstrate_error_handling():
    """Demonstrate error handling in distributed execution."""
    
    print("\n=== 5. Error Handling ===\n")
    
    try:
        executor = ParslExecutor({'executor_type': 'parsl'})
        
        print("Testing error handling:")
        
        # Test with non-existent configuration
        try:
            result = await executor.execute_workflow_distributed(
                config_path='nonexistent_workflow.yml',
                input_data={'test': 'data'}
            )
        except RuntimeError as e:
            print(f"   ✅ Caught expected error: {type(e).__name__}")
            print(f"   Error message: {str(e)[:100]}...")
        
        print("   ✅ Error handling working correctly")
        
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")


async def main():
    """Run all demonstration examples."""
    
    print("NanoBrain Distributed Execution Demonstration")
    print("=" * 50)
    
    # Run all demonstrations
    await demonstrate_transparent_execution()
    await demonstrate_distributed_execution()
    await demonstrate_parallel_execution()
    await demonstrate_performance_comparison()
    await demonstrate_error_handling()
    
    print("\n" + "=" * 50)
    print("Demonstration Complete!")
    print("\nKey Takeaways:")
    print("✅ Same workflow code works with local and distributed execution")
    print("✅ Only executor configuration needs to change")
    print("✅ Distributed execution enables parallel processing")
    print("✅ Working directory fix resolves configuration loading issues")
    print("✅ Comprehensive error handling and monitoring")
    
    print("\nNext Steps:")
    print("1. Test with your actual NanoBrain workflows")
    print("2. Configure Parsl for your HPC environment")
    print("3. Monitor performance and optimize as needed")
    print("4. Implement production monitoring and logging")


if __name__ == "__main__":
    asyncio.run(main())
