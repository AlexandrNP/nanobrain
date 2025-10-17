#!/usr/bin/env python3
"""
Test Shared Resources with PARSL Executor
==========================================

Demonstrates:
1. @shared decorator for vector database
2. Worker ID tracking in PARSL executor
3. Multiple workers accessing shared resources
4. Resource pooling and statistics
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Add the pbs_parallel_rag directory to path for local imports
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from shared_vector_database import SharedVectorDatabase
from nanobrain.core.shared_resource import get_resource_pool, get_worker_id
from nanobrain.core.executor import ParslExecutor, ExecutorConfig
import numpy as np


async def test_shared_vector_database():
    """Test shared vector database with multiple workers."""
    print("="*80)
    print("🧪 TEST 1: SHARED VECTOR DATABASE")
    print("="*80)
    
    # Create shared vector database
    print("\n1. Creating shared vector database...")
    vector_db = SharedVectorDatabase(dimension=384)
    await vector_db.initialize()
    
    print(f"✅ Created shared vector database")
    print(f"   Resource ID: {vector_db.get_resource_id()}")
    print(f"   Resource Type: {vector_db._shared_resource_type}")
    
    # Check resource pool
    pool = get_resource_pool()
    resources = pool.list_resources()
    print(f"\n📊 Resources in pool: {len(resources)}")
    for res_id, stats in resources.items():
        print(f"   - {res_id}: {stats['resource_type']}")
    
    # Simulate multiple workers
    print("\n2. Simulating multiple workers accessing shared database...")
    
    async def worker_task(worker_id: str, query_id: int):
        """Simulate a worker performing operations."""
        print(f"   🔧 Worker {worker_id} starting...")
        
        # Add vectors
        vectors = [np.random.rand(384).tolist() for _ in range(5)]
        metadata = [{'query_id': query_id, 'chunk_id': i} for i in range(5)]
        
        add_result = await vector_db.add_vectors(vectors, metadata, worker_id=worker_id)
        print(f"   ✅ Worker {worker_id}: Added {add_result['added_count']} vectors")
        
        # Search
        query_vector = np.random.rand(384).tolist()
        search_result = await vector_db.search(query_vector, top_k=3, worker_id=worker_id)
        print(f"   🔍 Worker {worker_id}: Found {len(search_result['results'])} results")
        
        return {
            'worker_id': worker_id,
            'added': add_result['added_count'],
            'found': len(search_result['results'])
        }
    
    # Run 4 workers in parallel
    tasks = [worker_task(f"worker_{i}", i) for i in range(4)]
    results = await asyncio.gather(*tasks)
    
    # Print results
    print(f"\n📊 Worker Results:")
    for result in results:
        print(f"   - {result['worker_id']}: added {result['added']}, found {result['found']}")
    
    # Get final stats
    stats = vector_db.get_stats()
    print(f"\n📈 Final Statistics:")
    print(f"   Total vectors: {stats['database_stats']['total_vectors']}")
    print(f"   Total accesses: {stats['database_stats']['access_count']}")
    print(f"   Unique workers: {stats['database_stats']['unique_workers']}")
    
    # Shutdown
    await vector_db.shutdown()
    
    print("\n✅ Test 1 PASSED")
    return True


async def test_parsl_worker_id_tracking():
    """Test PARSL executor worker ID tracking."""
    print("\n" + "="*80)
    print("🧪 TEST 2: PARSL EXECUTOR WORKER ID TRACKING")
    print("="*80)
    
    # Create PARSL executor
    print("\n1. Creating PARSL executor...")
    
    parsl_config = {
        'executor_type': 'parsl',
        'max_workers': 4,
        'timeout': 60,
        'parsl_config': {
            'strategy': None,
            'app_cache': True,
            'checkpoint_mode': None,
            'retries': 1,
            'executors': [{
                'label': 'htex_local_parallel',
                'class': 'parsl.executors.HighThroughputExecutor',
                'max_workers_per_node': 4,
                'worker_debug': False,
                'heartbeat_period': 5,
                'heartbeat_threshold': 10,
                'provider_config': {
                    'class': 'parsl.providers.LocalProvider',
                    'min_blocks': 1,
                    'init_blocks': 1,
                    'max_blocks': 1,
                    'worker_init': '',
                    'parallelism': 1.0
                }
            }]
        }
    }
    
    try:
        executor_config = ExecutorConfig(**parsl_config)
        parsl_executor = ParslExecutor(config=executor_config)
        await parsl_executor.initialize()
        
        print(f"✅ PARSL executor initialized")
        print(f"   Max workers: {parsl_config['max_workers']}")
        
        # Execute tasks with worker ID tracking
        print("\n2. Executing tasks with worker ID tracking...")
        
        def sample_task(task_id: int):
            """Sample task that returns some data."""
            return {
                'task_id': task_id,
                'result': f"Task {task_id} completed",
                'data': list(range(task_id, task_id + 5))
            }
        
        # Execute multiple tasks
        tasks = []
        for i in range(4):
            print(f"   📤 Submitting task {i}...")
            task = parsl_executor.execute(
                lambda tid=i: sample_task(tid),
                add_worker_id=True  # Enable worker ID tracking
            )
            tasks.append(task)
        
        # Wait for results
        results = await asyncio.gather(*tasks)
        
        # Check results
        print(f"\n📊 Task Results:")
        for i, result in enumerate(results):
            worker_id = result.get('_worker_id', 'unknown')
            executor_type = result.get('_executor_type', 'unknown')
            task_result = result.get('result', result)
            
            print(f"   Task {i}:")
            print(f"      Worker ID: {worker_id}")
            print(f"      Executor: {executor_type}")
            if isinstance(task_result, dict):
                print(f"      Result: {task_result.get('result', 'N/A')}")
        
        # Verify all have worker IDs
        all_have_worker_ids = all('_worker_id' in r for r in results)
        all_have_executor_type = all('_executor_type' in r for r in results)
        
        print(f"\n✅ Worker ID tracking:")
        print(f"   All results have worker_id: {all_have_worker_ids}")
        print(f"   All results have executor_type: {all_have_executor_type}")
        
        # Shutdown
        await parsl_executor.shutdown()
        
        print("\n✅ Test 2 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_shared_resource_with_parsl():
    """Test shared resource accessed by PARSL workers."""
    print("\n" + "="*80)
    print("🧪 TEST 3: SHARED RESOURCE WITH PARSL EXECUTOR")
    print("="*80)
    
    # Create shared vector database
    print("\n1. Creating shared vector database...")
    vector_db = SharedVectorDatabase(dimension=128)
    await vector_db.initialize()
    
    print(f"✅ Shared vector database created: {vector_db.get_resource_id()}")
    
    # Create PARSL executor
    print("\n2. Creating PARSL executor...")
    
    parsl_config = {
        'executor_type': 'parsl',
        'max_workers': 4,
        'timeout': 60,
        'parsl_config': {
            'executors': [{
                'label': 'htex_local',
                'class': 'parsl.executors.HighThroughputExecutor',
                'max_workers_per_node': 4,
                'provider_config': {
                    'class': 'parsl.providers.LocalProvider',
                    'min_blocks': 1,
                    'init_blocks': 1,
                    'max_blocks': 1
                }
            }]
        }
    }
    
    try:
        executor_config = ExecutorConfig(**parsl_config)
        parsl_executor = ParslExecutor(config=executor_config)
        await parsl_executor.initialize()
        
        print(f"✅ PARSL executor initialized")
        
        # Define task that accesses shared resource
        async def search_task(query_id: int):
            """Task that searches the shared vector database."""
            query_vector = np.random.rand(128).tolist()
            worker_id = get_worker_id()
            
            result = await vector_db.search(
                query_vector,
                top_k=3,
                worker_id=worker_id
            )
            
            return {
                'query_id': query_id,
                'worker_id': result['worker_id'],
                'results_count': len(result['results']),
                'access_count': result['access_count']
            }
        
        # Add some initial vectors
        print("\n3. Adding initial vectors...")
        vectors = [np.random.rand(128).tolist() for _ in range(20)]
        metadata = [{'id': i} for i in range(20)]
        await vector_db.add_vectors(vectors, metadata)
        print(f"✅ Added {len(vectors)} vectors")
        
        # Execute multiple search tasks via PARSL
        print("\n4. Executing parallel searches via PARSL...")
        
        tasks = []
        for i in range(4):
            task = parsl_executor.execute(
                lambda qid=i: search_task(qid),
                add_worker_id=True
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Print results
        print(f"\n📊 Search Results:")
        for result in results:
            parsl_worker_id = result.get('_worker_id', 'unknown')
            search_result = result.get('result', result)
            
            if isinstance(search_result, dict):
                print(f"   Query {search_result.get('query_id', '?')}:")
                print(f"      PARSL Worker: {parsl_worker_id}")
                print(f"      Search Worker: {search_result.get('worker_id', 'unknown')}")
                print(f"      Results: {search_result.get('results_count', 0)}")
        
        # Get final stats
        stats = vector_db.get_stats()
        print(f"\n📈 Final Statistics:")
        print(f"   Total vectors: {stats['database_stats']['total_vectors']}")
        print(f"   Total accesses: {stats['database_stats']['access_count']}")
        print(f"   Unique workers: {stats['database_stats']['unique_workers']}")
        
        # Shutdown
        await parsl_executor.shutdown()
        await vector_db.shutdown()
        
        print("\n✅ Test 3 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("="*80)
    print("🚀 SHARED RESOURCES AND PARSL EXECUTOR TESTS")
    print("="*80)
    
    results = []
    
    # Test 1: Shared vector database
    try:
        result1 = await test_shared_vector_database()
        results.append(('Shared Vector Database', result1))
    except Exception as e:
        print(f"\n❌ Test 1 failed with exception: {e}")
        results.append(('Shared Vector Database', False))
    
    # Test 2: PARSL worker ID tracking
    try:
        result2 = await test_parsl_worker_id_tracking()
        results.append(('PARSL Worker ID Tracking', result2))
    except Exception as e:
        print(f"\n❌ Test 2 failed with exception: {e}")
        results.append(('PARSL Worker ID Tracking', False))
    
    # Test 3: Shared resource with PARSL
    try:
        result3 = await test_shared_resource_with_parsl()
        results.append(('Shared Resource with PARSL', result3))
    except Exception as e:
        print(f"\n❌ Test 3 failed with exception: {e}")
        results.append(('Shared Resource with PARSL', False))
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠️  SOME TESTS FAILED")
    
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

