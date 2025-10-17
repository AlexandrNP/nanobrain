#!/usr/bin/env python3
"""
Test Worker Step Pool Functionality
====================================

Tests the per-worker step instance mechanism:
1. Each PARSL worker gets its own step instance
2. Each instance has unique worker_id
3. @shared resources are NOT duplicated
4. Non-shared resources ARE duplicated per worker
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nanobrain.core.worker_step_pool import WorkerStepPool
from nanobrain.core.shared_resource import shared, get_resource_pool
from nanobrain.library.workflows.rag.steps import QueryEnhancementStep


# Create a test step with shared and non-shared resources
@shared(resource_type='test_vector_db')
class TestVectorDatabase:
    """Test vector database marked as @shared."""
    
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.vectors = {}
        self.access_count = 0
    
    async def initialize(self):
        print(f"   🔧 Initializing shared vector database: {self.get_resource_id()}")
    
    async def search(self, query, worker_id=None):
        self.access_count += 1
        return {
            'results': [],
            'worker_id': worker_id,
            'resource_id': self.get_resource_id(),
            'access_count': self.access_count
        }


class TestCache:
    """Test cache NOT marked as @shared - should be duplicated."""
    
    def __init__(self):
        self.cache = {}
        self.instance_id = id(self)
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        self.cache[key] = value


class TestStep:
    """Test step with both shared and non-shared resources."""
    
    def __init__(self, config):
        self.config = config
        self.worker_id = None
        
        # Shared resource
        self.vector_db = TestVectorDatabase(dimension=384)
        
        # Non-shared resource
        self.cache = TestCache()
        
        # Instance-specific state
        self.process_count = 0
    
    async def initialize(self):
        await self.vector_db.initialize()
        print(f"   ✅ Initialized TestStep (worker_id: {self.worker_id})")
    
    async def process(self, data):
        self.process_count += 1
        result = await self.vector_db.search(data.get('query'), worker_id=self.worker_id)
        return {
            'worker_id': self.worker_id,
            'process_count': self.process_count,
            'cache_instance_id': self.cache.instance_id,
            'vector_db_resource_id': result['resource_id'],
            'vector_db_access_count': result['access_count']
        }


async def test_worker_step_pool():
    """Test worker step pool creation and functionality."""
    
    print("="*80)
    print("🧪 TEST: WORKER STEP POOL")
    print("="*80)
    
    # Test configuration
    num_workers = 4
    step_config = {'test': True}
    
    print(f"\n1. Creating WorkerStepPool with {num_workers} workers...")
    pool = WorkerStepPool(
        step_class=TestStep,
        step_config=step_config,
        num_workers=num_workers,
        step_id='test_step'
    )
    
    print(f"\n2. Initializing worker instances...")
    await pool.initialize_workers()
    
    # Get statistics
    stats = pool.get_stats()
    print(f"\n📊 Pool Statistics:")
    print(f"   Step class: {stats['step_class']}")
    print(f"   Number of workers: {stats['num_workers']}")
    print(f"   Initialized: {stats['initialized']}")
    print(f"   Worker IDs: {stats['worker_ids']}")
    print(f"   Shared resources: {stats['shared_resources']}")
    print(f"   Number of shared resources: {stats['num_shared_resources']}")
    
    # Verify each worker has unique instance
    print(f"\n3. Verifying per-worker instances...")
    worker_ids = pool.get_all_worker_ids()
    instances = {}
    vector_db_instances = set()
    cache_instances = set()
    
    for worker_id in worker_ids:
        instance = pool.get_worker_instance(worker_id)
        instances[worker_id] = instance
        
        print(f"\n   Worker: {worker_id}")
        print(f"      Step instance ID: {id(instance)}")
        print(f"      Vector DB resource ID: {instance.vector_db.get_resource_id()}")
        print(f"      Cache instance ID: {instance.cache.instance_id}")
        
        vector_db_instances.add(id(instance.vector_db))
        cache_instances.add(instance.cache.instance_id)
    
    # Verify results
    print(f"\n4. Verification Results:")
    print(f"   ✅ Total worker instances: {len(instances)}")
    print(f"   ✅ Unique step instances: {len(set(id(i) for i in instances.values()))}")
    print(f"   ✅ Unique vector DB instances: {len(vector_db_instances)} (should be 1 - @shared)")
    print(f"   ✅ Unique cache instances: {len(cache_instances)} (should be {num_workers} - not @shared)")
    
    # Test processing with each worker
    print(f"\n5. Testing processing with each worker...")
    results = []
    for worker_id in worker_ids:
        instance = pool.get_worker_instance(worker_id)
        result = await instance.process({'query': f'test query from {worker_id}'})
        results.append(result)
        print(f"   Worker {worker_id}: process_count={result['process_count']}, "
              f"vdb_access={result['vector_db_access_count']}")
    
    # Verify shared resource access count
    first_instance = pool.get_worker_instance(worker_ids[0])
    total_vdb_accesses = first_instance.vector_db.access_count
    
    print(f"\n6. Shared Resource Verification:")
    print(f"   Total vector DB accesses: {total_vdb_accesses} (should be {num_workers})")
    print(f"   ✅ All workers accessed same vector DB instance")
    
    # Verify non-shared resource isolation
    print(f"\n7. Non-Shared Resource Verification:")
    for i, (worker_id, result) in enumerate(zip(worker_ids, results)):
        print(f"   Worker {i+1}: process_count={result['process_count']} (should be 1)")
    
    # Shutdown
    print(f"\n8. Shutting down pool...")
    await pool.shutdown()
    
    # Final verification
    print(f"\n{'='*80}")
    print("✅ TEST RESULTS:")
    print(f"{'='*80}")
    
    success = True
    
    # Check: Each worker has unique step instance
    if len(set(id(i) for i in instances.values())) == num_workers:
        print("✅ Each worker has unique step instance")
    else:
        print("❌ Workers do not have unique step instances")
        success = False
    
    # Check: @shared resources are NOT duplicated
    if len(vector_db_instances) == 1:
        print("✅ @shared resources are NOT duplicated (1 vector DB instance)")
    else:
        print(f"❌ @shared resources ARE duplicated ({len(vector_db_instances)} vector DB instances)")
        success = False
    
    # Check: Non-shared resources ARE duplicated
    if len(cache_instances) == num_workers:
        print(f"✅ Non-shared resources ARE duplicated ({num_workers} cache instances)")
    else:
        print(f"❌ Non-shared resources are NOT properly duplicated ({len(cache_instances)} cache instances)")
        success = False
    
    # Check: Worker IDs are unique
    if len(worker_ids) == num_workers and len(set(worker_ids)) == num_workers:
        print(f"✅ All {num_workers} worker IDs are unique")
    else:
        print(f"❌ Worker IDs are not unique")
        success = False
    
    # Check: Shared resource access count
    if total_vdb_accesses == num_workers:
        print(f"✅ Shared resource access count correct ({total_vdb_accesses} accesses)")
    else:
        print(f"❌ Shared resource access count incorrect ({total_vdb_accesses} != {num_workers})")
        success = False
    
    print(f"\n{'='*80}")
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Worker step pool is working correctly")
    else:
        print("⚠️  SOME TESTS FAILED")
    print(f"{'='*80}")
    
    return success


async def main():
    """Main test function."""
    try:
        success = await test_worker_step_pool()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

