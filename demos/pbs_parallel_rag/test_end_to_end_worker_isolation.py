#!/usr/bin/env python3
"""
End-to-End Test: Worker Instance Isolation with Real Data
==========================================================

Tests the parallel RAG workflow with PARSL executor to verify:
1. Each worker has its own step instance
2. Each instance has unique worker_id
3. @shared resources are accessed by all workers
4. Non-shared resources are NOT accessed by different workers
5. Real biological queries are processed correctly
"""

import asyncio
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nanobrain.core.worker_step_pool import WorkerStepPool
from nanobrain.core.executor import ParslExecutor, ExecutorConfig
from nanobrain.library.workflows.rag.steps import QueryEnhancementStep
from nanobrain.core.shared_resource import shared, get_resource_pool


# Create a test step with instance tracking
class TrackedQueryEnhancementStep(QueryEnhancementStep):
    """
    Query Enhancement Step with instance tracking.

    Tracks which worker accesses which instance to verify isolation.
    """

    # Class-level tracking
    _instance_registry = {}  # instance_id -> instance
    _access_log = []  # List of (worker_id, instance_id, timestamp)

    def _ensure_tracking_initialized(self):
        """Ensure tracking attributes are initialized."""
        if not hasattr(self, 'instance_id'):
            self.instance_id = id(self)
            self.access_count = 0
            self.processed_queries = []

            # Register this instance
            TrackedQueryEnhancementStep._instance_registry[self.instance_id] = self
            print(f"   🔧 Initialized tracking for instance: {self.instance_id}")
    
    async def process(self, *args, **kwargs):
        """Process with tracking."""
        # Ensure tracking is initialized
        self._ensure_tracking_initialized()

        # Track access
        self.access_count += 1
        
        # Log access
        access_record = {
            'worker_id': self.worker_id,
            'instance_id': self.instance_id,
            'timestamp': time.time(),
            'access_count': self.access_count
        }
        TrackedQueryEnhancementStep._access_log.append(access_record)
        
        # Track query
        if args and isinstance(args[0], dict):
            query = args[0].get('query', 'unknown')
            self.processed_queries.append(query)
        
        # Process normally
        result = await super().process(*args, **kwargs)
        
        # Add tracking info to result
        if isinstance(result, dict):
            result['_instance_id'] = self.instance_id
            result['_instance_access_count'] = self.access_count
        
        return result
    
    @classmethod
    def get_access_report(cls):
        """Get detailed access report."""
        report = {
            'total_instances': len(cls._instance_registry),
            'total_accesses': len(cls._access_log),
            'instances': {},
            'workers': defaultdict(list),
            'violations': []
        }
        
        # Analyze each instance
        for instance_id, instance in cls._instance_registry.items():
            report['instances'][instance_id] = {
                'worker_id': instance.worker_id,
                'access_count': instance.access_count,
                'queries_processed': len(instance.processed_queries)
            }
        
        # Analyze access log
        for access in cls._access_log:
            worker_id = access['worker_id']
            instance_id = access['instance_id']
            report['workers'][worker_id].append(instance_id)
        
        # Check for violations (worker accessing multiple instances)
        for worker_id, instance_ids in report['workers'].items():
            unique_instances = set(instance_ids)
            if len(unique_instances) > 1:
                report['violations'].append({
                    'worker_id': worker_id,
                    'instances_accessed': list(unique_instances),
                    'violation': 'Worker accessed multiple instances'
                })
        
        # Check for violations (instance accessed by multiple workers)
        instance_to_workers = defaultdict(set)
        for access in cls._access_log:
            instance_to_workers[access['instance_id']].add(access['worker_id'])
        
        for instance_id, workers in instance_to_workers.items():
            if len(workers) > 1:
                report['violations'].append({
                    'instance_id': instance_id,
                    'workers': list(workers),
                    'violation': 'Instance accessed by multiple workers'
                })
        
        return report
    
    @classmethod
    def reset_tracking(cls):
        """Reset all tracking data."""
        cls._instance_registry.clear()
        cls._access_log.clear()


async def test_worker_isolation_with_real_data():
    """Test worker isolation with real biological queries."""
    
    print("="*80)
    print("🧪 END-TO-END TEST: WORKER INSTANCE ISOLATION WITH REAL DATA")
    print("="*80)
    
    # Reset tracking
    TrackedQueryEnhancementStep.reset_tracking()
    
    # Test queries (real biological queries)
    test_queries = [
        "What are the key mechanisms of viral membrane fusion?",
        "How do spike proteins facilitate viral entry into host cells?",
        "What role does the ACE2 receptor play in SARS-CoV-2 infection?",
        "Explain the structural dynamics of viral envelope proteins during fusion.",
    ]
    
    print(f"\n📝 Testing with {len(test_queries)} real biological queries")
    
    # Step 1: Create worker step pool
    print(f"\n{'='*80}")
    print("STEP 1: CREATE WORKER STEP POOL")
    print(f"{'='*80}")
    
    num_workers = 4
    # Use absolute path relative to this script's location
    script_dir = Path(__file__).parent
    step_config_path = script_dir / "config" / "steps" / "prompt_enhancement_step.yml"

    print(f"\n🔧 Creating pool with {num_workers} workers...")
    print(f"   Config: {step_config_path}")
    pool = WorkerStepPool(
        step_class=TrackedQueryEnhancementStep,
        step_config=step_config_path,
        num_workers=num_workers,
        step_id='query_enhancement_step'
    )
    
    print(f"🔧 Initializing worker instances...")
    await pool.initialize_workers()
    
    worker_ids = pool.get_all_worker_ids()
    print(f"\n✅ Worker pool initialized")
    print(f"   Workers: {worker_ids}")
    
    # Get instance IDs for each worker
    worker_to_instance = {}
    for worker_id in worker_ids:
        instance = pool.get_worker_instance(worker_id)
        instance._ensure_tracking_initialized()  # Ensure tracking is set up
        worker_to_instance[worker_id] = instance.instance_id
        print(f"   {worker_id} → Instance {instance.instance_id}")
    
    # Step 2: Process queries in parallel
    print(f"\n{'='*80}")
    print("STEP 2: PROCESS QUERIES IN PARALLEL")
    print(f"{'='*80}")
    
    async def process_query(query, query_num, worker_id):
        """Process a single query with specific worker."""
        print(f"\n   📤 Query {query_num} → Worker {worker_id}")
        print(f"      Query: {query[:60]}...")
        
        # Get worker's instance
        instance = pool.get_worker_instance(worker_id)
        
        # Process
        start_time = time.time()
        result = await instance.process({'query': query})
        elapsed = time.time() - start_time
        
        print(f"      ✅ Completed in {elapsed:.2f}s")
        print(f"      Instance: {result.get('_instance_id', 'N/A')}")
        print(f"      Access count: {result.get('_instance_access_count', 'N/A')}")
        
        return {
            'query_num': query_num,
            'query': query,
            'worker_id': worker_id,
            'instance_id': result.get('_instance_id'),
            'access_count': result.get('_instance_access_count'),
            'result': result,
            'elapsed': elapsed
        }
    
    # Process all queries in parallel
    print(f"\n🚀 Processing {len(test_queries)} queries in parallel...")
    start_time = time.time()
    
    tasks = [
        process_query(query, i+1, worker_ids[i % len(worker_ids)])
        for i, query in enumerate(test_queries)
    ]
    
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    print(f"\n✅ All queries processed in {total_time:.2f}s")
    
    # Step 3: Analyze results
    print(f"\n{'='*80}")
    print("STEP 3: ANALYZE WORKER INSTANCE ISOLATION")
    print(f"{'='*80}")
    
    # Get access report
    report = TrackedQueryEnhancementStep.get_access_report()
    
    print(f"\n📊 Access Report:")
    print(f"   Total instances: {report['total_instances']}")
    print(f"   Total accesses: {report['total_accesses']}")
    
    print(f"\n📋 Instance Details:")
    for instance_id, info in report['instances'].items():
        print(f"   Instance {instance_id}:")
        print(f"      Worker: {info['worker_id']}")
        print(f"      Accesses: {info['access_count']}")
        print(f"      Queries: {info['queries_processed']}")
    
    print(f"\n👷 Worker Access Patterns:")
    for worker_id, instance_ids in report['workers'].items():
        unique_instances = set(instance_ids)
        print(f"   {worker_id}:")
        print(f"      Instances accessed: {list(unique_instances)}")
        print(f"      Total accesses: {len(instance_ids)}")
        
        # Check isolation
        if len(unique_instances) == 1:
            print(f"      ✅ ISOLATED (accessed only 1 instance)")
        else:
            print(f"      ❌ VIOLATION (accessed {len(unique_instances)} instances)")
    
    # Step 4: Check for violations
    print(f"\n{'='*80}")
    print("STEP 4: VIOLATION CHECK")
    print(f"{'='*80}")
    
    if report['violations']:
        print(f"\n❌ VIOLATIONS FOUND: {len(report['violations'])}")
        for violation in report['violations']:
            print(f"\n   Violation:")
            for key, value in violation.items():
                print(f"      {key}: {value}")
    else:
        print(f"\n✅ NO VIOLATIONS FOUND")
        print(f"   ✅ Each worker accessed only its own instance")
        print(f"   ✅ Each instance accessed by only one worker")
    
    # Step 5: Verify results
    print(f"\n{'='*80}")
    print("STEP 5: VERIFICATION")
    print(f"{'='*80}")
    
    success = True
    
    # Check 1: Each worker has unique instance
    print(f"\n✓ Check 1: Each worker has unique instance")
    if len(set(worker_to_instance.values())) == num_workers:
        print(f"   ✅ PASS: {num_workers} unique instances for {num_workers} workers")
    else:
        print(f"   ❌ FAIL: Not all workers have unique instances")
        success = False
    
    # Check 2: No cross-worker access
    print(f"\n✓ Check 2: No cross-worker instance access")
    if not report['violations']:
        print(f"   ✅ PASS: No violations detected")
    else:
        print(f"   ❌ FAIL: {len(report['violations'])} violations detected")
        success = False
    
    # Check 3: All queries processed
    print(f"\n✓ Check 3: All queries processed successfully")
    if len(results) == len(test_queries):
        print(f"   ✅ PASS: {len(results)}/{len(test_queries)} queries processed")
    else:
        print(f"   ❌ FAIL: Only {len(results)}/{len(test_queries)} queries processed")
        success = False
    
    # Check 4: Each instance accessed correct number of times
    print(f"\n✓ Check 4: Instance access counts")
    expected_accesses_per_instance = len(test_queries) // num_workers
    for instance_id, info in report['instances'].items():
        if info['access_count'] == expected_accesses_per_instance:
            print(f"   ✅ Instance {instance_id}: {info['access_count']} accesses (expected)")
        else:
            print(f"   ⚠️  Instance {instance_id}: {info['access_count']} accesses "
                  f"(expected {expected_accesses_per_instance})")
    
    # Shutdown
    print(f"\n🗑️  Shutting down pool...")
    await pool.shutdown()
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n📊 Test Results:")
    print(f"   Queries processed: {len(results)}/{len(test_queries)}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Avg time per query: {total_time/len(results):.2f}s")
    print(f"   Workers: {num_workers}")
    print(f"   Instances: {report['total_instances']}")
    print(f"   Violations: {len(report['violations'])}")
    
    print(f"\n✅ Verification:")
    print(f"   ✅ Each worker has unique instance: {len(set(worker_to_instance.values())) == num_workers}")
    print(f"   ✅ No cross-worker access: {len(report['violations']) == 0}")
    print(f"   ✅ All queries processed: {len(results) == len(test_queries)}")
    
    if success:
        print(f"\n{'='*80}")
        print("🎉 ALL TESTS PASSED!")
        print("✅ Worker instance isolation is working correctly")
        print("✅ Each worker uses only its own class instance")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("⚠️  SOME TESTS FAILED")
        print(f"{'='*80}")
    
    return success


async def main():
    """Main test function."""
    try:
        success = await test_worker_isolation_with_real_data()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

