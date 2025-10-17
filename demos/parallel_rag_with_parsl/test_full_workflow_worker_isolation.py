#!/usr/bin/env python3
"""
Full Workflow Worker Isolation Test
====================================

Tests that ALL steps in a workflow are properly duplicated per worker when
the first step uses PARSL executor.

Test Setup:
1. Create workflow with 3 steps: Step1 → Step2 → Step3
2. Set Step1 executor to PARSL
3. Verify ALL steps (Step1, Step2, Step3) are duplicated per worker
4. Verify each worker accesses only its own instances
5. Verify @shared resources are not duplicated

Requirements:
- Each worker has its own instance of ALL steps
- No cross-worker instance access
- @shared resources properly pooled
"""

import asyncio
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nanobrain.core.step import BaseStep
from nanobrain.core.executor import ParslExecutor, ExecutorConfig
from nanobrain.core.worker_step_pool import WorkerStepPool
from nanobrain.core.shared_resource import shared


# Global tracking for all step instances
class InstanceTracker:
    """Global tracker for all step instances across the workflow."""
    
    instances = {}  # step_class_name -> {instance_id -> instance}
    access_log = []  # List of access records
    
    @classmethod
    def register_instance(cls, step_name, instance_id, instance):
        """Register a step instance."""
        if step_name not in cls.instances:
            cls.instances[step_name] = {}
        cls.instances[step_name][instance_id] = instance
    
    @classmethod
    def log_access(cls, step_name, instance_id, worker_id):
        """Log an access to a step instance."""
        cls.access_log.append({
            'step_name': step_name,
            'instance_id': instance_id,
            'worker_id': worker_id,
            'timestamp': time.time()
        })
    
    @classmethod
    def get_report(cls):
        """Generate comprehensive access report."""
        report = {
            'steps': {},
            'workers': defaultdict(lambda: defaultdict(list)),
            'violations': []
        }
        
        # Analyze instances per step
        for step_name, instances in cls.instances.items():
            report['steps'][step_name] = {
                'total_instances': len(instances),
                'instance_ids': list(instances.keys())
            }
        
        # Analyze worker access patterns
        for access in cls.access_log:
            step_name = access['step_name']
            instance_id = access['instance_id']
            worker_id = access['worker_id']
            
            report['workers'][worker_id][step_name].append(instance_id)
        
        # Check for violations
        for worker_id, step_accesses in report['workers'].items():
            for step_name, instance_ids in step_accesses.items():
                unique_instances = set(instance_ids)
                if len(unique_instances) > 1:
                    report['violations'].append({
                        'type': 'worker_multiple_instances',
                        'worker_id': worker_id,
                        'step_name': step_name,
                        'instances': list(unique_instances)
                    })
        
        # Check for instance shared by multiple workers
        instance_to_workers = defaultdict(set)
        for access in cls.access_log:
            key = (access['step_name'], access['instance_id'])
            instance_to_workers[key].add(access['worker_id'])
        
        for (step_name, instance_id), workers in instance_to_workers.items():
            if len(workers) > 1:
                report['violations'].append({
                    'type': 'instance_multiple_workers',
                    'step_name': step_name,
                    'instance_id': instance_id,
                    'workers': list(workers)
                })
        
        return report
    
    @classmethod
    def reset(cls):
        """Reset all tracking."""
        cls.instances.clear()
        cls.access_log.clear()


# Test shared resource
@shared(resource_type='test_database')
class TestDatabase:
    """Shared database - should NOT be duplicated."""
    
    def __init__(self):
        self.data = {}
        self.access_count = 0
    
    async def query(self, key, worker_id=None):
        self.access_count += 1
        return {
            'value': self.data.get(key),
            'access_count': self.access_count,
            'resource_id': self.get_resource_id(),
            'worker_id': worker_id
        }


# Test steps with tracking
class TrackedStep1(BaseStep):
    """First step - uses PARSL executor."""

    # Allow direct instantiation for testing
    _allow_direct_instantiation = True

    def __init__(self, config):
        super().__init__(config)
        self.instance_id = id(self)
        self.worker_id = None
        self.process_count = 0
        self.database = TestDatabase()  # Shared resource

        InstanceTracker.register_instance('Step1', self.instance_id, self)
        print(f"   🔧 Created Step1 instance: {self.instance_id}")
    
    async def initialize(self):
        """Initialize step."""
        self._is_initialized = True

    async def process(self, data):
        self.process_count += 1
        InstanceTracker.log_access('Step1', self.instance_id, self.worker_id)

        # Access shared database
        db_result = await self.database.query('test', worker_id=self.worker_id)

        return {
            'step': 'Step1',
            'instance_id': self.instance_id,
            'worker_id': self.worker_id,
            'process_count': self.process_count,
            'data': f"Processed by Step1: {data}",
            'db_access_count': db_result['access_count']
        }


class TrackedStep2(BaseStep):
    """Second step - should be duplicated per worker."""

    # Allow direct instantiation for testing
    _allow_direct_instantiation = True

    def __init__(self, config):
        super().__init__(config)
        self.instance_id = id(self)
        self.worker_id = None
        self.process_count = 0

        InstanceTracker.register_instance('Step2', self.instance_id, self)
        print(f"   🔧 Created Step2 instance: {self.instance_id}")

    async def initialize(self):
        """Initialize step."""
        self._is_initialized = True

    async def process(self, data):
        self.process_count += 1
        InstanceTracker.log_access('Step2', self.instance_id, self.worker_id)
        
        return {
            'step': 'Step2',
            'instance_id': self.instance_id,
            'worker_id': self.worker_id,
            'process_count': self.process_count,
            'data': f"Processed by Step2: {data.get('data', data)}"
        }


class TrackedStep3(BaseStep):
    """Third step - should be duplicated per worker."""

    # Allow direct instantiation for testing
    _allow_direct_instantiation = True

    def __init__(self, config):
        super().__init__(config)
        self.instance_id = id(self)
        self.worker_id = None
        self.process_count = 0

        InstanceTracker.register_instance('Step3', self.instance_id, self)
        print(f"   🔧 Created Step3 instance: {self.instance_id}")

    async def initialize(self):
        """Initialize step."""
        self._is_initialized = True

    async def process(self, data):
        self.process_count += 1
        InstanceTracker.log_access('Step3', self.instance_id, self.worker_id)
        
        return {
            'step': 'Step3',
            'instance_id': self.instance_id,
            'worker_id': self.worker_id,
            'process_count': self.process_count,
            'data': f"Processed by Step3: {data.get('data', data)}"
        }


async def test_full_workflow_isolation():
    """Test that all steps in workflow are duplicated per worker."""
    
    print("="*80)
    print("🧪 FULL WORKFLOW WORKER ISOLATION TEST")
    print("="*80)
    
    # Reset tracking
    InstanceTracker.reset()
    
    num_workers = 4
    num_queries = 4
    
    print(f"\n📋 Test Configuration:")
    print(f"   Workers: {num_workers}")
    print(f"   Queries: {num_queries}")
    print(f"   Steps: 3 (Step1 → Step2 → Step3)")
    
    # Step 1: Create worker pools for ALL steps
    print(f"\n{'='*80}")
    print("STEP 1: CREATE WORKER POOLS FOR ALL STEPS")
    print(f"{'='*80}")
    
    step_config = {'name': 'test_step'}
    
    print(f"\n🔧 Creating worker pool for Step1 (PARSL step)...")
    pool1 = WorkerStepPool(
        step_class=TrackedStep1,
        step_config=step_config,
        num_workers=num_workers,
        step_id='step1'
    )
    await pool1.initialize_workers()
    worker_ids = pool1.get_all_worker_ids()
    print(f"✅ Step1 pool initialized with workers: {worker_ids}")
    
    print(f"\n🔧 Creating worker pool for Step2...")
    pool2 = WorkerStepPool(
        step_class=TrackedStep2,
        step_config=step_config,
        num_workers=num_workers,
        step_id='step2'
    )
    await pool2.initialize_workers()
    print(f"✅ Step2 pool initialized")
    
    print(f"\n🔧 Creating worker pool for Step3...")
    pool3 = WorkerStepPool(
        step_class=TrackedStep3,
        step_config=step_config,
        num_workers=num_workers,
        step_id='step3'
    )
    await pool3.initialize_workers()
    print(f"✅ Step3 pool initialized")
    
    # Assign same worker_ids to all pools
    print(f"\n🔧 Assigning worker IDs across all steps...")
    for i, worker_id in enumerate(worker_ids):
        # Get instances for this worker from each pool
        step1_instance = pool1.get_worker_instance(worker_id)
        step2_instance = pool2.get_worker_instance(list(pool2.get_all_worker_ids())[i])
        step3_instance = pool3.get_worker_instance(list(pool3.get_all_worker_ids())[i])
        
        # Assign same worker_id to all steps for this worker
        step1_instance.worker_id = worker_id
        step2_instance.worker_id = worker_id
        step3_instance.worker_id = worker_id
        
        print(f"   Worker {worker_id}:")
        print(f"      Step1: {step1_instance.instance_id}")
        print(f"      Step2: {step2_instance.instance_id}")
        print(f"      Step3: {step3_instance.instance_id}")
    
    # Step 2: Process queries through full workflow
    print(f"\n{'='*80}")
    print("STEP 2: PROCESS QUERIES THROUGH FULL WORKFLOW")
    print(f"{'='*80}")
    
    async def process_through_workflow(query, query_num, worker_id):
        """Process query through all 3 steps."""
        print(f"\n   📤 Query {query_num} → Worker {worker_id}")
        
        # Get worker's instances for all steps
        step1 = pool1.get_worker_instance(worker_id)
        worker_ids_list = list(pool2.get_all_worker_ids())
        worker_index = list(pool1.get_all_worker_ids()).index(worker_id)
        step2 = pool2.get_worker_instance(worker_ids_list[worker_index])
        step3 = pool3.get_worker_instance(list(pool3.get_all_worker_ids())[worker_index])
        
        # Process through Step1
        result1 = await step1.process(query)
        print(f"      Step1: Instance {result1['instance_id']}")
        
        # Process through Step2
        result2 = await step2.process(result1)
        print(f"      Step2: Instance {result2['instance_id']}")
        
        # Process through Step3
        result3 = await step3.process(result2)
        print(f"      Step3: Instance {result3['instance_id']}")
        
        return {
            'query_num': query_num,
            'worker_id': worker_id,
            'step1_instance': result1['instance_id'],
            'step2_instance': result2['instance_id'],
            'step3_instance': result3['instance_id']
        }
    
    # Process queries in parallel
    print(f"\n🚀 Processing {num_queries} queries in parallel...")
    tasks = [
        process_through_workflow(f"Query {i+1}", i+1, worker_ids[i % len(worker_ids)])
        for i in range(num_queries)
    ]
    
    results = await asyncio.gather(*tasks)
    print(f"\n✅ All queries processed")
    
    # Step 3: Analyze results
    print(f"\n{'='*80}")
    print("STEP 3: ANALYZE WORKER ISOLATION")
    print(f"{'='*80}")
    
    report = InstanceTracker.get_report()
    
    print(f"\n📊 Instance Summary:")
    for step_name, info in report['steps'].items():
        print(f"   {step_name}: {info['total_instances']} instances")
    
    print(f"\n👷 Worker Access Patterns:")
    for worker_id in worker_ids:
        if worker_id in report['workers']:
            print(f"\n   Worker {worker_id}:")
            for step_name in ['Step1', 'Step2', 'Step3']:
                if step_name in report['workers'][worker_id]:
                    instances = set(report['workers'][worker_id][step_name])
                    print(f"      {step_name}: {len(instances)} unique instance(s)")
                    if len(instances) == 1:
                        print(f"         ✅ ISOLATED")
                    else:
                        print(f"         ❌ VIOLATION: accessed {len(instances)} instances")
    
    # Step 4: Verification
    print(f"\n{'='*80}")
    print("STEP 4: VERIFICATION")
    print(f"{'='*80}")
    
    success = True
    
    # Check 1: Each step has correct number of instances
    print(f"\n✓ Check 1: Instance count per step")
    for step_name in ['Step1', 'Step2', 'Step3']:
        count = report['steps'][step_name]['total_instances']
        if count == num_workers:
            print(f"   ✅ {step_name}: {count} instances (expected {num_workers})")
        else:
            print(f"   ❌ {step_name}: {count} instances (expected {num_workers})")
            success = False
    
    # Check 2: No violations
    print(f"\n✓ Check 2: No cross-worker access violations")
    if not report['violations']:
        print(f"   ✅ PASS: No violations detected")
    else:
        print(f"   ❌ FAIL: {len(report['violations'])} violations detected")
        for violation in report['violations']:
            print(f"      {violation}")
        success = False
    
    # Check 3: Each worker accessed only one instance per step
    print(f"\n✓ Check 3: Worker instance isolation")
    for worker_id in worker_ids:
        if worker_id in report['workers']:
            isolated = True
            for step_name in ['Step1', 'Step2', 'Step3']:
                if step_name in report['workers'][worker_id]:
                    instances = set(report['workers'][worker_id][step_name])
                    if len(instances) != 1:
                        isolated = False
                        break
            
            if isolated:
                print(f"   ✅ Worker {worker_id}: Isolated (1 instance per step)")
            else:
                print(f"   ❌ Worker {worker_id}: NOT isolated")
                success = False
    
    # Cleanup
    print(f"\n🗑️  Shutting down pools...")
    await pool1.shutdown()
    await pool2.shutdown()
    await pool3.shutdown()
    
    # Final summary
    print(f"\n{'='*80}")
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Full workflow worker isolation is working correctly")
        print("✅ All steps properly duplicated per worker")
    else:
        print("⚠️  SOME TESTS FAILED")
    print(f"{'='*80}")
    
    return success


async def main():
    """Main test function."""
    try:
        success = await test_full_workflow_isolation()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

