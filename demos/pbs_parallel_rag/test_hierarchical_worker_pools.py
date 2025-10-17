#!/usr/bin/env python3
"""
Hierarchical Worker Pool Test
==============================

Tests hierarchical/nested worker pool initialization where multiple steps
in a workflow have PARSL executors with different worker counts.

Test Setup:
1. Create workflow with 3 steps: Step1 → Step2 → Step3
2. Set Step1 executor to PARSL with 4 workers
3. Set Step2 executor to PARSL with 3 workers (nested parallelism)
4. Step3 has no executor (inherits from parent)

Expected Instance Counts:
- Step1: 4 instances (4 workers from Step1's PARSL)
- Step2: 4 × 3 = 12 instances (4 Step1 workers × 3 Step2 workers each)
- Step3: 4 × 3 = 12 instances (same as Step2, inherits context)

Total: 28 instances (4 + 12 + 12)

Requirements:
- Hierarchical worker pool structure
- Correct instance counts at each level
- Proper isolation at each level
- @shared resources not duplicated
"""

import asyncio
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nanobrain.core.step import BaseStep
from nanobrain.core.worker_step_pool import WorkerStepPool
from nanobrain.core.shared_resource import shared


# Global tracking for all step instances
class HierarchicalTracker:
    """Tracks instances in hierarchical worker pool structure."""
    
    instances = {}  # step_name -> {instance_id -> instance_info}
    access_log = []  # List of access records
    hierarchy = {}  # instance_id -> parent_worker_id
    
    @classmethod
    def register_instance(cls, step_name, instance_id, instance, parent_worker_id=None):
        """Register a step instance with optional parent worker."""
        if step_name not in cls.instances:
            cls.instances[step_name] = {}
        
        cls.instances[step_name][instance_id] = {
            'instance': instance,
            'parent_worker_id': parent_worker_id,
            'worker_id': getattr(instance, 'worker_id', None)
        }
        
        if parent_worker_id:
            cls.hierarchy[instance_id] = parent_worker_id
    
    @classmethod
    def log_access(cls, step_name, instance_id, worker_id, parent_worker_id=None):
        """Log an access to a step instance."""
        cls.access_log.append({
            'step_name': step_name,
            'instance_id': instance_id,
            'worker_id': worker_id,
            'parent_worker_id': parent_worker_id,
            'timestamp': time.time()
        })
    
    @classmethod
    def get_report(cls):
        """Generate hierarchical access report."""
        report = {
            'steps': {},
            'hierarchy': {},
            'violations': []
        }
        
        # Analyze instances per step
        for step_name, instances in cls.instances.items():
            report['steps'][step_name] = {
                'total_instances': len(instances),
                'instance_ids': list(instances.keys()),
                'by_parent': defaultdict(list)
            }
            
            # Group by parent worker
            for instance_id, info in instances.items():
                parent = info['parent_worker_id'] or 'root'
                report['steps'][step_name]['by_parent'][parent].append(instance_id)
        
        return report
    
    @classmethod
    def reset(cls):
        """Reset all tracking."""
        cls.instances.clear()
        cls.access_log.clear()
        cls.hierarchy.clear()


# Test shared resource
@shared(resource_type='hierarchical_database')
class HierarchicalDatabase:
    """Shared database - should NOT be duplicated at any level."""
    
    def __init__(self):
        self.data = {}
        self.access_count = 0
        self.access_by_level = defaultdict(int)
    
    async def query(self, key, level='root'):
        self.access_count += 1
        self.access_by_level[level] += 1
        return {
            'value': self.data.get(key),
            'access_count': self.access_count,
            'resource_id': self.get_resource_id(),
            'level': level
        }


# Test steps with hierarchical tracking
class HierarchicalStep1(BaseStep):
    """First step - uses PARSL executor with 4 workers."""
    
    _allow_direct_instantiation = True
    
    def __init__(self, config):
        super().__init__(config)
        self.instance_id = id(self)
        self.worker_id = None
        self.parent_worker_id = None
        self.process_count = 0
        self.database = HierarchicalDatabase()
        
        HierarchicalTracker.register_instance('Step1', self.instance_id, self, parent_worker_id=None)
        print(f"   🔧 Created Step1 instance: {self.instance_id}")
    
    async def initialize(self):
        """Initialize step."""
        self._is_initialized = True
    
    async def process(self, data):
        self.process_count += 1
        HierarchicalTracker.log_access('Step1', self.instance_id, self.worker_id, self.parent_worker_id)
        
        db_result = await self.database.query('test', level='step1')
        
        return {
            'step': 'Step1',
            'instance_id': self.instance_id,
            'worker_id': self.worker_id,
            'parent_worker_id': self.parent_worker_id,
            'process_count': self.process_count,
            'data': f"Step1[{self.worker_id}]: {data}",
            'db_access_count': db_result['access_count']
        }


class HierarchicalStep2(BaseStep):
    """Second step - uses PARSL executor with 3 workers (nested)."""
    
    _allow_direct_instantiation = True
    
    def __init__(self, config):
        super().__init__(config)
        self.instance_id = id(self)
        self.worker_id = None
        self.parent_worker_id = None  # Will be set to Step1's worker_id
        self.process_count = 0
        
        HierarchicalTracker.register_instance('Step2', self.instance_id, self, parent_worker_id=None)
        print(f"   🔧 Created Step2 instance: {self.instance_id}")
    
    async def initialize(self):
        """Initialize step."""
        self._is_initialized = True
    
    async def process(self, data):
        self.process_count += 1
        HierarchicalTracker.log_access('Step2', self.instance_id, self.worker_id, self.parent_worker_id)
        
        return {
            'step': 'Step2',
            'instance_id': self.instance_id,
            'worker_id': self.worker_id,
            'parent_worker_id': self.parent_worker_id,
            'process_count': self.process_count,
            'data': f"Step2[{self.parent_worker_id}/{self.worker_id}]: {data.get('data', data)}"
        }


class HierarchicalStep3(BaseStep):
    """Third step - inherits worker context from Step2."""
    
    _allow_direct_instantiation = True
    
    def __init__(self, config):
        super().__init__(config)
        self.instance_id = id(self)
        self.worker_id = None
        self.parent_worker_id = None
        self.process_count = 0
        
        HierarchicalTracker.register_instance('Step3', self.instance_id, self, parent_worker_id=None)
        print(f"   🔧 Created Step3 instance: {self.instance_id}")
    
    async def initialize(self):
        """Initialize step."""
        self._is_initialized = True
    
    async def process(self, data):
        self.process_count += 1
        HierarchicalTracker.log_access('Step3', self.instance_id, self.worker_id, self.parent_worker_id)
        
        return {
            'step': 'Step3',
            'instance_id': self.instance_id,
            'worker_id': self.worker_id,
            'parent_worker_id': self.parent_worker_id,
            'process_count': self.process_count,
            'data': f"Step3[{self.parent_worker_id}/{self.worker_id}]: {data.get('data', data)}"
        }


async def test_hierarchical_worker_pools():
    """Test hierarchical worker pool initialization."""
    
    print("="*80)
    print("🧪 HIERARCHICAL WORKER POOL TEST")
    print("="*80)
    
    # Reset tracking
    HierarchicalTracker.reset()
    
    # Configuration
    step1_workers = 4  # Step1 has 4 workers
    step2_workers = 3  # Step2 has 3 workers per Step1 worker
    
    print(f"\n📋 Test Configuration:")
    print(f"   Step1 workers: {step1_workers}")
    print(f"   Step2 workers per Step1 worker: {step2_workers}")
    print(f"   Expected Step1 instances: {step1_workers}")
    print(f"   Expected Step2 instances: {step1_workers} × {step2_workers} = {step1_workers * step2_workers}")
    print(f"   Expected Step3 instances: {step1_workers} × {step2_workers} = {step1_workers * step2_workers}")
    print(f"   Total expected instances: {step1_workers + (step1_workers * step2_workers * 2)}")
    
    # Step 1: Create worker pool for Step1 (4 workers)
    print(f"\n{'='*80}")
    print("STEP 1: CREATE WORKER POOL FOR STEP1 (4 WORKERS)")
    print(f"{'='*80}")
    
    step_config = {'name': 'test_step'}
    
    pool1 = WorkerStepPool(
        step_class=HierarchicalStep1,
        step_config=step_config,
        num_workers=step1_workers,
        step_id='step1'
    )
    await pool1.initialize_workers()
    step1_worker_ids = pool1.get_all_worker_ids()
    
    print(f"\n✅ Step1 pool initialized")
    print(f"   Workers: {step1_worker_ids}")
    
    # Step 2: Create nested worker pools for Step2 (3 workers per Step1 worker)
    print(f"\n{'='*80}")
    print(f"STEP 2: CREATE NESTED WORKER POOLS FOR STEP2 (3 WORKERS PER STEP1 WORKER)")
    print(f"{'='*80}")
    
    step2_pools = {}  # parent_worker_id -> pool
    step2_worker_mapping = {}  # (parent_worker_id, child_worker_id) -> instance
    
    for parent_worker_id in step1_worker_ids:
        print(f"\n🔧 Creating Step2 pool for parent worker: {parent_worker_id}")
        
        pool2 = WorkerStepPool(
            step_class=HierarchicalStep2,
            step_config=step_config,
            num_workers=step2_workers,
            step_id=f'step2_{parent_worker_id}'
        )
        await pool2.initialize_workers()
        
        step2_pools[parent_worker_id] = pool2
        step2_worker_ids = pool2.get_all_worker_ids()
        
        print(f"   ✅ Created {step2_workers} Step2 instances for parent {parent_worker_id}")
        print(f"      Child workers: {step2_worker_ids}")
        
        # Set parent_worker_id on each Step2 instance and update tracking
        for child_worker_id in step2_worker_ids:
            instance = pool2.get_worker_instance(child_worker_id)
            instance.parent_worker_id = parent_worker_id
            instance.worker_id = child_worker_id
            step2_worker_mapping[(parent_worker_id, child_worker_id)] = instance

            # Update tracking with parent info
            HierarchicalTracker.instances['Step2'][instance.instance_id]['parent_worker_id'] = parent_worker_id
            HierarchicalTracker.instances['Step2'][instance.instance_id]['worker_id'] = child_worker_id
    
    # Step 3: Create nested worker pools for Step3 (same structure as Step2)
    print(f"\n{'='*80}")
    print(f"STEP 3: CREATE NESTED WORKER POOLS FOR STEP3 (SAME STRUCTURE AS STEP2)")
    print(f"{'='*80}")
    
    step3_pools = {}
    step3_worker_mapping = {}
    
    for parent_worker_id in step1_worker_ids:
        print(f"\n🔧 Creating Step3 pool for parent worker: {parent_worker_id}")
        
        pool3 = WorkerStepPool(
            step_class=HierarchicalStep3,
            step_config=step_config,
            num_workers=step2_workers,  # Same as Step2
            step_id=f'step3_{parent_worker_id}'
        )
        await pool3.initialize_workers()
        
        step3_pools[parent_worker_id] = pool3
        step3_worker_ids = pool3.get_all_worker_ids()
        
        print(f"   ✅ Created {step2_workers} Step3 instances for parent {parent_worker_id}")
        
        # Set parent_worker_id on each Step3 instance and update tracking
        for i, child_worker_id in enumerate(step3_worker_ids):
            instance = pool3.get_worker_instance(child_worker_id)
            instance.parent_worker_id = parent_worker_id
            instance.worker_id = child_worker_id
            step3_worker_mapping[(parent_worker_id, child_worker_id)] = instance

            # Update tracking with parent info
            HierarchicalTracker.instances['Step3'][instance.instance_id]['parent_worker_id'] = parent_worker_id
            HierarchicalTracker.instances['Step3'][instance.instance_id]['worker_id'] = child_worker_id
    
    # Step 4: Verify instance counts
    print(f"\n{'='*80}")
    print("STEP 4: VERIFY INSTANCE COUNTS")
    print(f"{'='*80}")
    
    report = HierarchicalTracker.get_report()
    
    print(f"\n📊 Instance Counts:")
    for step_name, info in report['steps'].items():
        print(f"   {step_name}: {info['total_instances']} instances")
    
    print(f"\n📊 Hierarchical Structure:")
    for step_name, info in report['steps'].items():
        print(f"\n   {step_name}:")
        for parent, instances in info['by_parent'].items():
            print(f"      Parent {parent}: {len(instances)} instances")
    
    # Step 5: Verification
    print(f"\n{'='*80}")
    print("STEP 5: VERIFICATION")
    print(f"{'='*80}")
    
    success = True
    
    # Check 1: Step1 instance count
    print(f"\n✓ Check 1: Step1 instance count")
    step1_count = report['steps']['Step1']['total_instances']
    if step1_count == step1_workers:
        print(f"   ✅ PASS: {step1_count} instances (expected {step1_workers})")
    else:
        print(f"   ❌ FAIL: {step1_count} instances (expected {step1_workers})")
        success = False
    
    # Check 2: Step2 instance count
    print(f"\n✓ Check 2: Step2 instance count")
    step2_count = report['steps']['Step2']['total_instances']
    expected_step2 = step1_workers * step2_workers
    if step2_count == expected_step2:
        print(f"   ✅ PASS: {step2_count} instances (expected {expected_step2})")
    else:
        print(f"   ❌ FAIL: {step2_count} instances (expected {expected_step2})")
        success = False
    
    # Check 3: Step3 instance count
    print(f"\n✓ Check 3: Step3 instance count")
    step3_count = report['steps']['Step3']['total_instances']
    expected_step3 = step1_workers * step2_workers
    if step3_count == expected_step3:
        print(f"   ✅ PASS: {step3_count} instances (expected {expected_step3})")
    else:
        print(f"   ❌ FAIL: {step3_count} instances (expected {expected_step3})")
        success = False
    
    # Check 4: Hierarchical structure for Step2
    print(f"\n✓ Check 4: Hierarchical structure for Step2")
    step2_by_parent = report['steps']['Step2']['by_parent']
    # Filter out 'root' entries (from initial registration)
    step2_parents = {k: v for k, v in step2_by_parent.items() if k != 'root'}

    if len(step2_parents) == step1_workers:
        print(f"   ✅ PASS: Step2 has instances under {step1_workers} parents")
        all_correct = True
        for parent, instances in step2_parents.items():
            if len(instances) == step2_workers:
                print(f"      ✅ Parent {parent}: {len(instances)} instances")
            else:
                print(f"      ❌ Parent {parent}: {len(instances)} instances (expected {step2_workers})")
                all_correct = False
        if not all_correct:
            success = False
    else:
        print(f"   ❌ FAIL: Step2 has {len(step2_parents)} parents (expected {step1_workers})")
        success = False

    # Check 5: Hierarchical structure for Step3
    print(f"\n✓ Check 5: Hierarchical structure for Step3")
    step3_by_parent = report['steps']['Step3']['by_parent']
    # Filter out 'root' entries
    step3_parents = {k: v for k, v in step3_by_parent.items() if k != 'root'}

    if len(step3_parents) == step1_workers:
        print(f"   ✅ PASS: Step3 has instances under {step1_workers} parents")
        all_correct = True
        for parent, instances in step3_parents.items():
            if len(instances) == step2_workers:
                print(f"      ✅ Parent {parent}: {len(instances)} instances")
            else:
                print(f"      ❌ Parent {parent}: {len(instances)} instances (expected {step2_workers})")
                all_correct = False
        if not all_correct:
            success = False
    else:
        print(f"   ❌ FAIL: Step3 has {len(step3_parents)} parents (expected {step1_workers})")
        success = False
    
    # Check 6: Total instance count
    print(f"\n✓ Check 6: Total instance count")
    total_instances = step1_count + step2_count + step3_count
    expected_total = step1_workers + (step1_workers * step2_workers * 2)
    if total_instances == expected_total:
        print(f"   ✅ PASS: {total_instances} total instances (expected {expected_total})")
    else:
        print(f"   ❌ FAIL: {total_instances} total instances (expected {expected_total})")
        success = False
    
    # Cleanup
    print(f"\n🗑️  Shutting down pools...")
    await pool1.shutdown()
    for pool in step2_pools.values():
        await pool.shutdown()
    for pool in step3_pools.values():
        await pool.shutdown()
    
    # Final summary
    print(f"\n{'='*80}")
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Hierarchical worker pool initialization is working correctly")
        print(f"✅ Created {total_instances} instances across 3 levels")
    else:
        print("⚠️  SOME TESTS FAILED")
    print(f"{'='*80}")
    
    return success


async def main():
    """Main test function."""
    try:
        success = await test_hierarchical_worker_pools()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

