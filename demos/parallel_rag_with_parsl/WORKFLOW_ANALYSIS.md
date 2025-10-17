# Parallel RAG with PARSL Workflow Analysis 🔍

**Date**: October 7, 2025  
**Purpose**: Analyze current implementation and design per-worker step instances  
**Status**: 🔄 **IN PROGRESS**

---

## 🎯 CURRENT IMPLEMENTATION ANALYSIS

### Current Architecture

```
Workflow
  └── Step (Single Instance)
       └── PARSL Executor
            ├── Worker 1 (executes task)
            ├── Worker 2 (executes task)
            ├── Worker 3 (executes task)
            └── Worker 4 (executes task)
```

**Problem**: All workers share the same step instance, which can cause:
- State conflicts between workers
- Race conditions on instance variables
- Difficulty tracking per-worker state
- Shared resources not properly isolated

---

## 🎯 REQUIRED ARCHITECTURE

### Per-Worker Step Instances

```
Workflow
  └── PARSL Executor (Coordinator)
       ├── Worker 1
       │    └── Step Instance 1 (worker_id: worker_abc123)
       │         ├── Agent Instance 1
       │         ├── Data Units 1
       │         └── Shared Resources (via @shared decorator)
       │
       ├── Worker 2
       │    └── Step Instance 2 (worker_id: worker_def456)
       │         ├── Agent Instance 2
       │         ├── Data Units 2
       │         └── Shared Resources (same instances)
       │
       ├── Worker 3
       │    └── Step Instance 3 (worker_id: worker_ghi789)
       │         ├── Agent Instance 3
       │         ├── Data Units 3
       │         └── Shared Resources (same instances)
       │
       └── Worker 4
            └── Step Instance 4 (worker_id: worker_jkl012)
                 ├── Agent Instance 4
                 ├── Data Units 4
                 └── Shared Resources (same instances)
```

**Key Requirements**:
1. ✅ Each PARSL worker has its own step instance
2. ✅ Each step instance has unique worker_id
3. ✅ Objects marked with @shared are NOT duplicated
4. ✅ Non-shared objects are duplicated per worker
5. ✅ Worker instances are created when PARSL executor is set up

---

## 📊 CURRENT ISSUES

### Issue 1: Single Step Instance

**Current Code** (`parallel_query_enhancement_step.py`):
```python
class ParallelQueryEnhancementStep(QueryEnhancementStep):
    def __init__(self, config):
        super().__init__(config)
        self.parsl_executor = None  # Single instance
    
    async def process(self, *args, **kwargs):
        # All workers execute through same instance
        result = await self.parsl_executor.execute(
            lambda: super().process(*args, **kwargs)
        )
```

**Problem**: 
- Single step instance shared by all workers
- No per-worker state isolation
- Potential race conditions

### Issue 2: No Worker-Specific Instances

**Current Execution Flow**:
```
Query 1 → PARSL Worker 1 → Same Step Instance → Process
Query 2 → PARSL Worker 2 → Same Step Instance → Process
Query 3 → PARSL Worker 3 → Same Step Instance → Process
Query 4 → PARSL Worker 4 → Same Step Instance → Process
```

**Problem**:
- Workers compete for same instance
- No worker-specific configuration
- Difficult to track per-worker state

### Issue 3: Shared Resources Not Properly Marked

**Current Implementation**:
- Vector databases not marked with @shared
- LLM clients not marked with @shared
- Caches not marked with @shared

**Problem**:
- Resources duplicated unnecessarily
- Memory waste
- Inconsistent state across workers

---

## 🔧 REQUIRED SOLUTION

### 1. Per-Worker Step Instance Pool

**New Component**: `WorkerStepPool`

```python
class WorkerStepPool:
    """
    Manages per-worker step instances for PARSL execution.
    
    Features:
    - Creates one step instance per PARSL worker
    - Assigns unique worker_id to each instance
    - Shares @shared resources across instances
    - Isolates non-shared state per worker
    """
    
    def __init__(self, step_class, step_config, num_workers):
        self.step_class = step_class
        self.step_config = step_config
        self.num_workers = num_workers
        self.worker_instances = {}  # worker_id -> step_instance
        self.shared_resources = {}  # resource_id -> shared_resource
    
    async def initialize_workers(self):
        """Create and initialize step instance for each worker."""
        for i in range(self.num_workers):
            worker_id = f"worker_{uuid.uuid4().hex[:8]}"
            
            # Create step instance
            step_instance = self.step_class(self.step_config)
            step_instance.worker_id = worker_id
            
            # Identify and share @shared resources
            await self._setup_shared_resources(step_instance)
            
            # Initialize step
            await step_instance.initialize()
            
            self.worker_instances[worker_id] = step_instance
    
    def get_worker_instance(self, worker_id):
        """Get step instance for specific worker."""
        return self.worker_instances.get(worker_id)
```

### 2. Enhanced PARSL Executor

**Modified**: `ParslExecutor.execute()`

```python
class ParslExecutor(ExecutorBase):
    def __init__(self, config):
        super().__init__(config)
        self.worker_step_pools = {}  # step_id -> WorkerStepPool
    
    async def setup_worker_instances(self, step_class, step_config, step_id):
        """Set up per-worker instances for a step."""
        pool = WorkerStepPool(
            step_class=step_class,
            step_config=step_config,
            num_workers=self.config.max_workers
        )
        await pool.initialize_workers()
        self.worker_step_pools[step_id] = pool
    
    async def execute(self, task, step_id=None, **kwargs):
        """Execute task with worker-specific step instance."""
        # Generate worker ID
        worker_id = f"parsl_worker_{uuid.uuid4().hex[:8]}"
        
        # Get worker-specific step instance if available
        if step_id and step_id in self.worker_step_pools:
            pool = self.worker_step_pools[step_id]
            step_instance = pool.get_worker_instance(worker_id)
            
            # Execute with worker-specific instance
            result = await self._execute_with_instance(
                task, step_instance, worker_id
            )
        else:
            # Fallback to standard execution
            result = await self.submit(task, **kwargs)
        
        return result
```

### 3. Shared Resource Detection

**New Function**: `detect_shared_resources()`

```python
def detect_shared_resources(obj):
    """
    Detect all @shared resources in an object.
    
    Returns:
        Dict[str, Any]: resource_id -> resource_instance
    """
    shared_resources = {}
    
    # Inspect all attributes
    for attr_name in dir(obj):
        try:
            attr = getattr(obj, attr_name)
            
            # Check if marked with @shared
            if hasattr(attr.__class__, '_is_shared_resource'):
                resource_id = attr.get_resource_id()
                shared_resources[resource_id] = attr
        except:
            pass
    
    return shared_resources
```

---

## 📋 IMPLEMENTATION PLAN

### Phase 1: Core Infrastructure

1. ✅ Create `WorkerStepPool` class
2. ✅ Create `detect_shared_resources()` function
3. ✅ Enhance `ParslExecutor` with worker instance management
4. ✅ Add worker_id tracking to step instances

### Phase 2: Step Integration

5. ✅ Modify `ParallelQueryEnhancementStep` to support per-worker instances
6. ✅ Update workflow initialization to set up worker pools
7. ✅ Ensure @shared resources are properly detected and shared

### Phase 3: Testing

8. ✅ Test per-worker instance creation
9. ✅ Verify @shared resources are not duplicated
10. ✅ Verify non-shared resources are duplicated
11. ✅ Test worker_id tracking

### Phase 4: Documentation

12. ✅ Document new architecture
13. ✅ Create usage examples
14. ✅ Update existing tests

---

## 🎯 EXPECTED BEHAVIOR

### After Implementation

**Workflow Initialization**:
```python
# When PARSL executor is set up
workflow = Workflow.from_config('parallel_rag_workflow.yml')
await workflow.initialize()

# PARSL executor creates worker instances
# For each step with PARSL executor:
#   - Create 4 step instances (one per worker)
#   - Assign unique worker_id to each
#   - Share @shared resources across instances
#   - Initialize each instance
```

**Query Execution**:
```python
# Query 1 → Worker 1 → Step Instance 1 (worker_abc123)
# Query 2 → Worker 2 → Step Instance 2 (worker_def456)
# Query 3 → Worker 3 → Step Instance 3 (worker_ghi789)
# Query 4 → Worker 4 → Step Instance 4 (worker_jkl012)

# Each instance has:
# - Unique worker_id
# - Own agent instance
# - Own data units
# - Shared vector database (via @shared)
# - Shared LLM client (via @shared)
```

---

## ✅ SUCCESS CRITERIA

1. ✅ Each PARSL worker has dedicated step instance
2. ✅ Each instance has unique worker_id
3. ✅ @shared resources are NOT duplicated
4. ✅ Non-shared resources ARE duplicated per worker
5. ✅ Worker instances created during PARSL setup
6. ✅ All tests pass with new architecture
7. ✅ Performance maintained or improved

---

## 📊 COMPARISON

### Before (Current)

```
Memory Usage:
- 1 step instance
- 1 agent instance
- 1 vector database (not shared)
- 1 LLM client (not shared)

Execution:
- All workers share same instance
- Potential race conditions
- No worker-specific state
```

### After (Proposed)

```
Memory Usage:
- 4 step instances (one per worker)
- 4 agent instances (one per worker)
- 1 vector database (@shared)
- 1 LLM client (@shared)

Execution:
- Each worker has own instance
- No race conditions
- Worker-specific state isolated
- Shared resources properly pooled
```

---

**Status**: 🔄 **ANALYSIS COMPLETE - READY FOR IMPLEMENTATION**  
**Next**: Implement WorkerStepPool and enhanced PARSL executor

