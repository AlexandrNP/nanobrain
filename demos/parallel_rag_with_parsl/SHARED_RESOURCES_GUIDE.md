# Shared Resources Guide for Parallel RAG 🔧

**Date**: October 7, 2025  
**Feature**: Shared Resource Management with @shared Decorator  
**Status**: ✅ **IMPLEMENTED AND TESTED**

---

## 🎯 OVERVIEW

This guide documents the shared resource management system for the Nanobrain framework, specifically designed for parallel execution scenarios where multiple PARSL workers need to access shared resources like vector databases.

### Key Features

- ✅ **@shared Decorator** - Mark classes as shared resources
- ✅ **Automatic Resource Pooling** - Global pool manages all shared resources
- ✅ **Worker ID Tracking** - Track which worker accesses which resource
- ✅ **Thread-Safe Access** - Safe concurrent access from multiple workers
- ✅ **PARSL Integration** - Worker IDs added to PARSL executor outputs
- ✅ **Statistics & Monitoring** - Track access patterns and usage

---

## 📋 PROBLEM STATEMENT

### The Challenge

In parallel RAG workflows with PARSL executor:

1. **Multiple workers** process queries simultaneously
2. **Shared resources** (like vector databases) should be initialized once
3. **Concurrent access** must be thread-safe
4. **Worker tracking** is needed for debugging and monitoring
5. **Resource lifecycle** must be managed properly

### The Solution

**@shared Decorator + Resource Pool + Worker ID Tracking**

```python
@shared(resource_type='vector_database')
class VectorDatabase:
    # Initialized once, shared by all workers
    # Automatic pooling and tracking
    # Worker IDs tracked for all operations
    pass
```

---

## 🏗️ ARCHITECTURE

### Components

1. **SharedResourcePool** - Global singleton managing all shared resources
2. **@shared Decorator** - Marks classes as shared resources
3. **WorkerContext** - Tracks worker information
4. **get_worker_id()** - Generates unique worker IDs
5. **PARSL Executor Enhancement** - Adds worker IDs to outputs

### Data Flow

```
Query 1 → PARSL Worker 1 → Shared Vector DB → Results + Worker ID
Query 2 → PARSL Worker 2 → Shared Vector DB → Results + Worker ID
Query 3 → PARSL Worker 3 → Shared Vector DB → Results + Worker ID
Query 4 → PARSL Worker 4 → Shared Vector DB → Results + Worker ID
                              ↑
                    Single Shared Instance
                    (Thread-Safe Access)
```

---

## 🔧 IMPLEMENTATION

### 1. Creating a Shared Resource

```python
from nanobrain.core.shared_resource import shared, get_worker_id

@shared(resource_type='vector_database', auto_register=True)
class SharedVectorDatabase:
    """Vector database shared across all PARSL workers."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors = {}
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize once, shared by all workers."""
        if self.is_initialized:
            return
        
        # Build index (expensive operation done once)
        self.index = await self._build_index()
        self.is_initialized = True
    
    async def search(self, query, worker_id=None):
        """Search - called by multiple workers."""
        if worker_id is None:
            worker_id = get_worker_id()
        
        results = await self.index.search(query)
        
        return {
            'results': results,
            'worker_id': worker_id,  # Track which worker did this
            'resource_id': self.get_resource_id()
        }
```

### 2. Using Shared Resources

```python
# Create and initialize shared resource
vector_db = SharedVectorDatabase(dimension=384)
await vector_db.initialize()

# Multiple workers can access it
async def worker_task(query, worker_id):
    # Each worker gets the same shared instance
    results = await vector_db.search(query, worker_id=worker_id)
    return results

# Run in parallel
tasks = [worker_task(query, f"worker_{i}") for i in range(4)]
results = await asyncio.gather(*tasks)

# Each result has worker_id tracking
for result in results:
    print(f"Worker {result['worker_id']}: {len(result['results'])} results")
```

### 3. PARSL Executor with Worker IDs

```python
from nanobrain.core.executor import ParslExecutor, ExecutorConfig

# Create PARSL executor
executor_config = ExecutorConfig(**parsl_config)
parsl_executor = ParslExecutor(config=executor_config)
await parsl_executor.initialize()

# Execute with worker ID tracking
result = await parsl_executor.execute(
    lambda: some_task(),
    add_worker_id=True  # Enable worker ID tracking
)

# Result includes worker metadata
print(f"Worker ID: {result['_worker_id']}")
print(f"Executor: {result['_executor_type']}")
print(f"Result: {result['result']}")
```

---

## 📊 RESOURCE POOL MANAGEMENT

### Accessing the Resource Pool

```python
from nanobrain.core.shared_resource import get_resource_pool

# Get global pool
pool = get_resource_pool()

# List all resources
resources = pool.list_resources()
for res_id, stats in resources.items():
    print(f"{res_id}: {stats['access_count']} accesses")

# Get specific resource
vector_db = pool.get_resource(
    resource_id='vector_database_abc123',
    worker_id='worker_1',
    worker_type='parsl'
)

# Get statistics
stats = pool.get_resource_stats('vector_database_abc123')
print(f"Active workers: {stats['active_workers']}")
print(f"Total accesses: {stats['access_count']}")
```

### Resource Statistics

```python
# Get stats for a shared resource
stats = vector_db.get_resource_stats()

print(f"Resource ID: {stats['resource_id']}")
print(f"Resource Type: {stats['resource_type']}")
print(f"Access Count: {stats['access_count']}")
print(f"Active Workers: {stats['active_workers']}")
print(f"Worker IDs: {stats['worker_ids']}")
```

---

## 🎯 USE CASES

### 1. Shared Vector Database

**Scenario**: Multiple PARSL workers need to search the same vector database

```python
@shared(resource_type='vector_database')
class VectorDB:
    async def initialize(self):
        # Load index once (expensive)
        self.index = await load_faiss_index()
    
    async def search(self, query, worker_id=None):
        # Multiple workers search concurrently
        return await self.index.search(query)
```

**Benefits**:
- ✅ Index loaded once, not per worker
- ✅ Memory efficient (single copy)
- ✅ Thread-safe concurrent access
- ✅ Worker tracking for debugging

### 2. Shared LLM Client

**Scenario**: Multiple workers share an LLM API client

```python
@shared(resource_type='llm_client')
class LLMClient:
    async def initialize(self):
        # Initialize API client once
        self.client = OpenAI(api_key=...)
    
    async def generate(self, prompt, worker_id=None):
        # Multiple workers use same client
        response = await self.client.chat.completions.create(...)
        return {
            'response': response,
            'worker_id': worker_id
        }
```

**Benefits**:
- ✅ Single API client (connection pooling)
- ✅ Rate limiting coordination
- ✅ Cost tracking per worker

### 3. Shared Cache

**Scenario**: Workers share a cache for embeddings

```python
@shared(resource_type='embedding_cache')
class EmbeddingCache:
    def __init__(self):
        self.cache = {}
    
    async def get_or_compute(self, text, worker_id=None):
        if text in self.cache:
            return self.cache[text]
        
        embedding = await compute_embedding(text)
        self.cache[text] = embedding
        return embedding
```

**Benefits**:
- ✅ Avoid recomputing embeddings
- ✅ Memory efficient
- ✅ Faster processing

---

## 📈 MONITORING & DEBUGGING

### Worker ID Tracking

Every operation tracks which worker performed it:

```python
result = await vector_db.search(query, worker_id=get_worker_id())

print(f"Worker: {result['worker_id']}")
print(f"Resource: {result['resource_id']}")
print(f"Access count: {result['access_count']}")
```

### Access Logs

Shared resources maintain access logs:

```python
stats = vector_db.get_stats()

print(f"Total accesses: {stats['database_stats']['access_count']}")
print(f"Unique workers: {stats['database_stats']['unique_workers']}")

# Recent accesses
for access in stats['recent_accesses']:
    print(f"Worker {access['worker_id']}: {access['operation']}")
```

### Resource Pool Statistics

```python
pool = get_resource_pool()

# List all resources
for res_id, stats in pool.list_resources().items():
    print(f"\nResource: {res_id}")
    print(f"  Type: {stats['resource_type']}")
    print(f"  Accesses: {stats['access_count']}")
    print(f"  Active workers: {stats['active_workers']}")
    print(f"  Worker IDs: {stats['worker_ids']}")
```

---

## ✅ TESTING

### Test 1: Shared Vector Database

```bash
python demos/parallel_rag_demo/shared_vector_database.py
```

**Expected Output**:
```
✅ Created shared vector database: vector_database_a49d9aaf
Worker worker_0: Added 10 vectors
Worker worker_1: Added 10 vectors
Worker worker_2: Added 10 vectors
Worker worker_3: Added 10 vectors

📊 Final Statistics:
   Total vectors: 40
   Total accesses: 8
   Unique workers: 4
```

### Test 2: PARSL with Shared Resources

```bash
python demos/parallel_rag_demo/test_shared_resources.py
```

**Tests**:
1. ✅ Shared vector database with multiple workers
2. ✅ PARSL executor worker ID tracking
3. ✅ Shared resource accessed by PARSL workers

---

## 🏁 CONCLUSION

### Summary

The shared resource system provides:

- ✅ **@shared decorator** for easy resource marking
- ✅ **Automatic pooling** via global resource pool
- ✅ **Worker ID tracking** in PARSL executor
- ✅ **Thread-safe access** for concurrent workers
- ✅ **Statistics & monitoring** for debugging

### Benefits

1. **Memory Efficient** - Resources initialized once
2. **Performance** - No redundant initialization
3. **Debugging** - Worker ID tracking
4. **Monitoring** - Access statistics
5. **Thread-Safe** - Concurrent access handled

### Integration with Nanobrain

- ✅ Works with data unit→link→trigger execution strategy
- ✅ Compatible with PARSL executor
- ✅ Follows framework patterns
- ✅ Production ready

---

**Status**: ✅ **IMPLEMENTED AND TESTED**

**Ready for Production**: ✅ **YES**

