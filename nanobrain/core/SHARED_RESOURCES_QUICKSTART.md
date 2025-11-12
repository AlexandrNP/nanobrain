# Shared Resources Quick Start Guide 🚀

**For**: Nanobrain Framework Developers  
**Feature**: @shared Decorator and Resource Pooling  
**Version**: 1.0

---

## 🎯 QUICK START

### 1. Mark a Class as Shared

```python
from nanobrain.core.shared_resource import shared

@shared(resource_type='vector_database')
class VectorDatabase:
    def __init__(self, config):
        self.config = config
        self.index = None
    
    async def initialize(self):
        # Expensive initialization - done once
        self.index = await build_index()
```

### 2. Use the Shared Resource

```python
# Create and initialize
vector_db = VectorDatabase(config)
await vector_db.initialize()

# Multiple workers can access it
result = await vector_db.search(query, worker_id=get_worker_id())
```

### 3. Track Worker IDs

```python
from nanobrain.core.shared_resource import get_worker_id

# In your processing function
worker_id = get_worker_id()

result = {
    'data': processed_data,
    'worker_id': worker_id  # Track which worker did this
}
```

---

## 📋 COMMON PATTERNS

### Pattern 1: Shared Vector Database

```python
@shared(resource_type='vector_database')
class SharedVectorDB:
    async def search(self, query, worker_id=None):
        if worker_id is None:
            worker_id = get_worker_id()
        
        results = await self.index.search(query)
        return {
            'results': results,
            'worker_id': worker_id,
            'resource_id': self.get_resource_id()
        }
```

### Pattern 2: Shared LLM Client

```python
@shared(resource_type='llm_client')
class SharedLLMClient:
    async def generate(self, prompt, worker_id=None):
        if worker_id is None:
            worker_id = get_worker_id()
        
        response = await self.client.generate(prompt)
        return {
            'response': response,
            'worker_id': worker_id
        }
```

### Pattern 3: Shared Cache

```python
@shared(resource_type='cache')
class SharedCache:
    def __init__(self):
        self.cache = {}
        self._lock = Lock()
    
    def get_or_compute(self, key, compute_fn, worker_id=None):
        with self._lock:
            if key in self.cache:
                return self.cache[key]
            
            value = compute_fn()
            self.cache[key] = value
            return value
```

---

## 🔧 PARSL INTEGRATION

### Enable Worker ID Tracking

```python
from nanobrain.core.executor import ParslExecutor

# Execute with worker ID tracking
result = await parsl_executor.execute(
    task,
    add_worker_id=True  # Adds _worker_id to result
)

# Result includes metadata
print(result['_worker_id'])      # "parsl_worker_abc123"
print(result['_executor_type'])  # "parsl"
```

### Custom Step with PARSL

```python
class MyParallelStep(BaseStep):
    async def initialize(self):
        # Create PARSL executor
        self.parsl_executor = ParslExecutor(config=...)
        await self.parsl_executor.initialize()
    
    async def process(self, input_data):
        # Execute with worker ID tracking
        result = await self.parsl_executor.execute(
            lambda: self._process_task(input_data),
            add_worker_id=True
        )
        return result
```

---

## 📊 MONITORING

### Get Resource Statistics

```python
# For a specific resource
stats = vector_db.get_resource_stats()
print(f"Accesses: {stats['access_count']}")
print(f"Workers: {stats['active_workers']}")

# For all resources
from nanobrain.core.shared_resource import get_resource_pool

pool = get_resource_pool()
for res_id, stats in pool.list_resources().items():
    print(f"{res_id}: {stats['access_count']} accesses")
```

### Track Worker Access

```python
# In your shared resource
class MySharedResource:
    def __init__(self):
        self.access_log = []
    
    async def process(self, data, worker_id=None):
        if worker_id is None:
            worker_id = get_worker_id()
        
        # Log access
        self.access_log.append({
            'worker_id': worker_id,
            'timestamp': datetime.now(),
            'operation': 'process'
        })
        
        # Process data
        result = await self._do_processing(data)
        
        return {
            'result': result,
            'worker_id': worker_id
        }
```

---

## ✅ BEST PRACTICES

### 1. Always Track Worker IDs

```python
# ✅ GOOD
async def search(self, query, worker_id=None):
    if worker_id is None:
        worker_id = get_worker_id()
    
    return {'results': ..., 'worker_id': worker_id}

# ❌ BAD
async def search(self, query):
    return {'results': ...}  # No worker tracking
```

### 2. Use Thread-Safe Access

```python
# ✅ GOOD
@shared(resource_type='cache')
class Cache:
    def __init__(self):
        self._lock = Lock()
    
    def get(self, key):
        with self._lock:
            return self.cache.get(key)

# ❌ BAD
@shared(resource_type='cache')
class Cache:
    def get(self, key):
        return self.cache.get(key)  # Not thread-safe
```

### 3. Initialize Once

```python
# ✅ GOOD
async def initialize(self):
    if self.is_initialized:
        return  # Already initialized
    
    self.index = await build_index()
    self.is_initialized = True

# ❌ BAD
async def initialize(self):
    self.index = await build_index()  # Re-initializes every time
```

### 4. Clean Up Resources

```python
# ✅ GOOD
async def shutdown(self):
    # Unregister from pool
    self.unregister()
    
    # Clean up
    self.index = None
    self.is_initialized = False

# ❌ BAD
async def shutdown(self):
    self.index = None  # Doesn't unregister
```

---

## 🐛 DEBUGGING

### Check Resource Pool

```python
from nanobrain.core.shared_resource import get_resource_pool

pool = get_resource_pool()

# List all resources
print("Registered resources:")
for res_id, stats in pool.list_resources().items():
    print(f"  {res_id}:")
    print(f"    Type: {stats['resource_type']}")
    print(f"    Accesses: {stats['access_count']}")
    print(f"    Workers: {stats['worker_ids']}")
```

### Verify Worker IDs

```python
# In your step
async def process(self, input_data):
    worker_id = get_worker_id()
    print(f"Processing with worker: {worker_id}")
    
    result = await self.shared_resource.process(
        input_data,
        worker_id=worker_id
    )
    
    print(f"Result worker: {result['worker_id']}")
    assert result['worker_id'] == worker_id
```

### Monitor Access Patterns

```python
# In your shared resource
def get_access_summary(self):
    return {
        'total_accesses': len(self.access_log),
        'unique_workers': len(set(a['worker_id'] for a in self.access_log)),
        'recent_accesses': self.access_log[-10:]
    }
```

---

## 📚 API REFERENCE

### @shared Decorator

```python
@shared(
    resource_type: str = None,      # Type of resource (default: class name)
    resource_id: str = None,        # Unique ID (default: auto-generated)
    auto_register: bool = True      # Auto-register in pool
)
```

### SharedResourcePool Methods

```python
pool = get_resource_pool()

# Register a resource
pool.register_resource(resource_id, resource, resource_type)

# Get a resource
resource = pool.get_resource(resource_id, worker_id, worker_type)

# Get statistics
stats = pool.get_resource_stats(resource_id)

# List all resources
resources = pool.list_resources()

# Unregister a resource
pool.unregister_resource(resource_id)
```

### Shared Resource Methods (Auto-Added)

```python
# Get resource ID
resource_id = my_resource.get_resource_id()

# Get statistics
stats = my_resource.get_resource_stats()

# Unregister from pool
my_resource.unregister()
```

### Worker ID Functions

```python
from nanobrain.core.shared_resource import get_worker_id

# Get current worker ID
worker_id = get_worker_id()  # Returns: "worker_abc12345"
```

---

## 🎯 EXAMPLES

See:
- `demos/parallel_rag_demo/shared_vector_database.py` - Full example
- `demos/parallel_rag_demo/test_shared_resources.py` - Test suite
- `demos/parallel_rag_demo/SHARED_RESOURCES_GUIDE.md` - Complete guide

---

**Quick Start Complete!** 🎉

For more details, see the full documentation in `SHARED_RESOURCES_GUIDE.md`.

