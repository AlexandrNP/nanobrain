# PBS HPC Technical Architecture 🏗️

**Date**: October 7, 2025  
**Purpose**: Technical architecture for PARSL-based RAG on PBS HPC systems  
**Version**: 1.0.0

---

## 🎯 ARCHITECTURE OVERVIEW

### **System Components**

```
┌─────────────────────────────────────────────────────────────────┐
│                        PBS HPC Cluster                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Node 1     │  │   Node 2     │  │   Node 3     │  ...    │
│  │              │  │              │  │              │         │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │         │
│  │ │ Worker 1 │ │  │ │ Worker 5 │ │  │ │ Worker 9 │ │         │
│  │ │  GPU 0   │ │  │ │  GPU 0   │ │  │ │  GPU 0   │ │         │
│  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │         │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │         │
│  │ │ Worker 2 │ │  │ │ Worker 6 │ │  │ │ Worker 10│ │         │
│  │ │  GPU 1   │ │  │ │  GPU 1   │ │  │ │  GPU 1   │ │         │
│  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │         │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │         │
│  │ │ Worker 3 │ │  │ │ Worker 7 │ │  │ │ Worker 11│ │         │
│  │ │  GPU 2   │ │  │ │  GPU 2   │ │  │ │  GPU 2   │ │         │
│  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │         │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │         │
│  │ │ Worker 4 │ │  │ │ Worker 8 │ │  │ │ Worker 12│ │         │
│  │ │  GPU 3   │ │  │ │  GPU 3   │ │  │ │  GPU 3   │ │         │
│  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              PARSL Coordinator (Head Node)              │   │
│  │  ┌────────────────────────────────────────────────┐     │   │
│  │  │         Parallel RAG Workflow                  │     │   │
│  │  │  ┌──────────────────────────────────────────┐  │     │   │
│  │  │  │  Step 1: Query Enhancement (PARSL)       │  │     │   │
│  │  │  │    - 16 worker instances                 │  │     │   │
│  │  │  │    - GPU-accelerated LLM                 │  │     │   │
│  │  │  └──────────────────────────────────────────┘  │     │   │
│  │  │  ┌──────────────────────────────────────────┐  │     │   │
│  │  │  │  Step 2: Vector Search (PARSL)           │  │     │   │
│  │  │  │    - 16 worker instances                 │  │     │   │
│  │  │  │    - Distributed vector DB               │  │     │   │
│  │  │  └──────────────────────────────────────────┘  │     │   │
│  │  │  ┌──────────────────────────────────────────┐  │     │   │
│  │  │  │  Step 3: Response Generation (PARSL)     │  │     │   │
│  │  │  │    - 16 worker instances                 │  │     │   │
│  │  │  │    - GPU-accelerated LLM                 │  │     │   │
│  │  │  └──────────────────────────────────────────┘  │     │   │
│  │  └────────────────────────────────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 COMPONENT DETAILS

### **1. PBS Job Submission Layer**

**Purpose**: Submit and manage PBS jobs

**Components**:
- PBS submission script (`submit_pbs.sh`)
- Job configuration (`pbs_config.yml`)
- Environment setup
- Resource allocation

**Key Features**:
- Multi-node allocation
- GPU resource management
- Filesystem mounting
- Module loading
- Environment variables

### **2. PARSL Coordination Layer**

**Purpose**: Coordinate distributed execution

**Components**:
- PARSL DataFlowKernel (DFK)
- HighThroughputExecutor (HTEX)
- PBSProProvider
- MpiExecLauncher

**Key Features**:
- Task scheduling
- Worker management
- Load balancing
- Fault tolerance
- Monitoring

### **3. Worker Step Pool Layer**

**Purpose**: Manage per-worker step instances

**Components**:
- WorkerStepPool manager
- Step instance registry
- Shared resource pool
- Worker ID tracking

**Key Features**:
- Per-worker isolation
- Hierarchical pools
- Shared resource management
- Instance lifecycle management

### **4. Workflow Execution Layer**

**Purpose**: Execute RAG workflow steps

**Components**:
- Query Enhancement Step
- Vector Search Step
- Response Generation Step
- Data units
- Agents

**Key Features**:
- GPU-accelerated inference
- Distributed vector search
- Batch processing
- Checkpointing

---

## 📊 DATA FLOW

### **Query Processing Flow**

```
1. Job Submission
   ├── User submits PBS job
   ├── PBS allocates nodes
   └── PARSL initializes workers

2. Workflow Initialization
   ├── Load workflow configuration
   ├── Create worker step pools
   ├── Initialize shared resources
   └── Set up monitoring

3. Query Distribution
   ├── Read input queries
   ├── Batch queries
   ├── Distribute to workers
   └── Track assignments

4. Step 1: Query Enhancement
   ├── Worker receives query
   ├── Load LLM on GPU
   ├── Enhance query
   ├── Return enhanced query
   └── Log worker ID

5. Step 2: Vector Search
   ├── Worker receives enhanced query
   ├── Access shared vector DB
   ├── Perform similarity search
   ├── Return relevant documents
   └── Log worker ID

6. Step 3: Response Generation
   ├── Worker receives query + docs
   ├── Load LLM on GPU
   ├── Generate response
   ├── Return final response
   └── Log worker ID

7. Result Collection
   ├── Aggregate results
   ├── Write to output
   ├── Update checkpoints
   └── Generate reports

8. Job Completion
   ├── Shutdown workers
   ├── Clean up resources
   ├── Archive logs
   └── Exit PBS job
```

---

## 🔐 RESOURCE MANAGEMENT

### **GPU Allocation Strategy**

**Per-Node Configuration**:
```yaml
Node Configuration:
  CPUs: 32
  GPUs: 4 (NVIDIA A100 or similar)
  Memory: 256GB
  
Worker Configuration:
  Workers per node: 4
  CPUs per worker: 8
  GPU per worker: 1
  Memory per worker: 64GB
  
GPU Assignment:
  Worker 1 → GPU 0
  Worker 2 → GPU 1
  Worker 3 → GPU 2
  Worker 4 → GPU 3
```

**GPU Affinity**:
```bash
# Set in worker_init
export CUDA_VISIBLE_DEVICES=0,1,2,3

# PARSL configuration
cpu_affinity: block-reverse
available_accelerators: [0, 1, 2, 3]
```

### **Memory Management**

**Per-Worker Memory Budget**:
```
Total per worker: 64GB
├── LLM model: 20GB
├── Vector embeddings: 10GB
├── Working memory: 20GB
├── Cache: 10GB
└── System overhead: 4GB
```

### **Network Configuration**

**Inter-Node Communication**:
```
Protocol: MPI
Launcher: mpiexec
Binding: --cpu-bind none
Depth: --depth=64
PPN: --ppn 1
```

---

## 🔄 FAULT TOLERANCE

### **Checkpoint Strategy**

**Checkpoint Frequency**:
- Every 100 queries processed
- Every 30 minutes
- Before worker shutdown

**Checkpoint Data**:
```yaml
checkpoint:
  timestamp: 2025-10-07T12:00:00
  queries_processed: 500
  queries_remaining: 500
  worker_states:
    worker_1: active
    worker_2: active
    worker_3: failed
    worker_4: active
  results:
    - query_id: 1
      status: complete
      result: {...}
    - query_id: 2
      status: complete
      result: {...}
```

### **Failure Recovery**

**Worker Failure**:
1. Detect worker failure (heartbeat timeout)
2. Mark worker as failed
3. Reassign pending tasks
4. Spawn replacement worker
5. Resume processing

**Node Failure**:
1. Detect node failure (all workers timeout)
2. Mark node as failed
3. Reassign all tasks from node
4. Request replacement node from PBS
5. Resume processing

**Job Failure**:
1. Save checkpoint
2. Exit gracefully
3. User resubmits job
4. Load checkpoint
5. Resume from last checkpoint

---

## 📈 PERFORMANCE OPTIMIZATION

### **Batch Processing**

**Batch Configuration**:
```yaml
batch_processing:
  batch_size: 10
  prefetch_factor: 2
  max_queue_size: 100
  timeout: 300
```

**Benefits**:
- Reduced overhead
- Better GPU utilization
- Improved throughput
- Lower latency

### **Caching Strategy**

**Multi-Level Cache**:
```
L1: Worker-local cache (per-worker)
├── LLM model cache
├── Embedding cache
└── Result cache

L2: Node-local cache (shared per node)
├── Vector database cache
└── Document cache

L3: Cluster-wide cache (shared across nodes)
└── Persistent vector database
```

### **Load Balancing**

**Dynamic Load Balancing**:
- Monitor worker queue lengths
- Redistribute tasks to idle workers
- Adjust batch sizes based on load
- Prioritize fast workers

---

## 🔍 MONITORING

### **Metrics Collection**

**System Metrics**:
- CPU utilization per worker
- GPU utilization per worker
- Memory usage per worker
- Network bandwidth
- Disk I/O

**Application Metrics**:
- Queries processed per second
- Average query latency
- Worker throughput
- Error rate
- Queue depth

**PARSL Metrics**:
- Task submission rate
- Task completion rate
- Worker failures
- Executor status
- DFK status

### **Monitoring Dashboard**

**PARSL Monitoring**:
```bash
# Start monitoring
parsl-visualize --port 8080

# Access dashboard
http://localhost:8080
```

**Custom Monitoring**:
```python
# Log metrics
logger.info(f"Worker {worker_id}: Processed {count} queries")
logger.info(f"GPU {gpu_id}: Utilization {util}%")
logger.info(f"Throughput: {qps} queries/sec")
```

---

## 🔒 SECURITY

### **Access Control**

**PBS Account**:
- Require valid PBS account
- Enforce project allocation
- Track resource usage

**File Permissions**:
- Restrict access to checkpoints
- Protect configuration files
- Secure log files

### **Data Security**

**Sensitive Data**:
- Encrypt data at rest
- Secure data in transit
- Sanitize logs
- Implement access controls

---

## 📊 SCALABILITY

### **Scaling Strategy**

**Horizontal Scaling**:
```
Small:  4 nodes × 4 workers = 16 workers
Medium: 16 nodes × 4 workers = 64 workers
Large:  64 nodes × 4 workers = 256 workers
```

**Vertical Scaling**:
```
Workers per node: 1-8
CPUs per worker: 4-16
GPUs per worker: 1-4
Memory per worker: 32-128GB
```

### **Performance Scaling**

**Expected Throughput**:
```
16 workers:  100 queries/hour
64 workers:  400 queries/hour
256 workers: 1600 queries/hour
```

**Scaling Efficiency**:
```
Target: 90% efficiency up to 64 nodes
Acceptable: 80% efficiency up to 256 nodes
```

---

**Status**: 📋 **ARCHITECTURE DOCUMENTED**  
**Next Step**: Implement configuration templates

