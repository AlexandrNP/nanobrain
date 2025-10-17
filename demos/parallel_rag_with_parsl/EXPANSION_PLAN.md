# Parallel RAG with PARSL - Expansion Plan 📋

**Date**: October 7, 2025  
**Purpose**: Expand parallel RAG demo with comprehensive tracking and logging  
**Status**: 📋 **PLANNING PHASE**

---

## 🎯 OBJECTIVE

Enhance the parallel RAG with PARSL demo to track and log the complete RAG pipeline for each query:

1. **Original Query** - User's initial question
2. **Expanded Query** - Enhanced/expanded version from query enhancement step
3. **Relevant RAG Documents** - Retrieved documents from vector search
4. **Final Response** - Generated answer from response generation step

All data should be saved to structured log files for user review and analysis.

---

## 📊 CURRENT STATE ANALYSIS

### **Existing Components**

✅ **Worker Step Pools** - Implemented and tested
- Per-worker instance isolation
- Hierarchical pool support
- Shared resource management

✅ **Parallel RAG Workflow** - Basic structure exists
- Query enhancement step
- Vector search step
- Response generation step

✅ **Test Infrastructure** - Multiple tests passing
- `test_worker_step_pool.py` ✅
- `test_end_to_end_worker_isolation.py` ✅
- `test_full_workflow_worker_isolation.py` ✅
- `test_hierarchical_worker_pools.py` ✅

### **Current Gaps**

⚠️ **Missing Tracking**:
- No structured logging of query transformations
- No document retrieval tracking
- No end-to-end query journey logging
- No user-friendly output format

⚠️ **Missing Features**:
- No detailed RAG pipeline visualization
- No query-document-response correlation
- No performance metrics per query
- No debugging/analysis tools

---

## 🎯 EXPANSION GOALS

### **Primary Goals**

1. ✅ **Complete Query Tracking**
   - Track original query
   - Track expanded/enhanced query
   - Track retrieved documents
   - Track final response

2. ✅ **Structured Logging**
   - JSON format for machine readability
   - Human-readable text format
   - Per-query log files
   - Aggregated summary logs

3. ✅ **Performance Metrics**
   - Time per step
   - Document relevance scores
   - Worker assignments
   - Resource utilization

4. ✅ **User-Friendly Output**
   - Clear query journey visualization
   - Easy-to-read summaries
   - Searchable logs
   - Export capabilities

---

## 📋 IMPLEMENTATION PLAN

### **Phase 1: Data Structures** (Week 1, Days 1-2)

#### **Task 1.1: Create Query Journey Data Model**

**File**: `demos/parallel_rag_with_parsl/models/query_journey.py`

**Purpose**: Define data structures for tracking complete query journey

**Data Model**:
```python
@dataclass
class QueryJourney:
    """Complete journey of a query through RAG pipeline."""
    
    # Identifiers
    query_id: str
    timestamp: float
    
    # Original query
    original_query: str
    
    # Step 1: Query Enhancement
    enhanced_query: Optional[str] = None
    enhancement_metadata: Optional[Dict[str, Any]] = None
    enhancement_worker_id: Optional[str] = None
    enhancement_time: Optional[float] = None
    
    # Step 2: Document Retrieval
    retrieved_documents: Optional[List[Document]] = None
    retrieval_metadata: Optional[Dict[str, Any]] = None
    retrieval_worker_id: Optional[str] = None
    retrieval_time: Optional[float] = None
    
    # Step 3: Response Generation
    final_response: Optional[str] = None
    generation_metadata: Optional[Dict[str, Any]] = None
    generation_worker_id: Optional[str] = None
    generation_time: Optional[float] = None
    
    # Overall metrics
    total_time: Optional[float] = None
    status: str = "pending"  # pending, processing, complete, failed
    error: Optional[str] = None

@dataclass
class Document:
    """Retrieved document from vector search."""
    
    doc_id: str
    content: str
    relevance_score: float
    metadata: Dict[str, Any]
    source: Optional[str] = None
```

**Features**:
- Complete query lifecycle tracking
- Metadata for each step
- Worker ID tracking
- Performance timing
- Error handling

#### **Task 1.2: Create Logging Manager**

**File**: `demos/parallel_rag_with_parsl/logging/journey_logger.py`

**Purpose**: Manage logging of query journeys

**Class Structure**:
```python
class QueryJourneyLogger:
    """Logger for query journeys through RAG pipeline."""
    
    def __init__(self, output_dir: str, format: str = "both"):
        """
        Initialize logger.
        
        Args:
            output_dir: Directory for log files
            format: "json", "text", or "both"
        """
        self.output_dir = output_dir
        self.format = format
        self.journeys: Dict[str, QueryJourney] = {}
    
    def start_journey(self, query_id: str, original_query: str) -> QueryJourney
    def update_enhancement(self, query_id: str, enhanced_query: str, metadata: Dict)
    def update_retrieval(self, query_id: str, documents: List[Document], metadata: Dict)
    def update_generation(self, query_id: str, response: str, metadata: Dict)
    def complete_journey(self, query_id: str)
    def save_journey(self, query_id: str)
    def save_all_journeys(self)
    def get_summary(self) -> Dict[str, Any]
```

---

### **Phase 2: Step Integration** (Week 1, Days 3-5)

#### **Task 2.1: Enhance Query Enhancement Step**

**File**: `demos/parallel_rag_with_parsl/steps/tracked_query_enhancement_step.py`

**Purpose**: Add tracking to query enhancement step

**Enhancements**:
```python
class TrackedQueryEnhancementStep(QueryEnhancementStep):
    """Query enhancement step with journey tracking."""
    
    def __init__(self, config, journey_logger: QueryJourneyLogger):
        super().__init__(config)
        self.journey_logger = journey_logger
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query_id = data['query_id']
        original_query = data['query']
        
        # Start journey
        journey = self.journey_logger.start_journey(query_id, original_query)
        
        # Enhance query
        start_time = time.time()
        enhanced_result = await super().process(data)
        enhancement_time = time.time() - start_time
        
        # Update journey
        self.journey_logger.update_enhancement(
            query_id=query_id,
            enhanced_query=enhanced_result['enhanced_query'],
            metadata={
                'worker_id': self.worker_id,
                'instance_id': self.instance_id,
                'time': enhancement_time,
                'search_terms': enhanced_result.get('search_terms', []),
                'confidence_score': enhanced_result.get('confidence_score', 0.0)
            }
        )
        
        return enhanced_result
```

#### **Task 2.2: Enhance Vector Search Step**

**File**: `demos/parallel_rag_with_parsl/steps/tracked_vector_search_step.py`

**Purpose**: Add document retrieval tracking

**Enhancements**:
```python
class TrackedVectorSearchStep(VectorSearchStep):
    """Vector search step with document tracking."""
    
    def __init__(self, config, journey_logger: QueryJourneyLogger):
        super().__init__(config)
        self.journey_logger = journey_logger
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query_id = data['query_id']
        
        # Perform search
        start_time = time.time()
        search_result = await super().process(data)
        retrieval_time = time.time() - start_time
        
        # Convert to Document objects
        documents = [
            Document(
                doc_id=doc['id'],
                content=doc['content'],
                relevance_score=doc['score'],
                metadata=doc.get('metadata', {}),
                source=doc.get('source')
            )
            for doc in search_result['documents']
        ]
        
        # Update journey
        self.journey_logger.update_retrieval(
            query_id=query_id,
            documents=documents,
            metadata={
                'worker_id': self.worker_id,
                'instance_id': self.instance_id,
                'time': retrieval_time,
                'num_documents': len(documents),
                'avg_relevance_score': sum(d.relevance_score for d in documents) / len(documents)
            }
        )
        
        return search_result
```

#### **Task 2.3: Enhance Response Generation Step**

**File**: `demos/parallel_rag_with_parsl/steps/tracked_response_generation_step.py`

**Purpose**: Add final response tracking

**Enhancements**:
```python
class TrackedResponseGenerationStep(ResponseGenerationStep):
    """Response generation step with tracking."""
    
    def __init__(self, config, journey_logger: QueryJourneyLogger):
        super().__init__(config)
        self.journey_logger = journey_logger
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query_id = data['query_id']
        
        # Generate response
        start_time = time.time()
        generation_result = await super().process(data)
        generation_time = time.time() - start_time
        
        # Update journey
        self.journey_logger.update_generation(
            query_id=query_id,
            response=generation_result['response'],
            metadata={
                'worker_id': self.worker_id,
                'instance_id': self.instance_id,
                'time': generation_time,
                'response_length': len(generation_result['response']),
                'model': generation_result.get('model', 'unknown')
            }
        )
        
        # Complete journey
        self.journey_logger.complete_journey(query_id)
        self.journey_logger.save_journey(query_id)
        
        return generation_result
```

---

### **Phase 3: Logging Formats** (Week 2, Days 1-2)

#### **Task 3.1: JSON Log Format**

**File**: `demos/parallel_rag_with_parsl/output/logs/queries/query_{query_id}.json`

**Format**:
```json
{
  "query_id": "q_001",
  "timestamp": 1696723200.0,
  "status": "complete",
  
  "original_query": "What are the key mechanisms of viral membrane fusion?",
  
  "enhancement": {
    "enhanced_query": "What are the primary processes, key mechanisms, and fundamental principles of viral membrane fusion?",
    "worker_id": "worker_abc123",
    "instance_id": 4684434320,
    "time": 2.34,
    "metadata": {
      "search_terms": ["viral membrane fusion", "fusion proteins"],
      "confidence_score": 0.95
    }
  },
  
  "retrieval": {
    "worker_id": "worker_abc123",
    "instance_id": 4684434321,
    "time": 1.23,
    "documents": [
      {
        "doc_id": "doc_001",
        "content": "Viral membrane fusion is a critical step...",
        "relevance_score": 0.92,
        "metadata": {"source": "pubmed", "year": 2023}
      }
    ],
    "metadata": {
      "num_documents": 5,
      "avg_relevance_score": 0.87
    }
  },
  
  "generation": {
    "final_response": "Based on the analysis of viral membrane fusion mechanisms...",
    "worker_id": "worker_abc123",
    "instance_id": 4684434322,
    "time": 3.45,
    "metadata": {
      "response_length": 1234,
      "model": "gpt-4"
    }
  },
  
  "metrics": {
    "total_time": 7.02,
    "enhancement_time": 2.34,
    "retrieval_time": 1.23,
    "generation_time": 3.45
  }
}
```

#### **Task 3.2: Human-Readable Text Format**

**File**: `demos/parallel_rag_with_parsl/output/logs/queries/query_{query_id}.txt`

**Format**:
```
================================================================================
QUERY JOURNEY: q_001
================================================================================
Timestamp: 2025-10-07 12:00:00
Status: COMPLETE
Total Time: 7.02s

────────────────────────────────────────────────────────────────────────────────
ORIGINAL QUERY
────────────────────────────────────────────────────────────────────────────────
What are the key mechanisms of viral membrane fusion?

────────────────────────────────────────────────────────────────────────────────
STEP 1: QUERY ENHANCEMENT
────────────────────────────────────────────────────────────────────────────────
Worker: worker_abc123
Instance: 4684434320
Time: 2.34s

Enhanced Query:
What are the primary processes, key mechanisms, and fundamental principles 
of viral membrane fusion?

Search Terms:
  - viral membrane fusion
  - fusion proteins
  - envelope glycoproteins

Confidence Score: 0.95

────────────────────────────────────────────────────────────────────────────────
STEP 2: DOCUMENT RETRIEVAL
────────────────────────────────────────────────────────────────────────────────
Worker: worker_abc123
Instance: 4684434321
Time: 1.23s
Documents Retrieved: 5
Average Relevance: 0.87

Document 1 (Score: 0.92):
  ID: doc_001
  Source: pubmed (2023)
  Content: Viral membrane fusion is a critical step...

Document 2 (Score: 0.89):
  ID: doc_002
  Source: pubmed (2022)
  Content: The spike protein facilitates...

[... more documents ...]

────────────────────────────────────────────────────────────────────────────────
STEP 3: RESPONSE GENERATION
────────────────────────────────────────────────────────────────────────────────
Worker: worker_abc123
Instance: 4684434322
Time: 3.45s
Model: gpt-4
Response Length: 1234 characters

Final Response:
Based on the analysis of viral membrane fusion mechanisms, the key processes 
include: 1) Viral attachment to host cell receptors, 2) Conformational 
changes in fusion proteins, 3) Membrane hemifusion, and 4) Pore formation 
and expansion...

================================================================================
END OF QUERY JOURNEY
================================================================================
```

---

### **Phase 4: Summary and Analysis** (Week 2, Days 3-5)

#### **Task 4.1: Create Summary Logger**

**File**: `demos/parallel_rag_with_parsl/output/logs/summary.json`

**Content**:
```json
{
  "session_id": "session_20251007_120000",
  "timestamp": 1696723200.0,
  "total_queries": 100,
  "successful": 98,
  "failed": 2,
  
  "performance": {
    "total_time": 702.5,
    "avg_time_per_query": 7.02,
    "avg_enhancement_time": 2.34,
    "avg_retrieval_time": 1.23,
    "avg_generation_time": 3.45,
    "queries_per_second": 0.14
  },
  
  "workers": {
    "total_workers": 16,
    "queries_per_worker": {
      "worker_abc123": 25,
      "worker_def456": 24,
      ...
    }
  },
  
  "documents": {
    "total_retrieved": 500,
    "avg_per_query": 5.0,
    "avg_relevance_score": 0.87
  }
}
```

#### **Task 4.2: Create Analysis Tools**

**File**: `demos/parallel_rag_with_parsl/tools/analyze_logs.py`

**Features**:
- Search queries by keyword
- Filter by performance metrics
- Analyze document relevance
- Worker utilization analysis
- Export to CSV/Excel

---

## 📁 DELIVERABLES

### **New Files to Create**

1. ✅ `models/query_journey.py` - Data models
2. ✅ `logging/journey_logger.py` - Logging manager
3. ✅ `steps/tracked_query_enhancement_step.py` - Enhanced step
4. ✅ `steps/tracked_vector_search_step.py` - Enhanced step
5. ✅ `steps/tracked_response_generation_step.py` - Enhanced step
6. ✅ `tools/analyze_logs.py` - Analysis tools
7. ✅ `tools/export_logs.py` - Export utilities
8. ✅ `test_tracked_workflow.py` - Integration test

### **Updated Files**

9. ✅ `test_parallel_rag.py` - Add journey tracking
10. ✅ `README.md` - Document new features

---

## 📊 SUCCESS CRITERIA

### **Functional Requirements**

- [ ] All query journeys tracked end-to-end
- [ ] JSON logs generated for each query
- [ ] Text logs generated for each query
- [ ] Summary logs generated per session
- [ ] Analysis tools working
- [ ] Export functionality working

### **Performance Requirements**

- [ ] Logging overhead < 5% of total time
- [ ] Log files < 10MB per 1000 queries
- [ ] Real-time logging (no buffering delays)

### **Usability Requirements**

- [ ] Logs easy to read and understand
- [ ] Search/filter tools intuitive
- [ ] Export formats compatible with common tools

---

## 📅 DETAILED TIMELINE

### **Week 1: Core Implementation**

#### **Day 1: Data Structures**
- Morning: Create `models/query_journey.py`
  - Define QueryJourney dataclass
  - Define Document dataclass
  - Add serialization methods
  - Write unit tests

- Afternoon: Create `logging/journey_logger.py`
  - Implement QueryJourneyLogger class
  - Add JSON logging
  - Add text logging
  - Write unit tests

#### **Day 2: Logger Integration**
- Morning: Test logging infrastructure
  - Integration tests
  - Performance tests
  - Error handling tests

- Afternoon: Create directory structure
  - Set up output directories
  - Create log rotation
  - Add cleanup utilities

#### **Day 3: Query Enhancement Tracking**
- Morning: Create `steps/tracked_query_enhancement_step.py`
  - Extend QueryEnhancementStep
  - Add journey tracking
  - Add metadata collection

- Afternoon: Test enhancement tracking
  - Unit tests
  - Integration tests
  - Verify log output

#### **Day 4: Document Retrieval Tracking**
- Morning: Create `steps/tracked_vector_search_step.py`
  - Extend VectorSearchStep
  - Add document tracking
  - Add relevance scoring

- Afternoon: Test retrieval tracking
  - Unit tests
  - Integration tests
  - Verify document logs

#### **Day 5: Response Generation Tracking**
- Morning: Create `steps/tracked_response_generation_step.py`
  - Extend ResponseGenerationStep
  - Add response tracking
  - Complete journey

- Afternoon: End-to-end testing
  - Full workflow test
  - Verify all logs
  - Performance testing

### **Week 2: Analysis and Tools**

#### **Day 1: Summary Logging**
- Morning: Implement session summaries
  - Aggregate metrics
  - Worker statistics
  - Performance analysis

- Afternoon: Test summary generation
  - Multiple queries
  - Edge cases
  - Error scenarios

#### **Day 2: Analysis Tools**
- Morning: Create `tools/analyze_logs.py`
  - Search functionality
  - Filter by metrics
  - Statistical analysis

- Afternoon: Test analysis tools
  - Query search
  - Performance analysis
  - Worker utilization

#### **Day 3: Export Tools**
- Morning: Create `tools/export_logs.py`
  - CSV export
  - Excel export
  - JSON export

- Afternoon: Test export functionality
  - Format validation
  - Large datasets
  - Error handling

#### **Day 4: Integration Testing**
- Morning: Full workflow testing
  - 100+ queries
  - Multi-worker execution
  - Log verification

- Afternoon: Performance optimization
  - Reduce logging overhead
  - Optimize file I/O
  - Memory management

#### **Day 5: Documentation and Cleanup**
- Morning: Update documentation
  - README updates
  - API documentation
  - Usage examples

- Afternoon: Final testing and review
  - Code review
  - Test coverage
  - Performance validation

---

## 🔧 IMPLEMENTATION EXAMPLES

### **Example 1: Using Journey Logger**

```python
from demos.parallel_rag_with_parsl.logging.journey_logger import QueryJourneyLogger
from demos.parallel_rag_with_parsl.models.query_journey import QueryJourney, Document

# Initialize logger
logger = QueryJourneyLogger(
    output_dir="output/logs",
    format="both"  # JSON and text
)

# Start journey
journey = logger.start_journey(
    query_id="q_001",
    original_query="What are the key mechanisms of viral membrane fusion?"
)

# Update after enhancement
logger.update_enhancement(
    query_id="q_001",
    enhanced_query="What are the primary processes and key mechanisms...",
    metadata={
        'worker_id': 'worker_abc123',
        'time': 2.34,
        'confidence_score': 0.95
    }
)

# Update after retrieval
documents = [
    Document(
        doc_id="doc_001",
        content="Viral membrane fusion is...",
        relevance_score=0.92,
        metadata={'source': 'pubmed'}
    )
]
logger.update_retrieval(
    query_id="q_001",
    documents=documents,
    metadata={'worker_id': 'worker_abc123', 'time': 1.23}
)

# Update after generation
logger.update_generation(
    query_id="q_001",
    response="Based on the analysis...",
    metadata={'worker_id': 'worker_abc123', 'time': 3.45}
)

# Complete and save
logger.complete_journey("q_001")
logger.save_journey("q_001")
```

### **Example 2: Analyzing Logs**

```python
from demos.parallel_rag_with_parsl.tools.analyze_logs import LogAnalyzer

# Initialize analyzer
analyzer = LogAnalyzer("output/logs")

# Search queries
results = analyzer.search_queries(
    keyword="viral fusion",
    min_relevance=0.8,
    max_time=10.0
)

# Get performance statistics
stats = analyzer.get_performance_stats()
print(f"Average query time: {stats['avg_total_time']:.2f}s")
print(f"Average relevance: {stats['avg_relevance_score']:.2f}")

# Analyze worker utilization
worker_stats = analyzer.get_worker_stats()
for worker_id, count in worker_stats.items():
    print(f"{worker_id}: {count} queries")

# Export to CSV
analyzer.export_to_csv("output/analysis.csv")
```

### **Example 3: Viewing Query Journey**

```python
from demos.parallel_rag_with_parsl.tools.view_journey import view_journey

# View specific query
view_journey("q_001", format="text")

# Output:
# ================================================================================
# QUERY JOURNEY: q_001
# ================================================================================
# Original Query: What are the key mechanisms of viral membrane fusion?
# Enhanced Query: What are the primary processes and key mechanisms...
# Documents Retrieved: 5 (avg relevance: 0.87)
# Final Response: Based on the analysis of viral membrane fusion...
# Total Time: 7.02s
# ================================================================================
```

---

## 📊 LOG FILE STRUCTURE

### **Directory Layout**

```
demos/parallel_rag_with_parsl/output/logs/
├── queries/                           # Per-query logs
│   ├── q_001.json                    # JSON format
│   ├── q_001.txt                     # Text format
│   ├── q_002.json
│   ├── q_002.txt
│   └── ...
│
├── sessions/                          # Session summaries
│   ├── session_20251007_120000.json
│   ├── session_20251007_130000.json
│   └── ...
│
├── workers/                           # Worker-specific logs
│   ├── worker_abc123.json
│   ├── worker_def456.json
│   └── ...
│
├── summary.json                       # Overall summary
├── performance.json                   # Performance metrics
└── errors.json                        # Error log
```

### **Log Retention Policy**

```yaml
retention:
  query_logs: 30 days
  session_logs: 90 days
  summary_logs: 1 year

rotation:
  max_file_size: 100MB
  max_files_per_directory: 10000

compression:
  enabled: true
  format: gzip
  compress_after: 7 days
```

---

## 🎯 TESTING STRATEGY

### **Unit Tests**

1. **Data Models** (`test_query_journey.py`)
   - QueryJourney serialization
   - Document serialization
   - Metadata handling

2. **Logger** (`test_journey_logger.py`)
   - Journey creation
   - Updates
   - Saving
   - Format validation

3. **Tracked Steps** (`test_tracked_steps.py`)
   - Enhancement tracking
   - Retrieval tracking
   - Generation tracking

### **Integration Tests**

4. **End-to-End** (`test_tracked_workflow.py`)
   - Full workflow with tracking
   - Multiple queries
   - Worker isolation
   - Log verification

5. **Performance** (`test_logging_performance.py`)
   - Logging overhead
   - File I/O performance
   - Memory usage

### **Analysis Tests**

6. **Tools** (`test_analysis_tools.py`)
   - Search functionality
   - Export functionality
   - Statistics calculation

---

## 🚀 DEPLOYMENT

### **Installation**

```bash
# Install additional dependencies
pip install pandas openpyxl

# Create log directories
mkdir -p demos/parallel_rag_with_parsl/output/logs/{queries,sessions,workers}

# Run tests
pytest demos/parallel_rag_with_parsl/tests/test_tracked_workflow.py
```

### **Usage**

```bash
# Run with tracking enabled
python demos/parallel_rag_with_parsl/test_parallel_rag.py \
    --enable-tracking \
    --log-dir output/logs \
    --log-format both

# Analyze logs
python demos/parallel_rag_with_parsl/tools/analyze_logs.py \
    --log-dir output/logs \
    --export analysis.csv

# View specific query
python demos/parallel_rag_with_parsl/tools/view_journey.py \
    --query-id q_001 \
    --format text
```

---

## 📈 EXPECTED BENEFITS

### **For Users**

- ✅ **Transparency**: See exactly how queries are processed
- ✅ **Debugging**: Identify issues in specific steps
- ✅ **Analysis**: Understand document relevance and quality
- ✅ **Optimization**: Find bottlenecks and improve performance

### **For Developers**

- ✅ **Monitoring**: Track system behavior in production
- ✅ **Testing**: Verify correct operation
- ✅ **Debugging**: Detailed error information
- ✅ **Metrics**: Performance and quality metrics

### **For Researchers**

- ✅ **Reproducibility**: Complete query processing history
- ✅ **Analysis**: Study RAG pipeline effectiveness
- ✅ **Comparison**: Compare different configurations
- ✅ **Publication**: Data for papers and reports

---

**Status**: 📋 **PLAN COMPLETE - READY FOR IMPLEMENTATION**
**Timeline**: 2 weeks (10 working days)
**Estimated Effort**: 80 hours
**Next Step**: Begin Day 1 - Data Structures
**Priority**: High
**Dependencies**: None (all prerequisites met)

