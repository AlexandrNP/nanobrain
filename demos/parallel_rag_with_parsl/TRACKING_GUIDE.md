# Query Journey Tracking - User Guide 📊

**Version**: 1.0.0  
**Status**: ✅ **Production Ready**  
**Last Updated**: October 13, 2025

---

## 🎯 OVERVIEW

The parallel RAG demo now includes **complete query journey tracking** that logs every step of query processing through the RAG pipeline. This guide shows you how to use the tracking features.

### What Gets Tracked

For each query, the system tracks:

✅ **Original Query** - User's question  
✅ **Enhanced Query** - Expanded/improved version  
✅ **Retrieved Documents** - All documents with relevance scores  
✅ **Final Response** - Generated answer  
✅ **Timestamps** - Start and end time for each step  
✅ **Processing Time** - Duration of each step  
✅ **Worker Information** - Which worker processed each step  

---

## 🚀 QUICK START

### 1. Enable Tracking in Your Workflow

```python
from demos.parallel_rag_with_parsl.journey_logging import QueryJourneyLogger
from demos.parallel_rag_with_parsl.steps import (
    TrackedQueryEnhancementStep,
    TrackedVectorSearchStep,
    TrackedResponseGenerationStep
)

# Create journey logger
logger = QueryJourneyLogger(
    output_dir="output/logs",
    format="both"  # Generate both JSON and text files
)

# Use tracked steps in your workflow
# (These automatically log to the journey logger)
```

### 2. View Logs

After running queries, view the generated logs:

```bash
# List all tracked queries
python -m demos.parallel_rag_with_parsl.tools.view_journey --list

# View specific query (text format)
python -m demos.parallel_rag_with_parsl.tools.view_journey q_001

# View specific query (JSON format)
python -m demos.parallel_rag_with_parsl.tools.view_journey q_001 json
```

### 3. Analyze Results

```python
from demos.parallel_rag_with_parsl.tools import LogAnalyzer

# Load and analyze logs
analyzer = LogAnalyzer("output/logs")

# Get summary statistics
analyzer.print_summary()

# Search for specific queries
results = analyzer.search_queries(keyword="viral", min_relevance=0.85)
```

---

## 📁 LOG FILE STRUCTURE

### Directory Layout

```
output/logs/
├── queries/                    # Per-query logs
│   ├── q_001.json             # Machine-readable format
│   ├── q_001.txt              # Human-readable format
│   ├── q_002.json
│   ├── q_002.txt
│   └── ...
```

### JSON Format

```json
{
  "query_id": "q_001",
  "timestamp": 1697207264.123,
  "status": "complete",
  "original_query": "What are the key mechanisms of viral membrane fusion?",
  
  "enhanced_query": "What are the primary processes, key mechanisms...",
  "enhancement_metadata": {
    "worker_id": "worker_0",
    "instance_id": 12345,
    "start_time": 1697207264.123,
    "end_time": 1697207264.224,
    "duration": 0.101
  },
  
  "retrieved_documents": [
    {
      "doc_id": "doc_001",
      "content": "Viral membrane fusion is...",
      "relevance_score": 0.92,
      "metadata": {"source": "pubmed", "year": 2023}
    }
  ],
  "retrieval_metadata": {
    "worker_id": "worker_0",
    "start_time": 1697207264.224,
    "end_time": 1697207264.389,
    "duration": 0.165
  },
  
  "final_response": "Based on the analysis...",
  "generation_metadata": {
    "worker_id": "worker_0",
    "start_time": 1697207264.389,
    "end_time": 1697207264.590,
    "duration": 0.201
  },
  
  "total_time": 0.467
}
```

### Text Format

```
================================================================================
QUERY JOURNEY: q_001
================================================================================
Status: COMPLETE
Timestamp: 2025-10-13 14:07:44
Total Time: 0.47s

--------------------------------------------------------------------------------
ORIGINAL QUERY
--------------------------------------------------------------------------------
What are the key mechanisms of viral membrane fusion?

--------------------------------------------------------------------------------
STEP 1: QUERY ENHANCEMENT
--------------------------------------------------------------------------------
Worker: worker_0
Instance: 12345
Start Time: 2025-10-13 14:07:44
End Time: 2025-10-13 14:07:44
Duration: 0.10s

Enhanced Query:
What are the primary processes, key mechanisms, and fundamental principles...

--------------------------------------------------------------------------------
STEP 2: DOCUMENT RETRIEVAL
--------------------------------------------------------------------------------
Worker: worker_0
Start Time: 2025-10-13 14:07:44
End Time: 2025-10-13 14:07:44
Duration: 0.17s
Documents Retrieved: 3
Average Relevance: 0.89

Document 1 (Score: 0.92):
  ID: doc_001
  Source: pubmed
  Content: Viral membrane fusion is a critical step...

--------------------------------------------------------------------------------
STEP 3: RESPONSE GENERATION
--------------------------------------------------------------------------------
Worker: worker_0
Start Time: 2025-10-13 14:07:44
End Time: 2025-10-13 14:07:44
Duration: 0.20s
Response Length: 432 characters

Final Response:
Based on the analysis of viral membrane fusion mechanisms...

================================================================================
```

---

## 🔍 ANALYSIS TOOLS

### LogAnalyzer

Search, filter, and analyze query logs.

```python
from demos.parallel_rag_with_parsl.tools import LogAnalyzer

analyzer = LogAnalyzer("output/logs")

# Search by keyword
results = analyzer.search_queries(keyword="viral")
print(f"Found {len(results)} queries about viral topics")

# Filter by relevance
high_quality = analyzer.search_queries(min_relevance=0.85)
print(f"Found {len(high_quality)} high-quality results")

# Filter by processing time
fast_queries = analyzer.search_queries(max_time=1.0)
print(f"Found {len(fast_queries)} queries under 1 second")

# Filter by worker
worker_queries = analyzer.search_queries(worker_id="worker_0")
print(f"Worker 0 processed {len(worker_queries)} queries")

# Get performance statistics
stats = analyzer.get_performance_stats()
print(f"Average processing time: {stats['avg_total_time']:.3f}s")
print(f"Average relevance score: {stats['avg_relevance_score']:.2f}")

# Get worker utilization
worker_stats = analyzer.get_worker_stats()
for worker_id, count in worker_stats.items():
    print(f"{worker_id}: {count} tasks")

# Find slowest queries
slowest = analyzer.get_slowest_queries(10)
for stat in slowest:
    print(f"{stat.query_id}: {stat.total_time:.3f}s")

# Find lowest relevance queries
lowest = analyzer.get_lowest_relevance_queries(10)
for stat in lowest:
    print(f"{stat.query_id}: {stat.avg_relevance:.2f}")

# Print summary
analyzer.print_summary()
```

### LogExporter

Export logs to various formats.

```python
from demos.parallel_rag_with_parsl.tools import LogExporter

exporter = LogExporter(analyzer)

# Export to CSV (for Excel, Google Sheets, etc.)
exporter.export_to_csv("queries.csv")

# Export to JSON (for further processing)
exporter.export_to_json("queries.json")

# Export summary report
exporter.export_summary_to_text("summary.txt")

# Export detailed report
exporter.export_detailed_report("detailed_report.txt")

# Export specific queries only
query_ids = ["q_001", "q_002", "q_003"]
exporter.export_to_csv("selected_queries.csv", queries=query_ids)
```

### Journey Viewer

View and compare individual journeys.

```python
from demos.parallel_rag_with_parsl.tools.view_journey import (
    view_journey,
    list_journeys,
    compare_journeys
)

# List all available journeys
list_journeys("output/logs")

# View specific journey (text format)
view_journey("q_001", log_dir="output/logs", format="text")

# View specific journey (JSON format)
view_journey("q_001", log_dir="output/logs", format="json")

# Compare multiple journeys
compare_journeys(["q_001", "q_002", "q_003"], log_dir="output/logs")
```

---

## 📊 USE CASES

### 1. Debugging Slow Queries

```python
analyzer = LogAnalyzer("output/logs")

# Find slowest queries
slowest = analyzer.get_slowest_queries(5)

for stat in slowest:
    print(f"\nQuery: {stat.query_id}")
    print(f"Total time: {stat.total_time:.3f}s")
    print(f"  Enhancement: {stat.enhancement_time:.3f}s")
    print(f"  Retrieval: {stat.retrieval_time:.3f}s")
    print(f"  Generation: {stat.generation_time:.3f}s")
    
    # View full journey
    view_journey(stat.query_id, format="text")
```

### 2. Quality Analysis

```python
# Find queries with low relevance scores
lowest = analyzer.get_lowest_relevance_queries(10)

for stat in lowest:
    journey = analyzer.journeys[stat.query_id]
    print(f"\nQuery: {journey.original_query}")
    print(f"Avg relevance: {stat.avg_relevance:.2f}")
    print(f"Documents:")
    for doc in journey.retrieved_documents:
        print(f"  - {doc.doc_id}: {doc.relevance_score:.2f}")
```

### 3. Worker Load Balancing

```python
# Analyze worker utilization
worker_stats = analyzer.get_worker_stats()

print("Worker Utilization:")
for worker_id, count in sorted(worker_stats.items()):
    print(f"  {worker_id}: {count} tasks")

# Find queries processed by specific worker
worker_queries = analyzer.search_queries(worker_id="worker_0")
print(f"\nWorker 0 processed {len(worker_queries)} queries")
```

### 4. Performance Reporting

```python
exporter = LogExporter(analyzer)

# Generate comprehensive report
exporter.export_summary_to_text("weekly_report.txt")
exporter.export_to_csv("weekly_data.csv")

# Get statistics for presentation
stats = analyzer.get_performance_stats()
print(f"Queries processed: {stats['total_queries']}")
print(f"Success rate: {stats['complete'] / stats['total_queries'] * 100:.1f}%")
print(f"Avg processing time: {stats['avg_total_time']:.3f}s")
print(f"Avg quality score: {stats['avg_relevance_score']:.2f}")
```

---

## 🛠️ ADVANCED USAGE

### Custom Analysis

```python
# Load all journeys
analyzer = LogAnalyzer("output/logs")

# Custom filtering
filtered = []
for journey in analyzer.journeys.values():
    # Complex custom logic
    if (journey.get_total_time() and 
        journey.get_total_time() < 1.0 and
        journey.get_avg_relevance_score() and
        journey.get_avg_relevance_score() > 0.9):
        filtered.append(journey)

print(f"Found {len(filtered)} fast, high-quality queries")
```

### Programmatic Export

```python
import json

# Export custom format
custom_data = []
for journey in analyzer.journeys.values():
    custom_data.append({
        'id': journey.query_id,
        'query': journey.original_query,
        'time': journey.get_total_time(),
        'quality': journey.get_avg_relevance_score()
    })

with open('custom_export.json', 'w') as f:
    json.dump(custom_data, f, indent=2)
```

---

## 📈 BEST PRACTICES

### 1. Regular Monitoring

```python
# Run daily analysis
analyzer = LogAnalyzer("output/logs")
stats = analyzer.get_performance_stats()

# Alert if performance degrades
if stats['avg_total_time'] > 2.0:
    print("⚠️ Warning: Average processing time increased")

if stats['avg_relevance_score'] < 0.7:
    print("⚠️ Warning: Average relevance score decreased")
```

### 2. Archiving Old Logs

```bash
# Archive logs older than 30 days
find output/logs/queries -name "*.json" -mtime +30 -exec mv {} archive/ \;
```

### 3. Performance Optimization

```python
# Identify bottlenecks
stats = analyzer.get_performance_stats()

print("Step Performance:")
print(f"  Enhancement: {stats['avg_enhancement_time']:.3f}s")
print(f"  Retrieval: {stats['avg_retrieval_time']:.3f}s")
print(f"  Generation: {stats['avg_generation_time']:.3f}s")

# Focus optimization on slowest step
```

---

## 🔒 PRIVACY & SECURITY

### Data Retention

- Configure log retention policies
- Regularly archive or delete old logs
- Ensure compliance with data policies

### Sensitive Information

- Logs may contain user queries
- Implement access controls
- Consider encryption for sensitive data

---

## 📚 REFERENCE

### File Locations

- **Data Models**: `models/query_journey.py`
- **Logger**: `journey_logging/journey_logger.py`
- **Tracked Steps**: `steps/tracked_*.py`
- **Analysis Tools**: `tools/*.py`

### Documentation

- **Implementation Status**: `IMPLEMENTATION_STATUS.md`
- **Expansion Plan**: `EXPANSION_PLAN.md`
- **Phase Summaries**: `PHASE_*_COMPLETE.md`

---

**Version**: 1.0.0  
**Status**: ✅ **Production Ready**  
**Support**: See main README for contact information

🚀 **Happy tracking!**

