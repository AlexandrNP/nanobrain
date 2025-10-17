# PBS Parallel RAG Demo 🚀

**Purpose**: Demonstrate parallel processing of multiple RAG queries using PARSL with PBS provider  
**Status**: ✅ **PRODUCTION READY** with complete PBS submit scripts and stress testing  

## 🚀 Quick Start

### For HPC Clusters (Recommended)

```bash
# Copy to your cluster
scp -r demos/pbs_parallel_rag user@cluster:/path/to/nanobrain/demos/

# On cluster - Quick validation (10 queries, 30 min)
cd /path/to/nanobrain/demos/pbs_parallel_rag
qsub submit_quick_test.sh

# Full stress test (1000 queries, 2 hours)
qsub submit_stress_test.sh

# Monitor jobs
qstat -u onarykov

# Or use helper script
./run_pbs_tests.sh quick
./run_pbs_tests.sh stress
./run_pbs_tests.sh status
```

**Note**: Scripts are configured for:
- User: `onarykov`
- Project: `FoundEpidem`
- Environment: `module load frameworks; conda activate nanobrain`

Modify PBS directives and environment setup if using different settings.

**Dependencies**: Install required packages with:
```bash
pip install -r demos/pbs_parallel_rag/requirements.txt
```

### For Local Testing

```bash
cd /path/to/nanobrain/demos/pbs_parallel_rag

# Test configuration
python3 test_basic_structure.py
python3 test_comprehensive.py

# Test stress test setup
python3 test_stress_config.py
```

---

## 📁 DEMO STRUCTURE

```
demos/pbs_parallel_rag/
├── README.md                           # This file
├── 
├── submit_stress_test.sh               # Main PBS submit script ⭐ NEW
├── submit_quick_test.sh                # Quick validation script ⭐ NEW
├── run_pbs_tests.sh                    # Helper script for job management ⭐ NEW
│
├── test_parallel_rag.py                # Basic functionality test
├── test_stress_1000_queries.py         # Comprehensive stress test
├── test_comprehensive.py               # Configuration validation
├── test_stress_config.py               # Stress test validation ⭐ NEW
├── test_basic_structure.py             # Structure validation ⭐ NEW
├── analyze_node_distribution.py        # Aurora node tracking analyzer ⭐ NEW
│
├── pbs_query_enhancement_step.py       # PBS-enabled step class
├── shared_vector_database.py           # Shared resources
│
├── models/                             # Data models
│   ├── __init__.py
│   └── query_journey.py               # QueryJourney, Document, StepMetadata
│
├── journey_logging/                    # Journey logging
│   ├── __init__.py
│   └── journey_logger.py              # QueryJourneyLogger
│
├── steps/                              # Tracked workflow steps
│   ├── __init__.py
│   ├── tracked_query_enhancement_step.py
│   ├── tracked_vector_search_step.py
│   └── tracked_response_generation_step.py
│
├── tools/                              # Analysis tools
│   ├── __init__.py
│   ├── analyze_logs.py                # LogAnalyzer
│   ├── export_logs.py                 # LogExporter
│   └── view_journey.py                # Journey viewer
│
├── config/
│   ├── workflow/
│   │   └── parallel_rag_workflow.yml  # Workflow with PBS integration
│   ├── steps/
│   │   ├── prompt_enhancement_step.yml # Step configurations
│   │   ├── retrieval_specialist_step.yml
│   │   ├── analysis_specialist_step.yml
│   │   ├── synthesis_specialist_step.yml
│   │   └── quality_assurance_step.yml
│   └── executors/
│       ├── pbs_executor.yml           # Main PBS executor config ⭐ NEW
│       └── pbs_executor_small.yml     # Small cluster config ⭐ NEW
│
└── output/                            # Test results and logs
    ├── stress_test_*/                 # Stress test outputs
    ├── quick_test_*/                  # Quick test outputs
    └── test_logs/                     # Validation test logs
```

## 🎯 PBS Submit Scripts

### **submit_stress_test.sh** - Main Stress Test

**Purpose**: Comprehensive stress testing with 1000 queries  
**Resources**: 2 nodes, 8 CPUs each, 2 hours  
**Queue**: batch  

```bash
qsub submit_stress_test.sh
```

### **submit_quick_test.sh** - Quick Validation

**Purpose**: Fast validation with 10 queries  
**Resources**: 1 node, 4 CPUs, 30 minutes  
**Queue**: debug  

```bash
qsub submit_quick_test.sh
```

### **run_pbs_tests.sh** - Helper Script

**Purpose**: Convenience script for job management  

```bash
./run_pbs_tests.sh quick    # Submit quick test
./run_pbs_tests.sh stress   # Submit stress test  
./run_pbs_tests.sh status   # Check job status
./run_pbs_tests.sh help     # Show help
```

## ⚙️ Configuration

### PBS Directives

Both scripts are configured with:
- **User**: `onarykov`
- **Project**: `FoundEpidem`
- **Environment**: `module load frameworks; conda activate nanobrain`

### System Adaptation

For different systems, modify the PBS directives in the submit scripts:

```bash
# Edit submit_stress_test.sh or submit_quick_test.sh
#PBS -A your_project        # Your project allocation
#PBS -u your_username       # Your username
#PBS -q your_queue          # Your queue name

# Edit environment setup section
module load your_modules
conda activate your_env
```

## 🧪 Testing

### Configuration Validation

```bash
python3 test_basic_structure.py     # Test file structure
python3 test_comprehensive.py       # Test configuration
python3 test_stress_config.py       # Test stress test setup
```

**Note**: All Python calls use `python3` for Aurora compatibility.

### Job Submission

```bash
# Quick validation (recommended first)
qsub submit_quick_test.sh

# Monitor job
qstat -u onarykov

# Check output
cat quick_test_output.log

# Full stress test (after validation)
qsub submit_stress_test.sh

# Analyze Aurora node distribution (after completion)
python3 analyze_node_distribution.py output/stress_test_*/node_tracking.json
```

## 📊 Results

### Output Files

- `*_output.log` - PBS job logs
- `output/stress_test_*/` - Stress test results
- `output/quick_test_*/` - Quick test results
- `*.csv` - Performance analysis data
- `node_tracking.json` - Aurora node distribution data ⭐ NEW
- `node_allocation.txt` - PBS node allocation info ⭐ NEW

### Key Metrics

- Total queries processed
- Success rate
- Average processing time
- Throughput (queries/second)
- Resource utilization
- **Aurora Node Distribution** ⭐ NEW:
  - Node coverage percentage
  - Queries per node
  - Resource pool compliance
  - Load balancing analysis

## 🎯 Next Steps

### For New Users
1. **Copy to your cluster** and run quick validation test
2. **Modify PBS directives** for your system (user, project, queues)
3. **Update environment setup** for your Python environment
4. **Run stress tests** to validate performance

### For Advanced Users
1. **Customize configurations** in `config/executors/` for your cluster
2. **Modify submit scripts** for specific resource requirements
3. **Scale testing** with custom parameters via environment variables
4. **Integrate monitoring** with your cluster's monitoring systems

**Ready for production HPC deployment!** 🚀
