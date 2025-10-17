#!/bin/bash -l
#PBS -N nanobrain_pbs_rag_stress
#PBS -A FoundEpidem
#PBS -l nodes=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare
#PBS -q debug
#PBS -j oe
#PBS -o stress_test_output.log
#PBS -m abe

# PBS Parallel RAG Stress Test Submit Script
# ===========================================
#
# This script runs comprehensive stress tests for the PBS Parallel RAG system
# on HPC clusters using PBS job scheduler.
#
# Usage:
#   qsub submit_stress_test.pbs
#
# System Compatibility:
#   - Uses standard PBS syntax (compatible with PBS/Torque and PBS Pro)
#   - Configurable through environment variables
#   - System-specific settings via configuration files only
#
# Customization:
#   - Modify PBS directives above for your cluster requirements
#   - Adjust STRESS_TEST_SIZE for different workload sizes
#   - Set NANOBRAIN_PATH to your nanobrain installation directory
#   - Configure environment setup for your system

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE VARIABLES AS NEEDED
# =============================================================================

# Nanobrain installation path (parent of demos/pbs_parallel_rag)
NANOBRAIN_PATH="$(dirname $(dirname ${PBS_O_WORKDIR}))"

# System environment setup
module load frameworks
conda activate nanobrain

# Alternative setups for other systems:
# Option 1: Module system
# module load python/3.9
# module load cuda/11.8  # If GPU support needed

# Option 2: Custom conda environment
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env_name

# Option 3: Virtual environment
# source /path/to/venv/bin/activate

# Stress test configuration
STRESS_TEST_SIZE=1000  # Number of queries for stress test
MAX_WORKERS=16         # Maximum parallel workers
TIMEOUT=7200           # Timeout in seconds (2 hours)

# Output directory for results
OUTPUT_DIR="${PBS_O_WORKDIR}/output/stress_test_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

echo "================================================================================"
echo "🚀 PBS PARALLEL RAG STRESS TEST"
echo "================================================================================"
echo "Job ID: $PBS_JOBID"
echo "Job Name: $PBS_JOBNAME"
echo "Nodes: $PBS_NUM_NODES"
echo "CPUs: $PBS_NP"
echo "Start Time: $(date)"
echo "Working Directory: $PBS_O_WORKDIR"
echo "Nanobrain Path: $NANOBRAIN_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Stress Test Size: $STRESS_TEST_SIZE queries"
echo "Max Workers: $MAX_WORKERS"
echo "================================================================================"

# Change to nanobrain directory
cd "$NANOBRAIN_PATH" || {
    echo "❌ ERROR: Cannot access nanobrain directory: $NANOBRAIN_PATH"
    exit 1
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set Python path
export PYTHONPATH="$NANOBRAIN_PATH:$PYTHONPATH"

# Intel MPI environment setup for Aurora
export I_MPI_PIN_DOMAIN=auto
export I_MPI_FABRICS=shm:ofi
export I_MPI_HYDRA_BOOTSTRAP=pbs

# Force TCP communication for Parsl to avoid AF_UNIX path issues
export PARSL_MULTIPROCESSING_START_METHOD=spawn
export TMPDIR=/tmp

# =============================================================================
# SYSTEM INFORMATION
# =============================================================================

echo ""
echo "📊 SYSTEM INFORMATION"
echo "================================================================================"
echo "Hostname: $(hostname)"
echo "PBS Nodes: $(cat $PBS_NODEFILE | sort | uniq)"
echo "Total CPUs: $(cat $PBS_NODEFILE | wc -l)"
echo "Python Version: $(python3 --version 2>&1)"
echo "Python Path: $(which python3)"
echo "Working Directory: $(pwd)"
echo "Available Memory: $(free -h | grep Mem)"
echo "Disk Space: $(df -h . | tail -1)"
echo "================================================================================"

echo ""
echo "🔍 AURORA NODE ALLOCATION ANALYSIS"
echo "================================================================================"
if [ -f "$PBS_NODEFILE" ]; then
    echo "Node allocation details:"
    echo "  Total slots: $(cat $PBS_NODEFILE | wc -l)"
    echo "  Unique nodes: $(cat $PBS_NODEFILE | sort | uniq | wc -l)"
    echo "  Node distribution:"
    cat $PBS_NODEFILE | sort | uniq -c | while read count node; do
        echo "    $node: $count slots"
    done

    echo ""
    echo "Expected worker distribution:"
    echo "  Max workers: $MAX_WORKERS"
    echo "  Workers per node: approximately $((MAX_WORKERS / $(cat $PBS_NODEFILE | sort | uniq | wc -l)))"

    # Save node allocation for analysis
    echo "Saving node allocation to $OUTPUT_DIR/node_allocation.txt"
    cat $PBS_NODEFILE > "$OUTPUT_DIR/node_allocation.txt"
    cat $PBS_NODEFILE | sort | uniq -c > "$OUTPUT_DIR/node_distribution.txt"
else
    echo "⚠️  PBS_NODEFILE not found - node tracking may be limited"
fi
echo "================================================================================"

# =============================================================================
# DEPENDENCY CHECK
# =============================================================================

echo ""
echo "🔍 DEPENDENCY CHECK"
echo "================================================================================"

# Install/check Python dependencies
echo "Checking Python dependencies..."
if [ -f "demos/pbs_parallel_rag/requirements.txt" ]; then
    echo "Installing/updating dependencies from requirements.txt..."
    pip install -r demos/pbs_parallel_rag/requirements.txt --quiet --user
    echo "✓ Dependencies installed"
else
    echo "⚠️  requirements.txt not found, checking core dependencies manually..."
fi

python3 -c "
import sys
required_packages = ['parsl', 'asyncio', 'psutil', 'yaml', 'pathlib']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg} - MISSING')
        missing.append(pkg)

if missing:
    print(f'ERROR: Missing packages: {missing}')
    sys.exit(1)
else:
    print('✅ All required packages available')
"

if [ $? -ne 0 ]; then
    echo "❌ Dependency check failed"
    exit 1
fi

# Check nanobrain installation
echo "Checking nanobrain installation..."
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, '$NANOBRAIN_PATH')

# Add pbs_parallel_rag directory to path for local imports
pbs_dir = Path('$NANOBRAIN_PATH/demos/pbs_parallel_rag')
if str(pbs_dir) not in sys.path:
    sys.path.insert(0, str(pbs_dir))

try:
    from nanobrain.core.executor import ParslExecutor
    from pbs_query_enhancement_step import PBSQueryEnhancementStep
    print('✅ Nanobrain PBS components available')
except ImportError as e:
    print(f'❌ Nanobrain import failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Nanobrain check failed"
    exit 1
fi

echo "✅ All dependencies satisfied"
echo "================================================================================"

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

echo ""
echo "⚙️  CONFIGURATION VALIDATION"
echo "================================================================================"

# Run configuration tests (from nanobrain root directory)
echo "Running configuration validation tests..."
cd "$NANOBRAIN_PATH"

python3 demos/pbs_parallel_rag/test_comprehensive.py > "$OUTPUT_DIR/config_validation.log" 2>&1
CONFIG_EXIT_CODE=$?

if [ $CONFIG_EXIT_CODE -eq 0 ]; then
    echo "✅ Configuration validation passed"
    echo "📄 Validation log: $OUTPUT_DIR/config_validation.log"
else
    echo "⚠️  Configuration validation failed - proceeding with stress test anyway"
    echo "📄 Check validation log: $OUTPUT_DIR/config_validation.log"
    echo "Last 20 lines of validation log:"
    tail -20 "$OUTPUT_DIR/config_validation.log"
    echo "🚀 Continuing with MPI stress test to validate actual functionality..."
fi

echo "================================================================================"

# =============================================================================
# STRESS TEST EXECUTION
# =============================================================================

echo ""
echo "🧪 STRESS TEST EXECUTION"
echo "================================================================================"
echo "Starting stress test with $STRESS_TEST_SIZE queries..."
echo "Start Time: $(date)"
echo "Timeout: $TIMEOUT seconds"
echo "Output Directory: $OUTPUT_DIR"
echo "================================================================================"

# Set environment variables for stress test
export STRESS_TEST_OUTPUT_DIR="$OUTPUT_DIR"
export STRESS_TEST_SIZE="$STRESS_TEST_SIZE"
export MAX_WORKERS="$MAX_WORKERS"

# Run the stress test with timeout (from nanobrain root directory)
cd "$NANOBRAIN_PATH"
timeout "$TIMEOUT" python3 demos/pbs_parallel_rag/test_stress_1000_queries.py > "$OUTPUT_DIR/stress_test.log" 2>&1
STRESS_EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "📊 STRESS TEST RESULTS"
echo "================================================================================"

if [ $STRESS_EXIT_CODE -eq 0 ]; then
    echo "✅ Stress test completed successfully"
    echo "📄 Full log: $OUTPUT_DIR/stress_test.log"
    
    # Extract key metrics from log
    echo ""
    echo "📈 KEY METRICS:"
    grep -E "(queries processed|Total time|Average time|Throughput|Success rate)" "$OUTPUT_DIR/stress_test.log" || echo "Metrics extraction failed"

    # Aurora node tracking analysis
    echo ""
    echo "🔍 AURORA NODE TRACKING ANALYSIS:"
    if [ -f "$OUTPUT_DIR/node_tracking.json" ]; then
        echo "Node distribution analysis:"
        python3 -c "
import json
try:
    with open('$OUTPUT_DIR/node_tracking.json', 'r') as f:
        data = json.load(f)

    print(f'  Expected nodes: {len(data.get(\"expected_nodes\", []))}')
    print(f'  Actual nodes used: {len(data.get(\"node_usage\", {}))}')
    print(f'  Resource pool usage: {data.get(\"resource_pool_usage\", 0)}/{data.get(\"max_pool_size\", 0)}')

    queries_per_node = data.get('queries_per_node', {})
    if queries_per_node:
        print('  Queries per node:')
        for node, count in sorted(queries_per_node.items()):
            print(f'    {node}: {count} queries')

    # Check compliance
    pool_usage = data.get('resource_pool_usage', 0)
    max_pool = data.get('max_pool_size', 0)
    if pool_usage <= max_pool:
        print('  ✅ Resource pool compliance: PASSED')
    else:
        print('  ❌ Resource pool compliance: FAILED')

except Exception as e:
    print(f'  Error analyzing node tracking: {e}')
"
    else
        echo "  ⚠️  Node tracking data not found"
    fi
    
elif [ $STRESS_EXIT_CODE -eq 124 ]; then
    echo "⏰ Stress test timed out after $TIMEOUT seconds"
    echo "📄 Partial log: $OUTPUT_DIR/stress_test.log"
else
    echo "❌ Stress test failed with exit code: $STRESS_EXIT_CODE"
    echo "📄 Error log: $OUTPUT_DIR/stress_test.log"
    echo ""
    echo "Last 50 lines of stress test log:"
    tail -50 "$OUTPUT_DIR/stress_test.log"
fi

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

echo ""
echo "📊 RESULTS ANALYSIS"
echo "================================================================================"

# Check if output files were generated
# The stress test creates logs in a subdirectory based on the default output path
STRESS_LOG_DIR="$OUTPUT_DIR"
if [ -d "$STRESS_LOG_DIR/queries" ]; then
    QUERY_COUNT=$(find "$STRESS_LOG_DIR/queries" -name "*.json" | wc -l)
    echo "📁 Generated query logs: $QUERY_COUNT files"
    echo "📁 Output directory: $STRESS_LOG_DIR"
    
    # Generate summary report
    python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, '$NANOBRAIN_PATH')

# Add pbs_parallel_rag directory to path for local imports
pbs_dir = Path('$NANOBRAIN_PATH/demos/pbs_parallel_rag')
if str(pbs_dir) not in sys.path:
    sys.path.insert(0, str(pbs_dir))

from tools.analyze_logs import LogAnalyzer
from tools.export_logs import LogExporter

try:
    analyzer = LogAnalyzer('$OUTPUT_DIR')
    analyzer.print_summary()
    
    # Export results
    exporter = LogExporter(analyzer)
    exporter.export_to_csv('$OUTPUT_DIR/stress_test_results.csv')
    exporter.export_summary_to_text('$OUTPUT_DIR/stress_test_summary.txt')
    print('✅ Results exported to CSV and summary files')
except Exception as e:
    print(f'⚠️  Results analysis failed: {e}')
" 2>&1 | tee "$OUTPUT_DIR/analysis.log"

else
    echo "⚠️  No query logs found - stress test may have failed early"
    echo "📁 Expected directory: $STRESS_LOG_DIR/queries"
    echo "📁 Available directories:"
    ls -la "$OUTPUT_DIR" 2>/dev/null || echo "   No output directory found"
fi

# =============================================================================
# CLEANUP AND SUMMARY
# =============================================================================

echo ""
echo "🧹 CLEANUP AND SUMMARY"
echo "================================================================================"

# Collect system resource usage
echo "Final system status:"
echo "Memory usage: $(free -h | grep Mem)"
echo "Disk usage: $(df -h . | tail -1)"

# Create final summary
cat > "$OUTPUT_DIR/job_summary.txt" << EOF
PBS Parallel RAG Stress Test Summary
====================================

Job Information:
- Job ID: $PBS_JOBID
- Job Name: $PBS_JOBNAME
- Nodes: $PBS_NUM_NODES
- CPUs: $PBS_NP
- Start Time: $(date)
- Nanobrain Path: $NANOBRAIN_PATH

Test Configuration:
- Stress Test Size: $STRESS_TEST_SIZE queries
- Max Workers: $MAX_WORKERS
- Timeout: $TIMEOUT seconds

Results:
- Configuration Validation: $([ $CONFIG_EXIT_CODE -eq 0 ] && echo "PASSED" || echo "FAILED")
- Stress Test: $([ $STRESS_EXIT_CODE -eq 0 ] && echo "PASSED" || echo "FAILED (exit code: $STRESS_EXIT_CODE)")
- Output Directory: $OUTPUT_DIR

Files Generated:
- Configuration validation: config_validation.log
- Stress test log: stress_test.log
- Analysis log: analysis.log
- Results CSV: stress_test_results.csv (if successful)
- Summary report: stress_test_summary.txt (if successful)
EOF

echo "📄 Job summary written to: $OUTPUT_DIR/job_summary.txt"
echo ""
echo "================================================================================"
echo "🏁 PBS PARALLEL RAG STRESS TEST COMPLETE"
echo "================================================================================"
echo "End Time: $(date)"
echo "Output Directory: $OUTPUT_DIR"
echo "Overall Status: $([ $STRESS_EXIT_CODE -eq 0 ] && echo "SUCCESS ✅" || echo "FAILED ❌")"
echo "================================================================================"

# Exit with stress test exit code
exit $STRESS_EXIT_CODE
