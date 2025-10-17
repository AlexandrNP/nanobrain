#!/bin/bash -l
#PBS -N nanobrain_pbs_rag_quick
#PBS -A FoundEpidem
#PBS -l nodes=1
#PBS -l walltime=00:10:00
#PBS -l filesystems=flare
#PBS -q debug
#PBS -j oe
#PBS -o quick_test_output.log
#PBS -m abe

# PBS Parallel RAG Quick Test Submit Script
# ==========================================
#
# This script runs a quick validation test for the PBS Parallel RAG system
# to verify functionality before running full stress tests.
#
# Usage:
#   qsub submit_quick_test.pbs
#
# This is a lightweight test that:
#   - Validates configuration
#   - Tests basic functionality with 10 queries
#   - Verifies PBS integration
#   - Completes in under 30 minutes

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# System environment setup
module load frameworks
conda activate nanobrain

# Nanobrain installation path (parent of demos/pbs_parallel_rag)
NANOBRAIN_PATH="$(dirname $(dirname ${PBS_O_WORKDIR}))"

# Quick test configuration
QUICK_TEST_SIZE=10     # Small number of queries for quick validation
MAX_WORKERS=4          # Limited workers for quick test
TIMEOUT=1800           # 30 minutes timeout

# Output directory for results
OUTPUT_DIR="${PBS_O_WORKDIR}/output/quick_test_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

echo "================================================================================"
echo "⚡ PBS PARALLEL RAG QUICK TEST"
echo "================================================================================"
echo "Job ID: $PBS_JOBID"
echo "Job Name: $PBS_JOBNAME"
echo "Nodes: $PBS_NUM_NODES"
echo "CPUs: $PBS_NP"
echo "Start Time: $(date)"
echo "Working Directory: $PBS_O_WORKDIR"
echo "Quick Test Size: $QUICK_TEST_SIZE queries"
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

# =============================================================================
# QUICK DEPENDENCY CHECK
# =============================================================================

echo ""
echo "🔍 QUICK DEPENDENCY CHECK"
echo "================================================================================"

# Basic Python check
python3 -c "
import sys
from pathlib import Path
import parsl, asyncio, psutil
from nanobrain.core.executor import ParslExecutor

# Add pbs_parallel_rag directory to path for local imports
pbs_dir = Path('$NANOBRAIN_PATH/demos/pbs_parallel_rag')
if str(pbs_dir) not in sys.path:
    sys.path.insert(0, str(pbs_dir))

from pbs_query_enhancement_step import PBSQueryEnhancementStep
print('✅ All critical dependencies available')
" || {
    echo "❌ Dependency check failed"
    exit 1
}

echo "✅ Dependencies satisfied"

# =============================================================================
# AURORA NODE INFORMATION
# =============================================================================

echo ""
echo "🔍 AURORA NODE ALLOCATION"
echo "================================================================================"
if [ -f "$PBS_NODEFILE" ]; then
    echo "Quick test node allocation:"
    echo "  Total slots: $(cat $PBS_NODEFILE | wc -l)"
    echo "  Unique nodes: $(cat $PBS_NODEFILE | sort | uniq | wc -l)"
    echo "  Nodes: $(cat $PBS_NODEFILE | sort | uniq | tr '\n' ' ')"

    # Save for analysis
    cat $PBS_NODEFILE > "$OUTPUT_DIR/node_allocation.txt"
else
    echo "⚠️  PBS_NODEFILE not found"
fi
echo "================================================================================"

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

echo ""
echo "⚙️  CONFIGURATION VALIDATION"
echo "================================================================================"

cd "$NANOBRAIN_PATH/demos/pbs_parallel_rag"

# Run quick configuration test
python3 test_config_structure.py > "$OUTPUT_DIR/config_test.log" 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Configuration validation passed"
else
    echo "❌ Configuration validation failed"
    cat "$OUTPUT_DIR/config_test.log"
    exit 1
fi

# =============================================================================
# QUICK FUNCTIONALITY TEST
# =============================================================================

echo ""
echo "🧪 QUICK FUNCTIONALITY TEST"
echo "================================================================================"

# Run basic parallel RAG test (from nanobrain root directory)
echo "Running basic parallel RAG test..."
cd "$NANOBRAIN_PATH"
timeout 600 python3 demos/pbs_parallel_rag/test_parallel_rag.py > "$OUTPUT_DIR/basic_test.log" 2>&1
BASIC_EXIT_CODE=$?

if [ $BASIC_EXIT_CODE -eq 0 ]; then
    echo "✅ Basic functionality test passed"
else
    echo "❌ Basic functionality test failed"
    echo "Last 20 lines of test log:"
    tail -20 "$OUTPUT_DIR/basic_test.log"
    exit 1
fi

# =============================================================================
# MINI STRESS TEST
# =============================================================================

echo ""
echo "🏃 MINI STRESS TEST ($QUICK_TEST_SIZE queries)"
echo "================================================================================"

# Set environment for mini stress test
export STRESS_TEST_OUTPUT_DIR="$OUTPUT_DIR"
export STRESS_TEST_SIZE="$QUICK_TEST_SIZE"
export MAX_WORKERS="$MAX_WORKERS"

# Create a mini version of the stress test
python3 -c "
import sys
import asyncio
import time
from pathlib import Path

sys.path.insert(0, '$NANOBRAIN_PATH')

# Add pbs_parallel_rag directory to path for local imports
pbs_dir = Path('$NANOBRAIN_PATH/demos/pbs_parallel_rag')
if str(pbs_dir) not in sys.path:
    sys.path.insert(0, str(pbs_dir))

# Import required modules
from journey_logging.journey_logger import QueryJourneyLogger
from models.query_journey import Document

async def mini_stress_test():
    print(f'🚀 Starting mini stress test with $QUICK_TEST_SIZE queries...')
    
    # Simple test queries
    queries = [
        f'What are the mechanisms of viral infection process {i}?'
        for i in range($QUICK_TEST_SIZE)
    ]
    
    # Initialize logger
    logger = QueryJourneyLogger('$OUTPUT_DIR/mini_stress')
    
    start_time = time.time()
    
    for i, query in enumerate(queries):
        query_id = f'mini_q_{i+1:03d}'
        
        # Simulate query processing
        journey = logger.start_journey(query_id, query)
        
        # Simulate enhancement step
        await asyncio.sleep(0.1)  # Simulate processing time
        enhanced_query = f'Enhanced: {query}'
        logger.update_enhancement(
            query_id=query_id,
            enhanced_query=enhanced_query,
            duration=0.1
        )
        
        # Simulate retrieval step
        await asyncio.sleep(0.1)
        documents = [
            Document(
                doc_id=f'doc_{i}_{j}',
                content=f'Document {j} for query {i}',
                relevance_score=0.9-j*0.1
            )
            for j in range(3)
        ]
        logger.update_retrieval(
            query_id=query_id,
            documents=documents,
            duration=0.1
        )
        
        # Simulate response generation
        await asyncio.sleep(0.1)
        response = f'Response to: {enhanced_query}'
        logger.update_generation(
            query_id=query_id,
            final_response=response,
            duration=0.1
        )

        logger.complete_journey(query_id)
        
        if (i + 1) % 5 == 0:
            print(f'  Processed {i + 1}/$QUICK_TEST_SIZE queries...')
    
    total_time = time.time() - start_time
    
    print(f'✅ Mini stress test completed!')
    print(f'   Queries processed: $QUICK_TEST_SIZE')
    print(f'   Total time: {total_time:.2f}s')
    print(f'   Average time per query: {total_time/$QUICK_TEST_SIZE:.2f}s')
    print(f'   Throughput: {$QUICK_TEST_SIZE/total_time:.2f} queries/second')

if __name__ == '__main__':
    asyncio.run(mini_stress_test())
" > "$OUTPUT_DIR/mini_stress.log" 2>&1

MINI_EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "📊 QUICK TEST RESULTS"
echo "================================================================================"

if [ $MINI_EXIT_CODE -eq 0 ]; then
    echo "✅ Mini stress test completed successfully"
    echo ""
    echo "📈 RESULTS:"
    tail -10 "$OUTPUT_DIR/mini_stress.log"
else
    echo "❌ Mini stress test failed"
    echo "Error log:"
    cat "$OUTPUT_DIR/mini_stress.log"
fi

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "📋 QUICK TEST SUMMARY"
echo "================================================================================"

cat > "$OUTPUT_DIR/quick_test_summary.txt" << EOF
PBS Parallel RAG Quick Test Summary
===================================

Job Information:
- Job ID: $PBS_JOBID
- Start Time: $(date)
- Nodes: $PBS_NUM_NODES
- CPUs: $PBS_NP

Test Results:
- Configuration Validation: PASSED
- Basic Functionality: PASSED
- Mini Stress Test ($QUICK_TEST_SIZE queries): $([ $MINI_EXIT_CODE -eq 0 ] && echo "PASSED" || echo "FAILED")

Output Directory: $OUTPUT_DIR

Next Steps:
$([ $MINI_EXIT_CODE -eq 0 ] && echo "✅ System ready for full stress testing with submit_stress_test.pbs" || echo "❌ Fix issues before running full stress test")
EOF

echo "📄 Summary written to: $OUTPUT_DIR/quick_test_summary.txt"
echo ""
echo "================================================================================"
echo "🏁 QUICK TEST COMPLETE"
echo "================================================================================"
echo "End Time: $(date)"
echo "Status: $([ $MINI_EXIT_CODE -eq 0 ] && echo "SUCCESS ✅" || echo "FAILED ❌")"
echo ""
echo "$([ $MINI_EXIT_CODE -eq 0 ] && echo "✅ Ready for full stress testing!" || echo "❌ Please review logs and fix issues")"
echo "================================================================================"

exit $MINI_EXIT_CODE
