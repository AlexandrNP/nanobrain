#!/bin/bash

# PBS Parallel RAG Test Runner
# ============================
#
# Helper script to submit and monitor PBS parallel RAG tests
#
# Usage:
#   ./run_pbs_tests.sh [quick|stress|status|help]
#
# Commands:
#   quick   - Submit quick validation test (10 queries, 30 min)
#   stress  - Submit full stress test (1000 queries, 2 hours)
#   status  - Check status of running jobs
#   help    - Show this help message

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if PBS is available
check_pbs() {
    if ! command -v qsub &> /dev/null; then
        print_error "PBS not found. This script requires PBS/Torque or PBS Pro to be installed."
        exit 1
    fi

    if ! command -v qstat &> /dev/null; then
        print_error "qstat command not found. PBS may not be properly configured."
        exit 1
    fi
}

# Submit quick test
submit_quick_test() {
    print_header "SUBMITTING QUICK TEST"
    
    if [ ! -f "submit_quick_test.sh" ]; then
        print_error "submit_quick_test.sh not found in current directory"
        exit 1
    fi
    
    print_info "Submitting quick validation test..."
    print_info "  - 10 queries"
    print_info "  - 1 node, 4 CPUs"
    print_info "  - 30 minute timeout"
    print_info "  - Basic functionality validation"
    
    JOB_ID=$(qsub submit_quick_test.sh)
    
    if [ $? -eq 0 ]; then
        print_success "Quick test submitted successfully"
        print_info "Job ID: $JOB_ID"
        print_info "Monitor with: qstat $JOB_ID"
        print_info "Or use: ./run_pbs_tests.sh status"
        
        echo ""
        print_info "The quick test will:"
        echo "  1. Validate configuration"
        echo "  2. Test basic functionality"
        echo "  3. Run mini stress test with 10 queries"
        echo "  4. Generate validation report"
        
    else
        print_error "Failed to submit quick test"
        exit 1
    fi
}

# Submit stress test
submit_stress_test() {
    print_header "SUBMITTING STRESS TEST"
    
    if [ ! -f "submit_stress_test.sh" ]; then
        print_error "submit_stress_test.sh not found in current directory"
        exit 1
    fi
    
    print_warning "This will submit a comprehensive stress test:"
    print_info "  - 1000 queries"
    print_info "  - 2 nodes, 8 CPUs each"
    print_info "  - 2 hour timeout"
    print_info "  - Full performance analysis"
    
    echo ""
    read -p "Are you sure you want to submit the stress test? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Submitting stress test..."
        
        JOB_ID=$(qsub submit_stress_test.sh)
        
        if [ $? -eq 0 ]; then
            print_success "Stress test submitted successfully"
            print_info "Job ID: $JOB_ID"
            print_info "Monitor with: qstat $JOB_ID"
            print_info "Or use: ./run_pbs_tests.sh status"
            
            echo ""
            print_info "The stress test will:"
            echo "  1. Validate all configurations"
            echo "  2. Run comprehensive dependency checks"
            echo "  3. Execute 1000 parallel queries"
            echo "  4. Generate performance analysis"
            echo "  5. Export results to CSV"
            
        else
            print_error "Failed to submit stress test"
            exit 1
        fi
    else
        print_info "Stress test submission cancelled"
    fi
}

# Check job status
check_status() {
    print_header "PBS JOB STATUS"
    
    # Check for nanobrain jobs
    JOBS=$(qstat -u $USER | grep "nanobrain_pbs_rag" || true)

    if [ -z "$JOBS" ]; then
        print_info "No nanobrain PBS RAG jobs found"
        echo ""
        print_info "To submit tests:"
        echo "  Quick test:  ./run_pbs_tests.sh quick"
        echo "  Stress test: ./run_pbs_tests.sh stress"
    else
        print_info "Active nanobrain PBS RAG jobs:"
        echo ""
        echo "Job ID       Name                    User     Time Use S Queue"
        echo "------------ ----------------------- -------- -------- - -----"
        echo "$JOBS"
        
        echo ""
        print_info "Job details:"
        
        # Get job IDs
        JOB_IDS=$(echo "$JOBS" | awk '{print $1}' | cut -d'.' -f1)
        
        for JOB_ID in $JOB_IDS; do
            echo ""
            echo "Job $JOB_ID:"
            qstat -f "$JOB_ID" | grep -E "(Job_Name|job_state|queue|Resource_List|start_time|exec_host)" | sed 's/^/  /'
        done
    fi
    
    echo ""
    print_info "Recent output files:"
    find . -name "*test_output.log" -mtime -1 2>/dev/null | head -5 | while read file; do
        echo "  $file ($(stat -c %y "$file" | cut -d' ' -f1-2))"
    done
    
    echo ""
    print_info "Recent output directories:"
    find output -name "*test_*" -type d -mtime -1 2>/dev/null | head -5 | while read dir; do
        echo "  $dir ($(stat -c %y "$dir" | cut -d' ' -f1-2))"
    done
}

# Show help
show_help() {
    print_header "PBS PARALLEL RAG TEST RUNNER"
    
    echo "This script helps you submit and monitor PBS parallel RAG tests."
    echo ""
    echo "Usage:"
    echo "  ./run_pbs_tests.sh [command]"
    echo ""
    echo "Commands:"
    echo "  quick   - Submit quick validation test"
    echo "            • 10 queries, 1 node, 30 minutes"
    echo "            • Validates configuration and basic functionality"
    echo "            • Recommended before running stress test"
    echo ""
    echo "  stress  - Submit full stress test"
    echo "            • 1000 queries, 2 nodes, 2 hours"
    echo "            • Comprehensive performance testing"
    echo "            • Generates detailed analysis and reports"
    echo ""
    echo "  status  - Check status of running jobs"
    echo "            • Shows active nanobrain PBS jobs"
    echo "            • Lists recent output files"
    echo "            • Displays job details"
    echo ""
    echo "  help    - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_pbs_tests.sh quick     # Run quick validation"
    echo "  ./run_pbs_tests.sh stress    # Run full stress test"
    echo "  ./run_pbs_tests.sh status    # Check job status"
    echo ""
    echo "Prerequisites:"
    echo "  • PBS/Torque job scheduler"
    echo "  • Python 3.7+ with required packages"
    echo "  • Nanobrain installation"
    echo "  • Access to PBS queue (default: 'batch')"
    echo ""
    echo "Configuration:"
    echo "  • Edit PBS directives in submit_*.pbs files"
    echo "  • Modify resource requirements as needed"
    echo "  • Set appropriate queue names for your cluster"
    echo ""
    echo "Output:"
    echo "  • Job logs: *_output.log"
    echo "  • Results: output/[quick|stress]_test_YYYYMMDD_HHMMSS/"
    echo "  • Analysis: CSV files and summary reports"
}

# Main script logic
main() {
    # Check PBS availability
    check_pbs
    
    case "${1:-help}" in
        "quick")
            submit_quick_test
            ;;
        "stress")
            submit_stress_test
            ;;
        "status")
            check_status
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
