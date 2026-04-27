#!/bin/bash

# NanoBrain Playwright Test Runner
# Comprehensive test execution script for NanoBrain chatbot testing

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_RESULTS_DIR="$PROJECT_ROOT/test-results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Default values
TEST_TYPE="all"
BROWSER="chromium"
HEADLESS="true"
WORKERS="1"
RETRIES="1"
TIMEOUT="60000"
ENVIRONMENT="testing"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
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

# Function to show usage
show_usage() {
    cat << EOF
NanoBrain Playwright Test Runner

Usage: $0 [OPTIONS]

Options:
    -t, --type TYPE         Test type: all, ui, api, performance (default: all)
    -b, --browser BROWSER   Browser: chromium, firefox, webkit (default: chromium)
    -h, --headless BOOL     Run headless: true, false (default: true)
    -w, --workers NUM       Number of workers (default: 1)
    -r, --retries NUM       Number of retries (default: 1)
    -e, --env ENV          Environment: testing, staging, production (default: testing)
    --timeout MS           Test timeout in milliseconds (default: 60000)
    --help                 Show this help message

Examples:
    $0                                    # Run all tests with defaults
    $0 -t ui -b firefox --headless false # Run UI tests in Firefox with head
    $0 -t performance -w 2               # Run performance tests with 2 workers
    $0 -t api -e staging                 # Run API tests against staging

Test Types:
    all         - Run all test suites
    ui          - Run chatbot UI tests only
    api         - Run API integration tests only
    performance - Run performance and load tests only

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -b|--browser)
            BROWSER="$2"
            shift 2
            ;;
        -h|--headless)
            HEADLESS="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -r|--retries)
            RETRIES="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate test type
case $TEST_TYPE in
    all|ui|api|performance)
        ;;
    *)
        print_error "Invalid test type: $TEST_TYPE"
        show_usage
        exit 1
        ;;
esac

# Function to setup environment
setup_environment() {
    print_status "Setting up test environment..."
    
    # Create test results directory
    mkdir -p "$TEST_RESULTS_DIR"
    mkdir -p "$TEST_RESULTS_DIR/screenshots"
    mkdir -p "$TEST_RESULTS_DIR/videos"
    mkdir -p "$TEST_RESULTS_DIR/traces"
    
    # Set environment variables
    export NANOBRAIN_ENV="$ENVIRONMENT"
    export NANOBRAIN_LOG_LEVEL="INFO"
    export PLAYWRIGHT_BROWSERS_PATH="$PROJECT_ROOT/node_modules/.cache/ms-playwright"
    
    # Check if Playwright is installed
    if ! command -v npx &> /dev/null; then
        print_error "npx not found. Please install Node.js and npm."
        exit 1
    fi
    
    # Check if Playwright test is available
    if ! npx playwright --version &> /dev/null; then
        print_error "Playwright not found. Please run 'npm install @playwright/test'"
        exit 1
    fi
    
    print_success "Environment setup complete"
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Node.js version
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_status "Node.js version: $NODE_VERSION"
    else
        print_error "Node.js not found"
        exit 1
    fi
    
    # Check available memory
    if command -v free &> /dev/null; then
        AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.1f", $7/1024}')
        print_status "Available memory: ${AVAILABLE_MEM}GB"
    fi
    
    # Check disk space
    AVAILABLE_DISK=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    print_status "Available disk space: $AVAILABLE_DISK"
    
    print_success "System requirements check complete"
}

# Function to run specific test suite
run_test_suite() {
    local suite_name=$1
    local test_pattern=$2
    
    print_status "Running $suite_name tests..."
    
    local cmd="npx playwright test"
    
    # Add test pattern if specified
    if [[ -n "$test_pattern" ]]; then
        cmd="$cmd $test_pattern"
    fi
    
    # Add browser selection
    cmd="$cmd --project=$BROWSER"
    
    # Add workers
    cmd="$cmd --workers=$WORKERS"
    
    # Add retries
    cmd="$cmd --retries=$RETRIES"
    
    # Add headless mode
    if [[ "$HEADLESS" == "true" ]]; then
        cmd="$cmd --headed=false"
    else
        cmd="$cmd --headed=true"
    fi
    
    # Add timeout
    cmd="$cmd --timeout=$TIMEOUT"
    
    # Add reporter
    cmd="$cmd --reporter=html,line,json"
    
    print_status "Executing: $cmd"
    
    if eval "$cmd"; then
        print_success "$suite_name tests completed successfully"
        return 0
    else
        print_error "$suite_name tests failed"
        return 1
    fi
}

# Function to generate test report
generate_report() {
    print_status "Generating test report..."
    
    local report_file="$TEST_RESULTS_DIR/test-report-$TIMESTAMP.md"
    
    cat > "$report_file" << EOF
# NanoBrain Playwright Test Report

**Generated:** $(date)
**Test Type:** $TEST_TYPE
**Browser:** $BROWSER
**Environment:** $ENVIRONMENT
**Headless:** $HEADLESS

## Test Configuration

- Workers: $WORKERS
- Retries: $RETRIES
- Timeout: ${TIMEOUT}ms

## Test Results

EOF
    
    # Add results from JSON report if available
    if [[ -f "$TEST_RESULTS_DIR/playwright-results.json" ]]; then
        echo "Results available in: test-results/playwright-results.json" >> "$report_file"
    fi
    
    # Add links to artifacts
    echo "" >> "$report_file"
    echo "## Test Artifacts" >> "$report_file"
    echo "" >> "$report_file"
    echo "- HTML Report: test-results/playwright-report/index.html" >> "$report_file"
    echo "- Screenshots: test-results/screenshots/" >> "$report_file"
    echo "- Videos: test-results/videos/" >> "$report_file"
    echo "- Traces: test-results/traces/" >> "$report_file"
    
    print_success "Test report generated: $report_file"
}

# Function to cleanup
cleanup() {
    print_status "Cleaning up..."
    
    # Archive old test results
    if [[ -d "$TEST_RESULTS_DIR/archives" ]]; then
        find "$TEST_RESULTS_DIR/archives" -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true
    fi
    
    print_success "Cleanup complete"
}

# Main execution
main() {
    print_status "Starting NanoBrain Playwright Tests"
    print_status "Test Type: $TEST_TYPE | Browser: $BROWSER | Environment: $ENVIRONMENT"
    
    # Setup
    setup_environment
    check_requirements
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    local exit_code=0
    
    # Run tests based on type
    case $TEST_TYPE in
        "ui")
            run_test_suite "UI" "tests/playwright/chatbot/" || exit_code=1
            ;;
        "api")
            run_test_suite "API" "tests/playwright/api/" || exit_code=1
            ;;
        "performance")
            run_test_suite "Performance" "tests/playwright/performance/" || exit_code=1
            ;;
        "all")
            run_test_suite "UI" "tests/playwright/chatbot/" || exit_code=1
            run_test_suite "API" "tests/playwright/api/" || exit_code=1
            run_test_suite "Performance" "tests/playwright/performance/" || exit_code=1
            ;;
    esac
    
    # Generate report
    generate_report
    
    # Cleanup
    cleanup
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "All tests completed successfully!"
        print_status "View results: open test-results/playwright-report/index.html"
    else
        print_error "Some tests failed. Check the reports for details."
    fi
    
    exit $exit_code
}

# Run main function
main "$@"
