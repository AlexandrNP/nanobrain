#!/bin/bash

# NanoBrain Viral Protein Analysis - Production Deployment Script
# 
# This script provides a complete deployment solution for the NanoBrain
# viral protein analysis framework with Docker containers, monitoring,
# and production configuration.

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/nanobrain_deploy.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}❌ $1${NC}" | tee -a "$LOG_FILE"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root is not recommended for production"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    log_success "Docker found: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    log_success "Docker Compose found: $(docker-compose --version)"
    
    # Check available disk space (require at least 10GB)
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    required_space=10485760  # 10GB in KB
    
    if [[ $available_space -lt $required_space ]]; then
        log_error "Insufficient disk space. Required: 10GB, Available: $((available_space/1024/1024))GB"
        exit 1
    fi
    log_success "Sufficient disk space available: $((available_space/1024/1024))GB"
    
    # Check memory (require at least 4GB)
    total_memory=$(free -m | awk 'NR==2{print $2}')
    required_memory=4096  # 4GB in MB
    
    if [[ $total_memory -lt $required_memory ]]; then
        log_error "Insufficient memory. Required: 4GB, Available: ${total_memory}MB"
        exit 1
    fi
    log_success "Sufficient memory available: ${total_memory}MB"
}

# Setup environment
setup_environment() {
    log "Setting up environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create necessary directories
    mkdir -p data/{cache,results,logs} || true
    mkdir -p logs/{nginx,app} || true
    mkdir -p docker/{monitoring/grafana/{dashboards,provisioning},nginx/ssl} || true
    
    # Set permissions
    chmod 755 data logs docker
    
    # Create .env file if it doesn't exist
    if [[ ! -f .env ]]; then
        log "Creating .env file..."
        cat > .env << EOF
# NanoBrain Production Environment Configuration

# Database passwords
POSTGRES_PASSWORD=nanobrain_secure_$(openssl rand -hex 16)
JWT_SECRET=jwt_secret_$(openssl rand -hex 32)
GRAFANA_PASSWORD=grafana_admin_$(openssl rand -hex 16)

# Application settings
NANOBRAIN_ENV=production
NANOBRAIN_LOG_LEVEL=INFO

# External API settings
PUBMED_EMAIL=onarykov@anl.gov
BVBRC_ANONYMOUS_ACCESS=true

# Performance settings
MAX_MEMORY_GB=4
MAX_CPU_CORES=8
CACHE_SIZE_GB=2

# Security settings
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=http://localhost,https://localhost

# Monitoring settings
PROMETHEUS_RETENTION=200h
GRAFANA_ENABLE_SIGNUP=false
EOF
        log_success "Created .env file with secure random passwords"
    else
        log_success ".env file already exists"
    fi
    
    # Source environment variables
    source .env
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build main application image
    docker-compose build --no-cache nanobrain-app
    
    log_success "Docker images built successfully"
}

# Start services
start_services() {
    log "Starting services..."
    
    cd "$PROJECT_ROOT"
    
    # Start core services first
    docker-compose up -d postgres redis
    
    # Wait for database to be ready
    log "Waiting for database to be ready..."
    timeout=60
    while ! docker-compose exec -T postgres pg_isready -U nanobrain -d nanobrain_viral &>/dev/null; do
        sleep 2
        timeout=$((timeout - 2))
        if [[ $timeout -le 0 ]]; then
            log_error "Database failed to start within 60 seconds"
            exit 1
        fi
    done
    log_success "Database is ready"
    
    # Start monitoring services
    docker-compose up -d prometheus grafana
    
    # Start main application
    docker-compose up -d nanobrain-app
    
    # Start reverse proxy
    docker-compose up -d nginx
    
    log_success "All services started successfully"
}

# Perform health checks
health_check() {
    log "Performing health checks..."
    
    # Check application health
    timeout=120
    while ! curl -f http://localhost:8000/health &>/dev/null; do
        sleep 5
        timeout=$((timeout - 5))
        if [[ $timeout -le 0 ]]; then
            log_error "Application health check failed"
            return 1
        fi
    done
    log_success "Application is healthy"
    
    # Check Prometheus
    if curl -f http://localhost:9090/-/healthy &>/dev/null; then
        log_success "Prometheus is healthy"
    else
        log_warning "Prometheus health check failed"
    fi
    
    # Check Grafana
    if curl -f http://localhost:3000/api/health &>/dev/null; then
        log_success "Grafana is healthy"
    else
        log_warning "Grafana health check failed"
    fi
    
    # Test EEEV workflow endpoint
    if curl -f http://localhost:8000/api/v1/workflows/eeev &>/dev/null; then
        log_success "EEEV workflow endpoint is accessible"
    else
        log_warning "EEEV workflow endpoint check failed"
    fi
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    cd "$PROJECT_ROOT"
    
    # Create Grafana dashboards configuration
    mkdir -p docker/monitoring/grafana/provisioning/{dashboards,datasources}
    
    # Create datasource configuration
    cat > docker/monitoring/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    # Create dashboard provisioning configuration
    cat > docker/monitoring/grafana/provisioning/dashboards/dashboard.yml << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

    # Create basic dashboard
    cat > docker/monitoring/grafana/dashboards/nanobrain-overview.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "NanoBrain Viral Protein Analysis Overview",
    "tags": ["nanobrain", "viral-protein"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "System Overview",
        "type": "stat",
        "targets": [
          {
            "expr": "nanobrain_cpu_usage_percent",
            "legendFormat": "CPU Usage"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF

    log_success "Monitoring configuration created"
}

# Display deployment information
show_deployment_info() {
    log "Deployment completed successfully! 🎉"
    echo
    echo "======================================================================"
    echo "🧬 NanoBrain Viral Protein Analysis - Production Deployment"
    echo "======================================================================"
    echo
    echo "📊 Service URLs:"
    echo "  • Main Application:     http://localhost:8000"
    echo "  • Production Dashboard: http://localhost:8002"
    echo "  • Prometheus:          http://localhost:9090"
    echo "  • Grafana:             http://localhost:3000"
    echo
    echo "🔑 Default Credentials:"
    echo "  • Grafana Admin:       admin / $(grep GRAFANA_PASSWORD .env | cut -d'=' -f2)"
    echo
    echo "🧬 EEEV Workflow:"
    echo "  • API Endpoint:        http://localhost:8000/api/v1/workflows/eeev"
    echo "  • Documentation:       http://localhost:8000/docs"
    echo "  • Health Check:        http://localhost:8000/health"
    echo
    echo "📋 Management Commands:"
    echo "  • View logs:           docker-compose logs -f nanobrain-app"
    echo "  • Stop services:       docker-compose down"
    echo "  • Restart services:    docker-compose restart"
    echo "  • Scale app:           docker-compose up -d --scale nanobrain-app=2"
    echo
    echo "🔍 Monitoring:"
    echo "  • Metrics endpoint:    http://localhost:8000/metrics"
    echo "  • Prometheus targets:  http://localhost:9090/targets"
    echo "  • Container stats:     docker stats"
    echo
    echo "📁 Important Directories:"
    echo "  • Data directory:      $(pwd)/data"
    echo "  • Logs directory:      $(pwd)/logs"
    echo "  • Config directory:    $(pwd)/config"
    echo
    echo "⚠️  Security Notes:"
    echo "  • Change default passwords before production use"
    echo "  • Configure SSL certificates for HTTPS"
    echo "  • Update CORS_ORIGINS for your domain"
    echo "  • Review firewall settings"
    echo
    echo "🧬 Testing EEEV Workflow:"
    echo "  curl -X POST http://localhost:8000/api/v1/workflows/eeev \\"
    echo "       -H 'Content-Type: application/json' \\"
    echo "       -d '{\"organism\": \"Eastern equine encephalitis virus\"}'"
    echo
    echo "======================================================================"
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    docker-compose down 2>/dev/null || true
}

# Main deployment function
deploy() {
    local mode="${1:-production}"
    
    log "Starting NanoBrain Viral Protein Analysis deployment (mode: $mode)"
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    # Run deployment steps
    check_root
    check_requirements
    setup_environment
    setup_monitoring
    
    case $mode in
        "production")
            build_images
            start_services
            sleep 30  # Give services time to start
            health_check
            show_deployment_info
            ;;
        "development")
            log "Starting development environment..."
            docker-compose --profile development up -d
            log_success "Development environment started"
            ;;
        "testing")
            log "Running test suite..."
            docker-compose --profile testing up --build
            ;;
        *)
            log_error "Unknown deployment mode: $mode"
            exit 1
            ;;
    esac
}

# Script options
case "${1:-}" in
    "deploy")
        deploy "${2:-production}"
        ;;
    "start")
        log "Starting existing deployment..."
        cd "$PROJECT_ROOT"
        docker-compose up -d
        health_check
        log_success "Services started"
        ;;
    "stop")
        log "Stopping deployment..."
        cd "$PROJECT_ROOT"
        docker-compose down
        log_success "Services stopped"
        ;;
    "restart")
        log "Restarting deployment..."
        cd "$PROJECT_ROOT"
        docker-compose restart
        health_check
        log_success "Services restarted"
        ;;
    "logs")
        cd "$PROJECT_ROOT"
        docker-compose logs -f "${2:-nanobrain-app}"
        ;;
    "status")
        cd "$PROJECT_ROOT"
        docker-compose ps
        ;;
    "clean")
        log "Cleaning up deployment..."
        cd "$PROJECT_ROOT"
        docker-compose down -v --remove-orphans
        docker system prune -f
        log_success "Cleanup completed"
        ;;
    "test")
        deploy "testing"
        ;;
    "dev")
        deploy "development"
        ;;
    "health")
        health_check
        ;;
    *)
        echo "NanoBrain Viral Protein Analysis - Deployment Script"
        echo
        echo "Usage: $0 [command] [options]"
        echo
        echo "Commands:"
        echo "  deploy [production|development|testing]  Deploy the system"
        echo "  start                                     Start existing deployment"
        echo "  stop                                      Stop the deployment"
        echo "  restart                                   Restart services"
        echo "  logs [service]                            Show service logs"
        echo "  status                                    Show service status"
        echo "  health                                    Run health checks"
        echo "  clean                                     Clean up everything"
        echo "  test                                      Run test suite"
        echo "  dev                                       Start development mode"
        echo
        echo "Examples:"
        echo "  $0 deploy production     # Full production deployment"
        echo "  $0 deploy development    # Development environment"
        echo "  $0 test                  # Run tests"
        echo "  $0 logs nanobrain-app    # View application logs"
        echo "  $0 health                # Check system health"
        echo
        exit 1
        ;;
esac 