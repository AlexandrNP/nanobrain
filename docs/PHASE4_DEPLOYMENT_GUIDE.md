# PHASE 4 DEPLOYMENT GUIDE: PRODUCTION READY NANOBRAIN VIRAL PROTEIN ANALYSIS

**Version**: 4.0.0  
**Date**: December 2024  
**Status**: ✅ **PRODUCTION READY**

## Overview

This guide provides comprehensive instructions for deploying the NanoBrain Viral Protein Analysis framework in a production environment with Docker containers, monitoring, and performance optimization.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start Deployment](#quick-start-deployment)
3. [Production Deployment](#production-deployment)
4. [Monitoring & Observability](#monitoring--observability)
5. [Performance Optimization](#performance-optimization)
6. [Security Configuration](#security-configuration)
7. [Troubleshooting](#troubleshooting)
8. [Maintenance](#maintenance)

---

## Prerequisites

### System Requirements

**Minimum Requirements:**
- **CPU**: 4 cores
- **RAM**: 8GB
- **Disk**: 50GB free space
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows with WSL2

**Recommended for Production:**
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Disk**: 100GB+ SSD
- **Network**: 1Gbps connection

### Software Dependencies

```bash
# Docker & Docker Compose
docker --version          # >= 20.10.0
docker-compose --version  # >= 1.29.0

# System utilities
curl --version
git --version
openssl version
```

### Installation on Ubuntu 20.04+

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

---

## Quick Start Deployment

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/your-org/nanobrain.git
cd nanobrain

# Make deployment script executable
chmod +x docker/deploy.sh

# Quick production deployment
./docker/deploy.sh deploy production
```

### 2. Verify Deployment

After deployment completes (5-10 minutes), verify services:

```bash
# Check service status
./docker/deploy.sh status

# Run health checks
./docker/deploy.sh health

# View application logs
./docker/deploy.sh logs nanobrain-app
```

### 3. Access Services

| Service              | URL                   | Description             |
| -------------------- | --------------------- | ----------------------- |
| Main Application     | http://localhost:8000 | EEEV Workflow API       |
| Production Dashboard | http://localhost:8002 | Real-time monitoring    |
| Prometheus           | http://localhost:9090 | Metrics collection      |
| Grafana              | http://localhost:3000 | Visualization dashboard |

---

## Production Deployment

### Environment Configuration

Create and customize your environment file:

```bash
# Copy example environment
cp .env.example .env

# Edit configuration
nano .env
```

**Key Configuration Options:**

```bash
# Database Security
POSTGRES_PASSWORD=your_secure_password_here
JWT_SECRET=your_jwt_secret_here
GRAFANA_PASSWORD=your_grafana_password_here

# Application Settings
NANOBRAIN_ENV=production
NANOBRAIN_LOG_LEVEL=INFO
MAX_MEMORY_GB=8
MAX_CPU_CORES=16

# External APIs
PUBMED_EMAIL=your-email@domain.com
BVBRC_ANONYMOUS_ACCESS=true

# Security
ALLOWED_HOSTS=your-domain.com,localhost
CORS_ORIGINS=https://your-domain.com,http://localhost
```

### SSL/TLS Configuration

For production HTTPS deployment:

```bash
# Create SSL directory
mkdir -p docker/nginx/ssl

# Generate self-signed certificate (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout docker/nginx/ssl/nginx-selfsigned.key \
    -out docker/nginx/ssl/nginx-selfsigned.crt

# Or copy your existing certificates
cp your-domain.crt docker/nginx/ssl/
cp your-domain.key docker/nginx/ssl/
```

### Production Deployment Steps

```bash
# 1. Full production deployment
./docker/deploy.sh deploy production

# 2. Verify all services are healthy
./docker/deploy.sh health

# 3. Run integration tests
./docker/deploy.sh test

# 4. Monitor deployment
docker stats
```

### Service Scaling

Scale services based on load:

```bash
# Scale main application
docker-compose up -d --scale nanobrain-app=3

# Scale with load balancer
docker-compose up -d --scale nanobrain-app=3 nginx

# Monitor resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
```

---

## Monitoring & Observability

### Prometheus Metrics

The system exposes comprehensive metrics at `/metrics`:

**System Metrics:**
- `nanobrain_cpu_usage_percent`
- `nanobrain_memory_usage_percent`
- `nanobrain_disk_usage_percent`

**Workflow Metrics:**
- `nanobrain_workflow_duration_seconds`
- `nanobrain_proteins_analyzed`
- `nanobrain_boundaries_detected`

**Performance Metrics:**
- `nanobrain_cache_hit_rate`
- `nanobrain_api_response_time_seconds`
- `nanobrain_error_count`

### Grafana Dashboards

Access Grafana at http://localhost:3000:

**Default Credentials:**
- Username: `admin`
- Password: Check `.env` file for `GRAFANA_PASSWORD`

**Available Dashboards:**
1. **NanoBrain Overview** - System health and performance
2. **EEEV Workflow Monitoring** - Workflow-specific metrics
3. **Infrastructure Monitoring** - Container and resource metrics

### Production Dashboard

Real-time monitoring dashboard at http://localhost:8002:

**Features:**
- Live performance metrics
- Active workflow tracking
- System health status
- Alert management
- Resource utilization

### Alerting Configuration

Configure alerts in `docker/monitoring/prometheus.yml`:

```yaml
# Example alert rules
- alert: HighMemoryUsage
  expr: nanobrain_memory_usage_percent > 85
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High memory usage detected"
    
- alert: WorkflowFailures
  expr: increase(nanobrain_workflow_failures_total[10m]) > 3
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Multiple workflow failures detected"
```

---

## Performance Optimization

### Memory Optimization

```yaml
# docker/config/production_config.yml
resources:
  memory:
    workflow_max_gb: 8
    cache_max_gb: 4
    api_max_gb: 2
  
  optimization:
    garbage_collection: true
    cache_compression: true
    batch_processing: true
```

### Cache Configuration

```yaml
cache:
  strategies:
    pubmed_references:
      ttl_hours: 168  # 1 week
      max_size_mb: 1000
      compression: true
      
    bvbrc_data:
      ttl_hours: 72   # 3 days
      max_size_mb: 2000
      compression: true
```

### Database Optimization

```yaml
database:
  postgres:
    pool_size: 50
    max_overflow: 20
    pool_timeout: 30
    
    # Performance tuning
    shared_buffers: "256MB"
    effective_cache_size: "1GB"
    work_mem: "4MB"
```

---

## Security Configuration

### Network Security

```yaml
security:
  network:
    trusted_proxies: ["10.0.0.0/8", "172.16.0.0/12"]
    rate_limiting: "100/hour"
    max_request_size_mb: 100
    
  cors:
    allowed_origins: ["https://your-domain.com"]
    allowed_methods: ["GET", "POST"]
    allowed_headers: ["Authorization", "Content-Type"]
```

### Authentication

```yaml
security:
  api_security:
    require_authentication: true
    jwt_secret: "${JWT_SECRET}"
    token_expiry_hours: 24
    
  data_encryption:
    enabled: true
    algorithm: "AES-256-GCM"
```

### Firewall Configuration

```bash
# Ubuntu UFW example
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow necessary ports
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw allow 8000  # Application
sudo ufw allow 3000  # Grafana (if external access needed)

sudo ufw enable
```

---

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

```bash
# Check service logs
./docker/deploy.sh logs nanobrain-app

# Check container status
docker ps -a

# Inspect specific container
docker inspect nanobrain-viral-protein-app
```

#### 2. Database Connection Issues

```bash
# Check database logs
./docker/deploy.sh logs postgres

# Test database connection
docker-compose exec postgres psql -U nanobrain -d nanobrain_viral -c "SELECT 1;"

# Reset database
docker-compose down -v
./docker/deploy.sh deploy production
```

#### 3. High Memory Usage

```bash
# Check memory usage
docker stats

# Restart services to clear memory
./docker/deploy.sh restart

# Check application logs for memory leaks
./docker/deploy.sh logs nanobrain-app | grep -i memory
```

#### 4. Slow Performance

```bash
# Check system resources
top
iotop
nethogs

# Analyze application metrics
curl http://localhost:8000/metrics

# Check cache hit rates
curl http://localhost:8000/metrics | grep cache_hit_rate
```

### Log Analysis

```bash
# Application logs
tail -f logs/app/nanobrain.log

# Nginx access logs
tail -f logs/nginx/access.log

# System logs
journalctl -u docker.service -f
```

### Performance Profiling

```bash
# Generate performance report
docker-compose exec nanobrain-app python -m cProfile -o profile.prof -m nanobrain.workflows.eeev

# Memory profiling
docker-compose exec nanobrain-app python -m memory_profiler workflow_script.py
```

---

## Maintenance

### Regular Maintenance Tasks

#### Daily
```bash
# Check system health
./docker/deploy.sh health

# Review logs for errors
./docker/deploy.sh logs | grep -i error

# Monitor disk space
df -h
```

#### Weekly
```bash
# Update container images
docker-compose pull
./docker/deploy.sh restart

# Clean up old logs
find logs/ -name "*.log" -mtime +7 -delete

# Backup database
docker-compose exec postgres pg_dump -U nanobrain nanobrain_viral > backup_$(date +%Y%m%d).sql
```

#### Monthly
```bash
# System updates
sudo apt update && sudo apt upgrade -y

# Clean up Docker
docker system prune -a

# Review and rotate SSL certificates
openssl x509 -in docker/nginx/ssl/nginx-selfsigned.crt -text -noout
```

### Backup and Recovery

#### Database Backup
```bash
# Create backup
docker-compose exec postgres pg_dump -U nanobrain nanobrain_viral > nanobrain_backup_$(date +%Y%m%d).sql

# Restore from backup
docker-compose exec -T postgres psql -U nanobrain nanobrain_viral < nanobrain_backup_20241201.sql
```

#### Full System Backup
```bash
# Backup data and configuration
tar -czf nanobrain_full_backup_$(date +%Y%m%d).tar.gz \
    data/ \
    logs/ \
    config/ \
    .env \
    docker-compose.yml
```

### Updating the System

```bash
# 1. Backup current state
./docker/deploy.sh stop
cp -r data/ data_backup_$(date +%Y%m%d)/

# 2. Pull latest code
git pull origin main

# 3. Rebuild and restart
./docker/deploy.sh deploy production

# 4. Verify update
./docker/deploy.sh health
```

---

## EEEV Workflow Testing

### Quick Test

```bash
# Test EEEV workflow endpoint
curl -X POST http://localhost:8000/api/v1/workflows/eeev \
     -H "Content-Type: application/json" \
     -d '{"organism": "Eastern equine encephalitis virus"}'
```

### Comprehensive Testing

```bash
# Run full test suite
./docker/deploy.sh test

# Run integration tests
docker-compose exec nanobrain-app python -m pytest tests/test_eeev_production_integration.py -v

# Performance testing
docker-compose exec nanobrain-app python -m pytest tests/test_performance.py -v
```

### Load Testing

```bash
# Install Apache Bench
sudo apt install apache2-utils

# Load test EEEV endpoint
ab -n 100 -c 10 -H "Content-Type: application/json" \
   -p test_payload.json \
   http://localhost:8000/api/v1/workflows/eeev
```

---

## Support and Resources

### Documentation
- [API Reference](../API_REFERENCE.md)
- [Architecture Overview](../LIBRARY_ARCHITECTURE.md)
- [Phase 4 Implementation Plan](../PHASE4_IMPLEMENTATION_PLAN.md)

### Monitoring URLs
- **Application Health**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **API Documentation**: http://localhost:8000/docs

### Log Locations
- **Application**: `logs/app/nanobrain.log`
- **Nginx**: `logs/nginx/access.log`
- **Database**: `docker-compose logs postgres`

### Contact Information
- **Technical Support**: Check GitHub issues
- **Documentation**: See `docs/` directory
- **Configuration**: See `config/` directory

---

## Deployment Checklist

### Pre-Deployment
- [ ] System requirements verified
- [ ] Docker and Docker Compose installed
- [ ] Environment configuration completed
- [ ] SSL certificates configured (for HTTPS)
- [ ] Firewall rules configured
- [ ] DNS records configured (for production domain)

### Deployment
- [ ] Repository cloned and configured
- [ ] Environment variables set
- [ ] Docker images built successfully
- [ ] All services started
- [ ] Health checks passing
- [ ] Integration tests passing

### Post-Deployment
- [ ] Monitoring dashboards accessible
- [ ] Alerting configured and tested
- [ ] Backup procedures tested
- [ ] Performance baseline established
- [ ] Documentation updated
- [ ] Team training completed

### Production Readiness
- [ ] Load testing completed
- [ ] Security review completed
- [ ] Disaster recovery plan documented
- [ ] Monitoring and alerting operational
- [ ] Support procedures documented
- [ ] Performance optimization completed

---

**Phase 4 Deployment Complete! 🎉**

The NanoBrain Viral Protein Analysis framework is now production-ready with comprehensive monitoring, performance optimization, and operational excellence. 