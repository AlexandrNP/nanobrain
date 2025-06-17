# Multi-stage production Dockerfile for NanoBrain Viral Protein Analysis
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd -r nanobrain && useradd -r -g nanobrain nanobrain

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Dependencies stage
# ============================================================================
FROM base as dependencies

# Copy requirements
COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Install bioinformatics tools
RUN apt-get update && apt-get install -y \
    mmseqs2 \
    muscle \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Application stage
# ============================================================================
FROM dependencies as application

# Set working directory
WORKDIR /app

# Copy application code
COPY nanobrain/ /app/nanobrain/
COPY config/ /app/config/
COPY tests/ /app/tests/
COPY setup.py /app/

# Install NanoBrain package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/cache && \
    chown -R nanobrain:nanobrain /app

# Copy configuration templates
COPY docker/config/production_config.yml /app/config/
COPY docker/config/docker_env.yml /app/config/

# ============================================================================
# Production stage
# ============================================================================
FROM application as production

# Switch to non-root user
USER nanobrain

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import nanobrain; print('NanoBrain healthy')" || exit 1

# Expose port for web interface
EXPOSE 8000

# Set default command
CMD ["python", "-m", "nanobrain.library.workflows.viral_protein_analysis.web.standalone_interface"]

# ============================================================================
# Development stage
# ============================================================================
FROM application as development

# Install development dependencies
RUN pip install -r /tmp/requirements-dev.txt

# Install pre-commit hooks
RUN git init && pre-commit install || true

# Development command
CMD ["python", "-m", "pytest", "-v", "--cov=nanobrain", "tests/"]

# ============================================================================
# Testing stage
# ============================================================================
FROM dependencies as testing

# Copy only necessary files for testing
COPY nanobrain/ /app/nanobrain/
COPY tests/ /app/tests/
COPY setup.py /app/
COPY config/ /app/config/

# Install package and dev dependencies
RUN pip install -e . && \
    pip install -r /tmp/requirements-dev.txt

# Run tests
CMD ["python", "-m", "pytest", "-v", "--cov=nanobrain", "--cov-report=html", "tests/"] 