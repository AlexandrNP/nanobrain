# Nanobrain Framework

A research-preview event-driven AI agent framework for distributed workflows on HPC systems.

## Overview

Nanobrain is an experimental framework designed to orchestrate AI agents across distributed computing environments, with particular focus on High Performance Computing (HPC) systems. The framework provides event-driven workflow orchestration, agent coordination, and distributed execution capabilities.

## Current Status

⚠️ **Research Preview**: This framework is in active development and cleanup phase. Many components are experimental and subject to change.

### Core Components

- **Core Framework** (`nanobrain/core/`): Event-driven agent orchestration
- **Library** (`nanobrain/library/`): Reusable workflow components and tools
- **Cleanup System** (`nanobrain/cleanup/`): Repository maintenance and organization tools

### Working Demos

- **viral_pssm_workflow**: Viral protein sequence analysis pipeline
- **rag_database_creation**: RAG (Retrieval-Augmented Generation) database setup
- **academylink_aurora_demo**: Academy integration demonstration
- **simple_demo**: Basic framework usage example

## Installation

See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for detailed installation instructions.

## Quick Start

```bash
# Install the framework
pip install -e .

# Run a simple demo
python demos/simple_demo/run_demo.py
```

## Documentation

- **Framework Architecture**: `docs/01_FRAMEWORK_CORE_ARCHITECTURE.md`
- **Workflow Orchestration**: `docs/02_WORKFLOW_ORCHESTRATION.md`
- **API Reference**: `docs/API_REFERENCE.md`
- **Demo Documentation**: See individual demo directories

## Development Status

This repository is currently undergoing comprehensive cleanup and reorganization. Many temporary files, redundant documentation, and experimental code are being consolidated or removed.

### Cleanup Progress

- ✅ Backup system implementation
- ✅ Temporary file removal
- ✅ Demo consolidation
- ✅ Import resolution
- ✅ Configuration standardization
- 🔄 Documentation consolidation (in progress)
- ⏳ Dependency management
- ⏳ Repository structure reorganization

## Contributing

This is a research project. Contributions are welcome but please note the experimental nature of the codebase.

## License

[Add license information]

## Contact

[Add contact information]
