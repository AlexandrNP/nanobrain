# 🔥 Nanobrain Framework

**BRUTAL TRUTH: Event-driven AI agent framework for distributed workflows (RESEARCH PREVIEW)**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Development Status](https://img.shields.io/badge/status-alpha-red.svg)](https://pypi.org/project/nanobrain/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ⚠️ WARNING: RESEARCH FRAMEWORK

This is a **research framework in active development**. It is **NOT** production-ready and has significant limitations:

- **Hardcoded paths** and environment-specific configurations
- **External dependencies** on HPC systems and proprietary frameworks
- **Mock implementations** for many distributed features
- **Minimal documentation** and examples
- **Breaking changes** expected in future versions

## 🚀 Quick Install

```bash
# Minimal installation (workflow orchestration only)
pip install nanobrain

# With LLM support (requires API keys)
pip install nanobrain[llm]

# With distributed computing (requires HPC environment)
pip install nanobrain[distributed]

# Everything (WARNING: large download)
pip install nanobrain[all]
```

## 🔍 What Actually Works

### ✅ Core Framework
- **Configuration-driven architecture** with YAML configs
- **Event-driven data flow** with automatic triggers
- **Component composition** via `from_config` pattern
- **Basic workflow orchestration**

### ⚠️ Partially Working
- **LLM integration** (requires API keys and proper setup)
- **Web interfaces** (basic FastAPI endpoints)
- **Local execution** (single-machine workflows)

### ❌ Requires External Systems
- **Distributed execution** (needs Parsl + HPC)
- **Academy integration** (proprietary framework)
- **Aurora HPC demos** (specific supercomputer)
- **Production deployment** (complex setup)

## 📋 Basic Usage

```python
from nanobrain import ConversationalAgent

# Create agent from config (requires LLM setup)
agent = ConversationalAgent.from_config({
    'name': 'test_agent',
    'model': 'gpt-3.5-turbo',  # Requires OpenAI API key
    'temperature': 0.7
})

# Process request
response = await agent.aprocess("Hello!")
print(response.content)
```

## 🎯 Demo: AcademyLink Aurora

The main demo showcases mixed execution across local and HPC environments:

```bash
# Check if demo is available
nanobrain-demo academylink-aurora

# Run verification (if in development environment)
cd demos/academylink_aurora_demo
python verify_academylink_aurora.py
```

**BRUTAL TRUTH**: This demo only works in the original development environment with:
- Aurora supercomputer access
- Academy framework installation
- Specific conda environment setup
- Hardcoded node configurations

## 🛠️ CLI Commands

```bash
# Framework information
nanobrain info

# Validate configuration
nanobrain validate -c config.yml

# List demos
nanobrain demos

# Run demo (most won't work)
nanobrain-demo <demo-name>
```

## 📦 Dependencies

### Required (Always Installed)
- `pydantic>=2.0.0` - Data validation
- `pyyaml>=6.0` - Configuration files
- `fastapi>=0.100.0` - Web interfaces
- `structlog>=23.0.0` - Logging

### Optional (Install Separately)
- `openai>=1.0.0` - OpenAI models
- `anthropic>=0.3.0` - Claude models
- `parsl>=2024.6.10` - Distributed computing
- `proxystore>=0.6.0` - Academy integration

## 🔧 Development Setup

```bash
# Clone repository
git clone https://github.com/nanobrain/nanobrain.git
cd nanobrain

# Install in development mode
pip install -e .[dev]

# Run tests (many will fail without external systems)
pytest tests/

# Check code style
black nanobrain/
flake8 nanobrain/
```

## 🚨 Known Issues

1. **Hardcoded paths** throughout the codebase
2. **Missing __init__.py** files in some directories
3. **Circular imports** in some modules
4. **No proper error handling** for missing dependencies
5. **Configuration files not packaged** properly
6. **Mock implementations** don't match real interfaces
7. **No version management** for configurations

## 📚 Documentation

- **API Reference**: `nanobrain info` (basic information only)
- **Configuration Guide**: See `demos/` directory for examples
- **Development Guide**: Check source code comments
- **Issue Tracker**: GitHub Issues (expect many)

## 🤝 Contributing

**BRUTAL TRUTH**: This framework needs significant work before it's ready for contributions:

1. **Fix packaging issues** (hardcoded paths, missing files)
2. **Separate core from demos** (clean architecture)
3. **Add proper error handling** (graceful degradation)
4. **Create real documentation** (not just comments)
5. **Write comprehensive tests** (that actually pass)

## 📄 License

MIT License - See LICENSE file for details.

## 🔥 Final Brutal Truth

This package exists to demonstrate the framework's potential, but it's **not ready for production use**. The AcademyLink Aurora demo is impressive when it works, but it requires a very specific environment that most users won't have.

**Use at your own risk. Expect frustration. Prepare for debugging.**

If you need a production-ready AI agent framework, consider alternatives like:
- LangChain
- CrewAI  
- AutoGen
- Semantic Kernel

This framework may become production-ready in the future, but it's not there yet.
