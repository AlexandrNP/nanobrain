#!/usr/bin/env python3
"""
Comprehensive documentation generator for NanoBrain framework.
Automatically discovers and documents ALL Python modules in the package
and creates high-level architecture documentation in RST format.
"""

import yaml
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, List, Set
import importlib.util

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "sphinx_config.yml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def discover_all_modules(package_path: Path) -> List[str]:
    """Discover all Python modules in the nanobrain package."""
    modules = []
    
    def scan_directory(dir_path: Path, package_prefix: str = ""):
        """Recursively scan directory for Python modules."""
        if dir_path.name.startswith('.') or dir_path.name == '__pycache__':
            return
            
        init_file = dir_path / "__init__.py"
        
        # If directory has __init__.py, it's a package
        if init_file.exists():
            current_package = f"{package_prefix}.{dir_path.name}" if package_prefix else dir_path.name
            modules.append(current_package)
            
            # Scan for submodules
            for item in dir_path.iterdir():
                if item.is_file() and item.suffix == '.py' and item.name != '__init__.py':
                    module_name = item.stem
                    full_module = f"{current_package}.{module_name}"
                    modules.append(full_module)
                elif item.is_dir():
                    scan_directory(item, current_package)
    
    scan_directory(package_path)
    return sorted(modules)

def create_architecture_overview(source_dir: Path, config: Dict[str, Any]):
    """Create RST architecture overview instead of copying markdown files."""
    arch_dir = source_dir / "architecture"
    arch_dir.mkdir(exist_ok=True)
    
    print(f"📋 Creating architecture overview in RST format...")
    
    # Create architecture overview RST file
    arch_overview_content = """
NanoBrain Framework Architecture Overview
========================================

The NanoBrain Framework is an advanced AI agent framework with enterprise-grade capabilities,
built on configuration-driven architecture principles and event-driven processing patterns.

Framework Core Architecture
--------------------------

**Foundational Philosophy:**

The framework follows four core principles:

* **Configuration-Driven**: All behavior controlled via YAML configurations
* **Event-Driven Architecture**: Components communicate through data flows and triggers  
* **Mandatory from_config Pattern**: All components use unified creation patterns
* **Zero Hardcoding**: Complete system flexibility through configuration

**Core Component Hierarchy:**

The framework is built around a unified hierarchy of configurable components:

* **FromConfigBase**: Abstract foundation for all framework components
* **Agent**: AI processing components with A2A and MCP protocol support
* **BaseStep**: Event-driven data processing units  
* **DataUnitBase**: Type-safe data containers with validation
* **LinkBase**: Data flow connection management
* **TriggerBase**: Event-driven activation system
* **ExecutorBase**: Configurable execution backends
* **ToolBase**: Capability extension for AI agents

Workflow Orchestration
---------------------

**Neural Circuit Complex Design:**

Workflows are inspired by neural circuit complexes where specialized processing units
coordinate through defined connections and activation patterns.

**Execution Strategies:**

* **SEQUENTIAL**: Linear step execution
* **PARALLEL**: Concurrent processing
* **GRAPH_BASED**: Dependency-driven execution
* **EVENT_DRIVEN**: Reactive processing patterns

**Data Flow Architecture:**

Data flows through Links between Steps, triggered by configurable Triggers,
enabling sophisticated AI workflows without manual orchestration.

Web Architecture
---------------

**Universal Access Pattern:**

The web interface provides workflow-agnostic access to all NanoBrain workflows
through standardized HTTP/HTTPS protocols and real-time communication channels.

**Key Features:**

* **RESTful API Design**: OpenAPI/Swagger documentation
* **Real-Time Communication**: WebSocket support for streaming
* **Authentication & Security**: JWT-based access control
* **Request Processing**: Asynchronous handling with rate limiting
* **Response Formatting**: Flexible JSON, XML, and custom formats

LLM Code Generation Rules
-------------------------

**AI-Driven Development Principles:**

The framework enforces strict patterns for LLM-based code generation to ensure
production-ready output that integrates seamlessly with framework architecture.

**Mandatory Compliance Rules:**

* **Framework Pattern Compliance**: All generated code must follow from_config patterns
* **Configuration-Driven Behavior**: No hardcoding allowed in generated components
* **Security-First Generation**: Automatic security pattern enforcement
* **Enterprise Quality Standards**: Production-ready code generation
* **Performance Optimization**: Built-in performance patterns

Component Library System
-----------------------

**Philosophy:**

The component library provides production-ready, enterprise-grade components
organized by domain and functionality, following consistent architectural patterns.

**Component Categories:**

* **Agents**: AI processing components (Conversational, Collaborative, Specialized)
* **Tools**: External integrations (Bioinformatics, Search, Infrastructure)
* **Workflows**: Pre-built processing pipelines (Chat, Viral Analysis, Web)
* **Infrastructure**: Enterprise services (Docker, Load Balancing, Monitoring)
* **Interfaces**: Web and API access layers

Configuration Management
-----------------------

**Enterprise Configuration Architecture:**

Advanced configuration system with recursive loading, schema validation,
and protocol integration for complex enterprise deployments.

**Key Features:**

* **Recursive Reference Resolution**: Automatic component dependency management
* **Schema Validation**: Pydantic-based validation with custom constraints
* **Template System**: Configuration inheritance and templating
* **Protocol Integration**: A2A and MCP protocol configuration
* **Environment Management**: Multi-environment configuration support

Testing and Validation Architecture
----------------------------------

**LLM-Driven Testing Framework:**

Comprehensive testing architecture designed for enterprise AI agent frameworks
with specialized validation for configuration-driven and event-driven systems.

**Core Testing Principles:**

* **Multi-Phase Validation**: Component, Integration, and Live System testing phases
* **Framework Compliance Enforcement**: Systematic validation of NanoBrain patterns
* **Configuration-Driven Testing**: Test behavior controlled via YAML configurations
* **Quality Gates and Success Criteria**: Objective measurement of system readiness
* **Continuous Monitoring**: Production testing and feedback loops

**Testing Phases:**

* **Phase 1 - Component Testing**: Individual component validation and from_config pattern compliance
* **Phase 2 - Integration Testing**: Workflow assembly and component interaction validation
* **Phase 3 - Live System Testing**: Real-world query processing and end-to-end execution

For detailed API documentation, see the :doc:`api/index` section.
"""
    
    arch_overview_path = arch_dir / "overview.rst"
    arch_overview_path.write_text(arch_overview_content)
    print(f"  ✅ Created architecture overview")

def generate_conf_py(config: Dict[str, Any]) -> str:
    """Generate enhanced conf.py content from configuration without markdown support."""
    return f'''# Configuration file for the Sphinx documentation builder.
# Auto-generated from sphinx_config.yml for comprehensive NanoBrain documentation

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Project information
project = "{config['project']}"
copyright = "{config['copyright']}"
author = "{config['author']}"
version = "{config['version']}"
release = "{config['version']}"

# Extensions
extensions = {config['extensions']}

# Autodoc configuration
autodoc_default_options = {config['autodoc']['default_options']}
autodoc_typehints = "{config['autodoc']['typehints']}"
autodoc_member_order = "{config['autodoc']['member_order']}"
autodoc_mock_imports = {config['autodoc']['mock_imports']}

# Autosummary configuration
autosummary_generate = {config['autosummary']['generate']}
autosummary_recursive = {config['autosummary']['recursive']}
autosummary_imported_members = {config['autosummary']['imported_members']}

# HTML theme
html_theme = "{config['html_theme']}"
html_theme_options = {config['html_theme_options']}

# Paths
exclude_patterns = {config['exclude_patterns']}

# Intersphinx
intersphinx_mapping = {config['intersphinx_mapping']}

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Additional comprehensive documentation settings
add_module_names = {config['add_module_names']}
show_authors = {config['show_authors']}
todo_include_todos = {config['todo_include_todos']}
coverage_show_missing_items = {config['coverage_show_missing_items']}
'''

def generate_module_rst(module_name: str, title: str) -> str:
    """Generate RST content for a module."""
    return f'''{title}
{'=' * len(title)}

.. automodule:: {module_name}
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
'''

def organize_modules_by_category(modules: List[str]) -> Dict[str, List[str]]:
    """Organize modules into logical categories."""
    categories = {
        'core': [],
        'config': [],
        'agents': [],
        'tools': [],
        'workflows': [],
        'infrastructure': [],
        'interfaces': [],
        'guards': [],
        'testing': [],
        'other': []
    }
    
    for module in modules:
        if '.config' in module:
            categories['config'].append(module)
        elif '.agents' in module:
            categories['agents'].append(module)
        elif '.tools' in module:
            categories['tools'].append(module)
        elif '.workflows' in module:
            categories['workflows'].append(module)
        elif '.infrastructure' in module:
            categories['infrastructure'].append(module)
        elif '.interfaces' in module:
            categories['interfaces'].append(module)
        elif '.guards' in module:
            categories['guards'].append(module)
        elif '.testing' in module:
            categories['testing'].append(module)
        elif '.core' in module:
            categories['core'].append(module)
        else:
            categories['other'].append(module)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

def generate_comprehensive_api_docs():
    """Generate comprehensive API documentation for all discovered modules."""
    print("🔍 Discovering all Python modules in nanobrain package...")
    
    # Discover all modules
    nanobrain_path = Path(__file__).parent.parent / "nanobrain"
    all_modules = discover_all_modules(nanobrain_path)
    
    print(f"📊 Found {len(all_modules)} modules to document")
    
    # Create API directory
    api_dir = Path(__file__).parent / "source" / "api"
    api_dir.mkdir(exist_ok=True)
    
    # Organize modules by category
    categories = organize_modules_by_category(all_modules)
    
    # Generate main API index
    with open(api_dir / "index.rst", 'w') as f:
        f.write('''API Reference
=============

Complete API documentation for all NanoBrain framework components.

.. toctree::
   :maxdepth: 3
   :caption: API Documentation:

''')
        
        # Add main package
        f.write('   nanobrain\n')
        
        # Add category sections
        for category, modules in categories.items():
            if modules:
                f.write(f'   {category}_modules\n')
    
    # Generate main nanobrain module documentation
    with open(api_dir / "nanobrain.rst", 'w') as f:
        f.write(generate_module_rst("nanobrain", "NanoBrain Framework Package"))
    
    # Generate category documentation files
    for category, modules in categories.items():
        if not modules:
            continue
            
        category_title = category.replace('_', ' ').title() + " Modules"
        
        with open(api_dir / f"{category}_modules.rst", 'w') as f:
            f.write(f'''{category_title}
{'=' * len(category_title)}

.. toctree::
   :maxdepth: 2

''')
            
            # Add each module in the category
            for module in modules:
                safe_filename = module.replace('.', '_')
                f.write(f'   {safe_filename}\n')
        
        # Generate individual module documentation files
        for module in modules:
            safe_filename = module.replace('.', '_')
            module_title = module.split('.')[-1].replace('_', ' ').title()
            
            with open(api_dir / f"{safe_filename}.rst", 'w') as f:
                f.write(generate_module_rst(module, f"{module_title} ({module})"))
    
    print(f"✅ Generated comprehensive API documentation for {len(all_modules)} modules")
    return all_modules

def generate_architecture_index(config: Dict[str, Any]):
    """Generate index for architecture documentation."""
    arch_dir = Path(__file__).parent / "source" / "architecture"
    arch_dir.mkdir(exist_ok=True)
    
    with open(arch_dir / "index.rst", 'w') as f:
        f.write('''Architecture Documentation
===========================

Comprehensive architectural documentation covering the design principles, 
patterns, and philosophies behind the NanoBrain framework.

.. toctree::
   :maxdepth: 2
   :caption: Architecture Documentation:

   overview
''')
    
    print(f"✅ Generated architecture documentation index")

def main():
    """Generate comprehensive documentation configuration and structure."""
    print("🔄 Generating comprehensive Sphinx configuration with architecture documentation...")
    
    # Load configuration
    config = load_config()
    
    # Generate conf.py
    conf_content = generate_conf_py(config)
    source_dir = Path(__file__).parent / "source"
    source_dir.mkdir(exist_ok=True)
    
    with open(source_dir / "conf.py", 'w') as f:
        f.write(conf_content)
    
    # Create architecture documentation
    create_architecture_overview(source_dir, config)
    
    # Generate architecture documentation index
    generate_architecture_index(config)
    
    # Generate comprehensive API documentation structure
    documented_modules = generate_comprehensive_api_docs()
    
    # Generate enhanced main index with architecture documentation
    index_file = source_dir / "index.rst"
    with open(index_file, 'w') as f:
        f.write(f'''NanoBrain Framework Documentation
==================================

Welcome to the comprehensive documentation for the NanoBrain Framework - 
the most advanced AI agent framework with enterprise-grade capabilities.

**Framework Features:**

* 🤖 **Advanced AI Agents**: Multi-protocol support with A2A and MCP integration
* 🔧 **Configurable Architecture**: YAML-driven configuration system
* 🧬 **Bioinformatics Tools**: Specialized tools for scientific workflows
* 🏗️ **Enterprise Infrastructure**: Docker, monitoring, and load balancing
* 🌐 **Web Interfaces**: REST API and WebSocket support
* 📊 **Comprehensive Workflows**: Event-driven processing and orchestration

**Documentation Coverage:** {len(documented_modules)} modules fully documented

.. toctree::
   :maxdepth: 2
   :caption: Architecture Overview:

   architecture/index

.. toctree::
   :maxdepth: 3
   :caption: API Reference:

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
''')
    
    print("✅ Comprehensive documentation configuration generated successfully!")
    print(f"📊 Total modules documented: {len(documented_modules)}")
    print(f"📋 Architecture documentation created in RST format")
    print("📁 Run 'make html' in the docs directory to build comprehensive documentation")

if __name__ == "__main__":
    main() 