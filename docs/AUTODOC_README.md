# NanoBrain Documentation System

A barebones Sphinx + Autodoc documentation system that generates comprehensive API documentation from existing docstrings and Pydantic2 schemas.

## 🚀 Quick Start

### Build Documentation
```bash
cd docs
make html
```

### View Documentation
```bash
open build/html/index.html
```

### Clean and Rebuild
```bash
make clean
make html
```

## 📁 Structure

```
docs/
├── requirements.txt          # Minimal dependencies (4 lines)
├── sphinx_config.yml        # Single configuration file
├── generate_docs.py         # Simple generator script (~100 lines)
├── Makefile                 # Standard Sphinx Makefile
├── source/                  # Generated Sphinx source
│   ├── conf.py             # Auto-generated from YAML
│   ├── index.rst           # Main documentation index
│   └── api/                # API documentation
└── build/html/             # Generated HTML documentation
```

## ⚙️ Configuration

All settings are controlled by `sphinx_config.yml`:
- Project metadata
- Sphinx extensions
- Theme configuration  
- Autodoc settings
- Intersphinx mappings

## 🔄 Workflow

1. **Update code docstrings** in your NanoBrain modules
2. **Run `make html`** to regenerate documentation
3. **View results** at `build/html/index.html`

## ✨ Features

- **Automatic API Documentation**: Extracts from existing docstrings
- **Pydantic Schema Documentation**: Documents ConfigBase classes automatically  
- **Cross-References**: Links between related components
- **Search Functionality**: Built-in documentation search
- **Professional Theme**: ReadTheDocs theme with navigation
- **Type Hints**: Displays parameter and return types
- **Source Code Links**: Links to source code from documentation

## 🛠️ Maintenance

### Add New Modules
The system automatically discovers all modules in:
- `nanobrain.core`
- `nanobrain.library`

### Update Configuration
Edit `sphinx_config.yml` and run `make html`

### Dependencies
Only requires 4 packages:
- sphinx>=7.0.0
- sphinx-rtd-theme>=2.0.0  
- sphinx-autodoc-typehints>=1.25.0
- pydantic>=2.0.0

## 📊 What Gets Documented

✅ **All Classes**: Including inheritance relationships  
✅ **All Methods**: With parameters, returns, and exceptions  
✅ **Pydantic Models**: Field descriptions and validation rules  
✅ **Type Hints**: Parameter and return types  
✅ **Docstrings**: All existing documentation in your code  
✅ **Module Structure**: Complete package organization

## 🎯 Zero Complexity

- **No custom Python classes** for documentation management
- **No complex configuration files** - just one YAML file
- **No framework integration** - uses standard Sphinx patterns
- **No auxiliary management files** - everything is essential

The system provides 90% of documentation value with 10% of complexity! 