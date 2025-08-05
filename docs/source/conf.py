# Configuration file for the Sphinx documentation builder.
# Auto-generated from sphinx_config.yml for comprehensive NanoBrain documentation

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Project information
project = "NanoBrain Framework"
copyright = "2024, NanoBrain Team"
author = "NanoBrain Team"
version = "2.0.0"
release = "2.0.0"

# Extensions
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon', 'sphinx.ext.intersphinx', 'sphinx.ext.todo', 'sphinx.ext.coverage', 'sphinx.ext.inheritance_diagram']

# Autodoc configuration
autodoc_default_options = {'members': True, 'undoc-members': True, 'show-inheritance': True, 'special-members': '__init__', 'private-members': False, 'exclude-members': '__weakref__'}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_mock_imports = []

# Autosummary configuration
autosummary_generate = True
autosummary_recursive = True
autosummary_imported_members = True

# HTML theme
html_theme = "sphinx_rtd_theme"
html_theme_options = {'display_version': True, 'collapse_navigation': False, 'sticky_navigation': True, 'navigation_depth': 5, 'includehidden': True, 'titles_only': False}

# Paths
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/__pycache__', '**/.*', 'build']

# Intersphinx
intersphinx_mapping = {'python': ['https://docs.python.org/3', None], 'pydantic': ['https://docs.pydantic.dev/', None], 'sphinx': ['https://www.sphinx-doc.org/en/master/', None], 'numpy': ['https://numpy.org/doc/stable/', None]}

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Additional comprehensive documentation settings
add_module_names = True
show_authors = True
todo_include_todos = True
coverage_show_missing_items = True
