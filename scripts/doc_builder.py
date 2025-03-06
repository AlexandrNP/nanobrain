#!/usr/bin/env python3
"""
Documentation Builder for NanoBrain Framework

This script automatically generates documentation from:
1. Python docstrings in source files
2. YAML configuration files
3. Class and function signatures

The generated documentation follows the biological analogy theme of the framework.
"""

import os
import re
import ast
import yaml
import inspect
import importlib
import importlib.util
from typing import Dict, List, Any, Optional, Tuple, Set
import markdown
from pathlib import Path

# Configuration
SRC_DIR = "src"
CONFIG_DIR = "default_configs"
DOCS_DIR = "docs"
OUTPUT_DIR = os.path.join(DOCS_DIR, "auto_generated")


class DocBuilder:
    """Documentation builder for the NanoBrain framework."""
    
    def __init__(self):
        """Initialize the documentation builder."""
        self.src_files = []
        self.config_files = []
        self.class_docs = {}
        self.function_docs = {}
        self.module_docs = {}
        self.config_metadata = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def discover_files(self):
        """Discover all Python and YAML files in the project."""
        # Find all Python files in src directory
        for root, _, files in os.walk(SRC_DIR):
            for file in files:
                if file.endswith(".py"):
                    self.src_files.append(os.path.join(root, file))
        
        # Find all YAML files in config directory
        for root, _, files in os.walk(CONFIG_DIR):
            for file in files:
                if file.endswith(".yml") or file.endswith(".yaml"):
                    self.config_files.append(os.path.join(root, file))
    
    def parse_python_file(self, file_path: str):
        """Parse a Python file to extract docstrings and signatures."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the Python file
        try:
            tree = ast.parse(content)
            
            # Get the module docstring
            module_doc = ast.get_docstring(tree)
            if module_doc:
                module_name = os.path.basename(file_path).replace('.py', '')
                self.module_docs[module_name] = module_doc
            
            # Extract class and function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    class_doc = ast.get_docstring(node)
                    methods = {}
                    
                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_name = item.name
                            method_doc = ast.get_docstring(item)
                            if method_doc:
                                methods[method_name] = {
                                    'docstring': method_doc,
                                    'signature': self._get_function_signature(item)
                                }
                    
                    if class_doc:
                        self.class_docs[class_name] = {
                            'docstring': class_doc,
                            'methods': methods,
                            'file': file_path
                        }
                
                # Only include top-level functions (not methods inside classes)
                elif isinstance(node, ast.FunctionDef) and isinstance(node.parent, ast.Module):
                    function_name = node.name
                    function_doc = ast.get_docstring(node)
                    if function_doc:
                        self.function_docs[function_name] = {
                            'docstring': function_doc,
                            'signature': self._get_function_signature(node),
                            'file': file_path
                        }
        
        except SyntaxError as e:
            print(f"Error parsing {file_path}: {e}")
        except AttributeError as e:
            # In Python 3.8+, ast nodes don't have a parent attribute by default
            # We'll use a simpler approach to find top-level functions
            try:
                # Get all top-level function definitions
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef):
                        function_name = node.name
                        function_doc = ast.get_docstring(node)
                        if function_doc:
                            self.function_docs[function_name] = {
                                'docstring': function_doc,
                                'signature': self._get_function_signature(node),
                                'file': file_path
                            }
            except Exception as e2:
                print(f"Error parsing functions in {file_path}: {e2}")
    
    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature from AST node."""
        args = []
        
        # Add positional arguments
        for arg in node.args.args:
            arg_name = arg.arg
            arg_type = ""
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_type = f": {arg.annotation.id}"
                elif isinstance(arg.annotation, ast.Subscript):
                    arg_type = f": {ast.unparse(arg.annotation)}"
            args.append(f"{arg_name}{arg_type}")
        
        # Add *args if present
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        # Add keyword-only arguments
        for arg in node.args.kwonlyargs:
            arg_name = arg.arg
            arg_type = ""
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_type = f": {arg.annotation.id}"
                elif isinstance(arg.annotation, ast.Subscript):
                    arg_type = f": {ast.unparse(arg.annotation)}"
            args.append(f"{arg_name}{arg_type}")
        
        # Add **kwargs if present
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        # Get return type
        return_type = ""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return_type = f" -> {node.returns.id}"
            elif isinstance(node.returns, ast.Subscript):
                return_type = f" -> {ast.unparse(node.returns)}"
        
        return f"def {node.name}({', '.join(args)}){return_type}"
    
    def parse_yaml_file(self, file_path: str):
        """Parse a YAML configuration file to extract metadata."""
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        
        class_name = os.path.basename(file_path).replace('.yml', '').replace('.yaml', '')
        
        if 'metadata' in config:
            self.config_metadata[class_name] = config['metadata']
        
        # Also store defaults and examples for reference
        if 'defaults' in config:
            if class_name not in self.config_metadata:
                self.config_metadata[class_name] = {}
            self.config_metadata[class_name]['defaults'] = config['defaults']
        
        if 'examples' in config:
            if class_name not in self.config_metadata:
                self.config_metadata[class_name] = {}
            self.config_metadata[class_name]['examples'] = config['examples']
        
        if 'validation' in config:
            if class_name not in self.config_metadata:
                self.config_metadata[class_name] = {}
            self.config_metadata[class_name]['validation'] = config['validation']
    
    def generate_class_documentation(self):
        """Generate Markdown documentation for classes."""
        for class_name, class_info in self.class_docs.items():
            # Skip if no corresponding config metadata
            if class_name not in self.config_metadata and not class_info.get('docstring'):
                continue
            
            output_file = os.path.join(OUTPUT_DIR, f"{class_name}.md")
            
            with open(output_file, 'w') as f:
                # Class header
                f.write(f"# {class_name}\n\n")
                
                # Class docstring
                if class_info.get('docstring'):
                    f.write(f"{class_info['docstring']}\n\n")
                
                # Metadata from config
                if class_name in self.config_metadata:
                    metadata = self.config_metadata[class_name]
                    
                    if 'description' in metadata:
                        f.write(f"## Description\n\n{metadata['description']}\n\n")
                    
                    if 'biological_analogy' in metadata:
                        f.write(f"## Biological Analogy\n\n{metadata['biological_analogy']}\n\n")
                    
                    if 'justification' in metadata:
                        f.write(f"## Justification\n\n{metadata['justification']}\n\n")
                    
                    if 'objectives' in metadata:
                        f.write("## Objectives\n\n")
                        for objective in metadata['objectives']:
                            f.write(f"- {objective}\n")
                        f.write("\n")
                
                # Configuration
                if class_name in self.config_metadata and 'defaults' in self.config_metadata[class_name]:
                    f.write("## Default Configuration\n\n")
                    f.write("```yaml\n")
                    yaml.dump(self.config_metadata[class_name]['defaults'], f, default_flow_style=False)
                    f.write("```\n\n")
                
                # Validation
                if class_name in self.config_metadata and 'validation' in self.config_metadata[class_name]:
                    validation = self.config_metadata[class_name]['validation']
                    f.write("## Configuration Validation\n\n")
                    
                    if 'required' in validation:
                        f.write("### Required Parameters\n\n")
                        for param in validation['required']:
                            f.write(f"- `{param}`\n")
                        f.write("\n")
                    
                    if 'optional' in validation:
                        f.write("### Optional Parameters\n\n")
                        for param in validation['optional']:
                            f.write(f"- `{param}`\n")
                        f.write("\n")
                    
                    if 'constraints' in validation:
                        f.write("### Parameter Constraints\n\n")
                        for param, constraints in validation['constraints'].items():
                            f.write(f"#### `{param}`\n\n")
                            for key, value in constraints.items():
                                f.write(f"- {key}: `{value}`\n")
                            f.write("\n")
                
                # Examples
                if class_name in self.config_metadata and 'examples' in self.config_metadata[class_name]:
                    f.write("## Usage Examples\n\n")
                    examples = self.config_metadata[class_name]['examples']
                    
                    # Handle both dict and list examples
                    if isinstance(examples, dict):
                        for example_name, example in examples.items():
                            f.write(f"### {example_name.capitalize()}\n\n")
                            if 'description' in example:
                                f.write(f"{example['description']}\n\n")
                            if 'config' in example:
                                f.write("```yaml\n")
                                yaml.dump(example['config'], f, default_flow_style=False)
                                f.write("```\n\n")
                    elif isinstance(examples, list):
                        for i, example in enumerate(examples):
                            f.write(f"### Example {i+1}\n\n")
                            if isinstance(example, dict):
                                if 'description' in example:
                                    f.write(f"{example['description']}\n\n")
                                if 'config' in example:
                                    f.write("```yaml\n")
                                    yaml.dump(example['config'], f, default_flow_style=False)
                                    f.write("```\n\n")
                            else:
                                f.write(f"{example}\n\n")
                
                # Methods
                if class_info.get('methods'):
                    f.write("## Methods\n\n")
                    for method_name, method_info in class_info['methods'].items():
                        # Skip private methods
                        if method_name.startswith('_') and method_name != '__init__':
                            continue
                        
                        if method_name == '__init__':
                            f.write("### Constructor\n\n")
                        else:
                            f.write(f"### {method_name}\n\n")
                        
                        if method_info.get('signature'):
                            f.write("```python\n")
                            f.write(method_info['signature'])
                            f.write("\n```\n\n")
                        
                        if method_info.get('docstring'):
                            f.write(f"{method_info['docstring']}\n\n")
    
    def generate_index(self):
        """Generate an index page for all documentation."""
        output_file = os.path.join(OUTPUT_DIR, "index.md")
        
        with open(output_file, 'w') as f:
            f.write("# NanoBrain Framework Documentation\n\n")
            f.write("This documentation is automatically generated from source code and configuration files.\n\n")
            
            # Classes
            f.write("## Classes\n\n")
            for class_name in sorted(self.class_docs.keys()):
                if class_name in self.config_metadata or self.class_docs[class_name].get('docstring'):
                    f.write(f"- [{class_name}]({class_name}.md)\n")
            f.write("\n")
            
            # Modules
            if self.module_docs:
                f.write("## Modules\n\n")
                for module_name in sorted(self.module_docs.keys()):
                    f.write(f"- {module_name}\n")
                f.write("\n")
    
    def generate_framework_overview(self):
        """Generate a high-level framework overview document."""
        output_file = os.path.join(DOCS_DIR, "framework_overview.md")
        
        with open(output_file, 'w') as f:
            f.write("# NanoBrain Framework Overview\n\n")
            f.write("## Introduction\n\n")
            f.write("NanoBrain is a biologically-inspired framework for building adaptive, resilient systems. "
                   "It draws inspiration from neural systems to create software components that can adapt, "
                   "learn, and recover from failures.\n\n")
            
            f.write("## Core Principles\n\n")
            f.write("1. **Biological Analogies**: Components mirror biological systems for intuitive understanding\n")
            f.write("2. **Adaptive Behavior**: Connection strengths adapt based on usage patterns\n")
            f.write("3. **Resilience**: Built-in circuit breakers and recovery mechanisms\n")
            f.write("4. **Modularity**: Components can be combined in various ways\n\n")
            
            f.write("## Framework Architecture\n\n")
            
            f.write("### Core Components\n\n")
            f.write("- **Configuration Management**: ConfigManager, DirectoryTracer, ConfigLoader\n")
            f.write("- **Data Units**: DataUnitBase, DataUnitMemory, DataUnitFile\n")
            f.write("- **Links**: LinkBase, LinkDirect, LinkFile\n")
            f.write("- **Execution**: Runner, Router, ExecutorBase, ExecutorFunc, ExecutorParallel\n")
            f.write("- **Agents**: Agent, Step, Workflow\n\n")
            
            f.write("### Data Flow\n\n")
            f.write("```\n")
            f.write("[DataUnit] → [Link] → [Runner/Router] → [Executor] → [Output DataUnit]\n")
            f.write("```\n\n")
            
            f.write("### Control Flow\n\n")
            f.write("```\n")
            f.write("[Trigger] → [Runner] → [Router] → [Multiple Links] → [Multiple DataUnits]\n")
            f.write("```\n\n")
            
            f.write("## LLM Integration\n\n")
            f.write("The framework includes robust integration with various Language Model providers:\n\n")
            f.write("- **OpenAI**: GPT models (gpt-3.5-turbo, gpt-4, etc.)\n")
            f.write("- **Anthropic**: Claude models (claude-2, etc.)\n")
            f.write("- **Google**: Gemini models\n")
            f.write("- **Meta/Llama**: Llama models\n")
            f.write("- **Mistral**: Mistral models\n\n")
            
            f.write("The Agent class can work with both chat-based models (BaseChatModel) and completion-based models (BaseLLM).\n\n")
            
            f.write("## Tool Calling\n\n")
            f.write("The framework supports two approaches to tool calling:\n\n")
            f.write("1. **LangChain Tool Binding**: Wrapping Step classes as LangChain tools\n")
            f.write("2. **Custom Tool Prompts**: Using custom prompts for tool calling\n\n")
            
            f.write("For more details, see [Tool Calling Documentation](tool_calling.md).\n\n")
            
            f.write("## Getting Started\n\n")
            f.write("To get started with the NanoBrain framework:\n\n")
            f.write("1. Create instances of the components you need\n")
            f.write("2. Configure them using the provided configuration options\n")
            f.write("3. Connect them together to create your desired data and control flow\n")
            f.write("4. Run your system and observe its adaptive behavior\n\n")
            
            f.write("## Documentation\n\n")
            f.write("- [Auto-generated Class Documentation](auto_generated/index.md)\n")
            f.write("- [UML Diagrams](UML.md)\n")
            f.write("- [Tool Calling](tool_calling.md)\n\n")
    
    def run(self):
        """Run the documentation builder."""
        print("Discovering files...")
        self.discover_files()
        
        print(f"Found {len(self.src_files)} Python files and {len(self.config_files)} YAML files")
        
        print("Parsing Python files...")
        for file in self.src_files:
            self.parse_python_file(file)
        
        print("Parsing YAML files...")
        for file in self.config_files:
            self.parse_yaml_file(file)
        
        print("Generating class documentation...")
        self.generate_class_documentation()
        
        print("Generating index...")
        self.generate_index()
        
        print("Generating framework overview...")
        self.generate_framework_overview()
        
        print(f"Documentation generated in {OUTPUT_DIR}")


if __name__ == "__main__":
    builder = DocBuilder()
    builder.run() 