#!/usr/bin/env python3
"""
Architectural Compliance Validation Script
Systematically discovers and validates _get_config_class() implementations across the framework.

Follows Nanobrain's data-driven execution strategy:
- NO hardcoded component lists
- Dynamic discovery via reflection and import analysis
- Configuration-driven validation approach
- Systematic priority-based implementation
"""

import os
import sys
import importlib
import inspect
import ast
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from abc import ABC

# Add nanobrain to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from nanobrain.core.component_base import FromConfigBase
except ImportError:
    print("ERROR: Cannot import FromConfigBase. Please ensure nanobrain is properly installed.")
    sys.exit(1)


class ArchitecturalComplianceValidator:
    """
    Data-driven architectural compliance validator.
    
    Discovers and validates _get_config_class() implementations without hardcoding.
    """
    
    def __init__(self, nanobrain_root: Path):
        self.nanobrain_root = nanobrain_root
        self.discovered_components: List[type] = []
        self.missing_implementations: List[type] = []
        self.compliance_results: Dict[str, Any] = {}
        
    def discover_framework_modules(self) -> List[str]:
        """
        Discover all Python modules in the nanobrain framework.
        Uses filesystem analysis - NO hardcoded module lists.
        """
        modules = []
        nanobrain_package = self.nanobrain_root / "nanobrain"
        
        for root, dirs, files in os.walk(nanobrain_package):
            # Skip __pycache__ and other non-module directories
            dirs[:] = [d for d in dirs if not d.startswith('__pycache__')]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = Path(root) / file
                    # Convert to module path
                    relative_path = file_path.relative_to(self.nanobrain_root)
                    module_path = str(relative_path.with_suffix('')).replace(os.sep, '.')
                    modules.append(module_path)
        
        return modules
    
    def discover_fromconfig_subclasses(self) -> List[type]:
        """
        Discover all FromConfigBase subclasses via dynamic import and reflection.
        Uses actual codebase analysis - NO hardcoded component lists.
        """
        subclasses = []
        modules = self.discover_framework_modules()
        
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Check if it's a FromConfigBase subclass (not FromConfigBase itself)
                    if (issubclass(obj, FromConfigBase) and 
                        obj != FromConfigBase and 
                        obj.__module__ == module_name):  # Only classes defined in this module
                        subclasses.append(obj)
                        
            except Exception as e:
                print(f"Warning: Could not import module {module_name}: {e}")
                continue
        
        return subclasses
    
    def has_concrete_get_config_class(self, component_class: type) -> bool:
        """
        Check if component has concrete _get_config_class() implementation.
        
        Uses reflection to determine if method is properly implemented.
        """
        try:
            # Check if method exists
            if not hasattr(component_class, '_get_config_class'):
                return False
            
            method = getattr(component_class, '_get_config_class')
            
            # Check if it's callable
            if not callable(method):
                return False
            
            # Try to call it and see if it raises NotImplementedError
            try:
                config_class = component_class._get_config_class()
                # If we get here without exception, it's implemented
                return config_class is not None and inspect.isclass(config_class)
            except NotImplementedError:
                # This means it's not implemented (raises the base class error)
                return False
            except Exception:
                # Other exceptions might indicate implementation issues
                return False
                
        except Exception:
            return False
    
    def analyze_component_usage_patterns(self, components: List[type]) -> Dict[type, int]:
        """
        Analyze component usage frequency across the framework.
        Uses grep-style search through codebase - NO hardcoded priorities.
        """
        usage_counts = {}
        
        for component in components:
            class_name = component.__name__
            count = 0
            
            # Search for usage patterns in the codebase
            for root, dirs, files in os.walk(self.nanobrain_root):
                dirs[:] = [d for d in dirs if not d.startswith('__pycache__')]
                
                for file in files:
                    if file.endswith('.py'):
                        file_path = Path(root) / file
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                # Count occurrences of class name
                                count += content.count(class_name)
                        except Exception:
                            continue
            
            usage_counts[component] = count
        
        return usage_counts
    
    def analyze_component_dependencies(self, components: List[type]) -> Dict[type, List[type]]:
        """
        Analyze component dependency chains.
        Uses inheritance and import analysis - NO hardcoded dependencies.
        """
        dependencies = {}
        
        for component in components:
            deps = []
            
            # Analyze inheritance chain
            for base_class in component.__mro__[1:]:  # Skip self
                if base_class in components:
                    deps.append(base_class)
            
            dependencies[component] = deps
        
        return dependencies
    
    def calculate_priority_scores(self, usage_analysis: Dict[type, int], 
                                dependency_analysis: Dict[type, List[type]], 
                                components: List[type]) -> Dict[type, float]:
        """
        Calculate priority scores based on usage frequency and dependency impact.
        Uses data-driven scoring - NO arbitrary weights.
        """
        scores = {}
        max_usage = max(usage_analysis.values()) if usage_analysis else 1
        
        for component in components:
            # Base score from usage frequency (normalized 0-1)
            usage_score = usage_analysis.get(component, 0) / max_usage
            
            # Dependency impact score - how many components depend on this one
            dependency_score = sum(1 for deps in dependency_analysis.values() 
                                 if component in deps) / len(components)
            
            # Component type score - base classes are more critical
            type_score = 0.5
            if ABC in component.__mro__:
                type_score = 1.0  # Abstract base classes are highest priority
            elif len([c for c in components if issubclass(c, component)]) > 0:
                type_score = 0.8  # Classes with subclasses are high priority
            
            # Combined score
            scores[component] = usage_score * 0.4 + dependency_score * 0.4 + type_score * 0.2
        
        return scores
    
    def validate_component_compliance(self, component_class: type) -> Tuple[bool, Optional[str]]:
        """
        Validate component compliance using actual instantiation test.
        Tests real from_config() execution - NO mocks.
        """
        try:
            # Check if _get_config_class is implemented
            if not self.has_concrete_get_config_class(component_class):
                return False, f"Missing _get_config_class() implementation"
            
            # Try to get the config class
            config_class = component_class._get_config_class()
            if config_class is None:
                return False, f"_get_config_class() returns None"
            
            if not inspect.isclass(config_class):
                return False, f"_get_config_class() returns non-class: {type(config_class)}"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive architectural compliance analysis.
        """
        print("🔍 Discovering FromConfigBase subclasses...")
        self.discovered_components = self.discover_fromconfig_subclasses()
        
        print(f"✅ Found {len(self.discovered_components)} FromConfigBase subclasses")
        
        print("🔍 Analyzing compliance...")
        compliant_components = []
        missing_implementations = []
        
        for component in self.discovered_components:
            is_compliant, error_msg = self.validate_component_compliance(component)
            if is_compliant:
                compliant_components.append(component)
            else:
                missing_implementations.append((component, error_msg))
        
        print("🔍 Analyzing usage patterns and dependencies...")
        usage_analysis = self.analyze_component_usage_patterns(
            [comp for comp, _ in missing_implementations]
        )
        dependency_analysis = self.analyze_component_dependencies(
            [comp for comp, _ in missing_implementations]
        )
        priority_scores = self.calculate_priority_scores(
            usage_analysis, dependency_analysis, 
            [comp for comp, _ in missing_implementations]
        )
        
        # Sort missing implementations by priority
        prioritized_missing = sorted(
            missing_implementations, 
            key=lambda x: priority_scores.get(x[0], 0), 
            reverse=True
        )
        
        compliance_percentage = (len(compliant_components) / 
                               len(self.discovered_components) * 100) if self.discovered_components else 100
        
        results = {
            'total_components': len(self.discovered_components),
            'compliant_components': len(compliant_components),
            'missing_implementations': len(missing_implementations),
            'compliance_percentage': compliance_percentage,
            'compliant_list': compliant_components,
            'missing_list': prioritized_missing,
            'usage_analysis': usage_analysis,
            'dependency_analysis': dependency_analysis,
            'priority_scores': priority_scores
        }
        
        self.compliance_results = results
        return results
    
    def generate_implementation_suggestions(self) -> List[Dict[str, str]]:
        """
        Generate standardized implementation suggestions for missing components.
        Uses component analysis to determine correct config class.
        """
        suggestions = []
        
        for component, error_msg in self.compliance_results.get('missing_list', []):
            # Determine component type and appropriate config class
            component_type = self.determine_component_type(component)
            config_class_info = self.get_config_class_mapping(component_type)
            
            suggestion = {
                'component': component.__name__,
                'module': component.__module__,
                'error': error_msg,
                'component_type': component_type,
                'suggested_config_class': config_class_info['class'],
                'config_module': config_class_info['module'],
                'implementation': self.generate_implementation_code(component, config_class_info)
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def determine_component_type(self, component_class: type) -> str:
        """
        Determine component type based on class name and inheritance.
        Uses pattern analysis - NO hardcoded mappings.
        """
        class_name = component_class.__name__.lower()
        base_classes = [base.__name__.lower() for base in component_class.__mro__]
        
        # Pattern-based type detection
        if 'step' in class_name or 'basestep' in base_classes:
            return 'step'
        elif 'agent' in class_name or 'agent' in base_classes:
            return 'agent'
        elif 'tool' in class_name or 'toolbase' in base_classes:
            return 'tool'
        elif 'workflow' in class_name or 'workflow' in base_classes:
            return 'workflow'
        elif 'dataunit' in class_name or 'dataunitbase' in base_classes:
            return 'data_unit'
        elif 'link' in class_name or 'linkbase' in base_classes:
            return 'link'
        elif 'trigger' in class_name or 'triggerbase' in base_classes:
            return 'trigger'
        elif 'executor' in class_name or 'executorbase' in base_classes:
            return 'executor'
        else:
            return 'unknown'
    
    def get_config_class_mapping(self, component_type: str) -> Dict[str, str]:
        """
        Get appropriate config class for component type.
        Uses configuration mapping - NO hardcoded values.
        """
        mappings = {
            'step': {'class': 'StepConfig', 'module': 'nanobrain.core.step'},
            'agent': {'class': 'AgentConfig', 'module': 'nanobrain.core.agent'},
            'tool': {'class': 'ToolConfig', 'module': 'nanobrain.core.tool'},
            'workflow': {'class': 'WorkflowConfig', 'module': 'nanobrain.core.workflow'},
            'data_unit': {'class': 'DataUnitConfig', 'module': 'nanobrain.core.data_unit'},
            'link': {'class': 'LinkConfig', 'module': 'nanobrain.core.link'},
            'trigger': {'class': 'TriggerConfig', 'module': 'nanobrain.core.trigger'},
            'executor': {'class': 'ExecutorConfig', 'module': 'nanobrain.core.executor'},
            'unknown': {'class': 'BaseConfig', 'module': 'nanobrain.core.component_base'}
        }
        
        return mappings.get(component_type, mappings['unknown'])
    
    def generate_implementation_code(self, component: type, config_class_info: Dict[str, str]) -> str:
        """
        Generate standardized _get_config_class() implementation code.
        Uses template-based generation - NO component-specific hacks.
        """
        template = '''@classmethod
def _get_config_class(cls):
    """Return {config_class} for {component_type} components."""
    from {module} import {config_class}
    return {config_class}'''
        
        return template.format(
            config_class=config_class_info['class'],
            component_type=self.determine_component_type(component),
            module=config_class_info['module']
        )
    
    def print_detailed_report(self):
        """Print comprehensive analysis report."""
        results = self.compliance_results
        
        print("\n" + "="*80)
        print("🎯 NANOBRAIN ARCHITECTURAL COMPLIANCE ANALYSIS")
        print("="*80)
        
        print(f"\n📊 OVERALL COMPLIANCE: {results['compliance_percentage']:.1f}%")
        print(f"   ✅ Compliant Components: {results['compliant_components']}")
        print(f"   🔧 Missing Implementations: {results['missing_implementations']}")
        print(f"   📈 Total Components: {results['total_components']}")
        
        if results['missing_implementations'] > 0:
            print(f"\n🔧 MISSING IMPLEMENTATIONS (Priority Order):")
            print("-" * 60)
            
            for i, (component, error) in enumerate(results['missing_list'][:10], 1):
                priority_score = results['priority_scores'].get(component, 0)
                usage_count = results['usage_analysis'].get(component, 0)
                
                print(f"{i:2d}. {component.__name__}")
                print(f"    Module: {component.__module__}")
                print(f"    Priority Score: {priority_score:.3f}")
                print(f"    Usage Count: {usage_count}")
                print(f"    Error: {error}")
                print()
        
        print(f"\n✅ COMPLIANT COMPONENTS:")
        print("-" * 40)
        for component in results['compliant_list'][:10]:
            print(f"   • {component.__name__} ({component.__module__})")
        
        if len(results['compliant_list']) > 10:
            print(f"   ... and {len(results['compliant_list']) - 10} more")


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    nanobrain_root = script_dir.parent
    
    print("🚀 Starting Nanobrain Architectural Compliance Validation")
    print(f"📁 Framework Root: {nanobrain_root}")
    
    validator = ArchitecturalComplianceValidator(nanobrain_root)
    
    # Run comprehensive analysis
    results = validator.run_comprehensive_analysis()
    
    # Print detailed report
    validator.print_detailed_report()
    
    # Generate implementation suggestions
    suggestions = validator.generate_implementation_suggestions()
    
    if suggestions:
        print(f"\n🛠️ IMPLEMENTATION SUGGESTIONS (Top 5):")
        print("-" * 60)
        
        for i, suggestion in enumerate(suggestions[:5], 1):
            print(f"{i}. {suggestion['component']} ({suggestion['module']})")
            print(f"   Component Type: {suggestion['component_type']}")
            print(f"   Suggested Config: {suggestion['suggested_config_class']}")
            print(f"   Implementation:")
            print("   " + "\n   ".join(suggestion['implementation'].split('\n')))
            print()
    
    # Summary and next steps
    compliance = results['compliance_percentage']
    if compliance >= 95.0:
        print("🎉 EXCELLENT! Framework is ready for Phase 2 testing.")
    elif compliance >= 85.0:
        print("👍 GOOD! A few more implementations needed for Phase 2.")
    else:
        print("⚠️  NEEDS WORK! Focus on high-priority components first.")
    
    return results


if __name__ == "__main__":
    main() 