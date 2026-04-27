"""
Graceful degradation utilities for optional dependencies.

This module provides utilities to handle missing optional dependencies gracefully,
allowing the framework to function with reduced capabilities rather than failing.
"""

import importlib
import warnings
from typing import Any, Optional, Callable
from functools import wraps


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is not available."""
    pass


def optional_import(module_name: str, package_name: str = None) -> Any:
    """Import a module with graceful degradation.
    
    Args:
        module_name: Name of the module to import
        package_name: Name of the package (for better error messages)
        
    Returns:
        The imported module or None if not available
        
    Raises:
        OptionalDependencyError: If the module is required but not available
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        package_name = package_name or module_name
        warnings.warn(
            f"Optional dependency '{package_name}' not found. "
            f"Some features may not be available. "
            f"Install with: pip install {package_name}",
            UserWarning
        )
        return None


def requires_optional_dependency(dependency_name: str, install_command: str = None):
    """Decorator to mark functions that require optional dependencies.
    
    Args:
        dependency_name: Name of the required dependency
        install_command: Command to install the dependency
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (ImportError, ModuleNotFoundError) as e:
                install_cmd = install_command or f"pip install {dependency_name}"
                raise OptionalDependencyError(
                    f"Function '{func.__name__}' requires optional dependency '{dependency_name}'. "
                    f"Install with: {install_cmd}"
                ) from e
        return wrapper
    return decorator


# Optional dependency imports with graceful degradation
def get_llm_client():
    """Get LLM client with graceful degradation."""
    openai = optional_import('openai')
    anthropic = optional_import('anthropic')
    
    if not openai and not anthropic:
        raise OptionalDependencyError(
            "No LLM client available. Install with: pip install 'nanobrain[llm]'"
        )
    
    return {'openai': openai, 'anthropic': anthropic}


def get_vector_store():
    """Get vector store with graceful degradation."""
    faiss = optional_import('faiss', 'faiss-cpu')
    chromadb = optional_import('chromadb')
    
    if not faiss and not chromadb:
        raise OptionalDependencyError(
            "No vector store available. Install with: pip install 'nanobrain[rag]'"
        )
    
    return {'faiss': faiss, 'chromadb': chromadb}


def get_distributed_executor():
    """Get distributed executor with graceful degradation."""
    parsl = optional_import('parsl')
    
    if not parsl:
        raise OptionalDependencyError(
            "Distributed execution not available. Install with: pip install 'nanobrain[distributed]'"
        )
    
    return parsl


# Feature availability checks
def has_llm_support() -> bool:
    """Check if LLM support is available."""
    try:
        get_llm_client()
        return True
    except OptionalDependencyError:
        return False


def has_rag_support() -> bool:
    """Check if RAG support is available."""
    try:
        get_vector_store()
        return True
    except OptionalDependencyError:
        return False


def has_distributed_support() -> bool:
    """Check if distributed execution support is available."""
    try:
        get_distributed_executor()
        return True
    except OptionalDependencyError:
        return False


def check_feature_availability() -> dict:
    """Check availability of all optional features.
    
    Returns:
        Dictionary mapping feature names to availability status
    """
    return {
        'llm': has_llm_support(),
        'rag': has_rag_support(),
        'distributed': has_distributed_support(),
    }
