#!/usr/bin/env python3
"""
NanoBrain Framework Package Configuration
Framework-compliant setup configuration for development and production deployment.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

# Framework-required dependencies
requirements = [
    # Core framework dependencies
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "jsonschema>=4.0.0",
    
    # Web interface dependencies
    "fastapi>=0.100.0",
    "uvicorn>=0.20.0",
    
    # Async and messaging
    "asyncio-mqtt>=0.13.0",
    "aiofiles>=23.0.0",
    
    # Utilities
    "cachetools>=5.0.0",
    "pandas>=2.0.0",
    "psutil>=5.0.0",
    
    # AI/ML dependencies
    "openai>=1.0.0",
    "anthropic>=0.3.0",
    "langchain>=0.1.0",
    
    # Logging
    "structlog>=23.0.0",
]

# Development dependencies
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]

# Distributed execution dependencies
parsl_requirements = [
    "parsl>=2024.6.10",
    "dill>=0.3.0",
]

setup(
    name="nanobrain",
    version="0.1.0",
    description="A comprehensive AI agent framework with distributed execution capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NanoBrain Team",
    author_email="team@nanobrain.ai",
    url="https://github.com/nanobrain/nanobrain",
    
    # Package discovery
    packages=find_packages(),
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "parsl": parsl_requirements,
        "all": dev_requirements + parsl_requirements,
    },
    
    # Metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Package data
    include_package_data=True,
    zip_safe=False,
    
    # Framework compliance
    keywords="ai framework agents workflows distributed-computing",
)
