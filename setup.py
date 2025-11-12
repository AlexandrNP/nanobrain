#!/usr/bin/env python3
"""
Setup script for Nanobrain framework

🔥 BRUTAL TRUTH: Fallback setup.py for compatibility with older Python versions
This is needed because pyproject.toml doesn't work well with Python 3.6
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read version from __init__.py
def get_version():
    init_file = Path(__file__).parent / "nanobrain" / "__init__.py"
    with open(init_file) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()

# Core dependencies (compatible with Python 3.6+)
install_requires = [
    "pydantic>=1.8.0,<2.0.0",  # Compatible with Python 3.6, avoid v2
    "pyyaml>=5.0",
    "jsonschema>=3.0.0",
    "aiofiles>=0.8.0",
    "structlog>=20.0.0",
    "cachetools>=4.0.0",
    "psutil>=5.0.0",
    "pandas>=1.1.0",
    "python-dateutil>=2.8.0",
    "fastapi>=0.70.0",
    "uvicorn>=0.15.0",
    "httpx>=0.20.0",
    "requests>=2.25.0",
    "click>=7.0.0",
    "rich>=10.0.0",
    "tqdm>=4.50.0",
]

# Optional dependencies
extras_require = {
    "llm": [
        "openai>=0.27.0",
        "anthropic>=0.3.0",
        "langchain>=0.0.200",
    ],
    "distributed": [
        "parsl>=1.2.0",
        "dill>=0.3.0",
    ],
    "academy": [
        "proxystore>=0.4.0",
        "paho-mqtt>=1.5.0",
    ],
    "rag": [
        "sentence-transformers>=2.0.0",
        "faiss-cpu>=1.7.0",
        "numpy>=1.19.0",
        "scikit-learn>=1.0.0",
        "transformers>=4.20.0",
        "torch>=1.10.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
    ],
}

# Add 'all' extra
extras_require["all"] = []
for deps in extras_require.values():
    if deps != extras_require["all"]:
        extras_require["all"].extend(deps)

setup(
    name="nanobrain",
    version=get_version(),
    description="🔥 BRUTAL TRUTH: Event-driven AI agent framework for distributed workflows (RESEARCH PREVIEW)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nanobrain Team",
    author_email="team@nanobrain.ai",
    url="https://github.com/nanobrain/nanobrain",
    project_urls={
        "Documentation": "https://docs.nanobrain.ai",
        "Repository": "https://github.com/nanobrain/nanobrain.git",
        "Issues": "https://github.com/nanobrain/nanobrain/issues",
    },
    packages=find_packages(include=["nanobrain*", "demos*"]),
    include_package_data=True,
    package_data={
        "nanobrain": ["**/*.yml", "**/*.yaml", "**/*.txt", "**/*.md", "**/*.json"],
        "demos": ["**/*.yml", "**/*.yaml", "**/*.py", "**/*.md", "**/*.sh", "**/*.txt"],
    },
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "nanobrain=nanobrain.cli:main",
            "nanobrain-demo=nanobrain.demos.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    keywords=["ai", "agents", "workflow", "distributed", "hpc", "research"],
    license="MIT",
    zip_safe=False,
)
