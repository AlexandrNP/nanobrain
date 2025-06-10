#!/usr/bin/env python3
"""
NanoBrain Framework Setup

A comprehensive AI agent framework with distributed execution capabilities.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "NanoBrain: A comprehensive AI agent framework"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'asyncio-mqtt>=0.11.0',
        'pyyaml>=6.0',
        'openai>=1.0.0',
        'anthropic>=0.3.0',
        'parsl>=2024.6.10',
        'dill>=0.3.0',
        'psutil>=5.9.0',
        'aiofiles>=23.0.0',
        'typing-extensions>=4.0.0'
    ]

setup(
    name="nanobrain",
    version="0.1.0",
    author="NanoBrain Team",
    author_email="team@nanobrain.ai",
    description="A comprehensive AI agent framework with distributed execution capabilities",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/nanobrain/nanobrain",
    packages=find_packages(include=['nanobrain', 'nanobrain.*']),
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
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.21.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'parsl': [
            'parsl>=2024.6.10',
            'dill>=0.3.0',
        ],
        'all': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.21.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
            'parsl>=2024.6.10',
            'dill>=0.3.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'nanobrain=nanobrain.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'nanobrain': [
            'config/*.yml',
            'library/workflows/*/config/*.yml',
            'library/agents/*/config/*.yml',
            'docs/*.md',
            'docs/*/*.md',
        ],
    },
    zip_safe=False,
) 