import pytest
import os
import sys
import warnings
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add src directory to Python path
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

# Set testing environment variable
os.environ['NANOBRAIN_TESTING'] = '1'

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", message=".*Importing chat models from langchain is deprecated.*")
warnings.filterwarnings("ignore", message=".*Importing LLMs from langchain is deprecated.*")

# Test configuration
def pytest_configure(config):
    """Set up test configuration."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    
@pytest.fixture
def test_config_path():
    """Fixture providing path to test configuration files."""
    return os.path.join(os.path.dirname(__file__), 'test_configs')
    
@pytest.fixture
def temp_workspace(tmp_path):
    """Fixture providing a temporary workspace for file operations."""
    return tmp_path 

@pytest.fixture
def config_dir():
    """Return path to test config directory."""
    return os.path.join(os.path.dirname(__file__), 'test_default_configs') 