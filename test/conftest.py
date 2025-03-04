import pytest
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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