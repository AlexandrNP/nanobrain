#!/usr/bin/env python3
"""
Pytest configuration and fixtures for Chatbot Viral Integration Tests

This file provides pytest-asyncio compatible fixtures and configuration
for standard CI/CD pipeline integration while maintaining compatibility 
with the existing custom test runner.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import pytest
import pytest_asyncio
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.chatbot_viral_integration import (
    MockWorkflowComponents,
    MockBVBRCService,
    MockExternalTools,
    MockSessionManager
)

# Import actual workflow components
try:
    from nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow import ChatbotViralWorkflow
    from nanobrain.library.workflows.chatbot_viral_integration.steps.query_classification_step import QueryClassificationStep
    from nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow import InMemorySessionManager
    REAL_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Real components not available, using mocks only: {e}")
    REAL_COMPONENTS_AVAILABLE = False


# ===========================
# PYTEST CONFIGURATION
# ===========================

def pytest_configure(config):
    """Configure pytest for async tests"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "real_components: Tests requiring real components")
    config.addinivalue_line("markers", "mock_only: Tests using mocks only")
    config.addinivalue_line("markers", "chatbot_viral: Chatbot viral integration tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add chatbot_viral marker to all tests in this directory
        if "chatbot_viral_integration" in str(item.fspath):
            item.add_marker(pytest.mark.chatbot_viral)
        
        # Mark tests that use real components
        if hasattr(item, 'fixturenames') and 'real_workflow' in item.fixturenames:
            item.add_marker(pytest.mark.real_components)
        
        # Mark tests that use only mocks
        if hasattr(item, 'fixturenames') and 'mock_workflow_components' in item.fixturenames:
            item.add_marker(pytest.mark.mock_only)
        
        # Mark performance tests
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)


# ===========================
# ASYNC FIXTURES
# ===========================

@pytest_asyncio.fixture
async def mock_workflow_components() -> MockWorkflowComponents:
    """Async fixture for mock workflow components"""
    return MockWorkflowComponents()


@pytest_asyncio.fixture
async def mock_services() -> Dict[str, Any]:
    """Async fixture for all mock services"""
    return {
        "workflow_components": MockWorkflowComponents(),
        "bvbrc_service": MockBVBRCService(simulate_delays=True),
        "external_tools": MockExternalTools(simulate_processing=True),
        "session_manager": MockSessionManager()
    }


@pytest_asyncio.fixture
async def fast_mock_services() -> Dict[str, Any]:
    """Async fixture for fast mock services (no delays)"""
    return {
        "workflow_components": MockWorkflowComponents(),
        "bvbrc_service": MockBVBRCService(simulate_delays=False),
        "external_tools": MockExternalTools(simulate_processing=False),
        "session_manager": MockSessionManager()
    }


@pytest_asyncio.fixture
async def real_workflow() -> Optional[ChatbotViralWorkflow]:
    """Async fixture for real workflow if available"""
    if not REAL_COMPONENTS_AVAILABLE:
        pytest.skip("Real workflow components not available")
    
    try:
        from nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow import ChatbotViralWorkflow
        from nanobrain.core.config.component_factory import ComponentFactory
        
        # Create workflow using from_config pattern
        factory = ComponentFactory()
        config_path = Path(__file__).parent.parent.parent / "nanobrain" / "library" / "workflows" / "chatbot_viral_integration" / "ChatbotViralWorkflow.yml"
        
        if config_path.exists():
            workflow = factory.create_from_yaml_file(
                config_path,
                "nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow.ChatbotViralWorkflow"
            )
            await workflow.initialize()
            return workflow
        else:
            pytest.skip(f"Workflow config not found: {config_path}")
            
    except Exception as e:
        pytest.skip(f"Real workflow unavailable: {e}")


@pytest_asyncio.fixture
async def session_manager() -> Optional[Any]:
    """Async fixture for session manager"""
    if not REAL_COMPONENTS_AVAILABLE:
        return MockSessionManager()
    
    try:
        from nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow import InMemorySessionManager
        return InMemorySessionManager()
    except ImportError:
        return MockSessionManager()


# ===========================
# PERFORMANCE FIXTURES
# ===========================

@pytest_asyncio.fixture
async def performance_mock_services() -> Dict[str, Any]:
    """Async fixture optimized for performance testing"""
    return {
        "workflow_components": MockWorkflowComponents(),
        "bvbrc_service": MockBVBRCService(simulate_delays=False),
        "external_tools": MockExternalTools(simulate_processing=False),
        "session_manager": MockSessionManager()
    }


# ===========================
# SCOPE FIXTURES
# ===========================

@pytest_asyncio.fixture(scope="session")
async def session_scoped_mock_services() -> Dict[str, Any]:
    """Session-scoped mock services for efficiency"""
    return {
        "workflow_components": MockWorkflowComponents(),
        "bvbrc_service": MockBVBRCService(simulate_delays=False),
        "external_tools": MockExternalTools(simulate_processing=False),
        "session_manager": MockSessionManager()
    }


# ===========================
# UTILITY FIXTURES
# ===========================

# Using default pytest-asyncio event loop fixture

@pytest.fixture
def test_data_dir() -> Path:
    """Path to test data directory"""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def config_dir() -> Path:
    """Path to configuration directory"""
    return Path(__file__).parent.parent.parent / "nanobrain" / "library" / "workflows" / "chatbot_viral_integration" / "config"


# ===========================
# SKIP CONDITIONS
# ===========================

skip_if_no_real_components = pytest.mark.skipif(
    not REAL_COMPONENTS_AVAILABLE,
    reason="Real workflow components not available"
)


# ===========================
# PARAMETRIZED FIXTURES
# ===========================

@pytest.fixture(params=["mock", "real"])
def workflow_type(request):
    """Parametrized fixture to run tests with both mock and real workflows"""
    if request.param == "real" and not REAL_COMPONENTS_AVAILABLE:
        pytest.skip("Real components not available")
    return request.param


# ===========================
# CLEANUP FIXTURES
# ===========================

@pytest_asyncio.fixture
async def auto_cleanup():
    """Fixture that provides automatic cleanup after tests"""
    cleanup_tasks = []
    
    def register_cleanup(task):
        cleanup_tasks.append(task)
    
    yield register_cleanup
    
    # Cleanup after test
    for task in cleanup_tasks:
        try:
            if asyncio.iscoroutinefunction(task):
                await task()
            else:
                task()
        except Exception as e:
            print(f"Cleanup task failed: {e}")


# ===========================
# CONFIGURATION HELPERS
# ===========================

def pytest_report_header(config):
    """Add custom header to pytest reports"""
    return [
        "NanoBrain Framework - Chatbot Viral Integration Tests",
        f"Real components available: {REAL_COMPONENTS_AVAILABLE}",
        "Custom test runner also available at: tests/chatbot_viral_integration/test_runner.py"
    ]


def pytest_runtest_setup(item):
    """Setup before each test"""
    # Skip real component tests if not available
    if item.get_closest_marker("real_components") and not REAL_COMPONENTS_AVAILABLE:
        pytest.skip("Real components not available") 