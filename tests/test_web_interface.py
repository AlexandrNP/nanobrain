"""
Tests for NanoBrain Web Interface

Test cases for the web interface components including:
- Configuration loading and validation
- Request/response model validation
- API endpoint functionality
- Error handling
- Integration with ChatWorkflow
"""

import pytest
import asyncio
import json
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock

# Import web interface components
try:
    from nanobrain.library.interfaces.web import (
        WebInterface,
        WebInterfaceConfig,
        ChatRequest,
        ChatOptions,
        ChatResponse,
        ChatMetadata,
        ErrorResponse
    )
    from nanobrain.library.interfaces.web.config.web_interface_config import (
        ServerConfig,
        APIConfig,
        CORSConfig,
        ChatConfig,
        LoggingConfig,
        SecurityConfig
    )
    WEB_INTERFACE_AVAILABLE = True
except ImportError:
    WEB_INTERFACE_AVAILABLE = False
    pytest.skip("Web interface not available", allow_module_level=True)


class TestWebInterfaceConfig:
    """Test web interface configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = WebInterfaceConfig()
        
        assert config.name == "nanobrain_web_interface"
        assert config.version == "1.0.0"
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 8000
        assert config.api.prefix == "/api/v1"
        assert config.cors.allow_origins == ["*"]
        assert config.chat.default_temperature == 0.7
    
    def test_config_from_dict(self):
        """Test configuration loading from dictionary."""
        config_dict = {
            "web_interface": {
                "name": "test_interface",
                "version": "2.0.0",
                "server": {
                    "host": "localhost",
                    "port": 9000
                },
                "api": {
                    "prefix": "/api/v2",
                    "title": "Test API"
                },
                "chat": {
                    "default_temperature": 0.5,
                    "default_max_tokens": 1500
                }
            }
        }
        
        config = WebInterfaceConfig.from_dict(config_dict)
        
        assert config.name == "test_interface"
        assert config.version == "2.0.0"
        assert config.server.host == "localhost"
        assert config.server.port == 9000
        assert config.api.prefix == "/api/v2"
        assert config.api.title == "Test API"
        assert config.chat.default_temperature == 0.5
        assert config.chat.default_max_tokens == 1500
    
    def test_config_to_dict(self):
        """Test configuration conversion to dictionary."""
        config = WebInterfaceConfig()
        config_dict = config.to_dict()
        
        assert "web_interface" in config_dict
        assert config_dict["web_interface"]["name"] == "nanobrain_web_interface"
        assert config_dict["web_interface"]["server"]["port"] == 8000
        assert config_dict["web_interface"]["api"]["prefix"] == "/api/v1"


class TestRequestModels:
    """Test request model validation."""
    
    def test_chat_options_validation(self):
        """Test ChatOptions validation."""
        # Valid options
        options = ChatOptions(
            temperature=0.7,
            max_tokens=1000,
            use_rag=True,
            conversation_id="test-conv-123"
        )
        
        assert options.temperature == 0.7
        assert options.max_tokens == 1000
        assert options.use_rag is True
        assert options.conversation_id == "test-conv-123"
    
    def test_chat_options_defaults(self):
        """Test ChatOptions default values."""
        options = ChatOptions()
        
        assert options.temperature == 0.7
        assert options.max_tokens == 2000
        assert options.use_rag is False
        assert options.conversation_id is None
    
    def test_chat_options_validation_errors(self):
        """Test ChatOptions validation errors."""
        # Temperature too high
        with pytest.raises(ValueError):
            ChatOptions(temperature=3.0)
        
        # Temperature too low
        with pytest.raises(ValueError):
            ChatOptions(temperature=-1.0)
        
        # Invalid max_tokens
        with pytest.raises(ValueError):
            ChatOptions(max_tokens=0)
        
        # Invalid max_tokens (too high)
        with pytest.raises(ValueError):
            ChatOptions(max_tokens=10000)
    
    def test_chat_request_validation(self):
        """Test ChatRequest validation."""
        # Valid request
        request = ChatRequest(
            query="Hello, world!",
            user_id="test_user"
        )
        
        assert request.query == "Hello, world!"
        assert request.user_id == "test_user"
        assert request.request_id is not None  # Auto-generated
        assert isinstance(request.options, ChatOptions)
    
    def test_chat_request_validation_errors(self):
        """Test ChatRequest validation errors."""
        # Empty query
        with pytest.raises(ValueError):
            ChatRequest(query="")
        
        # Whitespace-only query
        with pytest.raises(ValueError):
            ChatRequest(query="   ")
        
        # Query too long
        with pytest.raises(ValueError):
            ChatRequest(query="x" * 10001)
    
    def test_chat_request_query_cleaning(self):
        """Test query cleaning in ChatRequest."""
        request = ChatRequest(query="  Hello, world!  ")
        assert request.query == "Hello, world!"


class TestResponseModels:
    """Test response model creation and validation."""
    
    def test_chat_metadata_creation(self):
        """Test ChatMetadata creation."""
        metadata = ChatMetadata(
            processing_time_ms=1500.5,
            token_count=25,
            model_used="gpt-3.5-turbo",
            rag_enabled=False,
            conversation_id="conv-123"
        )
        
        assert metadata.processing_time_ms == 1500.5
        assert metadata.token_count == 25
        assert metadata.model_used == "gpt-3.5-turbo"
        assert metadata.rag_enabled is False
        assert metadata.conversation_id == "conv-123"
        assert metadata.timestamp is not None
    
    def test_chat_response_creation(self):
        """Test ChatResponse creation."""
        metadata = ChatMetadata(processing_time_ms=1000.0)
        
        response = ChatResponse(
            response="Hello! How can I help you?",
            conversation_id="conv-123",
            request_id="req-456",
            metadata=metadata
        )
        
        assert response.response == "Hello! How can I help you?"
        assert response.conversation_id == "conv-123"
        assert response.request_id == "req-456"
        assert response.status == "success"
        assert response.metadata == metadata
        assert response.warnings == []
    
    def test_error_response_creation(self):
        """Test ErrorResponse creation."""
        error = ErrorResponse(
            error="validation_error",
            message="Invalid input",
            details={"field": "query"},
            request_id="req-123"
        )
        
        assert error.error == "validation_error"
        assert error.message == "Invalid input"
        assert error.details == {"field": "query"}
        assert error.request_id == "req-123"
        assert error.timestamp is not None


class TestWebInterface:
    """Test WebInterface class."""
    
    @pytest.fixture
    def mock_chat_workflow(self):
        """Create a mock chat workflow."""
        workflow = AsyncMock()
        workflow.is_initialized = True
        workflow.process_user_input.return_value = "Mock response"
        workflow.get_workflow_status.return_value = {"status": "healthy"}
        workflow.get_conversation_stats.return_value = {"total_messages": 5}
        return workflow
    
    def test_web_interface_creation(self):
        """Test WebInterface creation."""
        interface = WebInterface()
        
        assert interface.config is not None
        assert interface.is_initialized is False
        assert interface.is_running is False
        assert interface.app is None
        assert interface.chat_workflow is None
    
    def test_web_interface_with_config(self):
        """Test WebInterface creation with custom config."""
        config = WebInterfaceConfig()
        config.name = "test_interface"
        
        interface = WebInterface(config)
        
        assert interface.config.name == "test_interface"
    
    @pytest.mark.asyncio
    async def test_web_interface_initialization(self, mock_chat_workflow):
        """Test WebInterface initialization."""
        interface = WebInterface()
        
        # Mock the chat workflow creation
        interface.chat_workflow = mock_chat_workflow
        
        # This would normally initialize the workflow and FastAPI app
        # For testing, we'll just check the basic setup
        assert interface.config is not None
        assert not interface.is_initialized
    
    def test_web_interface_status(self):
        """Test WebInterface status reporting."""
        interface = WebInterface()
        status = interface.get_status()
        
        assert "name" in status
        assert "version" in status
        assert "is_initialized" in status
        assert "is_running" in status
        assert "uptime_seconds" in status
        assert status["is_initialized"] is False
        assert status["is_running"] is False
    
    def test_create_default(self):
        """Test WebInterface.create_default factory method."""
        interface = WebInterface.create_default()
        
        assert isinstance(interface, WebInterface)
        assert interface.config.name == "nanobrain_web_interface"
    
    def test_from_config_file_not_found(self):
        """Test WebInterface.from_config_file with missing file."""
        with pytest.raises(FileNotFoundError):
            WebInterface.from_config_file("non_existent_config.yml")


class TestIntegration:
    """Integration tests for web interface components."""
    
    @pytest.mark.asyncio
    async def test_request_response_flow(self):
        """Test complete request-response flow."""
        # Create request
        request = ChatRequest(
            query="Test message",
            options=ChatOptions(temperature=0.5),
            user_id="test_user"
        )
        
        # Validate request
        assert request.query == "Test message"
        assert request.options.temperature == 0.5
        assert request.user_id == "test_user"
        
        # Create mock response
        metadata = ChatMetadata(
            processing_time_ms=500.0,
            conversation_id=request.options.conversation_id or "new-conv"
        )
        
        response = ChatResponse(
            response="Test response",
            conversation_id="new-conv",
            request_id=request.request_id,
            metadata=metadata
        )
        
        # Validate response
        assert response.response == "Test response"
        assert response.request_id == request.request_id
        assert response.status == "success"
    
    def test_config_roundtrip(self):
        """Test configuration save/load roundtrip."""
        # Create config
        original_config = WebInterfaceConfig()
        original_config.name = "roundtrip_test"
        original_config.server.port = 9999
        original_config.chat.default_temperature = 0.9
        
        # Convert to dict and back
        config_dict = original_config.to_dict()
        loaded_config = WebInterfaceConfig.from_dict(config_dict)
        
        # Verify roundtrip
        assert loaded_config.name == "roundtrip_test"
        assert loaded_config.server.port == 9999
        assert loaded_config.chat.default_temperature == 0.9


# Helper functions for testing
def create_test_config() -> WebInterfaceConfig:
    """Create a test configuration."""
    config = WebInterfaceConfig()
    config.name = "test_web_interface"
    config.server.port = 8001  # Use different port for testing
    config.logging.log_level = "DEBUG"
    return config


def create_test_request() -> ChatRequest:
    """Create a test chat request."""
    return ChatRequest(
        query="This is a test message",
        options=ChatOptions(
            temperature=0.7,
            max_tokens=500,
            use_rag=False
        ),
        user_id="test_user_123"
    )


# Performance tests
class TestPerformance:
    """Performance tests for web interface."""
    
    def test_request_model_validation_performance(self):
        """Test performance of request model validation."""
        import time
        
        start_time = time.time()
        
        # Create many requests
        for i in range(1000):
            request = ChatRequest(
                query=f"Test message {i}",
                options=ChatOptions(temperature=0.7),
                user_id=f"user_{i}"
            )
            assert request.query.startswith("Test message")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should be fast (less than 1 second for 1000 validations)
        assert elapsed < 1.0, f"Validation took too long: {elapsed:.3f}s"
    
    def test_config_serialization_performance(self):
        """Test performance of configuration serialization."""
        import time
        
        config = WebInterfaceConfig()
        
        start_time = time.time()
        
        # Serialize and deserialize many times
        for i in range(100):
            config_dict = config.to_dict()
            loaded_config = WebInterfaceConfig.from_dict(config_dict)
            assert loaded_config.name == config.name
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should be fast (less than 0.5 seconds for 100 operations)
        assert elapsed < 0.5, f"Serialization took too long: {elapsed:.3f}s"


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"]) 