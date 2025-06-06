#!/usr/bin/env python3
"""
Test suite for the Parsl Chat Workflow Demo

Tests the integration and functionality of the Parsl-based chat workflow demo,
ensuring it works correctly with the NanoBrain framework and can handle
various execution scenarios.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import demo components
from demo.chat_workflow_parsl_demo import (
    ParslChatWorkflow, 
    ChatRequest, 
    ChatResponse
)

# Import NanoBrain components for testing
from core.executor import ParslExecutor, LocalExecutor, ExecutorConfig, ExecutorType
from core.agent import ConversationalAgent, AgentConfig
from core.logging_system import get_logger


class TestChatRequest:
    """Test the ChatRequest dataclass."""
    
    def test_chat_request_creation(self):
        """Test creating a ChatRequest with required fields."""
        request = ChatRequest(
            id="test-123",
            message="Hello, world!",
            timestamp=datetime.now()
        )
        
        assert request.id == "test-123"
        assert request.message == "Hello, world!"
        assert request.user_id == "default_user"
        assert request.priority == 1
        assert request.context == {}
    
    def test_chat_request_with_context(self):
        """Test creating a ChatRequest with custom context."""
        context = {"session_id": "abc123", "user_type": "premium"}
        request = ChatRequest(
            id="test-456",
            message="Test message",
            timestamp=datetime.now(),
            user_id="user123",
            priority=2,
            context=context
        )
        
        assert request.user_id == "user123"
        assert request.priority == 2
        assert request.context == context


class TestChatResponse:
    """Test the ChatResponse dataclass."""
    
    def test_chat_response_creation(self):
        """Test creating a ChatResponse."""
        response = ChatResponse(
            request_id="test-123",
            response="Hello back!",
            agent_id="agent_1",
            processing_time=0.5,
            timestamp=datetime.now(),
            tokens_used=10
        )
        
        assert response.request_id == "test-123"
        assert response.response == "Hello back!"
        assert response.agent_id == "agent_1"
        assert response.processing_time == 0.5
        assert response.tokens_used == 10
        assert response.error is None
    
    def test_chat_response_with_error(self):
        """Test creating a ChatResponse with error."""
        response = ChatResponse(
            request_id="test-456",
            response="Error occurred",
            agent_id="agent_2",
            processing_time=0.1,
            timestamp=datetime.now(),
            error="Connection timeout"
        )
        
        assert response.error == "Connection timeout"


class TestParslChatWorkflow:
    """Test the main ParslChatWorkflow class."""
    
    @pytest.fixture
    def workflow(self):
        """Create a workflow instance for testing."""
        return ParslChatWorkflow()
    
    def test_workflow_initialization(self, workflow):
        """Test workflow initialization."""
        assert workflow.logger is not None
        assert workflow.parsl_executor is None
        assert workflow.agents == []
        assert workflow.num_agents == 3
        assert workflow.api_key is None
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key-123'})
    def test_setup_api_keys_with_env_var(self, workflow):
        """Test API key setup with environment variable."""
        result = workflow._setup_api_keys()
        assert result is True
        assert workflow.api_key == 'test-key-123'
        assert os.environ.get('OPENAI_API_KEY') == 'test-key-123'
    
    def test_setup_api_keys_without_key(self, workflow):
        """Test API key setup without any key available."""
        with patch.dict(os.environ, {}, clear=True):
            result = workflow._setup_api_keys()
            assert result is False
            assert workflow.api_key is None
    
    @pytest.mark.asyncio
    async def test_setup_with_local_fallback(self, workflow):
        """Test workflow setup with fallback to local executor."""
        with patch('demo.chat_workflow_parsl_demo.ParslExecutor') as mock_parsl:
            # Mock Parsl executor to fail initialization
            mock_parsl_instance = Mock()
            mock_parsl_instance.initialize = AsyncMock(side_effect=Exception("Parsl not available"))
            mock_parsl.return_value = mock_parsl_instance
            
            with patch('demo.chat_workflow_parsl_demo.LocalExecutor') as mock_local:
                mock_local_instance = Mock()
                mock_local_instance.initialize = AsyncMock()
                mock_local.return_value = mock_local_instance
                
                # Setup should succeed with local executor fallback
                await workflow.setup()
                
                assert workflow.parsl_executor == mock_local_instance
                assert len(workflow.agents) == 3
    
    @pytest.mark.asyncio
    async def test_process_message(self, workflow):
        """Test message processing with mock agents."""
        # Setup workflow with mock agents
        mock_agent = Mock()
        # ConversationalAgent.process() returns a string directly
        mock_agent.process = AsyncMock(return_value='Test response from agent')
        workflow.agents = [mock_agent, mock_agent, mock_agent]
        
        # Process a message
        result = await workflow.process_message("Hello, test!")
        
        assert result == 'Test response from agent'
        # After fix: agent.process() is called with the message string directly
        mock_agent.process.assert_called_once_with('Hello, test!')
    
    @pytest.mark.asyncio
    async def test_process_message_with_error(self, workflow):
        """Test message processing when agent raises an error."""
        # Setup workflow with mock agent that raises an error
        mock_agent = Mock()
        mock_agent.process = AsyncMock(side_effect=Exception("Agent error"))
        workflow.agents = [mock_agent]
        
        # Process a message
        result = await workflow.process_message("Hello, test!")
        
        assert "Sorry, I encountered an error: Agent error" in result
    
    @pytest.mark.asyncio
    async def test_shutdown(self, workflow):
        """Test workflow shutdown."""
        # Setup workflow with mock executor
        mock_executor = Mock()
        mock_executor.shutdown = AsyncMock()
        workflow.parsl_executor = mock_executor
        
        # Shutdown should complete without errors
        await workflow.shutdown()
        
        mock_executor.shutdown.assert_called_once()


class TestParslIntegration:
    """Test Parsl executor integration."""
    
    @pytest.mark.asyncio
    async def test_parsl_executor_creation(self):
        """Test creating a Parsl executor with configuration."""
        config = ExecutorConfig(
            executor_type=ExecutorType.PARSL,
            max_workers=4,
            parsl_config={
                'executors': [{
                    'class': 'parsl.executors.HighThroughputExecutor',
                    'label': 'test_htex',
                    'max_workers': 4
                }]
            }
        )
        
        executor = ParslExecutor(config=config)
        assert executor.config.executor_type == ExecutorType.PARSL
        assert executor.config.max_workers == 4
    
    @pytest.mark.asyncio
    async def test_parsl_executor_fallback(self):
        """Test that workflow falls back gracefully when Parsl is not available."""
        workflow = ParslChatWorkflow()
        
        with patch('core.executor.parsl', side_effect=ImportError("Parsl not installed")):
            # Setup should fall back to local executor
            await workflow.setup()
            
            # Should have fallen back to LocalExecutor
            assert isinstance(workflow.parsl_executor, LocalExecutor)
            assert len(workflow.agents) == 3


class TestAgentConfiguration:
    """Test agent configuration and creation."""
    
    @pytest.mark.asyncio
    async def test_agent_creation_with_mock_model(self):
        """Test creating agents with mock model configuration."""
        workflow = ParslChatWorkflow()
        
        # Mock the executor
        mock_executor = Mock()
        workflow.parsl_executor = mock_executor
        
        # Create agent config
        agent_config = AgentConfig(
            name="test_agent",
            description="Test agent for parallel processing",
            model="mock",
            temperature=0.7,
            max_tokens=500,
            system_prompt="You are a test agent."
        )
        
        # Create agent
        agent = ConversationalAgent(
            config=agent_config,
            executor=mock_executor
        )
        
        assert agent.config.name == "test_agent"
        assert agent.config.model == "mock"
        assert agent.config.temperature == 0.7


class TestPerformanceMetrics:
    """Test performance monitoring and metrics collection."""
    
    def test_chat_request_timing(self):
        """Test that chat requests include timing information."""
        start_time = datetime.now()
        request = ChatRequest(
            id="perf-test",
            message="Performance test message",
            timestamp=start_time
        )
        
        # Simulate processing time
        import time
        time.sleep(0.01)  # 10ms
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        response = ChatResponse(
            request_id=request.id,
            response="Performance test response",
            agent_id="perf_agent",
            processing_time=processing_time,
            timestamp=end_time
        )
        
        assert response.processing_time > 0
        assert response.processing_time < 1.0  # Should be less than 1 second


class TestErrorHandling:
    """Test error handling in various scenarios."""
    
    @pytest.mark.asyncio
    async def test_workflow_setup_error_handling(self):
        """Test that workflow setup handles errors gracefully."""
        workflow = ParslChatWorkflow()
        
        with patch.object(workflow, '_setup_api_keys', side_effect=Exception("API setup error")):
            # Setup should handle the error and continue
            try:
                await workflow.setup()
                # If we get here, the error was handled gracefully
                assert True
            except Exception as e:
                # If an exception propagates, it should be a different one
                assert "API setup error" not in str(e)
    
    @pytest.mark.asyncio
    async def test_message_processing_error_recovery(self):
        """Test that message processing recovers from individual agent errors."""
        workflow = ParslChatWorkflow()
        
        # Create mix of working and failing agents
        working_agent = Mock()
        working_agent.process = AsyncMock(return_value={'response': 'Success'})
        
        failing_agent = Mock()
        failing_agent.process = AsyncMock(side_effect=Exception("Agent failure"))
        
        workflow.agents = [failing_agent, working_agent, failing_agent]
        
        # Should still get a response from working agent
        result = await workflow.process_message("Test message")
        
        # Should either get success response or error message
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.integration
class TestFullWorkflowIntegration:
    """Integration tests for the complete workflow."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow from setup to message processing."""
        workflow = ParslChatWorkflow()
        
        try:
            # Setup workflow
            await workflow.setup()
            
            # Verify setup completed
            assert workflow.parsl_executor is not None
            assert len(workflow.agents) == workflow.num_agents
            
            # Process a test message
            result = await workflow.process_message("Hello, this is a test!")
            
            # Should get some response
            assert isinstance(result, str)
            assert len(result) > 0
            
        finally:
            # Always cleanup
            await workflow.shutdown()
    
    @pytest.mark.asyncio
    async def test_multiple_message_processing(self):
        """Test processing multiple messages in sequence."""
        workflow = ParslChatWorkflow()
        
        try:
            await workflow.setup()
            
            messages = [
                "What is parallel processing?",
                "How does Parsl work?",
                "Tell me about distributed computing."
            ]
            
            results = []
            for message in messages:
                result = await workflow.process_message(message)
                results.append(result)
            
            # Should get responses for all messages
            assert len(results) == len(messages)
            for result in results:
                assert isinstance(result, str)
                assert len(result) > 0
                
        finally:
            await workflow.shutdown()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 