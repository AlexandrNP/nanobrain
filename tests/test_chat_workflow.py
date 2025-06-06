"""
Tests for Chat Workflow Demo

Tests the complete NanoBrain chat workflow including:
- ConversationalAgentStep
- CLIInterface (mocked)
- ChatWorkflow orchestration
- Component integration
- Data flow validation
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# Import the components we need to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))

from core.data_unit import DataUnitMemory, DataUnitConfig
from core.trigger import DataUpdatedTrigger, TriggerConfig
from core.link import DirectLink, LinkConfig
from core.step import StepConfig
from core.agent import ConversationalAgent, AgentConfig
from core.executor import LocalExecutor, ExecutorConfig

# Import chat workflow components
from chat_workflow_demo import ConversationalAgentStep, CLIInterface, ChatWorkflow


class TestConversationalAgentStep:
    """Test the ConversationalAgentStep class."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock conversational agent."""
        agent = AsyncMock(spec=ConversationalAgent)
        agent.process = AsyncMock()
        return agent
    
    @pytest.fixture
    def step_config(self):
        """Create step configuration."""
        return StepConfig(
            name="test_chat_step",
            description="Test conversational agent step",
            debug_mode=True
        )
    
    @pytest_asyncio.fixture
    async def agent_step(self, step_config, mock_agent):
        """Create ConversationalAgentStep instance."""
        step = ConversationalAgentStep(step_config, mock_agent)
        await step.initialize()
        return step
    
    @pytest.mark.asyncio
    async def test_process_valid_input(self, agent_step, mock_agent):
        """Test processing valid user input."""
        # Setup mock response
        mock_agent.process.return_value = "Hello! How can I help you today?"
        
        # Test input
        inputs = {'user_input': 'Hello there!'}
        
        # Process
        result = await agent_step.process(inputs)
        
        # Verify
        assert 'agent_response' in result
        assert result['agent_response'] == "Hello! How can I help you today?"
        mock_agent.process.assert_called_once_with('Hello there!')
        assert agent_step.conversation_count == 1
    
    @pytest.mark.asyncio
    async def test_process_empty_input(self, agent_step, mock_agent):
        """Test processing empty user input."""
        # Test empty inputs
        test_cases = [
            {'user_input': ''},
            {'user_input': '   '},
            {},
            {'other_key': 'value'}
        ]
        
        for inputs in test_cases:
            result = await agent_step.process(inputs)
            assert result['agent_response'] == ''
            
        # Agent should not be called for empty inputs
        mock_agent.process.assert_not_called()
        assert agent_step.conversation_count == 0
    
    @pytest.mark.asyncio
    async def test_process_agent_error(self, agent_step, mock_agent):
        """Test handling agent processing errors."""
        # Setup mock to raise exception
        mock_agent.process.side_effect = Exception("Network timeout")
        
        # Test input
        inputs = {'user_input': 'Test message'}
        
        # Process
        result = await agent_step.process(inputs)
        
        # Verify error handling
        assert 'agent_response' in result
        assert 'Sorry, I encountered an error' in result['agent_response']
        assert 'Network timeout' in result['agent_response']
        assert agent_step.conversation_count == 1
    
    @pytest.mark.asyncio
    async def test_process_none_response(self, agent_step, mock_agent):
        """Test handling None response from agent."""
        # Setup mock to return None
        mock_agent.process.return_value = None
        
        # Test input
        inputs = {'user_input': 'Test message'}
        
        # Process
        result = await agent_step.process(inputs)
        
        # Verify fallback response
        assert result['agent_response'] == 'I apologize, but I could not generate a response.'
        assert agent_step.conversation_count == 1
    
    @pytest.mark.asyncio
    async def test_conversation_counting(self, agent_step, mock_agent):
        """Test conversation counting functionality."""
        mock_agent.process.return_value = "Response"
        
        # Process multiple inputs
        for i in range(5):
            await agent_step.process({'user_input': f'Message {i}'})
        
        # Verify conversation count
        assert agent_step.conversation_count == 5
        assert mock_agent.process.call_count == 5


class TestCLIInterface:
    """Test the CLIInterface class."""
    
    @pytest.fixture
    def data_units(self):
        """Create test data units."""
        input_config = DataUnitConfig(
            name="test_input",
            data_type="memory",
            persistent=False,
            cache_size=10
        )
        output_config = DataUnitConfig(
            name="test_output", 
            data_type="memory",
            persistent=False,
            cache_size=10
        )
        
        input_du = DataUnitMemory(input_config)
        output_du = DataUnitMemory(output_config)
        
        return input_du, output_du
    
    @pytest.fixture
    def cli_interface(self, data_units):
        """Create CLIInterface instance."""
        input_du, output_du = data_units
        return CLIInterface(input_du, output_du)
    
    @pytest.mark.asyncio
    async def test_cli_initialization(self, cli_interface, data_units):
        """Test CLI interface initialization."""
        input_du, output_du = data_units
        
        assert cli_interface.input_data_unit is input_du
        assert cli_interface.output_data_unit is output_du
        assert cli_interface.running is False
        assert cli_interface.input_thread is None
    
    @pytest.mark.asyncio
    async def test_output_handling(self, cli_interface):
        """Test output handling functionality."""
        # Test data with response
        test_data = {'agent_response': 'Hello from the agent!'}
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            await cli_interface._on_output_received(test_data)
            mock_print.assert_called_once_with("\n🤖 Assistant: Hello from the agent!")
    
    @pytest.mark.asyncio
    async def test_output_handling_empty(self, cli_interface):
        """Test output handling with empty response."""
        test_cases = [
            {'agent_response': ''},
            {'agent_response': '   '},
            {},
            {'other_key': 'value'}
        ]
        
        with patch('builtins.print') as mock_print:
            for test_data in test_cases:
                await cli_interface._on_output_received(test_data)
            
            # Should not print anything for empty responses
            mock_print.assert_not_called()
    
    def test_help_display(self, cli_interface):
        """Test help display functionality."""
        with patch('builtins.print') as mock_print:
            cli_interface._show_help()
            
            # Verify help content is displayed
            assert mock_print.call_count > 0
            help_text = ''.join(str(call) for call in mock_print.call_args_list)
            assert 'Available Commands' in help_text
            assert 'help' in help_text
            assert 'quit' in help_text


class TestChatWorkflow:
    """Test the ChatWorkflow orchestration."""
    
    @pytest.fixture
    def workflow(self):
        """Create ChatWorkflow instance."""
        workflow = ChatWorkflow()
        return workflow
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, workflow):
        """Test workflow initialization."""
        assert workflow.factory is not None
        assert workflow.components == {}
        assert workflow.cli is None
        assert workflow.executor is None
    
    @pytest.mark.asyncio
    async def test_workflow_setup_components(self, workflow):
        """Test workflow component setup."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            # Mock the agent process method to avoid actual API calls
            with patch.object(ConversationalAgent, 'process', new_callable=AsyncMock) as mock_process:
                mock_process.return_value = "Test response"
                
                await workflow.setup()
                
                # Verify components were created
                assert 'user_input_du' in workflow.components
                assert 'agent_input_du' in workflow.components
                assert 'agent_output_du' in workflow.components
                assert 'agent_step' in workflow.components
                assert 'user_trigger' in workflow.components
                assert 'agent_trigger' in workflow.components
                assert 'output_trigger' in workflow.components
                
                # Verify component types
                assert isinstance(workflow.components['user_input_du'], DataUnitMemory)
                assert isinstance(workflow.components['agent_input_du'], DataUnitMemory)
                assert isinstance(workflow.components['agent_output_du'], DataUnitMemory)
                assert isinstance(workflow.components['agent_step'], ConversationalAgentStep)
                assert isinstance(workflow.components['user_trigger'], DataUpdatedTrigger)
                assert isinstance(workflow.components['agent_trigger'], DataUpdatedTrigger)
                assert isinstance(workflow.components['output_trigger'], DataUpdatedTrigger)
                
                # Verify executor and CLI
                assert workflow.executor is not None
                assert isinstance(workflow.executor, LocalExecutor)
                assert workflow.cli is not None
                assert isinstance(workflow.cli, CLIInterface)
                
                # Cleanup
                await workflow.shutdown()


class TestWorkflowIntegration:
    """Integration tests for the complete workflow."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_message_flow(self):
        """Test complete end-to-end message flow."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            # Create workflow
            workflow = ChatWorkflow()
            
            # Mock agent response
            with patch.object(ConversationalAgent, 'process', new_callable=AsyncMock) as mock_process:
                mock_process.return_value = "Hello! I'm working correctly."
                
                await workflow.setup()
                
                try:
                    # Get components
                    user_input_du = workflow.components['user_input_du']
                    agent_step = workflow.components['agent_step']
                    agent_output_du = workflow.components['agent_output_du']
                    
                    # Test direct step processing (bypassing triggers for simplicity)
                    test_input = {'user_input': 'Hello, are you working?'}
                    result = await agent_step.process(test_input)
                    
                    # Verify response
                    assert 'agent_response' in result
                    assert result['agent_response'] == "Hello! I'm working correctly."
                    
                    # Verify agent was called with correct input
                    mock_process.assert_called_once_with('Hello, are you working?')
                    
                finally:
                    await workflow.shutdown()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 