#!/usr/bin/env python3
"""
Tests for Enhanced NanoBrain Chat Workflow Demo

Comprehensive test suite covering:
- Conversation history management
- Performance metrics tracking
- Enhanced CLI interface
- Multi-turn context management
- Export/import functionality
- Real-time statistics
"""

import pytest
import asyncio
import tempfile
import os
import json
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))

from enhanced_chat_workflow_demo import (
    ConversationMessage, PerformanceMetrics, ConversationHistoryManager,
    PerformanceTracker, EnhancedConversationalAgentStep, EnhancedCLIInterface,
    EnhancedChatWorkflow
)
from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig
from nanobrain.core.step import StepConfig
from nanobrain.core.agent import ConversationalAgent, AgentConfig


class TestConversationMessage:
    """Test ConversationMessage dataclass."""
    
    def test_conversation_message_creation(self):
        """Test creating a conversation message."""
        timestamp = datetime.now()
        message = ConversationMessage(
            timestamp=timestamp,
            user_input="Hello",
            agent_response="Hi there!",
            response_time_ms=150.5,
            conversation_id="conv_123",
            message_id=1
        )
        
        assert message.timestamp == timestamp
        assert message.user_input == "Hello"
        assert message.agent_response == "Hi there!"
        assert message.response_time_ms == 150.5
        assert message.conversation_id == "conv_123"
        assert message.message_id == 1


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        start_time = datetime.now()
        metrics = PerformanceMetrics(
            total_conversations=5,
            total_messages=25,
            average_response_time_ms=200.0,
            total_response_time_ms=5000.0,
            error_count=2,
            uptime_seconds=3600.0,
            messages_per_minute=0.4,
            start_time=start_time
        )
        
        assert metrics.total_conversations == 5
        assert metrics.total_messages == 25
        assert metrics.average_response_time_ms == 200.0
        assert metrics.total_response_time_ms == 5000.0
        assert metrics.error_count == 2
        assert metrics.uptime_seconds == 3600.0
        assert metrics.messages_per_minute == 0.4
        assert metrics.start_time == start_time


class TestConversationHistoryManager:
    """Test ConversationHistoryManager functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def history_manager(self, temp_db):
        """Create a ConversationHistoryManager with temporary database."""
        return ConversationHistoryManager(temp_db)
    
    def test_database_initialization(self, history_manager, temp_db):
        """Test database initialization."""
        # Check that database file exists
        assert os.path.exists(temp_db)
        
        # Check that tables are created
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'conversations' in tables
        conn.close()
    
    @pytest.mark.asyncio
    async def test_save_and_retrieve_message(self, history_manager):
        """Test saving and retrieving conversation messages."""
        message = ConversationMessage(
            timestamp=datetime.now(),
            user_input="Test input",
            agent_response="Test response",
            response_time_ms=100.0,
            conversation_id="test_conv",
            message_id=1
        )
        
        # Save message
        await history_manager.save_message(message)
        
        # Retrieve messages
        history = await history_manager.get_conversation_history("test_conv")
        
        assert len(history) == 1
        retrieved = history[0]
        assert retrieved.user_input == "Test input"
        assert retrieved.agent_response == "Test response"
        assert retrieved.response_time_ms == 100.0
        assert retrieved.conversation_id == "test_conv"
        assert retrieved.message_id == 1
    
    @pytest.mark.asyncio
    async def test_multiple_messages_ordering(self, history_manager):
        """Test that multiple messages are retrieved in correct order."""
        messages = []
        for i in range(3):
            message = ConversationMessage(
                timestamp=datetime.now(),
                user_input=f"Input {i}",
                agent_response=f"Response {i}",
                response_time_ms=100.0 + i,
                conversation_id="test_conv",
                message_id=i + 1
            )
            messages.append(message)
            await history_manager.save_message(message)
        
        # Retrieve messages
        history = await history_manager.get_conversation_history("test_conv")
        
        assert len(history) == 3
        # Should be in chronological order (message_id 1, 2, 3)
        for i, msg in enumerate(history):
            assert msg.message_id == i + 1
            assert msg.user_input == f"Input {i}"
    
    @pytest.mark.asyncio
    async def test_get_recent_conversations(self, history_manager):
        """Test retrieving recent conversation IDs."""
        # Create messages in different conversations
        for conv_id in ["conv_1", "conv_2", "conv_3"]:
            message = ConversationMessage(
                timestamp=datetime.now(),
                user_input="Test",
                agent_response="Response",
                response_time_ms=100.0,
                conversation_id=conv_id,
                message_id=1
            )
            await history_manager.save_message(message)
        
        # Get recent conversations
        recent = await history_manager.get_recent_conversations(hours=24)
        
        assert len(recent) == 3
        assert "conv_1" in recent
        assert "conv_2" in recent
        assert "conv_3" in recent
    
    @pytest.mark.asyncio
    async def test_export_conversations(self, history_manager):
        """Test exporting conversations to JSON."""
        # Create test messages
        for conv_id in ["conv_1", "conv_2"]:
            for msg_id in [1, 2]:
                message = ConversationMessage(
                    timestamp=datetime.now(),
                    user_input=f"Input {msg_id}",
                    agent_response=f"Response {msg_id}",
                    response_time_ms=100.0,
                    conversation_id=conv_id,
                    message_id=msg_id
                )
                await history_manager.save_message(message)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_file = f.name
        
        try:
            await history_manager.export_conversations(export_file)
            
            # Verify export file
            with open(export_file, 'r') as f:
                data = json.load(f)
            
            assert "conv_1" in data
            assert "conv_2" in data
            assert len(data["conv_1"]) == 2
            assert len(data["conv_2"]) == 2
            
        finally:
            os.unlink(export_file)


class TestPerformanceTracker:
    """Test PerformanceTracker functionality."""
    
    @pytest.fixture
    def tracker(self):
        """Create a PerformanceTracker instance."""
        return PerformanceTracker()
    
    def test_initial_metrics(self, tracker):
        """Test initial performance metrics."""
        metrics = tracker.get_current_metrics()
        
        assert metrics.total_messages == 0
        assert metrics.total_response_time_ms == 0.0
        assert metrics.average_response_time_ms == 0.0
        assert metrics.error_count == 0
        assert metrics.start_time is not None
    
    def test_record_message_success(self, tracker):
        """Test recording successful messages."""
        tracker.record_message(100.0, error=False)
        tracker.record_message(200.0, error=False)
        
        metrics = tracker.get_current_metrics()
        
        assert metrics.total_messages == 2
        assert metrics.total_response_time_ms == 300.0
        assert metrics.average_response_time_ms == 150.0
        assert metrics.error_count == 0
    
    def test_record_message_error(self, tracker):
        """Test recording error messages."""
        tracker.record_message(100.0, error=True)
        tracker.record_message(200.0, error=False)
        
        metrics = tracker.get_current_metrics()
        
        assert metrics.total_messages == 2
        assert metrics.error_count == 1
    
    def test_recent_response_times(self, tracker):
        """Test tracking recent response times."""
        response_times = [100.0, 150.0, 200.0]
        
        for rt in response_times:
            tracker.record_message(rt, error=False)
        
        recent = tracker.get_recent_response_times()
        
        assert len(recent) == 3
        assert recent == response_times
    
    def test_response_times_limit(self, tracker):
        """Test that response times are limited to last 100."""
        # Record 150 messages
        for i in range(150):
            tracker.record_message(float(i), error=False)
        
        recent = tracker.get_recent_response_times()
        
        # Should only keep last 100
        assert len(recent) == 100
        assert recent[0] == 50.0  # Should start from message 50
        assert recent[-1] == 149.0  # Should end at message 149


class TestEnhancedConversationalAgentStep:
    """Test EnhancedConversationalAgentStep functionality."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock conversational agent."""
        agent = AsyncMock(spec=ConversationalAgent)
        agent.process = AsyncMock(return_value="Mock response")
        return agent
    
    @pytest.fixture
    def mock_history_manager(self):
        """Create a mock conversation history manager."""
        manager = AsyncMock(spec=ConversationHistoryManager)
        manager.save_message = AsyncMock()
        return manager
    
    @pytest.fixture
    def mock_performance_tracker(self):
        """Create a mock performance tracker."""
        tracker = MagicMock(spec=PerformanceTracker)
        tracker.record_message = MagicMock()
        return tracker
    
    @pytest.fixture
    def agent_step(self, mock_agent, mock_history_manager, mock_performance_tracker):
        """Create an EnhancedConversationalAgentStep instance."""
        config = StepConfig(
            name="test_step",
            description="Test step",
            debug_mode=True,
            enable_logging=True
        )
        
        step = EnhancedConversationalAgentStep(
            config, mock_agent, mock_history_manager, mock_performance_tracker
        )
        
        # Mock the logger
        step.nb_logger = MagicMock()
        
        return step
    
    @pytest.mark.asyncio
    async def test_process_valid_input(self, agent_step, mock_agent, mock_history_manager, mock_performance_tracker):
        """Test processing valid user input."""
        inputs = {'user_input': 'Hello there!'}
        
        result = await agent_step.process(inputs)
        
        # Verify agent was called
        mock_agent.process.assert_called_once_with('Hello there!')
        
        # Verify response
        assert result['agent_response'] == 'Mock response'
        
        # Verify history was saved
        mock_history_manager.save_message.assert_called_once()
        
        # Verify metrics were recorded
        mock_performance_tracker.record_message.assert_called_once()
        
        # Check that error=False was passed to metrics
        call_args = mock_performance_tracker.record_message.call_args
        assert call_args[1]['error'] is False
    
    @pytest.mark.asyncio
    async def test_process_empty_input(self, agent_step):
        """Test processing empty user input."""
        inputs = {'user_input': ''}
        
        result = await agent_step.process(inputs)
        
        assert result['agent_response'] == ''
    
    @pytest.mark.asyncio
    async def test_process_agent_error(self, agent_step, mock_agent, mock_history_manager, mock_performance_tracker):
        """Test handling agent processing errors."""
        inputs = {'user_input': 'Hello'}
        
        # Make agent raise an exception
        mock_agent.process.side_effect = Exception("Test error")
        
        result = await agent_step.process(inputs)
        
        # Verify error response
        assert 'Sorry, I encountered an error: Test error' in result['agent_response']
        
        # Verify error was recorded in metrics
        mock_performance_tracker.record_message.assert_called_once()
        call_args = mock_performance_tracker.record_message.call_args
        assert call_args[1]['error'] is True
        
        # Verify error was still saved to history
        mock_history_manager.save_message.assert_called_once()
    
    def test_start_new_conversation(self, agent_step):
        """Test starting a new conversation."""
        original_id = agent_step.current_conversation_id
        original_count = agent_step.conversation_count
        
        # Add a small delay to ensure timestamp difference
        import time
        time.sleep(0.001)
        
        agent_step.start_new_conversation()
        
        # Verify conversation ID changed
        assert agent_step.current_conversation_id != original_id
        
        # Verify conversation count increased
        assert agent_step.conversation_count == original_count + 1
        
        # Verify message ID reset
        assert agent_step.message_id == 0
    
    @pytest.mark.asyncio
    async def test_conversation_tracking(self, agent_step, mock_agent, mock_history_manager, mock_performance_tracker):
        """Test that conversation and message IDs are tracked correctly."""
        inputs = {'user_input': 'First message'}
        
        # Process first message
        await agent_step.process(inputs)
        
        # Check message ID incremented
        assert agent_step.message_id == 1
        
        # Process second message
        await agent_step.process(inputs)
        
        # Check message ID incremented again
        assert agent_step.message_id == 2
        
        # Verify both messages saved with correct IDs
        assert mock_history_manager.save_message.call_count == 2
        
        # Check the saved messages have correct message IDs
        calls = mock_history_manager.save_message.call_args_list
        assert calls[0][0][0].message_id == 1
        assert calls[1][0][0].message_id == 2


class TestEnhancedCLIInterface:
    """Test EnhancedCLIInterface functionality."""
    
    @pytest.fixture
    def mock_data_units(self):
        """Create mock data units."""
        input_du = AsyncMock(spec=DataUnitMemory)
        output_du = AsyncMock(spec=DataUnitMemory)
        input_du.store = AsyncMock()
        output_du.subscribe = AsyncMock()
        return input_du, output_du
    
    @pytest.fixture
    def mock_history_manager(self):
        """Create a mock conversation history manager."""
        manager = AsyncMock(spec=ConversationHistoryManager)
        manager.get_conversation_history = AsyncMock(return_value=[])
        manager.get_recent_conversations = AsyncMock(return_value=[])
        manager.export_conversations = AsyncMock()
        return manager
    
    @pytest.fixture
    def mock_performance_tracker(self):
        """Create a mock performance tracker."""
        tracker = MagicMock(spec=PerformanceTracker)
        tracker.get_current_metrics = MagicMock(return_value=PerformanceMetrics())
        tracker.get_recent_response_times = MagicMock(return_value=[])
        return tracker
    
    @pytest.fixture
    def mock_agent_step(self):
        """Create a mock agent step."""
        step = MagicMock()
        step.current_conversation_id = "test_conv"
        step.conversation_count = 1
        step.start_new_conversation = MagicMock()
        return step
    
    @pytest.fixture
    def cli_interface(self, mock_data_units, mock_history_manager, mock_performance_tracker, mock_agent_step):
        """Create an EnhancedCLIInterface instance."""
        input_du, output_du = mock_data_units
        
        cli = EnhancedCLIInterface(
            input_du, output_du, mock_history_manager, 
            mock_performance_tracker, mock_agent_step
        )
        
        return cli
    
    @pytest.mark.asyncio
    async def test_cli_initialization(self, cli_interface, mock_data_units):
        """Test CLI interface initialization."""
        input_du, output_du = mock_data_units
        
        # Test that CLI can be initialized
        assert cli_interface.running is False
        assert cli_interface.input_data_unit == input_du
        assert cli_interface.output_data_unit == output_du
    
    @pytest.mark.asyncio
    async def test_output_handling(self, cli_interface):
        """Test handling output from agent."""
        with patch('builtins.print') as mock_print:
            data = {'agent_response': 'Test response'}
            
            await cli_interface._on_output_received(data)
            
            # Verify response was printed
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            assert 'Test response' in call_args
    
    @pytest.mark.asyncio
    async def test_output_handling_empty(self, cli_interface):
        """Test handling empty output from agent."""
        with patch('builtins.print') as mock_print:
            data = {'agent_response': ''}
            
            await cli_interface._on_output_received(data)
            
            # Verify nothing was printed for empty response
            mock_print.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_command_help(self, cli_interface):
        """Test help command."""
        with patch('builtins.print') as mock_print:
            await cli_interface._handle_command('/help')
            
            # Verify help was displayed
            mock_print.assert_called()
            # Check that help content includes expected commands
            help_text = ''.join([call[0][0] for call in mock_print.call_args_list])
            assert '/quit' in help_text
            assert '/new' in help_text
            assert '/history' in help_text
    
    @pytest.mark.asyncio
    async def test_command_new_conversation(self, cli_interface, mock_agent_step):
        """Test new conversation command."""
        with patch('builtins.print'):
            await cli_interface._handle_command('/new')
            
            # Verify new conversation was started
            mock_agent_step.start_new_conversation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_command_quit(self, cli_interface):
        """Test quit command."""
        with patch('builtins.print'):
            await cli_interface._handle_command('/quit')
            
            # Verify CLI is set to stop running
            assert cli_interface.running is False
    
    @pytest.mark.asyncio
    async def test_command_history(self, cli_interface, mock_history_manager):
        """Test history command."""
        # Setup mock history
        mock_message = ConversationMessage(
            timestamp=datetime.now(),
            user_input="Test input",
            agent_response="Test response",
            response_time_ms=100.0,
            conversation_id="test_conv",
            message_id=1
        )
        mock_history_manager.get_conversation_history.return_value = [mock_message]
        
        with patch('builtins.print') as mock_print:
            await cli_interface._handle_command('/history')
            
            # Verify history was retrieved
            mock_history_manager.get_conversation_history.assert_called_once()
            
            # Verify history was displayed
            mock_print.assert_called()
            # Get all print calls and extract text
            history_text = ''
            for call in mock_print.call_args_list:
                if call[0]:  # Check if there are positional arguments
                    history_text += str(call[0][0])
            assert 'Test input' in history_text
            assert 'Test response' in history_text
    
    @pytest.mark.asyncio
    async def test_command_stats(self, cli_interface, mock_performance_tracker):
        """Test stats command."""
        # Setup mock metrics
        metrics = PerformanceMetrics(
            total_messages=10,
            average_response_time_ms=150.0,
            error_count=1
        )
        mock_performance_tracker.get_current_metrics.return_value = metrics
        mock_performance_tracker.get_recent_response_times.return_value = [100.0, 150.0, 200.0]
        
        with patch('builtins.print') as mock_print:
            await cli_interface._handle_command('/stats')
            
            # Verify stats were displayed
            mock_print.assert_called()
            stats_text = ''.join([call[0][0] for call in mock_print.call_args_list])
            assert '10' in stats_text  # total messages
            assert '150.0' in stats_text  # average response time
    
    @pytest.mark.asyncio
    async def test_command_export(self, cli_interface, mock_history_manager):
        """Test export command."""
        with patch('builtins.print') as mock_print:
            await cli_interface._handle_command('/export')
            
            # Verify export was called
            mock_history_manager.export_conversations.assert_called_once()
            
            # Verify success message was displayed
            mock_print.assert_called()
            export_text = ''.join([call[0][0] for call in mock_print.call_args_list])
            assert 'exported' in export_text.lower()


class TestEnhancedChatWorkflow:
    """Test EnhancedChatWorkflow functionality."""
    
    @pytest.fixture
    def workflow(self):
        """Create an EnhancedChatWorkflow instance."""
        return EnhancedChatWorkflow()
    
    def test_workflow_initialization(self, workflow):
        """Test workflow initialization."""
        assert workflow.factory is not None
        assert workflow.components == {}
        assert workflow.cli is None
        assert workflow.executor is None
        assert workflow.history_manager is not None
        assert workflow.performance_tracker is not None
    
    @pytest.mark.asyncio
    async def test_workflow_setup_components(self, workflow):
        """Test workflow component setup."""
        # Mock the agent initialization to avoid OpenAI API calls
        with patch('core.agent.ConversationalAgent.initialize', new_callable=AsyncMock):
            with patch('core.executor.LocalExecutor.initialize', new_callable=AsyncMock):
                await workflow.setup()
                
                # Verify components were created
                assert 'user_input_du' in workflow.components
                assert 'agent_input_du' in workflow.components
                assert 'agent_output_du' in workflow.components
                assert 'agent' in workflow.components
                assert 'agent_step' in workflow.components
                assert workflow.cli is not None
                assert workflow.executor is not None


class TestWorkflowIntegration:
    """Integration tests for the enhanced chat workflow."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_message_flow_with_history(self):
        """Test complete message flow with history tracking."""
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # Create components
            history_manager = ConversationHistoryManager(db_path)
            performance_tracker = PerformanceTracker()
            
            # Create mock agent
            mock_agent = AsyncMock(spec=ConversationalAgent)
            mock_agent.process = AsyncMock(return_value="Hello! How can I help you?")
            
            # Create data units
            user_input_config = DataUnitConfig(
                data_type="memory",
                name="user_input",
                description="User input",
                persistent=False,
                cache_size=100
            )
            user_input_du = DataUnitMemory(user_input_config)
            await user_input_du.initialize()
            
            agent_output_config = DataUnitConfig(
                data_type="memory",
                name="agent_output",
                description="Agent output",
                persistent=False,
                cache_size=100
            )
            agent_output_du = DataUnitMemory(agent_output_config)
            await agent_output_du.initialize()
            
            # Create enhanced agent step
            step_config = StepConfig(
                name="test_step",
                description="Test step",
                debug_mode=True,
                enable_logging=True
            )
            
            agent_step = EnhancedConversationalAgentStep(
                step_config, mock_agent, history_manager, performance_tracker
            )
            agent_step.nb_logger = MagicMock()  # Mock logger
            
            # Test message processing
            user_input = "Hello there!"
            result = await agent_step.process({'user_input': user_input})
            
            # Verify response
            assert result['agent_response'] == "Hello! How can I help you?"
            
            # Verify history was saved
            history = await history_manager.get_conversation_history(
                agent_step.current_conversation_id
            )
            assert len(history) == 1
            assert history[0].user_input == user_input
            assert history[0].agent_response == "Hello! How can I help you?"
            
            # Verify metrics were recorded
            metrics = performance_tracker.get_current_metrics()
            assert metrics.total_messages == 1
            assert metrics.error_count == 0
            
            # Test second message to verify conversation continuity
            second_input = "What can you do?"
            mock_agent.process.return_value = "I can help with many things!"
            
            result2 = await agent_step.process({'user_input': second_input})
            
            # Verify second response
            assert result2['agent_response'] == "I can help with many things!"
            
            # Verify history now has both messages
            history = await history_manager.get_conversation_history(
                agent_step.current_conversation_id
            )
            assert len(history) == 2
            assert history[0].message_id == 1
            assert history[1].message_id == 2
            
            # Verify metrics updated
            metrics = performance_tracker.get_current_metrics()
            assert metrics.total_messages == 2
            
        finally:
            # Cleanup
            os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_conversation_export_import_cycle(self):
        """Test exporting and importing conversation data."""
        # Create temporary database and export file
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            # Create history manager and add test data
            history_manager = ConversationHistoryManager(db_path)
            
            # Add multiple conversations
            conversations = {
                "conv_1": [
                    ("Hello", "Hi there!"),
                    ("How are you?", "I'm doing well, thanks!")
                ],
                "conv_2": [
                    ("What's the weather?", "I don't have weather data."),
                    ("Thanks anyway", "You're welcome!")
                ]
            }
            
            for conv_id, messages in conversations.items():
                for msg_id, (user_input, agent_response) in enumerate(messages, 1):
                    message = ConversationMessage(
                        timestamp=datetime.now(),
                        user_input=user_input,
                        agent_response=agent_response,
                        response_time_ms=100.0,
                        conversation_id=conv_id,
                        message_id=msg_id
                    )
                    await history_manager.save_message(message)
            
            # Export conversations
            await history_manager.export_conversations(export_path)
            
            # Verify export file
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            
            assert "conv_1" in exported_data
            assert "conv_2" in exported_data
            assert len(exported_data["conv_1"]) == 2
            assert len(exported_data["conv_2"]) == 2
            
            # Verify message content
            conv1_msg1 = exported_data["conv_1"][0]
            assert conv1_msg1["user_input"] == "Hello"
            assert conv1_msg1["agent_response"] == "Hi there!"
            
        finally:
            # Cleanup
            os.unlink(db_path)
            os.unlink(export_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])