"""
Test suite for logging configuration and third-party library logging suppression.

This test verifies that when logging is set to 'file' mode, INFO and DEBUG messages
from LLM models and PARSL are written to files instead of being displayed in the console.
"""

import pytest
import logging
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO
import sys

# Import the logging system components
from nanobrain.core.logging_system import (
    NanoBrainLogger, 
    _configure_global_logging,
    _suppress_third_party_console_logging,
    configure_third_party_loggers,
    get_logging_status,
    reconfigure_global_logging
)


class TestLoggingConfiguration:
    """Test logging configuration and third-party library suppression."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clear any existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Reset third-party loggers
        third_party_loggers = ['openai', 'parsl', 'httpx']
        for logger_name in third_party_loggers:
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
    
    def teardown_method(self):
        """Clean up after tests."""
        # Reset logging configuration
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        logging.basicConfig(level=logging.INFO)
    
    @patch('nanobrain.core.logging_system.should_log_to_console')
    @patch('nanobrain.core.logging_system.should_log_to_file')
    @patch('nanobrain.core.logging_system.get_logging_config')
    def test_file_mode_suppresses_console_output(self, mock_get_config, mock_should_file, mock_should_console):
        """Test that file mode suppresses console output for third-party libraries."""
        # Configure mocks for file-only mode
        mock_should_console.return_value = False
        mock_should_file.return_value = True
        mock_get_config.return_value = {
            'mode': 'file',
            'console': {'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'}
        }
        
        # Capture console output
        console_output = StringIO()
        
        # Configure global logging
        with patch('sys.stdout', console_output):
            _configure_global_logging()
            
            # Test third-party library logging
            openai_logger = logging.getLogger('openai')
            parsl_logger = logging.getLogger('parsl')
            
            # These should not appear in console output
            openai_logger.info("OpenAI API call")
            parsl_logger.debug("Parsl executor debug")
            
            # Get console output
            output = console_output.getvalue()
        
        # Verify no third-party library output in console
        assert "OpenAI API call" not in output
        assert "Parsl executor debug" not in output
        
        # Verify third-party loggers are configured correctly
        assert openai_logger.level >= logging.WARNING
        assert parsl_logger.level >= logging.WARNING
        assert not openai_logger.propagate
        assert not parsl_logger.propagate
    
    @patch('nanobrain.core.logging_system.should_log_to_console')
    @patch('nanobrain.core.logging_system.should_log_to_file')
    @patch('nanobrain.core.logging_system.get_logging_config')
    def test_console_mode_allows_output(self, mock_get_config, mock_should_file, mock_should_console):
        """Test that console mode allows third-party library output."""
        # Configure mocks for console mode
        mock_should_console.return_value = True
        mock_should_file.return_value = False
        mock_get_config.return_value = {
            'mode': 'console',
            'console': {'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'}
        }
        
        # Configure global logging
        _configure_global_logging()
        
        # Test third-party library logging
        openai_logger = logging.getLogger('openai')
        parsl_logger = logging.getLogger('parsl')
        
        # Verify third-party loggers are not suppressed
        assert openai_logger.level == logging.NOTSET or openai_logger.level <= logging.INFO
        assert parsl_logger.level == logging.NOTSET or parsl_logger.level <= logging.INFO
        assert openai_logger.propagate
        assert parsl_logger.propagate
    
    @patch('nanobrain.core.logging_system.should_log_to_console')
    @patch('nanobrain.core.logging_system.should_log_to_file')
    @patch('nanobrain.core.logging_system.get_logging_config')
    def test_both_mode_configuration(self, mock_get_config, mock_should_file, mock_should_console):
        """Test that both mode allows console output but also enables file logging."""
        # Configure mocks for both mode
        mock_should_console.return_value = True
        mock_should_file.return_value = True
        mock_get_config.return_value = {
            'mode': 'both',
            'console': {'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'}
        }
        
        # Configure global logging
        _configure_global_logging()
        
        # Verify root logger has console handler
        root_logger = logging.getLogger()
        has_console_handler = any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
        assert has_console_handler
        
        # Test third-party library logging
        openai_logger = logging.getLogger('openai')
        
        # Verify third-party loggers are not suppressed in both mode
        assert openai_logger.propagate
    
    def test_third_party_logger_suppression(self):
        """Test direct third-party logger suppression function."""
        # Set up third-party loggers
        test_loggers = ['openai', 'parsl', 'httpx']
        
        for logger_name in test_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            logger.propagate = True
        
        # Suppress third-party logging
        _suppress_third_party_console_logging()
        
        # Verify suppression
        for logger_name in test_loggers:
            logger = logging.getLogger(logger_name)
            assert logger.level >= logging.WARNING
            assert not logger.propagate
            # Verify no StreamHandler
            stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
            assert len(stream_handlers) == 0
    
    def test_configure_third_party_loggers_function(self):
        """Test the configure_third_party_loggers utility function."""
        # Test suppression
        configure_third_party_loggers(console_enabled=False)
        
        openai_logger = logging.getLogger('openai')
        assert openai_logger.level >= logging.WARNING
        assert not openai_logger.propagate
        
        # Test re-enabling
        configure_third_party_loggers(console_enabled=True)
        
        openai_logger = logging.getLogger('openai')
        assert openai_logger.level == logging.NOTSET
        assert openai_logger.propagate
    
    def test_nanobrain_logger_respects_file_mode(self):
        """Test that NanoBrainLogger respects file-only mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            # Mock file-only mode
            with patch('nanobrain.core.logging_system.should_log_to_console', return_value=False), \
                 patch('nanobrain.core.logging_system.should_log_to_file', return_value=True), \
                 patch('nanobrain.core.logging_system.get_logging_config', return_value={'console': {}}):
                
                logger = NanoBrainLogger("test_logger", log_file=log_file)
                
                # Verify console is disabled, file is enabled
                assert not logger.enable_console
                assert logger.enable_file
                
                # Verify no console handlers
                console_handlers = [h for h in logger.logger.handlers if isinstance(h, logging.StreamHandler)]
                assert len(console_handlers) == 0
                
                # Verify file handler exists
                file_handlers = [h for h in logger.logger.handlers if isinstance(h, logging.FileHandler)]
                assert len(file_handlers) == 1
    
    def test_get_logging_status(self):
        """Test the logging status utility function."""
        with patch('nanobrain.core.logging_system.get_logging_mode', return_value='file'), \
             patch('nanobrain.core.logging_system.should_log_to_console', return_value=False), \
             patch('nanobrain.core.logging_system.should_log_to_file', return_value=True):
            
            status = get_logging_status()
            
            assert status['nanobrain_mode'] == 'file'
            assert not status['console_enabled']
            assert status['file_enabled']
            assert 'root_logger_level' in status
            assert 'root_logger_handlers' in status
    
    def test_reconfigure_global_logging(self):
        """Test that global logging can be reconfigured dynamically."""
        # Initial configuration
        with patch('nanobrain.core.logging_system.should_log_to_console', return_value=True):
            _configure_global_logging()
            
            root_logger = logging.getLogger()
            initial_handler_count = len(root_logger.handlers)
        
        # Reconfigure to file-only mode
        with patch('nanobrain.core.logging_system.should_log_to_console', return_value=False), \
             patch('nanobrain.core.logging_system.should_log_to_file', return_value=True), \
             patch('nanobrain.core.logging_system.get_logging_config', return_value={'console': {}}):
            
            reconfigure_global_logging()
            
            # Verify configuration changed
            root_logger = logging.getLogger()
            # In file mode, we should have a NullHandler
            null_handlers = [h for h in root_logger.handlers if isinstance(h, logging.NullHandler)]
            assert len(null_handlers) > 0


class TestIntegrationWithDemos:
    """Test integration with demo applications."""
    
    @patch('nanobrain.core.logging_system.should_log_to_console')
    @patch('nanobrain.core.logging_system.should_log_to_file') 
    @patch('nanobrain.core.logging_system.get_logging_config')
    def test_parsl_demo_file_mode(self, mock_get_config, mock_should_file, mock_should_console):
        """Test that Parsl demo respects file-only logging mode."""
        # Configure file-only mode
        mock_should_console.return_value = False
        mock_should_file.return_value = True
        mock_get_config.return_value = {
            'mode': 'file',
            'console': {'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'}
        }
        
        # Configure logging
        _configure_global_logging()
        
        # Test Parsl-related loggers
        parsl_loggers = [
            'parsl',
            'parsl.executors',
            'parsl.providers',
            'parsl.monitoring'
        ]
        
        console_output = StringIO()
        
        with patch('sys.stdout', console_output):
            for logger_name in parsl_loggers:
                logger = logging.getLogger(logger_name)
                logger.info(f"Test message from {logger_name}")
                logger.debug(f"Debug message from {logger_name}")
        
        output = console_output.getvalue()
        
        # Verify no Parsl output in console
        for logger_name in parsl_loggers:
            assert f"Test message from {logger_name}" not in output
            assert f"Debug message from {logger_name}" not in output
    
    @patch('nanobrain.core.logging_system.should_log_to_console')
    @patch('nanobrain.core.logging_system.should_log_to_file')
    @patch('nanobrain.core.logging_system.get_logging_config')
    def test_openai_logging_suppression(self, mock_get_config, mock_should_file, mock_should_console):
        """Test that OpenAI library logging is suppressed in file mode."""
        # Configure file-only mode
        mock_should_console.return_value = False
        mock_should_file.return_value = True
        mock_get_config.return_value = {
            'mode': 'file',
            'console': {'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'}
        }
        
        # Configure logging
        _configure_global_logging()
        
        # Test OpenAI-related loggers
        openai_loggers = ['openai', 'httpx', 'httpcore']
        
        console_output = StringIO()
        
        with patch('sys.stdout', console_output):
            for logger_name in openai_loggers:
                logger = logging.getLogger(logger_name)
                logger.info(f"API call from {logger_name}")
                logger.debug(f"Debug info from {logger_name}")
        
        output = console_output.getvalue()
        
        # Verify no OpenAI-related output in console
        for logger_name in openai_loggers:
            assert f"API call from {logger_name}" not in output
            assert f"Debug info from {logger_name}" not in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 