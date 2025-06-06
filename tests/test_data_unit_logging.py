"""
Test suite for data unit content logging functionality.

This test verifies that data unit operations are properly logged with content details.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import data unit classes
from src.core.data_unit import (
    DataUnitMemory, DataUnitFile, DataUnitString, DataUnitConfig, DataUnitType
)


class TestDataUnitContentLogging:
    """Test data unit content logging functionality."""
    
    @pytest.mark.asyncio
    async def test_memory_data_unit_logging(self):
        """Test that memory data unit operations are logged with content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "memory_unit.log"
            
            # Create memory data unit with logging enabled
            memory_unit = DataUnitMemory(
                name="test_memory_unit",
                enable_logging=True,
                log_file=log_file,
                debug_mode=True
            )
            
            await memory_unit.initialize()
            
            # Test data
            test_data = {"key": "value", "number": 42}
            
            # Write data
            await memory_unit.write(test_data)
            
            # Read data
            read_data = await memory_unit.read()
            
            await memory_unit.shutdown()
            
            # Verify data was processed correctly
            assert read_data == test_data
            
            # Verify log file was created and contains content
            assert log_file.exists()
            log_content = log_file.read_text()
            
            # Check that log contains expected operations
            assert "DataUnit initialize" in log_content
            assert "DataUnit write" in log_content
            assert "DataUnit read" in log_content
            assert "DataUnit shutdown" in log_content
            
            # Check that actual data content is logged
            assert '"key": "value"' in log_content
            assert '"number": 42' in log_content
            
            # Check that metadata is included
            assert "data_unit_type" in log_content
            assert "DataUnitMemory" in log_content
    
    @pytest.mark.asyncio
    async def test_file_data_unit_logging(self):
        """Test that file data unit operations are logged with content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "file_unit.log"
            data_file = Path(temp_dir) / "test_data.json"
            
            # Create file data unit with logging enabled
            file_unit = DataUnitFile(
                file_path=str(data_file),
                name="test_file_unit",
                enable_logging=True,
                log_file=log_file,
                debug_mode=True
            )
            
            await file_unit.initialize()
            
            # Test data
            test_data = {
                "message": "Test file data",
                "items": [1, 2, 3],
                "metadata": {"author": "test"}
            }
            
            # Write and read data
            await file_unit.write(test_data)
            read_data = await file_unit.read()
            
            await file_unit.shutdown()
            
            # Verify data was processed correctly
            assert read_data == test_data
            
            # Verify log file contains content
            assert log_file.exists()
            log_content = log_file.read_text()
            
            # Check that actual data content is logged
            assert "Test file data" in log_content
            assert '"items": [' in log_content
            assert '"author": "test"' in log_content
    
    @pytest.mark.asyncio
    async def test_string_data_unit_logging(self):
        """Test that string data unit operations are logged with content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "string_unit.log"
            
            # Create string data unit with logging enabled
            string_unit = DataUnitString(
                initial_value="Initial content",
                name="test_string_unit",
                enable_logging=True,
                log_file=log_file,
                debug_mode=True
            )
            
            await string_unit.initialize()
            
            # Test string operations
            await string_unit.write("Hello, World!")
            content = await string_unit.read()
            
            await string_unit.shutdown()
            
            # Verify data was processed correctly
            assert content == "Hello, World!"
            
            # Verify log file contains content
            assert log_file.exists()
            log_content = log_file.read_text()
            
            # Check that actual string content is logged
            assert "Hello, World!" in log_content
            # Note: Initial content is not logged during initialization, only during read/write operations
    
    @pytest.mark.asyncio
    async def test_large_data_serialization(self):
        """Test that large data structures are properly serialized for logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "large_data.log"
            
            # Create memory data unit
            memory_unit = DataUnitMemory(
                name="large_data_unit",
                enable_logging=True,
                log_file=log_file,
                debug_mode=True
            )
            
            await memory_unit.initialize()
            
            # Create large data structure
            large_data = {
                "large_string": "x" * 2000,  # 2KB string
                "large_list": list(range(100)),  # 100 items
                "nested_data": {f"key_{i}": f"value_{i}" for i in range(50)}  # 50 nested items
            }
            
            # Write large data
            await memory_unit.write(large_data)
            
            await memory_unit.shutdown()
            
            # Verify log file contains truncated/summarized content
            assert log_file.exists()
            log_content = log_file.read_text()
            
            # Check that large data is handled appropriately
            # Should contain truncation indicators for large strings
            assert "truncated" in log_content or "additional_items" in log_content or "additional_keys" in log_content
            
            # Should still contain some actual data
            assert "large_string" in log_content
            assert "large_list" in log_content
    
    @pytest.mark.asyncio
    async def test_logging_disabled(self):
        """Test that logging can be disabled for data units."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "disabled_logging.log"
            
            # Create memory data unit with logging disabled
            memory_unit = DataUnitMemory(
                name="no_logging_unit",
                enable_logging=False,
                log_file=log_file,
                debug_mode=True
            )
            
            await memory_unit.initialize()
            
            # Perform operations
            await memory_unit.write("test data")
            await memory_unit.read()
            
            await memory_unit.shutdown()
            
            # Verify no log file was created or it's empty
            if log_file.exists():
                log_content = log_file.read_text()
                # Should not contain data unit operation logs
                assert "DataUnit write" not in log_content
                assert "DataUnit read" not in log_content
    
    def test_data_serialization_edge_cases(self):
        """Test edge cases in data serialization for logging."""
        from src.core.logging_system import NanoBrainLogger
        
        logger = NanoBrainLogger("test_logger")
        
        # Test None
        result = logger._serialize_data_for_logging(None)
        assert result is None
        
        # Test empty structures
        assert logger._serialize_data_for_logging({}) == {}
        assert logger._serialize_data_for_logging([]) == []
        
        # Test circular reference handling
        circular_dict = {"key": "value"}
        circular_dict["self"] = circular_dict
        
        # Should not raise an exception
        result = logger._serialize_data_for_logging(circular_dict)
        assert isinstance(result, dict)
        assert "key" in result
        
        # Test custom objects
        class CustomObject:
            def __init__(self):
                self.name = "test_object"
                self.value = 42
        
        obj = CustomObject()
        result = logger._serialize_data_for_logging(obj)
        assert isinstance(result, dict)
        assert result["__type__"] == "CustomObject"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 