#!/usr/bin/env python3
"""
Test script for the _process_query method in DataStorageCommandLine.
"""

import sys
import os
import pytest
from typing import List

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.DataStorageCommandLine import DataStorageCommandLine
from src.ExecutorBase import ExecutorBase
from src.DataUnitString import DataUnitString


@pytest.fixture
def queries():
    """Fixture that provides test queries."""
    return [
        "normal query string",
        "<TestClass>>Create a basic test class with methods for testing",
        "<ComplexProcessor>>Create a data processing class with methods for filtering, sorting, and transforming data",
        "query with <brackets> but not in the right format",
        "<ClassName>wrong format",
        "<ClassName>>>"  # Empty instructions
    ]


def test_process_query(queries):
    """
    Test the _process_query method with various inputs.
    
    Args:
        queries: List of query strings to process
    """
    # Create an instance of DataStorageCommandLine
    executor = ExecutorBase()
    command_line = DataStorageCommandLine(executor=executor, debug=True)
    command_line.output = DataUnitString()
    
    # Process each query
    for query in queries:
        print(f"\nInput query: {query}")
        result = command_line._process_query(query)
        print(f"Output result: {result}")
        print("=" * 50)


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 