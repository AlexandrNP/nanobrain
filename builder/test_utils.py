import unittest
import os
import sys
import asyncio
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from builder.utils import ensure_async, safe_async_run


class TestUtils(unittest.IsolatedAsyncioTestCase):
    """Test cases for utility functions."""
    
    async def test_ensure_async_decorator(self):
        """Test that the ensure_async decorator works correctly."""
        # Test with a properly defined async function
        @ensure_async
        async def good_function():
            return "good"
        
        # Verify that the function works correctly
        result = await good_function()
        self.assertEqual(result, "good")
        
        # Test with a non-async function - this should raise TypeError
        def not_async_function():
            return "not async"
        
        # Apply the decorator in a way that we can catch the exception
        with self.assertRaises(TypeError):
            ensure_async(not_async_function)
        
        # Test with a function that uses asyncio.run() - this should raise RuntimeError
        # We'll mock the source inspection to simulate finding asyncio.run()
        async def bad_function():
            return "bad"
        
        with patch('inspect.getsource', return_value="async def bad_function():\n    asyncio.run(something())"):
            # Apply the decorator
            decorated_bad = ensure_async(bad_function)
            
            # Call the decorated function - should raise RuntimeError
            with self.assertRaises(RuntimeError):
                await decorated_bad()
    
    async def test_safe_async_run_with_no_loop(self):
        """Test safe_async_run when no event loop exists."""
        # Create a simple async function
        async def simple_async_function():
            return "no loop"
        
        # Mock asyncio.get_event_loop to raise RuntimeError (no loop)
        with patch('asyncio.get_event_loop', side_effect=RuntimeError("No running event loop")), \
             patch('asyncio.run', return_value="no loop") as mock_run:
            
            # Call safe_async_run with a coroutine
            coro = simple_async_function()
            result = safe_async_run(coro)
            
            # Verify the result
            self.assertEqual(result, "no loop")
            
            # Verify that asyncio.run was called
            mock_run.assert_called_once()


if __name__ == '__main__':
    unittest.main() 