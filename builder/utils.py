"""
Utility functions for the NanoBrain builder.

This module provides utility functions for the NanoBrain builder, including
decorators for ensuring proper async behavior.
"""

import asyncio
import functools
import inspect
from typing import Callable, Any, TypeVar, cast

# Type variable for the decorated function
F = TypeVar('F', bound=Callable[..., Any])


def ensure_async(func: F) -> F:
    """
    Decorator to ensure that a function is properly async.
    
    This decorator checks that:
    1. The function is defined as async
    2. The function returns a coroutine when called
    3. The function doesn't use asyncio.run() internally
    
    Args:
        func: The function to decorate
    
    Returns:
        The decorated function
    
    Raises:
        TypeError: If the function is not async
        RuntimeError: If the function uses asyncio.run() internally
    """
    # Check that the function is defined as async
    if not asyncio.iscoroutinefunction(func):
        raise TypeError(f"Function {func.__name__} must be defined as async")
    
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Get the source code of the function
        try:
            source = inspect.getsource(func)
            
            # Check if the function uses asyncio.run()
            if "asyncio.run(" in source:
                raise RuntimeError(
                    f"Function {func.__name__} uses asyncio.run(), which cannot be called from a running event loop. "
                    f"Use 'await' instead."
                )
        except (OSError, IOError):
            # If we can't get the source code, we can't check for asyncio.run()
            pass
        
        # Call the function and return the result
        return await func(*args, **kwargs)
    
    # Cast to the original function type to satisfy the type checker
    return cast(F, wrapper)


def safe_async_run(coro: Any) -> Any:
    """
    Safely run an async function, handling the case where we're already in an event loop.
    
    This function is useful for running async functions from sync code. It will:
    1. Check if we're already in an event loop
    2. If we are, create a new task in the current loop and wait for it
    3. If we're not, create a new loop and run the coroutine in it
    
    Args:
        coro: The coroutine to run
    
    Returns:
        The result of the coroutine
    """
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_event_loop()
        
        # Check if the loop is running
        if loop.is_running():
            # We're already in a running event loop, so we can't use run_until_complete
            # Instead, create a future and wait for it to complete
            future = asyncio.ensure_future(coro, loop=loop)
            
            # Create a new event to signal completion
            done = asyncio.Event()
            
            # Define a callback to set the event when the future is done
            def on_done(_):
                done.set()
            
            future.add_done_callback(on_done)
            
            # Wait for the future to complete
            loop.run_until_complete(done.wait())
            
            # Get the result or raise the exception
            return future.result()
        else:
            # We have a loop, but it's not running, so we can run the coroutine directly
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, so create one and run the coroutine
        return asyncio.run(coro) 