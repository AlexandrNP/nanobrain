"""
Stream-based data storage implementation.

Real-time data streaming with subscription support.
"""

import asyncio
from typing import Any, Dict, Optional, List, AsyncIterator
from collections import deque
from .data_unit_base import DataUnitBase


class DataUnitStream(DataUnitBase):
    """Real-time data streaming with subscription support."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._current_data: Any = None
        self._subscribers: List[asyncio.Queue] = []
        self._history: deque = deque(maxlen=self.config.get('history_size', 100))
        self._stream_lock = asyncio.Lock()
        
    async def get(self) -> Any:
        """Get current data from stream."""
        async with self._stream_lock:
            return self._current_data
            
    async def set(self, data: Any) -> None:
        """Set data in stream and notify subscribers."""
        async with self._stream_lock:
            self._current_data = data
            self._history.append({
                'data': data,
                'timestamp': asyncio.get_event_loop().time()
            })
            
            # Notify all subscribers
            await self._notify_subscribers(data)
            self.set_metadata('total_messages', len(self._history))
            self.logger.debug(f"Data published to stream {self.name}")
            
    async def clear(self) -> None:
        """Clear stream data."""
        async with self._stream_lock:
            self._current_data = None
            self._history.clear()
            # Notify subscribers of clear
            await self._notify_subscribers(None)
            self.metadata.clear()
            self.logger.debug(f"Stream {self.name} cleared")
            
    async def subscribe(self) -> asyncio.Queue:
        """Subscribe to stream updates."""
        async with self._stream_lock:
            queue = asyncio.Queue(maxsize=self.config.get('queue_size', 1000))
            self._subscribers.append(queue)
            self.set_metadata('subscriber_count', len(self._subscribers))
            self.logger.debug(f"New subscriber added to stream {self.name}")
            return queue
            
    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unsubscribe from stream updates."""
        async with self._stream_lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)
                self.set_metadata('subscriber_count', len(self._subscribers))
                self.logger.debug(f"Subscriber removed from stream {self.name}")
                
    async def _notify_subscribers(self, data: Any) -> None:
        """Notify all subscribers of new data."""
        if not self._subscribers:
            return
            
        # Create notification tasks for all subscribers
        tasks = []
        for queue in self._subscribers[:]:  # Copy list to avoid modification during iteration
            tasks.append(self._notify_subscriber(queue, data))
            
        # Execute notifications concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def _notify_subscriber(self, queue: asyncio.Queue, data: Any) -> None:
        """Notify a single subscriber."""
        try:
            await asyncio.wait_for(queue.put(data), timeout=1.0)
        except asyncio.TimeoutError:
            # Remove slow subscribers
            self.logger.warning(f"Removing slow subscriber from stream {self.name}")
            if queue in self._subscribers:
                self._subscribers.remove(queue)
        except Exception as e:
            self.logger.error(f"Error notifying subscriber: {e}")
            
    async def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get stream history."""
        async with self._stream_lock:
            history = list(self._history)
            if limit:
                history = history[-limit:]
            return history
            
    async def stream_iterator(self) -> AsyncIterator[Any]:
        """Get an async iterator for the stream."""
        queue = await self.subscribe()
        try:
            while True:
                data = await queue.get()
                yield data
        finally:
            await self.unsubscribe(queue)
            
    async def publish_batch(self, data_list: List[Any]) -> None:
        """Publish multiple data items."""
        for data in data_list:
            await self.set(data)
            
    async def get_subscriber_count(self) -> int:
        """Get current number of subscribers."""
        async with self._stream_lock:
            return len(self._subscribers)
            
    async def _shutdown_impl(self) -> None:
        """Shutdown stream and notify subscribers."""
        async with self._stream_lock:
            # Notify all subscribers of shutdown
            for queue in self._subscribers[:]:
                try:
                    await asyncio.wait_for(queue.put(None), timeout=0.1)
                except:
                    pass
            self._subscribers.clear() 