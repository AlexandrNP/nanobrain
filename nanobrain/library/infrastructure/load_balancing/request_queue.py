"""
Request queuing and prioritization.

Request queue implementation with priority support.
"""

import asyncio
import heapq
import time
from enum import Enum
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from nanobrain.core.logging_system import get_logger


class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class QueuedRequest:
    """Represents a queued request."""
    priority: RequestPriority
    request_data: Any
    timestamp: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: str(time.time()))
    metadata: Optional[Dict[str, Any]] = None
    
    def __lt__(self, other):
        # Higher priority values come first, then by timestamp (FIFO for same priority)
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.timestamp < other.timestamp


class RequestQueue:
    """Request queuing and prioritization."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queue: List[QueuedRequest] = []
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._not_full = asyncio.Condition(self._lock)
        self.logger = get_logger("request_queue")
        
    async def put(self, request_data: Any, priority: RequestPriority = RequestPriority.NORMAL, 
                  request_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a request to the queue."""
        async with self._not_full:
            while len(self._queue) >= self.max_size:
                await self._not_full.wait()
                
            queued_request = QueuedRequest(
                priority=priority,
                request_data=request_data,
                request_id=request_id or str(time.time()),
                metadata=metadata
            )
            
            heapq.heappush(self._queue, queued_request)
            self.logger.debug(f"Added request {queued_request.request_id} with priority {priority.name}")
            self._not_empty.notify()
            return True
            
    async def get(self, timeout: Optional[float] = None) -> Optional[QueuedRequest]:
        """Get the highest priority request from the queue."""
        async with self._not_empty:
            if timeout is not None:
                try:
                    await asyncio.wait_for(self._wait_for_request(), timeout=timeout)
                except asyncio.TimeoutError:
                    return None
            else:
                await self._wait_for_request()
                
            if self._queue:
                request = heapq.heappop(self._queue)
                self.logger.debug(f"Retrieved request {request.request_id}")
                self._not_full.notify()
                return request
            return None
            
    async def _wait_for_request(self):
        """Wait for a request to be available."""
        while not self._queue:
            await self._not_empty.wait()
            
    async def peek(self) -> Optional[QueuedRequest]:
        """Peek at the highest priority request without removing it."""
        async with self._lock:
            return self._queue[0] if self._queue else None
            
    async def size(self) -> int:
        """Get current queue size."""
        async with self._lock:
            return len(self._queue)
            
    async def is_empty(self) -> bool:
        """Check if queue is empty."""
        async with self._lock:
            return len(self._queue) == 0
            
    async def is_full(self) -> bool:
        """Check if queue is full."""
        async with self._lock:
            return len(self._queue) >= self.max_size
            
    async def clear(self) -> int:
        """Clear all requests from the queue."""
        async with self._lock:
            count = len(self._queue)
            self._queue.clear()
            self.logger.info(f"Cleared {count} requests from queue")
            self._not_full.notify_all()
            return count
            
    async def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        async with self._lock:
            priority_counts = {priority.name: 0 for priority in RequestPriority}
            oldest_timestamp = None
            newest_timestamp = None
            
            for request in self._queue:
                priority_counts[request.priority.name] += 1
                if oldest_timestamp is None or request.timestamp < oldest_timestamp:
                    oldest_timestamp = request.timestamp
                if newest_timestamp is None or request.timestamp > newest_timestamp:
                    newest_timestamp = request.timestamp
                    
            current_time = time.time()
            avg_wait_time = 0.0
            if self._queue:
                total_wait_time = sum(current_time - req.timestamp for req in self._queue)
                avg_wait_time = total_wait_time / len(self._queue)
                
            return {
                'queue_size': len(self._queue),
                'max_size': self.max_size,
                'utilization': (len(self._queue) / self.max_size) * 100,
                'priority_distribution': priority_counts,
                'average_wait_time': avg_wait_time,
                'oldest_request_age': current_time - oldest_timestamp if oldest_timestamp else 0,
                'newest_request_age': current_time - newest_timestamp if newest_timestamp else 0
            }
            
    async def remove_expired_requests(self, max_age: float) -> int:
        """Remove requests older than max_age seconds."""
        async with self._lock:
            current_time = time.time()
            cutoff_time = current_time - max_age
            
            # Filter out expired requests
            original_count = len(self._queue)
            self._queue = [req for req in self._queue if req.timestamp >= cutoff_time]
            heapq.heapify(self._queue)  # Re-heapify after filtering
            
            removed_count = original_count - len(self._queue)
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} expired requests")
                self._not_full.notify_all()
                
            return removed_count 