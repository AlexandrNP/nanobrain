"""
Link System for NanoBrain Framework

Provides dataflow abstractions for connecting Steps together.
Links define how information flows between system components.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

# Import logging system
from .logging_system import get_logger, get_system_log_manager

logger = logging.getLogger(__name__)


class LinkType(Enum):
    """Types of links."""
    DIRECT = "direct"
    FILE = "file"
    QUEUE = "queue"
    TRANSFORM = "transform"
    CONDITIONAL = "conditional"


class LinkConfig(BaseModel):
    """Configuration for links."""
    link_type: LinkType = LinkType.DIRECT
    buffer_size: int = Field(default=100, ge=1)
    transform_function: Optional[str] = None
    condition: Optional[str] = None
    file_path: Optional[str] = None
    
    model_config = ConfigDict(use_enum_values=True)


class LinkBase(ABC):
    """
    Base class for links that connect data flow between Steps.
    
    Biological analogy: Neural pathways connecting brain regions.
    Justification: Like how neural pathways carry information between
    different brain regions, links carry data between different steps.
    """
    
    def __init__(self, source: Any, target: Any, config: Optional[LinkConfig] = None, **kwargs):
        self.config = config or LinkConfig()
        self.source = source
        self.target = target
        self.name = kwargs.get('name', f"{getattr(source, 'name', 'unknown')}->{getattr(target, 'name', 'unknown')}")
        self._is_active = False
        self._transfer_count = 0
        self._error_count = 0
        self._creation_time = time.time()
        self._last_transfer_time = None
        self._total_transfer_time = 0.0
        
        # Initialize centralized logging system
        self.enable_logging = kwargs.get('enable_logging', True)
        if self.enable_logging:
            # Use centralized logging system
            self.nb_logger = get_logger(self.name, category="links", debug_mode=kwargs.get('debug_mode', False))
            
            # Register with system log manager
            system_manager = get_system_log_manager()
            system_manager.register_component("links", self.name, self, {
                "link_type": self.config.link_type.value if hasattr(self.config.link_type, 'value') else str(self.config.link_type),
                "source": getattr(source, 'name', str(source)),
                "target": getattr(target, 'name', str(target)),
                "buffer_size": self.config.buffer_size,
                "enable_logging": True
            })
        else:
            self.nb_logger = None
        
    @abstractmethod
    async def transfer(self, data: Any) -> None:
        """Transfer data from source to target."""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the link."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the link."""
        pass
    
    def _get_internal_state(self) -> Dict[str, Any]:
        """Get comprehensive internal state for logging."""
        uptime = time.time() - self._creation_time
        avg_transfer_time = self._total_transfer_time / max(self._transfer_count, 1)
        
        return {
            "is_active": self._is_active,
            "transfer_count": self._transfer_count,
            "error_count": self._error_count,
            "success_rate": self._transfer_count / max(self._transfer_count + self._error_count, 1),
            "uptime_seconds": uptime,
            "last_transfer_time": self._last_transfer_time,
            "avg_transfer_time_ms": avg_transfer_time * 1000 if self._transfer_count > 0 else 0,
            "link_type": self.config.link_type.value if hasattr(self.config.link_type, 'value') else str(self.config.link_type),
            "source": getattr(self.source, 'name', str(self.source)),
            "target": getattr(self.target, 'name', str(self.target))
        }
    
    @property
    def is_active(self) -> bool:
        """Check if link is active."""
        return self._is_active
    
    @property
    def transfer_count(self) -> int:
        """Get number of successful transfers."""
        return self._transfer_count
    
    @property
    def error_count(self) -> int:
        """Get number of transfer errors."""
        return self._error_count
    
    async def _record_transfer(self, success: bool = True, duration_ms: float = None, data_info: Dict[str, Any] = None) -> None:
        """Record transfer statistics with comprehensive logging."""
        transfer_time = time.time()
        
        if success:
            self._transfer_count += 1
            self._last_transfer_time = transfer_time
            if duration_ms:
                self._total_transfer_time += duration_ms / 1000.0
        else:
            self._error_count += 1
        
        # Log transfer event
        if self.enable_logging and self.nb_logger:
            self.nb_logger.log_data_transfer(
                source=getattr(self.source, 'name', str(self.source)),
                destination=getattr(self.target, 'name', str(self.target)),
                data_type=data_info.get('data_type', 'unknown') if data_info else 'unknown',
                size_bytes=data_info.get('size_bytes', 0) if data_info else 0
            )
            
            self.nb_logger.info(f"Link transfer {'succeeded' if success else 'failed'}: {self.name}",
                               operation="transfer",
                               success=success,
                               duration_ms=duration_ms,
                               data_info=data_info or {},
                               internal_state=self._get_internal_state())


class DirectLink(LinkBase):
    """
    Direct link that immediately transfers data from source to target.
    """
    
    def __init__(self, source: Any, target: Any, config: Optional[LinkConfig] = None, **kwargs):
        config = config or LinkConfig(link_type=LinkType.DIRECT)
        super().__init__(source, target, config, **kwargs)
        
    async def start(self) -> None:
        """Start the direct link."""
        self._is_active = True
        logger.debug(f"DirectLink {self.name} started")
        
        # Log start event
        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"Link started: {self.name}",
                               operation="start",
                               internal_state=self._get_internal_state())
    
    async def stop(self) -> None:
        """Stop the direct link."""
        self._is_active = False
        logger.debug(f"DirectLink {self.name} stopped")
        
        # Log stop event with final statistics
        if self.enable_logging and self.nb_logger:
            uptime = time.time() - self._creation_time
            self.nb_logger.info(f"Link stopped: {self.name}",
                               operation="stop",
                               uptime_seconds=uptime,
                               final_stats={
                                   "total_transfers": self._transfer_count,
                                   "total_errors": self._error_count,
                                   "success_rate": self._transfer_count / max(self._transfer_count + self._error_count, 1)
                               },
                               internal_state=self._get_internal_state())
    
    async def transfer(self, data: Any) -> None:
        """Transfer data directly to target."""
        if not self._is_active:
            logger.warning(f"DirectLink {self.name} not active")
            return
        
        start_time = time.time()
        transfer_method = None
        
        try:
            # Prepare data info for logging
            data_info = {
                "data_type": type(data).__name__ if data is not None else "None",
                "size_bytes": len(str(data)) if data is not None else 0,
                "data_value": data if self.enable_logging else None
            }
            
            # Handle different target types
            if hasattr(self.target, 'set') and callable(getattr(self.target, 'set')):
                # Target is a DataUnit - call set method directly
                transfer_method = "DataUnit.set"
                await self.target.set(data)
                logger.debug(f"DirectLink {self.name} transferred data to DataUnit")
            elif hasattr(self.target, 'input_data_units') and self.target.input_data_units:
                # Target is a Step with input data units
                transfer_method = "Step.input_data_units"
                for input_unit in self.target.input_data_units.values():
                    await input_unit.set(data)
                logger.debug(f"DirectLink {self.name} transferred data to Step input units")
            elif hasattr(self.target, 'execute') and callable(getattr(self.target, 'execute')):
                # Target is a Step - trigger execution
                transfer_method = "Step.execute"
                await self.target.execute(data)
                logger.debug(f"DirectLink {self.name} executed target Step")
            elif hasattr(self.target, 'set_input'):
                # Direct method call
                transfer_method = "set_input"
                await self.target.set_input(data)
                logger.debug(f"DirectLink {self.name} called set_input on target")
            else:
                logger.warning(f"Target {getattr(self.target, 'name', str(self.target))} has no compatible input mechanism")
                
                # Log failed transfer attempt
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.warning(f"Transfer failed - no compatible target mechanism: {self.name}",
                                         operation="transfer_failed",
                                         reason="no_compatible_mechanism",
                                         target_type=type(self.target).__name__,
                                         data_info=data_info)
                return
            
            # Calculate duration and record success
            duration_ms = (time.time() - start_time) * 1000
            data_info["transfer_method"] = transfer_method
            await self._record_transfer(True, duration_ms, data_info)
            logger.debug(f"DirectLink {self.name} transferred data successfully")
            
        except Exception as e:
            # Calculate duration and record failure
            duration_ms = (time.time() - start_time) * 1000
            data_info = {
                "data_type": type(data).__name__ if data is not None else "None",
                "size_bytes": len(str(data)) if data is not None else 0,
                "transfer_method": transfer_method,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            await self._record_transfer(False, duration_ms, data_info)
            logger.error(f"DirectLink {self.name} transfer failed: {e}")
            
            # Log detailed error information
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(f"Transfer error in link: {self.name}",
                                   operation="transfer_error",
                                   error=str(e),
                                   error_type=type(e).__name__,
                                   duration_ms=duration_ms,
                                   data_info=data_info,
                                   internal_state=self._get_internal_state())
            raise


class QueueLink(LinkBase):
    """
    Queue-based link that buffers data between source and target.
    """
    
    def __init__(self, source: Any, target: Any, config: Optional[LinkConfig] = None, **kwargs):
        config = config or LinkConfig(link_type=LinkType.QUEUE)
        super().__init__(source, target, config, **kwargs)
        self._queue: Optional[asyncio.Queue] = None
        self._consumer_task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """Start the queue link."""
        if self._is_active:
            return
            
        self._queue = asyncio.Queue(maxsize=self.config.buffer_size)
        self._consumer_task = asyncio.create_task(self._consume_queue())
        self._is_active = True
        logger.debug(f"QueueLink {self.name} started with buffer size {self.config.buffer_size}")
    
    async def stop(self) -> None:
        """Stop the queue link."""
        self._is_active = False
        
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        
        self._queue = None
        logger.debug(f"QueueLink {self.name} stopped")
    
    async def transfer(self, data: Any) -> None:
        """Add data to queue for transfer."""
        if not self._is_active or not self._queue:
            logger.warning(f"QueueLink {self.name} not active")
            return
            
        try:
            await self._queue.put(data)
            logger.debug(f"QueueLink {self.name} queued data")
        except Exception as e:
            await self._record_transfer(False)
            logger.error(f"QueueLink {self.name} queue failed: {e}")
            raise
    
    async def _consume_queue(self) -> None:
        """Consume data from queue and transfer to target."""
        try:
            while self._is_active and self._queue:
                try:
                    # Wait for data with timeout
                    data = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                    
                    # Transfer to target
                    if hasattr(self.target, 'input_data_units') and self.target.input_data_units:
                        input_unit = self.target.input_data_units[0]
                        await input_unit.set(data)
                    elif hasattr(self.target, 'set_input'):
                        await self.target.set_input(data)
                    
                    await self._record_transfer(True)
                    
                except asyncio.TimeoutError:
                    # Continue loop on timeout
                    continue
                except Exception as e:
                    await self._record_transfer(False)
                    logger.error(f"QueueLink {self.name} consumer error: {e}")
                    
        except asyncio.CancelledError:
            logger.debug(f"QueueLink {self.name} consumer cancelled")


class TransformLink(LinkBase):
    """
    Link that transforms data before transferring to target.
    """
    
    def __init__(self, source: Any, target: Any, transform_func: Callable, 
                 config: Optional[LinkConfig] = None, **kwargs):
        config = config or LinkConfig(link_type=LinkType.TRANSFORM)
        super().__init__(source, target, config, **kwargs)
        self.transform_func = transform_func
        
    async def start(self) -> None:
        """Start the transform link."""
        self._is_active = True
        logger.debug(f"TransformLink {self.name} started")
    
    async def stop(self) -> None:
        """Stop the transform link."""
        self._is_active = False
        logger.debug(f"TransformLink {self.name} stopped")
    
    async def transfer(self, data: Any) -> None:
        """Transform data and transfer to target."""
        if not self._is_active:
            logger.warning(f"TransformLink {self.name} not active")
            return
            
        try:
            # Apply transformation
            if asyncio.iscoroutinefunction(self.transform_func):
                transformed_data = await self.transform_func(data)
            else:
                transformed_data = self.transform_func(data)
            
            # Transfer transformed data
            if hasattr(self.target, 'input_data_units') and self.target.input_data_units:
                input_unit = self.target.input_data_units[0]
                await input_unit.set(transformed_data)
            elif hasattr(self.target, 'set_input'):
                await self.target.set_input(transformed_data)
            
            await self._record_transfer(True)
            logger.debug(f"TransformLink {self.name} transformed and transferred data")
            
        except Exception as e:
            await self._record_transfer(False)
            logger.error(f"TransformLink {self.name} transform failed: {e}")
            raise


class ConditionalLink(LinkBase):
    """
    Link that transfers data only when condition is met.
    """
    
    def __init__(self, source: Any, target: Any, condition_func: Callable, 
                 config: Optional[LinkConfig] = None, **kwargs):
        config = config or LinkConfig(link_type=LinkType.CONDITIONAL)
        super().__init__(source, target, config, **kwargs)
        self.condition_func = condition_func
        
    async def start(self) -> None:
        """Start the conditional link."""
        self._is_active = True
        logger.debug(f"ConditionalLink {self.name} started")
    
    async def stop(self) -> None:
        """Stop the conditional link."""
        self._is_active = False
        logger.debug(f"ConditionalLink {self.name} stopped")
    
    async def transfer(self, data: Any) -> None:
        """Transfer data if condition is met."""
        if not self._is_active:
            logger.warning(f"ConditionalLink {self.name} not active")
            return
            
        try:
            # Check condition
            if asyncio.iscoroutinefunction(self.condition_func):
                should_transfer = await self.condition_func(data)
            else:
                should_transfer = self.condition_func(data)
            
            if should_transfer:
                # Transfer data
                if hasattr(self.target, 'input_data_units') and self.target.input_data_units:
                    input_unit = self.target.input_data_units[0]
                    await input_unit.set(data)
                elif hasattr(self.target, 'set_input'):
                    await self.target.set_input(data)
                
                await self._record_transfer(True)
                logger.debug(f"ConditionalLink {self.name} condition met, transferred data")
            else:
                logger.debug(f"ConditionalLink {self.name} condition not met, skipped transfer")
                
        except Exception as e:
            await self._record_transfer(False)
            logger.error(f"ConditionalLink {self.name} condition check failed: {e}")
            raise


class FileLink(LinkBase):
    """
    File-based link that transfers data through file system.
    """
    
    def __init__(self, source: Any, target: Any, file_path: str, 
                 config: Optional[LinkConfig] = None, **kwargs):
        config = config or LinkConfig(link_type=LinkType.FILE, file_path=file_path)
        super().__init__(source, target, config, **kwargs)
        self.file_path = file_path
        self._monitor_task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """Start the file link."""
        if self._is_active:
            return
            
        self._monitor_task = asyncio.create_task(self._monitor_file())
        self._is_active = True
        logger.debug(f"FileLink {self.name} started monitoring {self.file_path}")
    
    async def stop(self) -> None:
        """Stop the file link."""
        self._is_active = False
        
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.debug(f"FileLink {self.name} stopped")
    
    async def transfer(self, data: Any) -> None:
        """Write data to file for transfer."""
        if not self._is_active:
            logger.warning(f"FileLink {self.name} not active")
            return
            
        try:
            from pathlib import Path
            import json
            
            file_path = Path(self.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write data to file
            if isinstance(data, (dict, list)):
                content = json.dumps(data, indent=2)
            else:
                content = str(data)
            
            file_path.write_text(content)
            logger.debug(f"FileLink {self.name} wrote data to file")
            
        except Exception as e:
            await self._record_transfer(False)
            logger.error(f"FileLink {self.name} file write failed: {e}")
            raise
    
    async def _monitor_file(self) -> None:
        """Monitor file for changes and transfer to target."""
        from pathlib import Path
        import json
        
        file_path = Path(self.file_path)
        last_modified = 0.0
        
        try:
            while self._is_active:
                try:
                    if file_path.exists():
                        current_modified = file_path.stat().st_mtime
                        
                        if current_modified > last_modified:
                            last_modified = current_modified
                            
                            # Read and transfer data
                            content = file_path.read_text()
                            try:
                                data = json.loads(content)
                            except json.JSONDecodeError:
                                data = content
                            
                            # Transfer to target
                            if hasattr(self.target, 'input_data_units') and self.target.input_data_units:
                                input_unit = self.target.input_data_units[0]
                                await input_unit.set(data)
                            elif hasattr(self.target, 'set_input'):
                                await self.target.set_input(data)
                            
                            await self._record_transfer(True)
                            logger.debug(f"FileLink {self.name} detected file change and transferred")
                    
                    await asyncio.sleep(1.0)  # Check every second
                    
                except Exception as e:
                    await self._record_transfer(False)
                    logger.error(f"FileLink {self.name} monitor error: {e}")
                    await asyncio.sleep(5.0)  # Wait longer on error
                    
        except asyncio.CancelledError:
            logger.debug(f"FileLink {self.name} monitor cancelled")


def create_link(config: Union[Dict[str, Any], LinkConfig]) -> LinkBase:
    """
    Factory function to create links.
    
    Args:
        config: Link configuration (dict or LinkConfig)
        
    Returns:
        Configured link instance
    """
    if isinstance(config, dict):
        config = LinkConfig(**config)
    
    if config.link_type == LinkType.DIRECT:
        return DirectLink(config)
    elif config.link_type == LinkType.FILE:
        return FileLink(config)
    elif config.link_type == LinkType.QUEUE:
        return QueueLink(config)
    elif config.link_type == LinkType.TRANSFORM:
        return TransformLink(config)
    elif config.link_type == LinkType.CONDITIONAL:
        return ConditionalLink(config)
    else:
        raise ValueError(f"Unknown link type: {config.link_type}") 