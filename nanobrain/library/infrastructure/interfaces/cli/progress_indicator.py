"""
Progress Indicator

Progress tracking and status indication for CLI interfaces.

This module provides:
- Progress bars and indicators
- Status updates and notifications
- Long-running operation tracking
- Customizable progress display
"""

import sys
import os
import time
import asyncio
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

# Add src to path for core imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'src'))

from nanobrain.core.logging_system import get_logger


class ProgressStyle(Enum):
    """Progress indicator styles."""
    BAR = "bar"
    SPINNER = "spinner"
    DOTS = "dots"
    PERCENTAGE = "percentage"
    COUNTER = "counter"


@dataclass
class ProgressConfig:
    """Configuration for progress indicators."""
    
    # Display settings
    style: ProgressStyle = ProgressStyle.BAR
    width: int = 50
    show_percentage: bool = True
    show_eta: bool = True
    show_rate: bool = True
    
    # Bar settings
    fill_char: str = "█"
    empty_char: str = "░"
    bar_format: str = "{prefix} |{bar}| {percentage}% {suffix}"
    
    # Spinner settings
    spinner_chars: list = field(default_factory=lambda: ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    spinner_interval: float = 0.1
    
    # Update settings
    update_interval: float = 0.1
    min_update_interval: float = 0.05
    
    # Colors
    enable_colors: bool = True
    complete_color: str = "\033[92m"  # Green
    progress_color: str = "\033[94m"  # Blue
    reset_color: str = "\033[0m"


class ProgressIndicator:
    """
    Progress indicator for CLI interfaces.
    
    This class provides:
    - Multiple progress display styles
    - Real-time progress updates
    - ETA and rate calculations
    - Customizable appearance
    - Thread-safe operation
    """
    
    def __init__(self, config: Optional[ProgressConfig] = None):
        """
        Initialize the progress indicator.
        
        Args:
            config: Progress indicator configuration
        """
        self.config = config or ProgressConfig()
        self.logger = get_logger("cli.progress_indicator")
        
        # State
        self.active_indicators: Dict[str, Dict[str, Any]] = {}
        self.update_thread: Optional[threading.Thread] = None
        self.should_stop = False
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        self.logger.info("Progress indicator initialized")
    
    async def initialize(self) -> None:
        """Initialize the progress indicator."""
        # Start update thread
        self.should_stop = False
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self.logger.info("Progress indicator initialization complete")
    
    async def cleanup(self) -> None:
        """Cleanup progress indicator resources."""
        # Stop update thread
        self.should_stop = True
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        
        # Clear all indicators
        with self._lock:
            self.active_indicators.clear()
        
        self.logger.info("Progress indicator cleanup complete")
    
    def create_progress(self, 
                       name: str,
                       total: Optional[int] = None,
                       description: str = "",
                       style: Optional[ProgressStyle] = None) -> str:
        """
        Create a new progress indicator.
        
        Args:
            name: Unique name for the progress indicator
            total: Total number of items (None for indeterminate)
            description: Description text
            style: Progress style override
            
        Returns:
            Progress indicator ID
        """
        with self._lock:
            progress_id = f"{name}_{int(time.time() * 1000)}"
            
            self.active_indicators[progress_id] = {
                'name': name,
                'total': total,
                'current': 0,
                'description': description,
                'style': style or self.config.style,
                'start_time': time.time(),
                'last_update': 0,
                'rate': 0,
                'eta': None,
                'completed': False,
                'spinner_index': 0
            }
        
        self.logger.debug(f"Created progress indicator: {progress_id}")
        return progress_id
    
    def update_progress(self, 
                       progress_id: str,
                       current: Optional[int] = None,
                       increment: int = 1,
                       description: Optional[str] = None) -> None:
        """
        Update progress indicator.
        
        Args:
            progress_id: Progress indicator ID
            current: Current progress value
            increment: Amount to increment (if current not provided)
            description: Updated description
        """
        with self._lock:
            if progress_id not in self.active_indicators:
                return
            
            indicator = self.active_indicators[progress_id]
            
            # Update current value
            if current is not None:
                indicator['current'] = current
            else:
                indicator['current'] += increment
            
            # Update description
            if description is not None:
                indicator['description'] = description
            
            # Calculate rate and ETA
            now = time.time()
            elapsed = now - indicator['start_time']
            
            if elapsed > 0:
                indicator['rate'] = indicator['current'] / elapsed
                
                if indicator['total'] and indicator['rate'] > 0:
                    remaining = indicator['total'] - indicator['current']
                    indicator['eta'] = remaining / indicator['rate']
            
            indicator['last_update'] = now
    
    def complete_progress(self, progress_id: str, message: str = "Complete") -> None:
        """
        Mark progress as complete.
        
        Args:
            progress_id: Progress indicator ID
            message: Completion message
        """
        with self._lock:
            if progress_id not in self.active_indicators:
                return
            
            indicator = self.active_indicators[progress_id]
            indicator['completed'] = True
            indicator['description'] = message
            
            if indicator['total']:
                indicator['current'] = indicator['total']
        
        # Display final state
        self._display_progress(progress_id)
        
        # Remove after a short delay
        threading.Timer(1.0, lambda: self._remove_progress(progress_id)).start()
    
    def remove_progress(self, progress_id: str) -> None:
        """
        Remove progress indicator.
        
        Args:
            progress_id: Progress indicator ID
        """
        self._remove_progress(progress_id)
    
    def _remove_progress(self, progress_id: str) -> None:
        """Internal method to remove progress indicator."""
        with self._lock:
            if progress_id in self.active_indicators:
                del self.active_indicators[progress_id]
                # Clear the line
                print(f"\r{' ' * 80}\r", end='', flush=True)
    
    def _update_loop(self) -> None:
        """Update loop running in separate thread."""
        while not self.should_stop:
            try:
                with self._lock:
                    for progress_id in list(self.active_indicators.keys()):
                        if not self.should_stop:
                            self._display_progress(progress_id)
                
                time.sleep(self.config.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in progress update loop: {e}")
    
    def _display_progress(self, progress_id: str) -> None:
        """Display progress indicator."""
        if progress_id not in self.active_indicators:
            return
        
        indicator = self.active_indicators[progress_id]
        style = indicator['style']
        
        if style == ProgressStyle.BAR:
            display = self._render_bar(indicator)
        elif style == ProgressStyle.SPINNER:
            display = self._render_spinner(indicator)
        elif style == ProgressStyle.DOTS:
            display = self._render_dots(indicator)
        elif style == ProgressStyle.PERCENTAGE:
            display = self._render_percentage(indicator)
        elif style == ProgressStyle.COUNTER:
            display = self._render_counter(indicator)
        else:
            display = self._render_bar(indicator)
        
        # Display with carriage return for updating in place
        print(f"\r{display}", end='', flush=True)
        
        # If completed, add newline
        if indicator['completed']:
            print()
    
    def _render_bar(self, indicator: Dict[str, Any]) -> str:
        """Render progress bar."""
        total = indicator['total']
        current = indicator['current']
        description = indicator['description']
        
        if total and total > 0:
            # Determinate progress
            percentage = min(100, (current / total) * 100)
            filled_width = int((current / total) * self.config.width)
            empty_width = self.config.width - filled_width
            
            bar = (self.config.fill_char * filled_width + 
                   self.config.empty_char * empty_width)
            
            # Build suffix with additional info
            suffix_parts = []
            if self.config.show_percentage:
                suffix_parts.append(f"{percentage:.1f}%")
            
            if self.config.show_rate and indicator['rate'] > 0:
                suffix_parts.append(f"{indicator['rate']:.1f}/s")
            
            if self.config.show_eta and indicator['eta']:
                eta_str = self._format_time(indicator['eta'])
                suffix_parts.append(f"ETA: {eta_str}")
            
            suffix = " | ".join(suffix_parts)
            
        else:
            # Indeterminate progress
            percentage = 0
            bar = self.config.fill_char * min(current % self.config.width, self.config.width)
            bar += self.config.empty_char * (self.config.width - len(bar))
            suffix = f"{current} items"
        
        # Apply colors
        if self.config.enable_colors:
            if indicator['completed']:
                bar = f"{self.config.complete_color}{bar}{self.config.reset_color}"
            else:
                bar = f"{self.config.progress_color}{bar}{self.config.reset_color}"
        
        return self.config.bar_format.format(
            prefix=description,
            bar=bar,
            percentage=percentage,
            suffix=suffix
        )
    
    def _render_spinner(self, indicator: Dict[str, Any]) -> str:
        """Render spinner indicator."""
        spinner_char = self.config.spinner_chars[indicator['spinner_index']]
        indicator['spinner_index'] = (indicator['spinner_index'] + 1) % len(self.config.spinner_chars)
        
        description = indicator['description']
        current = indicator['current']
        
        if indicator['total']:
            percentage = (current / indicator['total']) * 100
            return f"{spinner_char} {description} ({current}/{indicator['total']}) {percentage:.1f}%"
        else:
            return f"{spinner_char} {description} ({current})"
    
    def _render_dots(self, indicator: Dict[str, Any]) -> str:
        """Render dots indicator."""
        dots_count = (indicator['current'] % 4) + 1
        dots = "." * dots_count + " " * (4 - dots_count)
        
        description = indicator['description']
        current = indicator['current']
        
        if indicator['total']:
            return f"{description}{dots} ({current}/{indicator['total']})"
        else:
            return f"{description}{dots} ({current})"
    
    def _render_percentage(self, indicator: Dict[str, Any]) -> str:
        """Render percentage indicator."""
        current = indicator['current']
        total = indicator['total']
        description = indicator['description']
        
        if total and total > 0:
            percentage = (current / total) * 100
            return f"{description}: {percentage:.1f}% ({current}/{total})"
        else:
            return f"{description}: {current} items"
    
    def _render_counter(self, indicator: Dict[str, Any]) -> str:
        """Render counter indicator."""
        current = indicator['current']
        total = indicator['total']
        description = indicator['description']
        
        if total:
            return f"{description}: {current}/{total}"
        else:
            return f"{description}: {current}"
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:.0f}m {secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    # Context manager support
    
    def progress_context(self, 
                        name: str,
                        total: Optional[int] = None,
                        description: str = "",
                        style: Optional[ProgressStyle] = None):
        """
        Create a progress context manager.
        
        Args:
            name: Progress name
            total: Total items
            description: Description
            style: Progress style
            
        Returns:
            Progress context manager
        """
        return ProgressContext(self, name, total, description, style)


class ProgressContext:
    """Context manager for progress indicators."""
    
    def __init__(self, 
                 indicator: ProgressIndicator,
                 name: str,
                 total: Optional[int] = None,
                 description: str = "",
                 style: Optional[ProgressStyle] = None):
        """
        Initialize progress context.
        
        Args:
            indicator: Progress indicator instance
            name: Progress name
            total: Total items
            description: Description
            style: Progress style
        """
        self.indicator = indicator
        self.name = name
        self.total = total
        self.description = description
        self.style = style
        self.progress_id = None
    
    def __enter__(self):
        """Enter context manager."""
        self.progress_id = self.indicator.create_progress(
            self.name, self.total, self.description, self.style
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.progress_id:
            if exc_type is None:
                self.indicator.complete_progress(self.progress_id, "Complete")
            else:
                self.indicator.complete_progress(self.progress_id, "Failed")
    
    def update(self, 
               current: Optional[int] = None,
               increment: int = 1,
               description: Optional[str] = None):
        """Update progress."""
        if self.progress_id:
            self.indicator.update_progress(self.progress_id, current, increment, description)


# Convenience functions

def create_progress_indicator(config: Optional[ProgressConfig] = None) -> ProgressIndicator:
    """
    Create a progress indicator with optional configuration.
    
    Args:
        config: Progress indicator configuration
        
    Returns:
        Progress indicator instance
    """
    return ProgressIndicator(config)


async def show_progress(name: str, 
                       total: Optional[int] = None,
                       description: str = "",
                       style: ProgressStyle = ProgressStyle.BAR) -> str:
    """
    Simple progress indicator function.
    
    Args:
        name: Progress name
        total: Total items
        description: Description
        style: Progress style
        
    Returns:
        Progress ID
    """
    indicator = ProgressIndicator()
    await indicator.initialize()
    return indicator.create_progress(name, total, description, style) 