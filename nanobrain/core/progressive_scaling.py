"""
Progressive Scaling Mixin for NanoBrain Framework

Provides progressive scaling capabilities for tools that need to handle
varying data sizes and computational loads.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from nanobrain.core.logging_system import get_logger


class ProgressiveScalingMixin:
    """
    Mixin for progressive scaling behavior.
    
    Progressive scaling allows tools to:
    1. Start with small data sets for validation
    2. Gradually increase scale based on success
    3. Handle large datasets efficiently
    4. Provide early feedback on configuration issues
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize progressive scaling attributes
        self.current_scale_level = getattr(self.config, 'initial_scale_level', 1)
        self.scale_config = getattr(self.config, 'progressive_scaling', {})
        self.scaling_logger = get_logger(f"{self.__class__.__name__}.scaling")
    
    async def execute_with_progressive_scaling(self, scale_level: int = None, 
                                             auto_advance: bool = False) -> Any:
        """
        Execute tool with progressive scaling.
        
        Args:
            scale_level: Specific scale level to use (None for current)
            auto_advance: Automatically advance to next level on success
            
        Returns:
            Execution result
        """
        if scale_level is None:
            scale_level = self.current_scale_level
        
        if scale_level not in self.scale_config:
            available_levels = list(self.scale_config.keys())
            raise ValueError(
                f"Scale level {scale_level} not configured. "
                f"Available levels: {available_levels}"
            )
        
        scale_config = self.scale_config[scale_level]
        description = scale_config.get('description', f'Scale level {scale_level}')
        
        self.scaling_logger.info(f"Executing at scale level {scale_level}: {description}")
        self.scaling_logger.debug(f"Scale configuration: {scale_config}")
        
        try:
            result = await self._execute_at_scale(scale_config)
            
            # Track successful execution
            self.current_scale_level = scale_level
            self.scaling_logger.info(f"✅ Scale level {scale_level} completed successfully")
            
            # Auto-advance to next level if requested and available
            if auto_advance:
                next_level = scale_level + 1
                if next_level in self.scale_config:
                    self.scaling_logger.info(f"🔄 Auto-advancing to scale level {next_level}")
                    return await self.execute_with_progressive_scaling(next_level, auto_advance)
                else:
                    self.scaling_logger.info(f"🏁 Maximum scale level {scale_level} reached")
            
            return result
            
        except Exception as e:
            self.scaling_logger.error(f"❌ Scale level {scale_level} failed: {e}")
            raise
    
    async def validate_all_scale_levels(self) -> Dict[int, bool]:
        """
        Validate all configured scale levels.
        
        Returns:
            Dictionary mapping scale levels to success status
        """
        results = {}
        
        for scale_level in sorted(self.scale_config.keys()):
            try:
                self.scaling_logger.info(f"Validating scale level {scale_level}")
                await self.execute_with_progressive_scaling(scale_level)
                results[scale_level] = True
                self.scaling_logger.info(f"✅ Scale level {scale_level} validation passed")
            except Exception as e:
                results[scale_level] = False
                self.scaling_logger.error(f"❌ Scale level {scale_level} validation failed: {e}")
        
        return results
    
    def get_scale_info(self, scale_level: int = None) -> Dict[str, Any]:
        """Get information about a scale level."""
        if scale_level is None:
            scale_level = self.current_scale_level
        
        if scale_level not in self.scale_config:
            return {"error": f"Scale level {scale_level} not configured"}
        
        scale_config = self.scale_config[scale_level].copy()
        scale_config.update({
            "level": scale_level,
            "is_current": scale_level == self.current_scale_level,
            "available_levels": list(self.scale_config.keys())
        })
        
        return scale_config
    
    def get_all_scale_levels(self) -> Dict[int, Dict[str, Any]]:
        """Get information about all configured scale levels."""
        return {
            level: self.get_scale_info(level) 
            for level in self.scale_config.keys()
        }
    
    def set_scale_level(self, scale_level: int) -> None:
        """Set the current scale level without execution."""
        if scale_level not in self.scale_config:
            available_levels = list(self.scale_config.keys())
            raise ValueError(
                f"Scale level {scale_level} not configured. "
                f"Available levels: {available_levels}"
            )
        
        self.current_scale_level = scale_level
        self.scaling_logger.info(f"Scale level set to {scale_level}")
    
    def get_recommended_scale_progression(self) -> List[int]:
        """Get recommended scale level progression."""
        return sorted(self.scale_config.keys())
    
    @abstractmethod
    async def _execute_at_scale(self, scale_config: Dict[str, Any]) -> Any:
        """
        Execute tool with specific scale configuration.
        
        This method must be implemented by classes using the mixin.
        
        Args:
            scale_config: Configuration for the specific scale level
            
        Returns:
            Execution result
        """
        pass


class ScalingValidationMixin:
    """Additional mixin for scale configuration validation."""
    
    def validate_scale_configuration(self) -> List[str]:
        """
        Validate progressive scaling configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        scale_config = getattr(self.config, 'progressive_scaling', {})
        
        if not scale_config:
            return errors  # No scaling configured, which is valid
        
        # Check that scale levels are positive integers
        for level in scale_config.keys():
            if not isinstance(level, int) or level < 1:
                errors.append(f"Scale level must be positive integer, got: {level}")
        
        # Check that initial scale level exists
        initial_level = getattr(self.config, 'initial_scale_level', 1)
        if initial_level not in scale_config:
            errors.append(f"Initial scale level {initial_level} not in configuration")
        
        # Validate scale level configurations
        for level, config in scale_config.items():
            if not isinstance(config, dict):
                errors.append(f"Scale level {level} configuration must be dict")
                continue
            
            # Tool-specific validation can be added by overriding this method
            tool_errors = self._validate_scale_level_config(level, config)
            errors.extend(tool_errors)
        
        return errors
    
    def _validate_scale_level_config(self, level: int, config: Dict[str, Any]) -> List[str]:
        """
        Validate individual scale level configuration.
        
        Override in tool classes for tool-specific validation.
        
        Args:
            level: Scale level number
            config: Scale level configuration
            
        Returns:
            List of validation errors
        """
        return []  # Base implementation accepts all configurations 