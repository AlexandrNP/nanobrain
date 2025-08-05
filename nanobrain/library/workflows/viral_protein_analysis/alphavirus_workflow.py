"""
Event-Driven Alphavirus Workflow - Enhanced from_config Compliance

Pure event routing workflow with minimal logic.
All business logic delegated to steps with event-driven execution.
Enhanced for framework compliance with Dict-based configuration format.
"""

import asyncio
import time
import importlib
from typing import Dict, Any, List, Optional
from pathlib import Path
import os

from nanobrain.core.workflow import Workflow, WorkflowConfig
from nanobrain.core.logging_system import get_logger


class AlphavirusWorkflow(Workflow):
    """
    Event-Driven Viral Protein Analysis Workflow
    
    Pure event routing workflow with minimal logic.
    All execution happens via event-driven data flow.
    
    Enhanced for framework compliance:
    - Dict-based steps configuration
    - Enhanced from_config patterns
    - Configuration-driven component creation
    - Step-level data units and triggers
    """
    
    def _init_from_config(self, config: WorkflowConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize event-driven workflow as pure event router"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Create ONLY workflow-level input/output data units
        self.input_data_unit = None
        self.output_data_unit = None
        
        # Initialize workflow input/output data units using proper from_config pattern
        if hasattr(config, 'input_data_units') and config.input_data_units:
            input_configs = config.input_data_units
            if 'workflow_input' in input_configs:
                unit_config = input_configs['workflow_input']
                # Use proper from_config pattern - import class and call its from_config method
                class_path = unit_config.get('class', 'nanobrain.core.data_unit.DataUnitMemory')
                module_path, class_name = class_path.rsplit('.', 1)
                import importlib
                module = importlib.import_module(module_path)
                data_unit_class = getattr(module, class_name)
                
                # Ensure config has class field for proper data unit creation
                enhanced_config = unit_config.copy()
                if 'class' not in enhanced_config:
                    enhanced_config['class'] = class_path
                
                self.input_data_unit = data_unit_class.from_config(enhanced_config)
        
        if hasattr(config, 'output_data_units') and config.output_data_units:
            output_configs = config.output_data_units
            if 'workflow_output' in output_configs:
                unit_config = output_configs['workflow_output']
                # Use proper from_config pattern - import class and call its from_config method
                class_path = unit_config.get('class', 'nanobrain.core.data_unit.DataUnitFile')
                module_path, class_name = class_path.rsplit('.', 1)
                import importlib
                module = importlib.import_module(module_path)
                data_unit_class = getattr(module, class_name)
                
                # Ensure config has class field for proper data unit creation
                enhanced_config = unit_config.copy()
                if 'class' not in enhanced_config:
                    enhanced_config['class'] = class_path
                
                self.output_data_unit = data_unit_class.from_config(enhanced_config)
        
        # Enhanced framework handles automatic component instantiation
        # No programmatic component creation needed
        
        # Steps are automatically instantiated from Dict-based configuration
        self.steps = {}
        if hasattr(config, 'steps') and isinstance(config.steps, dict):
            for step_id, step_config in config.steps.items():
                # Enhanced ConfigBase automatically handles class+config instantiation
                step_instance = self._resolve_step_from_config(step_config)
                self.steps[step_id] = step_instance
        
        # Links and triggers automatically resolved by enhanced framework
        # No manual creation required - framework handles via class+config pattern
        
        # NO triggers at workflow level - all are step-level
        # NO tools at workflow level - all are step-level  
        # NO business logic - pure event routing

    def _resolve_step_from_config(self, step_config):
        """Resolve step from configuration using enhanced patterns"""
        # Handle case where enhanced framework has already resolved steps to instances
        if hasattr(step_config, '__class__') and hasattr(step_config, 'process'):
            # step_config is already a step instance - return it directly
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"✅ Using pre-resolved step: {step_config.__class__.__name__}")
            return step_config
        
        # Handle case where step_config is still a dictionary configuration
        if isinstance(step_config, dict):
            step_class = step_config.get('class')
            config_path = step_config.get('config')
            
            if not step_class or not config_path:
                raise ValueError(f"Step configuration must include 'class' and 'config' fields")
            
            # Enhanced from_config handles automatic instantiation
            module_path, class_name = step_class.rsplit('.', 1)
            module = importlib.import_module(module_path)
            step_cls = getattr(module, class_name)
            
            try:
                step_instance = step_cls.from_config(
                    config_path, 
                    workflow_directory=getattr(self.config, 'workflow_directory', '')
                )
                
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.info(f"✅ Created step: {step_class}")
                return step_instance
                
            except Exception as e:
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.error(f"❌ Failed to create step {step_class}: {e}")
                raise
        
        # Invalid step_config type
        raise ValueError(f"Invalid step configuration type: {type(step_config)}. Expected dict or step instance.")



    async def initialize(self) -> None:
        """Initialize workflow and all components"""
        await super().initialize()
        
        # Initialize workflow data units
        if self.input_data_unit:
            await self.input_data_unit.initialize()
        if self.output_data_unit:
            await self.output_data_unit.initialize()
        
        # Initialize all steps
        for step_id, step in self.steps.items():
            await step.initialize()
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"Initialized step: {step_id}")
        
        # Initialize all links (sets up automatic transfer triggers)
        for link_id, link in self.links.items():
            await link.initialize()
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"Initialized link: {link_id}")

    async def process(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process workflow via pure event-driven flow (Enhanced Implementation)
        
        Args:
            input_data: Input data for the workflow
            
        Returns:
            Dict containing workflow results via event-driven completion
        """
        try:
            if not self.input_data_unit:
                raise ValueError("Workflow input data unit not configured")
            
            # Set input data unit (triggers the entire workflow via events)
            await self.input_data_unit.set(input_data or {})
            
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"🚀 Started enhanced event-driven workflow execution")
            
            # Wait for workflow completion (output data unit updated via events)
            timeout = 300  # 5 minutes timeout
            poll_interval = 0.1  # 100ms polling
            elapsed = 0
            
            while elapsed < timeout:
                if self.output_data_unit:
                    output_data = await self.output_data_unit.get()
                    if output_data is not None:
                        if hasattr(self, 'nb_logger') and self.nb_logger:
                            self.nb_logger.info(f"✅ Enhanced event-driven workflow completed successfully")
                        return output_data
                
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
            
            # Timeout handling
            raise TimeoutError(f"Workflow execution timed out after {timeout} seconds")
            
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ Enhanced event-driven workflow execution failed: {e}")
            raise
    
