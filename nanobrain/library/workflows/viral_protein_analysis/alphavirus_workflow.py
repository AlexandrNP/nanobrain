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
from nanobrain.core.data_unit import create_data_unit


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
        
        # Initialize workflow input/output data units
        if hasattr(config, 'input_data_units') and config.input_data_units:
            input_configs = config.input_data_units
            if 'workflow_input' in input_configs:
                from nanobrain.core.data_unit import DataUnitConfig, create_data_unit
                data_unit_config = DataUnitConfig(**input_configs['workflow_input'])
                class_path = data_unit_config.class_path
                self.input_data_unit = create_data_unit(class_path, data_unit_config)
        
        if hasattr(config, 'output_data_units') and config.output_data_units:
            output_configs = config.output_data_units
            if 'workflow_output' in output_configs:
                from nanobrain.core.data_unit import DataUnitConfig, create_data_unit
                data_unit_config = DataUnitConfig(**output_configs['workflow_output'])
                class_path = data_unit_config.class_path
                self.output_data_unit = create_data_unit(class_path, data_unit_config)
        
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
        # No manual creation required
        self.links = {}
        if hasattr(config, 'links') and isinstance(config.links, dict):
            for link_id, link_config in config.links.items():
                link = self._create_link_from_config(link_config)
                self.links[link_id] = link
        
        # NO triggers at workflow level - all are step-level
        # NO tools at workflow level - all are step-level  
        # NO business logic - pure event routing

    def _resolve_step_from_config(self, step_config: Dict[str, Any]):
        """Resolve step from configuration using enhanced patterns"""
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

    def _create_link_from_config(self, link_config: Dict[str, Any]):
        """Create link from configuration"""
        from nanobrain.core.link import create_link, LinkConfig
        
        try:
            # Create LinkConfig object  
            config = LinkConfig(**link_config)
            
            # Resolve source and target data units
            source_ref = link_config.get('source')
            target_ref = link_config.get('target')
            
            source_data_unit = self._resolve_data_unit_reference(source_ref)
            target_data_unit = self._resolve_data_unit_reference(target_ref)
            
            return create_link(config, source=source_data_unit, target=target_data_unit)
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"Failed to create link: {e}")
            raise

    def _resolve_data_unit_reference(self, ref: str):
        """Resolve data unit reference to actual data unit"""
        # This method resolves step.data_unit_name references
        if '.' in ref:
            step_id, data_unit_name = ref.split('.', 1)
            step = self.steps.get(step_id)
            if step and hasattr(step, data_unit_name):
                return getattr(step, data_unit_name)
        
        # Try workflow-level data units
        if ref == 'workflow_input':
            return self.input_data_unit
        elif ref == 'workflow_output':
            return self.output_data_unit
        
        raise ValueError(f"Could not resolve data unit reference: {ref}")

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
    
