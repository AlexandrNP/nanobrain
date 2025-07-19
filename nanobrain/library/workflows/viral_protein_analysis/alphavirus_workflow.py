"""
Event-Driven Alphavirus Workflow - Phase 1 Complete

Pure event routing workflow with minimal logic.
All business logic delegated to steps with event-driven execution.
Transformed according to EVENT_DRIVEN_ARCHITECTURE_COMPLIANCE_PLAN.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import os

from nanobrain.core.workflow import Workflow, WorkflowConfig
from nanobrain.core.logging_system import get_logger
from nanobrain.core.data_unit import create_data_unit
from nanobrain.core.config.component_factory import load_config_file, create_component


class AlphavirusWorkflow(Workflow):
    """
    Event-Driven Viral Protein Analysis Workflow
    
    Pure event routing workflow with minimal logic.
    All execution happens via event-driven data flow.
    
    PHASE 1: Core event system implementation
    - No business logic in workflow
    - Pure event routing
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
        
        # Load steps from configuration files (NO inline configuration)
        self.steps = {}
        if hasattr(config, 'steps') and config.steps:
            for step_config in config.steps:
                step_id = step_config['step_id']
                config_file = step_config.get('config_file')
                if config_file:
                    step = self._create_step_from_config_file(config_file)
                    self.steps[step_id] = step
        
        # Load links from configuration (NO business logic)
        self.links = {}
        if hasattr(config, 'links') and config.links:
            for link_config in config.links:
                link_id = link_config['link_id']
                link = self._create_link_from_config(link_config)
                self.links[link_id] = link
        
        # NO triggers at workflow level - all are step-level
        # NO tools at workflow level - all are step-level  
        # NO business logic - pure event routing
    
    def _create_step_from_config_file(self, config_file: str):
        """Create step from external configuration file with proper path resolution"""
        try:
            # Resolve config file path relative to workflow directory
            workflow_dir = getattr(self.config, 'workflow_directory', '')
            
            # Build full path
            if workflow_dir and not os.path.isabs(config_file):
                full_config_path = os.path.join(workflow_dir, config_file)
            else:
                full_config_path = config_file
            
            # Verify file exists
            if not os.path.exists(full_config_path):
                raise FileNotFoundError(f"Step configuration file not found: {full_config_path}")
            
            # Use existing load_config_file function for pure YAML loading
            step_config_data = load_config_file(full_config_path)
            step_class = step_config_data.get('class')
            
            if not step_class:
                raise ValueError(f"Step configuration must specify 'class' field: {full_config_path}")
            
            # Create StepConfig from loaded data
            from nanobrain.core.step import StepConfig
            step_config = StepConfig(**step_config_data)
            
            # Use pure from_config pattern with create_component function
            step = create_component(step_class, step_config)
            
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"✅ Created step from {full_config_path}")
            return step
            
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ Failed to create step from {config_file}: {e}")
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
    
    def _resolve_data_unit_reference(self, reference: str):
        """Resolve data unit reference (step.data_unit or workflow_input/workflow_output)"""
        if not reference:
            return None
        
        # Handle workflow-level data unit references
        if reference == 'workflow_input':
            return self.input_data_unit
        elif reference == 'workflow_output':
            return self.output_data_unit
            
        # Handle step-level data unit references (step.data_unit)
        if '.' in reference:
            step_id, data_unit_name = reference.split('.', 1)
            
            if step_id in self.steps:
                step = self.steps[step_id]
                
                # Check step input data units
                if hasattr(step, 'step_input_data_units') and data_unit_name in step.step_input_data_units:
                    return step.step_input_data_units[data_unit_name]
                
                # Check step output data units
                if hasattr(step, 'step_output_data_units') and data_unit_name in step.step_output_data_units:
                    return step.step_output_data_units[data_unit_name]
        
        return None
    
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
        Process workflow via pure event-driven flow (PHASE 1: Event-Driven Implementation)
        
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
                self.nb_logger.info(f"🚀 Started event-driven workflow execution")
            
            # Wait for workflow completion (output data unit updated via events)
            timeout = 300  # 5 minutes timeout
            poll_interval = 0.1  # 100ms polling
            elapsed = 0
            
            while elapsed < timeout:
                if self.output_data_unit:
                    output_data = await self.output_data_unit.get()
                    if output_data is not None:
                        if hasattr(self, 'nb_logger') and self.nb_logger:
                            self.nb_logger.info(f"✅ Event-driven workflow completed successfully")
                        return output_data
                
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
            
            # Timeout handling
            raise TimeoutError(f"Workflow execution timed out after {timeout} seconds")
            
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ Event-driven workflow execution failed: {e}")
            raise
    
