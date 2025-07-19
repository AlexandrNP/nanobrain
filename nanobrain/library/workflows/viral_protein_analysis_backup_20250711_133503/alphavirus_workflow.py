"""
Simplified Alphavirus Workflow

Framework-compliant workflow that uses NanoBrain's data-driven event execution.
This workflow relies ENTIRELY on the framework's built-in execution system with triggers and links.
NO CUSTOM EXECUTION LOGIC - pure configuration-driven approach.
"""

import time
from typing import Dict, Any
from nanobrain.core.workflow import Workflow, WorkflowConfig


class AlphavirusWorkflow(Workflow):
    """
    Viral Protein Analysis Workflow
    
    Comprehensive viral protein analysis workflow supporting any viral species.
    Originally designed for Alphavirus analysis but supports all viral types.
    
    This workflow uses NanoBrain's data-driven event execution model:
    - Data unit changes trigger step execution automatically
    - Framework handles step orchestration through links and triggers
    - NO manual step execution or data passing
    - PURE configuration-driven behavior
    """
    
    def _init_from_config(self, config: WorkflowConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize AlphavirusWorkflow with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Generic virus support attributes - configuration-driven only
        config_dict = getattr(config, 'config', {})
        self.virus_name = config_dict.get('virus_name', 'Viral species')
        
        self.workflow_logger.info(f"🧬 AlphavirusWorkflow initialized for viral protein analysis")
    
    async def execute(self, input_data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute method for Step interface compatibility
        
        This method allows AlphavirusWorkflow to be used as a step in another workflow.
        It uses the framework's built-in workflow execution system with pure configuration-driven behavior.
        """
        try:
            # Ensure input_data is not None
            if input_data is None:
                input_data = {}
            
            # CRITICAL FIX: Call our own process method, not the parent's
            result = await self.process(input_data, **kwargs)
            
            # Ensure result always has success field for step compatibility
            if result is None:
                return {
                    'success': True, 
                    'message': 'Viral protein analysis completed', 
                    'execution_model': 'data_driven_events'
                }
            
            # Add execution model metadata for step compatibility
            if isinstance(result, dict):
                result['execution_model'] = 'data_driven_events'
            
            return result
            
        except Exception as e:
            self.workflow_logger.error(f"❌ AlphavirusWorkflow execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_model': 'data_driven_events'
            }
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process method for data-driven execution
        
        This method sets the virus_input data unit to trigger the internal workflow steps.
        """
        try:
            self.workflow_logger.info("🧬 AlphavirusWorkflow processing input data")
            
            # Extract virus information from input data
            virus_species = input_data.get('virus_species', '')
            bvbrc_search_terms = input_data.get('bvbrc_search_terms', [])
            user_query = input_data.get('user_query', '')
            
            if not virus_species:
                self.workflow_logger.warning("⚠️ No virus species provided in input data")
                return {
                    'success': False,
                    'error': 'No virus species provided',
                    'execution_model': 'data_driven_events'
                }
            
            # Prepare virus input data for the workflow
            virus_input_data = {
                'virus_species': virus_species,
                'search_terms': bvbrc_search_terms,
                'user_query': user_query,
                'analysis_type': 'protein_analysis',
                'target_proteins': ['all'],  # Default to all proteins
                'output_format': 'pssm'  # PSSM requested
            }
            
            self.workflow_logger.info(f"📥 Setting virus_input data unit with: {virus_species}")
            
            # CRITICAL FIX: Access the workflow's input data units that are defined in YAML
            virus_input_set = False
            
            # Method 1: Try to access via workflow's input_data_units attribute
            if hasattr(self, 'input_data_units') and 'virus_input' in self.input_data_units:
                await self.input_data_units['virus_input'].set(virus_input_data)
                self.workflow_logger.info("✅ virus_input data unit set via input_data_units")
                virus_input_set = True
            
            # Method 2: Try to access via workflow's data_units attribute (framework standard)
            elif hasattr(self, 'data_units') and 'virus_input' in self.data_units:
                await self.data_units['virus_input'].set(virus_input_data)
                self.workflow_logger.info("✅ virus_input data unit set via data_units")
                virus_input_set = True
            
            # Method 3: Try to find virus_input in child steps' input data units
            elif hasattr(self, 'child_steps') and 'data_acquisition' in self.child_steps:
                data_acq_step = self.child_steps['data_acquisition']
                if hasattr(data_acq_step, 'input_data_units') and 'virus_input' in data_acq_step.input_data_units:
                    await data_acq_step.input_data_units['virus_input'].set(virus_input_data)
                    self.workflow_logger.info("✅ virus_input data unit set via data_acquisition step")
                    virus_input_set = True
            
            if not virus_input_set:
                self.workflow_logger.error("❌ virus_input data unit not found in any location")
                # Log available data units for debugging
                if hasattr(self, 'input_data_units'):
                    self.workflow_logger.info(f"Available input_data_units: {list(self.input_data_units.keys())}")
                if hasattr(self, 'data_units'):
                    self.workflow_logger.info(f"Available data_units: {list(self.data_units.keys())}")
                
                return {
                    'success': False,
                    'error': 'virus_input data unit not found',
                    'execution_model': 'data_driven_events'
                }
            
            # CRITICAL FIX: Let the framework handle trigger-based execution naturally
            # Do NOT call super().process() as it bypasses the trigger system
            # Instead, wait for the framework to trigger steps and return a success result
            
            # Give some time for the framework to process triggers (if needed)
            import asyncio
            await asyncio.sleep(0.1)  # Small delay to allow trigger processing
            
            self.workflow_logger.info("🎯 AlphavirusWorkflow processing completed - triggers should execute steps")
            
            # Return success - the actual analysis results will be generated by the triggered steps
            return {
                'success': True,
                'message': 'Viral protein analysis workflow triggered successfully',
                'virus_species': virus_species,
                'execution_model': 'data_driven_events'
            }
            
        except Exception as e:
            self.workflow_logger.error(f"❌ AlphavirusWorkflow processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_model': 'data_driven_events'
            }
    
    def get_workflow_config(self) -> Dict[str, Any]:
        """Get workflow configuration for validation"""
        return {
            'virus_name': self.virus_name,
            'execution_model': 'data_driven_events'
        } 