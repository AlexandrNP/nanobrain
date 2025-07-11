"""
Protein Synonym Agent Step for NanoBrain Workflow

This step wraps the ProteinSynonymAgent to provide synonym resolution
within the viral protein analysis workflow using Link-based communication.
"""

import time
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

from nanobrain.core.data_unit import DataUnit, DataUnitConfig
from nanobrain.core.step import Step
from nanobrain.core.config import YAMLConfig
from nanobrain.library.agents.specialized.protein_synonym_agent import ProteinSynonymAgent


class ProteinSynonymAgentStep(Step):
    """
    Workflow step that wraps the ProteinSynonymAgent for synonym resolution.
    
    This step:
    1. Receives synonym requests via DataUnit from AnnotationMappingStep
    2. Uses ProteinSynonymAgent to identify synonyms and classify proteins
    3. Updates the DataUnit context with results
    4. Sends the updated DataUnit back to AnnotationMappingStep
    """
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        """Initialize the step with configuration."""
        super().__init__(config, **kwargs)
        
        # Agent configuration
        self.agent = None
        self.agent_config_path = config.get('agent', {}).get('config_path')
        if not self.agent_config_path:
            raise ValueError("agent.config_path is required in configuration")
        
        # Processing configuration
        self.max_batch_size = config.get('processing', {}).get('max_batch_size', 100)
        self.timeout_per_batch = config.get('processing', {}).get('timeout_per_batch', 60)
        
        # Cache configuration
        self.cache_enabled = config.get('cache', {}).get('enable', True)
        self.cache_ttl_days = config.get('cache', {}).get('ttl_days', 30)
        self.cache_dir = config.get('cache', {}).get('cache_dir', 'data/synonym_cache')
        
    async def initialize(self):
        """Initialize the wrapped agent."""
        await super().initialize()
        
        try:
            # Load agent configuration
            agent_config_path = Path(self.agent_config_path)
            
            if not agent_config_path.is_absolute():
                # Make path relative to workspace root
                agent_config_path = Path.cwd() / agent_config_path
            
            if not agent_config_path.exists():
                raise FileNotFoundError(f"Agent config not found: {agent_config_path}")
            
            # Load configuration using YAMLConfig
            yaml_config = YAMLConfig()
            agent_config = yaml_config.load_from_file(str(agent_config_path))
            
            # Create agent via from_config
            self.agent = ProteinSynonymAgent.from_config(agent_config)
            await self.agent.initialize()
            
            self.nb_logger.info(f"Successfully initialized ProteinSynonymAgent from {agent_config_path}")
            
        except Exception as e:
            self.nb_logger.error(f"Failed to initialize agent: {str(e)}")
            raise
    
    async def cleanup(self):
        """Cleanup agent resources."""
        if self.agent:
            try:
                await self.agent.shutdown()
            except Exception as e:
                self.nb_logger.error(f"Error shutting down agent: {str(e)}")
        await super().cleanup()
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process synonym requests received via Link.
        
        Args:
            input_data: Dictionary containing 'data_unit' with the request
            
        Returns:
            Status dictionary with processing results
        """
        start_time = time.time()
        
        # Extract DataUnit from input
        if 'data_unit' not in input_data:
            self.nb_logger.warning("No data_unit in input")
            return {'error': 'No data unit provided', 'status': 'failed'}
            
        data_unit = input_data['data_unit']
        if not isinstance(data_unit, DataUnit):
            self.nb_logger.warning(f"Invalid data unit type: {type(data_unit)}")
            return {'error': 'Invalid data unit type', 'status': 'failed'}
        
        try:
            # Extract context from DataUnit
            context = await data_unit.get()
            if not isinstance(context, dict):
                self.nb_logger.warning("Data unit does not contain dict")
                return {'error': 'Invalid context type', 'status': 'failed'}
            
            # Verify this is a request
            if context.get('stage') != 'synonym_request':
                self.nb_logger.warning(f"Invalid stage: {context.get('stage')}")
                return {'error': 'Not a synonym request', 'status': 'failed'}
            
            # Extract required data
            products = context.get('products', [])
            virus_info = context.get('virus_info', {})
            request_id = context.get('request_id', 'unknown')
            
            self.nb_logger.info(
                f"Processing synonym request {request_id}: "
                f"{len(products)} products for {virus_info.get('genus', 'Unknown')}"
            )
            
            # Call agent to identify synonyms
            synonym_groups = await self._identify_synonyms_with_timeout(products, virus_info)
            
            # Call agent to classify proteins (with synonyms for context)
            classifications = await self._classify_proteins_with_timeout(
                products, 
                virus_info,
                synonym_groups
            )
            
            # Update context object IN PLACE (it's passed by reference)
            context['synonym_groups'] = synonym_groups
            context['protein_classifications'] = classifications
            context['stage'] = 'synonym_response'
            context['processing_time'] = time.time() - start_time
            
            # Log results
            self.nb_logger.info(
                f"Request {request_id} completed: "
                f"Found {len(synonym_groups)} synonym groups, "
                f"Classified {len(classifications)} proteins"
            )
            
            # Update DataUnit metadata
            await data_unit.set_metadata('stage', 'response')
            await data_unit.set_metadata('processed_by', 'synonym_agent')
            await data_unit.set_metadata('processing_time', time.time() - start_time)
            
            # Deposit the same DataUnit (with updated context) back
            await self.deposit_output(data_unit)
            
            return {
                'status': 'completed',
                'request_id': request_id,
                'groups_found': len(synonym_groups),
                'classifications_made': len(classifications),
                'processing_time': time.time() - start_time
            }
            
        except asyncio.TimeoutError:
            self.nb_logger.error(f"Timeout processing synonym request")
            return {
                'status': 'timeout',
                'error': 'Processing timeout',
                'processing_time': time.time() - start_time
            }
        except Exception as e:
            self.nb_logger.error(f"Error processing synonym request: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def _identify_synonyms_with_timeout(
        self, 
        products: list, 
        virus_info: Dict[str, Any]
    ) -> Dict[str, list]:
        """Identify synonyms with timeout protection."""
        try:
            return await asyncio.wait_for(
                self.agent.identify_synonyms(products, virus_info),
                timeout=self.timeout_per_batch
            )
        except asyncio.TimeoutError:
            self.nb_logger.warning("Synonym identification timed out, returning empty groups")
            return {}
    
    async def _classify_proteins_with_timeout(
        self,
        products: list,
        virus_info: Dict[str, Any],
        synonym_groups: Dict[str, list]
    ) -> Dict[str, str]:
        """Classify proteins with timeout protection."""
        try:
            return await asyncio.wait_for(
                self.agent.classify_proteins(products, virus_info, synonym_groups),
                timeout=self.timeout_per_batch
            )
        except asyncio.TimeoutError:
            self.nb_logger.warning("Protein classification timed out, returning default classifications")
            # Return default classifications
            return {product: 'unknown' for product in products} 