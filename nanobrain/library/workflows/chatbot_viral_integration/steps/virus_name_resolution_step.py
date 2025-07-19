"""
Enhanced Virus Name Resolution Step for Chatbot Viral Integration

This step provides ultra-high-confidence synonym detection with multi-agent processing.
- Enhanced Query Analysis Agent for virus species extraction
- Virus Synonym Detection Agent for comprehensive synonym generation
- Species validation criteria for downstream CSV matching
- Configurable confidence filtering (>0.9 for zero contamination)
- NO hardcoded virus mappings - all processing via configurable agents
"""

import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from nanobrain.core.step import Step
from nanobrain.core.data_unit import DataUnit
from nanobrain.core.agent import SimpleAgent
from nanobrain.core.config.component_factory import create_component, load_config_file


class UltraHighConfidenceCacheManager:
    """
    Ultra-high-confidence cache manager for virus name resolution
    
    PRECISION REQUIREMENT: Only cache results with confidence >= threshold
    Prevents low-confidence data from contaminating future queries
    """
    
    def __init__(self, cache_directory: str, confidence_threshold: float = 0.9):
        self.cache_directory = Path(cache_directory)
        self.confidence_threshold = confidence_threshold
        self.cache_directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'UltraHighConfidenceCacheManager':
        """Create cache manager from configuration"""
        cache_directory = config.get('cache_directory', 'data/ultra_cache')
        confidence_threshold = config.get('confidence_threshold', 0.9)
        
        return cls(
            cache_directory=cache_directory,
            confidence_threshold=confidence_threshold
        )
    
    async def get_cached_resolution(self, virus_key: str) -> Optional[Dict[str, Any]]:
        """Get cached virus resolution if confidence meets threshold"""
        cache_file = self.cache_directory / f"{virus_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Check if cached data meets confidence threshold
                overall_confidence = cached_data.get('overall_confidence', 0.0)
                
                if overall_confidence >= self.confidence_threshold:
                    return cached_data
                else:
                    # Remove low-confidence cache entry
                    cache_file.unlink()
                    return None
                    
            except Exception:
                return None
        
        return None
    
    async def cache_resolution(self, virus_key: str, resolution_data: Dict[str, Any]) -> None:
        """Cache virus resolution if it meets confidence threshold"""
        overall_confidence = resolution_data.get('overall_confidence', 0.0)
        
        if overall_confidence >= self.confidence_threshold:
            cache_file = self.cache_directory / f"{virus_key}.json"
            
            # Add cache metadata
            resolution_data['cache_metadata'] = {
                'cached_at': datetime.now().isoformat(),
                'confidence_threshold': self.confidence_threshold,
                'cache_source': 'ultra_high_confidence'
            }
            
            with open(cache_file, 'w') as f:
                json.dump(resolution_data, f, indent=2)


class EnhancedVirusNameResolutionStep(Step):
    """
    ENHANCED RESPONSIBILITY: Ultra-high-confidence virus synonym detection + species validation criteria
    
    This step provides:
    1. Enhanced virus species extraction using specialized agents
    2. Ultra-high-confidence synonym detection (>0.9 confidence)
    3. Species validation criteria for CSV matching
    4. Multi-stage confidence filtering for zero contamination
    5. Comprehensive taxonomic information for validation
    """
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.query_analysis_agent = None
        self.synonym_detection_agent = None
        self.cache_manager = None
        
    def _init_from_config(self, config, component_config: Dict[str, Any], dependencies: Dict[str, Any]) -> None:
        """Initialize EnhancedVirusNameResolutionStep with multiple specialized agents"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Initialize enhanced query analysis agent
        self.query_analysis_agent = self._create_query_analysis_agent(component_config)
        
        # Initialize virus synonym detection agent
        self.synonym_detection_agent = self._create_synonym_detection_agent(component_config)
        
        # Initialize ultra-high-confidence cache manager
        self.cache_manager = self._create_ultra_cache_manager(component_config)
        
        # Get confidence threshold from configuration
        self.confidence_threshold = component_config.get('confidence_threshold', 0.9)
        
        if self.nb_logger:
            self.nb_logger.info(f"🔬 Enhanced Virus Name Resolution Step initialized with ultra-high-confidence threshold: {self.confidence_threshold}")
    
    def _create_query_analysis_agent(self, component_config: Dict[str, Any]) -> SimpleAgent:
        """Create enhanced query analysis agent for virus species extraction"""
        agent_config_ref = component_config.get('query_analysis_agent', {})
        
        if 'config_file' in agent_config_ref:
            # Load agent from external configuration file
            config_file_path = agent_config_ref['config_file']
            agent_config = load_config_file(config_file_path)
            agent_class_path = agent_config.get('class')
            
            if not agent_class_path:
                raise ValueError(f"Agent configuration must specify 'class' field: {config_file_path}")
            
            return create_component(agent_class_path, agent_config)
        else:
            # Fallback to inline configuration
            if not agent_config_ref:
                raise ValueError("No query_analysis_agent configuration found")
            
            # Ensure name field for agent creation
            if 'name' not in agent_config_ref:
                agent_config_ref['name'] = 'enhanced_query_analysis_agent'
            
            # Create agent using from_config pattern
            from nanobrain.library.agents.specialized.base import SimpleSpecializedAgent
            from nanobrain.core.agent import AgentConfig
            
            config_obj = AgentConfig(**agent_config_ref)
            return SimpleSpecializedAgent.from_config(config_obj)
    
    def _create_synonym_detection_agent(self, component_config: Dict[str, Any]) -> SimpleAgent:
        """Create virus synonym detection agent for ultra-high-confidence synonyms"""
        agent_config_ref = component_config.get('synonym_detection_agent', {})
        
        if 'config_file' in agent_config_ref:
            # Load agent from external configuration file
            config_file_path = agent_config_ref['config_file']
            agent_config = load_config_file(config_file_path)
            agent_class_path = agent_config.get('class')
            
            if not agent_class_path:
                raise ValueError(f"Agent configuration must specify 'class' field: {config_file_path}")
            
            return create_component(agent_class_path, agent_config)
        else:
            # Fallback to inline configuration
            if not agent_config_ref:
                raise ValueError("No synonym_detection_agent configuration found")
            
            # Ensure name field for agent creation
            if 'name' not in agent_config_ref:
                agent_config_ref['name'] = 'virus_synonym_detection_agent'
            
            # Create agent using from_config pattern
            from nanobrain.library.agents.specialized.base import SimpleSpecializedAgent
            from nanobrain.core.agent import AgentConfig
            
            config_obj = AgentConfig(**agent_config_ref)
            return SimpleSpecializedAgent.from_config(config_obj)
    
    def _create_ultra_cache_manager(self, component_config: Dict[str, Any]) -> UltraHighConfidenceCacheManager:
        """Create ultra-high-confidence cache manager from configuration"""
        cache_config = component_config.get('cache_config', {})
        
        # Add confidence threshold to cache config
        cache_config['confidence_threshold'] = component_config.get('confidence_threshold', 0.9)
        
        return UltraHighConfidenceCacheManager.from_config(cache_config)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process ultra-high-confidence virus species resolution with multi-agent pipeline
        
        Args:
            input_data: Contains 'extracted_virus_species' or 'user_query' key
            
        Returns:
            Dict with ultra-high-confidence synonyms and species validation criteria
        """
        # Extract virus species from input
        extracted_species = input_data.get('extracted_virus_species')
        user_query = input_data.get('user_query', '')
        
        # If no extracted species, use query analysis agent to extract
        if not extracted_species and user_query:
            self.nb_logger.info("🔍 No extracted species provided, using query analysis agent")
            extracted_species = await self._extract_virus_species_enhanced(user_query)
        
        if not extracted_species:
            raise ValueError("No virus species could be extracted from input")
        
        self.nb_logger.info(f"🧬 Processing virus species: {extracted_species}")
        
        # Check ultra-high-confidence cache first
        cache_key = self._generate_cache_key(extracted_species)
        cached_result = await self.cache_manager.get_cached_resolution(cache_key)
        
        if cached_result:
            self.nb_logger.info(f"🎯 Found ultra-high-confidence cached result for {extracted_species}")
            return cached_result
        
        try:
            # Generate ultra-high-confidence synonyms
            result = await self._generate_ultra_synonyms(extracted_species)
            
            # Cache if meets confidence threshold
            if result.get('overall_confidence', 0.0) >= self.confidence_threshold:
                await self.cache_manager.cache_resolution(cache_key, result)
                self.nb_logger.info(f"✅ Cached ultra-high-confidence result for {extracted_species}")
            else:
                self.nb_logger.warning(f"⚠️ Result for {extracted_species} below confidence threshold, not cached")
            
            return result
        
        except Exception as e:
            self.nb_logger.error(f"❌ Error in enhanced virus name resolution: {e}")
            raise
    
    async def _extract_virus_species_enhanced(self, user_query: str) -> Optional[str]:
        """Extract virus species using enhanced query analysis agent"""
        try:
            # Use query analysis agent to extract virus species
            prompt_template = self.step_config.get('virus_extraction_prompt', 
                'Analyze this query and extract virus species: "{user_query}". Return JSON with virus_species field.')
            
            formatted_prompt = prompt_template.format(user_query=user_query)
            
            response = await self.query_analysis_agent.process({
                'prompt': formatted_prompt,
                'expected_format': 'json'
            })
            
            # Parse response to extract virus species
            parsed_response = self._parse_agent_response(response)
            return parsed_response.get('virus_species')
            
        except Exception as e:
            self.nb_logger.error(f"❌ Error in enhanced virus species extraction: {e}")
            return None
    
    async def _generate_ultra_synonyms(self, virus_species: str) -> Dict[str, Any]:
        """Generate ultra-high-confidence synonyms using specialized agent"""
        try:
            # Use virus synonym detection agent for ultra-high-confidence processing
            synonym_prompt = self._build_synonym_detection_prompt(virus_species)
            
            response = await self.synonym_detection_agent.process({
                'prompt': synonym_prompt,
                'expected_format': 'json',
                'confidence_threshold': self.confidence_threshold
            })
            
            # Parse and validate response
            parsed_response = self._parse_agent_response(response)
            
            # Ensure ultra-high-confidence format
            result = self._format_ultra_high_confidence_result(virus_species, parsed_response)
            
            return result
            
        except Exception as e:
            self.nb_logger.error(f"❌ Error in ultra synonym generation: {e}")
            raise
    
    def _build_synonym_detection_prompt(self, virus_species: str) -> str:
        """Build comprehensive prompt for virus synonym detection"""
        return f"""
        As a virology expert, generate ultra-high-confidence synonyms for: {virus_species}
        
        Requirements:
        1. Only include synonyms with >90% confidence
        2. Include taxonomic lineage information
        3. Generate species validation criteria
        4. Provide confidence scores for each synonym
        
        Return JSON with:
        - canonical_name: Official ICTV name
        - ultra_high_confidence_synonyms: List of >90% confidence synonyms
        - taxonomic_lineage: genus, family, order
        - confidence_scores: confidence for each synonym
        - overall_confidence: overall confidence score
        """
    
    def _format_ultra_high_confidence_result(self, virus_species: str, parsed_response: Dict[str, Any]) -> Dict[str, Any]:
        """Format response into ultra-high-confidence result structure"""
        canonical_name = parsed_response.get('canonical_name', virus_species)
        synonyms = parsed_response.get('ultra_high_confidence_synonyms', [virus_species])
        taxonomic_lineage = parsed_response.get('taxonomic_lineage', {})
        confidence_scores = parsed_response.get('confidence_scores', {})
        overall_confidence = parsed_response.get('overall_confidence', 0.95)
        
        # Generate species validation criteria
        validation_criteria = self._generate_species_validation_criteria(
            canonical_name, synonyms, taxonomic_lineage, overall_confidence
        )
        
        return {
            'virus_species': canonical_name,
            'ultra_high_confidence_synonyms': synonyms,
            'species_validation_criteria': validation_criteria,
            'confidence_scores': confidence_scores,
            'overall_confidence': overall_confidence,
            'taxonomic_lineage': taxonomic_lineage,
            'validation_metadata': {
                'generation_method': 'multi_agent_ultra_precision',
                'confidence_threshold': self.confidence_threshold,
                'contamination_prevention': True,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _generate_species_validation_criteria(self, canonical_name: str, synonyms: List[str], 
                                           taxonomic_lineage: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        """Generate comprehensive species validation criteria for CSV matching"""
        return {
            'primary_species': canonical_name,
            'acceptable_variations': synonyms,
            'taxonomic_constraints': {
                'genus': taxonomic_lineage.get('genus', ''),
                'family': taxonomic_lineage.get('family', ''),
                'order': taxonomic_lineage.get('order', '')
            },
            'validation_rules': {
                'exact_match_required': confidence > 0.95,
                'case_sensitive': False,
                'allow_abbreviations': True,
                'require_species_confirmation': True
            },
            'contamination_prevention': {
                'reject_on_genus_mismatch': True,
                'require_species_confirmation': True,
                'cross_validation_required': True,
                'zero_contamination_tolerance': 0.0
            }
        }
    
    def _generate_cache_key(self, virus_species: str) -> str:
        """Generate normalized cache key for virus species"""
        normalized = virus_species.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
        return f"ultra_resolution_{normalized}"
    
    def _parse_agent_response(self, response: Any) -> Dict[str, Any]:
        """Parse agent response ensuring JSON format"""
        if isinstance(response, dict):
            return response
        elif isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse agent response as JSON: {response}")
        else:
            raise ValueError(f"Unexpected agent response type: {type(response)}")


# Maintain backward compatibility by keeping original class as alias
VirusNameResolutionStep = EnhancedVirusNameResolutionStep 