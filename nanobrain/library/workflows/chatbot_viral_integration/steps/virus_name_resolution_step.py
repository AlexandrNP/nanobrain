"""
Virus Name Resolution Step for Chatbot Viral Integration

This step resolves virus species using LLM with cache-first approach.
NO hardcoded virus name mappings - all resolution via LLM with configurable prompts.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from nanobrain.core.step import Step
from nanobrain.core.data_unit import DataUnit
from nanobrain.core.agent import SimpleAgent


class CacheManager:
    """
    Cache manager for virus resolution data
    NO session ID usage - dynamic cache key generation only
    """
    
    def __init__(self, cache_directory: str, ttl: int = 86400):
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CacheManager':
        """Create CacheManager from configuration"""
        return cls(
            cache_directory=config.get('cache_directory', 'data/virus_resolution_cache'),
            ttl=config.get('ttl', 86400)
        )
    
    async def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached resolution data"""
        cache_file = self.cache_directory / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            # Check if cache is still valid
            if self._is_cache_expired(cache_file):
                os.remove(cache_file)
                return None
            
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        except Exception:
            return None
    
    async def set(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Set cached resolution data"""
        cache_file = self.cache_directory / f"{cache_key}.json"
        
        try:
            # Add timestamp to cached data
            data['_cache_timestamp'] = datetime.now().isoformat()
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception:
            pass  # Fail silently if cache write fails
    
    def _is_cache_expired(self, cache_file: Path) -> bool:
        """Check if cache file is expired"""
        try:
            file_age = datetime.now().timestamp() - cache_file.stat().st_mtime
            return file_age > self.ttl
        except Exception:
            return True


class VirusNameResolutionStep(Step):
    """
    SINGLE RESPONSIBILITY: Resolve virus species using LLM + cache
    
    This step resolves virus species to canonical ICTV names and BV-BRC search terms
    using LLM agents with cache-first approach. NO hardcoded mappings.
    """
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.resolution_agent = None
        self.cache_manager = None
        
    def _init_from_config(self, config, component_config: Dict[str, Any], dependencies: Dict[str, Any]) -> None:
        """Initialize VirusNameResolutionStep with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Initialize resolution agent from configuration
        self.resolution_agent = self._create_resolution_agent(component_config)
        
        # Initialize cache manager from configuration
        self.cache_manager = self._create_cache_manager(component_config)
        
        if self.nb_logger:
            self.nb_logger.info("🔬 Virus Name Resolution Step initialized with LLM + cache")
    
    def _create_resolution_agent(self, component_config: Dict[str, Any]) -> SimpleAgent:
        """Create LLM agent for virus name resolution from configuration"""
        # Get agent config from the component configuration
        agent_config_dict = component_config.get('resolution_agent', {})
        
        # If not found, try to get from the main config object (the full YAML configuration)
        if not agent_config_dict and hasattr(self.config, 'resolution_agent'):
            agent_config_dict = getattr(self.config, 'resolution_agent')
        elif not agent_config_dict and isinstance(self.config, dict):
            agent_config_dict = self.config.get('resolution_agent', {})
        
        if not agent_config_dict:
            raise ValueError("No resolution_agent configuration found in step configuration")
        
        # Ensure the name field is present for SimpleAgent
        if 'name' not in agent_config_dict:
            agent_config_dict['name'] = 'virus_resolution_agent'
        
        # Create AgentConfig object from dictionary - this is what SimpleAgent expects
        from nanobrain.core.agent import AgentConfig
        
        # Create A2A protocol compliant agent_card
        agent_card = {
            "version": "1.0.0",
            "purpose": "Virus species resolution to canonical ICTV taxonomy",
            "detailed_description": "Specialized agent for resolving virus species names to canonical ICTV taxonomy with BV-BRC search terms",
            "domain": "bioinformatics",
            "expertise_level": "expert",
            "input_format": {
                "primary_mode": "text",
                "supported_modes": ["text"],
                "content_types": ["text/plain"]
            },
            "output_format": {
                "primary_mode": "json",
                "supported_modes": ["json"],
                "content_types": ["application/json"]
            },
            "capabilities": {
                "streaming": False,
                "multi_turn_conversation": False,
                "context_retention": False,
                "tool_usage": False
            }
        }
        
        agent_config = AgentConfig(
            name=agent_config_dict.get('name', 'virus_resolution_agent'),
            description=agent_config_dict.get('description', 'Agent for resolving virus species to canonical ICTV taxonomy'),
            model=agent_config_dict.get('model', 'gpt-4'),
            temperature=agent_config_dict.get('temperature', 0.1),
            max_tokens=agent_config_dict.get('max_tokens', 200),
            system_prompt=agent_config_dict.get('system_prompt', ''),
            timeout=agent_config_dict.get('timeout', 30),
            agent_card=agent_card
        )
        
        # Create agent using mandatory from_config pattern with proper AgentConfig object
        return SimpleAgent.from_config(agent_config)
    
    def _create_cache_manager(self, component_config: Dict[str, Any]) -> CacheManager:
        """Create cache manager for virus resolution data from configuration"""
        cache_config = component_config.get('cache_config', {})
        
        # Create cache manager using from_config pattern
        return CacheManager.from_config(cache_config)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process virus species resolution using LLM + cache
        
        Args:
            input_data: Contains 'extracted_virus_species' key
            
        Returns:
            Dict with resolved virus species information
        """
        extracted_species = input_data.get('extracted_virus_species')
        if not extracted_species:
            raise ValueError("No virus species extracted from query")
        
        try:
            # Check cache first (NO hardcoded cache keys)
            cache_key = self._generate_cache_key(extracted_species)
            cached_resolution = await self.cache_manager.get(cache_key)
            
            if cached_resolution:
                self.nb_logger.info(f"✅ Cache hit for virus species: {extracted_species}")
                resolution_result = cached_resolution
            else:
                # Use LLM for resolution
                self.nb_logger.info(f"🔄 Cache miss, using LLM for virus species: {extracted_species}")
                resolution_result = await self._resolve_virus_species_llm(extracted_species)
                
                # Cache the result
                await self.cache_manager.set(cache_key, resolution_result)
            
            return {
                'virus_species': resolution_result['canonical_name'],
                'bvbrc_search_terms': resolution_result['bvbrc_search_terms'],
                'genome_characteristics': resolution_result.get('genome_characteristics', {}),
                'taxonomic_lineage': resolution_result.get('taxonomic_lineage', []),
                'confidence': resolution_result.get('confidence', 0.0),
                'user_query': input_data.get('user_query')
            }
        
        except Exception as e:
            self.nb_logger.error(f"❌ Error in virus name resolution: {e}")
            raise
    
    def _generate_cache_key(self, virus_species: str) -> str:
        """
        Generate cache key dynamically (NO hardcoding, NO session ID)
        
        Args:
            virus_species: Virus species name
            
        Returns:
            Cache key string
        """
        # Use virus species name only (NO session ID)
        normalized_name = virus_species.lower().replace(' ', '_').replace('-', '_')
        return f"virus_resolution_{normalized_name}"
    
    async def _resolve_virus_species_llm(self, virus_species: str) -> Dict[str, Any]:
        """
        Resolve virus species using LLM agent
        
        Args:
            virus_species: Virus species name to resolve
            
        Returns:
            Dict with resolved virus species information
        """
        try:
            # Get prompt template from configuration
            prompt_template = self.step_config.get('resolution_prompt', '')
            
            if not prompt_template:
                raise ValueError("No resolution prompt configured")
            
            # Format prompt with virus species
            formatted_prompt = prompt_template.format(virus_species=virus_species)
            
            # Call LLM agent with formatted prompt
            response = await self.resolution_agent.process({
                'prompt': formatted_prompt,
                'expected_format': 'json'
            })
            
            # Parse the response
            return self._parse_resolution_response(response)
            
        except Exception as e:
            self.nb_logger.error(f"❌ Error in LLM virus species resolution: {e}")
            raise
    
    def _parse_resolution_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse LLM response to extract virus species resolution
        
        Args:
            response: Raw response from LLM agent
            
        Returns:
            Dict with resolved virus species data
        """
        try:
            # Handle different response formats
            if isinstance(response, dict):
                if 'content' in response:
                    content = response['content']
                elif 'text' in response:
                    content = response['text']
                else:
                    content = str(response)
            else:
                content = str(response)
            
            # Try to parse as JSON
            try:
                parsed_data = json.loads(content)
                
                return {
                    'canonical_name': parsed_data.get('canonical_name'),
                    'bvbrc_search_terms': parsed_data.get('bvbrc_search_terms', []),
                    'genome_characteristics': parsed_data.get('genome_characteristics', {}),
                    'taxonomic_lineage': parsed_data.get('taxonomic_lineage', []),
                    'confidence': parsed_data.get('confidence', 0.0)
                }
            
            except json.JSONDecodeError:
                # If not JSON, create fallback response
                self.nb_logger.warning("⚠️ LLM response not in JSON format, creating fallback")
                
                # Create fallback resolution
                return {
                    'canonical_name': virus_species,
                    'bvbrc_search_terms': [virus_species],
                    'genome_characteristics': {},
                    'taxonomic_lineage': ['Viruses'],
                    'confidence': 0.5
                }
        
        except Exception as e:
            self.nb_logger.error(f"❌ Error parsing resolution response: {e}")
            raise 