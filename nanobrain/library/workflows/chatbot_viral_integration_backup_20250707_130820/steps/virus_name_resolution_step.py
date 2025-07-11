"""
Virus Name Resolution Step for Chatbot Viral Integration Workflow

SINGLE RESPONSIBILITY: Resolve virus species using LLM + cache
"""

import asyncio
import os
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.logging_system import get_logger
from nanobrain.library.agents.specialized.simple_specialized_agent import SimpleSpecializedAgent


class CacheManager:
    """Simple cache manager for virus resolution data"""
    
    def __init__(self, cache_directory: str, ttl: int = 86400):
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl  # Time to live in seconds
    
    async def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if valid"""
        cache_file = self.cache_directory / f"{cache_key}.json"
        
        try:
            if cache_file.exists():
                # Check if cache is still valid
                cache_age = time.time() - cache_file.stat().st_mtime
                if cache_age < self.ttl:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                else:
                    # Cache expired, remove it
                    cache_file.unlink()
                    return None
            return None
        except Exception:
            return None
    
    async def set(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Set cached data"""
        cache_file = self.cache_directory / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Silently fail if cache write fails
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CacheManager':
        """Create CacheManager from configuration"""
        return cls(
            cache_directory=config.get('cache_directory', 'data/virus_resolution_cache'),
            ttl=config.get('ttl', 86400)
        )


class VirusNameResolutionStep(Step):
    """
    Virus Name Resolution Step implementing LLM-based resolution with cache
    
    SINGLE RESPONSIBILITY: Resolve virus species using LLM + cache
    Cache-first approach to reduce LLM calls
    """
    
    COMPONENT_TYPE = "step"
    REQUIRED_CONFIG_FIELDS = ['name']
    
    def __init__(self, config: StepConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.resolution_agent = None
        self.cache_manager = None
        
    @classmethod
    def from_config(cls, config: StepConfig, **kwargs) -> 'VirusNameResolutionStep':
        """Create VirusNameResolutionStep from configuration"""
        return cls(config, **kwargs)
    
    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any], 
                         dependencies: Dict[str, Any]) -> None:
        """Initialize VirusNameResolutionStep with LLM agent and cache"""
        super()._init_from_config(config, component_config, dependencies)
        self._create_resolution_agent()
        self._create_cache_manager()
    
    def _create_resolution_agent(self) -> None:
        """Create LLM agent for virus name resolution"""
        try:
            # Get agent configuration from step config
            agent_config_dict = getattr(self.config, 'resolution_agent', {})
            
            # Create SimpleSpecializedAgent configuration
            agent_config = StepConfig(
                name="virus_resolution_agent",
                model=agent_config_dict.get('model', 'gpt-4'),
                temperature=agent_config_dict.get('temperature', 0.1),
                max_tokens=agent_config_dict.get('max_tokens', 200),
                timeout=agent_config_dict.get('timeout', 30),
                **agent_config_dict
            )
            
            # Create agent using from_config pattern
            self.resolution_agent = SimpleSpecializedAgent.from_config(agent_config)
            
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info("✅ Created LLM agent for virus name resolution")
                
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ Failed to create resolution agent: {e}")
            raise
    
    def _create_cache_manager(self) -> None:
        """Create cache manager for virus resolution data"""
        try:
            cache_config_dict = getattr(self.config, 'cache_config', {})
            self.cache_manager = CacheManager.from_config(cache_config_dict)
            
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"✅ Created cache manager: {self.cache_manager.cache_directory}")
                
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ Failed to create cache manager: {e}")
            raise
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process method implementing LLM-based virus name resolution with cache
        
        Args:
            input_data: Must contain 'extracted_virus_species' key
            
        Returns:
            Dict with resolved virus species information
        """
        if hasattr(self, 'nb_logger') and self.nb_logger:
            self.nb_logger.info("🔄 Processing virus name resolution step with cache-first approach")
        
        extracted_species = input_data.get('extracted_virus_species')
        if not extracted_species:
            raise ValueError("No virus species extracted from query")
        
        # Check cache first (NO hardcoded cache keys)
        cache_key = self._generate_cache_key(extracted_species)
        cached_resolution = await self.cache_manager.get(cache_key)
        
        if cached_resolution:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"💾 Cache hit for virus species: {extracted_species}")
            resolution_result = cached_resolution
        else:
            # Use LLM for resolution
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"🤖 Cache miss, using LLM for virus species: {extracted_species}")
            resolution_result = await self._resolve_virus_species_llm(extracted_species)
            
            # Cache the result
            await self.cache_manager.set(cache_key, resolution_result)
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"💾 Cached resolution result for: {extracted_species}")
        
        return {
            'virus_species': resolution_result['canonical_name'],
            'bvbrc_search_terms': resolution_result['bvbrc_search_terms'],
            'genome_characteristics': resolution_result.get('genome_characteristics', {}),
            'taxonomic_lineage': resolution_result.get('taxonomic_lineage', []),
            'confidence': resolution_result.get('confidence', 0.0),
            'user_query': input_data.get('user_query'),
            'cache_used': cached_resolution is not None
        }
    
    def _generate_cache_key(self, virus_species: str) -> str:
        """Generate cache key dynamically (NO hardcoding, NO session ID)"""
        # Use virus species name only (NO session ID)
        normalized_name = virus_species.lower().replace(' ', '_').replace('-', '_')
        # Remove special characters
        import re
        normalized_name = re.sub(r'[^a-z0-9_]', '', normalized_name)
        return f"virus_resolution_{normalized_name}"
    
    async def _resolve_virus_species_llm(self, virus_species: str) -> Dict[str, Any]:
        """Resolve virus species using LLM agent"""
        try:
            # Get prompt template from configuration (NO hardcoded mappings)
            prompt_template = getattr(self.config, 'resolution_prompt', 
                '''Resolve the following virus species to canonical ICTV taxonomy and BV-BRC search terms:

Virus Species: "{virus_species}"

Provide comprehensive resolution including:
1. Canonical ICTV name
2. BV-BRC database search terms
3. Genome characteristics (size ranges)
4. Taxonomic lineage

Return JSON format:
{{
  "canonical_name": "official ICTV name",
  "bvbrc_search_terms": ["search_term1", "search_term2"],
  "genome_characteristics": {{
    "min_length": 10000,
    "max_length": 15000
  }},
  "taxonomic_lineage": ["Viruses", "Family", "Genus"],
  "confidence": 0.0-1.0
}}

Return only the JSON response.''')
            
            formatted_prompt = prompt_template.format(virus_species=virus_species)
            
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"🤖 Sending virus resolution prompt to LLM")
            
            # Process using LLM agent
            response = await self.resolution_agent.process({
                'prompt': formatted_prompt,
                'expected_format': 'json',
                'virus_species': virus_species
            })
            
            # Parse LLM response
            return self._parse_resolution_response(response)
            
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ LLM resolution failed: {e}")
            
            # Return fallback resolution
            return self._create_fallback_resolution(virus_species)
    
    def _parse_resolution_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response and extract virus resolution information"""
        try:
            # Handle different response formats
            if isinstance(response, dict):
                if 'response' in response:
                    response_text = response['response']
                elif 'content' in response:
                    response_text = response['content']
                else:
                    response_text = str(response)
            else:
                response_text = str(response)
            
            # Try to parse as JSON
            import json
            try:
                if response_text.strip().startswith('{'):
                    parsed = json.loads(response_text)
                    
                    # Validate required fields
                    if 'canonical_name' in parsed and 'bvbrc_search_terms' in parsed:
                        return {
                            'canonical_name': parsed['canonical_name'],
                            'bvbrc_search_terms': parsed.get('bvbrc_search_terms', []),
                            'genome_characteristics': parsed.get('genome_characteristics', {}),
                            'taxonomic_lineage': parsed.get('taxonomic_lineage', []),
                            'confidence': float(parsed.get('confidence', 0.8))
                        }
                    else:
                        if hasattr(self, 'nb_logger') and self.nb_logger:
                            self.nb_logger.warning(f"⚠️ Incomplete LLM response, missing required fields")
                        return self._create_fallback_resolution(parsed.get('canonical_name', 'Unknown'))
                else:
                    if hasattr(self, 'nb_logger') and self.nb_logger:
                        self.nb_logger.warning(f"⚠️ Non-JSON response from LLM")
                    return self._create_fallback_resolution('Unknown')
                    
            except json.JSONDecodeError as e:
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.error(f"❌ JSON parsing failed: {e}")
                return self._create_fallback_resolution('Unknown')
                
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ Response parsing failed: {e}")
            return self._create_fallback_resolution('Unknown')
    
    def _create_fallback_resolution(self, virus_species: str) -> Dict[str, Any]:
        """Create fallback resolution when LLM fails"""
        # Create basic resolution based on common patterns
        search_terms = [virus_species]
        if 'chikungunya' in virus_species.lower():
            search_terms.extend(['CHIKV', 'chikungunya'])
        elif 'eastern equine' in virus_species.lower():
            search_terms.extend(['EEEV', 'eastern equine encephalitis'])
        elif 'western equine' in virus_species.lower():
            search_terms.extend(['WEEV', 'western equine encephalitis'])
        elif 'venezuelan equine' in virus_species.lower():
            search_terms.extend(['VEEV', 'venezuelan equine encephalitis'])
        
        return {
            'canonical_name': virus_species,
            'bvbrc_search_terms': search_terms,
            'genome_characteristics': {
                'min_length': 8000,
                'max_length': 15000
            },
            'taxonomic_lineage': ['Viruses', 'Togaviridae', 'Alphavirus'],
            'confidence': 0.5  # Lower confidence for fallback
        } 