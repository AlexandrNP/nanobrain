"""
Annotation Mapping Step for Viral Protein Analysis Workflow

SINGLE RESPONSIBILITY: Cache-based synonym resolution with LLM fallback
"""

import asyncio
import os
import json
import time
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.logging_system import get_logger


class AnnotationMappingStep(Step):
    """
    Annotation Mapping Step implementing cache-based synonym resolution
    
    SINGLE RESPONSIBILITY: Cache-based synonym resolution with LLM fallback
    NO session ID contamination - cache keys generated dynamically
    """
    
    COMPONENT_TYPE = "step"
    REQUIRED_CONFIG_FIELDS = ['name']
    
    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return StepConfig - ONLY method that differs from other components"""
        return StepConfig
    
    def __init__(self, config: StepConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.synonym_agent = None
        self.cache_manager = None
        
    # Now inherits unified from_config implementation from FromConfigBase
    
    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any], 
                         dependencies: Dict[str, Any]) -> None:
        """Initialize AnnotationMappingStep with LLM agent and cache"""
        super()._init_from_config(config, component_config, dependencies)
        self._create_synonym_agent()
        self._create_cache_manager()
    
    def _create_synonym_agent(self) -> None:
        """Create LLM agent for synonym resolution"""
        try:
            # Import SimpleAgent instead of SimpleSpecializedAgent
            from nanobrain.core.agent import AgentConfig, SimpleAgent
            
            # Get agent configuration from step config
            agent_config_dict = getattr(self.config, 'synonym_agent', {})
            
            # Create AgentConfig with proper fields
            agent_config = AgentConfig(
                name=agent_config_dict.get('name', 'synonym_resolution_agent'),
                description=agent_config_dict.get('description', 'Specialized agent for protein product synonym resolution'),
                model=agent_config_dict.get('model', 'gpt-4'),
                temperature=agent_config_dict.get('temperature', 0.1),
                max_tokens=agent_config_dict.get('max_tokens', 1000),
                system_prompt=agent_config_dict.get('system_prompt', '''You are a specialized bioinformatics agent focused on identifying protein product synonyms.
You have expert knowledge of:
- International Committee on Taxonomy of Viruses (ICTV) nomenclature standards
- Viral protein naming conventions across different databases
- Common abbreviations and alternative names for viral proteins
- Polyprotein processing and fragment relationships

Your task is to identify synonym relationships with high accuracy and provide confidence scores.
Always prioritize ICTV standard nomenclature when available.'''),
                # Add required agent_card for A2A compliance
                agent_card={
                    "version": "1.0.0",
                    "purpose": "Specialized agent for protein product synonym resolution",
                    "detailed_description": "This agent provides expert bioinformatics capabilities for identifying protein product synonyms and grouping them according to ICTV standards.",
                    "domain": "bioinformatics",
                    "expertise_level": "expert",
                    "input_format": {
                        "primary_mode": "json",
                        "supported_modes": ["json", "text"],
                        "content_types": ["application/json", "text/plain"],
                        "format_schema": {
                            "type": "object",
                            "required_fields": {
                                "prompt": {
                                    "type": "string",
                                    "description": "Natural language prompt for synonym resolution"
                                }
                            }
                        }
                    },
                    "output_format": {
                        "primary_mode": "json",
                        "supported_modes": ["json"],
                        "content_types": ["application/json"],
                        "format_schema": {
                            "type": "object",
                            "guaranteed_fields": {
                                "response": {
                                    "type": "string",
                                    "description": "JSON response with synonym groups"
                                }
                            }
                        }
                    },
                    "capabilities": {
                        "streaming": False,
                        "push_notifications": False,
                        "state_transition_history": False,
                        "multi_turn_conversation": False,
                        "context_retention": False,
                        "tool_usage": False,
                        "delegation": False,
                        "collaboration": False
                    },
                    "performance": {
                        "typical_response_time": "5-15 seconds",
                        "max_response_time": "60 seconds",
                        "memory_usage": "200-500 MB",
                        "cpu_requirements": "Medium"
                    }
                }
            )
            
            # Create agent using SimpleAgent.from_config pattern
            self.synonym_agent = SimpleAgent.from_config(agent_config)
            
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info("✅ Created LLM agent for synonym resolution")
                
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ Failed to create synonym agent: {e}")
            raise
    
    def _create_cache_manager(self) -> None:
        """Create cache manager for synonym resolution data"""
        try:
            from nanobrain.library.workflows.chatbot_viral_integration.steps.virus_name_resolution_step import CacheManager
            
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
        Process method implementing cache-based synonym resolution
        
        Args:
            input_data: Must contain virus_species, annotated_fasta, and protein_annotations
            
        Returns:
            Dict with standardized annotations and synonym resolution results
        """
        if hasattr(self, 'nb_logger') and self.nb_logger:
            self.nb_logger.info("🔄 Processing annotation mapping step with cache-based synonym resolution")
        
        virus_species = input_data.get('virus_species')
        annotated_fasta = input_data.get('annotated_fasta')
        protein_annotations = input_data.get('protein_annotations', [])
        unique_protein_products = input_data.get('unique_protein_products', [])
        
        if not virus_species:
            raise ValueError("No virus species provided")
        if not annotated_fasta:
            raise ValueError("No FASTA content provided")
        
        # Check cache for ICTV standards (NO hardcoded cache keys, NO session ID)
        ictv_cache_key = self._generate_ictv_cache_key(virus_species)
        ictv_standards = await self.cache_manager.get(ictv_cache_key)
        
        # Check cache for existing synonym mappings
        synonym_cache_key = self._generate_synonym_cache_key(virus_species)
        synonym_groups = await self.cache_manager.get(synonym_cache_key)
        
        if not ictv_standards or not synonym_groups:
            # Cache miss - use LLM for synonym resolution
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"💾 Cache miss for {virus_species}, using LLM for synonym resolution")
            
            ictv_standards, synonym_groups = await self._llm_based_synonym_resolution(
                virus_species=virus_species,
                protein_products=unique_protein_products
            )
            
            # Cache the results
            await self.cache_manager.set(ictv_cache_key, ictv_standards)
            await self.cache_manager.set(synonym_cache_key, synonym_groups)
            
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"💾 Cached ICTV standards and synonym groups for: {virus_species}")
        else:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"💾 Cache hit for {virus_species} synonym resolution")
        
        # Apply synonym resolution to annotations
        processed_annotations = self._apply_synonym_resolution(
            annotations=protein_annotations,
            synonym_groups=synonym_groups,
            virus_species=virus_species
        )
        
        # Save FASTA content to file and update with canonical names
        fasta_file_path = await self._save_fasta_content(annotated_fasta, virus_species)
        updated_fasta_path = await self._update_fasta_with_canonical_names(
            fasta_file_path=fasta_file_path,
            synonym_groups=synonym_groups
        )
        
        return {
            'virus_species': virus_species,
            'standardized_annotations': processed_annotations,
            'canonical_fasta_path': updated_fasta_path,
            'original_fasta_path': fasta_file_path,
            'synonym_groups': synonym_groups,
            'ictv_standards': ictv_standards,
            'original_product_count': len(unique_protein_products),
            'reduced_product_count': len(synonym_groups),
            'cluster_reduction_achieved': len(unique_protein_products) > len(synonym_groups),
            'reduction_percentage': ((len(unique_protein_products) - len(synonym_groups)) / len(unique_protein_products) * 100) if unique_protein_products else 0
        }
    
    def _generate_ictv_cache_key(self, virus_species: str) -> str:
        """Generate ICTV cache key dynamically (NO hardcoding, NO session ID)"""
        normalized_name = virus_species.lower().replace(' ', '_').replace('-', '_')
        # Remove special characters
        import re
        normalized_name = re.sub(r'[^a-z0-9_]', '', normalized_name)
        return f"ictv_standards_{normalized_name}"
    
    def _generate_synonym_cache_key(self, virus_species: str) -> str:
        """Generate synonym cache key dynamically (NO hardcoding, NO session ID)"""
        normalized_name = virus_species.lower().replace(' ', '_').replace('-', '_')
        # Remove special characters
        import re
        normalized_name = re.sub(r'[^a-z0-9_]', '', normalized_name)
        return f"synonym_groups_{normalized_name}"
    
    async def _llm_based_synonym_resolution(self, virus_species: str, protein_products: List[str]) -> Tuple[Dict, Dict]:
        """Use LLM agent for synonym resolution"""
        try:
            # Get prompt template from configuration (NO hardcoded mappings)
            prompt_template = getattr(self.config, 'synonym_resolution_prompt', 
                '''Analyze the following protein products from {virus_species} and create canonical ICTV-compliant synonym groups:

Protein Products:
{protein_products}

Create synonym groups that:
1. Follow ICTV naming conventions
2. Group functionally equivalent proteins
3. Reduce redundancy in protein annotations
4. Maintain biological accuracy

Return JSON format:
{{
  "ictv_standards": {{
    "canonical_protein_names": ["polyprotein", "capsid protein", "envelope protein"],
    "naming_guidelines": "ICTV guidelines summary"
  }},
  "synonym_groups": {{
    "canonical_name_1": ["synonym1", "synonym2", "synonym3"],
    "canonical_name_2": ["synonym4", "synonym5"]
  }}
}}

Aim to reduce the number of unique protein products by at least 50%.
Return only the JSON response.''')
            
            formatted_prompt = prompt_template.format(
                virus_species=virus_species,
                protein_products='\n'.join(protein_products[:50])  # Limit to avoid prompt length issues
            )
            
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"🤖 Sending synonym resolution prompt to LLM")
            
            # Process using LLM agent
            response = await self.synonym_agent.process({
                'prompt': formatted_prompt,
                'expected_format': 'json',
                'virus_species': virus_species
            })
            
            # Parse LLM response
            return self._parse_synonym_response(response)
            
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ LLM synonym resolution failed: {e}")
            
            # NO FALLBACKS - Workflow must fail when synonym resolution cannot be completed
            raise ValueError(
                f"Synonym resolution failed and cannot proceed without proper resolution. "
                f"LLM processing failed: {e}. "
                f"Please ensure proper agent configuration and data availability."
            )
    
    def _parse_synonym_response(self, response: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """Parse LLM response and extract synonym resolution information"""
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
                    
                    # Extract ICTV standards and synonym groups
                    ictv_standards = parsed.get('ictv_standards', {})
                    synonym_groups = parsed.get('synonym_groups', {})
                    
                    if ictv_standards and synonym_groups:
                        return ictv_standards, synonym_groups
                    else:
                        if hasattr(self, 'nb_logger') and self.nb_logger:
                            self.nb_logger.warning(f"⚠️ Incomplete LLM response for synonym resolution")
                        raise ValueError(
                            "Incomplete LLM response for synonym resolution. "
                            "Cannot proceed without complete ICTV standards and synonym groups."
                        )
                else:
                    if hasattr(self, 'nb_logger') and self.nb_logger:
                        self.nb_logger.warning(f"⚠️ Non-JSON response from LLM")
                    raise ValueError(
                        "Non-JSON response from LLM. "
                        "Cannot proceed without properly formatted synonym resolution data."
                    )
                    
            except json.JSONDecodeError as e:
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.error(f"❌ JSON parsing failed: {e}")
                raise ValueError(
                    f"JSON parsing failed for synonym resolution: {e}. "
                    f"Cannot proceed without valid response format."
                )
                    
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ Response parsing failed: {e}")
            raise ValueError(
                f"Response parsing failed for synonym resolution: {e}. "
                f"Cannot proceed without successful data parsing."
            )
    
    def _apply_synonym_resolution(self, annotations: List[Dict], synonym_groups: Dict, virus_species: str) -> List[Dict]:
        """Apply synonym resolution to protein annotations"""
        processed_annotations = []
        
        # Create reverse lookup for synonyms
        synonym_lookup = {}
        for canonical_name, synonyms in synonym_groups.items():
            for synonym in synonyms:
                synonym_lookup[synonym.lower()] = canonical_name
        
        for annotation in annotations:
            processed_annotation = annotation.copy()
            
            # Get original product name
            original_product = annotation.get('product', '')
            
            # Look up canonical name
            canonical_product = synonym_lookup.get(original_product.lower(), original_product)
            
            # Update annotation with canonical name
            processed_annotation['canonical_product'] = canonical_product
            processed_annotation['original_product'] = original_product
            processed_annotation['synonym_resolved'] = canonical_product != original_product
            
            processed_annotations.append(processed_annotation)
        
        return processed_annotations
    
    async def _save_fasta_content(self, fasta_content: str, virus_species: str) -> str:
        """Save FASTA content to file and return path"""
        cache_dir = Path(getattr(self.config, 'cache_directory', 'data/annotation_cache'))
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename without session ID
        normalized_name = virus_species.lower().replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{normalized_name}_original_{timestamp}.fasta"
        
        file_path = cache_dir / filename
        
        with open(file_path, 'w') as f:
            f.write(fasta_content)
        
        return str(file_path)
    
    async def _update_fasta_with_canonical_names(self, fasta_file_path: str, synonym_groups: Dict) -> str:
        """Update FASTA file with canonical protein names"""
        try:
            # Read original FASTA content
            with open(fasta_file_path, 'r') as f:
                original_content = f.read()
            
            # Apply synonym resolution to headers
            updated_content = self._apply_synonyms_to_fasta(original_content, synonym_groups)
            
            # Save updated FASTA file
            base_path = Path(fasta_file_path)
            updated_path = base_path.parent / f"{base_path.stem}_canonical{base_path.suffix}"
            
            with open(updated_path, 'w') as f:
                f.write(updated_content)
            
            return str(updated_path)
            
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ Failed to update FASTA with canonical names: {e}")
            return fasta_file_path  # Return original path if update fails
    
    def _apply_synonyms_to_fasta(self, fasta_content: str, synonym_groups: Dict) -> str:
        """Apply synonym resolution to FASTA headers"""
        # Create reverse lookup for synonyms
        synonym_lookup = {}
        for canonical_name, synonyms in synonym_groups.items():
            for synonym in synonyms:
                synonym_lookup[synonym.lower()] = canonical_name
        
        lines = fasta_content.split('\n')
        updated_lines = []
        
        for line in lines:
            if line.startswith('>'):
                # This is a header line - apply synonym resolution
                header = line[1:]  # Remove '>' prefix
                
                # Extract product information from header
                # Typical header format: >patric_id|product|genome_id
                parts = header.split('|')
                
                if len(parts) >= 2:
                    product = parts[1]
                    canonical_product = synonym_lookup.get(product.lower(), product)
                    
                    # Update header with canonical product name
                    parts[1] = canonical_product
                    updated_header = '>' + '|'.join(parts)
                    updated_lines.append(updated_header)
                else:
                    # Keep original header if format is unexpected
                    updated_lines.append(line)
            else:
                # This is a sequence line - keep as is
                updated_lines.append(line)
        
        return '\n'.join(updated_lines) 