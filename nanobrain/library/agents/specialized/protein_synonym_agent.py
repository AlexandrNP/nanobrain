"""
Protein Synonym Identification Agent

Specialized agent for identifying protein product synonyms with dynamic ICTV standards.
Uses the core PromptTemplateManager for all prompt handling.
"""

from typing import Dict, List, Tuple, Any, Optional, Set
from pathlib import Path
import json
import hashlib
import time
import logging

from nanobrain.library.agents.specialized.base import SimpleSpecializedAgent
from nanobrain.core import AgentConfig

logger = logging.getLogger(__name__)


class ProteinSynonymAgent(SimpleSpecializedAgent):
    """
    Specialized agent for identifying protein product synonyms using LLM.
    
    This agent analyzes protein product names and identifies synonyms
    with confidence scoring, using dynamically inferred ICTV nomenclature.
    """
    
    def __init__(self, config: AgentConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        # Configuration - read from top-level AgentConfig fields
        self.min_confidence_threshold = getattr(config, 'min_confidence_threshold', 0.8)
        self.max_products_per_request = getattr(config, 'max_products_per_request', 200)
        self.use_ictv_standards = getattr(config, 'use_ictv_standards', True)
        
        # ICTV standards cache
        cache_dir = getattr(config, 'cache_dir', 'data/ictv_cache')
        self.ictv_cache_dir = Path(cache_dir)
        self.ictv_cache_dir.mkdir(exist_ok=True, parents=True)
        self.ictv_cache_ttl_days = getattr(config, 'ictv_cache_ttl_days', 30)
        
        # Verify prompt manager is available
        if not self.prompt_manager:
            logger.warning("No prompt templates configured for ProteinSynonymAgent")
    
    @classmethod
    def from_config(cls, config: AgentConfig, **kwargs) -> 'ProteinSynonymAgent':
        """Create agent from configuration following mandatory from_config pattern."""
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Convert dictionary to AgentConfig if needed
        if isinstance(config, dict):
            logger.debug(f"Converting dictionary config to AgentConfig for {cls.__name__}")
            from nanobrain.core.agent import AgentConfig
            config = AgentConfig(**config)
        
        # Step 1: Validate configuration schema
        cls.validate_config_schema(config)
        
        # Step 2: Extract component-specific configuration  
        component_config = cls.extract_component_config(config)
        
        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 4: Create instance
        instance = cls.create_instance(config, component_config, dependencies)
        
        # Step 5: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
    
    async def _process_specialized_request(self, input_text: str, **kwargs) -> Optional[str]:
        """
        Process specialized requests for protein synonym identification.
        
        This method handles direct synonym identification requests.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional parameters including:
                - products: List of product names to analyze
                - virus_info: Dictionary with virus metadata
                
        Returns:
            JSON string with synonym groups if handled, None otherwise
        """
        # Check if this is a synonym identification request
        if 'products' in kwargs and 'virus_info' in kwargs:
            products = kwargs['products']
            virus_info = kwargs['virus_info']
            
            # Perform synonym identification
            synonym_groups = await self.identify_synonyms(products, virus_info)
            
            # Return as JSON
            return json.dumps({
                'synonym_groups': synonym_groups,
                'products_analyzed': len(products),
                'groups_found': len(synonym_groups)
            }, indent=2)
        
        # Not a specialized request, return None to use LLM
        return None
    
    async def identify_synonyms(self, 
                              product_names: List[str], 
                              virus_info: Dict[str, str]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Identify synonyms among protein product names using three-phase approach.
        
        Phase 1: Infer/retrieve ICTV standards for the specific virus
        Phase 2: Extract key terms and patterns
        Phase 3: Identify relationships globally using ICTV context
        
        Args:
            product_names: List of protein product names to analyze
            virus_info: Dictionary containing virus metadata (genus, species, strain)
            
        Returns:
            Dictionary mapping canonical names to list of (synonym, confidence) tuples
        """
        
        # Phase 1: Get ICTV standards for this virus
        ictv_standards = await self._get_ictv_standards(virus_info, product_names)
        
        # Phase 2: Extract key terms and create initial groupings
        initial_groups = await self._extract_key_terms(product_names, ictv_standards)
        
        # Phase 3: Global synonym analysis with ICTV context
        if len(product_names) <= self.max_products_per_request:
            # Small dataset: analyze all at once
            synonym_groups = await self._analyze_synonyms_global(
                product_names, initial_groups, ictv_standards
            )
        else:
            # Large dataset: use smart grouping based on key terms
            synonym_groups = await self._analyze_synonyms_smart(
                product_names, initial_groups, ictv_standards
            )
        
        # Apply ICTV standards if enabled
        if self.use_ictv_standards and ictv_standards:
            synonym_groups = self._apply_ictv_standards(synonym_groups, ictv_standards)
        
        return synonym_groups
    
    async def _get_ictv_standards(self, 
                                virus_info: Dict[str, str], 
                                sample_products: List[str]) -> Dict[str, str]:
        """
        Get ICTV standards for a specific virus, either from cache or by inference.
        
        Args:
            virus_info: Contains genus, species, strain information
            sample_products: Sample product names from the dataset for context
            
        Returns:
            Dictionary mapping abbreviated names to ICTV standard full names
        """
        
        # Generate cache key from virus info
        cache_key = self._generate_ictv_cache_key(virus_info)
        
        # Check cache first
        cached_standards = self._load_ictv_cache(cache_key)
        if cached_standards is not None:
            logger.info(f"📦 Using cached ICTV standards for {virus_info.get('genus', 'unknown')} {virus_info.get('species', '')}")
            return cached_standards
        
        # Infer ICTV standards using LLM
        logger.info(f"🔍 Inferring ICTV standards for {virus_info.get('genus', 'unknown')} {virus_info.get('species', '')}")
        
        prompt = await self._create_ictv_inference_prompt(virus_info, sample_products)
        
        # Execute the prompt using the agent's process method
        response = await self.process(prompt)
        
        # Parse and validate ICTV standards
        ictv_standards = self._parse_ictv_response(response)
        
        # Cache the results
        self._save_ictv_cache(cache_key, ictv_standards)
        
        return ictv_standards
    
    async def _create_ictv_inference_prompt(self, 
                                          virus_info: Dict[str, str], 
                                          sample_products: List[str]) -> str:
        """Create prompt for inferring ICTV standards using core template manager."""
        
        if not self.prompt_manager:
            return self._get_fallback_ictv_prompt(virus_info, sample_products)
        
        try:
            # Use the core prompt manager
            return self.get_prompt(
                'ictv_inference',
                genus=virus_info.get('genus', 'Unknown'),
                species=virus_info.get('species', 'Unknown'),
                strain=virus_info.get('strain', 'Unknown'),
                product_sample=json.dumps(sample_products[:30], indent=2)
            )
        except KeyError:
            logger.warning("ICTV inference prompt template not found, using fallback")
            return self._get_fallback_ictv_prompt(virus_info, sample_products)
    
    async def _extract_key_terms(self, 
                               products: List[str], 
                               ictv_standards: Dict[str, str]) -> Dict[str, Set[str]]:
        """Extract key protein terms from all products with ICTV context."""
        
        if not self.prompt_manager:
            logger.warning("No prompt manager available for key term extraction")
            return {}
        
        # Build ICTV context
        ictv_context = ""
        if ictv_standards and self.has_prompt_template('ictv_standards_context'):
            try:
                ictv_context = self.get_prompt(
                    'ictv_standards_context',
                    ictv_standards=json.dumps(ictv_standards, indent=2)
                )
            except Exception as e:
                logger.warning(f"Failed to get ICTV context: {e}")
        
        try:
            # Get the prompt
            prompt = self.get_prompt(
                'key_term_extraction',
                products=json.dumps(products, indent=2),
                ictv_context=ictv_context
            )
            
            response = await self.process(prompt)
            
            return self._parse_key_terms(response)
            
        except KeyError:
            logger.warning("Key term extraction prompt template not found")
            return {}
    
    async def _analyze_synonyms_global(self, 
                                     products: List[str], 
                                     initial_groups: Dict[str, Set[str]],
                                     ictv_standards: Dict[str, str]) -> Dict[str, List[Tuple[str, float]]]:
        """Analyze all products globally for synonym relationships."""
        
        if not self.prompt_manager:
            logger.warning("No prompt manager available for synonym analysis")
            return {}
        
        # Group products by their key terms for context
        grouped_products = self._group_by_key_terms(products, initial_groups)
        
        # Build ICTV context
        ictv_context = ""
        if ictv_standards and self.has_prompt_template('ictv_standards_context'):
            try:
                ictv_context = self.get_prompt(
                    'ictv_standards_context',
                    ictv_standards=json.dumps(ictv_standards, indent=2)
                )
            except Exception:
                pass
        
        try:
            prompt = self.get_prompt(
                'global_synonym_analysis',
                grouped_products=json.dumps(grouped_products, indent=2),
                ictv_context=ictv_context
            )
            
            response = await self.process(prompt)
            
            return self._parse_synonym_response(response, products)
            
        except KeyError:
            logger.warning("Global synonym analysis prompt template not found")
            return {}
    
    async def _analyze_synonyms_smart(self, 
                                    products: List[str], 
                                    initial_groups: Dict[str, Set[str]],
                                    ictv_standards: Dict[str, str]) -> Dict[str, List[Tuple[str, float]]]:
        """Smart analysis for large datasets using key term grouping."""
        
        # Group products by core protein identifier
        protein_groups = self._group_by_core_protein(products, initial_groups)
        
        # Analyze each protein group
        all_synonyms = {}
        
        for core_protein, product_list in protein_groups.items():
            if len(product_list) > 1:  # Only analyze if there are potential synonyms
                group_synonyms = await self._analyze_protein_group(core_protein, product_list, ictv_standards)
                all_synonyms.update(group_synonyms)
        
        # Cross-reference phase: check for relationships between groups
        cross_references = await self._analyze_cross_references(protein_groups, ictv_standards)
        all_synonyms = self._merge_cross_references(all_synonyms, cross_references)
        
        return all_synonyms
    
    async def _analyze_cross_references(self, 
                                      protein_groups: Dict[str, List[str]],
                                      ictv_standards: Dict[str, str]) -> List[Dict[str, Any]]:
        """Analyze potential relationships between different protein groups."""
        
        if not self.prompt_manager or not self.has_prompt_template('cross_reference_analysis'):
            return []
        
        # Get representative samples from each group
        group_samples = {
            core: products[:5]  # Top 5 from each group
            for core, products in protein_groups.items()
        }
        
        try:
            prompt = self.get_prompt(
                'cross_reference_analysis',
                group_samples=json.dumps(group_samples, indent=2)
            )
            
            response = await self.process(prompt)
            
            return self._parse_cross_references(response)
            
        except Exception as e:
            logger.warning(f"Cross-reference analysis failed: {e}")
            return []
    
    # Helper methods for ICTV cache management
    
    def _generate_ictv_cache_key(self, virus_info: Dict[str, str]) -> str:
        """Generate unique cache key for virus-specific ICTV standards."""
        
        key_parts = [
            virus_info.get('genus', 'unknown'),
            virus_info.get('species', ''),
            virus_info.get('family', ''),
            'v1'  # Version for cache invalidation
        ]
        
        key_string = '|'.join(filter(None, key_parts)).lower()
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _load_ictv_cache(self, cache_key: str) -> Optional[Dict[str, str]]:
        """Load ICTV standards from cache if available and not expired."""
        
        cache_file = self.ictv_cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            # Check age
            age_days = (time.time() - cache_file.stat().st_mtime) / 86400
            if age_days < self.ictv_cache_ttl_days:
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                        return cache_data.get('ictv_standards', {})
                except Exception as e:
                    logger.warning(f"Failed to load ICTV cache: {e}")
        
        return None
    
    def _save_ictv_cache(self, cache_key: str, ictv_standards: Dict[str, str]) -> None:
        """Save ICTV standards to cache."""
        
        cache_file = self.ictv_cache_dir / f"{cache_key}.json"
        
        try:
            cache_data = {
                'ictv_standards': ictv_standards,
                'timestamp': time.time(),
                'version': 'v1'
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            logger.info(f"💾 Cached ICTV standards for future use")
        except Exception as e:
            logger.warning(f"Failed to save ICTV cache: {e}")
    
    # Fallback methods
    
    def _get_fallback_ictv_prompt(self, virus_info: Dict[str, str], sample_products: List[str]) -> str:
        """Fallback prompt if template not available."""
        
        return f"""Based on the International Committee on Taxonomy of Viruses (ICTV) standards,
identify the standardized protein nomenclature for {virus_info.get('genus', 'unknown genus')} viruses,
specifically {virus_info.get('species', 'this species')}.

Virus Information:
- Genus: {virus_info.get('genus', 'Unknown')}
- Species: {virus_info.get('species', 'Unknown')}
- Strain: {virus_info.get('strain', 'Unknown')}

Sample protein products from this dataset:
{json.dumps(sample_products[:30], indent=2)}

Please provide the ICTV standard nomenclature mapping for this virus, including:
1. Nonstructural proteins (e.g., nsP1 -> nonstructural protein 1)
2. Structural proteins (e.g., E1 -> envelope protein E1)
3. Polyproteins (e.g., P1234 -> nonstructural polyprotein P1234)
4. Any virus-specific proteins

Return JSON format:
{{
    "ictv_standards": {{
        "abbreviated_form": "full_ictv_name",
        "nsP1": "nonstructural protein 1",
        ...
    }},
    "virus_specific_notes": "any special considerations for this virus",
    "confidence": "high/medium/low"
}}"""
    
    # Parsing and utility methods (to be implemented based on response format)
    
    def _parse_ictv_response(self, response: str) -> Dict[str, str]:
        """Parse ICTV standards from LLM response."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('ictv_standards', {})
            
            # Fallback: try to parse the entire response as JSON
            data = json.loads(response)
            return data.get('ictv_standards', {})
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse ICTV response: {e}")
            return {}
    
    def _parse_key_terms(self, response: str) -> Dict[str, Set[str]]:
        """Parse key terms from LLM response."""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            # Build key term groups
            key_term_groups = {}
            
            for item in data.get('product_analysis', []):
                product = item.get('product', '')
                core_id = item.get('core_id', '')
                key_terms = item.get('key_terms', [])
                
                if core_id:
                    if core_id not in key_term_groups:
                        key_term_groups[core_id] = set()
                    key_term_groups[core_id].add(product)
                    
                    # Also group by key terms
                    for term in key_terms:
                        if term not in key_term_groups:
                            key_term_groups[term] = set()
                        key_term_groups[term].add(product)
            
            return key_term_groups
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse key terms response: {e}")
            return {}
    
    def _parse_synonym_response(self, response: str, products: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """Parse synonym groups from LLM response."""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            # Build synonym groups
            synonym_groups = {}
            
            for group in data.get('synonym_groups', []):
                canonical = group.get('canonical', '')
                if not canonical:
                    continue
                
                synonyms = []
                for syn in group.get('synonyms', []):
                    name = syn.get('name', '')
                    confidence = float(syn.get('confidence', 0.5))
                    
                    if name and name != canonical and confidence >= self.min_confidence_threshold:
                        synonyms.append((name, confidence))
                
                if synonyms:
                    synonym_groups[canonical] = synonyms
            
            return synonym_groups
            
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.warning(f"Failed to parse synonym response: {e}")
            return {}
    
    def _parse_cross_references(self, response: str) -> List[Dict[str, Any]]:
        """Parse cross-references from LLM response."""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            return data.get('cross_references', [])
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse cross-references: {e}")
            return []
    
    def _group_by_key_terms(self, products: List[str], initial_groups: Dict[str, Set[str]]) -> Dict[str, List[str]]:
        """Group products by their key terms."""
        grouped = {}
        
        # Invert the initial_groups mapping
        product_to_terms = {}
        for term, product_set in initial_groups.items():
            for product in product_set:
                if product not in product_to_terms:
                    product_to_terms[product] = set()
                product_to_terms[product].add(term)
        
        # Group products with similar key terms
        for product in products:
            terms = product_to_terms.get(product, set())
            if terms:
                # Use the most specific term as the group key
                group_key = min(terms, key=lambda t: len(initial_groups.get(t, [])))
                if group_key not in grouped:
                    grouped[group_key] = []
                grouped[group_key].append(product)
            else:
                # Products without key terms go to 'other'
                if 'other' not in grouped:
                    grouped['other'] = []
                grouped['other'].append(product)
        
        return grouped
    
    def _group_by_core_protein(self, products: List[str], initial_groups: Dict[str, Set[str]]) -> Dict[str, List[str]]:
        """Group products by core protein identifier."""
        import re
        
        protein_groups = {}
        
        # Common protein patterns
        protein_patterns = [
            r'\b(nsP\d+)\b',  # nsP1, nsP2, etc.
            r'\b(nsp\d+)\b',  # nsp1, nsp2, etc.
            r'\b([EC]\d+)\b',  # E1, E2, C, etc.
            r'\b(NS\d+[AB]?)\b',  # NS1, NS2A, etc.
            r'\b(VP\d+)\b',  # VP1, VP2, etc.
            r'\b(ORF\d+[ab]?)\b',  # ORF1a, ORF1b, etc.
            r'\b(P\d+)\b',  # P1, P2, etc.
            r'\b(capsid|envelope|spike|helicase|protease|polymerase)\b',
        ]
        
        for product in products:
            matched = False
            
            # Try to match against known patterns
            for pattern in protein_patterns:
                match = re.search(pattern, product, re.IGNORECASE)
                if match:
                    core_protein = match.group(1).upper()
                    if core_protein not in protein_groups:
                        protein_groups[core_protein] = []
                    protein_groups[core_protein].append(product)
                    matched = True
                    break
            
            # If no pattern matched, use key terms from initial groups
            if not matched:
                for term, product_set in initial_groups.items():
                    if product in product_set:
                        if term not in protein_groups:
                            protein_groups[term] = []
                        protein_groups[term].append(product)
                        matched = True
                        break
            
            # Last resort: group as 'unknown'
            if not matched:
                if 'unknown' not in protein_groups:
                    protein_groups['unknown'] = []
                protein_groups['unknown'].append(product)
        
        return protein_groups
    
    async def _analyze_protein_group(self, core_protein: str, products: List[str], ictv_standards: Dict[str, str]) -> Dict[str, List[Tuple[str, float]]]:
        """Analyze a single protein group for synonyms."""
        if len(products) < 2:
            return {}
        
        # Use the global analysis method for the group
        return await self._analyze_synonyms_global(products, {core_protein: set(products)}, ictv_standards)
    
    def _merge_cross_references(self, synonyms: Dict[str, List[Tuple[str, float]]], cross_refs: List[Dict[str, Any]]) -> Dict[str, List[Tuple[str, float]]]:
        """Merge cross-references into synonym groups."""
        # Create a mapping of all products to their canonical forms
        product_to_canonical = {}
        for canonical, syn_list in synonyms.items():
            product_to_canonical[canonical] = canonical
            for syn, _ in syn_list:
                product_to_canonical[syn] = canonical
        
        # Process cross-references
        for ref in cross_refs:
            group1 = ref.get('group1', '')
            group2 = ref.get('group2', '')
            relationship = ref.get('relationship', '')
            confidence = float(ref.get('confidence', 0.5))
            
            if confidence < self.min_confidence_threshold:
                continue
            
            # Handle different relationship types
            if relationship == 'alternative_name':
                # Merge the groups
                canonical1 = product_to_canonical.get(group1, group1)
                canonical2 = product_to_canonical.get(group2, group2)
                
                if canonical1 != canonical2:
                    # Add group2 as synonym of group1
                    if canonical1 not in synonyms:
                        synonyms[canonical1] = []
                    synonyms[canonical1].append((canonical2, confidence))
                    
                    # Move all synonyms of group2 to group1
                    if canonical2 in synonyms:
                        for syn, syn_conf in synonyms[canonical2]:
                            synonyms[canonical1].append((syn, min(syn_conf, confidence)))
                        del synonyms[canonical2]
        
        return synonyms
    
    def _apply_ictv_standards(self, synonym_groups: Dict[str, List[Tuple[str, float]]], ictv_standards: Dict[str, str]) -> Dict[str, List[Tuple[str, float]]]:
        """Apply ICTV standards to synonym groups."""
        if not ictv_standards:
            return synonym_groups
        
        updated_groups = {}
        
        for canonical, synonyms in synonym_groups.items():
            # Check if canonical name has an ICTV standard
            ictv_canonical = None
            
            # Direct match
            for abbrev, full_name in ictv_standards.items():
                if abbrev.lower() in canonical.lower() or canonical.lower() in full_name.lower():
                    ictv_canonical = full_name
                    break
            
            if ictv_canonical:
                # Use ICTV standard as canonical
                updated_groups[ictv_canonical] = [(canonical, 0.95)]  # High confidence for original canonical
                updated_groups[ictv_canonical].extend(synonyms)
            else:
                # Keep original canonical
                updated_groups[canonical] = synonyms
        
        return updated_groups 