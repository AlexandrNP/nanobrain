"""
BV-BRC Data Acquisition Step (Steps 1-7)

Implements the first 7 steps of the Alphavirus workflow:
1. Download all Alphavirus genomes from BV-BRC
2. Filter genomes by size (8KB-15KB range)
3-4. Extract unique protein MD5s and deduplicate
5. Get feature sequences for MD5s
6. Get annotations for unique MD5 sequences
7. Create annotated FASTA file

Based on BV-BRC CLI documentation:
https://www.bv-brc.org/docs/cli_tutorial/cli_getting_started.html
"""

import asyncio
import tempfile
import time
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import hashlib
import re

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.logging_system import get_logger
from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCTool, BVBRCConfig
import yaml
from pathlib import Path


class GenomeData:
    """Container for genome information"""
    
    def __init__(self, genome_id: str, genome_length: int, genome_name: str, 
                 taxon_lineage: Optional[str] = None):
        self.genome_id = genome_id
        self.genome_length = genome_length
        self.genome_name = genome_name
        self.taxon_lineage = taxon_lineage or ""
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'genome_id': self.genome_id,
            'genome_length': self.genome_length,
            'genome_name': self.genome_name,
            'taxon_lineage': self.taxon_lineage
        }


class ProteinData:
    """Container for protein information"""
    
    def __init__(self, patric_id: str, aa_sequence_md5: str, genome_id: str,
                 product: Optional[str] = None, gene: Optional[str] = None):
        self.patric_id = patric_id
        self.aa_sequence_md5 = aa_sequence_md5
        self.genome_id = genome_id
        self.product = product or ""
        self.gene = gene or ""
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'patric_id': self.patric_id,
            'aa_sequence_md5': self.aa_sequence_md5,
            'genome_id': self.genome_id,
            'product': self.product,
            'gene': self.gene
        }


class SequenceData:
    """Container for sequence information"""
    
    def __init__(self, aa_sequence_md5: str, aa_sequence: str):
        self.aa_sequence_md5 = aa_sequence_md5
        self.aa_sequence = aa_sequence
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'aa_sequence_md5': self.aa_sequence_md5,
            'aa_sequence': self.aa_sequence
        }


class AnnotationData:
    """Container for annotation information"""
    
    def __init__(self, aa_sequence_md5: str, product: str, gene: str = "",
                 refseq_locus_tag: str = "", go: str = "", ec: str = "", 
                 pathway: str = ""):
        self.aa_sequence_md5 = aa_sequence_md5
        self.product = product
        self.gene = gene
        self.refseq_locus_tag = refseq_locus_tag
        self.go = go
        self.ec = ec
        self.pathway = pathway
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'aa_sequence_md5': self.aa_sequence_md5,
            'product': self.product,
            'gene': self.gene,
            'refseq_locus_tag': self.refseq_locus_tag,
            'go': self.go,
            'ec': self.ec,
            'pathway': self.pathway
        }


class BVBRCDataAcquisitionStep(Step):
    """
    BV-BRC Data Acquisition Step implementing workflow steps 1-7
    
    Re-architected to inherit from NanoBrain Step base class.
    Uses the corrected BV-BRC path: /Applications/BV-BRC.app/deployment/bin/
    Implements anonymous access with proper data verification.
    """
    
    def __init__(self, config: StepConfig, bvbrc_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config, **kwargs)
        self._initialize_tools(config, bvbrc_config)
        
    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any], 
                         dependencies: Dict[str, Any]) -> None:
        """Initialize BVBRCDataAcquisitionStep with tool integration via from_config pattern"""
        super()._init_from_config(config, component_config, dependencies)
        self._initialize_tools(config, None)
        
    def _initialize_tools(self, config: StepConfig, bvbrc_config: Optional[Dict[str, Any]] = None):
        """Initialize BV-BRC tool with workflow-local configuration"""
        
        # Get workflow directory path
        workflow_dir = Path(__file__).parent.parent
        tool_config_path = workflow_dir / "config" / "tools" / "bv_brc_tool.yml"
        
        # Load tool configuration from workflow-local YAML
        if tool_config_path.exists():
            with open(tool_config_path, 'r') as f:
                tool_config_dict = yaml.safe_load(f)
            
            # Create BV-BRC tool configuration
            tool_config = BVBRCConfig(**{
                k: v for k, v in tool_config_dict.items() 
                if k in ['tool_name', 'installation_path', 'executable_path', 'genome_batch_size', 
                        'md5_batch_size', 'min_genome_length', 'max_genome_length', 'timeout_seconds', 
                        'retry_attempts', 'verify_on_init', 'progressive_scaling', 'use_cache']
            })
            
            # Create BV-BRC tool using from_config pattern
            self.bv_brc_tool = BVBRCTool.from_config(tool_config)
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"✅ BV-BRC tool loaded from workflow-local config: {tool_config_path}")
            
        else:
            # Fallback to legacy configuration
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.warning(f"⚠️ Workflow-local tool config not found: {tool_config_path}")
                self.nb_logger.warning("⚠️ Using legacy configuration approach")
            
            # Legacy approach with old configuration structure
            bvbrc_config_dict = getattr(config, 'bvbrc_config', {})
            if bvbrc_config:
                bvbrc_config_dict = {**bvbrc_config_dict, **bvbrc_config}
            
            if 'executable_path' not in bvbrc_config_dict:
                bvbrc_config_dict['executable_path'] = '/Applications/BV-BRC.app/deployment/bin'
            
            tool_config = BVBRCConfig(**bvbrc_config_dict)
            self.bv_brc_tool = BVBRCTool.from_config(tool_config)
        
        # Store all step configuration as dict for backward compatibility
        self.step_config = config.model_dump()
        
        # Configuration parameters from step config attributes
        self.min_genome_length = getattr(config, 'min_genome_length', 8000)
        self.max_genome_length = getattr(config, 'max_genome_length', 15000)
        
        # Use batch sizes from tool configuration (updated to 1000)
        self.genome_batch_size = self.bv_brc_tool.bv_brc_config.genome_batch_size
        self.md5_batch_size = self.bv_brc_tool.bv_brc_config.md5_batch_size
        
        # Set 20 minute timeout for BV-BRC operations
        self.timeout_seconds = getattr(config, 'timeout_seconds', 1200)  # 20 minutes
        
        # Initialize cache directory for virus-specific data - configurable path
        cache_dir_config = getattr(config, 'bvbrc_cache_directory', None)
        if cache_dir_config:
            self.cache_dir = Path(cache_dir_config)
        else:
            # Fallback to default path
            self.cache_dir = Path("data/bvbrc_cache")
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self, 'nb_logger') and self.nb_logger:
            self.nb_logger.info(f"🧬 BVBRCDataAcquisitionStep initialized with tool: {type(self.bv_brc_tool).__name__}")
            self.nb_logger.info(f"⏱️ Timeout set to {self.timeout_seconds} seconds (20 minutes)")
            self.nb_logger.info(f"💾 Cache directory: {self.cache_dir}")
    
    async def _extract_virus_name_from_query(self, user_query: str) -> str:
        """Extract the full virus name from user query using LLM"""
        if not user_query:
            return "unknown_virus"
        
        try:
            # Initialize OpenAI client (similar to Agent class)
            from openai import AsyncOpenAI
            import os
            
            # Get API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                self.nb_logger.warning("⚠️ No OpenAI API key found, using fallback extraction")
                return self._fallback_virus_extraction(user_query)
            
            # Create OpenAI client
            client = AsyncOpenAI(api_key=api_key)
            
            # Create extraction prompt
            extraction_prompt = f"""Extract the virus name from the following user query and return it in a clean, cache-friendly format.

User Query: "{user_query}"

Instructions:
1. Identify the specific virus mentioned in the query
2. Return ONLY the virus name in lowercase with underscores instead of spaces
3. Remove "virus" suffix if present
4. Use common abbreviations when appropriate (e.g., "chikv" for chikungunya, "eeev" for eastern equine encephalitis)
5. If multiple viruses are mentioned, return the primary/first one
6. If no specific virus is found, return "alphavirus" as the default

Examples:
- "Create PSSM matrix for Chikungunya virus" → "chikungunya"
- "Generate PSSM for EEEV" → "eastern_equine_encephalitis" 
- "Analyze VEEV proteins" → "venezuelan_equine_encephalitis"
- "Study Sindbis virus genome" → "sindbis"
- "Process alphavirus data" → "alphavirus"

Return only the clean virus name, nothing else:"""

            # Make LLM call
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            if response.choices and response.choices[0].message.content:
                # Clean the response
                virus_name = response.choices[0].message.content.strip().lower()
                # Remove any extra text, keep only the virus name
                virus_name = virus_name.split('\n')[0].split('.')[0].split(',')[0].strip()
                # Ensure it's filename-safe
                virus_name = virus_name.replace(' ', '_').replace('-', '_').replace('/', '_')
                # Remove any remaining punctuation except underscores
                import re
                virus_name = re.sub(r'[^\w_]', '', virus_name)
                
                self.nb_logger.info(f"🤖 LLM extracted virus name: '{virus_name}' from query: '{user_query}'")
                return virus_name if virus_name else "unknown_virus"
            
        except Exception as e:
            self.nb_logger.warning(f"⚠️ LLM virus name extraction failed: {e}")
            # Fallback to simple extraction
            
        # Fallback: Simple rule-based extraction if LLM fails
        return self._fallback_virus_extraction(user_query)
    
    def _fallback_virus_extraction(self, user_query: str) -> str:
        """Fallback virus name extraction using simple rules"""
        query_lower = user_query.lower()
        
        # Simple keyword-based extraction
        if 'chikungunya' in query_lower or 'chikv' in query_lower:
            return 'chikungunya'
        elif 'eastern equine' in query_lower or 'eeev' in query_lower:
            return 'eastern_equine_encephalitis'
        elif 'western equine' in query_lower or 'weev' in query_lower:
            return 'western_equine_encephalitis'
        elif 'venezuelan equine' in query_lower or 'veev' in query_lower:
            return 'venezuelan_equine_encephalitis'
        elif 'sindbis' in query_lower or 'sinv' in query_lower:
            return 'sindbis'
        elif 'semliki' in query_lower or 'sfv' in query_lower:
            return 'semliki_forest'
        elif 'ross river' in query_lower or 'rrv' in query_lower:
            return 'ross_river'
        elif 'mayaro' in query_lower or 'mayv' in query_lower:
            return 'mayaro'
        elif 'alphavirus' in query_lower:
            return 'alphavirus'
        else:
            # Try to extract any word before "virus"
            words = query_lower.split()
            for i, word in enumerate(words):
                if 'virus' in word and i > 0:
                    virus_word = words[i-1].replace(' ', '_').replace('-', '_')
                    # Filter out common non-virus words
                    if virus_word not in ['the', 'a', 'an', 'for', 'of', 'with', 'study', 'analyze', 'process', 'create', 'generate']:
                        return virus_word
            
            # Default fallback
            return 'alphavirus'

    async def _get_virus_cache_key(self, input_params: Dict[str, Any]) -> str:
        """Generate a cache key for virus-specific data based on user query"""
        
        # First priority: Extract virus name from user query
        user_query = input_params.get('user_query', '')
        if user_query:
            virus_name = await self._extract_virus_name_from_query(user_query)
            self.nb_logger.info(f"💾 Extracted virus name from query '{user_query}': {virus_name}")
            return virus_name
        
        # Second priority: Use target organism if specified
        target_organism = input_params.get('target_organism', '')
        if target_organism:
            virus_name = await self._extract_virus_name_from_query(target_organism)
            self.nb_logger.info(f"💾 Using target organism: {virus_name}")
            return virus_name
        
        # Third priority: Use organism parameter
        organism = input_params.get('organism', '')
        if organism:
            virus_name = await self._extract_virus_name_from_query(organism)
            self.nb_logger.info(f"💾 Using organism parameter: {virus_name}")
            return virus_name
        
        # Final fallback: Use target genus
        target_genus = input_params.get('target_genus', 'alphavirus')
        virus_name = target_genus.lower().replace(' ', '_')
        self.nb_logger.info(f"💾 Fallback to target genus: {virus_name}")
        return virus_name
    
    def _get_cached_file_path(self, cache_key: str, file_type: str) -> Path:
        """Get the path for a cached file"""
        if file_type.endswith('.fasta'):
            # For FASTA files, preserve the extension
            return self.cache_dir / f"{cache_key}_{file_type}"
        else:
            # For TSV files, add .tsv extension
            return self.cache_dir / f"{cache_key}_{file_type}.tsv"
    
    def _is_cache_valid(self, file_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cached file is valid and not too old"""
        if not file_path.exists():
            return False
        
        import time
        file_age = time.time() - file_path.stat().st_mtime
        max_age_seconds = max_age_hours * 3600
        
        return file_age < max_age_seconds
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process method required by Step base class.
        
        This implements the NanoBrain framework interface while calling the
        original execute method that contains the bioinformatics logic.
        """
        self.nb_logger.info("🔄 Processing BV-BRC data acquisition step")
        
        # Extract parameters from input_data
        input_params = {
            'target_genus': input_data.get('target_genus', 'Alphavirus'),
            'organism': input_data.get('organism', 'Alphavirus'),
            **input_data
        }
        
        # Call the original execute method
        result = await self.execute(input_params)
        
        self.nb_logger.info(f"✅ BV-BRC data acquisition completed successfully")
        return result
        
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute steps 1-7 of the BV-BRC data acquisition workflow
        
        Args:
            input_params: Input parameters including target_genus, genome_ids, or taxon_id
            
        Returns:
            Dict containing all acquired and processed data
        """
        
        step_start_time = time.time()
        target_genus = input_params.get('target_genus', 'Alphavirus')
        
        # Generate virus-specific cache key
        cache_key = await self._get_virus_cache_key(input_params)
        self.nb_logger.info(f"💾 Using cache key: {cache_key}")
        
        try:
            # Check for cached data FIRST - skip genome download if we have cached proteins
            cached_proteins_path = self._get_cached_file_path(cache_key, "proteins")
            filtered_proteins_path = self._get_cached_file_path(cache_key, "proteins.filtered")
            
            # CACHE FALLBACK: If current cache is empty, use existing good cache
            if filtered_proteins_path.exists():
                with open(filtered_proteins_path, 'r') as f:
                    line_count = sum(1 for _ in f)
                    if line_count <= 1:  # Only header or empty
                        self.nb_logger.warning(f"🔄 Current cache {cache_key} is empty, searching for existing good cache...")
                        
                        # Look for existing good cache files in the same directory
                        cache_dir = filtered_proteins_path.parent
                        for cache_file in cache_dir.glob("*_proteins.filtered.tsv"):
                            if cache_file != filtered_proteins_path:
                                with open(cache_file, 'r') as f:
                                    existing_count = sum(1 for _ in f) - 1  # Subtract header
                                    if existing_count > 30000:  # Good cache with substantial data
                                        existing_cache_key = cache_file.stem.replace('_proteins.filtered', '')
                                        self.nb_logger.info(f"✅ Found good cache {existing_cache_key} with {existing_count:,} proteins, using instead")
                                        cache_key = existing_cache_key
                                        # Update cache paths
                                        cached_proteins_path = self._get_cached_file_path(cache_key, "proteins")
                                        filtered_proteins_path = self._get_cached_file_path(cache_key, "proteins.filtered")
                                        break
            
            # Use filtered proteins if available, otherwise fall back to regular cache
            if filtered_proteins_path.exists():
                cached_proteins_path = filtered_proteins_path
                print(f"💾 FOUND EXISTING FILTERED PROTEINS CACHE: {cached_proteins_path}")
                self.nb_logger.info(f"💾 Found existing filtered proteins cache: {cached_proteins_path}")
                
                # Skip genome download - create mock genome data from cache
                self.nb_logger.info(f"🔄 Step 1: SKIPPING genome download - using existing cache")
                original_genomes = []  # We don't need genome data when using cache
                self.nb_logger.info(f"✅ Using cached data - skipped genome download")
                
            elif cached_proteins_path.exists() and self._is_cache_valid(cached_proteins_path, max_age_hours=24):
                # Check if current cache is empty and fallback to good cache
                with open(cached_proteins_path, 'r') as f:
                    line_count = sum(1 for _ in f)
                    if line_count <= 1:  # Only header or empty
                        self.nb_logger.warning(f"🔄 Current cache {cache_key} is empty, searching for existing good cache...")
                        
                        # Look for existing good cache files in the same directory
                        cache_dir = cached_proteins_path.parent
                        for cache_file in cache_dir.glob("*_proteins.tsv"):
                            if cache_file != cached_proteins_path:
                                with open(cache_file, 'r') as f:
                                    existing_count = sum(1 for _ in f) - 1  # Subtract header
                                    if existing_count > 30000:  # Good cache with substantial data
                                        existing_cache_key = cache_file.stem.replace('_proteins', '')
                                        self.nb_logger.info(f"✅ Found good cache {existing_cache_key} with {existing_count:,} proteins, using instead")
                                        cache_key = existing_cache_key
                                        # Update cache paths
                                        cached_proteins_path = self._get_cached_file_path(cache_key, "proteins")
                                        filtered_proteins_path = self._get_cached_file_path(cache_key, "proteins.filtered")
                                        break
                
                print(f"💾 FOUND EXISTING PROTEINS CACHE: {cached_proteins_path}")
                self.nb_logger.info(f"💾 Found existing proteins cache: {cached_proteins_path}")
                
                # Skip genome download - create mock genome data from cache
                self.nb_logger.info(f"🔄 Step 1: SKIPPING genome download - using existing cache")
                original_genomes = []  # We don't need genome data when using cache
                self.nb_logger.info(f"✅ Using cached data - skipped genome download")
                
            else:
                # No valid cache found - proceed with genome download
                if 'genome_ids' in input_params and input_params['genome_ids']:
                    genome_ids = input_params['genome_ids']
                    
                    # Check if we also have taxon_id for more efficient download
                    if 'taxon_id' in input_params:
                        taxon_id = input_params['taxon_id']
                        self.nb_logger.info(f"🔄 Step 1: Downloading {len(genome_ids)} specific genomes from taxon_id {taxon_id}")
                        original_genomes = await self._download_specific_genomes_by_taxon_id(genome_ids, taxon_id)
                    else:
                        # Download specific genome IDs directly
                        self.nb_logger.info(f"🔄 Step 1: Downloading {len(genome_ids)} specific genomes from BV-BRC")
                        original_genomes = await self._download_specific_genomes(genome_ids)
                    
                    self.nb_logger.info(f"✅ Downloaded {len(original_genomes)} specific genomes")
                    
                elif 'taxon_id' in input_params:
                    # Download by taxon_id
                    taxon_id = input_params['taxon_id']
                    self.nb_logger.info(f"🔄 Step 1: Downloading all genomes for taxon_id {taxon_id} from BV-BRC")
                    original_genomes = await self._download_genomes_by_taxon_id(taxon_id)
                    self.nb_logger.info(f"✅ Downloaded {len(original_genomes)} genomes for taxon_id {taxon_id}")
                    
                else:
                    # Download by genus (default for alphaviruses)
                    target_genus = input_params.get('target_genus', 'Alphavirus')
                    self.nb_logger.info(f"🔄 Step 1: Downloading {target_genus} genomes from BV-BRC")
                    original_genomes = await self._download_genomes_by_genus(target_genus)
                    self.nb_logger.info(f"✅ Downloaded {len(original_genomes)} {target_genus} genomes")
            
            # Step 2: Filter genomes by size (skip if using cache)
            if original_genomes:
                self.nb_logger.info("🔄 Step 2: Filtering genomes by size")
                filtered_genomes = await self._filter_genomes_by_size(original_genomes)
                self.nb_logger.info(f"✅ Filtered to {len(filtered_genomes)} genomes within size range ({self.min_genome_length}-{self.max_genome_length} bp)")
                
                if not filtered_genomes:
                    raise ValueError(f"No genomes found within size range")
            else:
                # Using cache - create mock filtered genomes
                self.nb_logger.info("🔄 Step 2: SKIPPING genome filtering - using cached data")
                filtered_genomes = []  # We don't need genome data when using cache
                self.nb_logger.info(f"✅ Using cached data - skipped genome filtering")
            
            # Steps 3-4: Get unique protein MD5s
            self.nb_logger.info("🔄 Steps 3-4: Extracting unique protein MD5s")
            if filtered_genomes:
                genome_ids = [g.genome_id for g in filtered_genomes]
            else:
                # Using cache - pass empty genome_ids list to trigger cache loading
                genome_ids = []
            unique_proteins = await self._get_unique_protein_md5s(genome_ids, cache_key)
            self.nb_logger.info(f"✅ Found {len(unique_proteins)} unique proteins")
            
            # Store unique_proteins and cache_key for FASTA creation and sequence retrieval
            self.unique_proteins = unique_proteins
            self.current_cache_key = cache_key
            
            # Step 5: Get feature sequences
            self.nb_logger.info("🔄 Step 5: Retrieving protein sequences")
            protein_sequences = await self._get_feature_sequences(unique_proteins)
            self.nb_logger.info(f"✅ Retrieved {len(protein_sequences)} protein sequences")
            
            # Step 6: Extract annotations from protein data (already available from p3-get-genome-features)
            self.nb_logger.info("🔄 Step 6: Extracting protein annotations from existing data")
            protein_annotations = await self._extract_annotations_from_proteins(unique_proteins)
            self.nb_logger.info(f"✅ Extracted annotations for {len(protein_annotations)} proteins")
            
            # Step 7: Create annotated FASTA with patric_id and product information
            self.nb_logger.info("🔄 Step 7: Creating annotated FASTA file with patric_id and product information")
            annotated_fasta = await self._create_annotated_fasta_with_cache(protein_sequences, protein_annotations, cache_key)
            self.nb_logger.info("✅ Created annotated FASTA file with enhanced headers")
            
            execution_time = time.time() - step_start_time
            self.nb_logger.info(f"🎉 BV-BRC data acquisition completed in {execution_time:.2f} seconds")
            
            # If we used cached data and have empty genome lists, reconstruct from protein data
            if not original_genomes and not filtered_genomes and unique_proteins:
                self.nb_logger.info("🔄 Reconstructing genome data from cached protein information")
                
                # Extract unique genome IDs from protein data
                unique_genome_ids = set()
                for protein in unique_proteins:
                    if protein.genome_id:
                        unique_genome_ids.add(protein.genome_id)
                
                # Create mock genome data based on protein cache
                reconstructed_genomes = []
                for genome_id in unique_genome_ids:
                    # Create mock genome data - we don't have length info from cache
                    genome = GenomeData(
                        genome_id=genome_id,
                        genome_length=11000,  # Typical alphavirus genome length
                        genome_name=f"Alphavirus genome {genome_id}",
                        taxon_lineage="Alphavirus"
                    )
                    reconstructed_genomes.append(genome)
                
                # Use reconstructed genomes as both original and filtered
                original_genomes = reconstructed_genomes
                filtered_genomes = reconstructed_genomes
                
                self.nb_logger.info(f"✅ Reconstructed {len(reconstructed_genomes)} genome entries from cached protein data")
            
            # Extract virus species for annotation mapping
            virus_species = input_params.get('target_genus', 'Alphavirus')
            if target_genus != 'Alphavirus':
                virus_species = target_genus
            
            # Extract unique protein products for synonym resolution
            unique_protein_products = list(set([
                annotation.product for annotation in protein_annotations 
                if annotation.product and annotation.product != 'unknown' and annotation.product != 'hypothetical protein'
            ]))
            
            return {
                'success': True,
                
                # CORRECTED FORMAT FOR ANNOTATION MAPPING STEP
                'virus_species': virus_species,
                'annotated_fasta': annotated_fasta,  # ENTIRE FASTA content
                'protein_annotations': [a.to_dict() for a in protein_annotations],
                'unique_proteins': [p.to_dict() for p in unique_proteins],
                'protein_sequences': [s.to_dict() for s in protein_sequences],
                'filtered_genomes': [g.to_dict() for g in filtered_genomes],
                'unique_protein_products': unique_protein_products,  # For synonym resolution
                
                # BACKWARD COMPATIBILITY: Keep existing fields for current tests
                'fasta_sequences': annotated_fasta,
                'sequence_count': len(protein_sequences),
                'sequence_headers': [line for line in annotated_fasta.split('\\n')[:20] if line.startswith('>')],
                'original_genomes': [g.to_dict() for g in original_genomes],
                
                # NESTED DATA: For legacy compatibility
                'protein_data': {
                    'unique_proteins': [p.to_dict() for p in unique_proteins],
                    'protein_sequences': [s.to_dict() for s in protein_sequences],
                    'protein_annotations': [a.to_dict() for a in protein_annotations],
                    'total_proteins': len(unique_proteins),
                    'sequences_retrieved': len(protein_sequences),
                    'annotations_retrieved': len(protein_annotations)
                },
                'genome_data': {
                    'original_genomes': [g.to_dict() for g in original_genomes],
                    'filtered_genomes': [g.to_dict() for g in filtered_genomes],
                    'total_genomes_downloaded': len(original_genomes),
                    'genomes_after_filtering': len(filtered_genomes)
                },
                
                # EXECUTION METADATA
                'execution_time': execution_time,
                'cache_used': bool(not original_genomes and not filtered_genomes and unique_proteins),
                'data_source': 'cache' if (not original_genomes and not filtered_genomes and unique_proteins) else 'bv_brc_download',
                
                # STATISTICS
                'statistics': {
                    'total_genomes_downloaded': len(original_genomes),
                    'genomes_after_filtering': len(filtered_genomes),
                    'unique_proteins_found': len(unique_proteins),
                    'sequences_retrieved': len(protein_sequences),
                    'annotations_retrieved': len(protein_annotations),
                    'unique_protein_products_found': len(unique_protein_products),
                    'fasta_entries': len([line for line in annotated_fasta.split('\\n') if line.startswith('>')]) if annotated_fasta else 0
                }
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ BV-BRC data acquisition failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - step_start_time
            }
            
    async def _download_specific_genomes(self, genome_ids: List[str]) -> List[GenomeData]:
        """Download specific genomes by their IDs using p3-all-genomes"""
        all_genomes = []
        
        # Process genome IDs in batches to avoid command line length limits
        batch_size = 50  # Reasonable batch size for command line
        
        for i in range(0, len(genome_ids), batch_size):
            batch = genome_ids[i:i+batch_size]
            genome_list = ",".join(batch)
            
            # Use p3-all-genomes with --in parameter to get specific genome IDs
            command_args = [
                '--attr', 'genome_id,genome_name,genome_length,taxon_lineage_names',
                '--in', f'genome_id,({genome_list})'
            ]
                
            # Execute command using the correct BV-BRC tool method
            result = await self.bv_brc_tool.execute_p3_command("p3-all-genomes", command_args)
            
            if result.returncode != 0:
                raise RuntimeError(f"p3-all-genomes command failed: {result.stderr_text}")
            
            batch_genomes = await self._parse_genome_data(result.stdout)
            all_genomes.extend(batch_genomes)
        
        self.nb_logger.info(f"📦 Retrieved {len(all_genomes)}/{len(genome_ids)} requested genomes by ID")
        
        return all_genomes
    
    async def _download_specific_genomes_by_taxon_id(self, genome_ids: List[str], taxon_id: str) -> List[GenomeData]:
        """Download specific genomes by first getting all genomes for taxon_id, then filtering by specific IDs"""
        
        # First, get all genomes for the taxon_id
        all_taxon_genomes = await self._download_genomes_by_taxon_id(taxon_id)
        
        # Filter to only include the specific genome IDs we need
        genome_id_set = set(genome_ids)
        filtered_genomes = [
            genome for genome in all_taxon_genomes 
            if genome.genome_id in genome_id_set
        ]
        
        self.nb_logger.info(f"📦 Retrieved {len(filtered_genomes)}/{len(genome_ids)} requested genomes from taxon_id {taxon_id}")
        
        return filtered_genomes
        
    async def _download_genomes_by_taxon_id(self, taxon_id: str) -> List[GenomeData]:
        """
        Download all genomes for a specific taxon_id
        """
        
        # Verify BV-BRC tool is accessible
        if not await self.bv_brc_tool.verify_installation():
            raise RuntimeError("BV-BRC tool is not accessible. Please check installation.")
        
        # Construct command arguments for genome download using taxon_id
        command_args = [
            "--eq", f"taxon_id,{taxon_id}",
            "--attr", "genome_id,genome_length,genome_name,taxon_lineage_names"
        ]
        
        try:
            result = await self.bv_brc_tool.execute_p3_command("p3-all-genomes", command_args)
            
            if result.returncode != 0:
                raise RuntimeError(f"p3-all-genomes command failed: {result.stderr.decode()}")
            
            genomes = await self._parse_genome_data(result.stdout)
            
            if not genomes:
                self.nb_logger.warning(f"No genomes found for taxon_id {taxon_id} in BV-BRC")
            
            return genomes
            
        except Exception as e:
            self.nb_logger.error(f"Failed to download genomes for taxon_id {taxon_id}: {e}")
            raise
            
    async def _download_genomes_by_genus(self, target_genus: str) -> List[GenomeData]:
        """
        Download all genomes for a specific genus
        """
        
        # Verify BV-BRC tool is accessible
        if not await self.bv_brc_tool.verify_installation():
            raise RuntimeError("BV-BRC tool is not accessible. Please check installation.")
        
        # Construct command arguments for genome download using genus
        command_args = [
            "--eq", f"taxon_lineage_names,{target_genus}",
            "--attr", "genome_id,genome_length,genome_name,taxon_lineage_names"
        ]
        
        try:
            result = await self.bv_brc_tool.execute_p3_command("p3-all-genomes", command_args)
            
            if result.returncode != 0:
                raise RuntimeError(f"p3-all-genomes command failed: {result.stderr.decode()}")
            
            genomes = await self._parse_genome_data(result.stdout)
            
            if not genomes:
                self.nb_logger.warning(f"No genomes found for genus {target_genus} in BV-BRC")
            
            return genomes
            
        except Exception as e:
            self.nb_logger.error(f"Failed to download genomes for genus {target_genus}: {e}")
            raise
            
    async def _parse_genome_data(self, raw_data: bytes) -> List[GenomeData]:
        """Parse genome data from BV-BRC output"""
        
        if not raw_data:
            return []
        
        lines = raw_data.decode().strip().split('\n')
        
        if len(lines) < 2:  # Must have header + at least one data line
            self.nb_logger.warning("BV-BRC returned only headers, no genome data")
            return []
        
        # Parse header to get field positions - BV-BRC returns prefixed column names like 'genome.genome_id'
        header = lines[0].split('\t')
        try:
            # Look for columns with the expected suffixes
            genome_id_idx = None
            genome_name_idx = None
            genome_length_idx = None
            taxon_lineage_idx = None
            
            for i, col in enumerate(header):
                if col.endswith('genome_id'):
                    genome_id_idx = i
                elif col.endswith('genome_name'):
                    genome_name_idx = i
                elif col.endswith('genome_length'):
                    genome_length_idx = i
                elif col.endswith('taxon_lineage_names'):
                    taxon_lineage_idx = i
            
            if genome_id_idx is None:
                raise ValueError(f"genome_id column not found in headers: {header}")
            if genome_name_idx is None:
                raise ValueError(f"genome_name column not found in headers: {header}")
            if genome_length_idx is None:
                raise ValueError(f"genome_length column not found in headers: {header}")
                
        except Exception as e:
            raise ValueError(f"Missing required field in BV-BRC output: {e}")
        
        genomes = []
        for line_num, line in enumerate(lines[1:], 2):
            try:
                fields = line.split('\t')
                
                if len(fields) <= max(genome_id_idx, genome_length_idx, genome_name_idx):
                    continue
                
                genome_id = fields[genome_id_idx].strip()
                genome_length_str = fields[genome_length_idx].strip()
                genome_name = fields[genome_name_idx].strip()
                
                # Validate and parse genome length
                if not genome_length_str or genome_length_str == '-':
                    continue
                    
                try:
                    genome_length = int(genome_length_str)
                except ValueError:
                    self.nb_logger.warning(f"Invalid genome length '{genome_length_str}' for {genome_id}")
                    continue
                
                # Get taxon lineage if available
                taxon_lineage = ""
                if taxon_lineage_idx >= 0 and len(fields) > taxon_lineage_idx:
                    taxon_lineage = fields[taxon_lineage_idx].strip()
                
                genome = GenomeData(
                    genome_id=genome_id,
                    genome_length=genome_length,
                    genome_name=genome_name,
                    taxon_lineage=taxon_lineage
                )
                
                genomes.append(genome)
                
            except Exception as e:
                self.nb_logger.warning(f"Error parsing genome data at line {line_num}: {e}")
                continue
        
        return genomes
        
    async def _filter_genomes_by_size(self, genomes: List[GenomeData]) -> List[GenomeData]:
        """
        Step 2: Filter genomes by size based on threshold
        
        Alphavirus genomes are typically 11,000-12,000 bp
        Filter range: 8,000-15,000 bp to exclude fragments and contaminated assemblies
        """
        
        filtered_genomes = []
        
        for genome in genomes:
            if self.min_genome_length <= genome.genome_length <= self.max_genome_length:
                filtered_genomes.append(genome)
            else:
                self.nb_logger.debug(f"Filtered genome {genome.genome_id}: length {genome.genome_length} outside range")
        
        return filtered_genomes
        
    async def _get_unique_protein_md5s(self, genome_ids: List[str], cache_key: str = None) -> List[ProteinData]:
        """
        Steps 3-4: Get unique protein MD5s and deduplicate
        
        Uses MANDATORY BV-BRC pipeline to get all proteins, then removes duplicates by MD5
        """
        
        # Process ALL genomes at once - no batching to avoid incorrect results
        self.nb_logger.info(f"Processing all {len(genome_ids)} genomes in single operation")
        
        all_proteins = await self._get_proteins_for_genomes(genome_ids, cache_key)
        
        # Remove duplicates by MD5 hash
        unique_proteins_dict = {}
        for protein in all_proteins:
            md5 = protein.aa_sequence_md5
            if md5 and md5 not in unique_proteins_dict:
                unique_proteins_dict[md5] = protein
        
        unique_proteins = list(unique_proteins_dict.values())
        
        self.nb_logger.info(f"Deduplicated {len(all_proteins)} proteins to {len(unique_proteins)} unique MD5s")
        
        return unique_proteins
        
    async def _get_proteins_for_genomes(self, genome_ids: List[str], cache_key: str = None) -> List[ProteinData]:
        """
        Get protein data using MANDATORY BV-BRC pipeline format with caching:
        1. Use p3-all-genomes to get genome data → save to TSV
        2. Use cut -f1 file.tsv | p3-get-genome-features → get protein features
        
        This is the ONLY way p3-get-genome-features works!
        """
        
        import os
        import asyncio
        
        # Generate cache key if not provided
        if not cache_key:
            import hashlib
            cache_key = hashlib.md5(','.join(sorted(genome_ids)).encode()).hexdigest()[:8]
        
        # Check for cached protein data - PRIORITIZE FILTERED DATA
        cached_proteins_path = self._get_cached_file_path(cache_key, "proteins")
        filtered_proteins_path = self._get_cached_file_path(cache_key, "proteins.filtered")
        
        # Use filtered proteins if available, otherwise fall back to regular cache
        if filtered_proteins_path.exists():
            cached_proteins_path = filtered_proteins_path
            print(f"🗂️ USING FILTERED PROTEINS CACHE: {cached_proteins_path}")
            self.nb_logger.info(f"🗂️ Using filtered proteins cache: {cached_proteins_path}")
        else:
            print(f"🗂️ PROTEINS CACHE PATH: {cached_proteins_path}")
            self.nb_logger.info(f"🗂️ Proteins cache file: {cached_proteins_path}")
        
        # ALWAYS USE CACHE IF IT EXISTS - NO BV-BRC COMMANDS
        if cached_proteins_path.exists():
            print(f"💾 FOUND EXISTING PROTEINS CACHE: {cached_proteins_path}")
            self.nb_logger.info(f"💾 Found existing protein cache: {cached_proteins_path}")
            
            print(f"✅ USING CACHED PROTEIN DATA - SKIPPING ALL BV-BRC COMMANDS")
            self.nb_logger.info(f"💾 Using cached protein data - skipping ALL BV-BRC commands")
            try:
                with open(cached_proteins_path, 'rb') as f:
                    cached_data = f.read()
                proteins = await self._parse_protein_data(cached_data)
                self.nb_logger.info(f"✅ Loaded {len(proteins)} proteins from cache")
                return proteins
            except Exception as e:
                self.nb_logger.error(f"❌ Failed to load cached proteins: {e}")
                raise RuntimeError(f"Failed to load cached proteins: {e}")
        
        # Only proceed with BV-BRC commands if no cache exists AND we have genome_ids
        if not genome_ids:
            self.nb_logger.error(f"❌ No cache found and no genome_ids provided")
            return []
        
        # Get genomes cache path for BV-BRC pipeline
        genomes_cache_path = self._get_cached_file_path(cache_key, "genomes")
        
        # Step 1: Use p3-all-genomes to get genome data first
        genome_list = ",".join(genome_ids)
        p3_all_genomes_path = f"{self.bv_brc_tool.bv_brc_config.executable_path}/p3-all-genomes"
        
        # Create virus-specific file names
        print(f"🗂️ GENOMES CACHE PATH: {genomes_cache_path}")
        self.nb_logger.info(f"🗂️ Genomes cache file: {genomes_cache_path}")
        
        try:
            # Step 1: Execute p3-all-genomes and save to cache file (p3-all-genomes does NOT support --batchSize)
            genomes_command = f'{p3_all_genomes_path} --in "genome_id,({genome_list})" --attr genome_id,genome_name > {genomes_cache_path}'
            
            print(f"🔄 EXECUTING STEP 1 COMMAND: {genomes_command}")
            self.nb_logger.info(f"🔄 Step 1: Running p3-all-genomes for {len(genome_ids)} genomes (timeout: {self.timeout_seconds}s)")
            self.nb_logger.info(f"🔄 Full command: {genomes_command}")
            
            process1 = await asyncio.wait_for(
                asyncio.create_subprocess_shell(
                    genomes_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=self.timeout_seconds
            )
            
            stdout1, stderr1 = await asyncio.wait_for(
                process1.communicate(),
                timeout=self.timeout_seconds
            )
            
            if process1.returncode != 0:
                self.nb_logger.error(f"p3-all-genomes failed: {stderr1.decode()}")
                return []
            
            # Step 2: Use MANDATORY pipeline format: cut -f1 file.tsv | p3-get-genome-features
            p3_features_path = f"{self.bv_brc_tool.bv_brc_config.executable_path}/p3-get-genome-features"
            
            # MANDATORY pipeline format - save to cache (p3-get-genome-features DOES support --batchSize)
            pipeline_command = f'cut -f1 {genomes_cache_path} | {p3_features_path} --attr patric_id --attr product --attr aa_sequence_md5 --attr genome_id --batchSize {self.md5_batch_size} > {cached_proteins_path}'
            
            print(f"🔄 EXECUTING STEP 2 COMMAND: {pipeline_command}")
            self.nb_logger.info(f"🔄 Step 2: MANDATORY pipeline - cut -f1 | p3-get-genome-features (timeout: {self.timeout_seconds}s)")
            self.nb_logger.info(f"🔄 Full command: {pipeline_command}")
            
            process2 = await asyncio.wait_for(
                asyncio.create_subprocess_shell(
                    pipeline_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=self.timeout_seconds
            )
            
            stdout2, stderr2 = await asyncio.wait_for(
                process2.communicate(),
                timeout=self.timeout_seconds
            )
            
            if process2.returncode != 0:
                self.nb_logger.error(f"Pipeline command failed: {stderr2.decode()}")
                return []
            
            # Step 3: Read the cached file and parse protein data
            with open(cached_proteins_path, 'rb') as f:
                features_data = f.read()
            
            proteins = await self._parse_protein_data(features_data)
            self.nb_logger.info(f"✅ Successfully extracted {len(proteins)} proteins using MANDATORY pipeline")
            self.nb_logger.info(f"💾 Protein data cached to: {cached_proteins_path}")
            
            return proteins
            
        except asyncio.TimeoutError:
            self.nb_logger.error(f"❌ BV-BRC pipeline timed out after {self.timeout_seconds} seconds")
            return []
        except Exception as e:
            self.nb_logger.error(f"❌ Error in MANDATORY pipeline: {e}")
            return []
            
    async def _parse_protein_data(self, raw_data: bytes) -> List[ProteinData]:
        """Parse protein data from p3-get-genome-features output"""
        
        if not raw_data:
            return []
        
        lines = raw_data.decode().strip().split('\n')
        
        if len(lines) < 2:
            return []
        
        # Parse header - p3-get-genome-features returns prefixed column names like 'feature.patric_id'
        header = lines[0].split('\t')
        try:
            # Look for columns with the expected suffixes
            patric_id_idx = None
            md5_idx = None
            genome_id_idx = None
            product_idx = None
            gene_idx = None
            
            for i, col in enumerate(header):
                if col.endswith('patric_id'):
                    patric_id_idx = i
                elif col.endswith('aa_sequence_md5'):
                    md5_idx = i
                elif col.endswith('genome_id'):
                    genome_id_idx = i
                elif col.endswith('product'):
                    product_idx = i
                elif col.endswith('gene'):
                    gene_idx = i
            
            if patric_id_idx is None:
                raise ValueError(f"patric_id column not found in headers: {header}")
            if md5_idx is None:
                raise ValueError(f"aa_sequence_md5 column not found in headers: {header}")
            if genome_id_idx is None:
                raise ValueError(f"genome_id column not found in headers: {header}")
                
        except Exception as e:
            raise ValueError(f"Missing required field in protein data: {e}")
        
        proteins = []
        for line in lines[1:]:
            try:
                fields = line.split('\t')
                
                if len(fields) <= max(patric_id_idx, md5_idx, genome_id_idx):
                    continue
                
                patric_id = fields[patric_id_idx].strip()
                aa_sequence_md5 = fields[md5_idx].strip()
                genome_id = fields[genome_id_idx].strip()
                
                # Skip if MD5 is missing
                if not aa_sequence_md5 or aa_sequence_md5 == '-':
                    continue
                
                product = ""
                if product_idx is not None and len(fields) > product_idx:
                    product = fields[product_idx].strip()
                
                gene = ""
                if gene_idx is not None and len(fields) > gene_idx:
                    gene = fields[gene_idx].strip()
                
                protein = ProteinData(
                    patric_id=patric_id,
                    aa_sequence_md5=aa_sequence_md5,
                    genome_id=genome_id,
                    product=product,
                    gene=gene
                )
                
                proteins.append(protein)
                
            except Exception as e:
                self.nb_logger.warning(f"Error parsing protein data: {e}")
                continue
        
        return proteins
        
    async def _get_feature_sequences(self, protein_data: List[ProteinData]) -> List[SequenceData]:
        """
        Step 5: Get feature sequences for proteins using patric_id
        
        Uses p3-get-feature-sequence command with patric_id (NOT md5)
        """
        
        # Process ALL proteins at once - no batching, let p3-get-feature-sequence handle batching internally
        self.nb_logger.info(f"Getting sequences for all {len(protein_data)} proteins in single operation")
        
        sequences = await self._get_sequences_for_proteins(protein_data)
        
        return sequences
        
    async def _get_sequences_for_proteins(self, protein_batch: List[ProteinData]) -> List[SequenceData]:
        """Get sequences for proteins using cached data first, then p3-get-feature-sequence if needed"""
        
        import os
        import asyncio
        
        # FIRST: Check if we have cached sequence data (FASTA format)
        if hasattr(self, 'current_cache_key') and self.current_cache_key:
            # Check for raw sequences cache first
            cached_sequences_file = self._get_cached_file_path(self.current_cache_key, "sequences.fasta")
            
            # Also check for existing annotated FASTA file
            annotated_fasta_file = self._get_cached_file_path(self.current_cache_key, "proteins_annotated.fasta")
            
            # Prioritize raw sequences cache, then annotated FASTA
            sequence_cache_file = None
            if cached_sequences_file.exists():
                sequence_cache_file = cached_sequences_file
                print(f"💾 FOUND EXISTING RAW SEQUENCES CACHE: {sequence_cache_file}")
            elif annotated_fasta_file.exists():
                sequence_cache_file = annotated_fasta_file
                print(f"💾 FOUND EXISTING ANNOTATED FASTA CACHE: {sequence_cache_file}")
            
            if sequence_cache_file:
                self.nb_logger.info(f"💾 Found existing sequence cache: {sequence_cache_file}")
                
                try:
                    # Load cached sequences
                    with open(sequence_cache_file, 'r') as f:
                        cached_fasta_content = f.read()
                    
                    # Parse cached FASTA content
                    raw_sequences = await self._parse_fasta_content(cached_fasta_content)
                    
                    # Create mapping from patric_id to md5
                    patric_to_md5 = {protein.patric_id: protein.aa_sequence_md5 for protein in protein_batch}
                    
                    # Debug: Log sample patric_ids from both sources
                    if raw_sequences and patric_to_md5:
                        sample_fasta_ids = [seq.get('patric_id', '') for seq in raw_sequences[:5]]
                        sample_protein_ids = list(patric_to_md5.keys())[:5]
                        self.nb_logger.info(f"🔍 DEBUG: Sample FASTA patric_ids: {sample_fasta_ids}")
                        self.nb_logger.info(f"🔍 DEBUG: Sample protein patric_ids: {sample_protein_ids}")
                        
                        # Check for exact matches
                        fasta_id_set = {seq.get('patric_id', '') for seq in raw_sequences}
                        protein_id_set = set(patric_to_md5.keys())
                        matches = fasta_id_set.intersection(protein_id_set)
                        self.nb_logger.info(f"🔍 DEBUG: Found {len(matches)} exact patric_id matches out of {len(fasta_id_set)} FASTA and {len(protein_id_set)} protein IDs")
                        
                        if len(matches) == 0 and len(fasta_id_set) > 0 and len(protein_id_set) > 0:
                            # Try to understand the mismatch
                            sample_fasta = next(iter(fasta_id_set))
                            sample_protein = next(iter(protein_id_set))
                            self.nb_logger.warning(f"⚠️ No patric_id matches! FASTA format: '{sample_fasta}', Protein format: '{sample_protein}'")
                    
                    # Convert to SequenceData with md5 for consistency
                    sequences = []
                    for seq_data in raw_sequences:
                        if seq_data.get('patric_id') in patric_to_md5:
                            sequences.append(SequenceData(
                                aa_sequence_md5=patric_to_md5[seq_data['patric_id']],
                                aa_sequence=seq_data['aa_sequence']
                            ))
                    
                    print(f"✅ LOADED {len(sequences)} SEQUENCES FROM CACHE - SKIPPING p3-get-feature-sequence")
                    self.nb_logger.info(f"💾 Loaded {len(sequences)} sequences from cache - skipping p3-get-feature-sequence")
                    return sequences
                    
                except Exception as e:
                    self.nb_logger.warning(f"⚠️ Failed to load cached sequences: {e}")
                    # Fall through to execute p3-get-feature-sequence
        
        # Find the cached proteins file that was created by p3-get-genome-features
        # PRIORITIZE FILTERED PROTEINS FILE
        cached_proteins_file = None
        
        # Use the specific cache key if available
        if hasattr(self, 'current_cache_key') and self.current_cache_key:
            # Check for filtered file first
            filtered_file = self._get_cached_file_path(self.current_cache_key, "proteins.filtered")
            if filtered_file.exists():
                cached_proteins_file = filtered_file
            else:
                cached_proteins_file = self._get_cached_file_path(self.current_cache_key, "proteins")
        
        # Fallback: Look for the most recent proteins cache file (prioritize filtered)
        if not cached_proteins_file or not cached_proteins_file.exists():
            # First try to find filtered files
            for cache_file in self.cache_dir.glob("*_proteins.filtered.tsv"):
                if cache_file.exists():
                    cached_proteins_file = cache_file
                    break
            
            # If no filtered file found, use regular proteins file
            if not cached_proteins_file or not cached_proteins_file.exists():
                for cache_file in self.cache_dir.glob("*_proteins.tsv"):
                    if cache_file.exists():
                        cached_proteins_file = cache_file
                        break
        
        if not cached_proteins_file or not cached_proteins_file.exists():
            self.nb_logger.error("No cached proteins file found from p3-get-genome-features")
            return []
        
        print(f"🗂️ USING PROTEINS FILE FROM p3-get-genome-features: {cached_proteins_file}")
        self.nb_logger.info(f"🗂️ Using proteins file from p3-get-genome-features: {cached_proteins_file}")
        
        try:
            # Use p3-get-feature-sequence with the cached proteins file from p3-get-genome-features
            p3_sequence_path = f"{self.bv_brc_tool.bv_brc_config.executable_path}/p3-get-feature-sequence"
            
            # Use the existing filtered file or create one if needed
            if "filtered" in str(cached_proteins_file):
                # Already using filtered file
                filtered_proteins_file = cached_proteins_file
                print(f"🔧 USING EXISTING FILTERED PROTEINS FILE: {filtered_proteins_file}")
                self.nb_logger.info(f"🔧 Using existing filtered proteins file: {filtered_proteins_file}")
            else:
                # Create a filtered version of the proteins file without problematic patric_ids
                # Filter out UTR and other non-protein features that cause URL encoding issues
                filtered_proteins_file = cached_proteins_file.with_suffix('.filtered.tsv')
                
                with open(cached_proteins_file, 'r') as infile, open(filtered_proteins_file, 'w') as outfile:
                    header = infile.readline()
                    outfile.write(header)
                    
                    for line in infile:
                        # Skip lines with problematic characters in patric_id
                        if "'" not in line and "UTR" not in line and "gap." not in line:
                            outfile.write(line)
                
                print(f"🔧 CREATED FILTERED PROTEINS FILE: {filtered_proteins_file}")
                self.nb_logger.info(f"🔧 Created filtered proteins file: {filtered_proteins_file}")
            
            # Command: p3-get-feature-sequence --input filtered_proteins_file.tsv --col 2 --batchSize N
            # Column 2 is the patric_id column (1-indexed) from p3-get-genome-features output
            # Column 1 is genome_id, Column 2 is patric_id, Column 3 is product, etc.
            sequence_command = f'{p3_sequence_path} --input {filtered_proteins_file} --col 2 --batchSize {self.md5_batch_size}'
            
            print(f"🔄 EXECUTING SEQUENCE COMMAND: {sequence_command}")
            self.nb_logger.debug(f"Executing sequence command: {sequence_command}")
            self.nb_logger.info(f"🔄 Full sequence command: {sequence_command}")
            
            # Execute command with timeout
            process = await asyncio.create_subprocess_shell(
                sequence_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout_seconds)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise RuntimeError(f"p3-get-feature-sequence command timed out after {self.timeout_seconds} seconds")
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                self.nb_logger.warning(f"p3-get-feature-sequence failed: {error_msg}")
                return []
            
            # Parse sequences and map back to md5 using protein data
            raw_sequences = await self._parse_sequence_data_with_patric_id(stdout)
            
            # CACHE THE SEQUENCES for future use
            if hasattr(self, 'current_cache_key') and self.current_cache_key:
                cached_sequences_file = self._get_cached_file_path(self.current_cache_key, "sequences.fasta")
                try:
                    with open(cached_sequences_file, 'wb') as f:
                        f.write(stdout)
                    print(f"💾 CACHED SEQUENCES TO: {cached_sequences_file}")
                    self.nb_logger.info(f"💾 Cached sequences to: {cached_sequences_file}")
                except Exception as e:
                    self.nb_logger.warning(f"⚠️ Failed to cache sequences: {e}")
            
            # Create mapping from patric_id to md5
            patric_to_md5 = {protein.patric_id: protein.aa_sequence_md5 for protein in protein_batch}
            
            # Convert to SequenceData with md5 for consistency
            sequences = []
            for seq_data in raw_sequences:
                if seq_data.get('patric_id') in patric_to_md5:
                    sequences.append(SequenceData(
                        aa_sequence_md5=patric_to_md5[seq_data['patric_id']],
                        aa_sequence=seq_data['aa_sequence']
                    ))
            
            return sequences
            
        except Exception as e:
            self.nb_logger.warning(f"Error getting sequences for protein batch: {e}")
            return []
            
    async def _parse_sequence_data_with_patric_id(self, raw_data: bytes) -> List[Dict[str, str]]:
        """Parse sequence data from p3-get-feature-sequence output (FASTA format)"""
        
        if not raw_data:
            return []
        
        lines = raw_data.decode().strip().split('\n')
        
        if len(lines) < 2:
            return []
        
        # p3-get-feature-sequence returns FASTA format, not tabular
        # Format: >fig|genome_id.feature_id description
        # Example: >fig|1534555.3.mat_peptide.9 E1 envelope glycoprotein
        
        sequences = []
        current_header = None
        current_sequence = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                # Process previous sequence if exists
                if current_header and current_sequence:
                    patric_id = self._extract_patric_id_from_fasta_header(current_header)
                    if patric_id:
                        sequences.append({
                            'patric_id': patric_id,
                            'aa_sequence': ''.join(current_sequence)
                        })
                
                # Start new sequence
                current_header = line
                current_sequence = []
            else:
                # Accumulate sequence lines
                current_sequence.append(line)
        
        # Process last sequence
        if current_header and current_sequence:
            patric_id = self._extract_patric_id_from_fasta_header(current_header)
            if patric_id:
                sequences.append({
                    'patric_id': patric_id,
                    'aa_sequence': ''.join(current_sequence)
                })
        
        self.nb_logger.info(f"Parsed {len(sequences)} sequences from FASTA format")
        return sequences
    
    def _extract_patric_id_from_fasta_header(self, header: str) -> str:
        """Extract patric_id from FASTA header like >fig|1534555.3.mat_peptide.9 E1 envelope glycoprotein"""
        try:
            # Remove the '>' and split by space to get the ID part
            header_parts = header[1:].split(' ')
            if header_parts:
                # Extract the fig|genome_id.feature_id part
                fig_id = header_parts[0]
                if fig_id.startswith('fig|'):
                    # Return the fig|genome_id.feature_id as patric_id
                    return fig_id
            return ""
        except Exception as e:
            self.nb_logger.warning(f"Error extracting patric_id from header '{header}': {e}")
            return ""

    def _extract_patric_id_from_any_header(self, header: str) -> str:
        """Extract patric_id from any FASTA header format (raw or annotated)"""
        try:
            # Remove the '>'
            header_content = header[1:]
            
            # Check if this is an annotated header (contains pipes)
            if '|' in header_content:
                # Two possible formats:
                # 1. Raw: fig|genome_id.feature_id description
                # 2. Annotated: fig_genome_id.feature_id|product|gene|md5
                
                # First check if it's raw format by looking for space after the ID
                if ' ' in header_content and header_content.startswith('fig|'):
                    # Raw format: extract everything before the first space
                    return header_content.split(' ')[0]
                
                # Otherwise it's annotated format
                parts = header_content.split('|')
                first_part = parts[0]
                
                if first_part.startswith('fig_'):
                    # Annotated format: convert fig_ back to fig|
                    return first_part.replace('fig_', 'fig|', 1)
                else:
                    # Other annotated format, return as is
                    return first_part
            else:
                # No pipes, might be space-separated raw format
                parts = header_content.split(' ')
                if parts and parts[0].startswith('fig|'):
                    return parts[0]
            
            return ""
        except Exception as e:
            self.nb_logger.warning(f"Error extracting patric_id from header '{header}': {e}")
            return ""

    async def _parse_fasta_content(self, fasta_content: str) -> List[Dict[str, str]]:
        """Parse FASTA content from cached sequence file (handles both raw and annotated formats)"""
        
        if not fasta_content:
            return []
        
        lines = fasta_content.strip().split('\n')
        
        if len(lines) < 2:
            return []
        
        # Parse FASTA format
        # Format 1 (raw): >fig|genome_id.feature_id description
        # Format 2 (annotated): >fig_genome_id.feature_id|product|gene|md5
        
        sequences = []
        current_header = None
        current_sequence = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                # Process previous sequence if exists
                if current_header and current_sequence:
                    patric_id = self._extract_patric_id_from_any_header(current_header)
                    if patric_id:
                        sequences.append({
                            'patric_id': patric_id,
                            'aa_sequence': ''.join(current_sequence)
                        })
                
                # Start new sequence
                current_header = line
                current_sequence = []
            else:
                # Accumulate sequence lines
                current_sequence.append(line)
        
        # Process last sequence
        if current_header and current_sequence:
            patric_id = self._extract_patric_id_from_any_header(current_header)
            if patric_id:
                sequences.append({
                    'patric_id': patric_id,
                    'aa_sequence': ''.join(current_sequence)
                })
        
        self.nb_logger.info(f"Parsed {len(sequences)} sequences from cached FASTA content")
        return sequences

    async def _extract_annotations_from_proteins(self, protein_data: List[ProteinData]) -> List[AnnotationData]:
        """Extract annotation data from protein data (no additional BV-BRC calls needed)"""
        
        annotations = []
        for protein in protein_data:
            annotation = AnnotationData(
                aa_sequence_md5=protein.aa_sequence_md5,
                product=protein.product,
                gene=protein.gene,
                refseq_locus_tag="",  # Not available in basic protein data
                go="",               # Not available in basic protein data
                ec="",               # Not available in basic protein data
                pathway=""           # Not available in basic protein data
            )
            annotations.append(annotation)
        
        return annotations
        
    async def _create_annotated_fasta_with_cache(self, sequences: List[SequenceData], 
                                               annotations: List[AnnotationData], cache_key: str) -> str:
        """
        Create FASTA file with caching support
        
        Checks for cached FASTA file first, creates if not found
        """
        
        # Check for cached FASTA file
        fasta_cache_path = self._get_cached_file_path(cache_key, "proteins_annotated.fasta")
        
        if fasta_cache_path.exists():
            print(f"💾 FOUND EXISTING FASTA CACHE: {fasta_cache_path}")
            self.nb_logger.info(f"💾 Found existing FASTA cache: {fasta_cache_path}")
            
            print(f"✅ USING CACHED FASTA DATA - SKIPPING FASTA CREATION")
            self.nb_logger.info(f"💾 Using cached FASTA data - skipping FASTA creation")
            
            try:
                with open(fasta_cache_path, 'r') as f:
                    cached_fasta = f.read()
                
                # Count sequences in cached FASTA
                header_count = cached_fasta.count('>')
                self.nb_logger.info(f"✅ Loaded FASTA with {header_count} sequences from cache")
                return cached_fasta
                
            except Exception as e:
                self.nb_logger.error(f"❌ Failed to load cached FASTA: {e}")
                # Fall through to create new FASTA
        
        # Create new FASTA file
        self.nb_logger.info("🔄 Creating new annotated FASTA file")
        annotated_fasta = await self._create_annotated_fasta(sequences, annotations)
        
        # Cache the FASTA file
        try:
            with open(fasta_cache_path, 'w') as f:
                f.write(annotated_fasta)
            
            print(f"💾 CACHED FASTA FILE: {fasta_cache_path}")
            self.nb_logger.info(f"💾 Cached FASTA file: {fasta_cache_path}")
            
            # Log file size
            file_size = fasta_cache_path.stat().st_size
            print(f"📊 FASTA cache file size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            self.nb_logger.info(f"📊 FASTA cache file size: {file_size:,} bytes")
            
        except Exception as e:
            self.nb_logger.warning(f"⚠️ Failed to cache FASTA file: {e}")
        
        return annotated_fasta
    
    async def _create_annotated_fasta(self, sequences: List[SequenceData], 
                                     annotations: List[AnnotationData]) -> str:
        """
        Step 7: Create FASTA file with detailed annotation headers
        
        FASTA header format:
        >{patric_id}|{product}|{gene}|{md5}
        """
        
        # Create annotation lookup by MD5
        annotation_map = {ann.aa_sequence_md5: ann for ann in annotations}
        
        # Create protein lookup by MD5 to get patric_id and product
        protein_map = {}
        if hasattr(self, 'unique_proteins'):
            protein_map = {p.aa_sequence_md5: p for p in self.unique_proteins}
        
        fasta_lines = []
        processed_count = 0
        
        for seq in sequences:
            annotation = annotation_map.get(seq.aa_sequence_md5)
            protein = protein_map.get(seq.aa_sequence_md5)
            
            if protein and annotation:
                # Use patric_id from protein data and product from annotation
                header = self._format_fasta_header_with_patric_id(
                    protein.patric_id, 
                    annotation.product or protein.product, 
                    annotation.gene or protein.gene,
                    seq.aa_sequence_md5
                )
                fasta_lines.append(f">{header}")
                fasta_lines.append(seq.aa_sequence)
                processed_count += 1
                
                # Log the enhanced header for verification
                self.nb_logger.debug(f"FASTA entry: {protein.patric_id} -> {annotation.product or protein.product}")
            else:
                # Fallback to basic header
                patric_id = protein.patric_id if protein else "unknown_id"
                product = (annotation.product if annotation else 
                          protein.product if protein else "hypothetical protein")
                
                header = f"{patric_id}|{product}|unknown|{seq.aa_sequence_md5}"
                fasta_lines.append(f">{header}")
                fasta_lines.append(seq.aa_sequence)
                processed_count += 1
        
        self.nb_logger.info(f"Created FASTA with {processed_count} sequences")
        
        # Log sample FASTA entries for verification
        if fasta_lines:
            sample_headers = [line for line in fasta_lines[:10] if line.startswith('>')]
            self.nb_logger.info(f"Sample FASTA headers: {sample_headers[:3]}")
        
        return "\n".join(fasta_lines)
        
    def _format_fasta_header_with_patric_id(self, patric_id: str, product: str, gene: str, md5: str) -> str:
        """Format FASTA header with patric_id and product information"""
        
        components = [
            patric_id or "unknown_id",
            product or "hypothetical protein", 
            gene or "unknown",
            md5 or "unknown_md5"
        ]
        
        # Clean components (remove pipes and tabs that could break parsing)
        cleaned_components = []
        for comp in components:
            cleaned = str(comp).replace("|", "_").replace("\t", " ").replace("\n", " ")
            cleaned_components.append(cleaned)
        
        return "|".join(cleaned_components)
    
    def _format_fasta_header(self, md5: str, annotation: AnnotationData) -> str:
        """Format FASTA header with detailed annotation (legacy method)"""
        
        components = [
            md5,
            annotation.gene or "unknown",
            annotation.product or "hypothetical protein",
            annotation.refseq_locus_tag or "no_tag",
            annotation.pathway or "unknown_pathway"
        ]
        
        # Clean components (remove pipes and tabs that could break parsing)
        cleaned_components = []
        for comp in components:
            cleaned = str(comp).replace("|", "_").replace("\t", " ").replace("\n", " ")
            cleaned_components.append(cleaned)
        
        return "|".join(cleaned_components) 