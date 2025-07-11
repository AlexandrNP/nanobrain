"""
Annotation Mapping Step

Re-architected to inherit from NanoBrain Step base class.
Integrates ProteinSynonymAgent for advanced synonym resolution.
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.logging_system import get_logger
from nanobrain.core import AgentConfig
from nanobrain.core.data_unit import DataUnit, DataUnitConfig, DataUnitType


class AnnotationMappingStep(Step):
    """
    Annotation mapping functionality with synonym resolution
    
    Re-architected to inherit from NanoBrain Step base class.
    Integrates ProteinSynonymAgent for ICTV-compliant synonym resolution.
    """
    
    def __init__(self, config: StepConfig, annotation_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config, **kwargs)
        
        # Extract configuration from step config or provided annotation_config
        step_config_dict = config.config if hasattr(config, 'config') else {}
        if annotation_config:
            step_config_dict.update(annotation_config)
        
        self.annotation_config = step_config_dict.get('annotation_config', {})
        self.step_config = step_config_dict
        
        # Synonym resolution configuration
        synonym_config = self.annotation_config.get('synonym_resolution', {})
        self.enable_synonym_resolution = synonym_config.get('enabled', False)
        self.synonym_wait_timeout = synonym_config.get('wait_timeout', 120)
        self.fallback_on_timeout = synonym_config.get('fallback_on_timeout', True)
        
        # Track pending synonym requests
        self.pending_requests = {}
        
        # Legacy synonym agent support (will be removed)
        self.use_synonym_agent = False  # Disabled - using Link-based approach
        self.synonym_agent = None
        
        self.nb_logger.info(f"🧬 AnnotationMappingStep initialized")
        self.nb_logger.info(f"🔗 Link-based synonym resolution enabled: {self.enable_synonym_resolution}")
    
    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any], 
                         dependencies: Dict[str, Any]) -> None:
        """Initialize AnnotationMappingStep with resolved dependencies"""
        # Initialize base step
        super()._init_from_config(config, component_config, dependencies)
        
        # Initialize specific attributes
        self.tools = {}
        self.pending_requests = {}
        
        # Configuration
        self.enable_synonym_resolution = component_config.get('enable_synonym_resolution', False)
        self.synonym_agent_config_path = component_config.get('synonym_agent_config_path', 
                                                             'nanobrain/library/agents/specialized/config/ProteinSynonymAgent.yml')
        
        # CRITICAL FIX: Access original config directly since component_config is filtered
        original_config = config.config if hasattr(config, 'config') else {}
        self._initialize_tools(original_config)
        
        # Logging
        self.nb_logger.info(f"🧪 AnnotationMappingStep initialized via from_config")
        self.nb_logger.info(f"🔧 Tools initialized: {list(self.tools.keys())}")
        self.nb_logger.info(f"🔗 Link-based synonym resolution enabled: {self.enable_synonym_resolution}")
    
    def _initialize_tools(self, component_config: Dict[str, Any]) -> None:
        """Initialize tools based on configuration"""
        # DEBUG: Log what we're receiving
        self.nb_logger.info(f"🔍 _initialize_tools called with config keys: {list(component_config.keys())}")
        
        tools_config = component_config.get('tools', {})
        self.nb_logger.info(f"🔍 Found tools_config: {tools_config}")
        
        # Initialize synonym agent if enabled
        synonym_config = tools_config.get('synonym_agent', {})
        self.nb_logger.info(f"🔍 Found synonym_config: {synonym_config}")
        
        if synonym_config.get('enabled', False):
            self.nb_logger.info("🤖 Initializing synonym agent from tools configuration")
            try:
                agent_class = self._load_class(synonym_config['class'])
                config_file = synonym_config.get('config_file')
                
                if config_file:
                    # FIXED: Load agent configuration properly
                    import yaml
                    from nanobrain.core import AgentConfig
                    from pathlib import Path
                    
                    # Resolve config file path
                    config_path = Path(config_file)
                    if not config_path.is_absolute():
                        # Try multiple fallback strategies
                        strategies = [
                            Path(config_file),  # Direct path
                            Path(__file__).parent.parents[2] / config_file,  # Relative to library
                            Path(__file__).parent.parents[3] / config_file,  # Relative to project
                        ]
                        
                        resolved_path = None
                        for path in strategies:
                            if path.exists():
                                resolved_path = path
                                break
                        
                        if not resolved_path:
                            raise FileNotFoundError(f"Config file not found: {config_file}")
                        
                        config_path = resolved_path
                    
                    # Load YAML configuration
                    with open(config_path, 'r') as f:
                        agent_config_data = yaml.safe_load(f)
                    
                    # Create AgentConfig object
                    agent_config = AgentConfig(**agent_config_data)
                    
                    # CRITICAL FIX: Pass AgentConfig object, not file path
                    self.tools['synonym_agent'] = agent_class.from_config(agent_config)
                    self.nb_logger.info(f"✅ Synonym agent loaded from {config_path}")
                else:
                    # Load agent from inline config
                    agent_config_data = synonym_config.get('config', {})
                    agent_config = AgentConfig(**agent_config_data)
                    self.tools['synonym_agent'] = agent_class.from_config(agent_config)
                    self.nb_logger.info("✅ Synonym agent loaded from inline config")
                    
            except Exception as e:
                self.nb_logger.error(f"❌ Failed to initialize synonym agent: {e}")
                import traceback
                self.nb_logger.error(f"Full traceback: {traceback.format_exc()}")
                # Don't raise - allow step to continue without synonym agent
        else:
            self.nb_logger.info("ℹ️ Synonym agent not enabled in configuration")
    
    def _load_class(self, class_path: str):
        """Load a class from its full module path"""
        module_path, class_name = class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    
    async def initialize(self) -> None:
        """Initialize the step and its dependencies"""
        await super().initialize()
        
        # Initialize tools if they exist
        for tool_name, tool in self.tools.items():
            if hasattr(tool, 'initialize'):
                try:
                    await tool.initialize()
                    self.nb_logger.info(f"✅ Tool {tool_name} initialized successfully")
                except Exception as e:
                    self.nb_logger.error(f"❌ Failed to initialize tool {tool_name}: {e}")
                    # Continue with other tools
    
    async def _initialize_synonym_agent(self) -> None:
        """Initialize the protein synonym agent following NanoBrain patterns"""
        try:
            self.nb_logger.info(f"🤖 Initializing ProteinSynonymAgent from {self.synonym_agent_config_path}")
            
            # Import here to avoid circular dependencies
            from nanobrain.library.agents.specialized import ProteinSynonymAgent
            
            # Load YAML manually since AgentConfig doesn't have from_yaml
            import yaml
            from pathlib import Path
            
            # Resolve config path
            config_path = self._resolve_agent_config_path(self.synonym_agent_config_path)
            
            with open(config_path, 'r') as f:
                agent_config_data = yaml.safe_load(f)
            
            # Create AgentConfig from dict
            agent_config = AgentConfig(**agent_config_data)
            
            # Create agent using from_config pattern
            self.synonym_agent = ProteinSynonymAgent.from_config(agent_config)
            
            # Initialize the agent
            await self.synonym_agent.initialize()
            
            self.nb_logger.info("✅ ProteinSynonymAgent initialized successfully")
            
        except Exception as e:
            self.nb_logger.error(f"❌ Failed to initialize synonym agent: {e}")
            import traceback
            self.nb_logger.error(f"Full traceback: {traceback.format_exc()}")
            raise  # Re-raise to fail fast instead of degrading gracefully
    
    def _resolve_agent_config_path(self, config_path: str) -> Path:
        """Resolve agent config path relative to project root."""
        from pathlib import Path
        
        # If absolute path, use as-is
        if Path(config_path).is_absolute():
            return Path(config_path)
        
        # Try relative to current file
        current_dir = Path(__file__).parent
        
        # Try multiple resolution strategies
        strategies = [
            # Direct path from current working directory
            Path(config_path),
            # Relative to current file's directory
            current_dir / config_path,
            # Relative to project root (5 levels up from current file)
            current_dir.parents[4] / config_path,
            # Relative to library directory
            current_dir.parents[2] / "agents" / "specialized" / "config" / "ProteinSynonymAgent.yml",
            # Alternative path structure
            current_dir.parents[3] / "agents" / "specialized" / "config" / "ProteinSynonymAgent.yml"
        ]
        
        for path in strategies:
            if path.exists():
                self.nb_logger.info(f"✅ Found agent config at: {path}")
                return path
        
        # Log all attempted paths for debugging
        self.nb_logger.error(f"❌ Could not find agent config file. Attempted paths:")
        for i, path in enumerate(strategies):
            self.nb_logger.error(f"  {i+1}. {path}")
        
        raise FileNotFoundError(f"Could not find agent config: {config_path}")
    
    async def cleanup(self) -> None:
        """Cleanup resources including tools"""
        # Cleanup tools
        for tool_name, tool in self.tools.items():
            if hasattr(tool, 'shutdown'):
                try:
                    await tool.shutdown()
                    self.nb_logger.info(f"🧹 Tool {tool_name} shutdown completed")
                except Exception as e:
                    self.nb_logger.error(f"Error shutting down tool {tool_name}: {e}")
        
        await super().cleanup()
        
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process method required by Step base class.
        
        This method now handles two types of inputs:
        1. Initial data from BV-BRC acquisition
        2. Synonym response from ProteinSynonymAgentStep
        """
        self.nb_logger.info("🔄 Processing annotation mapping step")
        
        # Check if this is a response from synonym agent
        if 'data_unit' in input_data:
            data_unit = input_data['data_unit']
            if isinstance(data_unit, DataUnit):
                context = await data_unit.get()
                if isinstance(context, dict) and context.get('stage') == 'synonym_response':
                    return await self._handle_synonym_response(data_unit)
        
        # Otherwise, handle as initial data
        return await self._handle_initial_data(input_data)
    
    async def _handle_initial_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process initial annotation data and optionally send to synonym agent"""
        
        # If synonym resolution is not enabled, process normally
        if not self.enable_synonym_resolution:
            result = await self.execute(input_data)
            self.nb_logger.info(f"✅ Annotation mapping completed without synonym resolution")
            return result
        
        # Extract unique products for synonym analysis
        protein_annotations = input_data.get('protein_annotations', [])
        unique_products = self._extract_unique_products(protein_annotations)
        
        if not unique_products:
            # No products to resolve, process normally
            result = await self.execute(input_data)
            return result
        
        # Extract virus information
        filtered_genomes = input_data.get('filtered_genomes', [])
        virus_info = self._extract_virus_info(input_data, filtered_genomes)
        
        # Create context object for synonym request
        request_id = str(uuid.uuid4())
        context = {
            'request_id': request_id,
            'stage': 'synonym_request',
            'timestamp': time.time(),
            
            # Data for synonym agent
            'virus_info': virus_info,
            'products': list(unique_products),
            
            # Store original data for later processing
            'original_data': input_data
        }
        
        # Store context for correlation
        self.pending_requests[request_id] = context
        
        # Create DataUnit configuration
        data_unit_config = DataUnitConfig(
            data_type=DataUnitType.OBJECT,
            name=f"synonym_request_{request_id}"
        )
        
        # Create DataUnit with context reference
        request_unit = DataUnit.from_config(data_unit_config)
        await request_unit.set(context)
        
        # Deposit to output port - Link will deliver to synonym agent
        await self.deposit_output(request_unit)
        
        self.nb_logger.info(f"📤 Sent synonym request {request_id} with {len(unique_products)} products")
        
        # Return status while waiting for response
        return {
            'status': 'processing',
            'message': 'Synonym resolution in progress',
            'request_id': request_id
        }
    
    async def _handle_synonym_response(self, data_unit: DataUnit) -> Dict[str, Any]:
        """Handle response from synonym agent and complete processing"""
        
        context = await data_unit.get()
        request_id = context.get('request_id')
        
        # Verify this is a valid response
        if request_id not in self.pending_requests:
            self.nb_logger.error(f"Received unknown synonym response: {request_id}")
            return {'error': 'Invalid response', 'status': 'failed'}
        
        # Remove from pending
        stored_context = self.pending_requests.pop(request_id)
        
        # Extract results added by synonym agent
        synonym_groups = context.get('synonym_groups', {})
        protein_classifications = context.get('protein_classifications', {})
        
        self.nb_logger.info(f"📥 Received synonym response {request_id}: {len(synonym_groups)} groups")
        
        # Get original data
        original_data = context['original_data']
        
        # Add synonym groups to original data for execute method
        original_data['_synonym_groups'] = synonym_groups
        original_data['_protein_classifications'] = protein_classifications
        
        # Now process with synonym information
        result = await self.execute(original_data)
        
        # Add synonym resolution metadata
        result['synonym_resolution_applied'] = True
        result['synonym_groups'] = synonym_groups
        result['synonym_processing_time'] = context.get('processing_time', 0)
        
        self.nb_logger.info(f"✅ Annotation mapping completed with synonym resolution")
        return result

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute annotation mapping step with synonym resolution
        
        Args:
            input_data: Contains BV-BRC acquisition results with protein_annotations, unique_proteins, etc.
            
        Returns:
            Dict with mapped_proteins and standardized annotations
        """
        
        step_start_time = time.time()
        
        try:
            self.nb_logger.info("🔍 Starting annotation mapping with ICTV integration")
            
            # Extract data from BV-BRC acquisition result
            protein_annotations = input_data.get('protein_annotations', [])
            unique_proteins = input_data.get('unique_proteins', [])
            protein_sequences = input_data.get('protein_sequences', [])
            filtered_genomes = input_data.get('filtered_genomes', [])
            
            self.nb_logger.info(f"🔍 Input data keys: {list(input_data.keys())}")
            self.nb_logger.info(f"📊 Received {len(protein_annotations)} protein annotations")
            self.nb_logger.info(f"🧬 Received {len(unique_proteins)} unique proteins")
            self.nb_logger.info(f"🧬 Received {len(protein_sequences)} protein sequences")
            
            if not protein_annotations:
                self.nb_logger.warning("⚠️ No protein annotations found in input data")
                return {
                    'success': False,
                    'mapped_proteins': {},
                    'standardized_annotations': [],
                    'execution_time': time.time() - step_start_time,
                    'ictv_mapping_applied': False,
                    'synonym_resolution_applied': False
                }
            
            # Extract virus information for synonym agent
            virus_info = self._extract_virus_info(input_data, filtered_genomes)
            
            # Extract all unique product names for synonym analysis
            all_products = self._extract_unique_products(protein_annotations)
            self.nb_logger.info(f"📋 Found {len(all_products)} unique product names")
            
            # Perform synonym resolution if enabled
            synonym_groups = {}
            if 'synonym_agent' in self.tools and all_products:
                try:
                    self.nb_logger.info(f"🤖 Performing synonym analysis on {len(all_products)} products")
                    self.nb_logger.info(f"🦠 Virus context: {virus_info}")
                    
                    synonym_agent = self.tools['synonym_agent']
                    synonym_groups = await synonym_agent.identify_synonyms(
                        list(all_products), 
                        virus_info
                    )
                    
                    self.nb_logger.info(f"✅ Found {len(synonym_groups)} synonym groups")
                    
                    # Log sample synonym groups
                    for canonical, synonyms in list(synonym_groups.items())[:3]:
                        self.nb_logger.info(f"  📌 {canonical} -> {[s[0] for s in synonyms[:3]]}")
                        
                except Exception as e:
                    self.nb_logger.error(f"❌ Synonym resolution failed: {e}")
                    synonym_groups = {}
            
            # Check if synonym groups were provided via Link-based communication
            if '_synonym_groups' in input_data:
                synonym_groups = input_data.get('_synonym_groups', {})
                self.nb_logger.info(f"🔗 Using {len(synonym_groups)} synonym groups from Link communication")
            
            # Create sequence lookup for merging
            sequence_lookup = self._create_sequence_lookup(protein_sequences)
            
            # Create lookup for unique proteins by aa_sequence_md5
            protein_lookup = self._create_protein_lookup(unique_proteins)
            
            # Process annotations with synonym resolution
            standardized_annotations = await self._process_annotations_with_synonyms(
                protein_annotations,
                protein_lookup,
                sequence_lookup,
                synonym_groups
            )
            
            # Group proteins by classification
            mapped_proteins = self._group_proteins_by_class(standardized_annotations)
            
            # Create genome schematics
            genome_schematics = self._create_genome_schematics(filtered_genomes, len(protein_annotations))
            
            execution_time = time.time() - step_start_time
            
            # Calculate statistics
            stats = self._calculate_annotation_statistics(
                standardized_annotations, 
                synonym_groups,
                len(protein_annotations)
            )
            
            self.nb_logger.info(f"✅ Annotation mapping completed in {execution_time:.2f} seconds")
            self.nb_logger.info(f"📈 Mapped {len(standardized_annotations)} proteins into {len(mapped_proteins)} classes")
            if synonym_groups:
                self.nb_logger.info(f"🔗 Applied {len(synonym_groups)} synonym groups")
            
            return {
                'success': True,
                'mapped_proteins': mapped_proteins,
                'standardized_annotations': standardized_annotations,
                'genome_schematics': genome_schematics,
                'execution_time': execution_time,
                'ictv_mapping_applied': True,
                'synonym_resolution_applied': bool(synonym_groups),
                'annotation_statistics': stats,
                'synonym_groups': synonym_groups  # Include for downstream analysis
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Annotation mapping failed: {e}")
            raise
    
    def _extract_virus_info(self, input_data: Dict[str, Any], 
                           filtered_genomes: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract virus information from input data"""
        virus_info = {
            'genus': 'Alphavirus',  # Default
            'species': '',
            'strain': '',
            'family': 'Togaviridae'
        }
        
        # Try to extract from workflow metadata
        if 'workflow_metadata' in input_data:
            metadata = input_data['workflow_metadata']
            virus_info['genus'] = metadata.get('target_genus', virus_info['genus'])
            virus_info['species'] = metadata.get('target_species', '')
        
        # Try to extract from genome data
        if filtered_genomes:
            sample_genome = filtered_genomes[0]
            genome_name = sample_genome.get('genome_name', '')
            
            # Parse species from genome name
            if 'Chikungunya' in genome_name:
                virus_info['species'] = 'Chikungunya virus'
            elif 'Eastern equine' in genome_name:
                virus_info['species'] = 'Eastern equine encephalitis virus'
            elif 'Venezuelan equine' in genome_name:
                virus_info['species'] = 'Venezuelan equine encephalitis virus'
            # Add more species patterns as needed
            
            # Extract strain if present
            if 'strain' in sample_genome:
                virus_info['strain'] = sample_genome['strain']
        
        return virus_info
    
    def _extract_unique_products(self, protein_annotations: List[Dict[str, Any]]) -> Set[str]:
        """Extract all unique product names from annotations"""
        products = set()
        
        for annotation in protein_annotations:
            if isinstance(annotation, dict):
                product = annotation.get('product', '')
                if product and product != 'unknown' and product != 'hypothetical protein':
                    products.add(product)
        
        return products
    
    def _create_sequence_lookup(self, protein_sequences: List[Dict[str, Any]]) -> Dict[str, str]:
        """Create lookup dictionary for protein sequences"""
        sequence_lookup = {}
        
        if protein_sequences:
            for seq in protein_sequences:
                if isinstance(seq, dict) and 'aa_sequence_md5' in seq:
                    sequence_lookup[seq['aa_sequence_md5']] = seq.get('aa_sequence', '')
            
            self.nb_logger.info(f"🔍 Created sequence lookup for {len(sequence_lookup)} sequences")
        
        return sequence_lookup
    
    def _create_protein_lookup(self, unique_proteins: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Create lookup dictionary for unique proteins"""
        protein_lookup = {}
        
        for protein in unique_proteins:
            if isinstance(protein, dict) and 'aa_sequence_md5' in protein:
                protein_lookup[protein['aa_sequence_md5']] = protein
        
        self.nb_logger.info(f"🔍 Created lookup for {len(protein_lookup)} unique proteins")
        
        return protein_lookup
    
    async def _process_annotations_with_synonyms(self,
                                               protein_annotations: List[Dict[str, Any]],
                                               protein_lookup: Dict[str, Dict[str, Any]],
                                               sequence_lookup: Dict[str, str],
                                               synonym_groups: Dict[str, List[Tuple[str, float]]]) -> List[Dict[str, Any]]:
        """Process annotations and apply synonym resolution"""
        
        # Create reverse synonym mapping for quick lookup
        product_to_canonical = {}
        if synonym_groups:
            for canonical, synonyms in synonym_groups.items():
                product_to_canonical[canonical] = canonical
                for synonym, confidence in synonyms:
                    product_to_canonical[synonym] = canonical
        
        standardized_annotations = []
        
        for annotation in protein_annotations:
            if not isinstance(annotation, dict):
                continue
            
            product = annotation.get('product', 'unknown')
            aa_sequence_md5 = annotation.get('aa_sequence_md5', '')
            
            # Apply synonym resolution
            canonical_product = product_to_canonical.get(product, product)
            is_synonym_resolved = (canonical_product != product)
            
            # Get protein data and sequence
            protein_data = protein_lookup.get(aa_sequence_md5, {})
            aa_sequence = sequence_lookup.get(aa_sequence_md5, '') or protein_data.get('aa_sequence', '')
            genome_id = protein_data.get('genome_id', '') or annotation.get('genome_id', '')
            
            # Classify protein
            protein_class = self._classify_protein(canonical_product)
            
            # Create standardized annotation
            standardized_annotation = {
                'patric_id': annotation.get('patric_id', '') or protein_data.get('patric_id', ''),
                'aa_sequence_md5': aa_sequence_md5,
                'aa_sequence': aa_sequence,
                'product': product,  # Original product name
                'canonical_product': canonical_product,  # Resolved canonical name
                'gene': annotation.get('gene', ''),
                'genome_id': genome_id,
                'original_annotation': annotation,
                'standard_name': canonical_product,
                'protein_class': protein_class,
                'confidence': 0.9 if is_synonym_resolved else 0.8,
                'ictv_mapped': True,
                'synonym_resolved': is_synonym_resolved
            }
            
            standardized_annotations.append(standardized_annotation)
        
        return standardized_annotations
    
    def _group_proteins_by_class(self, standardized_annotations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group proteins by their classification"""
        mapped_proteins = {}
        
        for annotation in standardized_annotations:
            protein_class = annotation['protein_class']
            
            if protein_class not in mapped_proteins:
                mapped_proteins[protein_class] = []
            
            mapped_proteins[protein_class].append(annotation)
        
        return mapped_proteins
    
    def _create_genome_schematics(self, filtered_genomes: List[Dict[str, Any]], 
                                 annotation_count: int) -> List[Dict[str, Any]]:
        """Create genome schematics based on actual data"""
        genome_schematics = []
        
        for genome in filtered_genomes:
            if isinstance(genome, dict):
                genome_schematic = {
                    'genome_id': genome.get('genome_id', 'unknown'),
                    'genome_name': genome.get('genome_name', 'unknown'),
                    'genome_organization': "5'-nsP1-nsP2-nsP3-nsP4-Capsid-E3-E2-6K-E1-3'",
                    'protein_count': annotation_count,
                    'ictv_mapping_applied': True
                }
                genome_schematics.append(genome_schematic)
        
        return genome_schematics
    
    def _calculate_annotation_statistics(self, 
                                       standardized_annotations: List[Dict[str, Any]],
                                       synonym_groups: Dict[str, List[Tuple[str, float]]],
                                       total_annotations: int) -> Dict[str, Any]:
        """Calculate comprehensive annotation statistics"""
        
        # Count synonym-resolved annotations
        synonym_resolved_count = sum(1 for ann in standardized_annotations 
                                   if ann.get('synonym_resolved', False))
        
        # Get protein class distribution
        class_distribution = {}
        for ann in standardized_annotations:
            protein_class = ann['protein_class']
            class_distribution[protein_class] = class_distribution.get(protein_class, 0) + 1
        
        # Calculate confidence distribution
        confidence_dist = self._calculate_confidence_distribution(standardized_annotations)
        
        return {
            'total_annotations': total_annotations,
            'standardized_annotations': len(standardized_annotations),
            'synonym_groups_applied': len(synonym_groups),
            'annotations_with_synonyms': synonym_resolved_count,
            'protein_classes': list(class_distribution.keys()),
            'protein_class_distribution': class_distribution,
            'confidence_distribution': confidence_dist
        }
            
    def _classify_protein(self, product: str) -> str:
        """Classify protein based on product description"""
        product_lower = product.lower()
        
        # Check for nonstructural proteins
        if any(keyword in product_lower for keyword in ['nsp1', 'nonstructural protein 1', 'replicase']):
            return 'nsP1'
        elif any(keyword in product_lower for keyword in ['nsp2', 'nonstructural protein 2', 'protease']):
            return 'nsP2'
        elif any(keyword in product_lower for keyword in ['nsp3', 'nonstructural protein 3']):
            return 'nsP3'
        elif any(keyword in product_lower for keyword in ['nsp4', 'nonstructural protein 4', 'polymerase']):
            return 'nsP4'
        
        # Check for structural proteins
        elif any(keyword in product_lower for keyword in ['capsid', 'structural protein c', 'core protein']):
            return 'capsid'
        elif any(keyword in product_lower for keyword in ['envelope protein e3', 'glycoprotein e3', 'e3 protein']):
            return 'E3'
        elif any(keyword in product_lower for keyword in ['envelope protein e2', 'glycoprotein e2', 'e2 protein']):
            return 'E2'
        elif any(keyword in product_lower for keyword in ['6k protein', 'small membrane', '6k']):
            return '6K'
        elif any(keyword in product_lower for keyword in ['envelope protein e1', 'glycoprotein e1', 'e1 protein']):
            return 'E1'
        
        # Check for polyproteins
        elif 'polyprotein' in product_lower:
            if 'structural' in product_lower:
                return 'structural_polyprotein'
            elif 'nonstructural' in product_lower:
                return 'nonstructural_polyprotein'
            else:
                return 'polyprotein'
        
        else:
            return 'unknown'
            
    def _calculate_confidence_distribution(self, annotations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate confidence distribution of annotations"""
        distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for annotation in annotations:
            confidence = annotation.get('confidence', 0.0)
            if confidence >= 0.8:
                distribution['high'] += 1
            elif confidence >= 0.5:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
                
        return distribution 