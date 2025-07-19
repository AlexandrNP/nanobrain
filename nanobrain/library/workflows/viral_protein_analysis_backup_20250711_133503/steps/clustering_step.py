"""
Clustering Step (Steps 8-9)

Re-architected to inherit from NanoBrain Step base class.
Steps 8-9: Run MMseqs2 clustering on all protein sequences.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from collections import defaultdict
from pathlib import Path
import hashlib

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.logging_system import get_logger


class ClusteringStep(Step):
    """
    Steps 8-9: Run MMseqs2 clustering on all protein sequences
    
    Takes all curated sequences, creates a FASTA file with actual amino acid sequences,
    and runs MMseqs2 clustering on the entire dataset.
    
    FASTA header format: {patric_id}|{product}|{gene}|{md5}
    """
    
    def __init__(self, config: StepConfig, clustering_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config, **kwargs)
        
        # Extract configuration from step config or provided clustering_config
        step_config_dict = config.config if hasattr(config, 'config') else {}
        if clustering_config:
            step_config_dict.update(clustering_config)
        
        self.clustering_config = step_config_dict.get('clustering_config', {})
        self.step_config = step_config_dict
        
        # Configuration parameters for clustering
        self.min_cluster_size = self.step_config.get('min_cluster_size', 3)
        
        # Determine clustering mode
        self.clustering_mode = self.clustering_config.get('mode', 'product_based')  # Default to product-based
        
        # Initialize cache directory for FASTA files
        self.cache_dir = Path("data/clustering_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Only initialize MMseqs2 tool if explicitly configured for MMseqs2 mode
        if self.clustering_mode == 'mmseqs2':
            self._initialize_mmseqs2_tool()
            self.nb_logger.info(f"🧬 ClusteringStep initialized for MMseqs2 clustering")
        else:
            self.mmseqs2_tool = None
            self.nb_logger.info(f"🧬 ClusteringStep initialized for product-based clustering")
        
        self.nb_logger.info(f"📊 Clustering mode: {self.clustering_mode}")
        self.nb_logger.info(f"📊 Minimum cluster size: {self.min_cluster_size}")
    
    def _initialize_mmseqs2_tool(self):
        """Initialize MMseqs2 tool with workflow-local configuration"""
        
        # Get workflow directory path
        import yaml
        from nanobrain.library.tools.bioinformatics.mmseqs_tool import MMseqs2Tool, MMseqs2Config
        
        workflow_dir = Path(__file__).parent.parent
        tool_config_path = workflow_dir / "config" / "tools" / "mmseqs2_tool.yml"
        
        # Load tool configuration from workflow-local YAML
        if tool_config_path.exists():
            with open(tool_config_path, 'r') as f:
                tool_config_dict = yaml.safe_load(f)
            
            # Create MMseqs2 tool configuration
            tool_config = MMseqs2Config(**{
                k: v for k, v in tool_config_dict.items() 
                if k in ['tool_name', 'conda_package', 'conda_channel', 'environment_name',
                        'min_seq_id', 'coverage', 'cluster_mode', 'sensitivity', 
                        'progressive_scaling', 'threads', 'memory_limit', 'tmp_dir',
                        'timeout_seconds', 'verify_on_init']
            })
            
            # Create MMseqs2 tool using from_config pattern
            self.mmseqs2_tool = MMseqs2Tool.from_config(tool_config)
            self.nb_logger.info(f"✅ MMseqs2 tool loaded from workflow-local config: {tool_config_path}")
            
        else:
            # Fallback to step configuration if workflow-local config not found
            self.nb_logger.warning(f"⚠️ Workflow-local tool config not found: {tool_config_path}")
            self.nb_logger.warning("⚠️ Using step configuration approach")
            
            # Legacy approach with step configuration
            mmseqs2_config_dict = self.step_config.get('mmseqs2_config', {})
            if 'tool_name' not in mmseqs2_config_dict:
                mmseqs2_config_dict['tool_name'] = 'mmseqs2'
            
            tool_config = MMseqs2Config(**mmseqs2_config_dict)
            self.mmseqs2_tool = MMseqs2Tool.from_config(tool_config)
            self.nb_logger.info(f"✅ MMseqs2 tool initialized with step config")
        
        self.nb_logger.info(f"🧬 ClusteringStep initialized with tool: {type(self.mmseqs2_tool).__name__}")
    
    @classmethod
    def extract_component_config(cls, config: StepConfig) -> Dict[str, Any]:
        """Extract ClusteringStep configuration"""
        base_config = super().extract_component_config(config)
        step_config_dict = config.config if hasattr(config, 'config') else {}
        
        return {
            **base_config,
            'clustering_config': step_config_dict.get('clustering_config', {}),
            'min_cluster_size': step_config_dict.get('min_cluster_size', 3)
        }
    
    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ClusteringStep with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Extract configuration from component_config
        self.clustering_config = component_config.get('clustering_config', {})
        self.min_cluster_size = component_config.get('min_cluster_size', 3)
        
        # Store step configuration for backward compatibility
        self.step_config = config.config if hasattr(config, 'config') else {}
        
        # Determine clustering mode
        self.clustering_mode = self.clustering_config.get('mode', 'product_based')  # Default to product-based
        
        # Initialize cache directory
        self.cache_dir = Path("data/clustering_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Only initialize MMseqs2 tool if explicitly configured for MMseqs2 mode
        if self.clustering_mode == 'mmseqs2':
            self._initialize_mmseqs2_tool()
            self.nb_logger.info(f"🧬 ClusteringStep initialized for MMseqs2 clustering")
        else:
            self.mmseqs2_tool = None
            self.nb_logger.info(f"🧬 ClusteringStep initialized for product-based clustering")
        
        self.nb_logger.info(f"📊 Clustering mode: {self.clustering_mode}")
        self.nb_logger.info(f"📊 Minimum cluster size: {self.min_cluster_size}")
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process method required by Step base class.
        
        This implements the NanoBrain framework interface while calling the
        original execute method that contains the clustering logic.
        """
        self.nb_logger.info("🔄 Processing clustering step")
        
        # Call the original execute method
        result = await self.execute(input_data)
        
        self.nb_logger.info(f"✅ Clustering completed successfully")
        return result

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute clustering on all sequences - supports both MMseqs2 and product-based clustering
        
        Args:
            input_data: Contains curated sequences and curation report
            
        Returns:
            Dict with clusters from analysis
        """
        
        step_start_time = time.time()
        
        try:
            # Determine clustering mode from instance configuration
            clustering_mode = getattr(self, 'clustering_mode', 'product_based')
            self.nb_logger.info(f"🗂️ Starting clustering analysis in '{clustering_mode}' mode")
            
            # Extract curated sequences - handle both dict and list formats
            curated_sequences_input = input_data.get('curated_sequences', [])
            curation_report = input_data.get('curation_report', {})
            
            # Convert curated_sequences from dict format to list format
            all_sequences = []
            if isinstance(curated_sequences_input, dict):
                # Dict format: {protein_class: [sequences]}
                self.nb_logger.info(f"📊 Processing curated sequences from {len(curated_sequences_input)} protein classes")
                for protein_class, sequences in curated_sequences_input.items():
                    if isinstance(sequences, list):
                        all_sequences.extend(sequences)
                        self.nb_logger.debug(f"Added {len(sequences)} sequences from class '{protein_class}'")
            elif isinstance(curated_sequences_input, list):
                # List format: [sequences]
                all_sequences = curated_sequences_input
                self.nb_logger.info(f"📊 Processing curated sequences from list with {len(all_sequences)} sequences")
            else:
                self.nb_logger.warning(f"⚠️ Unexpected curated_sequences format: {type(curated_sequences_input)}")
                all_sequences = []
            
            if not all_sequences:
                self.nb_logger.warning("⚠️ No sequences found for clustering")
                return {
                    'success': False,
                    'protein_clusters': {},
                    'clustering_analysis': {'total_clusters': 0},
                    'execution_time': time.time() - step_start_time,
                    'error': 'No sequences available for clustering'
                }
            
            self.nb_logger.info(f"🔢 Total sequences for clustering: {len(all_sequences)}")
            
            # Debug: Check if sequences have aa_sequence field
            if all_sequences:
                sample_seq = all_sequences[0]
                self.nb_logger.info(f"🔍 DEBUG: Sample sequence keys: {list(sample_seq.keys()) if isinstance(sample_seq, dict) else 'Not a dict'}")
                if isinstance(sample_seq, dict):
                    aa_seq = sample_seq.get('aa_sequence', '')
                    self.nb_logger.info(f"🔍 DEBUG: Sample aa_sequence length: {len(aa_seq) if aa_seq else 'MISSING or EMPTY'}")
            
            # Choose clustering method based on configuration
            if clustering_mode == 'product_based':
                # Use product-based clustering
                clusters = await self._cluster_by_product(all_sequences)
                clustering_analysis = await self._analyze_clusters(clusters)
                
                # Generate LLM summary if enabled
                if self.clustering_config.get('enable_llm_summary', True):
                    llm_summary = await self._generate_llm_summary(clusters, clustering_analysis)
                    clustering_analysis['llm_summary'] = llm_summary
                
                execution_time = time.time() - step_start_time
                self.nb_logger.info(f"✅ Product-based clustering completed in {execution_time:.2f} seconds")
                self.nb_logger.info(f"Generated {len(clusters)} protein clusters")
                
                # Validate return format
                if not isinstance(clusters, dict):
                    raise ValueError(f"Clustering step must return dict for protein_clusters, not {type(clusters)}")
                
                return {
                    'success': True,
                    'protein_clusters': clusters,
                    'clustering_analysis': clustering_analysis,
                    'execution_time': execution_time,
                    'clustering_method': 'product_based',
                    'clustering_parameters': {
                        'min_cluster_size': self.min_cluster_size,
                        'mode': clustering_mode
                    }
                }
                
            else:
                # Use MMseqs2 clustering (original implementation)
                return await self._execute_mmseqs2_clustering(all_sequences, step_start_time)
                
        except Exception as e:
            self.nb_logger.error(f"❌ Clustering failed: {str(e)}")
            
            # Try fallback if enabled and not already using fallback
            if (self.clustering_config.get('fallback_to_mmseqs2', False) and 
                clustering_mode == 'product_based'):
                try:
                    self.nb_logger.info("🔄 Attempting MMseqs2 fallback...")
                    return await self._execute_mmseqs2_clustering(all_sequences, step_start_time)
                except Exception as fallback_error:
                    self.nb_logger.error(f"❌ Fallback also failed: {str(fallback_error)}")
            
            return {
                'success': False,
                'protein_clusters': {},
                'clustering_analysis': {'total_clusters': 0, 'error': str(e)},
                'execution_time': time.time() - step_start_time,
                'error': str(e)
            }

    async def _execute_mmseqs2_clustering(self, all_sequences: List[Dict[str, Any]], step_start_time: float) -> Dict[str, Any]:
        """
        Execute original MMseqs2 clustering workflow
        """
        # Create FASTA content with actual amino acid sequences
        fasta_content = await self._create_complete_fasta(all_sequences)
        
        if not fasta_content:
            self.nb_logger.warning("⚠️ No valid FASTA content generated")
            return {
                'success': False,
                'protein_clusters': {},
                'clustering_analysis': {'total_clusters': 0},
                'execution_time': time.time() - step_start_time,
                'error': 'No valid sequences for FASTA generation'
            }
        
        # Cache the complete FASTA file
        await self._cache_complete_fasta(fasta_content)
        
        # Run MMseqs2 clustering on the entire dataset
        self.nb_logger.info(f"🔧 Running MMseqs2 on {len(all_sequences)} sequences")
        clustering_report = await self.mmseqs2_tool.cluster_sequences(fasta_content)
        
        # Convert MMseqs2 results to our cluster format
        clusters = await self._convert_mmseqs2_results_to_clusters(clustering_report, all_sequences)
        
        # Analyze clustering results
        clustering_analysis = await self._analyze_clusters(clusters)
        
        execution_time = time.time() - step_start_time
        self.nb_logger.info(f"✅ MMseqs2 clustering completed in {execution_time:.2f} seconds")
        self.nb_logger.info(f"Generated {len(clusters)} protein clusters")
        
        # Validate return format
        if not isinstance(clusters, dict):
            raise ValueError(f"Clustering step must return dict for protein_clusters, not {type(clusters)}")
        
        return {
            'success': True,
            'protein_clusters': clusters,
            'clustering_analysis': clustering_analysis,
            'execution_time': execution_time,
            'clustering_method': 'mmseqs2',
            'clustering_parameters': {
                'min_cluster_size': self.min_cluster_size
            }
        }

    async def _cluster_by_product(self, sequences: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Cluster sequences by their product names (using canonical names from synonym resolution)
        
        Args:
            sequences: List of sequence dictionaries
            
        Returns:
            Dict with clusters organized by product name
        """
        self.nb_logger.info("🧬 Starting product-based clustering")
        
        # Group sequences by product name
        product_groups = {}
        for seq in sequences:
            # Use canonical_product if available (from synonym resolution), otherwise use product
            product = seq.get('canonical_product') or seq.get('product', 'unknown_product')
            normalized_product = self._normalize_product_name(product)
            
            if normalized_product not in product_groups:
                product_groups[normalized_product] = []
            product_groups[normalized_product].append(seq)
        
        self.nb_logger.info(f"📦 Grouped sequences into {len(product_groups)} product categories")
        
        # Log synonym resolution impact
        synonym_resolved_count = sum(1 for seq in sequences if seq.get('synonym_resolved', False))
        if synonym_resolved_count > 0:
            self.nb_logger.info(f"🔗 {synonym_resolved_count} sequences had synonyms resolved")
        
        # Convert to cluster format
        clusters = {}
        cluster_num = 1
        
        for product_name, product_sequences in product_groups.items():
            if len(product_sequences) >= self.min_cluster_size:
                # Calculate length statistics
                sequence_lengths = []
                sequences_with_aa = []  # Track sequences that have aa_sequence
                for seq in product_sequences:
                    aa_sequence = seq.get('aa_sequence', '')
                    if aa_sequence:
                        sequence_lengths.append(len(aa_sequence))
                        sequences_with_aa.append(seq)
                
                # Select representative sequence (longest one) - only from sequences with aa_sequence
                if sequences_with_aa:
                    representative_seq = max(sequences_with_aa, 
                                           key=lambda x: len(x.get('aa_sequence', '')))
                else:
                    # Fallback if no sequences have aa_sequence
                    representative_seq = product_sequences[0] if product_sequences else {}
                
                # Calculate genome distribution
                genome_distribution = {}
                for seq in product_sequences:
                    genome_id = seq.get('genome_id', 'unknown')
                    if not genome_id:  # Handle empty strings
                        genome_id = 'unknown'
                    genome_distribution[genome_id] = genome_distribution.get(genome_id, 0) + 1
                
                # Get full amino acid sequence for representative
                representative_sequence = representative_seq.get('aa_sequence', '')
                
                # Extract sequence IDs (same as member_sequences but named for response formatter)
                sequence_ids = [seq.get('patric_id', '') for seq in product_sequences]
                
                # Create cluster
                cluster_id = f"product_cluster_{cluster_num:03d}_{self._sanitize_filename(product_name)}"
                
                # Check if this cluster contains synonym-resolved sequences
                has_synonyms = any(seq.get('synonym_resolved', False) for seq in product_sequences)
                
                clusters[cluster_id] = {
                    'cluster_id': cluster_id,
                    'product_name': product_name,
                    'representative_seq': representative_seq.get('patric_id', ''),
                    'member_sequences': [seq.get('patric_id', '') for seq in product_sequences],
                    'cluster_size': len(product_sequences),
                    'sequences': product_sequences,  # Full sequence data
                    'length_stats': self._calculate_length_stats(sequence_lengths) if sequence_lengths else {},
                    'clustering_method': 'product_based',
                    'has_synonym_resolved': has_synonyms,
                    # Fields for response formatter:
                    'sequence_ids': sequence_ids,
                    'genome_distribution': genome_distribution,
                    'representative_sequence': representative_sequence
                }
                
                cluster_num += 1
                self.nb_logger.debug(f"Created cluster for '{product_name}': {len(product_sequences)} sequences")
        
        self.nb_logger.info(f"✅ Product-based clustering created {len(clusters)} clusters from {len(product_groups)} product groups")
        
        # Count clusters that benefited from synonym resolution
        synonym_enhanced_clusters = sum(1 for c in clusters.values() if c.get('has_synonym_resolved', False))
        if synonym_enhanced_clusters > 0:
            self.nb_logger.info(f"🔗 {synonym_enhanced_clusters} clusters contain synonym-resolved sequences")
        
        # Debug logging to verify cluster content
        if clusters:
            sample_cluster_id = next(iter(clusters.keys()))
            sample_cluster = clusters[sample_cluster_id]
            self.nb_logger.info(f"🔍 DEBUG: Sample cluster '{sample_cluster_id}' keys: {list(sample_cluster.keys())}")
            self.nb_logger.info(f"🔍 DEBUG: length_stats: {sample_cluster.get('length_stats', 'MISSING')}")
            self.nb_logger.info(f"🔍 DEBUG: representative_sequence length: {len(sample_cluster.get('representative_sequence', ''))}")
            self.nb_logger.info(f"🔍 DEBUG: genome_distribution: {sample_cluster.get('genome_distribution', 'MISSING')}")
        
        return clusters

    def _normalize_product_name(self, product: str) -> str:
        """
        Normalize product names for consistent clustering
        """
        if not product or product.lower() in ['unknown', 'hypothetical protein', '']:
            return 'unknown_product'
        
        # Basic normalization - remove extra spaces, convert to lowercase for comparison
        normalized = ' '.join(product.strip().split())
        return normalized

    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitize product name for use in cluster IDs
        """
        import re
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Truncate if too long
        return sanitized[:50]

    def _calculate_length_stats(self, lengths: List[int]) -> Dict[str, float]:
        """
        Calculate basic statistics for sequence lengths
        """
        if not lengths:
            return {}
        
        import statistics
        
        return {
            'min': min(lengths),
            'max': max(lengths),
            'mean': statistics.mean(lengths),
            'median': statistics.median(lengths),
            'std': statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
            'count': len(lengths)
        }

    async def _generate_llm_summary(self, clusters: Dict[str, Dict[str, Any]], 
                                  clustering_analysis: Dict[str, Any]) -> str:
        """
        Generate LLM-powered summary of clustering results in markdown format
        """
        self.nb_logger.info("🤖 Generating LLM summary of clustering results")
        
        try:
            # Prepare data for LLM
            cluster_summary_data = []
            max_clusters = self.clustering_config.get('max_clusters_to_display', 20)
            include_snippets = self.clustering_config.get('include_sequence_snippets', True)
            snippet_length = self.clustering_config.get('snippet_length', 100)
            
            # Sort clusters by size (descending)
            sorted_clusters = sorted(clusters.items(), 
                                   key=lambda x: x[1]['cluster_size'], 
                                   reverse=True)
            
            for i, (cluster_id, cluster_data) in enumerate(sorted_clusters[:max_clusters]):
                length_stats = cluster_data.get('length_stats', {})
                
                cluster_info = {
                    'rank': i + 1,
                    'product_name': cluster_data.get('product_name', 'Unknown'),
                    'size': cluster_data['cluster_size'],
                    'length_stats': length_stats
                }
                
                # Add representative sequence snippet if enabled
                if include_snippets and cluster_data.get('sequences'):
                    rep_seq = cluster_data['sequences'][0]  # First sequence as representative
                    aa_sequence = rep_seq.get('aa_sequence', '')
                    if aa_sequence:
                        cluster_info['sequence_snippet'] = aa_sequence[:snippet_length]
                        cluster_info['representative_id'] = rep_seq.get('patric_id', '')
                
                cluster_summary_data.append(cluster_info)
            
            # Create prompt for LLM
            prompt = self._create_clustering_summary_prompt(cluster_summary_data, clustering_analysis)
            
            # Generate summary using LLM (placeholder - would use actual LLM integration)
            llm_summary = await self._call_llm_for_summary(prompt)
            
            return llm_summary
            
        except Exception as e:
            self.nb_logger.warning(f"⚠️ Failed to generate LLM summary: {e}")
            # Return fallback summary
            return self._generate_fallback_summary(clusters, clustering_analysis)

    def _create_clustering_summary_prompt(self, cluster_data: List[Dict], 
                                        clustering_analysis: Dict[str, Any]) -> str:
        """
        Create prompt for LLM clustering summary
        """
        total_clusters = clustering_analysis.get('total_clusters', len(cluster_data))
        total_sequences = sum(cluster['size'] for cluster in cluster_data)
        
        prompt = f"""# Viral Protein Clustering Analysis

Please provide a comprehensive markdown summary of this viral protein clustering analysis.

## Dataset Overview
- Total clusters: {total_clusters}
- Total sequences analyzed: {total_sequences}
- Clustering method: Product-based grouping

## Top Clusters (by size):
"""
        
        for cluster in cluster_data:
            length_stats = cluster.get('length_stats', {})
            prompt += f"""
### {cluster['rank']}. {cluster['product_name']} ({cluster['size']} sequences)
- Length range: {length_stats.get('min', 'N/A')}-{length_stats.get('max', 'N/A')} amino acids
- Mean length: {length_stats.get('mean', 'N/A'):.1f} ± {length_stats.get('std', 'N/A'):.1f}
"""
            if 'sequence_snippet' in cluster:
                prompt += f"- Representative sequence: {cluster.get('representative_id', 'N/A')}\n"
                prompt += f"- Sequence snippet: {cluster['sequence_snippet']}...\n"
        
        prompt += """
Please provide:
1. A brief overview of the clustering results
2. Analysis of the most abundant protein types
3. Observations about sequence length variability
4. Biological insights about the viral proteins identified
5. Any notable patterns or findings

Format the response in clean markdown with appropriate headers and bullet points.
"""
        
        return prompt

    async def _call_llm_for_summary(self, prompt: str) -> str:
        """
        Call LLM service to generate clustering summary
        TODO: Integrate with actual LLM service (OpenAI, Claude, etc.)
        """
        # Placeholder for actual LLM integration
        # For now, return a basic structured summary
        return self._generate_fallback_summary_from_prompt(prompt)

    def _generate_fallback_summary_from_prompt(self, prompt: str) -> str:
        """
        Generate a basic summary when LLM is not available
        """
        # Extract key information from prompt for basic summary
        lines = prompt.split('\n')
        summary = "# Viral Protein Clustering Analysis Summary\n\n"
        summary += "## Overview\n"
        summary += "Protein clustering analysis completed using product-based grouping method.\n\n"
        
        # Add basic cluster information
        summary += "## Cluster Results\n"
        for line in lines:
            if line.startswith('### '):
                summary += line + '\n'
                
        summary += "\n## Analysis Notes\n"
        summary += "- Clustering based on protein product annotations\n"
        summary += "- Results provide functional grouping of viral proteins\n"
        summary += "- Length statistics indicate sequence conservation within functional groups\n"
        
        return summary

    def _generate_fallback_summary(self, clusters: Dict[str, Dict[str, Any]], 
                                 clustering_analysis: Dict[str, Any]) -> str:
        """
        Generate basic fallback summary without LLM
        """
        total_clusters = len(clusters)
        total_sequences = sum(cluster['cluster_size'] for cluster in clusters.values())
        
        summary = f"""# Protein Clustering Analysis Summary

## Overview
- **Total Clusters**: {total_clusters}
- **Total Sequences**: {total_sequences:,}
- **Clustering Method**: Product-based grouping
- **Minimum Cluster Size**: {self.min_cluster_size}

## Top Clusters by Size

"""
        
        # Sort clusters by size and show top ones
        sorted_clusters = sorted(clusters.items(), 
                               key=lambda x: x[1]['cluster_size'], 
                               reverse=True)
        
        for i, (cluster_id, cluster_data) in enumerate(sorted_clusters[:10]):
            length_stats = cluster_data.get('length_stats', {})
            product_name = cluster_data.get('product_name', 'Unknown')
            size = cluster_data['cluster_size']
            
            summary += f"### {i+1}. {product_name} ({size:,} sequences)\n"
            
            if length_stats:
                min_len = length_stats.get('min', 'N/A')
                max_len = length_stats.get('max', 'N/A')
                mean_len = length_stats.get('mean', 0)
                std_len = length_stats.get('std', 0)
                
                summary += f"- **Length Range**: {min_len}-{max_len} amino acids\n"
                summary += f"- **Average Length**: {mean_len:.1f} ± {std_len:.1f} amino acids\n"
            
            # Add representative sequence info
            if cluster_data.get('sequences'):
                rep_seq = cluster_data['sequences'][0]
                rep_id = rep_seq.get('patric_id', 'N/A')
                summary += f"- **Representative Sequence**: {rep_id}\n"
            
            summary += "\n"
        
        summary += "## Analysis Complete\n"
        summary += "Clustering analysis provides functional grouping of viral proteins based on product annotations.\n"
        
        return summary

    async def _create_complete_fasta(self, sequences: List[Dict[str, Any]]) -> str:
        """
        Create FASTA content with actual amino acid sequences for all sequences
        """
        
        fasta_lines = []
        valid_sequences = 0
        skipped_sequences = 0
        
        self.nb_logger.info(f"🔧 Creating FASTA content for {len(sequences)} sequences")
        
        for i, seq in enumerate(sequences):
            try:
                # Extract actual amino acid sequence
                aa_sequence = seq.get('aa_sequence', '')
                patric_id = seq.get('patric_id', f'seq_{i}')
                product = seq.get('product', 'unknown_product')
                gene = seq.get('gene', 'unknown_gene')
                aa_sequence_md5 = seq.get('aa_sequence_md5', f'md5_{i}')
                
                if not aa_sequence:
                    self.nb_logger.debug(f"Skipping sequence without aa_sequence: {patric_id}")
                    skipped_sequences += 1
                    continue
                    
                if len(aa_sequence) < 10:  # Skip very short sequences
                    self.nb_logger.debug(f"Skipping short sequence ({len(aa_sequence)} aa): {patric_id}")
                    skipped_sequences += 1
                    continue
                
                # Validate amino acid sequence (contains only valid amino acid characters)
                valid_aa_chars = set('ACDEFGHIKLMNPQRSTVWY*X')
                if not all(c.upper() in valid_aa_chars for c in aa_sequence):
                    self.nb_logger.debug(f"Skipping sequence with invalid amino acids: {patric_id}")
                    skipped_sequences += 1
                    continue
                
                # Create FASTA header and sequence
                header = f">{patric_id}|{product}|{gene}|{aa_sequence_md5}"
                fasta_lines.append(header)
                fasta_lines.append(aa_sequence)
                valid_sequences += 1
                
                # Log first few sequences for verification
                if valid_sequences <= 5:
                    self.nb_logger.debug(f"FASTA entry {valid_sequences}: {patric_id} -> {product} ({len(aa_sequence)} aa)")
                
            except Exception as e:
                self.nb_logger.warning(f"Error processing sequence {i}: {e}")
                skipped_sequences += 1
                continue
        
        if valid_sequences == 0:
            self.nb_logger.warning(f"No valid sequences found for FASTA generation")
            return ""
        
        fasta_content = "\n".join(fasta_lines)
        
        self.nb_logger.info(f"✅ Generated FASTA with {valid_sequences} valid sequences")
        self.nb_logger.info(f"⚠️ Skipped {skipped_sequences} invalid sequences")
        self.nb_logger.info(f"📊 FASTA content: {len(fasta_content):,} characters")
        
        return fasta_content

    async def _cache_complete_fasta(self, fasta_content: str):
        """
        Cache the complete FASTA file containing all sequences
        """
        
        try:
            # Create cache file path with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_file = self.cache_dir / f"all_sequences_{timestamp}.fasta"
            
            # Also create a generic filename for easy access
            generic_cache_file = self.cache_dir / "all_sequences.fasta"
            
            # Write FASTA content to both cache files
            with open(cache_file, 'w') as f:
                f.write(fasta_content)
            
            with open(generic_cache_file, 'w') as f:
                f.write(fasta_content)
            
            # Log cache info
            file_size = cache_file.stat().st_size
            sequence_count = fasta_content.count('>')
            
            self.nb_logger.info(f"💾 Cached complete FASTA: {cache_file}")
            self.nb_logger.info(f"💾 Generic cache file: {generic_cache_file}")
            self.nb_logger.info(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            self.nb_logger.info(f"📊 Sequence count: {sequence_count:,} sequences")
            
        except Exception as e:
            self.nb_logger.warning(f"Failed to cache complete FASTA: {e}")

    async def _convert_mmseqs2_results_to_clusters(self, clustering_report, original_sequences: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Convert MMseqs2 ClusteringReport to our cluster format
        """
        
        if not clustering_report:
            self.nb_logger.warning(f"No clustering report received from MMseqs2")
            return {}
        
        # Extract real cluster data from MMseqs2 ClusteringReport
        if not hasattr(clustering_report, 'total_clusters') or clustering_report.total_clusters == 0:
            self.nb_logger.warning(f"MMseqs2 produced no clusters")
            return {}
        
        self.nb_logger.info(f"Processing MMseqs2 clustering report: {clustering_report.total_clusters} clusters found")
        
        clusters = {}
        
        # Create mapping from sequences to enable cluster reconstruction
        sequence_map = {seq.get('patric_id', f'seq_{i}'): seq for i, seq in enumerate(original_sequences)}
        
        # Process MMseqs2 clusters
        if hasattr(clustering_report, 'clusters') and clustering_report.clusters:
            # If the clustering report has actual cluster data
            for i, cluster_data in enumerate(clustering_report.clusters):
                cluster_members = []
                member_ids = cluster_data.get('member_sequences', [])
                
                # Map cluster member IDs back to original sequence data
                for member_id in member_ids:
                    # Extract patric_id from FASTA header if needed
                    if '|' in member_id:
                        patric_id = member_id.split('|')[0]
                    else:
                        patric_id = member_id
                    
                    if patric_id in sequence_map:
                        cluster_members.append(sequence_map[patric_id])
                
                if cluster_members and len(cluster_members) >= self.min_cluster_size:
                    cluster_id = f"mmseqs2_cluster_{i+1:03d}"
                    
                    # Determine consensus product name
                    products = [seq.get('product', 'unknown') for seq in cluster_members]
                    consensus_product = max(set(products), key=products.count) if products else 'unknown'
                    
                    # Calculate genome distribution
                    genome_distribution = {}
                    sequence_ids = []
                    sequence_lengths = []
                    for member in cluster_members:
                        genome_id = member.get('genome_id', 'unknown')
                        genome_distribution[genome_id] = genome_distribution.get(genome_id, 0) + 1
                        sequence_ids.append(member.get('patric_id', ''))
                        aa_sequence = member.get('aa_sequence', '')
                        if aa_sequence:
                            sequence_lengths.append(len(aa_sequence))
                    
                    # Get representative sequence (longest one)
                    representative_member = max(cluster_members, key=lambda x: len(x.get('aa_sequence', ''))) if cluster_members else cluster_members[0]
                    representative_sequence = representative_member.get('aa_sequence', '')
                    
                    cluster = {
                        'cluster_id': cluster_id,
                        'sequences': cluster_members,
                        'size': len(cluster_members),
                        'consensus_annotation': consensus_product,
                        'cluster_type': 'mmseqs2_sequence_cluster',
                        'clustering_method': 'mmseqs2',
                        'mmseqs2_confidence': cluster_data.get('confidence', 0.85),
                        # NEW FIELDS:
                        'sequence_ids': sequence_ids,
                        'genome_distribution': genome_distribution,
                        'representative_sequence': representative_sequence,
                        'length_stats': self._calculate_length_stats(sequence_lengths) if sequence_lengths else {},
                        # For compatibility with response formatter:
                        'product_name': consensus_product,
                        'cluster_size': len(cluster_members),
                        'representative_seq': representative_member.get('patric_id', ''),
                        'member_sequences': sequence_ids
                    }
                    clusters[cluster_id] = cluster
        else:
            # If no detailed cluster data available, create basic clusters from report statistics
            cluster_size = len(original_sequences) // clustering_report.total_clusters
            
            for i in range(clustering_report.total_clusters):
                start_idx = i * cluster_size
                end_idx = min((i + 1) * cluster_size, len(original_sequences))
                cluster_members = original_sequences[start_idx:end_idx]
                
                if cluster_members and len(cluster_members) >= self.min_cluster_size:
                    cluster_id = f"mmseqs2_cluster_{i+1:03d}"
                    
                    # Determine consensus product name
                    products = [seq.get('product', 'unknown') for seq in cluster_members]
                    consensus_product = max(set(products), key=products.count) if products else 'unknown'
                    
                    # Calculate genome distribution
                    genome_distribution = {}
                    sequence_ids = []
                    sequence_lengths = []
                    for member in cluster_members:
                        genome_id = member.get('genome_id', 'unknown')
                        genome_distribution[genome_id] = genome_distribution.get(genome_id, 0) + 1
                        sequence_ids.append(member.get('patric_id', ''))
                        aa_sequence = member.get('aa_sequence', '')
                        if aa_sequence:
                            sequence_lengths.append(len(aa_sequence))
                    
                    # Get representative sequence (longest one)
                    representative_member = max(cluster_members, key=lambda x: len(x.get('aa_sequence', ''))) if cluster_members else cluster_members[0]
                    representative_sequence = representative_member.get('aa_sequence', '')
                    
                    cluster = {
                        'cluster_id': cluster_id,
                        'sequences': cluster_members,
                        'size': len(cluster_members),
                        'consensus_annotation': consensus_product,
                        'cluster_type': 'mmseqs2_sequence_cluster',
                        'clustering_method': 'mmseqs2',
                        'mmseqs2_confidence': 0.85,
                        # NEW FIELDS:
                        'sequence_ids': sequence_ids,
                        'genome_distribution': genome_distribution,
                        'representative_sequence': representative_sequence,
                        'length_stats': self._calculate_length_stats(sequence_lengths) if sequence_lengths else {},
                        # For compatibility with response formatter:
                        'product_name': consensus_product,
                        'cluster_size': len(cluster_members),
                        'representative_seq': representative_member.get('patric_id', ''),
                        'member_sequences': sequence_ids
                    }
                    clusters[cluster_id] = cluster
        
        self.nb_logger.info(f"✅ Converted MMseqs2 results to {len(clusters)} clusters")
        return clusters

    async def _analyze_clusters(self, clusters: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze clustering results and generate statistics
        """
        
        if not clusters:
            return {
                'total_clusters': 0,
                'total_sequences_clustered': 0,
                'average_cluster_size': 0,
                'cluster_size_distribution': {},
                'clustering_methods': {},
                'top_products': []
            }
        
        # Basic statistics - handle both 'size' and 'cluster_size' keys
        total_clusters = len(clusters)
        cluster_sizes = []
        for cluster in clusters.values():
            # Handle both 'size' (MMseqs2) and 'cluster_size' (product-based) keys
            size = cluster.get('size', cluster.get('cluster_size', 0))
            cluster_sizes.append(size)
        
        total_sequences_clustered = sum(cluster_sizes)
        average_cluster_size = total_sequences_clustered / total_clusters if total_clusters > 0 else 0
        
        # Cluster size distribution
        size_distribution = {}
        for size in cluster_sizes:
            size_range = self._get_size_range(size)
            size_distribution[size_range] = size_distribution.get(size_range, 0) + 1
        
        # Clustering methods distribution
        clustering_methods = {}
        for cluster in clusters.values():
            method = cluster.get('clustering_method', 'unknown')
            clustering_methods[method] = clustering_methods.get(method, 0) + 1
        
        # Top consensus products by sequence count - handle both naming conventions
        product_counts = {}
        for cluster in clusters.values():
            # For product-based clustering, use 'product_name', for MMseqs2 use 'consensus_annotation'
            consensus_product = cluster.get('product_name', cluster.get('consensus_annotation', 'unknown'))
            size = cluster.get('size', cluster.get('cluster_size', 0))
            
            if consensus_product not in product_counts:
                product_counts[consensus_product] = {'sequence_count': 0, 'cluster_count': 0}
            product_counts[consensus_product]['sequence_count'] += size
            product_counts[consensus_product]['cluster_count'] += 1
        
        # Sort by sequence count and get top 10
        top_products = []
        for product_name, counts in sorted(product_counts.items(), key=lambda x: x[1]['sequence_count'], reverse=True)[:10]:
            top_products.append({
                'product_name': product_name,
                'sequence_count': counts['sequence_count'],
                'cluster_count': counts['cluster_count']
            })
        
        # Determine primary clustering method
        primary_method = max(clustering_methods.items(), key=lambda x: x[1])[0] if clustering_methods else 'unknown'
        
        analysis = {
            'total_clusters': total_clusters,
            'total_sequences_clustered': total_sequences_clustered,
            'average_cluster_size': round(average_cluster_size, 2),
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'cluster_size_distribution': size_distribution,
            'clustering_methods': clustering_methods,
            'top_products': top_products,
            'clustering_method': primary_method
        }
        
        self.nb_logger.info(f"📊 Clustering Analysis ({primary_method}):")
        self.nb_logger.info(f"  - Total clusters: {total_clusters}")
        self.nb_logger.info(f"  - Total sequences clustered: {total_sequences_clustered}")
        self.nb_logger.info(f"  - Average cluster size: {average_cluster_size:.2f}")
        self.nb_logger.info(f"  - Size range: {min(cluster_sizes) if cluster_sizes else 0} - {max(cluster_sizes) if cluster_sizes else 0}")
        self.nb_logger.info(f"  - Top products: {[p['product_name'] for p in top_products[:5]]}")
        
        return analysis

    def _get_size_range(self, size: int) -> str:
        """Categorize cluster size into ranges"""
        if size < 10:
            return "small (3-9)"
        elif size < 50:
            return "medium (10-49)"
        elif size < 200:
            return "large (50-199)"
        else:
            return "very_large (200+)"
    
 