"""
Clustering Step (Steps 8-9)

Re-architected to inherit from NanoBrain Step base class.
Steps 8-9: Cluster proteins and prepare for alignment.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.logging_system import get_logger
import yaml
from pathlib import Path


class ClusteringStep(Step):
    """
    Steps 8-9: Cluster proteins and prepare for alignment
    
    Re-architected to inherit from NanoBrain Step base class.
    """
    
    def __init__(self, config: StepConfig, clustering_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config, **kwargs)
        self._initialize_tools(config, clustering_config)
        
    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any], 
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ClusteringStep with tool integration via from_config pattern"""
        super()._init_from_config(config, component_config, dependencies)
        self._initialize_tools(config, None)
        
    def _initialize_tools(self, config: StepConfig, clustering_config: Optional[Dict[str, Any]] = None):
        """Initialize MMseqs2 tool with workflow-local configuration"""
        
        # Get workflow directory path
        workflow_dir = Path(__file__).parent.parent
        tool_config_path = workflow_dir / "config" / "tools" / "mmseqs2_tool.yml"
        
        # Load MMseqs2 tool configuration from workflow-local YAML
        if tool_config_path.exists():
            with open(tool_config_path, 'r') as f:
                tool_config_dict = yaml.safe_load(f)
            
            # Import MMseqs2 tool and config
            try:
                from nanobrain.library.tools.bioinformatics.mmseqs_tool import MMseqs2Tool, MMseqs2Config
                
                # Create MMseqs2 tool configuration
                tool_config = MMseqs2Config(**{
                    k: v for k, v in tool_config_dict.items() 
                    if k in ['tool_name', 'conda_package', 'conda_channel', 'git_repository',
                            'environment_name', 'min_seq_id', 'coverage', 'cluster_mode', 
                            'sensitivity', 'progressive_scaling', 'threads', 'memory_limit', 
                            'tmp_dir']
                })
                
                # Create MMseqs2 tool using from_config pattern
                self.mmseqs2_tool = MMseqs2Tool.from_config(tool_config)
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.info(f"✅ MMseqs2 tool loaded from workflow-local config: {tool_config_path}")
                
                # Extract clustering parameters from tool config
                self.similarity_threshold = tool_config_dict.get('similarity_threshold', 0.7)
                self.min_cluster_size = tool_config_dict.get('min_cluster_size', 3)
                
            except ImportError:
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.warning("⚠️ MMseqs2Tool not available, using placeholder implementation")
                self.mmseqs2_tool = None
                self.similarity_threshold = 0.7
                self.min_cluster_size = 3
                
        else:
            # Fallback to legacy configuration
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.warning(f"⚠️ Workflow-local tool config not found: {tool_config_path}")
                self.nb_logger.warning("⚠️ Using legacy configuration approach")
            
            self.mmseqs2_tool = None
            step_config_dict = config.config if hasattr(config, 'config') else {}
            if clustering_config:
                step_config_dict.update(clustering_config)
            
            self.clustering_config = step_config_dict.get('clustering_config', {})
            self.similarity_threshold = step_config_dict.get('similarity_threshold', 0.5)
            self.min_cluster_size = step_config_dict.get('min_cluster_size', 3)
        
        # Store step configuration
        self.step_config = config.model_dump()
        
        if hasattr(self, 'nb_logger') and self.nb_logger:
            self.nb_logger.info(f"🧬 ClusteringStep initialized with threshold: {self.similarity_threshold}")
        
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process method required by Step base class.
        
        This implements the NanoBrain framework interface while calling the
        original execute method that contains the clustering logic.
        """
        self.nb_logger.info("🔄 Processing clustering step")
        
        # Call the original execute method
        result = await self.execute(input_data)
        
        self.nb_logger.info(f"✅ Clustering step completed successfully")
        return result

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute clustering step
        
        Args:
            input_data: Contains curated sequences and curation report
            
        Returns:
            Dict with clusters and clustering analysis
        """
        
        step_start_time = time.time()
        
        try:
            self.nb_logger.info("🗂️ Starting MMseqs2 clustering analysis")
            
            curated_sequences = input_data.get('curated_sequences', [])
            curation_report = input_data.get('curation_report', {})
            
            # Placeholder clustering implementation
            # In a full implementation, this would:
            # 1. Run MMseqs2 with specified parameters
            # 2. Analyze cluster quality focusing on short well-conserved regions
            # 3. Generate cluster statistics and quality metrics
            
            clusters = await self._perform_clustering(curated_sequences)
            clustering_analysis = await self._analyze_clusters(clusters)
            
            execution_time = time.time() - step_start_time
            self.nb_logger.info(f"✅ Clustering completed in {execution_time:.2f} seconds")
            self.nb_logger.info(f"Generated {len(clusters)} protein clusters")
            
            return {
                'clusters': clusters,
                'clustering_analysis': clustering_analysis,
                'execution_time': execution_time,
                'clustering_parameters': {
                    'min_seq_id': self.step_config.get('min_seq_id', 0.7),
                    'coverage': self.step_config.get('coverage', 0.8),
                    'sensitivity': self.step_config.get('sensitivity', 7.5)
                },
                'cluster_statistics': {
                    'total_clusters': len(clusters),
                    'total_sequences_clustered': sum(len(cluster.get('members', [])) for cluster in clusters),
                    'average_cluster_size': sum(len(cluster.get('members', [])) for cluster in clusters) / len(clusters) if clusters else 0
                }
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Clustering failed: {e}")
            raise
            
    async def _perform_clustering(self, sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform MMseqs2 clustering with real tool or placeholder implementation
        """
        
        if self.mmseqs2_tool and hasattr(self.mmseqs2_tool, 'cluster_sequences'):
            self.nb_logger.info("🔧 Using real MMseqs2Tool for clustering")
            try:
                # Use real MMseqs2 tool via from_config
                return await self.mmseqs2_tool.cluster_sequences(sequences)
            except Exception as e:
                self.nb_logger.warning(f"⚠️ MMseqs2Tool failed, falling back to placeholder: {e}")
                return await self._placeholder_clustering(sequences)
        else:
            # Fallback to placeholder implementation
            self.nb_logger.info("🔧 Using placeholder clustering implementation")
            return await self._placeholder_clustering(sequences)
    
    async def _placeholder_clustering(self, sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Placeholder clustering implementation for when MMseqs2 is not available
        """
        
        self.nb_logger.info("Performing sequence clustering (placeholder)")
        
        # Placeholder: Create simple clusters based on sequence similarity
        clusters = []
        
        # Group sequences by rough similarity (placeholder logic)
        processed_sequences = set()
        cluster_id = 1
        
        for i, seq in enumerate(sequences):
            if i in processed_sequences:
                continue
                
            if not isinstance(seq, dict) or 'aa_sequence' not in seq:
                continue
                
            # Create cluster with this sequence as representative
            cluster_members = [seq]
            processed_sequences.add(i)
            
            # Find similar sequences (placeholder: by length similarity)
            seq_length = len(seq['aa_sequence'])
            
            for j, other_seq in enumerate(sequences[i+1:], i+1):
                if j in processed_sequences:
                    continue
                    
                if not isinstance(other_seq, dict) or 'aa_sequence' not in other_seq:
                    continue
                    
                other_length = len(other_seq['aa_sequence'])
                
                # Placeholder similarity: sequences within 20% length difference
                length_ratio = min(seq_length, other_length) / max(seq_length, other_length)
                if length_ratio > 0.8:  # 80% length similarity
                    cluster_members.append(other_seq)
                    processed_sequences.add(j)
                    
            # Only create cluster if it has multiple members or meets size criteria
            if len(cluster_members) >= self.min_cluster_size or len(cluster_members) == 1:
                cluster = {
                    'id': f'cluster_{cluster_id}',
                    'representative': cluster_members[0],
                    'members': cluster_members,
                    'member_count': len(cluster_members),
                    'consensus_annotation': self._determine_consensus_annotation(cluster_members),
                    'protein_class': self._classify_cluster(cluster_members),
                    'consensus_score': 0.8,  # Placeholder
                    'overall_confidence': 0.75  # Placeholder
                }
                clusters.append(cluster)
                cluster_id += 1
                
        self.nb_logger.info(f"Created {len(clusters)} clusters from {len(sequences)} sequences")
        return clusters
        
    def _determine_consensus_annotation(self, cluster_members: List[Dict[str, Any]]) -> str:
        """Determine consensus annotation for cluster"""
        
        # Count product annotations
        product_counts = {}
        for member in cluster_members:
            product = member.get('product', 'hypothetical protein').lower()
            product_counts[product] = product_counts.get(product, 0) + 1
            
        if product_counts:
            # Return most common annotation
            consensus_product = max(product_counts.items(), key=lambda x: x[1])[0]
            return consensus_product
        else:
            return 'hypothetical protein'
            
    def _classify_cluster(self, cluster_members: List[Dict[str, Any]]) -> str:
        """Classify cluster based on member annotations"""
        
        consensus_annotation = self._determine_consensus_annotation(cluster_members)
        
        # Simple classification based on annotation
        if any(keyword in consensus_annotation for keyword in ['nsp', 'nonstructural', 'replicase', 'protease', 'polymerase']):
            return 'non_structural'
        elif any(keyword in consensus_annotation for keyword in ['capsid', 'envelope', 'structural', 'glycoprotein']):
            return 'structural'
        else:
            return 'unknown'
            
    async def _analyze_clusters(self, clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze cluster quality focusing on short well-conserved regions
        """
        
        analysis = {
            'total_clusters': len(clusters),
            'cluster_size_distribution': {},
            'protein_class_distribution': {},
            'quality_metrics': {},
            'conservation_analysis': {}
        }
        
        # Analyze cluster size distribution
        size_counts = {}
        for cluster in clusters:
            size = cluster.get('member_count', 0)
            size_range = self._get_size_range(size)
            size_counts[size_range] = size_counts.get(size_range, 0) + 1
            
        analysis['cluster_size_distribution'] = size_counts
        
        # Analyze protein class distribution
        class_counts = {}
        for cluster in clusters:
            protein_class = cluster.get('protein_class', 'unknown')
            class_counts[protein_class] = class_counts.get(protein_class, 0) + 1
            
        analysis['protein_class_distribution'] = class_counts
        
        # Quality metrics
        if clusters:
            average_confidence = sum(cluster.get('overall_confidence', 0) for cluster in clusters) / len(clusters)
            high_confidence_clusters = sum(1 for cluster in clusters if cluster.get('overall_confidence', 0) > 0.8)
            
            analysis['quality_metrics'] = {
                'average_confidence': average_confidence,
                'high_confidence_clusters': high_confidence_clusters,
                'high_confidence_ratio': high_confidence_clusters / len(clusters)
            }
        else:
            analysis['quality_metrics'] = {
                'average_confidence': 0.0,
                'high_confidence_clusters': 0,
                'high_confidence_ratio': 0.0
            }
            
        return analysis
        
    def _get_size_range(self, size: int) -> str:
        """Get size range category for cluster"""
        if size == 1:
            return 'singleton'
        elif size <= 5:
            return 'small (2-5)'
        elif size <= 20:
            return 'medium (6-20)'
        else:
            return 'large (>20)' 