"""
Alignment Step (Steps 10-13)

Re-architected to inherit from NanoBrain Step base class.
Steps 10-13: Align protein clusters and prepare for PSSM analysis.
"""

import asyncio
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.logging_system import get_logger


class AlignmentStep(Step):
    """
    Steps 10-13: Align protein clusters and prepare for PSSM analysis
    
    Re-architected to inherit from NanoBrain Step base class.
    """
    
    def __init__(self, config: StepConfig, alignment_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config, **kwargs)
        
        # Extract configuration from step config or provided alignment_config
        step_config_dict = config.config if hasattr(config, 'config') else {}
        if alignment_config:
            step_config_dict.update(alignment_config)
        
        self.alignment_config = step_config_dict.get('alignment_config', {})
        self.step_config = step_config_dict
        
        # Configuration parameters
        self.alignment_tool = self.step_config.get('alignment_tool', 'muscle')
        self.min_cluster_size = self.step_config.get('min_cluster_size', 3)
        self.max_cluster_size = self.step_config.get('max_cluster_size', 1000)
        
        self.nb_logger.info(f"🧬 AlignmentStep initialized with tool: {self.alignment_tool}")
        
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process method required by Step base class.
        
        This implements the NanoBrain framework interface while calling the
        original execute method that contains the alignment logic.
        """
        self.nb_logger.info("🔄 Processing alignment step")
        
        # Call the original execute method
        result = await self.execute(input_data)
        
        self.nb_logger.info(f"✅ Alignment step completed successfully")
        return result

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute alignment step
        
        Args:
            input_data: Contains clusters from clustering step
            
        Returns:
            Dict with aligned clusters and alignment quality stats
        """
        
        step_start_time = time.time()
        
        try:
            self.nb_logger.info("📏 Starting multiple sequence alignment")
            
            clusters = input_data.get('clusters', [])
            
            # Placeholder alignment implementation
            # In a full implementation, this would:
            # 1. Run MUSCLE alignment for each cluster
            # 2. Assess alignment quality
            # 3. Prepare sequences for PSSM generation
            
            aligned_clusters = await self._align_protein_clusters(clusters)
            alignment_quality_stats = await self._calculate_alignment_statistics(aligned_clusters)
            
            execution_time = time.time() - step_start_time
            self.nb_logger.info(f"✅ Alignment completed in {execution_time:.2f} seconds")
            self.nb_logger.info(f"Aligned {len(aligned_clusters)} protein clusters")
            
            return {
                'aligned_clusters': aligned_clusters,
                'alignment_quality_stats': alignment_quality_stats,
                'execution_time': execution_time,
                'alignment_parameters': {
                    'max_iterations': self.step_config.get('max_iterations', 16),
                    'gap_open_penalty': self.step_config.get('gap_open_penalty', -12),
                    'gap_extend_penalty': self.step_config.get('gap_extend_penalty', -1)
                },
                'alignment_statistics': {
                    'total_clusters_aligned': len(aligned_clusters),
                    'successful_alignments': sum(1 for cluster in aligned_clusters if cluster.get('alignment_successful', False)),
                    'average_alignment_length': alignment_quality_stats.get('average_alignment_length', 0)
                }
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Alignment failed: {e}")
            raise
            
    async def _align_protein_clusters(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform multiple sequence alignment for each cluster using MUSCLE
        """
        
        aligned_clusters = []
        
        for cluster in clusters:
            cluster_members = cluster.get('members', [])
            
            if len(cluster_members) >= 3:  # Minimum sequences for meaningful alignment
                self.nb_logger.debug(f"Aligning cluster {cluster.get('id')} with {len(cluster_members)} sequences")
                
                aligned_cluster = await self._align_cluster_sequences(cluster)
                aligned_clusters.append(aligned_cluster)
                
            else:
                self.nb_logger.warning(f"Cluster {cluster.get('id')} has too few sequences ({len(cluster_members)}) for alignment")
                # Still include the cluster but mark as not aligned
                aligned_cluster = cluster.copy()
                aligned_cluster['alignment_successful'] = False
                aligned_cluster['alignment_quality'] = {
                    'mean_conservation': 0.0,
                    'alignment_length': 0,
                    'sequence_count': len(cluster_members)
                }
                aligned_clusters.append(aligned_cluster)
                
        return aligned_clusters
        
    async def _align_cluster_sequences(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        """
        Align sequences within a cluster using MUSCLE (placeholder)
        """
        
        cluster_members = cluster.get('members', [])
        
        # Placeholder alignment implementation
        # In a real implementation, this would:
        # 1. Create temporary FASTA file for cluster sequences
        # 2. Run MUSCLE alignment
        # 3. Parse alignment results
        # 4. Calculate alignment quality metrics
        
        # For now, simulate alignment by adding alignment metadata
        aligned_cluster = cluster.copy()
        
        # Simulate alignment quality assessment
        sequence_count = len(cluster_members)
        
        # Placeholder: calculate mock alignment statistics
        if cluster_members and 'aa_sequence' in cluster_members[0]:
            # Use average sequence length as mock alignment length
            avg_length = sum(len(member.get('aa_sequence', '')) for member in cluster_members) / len(cluster_members)
            alignment_length = int(avg_length * 1.1)  # Assume some gaps added
            
            # Mock conservation score based on cluster consensus score
            mean_conservation = cluster.get('consensus_score', 0.8)
            
            alignment_quality = {
                'mean_conservation': mean_conservation,
                'highly_conserved_positions': int(alignment_length * mean_conservation * 0.7),
                'alignment_length': alignment_length,
                'sequence_count': sequence_count,
                'gap_percentage': 15.0,  # Mock gap percentage
                'alignment_score': mean_conservation * 100
            }
        else:
            alignment_quality = {
                'mean_conservation': 0.0,
                'highly_conserved_positions': 0,
                'alignment_length': 0,
                'sequence_count': sequence_count,
                'gap_percentage': 0.0,
                'alignment_score': 0.0
            }
            
        aligned_cluster['alignment_quality'] = alignment_quality
        aligned_cluster['alignment_successful'] = alignment_quality['mean_conservation'] > 0.3
        aligned_cluster['alignment_method'] = 'MUSCLE (placeholder)'
        
        return aligned_cluster
        
    async def _calculate_alignment_statistics(self, aligned_clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate overall alignment quality statistics
        """
        
        if not aligned_clusters:
            return {
                'total_alignments': 0,
                'successful_alignments': 0,
                'average_conservation': 0.0,
                'average_alignment_length': 0,
                'quality_distribution': {'high': 0, 'medium': 0, 'low': 0}
            }
            
        successful_alignments = [cluster for cluster in aligned_clusters 
                               if cluster.get('alignment_successful', False)]
        
        total_alignments = len(aligned_clusters)
        successful_count = len(successful_alignments)
        
        if successful_alignments:
            # Calculate average metrics for successful alignments
            avg_conservation = sum(
                cluster['alignment_quality']['mean_conservation'] 
                for cluster in successful_alignments
            ) / len(successful_alignments)
            
            avg_alignment_length = sum(
                cluster['alignment_quality']['alignment_length']
                for cluster in successful_alignments  
            ) / len(successful_alignments)
            
            # Quality distribution
            quality_distribution = {'high': 0, 'medium': 0, 'low': 0}
            for cluster in successful_alignments:
                conservation = cluster['alignment_quality']['mean_conservation']
                if conservation >= 0.8:
                    quality_distribution['high'] += 1
                elif conservation >= 0.5:
                    quality_distribution['medium'] += 1
                else:
                    quality_distribution['low'] += 1
                    
        else:
            avg_conservation = 0.0
            avg_alignment_length = 0
            quality_distribution = {'high': 0, 'medium': 0, 'low': 0}
            
        return {
            'total_alignments': total_alignments,
            'successful_alignments': successful_count,
            'success_rate': successful_count / total_alignments if total_alignments > 0 else 0,
            'average_conservation': avg_conservation,
            'average_alignment_length': avg_alignment_length,
            'quality_distribution': quality_distribution,
            'alignment_method': 'MUSCLE',
            'min_sequences_required': 3
        } 