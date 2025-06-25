"""
MUSCLE Multiple Sequence Alignment Tool

This module provides a wrapper for MUSCLE multiple sequence alignment tool,
used to prepare clustered sequences for PSSM generation in the Alphavirus workflow.
"""

import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import Field

from nanobrain.core.external_tool import (
    ExternalTool,
    ToolResult,
    ToolExecutionError,
    InstallationStatus,
    DiagnosticReport,
    ToolInstallationError,
    ExternalToolConfig
)
from nanobrain.core.progressive_scaling import ProgressiveScalingMixin
from nanobrain.core.tool import ToolConfig
from nanobrain.core.logging_system import get_logger


@dataclass
class MUSCLEConfig(ExternalToolConfig):
    """Configuration for MUSCLE multiple sequence alignment tool"""
    # Tool identification
    tool_name: str = "muscle"
    
    # Default tool card
    tool_card: Dict[str, Any] = field(default_factory=lambda: {
        "name": "muscle",
        "description": "MUSCLE tool for multiple sequence alignment",
        "version": "1.0.0",
        "category": "bioinformatics",
        "capabilities": ["sequence_alignment", "msa_generation", "conservation_analysis"]
    })
    
    # Installation configuration
    conda_package: str = "muscle"
    conda_channel: str = "bioconda"
    environment_name: str = "nanobrain-viral_protein-muscle"
    create_isolated_environment: bool = True
    
    # Alignment parameters
    max_iterations: int = 16
    diagonal_optimization: bool = True
    gap_open_penalty: float = -12.0
    gap_extend_penalty: float = -1.0
    
    # Quality parameters
    min_sequences: int = 3
    max_sequences: int = 1000
    output_format: str = "fasta"  # fasta, clustal, msf, phylip
    
    # Conservation analysis
    calculate_profile: bool = True
    highly_conserved_threshold: float = 0.8
    position_scoring: bool = True
    
    # Quality scoring
    quality_scoring: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "weights": {
            "conservation_score": 0.5,
            "gap_penalty": 0.3,
            "size_bonus": 0.2
        },
        "thresholds": {
            "high_quality_threshold": 0.7,
            "acceptable_quality_threshold": 0.5
        }
    })
    
    # Performance settings
    parallel_processing: bool = False
    memory_optimization: bool = True
    
    # Output settings
    export_conservation_profile: bool = True
    export_quality_metrics: bool = True
    keep_intermediate_files: bool = False
    
    # Error handling
    retry_failed_alignments: bool = True
    max_retry_attempts: int = 2
    skip_problematic_clusters: bool = True
    
    # Installation paths
    local_installation_paths: List[str] = field(default_factory=lambda: [
        "/usr/local/bin",
        "/opt/homebrew/bin",
        "~/bin"
    ])


@dataclass
class AlignedSequence:
    """Single sequence in an alignment"""
    sequence_id: str
    aligned_sequence: str
    original_sequence: str
    gaps_count: int
    
    def __post_init__(self):
        self.gaps_count = self.aligned_sequence.count('-')


@dataclass
class MultipleSeqAlignment:
    """Multiple sequence alignment result"""
    sequences: List[AlignedSequence]
    alignment_length: int
    sequence_count: int
    gap_percentage: float
    conservation_profile: List[float]
    
    def __post_init__(self):
        self.sequence_count = len(self.sequences)
        if self.sequences:
            self.alignment_length = len(self.sequences[0].aligned_sequence)
            total_positions = self.alignment_length * self.sequence_count
            total_gaps = sum(seq.gaps_count for seq in self.sequences)
            self.gap_percentage = (total_gaps / total_positions) * 100 if total_positions > 0 else 0
        else:
            self.alignment_length = 0
            self.gap_percentage = 0


@dataclass
class AlignmentQuality:
    """Quality metrics for alignment"""
    mean_conservation: float
    highly_conserved_positions: int
    alignment_length: int
    sequence_count: int
    gap_percentage: float
    quality_score: float
    
    def __post_init__(self):
        # Calculate overall quality score
        conservation_score = self.mean_conservation * 0.5
        gap_penalty = (self.gap_percentage / 100) * 0.3
        size_bonus = min(self.sequence_count / 10, 1.0) * 0.2
        
        self.quality_score = max(0, conservation_score - gap_penalty + size_bonus)


@dataclass
class AlignmentResult:
    """Result of MUSCLE alignment"""
    alignment: MultipleSeqAlignment
    quality: AlignmentQuality
    execution_time: float
    parameters: MUSCLEConfig


class MUSCLETool(ProgressiveScalingMixin, ExternalTool):
    """
    MUSCLE multiple sequence alignment tool wrapper.
    Enhanced with mandatory from_config pattern implementation.
    
    Provides high-quality multiple sequence alignments for PSSM generation
    in the Alphavirus protein analysis workflow.
    """
    
    @classmethod
    def from_config(cls, config: Union[ToolConfig, MUSCLEConfig, Dict], **kwargs) -> 'MUSCLETool':
        """Mandatory from_config implementation for MUSCLETool"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Convert any input to MUSCLEConfig
        if isinstance(config, MUSCLEConfig):
            # Already specific config, use as-is
            pass
        else:
            # Convert ToolConfig, dict, or any other input to MUSCLEConfig
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif isinstance(config, dict):
                config_dict = config
            else:
                # Handle object with attributes
                config_dict = {}
                # Extract fields that are common to both ToolConfig and MUSCLEConfig
                for attr in ['name', 'description', 'tool_card']:
                    if hasattr(config, attr):
                        config_dict[attr] = getattr(config, attr)
            
            # Filter config_dict to only include fields that MUSCLEConfig accepts
            # Remove ToolConfig-specific fields that MUSCLEConfig doesn't inherit
            muscle_compatible_fields = {
                # Core fields from MUSCLEConfig
                'tool_name', 'tool_card', 'conda_package', 'conda_channel', 
                'environment_name', 'create_isolated_environment', 'max_iterations',
                'diagonal_optimization', 'gap_open_penalty', 'gap_extend_penalty',
                'min_sequences', 'max_sequences', 'output_format', 'calculate_profile',
                'highly_conserved_threshold', 'position_scoring', 'quality_scoring',
                'parallel_processing', 'memory_optimization', 'export_conservation_profile',
                'export_quality_metrics', 'keep_intermediate_files', 'retry_failed_alignments',
                'max_retry_attempts', 'skip_problematic_clusters', 'local_installation_paths',
                # Core fields from ExternalToolConfig
                'installation_path', 'executable_path', 'environment', 'timeout_seconds',
                'retry_attempts', 'verify_on_init', 'pip_package', 'git_repository',
                'initial_scale_level', 'detailed_diagnostics', 'suggest_fixes'
            }
            
            filtered_config_dict = {k: v for k, v in config_dict.items() 
                                   if k in muscle_compatible_fields}
            
            logger.debug(f"Filtered config keys: {list(filtered_config_dict.keys())}")
            logger.debug(f"Removed incompatible keys: {set(config_dict.keys()) - set(filtered_config_dict.keys())}")
            
            # Create MUSCLEConfig from the filtered data
            config = MUSCLEConfig(**filtered_config_dict)
        
        # Mandatory tool_card validation and extraction
        if hasattr(config, 'tool_card') and config.tool_card:
            tool_card_data = config.tool_card.model_dump() if hasattr(config.tool_card, 'model_dump') else config.tool_card
            logger.info(f"Tool {config.tool_name} loaded with tool card metadata")
        elif isinstance(config, dict) and 'tool_card' in config:
            tool_card_data = config['tool_card']
            logger.info(f"Tool {config.tool_name} loaded with tool card metadata")
        else:
            raise ValueError(
                f"Missing mandatory 'tool_card' section in configuration for {cls.__name__}. "
                f"All tools must include tool card metadata for proper discovery and usage."
            )
        
        # Create instance
        instance = cls(config, **kwargs)
        instance._tool_card_data = tool_card_data
        
        logger.info(f"Successfully created {cls.__name__} with tool card compliance")
        return instance
    
    def __init__(self, config: MUSCLEConfig, **kwargs):
        """Initialize MUSCLETool with configuration"""
        if config is None:
            config = MUSCLEConfig(
                tool_name="muscle",
                max_iterations=16,
                diagonal_optimization=True,
                gap_open_penalty=-12.0,
                gap_extend_penalty=-1.0
            )
        
        # Ensure name is set consistently
        if not hasattr(config, 'tool_name') or not config.tool_name:
            config.tool_name = "muscle"
        
        # Initialize parent classes
        super().__init__(config, **kwargs)
        
        # MUSCLE specific initialization
        self.muscle_config = config
        self.name = config.tool_name
        self.logger = get_logger(f"bio_tool_{self.name}")
        
        # MUSCLE specific attributes
        self.max_iterations = getattr(config, 'max_iterations', 16)
        self.diagonal_optimization = getattr(config, 'diagonal_optimization', True)
        self.gap_open_penalty = getattr(config, 'gap_open_penalty', -12.0)
        self.gap_extend_penalty = getattr(config, 'gap_extend_penalty', -1.0)
        self.min_sequences = getattr(config, 'min_sequences', 3)
        self.max_sequences = getattr(config, 'max_sequences', 1000)
        self.output_format = getattr(config, 'output_format', "fasta")
        
    async def verify_installation(self) -> bool:
        """Verify MUSCLE installation"""
        try:
            result = await self.execute_command(["muscle", "-version"])
            if result.success:
                version = result.stdout_text.strip()
                self.logger.info(f"MUSCLE version detected: {version}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"MUSCLE verification failed: {e}")
            return False
    
    async def execute_command(self, command: List[str], **kwargs) -> ToolResult:
        """
        Execute MUSCLE command with retry logic.
        
        BREAKING CHANGE: Now enforces mandatory initialization before execution.
        """
        # MANDATORY: Ensure tool is initialized before any execution
        await self.ensure_initialized()
        
        return await self._execute_with_retry(command, **kwargs)
    
    async def parse_output(self, raw_output: str) -> Dict[str, Any]:
        """Parse MUSCLE alignment output"""
        return await self._parse_fasta_output(raw_output)
    
    async def _parse_fasta_output(self, fasta_content: str) -> Dict[str, Any]:
        """Parse FASTA format output from MUSCLE"""
        sequences = {}
        current_header = None
        current_sequence = []
        
        for line in fasta_content.strip().split('\n'):
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_header and current_sequence:
                    sequences[current_header] = ''.join(current_sequence)
                
                # Start new sequence
                current_header = line[1:]  # Remove '>' prefix
                current_sequence = []
            elif line and current_header:
                current_sequence.append(line)
        
        # Save last sequence
        if current_header and current_sequence:
            sequences[current_header] = ''.join(current_sequence)
        
        return {"sequences": sequences, "total_sequences": len(sequences)}
    
    async def align_sequences(self, sequences: Dict[str, str], 
                            custom_params: Optional[Dict[str, Any]] = None) -> AlignmentResult:
        """
        Align sequences using MUSCLE
        
        Args:
            sequences: Dictionary of {seq_id: sequence}
            custom_params: Optional custom parameters
            
        Returns:
            AlignmentResult: Alignment with quality metrics
        """
        import time
        start_time = time.time()
        
        # Validate input
        if len(sequences) < self.muscle_config.min_sequences:
            raise ValueError(f"Need at least {self.muscle_config.min_sequences} sequences for alignment")
        
        if len(sequences) > self.muscle_config.max_sequences:
            self.logger.warning(f"Too many sequences ({len(sequences)}), truncating to {self.muscle_config.max_sequences}")
            sequences = dict(list(sequences.items())[:self.muscle_config.max_sequences])
        
        # Apply custom parameters if provided
        config = self._merge_params(custom_params) if custom_params else self.muscle_config
        
        # Create temporary files
        input_fasta = await self._create_input_fasta(sequences)
        output_fasta = await self._get_output_file()
        
        try:
            # Run MUSCLE alignment
            result = await self._run_muscle_alignment(input_fasta, output_fasta, config)
            
            if not result.success:
                raise ToolExecutionError(f"MUSCLE alignment failed: {result.stderr_text}")
            
            # Parse alignment results
            alignment = await self._parse_alignment_output(output_fasta, sequences)
            
            # Calculate quality metrics
            quality = await self._assess_alignment_quality(alignment)
            
            execution_time = time.time() - start_time
            
            return AlignmentResult(
                alignment=alignment,
                quality=quality,
                execution_time=execution_time,
                parameters=config
            )
            
        finally:
            # Cleanup temporary files
            await self._cleanup_temp_files([input_fasta, output_fasta])
    
    async def _run_muscle_alignment(self, input_file: str, output_file: str, 
                                   config: MUSCLEConfig) -> ToolResult:
        """Run MUSCLE alignment command"""
        command = [
            "muscle",
            "-in", input_file,
            "-out", output_file,
            "-maxiters", str(config.max_iterations)
        ]
        
        # Add diagonal optimization if enabled
        if config.diagonal_optimization:
            command.append("-diags")
        
        # Add gap penalties
        command.extend([
            "-gapopen", str(config.gap_open_penalty),
            "-gapextend", str(config.gap_extend_penalty)
        ])
        
        # Set output format
        if config.output_format.lower() == "clustal":
            command.extend(["-clw"])
        elif config.output_format.lower() == "msf":
            command.extend(["-msf"])
        elif config.output_format.lower() == "phylip":
            command.extend(["-phyi"])
        # Default is FASTA, no additional flag needed
        
        self.logger.info(f"Running MUSCLE alignment: {' '.join(command)}")
        return await self.execute_command(command)
    
    async def _create_input_fasta(self, sequences: Dict[str, str]) -> str:
        """Create input FASTA file from sequences"""
        fasta_content = []
        for seq_id, sequence in sequences.items():
            fasta_content.append(f">{seq_id}")
            fasta_content.append(sequence)
        
        temp_file = await self.create_temp_file("\n".join(fasta_content), suffix=".fasta")
        return temp_file
    
    async def _get_output_file(self) -> str:
        """Get temporary output file path"""
        fd, temp_file = tempfile.mkstemp(suffix=".aln", prefix="muscle_output_")
        os.close(fd)  # Close the file descriptor, keep the path
        return temp_file
    
    async def _parse_alignment_output(self, output_file: str, 
                                    original_sequences: Dict[str, str]) -> MultipleSeqAlignment:
        """Parse MUSCLE alignment output"""
        aligned_sequences = []
        
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            
            # Parse FASTA format alignment
            sequences_data = await self._parse_fasta_output(content)
            
            for seq_id, aligned_seq in sequences_data.get("sequences", {}).items():
                original_seq = original_sequences.get(seq_id, "")
                
                aligned_sequence = AlignedSequence(
                    sequence_id=seq_id,
                    aligned_sequence=aligned_seq,
                    original_sequence=original_seq,
                    gaps_count=aligned_seq.count('-')
                )
                aligned_sequences.append(aligned_sequence)
            
        except FileNotFoundError:
            self.logger.error(f"Alignment output file not found: {output_file}")
            return MultipleSeqAlignment([], 0, 0, 0, [])
        
        # Calculate conservation profile
        conservation_profile = await self._calculate_conservation_profile(aligned_sequences)
        
        alignment = MultipleSeqAlignment(
            sequences=aligned_sequences,
            alignment_length=len(aligned_sequences[0].aligned_sequence) if aligned_sequences else 0,
            sequence_count=len(aligned_sequences),
            gap_percentage=0,  # Will be calculated in __post_init__
            conservation_profile=conservation_profile
        )
        
        return alignment
    
    async def _calculate_conservation_profile(self, aligned_sequences: List[AlignedSequence]) -> List[float]:
        """Calculate conservation score for each position in the alignment"""
        if not aligned_sequences:
            return []
        
        alignment_length = len(aligned_sequences[0].aligned_sequence)
        conservation_scores = []
        
        for pos in range(alignment_length):
            # Get residues at this position (excluding gaps)
            residues = []
            for seq in aligned_sequences:
                if pos < len(seq.aligned_sequence):
                    residue = seq.aligned_sequence[pos]
                    if residue != '-':
                        residues.append(residue.upper())
            
            if not residues:
                conservation_scores.append(0.0)
                continue
            
            # Calculate conservation as frequency of most common residue
            from collections import Counter
            residue_counts = Counter(residues)
            most_common_count = residue_counts.most_common(1)[0][1]
            conservation = most_common_count / len(residues)
            
            conservation_scores.append(conservation)
        
        return conservation_scores
    
    async def _assess_alignment_quality(self, alignment: MultipleSeqAlignment) -> AlignmentQuality:
        """Assess the quality of the multiple sequence alignment"""
        if not alignment.sequences:
            return AlignmentQuality(0, 0, 0, 0, 0, 0)
        
        # Calculate mean conservation
        mean_conservation = sum(alignment.conservation_profile) / len(alignment.conservation_profile) if alignment.conservation_profile else 0
        
        # Count highly conserved positions (>80% conservation)
        highly_conserved = sum(1 for score in alignment.conservation_profile if score > 0.8)
        
        return AlignmentQuality(
            mean_conservation=mean_conservation,
            highly_conserved_positions=highly_conserved,
            alignment_length=alignment.alignment_length,
            sequence_count=alignment.sequence_count,
            gap_percentage=alignment.gap_percentage,
            quality_score=0  # Will be calculated in __post_init__
        )
    
    def _merge_params(self, custom_params: Dict[str, Any]) -> MUSCLEConfig:
        """Merge custom parameters with default config"""
        config_dict = self.muscle_config.__dict__.copy()
        config_dict.update(custom_params)
        return MUSCLEConfig(**config_dict)
    
    async def _cleanup_temp_files(self, file_paths: List[str]) -> None:
        """Clean up temporary files"""
        for path in file_paths:
            try:
                if os.path.isfile(path):
                    os.unlink(path)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup {path}: {e}")
    
    async def align_cluster_sequences(self, cluster_sequences: List[Dict[str, str]]) -> List[AlignmentResult]:
        """
        Align multiple clusters of sequences
        
        Args:
            cluster_sequences: List of sequence dictionaries for each cluster
            
        Returns:
            List[AlignmentResult]: Alignment results for each cluster
        """
        alignment_results = []
        
        for i, sequences in enumerate(cluster_sequences):
            if len(sequences) >= self.muscle_config.min_sequences:
                try:
                    self.logger.info(f"Aligning cluster {i+1}/{len(cluster_sequences)} with {len(sequences)} sequences")
                    result = await self.align_sequences(sequences)
                    alignment_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to align cluster {i+1}: {e}")
                    continue
            else:
                self.logger.warning(f"Cluster {i+1} has too few sequences ({len(sequences)}) for alignment")
        
        return alignment_results
    
    async def get_alignment_statistics(self, results: List[AlignmentResult]) -> Dict[str, Any]:
        """Get detailed statistics about alignment results"""
        if not results:
            return {"error": "No alignment results"}
        
        # Collect quality metrics
        quality_scores = [r.quality.quality_score for r in results]
        conservation_scores = [r.quality.mean_conservation for r in results]
        gap_percentages = [r.quality.gap_percentage for r in results]
        sequence_counts = [r.quality.sequence_count for r in results]
        execution_times = [r.execution_time for r in results]
        
        # High-quality alignments (quality score > 0.7)
        high_quality = [r for r in results if r.quality.quality_score > 0.7]
        
        stats = {
            "total_alignments": len(results),
            "high_quality_alignments": len(high_quality),
            "quality_stats": {
                "min": min(quality_scores),
                "max": max(quality_scores),
                "mean": sum(quality_scores) / len(quality_scores),
                "median": sorted(quality_scores)[len(quality_scores) // 2]
            },
            "conservation_stats": {
                "min": min(conservation_scores),
                "max": max(conservation_scores),
                "mean": sum(conservation_scores) / len(conservation_scores)
            },
            "gap_stats": {
                "min": min(gap_percentages),
                "max": max(gap_percentages),
                "mean": sum(gap_percentages) / len(gap_percentages)
            },
            "sequence_count_stats": {
                "min": min(sequence_counts),
                "max": max(sequence_counts),
                "mean": sum(sequence_counts) / len(sequence_counts)
            },
            "performance": {
                "total_time": sum(execution_times),
                "avg_time_per_alignment": sum(execution_times) / len(execution_times)
            }
        }
        
        return stats
    
    async def _execute_at_scale(self, scale_config: Dict[str, Any]) -> Any:
        """Execute MUSCLE at specified scale"""
        max_sequences = scale_config.get("max_sequences", 100)
        self.logger.info(f"Executing MUSCLE at scale: max_sequences={max_sequences}")
        return {"status": "scale_executed", "max_sequences": max_sequences}
    
    async def _find_executable_in_path(self) -> Optional[str]:
        """Find MUSCLE executable in system PATH"""
        import shutil
        return shutil.which("muscle")
    
    async def _check_tool_in_environment(self, env_path: str, env_name: str) -> bool:
        """Check if MUSCLE exists in conda environment"""
        muscle_path = Path(env_path) / "bin" / "muscle"
        return muscle_path.exists() and muscle_path.is_file()
    
    async def _check_tool_in_directory(self, directory: str) -> bool:
        """Check if MUSCLE exists in specified directory"""
        muscle_path = Path(directory) / "muscle"
        return muscle_path.exists() and muscle_path.is_file()
    
    async def _build_tool_in_environment(self, source_dir: str) -> bool:
        """Build MUSCLE in environment from source"""
        self.logger.info(f"Building MUSCLE from source in {source_dir}")
        # For MUSCLE, typically installed via package manager
        return False
    
    async def _generate_specific_suggestions(self) -> List[str]:
        """Generate MUSCLE-specific installation suggestions"""
        return [
            "Install MUSCLE via conda: conda install -c bioconda muscle",
            "Install MUSCLE via homebrew: brew install muscle",
            "Download from: https://www.drive5.com/muscle/downloads.htm"
        ]
    
    async def _get_alternative_methods(self) -> List[str]:
        """Get alternative alignment methods"""
        return [
            "ClustalW multiple sequence alignment",
            "MAFFT multiple sequence alignment",
            "T-Coffee multiple sequence alignment"
        ]

    async def export_alignment(self, alignment: MultipleSeqAlignment, 
                             output_file: str, format: str = "fasta") -> str:
        """
        Export alignment to file in specified format
        
        Args:
            alignment: Multiple sequence alignment
            output_file: Output file path
            format: Output format (fasta, clustal, phylip)
            
        Returns:
            str: Path to exported file
        """
        if format.lower() == "fasta":
            content = []
            for seq in alignment.sequences:
                content.append(f">{seq.sequence_id}")
                content.append(seq.aligned_sequence)
            
            with open(output_file, 'w') as f:
                f.write("\n".join(content))
                
        elif format.lower() == "clustal":
            # Basic Clustal format export
            content = ["CLUSTAL W (1.83) multiple sequence alignment\n"]
            
            # Write sequences in blocks
            block_size = 60
            for start in range(0, alignment.alignment_length, block_size):
                for seq in alignment.sequences:
                    seq_block = seq.aligned_sequence[start:start+block_size]
                    content.append(f"{seq.sequence_id:<16} {seq_block}")
                content.append("")  # Empty line between blocks
            
            with open(output_file, 'w') as f:
                f.write("\n".join(content))
                
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported alignment to {output_file} in {format} format")
        return output_file 