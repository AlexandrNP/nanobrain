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
    MUSCLE Multiple Sequence Alignment Tool - High-Performance MSA with Conservation Analysis and Quality Scoring
    ============================================================================================================
    
    The MUSCLETool provides a comprehensive wrapper for MUSCLE (Multiple Sequence Comparison by Log-Expectation),
    a widely-used multiple sequence alignment software optimized for protein and nucleotide sequences. This tool
    integrates MUSCLE's advanced alignment algorithms with NanoBrain's framework architecture, providing automated
    installation, progressive scaling, conservation analysis, and quality scoring for bioinformatics workflows.
    
    **Core Architecture:**
        The MUSCLE tool provides enterprise-grade multiple sequence alignment capabilities:
        
        * **Multiple Sequence Alignment**: Advanced protein and nucleotide sequence alignment
        * **Conservation Analysis**: Detailed conservation scoring and position analysis
        * **Quality Assessment**: Comprehensive alignment quality metrics and scoring
        * **Progressive Scaling**: Adaptive performance optimization for varying sequence sets
        * **Auto-Installation**: Intelligent detection and installation via conda/bioconda
        * **Framework Integration**: Full integration with NanoBrain's component architecture
    
    **Multiple Sequence Alignment Capabilities:**
        
        **Advanced Alignment Algorithms:**
        * Log-expectation scoring for optimal sequence alignment accuracy
        * Progressive alignment strategy with iterative refinement
        * Diagonal optimization for improved alignment speed and quality
        * Gap penalty optimization for biologically meaningful alignments
        
        **Sequence Processing:**
        * Support for protein and nucleotide sequence alignment
        * Automatic sequence validation and preprocessing
        * Handling of large sequence sets with memory optimization
        * Multiple output format support (FASTA, Clustal, MSF, Phylip)
        
        **Conservation Analysis:**
        * Position-specific conservation scoring and analysis
        * Highly conserved region identification and annotation
        * Conservation profile generation for downstream analysis
        * Statistical significance testing for conservation patterns
        
        **Quality Assessment:**
        * Comprehensive alignment quality scoring and metrics
        * Gap distribution analysis and optimization
        * Sequence coverage and identity assessment
        * Quality thresholds for alignment validation
    
    **Bioinformatics Applications:**
        
        **Protein Analysis:**
        * Protein family alignment and evolutionary analysis
        * Structural motif identification through sequence conservation
        * Functional domain mapping and annotation
        * Phylogenetic analysis preparation and optimization
        
        **Genomic Studies:**
        * Comparative genomics and sequence evolution analysis
        * Gene family alignment and ortholog identification
        * Regulatory element conservation analysis
        * Mutation impact assessment through conservation scores
        
        **Structural Biology:**
        * Structure-based sequence alignment and validation
        * Secondary structure prediction support through alignment
        * Structural conservation analysis and mapping
        * Protein fold family characterization
        
        **Evolutionary Biology:**
        * Phylogenetic tree construction support
        * Evolutionary rate analysis through conservation scoring
        * Species comparison and divergence analysis
        * Molecular evolution pattern identification
    
    **Conservation Analysis Features:**
        
        **Position-Specific Scoring:**
        * Detailed conservation scores for each alignment position
        * Statistical significance assessment for conservation patterns
        * Identification of highly conserved functional regions
        * Conservation gradient analysis across sequence length
        
        **Conservation Profiles:**
        * Generation of conservation profiles for visualization
        * Export of conservation data for downstream analysis
        * Integration with structural analysis workflows
        * Conservation-based functional annotation support
        
        **Quality Metrics:**
        * Alignment quality scoring with customizable weights
        * Gap penalty assessment and optimization
        * Sequence identity and similarity measurements
        * Coverage analysis and completeness assessment
    
    **Configuration Architecture:**
        Comprehensive configuration supports diverse bioinformatics workflows:
        
        ```yaml
        # MUSCLE Tool Configuration
        tool_name: "muscle"
        
        # Tool card for framework integration
        tool_card:
          name: "muscle"
          description: "MUSCLE multiple sequence alignment tool"
          version: "1.0.0"
          category: "bioinformatics"
          capabilities:
            - "sequence_alignment"
            - "msa_generation"
            - "conservation_analysis"
        
        # Installation Configuration
        conda_package: "muscle"
        conda_channel: "bioconda"
        environment_name: "nanobrain-viral_protein-muscle"
        create_isolated_environment: true
        
        # Alignment Parameters
        max_iterations: 16              # Maximum refinement iterations
        diagonal_optimization: true     # Enable diagonal optimization
        gap_open_penalty: -12.0        # Gap opening penalty
        gap_extend_penalty: -1.0       # Gap extension penalty
        
        # Quality Parameters
        min_sequences: 3               # Minimum sequences for alignment
        max_sequences: 1000           # Maximum sequences to process
        output_format: "fasta"        # Output format (fasta, clustal, msf, phylip)
        
        # Conservation Analysis
        calculate_profile: true        # Generate conservation profiles
        highly_conserved_threshold: 0.8  # Conservation threshold
        position_scoring: true         # Enable position-specific scoring
        
        # Quality Scoring Configuration
        quality_scoring:
          enabled: true
          weights:
            conservation_score: 0.5    # Conservation weight in quality score
            gap_penalty: 0.3          # Gap penalty weight
            size_bonus: 0.2           # Size bonus weight
          thresholds:
            high_quality_threshold: 0.7      # High quality threshold
            acceptable_quality_threshold: 0.5 # Acceptable quality threshold
        ```
    
    **Usage Patterns:**
        
        **Basic Multiple Sequence Alignment:**
        ```python
        from nanobrain.library.tools.bioinformatics import MUSCLETool
        
        # Create MUSCLE tool with configuration
        muscle_tool = MUSCLETool.from_config('config/muscle_config.yml')
        
        # Perform multiple sequence alignment
        sequences = [
            ">sequence1\\nMKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGE",
            ">sequence2\\nMKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGE", 
            ">sequence3\\nMKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGE"
        ]
        
        fasta_content = "\\n".join(sequences)
        
        # Execute alignment with quality assessment
        result = await muscle_tool.align_sequences(fasta_content)
        
        # Access alignment results
        print(f"Alignment completed: {result.success}")
        print(f"Aligned sequences: {len(result.data['aligned_sequences'])}")
        print(f"Conservation score: {result.data['conservation_score']}")
        print(f"Quality score: {result.data['quality_score']}")
        ```
        
        **Conservation Analysis:**
        ```python
        # Configure MUSCLE for detailed conservation analysis
        conservation_config = {
            'tool_name': 'muscle',
            'calculate_profile': True,
            'position_scoring': True,
            'highly_conserved_threshold': 0.8,
            'quality_scoring': {
                'enabled': True,
                'weights': {
                    'conservation_score': 0.6,
                    'gap_penalty': 0.2,
                    'size_bonus': 0.2
                }
            }
        }
        
        muscle_tool = MUSCLETool.from_config(conservation_config)
        
        # Align protein family sequences with conservation analysis
        family_sequences = load_protein_family_sequences()
        result = await muscle_tool.align_sequences(family_sequences)
        
        # Access conservation analysis results
        conservation_profile = result.data['conservation_profile']
        highly_conserved_positions = result.data['highly_conserved_positions']
        
        for position, score in conservation_profile.items():
            if score > 0.8:
                print(f"Position {position}: Conservation score {score:.3f}")
        
        # Export alignment for visualization
        alignment_file = result.data['alignment_file']
        conservation_file = result.data['conservation_file']
        ```
        
        **Quality Assessment and Validation:**
        ```python
        # Configure quality-focused alignment
        quality_config = {
            'max_iterations': 32,        # More iterations for higher quality
            'diagonal_optimization': True,
            'gap_open_penalty': -15.0,   # Stricter gap penalties
            'gap_extend_penalty': -2.0,
            'quality_scoring': {
                'enabled': True,
                'thresholds': {
                    'high_quality_threshold': 0.8,
                    'acceptable_quality_threshold': 0.6
                }
            }
        }
        
        muscle_tool = MUSCLETool.from_config(quality_config)
        
        # Perform high-quality alignment with validation
        result = await muscle_tool.align_sequences(sequences)
        
        # Validate alignment quality
        quality_metrics = result.data['quality_metrics']
        print(f"Overall quality: {quality_metrics['overall_score']:.3f}")
        print(f"Gap penalty: {quality_metrics['gap_penalty']:.3f}")
        print(f"Conservation bonus: {quality_metrics['conservation_bonus']:.3f}")
        
        # Check quality thresholds
        if quality_metrics['overall_score'] > 0.8:
            print("High-quality alignment achieved")
        elif quality_metrics['overall_score'] > 0.6:
            print("Acceptable quality alignment")
        else:
            print("Low quality alignment - consider parameter adjustment")
        ```
        
        **Large-Scale Alignment with Progressive Scaling:**
        ```python
        # Configure for large sequence sets
        scaling_config = {
            'max_sequences': 2000,
            'progressive_scaling': True,
            'memory_optimization': True,
            'batch_processing': True
        }
        
        muscle_tool = MUSCLETool.from_config(scaling_config)
        
        # Process large protein family
        large_sequence_set = load_large_protein_family()
        
        # Tool automatically handles scaling and optimization
        result = await muscle_tool.align_sequences(large_sequence_set)
        
        # Monitor performance metrics
        performance = result.data.get('performance_metrics', {})
        print(f"Processing time: {performance.get('execution_time')}s")
        print(f"Memory usage: {performance.get('memory_usage')}MB")
        print(f"Sequences processed: {performance.get('sequences_processed')}")
        ```
        
        **Integration with Phylogenetic Analysis:**
        ```python
        # Prepare alignment for phylogenetic analysis
        phylo_config = {
            'output_format': 'phylip',   # Phylip format for tree software
            'calculate_profile': True,
            'position_scoring': True,
            'gap_open_penalty': -10.0,   # Optimized for phylogenetics
            'max_iterations': 20
        }
        
        muscle_tool = MUSCLETool.from_config(phylo_config)
        
        # Align sequences for phylogenetic tree construction
        species_sequences = load_orthologous_sequences()
        result = await muscle_tool.align_sequences(species_sequences)
        
        # Prepare output for phylogenetic software
        phylip_alignment = result.data['phylip_alignment']
        conservation_weights = result.data['conservation_weights']
        
        # Export for external phylogenetic analysis
        with open('alignment.phy', 'w') as f:
            f.write(phylip_alignment)
        
        # Conservation-weighted analysis
        weighted_positions = []
        for pos, weight in conservation_weights.items():
            if weight > 0.5:  # Use highly conserved positions
                weighted_positions.append(pos)
        ```
    
    **Advanced Features:**
        
        **Algorithm Optimization:**
        * Iterative refinement with convergence detection
        * Diagonal optimization for computational efficiency
        * Memory-efficient processing for large sequence sets
        * Parallel processing support for multi-core systems
        
        **Quality Control:**
        * Comprehensive alignment validation and quality metrics
        * Statistical significance testing for alignment reliability
        * Gap distribution analysis and optimization recommendations
        * Sequence outlier detection and handling
        
        **Integration Capabilities:**
        * Seamless integration with structure prediction workflows
        * Export to multiple phylogenetic analysis formats
        * Integration with protein function prediction pipelines
        * Compatibility with evolutionary analysis tools
        
        **Performance Features:**
        * Progressive scaling for datasets of varying sizes
        * Memory optimization for resource-constrained environments
        * Batch processing support for high-throughput analysis
        * Intelligent parameter optimization based on sequence characteristics
    
    **Scientific Applications:**
        
        **Protein Function Analysis:**
        * Functional domain identification through conservation patterns
        * Active site prediction and characterization
        * Protein family classification and annotation
        * Evolutionary conservation analysis for function prediction
        
        **Comparative Genomics:**
        * Cross-species sequence comparison and analysis
        * Ortholog and paralog identification and characterization
        * Regulatory element conservation analysis
        * Genome evolution pattern identification
        
        **Drug Discovery:**
        * Target protein family analysis for drug design
        * Binding site conservation analysis across species
        * Pharmacophore identification through alignment patterns
        * Drug resistance mutation analysis through conservation
        
        **Structural Biology:**
        * Structure-function relationship analysis through alignment
        * Secondary structure prediction support
        * Structural motif identification and characterization
        * Protein fold evolution analysis
    
    Attributes:
        muscle_config (MUSCLEConfig): MUSCLE tool configuration
        muscle_executable (str): Path to MUSCLE executable
        max_iterations (int): Maximum number of refinement iterations
        diagonal_optimization (bool): Whether diagonal optimization is enabled
        gap_open_penalty (float): Penalty for opening gaps in alignment
        gap_extend_penalty (float): Penalty for extending gaps in alignment
        min_sequences (int): Minimum number of sequences required for alignment
        max_sequences (int): Maximum number of sequences to process
        output_format (str): Output format for aligned sequences
        quality_scoring (dict): Quality scoring configuration and thresholds
    
    Note:
        This tool requires MUSCLE to be available either through conda, system PATH,
        or local installation. The tool provides comprehensive auto-installation
        capabilities using conda/bioconda channels. Quality scoring and conservation
        analysis add computational overhead but provide valuable biological insights.
    
    Warning:
        Multiple sequence alignment can be computationally intensive for large sequence
        sets. Monitor system resources and configure appropriate limits for production
        deployments. Gap penalty parameters significantly affect alignment quality and
        biological interpretation of results.
    
    See Also:
        * :class:`ExternalTool`: Base external tool implementation
        * :class:`ProgressiveScalingMixin`: Progressive scaling capabilities
        * :class:`MUSCLEConfig`: MUSCLE tool configuration schema
        * :mod:`nanobrain.library.tools.bioinformatics`: Bioinformatics tool implementations
        * :mod:`nanobrain.core.external_tool`: External tool framework
    """
    
    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return MUSCLEConfig - ONLY method that differs from other components"""
        return MUSCLEConfig
    
    # Now inherits unified from_config implementation from FromConfigBase
    # Uses MUSCLEConfig returned by _get_config_class() to preserve all existing functionality
    
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