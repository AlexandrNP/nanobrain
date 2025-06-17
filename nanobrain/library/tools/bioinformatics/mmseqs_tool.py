"""
MMseqs2 Tool Implementation for NanoBrain Framework

This module provides a comprehensive wrapper for MMseqs2 clustering tool with:
- Auto-detection of existing installations (conda, system PATH, local)
- Automated conda environment creation (nanobrain-viral_protein-mmseqs2)
- Real protein clustering with progressive scaling
- Automatic retry with exponential backoff
- Detailed error diagnostics and troubleshooting
"""

import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .base_bioinformatics_tool import (
    BioinformaticsExternalTool,
    BioinformaticsToolConfig,
    InstallationStatus,
    DiagnosticReport,
    ToolInstallationError,
    ToolExecutionError
)
from nanobrain.core.external_tool import ToolResult
from nanobrain.core.logging_system import get_logger


@dataclass
class MMseqs2Config(BioinformaticsToolConfig):
    """Configuration for MMseqs2 tool"""
    # Installation configuration
    conda_package: str = "mmseqs2"
    conda_channel: str = "bioconda"
    git_repository: str = "https://github.com/soedinglab/MMseqs2.git"
    environment_name: str = "nanobrain-viral_protein-mmseqs2"
    
    # Clustering parameters
    min_seq_id: float = 0.3
    coverage: float = 0.8
    cluster_mode: int = 0
    sensitivity: float = 7.5
    
    # Progressive scaling configuration
    progressive_scaling: Dict[int, Dict[str, Any]] = field(default_factory=lambda: {
        1: {"max_sequences": 50, "sensitivity": 4.0},      # Fast test
        2: {"max_sequences": 100, "sensitivity": 5.5},     # Basic validation
        3: {"max_sequences": 500, "sensitivity": 7.0},     # Medium scale
        4: {"max_sequences": 2000, "sensitivity": 7.5}     # Full scale
    })
    
    # Performance settings
    threads: int = 4
    memory_limit: str = "8G"
    tmp_dir: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        # Set tool name
        self.tool_name = "mmseqs2"
        # MMseqs2 installation locations to check
        self.local_installation_paths = [
            "/usr/local/bin",
            "/opt/homebrew/bin",
            f"{os.path.expanduser('~')}/bin"
        ]


@dataclass
class ClusterResult:
    """MMseqs2 clustering result"""
    cluster_id: str
    representative_seq: str
    member_sequences: List[str]
    cluster_size: int
    
    @property
    def is_singleton(self) -> bool:
        """Check if cluster contains only one sequence"""
        return self.cluster_size == 1


@dataclass
class ClusteringReport:
    """Comprehensive clustering analysis report"""
    total_input_sequences: int
    total_clusters: int
    singleton_clusters: int
    multi_member_clusters: int
    largest_cluster_size: int
    clustering_efficiency: float
    execution_time: float
    
    @property
    def reduction_ratio(self) -> float:
        """Calculate sequence reduction ratio"""
        if self.total_input_sequences == 0:
            return 0.0
        return 1.0 - (self.total_clusters / self.total_input_sequences)


class MMseqs2DataError(ToolExecutionError):
    """Raised when MMseqs2 data processing fails"""
    pass


class MMseqs2InstallationError(ToolInstallationError):
    """Raised when MMseqs2 installation fails"""
    pass


class MMseqs2Tool(BioinformaticsExternalTool):
    """
    MMseqs2 (Many-against-Many sequence searching) tool wrapper.
    
    Provides protein sequence clustering with:
    - Auto-detection and installation via conda/bioconda
    - Progressive scaling for different data volumes
    - Comprehensive clustering analysis and reporting
    - Automatic retry with exponential backoff
    - Real-time performance monitoring
    """
    
    def __init__(self, config: Optional[MMseqs2Config] = None):
        if config is None:
            config = MMseqs2Config()
            
        super().__init__(config)
        
        self.mmseqs_config = config
        self.logger = get_logger("mmseqs2_tool")
        
        # Executable path
        self.mmseqs_executable = None
        
        # Temporary directories
        self.temp_dirs = []
        
    async def initialize_tool(self) -> InstallationStatus:
        """Initialize MMseqs2 tool with detection and installation"""
        self.logger.info("🔄 Initializing MMseqs2 tool...")
        
        try:
            # Step 1: Try to detect existing installation
            status = await self.detect_existing_installation()
            
            if status.found:
                self.mmseqs_executable = status.executable_path or "mmseqs"
                self.logger.info(f"✅ MMseqs2 found: {status.installation_type}")
                return status
            
            # Step 2: Attempt automated installation
            self.logger.info("🔄 MMseqs2 not found, attempting installation...")
            installation_success = await self.install_if_missing()
            
            if installation_success:
                # Re-detect after installation
                status = await self.detect_existing_installation()
                if status.found:
                    self.mmseqs_executable = status.executable_path or "mmseqs"
                    self.logger.info("✅ MMseqs2 installed and configured successfully")
                    return status
            
            # Step 3: Installation failed
            raise MMseqs2InstallationError(
                "Failed to install MMseqs2. Please install manually via conda or from source."
            )
            
        except Exception as e:
            self.logger.error(f"❌ MMseqs2 initialization failed: {e}")
            raise
    
    async def cluster_sequences(self, fasta_content: str, 
                               output_prefix: str = "clusters") -> ClusteringReport:
        """
        Cluster protein sequences using MMseqs2
        
        Args:
            fasta_content: Input FASTA sequences
            output_prefix: Prefix for output files
            
        Returns:
            ClusteringReport: Comprehensive clustering analysis
        """
        self.logger.info("🔄 Starting MMseqs2 protein clustering...")
        
        # Get scale configuration
        scale_config = self.scale_config.get(self.current_scale_level, {})
        max_sequences = scale_config.get("max_sequences", 1000)
        sensitivity = scale_config.get("sensitivity", self.mmseqs_config.sensitivity)
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="mmseqs2_")
        self.temp_dirs.append(temp_dir)
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Step 1: Prepare input sequences
            input_sequences = await self._prepare_input_sequences(
                fasta_content, temp_dir, max_sequences
            )
            
            if len(input_sequences) == 0:
                raise MMseqs2DataError("No valid sequences found for clustering")
            
            # Step 2: Create MMseqs2 database
            db_path = await self._create_mmseqs_database(
                input_sequences, temp_dir, output_prefix
            )
            
            # Step 3: Perform clustering
            cluster_db_path = await self._perform_clustering(
                db_path, temp_dir, output_prefix, sensitivity
            )
            
            # Step 4: Parse clustering results
            clusters = await self._parse_clustering_results(
                db_path, cluster_db_path, input_sequences
            )
            
            # Step 5: Generate comprehensive report
            execution_time = asyncio.get_event_loop().time() - start_time
            report = await self._generate_clustering_report(
                clusters, len(input_sequences), execution_time
            )
            
            self.logger.info(
                f"✅ Clustering completed: {report.total_input_sequences} → "
                f"{report.total_clusters} clusters in {execution_time:.2f}s"
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ MMseqs2 clustering failed: {e}")
            raise MMseqs2DataError(f"Clustering failed: {e}")
        finally:
            # Cleanup temporary directory
            await self._cleanup_temp_dir(temp_dir)
    
    async def _prepare_input_sequences(self, fasta_content: str, 
                                      temp_dir: str, max_sequences: int) -> List[Dict[str, str]]:
        """Prepare and validate input sequences"""
        sequences = []
        current_header = None
        current_sequence = ""
        
        for line in fasta_content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                # Save previous sequence
                if current_header and current_sequence:
                    sequences.append({
                        'header': current_header,
                        'sequence': current_sequence
                    })
                    
                current_header = line
                current_sequence = ""
            else:
                current_sequence += line
        
        # Save last sequence
        if current_header and current_sequence:
            sequences.append({
                'header': current_header,
                'sequence': current_sequence
            })
        
        # Apply sequence limit for scaling
        if len(sequences) > max_sequences:
            self.logger.info(f"Limiting sequences to {max_sequences} for current scale level")
            sequences = sequences[:max_sequences]
        
        # Validate sequences
        valid_sequences = []
        for seq_data in sequences:
            if len(seq_data['sequence']) >= 10:  # Minimum length filter
                valid_sequences.append(seq_data)
            else:
                self.logger.warning(f"Skipping short sequence: {seq_data['header']}")
        
        self.logger.info(f"Prepared {len(valid_sequences)} valid sequences for clustering")
        return valid_sequences
    
    async def _create_mmseqs_database(self, sequences: List[Dict[str, str]], 
                                     temp_dir: str, output_prefix: str) -> str:
        """Create MMseqs2 sequence database"""
        # Write sequences to FASTA file
        fasta_path = os.path.join(temp_dir, f"{output_prefix}_input.fasta")
        with open(fasta_path, 'w') as f:
            for seq_data in sequences:
                f.write(f"{seq_data['header']}\n{seq_data['sequence']}\n")
        
        # Create MMseqs2 database
        db_path = os.path.join(temp_dir, f"{output_prefix}_db")
        
        createdb_cmd = [
            self.mmseqs_executable, "createdb",
            fasta_path, db_path
        ]
        
        result = await self.execute_with_retry(createdb_cmd)
        
        if not result.success:
            raise MMseqs2DataError(f"Failed to create MMseqs2 database: {result.stderr_text}")
        
        self.logger.debug(f"Created MMseqs2 database: {db_path}")
        return db_path
    
    async def _perform_clustering(self, db_path: str, temp_dir: str, 
                                 output_prefix: str, sensitivity: float) -> str:
        """Perform MMseqs2 clustering"""
        cluster_db_path = os.path.join(temp_dir, f"{output_prefix}_cluster")
        tmp_path = os.path.join(temp_dir, "tmp")
        
        # Create tmp directory
        os.makedirs(tmp_path, exist_ok=True)
        
        # Clustering command
        cluster_cmd = [
            self.mmseqs_executable, "cluster",
            db_path, cluster_db_path, tmp_path,
            "--min-seq-id", str(self.mmseqs_config.min_seq_id),
            "--coverage", str(self.mmseqs_config.coverage),
            "--cluster-mode", str(self.mmseqs_config.cluster_mode),
            "-s", str(sensitivity),
            "--threads", str(self.mmseqs_config.threads)
        ]
        
        # Add memory limit if specified
        if self.mmseqs_config.memory_limit:
            cluster_cmd.extend(["--split-memory-limit", self.mmseqs_config.memory_limit])
        
        result = await self.execute_with_retry(cluster_cmd, timeout=600)  # 10 minute timeout
        
        if not result.success:
            raise MMseqs2DataError(f"MMseqs2 clustering failed: {result.stderr_text}")
        
        self.logger.debug(f"Clustering completed: {cluster_db_path}")
        return cluster_db_path
    
    async def _parse_clustering_results(self, db_path: str, cluster_db_path: str, 
                                       input_sequences: List[Dict[str, str]]) -> List[ClusterResult]:
        """Parse MMseqs2 clustering results"""
        # Convert cluster results to TSV format
        temp_dir = os.path.dirname(cluster_db_path)
        cluster_tsv_path = os.path.join(temp_dir, "clusters.tsv")
        
        createtsv_cmd = [
            self.mmseqs_executable, "createtsv",
            db_path, db_path, cluster_db_path, cluster_tsv_path
        ]
        
        result = await self.execute_with_retry(createtsv_cmd)
        
        if not result.success:
            raise MMseqs2DataError(f"Failed to create cluster TSV: {result.stderr_text}")
        
        # Parse TSV results
        clusters_dict = {}
        
        try:
            with open(cluster_tsv_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        representative = parts[0]
                        member = parts[1]
                        
                        if representative not in clusters_dict:
                            clusters_dict[representative] = {
                                'representative': representative,
                                'members': []
                            }
                        
                        clusters_dict[representative]['members'].append(member)
        
        except Exception as e:
            raise MMseqs2DataError(f"Failed to parse clustering results: {e}")
        
        # Convert to ClusterResult objects
        clusters = []
        for i, (rep, cluster_data) in enumerate(clusters_dict.items()):
            cluster = ClusterResult(
                cluster_id=f"cluster_{i+1}",
                representative_seq=rep,
                member_sequences=cluster_data['members'],
                cluster_size=len(cluster_data['members'])
            )
            clusters.append(cluster)
        
        self.logger.debug(f"Parsed {len(clusters)} clusters from results")
        return clusters
    
    async def _generate_clustering_report(self, clusters: List[ClusterResult], 
                                         total_input: int, execution_time: float) -> ClusteringReport:
        """Generate comprehensive clustering analysis report"""
        singleton_clusters = sum(1 for c in clusters if c.is_singleton)
        multi_member_clusters = len(clusters) - singleton_clusters
        largest_cluster_size = max((c.cluster_size for c in clusters), default=0)
        
        # Calculate clustering efficiency (reduction ratio)
        clustering_efficiency = 1.0 - (len(clusters) / total_input) if total_input > 0 else 0.0
        
        report = ClusteringReport(
            total_input_sequences=total_input,
            total_clusters=len(clusters),
            singleton_clusters=singleton_clusters,
            multi_member_clusters=multi_member_clusters,
            largest_cluster_size=largest_cluster_size,
            clustering_efficiency=clustering_efficiency,
            execution_time=execution_time
        )
        
        # Log summary
        self.logger.info(f"Clustering report: {total_input} → {len(clusters)} clusters")
        self.logger.info(f"Singletons: {singleton_clusters}, Multi-member: {multi_member_clusters}")
        self.logger.info(f"Efficiency: {clustering_efficiency:.2%}, Time: {execution_time:.2f}s")
        
        return report
    
    async def _cleanup_temp_dir(self, temp_dir: str) -> None:
        """Clean up temporary directory"""
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            if temp_dir in self.temp_dirs:
                self.temp_dirs.remove(temp_dir)
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")
    
    async def cleanup_all_temp_dirs(self) -> None:
        """Clean up all temporary directories"""
        for temp_dir in self.temp_dirs.copy():
            await self._cleanup_temp_dir(temp_dir)
    
    # Implementation of abstract methods from base class
    
    async def _execute_at_scale(self, scale_config: Dict[str, Any]) -> Any:
        """Execute MMseqs2 at specified scale"""
        max_sequences = scale_config.get("max_sequences", 500)
        sensitivity = scale_config.get("sensitivity", 7.5)
        
        self.logger.info(f"Executing MMseqs2 at scale: max_sequences={max_sequences}, sensitivity={sensitivity}")
        
        # This would be called with actual FASTA content in real usage
        return {
            "scale_config": scale_config,
            "max_sequences": max_sequences,
            "sensitivity": sensitivity,
            "ready": True
        }
    
    async def _find_executable_in_path(self) -> Optional[str]:
        """Find MMseqs2 executable in system PATH"""
        try:
            result = await asyncio.create_subprocess_exec(
                "which", "mmseqs",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                return stdout.decode().strip()
                
        except Exception:
            pass
            
        return None
    
    async def _check_tool_in_environment(self, env_path: str, env_name: str) -> bool:
        """Check if MMseqs2 is available in conda environment"""
        try:
            # Check if mmseqs executable exists in environment
            mmseqs_path = Path(env_path) / "bin" / "mmseqs"
            if mmseqs_path.exists():
                self.mmseqs_executable = str(mmseqs_path)
                return True
        except Exception:
            pass
            
        return False
    
    async def _check_tool_in_directory(self, directory: str) -> bool:
        """Check if MMseqs2 is available in specific directory"""
        try:
            mmseqs_path = Path(directory) / "mmseqs"
            return mmseqs_path.exists() and os.access(mmseqs_path, os.X_OK)
        except Exception:
            return False
    
    async def _build_tool_in_environment(self, source_dir: str) -> bool:
        """Build MMseqs2 from source in conda environment"""
        try:
            self.logger.info("🔄 Building MMseqs2 from source...")
            
            # Create build directory
            build_dir = os.path.join(source_dir, "build")
            os.makedirs(build_dir, exist_ok=True)
            
            # Configure with cmake
            cmake_cmd = [
                "conda", "run", "-n", self.environment_name,
                "cmake", "-B", build_dir, "-S", source_dir
            ]
            
            cmake_process = await asyncio.create_subprocess_exec(
                *cmake_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await cmake_process.communicate()
            
            if cmake_process.returncode != 0:
                self.logger.error("CMake configuration failed")
                return False
            
            # Build with make
            make_cmd = [
                "conda", "run", "-n", self.environment_name,
                "make", "-C", build_dir, "-j", str(self.mmseqs_config.threads)
            ]
            
            make_process = await asyncio.create_subprocess_exec(
                *make_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await make_process.communicate()
            
            if make_process.returncode == 0:
                self.logger.info("✅ MMseqs2 built successfully from source")
                return True
            else:
                self.logger.error("Make build failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Source build failed: {e}")
            return False
    
    async def _generate_specific_suggestions(self) -> List[str]:
        """Generate MMseqs2 specific installation suggestions"""
        return [
            f"Install via conda: conda create -n {self.environment_name} -c bioconda mmseqs2",
            "Install via homebrew: brew install mmseqs2",
            "Download pre-compiled binaries from https://github.com/soedinglab/MMseqs2/releases",
            "Build from source: git clone https://github.com/soedinglab/MMseqs2.git",
            "Ensure sufficient memory for clustering large datasets"
        ]
    
    async def _get_alternative_methods(self) -> List[str]:
        """Get alternative installation methods for MMseqs2"""
        return [
            "Use Docker: docker pull soedinglab/mmseqs2",
            "Install via package manager (apt/yum): sudo apt install mmseqs2",
            "Use online clustering services (for small datasets)",
            "Alternative clustering tools: CD-HIT, USEARCH, VSEARCH"
        ] 