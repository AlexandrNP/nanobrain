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
from pydantic import Field

from nanobrain.core.external_tool import (
    ExternalTool,
    ToolResult,
    InstallationStatus,
    DiagnosticReport,
    ToolInstallationError,
    ToolExecutionError,
    ExternalToolConfig
)
from nanobrain.core.progressive_scaling import ProgressiveScalingMixin
from nanobrain.core.tool import ToolConfig
from nanobrain.core.logging_system import get_logger


@dataclass
class MMseqs2Config(ExternalToolConfig):
    """Configuration for MMseqs2 protein clustering tool"""
    # Tool identification
    tool_name: str = "mmseqs2"
    
    # Default tool card
    tool_card: Dict[str, Any] = field(default_factory=lambda: {
        "name": "mmseqs2",
        "description": "MMseqs2 tool for protein sequence clustering and analysis",
        "version": "1.0.0",
        "category": "bioinformatics", 
        "capabilities": ["protein_clustering", "sequence_analysis", "similarity_search"]
    })
    
    # Installation configuration
    conda_package: str = "mmseqs2"
    conda_channel: str = "bioconda"
    git_repository: str = "https://github.com/soedinglab/MMseqs2.git"
    environment_name: str = "nanobrain-viral_protein-mmseqs2"
    create_isolated_environment: bool = True
    
    # Clustering parameters
    min_seq_id: float = 0.3
    coverage: float = 0.8
    cluster_mode: int = 0
    sensitivity: float = 7.5
    
    # Progressive scaling configuration
    progressive_scaling: Dict[int, Dict[str, Any]] = field(default_factory=lambda: {
        1: {"max_sequences": 50, "sensitivity": 4.0, "description": "Fast test"},
        2: {"max_sequences": 100, "sensitivity": 5.5, "description": "Basic validation"},
        3: {"max_sequences": 500, "sensitivity": 7.0, "description": "Medium scale"},
        4: {"max_sequences": 2000, "sensitivity": 7.5, "description": "Full scale"}
    })
    
    # Performance settings
    threads: int = 4
    memory_limit: str = "8G"
    tmp_dir: Optional[str] = None
    
    # Cache directory for MMseqs2 databases
    cache_dir: Optional[str] = None
    
    # Installation paths
    local_installation_paths: List[str] = field(default_factory=lambda: [
        "/usr/local/bin",
        "/opt/homebrew/bin",
        "~/bin"
    ])


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


class MMseqs2Tool(ProgressiveScalingMixin, ExternalTool):
    """
    MMseqs2 (Many-against-Many sequence searching) tool wrapper.
    Enhanced with mandatory from_config pattern implementation.
    
    Provides protein sequence clustering with:
    - Auto-detection and installation via conda/bioconda
    - Progressive scaling for different data volumes
    - Comprehensive clustering analysis and reporting
    - Automatic retry with exponential backoff
    - Real-time performance monitoring
    """
    
    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return MMseqs2Config - ONLY method that differs from other components"""
        return MMseqs2Config
    
    # REMOVED: Custom from_config method - now inherits unified implementation
    # Now inherits unified from_config implementation from FromConfigBase
    # Uses MMseqs2Config returned by _get_config_class() to preserve all existing functionality
    
    def __init__(self, config: MMseqs2Config, **kwargs):
        """Initialize MMseqs2Tool with configuration"""
        if config is None:
            config = MMseqs2Config()
            
        # Ensure name is set consistently
        if not hasattr(config, 'tool_name') or not config.tool_name:
            config.tool_name = "mmseqs2"
        
        # Initialize parent classes
        super().__init__(config, **kwargs)
        
        # MMseqs2 specific initialization
        self.mmseqs_config = config
        self.name = config.tool_name
        self.logger = get_logger(f"bio_tool_{self.name}")
        
        # MMseqs2 specific attributes
        self.min_seq_id = getattr(config, 'min_seq_id', 0.3)
        self.coverage = getattr(config, 'coverage', 0.8)
        self.cluster_mode = getattr(config, 'cluster_mode', 0)
        self.sensitivity = getattr(config, 'sensitivity', 7.5)
        self.threads = getattr(config, 'threads', 4)
        self.memory_limit = getattr(config, 'memory_limit', "8G")
        self.tmp_dir = getattr(config, 'tmp_dir', None)
        
        # Cache directory for MMseqs2 databases - configurable path
        cache_dir_config = getattr(config, 'cache_directory', None) or getattr(config, 'mmseqs2_cache_directory', None)
        if cache_dir_config:
            self.cache_dir = cache_dir_config
        else:
            # Fallback to default path
            self.cache_dir = "data/mmseqs2_cache"
        
        # Initialize cache directory
        from pathlib import Path
        cache_path = Path(self.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"MMseqs2 cache directory: {cache_path.absolute()}")
        
        # Executable path
        self.mmseqs_executable = None
        
        # Temporary directories
        self.temp_dirs = []
    
    # MMseqs2Tool inherits FromConfigBase.__init__ which prevents direct instantiation
        
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
        Cluster protein sequences using proper MMseqs2 workflow:
        1. mmseqs createdb - Convert FASTA to MMseqs2 database format
        2. mmseqs cluster - Perform clustering on the database  
        3. mmseqs createtsv - Generate TSV output with cluster assignments
        4. mmseqs createseqfiledb and result2flat - Generate FASTA output
        
        Args:
            fasta_content: Input FASTA sequences
            output_prefix: Prefix for output files
            
        Returns:
            ClusteringReport: Comprehensive clustering analysis
        """
        self.logger.info("🔄 Starting MMseqs2 protein clustering with proper workflow...")
        
        # Get scale configuration
        scale_config = self.scale_config.get(self.current_scale_level, {})
        max_sequences = scale_config.get("max_sequences", 1000)
        sensitivity = scale_config.get("sensitivity", self.mmseqs_config.sensitivity)
        
        # Use cache directory for database files
        from pathlib import Path
        cache_path = Path(self.cache_dir)
        
        # Create temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp(prefix="mmseqs2_tmp_")
        self.temp_dirs.append(temp_dir)
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Step 1: Prepare input sequences and save FASTA to cache
            input_sequences, fasta_path = await self._prepare_input_fasta(
                fasta_content, cache_path, output_prefix, max_sequences
            )
            
            if len(input_sequences) == 0:
                raise MMseqs2DataError("No valid sequences found for clustering")
            
            # Step 2: mmseqs createdb - Convert FASTA to MMseqs2 database format
            db_path = await self._mmseqs_createdb(fasta_path, cache_path, output_prefix)
            
            # Step 3: mmseqs cluster - Perform clustering on the database
            cluster_db_path = await self._mmseqs_cluster(db_path, cache_path, temp_dir, output_prefix, sensitivity)
            
            # Step 4: mmseqs createtsv - Generate TSV output with cluster assignments
            cluster_tsv_path = await self._mmseqs_createtsv(db_path, cluster_db_path, cache_path, output_prefix)
            
            # Step 5: mmseqs createseqfiledb and result2flat - Generate FASTA output
            cluster_fasta_path = await self._mmseqs_create_cluster_fasta(db_path, cluster_db_path, cache_path, output_prefix)
            
            # Step 6: Parse clustering results from TSV
            clusters = await self._parse_mmseqs_tsv_results(cluster_tsv_path, input_sequences)
            
            # Step 7: Generate comprehensive report
            execution_time = asyncio.get_event_loop().time() - start_time
            report = await self._generate_clustering_report(
                clusters, len(input_sequences), execution_time
            )
            
            self.logger.info(
                f"✅ MMseqs2 clustering completed: {report.total_input_sequences} → "
                f"{report.total_clusters} clusters in {execution_time:.2f}s"
            )
            self.logger.info(f"💾 Database files cached in: {cache_path}")
            self.logger.info(f"📄 TSV results: {cluster_tsv_path}")
            self.logger.info(f"🧬 FASTA results: {cluster_fasta_path}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ MMseqs2 clustering failed: {e}")
            raise MMseqs2DataError(f"Clustering failed: {e}")
        finally:
            # Cleanup temporary directory (keep cache directory)
            await self._cleanup_temp_dir(temp_dir)
    
    async def _prepare_input_fasta(self, fasta_content: str, cache_path: Path, 
                                  output_prefix: str, max_sequences: int) -> tuple[List[Dict[str, str]], str]:
        """Prepare input sequences and save FASTA file to cache directory"""
        sequences = []
        current_header = None
        current_sequence = ""
        
        # Parse FASTA content
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
        
        # Save FASTA file to cache directory
        fasta_path = cache_path / f"{output_prefix}.fasta"
        with open(fasta_path, 'w') as f:
            for seq_data in valid_sequences:
                f.write(f"{seq_data['header']}\n{seq_data['sequence']}\n")
        
        self.logger.info(f"Prepared {len(valid_sequences)} valid sequences for clustering")
        self.logger.info(f"💾 Saved FASTA to cache: {fasta_path}")
        
        return valid_sequences, str(fasta_path)
    
    async def _mmseqs_createdb(self, fasta_path: str, cache_path: Path, output_prefix: str) -> str:
        """Step 2: mmseqs createdb - Convert FASTA to MMseqs2 database format"""
        
        # Database will be saved in cache directory
        db_path = cache_path / f"{output_prefix}_DB"
        
        createdb_cmd = [
            self.mmseqs_executable, "createdb",
            fasta_path, str(db_path)
        ]
        
        self.logger.info(f"🔧 Running: mmseqs createdb {fasta_path} {db_path}")
        result = await self._execute_with_retry(createdb_cmd)
        
        if not result.success:
            raise MMseqs2DataError(f"Failed to create MMseqs2 database: {result.stderr_text}")
        
        self.logger.info(f"✅ Created MMseqs2 database: {db_path}")
        return str(db_path)
    
    async def _mmseqs_cluster(self, db_path: str, cache_path: Path, temp_dir: str, 
                             output_prefix: str, sensitivity: float) -> str:
        """Step 3: mmseqs cluster - Perform clustering on the database"""
        
        # Cluster database will be saved in cache directory  
        cluster_db_path = cache_path / f"{output_prefix}_DB_clu"
        tmp_path = os.path.join(temp_dir, "tmp")
        
        # Create tmp directory
        os.makedirs(tmp_path, exist_ok=True)
        
        # Clustering command: mmseqs cluster DB DB_clu tmp
        cluster_cmd = [
            self.mmseqs_executable, "cluster",
            db_path, str(cluster_db_path), tmp_path,
            "--min-seq-id", str(self.mmseqs_config.min_seq_id),
            "--coverage", str(self.mmseqs_config.coverage),
            "--cluster-mode", str(self.mmseqs_config.cluster_mode),
            "-s", str(sensitivity),
            "--threads", str(self.mmseqs_config.threads)
        ]
        
        # Add memory limit if specified
        if self.mmseqs_config.memory_limit:
            cluster_cmd.extend(["--split-memory-limit", self.mmseqs_config.memory_limit])
        
        self.logger.info(f"🔧 Running: mmseqs cluster {db_path} {cluster_db_path} {tmp_path}")
        result = await self._execute_with_retry(cluster_cmd, timeout=600)  # 10 minute timeout
        
        if not result.success:
            raise MMseqs2DataError(f"MMseqs2 clustering failed: {result.stderr_text}")
        
        self.logger.info(f"✅ Clustering completed: {cluster_db_path}")
        return str(cluster_db_path)
    
    async def _mmseqs_createtsv(self, db_path: str, cluster_db_path: str, cache_path: Path, output_prefix: str) -> str:
        """Step 4: mmseqs createtsv - Generate TSV output with cluster assignments"""
        
        # TSV output will be saved in cache directory
        cluster_tsv_path = cache_path / f"{output_prefix}_DB_clu.tsv"
        
        # Command: mmseqs createtsv DB DB DB_clu DB_clu.tsv
        createtsv_cmd = [
            self.mmseqs_executable, "createtsv",
            db_path, db_path, cluster_db_path, str(cluster_tsv_path)
        ]
        
        self.logger.info(f"🔧 Running: mmseqs createtsv {db_path} {db_path} {cluster_db_path} {cluster_tsv_path}")
        result = await self._execute_with_retry(createtsv_cmd)
        
        if not result.success:
            raise MMseqs2DataError(f"Failed to create cluster TSV: {result.stderr_text}")
        
        self.logger.info(f"✅ Created cluster TSV: {cluster_tsv_path}")
        return str(cluster_tsv_path)
    
    async def _mmseqs_create_cluster_fasta(self, db_path: str, cluster_db_path: str, cache_path: Path, output_prefix: str) -> str:
        """Step 5: mmseqs createseqfiledb and result2flat - Generate FASTA output with clustered sequences"""
        
        # Intermediate sequence file database
        cluster_seq_db_path = cache_path / f"{output_prefix}_DB_clu_seq"
        
        # Final FASTA output
        cluster_fasta_path = cache_path / f"{output_prefix}_DB_clu_seq.fasta"
        
        # Step 5a: mmseqs createseqfiledb DB DB_clu DB_clu_seq
        createseqfiledb_cmd = [
            self.mmseqs_executable, "createseqfiledb",
            db_path, cluster_db_path, str(cluster_seq_db_path)
        ]
        
        self.logger.info(f"🔧 Running: mmseqs createseqfiledb {db_path} {cluster_db_path} {cluster_seq_db_path}")
        result = await self._execute_with_retry(createseqfiledb_cmd)
        
        if not result.success:
            raise MMseqs2DataError(f"Failed to create sequence file database: {result.stderr_text}")
        
        # Step 5b: mmseqs result2flat DB DB DB_clu_seq DB_clu_seq.fasta
        result2flat_cmd = [
            self.mmseqs_executable, "result2flat",
            db_path, db_path, str(cluster_seq_db_path), str(cluster_fasta_path)
        ]
        
        self.logger.info(f"🔧 Running: mmseqs result2flat {db_path} {db_path} {cluster_seq_db_path} {cluster_fasta_path}")
        result = await self._execute_with_retry(result2flat_cmd)
        
        if not result.success:
            raise MMseqs2DataError(f"Failed to create cluster FASTA: {result.stderr_text}")
        
        self.logger.info(f"✅ Created cluster FASTA: {cluster_fasta_path}")
        return str(cluster_fasta_path)
    
    async def _parse_mmseqs_tsv_results(self, cluster_tsv_path: str, input_sequences: List[Dict[str, str]]) -> List[ClusterResult]:
        """Parse MMseqs2 TSV clustering results"""
        
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
            
            # Convert to ClusterResult objects
            clusters = []
            for i, (rep_id, cluster_data) in enumerate(clusters_dict.items()):
                cluster = ClusterResult(
                    cluster_id=f"cluster_{i+1:03d}",
                    representative_seq=rep_id,
                    member_sequences=cluster_data['members'],
                    cluster_size=len(cluster_data['members'])
                )
                clusters.append(cluster)
            
            self.logger.info(f"✅ Parsed {len(clusters)} clusters from TSV results")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Failed to parse TSV results: {e}")
            raise MMseqs2DataError(f"Failed to parse clustering results: {e}")
    
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
        
        result = await self._execute_with_retry(createtsv_cmd)
        
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
                "conda", "run", "-n", self.mmseqs_config.environment_name,
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
                "conda", "run", "-n", self.mmseqs_config.environment_name,
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
            f"Install via conda: conda create -n {self.mmseqs_config.environment_name} -c bioconda mmseqs2",
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
    
    async def execute_command(self, command: List[str], **kwargs) -> ToolResult:
        """Execute MMseqs2 command with retry logic"""
        return await self._execute_with_retry(command, **kwargs)
    
    async def parse_output(self, raw_output: str, output_type: str = "clustering") -> Any:
        """Parse MMseqs2 clustering output"""
        if output_type == "clustering":
            # Parse clustering results from raw output
            return {"clusters": [], "total_sequences": 0}
        return {"raw_output": raw_output}
    
    async def verify_installation(self) -> bool:
        """Verify MMseqs2 installation"""
        try:
            result = await self.execute_command(["mmseqs", "version"])
            if result.success:
                version = result.stdout_text.strip()
                self.logger.info(f"MMseqs2 version detected: {version}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"MMseqs2 verification failed: {e}")
            return False 