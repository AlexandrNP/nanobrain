"""
Workflow Configuration Management

Handles loading and validation of Alphavirus workflow configurations
from YAML files with proper type checking and defaults.
"""

import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class BVBRCConfig:
    """BV-BRC tool configuration"""
    tool_name: str = "bv_brc"
    installation_path: str = "/Applications/BV-BRC.app/"
    executable_path: str = "/Applications/BV-BRC.app/deployment/bin/"
    genus: str = "Alphavirus"
    min_length: int = 8000
    max_length: int = 15000
    exclude_incomplete: bool = True
    genome_batch: int = 100
    md5_batch: int = 50
    timeout_seconds: int = 300
    retry_attempts: int = 3
    verify_on_init: bool = False


@dataclass
class ClusteringConfig:
    """MMseqs2 clustering configuration"""
    min_seq_id: float = 0.7
    coverage: float = 0.8
    sensitivity: float = 7.5
    coverage_mode: int = 1
    cluster_mode: int = 0
    prefer_short_conserved: bool = True
    min_cluster_size: int = 3
    max_cluster_size: int = 1000


@dataclass
class AlignmentConfig:
    """MUSCLE alignment configuration"""
    max_iterations: int = 16
    gap_open_penalty: int = -12
    gap_extend_penalty: int = -1
    diagonal_optimization: bool = True


@dataclass
class QualityControlConfig:
    """Quality control and validation configuration"""
    min_alignment_conservation: float = 0.5
    max_ambiguous_aa_percent: int = 5
    min_sequence_length: int = 10
    max_sequence_length: int = 10000
    expected_lengths: Dict[str, List[int]] = field(default_factory=lambda: {
        "nsP1": [500, 600],
        "nsP2": [750, 850],
        "nsP3": [480, 580],
        "nsP4": [560, 660],
        "capsid": [200, 320],
        "E3": [40, 80],
        "E2": [370, 470],
        "6K": [40, 70],
        "E1": [390, 490]
    })


@dataclass
class OutputConfig:
    """Output configuration"""
    base_directory: str = "data/alphavirus_analysis"
    create_html_report: bool = True
    create_summary_plots: bool = True
    save_intermediate_files: bool = True
    output_files: Dict[str, str] = field(default_factory=lambda: {
        "filtered_genomes": "alphavirus_filtered_genomes.json",
        "unique_proteins": "alphavirus_unique_proteins.fasta",
        "clusters": "alphavirus_clusters.json",
        "pssm_matrices": "alphavirus_pssm_matrices.json",
        "curation_report": "alphavirus_curation_report.json",
        "viral_pssm_json": "alphavirus_viral_pssm.json"
    })


@dataclass
class ResourceConfig:
    """Resource monitoring configuration"""
    disk_warning_gb: float = 1.0
    disk_critical_gb: float = 0.5
    memory_limit_gb: float = 8.0


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "logs/alphavirus_workflow.log"
    include_timestamps: bool = True
    include_step_timing: bool = True


@dataclass
class AlphavirusWorkflowConfig:
    """Main workflow configuration container"""
    name: str = "alphavirus_analysis"
    version: str = "1.0.0"
    description: str = "Comprehensive Alphavirus protein analysis using BV-BRC and MMseqs2"
    
    # Step/Workflow configuration attributes
    debug_mode: bool = False
    enable_logging: bool = True
    log_data_transfers: bool = True
    log_executions: bool = True
    auto_initialize: bool = True
    
    # Sub-configurations
    bvbrc: BVBRCConfig = field(default_factory=BVBRCConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    quality_control: QualityControlConfig = field(default_factory=QualityControlConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AlphavirusWorkflowConfig':
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AlphavirusWorkflowConfig':
        """Create configuration from dictionary"""
        
        # Extract workflow section
        workflow_config = config_dict.get('workflow', {})
        
        # Create sub-configurations
        bvbrc_data = workflow_config.get('bvbrc', {})
        # Flatten nested structures for easier access
        if 'genome_filters' in bvbrc_data:
            bvbrc_data.update(bvbrc_data.pop('genome_filters'))
        if 'batch_sizes' in bvbrc_data:
            batch_sizes = bvbrc_data.pop('batch_sizes')
            bvbrc_data['genome_batch'] = batch_sizes.get('genome_batch', 100)
            bvbrc_data['md5_batch'] = batch_sizes.get('md5_batch', 50)
        
        return cls(
            name=workflow_config.get('name', 'alphavirus_analysis'),
            version=workflow_config.get('version', '1.0.0'),
            description=workflow_config.get('description', ''),
            bvbrc=BVBRCConfig(**bvbrc_data),
            clustering=ClusteringConfig(**workflow_config.get('clustering', {})),
            alignment=AlignmentConfig(**workflow_config.get('alignment', {})),
            quality_control=QualityControlConfig(**workflow_config.get('quality_control', {})),
            output=OutputConfig(**workflow_config.get('output', {})),
            resources=ResourceConfig(**workflow_config.get('resources', {})),
            logging=LoggingConfig(**workflow_config.get('logging', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'workflow': {
                'name': self.name,
                'version': self.version,
                'description': self.description,
                'bvbrc': self.bvbrc.__dict__,
                'clustering': self.clustering.__dict__,
                'alignment': self.alignment.__dict__,
                'quality_control': self.quality_control.__dict__,
                'output': self.output.__dict__,
                'resources': self.resources.__dict__,
                'logging': self.logging.__dict__
            }
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate BV-BRC path
        bvbrc_path = Path(self.bvbrc.installation_path)
        if not bvbrc_path.exists():
            issues.append(f"BV-BRC installation path does not exist: {self.bvbrc.installation_path}")
        
        # Validate executable path
        exec_path = Path(self.bvbrc.executable_path)
        if not exec_path.exists():
            issues.append(f"BV-BRC executable path does not exist: {self.bvbrc.executable_path}")
        
        # Validate clustering parameters
        if not (0.0 <= self.clustering.min_seq_id <= 1.0):
            issues.append("Clustering min_seq_id must be between 0.0 and 1.0")
        
        if not (0.0 <= self.clustering.coverage <= 1.0):
            issues.append("Clustering coverage must be between 0.0 and 1.0")
        
        # Validate output directory
        try:
            output_dir = Path(self.output.base_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create output directory: {e}")
        
        return issues


def load_default_config() -> AlphavirusWorkflowConfig:
    """Load default configuration"""
    config_path = Path(__file__).parent / "AlphavirusWorkflow.yml"
    
    if config_path.exists():
        return AlphavirusWorkflowConfig.from_file(str(config_path))
    else:
        # Return default configuration
        return AlphavirusWorkflowConfig()


def validate_config_file(config_path: str) -> List[str]:
    """Validate a configuration file and return issues"""
    try:
        config = AlphavirusWorkflowConfig.from_file(config_path)
        return config.validate()
    except Exception as e:
        return [f"Failed to load configuration: {e}"] 