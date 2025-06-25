"""
Bioinformatics Extensions for NanoBrain Framework

Provides specialized bioinformatics functionality including sequence management,
coordinate systems, and external tool integration.
"""

import asyncio
import logging
import os
import tempfile
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union, Tuple
from enum import Enum
from pathlib import Path
import json
from pydantic import BaseModel, Field, ConfigDict, field_validator

from .step import Step, StepConfig
from .agent import Agent, AgentConfig
from .tool import ToolBase, ToolConfig, ToolType
from .data_unit import DataUnitBase, DataUnitConfig
from .logging_system import get_logger, OperationType

logger = logging.getLogger(__name__)


class CoordinateSystem(Enum):
    """Coordinate system types for biological sequences."""
    ZERO_BASED = "0-based"  # Computational standard (0-based, half-open)
    ONE_BASED = "1-based"   # Biological standard (1-based, closed)


class SequenceType(Enum):
    """Types of biological sequences."""
    DNA = "dna"
    RNA = "rna"
    PROTEIN = "protein"
    NUCLEOTIDE = "nucleotide"


class BioinformaticsConfig(BaseModel):
    """Configuration for bioinformatics components."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    coordinate_system: CoordinateSystem = CoordinateSystem.ONE_BASED
    sequence_type: SequenceType = SequenceType.DNA
    temp_dir: Optional[str] = None
    cleanup_temp_files: bool = True
    external_tools_timeout: float = 300.0  # 5 minutes default


class SequenceCoordinate(BaseModel):
    """Represents a sequence coordinate with proper system handling."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    start: int
    end: int
    coordinate_system: CoordinateSystem = CoordinateSystem.ONE_BASED
    strand: Optional[str] = None  # '+', '-', or None
    
    @field_validator('start', 'end')
    @classmethod
    def validate_coordinates(cls, v):
        """Validate coordinate values."""
        if v < 0:
            raise ValueError("Coordinates cannot be negative")
        return v
    
    @field_validator('end')
    @classmethod
    def validate_end_after_start(cls, v, info):
        """Ensure end coordinate is after start."""
        if info.data and 'start' in info.data and v <= info.data['start']:
            raise ValueError("End coordinate must be greater than start coordinate")
        return v
    
    def to_zero_based(self) -> 'SequenceCoordinate':
        """Convert to 0-based coordinate system."""
        if self.coordinate_system == CoordinateSystem.ZERO_BASED:
            return self
        
        return SequenceCoordinate(
            start=self.start - 1,
            end=self.end,  # Keep end as-is for half-open interval
            coordinate_system=CoordinateSystem.ZERO_BASED,
            strand=self.strand
        )
    
    def to_one_based(self) -> 'SequenceCoordinate':
        """Convert to 1-based coordinate system."""
        if self.coordinate_system == CoordinateSystem.ONE_BASED:
            return self
        
        return SequenceCoordinate(
            start=self.start + 1,
            end=self.end,  # Keep end as-is for closed interval
            coordinate_system=CoordinateSystem.ONE_BASED,
            strand=self.strand
        )
    
    def length(self) -> int:
        """Calculate the length of the coordinate span."""
        if self.coordinate_system == CoordinateSystem.ZERO_BASED:
            return self.end - self.start
        else:  # ONE_BASED
            return self.end - self.start + 1
    
    def overlaps(self, other: 'SequenceCoordinate') -> bool:
        """Check if this coordinate overlaps with another."""
        # Convert both to same coordinate system for comparison
        self_zero = self.to_zero_based()
        other_zero = other.to_zero_based()
        
        return not (self_zero.end <= other_zero.start or other_zero.end <= self_zero.start)
    
    def contains(self, other: 'SequenceCoordinate') -> bool:
        """Check if this coordinate contains another."""
        self_zero = self.to_zero_based()
        other_zero = other.to_zero_based()
        
        return self_zero.start <= other_zero.start and other_zero.end <= self_zero.end
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'start': self.start,
            'end': self.end,
            'coordinate_system': self.coordinate_system.value,
            'strand': self.strand,
            'length': self.length()
        }


class SequenceRegion(BaseModel):
    """Represents a biological sequence region with metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    sequence_id: str
    coordinates: SequenceCoordinate
    sequence_type: SequenceType
    sequence_data: Optional[str] = None
    annotations: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: Optional[float] = None
    
    def get_fasta_header(self) -> str:
        """Generate FASTA header for this region."""
        coord_str = f"{self.coordinates.start}-{self.coordinates.end}"
        if self.coordinates.strand:
            coord_str += f"({self.coordinates.strand})"
        
        return f">{self.sequence_id}_{coord_str}"
    
    def to_fasta(self) -> str:
        """Convert to FASTA format."""
        if not self.sequence_data:
            raise ValueError("No sequence data available for FASTA conversion")
        
        header = self.get_fasta_header()
        return f"{header}\n{self.sequence_data}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'sequence_id': self.sequence_id,
            'coordinates': self.coordinates.to_dict(),
            'sequence_type': self.sequence_type.value,
            'sequence_data': self.sequence_data,
            'annotations': self.annotations,
            'confidence_score': self.confidence_score
        }


class BioinformaticsDataUnit(DataUnitBase):
    """Specialized data unit for bioinformatics data."""
    
    def __init__(self, config: DataUnitConfig, bio_config: Optional[BioinformaticsConfig] = None):
        super().__init__(config)
        self.bio_config = bio_config or BioinformaticsConfig()
        self.nb_logger = get_logger(f"biodata.{config.name}")
        self._sequence_regions: List[SequenceRegion] = []
    
    async def get(self) -> List[SequenceRegion]:
        """Get stored sequence regions."""
        return self._sequence_regions.copy()
    
    async def set(self, data: Any) -> None:
        """Set sequence regions data."""
        if isinstance(data, list) and all(isinstance(item, SequenceRegion) for item in data):
            self._sequence_regions = data.copy()
        else:
            raise ValueError("BioinformaticsDataUnit only accepts List[SequenceRegion]")
    
    async def clear(self) -> None:
        """Clear stored sequence regions."""
        self._sequence_regions.clear()
        
    async def store_sequence_regions(self, regions: List[SequenceRegion]) -> None:
        """Store sequence regions with proper coordinate handling."""
        async with self.nb_logger.async_execution_context(
            OperationType.DATA_TRANSFER,
            f"store_sequence_regions"
        ) as context:
            
            # Convert regions to standardized format
            standardized_regions = []
            for region in regions:
                # Ensure coordinates match our coordinate system
                if region.coordinates.coordinate_system != self.bio_config.coordinate_system:
                    if self.bio_config.coordinate_system == CoordinateSystem.ONE_BASED:
                        region.coordinates = region.coordinates.to_one_based()
                    else:
                        region.coordinates = region.coordinates.to_zero_based()
                
                standardized_regions.append(region)
            
            await self.set(standardized_regions)
            context.metadata['regions_count'] = len(regions)
            context.metadata['coordinate_system'] = self.bio_config.coordinate_system.value
    
    async def retrieve_sequence_regions(self) -> List[SequenceRegion]:
        """Retrieve sequence regions from storage."""
        data = await self.get()
        if not data:
            return []
        
        # If data is already SequenceRegion objects, return directly
        if all(isinstance(item, SequenceRegion) for item in data):
            return data
        
        # Otherwise, reconstruct from dictionaries
        regions = []
        for region_dict in data:
            # Reconstruct coordinate object
            coord_data = region_dict['coordinates']
            coordinates = SequenceCoordinate(
                start=coord_data['start'],
                end=coord_data['end'],
                coordinate_system=CoordinateSystem(coord_data['coordinate_system']),
                strand=coord_data.get('strand')
            )
            
            # Reconstruct region object
            region = SequenceRegion(
                sequence_id=region_dict['sequence_id'],
                coordinates=coordinates,
                sequence_type=SequenceType(region_dict['sequence_type']),
                sequence_data=region_dict.get('sequence_data'),
                annotations=region_dict.get('annotations', {}),
                confidence_score=region_dict.get('confidence_score')
            )
            
            regions.append(region)
        
        return regions


class ExternalToolManager:
    """Manages external bioinformatics tools and their execution."""
    
    def __init__(self, config: BioinformaticsConfig):
        self.config = config
        self.nb_logger = get_logger("external_tools")
        self.temp_dir = Path(config.temp_dir) if config.temp_dir else Path(tempfile.gettempdir()) / "nanobrain_bio"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Tool environment configurations
        self.tool_environments = {}
        
    async def setup_conda_environment(self, env_name: str, packages: List[str]) -> bool:
        """Set up a conda environment for a specific tool."""
        async with self.nb_logger.async_execution_context(
            OperationType.EXECUTOR_RUN,
            f"setup_conda_env_{env_name}"
        ) as context:
            
            try:
                # Check if conda is available
                result = await asyncio.create_subprocess_exec(
                    'conda', '--version',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await result.wait()
                
                if result.returncode != 0:
                    raise RuntimeError("Conda not available")
                
                # Create environment
                cmd = ['conda', 'create', '-n', env_name, '-y'] + packages
                result = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await result.communicate()
                
                if result.returncode == 0:
                    self.tool_environments[env_name] = {
                        'type': 'conda',
                        'name': env_name,
                        'packages': packages,
                        'active': True
                    }
                    context.metadata['env_name'] = env_name
                    context.metadata['packages'] = packages
                    return True
                else:
                    self.nb_logger.error(f"Failed to create conda environment {env_name}: {stderr.decode()}")
                    return False
                    
            except Exception as e:
                self.nb_logger.error(f"Error setting up conda environment {env_name}: {e}")
                return False
    
    async def execute_tool(self, tool_name: str, command: List[str], 
                          input_files: Optional[List[str]] = None,
                          output_files: Optional[List[str]] = None,
                          env_name: Optional[str] = None) -> Dict[str, Any]:
        """Execute an external bioinformatics tool."""
        async with self.nb_logger.async_execution_context(
            OperationType.EXECUTOR_RUN,
            f"execute_{tool_name}"
        ) as context:
            
            try:
                # Prepare command with environment activation if specified
                if env_name and env_name in self.tool_environments:
                    env_info = self.tool_environments[env_name]
                    if env_info['type'] == 'conda':
                        # Activate conda environment
                        full_command = ['conda', 'run', '-n', env_name] + command
                    else:
                        full_command = command
                else:
                    full_command = command
                
                # Execute command
                result = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *full_command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=self.temp_dir
                    ),
                    timeout=self.config.external_tools_timeout
                )
                
                stdout, stderr = await result.communicate()
                
                # Prepare result
                execution_result = {
                    'tool_name': tool_name,
                    'command': ' '.join(full_command),
                    'return_code': result.returncode,
                    'stdout': stdout.decode(),
                    'stderr': stderr.decode(),
                    'success': result.returncode == 0,
                    'input_files': input_files or [],
                    'output_files': output_files or []
                }
                
                # Check output files exist
                if output_files:
                    for output_file in output_files:
                        output_path = self.temp_dir / output_file
                        execution_result[f'{output_file}_exists'] = output_path.exists()
                        if output_path.exists():
                            execution_result[f'{output_file}_size'] = output_path.stat().st_size
                
                context.metadata.update({
                    'tool_name': tool_name,
                    'return_code': result.returncode,
                    'success': result.returncode == 0,
                    'env_name': env_name
                })
                
                return execution_result
                
            except asyncio.TimeoutError:
                self.nb_logger.error(f"Tool {tool_name} execution timed out")
                return {
                    'tool_name': tool_name,
                    'success': False,
                    'error': 'Execution timeout',
                    'return_code': -1
                }
            except Exception as e:
                self.nb_logger.error(f"Error executing tool {tool_name}: {e}")
                return {
                    'tool_name': tool_name,
                    'success': False,
                    'error': str(e),
                    'return_code': -1
                }
    
    def create_temp_file(self, content: str, suffix: str = ".tmp") -> str:
        """Create a temporary file with given content."""
        temp_file = self.temp_dir / f"temp_{asyncio.get_event_loop().time()}{suffix}"
        temp_file.write_text(content)
        return str(temp_file)
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        if self.config.cleanup_temp_files:
            try:
                for temp_file in self.temp_dir.glob("temp_*"):
                    temp_file.unlink()
                self.nb_logger.debug("Cleaned up temporary files")
            except Exception as e:
                self.nb_logger.error(f"Error cleaning up temp files: {e}")


class BioinformaticsStep(Step):
    """Base class for bioinformatics-specific steps."""
    
    def __init__(self, config: StepConfig, bio_config: Optional[BioinformaticsConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.bio_config = bio_config or BioinformaticsConfig()
        self.tool_manager = ExternalToolManager(self.bio_config)
        
    async def initialize(self) -> None:
        """Initialize bioinformatics step."""
        await super().initialize()
        # Additional bioinformatics-specific initialization can go here
        
    async def process_sequences(self, sequences: List[SequenceRegion]) -> List[SequenceRegion]:
        """Process sequence regions - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement process_sequences")
    
    def standardize_coordinates(self, regions: List[SequenceRegion]) -> List[SequenceRegion]:
        """Standardize coordinate systems across regions."""
        standardized = []
        for region in regions:
            if region.coordinates.coordinate_system != self.bio_config.coordinate_system:
                if self.bio_config.coordinate_system == CoordinateSystem.ONE_BASED:
                    region.coordinates = region.coordinates.to_one_based()
                else:
                    region.coordinates = region.coordinates.to_zero_based()
            standardized.append(region)
        return standardized


class BioinformaticsAgent(Agent):
    """Base class for bioinformatics-specific agents."""
    
    def __init__(self, config: AgentConfig, bio_config: Optional[BioinformaticsConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.bio_config = bio_config or BioinformaticsConfig()
        self.tool_manager = ExternalToolManager(self.bio_config)
        
    async def initialize(self) -> None:
        """Initialize bioinformatics agent."""
        await super().initialize()
        # Load bioinformatics-specific system prompts if available
        await self._load_bioinformatics_prompts()
        
    async def _load_bioinformatics_prompts(self) -> None:
        """Load bioinformatics-specific system prompts."""
        # This will be implemented when we create the prompt templates
        pass
    
    async def analyze_sequences(self, sequences: List[SequenceRegion]) -> Dict[str, Any]:
        """Analyze sequence regions - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement analyze_sequences")


class BioinformaticsTool(ToolBase):
    """
    Base class for bioinformatics-specific tools.
    Enhanced with mandatory from_config pattern implementation.
    """
    
    @classmethod
    def from_config(cls, config: ToolConfig, **kwargs) -> 'BioinformaticsTool':
        """Mandatory from_config implementation for BioinformaticsTool"""
        from .logging_system import get_logger
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Validate configuration schema
        cls.validate_config_schema(config)
        
        # Step 2: Extract component-specific configuration  
        component_config = cls.extract_component_config(config)
        
        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 4: Create instance
        instance = cls.create_instance(config, component_config, dependencies)
        
        # Step 5: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
    
    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve BioinformaticsTool dependencies"""
        bio_config = kwargs.get('bio_config')
        if bio_config is None:
            bio_config = BioinformaticsConfig()
        
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        return {
            **base_deps,
            'bio_config': bio_config
        }
    
    def _init_from_config(self, config: ToolConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize BioinformaticsTool with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.bio_config = dependencies['bio_config']
        self.tool_manager = ExternalToolManager(self.bio_config)
    
    # BioinformaticsTool inherits FromConfigBase.__init__ which prevents direct instantiation
        
    async def initialize(self) -> None:
        """Initialize bioinformatics tool."""
        await super().initialize()
        # Set up any required external tools or environments
        await self._setup_external_dependencies()
        
    async def _setup_external_dependencies(self) -> None:
        """Set up external tool dependencies - to be implemented by subclasses."""
        pass
    
    async def process_biological_data(self, data: Any) -> Any:
        """Process biological data - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement process_biological_data")


# Factory functions for creating bioinformatics components
def create_bioinformatics_data_unit(name: str, bio_config: Optional[BioinformaticsConfig] = None) -> BioinformaticsDataUnit:
    """Create a bioinformatics data unit."""
    from .data_unit import DataUnitType
    config = DataUnitConfig(name=name, data_type=DataUnitType.MEMORY)
    return BioinformaticsDataUnit(config, bio_config)


def create_sequence_coordinate(start: int, end: int, 
                             coordinate_system: CoordinateSystem = CoordinateSystem.ONE_BASED,
                             strand: Optional[str] = None) -> SequenceCoordinate:
    """Create a sequence coordinate with validation."""
    return SequenceCoordinate(
        start=start,
        end=end,
        coordinate_system=coordinate_system,
        strand=strand
    )


def create_sequence_region(sequence_id: str, start: int, end: int,
                          sequence_type: SequenceType = SequenceType.DNA,
                          coordinate_system: CoordinateSystem = CoordinateSystem.ONE_BASED,
                          sequence_data: Optional[str] = None,
                          **kwargs) -> SequenceRegion:
    """Create a sequence region with proper coordinate handling."""
    coordinates = create_sequence_coordinate(start, end, coordinate_system)
    return SequenceRegion(
        sequence_id=sequence_id,
        coordinates=coordinates,
        sequence_type=sequence_type,
        sequence_data=sequence_data,
        **kwargs
    )