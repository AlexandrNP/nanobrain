"""
Tests for NanoBrain Bioinformatics Core Framework Extensions
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from nanobrain.core.bioinformatics import (
    BioinformaticsConfig, CoordinateSystem, SequenceType,
    SequenceCoordinate, SequenceRegion, BioinformaticsDataUnit,
    ExternalToolManager, BioinformaticsStep, BioinformaticsAgent, BioinformaticsTool,
    create_bioinformatics_data_unit, create_sequence_coordinate, create_sequence_region
)

from nanobrain.core.sequence_manager import (
    SequenceManager, SequenceValidator, FastaParser, SequenceFormat,
    SequenceStats, SequenceValidationError,
    create_sequence_manager, create_fasta_parser, create_sequence_validator
)

from nanobrain.core import StepConfig, AgentConfig, ToolConfig, DataUnitConfig


class TestCoordinateSystem:
    """Test coordinate system handling."""
    
    def test_coordinate_creation(self):
        """Test creating coordinates in different systems."""
        # 1-based coordinate
        coord_1based = SequenceCoordinate(start=100, end=200, coordinate_system=CoordinateSystem.ONE_BASED)
        assert coord_1based.start == 100
        assert coord_1based.end == 200
        assert coord_1based.coordinate_system == CoordinateSystem.ONE_BASED
        assert coord_1based.length() == 101  # 1-based inclusive
        
        # 0-based coordinate
        coord_0based = SequenceCoordinate(start=99, end=200, coordinate_system=CoordinateSystem.ZERO_BASED)
        assert coord_0based.start == 99
        assert coord_0based.end == 200
        assert coord_0based.coordinate_system == CoordinateSystem.ZERO_BASED
        assert coord_0based.length() == 101  # 0-based half-open
    
    def test_coordinate_conversion(self):
        """Test coordinate system conversion."""
        # Convert 1-based to 0-based
        coord_1based = SequenceCoordinate(start=100, end=200, coordinate_system=CoordinateSystem.ONE_BASED)
        coord_0based = coord_1based.to_zero_based()
        
        assert coord_0based.start == 99
        assert coord_0based.end == 200
        assert coord_0based.coordinate_system == CoordinateSystem.ZERO_BASED
        
        # Convert back to 1-based
        coord_back = coord_0based.to_one_based()
        assert coord_back.start == 100
        assert coord_back.end == 200
        assert coord_back.coordinate_system == CoordinateSystem.ONE_BASED
    
    def test_coordinate_operations(self):
        """Test coordinate overlap and containment."""
        coord1 = SequenceCoordinate(start=100, end=200)
        coord2 = SequenceCoordinate(start=150, end=250)
        coord3 = SequenceCoordinate(start=120, end=180)
        coord4 = SequenceCoordinate(start=300, end=400)
        
        # Test overlap
        assert coord1.overlaps(coord2)
        assert coord2.overlaps(coord1)
        assert not coord1.overlaps(coord4)
        
        # Test containment
        assert coord1.contains(coord3)
        assert not coord3.contains(coord1)
        assert not coord1.contains(coord2)
    
    def test_coordinate_validation(self):
        """Test coordinate validation."""
        # Valid coordinate
        coord = SequenceCoordinate(start=100, end=200)
        assert coord.start == 100
        
        # Invalid coordinates should raise validation error
        with pytest.raises(ValueError):
            SequenceCoordinate(start=-1, end=100)  # Negative start
        
        with pytest.raises(ValueError):
            SequenceCoordinate(start=200, end=100)  # End before start


class TestSequenceRegion:
    """Test sequence region functionality."""
    
    def test_sequence_region_creation(self):
        """Test creating sequence regions."""
        coordinates = SequenceCoordinate(start=1, end=100)
        region = SequenceRegion(
            sequence_id="test_seq",
            coordinates=coordinates,
            sequence_type=SequenceType.DNA,
            sequence_data="ATCGATCGATCG"
        )
        
        assert region.sequence_id == "test_seq"
        assert region.coordinates.start == 1
        assert region.sequence_type == SequenceType.DNA
        assert region.sequence_data == "ATCGATCGATCG"
    
    def test_fasta_conversion(self):
        """Test FASTA format conversion."""
        coordinates = SequenceCoordinate(start=1, end=12, strand="+")
        region = SequenceRegion(
            sequence_id="test_seq",
            coordinates=coordinates,
            sequence_type=SequenceType.DNA,
            sequence_data="ATCGATCGATCG"
        )
        
        fasta = region.to_fasta()
        expected = ">test_seq_1-12(+)\nATCGATCGATCG"
        assert fasta == expected
        
        header = region.get_fasta_header()
        assert header == ">test_seq_1-12(+)"
    
    def test_sequence_region_dict_conversion(self):
        """Test dictionary conversion."""
        coordinates = SequenceCoordinate(start=1, end=12)
        region = SequenceRegion(
            sequence_id="test_seq",
            coordinates=coordinates,
            sequence_type=SequenceType.DNA,
            sequence_data="ATCGATCGATCG",
            annotations={"gene": "test_gene"},
            confidence_score=0.95
        )
        
        region_dict = region.to_dict()
        assert region_dict["sequence_id"] == "test_seq"
        assert region_dict["sequence_type"] == "dna"
        assert region_dict["annotations"]["gene"] == "test_gene"
        assert region_dict["confidence_score"] == 0.95


class TestSequenceValidator:
    """Test sequence validation."""
    
    def test_dna_validation(self):
        """Test DNA sequence validation."""
        validator = SequenceValidator(strict=True)
        
        # Valid DNA
        is_valid, errors = validator.validate_sequence("ATCGATCG", SequenceType.DNA)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid DNA (contains U)
        is_valid, errors = validator.validate_sequence("ATCGAUCG", SequenceType.DNA)
        assert not is_valid
        assert len(errors) > 0
    
    def test_rna_validation(self):
        """Test RNA sequence validation."""
        validator = SequenceValidator(strict=True)
        
        # Valid RNA
        is_valid, errors = validator.validate_sequence("AUCGAUCG", SequenceType.RNA)
        assert is_valid
        
        # Invalid RNA (contains T)
        is_valid, errors = validator.validate_sequence("ATCGATCG", SequenceType.RNA)
        assert not is_valid
    
    def test_protein_validation(self):
        """Test protein sequence validation."""
        validator = SequenceValidator()
        
        # Valid protein
        is_valid, errors = validator.validate_sequence("MKLLVVVAG", SequenceType.PROTEIN)
        assert is_valid
        
        # Invalid protein (contains nucleotides not found in protein alphabet)
        is_valid, errors = validator.validate_sequence("BJOUZJOU", SequenceType.PROTEIN)
        assert not is_valid
    
    def test_sequence_stats(self):
        """Test sequence statistics calculation."""
        validator = SequenceValidator()
        
        dna_seq = "ATCGATCGATCG"
        stats = validator.calculate_stats(dna_seq, SequenceType.DNA)
        
        assert stats.length == 12
        assert stats.gc_content == 0.5  # 6 GC out of 12
        assert stats.n_count == 0
        assert stats.valid_bases == 12
        assert stats.composition["A"] == 3
        assert stats.composition["T"] == 3
        assert stats.composition["C"] == 3
        assert stats.composition["G"] == 3


class TestFastaParser:
    """Test FASTA parsing functionality."""
    
    def test_parse_fasta_string(self):
        """Test parsing FASTA from string."""
        parser = FastaParser()
        
        fasta_content = """>seq1
ATCGATCGATCG
>seq2
GCTAGCTAGCTA"""
        
        sequences = parser.parse_fasta_string(fasta_content)
        assert len(sequences) == 2
        assert sequences[0] == ("seq1", "ATCGATCGATCG")
        assert sequences[1] == ("seq2", "GCTAGCTAGCTA")
    
    def test_write_fasta_string(self):
        """Test writing FASTA to string."""
        parser = FastaParser()
        
        sequences = [("seq1", "ATCGATCGATCG"), ("seq2", "GCTAGCTAGCTA")]
        fasta_content = parser.write_fasta_string(sequences, line_width=6)
        
        expected = """>seq1
ATCGAT
CGATCG
>seq2
GCTAGC
TAGCTA"""
        assert fasta_content == expected
    
    @pytest.mark.asyncio
    async def test_parse_fasta_file(self):
        """Test parsing FASTA from file."""
        parser = FastaParser()
        
        # Create temporary FASTA file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(""">seq1
ATCGATCGATCG
>seq2
GCTAGCTAGCTA""")
            temp_file = f.name
        
        try:
            sequences = await parser.parse_fasta_file(temp_file)
            assert len(sequences) == 2
            assert sequences[0] == ("seq1", "ATCGATCGATCG")
            assert sequences[1] == ("seq2", "GCTAGCTAGCTA")
        finally:
            os.unlink(temp_file)


class TestSequenceManager:
    """Test sequence manager functionality."""
    
    @pytest.mark.asyncio
    async def test_load_sequences_from_fasta(self):
        """Test loading sequences from FASTA file."""
        manager = SequenceManager()
        
        # Create temporary FASTA file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(""">seq1_100-200
ATCGATCGATCG
>seq2
GCTAGCTAGCTA""")
            temp_file = f.name
        
        try:
            regions = await manager.load_sequences_from_fasta(temp_file, SequenceType.DNA)
            assert len(regions) == 2
            
            # First sequence should have parsed coordinates
            assert regions[0].sequence_id == "seq1"
            assert regions[0].coordinates.start == 100
            assert regions[0].coordinates.end == 200
            
            # Second sequence should have default coordinates
            assert regions[1].sequence_id == "seq2"
            assert regions[1].coordinates.start == 1
            assert regions[1].coordinates.end == 12
            
        finally:
            os.unlink(temp_file)
    
    def test_extract_subsequence(self):
        """Test subsequence extraction."""
        manager = SequenceManager()
        
        coordinates = SequenceCoordinate(start=1, end=20)
        region = SequenceRegion(
            sequence_id="test_seq",
            coordinates=coordinates,
            sequence_type=SequenceType.DNA,
            sequence_data="ATCGATCGATCGATCGATCG"
        )
        
        subregion = manager.extract_subsequence(region, 5, 10)
        assert subregion.sequence_data == "ATCGAT"  # 1-based extraction: positions 4-9 (0-based)
        assert subregion.coordinates.start == 5
        assert subregion.coordinates.end == 10
    
    def test_merge_regions(self):
        """Test merging sequence regions."""
        manager = SequenceManager()
        
        coord1 = SequenceCoordinate(start=1, end=10)
        region1 = SequenceRegion(
            sequence_id="seq1",
            coordinates=coord1,
            sequence_type=SequenceType.DNA,
            sequence_data="ATCGATCGAT"
        )
        
        coord2 = SequenceCoordinate(start=11, end=20)
        region2 = SequenceRegion(
            sequence_id="seq2",
            coordinates=coord2,
            sequence_type=SequenceType.DNA,
            sequence_data="GCTAGCTAGC"
        )
        
        merged = manager.merge_regions([region1, region2], "merged_seq")
        assert merged.sequence_id == "merged_seq"
        assert merged.sequence_data == "ATCGATCGATGCTAGCTAGC"
        assert merged.coordinates.start == 1
        assert merged.coordinates.end == 20
    
    def test_find_orfs(self):
        """Test ORF finding."""
        manager = SequenceManager()
        
        # Create sequence with ORF (ATG...TAA)
        coordinates = SequenceCoordinate(start=1, end=30)
        region = SequenceRegion(
            sequence_id="test_seq",
            coordinates=coordinates,
            sequence_type=SequenceType.DNA,
            sequence_data="ATGAAAAAAAAAAAAAAAAAAAAATAA"  # ATG + 21 A's + TAA = 27 bp
        )
        
        orfs = manager.find_orfs(region, min_length=20)
        assert len(orfs) == 1
        assert orfs[0].annotations["start_codon"] == "ATG"
        assert orfs[0].annotations["stop_codon"] == "TAA"
        assert orfs[0].annotations["strand"] == "+"
    
    def test_translate_sequence(self):
        """Test sequence translation."""
        manager = SequenceManager()
        
        coordinates = SequenceCoordinate(start=1, end=9)
        region = SequenceRegion(
            sequence_id="test_seq",
            coordinates=coordinates,
            sequence_type=SequenceType.DNA,
            sequence_data="ATGAAATAA"  # ATG AAA TAA = M K *
        )
        
        protein = manager.translate_sequence(region)
        assert protein.sequence_data == "MK*"
        assert protein.sequence_type == SequenceType.PROTEIN
        assert protein.annotations["translated_from"] == "test_seq"


class TestExternalToolManager:
    """Test external tool management."""
    
    def test_external_tool_manager_creation(self):
        """Test creating external tool manager."""
        config = BioinformaticsConfig()
        manager = ExternalToolManager(config)
        
        assert manager.config == config
        assert manager.temp_dir.exists()
    
    def test_create_temp_file(self):
        """Test temporary file creation."""
        config = BioinformaticsConfig()
        manager = ExternalToolManager(config)
        
        content = "test content"
        temp_file = manager.create_temp_file(content, ".txt")
        
        assert os.path.exists(temp_file)
        assert Path(temp_file).read_text() == content
        
        # Cleanup
        manager.cleanup_temp_files()
    
    @pytest.mark.asyncio
    async def test_setup_conda_environment(self):
        """Test conda environment setup."""
        config = BioinformaticsConfig()
        manager = ExternalToolManager(config)
        
        # Mock conda command execution
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock conda --version check
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.wait.return_value = None
            mock_process.communicate.return_value = (b"conda 4.10.0", b"")
            mock_subprocess.return_value = mock_process
            
            result = await manager.setup_conda_environment("test_env", ["python=3.8"])
            
            # Should succeed with mocked conda
            assert result is True
            assert "test_env" in manager.tool_environments


class TestBioinformaticsDataUnit:
    """Test bioinformatics data unit."""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_regions(self):
        """Test storing and retrieving sequence regions."""
        from nanobrain.core.data_unit import DataUnitType
        config = DataUnitConfig(name="test_bio_data", data_type=DataUnitType.MEMORY)
        bio_config = BioinformaticsConfig()
        data_unit = BioinformaticsDataUnit(config, bio_config)
        
        await data_unit.initialize()
        
        # Create test regions
        coord1 = SequenceCoordinate(start=1, end=10)
        region1 = SequenceRegion(
            sequence_id="seq1",
            coordinates=coord1,
            sequence_type=SequenceType.DNA,
            sequence_data="ATCGATCGAT"
        )
        
        regions = [region1]
        
        # Store regions
        await data_unit.store_sequence_regions(regions)
        
        # Retrieve regions
        retrieved_regions = await data_unit.retrieve_sequence_regions()
        
        assert len(retrieved_regions) == 1
        assert retrieved_regions[0].sequence_id == "seq1"
        assert retrieved_regions[0].sequence_data == "ATCGATCGAT"
        
        await data_unit.shutdown()


class TestBioinformaticsComponents:
    """Test bioinformatics step, agent, and tool base classes."""
    
    @pytest.mark.asyncio
    async def test_bioinformatics_step(self):
        """Test bioinformatics step."""
        config = StepConfig(name="test_bio_step", description="Test step")
        bio_config = BioinformaticsConfig()
        
        class TestBioStep(BioinformaticsStep):
            async def process(self, input_data, **kwargs):
                return {"processed": True}
            
            async def process_sequences(self, sequences):
                return sequences
        
        step = TestBioStep(config, bio_config)
        await step.initialize()
        
        assert step.bio_config == bio_config
        assert step.tool_manager is not None
        
        await step.shutdown()
    
    @pytest.mark.asyncio
    async def test_bioinformatics_agent(self):
        """Test bioinformatics agent."""
        config = AgentConfig(name="test_bio_agent", description="Test agent")
        bio_config = BioinformaticsConfig()
        
        class TestBioAgent(BioinformaticsAgent):
            async def process(self, input_text, **kwargs):
                return "processed"
            
            async def analyze_sequences(self, sequences):
                return {"analysis": "complete"}
        
        agent = TestBioAgent(config, bio_config)
        await agent.initialize()
        
        assert agent.bio_config == bio_config
        assert agent.tool_manager is not None
        
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_bioinformatics_tool(self):
        """Test bioinformatics tool."""
        config = ToolConfig(name="test_bio_tool", description="Test tool")
        bio_config = BioinformaticsConfig()
        
        class TestBioTool(BioinformaticsTool):
            async def execute(self, **kwargs):
                return {"executed": True}
            
            async def process_biological_data(self, data):
                return data
        
        tool = TestBioTool(config, bio_config)
        await tool.initialize()
        
        assert tool.bio_config == bio_config
        assert tool.tool_manager is not None
        
        await tool.shutdown()


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_bioinformatics_data_unit(self):
        """Test bioinformatics data unit factory."""
        data_unit = create_bioinformatics_data_unit("test_data")
        assert isinstance(data_unit, BioinformaticsDataUnit)
        assert data_unit.config.name == "test_data"
    
    def test_create_sequence_coordinate(self):
        """Test sequence coordinate factory."""
        coord = create_sequence_coordinate(100, 200, CoordinateSystem.ONE_BASED, "+")
        assert coord.start == 100
        assert coord.end == 200
        assert coord.coordinate_system == CoordinateSystem.ONE_BASED
        assert coord.strand == "+"
    
    def test_create_sequence_region(self):
        """Test sequence region factory."""
        region = create_sequence_region(
            "test_seq", 1, 10, SequenceType.DNA, 
            sequence_data="ATCGATCGAT"
        )
        assert region.sequence_id == "test_seq"
        assert region.coordinates.start == 1
        assert region.coordinates.end == 10
        assert region.sequence_type == SequenceType.DNA
        assert region.sequence_data == "ATCGATCGAT"
    
    def test_create_sequence_manager(self):
        """Test sequence manager factory."""
        manager = create_sequence_manager()
        assert isinstance(manager, SequenceManager)
        
        config = BioinformaticsConfig(coordinate_system=CoordinateSystem.ZERO_BASED)
        manager_with_config = create_sequence_manager(config)
        assert manager_with_config.config.coordinate_system == CoordinateSystem.ZERO_BASED
    
    def test_create_fasta_parser(self):
        """Test FASTA parser factory."""
        parser = create_fasta_parser()
        assert isinstance(parser, FastaParser)
    
    def test_create_sequence_validator(self):
        """Test sequence validator factory."""
        validator = create_sequence_validator()
        assert isinstance(validator, SequenceValidator)
        assert not validator.strict
        
        strict_validator = create_sequence_validator(strict=True)
        assert strict_validator.strict


class TestIntegration:
    """Test integration with existing NanoBrain framework."""
    
    @pytest.mark.asyncio
    async def test_bioinformatics_workflow_integration(self):
        """Test that bioinformatics components integrate with workflows."""
        # This test verifies that our bioinformatics components can be used
        # in the existing NanoBrain workflow system
        
        config = StepConfig(name="bio_step", description="Bioinformatics step")
        bio_config = BioinformaticsConfig()
        
        class TestBioStep(BioinformaticsStep):
            async def process(self, input_data, **kwargs):
                # Process some biological data
                if "sequences" in input_data:
                    sequences = input_data["sequences"]
                    processed = self.standardize_coordinates(sequences)
                    return {"processed_sequences": processed}
                return {"status": "no sequences found"}
            
            async def process_sequences(self, sequences):
                return self.standardize_coordinates(sequences)
        
        step = TestBioStep(config, bio_config)
        await step.initialize()
        
        # Test with sequence data
        coord = SequenceCoordinate(start=1, end=10, coordinate_system=CoordinateSystem.ZERO_BASED)
        region = SequenceRegion(
            sequence_id="test",
            coordinates=coord,
            sequence_type=SequenceType.DNA,
            sequence_data="ATCGATCGAT"
        )
        
        input_data = {"sequences": [region]}
        result = await step.process(input_data)
        
        assert "processed_sequences" in result
        # Should be converted to 1-based (default config)
        assert result["processed_sequences"][0].coordinates.coordinate_system == CoordinateSystem.ONE_BASED
        assert result["processed_sequences"][0].coordinates.start == 2  # 1 -> 2 (0-based to 1-based)
        
        await step.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])