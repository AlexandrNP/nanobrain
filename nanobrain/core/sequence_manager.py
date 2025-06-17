"""
Sequence Management for NanoBrain Bioinformatics Framework

Provides comprehensive sequence handling, FASTA processing, and validation.
"""

import asyncio
import logging
import re
from typing import Any, Dict, Optional, List, Union, Tuple, Iterator
from pathlib import Path
from io import StringIO
from dataclasses import dataclass
from enum import Enum

from .bioinformatics import (
    SequenceRegion, SequenceCoordinate, SequenceType, CoordinateSystem,
    BioinformaticsConfig, create_sequence_region
)
from .logging_system import get_logger, OperationType

logger = logging.getLogger(__name__)


class SequenceFormat(Enum):
    """Supported sequence formats."""
    FASTA = "fasta"
    FASTQ = "fastq"
    GENBANK = "genbank"
    EMBL = "embl"


class SequenceValidationError(Exception):
    """Exception raised for sequence validation errors."""
    pass


@dataclass
class SequenceStats:
    """Statistics for a sequence."""
    length: int
    gc_content: float
    n_count: int
    valid_bases: int
    composition: Dict[str, int]


class SequenceValidator:
    """Validates biological sequences."""
    
    # Valid nucleotide characters
    DNA_CHARS = set('ATCGN')
    RNA_CHARS = set('AUCGN')
    PROTEIN_CHARS = set('ACDEFGHIKLMNPQRSTVWY*X')
    
    # IUPAC nucleotide codes
    IUPAC_DNA = set('ATCGRYSWKMBDHVN')
    IUPAC_RNA = set('AUCGRYSWKMBDHVN')
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.nb_logger = get_logger("sequence_validator")
    
    def validate_sequence(self, sequence: str, sequence_type: SequenceType) -> Tuple[bool, List[str]]:
        """Validate a sequence against its type."""
        errors = []
        sequence = sequence.upper().strip()
        
        if not sequence:
            errors.append("Empty sequence")
            return False, errors
        
        # Check for valid characters
        if sequence_type == SequenceType.DNA:
            valid_chars = self.DNA_CHARS if self.strict else self.IUPAC_DNA
            invalid_chars = set(sequence) - valid_chars
        elif sequence_type == SequenceType.RNA:
            valid_chars = self.RNA_CHARS if self.strict else self.IUPAC_RNA
            invalid_chars = set(sequence) - valid_chars
        elif sequence_type == SequenceType.PROTEIN:
            valid_chars = self.PROTEIN_CHARS
            invalid_chars = set(sequence) - valid_chars
        else:
            # Generic nucleotide - allow both DNA and RNA
            valid_chars = self.IUPAC_DNA | self.IUPAC_RNA
            invalid_chars = set(sequence) - valid_chars
        
        if invalid_chars:
            errors.append(f"Invalid characters for {sequence_type.value}: {invalid_chars}")
        
        # Additional validation rules
        if sequence_type in [SequenceType.DNA, SequenceType.RNA, SequenceType.NUCLEOTIDE]:
            # Check for excessive N content
            n_content = sequence.count('N') / len(sequence)
            if n_content > 0.5:
                errors.append(f"Excessive N content: {n_content:.2%}")
        
        return len(errors) == 0, errors
    
    def calculate_stats(self, sequence: str, sequence_type: SequenceType) -> SequenceStats:
        """Calculate sequence statistics."""
        sequence = sequence.upper().strip()
        length = len(sequence)
        
        # Count composition
        composition = {}
        for char in set(sequence):
            composition[char] = sequence.count(char)
        
        # Calculate GC content for nucleotides
        gc_content = 0.0
        if sequence_type in [SequenceType.DNA, SequenceType.RNA, SequenceType.NUCLEOTIDE]:
            gc_count = sequence.count('G') + sequence.count('C')
            if length > 0:
                gc_content = gc_count / length
        
        # Count N's and valid bases
        n_count = sequence.count('N')
        valid_bases = length - n_count
        
        return SequenceStats(
            length=length,
            gc_content=gc_content,
            n_count=n_count,
            valid_bases=valid_bases,
            composition=composition
        )


class FastaParser:
    """Parser for FASTA format sequences."""
    
    def __init__(self):
        self.nb_logger = get_logger("fasta_parser")
    
    def parse_fasta_string(self, fasta_content: str) -> List[Tuple[str, str]]:
        """Parse FASTA content string into header-sequence pairs."""
        sequences = []
        current_header = None
        current_sequence = []
        
        for line in fasta_content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_sequence)))
                
                # Start new sequence
                current_header = line[1:]  # Remove '>'
                current_sequence = []
            else:
                if current_header is None:
                    raise ValueError("FASTA sequence data found before header")
                current_sequence.append(line)
        
        # Save last sequence
        if current_header is not None:
            sequences.append((current_header, ''.join(current_sequence)))
        
        return sequences
    
    async def parse_fasta_file(self, file_path: Union[str, Path]) -> List[Tuple[str, str]]:
        """Parse FASTA file asynchronously."""
        file_path = Path(file_path)
        
        async with self.nb_logger.async_execution_context(
            OperationType.DATA_TRANSFER,
            f"parse_fasta_{file_path.name}"
        ) as context:
            
            try:
                content = file_path.read_text()
                sequences = self.parse_fasta_string(content)
                
                context.metadata['file_size'] = file_path.stat().st_size
                context.metadata['sequences_count'] = len(sequences)
                
                return sequences
                
            except Exception as e:
                self.nb_logger.error(f"Error parsing FASTA file {file_path}: {e}")
                raise
    
    def write_fasta_string(self, sequences: List[Tuple[str, str]], line_width: int = 80) -> str:
        """Write sequences to FASTA format string."""
        fasta_lines = []
        
        for header, sequence in sequences:
            # Add header line
            fasta_lines.append(f">{header}")
            
            # Add sequence lines with proper wrapping
            for i in range(0, len(sequence), line_width):
                fasta_lines.append(sequence[i:i + line_width])
        
        return '\n'.join(fasta_lines)
    
    async def write_fasta_file(self, sequences: List[Tuple[str, str]], 
                              file_path: Union[str, Path], line_width: int = 80) -> None:
        """Write sequences to FASTA file asynchronously."""
        file_path = Path(file_path)
        
        async with self.nb_logger.async_execution_context(
            OperationType.DATA_TRANSFER,
            f"write_fasta_{file_path.name}"
        ) as context:
            
            try:
                fasta_content = self.write_fasta_string(sequences, line_width)
                file_path.write_text(fasta_content)
                
                context.metadata['file_size'] = file_path.stat().st_size
                context.metadata['sequences_count'] = len(sequences)
                
            except Exception as e:
                self.nb_logger.error(f"Error writing FASTA file {file_path}: {e}")
                raise


class SequenceManager:
    """Comprehensive sequence management system."""
    
    def __init__(self, config: Optional[BioinformaticsConfig] = None):
        self.config = config or BioinformaticsConfig()
        self.validator = SequenceValidator()
        self.fasta_parser = FastaParser()
        self.nb_logger = get_logger("sequence_manager")
        
        # Cache for sequence regions
        self._sequence_cache: Dict[str, SequenceRegion] = {}
        
    async def load_sequences_from_fasta(self, file_path: Union[str, Path], 
                                       sequence_type: SequenceType = SequenceType.DNA,
                                       validate: bool = True) -> List[SequenceRegion]:
        """Load sequences from FASTA file into SequenceRegion objects."""
        async with self.nb_logger.async_execution_context(
            OperationType.DATA_TRANSFER,
            f"load_sequences_from_fasta"
        ) as context:
            
            # Parse FASTA file
            sequences = await self.fasta_parser.parse_fasta_file(file_path)
            
            # Convert to SequenceRegion objects
            regions = []
            validation_errors = []
            
            for header, sequence_data in sequences:
                # Parse header to extract sequence ID and coordinates if present
                sequence_id, coordinates = self._parse_fasta_header(header)
                
                # Validate sequence if requested
                if validate:
                    is_valid, errors = self.validator.validate_sequence(sequence_data, sequence_type)
                    if not is_valid:
                        validation_errors.extend([f"{sequence_id}: {error}" for error in errors])
                        continue
                
                # Create coordinates if not parsed from header
                if coordinates is None:
                    coordinates = SequenceCoordinate(
                        start=1,
                        end=len(sequence_data),
                        coordinate_system=self.config.coordinate_system
                    )
                
                # Create sequence region
                region = SequenceRegion(
                    sequence_id=sequence_id,
                    coordinates=coordinates,
                    sequence_type=sequence_type,
                    sequence_data=sequence_data
                )
                
                regions.append(region)
                
                # Cache the region
                self._sequence_cache[sequence_id] = region
            
            # Log validation errors if any
            if validation_errors:
                self.nb_logger.warning(f"Sequence validation errors: {validation_errors}")
            
            context.metadata.update({
                'total_sequences': len(sequences),
                'valid_sequences': len(regions),
                'validation_errors': len(validation_errors),
                'sequence_type': sequence_type.value
            })
            
            return regions
    
    def _parse_fasta_header(self, header: str) -> Tuple[str, Optional[SequenceCoordinate]]:
        """Parse FASTA header to extract sequence ID and coordinates."""
        # Try to extract coordinates from header like: sequence_id_start-end or sequence_id_start-end(strand)
        coord_pattern = r'(.+)_(\d+)-(\d+)(?:\(([+-])\))?$'
        match = re.match(coord_pattern, header)
        
        if match:
            sequence_id = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))
            strand = match.group(4)
            
            coordinates = SequenceCoordinate(
                start=start,
                end=end,
                coordinate_system=self.config.coordinate_system,
                strand=strand
            )
            
            return sequence_id, coordinates
        else:
            # No coordinates found, use entire header as sequence ID
            return header, None
    
    async def save_sequences_to_fasta(self, regions: List[SequenceRegion], 
                                     file_path: Union[str, Path]) -> None:
        """Save sequence regions to FASTA file."""
        async with self.nb_logger.async_execution_context(
            OperationType.DATA_TRANSFER,
            f"save_sequences_to_fasta"
        ) as context:
            
            # Convert regions to header-sequence pairs
            sequences = []
            for region in regions:
                if not region.sequence_data:
                    self.nb_logger.warning(f"Skipping region {region.sequence_id} - no sequence data")
                    continue
                
                header = region.get_fasta_header()[1:]  # Remove '>' prefix
                sequences.append((header, region.sequence_data))
            
            # Write to file
            await self.fasta_parser.write_fasta_file(sequences, file_path)
            
            context.metadata.update({
                'regions_count': len(regions),
                'sequences_written': len(sequences),
                'file_path': str(file_path)
            })
    
    def extract_subsequence(self, region: SequenceRegion, 
                           start: int, end: int,
                           coordinate_system: Optional[CoordinateSystem] = None) -> SequenceRegion:
        """Extract a subsequence from a sequence region."""
        if not region.sequence_data:
            raise ValueError("No sequence data available for extraction")
        
        # Use default coordinate system if not specified
        if coordinate_system is None:
            coordinate_system = self.config.coordinate_system
        
        # Convert coordinates to 0-based for extraction
        extract_coords = SequenceCoordinate(
            start=start,
            end=end,
            coordinate_system=coordinate_system
        ).to_zero_based()
        
        # Extract subsequence
        subseq_data = region.sequence_data[extract_coords.start:extract_coords.end]
        
        # Create new coordinates in the requested system
        new_coordinates = SequenceCoordinate(
            start=start,
            end=end,
            coordinate_system=coordinate_system,
            strand=region.coordinates.strand
        )
        
        # Create new region
        subregion = SequenceRegion(
            sequence_id=f"{region.sequence_id}_sub_{start}_{end}",
            coordinates=new_coordinates,
            sequence_type=region.sequence_type,
            sequence_data=subseq_data,
            annotations=region.annotations.copy()
        )
        
        return subregion
    
    def merge_regions(self, regions: List[SequenceRegion], 
                     new_sequence_id: str) -> SequenceRegion:
        """Merge multiple sequence regions into one."""
        if not regions:
            raise ValueError("No regions to merge")
        
        # Sort regions by start coordinate
        sorted_regions = sorted(regions, key=lambda r: r.coordinates.to_zero_based().start)
        
        # Check if regions are contiguous or overlapping
        merged_sequence = ""
        min_start = float('inf')
        max_end = 0
        sequence_type = sorted_regions[0].sequence_type
        
        for region in sorted_regions:
            if not region.sequence_data:
                raise ValueError(f"Region {region.sequence_id} has no sequence data")
            
            if region.sequence_type != sequence_type:
                raise ValueError("Cannot merge regions of different sequence types")
            
            merged_sequence += region.sequence_data
            zero_coords = region.coordinates.to_zero_based()
            min_start = min(min_start, zero_coords.start)
            max_end = max(max_end, zero_coords.end)
        
        # Create merged coordinates
        merged_coordinates = SequenceCoordinate(
            start=int(min_start) + 1 if self.config.coordinate_system == CoordinateSystem.ONE_BASED else int(min_start),
            end=int(max_end),
            coordinate_system=self.config.coordinate_system
        )
        
        # Merge annotations
        merged_annotations = {}
        for region in sorted_regions:
            merged_annotations.update(region.annotations)
        
        # Create merged region
        merged_region = SequenceRegion(
            sequence_id=new_sequence_id,
            coordinates=merged_coordinates,
            sequence_type=sequence_type,
            sequence_data=merged_sequence,
            annotations=merged_annotations
        )
        
        return merged_region
    
    def get_sequence_stats(self, region: SequenceRegion) -> SequenceStats:
        """Get statistics for a sequence region."""
        if not region.sequence_data:
            raise ValueError("No sequence data available for statistics")
        
        return self.validator.calculate_stats(region.sequence_data, region.sequence_type)
    
    def find_orfs(self, region: SequenceRegion, min_length: int = 100) -> List[SequenceRegion]:
        """Find open reading frames in a DNA sequence."""
        if region.sequence_type not in [SequenceType.DNA, SequenceType.NUCLEOTIDE]:
            raise ValueError("ORF finding only supported for DNA sequences")
        
        if not region.sequence_data:
            raise ValueError("No sequence data available")
        
        sequence = region.sequence_data.upper()
        orfs = []
        
        # Start codons and stop codons
        start_codons = ['ATG', 'GTG', 'TTG']
        stop_codons = ['TAA', 'TAG', 'TGA']
        
        # Check all three reading frames in both directions
        for frame in range(3):
            # Forward strand
            for start_pos in range(frame, len(sequence) - 2, 3):
                codon = sequence[start_pos:start_pos + 3]
                if codon in start_codons:
                    # Look for stop codon
                    for stop_pos in range(start_pos + 3, len(sequence) - 2, 3):
                        stop_codon = sequence[stop_pos:stop_pos + 3]
                        if stop_codon in stop_codons:
                            orf_length = stop_pos + 3 - start_pos
                            if orf_length >= min_length:
                                # Create ORF coordinates (convert to 1-based if needed)
                                orf_start = start_pos + 1 if self.config.coordinate_system == CoordinateSystem.ONE_BASED else start_pos
                                orf_end = stop_pos + 3 if self.config.coordinate_system == CoordinateSystem.ONE_BASED else stop_pos + 2
                                
                                orf_coords = SequenceCoordinate(
                                    start=orf_start,
                                    end=orf_end,
                                    coordinate_system=self.config.coordinate_system,
                                    strand='+'
                                )
                                
                                orf_region = SequenceRegion(
                                    sequence_id=f"{region.sequence_id}_orf_{orf_start}_{orf_end}_+",
                                    coordinates=orf_coords,
                                    sequence_type=SequenceType.DNA,
                                    sequence_data=sequence[start_pos:stop_pos + 3],
                                    annotations={
                                        'type': 'ORF',
                                        'frame': frame + 1,
                                        'strand': '+',
                                        'start_codon': codon,
                                        'stop_codon': stop_codon
                                    }
                                )
                                
                                orfs.append(orf_region)
                            break
        
        return orfs
    
    def translate_sequence(self, region: SequenceRegion, 
                          genetic_code: int = 1) -> SequenceRegion:
        """Translate a DNA/RNA sequence to protein."""
        if region.sequence_type not in [SequenceType.DNA, SequenceType.RNA, SequenceType.NUCLEOTIDE]:
            raise ValueError("Translation only supported for nucleotide sequences")
        
        if not region.sequence_data:
            raise ValueError("No sequence data available")
        
        # Standard genetic code (simplified)
        codon_table = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        }
        
        sequence = region.sequence_data.upper().replace('U', 'T')  # Convert RNA to DNA
        protein_sequence = ""
        
        # Translate in triplets
        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i:i + 3]
            if len(codon) == 3:
                amino_acid = codon_table.get(codon, 'X')  # X for unknown
                protein_sequence += amino_acid
        
        # Create protein region
        protein_region = SequenceRegion(
            sequence_id=f"{region.sequence_id}_translated",
            coordinates=SequenceCoordinate(
                start=1,
                end=len(protein_sequence),
                coordinate_system=self.config.coordinate_system,
                strand=region.coordinates.strand
            ),
            sequence_type=SequenceType.PROTEIN,
            sequence_data=protein_sequence,
            annotations={
                **region.annotations,
                'translated_from': region.sequence_id,
                'genetic_code': genetic_code
            }
        )
        
        return protein_region
    
    def get_cached_sequence(self, sequence_id: str) -> Optional[SequenceRegion]:
        """Get a cached sequence region."""
        return self._sequence_cache.get(sequence_id)
    
    def cache_sequence(self, region: SequenceRegion) -> None:
        """Cache a sequence region."""
        self._sequence_cache[region.sequence_id] = region
    
    def clear_cache(self) -> None:
        """Clear the sequence cache."""
        self._sequence_cache.clear()
        self.nb_logger.debug("Sequence cache cleared")


# Factory functions
def create_sequence_manager(config: Optional[BioinformaticsConfig] = None) -> SequenceManager:
    """Create a sequence manager with the given configuration."""
    return SequenceManager(config)


def create_fasta_parser() -> FastaParser:
    """Create a FASTA parser."""
    return FastaParser()


def create_sequence_validator(strict: bool = False) -> SequenceValidator:
    """Create a sequence validator."""
    return SequenceValidator(strict)