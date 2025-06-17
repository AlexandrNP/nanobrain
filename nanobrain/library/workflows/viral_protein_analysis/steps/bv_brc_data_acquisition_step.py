"""
BV-BRC Data Acquisition Step (Steps 1-7)

Implements the first 7 steps of the Alphavirus workflow:
1. Download all Alphavirus genomes from BV-BRC
2. Filter genomes by size (8KB-15KB range)
3-4. Extract unique protein MD5s and deduplicate
5. Get feature sequences for MD5s
6. Get annotations for unique MD5 sequences
7. Create annotated FASTA file

Based on BV-BRC CLI documentation:
https://www.bv-brc.org/docs/cli_tutorial/cli_getting_started.html
"""

import asyncio
import tempfile
import time
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import hashlib
import re

from nanobrain.core.logging_system import get_logger
from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCTool, BVBRCConfig


class GenomeData:
    """Container for genome information"""
    
    def __init__(self, genome_id: str, genome_length: int, genome_name: str, 
                 taxon_lineage: Optional[str] = None):
        self.genome_id = genome_id
        self.genome_length = genome_length
        self.genome_name = genome_name
        self.taxon_lineage = taxon_lineage or ""
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'genome_id': self.genome_id,
            'genome_length': self.genome_length,
            'genome_name': self.genome_name,
            'taxon_lineage': self.taxon_lineage
        }


class ProteinData:
    """Container for protein information"""
    
    def __init__(self, patric_id: str, aa_sequence_md5: str, genome_id: str,
                 product: Optional[str] = None, gene: Optional[str] = None):
        self.patric_id = patric_id
        self.aa_sequence_md5 = aa_sequence_md5
        self.genome_id = genome_id
        self.product = product or "hypothetical protein"
        self.gene = gene or ""
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'patric_id': self.patric_id,
            'aa_sequence_md5': self.aa_sequence_md5,
            'genome_id': self.genome_id,
            'product': self.product,
            'gene': self.gene
        }


class SequenceData:
    """Container for sequence information"""
    
    def __init__(self, aa_sequence_md5: str, aa_sequence: str):
        self.aa_sequence_md5 = aa_sequence_md5
        self.aa_sequence = aa_sequence
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'aa_sequence_md5': self.aa_sequence_md5,
            'aa_sequence': self.aa_sequence
        }


class AnnotationData:
    """Container for annotation information"""
    
    def __init__(self, aa_sequence_md5: str, product: str, gene: str = "",
                 refseq_locus_tag: str = "", go: str = "", ec: str = "", 
                 pathway: str = ""):
        self.aa_sequence_md5 = aa_sequence_md5
        self.product = product
        self.gene = gene
        self.refseq_locus_tag = refseq_locus_tag
        self.go = go
        self.ec = ec
        self.pathway = pathway
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'aa_sequence_md5': self.aa_sequence_md5,
            'product': self.product,
            'gene': self.gene,
            'refseq_locus_tag': self.refseq_locus_tag,
            'go': self.go,
            'ec': self.ec,
            'pathway': self.pathway
        }


class BVBRCDataAcquisitionStep:
    """
    BV-BRC Data Acquisition Step implementing workflow steps 1-7
    
    Uses the corrected BV-BRC path: /Applications/BV-BRC.app/deployment/bin/
    Implements anonymous access with proper data verification
    """
    
    def __init__(self, bvbrc_config: BVBRCConfig, step_config: Dict[str, Any]):
        self.bvbrc_config = bvbrc_config
        self.step_config = step_config
        self.logger = get_logger("bvbrc_data_acquisition")
        
        # Convert workflow BVBRCConfig to tool BVBRCConfig
        from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCConfig as ToolBVBRCConfig
        
        tool_config = ToolBVBRCConfig(
            tool_name="bv_brc",
            installation_path=bvbrc_config.installation_path,
            executable_path=bvbrc_config.executable_path,
            genome_batch_size=bvbrc_config.genome_batch,
            md5_batch_size=bvbrc_config.md5_batch,
            min_genome_length=bvbrc_config.min_length,
            max_genome_length=bvbrc_config.max_length,
            timeout_seconds=bvbrc_config.timeout_seconds,
            retry_attempts=bvbrc_config.retry_attempts,
            verify_on_init=bvbrc_config.verify_on_init
        )
        
        # Initialize BV-BRC tool with converted configuration
        self.bv_brc_tool = BVBRCTool(config=tool_config)
        
        # Configuration parameters
        self.min_genome_length = step_config.get('min_length', 8000)
        self.max_genome_length = step_config.get('max_length', 15000)
        self.genome_batch_size = step_config.get('genome_batch_size', 100)
        self.md5_batch_size = step_config.get('md5_batch_size', 50)
        
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute steps 1-7 of the BV-BRC data acquisition workflow
        
        Args:
            input_params: Input parameters including target_genus
            
        Returns:
            Dict containing all acquired and processed data
        """
        
        step_start_time = time.time()
        target_genus = input_params.get('target_genus', 'Alphavirus')
        
        try:
            # Step 1: Download Alphavirus genomes
            self.logger.info("🔄 Step 1: Downloading Alphavirus genomes from BV-BRC")
            original_genomes = await self._download_alphavirus_genomes(target_genus)
            self.logger.info(f"✅ Downloaded {len(original_genomes)} {target_genus} genomes")
            
            # Step 2: Filter genomes by size
            self.logger.info("🔄 Step 2: Filtering genomes by size")
            filtered_genomes = await self._filter_genomes_by_size(original_genomes)
            self.logger.info(f"✅ Filtered to {len(filtered_genomes)} genomes within size range ({self.min_genome_length}-{self.max_genome_length} bp)")
            
            if not filtered_genomes:
                raise ValueError(f"No {target_genus} genomes found within size range")
            
            # Steps 3-4: Get unique protein MD5s
            self.logger.info("🔄 Steps 3-4: Extracting unique protein MD5s")
            genome_ids = [g.genome_id for g in filtered_genomes]
            unique_proteins = await self._get_unique_protein_md5s(genome_ids)
            self.logger.info(f"✅ Found {len(unique_proteins)} unique proteins")
            
            # Step 5: Get feature sequences
            self.logger.info("🔄 Step 5: Retrieving protein sequences")
            md5_list = [p.aa_sequence_md5 for p in unique_proteins]
            protein_sequences = await self._get_feature_sequences(md5_list)
            self.logger.info(f"✅ Retrieved {len(protein_sequences)} protein sequences")
            
            # Step 6: Get annotations
            self.logger.info("🔄 Step 6: Retrieving protein annotations")
            protein_annotations = await self._get_protein_annotations(md5_list)
            self.logger.info(f"✅ Retrieved annotations for {len(protein_annotations)} proteins")
            
            # Step 7: Create annotated FASTA
            self.logger.info("🔄 Step 7: Creating annotated FASTA file")
            annotated_fasta = await self._create_annotated_fasta(protein_sequences, protein_annotations)
            self.logger.info("✅ Created annotated FASTA file")
            
            execution_time = time.time() - step_start_time
            self.logger.info(f"🎉 BV-BRC data acquisition completed in {execution_time:.2f} seconds")
            
            return {
                'original_genomes': [g.to_dict() for g in original_genomes],
                'filtered_genomes': [g.to_dict() for g in filtered_genomes],
                'unique_proteins': [p.to_dict() for p in unique_proteins],
                'protein_sequences': [s.to_dict() for s in protein_sequences],
                'protein_annotations': [a.to_dict() for a in protein_annotations],
                'annotated_fasta': annotated_fasta,
                'execution_time': execution_time,
                'statistics': {
                    'total_genomes_downloaded': len(original_genomes),
                    'genomes_after_filtering': len(filtered_genomes),
                    'unique_proteins_found': len(unique_proteins),
                    'sequences_retrieved': len(protein_sequences),
                    'annotations_retrieved': len(protein_annotations)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ BV-BRC data acquisition failed: {e}")
            raise
            
    async def _download_alphavirus_genomes(self, target_genus: str) -> List[GenomeData]:
        """
        Step 1: Download all Alphavirus genomes with metadata
        
        Uses p3-all-genomes command with proper parameters
        """
        
        # Verify BV-BRC tool is accessible
        if not await self.bv_brc_tool.verify_installation():
            raise RuntimeError("BV-BRC tool is not accessible. Please check installation.")
        
        # Construct command arguments for genome download
        command_args = [
            "--eq", f"taxon_lineage_names,{target_genus}",
            "--attr", "genome_id,genome_length,genome_name,taxon_lineage_names"
        ]
        
        try:
            result = await self.bv_brc_tool.execute_p3_command("p3-all-genomes", command_args)
            
            if result.returncode != 0:
                raise RuntimeError(f"p3-all-genomes command failed: {result.stderr.decode()}")
            
            genomes = await self._parse_genome_data(result.stdout)
            
            if not genomes:
                self.logger.warning(f"No {target_genus} genomes found in BV-BRC")
            
            return genomes
            
        except Exception as e:
            self.logger.error(f"Failed to download {target_genus} genomes: {e}")
            raise
            
    async def _parse_genome_data(self, raw_data: bytes) -> List[GenomeData]:
        """Parse genome data from BV-BRC output"""
        
        if not raw_data:
            return []
        
        lines = raw_data.decode().strip().split('\n')
        
        if len(lines) < 2:  # Must have header + at least one data line
            self.logger.warning("BV-BRC returned only headers, no genome data")
            return []
        
        # Parse header to get field positions
        header = lines[0].split('\t')
        try:
            genome_id_idx = header.index('genome_id')
            genome_length_idx = header.index('genome_length')
            genome_name_idx = header.index('genome_name')
            taxon_lineage_idx = header.index('taxon_lineage_names') if 'taxon_lineage_names' in header else -1
        except ValueError as e:
            raise ValueError(f"Missing required field in BV-BRC output: {e}")
        
        genomes = []
        for line_num, line in enumerate(lines[1:], 2):
            try:
                fields = line.split('\t')
                
                if len(fields) <= max(genome_id_idx, genome_length_idx, genome_name_idx):
                    continue
                
                genome_id = fields[genome_id_idx].strip()
                genome_length_str = fields[genome_length_idx].strip()
                genome_name = fields[genome_name_idx].strip()
                
                # Validate and parse genome length
                if not genome_length_str or genome_length_str == '-':
                    continue
                    
                try:
                    genome_length = int(genome_length_str)
                except ValueError:
                    self.logger.warning(f"Invalid genome length '{genome_length_str}' for {genome_id}")
                    continue
                
                # Get taxon lineage if available
                taxon_lineage = ""
                if taxon_lineage_idx >= 0 and len(fields) > taxon_lineage_idx:
                    taxon_lineage = fields[taxon_lineage_idx].strip()
                
                genome = GenomeData(
                    genome_id=genome_id,
                    genome_length=genome_length,
                    genome_name=genome_name,
                    taxon_lineage=taxon_lineage
                )
                
                genomes.append(genome)
                
            except Exception as e:
                self.logger.warning(f"Error parsing genome data at line {line_num}: {e}")
                continue
        
        return genomes
        
    async def _filter_genomes_by_size(self, genomes: List[GenomeData]) -> List[GenomeData]:
        """
        Step 2: Filter genomes by size based on threshold
        
        Alphavirus genomes are typically 11,000-12,000 bp
        Filter range: 8,000-15,000 bp to exclude fragments and contaminated assemblies
        """
        
        filtered_genomes = []
        
        for genome in genomes:
            if self.min_genome_length <= genome.genome_length <= self.max_genome_length:
                filtered_genomes.append(genome)
            else:
                self.logger.debug(f"Filtered genome {genome.genome_id}: length {genome.genome_length} outside range")
        
        return filtered_genomes
        
    async def _get_unique_protein_md5s(self, genome_ids: List[str]) -> List[ProteinData]:
        """
        Steps 3-4: Get unique protein MD5s and deduplicate
        
        Uses p3-get-feature-data to get all proteins, then removes duplicates by MD5
        """
        
        all_proteins = []
        
        # Process genomes in batches to avoid command line length limits
        for i in range(0, len(genome_ids), self.genome_batch_size):
            batch = genome_ids[i:i+self.genome_batch_size]
            
            self.logger.debug(f"Processing genome batch {i//self.genome_batch_size + 1}: {len(batch)} genomes")
            
            batch_proteins = await self._get_proteins_for_genomes(batch)
            all_proteins.extend(batch_proteins)
        
        # Remove duplicates by MD5 hash
        unique_proteins_dict = {}
        for protein in all_proteins:
            md5 = protein.aa_sequence_md5
            if md5 and md5 not in unique_proteins_dict:
                unique_proteins_dict[md5] = protein
        
        unique_proteins = list(unique_proteins_dict.values())
        
        self.logger.info(f"Deduplicated {len(all_proteins)} proteins to {len(unique_proteins)} unique MD5s")
        
        return unique_proteins
        
    async def _get_proteins_for_genomes(self, genome_ids: List[str]) -> List[ProteinData]:
        """Get protein data for a batch of genomes"""
        
        genome_list = ",".join(genome_ids)
        command_args = [
            "--in", f"genome_id,{genome_list}",
            "--attr", "patric_id,aa_sequence_md5,genome_id,product,gene",
            "--eq", "feature_type,CDS"  # Only coding sequences
        ]
        
        try:
            result = await self.bv_brc_tool.execute_p3_command("p3-get-feature-data", command_args)
            
            if result.returncode != 0:
                self.logger.warning(f"p3-get-feature-data failed for batch: {result.stderr.decode()}")
                return []
            
            return await self._parse_protein_data(result.stdout)
            
        except Exception as e:
            self.logger.warning(f"Error getting proteins for genome batch: {e}")
            return []
            
    async def _parse_protein_data(self, raw_data: bytes) -> List[ProteinData]:
        """Parse protein data from BV-BRC output"""
        
        if not raw_data:
            return []
        
        lines = raw_data.decode().strip().split('\n')
        
        if len(lines) < 2:
            return []
        
        # Parse header
        header = lines[0].split('\t')
        try:
            patric_id_idx = header.index('patric_id')
            md5_idx = header.index('aa_sequence_md5')
            genome_id_idx = header.index('genome_id')
            product_idx = header.index('product') if 'product' in header else -1
            gene_idx = header.index('gene') if 'gene' in header else -1
        except ValueError as e:
            raise ValueError(f"Missing required field in protein data: {e}")
        
        proteins = []
        for line in lines[1:]:
            try:
                fields = line.split('\t')
                
                if len(fields) <= max(patric_id_idx, md5_idx, genome_id_idx):
                    continue
                
                patric_id = fields[patric_id_idx].strip()
                aa_sequence_md5 = fields[md5_idx].strip()
                genome_id = fields[genome_id_idx].strip()
                
                # Skip if MD5 is missing
                if not aa_sequence_md5 or aa_sequence_md5 == '-':
                    continue
                
                product = ""
                if product_idx >= 0 and len(fields) > product_idx:
                    product = fields[product_idx].strip()
                
                gene = ""
                if gene_idx >= 0 and len(fields) > gene_idx:
                    gene = fields[gene_idx].strip()
                
                protein = ProteinData(
                    patric_id=patric_id,
                    aa_sequence_md5=aa_sequence_md5,
                    genome_id=genome_id,
                    product=product,
                    gene=gene
                )
                
                proteins.append(protein)
                
            except Exception as e:
                self.logger.warning(f"Error parsing protein data: {e}")
                continue
        
        return proteins
        
    async def _get_feature_sequences(self, md5_list: List[str]) -> List[SequenceData]:
        """
        Step 5: Get feature sequences for MD5s
        
        Uses p3-get-feature-sequence command
        """
        
        sequences = []
        
        # Process MD5s in batches
        for i in range(0, len(md5_list), self.md5_batch_size):
            batch = md5_list[i:i+self.md5_batch_size]
            
            self.logger.debug(f"Getting sequences for batch {i//self.md5_batch_size + 1}: {len(batch)} MD5s")
            
            batch_sequences = await self._get_sequences_for_md5s(batch)
            sequences.extend(batch_sequences)
        
        return sequences
        
    async def _get_sequences_for_md5s(self, md5_list: List[str]) -> List[SequenceData]:
        """Get sequences for a batch of MD5s"""
        
        md5_string = ",".join(md5_list)
        command_args = [
            "--in", f"aa_sequence_md5,{md5_string}",
            "--attr", "aa_sequence_md5,aa_sequence"
        ]
        
        try:
            result = await self.bv_brc_tool.execute_p3_command("p3-get-feature-sequence", command_args)
            
            if result.returncode != 0:
                self.logger.warning(f"p3-get-feature-sequence failed: {result.stderr.decode()}")
                return []
            
            return await self._parse_sequence_data(result.stdout)
            
        except Exception as e:
            self.logger.warning(f"Error getting sequences for MD5 batch: {e}")
            return []
            
    async def _parse_sequence_data(self, raw_data: bytes) -> List[SequenceData]:
        """Parse sequence data from BV-BRC output"""
        
        if not raw_data:
            return []
        
        lines = raw_data.decode().strip().split('\n')
        
        if len(lines) < 2:
            return []
        
        # Parse header
        header = lines[0].split('\t')
        try:
            md5_idx = header.index('aa_sequence_md5')
            sequence_idx = header.index('aa_sequence')
        except ValueError as e:
            raise ValueError(f"Missing required field in sequence data: {e}")
        
        sequences = []
        for line in lines[1:]:
            try:
                fields = line.split('\t')
                
                if len(fields) <= max(md5_idx, sequence_idx):
                    continue
                
                aa_sequence_md5 = fields[md5_idx].strip()
                aa_sequence = fields[sequence_idx].strip()
                
                # Skip if sequence is missing
                if not aa_sequence or aa_sequence == '-':
                    continue
                
                sequence = SequenceData(
                    aa_sequence_md5=aa_sequence_md5,
                    aa_sequence=aa_sequence
                )
                
                sequences.append(sequence)
                
            except Exception as e:
                self.logger.warning(f"Error parsing sequence data: {e}")
                continue
        
        return sequences
        
    async def _get_protein_annotations(self, md5_list: List[str]) -> List[AnnotationData]:
        """
        Step 6: Get annotations for unique MD5 sequences
        
        Retrieves comprehensive annotation data including GO terms, EC numbers, pathways
        """
        
        annotations = []
        
        # Process MD5s in batches
        for i in range(0, len(md5_list), self.md5_batch_size):
            batch = md5_list[i:i+self.md5_batch_size]
            
            self.logger.debug(f"Getting annotations for batch {i//self.md5_batch_size + 1}: {len(batch)} MD5s")
            
            batch_annotations = await self._get_annotations_for_md5s(batch)
            annotations.extend(batch_annotations)
        
        return annotations
        
    async def _get_annotations_for_md5s(self, md5_list: List[str]) -> List[AnnotationData]:
        """Get annotations for a batch of MD5s"""
        
        md5_string = ",".join(md5_list)
        command_args = [
            "--in", f"aa_sequence_md5,{md5_string}",
            "--attr", "aa_sequence_md5,product,gene,refseq_locus_tag,go,ec,pathway"
        ]
        
        try:
            result = await self.bv_brc_tool.execute_p3_command("p3-get-feature-data", command_args)
            
            if result.returncode != 0:
                self.logger.warning(f"p3-get-feature-data for annotations failed: {result.stderr.decode()}")
                return []
            
            return await self._parse_annotation_data(result.stdout)
            
        except Exception as e:
            self.logger.warning(f"Error getting annotations for MD5 batch: {e}")
            return []
            
    async def _parse_annotation_data(self, raw_data: bytes) -> List[AnnotationData]:
        """Parse annotation data from BV-BRC output"""
        
        if not raw_data:
            return []
        
        lines = raw_data.decode().strip().split('\n')
        
        if len(lines) < 2:
            return []
        
        # Parse header
        header = lines[0].split('\t')
        try:
            md5_idx = header.index('aa_sequence_md5')
            product_idx = header.index('product')
        except ValueError as e:
            raise ValueError(f"Missing required field in annotation data: {e}")
        
        # Optional fields
        gene_idx = header.index('gene') if 'gene' in header else -1
        refseq_idx = header.index('refseq_locus_tag') if 'refseq_locus_tag' in header else -1
        go_idx = header.index('go') if 'go' in header else -1
        ec_idx = header.index('ec') if 'ec' in header else -1
        pathway_idx = header.index('pathway') if 'pathway' in header else -1
        
        annotations = []
        for line in lines[1:]:
            try:
                fields = line.split('\t')
                
                if len(fields) <= max(md5_idx, product_idx):
                    continue
                
                aa_sequence_md5 = fields[md5_idx].strip()
                product = fields[product_idx].strip()
                
                # Get optional fields
                gene = fields[gene_idx].strip() if gene_idx >= 0 and len(fields) > gene_idx else ""
                refseq_locus_tag = fields[refseq_idx].strip() if refseq_idx >= 0 and len(fields) > refseq_idx else ""
                go = fields[go_idx].strip() if go_idx >= 0 and len(fields) > go_idx else ""
                ec = fields[ec_idx].strip() if ec_idx >= 0 and len(fields) > ec_idx else ""
                pathway = fields[pathway_idx].strip() if pathway_idx >= 0 and len(fields) > pathway_idx else ""
                
                annotation = AnnotationData(
                    aa_sequence_md5=aa_sequence_md5,
                    product=product,
                    gene=gene,
                    refseq_locus_tag=refseq_locus_tag,
                    go=go,
                    ec=ec,
                    pathway=pathway
                )
                
                annotations.append(annotation)
                
            except Exception as e:
                self.logger.warning(f"Error parsing annotation data: {e}")
                continue
        
        return annotations
        
    async def _create_annotated_fasta(self, sequences: List[SequenceData], 
                                     annotations: List[AnnotationData]) -> str:
        """
        Step 7: Create FASTA file with detailed annotation headers
        
        FASTA header format:
        >{md5}|{gene}|{product}|{refseq_locus_tag}|{pathway}
        """
        
        # Create annotation lookup
        annotation_map = {ann.aa_sequence_md5: ann for ann in annotations}
        
        fasta_lines = []
        processed_count = 0
        
        for seq in sequences:
            annotation = annotation_map.get(seq.aa_sequence_md5)
            
            if annotation:
                header = self._format_fasta_header(seq.aa_sequence_md5, annotation)
                fasta_lines.append(f">{header}")
                fasta_lines.append(seq.aa_sequence)
                processed_count += 1
            else:
                # Use basic header if no annotation available
                fasta_lines.append(f">{seq.aa_sequence_md5}|unknown|hypothetical protein|no_tag|unknown_pathway")
                fasta_lines.append(seq.aa_sequence)
                processed_count += 1
        
        self.logger.info(f"Created FASTA with {processed_count} sequences")
        
        return "\n".join(fasta_lines)
        
    def _format_fasta_header(self, md5: str, annotation: AnnotationData) -> str:
        """Format FASTA header with detailed annotation"""
        
        components = [
            md5,
            annotation.gene or "unknown",
            annotation.product or "hypothetical protein",
            annotation.refseq_locus_tag or "no_tag",
            annotation.pathway or "unknown_pathway"
        ]
        
        # Clean components (remove pipes and tabs that could break parsing)
        cleaned_components = []
        for comp in components:
            cleaned = str(comp).replace("|", "_").replace("\t", " ").replace("\n", " ")
            cleaned_components.append(cleaned)
        
        return "|".join(cleaned_components) 