# PHASE 2 IMPLEMENTATION PLAN: EXTERNAL TOOL FRAMEWORK & ALPHAVIRUS WORKFLOW

## OVERVIEW

This document outlines the detailed implementation plan for Phase 2 of the NanoBrain bioinformatics framework, focusing on external tool integration and a comprehensive Alphavirus protein analysis workflow using BV-BRC tools.

## ARCHITECTURE OVERVIEW

### External Tool Framework
```
nanobrain/core/
├── external_tool.py                    # NEW - Base class for external tools
├── bioinformatics.py                   # EXISTING - Enhanced with tool integration
└── sequence_manager.py                 # EXISTING - Enhanced with BV-BRC data

nanobrain/library/tools/bioinformatics/
├── base_external_tool.py               # NEW - Bioinformatics tool base class
├── bv_brc_tool.py                      # NEW - BV-BRC CLI wrapper
├── mmseqs_tool.py                      # NEW - MMseqs2 clustering tool
├── muscle_tool.py                      # NEW - MUSCLE alignment tool
└── pssm_generator_tool.py              # NEW - PSSM matrix generation
```

### Workflow Implementation
```
nanobrain/library/workflows/alphavirus_analysis/
├── alphavirus_workflow.py              # NEW - Main workflow orchestrator
├── steps/
│   ├── bv_brc_data_acquisition_step.py # NEW - Steps 1-7
│   ├── annotation_mapping_step.py      # NEW - Step 8
│   ├── sequence_curation_step.py       # NEW - Steps 9-11
│   ├── clustering_step.py              # NEW - Step 12
│   ├── alignment_step.py               # NEW - Step 13
│   └── pssm_analysis_step.py           # NEW - Step 14
└── config/
    ├── AlphavirusWorkflow.yml          # NEW - Main workflow config
    ├── bv_brc_config.yml               # NEW - BV-BRC tool configuration
    └── clustering_config.yml           # NEW - MMseqs2 parameters
```

## DETAILED STEP-BY-STEP IMPLEMENTATION PLAN

### PHASE 2.1: EXTERNAL TOOL BASE FRAMEWORK

#### Step 2.1.1: Create External Tool Base Class
**File**: `nanobrain/core/external_tool.py`

**Objective**: Create a universal base class for all external tool integrations

**Implementation Details**:
```python
class ExternalTool(ABC):
    """
    Base class for external tool integration in NanoBrain framework.
    
    This class provides:
    - Tool installation and verification
    - Command execution with proper error handling
    - Environment management (conda, docker, etc.)
    - Result parsing and validation
    - Integration with NanoBrain logging and monitoring
    """
    
    def __init__(self, config: ExternalToolConfig):
        self.config = config
        self.tool_name = config.tool_name
        self.installation_path = config.installation_path
        self.environment = config.environment
        self.logger = get_logger(f"external_tool_{self.tool_name}")
        
    @abstractmethod
    async def verify_installation(self) -> bool:
        """Verify tool is properly installed and accessible"""
        pass
        
    @abstractmethod
    async def execute_command(self, command: List[str], **kwargs) -> ToolResult:
        """Execute tool command with proper error handling"""
        pass
        
    @abstractmethod
    async def parse_output(self, raw_output: str) -> Dict[str, Any]:
        """Parse tool-specific output format"""
        pass
        
    async def install_if_missing(self) -> bool:
        """Install tool if not found"""
        pass
        
    async def get_version(self) -> str:
        """Get tool version information"""
        pass
```

**Key Features**:
- Abstract base class following Python inheritance patterns ([GeeksforGeeks inheritance guide](https://www.geeksforgeeks.org/create-derived-class-from-base-class-universally-in-python/))
- Async/await support for non-blocking operations
- Comprehensive error handling and logging
- Environment management (conda, PATH, etc.)
- Tool verification and auto-installation

#### Step 2.1.2: Create Bioinformatics Tool Base Class
**File**: `nanobrain/library/tools/bioinformatics/base_external_tool.py`

**Objective**: Specialized base class for bioinformatics external tools

**Implementation Details**:
```python
class BioinformaticsExternalTool(ExternalTool):
    """
    Specialized base class for bioinformatics external tools.
    
    Extends ExternalTool with bioinformatics-specific functionality:
    - Sequence format validation
    - Biological data type handling
    - Standard bioinformatics file formats (FASTA, GenBank, etc.)
    - Integration with sequence managers
    """
    
    def __init__(self, config: BioinformaticsToolConfig):
        super().__init__(config)
        self.sequence_manager = SequenceManager()
        self.supported_formats = config.supported_formats
        self.coordinate_system = config.coordinate_system
        
    async def validate_input_sequences(self, sequences: Any) -> bool:
        """Validate biological sequence input"""
        pass
        
    async def convert_coordinates(self, coordinates: Any) -> Any:
        """Convert between coordinate systems if needed"""
        pass
        
    async def handle_biological_output(self, output: str) -> BiologicalData:
        """Parse and validate biological data output"""
        pass
```

### PHASE 2.2: BV-BRC TOOL INTEGRATION

#### Step 2.2.1: Create BV-BRC Tool Wrapper
**File**: `nanobrain/library/tools/bioinformatics/bv_brc_tool.py`

**Objective**: Comprehensive wrapper for BV-BRC CLI tools on macOS

**Implementation Details**:
```python
class BVBRCTool(BioinformaticsExternalTool):
    """
    BV-BRC CLI tool wrapper for macOS.
    
    Handles all BV-BRC command-line tools including:
    - p3-all-genomes
    - p3-get-genome-data
    - p3-get-feature-data
    - p3-get-feature-sequence
    - And all other p3-* commands
    """
    
    def __init__(self, config: Optional[BVBRCConfig] = None):
        config = config or BVBRCConfig(
            tool_name="bv_brc",
            installation_path="/Applications/BV-BRC.app/",
            executable_path="/Applications/BV-BRC.app/Contents/Resources/deployment/bin/",
            environment_setup=True
        )
        super().__init__(config)
        
    async def verify_installation(self) -> bool:
        """Verify BV-BRC installation on macOS"""
        bv_brc_path = Path(self.config.installation_path)
        if not bv_brc_path.exists():
            raise BVBRCInstallationError(f"BV-BRC not found at {bv_brc_path}")
            
        # Test p3-all-genomes command
        try:
            result = await self.execute_p3_command("p3-all-genomes", ["--help"])
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"BV-BRC verification failed: {e}")
            return False
    
    async def execute_p3_command(self, command: str, args: List[str]) -> ToolResult:
        """Execute any p3-* command with proper environment setup"""
        full_command = [
            f"{self.config.executable_path}/{command}"
        ] + args
        
        # Set up BV-BRC environment
        env = os.environ.copy()
        env["PATH"] = f"{self.config.executable_path}:{env.get('PATH', '')}"
        
        return await self._execute_with_env(full_command, env)
    
    # Specific methods for workflow steps
    async def get_alphavirus_genomes(self) -> GenomeDataset:
        """Step 1: Download all Alphavirus genomes with metadata"""
        pass
        
    async def get_unique_proteins(self, genome_ids: List[str]) -> ProteinDataset:
        """Steps 3-4: Get unique protein MD5s"""
        pass
        
    async def get_protein_sequences(self, md5_list: List[str]) -> SequenceDataset:
        """Step 5: Get feature sequences for MD5s"""
        pass
        
    async def get_protein_annotations(self, md5_list: List[str]) -> AnnotationDataset:
        """Step 6: Get annotations for unique MD5 sequences"""
        pass
```

**BV-BRC Configuration**:
```yaml
# bv_brc_config.yml
bv_brc:
  installation_path: "/Applications/BV-BRC.app/"
  executable_path: "/Applications/BV-BRC.app/Contents/Resources/deployment/bin/"
  
  # Command configurations
  commands:
    p3_all_genomes:
      timeout: 300
      retry_attempts: 3
      output_format: "tsv"
      
    p3_get_genome_data:
      timeout: 600
      batch_size: 100
      fields: ["genome_id", "genome_length", "genome_name", "taxon_lineage_names"]
      
    p3_get_feature_data:
      timeout: 900
      batch_size: 50
      fields: ["patric_id", "aa_sequence_md5", "product", "gene", "refseq_locus_tag"]
      
  # Alphavirus-specific settings
  alphavirus:
    genus_filter: "Alphavirus"
    min_genome_length: 8000  # Minimum size threshold
    max_genome_length: 15000 # Maximum size threshold
    exclude_incomplete: true
```

### PHASE 2.3: ALPHAVIRUS WORKFLOW IMPLEMENTATION

#### Step 2.3.1: Workflow Step 1 - BV-BRC Data Acquisition
**File**: `nanobrain/library/workflows/alphavirus_analysis/steps/bv_brc_data_acquisition_step.py`

**Objective**: Implement Steps 1-7 of the workflow

**Detailed Implementation**:

##### Sub-step 1.1: Download Alphavirus Genomes
```python
async def download_alphavirus_genomes(self) -> GenomeDataset:
    """
    Step 1: From BV-BRC, download all genomes for Alphavirus genus
    Fields: genome_id, genome_length, genome_name
    
    Command: p3-all-genomes --eq "taxon_lineage_names,Alphavirus" 
             --attr genome_id,genome_length,genome_name
    """
    command_args = [
        "--eq", "taxon_lineage_names,Alphavirus",
        "--attr", "genome_id,genome_length,genome_name"
    ]
    
    result = await self.bv_brc_tool.execute_p3_command("p3-all-genomes", command_args)
    genomes = await self._parse_genome_data(result.stdout)
    
    self.logger.info(f"Downloaded {len(genomes)} Alphavirus genomes")
    return genomes
```

##### Sub-step 1.2: Filter Small Genomes
```python
async def filter_genomes_by_size(self, genomes: GenomeDataset) -> GenomeDataset:
    """
    Step 2: Filter all small genomes based on threshold
    
    Analysis of typical Alphavirus genome sizes:
    - Complete genomes: 11,000-12,000 bp
    - Minimum threshold: 8,000 bp (exclude fragments)
    - Maximum threshold: 15,000 bp (exclude contaminated assemblies)
    """
    min_length = self.config.min_genome_length
    max_length = self.config.max_genome_length
    
    filtered_genomes = []
    for genome in genomes:
        if min_length <= genome.genome_length <= max_length:
            filtered_genomes.append(genome)
        else:
            self.logger.debug(f"Filtered genome {genome.genome_id}: length {genome.genome_length}")
    
    self.logger.info(f"Filtered to {len(filtered_genomes)} genomes within size range")
    return filtered_genomes
```

##### Sub-step 1.3: Get Unique Protein MD5s
```python
async def get_unique_protein_md5s(self, genome_ids: List[str]) -> ProteinMD5Dataset:
    """
    Steps 3-4: Get every unique MD5 'patric_id aa_sequence_md5'
    Sort out identical proteins, create file with unique MD5
    
    Command: p3-get-feature-data --in genome_id,{genome_ids} 
             --attr patric_id,aa_sequence_md5 --eq feature_type,CDS
    """
    # Process genomes in batches to avoid command line length limits
    batch_size = self.config.genome_batch_size
    all_proteins = []
    
    for i in range(0, len(genome_ids), batch_size):
        batch = genome_ids[i:i+batch_size]
        genome_list = ",".join(batch)
        
        command_args = [
            "--in", f"genome_id,{genome_list}",
            "--attr", "patric_id,aa_sequence_md5",
            "--eq", "feature_type,CDS"
        ]
        
        result = await self.bv_brc_tool.execute_p3_command("p3-get-feature-data", command_args)
        batch_proteins = await self._parse_protein_data(result.stdout)
        all_proteins.extend(batch_proteins)
    
    # Remove duplicates by MD5
    unique_md5s = {}
    for protein in all_proteins:
        if protein.aa_sequence_md5 not in unique_md5s:
            unique_md5s[protein.aa_sequence_md5] = protein
    
    self.logger.info(f"Found {len(unique_md5s)} unique proteins from {len(all_proteins)} total")
    return list(unique_md5s.values())
```

##### Sub-step 1.4: Get Feature Sequences
```python
async def get_feature_sequences(self, md5_list: List[str]) -> SequenceDataset:
    """
    Step 5: Get feature sequence for MD5 (-r "md5 sequence")
    
    Command: p3-get-feature-sequence --in aa_sequence_md5,{md5_list} 
             --attr aa_sequence_md5,aa_sequence
    """
    sequences = []
    batch_size = self.config.md5_batch_size
    
    for i in range(0, len(md5_list), batch_size):
        batch = md5_list[i:i+batch_size]
        md5_string = ",".join(batch)
        
        command_args = [
            "--in", f"aa_sequence_md5,{md5_string}",
            "--attr", "aa_sequence_md5,aa_sequence"
        ]
        
        result = await self.bv_brc_tool.execute_p3_command("p3-get-feature-sequence", command_args)
        batch_sequences = await self._parse_sequence_data(result.stdout)
        sequences.extend(batch_sequences)
    
    return sequences
```

##### Sub-step 1.5: Get Annotations
```python
async def get_protein_annotations(self, md5_list: List[str]) -> AnnotationDataset:
    """
    Step 6: Get annotations for unique MD5 sequences
    
    Command: p3-get-feature-data --in aa_sequence_md5,{md5_list}
             --attr aa_sequence_md5,product,gene,refseq_locus_tag,go,ec,pathway
    """
    annotations = []
    batch_size = self.config.md5_batch_size
    
    for i in range(0, len(md5_list), batch_size):
        batch = md5_list[i:i+batch_size]
        md5_string = ",".join(batch)
        
        command_args = [
            "--in", f"aa_sequence_md5,{md5_string}",
            "--attr", "aa_sequence_md5,product,gene,refseq_locus_tag,go,ec,pathway"
        ]
        
        result = await self.bv_brc_tool.execute_p3_command("p3-get-feature-data", command_args)
        batch_annotations = await self._parse_annotation_data(result.stdout)
        annotations.extend(batch_annotations)
    
    return annotations
```

##### Sub-step 1.6: Create Annotated FASTA
```python
async def create_annotated_fasta(self, sequences: SequenceDataset, 
                               annotations: AnnotationDataset) -> FASTADataset:
    """
    Step 7: Create FASTA file of every protein with detailed annotation
    
    FASTA header format:
    >{md5}|{gene}|{product}|{refseq_locus_tag}|{pathway}
    {sequence}
    """
    # Merge sequences with annotations
    annotation_map = {ann.aa_sequence_md5: ann for ann in annotations}
    
    fasta_entries = []
    for seq in sequences:
        annotation = annotation_map.get(seq.aa_sequence_md5)
        if annotation:
            header = self._format_fasta_header(seq.aa_sequence_md5, annotation)
            fasta_entries.append(FASTAEntry(
                header=header,
                sequence=seq.aa_sequence,
                md5=seq.aa_sequence_md5,
                annotation=annotation
            ))
    
    return fasta_entries

def _format_fasta_header(self, md5: str, annotation: ProteinAnnotation) -> str:
    """Format FASTA header with detailed annotation"""
    components = [
        md5,
        annotation.gene or "unknown",
        annotation.product or "hypothetical protein",
        annotation.refseq_locus_tag or "no_tag",
        annotation.pathway or "unknown_pathway"
    ]
    return "|".join(components)
```

#### Step 2.3.2: Workflow Step 8 - Annotation Mapping
**File**: `nanobrain/library/workflows/alphavirus_analysis/steps/annotation_mapping_step.py`

**Objective**: Handle annotation inconsistencies using ICTV resources

**Implementation Details**:
```python
class AnnotationMappingStep(BioinformaticsStep):
    """
    Step 8: Handle annotation inconsistencies using ICTV mapping
    
    Issues to address:
    - Inconsistent protein names across different genomes
    - Different naming conventions (e.g., "capsid" vs "structural protein C")
    - Missing or incomplete annotations
    - Need for standardized protein classification
    """
    
    async def create_ictv_mapping(self) -> ICTVMapping:
        """
        Create mapping based on ICTV genome schematics
        
        ICTV Alphavirus genome organization:
        5' - nsP1 - nsP2 - nsP3 - nsP4 - Capsid - E3 - E2 - 6K - E1 - 3'
        
        Non-structural proteins: nsP1, nsP2, nsP3, nsP4
        Structural proteins: Capsid, E3, E2, 6K, E1
        """
        return ICTVMapping({
            "non_structural": {
                "nsP1": ["nonstructural protein 1", "nsp1", "replicase"],
                "nsP2": ["nonstructural protein 2", "nsp2", "protease"],
                "nsP3": ["nonstructural protein 3", "nsp3"],
                "nsP4": ["nonstructural protein 4", "nsp4", "RNA polymerase"]
            },
            "structural": {
                "capsid": ["capsid protein", "structural protein C", "core protein"],
                "E3": ["envelope protein E3", "glycoprotein E3"],
                "E2": ["envelope protein E2", "glycoprotein E2"],
                "6K": ["6K protein", "small membrane protein"],
                "E1": ["envelope protein E1", "glycoprotein E1"]
            }
        })
    
    async def standardize_annotations(self, annotations: AnnotationDataset) -> StandardizedAnnotations:
        """Standardize protein annotations using ICTV mapping"""
        mapping = await self.create_ictv_mapping()
        standardized = []
        
        for annotation in annotations:
            standard_name = self._map_to_standard_name(annotation.product, mapping)
            protein_class = self._classify_protein(standard_name, mapping)
            
            standardized.append(StandardizedAnnotation(
                original_annotation=annotation,
                standard_name=standard_name,
                protein_class=protein_class,
                confidence=self._calculate_mapping_confidence(annotation, standard_name)
            ))
        
        return standardized
    
    async def generate_genome_schematic(self, proteins: List[StandardizedAnnotation]) -> GenomeSchematic:
        """Generate genome organization schematic"""
        # Order proteins based on typical Alphavirus genome organization
        ordered_proteins = self._order_proteins_by_genome_position(proteins)
        
        return GenomeSchematic(
            proteins=ordered_proteins,
            genome_organization="5'-nsP1-nsP2-nsP3-nsP4-Capsid-E3-E2-6K-E1-3'",
            protein_boundaries=self._calculate_protein_boundaries(ordered_proteins)
        )
```

#### Step 2.3.3: Workflow Steps 9-11 - Sequence Curation
**File**: `nanobrain/library/workflows/alphavirus_analysis/steps/sequence_curation_step.py`

**Implementation Details**:
```python
class SequenceCurationStep(BioinformaticsStep):
    """
    Steps 9-11: Sequence curation and quality control
    
    Step 9: Create FASTA file of selected proteins
    Step 10: Analyze length distribution
    Step 11: Identify mangled sequences
    """
    
    async def analyze_length_distribution(self, sequences: SequenceDataset) -> LengthAnalysis:
        """
        Step 10: Analyze protein length distribution
        
        Expected Alphavirus protein lengths:
        - nsP1: ~535 aa
        - nsP2: ~800 aa  
        - nsP3: ~530 aa
        - nsP4: ~610 aa
        - Capsid: ~260 aa
        - E3: ~60 aa
        - E2: ~420 aa
        - 6K: ~55 aa
        - E1: ~440 aa
        """
        length_stats = {}
        
        for protein_class in ["nsP1", "nsP2", "nsP3", "nsP4", "capsid", "E3", "E2", "6K", "E1"]:
            class_sequences = [s for s in sequences if s.protein_class == protein_class]
            lengths = [len(s.sequence) for s in class_sequences]
            
            if lengths:
                length_stats[protein_class] = LengthStatistics(
                    mean=np.mean(lengths),
                    median=np.median(lengths),
                    std=np.std(lengths),
                    min=min(lengths),
                    max=max(lengths),
                    outliers=self._identify_length_outliers(lengths, protein_class)
                )
        
        return LengthAnalysis(statistics=length_stats)
    
    async def identify_mangled_sequences(self, sequences: SequenceDataset) -> CurationReport:
        """
        Step 11: Identify mangled or problematic sequences
        
        Issues to detect:
        - Sequences with unusual amino acid composition
        - Sequences with stop codons (*)
        - Sequences with ambiguous amino acids (X)
        - Sequences that are too short or too long
        - Sequences with unusual N-terminal or C-terminal regions
        """
        problematic_sequences = []
        
        for seq in sequences:
            issues = []
            
            # Check for stop codons
            if '*' in seq.sequence:
                issues.append("Contains stop codons")
            
            # Check for ambiguous amino acids
            ambiguous_count = seq.sequence.count('X')
            if ambiguous_count > len(seq.sequence) * 0.05:  # >5% ambiguous
                issues.append(f"High ambiguous AA content: {ambiguous_count}")
            
            # Check length against expected ranges
            expected_range = self._get_expected_length_range(seq.protein_class)
            if not (expected_range[0] <= len(seq.sequence) <= expected_range[1]):
                issues.append(f"Length outside expected range: {len(seq.sequence)}")
            
            # Check amino acid composition
            composition_issues = self._check_amino_acid_composition(seq.sequence)
            issues.extend(composition_issues)
            
            if issues:
                problematic_sequences.append(ProblematicSequence(
                    sequence=seq,
                    issues=issues,
                    severity=self._assess_issue_severity(issues)
                ))
        
        return CurationReport(
            total_sequences=len(sequences),
            problematic_sequences=problematic_sequences,
            curation_recommendations=self._generate_curation_recommendations(problematic_sequences)
        )
```

#### Step 2.3.4: Workflow Step 12 - MMseqs2 Clustering
**File**: `nanobrain/library/workflows/alphavirus_analysis/steps/clustering_step.py`

**Implementation Details**:
```python
class ClusteringStep(BioinformaticsStep):
    """
    Step 12: Use MMseqs2 to build protein clusters
    
    Equivalent to fasta-cluster-pssm.pl functionality
    Parameters (-f, -n, -c) are critical for short well-conserved regions
    """
    
    async def perform_mmseqs_clustering(self, sequences: SequenceDataset) -> ClusteringResult:
        """
        Execute MMseqs2 clustering with optimized parameters
        
        Key parameters:
        -f: Coverage fraction (0.8 = 80% coverage required)
        -n: Sensitivity (7.5 for high sensitivity)
        -c: Minimum sequence identity (0.7 = 70% identity)
        
        Command: mmseqs easy-cluster input.fasta clusters tmp 
                 --min-seq-id 0.7 -c 0.8 --cov-mode 1 -s 7.5
        """
        # Create temporary input file
        input_fasta = await self._create_clustering_input(sequences)
        
        mmseqs_params = MMseqs2Parameters(
            min_seq_id=0.7,        # -c parameter: minimum identity
            coverage=0.8,          # -f parameter: coverage fraction
            sensitivity=7.5,       # -n parameter: sensitivity
            coverage_mode=1,       # Bidirectional coverage
            cluster_mode=0,        # Greedy clustering
            prefer_short_conserved=True  # Prioritize short well-conserved regions
        )
        
        result = await self.mmseqs_tool.easy_cluster(
            input_file=input_fasta,
            output_prefix="alphavirus_clusters",
            parameters=mmseqs_params
        )
        
        # Parse clustering results
        clusters = await self._parse_mmseqs_output(result)
        
        # Analyze cluster quality
        cluster_analysis = await self._analyze_cluster_quality(clusters)
        
        return ClusteringResult(
            clusters=clusters,
            analysis=cluster_analysis,
            parameters=mmseqs_params
        )
    
    async def _analyze_cluster_quality(self, clusters: List[ProteinCluster]) -> ClusterAnalysis:
        """
        Analyze cluster quality focusing on short well-conserved regions
        """
        analysis = ClusterAnalysis()
        
        for cluster in clusters:
            # Identify conserved regions within cluster
            conserved_regions = await self._identify_conserved_regions(cluster)
            
            # Prioritize shorter well-conserved regions
            short_conserved = [r for r in conserved_regions if r.length <= 50 and r.conservation > 0.9]
            
            cluster.quality_metrics = ClusterQualityMetrics(
                conserved_regions=conserved_regions,
                short_well_conserved=short_conserved,
                overall_conservation=np.mean([r.conservation for r in conserved_regions]),
                cluster_coherence=self._calculate_cluster_coherence(cluster)
            )
        
        return analysis
```

#### Step 2.3.5: Workflow Step 13 - Multiple Sequence Alignment
**File**: `nanobrain/library/workflows/alphavirus_analysis/steps/alignment_step.py`

**Implementation Details**:
```python
class AlignmentStep(BioinformaticsStep):
    """
    Step 13: Add alignment before constructing PSSM
    
    Perform multiple sequence alignment for each protein cluster
    to prepare for PSSM generation
    """
    
    async def align_protein_clusters(self, clusters: List[ProteinCluster]) -> AlignmentResult:
        """
        Perform multiple sequence alignment for each cluster using MUSCLE
        """
        aligned_clusters = []
        
        for cluster in clusters:
            if len(cluster.sequences) >= 3:  # Minimum sequences for meaningful alignment
                alignment = await self._align_cluster_sequences(cluster)
                aligned_cluster = AlignedCluster(
                    original_cluster=cluster,
                    alignment=alignment,
                    alignment_quality=await self._assess_alignment_quality(alignment)
                )
                aligned_clusters.append(aligned_cluster)
            else:
                self.logger.warning(f"Cluster {cluster.id} has too few sequences for alignment")
        
        return AlignmentResult(aligned_clusters=aligned_clusters)
    
    async def _align_cluster_sequences(self, cluster: ProteinCluster) -> MultipleSeqAlignment:
        """Align sequences within a cluster using MUSCLE"""
        # Create temporary FASTA file for cluster sequences
        cluster_fasta = await self._create_cluster_fasta(cluster)
        
        # Run MUSCLE alignment
        muscle_params = MUSCLEParameters(
            max_iterations=16,
            diagonal_optimization=True,
            gap_open_penalty=-12,
            gap_extend_penalty=-1
        )
        
        alignment_result = await self.muscle_tool.align_sequences(
            input_file=cluster_fasta,
            parameters=muscle_params
        )
        
        return alignment_result.alignment
    
    async def _assess_alignment_quality(self, alignment: MultipleSeqAlignment) -> AlignmentQuality:
        """Assess quality of multiple sequence alignment"""
        # Calculate conservation scores
        conservation_scores = []
        for pos in range(alignment.get_alignment_length()):
            column = [record.seq[pos] for record in alignment]
            conservation = self._calculate_position_conservation(column)
            conservation_scores.append(conservation)
        
        return AlignmentQuality(
            mean_conservation=np.mean(conservation_scores),
            highly_conserved_positions=len([s for s in conservation_scores if s > 0.8]),
            alignment_length=alignment.get_alignment_length(),
            sequence_count=len(alignment),
            gap_percentage=self._calculate_gap_percentage(alignment)
        )
```

#### Step 2.3.6: Workflow Step 14 - PSSM Analysis
**File**: `nanobrain/library/workflows/alphavirus_analysis/steps/pssm_analysis_step.py`

**Implementation Details**:
```python
class PSSMAnalysisStep(BioinformaticsStep):
    """
    Step 14: Analyze curation report and generate PSSM matrices
    
    Generate position-specific scoring matrices from aligned clusters
    and create comprehensive curation report
    """
    
    async def generate_pssm_matrices(self, aligned_clusters: List[AlignedCluster]) -> PSSMResult:
        """Generate PSSM matrices from aligned protein clusters"""
        pssm_matrices = []
        
        for aligned_cluster in aligned_clusters:
            if aligned_cluster.alignment_quality.mean_conservation > 0.5:
                pssm = await self._create_pssm_matrix(aligned_cluster)
                pssm_matrices.append(pssm)
            else:
                self.logger.warning(f"Cluster {aligned_cluster.original_cluster.id} has low conservation")
        
        return PSSMResult(matrices=pssm_matrices)
    
    async def _create_pssm_matrix(self, aligned_cluster: AlignedCluster) -> PSSMMatrix:
        """Create PSSM matrix from aligned cluster"""
        alignment = aligned_cluster.alignment
        
        # Calculate amino acid frequencies at each position
        matrix = np.zeros((20, alignment.get_alignment_length()))
        aa_to_index = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        
        for pos in range(alignment.get_alignment_length()):
            column = [record.seq[pos] for record in alignment]
            
            # Count amino acids (excluding gaps)
            aa_counts = Counter([aa for aa in column if aa in aa_to_index])
            total_valid = sum(aa_counts.values())
            
            if total_valid > 0:
                for aa, count in aa_counts.items():
                    matrix[aa_to_index[aa], pos] = count / total_valid
        
        # Convert to log-odds scores
        background_freq = np.array([0.05] * 20)  # Uniform background
        log_odds_matrix = np.log2((matrix + 0.01) / (background_freq.reshape(-1, 1) + 0.01))
        
        return PSSMMatrix(
            matrix=log_odds_matrix,
            alphabet='ACDEFGHIKLMNPQRSTVWY',
            cluster_id=aligned_cluster.original_cluster.id,
            protein_class=aligned_cluster.original_cluster.protein_class,
            conservation_profile=aligned_cluster.alignment_quality
        )
    
    async def generate_curation_report(self, workflow_data: WorkflowData) -> CurationReport:
        """
        Generate comprehensive curation report analyzing entire workflow
        """
        report = CurationReport(
            workflow_summary=WorkflowSummary(
                total_genomes_downloaded=len(workflow_data.original_genomes),
                genomes_after_filtering=len(workflow_data.filtered_genomes),
                unique_proteins=len(workflow_data.unique_proteins),
                clusters_generated=len(workflow_data.clusters),
                pssm_matrices_created=len(workflow_data.pssm_matrices)
            ),
            
            quality_assessment=QualityAssessment(
                genome_size_distribution=workflow_data.genome_size_stats,
                protein_length_analysis=workflow_data.length_analysis,
                annotation_consistency=workflow_data.annotation_mapping_quality,
                clustering_effectiveness=workflow_data.clustering_analysis,
                alignment_quality=workflow_data.alignment_quality_stats
            ),
            
            recommendations=self._generate_recommendations(workflow_data),
            
            data_files=DataFiles(
                filtered_genomes="data/alphavirus_filtered_genomes.tsv",
                unique_proteins="data/alphavirus_unique_proteins.fasta",
                cluster_assignments="data/alphavirus_clusters.tsv",
                pssm_matrices="data/alphavirus_pssm_matrices.json",
                curation_report="data/alphavirus_curation_report.html"
            )
        )
        
        return report
```

### PHASE 2.4: WORKFLOW ORCHESTRATION

#### Main Workflow Class
**File**: `nanobrain/library/workflows/alphavirus_analysis/alphavirus_workflow.py`

```python
class AlphavirusWorkflow(Workflow):
    """
    Main orchestrator for Alphavirus protein analysis workflow
    
    Executes all 14 steps in sequence with proper error handling,
    checkpointing, and progress tracking
    """
    
    def __init__(self, config: AlphavirusWorkflowConfig):
        super().__init__(config)
        self.steps = self._initialize_workflow_steps()
        
    async def execute_full_workflow(self, input_params: WorkflowInput) -> WorkflowResult:
        """Execute complete 14-step Alphavirus analysis workflow"""
        
        workflow_data = WorkflowData()
        
        try:
            # Steps 1-7: BV-BRC Data Acquisition
            self.logger.info("Starting BV-BRC data acquisition (Steps 1-7)")
            acquisition_result = await self.steps['data_acquisition'].execute(input_params)
            workflow_data.update_from_acquisition(acquisition_result)
            
            # Step 8: Annotation Mapping
            self.logger.info("Starting annotation mapping (Step 8)")
            mapping_result = await self.steps['annotation_mapping'].execute(workflow_data)
            workflow_data.update_from_mapping(mapping_result)
            
            # Steps 9-11: Sequence Curation
            self.logger.info("Starting sequence curation (Steps 9-11)")
            curation_result = await self.steps['sequence_curation'].execute(workflow_data)
            workflow_data.update_from_curation(curation_result)
            
            # Step 12: Clustering
            self.logger.info("Starting MMseqs2 clustering (Step 12)")
            clustering_result = await self.steps['clustering'].execute(workflow_data)
            workflow_data.update_from_clustering(clustering_result)
            
            # Step 13: Alignment
            self.logger.info("Starting multiple sequence alignment (Step 13)")
            alignment_result = await self.steps['alignment'].execute(workflow_data)
            workflow_data.update_from_alignment(alignment_result)
            
            # Step 14: PSSM Analysis
            self.logger.info("Starting PSSM analysis and curation report (Step 14)")
            pssm_result = await self.steps['pssm_analysis'].execute(workflow_data)
            workflow_data.update_from_pssm(pssm_result)
            
            # Generate final results
            final_result = WorkflowResult(
                success=True,
                workflow_data=workflow_data,
                execution_time=self.get_execution_time(),
                output_files=self._collect_output_files(workflow_data)
            )
            
            self.logger.info("Alphavirus workflow completed successfully")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            return WorkflowResult(
                success=False,
                error=str(e),
                partial_data=workflow_data
            )
```

### PHASE 2.5: CONFIGURATION AND TESTING

#### Main Configuration File
**File**: `nanobrain/library/workflows/alphavirus_analysis/config/AlphavirusWorkflow.yml`

```yaml
# Alphavirus Analysis Workflow Configuration
workflow:
  name: "alphavirus_analysis"
  version: "1.0.0"
  description: "Comprehensive Alphavirus protein analysis using BV-BRC and MMseqs2"
  
  # BV-BRC configuration
  bv_brc:
    installation_path: "/Applications/BV-BRC.app/"
    executable_path: "/Applications/BV-BRC.app/Contents/Resources/deployment/bin/"
    
    # Genome filtering parameters
    genome_filters:
      genus: "Alphavirus"
      min_length: 8000
      max_length: 15000
      exclude_incomplete: true
      
    # Batch processing settings
    batch_sizes:
      genome_batch: 100
      md5_batch: 50
      
  # MMseqs2 clustering parameters
  clustering:
    min_seq_id: 0.7      # -c parameter
    coverage: 0.8        # -f parameter  
    sensitivity: 7.5     # -n parameter
    coverage_mode: 1     # Bidirectional
    prefer_short_conserved: true
    
  # MUSCLE alignment parameters
  alignment:
    max_iterations: 16
    gap_open_penalty: -12
    gap_extend_penalty: -1
    
  # Quality control thresholds
  quality_control:
    min_cluster_size: 3
    min_alignment_conservation: 0.5
    max_ambiguous_aa_percent: 5
    
  # Output settings
  output:
    base_directory: "data/alphavirus_analysis"
    create_html_report: true
    create_summary_plots: true
```

#### Comprehensive Test Suite
**File**: `tests/test_alphavirus_workflow.py`

```python
class TestAlphavirusWorkflow:
    """Comprehensive test suite for Alphavirus workflow"""
    
    @pytest.fixture
    async def workflow(self):
        config = AlphavirusWorkflowConfig.from_file("config/AlphavirusWorkflow.yml")
        return AlphavirusWorkflow(config)
    
    @pytest.mark.asyncio
    async def test_bv_brc_tool_installation(self):
        """Test BV-BRC tool installation and verification"""
        bv_brc_tool = BVBRCTool()
        assert await bv_brc_tool.verify_installation()
        
    @pytest.mark.asyncio
    async def test_step_01_download_genomes(self, workflow):
        """Test Step 1: Download Alphavirus genomes"""
        genomes = await workflow.steps['data_acquisition'].download_alphavirus_genomes()
        assert len(genomes) > 0
        assert all(g.genome_length > 0 for g in genomes)
        
    @pytest.mark.asyncio
    async def test_step_02_filter_genomes(self, workflow):
        """Test Step 2: Filter genomes by size"""
        # Test with known genome data
        test_genomes = [
            GenomeData(genome_id="test1", genome_length=11500, genome_name="Test1"),
            GenomeData(genome_id="test2", genome_length=5000, genome_name="Test2"),  # Too small
            GenomeData(genome_id="test3", genome_length=20000, genome_name="Test3")  # Too large
        ]
        
        filtered = await workflow.steps['data_acquisition'].filter_genomes_by_size(test_genomes)
        assert len(filtered) == 1
        assert filtered[0].genome_id == "test1"
        
    @pytest.mark.asyncio  
    async def test_full_workflow_integration(self, workflow):
        """Test complete workflow execution"""
        input_params = WorkflowInput(
            target_genus="Alphavirus",
            test_mode=True,
            max_genomes=10  # Limit for testing
        )
        
        result = await workflow.execute_full_workflow(input_params)
        assert result.success
        assert result.workflow_data.pssm_matrices is not None
        assert len(result.output_files) > 0
```

## USER REQUIREMENTS CLARIFIED ✅

Based on user feedback, the following requirements have been confirmed:

### 1. **Authentication & Data Verification** ✅
- **Requirement**: Anonymous authorization for BV-BRC access
- **Data Validation**: Ensure downloaded data contains actual content, not just headers
- **Implementation**: Verify data completeness after each download step

### 2. **Data Volume Expectations** ✅
- **Expected Size**: Hundreds of MB (manageable size)
- **Processing Strategy**: Standard processing, no need for streaming/chunked processing
- **Storage**: Monitor disk space and warn at 1GB remaining

### 3. **ICTV Integration** ✅
- **Approach**: Automated ICTV data integration
- **Method**: Fetch ICTV genome schematics programmatically
- **Fallback**: Static mapping files as backup

### 4. **Clustering Parameters** ✅
- **Default Settings**: Global MMseqs2 parameters (-f, -n, -c) from configuration
- **Fine-tuning**: Protein-specific parameter optimization when needed
- **Flexibility**: Allow per-protein class customization

### 5. **Output Format** ✅
- **Format**: Match [Viral_PSSM.json](https://github.com/jimdavis1/Viral_Annotation/blob/main/Viral_PSSM.json) exactly
- **Enhancement**: Include PubMed article references from RAG/PubMed search
- **Integration**: Combine PSSM data with literature references

### 6. **Error Handling Strategy** ✅
- **Retry Policy**: Up to 3 attempts for partial failures
- **User Interaction**: Ask user after 3 failed attempts
- **Graceful Degradation**: Continue with available data when possible

### 7. **Resource Management** ✅
- **Disk Space Monitoring**: Continuous monitoring during workflow
- **Warning Threshold**: Alert when 1GB disk space remaining
- **Workflow Pause**: Automatically pause workflow at low disk space
- **User Notification**: Prompt user for action on resource constraints

## UPDATED IMPLEMENTATION PLAN

### **PHASE 2A: ENHANCED EXTERNAL TOOL FRAMEWORK**

#### **Step 2A.1: Resource Monitoring System**
**File**: `nanobrain/core/resource_monitor.py`

**Objective**: Monitor system resources and manage workflow execution

**Implementation Details**:
```python
class ResourceMonitor:
    """
    Monitor system resources during workflow execution.
    
    Features:
    - Disk space monitoring with configurable thresholds
    - Memory usage tracking
    - Automatic workflow pausing on resource constraints
    - User notification system for resource issues
    """
    
    def __init__(self, config: ResourceMonitorConfig):
        self.config = config
        self.disk_warning_threshold = config.disk_warning_gb  # 1GB default
        self.monitoring_enabled = True
        self.logger = get_logger("resource_monitor")
        
    async def check_disk_space(self, path: str = ".") -> DiskSpaceInfo:
        """Check available disk space at specified path"""
        statvfs = os.statvfs(path)
        free_bytes = statvfs.f_frsize * statvfs.f_bavail
        free_gb = free_bytes / (1024**3)
        
        return DiskSpaceInfo(
            free_bytes=free_bytes,
            free_gb=free_gb,
            warning_triggered=free_gb < self.disk_warning_threshold,
            critical_triggered=free_gb < 0.5  # Critical at 500MB
        )
    
    async def monitor_workflow_resources(self, workflow_instance) -> None:
        """Continuously monitor resources during workflow execution"""
        while self.monitoring_enabled and workflow_instance.is_running():
            disk_info = await self.check_disk_space(workflow_instance.working_directory)
            
            if disk_info.critical_triggered:
                await self._handle_critical_disk_space(workflow_instance, disk_info)
            elif disk_info.warning_triggered:
                await self._handle_disk_space_warning(workflow_instance, disk_info)
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _handle_disk_space_warning(self, workflow_instance, disk_info: DiskSpaceInfo) -> None:
        """Handle disk space warning (1GB threshold)"""
        warning_message = (
            f"⚠️ LOW DISK SPACE WARNING ⚠️\n"
            f"Available space: {disk_info.free_gb:.2f} GB\n"
            f"Workflow will pause if space drops below 500MB\n"
            f"Consider freeing up disk space to continue safely."
        )
        
        self.logger.warning(warning_message)
        await workflow_instance.notify_user(warning_message, severity="warning")
    
    async def _handle_critical_disk_space(self, workflow_instance, disk_info: DiskSpaceInfo) -> None:
        """Handle critical disk space (500MB threshold) - pause workflow"""
        critical_message = (
            f"🛑 CRITICAL DISK SPACE - WORKFLOW PAUSED 🛑\n"
            f"Available space: {disk_info.free_gb:.2f} GB\n"
            f"Workflow has been paused to prevent data loss.\n"
            f"Please free up disk space and resume manually."
        )
        
        self.logger.critical(critical_message)
        await workflow_instance.pause_workflow("critical_disk_space")
        await workflow_instance.notify_user(critical_message, severity="critical")
```

#### **Step 2A.2: Enhanced BV-BRC Tool with Data Verification**
**File**: `nanobrain/library/tools/bioinformatics/bv_brc_tool.py`

**Objective**: BV-BRC integration with anonymous access and data verification

**Implementation Details**:
```python
class BVBRCTool(BioinformaticsExternalTool):
    """
    Enhanced BV-BRC CLI tool wrapper with data verification and anonymous access.
    
    Features:
    - Anonymous authentication (no login required)
    - Data completeness verification after downloads
    - Retry mechanism with exponential backoff
    - Integration with resource monitoring
    """
    
    def __init__(self, config: Optional[BVBRCConfig] = None):
        config = config or BVBRCConfig(
            tool_name="bv_brc",
            installation_path="/Applications/BV-BRC.app/",
            executable_path="/Applications/BV-BRC.app/Contents/Resources/deployment/bin/",
            anonymous_access=True,  # No login required
            data_verification_enabled=True,
            retry_attempts=3
        )
        super().__init__(config)
        self.resource_monitor = ResourceMonitor()
        
    async def verify_installation(self) -> bool:
        """Verify BV-BRC installation and test data access"""
        # Check installation path
        bv_brc_path = Path(self.config.installation_path)
        if not bv_brc_path.exists():
            raise BVBRCInstallationError(f"BV-BRC not found at {bv_brc_path}")
        
        # Test anonymous access with a simple query
        try:
            test_result = await self.execute_p3_command("p3-all-genomes", [
                "--eq", "genome_id,511145.12",  # E. coli test genome
                "--attr", "genome_id,genome_name",
                "--limit", "1"
            ])
            
            if test_result.returncode == 0 and test_result.stdout:
                # Verify data is not just headers
                lines = test_result.stdout.decode().strip().split('\n')
                if len(lines) > 1:  # Header + at least one data line
                    self.logger.info("BV-BRC anonymous access verified with actual data")
                    return True
                else:
                    raise BVBRCDataError("BV-BRC returned only headers, no actual data")
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"BV-BRC verification failed: {e}")
            return False
    
    async def download_alphavirus_genomes_with_verification(self) -> GenomeDataset:
        """
        Step 1: Download Alphavirus genomes with data verification
        
        Enhanced with:
        - Data completeness verification
        - Retry mechanism
        - Progress tracking
        """
        for attempt in range(self.config.retry_attempts):
            try:
                self.logger.info(f"Downloading Alphavirus genomes (attempt {attempt + 1})")
                
                # Check disk space before download
                disk_info = await self.resource_monitor.check_disk_space()
                if disk_info.warning_triggered:
                    self.logger.warning(f"Low disk space: {disk_info.free_gb:.2f} GB remaining")
                
                command_args = [
                    "--eq", "taxon_lineage_names,Alphavirus",
                    "--attr", "genome_id,genome_length,genome_name,taxon_lineage_names"
                ]
                
                result = await self.execute_p3_command("p3-all-genomes", command_args)
                
                if result.returncode != 0:
                    raise BVBRCCommandError(f"p3-all-genomes failed: {result.stderr}")
                
                # Verify data completeness
                genomes = await self._parse_and_verify_genome_data(result.stdout)
                
                if len(genomes) == 0:
                    raise BVBRCDataError("No Alphavirus genomes found - data may be incomplete")
                
                # Additional verification: check for reasonable genome count
                if len(genomes) < 10:  # Expect at least 10 Alphavirus genomes
                    self.logger.warning(f"Only {len(genomes)} Alphavirus genomes found - unusually low")
                
                self.logger.info(f"Successfully downloaded {len(genomes)} Alphavirus genomes")
                return genomes
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise BVBRCDownloadError(f"Failed to download genomes after {self.config.retry_attempts} attempts: {e}")
    
    async def _parse_and_verify_genome_data(self, raw_data: bytes) -> List[GenomeData]:
        """Parse genome data and verify completeness"""
        if not raw_data:
            raise BVBRCDataError("Empty response from BV-BRC")
        
        lines = raw_data.decode().strip().split('\n')
        
        if len(lines) < 2:  # Must have header + at least one data line
            raise BVBRCDataError("BV-BRC response contains only headers, no actual data")
        
        # Parse header
        header = lines[0].split('\t')
        expected_fields = ['genome_id', 'genome_length', 'genome_name']
        
        for field in expected_fields:
            if field not in header:
                raise BVBRCDataError(f"Missing expected field '{field}' in BV-BRC response")
        
        # Parse data lines
        genomes = []
        for line_num, line in enumerate(lines[1:], 2):
            try:
                fields = line.split('\t')
                if len(fields) != len(header):
                    self.logger.warning(f"Line {line_num}: field count mismatch")
                    continue
                
                data_dict = dict(zip(header, fields))
                
                # Verify required fields are not empty
                if not data_dict.get('genome_id') or not data_dict.get('genome_length'):
                    self.logger.warning(f"Line {line_num}: missing required data")
                    continue
                
                genome = GenomeData(
                    genome_id=data_dict['genome_id'],
                    genome_length=int(data_dict['genome_length']),
                    genome_name=data_dict.get('genome_name', 'Unknown'),
                    taxon_lineage=data_dict.get('taxon_lineage_names', '')
                )
                
                genomes.append(genome)
                
            except (ValueError, KeyError) as e:
                self.logger.warning(f"Line {line_num}: parsing error - {e}")
                continue
        
        if len(genomes) == 0:
            raise BVBRCDataError("No valid genome data could be parsed")
        
        return genomes
```

#### **Step 2A.3: PubMed Integration System**
**File**: `nanobrain/library/tools/bioinformatics/pubmed_integration_tool.py`

**Objective**: Integrate PubMed literature search with protein analysis

**Implementation Details**:
```python
class PubMedIntegrationTool(BioinformaticsExternalTool):
    """
    PubMed integration for literature search and reference collection.
    
    Features:
    - Automated PubMed searches for protein functions
    - RAG-based literature analysis
    - Reference formatting for Viral_PSSM.json output
    - Caching to avoid redundant searches
    """
    
    def __init__(self, config: Optional[PubMedConfig] = None):
        config = config or PubMedConfig(
            tool_name="pubmed_integration",
            email="nanobrain@example.com",  # Required by NCBI
            max_results_per_search=10,
            cache_enabled=True
        )
        super().__init__(config)
        self.entrez_email = config.email
        self.search_cache = {}
        
    async def search_protein_literature(self, protein_annotation: str, organism: str = "Alphavirus") -> LiteratureResult:
        """
        Search PubMed for literature related to specific protein annotation.
        
        Args:
            protein_annotation: Protein function/name (e.g., "capsid protein")
            organism: Organism context for search
            
        Returns:
            LiteratureResult with relevant papers and references
        """
        # Create cache key
        cache_key = f"{protein_annotation}_{organism}".lower().replace(" ", "_")
        
        if cache_key in self.search_cache:
            self.logger.debug(f"Using cached literature for {protein_annotation}")
            return self.search_cache[cache_key]
        
        try:
            # Construct search query
            search_terms = self._construct_search_terms(protein_annotation, organism)
            
            # Execute PubMed search
            search_results = await self._execute_pubmed_search(search_terms)
            
            # Process and rank results
            literature_result = await self._process_search_results(search_results, protein_annotation)
            
            # Cache results
            self.search_cache[cache_key] = literature_result
            
            return literature_result
            
        except Exception as e:
            self.logger.warning(f"PubMed search failed for {protein_annotation}: {e}")
            return LiteratureResult(
                protein_annotation=protein_annotation,
                references=[],
                search_successful=False,
                error_message=str(e)
            )
    
    def _construct_search_terms(self, protein_annotation: str, organism: str) -> str:
        """Construct optimized PubMed search terms"""
        # Base protein terms
        protein_terms = protein_annotation.lower()
        
        # Add organism context
        organism_terms = f"({organism} OR alphavirus OR togavirus)"
        
        # Combine with relevant keywords
        search_query = f"({protein_terms}) AND {organism_terms} AND (structure OR function OR domain OR motif)"
        
        return search_query
    
    async def _execute_pubmed_search(self, search_terms: str) -> List[Dict[str, Any]]:
        """Execute PubMed search using Entrez utilities"""
        from Bio import Entrez
        
        Entrez.email = self.entrez_email
        
        try:
            # Search for PMIDs
            handle = Entrez.esearch(
                db="pubmed",
                term=search_terms,
                retmax=self.config.max_results_per_search,
                sort="relevance"
            )
            search_results = Entrez.read(handle)
            handle.close()
            
            pmids = search_results["IdList"]
            
            if not pmids:
                return []
            
            # Fetch article details
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(pmids),
                rettype="medline",
                retmode="xml"
            )
            articles = Entrez.read(handle)
            handle.close()
            
            return articles["PubmedArticle"]
            
        except Exception as e:
            self.logger.error(f"PubMed API error: {e}")
            return []
    
    async def _process_search_results(self, articles: List[Dict], protein_annotation: str) -> LiteratureResult:
        """Process PubMed search results and extract relevant information"""
        references = []
        
        for article in articles:
            try:
                # Extract article information
                medline_citation = article["MedlineCitation"]
                article_info = medline_citation["Article"]
                
                pmid = medline_citation["PMID"]
                title = article_info["ArticleTitle"]
                
                # Extract authors
                authors = []
                if "AuthorList" in article_info:
                    for author in article_info["AuthorList"][:3]:  # First 3 authors
                        if "LastName" in author and "ForeName" in author:
                            authors.append(f"{author['LastName']}, {author['ForeName']}")
                
                # Extract journal and year
                journal = article_info.get("Journal", {}).get("Title", "Unknown Journal")
                pub_date = article_info.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                year = pub_date.get("Year", "Unknown")
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(title, protein_annotation)
                
                reference = LiteratureReference(
                    pmid=str(pmid),
                    title=title,
                    authors=authors,
                    journal=journal,
                    year=year,
                    relevance_score=relevance_score,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                )
                
                references.append(reference)
                
            except Exception as e:
                self.logger.warning(f"Error processing article: {e}")
                continue
        
        # Sort by relevance score
        references.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return LiteratureResult(
            protein_annotation=protein_annotation,
            references=references[:5],  # Top 5 most relevant
            search_successful=True,
            total_found=len(articles)
        )
    
    def _calculate_relevance_score(self, title: str, protein_annotation: str) -> float:
        """Calculate relevance score based on title content"""
        title_lower = title.lower()
        protein_lower = protein_annotation.lower()
        
        score = 0.0
        
        # Exact protein name match
        if protein_lower in title_lower:
            score += 2.0
        
        # Individual word matches
        protein_words = protein_lower.split()
        for word in protein_words:
            if len(word) > 3 and word in title_lower:
                score += 0.5
        
        # Structural/functional keywords
        keywords = ["structure", "function", "domain", "motif", "binding", "activity"]
        for keyword in keywords:
            if keyword in title_lower:
                score += 0.3
        
        return score
```

### **PHASE 2B: ENHANCED WORKFLOW WITH VIRAL_PSSM.JSON OUTPUT**

#### **Step 2B.1: Viral_PSSM.json Output Formatter**
**File**: `nanobrain/library/workflows/alphavirus_analysis/formatters/viral_pssm_formatter.py`

**Objective**: Format output to match exact Viral_PSSM.json structure with literature references

**Implementation Details**:
```python
class ViralPSSMFormatter:
    """
    Format workflow output to match Viral_PSSM.json structure exactly.
    
    Based on: https://github.com/jimdavis1/Viral_Annotation/blob/main/Viral_PSSM.json
    Enhanced with PubMed literature references
    """
    
    def __init__(self, config: ViralPSSMFormatterConfig):
        self.config = config
        self.pubmed_tool = PubMedIntegrationTool()
        self.logger = get_logger("viral_pssm_formatter")
        
    async def format_workflow_output(self, workflow_data: WorkflowData) -> Dict[str, Any]:
        """
        Format complete workflow output to Viral_PSSM.json structure.
        
        Structure based on reference implementation:
        {
          "metadata": {...},
          "proteins": [
            {
              "id": "protein_1",
              "function": "capsid protein", 
              "boundaries": {...},
              "pssm_profile": {...},
              "literature_references": [...]  # Enhanced with PubMed
            }
          ]
        }
        """
        # Generate metadata
        metadata = await self._generate_metadata(workflow_data)
        
        # Process each protein cluster
        proteins = []
        for cluster_id, cluster_data in workflow_data.clusters.items():
            protein_entry = await self._format_protein_entry(cluster_id, cluster_data)
            proteins.append(protein_entry)
        
        viral_pssm_output = {
            "metadata": metadata,
            "proteins": proteins,
            "analysis_summary": await self._generate_analysis_summary(workflow_data),
            "quality_metrics": await self._generate_quality_metrics(workflow_data)
        }
        
        return viral_pssm_output
    
    async def _generate_metadata(self, workflow_data: WorkflowData) -> Dict[str, Any]:
        """Generate metadata section matching Viral_PSSM.json format"""
        return {
            "organism": "Alphavirus",
            "analysis_date": datetime.now().isoformat(),
            "coordinate_system": "1-based",
            "method": "nanobrain_alphavirus_analysis",
            "version": "1.0.0",
            "data_source": "BV-BRC",
            "total_genomes_analyzed": len(workflow_data.filtered_genomes),
            "clustering_method": "MMseqs2",
            "clustering_parameters": {
                "min_seq_id": workflow_data.clustering_params.min_seq_id,
                "coverage": workflow_data.clustering_params.coverage,
                "sensitivity": workflow_data.clustering_params.sensitivity
            },
            "alignment_method": "MUSCLE",
            "pssm_generation_method": "custom_nanobrain"
        }
    
    async def _format_protein_entry(self, cluster_id: str, cluster_data: ClusterData) -> Dict[str, Any]:
        """Format individual protein entry with literature references"""
        
        # Get protein annotation
        protein_annotation = cluster_data.consensus_annotation.standard_name
        
        # Search for literature references
        literature_result = await self.pubmed_tool.search_protein_literature(
            protein_annotation, 
            organism="Alphavirus"
        )
        
        # Format PSSM profile
        pssm_profile = await self._format_pssm_profile(cluster_data.pssm_matrix)
        
        # Format boundaries (if available)
        boundaries = await self._format_boundaries(cluster_data.boundaries)
        
        protein_entry = {
            "id": cluster_id,
            "function": protein_annotation,
            "protein_class": cluster_data.protein_class,
            "boundaries": boundaries,
            "pssm_profile": pssm_profile,
            "cluster_info": {
                "member_count": len(cluster_data.members),
                "representative_sequence": cluster_data.representative.sequence,
                "consensus_score": cluster_data.consensus_score,
                "alignment_quality": cluster_data.alignment_quality.mean_conservation
            },
            "literature_references": await self._format_literature_references(literature_result),
            "confidence_metrics": {
                "annotation_confidence": cluster_data.annotation_confidence,
                "boundary_confidence": cluster_data.boundary_confidence,
                "overall_confidence": cluster_data.overall_confidence
            }
        }
        
        return protein_entry
    
    async def _format_literature_references(self, literature_result: LiteratureResult) -> List[Dict[str, Any]]:
        """Format literature references for inclusion in output"""
        if not literature_result.search_successful:
            return []
        
        references = []
        for ref in literature_result.references:
            reference_entry = {
                "pmid": ref.pmid,
                "title": ref.title,
                "authors": ref.authors,
                "journal": ref.journal,
                "year": ref.year,
                "url": ref.url,
                "relevance_score": ref.relevance_score,
                "citation": f"{', '.join(ref.authors[:2])} et al. ({ref.year}). {ref.title}. {ref.journal}."
            }
            references.append(reference_entry)
        
        return references
    
    async def _format_pssm_profile(self, pssm_matrix: PSSMMatrix) -> Dict[str, Any]:
        """Format PSSM matrix for JSON output"""
        if not pssm_matrix:
            return {}
        
        return {
            "matrix": pssm_matrix.matrix.tolist(),  # Convert numpy array to list
            "alphabet": pssm_matrix.alphabet,
            "length": pssm_matrix.length,
            "conservation_profile": pssm_matrix.conservation_profile.tolist() if hasattr(pssm_matrix, 'conservation_profile') else [],
            "consensus_sequence": await self._generate_consensus_sequence(pssm_matrix)
        }
    
    async def _format_boundaries(self, boundaries: Optional[ProteinBoundaries]) -> Dict[str, Any]:
        """Format protein boundaries in 1-based coordinates"""
        if not boundaries:
            return {
                "start": None,
                "end": None,
                "confidence": 0.0,
                "method": "not_determined"
            }
        
        return {
            "start": boundaries.start,  # Already in 1-based coordinates
            "end": boundaries.end,
            "confidence": boundaries.confidence,
            "method": "pssm_based",
            "supporting_evidence": boundaries.supporting_evidence
        }
```

#### **Step 2B.2: Enhanced Error Handling and Retry System**
**File**: `nanobrain/core/retry_manager.py`

**Implementation Details**:
```python
class RetryManager:
    """
    Enhanced retry system for workflow steps with user interaction.
    
    Features:
    - Configurable retry attempts (default: 3)
    - Exponential backoff
    - User interaction after max retries
    - Partial failure handling
    """
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.max_retries = config.max_retries  # 3 as specified
        self.logger = get_logger("retry_manager")
        
    async def execute_with_retry(self, operation: Callable, operation_name: str, **kwargs) -> Any:
        """
        Execute operation with retry logic and user interaction.
        
        Args:
            operation: Async function to execute
            operation_name: Human-readable name for logging
            **kwargs: Arguments to pass to operation
            
        Returns:
            Operation result
            
        Raises:
            RetryExhaustedException: After max retries and user declines to continue
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Executing {operation_name} (attempt {attempt + 1}/{self.max_retries})")
                
                result = await operation(**kwargs)
                
                # Validate result if validator provided
                if hasattr(self.config, 'result_validator') and self.config.result_validator:
                    if not await self.config.result_validator(result):
                        raise ValidationError(f"Result validation failed for {operation_name}")
                
                self.logger.info(f"{operation_name} succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"{operation_name} failed on attempt {attempt + 1}: {e}")
                
                if attempt < self.max_retries - 1:
                    wait_time = self._calculate_backoff_time(attempt)
                    self.logger.info(f"Retrying {operation_name} in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    # Max retries reached - ask user
                    user_decision = await self._ask_user_for_retry_decision(operation_name, last_exception)
                    
                    if user_decision.retry:
                        # User wants to retry - extend max retries
                        self.max_retries += user_decision.additional_attempts
                        self.logger.info(f"User requested {user_decision.additional_attempts} additional attempts")
                    elif user_decision.skip:
                        # User wants to skip this operation
                        self.logger.warning(f"User chose to skip {operation_name}")
                        return None
                    else:
                        # User wants to abort
                        raise RetryExhaustedException(
                            f"Operation {operation_name} failed after {self.max_retries} attempts. "
                            f"Last error: {last_exception}"
                        )
        
        # Should not reach here, but just in case
        raise RetryExhaustedException(f"Unexpected end of retry loop for {operation_name}")
    
    def _calculate_backoff_time(self, attempt: int) -> float:
        """Calculate exponential backoff time"""
        base_wait = self.config.base_wait_time  # Default: 2 seconds
        max_wait = self.config.max_wait_time    # Default: 60 seconds
        
        wait_time = base_wait * (2 ** attempt)
        return min(wait_time, max_wait)
    
    async def _ask_user_for_retry_decision(self, operation_name: str, exception: Exception) -> RetryDecision:
        """
        Ask user what to do after max retries reached.
        
        Options:
        1. Retry with additional attempts
        2. Skip this operation and continue
        3. Abort the entire workflow
        """
        message = (
            f"❌ Operation '{operation_name}' failed after {self.max_retries} attempts.\n"
            f"Last error: {exception}\n\n"
            f"What would you like to do?\n"
            f"1. Retry (with additional attempts)\n"
            f"2. Skip this operation and continue\n"
            f"3. Abort the workflow\n\n"
            f"Please choose an option (1-3):"
        )
        
        # This would be implemented based on the interface (web, CLI, etc.)
        # For now, we'll simulate user input
        user_choice = await self._get_user_input(message)
        
        if user_choice == "1":
            additional_attempts = await self._get_additional_attempts()
            return RetryDecision(retry=True, additional_attempts=additional_attempts)
        elif user_choice == "2":
            return RetryDecision(skip=True)
        else:
            return RetryDecision(abort=True)
    
    async def _get_user_input(self, message: str) -> str:
        """Get user input - implementation depends on interface type"""
        # This is a placeholder - actual implementation would depend on
        # whether we're running in web interface, CLI, etc.
        self.logger.info(f"User prompt: {message}")
        return "1"  # Default to retry for now
    
    async def _get_additional_attempts(self) -> int:
        """Ask user how many additional attempts they want"""
        # Placeholder implementation
        return 2  # Default to 2 additional attempts
```

## COMPREHENSIVE STEP-BY-STEP IMPLEMENTATION PLAN

### **Phase 1: Foundation Setup (Days 1-3)**

#### **Day 1: Core Infrastructure**
1. **Create resource monitoring system** (`resource_monitor.py`)
   - Implement disk space monitoring with 1GB warning threshold
   - Add workflow pause capability at 500MB critical threshold
   - Create user notification system

2. **Enhance external tool base classes** (`external_tool.py`, `base_external_tool.py`)
   - Add retry mechanism support
   - Integrate resource monitoring
   - Implement data verification interfaces

#### **Day 2: BV-BRC Integration**
3. **Implement enhanced BV-BRC tool** (`bv_brc_tool.py`)
   - Anonymous authentication setup
   - Data verification after downloads
   - Retry mechanism with exponential backoff

4. **Test BV-BRC connectivity and data verification**
   - Verify `/Applications/BV-BRC.app/` access
   - Test anonymous data downloads
   - Validate data completeness (not just headers)

#### **Day 3: Literature Integration**
5. **Implement PubMed integration tool** (`pubmed_integration_tool.py`)
   - Set up Entrez utilities access
   - Implement protein-specific literature search
   - Add result caching and relevance scoring

6. **Create Viral_PSSM.json formatter** (`viral_pssm_formatter.py`)
   - Match exact structure from reference implementation
   - Integrate literature references
   - Add comprehensive metadata

### **Phase 2: Workflow Implementation (Days 4-10)**

#### **Days 4-5: Data Acquisition Steps (1-7)**
7. **Implement BV-BRC data acquisition step** (`bv_brc_data_acquisition_step.py`)
   - Step 1: Download Alphavirus genomes with verification
   - Step 2: Filter genomes by size (8KB-15KB range)
   - Steps 3-4: Extract unique protein MD5s with deduplication
   - Step 5: Get feature sequences with batch processing
   - Step 6: Get annotations with comprehensive metadata
   - Step 7: Create annotated FASTA with detailed headers

#### **Days 6-7: Annotation and Curation Steps (8-11)**
8. **Implement annotation mapping step** (`annotation_mapping_step.py`)
   - Step 8: Automated ICTV genome schematic integration
   - Standardize protein annotations using ICTV mapping
   - Handle annotation inconsistencies

9. **Implement sequence curation step** (`sequence_curation_step.py`)
   - Step 9: Create curated FASTA files
   - Step 10: Analyze length distributions with expected ranges
   - Step 11: Identify mangled sequences and quality issues

#### **Days 8-9: Clustering and Alignment Steps (12-13)**
10. **Implement clustering step** (`clustering_step.py`)
    - Step 12: MMseqs2 clustering with configurable parameters
    - Global settings with protein-specific fine-tuning
    - Quality analysis focusing on short well-conserved regions

11. **Implement alignment step** (`alignment_step.py`)
    - Step 13: MUSCLE multiple sequence alignment
    - Alignment quality assessment
    - Preparation for PSSM generation

#### **Day 10: PSSM Analysis and Report (Step 14)**
12. **Implement PSSM analysis step** (`pssm_analysis_step.py`)
    - Step 14: Generate PSSM matrices from alignments
    - Create comprehensive curation report
    - Integrate literature references for each protein

### **Phase 3: Integration and Testing (Days 11-14)**

#### **Days 11-12: Workflow Orchestration**
13. **Implement main workflow class** (`alphavirus_workflow.py`)
    - Orchestrate all 14 steps with proper error handling
    - Integrate resource monitoring throughout
    - Add progress tracking and user notifications

14. **Implement retry and error handling system** (`retry_manager.py`)
    - 3-attempt retry mechanism with user interaction
    - Partial failure handling with user choices
    - Graceful degradation strategies

#### **Days 13-14: Testing and Validation**
15. **Create comprehensive test suite** (`test_alphavirus_workflow.py`)
    - Unit tests for each workflow step
    - Integration tests with real BV-BRC data
    - Resource monitoring and error handling tests

16. **Perform end-to-end testing**
    - Test complete workflow with small dataset
    - Validate Viral_PSSM.json output format
    - Verify literature integration functionality

### **Phase 4: Configuration and Documentation (Days 15-16)**

#### **Day 15: Configuration Management**
17. **Create configuration files**
    - `AlphavirusWorkflow.yml` - Main workflow configuration
    - `bv_brc_config.yml` - BV-BRC tool settings
    - `clustering_config.yml` - MMseqs2 parameters
    - `pubmed_config.yml` - Literature search settings

#### **Day 16: Documentation and Finalization**
18. **Create documentation**
    - Update implementation plan with final details
    - Create user guide for workflow execution
    - Document configuration options

19. **Final integration testing**
    - Test with realistic data volumes (hundreds of MB)
    - Verify resource monitoring triggers
    - Validate complete Viral_PSSM.json output

## IMPLEMENTATION READINESS CHECKLIST ✅

**User Requirements Addressed**:
- ✅ Anonymous BV-BRC authentication with data verification
- ✅ Hundreds of MB data volume handling
- ✅ Automated ICTV integration
- ✅ Global clustering parameters with protein-specific fine-tuning
- ✅ Viral_PSSM.json output format with PubMed references
- ✅ 3-attempt retry mechanism with user interaction
- ✅ 1GB disk space warning with workflow pause

**Technical Implementation**:
- ✅ Proper base class hierarchy for external tools
- ✅ Comprehensive resource monitoring system
- ✅ Enhanced error handling and retry mechanisms
- ✅ PubMed literature integration
- ✅ Complete 14-step workflow implementation
- ✅ Viral_PSSM.json formatter matching reference structure

**Estimated Timeline**: 16 days for complete implementation and testing

## FINAL CLARIFICATION QUESTIONS

Before proceeding with implementation, I need confirmation on these final details:

1. **Email for PubMed Access**: What email address should be used for NCBI Entrez utilities? (Required by NCBI API)

2. **ICTV Data Source**: Should I use the official ICTV website API or a specific dataset for genome schematics?

3. **User Interface for Retry Decisions**: How should user interaction be handled? Web interface prompts, CLI prompts, or configuration-based decisions?

4. **Literature Reference Limits**: How many PubMed references should be included per protein in the final output?

**Ready for Implementation**: All major requirements are addressed. Please confirm the above details and provide the green light to proceed with implementation. 