# Data Flow Analysis: Steps 12-14 Integration - BRUTAL TRUTH

**Date**: December 2, 2025
**Status**: CRITICAL DATA FLOW ISSUES IDENTIFIED
**Severity**: HIGH - Proposed implementation has fundamental incompatibilities

---

## 🔥 EXECUTIVE SUMMARY - WHAT'S ACTUALLY BROKEN

### The Proposed Plan vs. Reality

| Component | What I Proposed | What Actually Exists | Compatibility |
|-----------|----------------|---------------------|---------------|
| **Input Data** | `quality_controlled_sequences` | Doesn't match real output | ❌ **BROKEN** |
| **Clustering Step** | New PSI-BLAST integration | MMseqs2Tool already works | ✅ **COMPATIBLE** |
| **Output Format** | `pssm_results` dict | Unknown downstream expectations | ⚠️ **UNCLEAR** |
| **Data Extraction** | Simple dict access | Nested complex structures | ❌ **BROKEN** |

### Critical Problems Discovered

1. **INPUT DATA MISMATCH** - Steps 10-11 output format doesn't match what I assumed
2. **NO CLEAR DOWNSTREAM CONSUMER** - No documentation of what uses PSSM results
3. **NESTED DATA STRUCTURES** - Real data is deeply nested, not flat dicts
4. **MISSING DATA UNITS** - Workflow doesn't define intermediate data units properly

---

## 📊 ACTUAL DATA FLOW THROUGH THE WORKFLOW

### Step-by-Step Reality Check

```
Step 1-2: Genome Acquisition & Filtering (BVBRCDataAcquisitionStep)
    ↓
    Output: {
        'fasta_sequences': str,
        'protein_data': {
            'unique_proteins': List[Dict],
            'protein_sequences': List[Dict],
            'protein_annotations': List[Dict]
        },
        'original_genomes': List[Dict]
    }
    ↓
Step 3-6: Protein MD5 Processing (ALREADY EXISTS - no separate step needed)
    ↓
    Same output structure (integrated into Step 1-2)
    ↓
Step 7-9: FASTA Creation & Annotation Mapping (AutomatedAnnotationMappingStep)
    ↓
    Expected Output: {
        'annotated_fasta': {
            'fasta_content': str,
            'standardized_proteins': List[Dict],
            'mapping_metadata': Dict
        }
    }
    ↓
Step 10-11: Sequence Quality Control (SequenceQualityControlStep)
    ↓
    Expected Input: 'annotated_fasta'
    Expected Output: {
        'quality_controlled_sequences': {
            'filtered_sequences': List[Dict],
            'quality_stats': Dict,
            'filtering_metadata': Dict
        }
    }
    ↓
Step 12-14: Clustering & PSSM (ClusteringPSSMGenerationStep) ← MY PROPOSAL
    ↓
    Expected Input: 'quality_controlled_sequences'  ❌ BUT WHAT FORMAT?
    Proposed Output: {
        'pssm_results': {
            'cluster_results': ???,
            'pssm_matrices': ???,
            'curated_pssms': ???
        }
    }
    ↓
    ??? WHO CONSUMES THIS OUTPUT ???
```

---

## 🚨 CRITICAL PROBLEM #1: Input Data Structure Mismatch

### What My Proposed Step Expects

```python
# From my ClusteringPSSMGenerationStep:
async def process(self, input_data: Any, **kwargs) -> Optional[Dict[str, Any]]:
    # Extract input data
    quality_data = input_data.get('quality_controlled_sequences', input_data)

    if not quality_data:
        self.nb_logger.error("❌ No quality-controlled sequences provided")
        return None
```

**I ASSUMED**: `quality_controlled_sequences` is a simple list of sequences or dict with sequences

### What Step 10-11 Actually Outputs

```python
# From step_10_11_sequence_quality_control.md:
return {
    'quality_controlled_sequences': {
        'filtered_sequences': filtered_sequences,  # List[Dict] - BUT WHAT STRUCTURE?
        'quality_stats': length_stats,              # Dict with statistics
        'filtering_metadata': {                     # Dict with metadata
            'original_count': len(fasta_data.get('standardized_proteins', [])),
            'filtered_count': len(filtered_sequences),
            'min_length': self.min_length,
            'max_length': self.max_length,
            'outlier_threshold': self.outlier_std_dev
        }
    }
}
```

### The ACTUAL Problem

**MY CODE**:
```python
quality_data = input_data.get('quality_controlled_sequences', input_data)
# This gets the ENTIRE nested dict {'filtered_sequences': [...], 'quality_stats': {...}, ...}

if not quality_data:  # This will NEVER be False
    # Because quality_data is a dict, which is truthy even if empty
```

**CORRECT CODE SHOULD BE**:
```python
quality_container = input_data.get('quality_controlled_sequences', {})
filtered_sequences = quality_container.get('filtered_sequences', [])

if not filtered_sequences:
    self.nb_logger.error("❌ No filtered sequences provided")
    return None

# Now extract actual sequence data
sequences_for_clustering = self._extract_sequences_from_filtered(filtered_sequences)
```

---

## 🚨 CRITICAL PROBLEM #2: What IS `filtered_sequences`?

### The Documentation Says:

```python
# From step_10_11_sequence_quality_control.md:
filtered_sequences = self._filter_sequences_by_quality(fasta_data, length_stats)
```

But **WHAT IS THE STRUCTURE** of each item in `filtered_sequences`?

### Three Possible Structures (ALL DIFFERENT):

**Option A: Copy from input (standardized_proteins)**
```python
{
    'aa_sequence_md5': str,
    'product': str,  # Standardized name
    'original_product': str,
    'gene': str,
    'patric_id': str,
    'genome_id': str,
    'viral_family': str,
    'synonym_confidence': float
}
```

**Option B: Original protein data structure**
```python
{
    'patric_id': str,
    'aa_sequence_md5': str,
    'genome_id': str,
    'product': str,
    'gene': str
}
```

**Option C: FASTA-based structure**
```python
{
    'header': str,  # FASTA header
    'sequence': str,  # Amino acid sequence
    'length': int,
    'quality_score': float
}
```

### Which One is It?

**I HAVE NO IDEA** because the quality control step documentation doesn't specify!

---

## 🚨 CRITICAL PROBLEM #3: Sequence Data for Clustering

### What MMseqs2/PSI-BLAST Actually Needs

**FASTA format input**:
```
>sequence_id1 protein_name
MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQT
>sequence_id2 protein_name
MKFLVNVALVFMVVYISYIYALHHHHHDDDDDPPPPP
```

### What My Code Tries to Do

```python
# From my implementation plan:
async def _perform_mmseqs2_clustering(self, quality_data: Dict[str, Any]) -> List[ClusterResult]:
    # Extract sequences from quality control output
    sequences = quality_data.get('filtered_sequences', [])

    # Convert to FASTA format
    fasta_content = self._sequences_to_fasta(sequences)  # ❌ UNDEFINED METHOD
```

### The Missing Logic

I NEVER DEFINED `_sequences_to_fasta()` because **I don't know the structure of `sequences`**!

```python
def _sequences_to_fasta(self, sequences: List[Dict]) -> str:
    """Convert sequences to FASTA format - BUT HOW?"""

    # Option A: sequences have 'aa_sequence_md5' but NO sequence data
    # Option B: sequences have 'sequence' field directly
    # Option C: need to fetch sequences from somewhere else

    # ??? WHICH IS IT ???
```

---

## 🚨 CRITICAL PROBLEM #4: No Sequence Data in Quality Control Output?

### Brutal Realization

Looking at the quality control documentation:

```python
# Step 10: Analyze sequence lengths
length_stats = self._analyze_sequence_lengths(fasta_data)

# Step 11: Filter sequences by quality
filtered_sequences = self._filter_sequences_by_quality(fasta_data, length_stats)
```

**WHERE DOES SEQUENCE DATA COME FROM?**

The input is `annotated_fasta`, which contains:
```python
{
    'fasta_content': str,  # FULL FASTA STRING
    'standardized_proteins': List[Dict],  # METADATA ONLY, NO SEQUENCES
    'mapping_metadata': Dict
}
```

**PROBLEM**:
- `fasta_content` has the sequences (in FASTA string format)
- `standardized_proteins` has metadata (product names, IDs, etc.)
- **But these are SEPARATE!**

### How to Match Them?

**The quality control step needs to**:
1. Parse `fasta_content` string into individual sequences
2. Match each sequence to its metadata in `standardized_proteins` (by what? MD5? Header?)
3. Filter based on sequence length
4. Re-associate filtered sequences with their metadata

**MY PROPOSED CLUSTERING STEP NEEDS**:
- Sequence data (amino acid strings) for clustering
- Metadata (protein names, IDs) for annotation
- **But where does the sequence come from if quality control only passes metadata?**

---

## 🚨 CRITICAL PROBLEM #5: Missing Link Between Steps

### The Real Data Flow Problem

```
Step 7-9 Output:
  ├─ fasta_content: "STRING WITH ALL SEQUENCES"
  └─ standardized_proteins: [{metadata}, {metadata}, ...]

Step 10-11 Processing:
  ├─ Parse fasta_content → List of (header, sequence) tuples
  ├─ Analyze sequence lengths
  ├─ Filter by length thresholds
  └─ Return... WHAT?
      Option A: Filtered FASTA string
      Option B: Filtered metadata list
      Option C: Both
      Option D: Neither (just statistics)

Step 12-14 Input:
  ├─ Need: FASTA content for clustering
  ├─ Need: Metadata for annotation
  └─ Get: ??? UNKNOWN FORMAT ???
```

### The Missing Specification

**NONE OF THE STEP DOCUMENTATION SPECIFIES**:
1. Exact structure of output data units
2. How sequences and metadata stay connected
3. Whether FASTA strings are passed or parsed structures
4. How to re-create FASTA for clustering if needed

---

## 🔥 BRUTAL TRUTH: What's Actually Missing

### Documentation Gaps

1. **No data unit specifications** - "quality_controlled_sequences" is referenced but never fully defined
2. **No interface contracts** - Steps don't specify exact input/output structures
3. **No data transformation docs** - How to convert between FASTA strings and structured data
4. **No integration tests** - No end-to-end examples showing actual data flow

### Implementation Gaps

1. **Sequence extraction logic** - How to get amino acid strings from quality control output
2. **Metadata preservation** - How to keep protein names/IDs associated with sequences
3. **FASTA regeneration** - How to create FASTA input for MMseqs2/PSI-BLAST
4. **Data unit mapping** - How output data unit keys map to input keys in next step

---

## 💡 PROPOSED FIX - Complete Data Flow Specification

### 1. Define Explicit Data Structures

**QualityControlledSequence** (NEW data structure):
```python
@dataclass
class QualityControlledSequence:
    """Output from Step 10-11, Input to Step 12-14"""
    # Sequence data
    aa_sequence: str  # Actual amino acid sequence
    aa_sequence_md5: str  # MD5 hash
    sequence_length: int  # Pre-computed for quality metrics

    # Metadata
    product: str  # Standardized protein name
    original_product: str  # Original name before standardization
    gene: str  # Gene name
    patric_id: str  # BV-BRC identifier
    genome_id: str  # Genome identifier
    viral_family: str  # Taxonomic context

    # Quality metrics
    quality_score: float  # Overall quality score
    length_zscore: float  # Z-score for length distribution
    passed_quality_control: bool  # Whether it passed filtering

    def to_fasta_entry(self) -> str:
        """Convert to FASTA format entry"""
        header = f">{self.patric_id}|{self.product}|{self.genome_id}"
        return f"{header}\n{self.aa_sequence}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'aa_sequence': self.aa_sequence,
            'aa_sequence_md5': self.aa_sequence_md5,
            'sequence_length': self.sequence_length,
            'product': self.product,
            'original_product': self.original_product,
            'gene': self.gene,
            'patric_id': self.patric_id,
            'genome_id': self.genome_id,
            'viral_family': self.viral_family,
            'quality_score': self.quality_score,
            'length_zscore': self.length_zscore,
            'passed_quality_control': self.passed_quality_control
        }
```

### 2. Fix Step 10-11 Output

**Correct SequenceQualityControlStep output**:
```python
async def process(self, input_data: Any, **kwargs) -> Optional[Dict[str, Any]]:
    # Extract input data
    annotated_fasta_container = input_data.get('annotated_fasta', {})
    fasta_content = annotated_fasta_container.get('fasta_content', '')
    standardized_proteins = annotated_fasta_container.get('standardized_proteins', [])

    # Parse FASTA content
    parsed_sequences = self._parse_fasta_content(fasta_content)

    # Match sequences with metadata
    sequences_with_metadata = self._match_sequences_to_metadata(
        parsed_sequences, standardized_proteins
    )

    # Analyze quality
    quality_controlled = []
    for seq_data in sequences_with_metadata:
        quality_metrics = self._analyze_sequence_quality(seq_data)

        qc_sequence = QualityControlledSequence(
            aa_sequence=seq_data['sequence'],
            aa_sequence_md5=seq_data['md5'],
            sequence_length=len(seq_data['sequence']),
            product=seq_data['product'],
            original_product=seq_data.get('original_product', seq_data['product']),
            gene=seq_data.get('gene', ''),
            patric_id=seq_data.get('patric_id', ''),
            genome_id=seq_data.get('genome_id', ''),
            viral_family=seq_data.get('viral_family', ''),
            quality_score=quality_metrics['score'],
            length_zscore=quality_metrics['length_zscore'],
            passed_quality_control=quality_metrics['passed']
        )
        quality_controlled.append(qc_sequence)

    # Filter to only passing sequences
    filtered_sequences = [qc for qc in quality_controlled if qc.passed_quality_control]

    return {
        'quality_controlled_sequences': {
            'filtered_sequences': [seq.to_dict() for seq in filtered_sequences],
            'all_sequences': [seq.to_dict() for seq in quality_controlled],
            'quality_stats': self._generate_quality_stats(quality_controlled),
            'filtering_metadata': {
                'total_sequences': len(quality_controlled),
                'passed_sequences': len(filtered_sequences),
                'failed_sequences': len(quality_controlled) - len(filtered_sequences),
                'pass_rate': len(filtered_sequences) / len(quality_controlled),
                'min_length': self.min_length,
                'max_length': self.max_length
            }
        }
    }
```

### 3. Fix Step 12-14 Input Processing

**Correct ClusteringPSSMGenerationStep input extraction**:
```python
async def process(self, input_data: Any, **kwargs) -> Optional[Dict[str, Any]]:
    try:
        self.nb_logger.info("🚀 STARTING CLUSTERING & PSSM PIPELINE")

        # CORRECT: Extract from nested structure
        qc_container = input_data.get('quality_controlled_sequences', {})
        filtered_sequences = qc_container.get('filtered_sequences', [])

        if not filtered_sequences:
            self.nb_logger.error("❌ No filtered sequences provided")
            return None

        self.nb_logger.info(f"📊 Processing {len(filtered_sequences)} quality-controlled sequences")

        # Convert to FASTA content for clustering
        fasta_content = self._create_fasta_from_sequences(filtered_sequences)

        # Step 12: MMseqs2 clustering
        cluster_results = await self._perform_mmseqs2_clustering(fasta_content)

        # Step 13: PSI-BLAST PSSM generation
        pssm_results = await self._generate_pssms_for_clusters(
            cluster_results, filtered_sequences
        )

        # Step 14: PSSM quality assessment
        curated_pssms = await self._assess_pssm_quality(pssm_results)

        return {
            'pssm_results': {
                'cluster_results': [c.to_dict() for c in cluster_results],
                'pssm_matrices': pssm_results,
                'curated_pssms': curated_pssms,
                'pipeline_metadata': {
                    'input_sequences': len(filtered_sequences),
                    'clusters_generated': len(cluster_results),
                    'pssms_generated': len(pssm_results),
                    'high_quality_pssms': len([p for p in curated_pssms if p['quality_score'] > 0.7])
                }
            }
        }

    except Exception as e:
        self.nb_logger.error(f"❌ Clustering & PSSM pipeline failed: {e}")
        import traceback
        self.nb_logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def _create_fasta_from_sequences(self, filtered_sequences: List[Dict]) -> str:
    """Convert filtered sequences to FASTA format for clustering"""
    fasta_entries = []

    for seq_dict in filtered_sequences:
        header = f">{seq_dict['patric_id']}|{seq_dict['product']}|{seq_dict['genome_id']}"
        sequence = seq_dict['aa_sequence']
        fasta_entries.append(f"{header}\n{sequence}")

    return "\n".join(fasta_entries)
```

---

## 🔥 CRITICAL MISSING COMPONENT: Workflow Configuration

### The Actual Problem

**NONE OF THE STEP DOCS SHOW THE WORKFLOW CONFIGURATION!**

A complete workflow needs:

```yaml
name: viral_pssm_workflow
version: "2.0"

# Workflow-level data units (MISSING FROM ALL DOCS)
input_data_units:
  raw_viral_genomes:
    class: "nanobrain.core.data_unit.DataUnitMemory"
    name: "raw_viral_genomes"

output_data_units:
  viral_pssm_json:
    class: "nanobrain.core.data_unit.DataUnitMemory"
    name: "viral_pssm_json"

# ALL INTERMEDIATE DATA UNITS (MISSING)
intermediate_data_units:
  protein_data:
    class: "nanobrain.core.data_unit.DataUnitMemory"
    name: "protein_data"

  annotated_fasta:
    class: "nanobrain.core.data_unit.DataUnitMemory"
    name: "annotated_fasta"

  quality_controlled_sequences:
    class: "nanobrain.core.data_unit.DataUnitMemory"
    name: "quality_controlled_sequences"

  pssm_results:
    class: "nanobrain.core.data_unit.DataUnitMemory"
    name: "pssm_results"

# Steps
steps:
  data_acquisition:
    class: BVBRCDataAcquisitionStep
    config: config/data_acquisition.yml

  annotation_mapping:
    class: AutomatedAnnotationMappingStep
    config: config/annotation_mapping.yml

  quality_control:
    class: SequenceQualityControlStep
    config: config/quality_control.yml

  clustering_pssm:
    class: ClusteringPSSMGenerationStep
    config: config/clustering_pssm.yml

  final_output:
    class: ViralPSSMGenerationStep
    config: config/final_output.yml

# Links (CRITICAL - DEFINES DATA FLOW)
links:
  acquisition_to_mapping:
    class: "nanobrain.core.link.DirectLink"
    config:
      source: "data_acquisition.protein_data"
      target: "annotation_mapping.proteins_with_sequences"

  mapping_to_quality:
    class: "nanobrain.core.link.DirectLink"
    config:
      source: "annotation_mapping.annotated_fasta"
      target: "quality_control.annotated_fasta"

  quality_to_clustering:
    class: "nanobrain.core.link.DirectLink"
    config:
      source: "quality_control.quality_controlled_sequences"
      target: "clustering_pssm.quality_controlled_sequences"

  clustering_to_output:
    class: "nanobrain.core.link.DirectLink"
    config:
      source: "clustering_pssm.pssm_results"
      target: "final_output.pssm_data"
```

---

## 🔥 FINAL BRUTAL ASSESSMENT

### What's Actually Wrong

**EVERY STEP DOCUMENTATION IS INCOMPLETE**:
1. ❌ No explicit data structure definitions
2. ❌ No interface contracts (exact input/output formats)
3. ❌ No data transformation logic specified
4. ❌ No workflow configuration showing the full picture
5. ❌ No integration between steps validated

### What I Got Wrong

**MY IMPLEMENTATION PLAN HAS CRITICAL FLAWS**:
1. ❌ Wrong input data extraction (assumed flat dict, reality is nested)
2. ❌ Missing sequence data handling (how to get amino acid strings)
3. ❌ No FASTA regeneration logic (how to create input for tools)
4. ❌ Undefined output consumers (who uses PSSM results?)
5. ❌ No validation against actual existing code

### What Needs to Happen

**BEFORE IMPLEMENTING STEPS 12-14**:
1. **Define explicit data structures** - QualityControlledSequence, PSSMResult, etc.
2. **Write interface contracts** - Exact format for each step's input/output
3. **Create workflow configuration** - Full YAML with all data units and links
4. **Write integration tests** - End-to-end test showing data flow through all steps
5. **Validate against existing code** - Make sure it connects to what already works

### Realistic Assessment

**CURRENT STATE**:
- Individual step logic is mostly sound
- Data flow between steps is **COMPLETELY UNDEFINED**
- No way to validate correctness without running end-to-end

**REQUIRED WORK** (before coding any steps):
- 1 week: Define all data structures
- 1 week: Write interface contracts
- 1 week: Create complete workflow configuration
- 1 week: Write integration tests
- **THEN** start implementing actual steps

**TOTAL BEFORE STEP IMPLEMENTATION: 4 weeks of foundational work**

This is NOT a criticism of your request - this is a criticism of **MY INCOMPLETE ANALYSIS** that failed to trace the actual data structures through the workflow.

---

## 💡 RECOMMENDED NEXT STEPS

1. **STOP** - Don't implement ClusteringPSSMGenerationStep yet
2. **DEFINE** - Create explicit data structure classes for all intermediate data
3. **SPECIFY** - Write interface contracts for each step's input/output
4. **CONFIGURE** - Write complete workflow YAML showing all connections
5. **TEST** - Create end-to-end integration test with mock data
6. **VALIDATE** - Run test to prove data flows correctly
7. **THEN IMPLEMENT** - Only after validation, start coding actual steps

**This is the HONEST, BRUTAL TRUTH about what's actually needed.**
