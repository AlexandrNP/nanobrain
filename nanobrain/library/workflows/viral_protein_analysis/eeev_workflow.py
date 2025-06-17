"""
EEEV (Eastern Equine Encephalitis Virus) Specific Workflow

Specialized implementation of the Alphavirus workflow customized for EEEV analysis.
Includes EEEV-specific validation, configuration, and metadata generation.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path

from .alphavirus_workflow import AlphavirusWorkflow, WorkflowData, WorkflowResult
from .config.workflow_config import AlphavirusWorkflowConfig


class EEEVWorkflow(AlphavirusWorkflow):
    """
    Eastern Equine Encephalitis Virus specific workflow
    
    Extends the base Alphavirus workflow with EEEV-specific:
    - Configuration parameters
    - Validation criteria  
    - Output file naming
    - Metadata generation
    """
    
    def __init__(self, config: Optional[AlphavirusWorkflowConfig] = None):
        """Initialize EEEV workflow with customized configuration"""
        
        # Load and customize config for EEEV
        if config is None:
            config_path = Path(__file__).parent / "config" / "AlphavirusWorkflow.yml"
            config = AlphavirusWorkflowConfig.from_file(str(config_path))
        
        # Apply EEEV-specific customizations
        self._customize_config_for_eeev(config)
        
        # Initialize parent workflow
        super().__init__(config)
        
        # Create EEEV-specific logger
        from nanobrain.core.logging_system import get_logger
        self.logger = get_logger("eeev_workflow")
        self.logger.info("Initialized EEEV-specific workflow")
    
    def _customize_config_for_eeev(self, config: AlphavirusWorkflowConfig) -> None:
        """Customize configuration specifically for EEEV analysis"""
        
        # EEEV-specific naming
        config.name = "eeev_analysis"
        config.description = "Eastern Equine Encephalitis Virus protein analysis using BV-BRC and MMseqs2"
        
        # EEEV genome size constraints (typical EEEV ~11.7kb)
        config.bvbrc.min_length = 10500  # Allow some variation
        config.bvbrc.max_length = 12500
        config.bvbrc.genus = "Alphavirus"
        
        # EEEV-specific output directory
        if not config.output.base_directory.endswith("eeev_analysis"):
            config.output.base_directory = str(Path(config.output.base_directory) / "eeev_analysis")
        
        # EEEV-specific output file naming
        config.output.output_files = {
            "filtered_genomes": "eeev_filtered_genomes.json",
            "unique_proteins": "eeev_unique_proteins.fasta",
            "clusters": "eeev_clusters.json", 
            "alignments": "eeev_alignments.json",
            "pssm_matrices": "eeev_pssm_matrices.json",
            "curation_report": "eeev_curation_report.json",
            "viral_pssm_json": "eeev_viral_pssm.json"
        }
        
        # EEEV-specific protein length expectations
        config.quality_control.expected_lengths = {
            "nsP1": [520, 580],    # EEEV nsP1 ~549 aa
            "nsP2": [770, 820],    # EEEV nsP2 ~798 aa
            "nsP3": [500, 560],    # EEEV nsP3 ~530 aa  
            "nsP4": [580, 640],    # EEEV nsP4 ~611 aa
            "capsid": [250, 290],  # EEEV capsid ~264 aa
            "E3": [50, 80],        # EEEV E3 ~64 aa
            "E2": [400, 450],      # EEEV E2 ~423 aa
            "6K": [45, 65],        # EEEV 6K ~55 aa
            "E1": [420, 460]       # EEEV E1 ~439 aa
        }
        
        # EEEV-specific quality thresholds
        config.quality_control.min_sequence_length = 30  # Minimum protein length
        config.quality_control.max_sequence_length = 1000  # Maximum expected protein length
        config.quality_control.remove_duplicates = True
        config.quality_control.validate_protein_families = True
    
    async def execute_full_workflow(self, target_organism: str = "Eastern equine encephalitis virus") -> WorkflowResult:
        """
        Execute complete EEEV-specific workflow
        
        Args:
            target_organism: Target organism name (defaults to EEEV)
            
        Returns:
            WorkflowResult with EEEV-specific validation and metadata
        """
        
        self.logger.info(f"🦠 Starting EEEV analysis workflow for: {target_organism}")
        
        # Execute base workflow
        result = await super().execute_full_workflow({
            "target_genus": "Alphavirus",
            "species_filter": target_organism,
            "output_directory": self.config.output.base_directory
        })
        
        if result.success:
            # Apply EEEV-specific validation
            self.logger.info("🔍 Running EEEV-specific validation")
            validation_results = self._validate_eeev_specific_criteria(result.workflow_data)
            
            # Update viral_pssm.json with EEEV-specific metadata
            if result.viral_pssm_json:
                result.viral_pssm_json = self._enhance_viral_pssm_for_eeev(
                    result.viral_pssm_json, 
                    validation_results
                )
            
            # Add EEEV validation to result
            result.eeev_validation = validation_results
            
            # Log EEEV-specific results
            self._log_eeev_results(result, validation_results)
        
        return result
    
    def _validate_eeev_specific_criteria(self, workflow_data: WorkflowData) -> Dict[str, Any]:
        """
        Validate EEEV-specific criteria
        
        Args:
            workflow_data: Workflow data to validate
            
        Returns:
            Dictionary containing validation results
        """
        
        validation_results = {
            "validation_type": "eeev_specific",
            "timestamp": self.execution_start_time
        }
        
        # Genome count validation (should have multiple EEEV genomes)
        genome_count = len(workflow_data.filtered_genomes)
        validation_results["genome_count"] = genome_count
        validation_results["genome_count_check"] = genome_count >= 3  # At least 3 EEEV genomes
        
        # Expected EEEV proteins (9 total)
        expected_eeev_proteins = {
            'nsP1', 'nsP2', 'nsP3', 'nsP4',      # Non-structural proteins
            'capsid', 'E3', 'E2', '6K', 'E1'     # Structural proteins
        }
        
        # Check protein coverage
        found_proteins = set()
        if hasattr(workflow_data, 'clusters') and workflow_data.clusters:
            for cluster in workflow_data.clusters:
                protein_class = cluster.get('protein_class', '').lower()
                for expected in expected_eeev_proteins:
                    if expected.lower() in protein_class:
                        found_proteins.add(expected)
                        break
        
        validation_results["expected_proteins_found"] = list(found_proteins)
        validation_results["missing_proteins"] = list(expected_eeev_proteins - found_proteins)
        validation_results["protein_coverage_check"] = len(found_proteins) >= 7  # At least 7 out of 9
        
        # Genome size validation (EEEV should be ~11.7kb)
        genome_sizes = []
        if hasattr(workflow_data, 'filtered_genomes'):
            for genome in workflow_data.filtered_genomes:
                if 'genome_length' in genome:
                    genome_sizes.append(genome['genome_length'])
        
        validation_results["genome_sizes"] = genome_sizes
        validation_results["genome_size_check"] = all(10500 <= size <= 12500 for size in genome_sizes)
        
        # Calculate overall quality score
        checks_passed = sum([
            validation_results["genome_count_check"],
            validation_results["protein_coverage_check"], 
            validation_results["genome_size_check"]
        ])
        validation_results["quality_score"] = checks_passed / 3.0
        validation_results["overall_pass"] = validation_results["quality_score"] >= 0.67
        
        return validation_results
    
    def _enhance_viral_pssm_for_eeev(self, viral_pssm_json: Dict[str, Any], validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance viral_pssm.json with EEEV-specific metadata
        
        Args:
            viral_pssm_json: Base viral_pssm.json structure
            validation_results: EEEV validation results
            
        Returns:
            Enhanced viral_pssm.json with EEEV-specific data
        """
        
        # Update metadata with EEEV-specific information
        viral_pssm_json['metadata']['organism'] = 'Eastern Equine Encephalitis Virus'
        viral_pssm_json['metadata']['virus_family'] = 'Togaviridae'
        viral_pssm_json['metadata']['genus'] = 'Alphavirus'
        viral_pssm_json['metadata']['species'] = 'Eastern equine encephalitis virus'
        
        # Add EEEV-specific metadata section
        viral_pssm_json['metadata']['eeev_specific'] = {
            "genome_size_range": "11.5-11.8 kb",
            "typical_proteins": 9,
            "virulence_factors": [
                "E2 receptor binding",
                "E1 fusion protein", 
                "nsP2 protease activity",
                "capsid nucleocapsid formation"
            ],
            "geographic_distribution": "Eastern North America",
            "primary_vectors": ["Culiseta melanura", "Aedes species"],
            "primary_hosts": ["Birds", "Horses", "Humans"],
            "pathogenicity": "High neurotropism, severe encephalitis",
            "validation_results": validation_results
        }
        
        # Update protein IDs to be EEEV-specific
        if 'proteins' in viral_pssm_json:
            for protein in viral_pssm_json['proteins']:
                original_id = protein.get('id', '')
                if not original_id.startswith('eeev_'):
                    protein['id'] = f"eeev_{original_id}"
        
        # Add EEEV-specific conservation notes
        viral_pssm_json['metadata']['conservation_notes'] = {
            "structural_proteins": "E1/E2 envelope proteins highly conserved for receptor binding",
            "nonstructural_proteins": "nsP2 protease domain critical for replication",
            "antigenic_sites": "E2 protein contains major neutralizing epitopes",
            "drug_targets": "nsP4 RNA polymerase domain, nsP2 protease active site"
        }
        
        return viral_pssm_json
    
    def _log_eeev_results(self, result: WorkflowResult, validation_results: Dict[str, Any]) -> None:
        """Log EEEV-specific analysis results"""
        
        if result.success:
            quality_score = validation_results.get('quality_score', 0)
            genome_count = validation_results.get('genome_count', 0)
            proteins_found = len(validation_results.get('expected_proteins_found', []))
            
            self.logger.info(
                f"✅ EEEV workflow completed successfully",
                quality_score=quality_score,
                genomes_analyzed=genome_count, 
                proteins_identified=proteins_found,
                execution_time=result.execution_time
            )
            
            if quality_score >= 0.8:
                self.logger.info("🏆 High quality EEEV analysis achieved")
            elif quality_score >= 0.6:
                self.logger.warning("⚠️ Moderate quality EEEV analysis")
            else:
                self.logger.warning("⚠️ Low quality EEEV analysis - review results carefully")
                
            # Log missing proteins if any
            missing_proteins = validation_results.get('missing_proteins', [])
            if missing_proteins:
                self.logger.warning(f"🚨 Missing expected EEEV proteins: {missing_proteins}")
        else:
            self.logger.error(f"❌ EEEV workflow failed: {result.error}")


# For backward compatibility
class EEEVAlphavirusWorkflow(EEEVWorkflow):
    """Alias for backward compatibility"""
    pass 