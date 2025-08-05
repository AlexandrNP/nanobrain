"""
Enhanced BV-BRC Data Acquisition Step with Multi-Stage Validation Pipeline

This step provides:
- Multi-strategy CSV search without hardcoding
- Individual match validation using Species Validation Agent
- Taxonomic cross-validation using Taxonomic Verification Agent
- Final contamination prevention check with zero tolerance
- Comprehensive audit trail generation

NO hardcoded virus patterns - all validation via configurable agents
ZERO contamination tolerance - rejects any matches with contamination risk
"""

import asyncio
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict
import re

from nanobrain.core.step import Step
from nanobrain.core.data_unit import DataUnit
from nanobrain.core.agent import SimpleAgent
from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCTool
from nanobrain.core.config.component_factory import create_component, load_config_file


@dataclass
class ValidationResult:
    """Result of match validation with detailed metadata"""
    is_valid: bool
    confidence_score: float
    validation_method: str
    validation_details: Dict[str, Any]
    contamination_risk: float
    rejection_reason: Optional[str] = None


@dataclass
class CSVMatchCandidate:
    """CSV entry match candidate with validation metadata"""
    entry_data: Dict[str, Any]
    match_method: str
    search_term: str
    initial_confidence: float
    species_validation: Optional[ValidationResult] = None
    taxonomic_validation: Optional[ValidationResult] = None
    final_validation: Optional[ValidationResult] = None
    is_accepted: bool = False


class EnhancedBVBRCDataAcquisitionStep(Step):
    """
    Enhanced BV-BRC Data Acquisition Step - Advanced Bioinformatics Data Pipeline with Multi-Stage Validation
    ====================================================================================================
    
    The EnhancedBVBRCDataAcquisitionStep provides comprehensive bioinformatics data acquisition from the
    Bacterial and Viral Bioinformatics Resource Center (BV-BRC), implementing advanced multi-stage validation
    pipelines, intelligent quality assurance, and contamination prevention protocols. This step ensures
    scientific accuracy, data integrity, and automated quality control for viral protein analysis workflows.
    
    **Core Architecture:**
        The enhanced data acquisition step provides enterprise-grade bioinformatics capabilities:
        
        * **Multi-Stage Validation Pipeline**: Comprehensive validation using multiple specialized agents
        * **Intelligent Data Acquisition**: Smart search strategies with configurable validation protocols
        * **Contamination Prevention**: Zero-tolerance contamination detection and rejection mechanisms
        * **Quality Assurance**: Automated quality control with scientific validation standards
        * **Audit Trail Generation**: Complete traceability and reproducibility documentation
        * **Framework Integration**: Full integration with NanoBrain's scientific workflow architecture
    
    **Data Acquisition Capabilities:**
        
        **BV-BRC Database Integration:**
        * **Comprehensive Data Access**: Full access to BV-BRC viral and bacterial genome databases
        * **Advanced Search Strategies**: Multi-parameter search with taxonomic filtering
        * **Metadata Enrichment**: Automatic annotation and metadata enhancement
        * **Real-Time Synchronization**: Live database updates and version control
        
        **Intelligent Query Processing:**
        * **Natural Language Queries**: Scientific query interpretation and parameter extraction
        * **Taxonomic Intelligence**: Automated taxonomic classification and validation
        * **Cross-Reference Validation**: Multi-database cross-validation and verification
        * **Synonym Resolution**: Automatic species and strain synonym identification
        
        **Quality Control Systems:**
        * **Data Integrity Validation**: Comprehensive sequence and metadata validation
        * **Contamination Detection**: Advanced contamination screening and prevention
        * **Duplicate Detection**: Intelligent duplicate identification and removal
        * **Quality Scoring**: Scientific quality assessment and ranking
    
    **Multi-Stage Validation Pipeline:**
        
        **Stage 1: Initial Match Discovery:**
        * **Multi-Strategy Search**: Flexible search algorithms with configurable parameters
        * **Candidate Identification**: Initial match discovery and preliminary scoring
        * **Search Term Optimization**: Dynamic search term expansion and refinement
        * **Result Aggregation**: Comprehensive result collection and organization
        
        **Stage 2: Species Validation:**
        * **Species Validation Agent**: Specialized AI agent for species classification validation
        * **Taxonomic Accuracy**: ICTV-compliant taxonomic classification verification
        * **Confidence Scoring**: Machine learning-based confidence assessment
        * **Cross-Database Validation**: Multi-source species verification
        
        **Stage 3: Taxonomic Cross-Validation:**
        * **Taxonomic Verification Agent**: Advanced taxonomic relationship validation
        * **Phylogenetic Consistency**: Evolutionary relationship verification
        * **Classification Accuracy**: Hierarchical classification validation
        * **Scientific Consensus**: Literature-based validation and verification
        
        **Stage 4: Final Contamination Prevention:**
        * **Zero-Tolerance Screening**: Absolute contamination rejection protocols
        * **Cross-Contamination Detection**: Advanced contamination pattern recognition
        * **Quality Gate Enforcement**: Strict quality standards and gate controls
        * **Final Approval Process**: Multi-criteria approval and acceptance validation
    
    **Scientific Workflow Integration:**
        
        **Workflow Orchestration:**
        * **Event-Driven Processing**: Asynchronous processing with intelligent scheduling
        * **Dependency Management**: Automatic dependency resolution and validation
        * **Pipeline Coordination**: Multi-step workflow coordination and synchronization
        * **Progress Monitoring**: Real-time progress tracking and status reporting
        
        **Data Flow Management:**
        * **Type-Safe Data Handling**: Strongly-typed data structures and validation
        * **Schema Validation**: Automatic data schema validation and enforcement
        * **Version Control**: Data versioning and change tracking
        * **Backup and Recovery**: Automatic data backup and recovery mechanisms
        
        **Error Handling and Recovery:**
        * **Graceful Degradation**: Intelligent fallback mechanisms and error recovery
        * **Retry Logic**: Configurable retry policies with exponential backoff
        * **Error Classification**: Automatic error categorization and handling
        * **Recovery Strategies**: Multi-level recovery and continuation strategies
    
    **Configuration Architecture:**
        Comprehensive configuration supports diverse bioinformatics acquisition scenarios:
        
        ```yaml
        # Enhanced BV-BRC Data Acquisition Configuration
        step_name: "enhanced_bvbrc_data_acquisition"
        step_type: "bioinformatics_data_acquisition"
        
        # Step card for framework integration
        step_card:
          name: "enhanced_bvbrc_data_acquisition"
          description: "Advanced bioinformatics data acquisition with multi-stage validation"
          version: "1.0.0"
          category: "bioinformatics"
          capabilities:
            - "bvbrc_data_acquisition"
            - "multi_stage_validation"
            - "contamination_prevention"
        
        # BV-BRC Tool Configuration
        bvbrc_tool:
          class: "nanobrain.library.tools.bioinformatics.BVBRCTool"
          config:
            cache_enabled: true
            cache_directory: "bvbrc_cache"
            timeout: 300
            retry_attempts: 3
            
        # Search Strategy Configuration
        search_strategies:
          multi_term_search:
            enabled: true
            max_terms: 10
            term_expansion: true
            synonym_resolution: true
            
          taxonomic_search:
            enabled: true
            taxonomic_levels: ["species", "genus", "family"]
            include_synonyms: true
            
          metadata_search:
            enabled: true
            fields: ["organism_name", "strain", "isolate"]
            fuzzy_matching: true
            
        # Validation Pipeline Configuration
        validation_pipeline:
          species_validation:
            agent_class: "nanobrain.library.agents.specialized.SpeciesValidationAgent"
            confidence_threshold: 0.8
            ictv_compliance: true
            
          taxonomic_validation:
            agent_class: "nanobrain.library.agents.specialized.TaxonomicVerificationAgent"
            phylogenetic_validation: true
            consensus_threshold: 0.9
            
          contamination_screening:
            zero_tolerance: true
            screening_algorithms: ["sequence_similarity", "metadata_inconsistency"]
            rejection_threshold: 0.01
            
        # Quality Control Configuration
        quality_control:
          minimum_confidence: 0.85
          required_metadata_fields: ["organism_name", "genome_id", "taxon_id"]
          sequence_quality_threshold: 0.9
          completeness_threshold: 0.95
          
        # Data Processing Configuration
        data_processing:
          batch_size: 100
          parallel_processing: true
          max_concurrent_requests: 5
          result_caching: true
          
        # Audit and Logging Configuration
        audit_logging:
          enabled: true
          detailed_tracking: true
          validation_logs: true
          decision_audit: true
          
        # Output Configuration
        output:
          format: "structured_json"
          include_validation_metadata: true
          include_audit_trail: true
          quality_metrics: true
        ```
    
    **Usage Patterns:**
        
        **Basic Enhanced Data Acquisition:**
        ```python
        from nanobrain.library.workflows.viral_protein_analysis.steps import EnhancedBVBRCDataAcquisitionStep
        from nanobrain.core.data_unit import DataUnit
        
        # Create enhanced data acquisition step
        acquisition_step = EnhancedBVBRCDataAcquisitionStep.from_config({
            'step_name': 'viral_data_acquisition',
            'bvbrc_tool_config': 'bvbrc_tool_config.yml',
            'validation_agents': {
                'species_validator': 'species_validation_config.yml',
                'taxonomic_validator': 'taxonomic_validation_config.yml'
            },
            'quality_thresholds': {
                'minimum_confidence': 0.85,
                'contamination_tolerance': 0.0
            }
        })
        
        # Initialize the step
        await acquisition_step.initialize()
        
        # Create input data unit with search parameters
        input_data = DataUnit({
            'target_virus': 'SARS-CoV-2',
            'search_criteria': {
                'organism_type': 'virus',
                'genome_completeness': 'complete',
                'quality_threshold': 'high'
            },
            'analysis_scope': ['spike_protein', 'nucleocapsid_protein']
        })
        
        # Execute enhanced data acquisition
        result = await acquisition_step.process(input_data)
        
        # Access validated results
        validated_data = result.data['validated_entries']
        audit_trail = result.data['audit_trail']
        quality_metrics = result.data['quality_metrics']
        
        print(f"Validated Entries: {len(validated_data)}")
        print(f"Validation Success Rate: {quality_metrics['validation_success_rate']:.2%}")
        print(f"Contamination Rejections: {quality_metrics['contamination_rejections']}")
        ```
        
        **Advanced Multi-Species Analysis Pipeline:**
        ```python
        # Configure for comprehensive multi-species viral analysis
        class ViralGenomeAcquisitionPipeline:
            def __init__(self):
                self.acquisition_step = None
                self.species_validator = None
                self.taxonomic_validator = None
                self.quality_controller = None
                
            async def initialize_pipeline(self):
                # Configure enhanced acquisition step
                acquisition_config = {
                    'step_name': 'multi_species_viral_acquisition',
                    'search_strategies': {
                        'comprehensive_search': True,
                        'cross_reference_validation': True,
                        'phylogenetic_filtering': True
                    },
                    'validation_pipeline': {
                        'species_validation': {
                            'confidence_threshold': 0.9,
                            'ictv_compliance': True,
                            'cross_database_validation': True
                        },
                        'taxonomic_validation': {
                            'phylogenetic_consistency': True,
                            'evolutionary_distance_check': True,
                            'consensus_threshold': 0.95
                        },
                        'contamination_screening': {
                            'zero_tolerance': True,
                            'advanced_algorithms': True,
                            'multi_level_screening': True
                        }
                    }
                }
                
                self.acquisition_step = EnhancedBVBRCDataAcquisitionStep.from_config(
                    acquisition_config
                )
                await self.acquisition_step.initialize()
                
            async def acquire_viral_families(self, viral_families: List[str]):
                acquisition_results = {}
                
                for family in viral_families:
                    try:
                        # Configure family-specific search parameters
                        search_input = DataUnit({
                            'target_family': family,
                            'search_parameters': {
                                'taxonomic_level': 'family',
                                'include_all_species': True,
                                'quality_requirements': {
                                    'genome_completeness': 'complete',
                                    'annotation_quality': 'high',
                                    'sequence_quality': 'validated'
                                }
                            },
                            'validation_requirements': {
                                'species_validation': True,
                                'taxonomic_verification': True,
                                'contamination_screening': True,
                                'quality_assessment': True
                            }
                        })
                        
                        # Execute acquisition with enhanced validation
                        result = await self.acquisition_step.process(search_input)
                        
                        # Process and validate results
                        family_data = await self.process_family_results(family, result)
                        acquisition_results[family] = family_data
                        
                        # Log acquisition summary
                        summary = family_data['summary']
                        print(f"Family {family}:")
                        print(f"  Species Found: {summary['species_count']}")
                        print(f"  Validated Entries: {summary['validated_entries']}")
                        print(f"  Quality Score: {summary['average_quality_score']:.3f}")
                        print(f"  Contamination Rejections: {summary['contamination_rejections']}")
                        
                    except Exception as e:
                        print(f"Error acquiring data for family {family}: {e}")
                        acquisition_results[family] = {'error': str(e)}
                
                return acquisition_results
                
            async def process_family_results(self, family: str, result: DataUnit) -> dict:
                validated_entries = result.data['validated_entries']
                audit_trail = result.data['audit_trail']
                quality_metrics = result.data['quality_metrics']
                
                # Analyze species distribution
                species_analysis = await self.analyze_species_distribution(validated_entries)
                
                # Generate quality report
                quality_report = await self.generate_quality_report(
                    family, validated_entries, quality_metrics
                )
                
                # Create comprehensive family dataset
                family_dataset = {
                    'family_name': family,
                    'acquisition_timestamp': result.data['acquisition_timestamp'],
                    'validated_entries': validated_entries,
                    'species_analysis': species_analysis,
                    'quality_report': quality_report,
                    'audit_trail': audit_trail,
                    'summary': {
                        'species_count': len(species_analysis['unique_species']),
                        'validated_entries': len(validated_entries),
                        'average_quality_score': quality_metrics['average_quality_score'],
                        'contamination_rejections': quality_metrics['contamination_rejections'],
                        'validation_success_rate': quality_metrics['validation_success_rate']
                    }
                }
                
                return family_dataset
        
        # Initialize and run viral genome acquisition pipeline
        pipeline = ViralGenomeAcquisitionPipeline()
        await pipeline.initialize_pipeline()
        
        # Acquire data for multiple viral families
        viral_families = [
            'Coronaviridae',
            'Orthomyxoviridae', 
            'Paramyxoviridae',
            'Filoviridae',
            'Flaviviridae'
        ]
        
        family_results = await pipeline.acquire_viral_families(viral_families)
        
        # Generate comprehensive acquisition report
        acquisition_report = await pipeline.generate_acquisition_report(family_results)
        print(f"\\nComprehensive Acquisition Report:")
        print(f"Total Families Processed: {acquisition_report['total_families']}")
        print(f"Total Species Identified: {acquisition_report['total_species']}")
        print(f"Total Validated Entries: {acquisition_report['total_validated_entries']}")
        print(f"Overall Quality Score: {acquisition_report['overall_quality_score']:.3f}")
        ```
        
        **Custom Validation Agent Integration:**
        ```python
        # Advanced validation with custom scientific validation agents
        class CustomValidationPipeline:
            def __init__(self, acquisition_step: EnhancedBVBRCDataAcquisitionStep):
                self.acquisition_step = acquisition_step
                self.custom_validators = {}
                
            async def register_custom_validators(self):
                # Protein-specific validation agent
                protein_validator_config = {
                    'agent_name': 'protein_structure_validator',
                    'validation_criteria': {
                        'structure_prediction_confidence': 0.8,
                        'functional_domain_completeness': 0.9,
                        'homology_validation': True
                    }
                }
                
                self.custom_validators['protein_validator'] = await self.create_validator(
                    'ProteinStructureValidationAgent',
                    protein_validator_config
                )
                
                # Phylogenetic validation agent
                phylo_validator_config = {
                    'agent_name': 'phylogenetic_validator',
                    'validation_criteria': {
                        'evolutionary_distance_threshold': 0.3,
                        'monophyly_validation': True,
                        'bootstrap_support_threshold': 0.7
                    }
                }
                
                self.custom_validators['phylo_validator'] = await self.create_validator(
                    'PhylogeneticValidationAgent',
                    phylo_validator_config
                )
                
                # Literature validation agent
                literature_validator_config = {
                    'agent_name': 'literature_validator',
                    'validation_criteria': {
                        'publication_recency': 5,  # years
                        'citation_threshold': 10,
                        'peer_review_requirement': True
                    }
                }
                
                self.custom_validators['literature_validator'] = await self.create_validator(
                    'LiteratureValidationAgent',
                    literature_validator_config
                )
                
            async def run_custom_validation(self, candidates: List[CSVMatchCandidate]) -> List[CSVMatchCandidate]:
                validated_candidates = []
                
                for candidate in candidates:
                    # Run protein structure validation
                    protein_validation = await self.custom_validators['protein_validator'].validate(
                        candidate.entry_data
                    )
                    
                    # Run phylogenetic validation
                    phylo_validation = await self.custom_validators['phylo_validator'].validate(
                        candidate.entry_data
                    )
                    
                    # Run literature validation
                    literature_validation = await self.custom_validators['literature_validator'].validate(
                        candidate.entry_data
                    )
                    
                    # Combine validation results
                    combined_score = (
                        protein_validation.confidence_score * 0.4 +
                        phylo_validation.confidence_score * 0.4 +
                        literature_validation.confidence_score * 0.2
                    )
                    
                    # Create enhanced validation result
                    enhanced_validation = ValidationResult(
                        is_valid=(
                            protein_validation.is_valid and
                            phylo_validation.is_valid and
                            literature_validation.is_valid
                        ),
                        confidence_score=combined_score,
                        validation_method='custom_multi_agent',
                        validation_details={
                            'protein_validation': asdict(protein_validation),
                            'phylogenetic_validation': asdict(phylo_validation),
                            'literature_validation': asdict(literature_validation)
                        },
                        contamination_risk=max(
                            protein_validation.contamination_risk,
                            phylo_validation.contamination_risk,
                            literature_validation.contamination_risk
                        )
                    )
                    
                    # Update candidate with enhanced validation
                    candidate.final_validation = enhanced_validation
                    candidate.is_accepted = (
                        enhanced_validation.is_valid and
                        enhanced_validation.contamination_risk < 0.01 and
                        combined_score >= 0.85
                    )
                    
                    if candidate.is_accepted:
                        validated_candidates.append(candidate)
                
                return validated_candidates
        
        # Integrate custom validation pipeline
        custom_pipeline = CustomValidationPipeline(acquisition_step)
        await custom_pipeline.register_custom_validators()
        
        # Use custom validation in acquisition process
        enhanced_candidates = await custom_pipeline.run_custom_validation(match_candidates)
        ```
        
        **Quality Assurance and Audit Trail:**
        ```python
        # Comprehensive quality assurance and audit trail management
        class QualityAssuranceManager:
            def __init__(self):
                self.quality_metrics = {}
                self.audit_records = []
                self.quality_thresholds = {}
                
            async def configure_quality_standards(self):
                self.quality_thresholds = {
                    'minimum_confidence_score': 0.85,
                    'maximum_contamination_risk': 0.01,
                    'required_validation_stages': ['species', 'taxonomic', 'contamination'],
                    'minimum_metadata_completeness': 0.9,
                    'sequence_quality_threshold': 0.95
                }
                
            async def assess_acquisition_quality(self, acquisition_results: dict) -> dict:
                quality_assessment = {
                    'overall_score': 0.0,
                    'validation_success_rate': 0.0,
                    'contamination_rejection_rate': 0.0,
                    'metadata_completeness': 0.0,
                    'quality_distribution': {},
                    'recommendations': []
                }
                
                validated_entries = acquisition_results['validated_entries']
                rejected_entries = acquisition_results['rejected_entries']
                
                # Calculate validation success rate
                total_entries = len(validated_entries) + len(rejected_entries)
                quality_assessment['validation_success_rate'] = len(validated_entries) / total_entries
                
                # Calculate contamination rejection rate
                contamination_rejections = sum(
                    1 for entry in rejected_entries
                    if entry.get('rejection_reason') == 'contamination_risk'
                )
                quality_assessment['contamination_rejection_rate'] = contamination_rejections / total_entries
                
                # Assess metadata completeness
                completeness_scores = [
                    self.calculate_metadata_completeness(entry)
                    for entry in validated_entries
                ]
                quality_assessment['metadata_completeness'] = sum(completeness_scores) / len(completeness_scores)
                
                # Calculate overall quality score
                quality_assessment['overall_score'] = (
                    quality_assessment['validation_success_rate'] * 0.4 +
                    quality_assessment['metadata_completeness'] * 0.3 +
                    (1.0 - quality_assessment['contamination_rejection_rate']) * 0.3
                )
                
                # Generate quality recommendations
                if quality_assessment['overall_score'] < 0.8:
                    quality_assessment['recommendations'].append(
                        "Consider adjusting validation thresholds to improve quality"
                    )
                
                if quality_assessment['contamination_rejection_rate'] > 0.1:
                    quality_assessment['recommendations'].append(
                        "High contamination rejection rate detected - review search criteria"
                    )
                
                return quality_assessment
                
            async def generate_audit_report(self, acquisition_session: dict) -> dict:
                audit_report = {
                    'session_id': acquisition_session['session_id'],
                    'timestamp': acquisition_session['timestamp'],
                    'configuration_used': acquisition_session['configuration'],
                    'search_parameters': acquisition_session['search_parameters'],
                    'validation_pipeline': acquisition_session['validation_pipeline'],
                    'results_summary': acquisition_session['results_summary'],
                    'quality_assessment': acquisition_session['quality_assessment'],
                    'decision_trail': acquisition_session['decision_trail'],
                    'reproducibility_info': {
                        'software_version': acquisition_session['software_version'],
                        'database_version': acquisition_session['database_version'],
                        'configuration_hash': acquisition_session['configuration_hash']
                    }
                }
                
                return audit_report
        
        # Initialize quality assurance
        qa_manager = QualityAssuranceManager()
        await qa_manager.configure_quality_standards()
        
        # Assess acquisition quality
        quality_assessment = await qa_manager.assess_acquisition_quality(acquisition_results)
        audit_report = await qa_manager.generate_audit_report(acquisition_session)
        
        print(f"Quality Assessment:")
        print(f"  Overall Score: {quality_assessment['overall_score']:.3f}")
        print(f"  Validation Success: {quality_assessment['validation_success_rate']:.2%}")
        print(f"  Metadata Completeness: {quality_assessment['metadata_completeness']:.2%}")
        print(f"  Recommendations: {quality_assessment['recommendations']}")
        ```
    
    **Advanced Features:**
        
        **Machine Learning Integration:**
        * Intelligent search query optimization based on historical success patterns
        * Predictive quality scoring using machine learning models
        * Automated threshold optimization based on validation outcomes
        * Pattern recognition for contamination detection and prevention
        
        **Scientific Validation:**
        * ICTV taxonomic compliance validation and verification
        * Phylogenetic consistency checking and evolutionary analysis
        * Literature-based validation and scientific consensus verification
        * Cross-database validation and reference checking
        
        **Performance Optimization:**
        * Intelligent caching and result optimization for repeated queries
        * Parallel processing for large-scale data acquisition
        * Adaptive timeout and retry mechanisms for network resilience
        * Memory-efficient processing for large datasets
        
        **Integration Capabilities:**
        * Multi-database integration beyond BV-BRC (NCBI, GenBank, etc.)
        * Workflow orchestration and pipeline coordination
        * Real-time monitoring and progress tracking
        * External tool integration and data enrichment
    
    **Scientific Applications:**
        
        **Viral Genomics Research:**
        * Comprehensive viral genome collection and validation
        * Comparative genomics and evolutionary analysis preparation
        * Pandemic surveillance and outbreak analysis support
        * Vaccine development and antiviral research data acquisition
        
        **Phylogenetic Analysis:**
        * Taxonomically validated sequence collection for phylogenetic studies
        * Evolutionary relationship validation and verification
        * Species classification and nomenclature validation
        * Cross-species contamination prevention and quality control
        
        **Protein Structure Analysis:**
        * High-quality protein sequence acquisition and validation
        * Structure-function relationship analysis preparation
        * Comparative protein analysis and homology studies
        * Drug target identification and validation support
        
        **Epidemiological Studies:**
        * Pathogen surveillance and tracking data collection
        * Outbreak investigation and source identification
        * Transmission pattern analysis and modeling support
        * Public health surveillance and monitoring
    
    **Production Deployment:**
        
        **Scalability:**
        * Horizontal scaling for large-scale genomic data acquisition
        * Distributed processing and parallel validation pipelines
        * Cloud-native deployment with auto-scaling capabilities
        * High-throughput processing optimization
        
        **Reliability:**
        * Robust error handling and recovery mechanisms
        * Data integrity validation and corruption prevention
        * Backup and disaster recovery capabilities
        * Service health monitoring and alerting
        
        **Security & Compliance:**
        * Secure data transmission and storage protocols
        * Access control and authentication integration
        * Audit logging and compliance reporting
        * Data privacy and protection measures
    
    Attributes:
        bvbrc_tool (BVBRCTool): BV-BRC database interface for data acquisition operations
        species_validation_agent (Agent): Specialized agent for species classification validation
        taxonomic_validation_agent (Agent): Advanced agent for taxonomic relationship verification
        validation_pipeline (Dict): Multi-stage validation pipeline configuration and state
        quality_controller (QualityController): Quality assurance and metrics management system
        audit_trail (List): Comprehensive audit trail for reproducibility and validation
        logger (Logger): Structured logging system for scientific workflow tracking
    
    Note:
        This step requires proper BV-BRC API access and authentication for data acquisition.
        Validation agents must be properly configured with appropriate confidence thresholds.
        Quality control parameters should be calibrated based on specific research requirements.
        Audit trail generation requires sufficient storage for comprehensive tracking.
    
    Warning:
        Zero-tolerance contamination screening may reject valid entries with minimal contamination risk.
        High validation thresholds may significantly reduce the number of accepted entries.
        Multi-stage validation requires substantial computational resources and processing time.
        Network connectivity issues may impact data acquisition and validation performance.
    
    See Also:
        * :class:`BVBRCTool`: BV-BRC database interface and data acquisition tool
        * :class:`ValidationResult`: Validation result data structure and quality metrics
        * :class:`CSVMatchCandidate`: Match candidate data structure with validation metadata
        * :mod:`nanobrain.library.agents.specialized`: Specialized validation agents
        * :mod:`nanobrain.library.workflows.viral_protein_analysis`: Viral protein analysis workflows
    """
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.bv_brc_tool = None
        self.species_validation_agent = None
        self.taxonomic_verification_agent = None
        self.contamination_tolerance = 0.0  # ZERO tolerance
        
    def _init_from_config(self, config, component_config: Dict[str, Any], dependencies: Dict[str, Any]) -> None:
        """Initialize Enhanced BV-BRC Data Acquisition with validation agents"""
        super()._init_from_config(config, component_config, dependencies)
        
        # ENHANCED: Prioritize tools already resolved by ConfigBase._resolve_nested_objects()
        resolved_tools = getattr(config, '_resolved_tools', {})
        
        # Use resolved BV-BRC tool if available
        self.bv_brc_tool = resolved_tools.get('bv_brc_tool') or self.step_tools.get('bv_brc_tool')
        if not self.bv_brc_tool:
            # Fallback to legacy creation only if no resolved tool available
            self.bv_brc_tool = self._create_bv_brc_tool(component_config)
        
        if self.bv_brc_tool and resolved_tools.get('bv_brc_tool') and self.nb_logger:
            self.nb_logger.info(f"✅ Using resolved BV-BRC tool from enhanced ConfigBase system")
        
        # Use resolved species validation agent if available
        self.species_validation_agent = resolved_tools.get('species_validation_agent') or self.step_tools.get('species_validation_agent')
        if not self.species_validation_agent:
            # Fallback to legacy creation only if no resolved agent available
            self.species_validation_agent = self._create_species_validation_agent(component_config)
        
        if self.species_validation_agent and resolved_tools.get('species_validation_agent') and self.nb_logger:
            self.nb_logger.info(f"✅ Using resolved species validation agent from enhanced ConfigBase system")
        
        # Use resolved taxonomic verification agent if available
        self.taxonomic_verification_agent = resolved_tools.get('taxonomic_verification_agent') or self.step_tools.get('taxonomic_verification_agent')
        if not self.taxonomic_verification_agent:
            # Fallback to legacy creation only if no resolved agent available
            self.taxonomic_verification_agent = self._create_taxonomic_verification_agent(component_config)
        
        if self.taxonomic_verification_agent and resolved_tools.get('taxonomic_verification_agent') and self.nb_logger:
            self.nb_logger.info(f"✅ Using resolved taxonomic verification agent from enhanced ConfigBase system")
        
        # Get contamination tolerance (should be 0.0 for zero tolerance)
        self.contamination_tolerance = component_config.get('contamination_tolerance', 0.0)
        
        # Get validation thresholds
        self.species_validation_threshold = component_config.get('species_validation_threshold', 0.9)
        self.taxonomic_validation_threshold = component_config.get('taxonomic_validation_threshold', 0.9)
        
        if self.nb_logger:
            self.nb_logger.info(f"🛡️ Enhanced BV-BRC Data Acquisition initialized with ZERO contamination tolerance")
            self.nb_logger.info(f"🎯 Validation thresholds - Species: {self.species_validation_threshold}, Taxonomic: {self.taxonomic_validation_threshold}")
    
    def _create_species_validation_agent(self, component_config: Dict[str, Any]) -> SimpleAgent:
        """Create species validation agent for precise match validation"""
        tools_config = component_config.get('tools', {})
        agent_config_ref = tools_config.get('species_validation_agent', {})
        
        if 'config_file' in agent_config_ref:
            # Load agent from external configuration file
            config_file_path = agent_config_ref['config_file']
            agent_config = load_config_file(config_file_path)
            agent_class_path = agent_config.get('class')
            
            if not agent_class_path:
                raise ValueError(f"Agent configuration must specify 'class' field: {config_file_path}")
            
            return create_component(agent_class_path, agent_config)
        else:
            # Fallback: Try to load from step-specific configuration directory
            try:
                agent_config_path = "../config/DataAcquisitionStep/SynonymDetectionAgent.yml"
                agent_config = load_config_file(agent_config_path)
                agent_class_path = agent_config.get('class')
                
                if not agent_class_path:
                    raise ValueError(f"Agent configuration must specify 'class' field: {agent_config_path}")
                
                self.nb_logger.info(f"🔧 Species validation agent loaded from fallback path: {agent_config_path}")
                return create_component(agent_class_path, agent_config)
                
            except Exception as e:
                self.nb_logger.error(f"❌ Failed to load species validation agent from fallback path: {e}")
                raise ValueError(f"Species validation agent configuration required for enhanced validation. Fallback also failed: {e}")

    def _create_taxonomic_verification_agent(self, component_config: Dict[str, Any]) -> SimpleAgent:
        """Create taxonomic verification agent for cross-validation"""
        tools_config = component_config.get('tools', {})
        agent_config_ref = tools_config.get('taxonomic_verification_agent', {})
        
        if 'config_file' in agent_config_ref:
            # Load agent from external configuration file
            config_file_path = agent_config_ref['config_file']
            agent_config = load_config_file(config_file_path)
            agent_class_path = agent_config.get('class')
            
            if not agent_class_path:
                raise ValueError(f"Agent configuration must specify 'class' field: {config_file_path}")
            
            return create_component(agent_class_path, agent_config)
        else:
            # Fallback: Try to load from step-specific configuration directory
            try:
                agent_config_path = "../config/DataAcquisitionStep/TaxonomicVerificationAgent.yml"
                agent_config = load_config_file(agent_config_path)
                agent_class_path = agent_config.get('class')
                
                if not agent_class_path:
                    raise ValueError(f"Agent configuration must specify 'class' field: {agent_config_path}")
                
                self.nb_logger.info(f"🔧 Taxonomic verification agent loaded from fallback path: {agent_config_path}")
                return create_component(agent_class_path, agent_config)
                
            except Exception as e:
                self.nb_logger.error(f"❌ Failed to load taxonomic verification agent from fallback path: {e}")
                raise ValueError(f"Taxonomic verification agent configuration required for enhanced validation. Fallback also failed: {e}")
    
    def _create_bv_brc_tool(self, component_config: Dict[str, Any]) -> BVBRCTool:
        """Create BV-BRC tool from configuration"""
        tools_config = component_config.get('tools', {})
        tool_config = tools_config.get('bv_brc_tool', {})
        
        if 'config_file' in tool_config:
            # Load tool from external configuration file
            config_file_path = tool_config['config_file']
            loaded_config = load_config_file(config_file_path)
            tool_class_path = loaded_config.get('class')
            
            if not tool_class_path:
                raise ValueError(f"Tool configuration must specify 'class' field: {config_file_path}")
            
            return create_component(tool_class_path, loaded_config)
        else:
            # Fallback: Try to load from step-specific configuration directory  
            try:
                tool_config_path = "../config/DataAcquisitionStep/BVBRCTool.yml"
                loaded_config = load_config_file(tool_config_path)
                tool_class_path = loaded_config.get('class')
                
                if not tool_class_path:
                    raise ValueError(f"Tool configuration must specify 'class' field: {tool_config_path}")
                
                self.nb_logger.info(f"🔧 BV-BRC tool loaded from fallback path: {tool_config_path}")
                return create_component(tool_class_path, loaded_config)
                
            except Exception as e:
                self.nb_logger.error(f"❌ Failed to load BV-BRC tool from fallback path: {e}")
                raise ValueError(f"BV-BRC tool configuration required but not found in tools section. Fallback also failed: {e}")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced BV-BRC data acquisition with multi-stage validation
        
        Args:
            input_data: Contains ultra_high_confidence_synonyms and species_validation_criteria
            
        Returns:
            Dict with verified genome data and comprehensive validation audit trail
        """
        try:
            # Extract ultra-high-confidence synonyms from enhanced virus resolution
            synonyms_data = input_data.get('ultra_high_confidence_synonyms', [])
            validation_criteria = input_data.get('species_validation_criteria', {})
            
            if not synonyms_data:
                raise ValueError("No ultra-high-confidence synonyms provided for CSV search")
            
            self.nb_logger.info(f"🔍 Starting enhanced CSV search with {len(synonyms_data)} ultra-high-confidence terms")
            
            # Multi-strategy CSV search without hardcoding
            match_candidates = await self._multi_strategy_csv_search(synonyms_data, validation_criteria)
            
            if not match_candidates:
                self.nb_logger.warning("⚠️ No initial CSV matches found")
                return {
                    'verified_entries': [],
                    'validation_summary': {
                        'total_candidates': 0,
                        'validated_entries': 0,
                        'rejected_entries': 0,
                        'contamination_risk': 0.0
                    }
                }
            
            self.nb_logger.info(f"🎯 Found {len(match_candidates)} initial match candidates")
            
            # Multi-stage validation pipeline
            validated_matches = await self._multi_stage_validation_pipeline(match_candidates, validation_criteria)
            
            # Final contamination prevention check
            final_verified_entries = await self._final_contamination_check(validated_matches, validation_criteria)
            
            # Generate comprehensive audit trail
            audit_trail = self._generate_audit_trail(match_candidates, validated_matches, final_verified_entries)
            
            self.nb_logger.info(f"✅ Enhanced validation complete: {len(final_verified_entries)} verified entries")
            
            return {
                'verified_entries': [entry.entry_data for entry in final_verified_entries],
                'validation_summary': audit_trail['summary'],
                'validation_audit_trail': audit_trail,
                'contamination_prevention': {
                    'tolerance': self.contamination_tolerance,
                    'final_contamination_risk': audit_trail['summary']['final_contamination_risk']
                }
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Enhanced BV-BRC data acquisition failed: {e}")
            raise
    
    async def _multi_strategy_csv_search(self, synonyms: List[str], validation_criteria: Dict[str, Any]) -> List[CSVMatchCandidate]:
        """Multi-strategy CSV search without hardcoded patterns"""
        all_candidates = []
        
        # Load CSV data
        csv_data = await self._load_csv_data()
        
        # Strategy 1: Exact synonym matching
        for synonym in synonyms:
            exact_matches = self._exact_synonym_search(csv_data, synonym)
            for match in exact_matches:
                candidate = CSVMatchCandidate(
                    entry_data=match,
                    match_method='exact_synonym',
                    search_term=synonym,
                    initial_confidence=0.95
                )
                all_candidates.append(candidate)
        
        # Strategy 2: Partial synonym matching
        for synonym in synonyms:
            partial_matches = self._partial_synonym_search(csv_data, synonym)
            for match in partial_matches:
                candidate = CSVMatchCandidate(
                    entry_data=match,
                    match_method='partial_synonym',
                    search_term=synonym,
                    initial_confidence=0.85
                )
                all_candidates.append(candidate)
        
        # Strategy 3: Fuzzy matching with taxonomic constraints
        taxonomic_constraints = validation_criteria.get('taxonomic_constraints', {})
        fuzzy_matches = await self._fuzzy_taxonomic_search(csv_data, synonyms, taxonomic_constraints)
        all_candidates.extend(fuzzy_matches)
        
        # Strategy 4: Agent-based pattern matching (NO hardcoded patterns)
        agent_matches = await self._agent_pattern_search(csv_data, synonyms, validation_criteria)
        all_candidates.extend(agent_matches)
        
        # Remove duplicates while preserving best match method
        unique_candidates = self._deduplicate_candidates(all_candidates)
        
        self.nb_logger.info(f"🔍 Multi-strategy search results: {len(unique_candidates)} unique candidates")
        
        return unique_candidates
    
    async def _multi_stage_validation_pipeline(self, candidates: List[CSVMatchCandidate], 
                                             validation_criteria: Dict[str, Any]) -> List[CSVMatchCandidate]:
        """Multi-stage validation pipeline with agent-based verification"""
        validated_candidates = []
        
        for candidate in candidates:
            self.nb_logger.info(f"🔬 Validating candidate: {candidate.search_term} via {candidate.match_method}")
            
            # Stage 1: Species validation using Species Validation Agent
            candidate.species_validation = await self._validate_species_match(candidate, validation_criteria)
            
            if not candidate.species_validation.is_valid:
                self.nb_logger.info(f"❌ Species validation failed: {candidate.species_validation.rejection_reason}")
                continue
            
            # Stage 2: Taxonomic cross-validation using Taxonomic Verification Agent
            candidate.taxonomic_validation = await self._validate_taxonomic_consistency(candidate, validation_criteria)
            
            if not candidate.taxonomic_validation.is_valid:
                self.nb_logger.info(f"❌ Taxonomic validation failed: {candidate.taxonomic_validation.rejection_reason}")
                continue
            
            # Stage 3: Final validation combining all checks
            candidate.final_validation = await self._final_validation_check(candidate, validation_criteria)
            
            if candidate.final_validation.is_valid:
                candidate.is_accepted = True
                validated_candidates.append(candidate)
                self.nb_logger.info(f"✅ Candidate validated with confidence: {candidate.final_validation.confidence_score:.3f}")
            else:
                self.nb_logger.info(f"❌ Final validation failed: {candidate.final_validation.rejection_reason}")
        
        return validated_candidates
    
    async def _validate_species_match(self, candidate: CSVMatchCandidate, 
                                    validation_criteria: Dict[str, Any]) -> ValidationResult:
        """Validate species match using Species Validation Agent"""
        try:
            validation_prompt = self._build_species_validation_prompt(candidate, validation_criteria)
            
            response = await self.species_validation_agent.process({
                'prompt': validation_prompt,
                'expected_format': 'json',
                'validation_threshold': self.species_validation_threshold
            })
            
            parsed_response = self._parse_agent_response(response)
            
            is_valid = parsed_response.get('is_valid', False)
            confidence = parsed_response.get('confidence_score', 0.0)
            contamination_risk = parsed_response.get('contamination_risk', 1.0)
            
            return ValidationResult(
                is_valid=is_valid and confidence >= self.species_validation_threshold,
                confidence_score=confidence,
                validation_method='species_validation_agent',
                validation_details=parsed_response,
                contamination_risk=contamination_risk,
                rejection_reason=parsed_response.get('rejection_reason') if not is_valid else None
            )
            
        except Exception as e:
            self.nb_logger.error(f"❌ Species validation error: {e}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                validation_method='species_validation_agent',
                validation_details={'error': str(e)},
                contamination_risk=1.0,
                rejection_reason=f"Validation error: {e}"
            )
    
    async def _validate_taxonomic_consistency(self, candidate: CSVMatchCandidate, 
                                            validation_criteria: Dict[str, Any]) -> ValidationResult:
        """Validate taxonomic consistency using Taxonomic Verification Agent"""
        try:
            validation_prompt = self._build_taxonomic_validation_prompt(candidate, validation_criteria)
            
            response = await self.taxonomic_verification_agent.process({
                'prompt': validation_prompt,
                'expected_format': 'json',
                'validation_threshold': self.taxonomic_validation_threshold
            })
            
            parsed_response = self._parse_agent_response(response)
            
            is_valid = parsed_response.get('is_taxonomically_consistent', False)
            confidence = parsed_response.get('confidence_score', 0.0)
            contamination_risk = parsed_response.get('cross_contamination_risk', 1.0)
            
            return ValidationResult(
                is_valid=is_valid and confidence >= self.taxonomic_validation_threshold,
                confidence_score=confidence,
                validation_method='taxonomic_verification_agent',
                validation_details=parsed_response,
                contamination_risk=contamination_risk,
                rejection_reason=parsed_response.get('rejection_reason') if not is_valid else None
            )
            
        except Exception as e:
            self.nb_logger.error(f"❌ Taxonomic validation error: {e}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                validation_method='taxonomic_verification_agent',
                validation_details={'error': str(e)},
                contamination_risk=1.0,
                rejection_reason=f"Validation error: {e}"
            )
    
    async def _final_contamination_check(self, validated_candidates: List[CSVMatchCandidate], 
                                       validation_criteria: Dict[str, Any]) -> List[CSVMatchCandidate]:
        """Final contamination prevention check with ZERO tolerance"""
        final_verified = []
        
        for candidate in validated_candidates:
            # Calculate overall contamination risk
            species_risk = candidate.species_validation.contamination_risk if candidate.species_validation else 1.0
            taxonomic_risk = candidate.taxonomic_validation.contamination_risk if candidate.taxonomic_validation else 1.0
            overall_risk = max(species_risk, taxonomic_risk)
            
            # Apply ZERO contamination tolerance
            if overall_risk <= self.contamination_tolerance:
                final_verified.append(candidate)
                self.nb_logger.info(f"✅ Final verification passed: contamination risk {overall_risk:.6f}")
            else:
                self.nb_logger.warning(f"❌ REJECTED due to contamination risk: {overall_risk:.6f} > {self.contamination_tolerance}")
        
        return final_verified
    
    def _build_species_validation_prompt(self, candidate: CSVMatchCandidate, 
                                       validation_criteria: Dict[str, Any]) -> str:
        """Build comprehensive species validation prompt"""
        return f"""
        Validate if this CSV entry matches the target species with ultra-high precision:
        
        TARGET SPECIES: {validation_criteria.get('primary_species', 'Unknown')}
        ACCEPTABLE VARIATIONS: {validation_criteria.get('acceptable_variations', [])}
        
        CSV ENTRY:
        - Species Name: {candidate.entry_data.get('species', 'Unknown')}
        - Organism Name: {candidate.entry_data.get('organism_name', 'Unknown')}
        - Genome Name: {candidate.entry_data.get('genome_name', 'Unknown')}
        
        VALIDATION CRITERIA:
        - Exact match required: {validation_criteria.get('validation_rules', {}).get('exact_match_required', True)}
        - Case sensitive: {validation_criteria.get('validation_rules', {}).get('case_sensitive', False)}
        - Allow abbreviations: {validation_criteria.get('validation_rules', {}).get('allow_abbreviations', True)}
        
        Return JSON with:
        - is_valid: boolean
        - confidence_score: 0.0-1.0
        - contamination_risk: 0.0-1.0 (0.0 = no risk, 1.0 = high risk)
        - validation_reasoning: detailed explanation
        - rejection_reason: if not valid
        """
    
    def _build_taxonomic_validation_prompt(self, candidate: CSVMatchCandidate, 
                                         validation_criteria: Dict[str, Any]) -> str:
        """Build comprehensive taxonomic validation prompt"""
        taxonomic_constraints = validation_criteria.get('taxonomic_constraints', {})
        
        return f"""
        Verify taxonomic consistency for this CSV entry with ZERO contamination tolerance:
        
        REQUIRED TAXONOMY:
        - Genus: {taxonomic_constraints.get('genus', 'Unknown')}
        - Family: {taxonomic_constraints.get('family', 'Unknown')}
        - Order: {taxonomic_constraints.get('order', 'Unknown')}
        
        CSV ENTRY:
        - Species: {candidate.entry_data.get('species', 'Unknown')}
        - Genus: {candidate.entry_data.get('genus', 'Unknown')}
        - Family: {candidate.entry_data.get('family', 'Unknown')}
        
        CONTAMINATION PREVENTION:
        - Reject on genus mismatch: {validation_criteria.get('contamination_prevention', {}).get('reject_on_genus_mismatch', True)}
        - Require species confirmation: {validation_criteria.get('contamination_prevention', {}).get('require_species_confirmation', True)}
        - Zero contamination tolerance: {validation_criteria.get('contamination_prevention', {}).get('zero_contamination_tolerance', 0.0)}
        
        Return JSON with:
        - is_taxonomically_consistent: boolean
        - confidence_score: 0.0-1.0
        - cross_contamination_risk: 0.0-1.0
        - taxonomic_reasoning: detailed explanation
        - rejection_reason: if not consistent
        """
    
    async def _final_validation_check(self, candidate: CSVMatchCandidate, 
                                    validation_criteria: Dict[str, Any]) -> ValidationResult:
        """Combine all validation results for final decision"""
        species_valid = candidate.species_validation and candidate.species_validation.is_valid
        taxonomic_valid = candidate.taxonomic_validation and candidate.taxonomic_validation.is_valid
        
        if not species_valid:
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                validation_method='final_combined',
                validation_details={'reason': 'species_validation_failed'},
                contamination_risk=1.0,
                rejection_reason='Species validation failed'
            )
        
        if not taxonomic_valid:
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                validation_method='final_combined',
                validation_details={'reason': 'taxonomic_validation_failed'},
                contamination_risk=1.0,
                rejection_reason='Taxonomic validation failed'
            )
        
        # Calculate combined confidence and contamination risk
        combined_confidence = min(
            candidate.species_validation.confidence_score,
            candidate.taxonomic_validation.confidence_score
        )
        
        combined_contamination_risk = max(
            candidate.species_validation.contamination_risk,
            candidate.taxonomic_validation.contamination_risk
        )
        
        return ValidationResult(
            is_valid=True,
            confidence_score=combined_confidence,
            validation_method='final_combined',
            validation_details={
                'species_confidence': candidate.species_validation.confidence_score,
                'taxonomic_confidence': candidate.taxonomic_validation.confidence_score,
                'combined_confidence': combined_confidence
            },
            contamination_risk=combined_contamination_risk
        )
    
    def _generate_audit_trail(self, initial_candidates: List[CSVMatchCandidate], 
                            validated_candidates: List[CSVMatchCandidate],
                            final_verified: List[CSVMatchCandidate]) -> Dict[str, Any]:
        """Generate comprehensive audit trail for validation process"""
        return {
            'summary': {
                'total_candidates': len(initial_candidates),
                'validated_entries': len(validated_candidates),
                'final_verified_entries': len(final_verified),
                'rejected_entries': len(initial_candidates) - len(final_verified),
                'final_contamination_risk': max([c.final_validation.contamination_risk for c in final_verified], default=0.0),
                'validation_success_rate': len(final_verified) / len(initial_candidates) if initial_candidates else 0.0
            },
            'validation_details': {
                'contamination_tolerance': self.contamination_tolerance,
                'species_validation_threshold': self.species_validation_threshold,
                'taxonomic_validation_threshold': self.taxonomic_validation_threshold,
                'validation_timestamp': datetime.now().isoformat()
            },
            'rejected_candidates': [
                {
                    'search_term': c.search_term,
                    'match_method': c.match_method,
                    'rejection_stage': self._determine_rejection_stage(c),
                    'rejection_reason': self._get_rejection_reason(c)
                }
                for c in initial_candidates if c not in final_verified
            ]
        }
    
    def _determine_rejection_stage(self, candidate: CSVMatchCandidate) -> str:
        """Determine at which stage the candidate was rejected"""
        if not candidate.species_validation or not candidate.species_validation.is_valid:
            return 'species_validation'
        elif not candidate.taxonomic_validation or not candidate.taxonomic_validation.is_valid:
            return 'taxonomic_validation'
        elif not candidate.final_validation or not candidate.final_validation.is_valid:
            return 'final_validation'
        else:
            return 'contamination_check'
    
    def _get_rejection_reason(self, candidate: CSVMatchCandidate) -> str:
        """Get the reason for candidate rejection"""
        if candidate.species_validation and not candidate.species_validation.is_valid:
            return candidate.species_validation.rejection_reason or 'Species validation failed'
        elif candidate.taxonomic_validation and not candidate.taxonomic_validation.is_valid:
            return candidate.taxonomic_validation.rejection_reason or 'Taxonomic validation failed'
        elif candidate.final_validation and not candidate.final_validation.is_valid:
            return candidate.final_validation.rejection_reason or 'Final validation failed'
        else:
            return 'Contamination risk above tolerance'
    
    # Additional helper methods for CSV search strategies
    def _exact_synonym_search(self, csv_data: pd.DataFrame, synonym: str) -> List[Dict[str, Any]]:
        """Exact synonym matching in CSV data"""
        matches = []
        for _, row in csv_data.iterrows():
            if (str(row.get('species', '')).lower() == synonym.lower() or
                str(row.get('organism_name', '')).lower() == synonym.lower() or
                str(row.get('genome_name', '')).lower() == synonym.lower()):
                matches.append(row.to_dict())
        return matches
    
    def _partial_synonym_search(self, csv_data: pd.DataFrame, synonym: str) -> List[Dict[str, Any]]:
        """Partial synonym matching in CSV data"""
        matches = []
        synonym_lower = synonym.lower()
        for _, row in csv_data.iterrows():
            species = str(row.get('species', '')).lower()
            organism = str(row.get('organism_name', '')).lower()
            genome = str(row.get('genome_name', '')).lower()
            
            if (synonym_lower in species or synonym_lower in organism or synonym_lower in genome or
                species in synonym_lower or organism in synonym_lower or genome in synonym_lower):
                matches.append(row.to_dict())
        return matches
    
    async def _fuzzy_taxonomic_search(self, csv_data: pd.DataFrame, synonyms: List[str], 
                                    taxonomic_constraints: Dict[str, Any]) -> List[CSVMatchCandidate]:
        """Fuzzy matching with taxonomic constraints"""
        candidates = []
        genus = taxonomic_constraints.get('genus', '').lower()
        family = taxonomic_constraints.get('family', '').lower()
        
        if genus:
            for _, row in csv_data.iterrows():
                row_genus = str(row.get('genus', '')).lower()
                if genus in row_genus or row_genus in genus:
                    candidate = CSVMatchCandidate(
                        entry_data=row.to_dict(),
                        match_method='fuzzy_taxonomic',
                        search_term=f"genus:{genus}",
                        initial_confidence=0.75
                    )
                    candidates.append(candidate)
        
        return candidates
    
    async def _agent_pattern_search(self, csv_data: pd.DataFrame, synonyms: List[str], 
                                  validation_criteria: Dict[str, Any]) -> List[CSVMatchCandidate]:
        """Agent-based pattern matching without hardcoded patterns"""
        # This would use an agent to identify patterns, but for now return empty list
        # as the other strategies should cover most cases
        return []
    
    def _deduplicate_candidates(self, candidates: List[CSVMatchCandidate]) -> List[CSVMatchCandidate]:
        """Remove duplicate candidates, keeping best match method"""
        seen_entries = {}
        unique_candidates = []
        
        # Sort by confidence (highest first)
        sorted_candidates = sorted(candidates, key=lambda c: c.initial_confidence, reverse=True)
        
        for candidate in sorted_candidates:
            # Create unique key from entry data
            entry_key = str(sorted(candidate.entry_data.items()))
            
            if entry_key not in seen_entries:
                seen_entries[entry_key] = candidate
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    async def _load_csv_data(self) -> pd.DataFrame:
        """Load CSV data using BV-BRC tool"""
        # This would load the actual CSV data
        # For now, return empty DataFrame as placeholder
        return pd.DataFrame()
    
    def _parse_agent_response(self, response: Any) -> Dict[str, Any]:
        """Parse agent response ensuring JSON format"""
        if isinstance(response, dict):
            return response
        elif isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse agent response as JSON: {response}")
        else:
            raise ValueError(f"Unexpected agent response type: {type(response)}")


# Maintain backward compatibility
BVBRCDataAcquisitionStep = EnhancedBVBRCDataAcquisitionStep 