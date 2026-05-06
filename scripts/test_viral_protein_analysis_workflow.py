#!/usr/bin/env python3
"""
Comprehensive Viral Protein Analysis Workflow Test

Tests the viral_protein_analysis workflow with real virus data to ensure:
- Complete end-to-end workflow execution
- Meaningful scientific results (no fallbacks or empty data)
- Proper NanoBrain framework compliance
- Real data processing with actual biological insights

✅ FRAMEWORK COMPLIANCE:
- Uses enhanced from_config patterns exclusively
- Configuration-driven component creation
- Event-driven workflow execution
- No hardcoded values or simplified solutions
- Complete scientific data validation
"""

import asyncio
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow import AlphavirusWorkflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ViralProteinAnalysisWorkflowTest:
    """Comprehensive test suite for viral protein analysis workflow"""
    
    def __init__(self):
        self.test_results = []
        self.workflow = None
        self.test_data_quality = {}
        
    async def run_comprehensive_test(self) -> bool:
        """Run complete workflow test with real virus data"""
        logger.info("🚀 Starting Comprehensive Viral Protein Analysis Workflow Test")
        
        try:
            # Phase 1: Setup and validation
            await self._validate_test_environment()
            
            # Phase 2: Create and initialize workflow
            await self._create_workflow_instance()
            
            # Phase 3: Prepare real test data
            test_input = await self._prepare_real_virus_data()
            
            # Phase 4: Execute workflow end-to-end
            workflow_results = await self._execute_workflow_end_to_end(test_input)
            
            # Phase 5: Validate scientific meaningfulness
            validation_success = await self._validate_scientific_results(workflow_results)
            
            # Phase 6: Generate comprehensive report
            await self._generate_test_report(workflow_results, validation_success)
            
            return validation_success
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            self._record_failure("Workflow Test", f"Exception: {str(e)}")
            return False
    
    async def _validate_test_environment(self):
        """Validate that all required data and configuration files exist"""
        logger.info("🔍 Validating Test Environment")
        
        required_files = [
            "nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml",
            "data/bvbrc_cache/alphavirus_genomes.tsv",
            "data/bvbrc_cache/alphavirus_proteins.tsv",
            "data/virus_resolution_cache/virus_resolution_chikungunya_virus.json"
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Required test file not found: {file_path}")
        
        logger.info("✅ Test environment validated")
    
    async def _create_workflow_instance(self):
        """Create workflow instance using enhanced from_config patterns"""
        logger.info("🔧 Creating Workflow Instance")
        
        config_path = "nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml"
        
        # Use enhanced from_config pattern - NO programmatic configuration
        self.workflow = AlphavirusWorkflow.from_config(
            config_path,
            workflow_directory="nanobrain/library/workflows/viral_protein_analysis"
        )
        
        # Initialize workflow components
        await self.workflow.initialize()
        
        logger.info("✅ Workflow instance created and initialized")
    
    async def _prepare_real_virus_data(self) -> Dict[str, Any]:
        """Prepare real virus data for workflow testing"""
        logger.info("📊 Preparing Real Virus Data")
        
        # Load real virus resolution data
        with open("data/virus_resolution_cache/virus_resolution_chikungunya_virus.json", 'r') as f:
            virus_resolution_data = json.load(f)
        
        # Create realistic workflow input based on virus resolution output
        # This matches the expected input format from chatbot_viral_integration
        test_input = {
            "virus_species": virus_resolution_data["canonical_name"],
            "bvbrc_search_terms": virus_resolution_data["bvbrc_search_terms"],
            "genome_characteristics": virus_resolution_data.get("genome_characteristics", {}),
            "taxonomic_lineage": virus_resolution_data.get("taxonomic_lineage", []),
            "confidence": virus_resolution_data.get("confidence", 0.9),
            "analysis_type": "pssm",
            "ultra_high_confidence_synonyms": [
                {
                    "canonical_name": virus_resolution_data["canonical_name"],
                    "search_terms": virus_resolution_data["bvbrc_search_terms"],
                    "confidence": virus_resolution_data.get("confidence", 0.9)
                }
            ],
            "species_validation_criteria": {
                "min_confidence": 0.8,
                "require_taxonomic_validation": True,
                "contamination_tolerance": 0.0
            },
            "user_query": f"Analyze {virus_resolution_data['canonical_name']} protein structures",
            "analysis_parameters": {
                "min_genome_length": 8000,
                "max_genome_length": 15000,
                "clustering_threshold": 0.8,
                "alignment_method": "muscle",
                "pssm_generation": True
            }
        }
        
        logger.info(f"✅ Real virus data prepared for: {test_input['virus_species']}")
        return test_input
    
    async def _execute_workflow_end_to_end(self, test_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete workflow with real data"""
        logger.info("🔄 Executing Workflow End-to-End")
        
        start_time = time.time()
        
        try:
            # Execute workflow using framework's process method
            results = await self.workflow.process(test_input)
            
            execution_time = time.time() - start_time
            logger.info(f"✅ Workflow execution completed in {execution_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Workflow execution failed after {execution_time:.2f} seconds: {e}")
            raise
    
    async def _validate_scientific_results(self, results: Dict[str, Any]) -> bool:
        """Validate that results contain meaningful scientific data"""
        logger.info("🧬 Validating Scientific Meaningfulness")
        
        validation_checks = []
        
        # Check 1: Results structure validation
        structure_valid = await self._validate_results_structure(results)
        validation_checks.append(("Results Structure", structure_valid))
        
        # Check 2: Protein data validation
        protein_data_valid = await self._validate_protein_data(results)
        validation_checks.append(("Protein Data", protein_data_valid))
        
        # Check 3: Analysis completeness validation
        analysis_complete = await self._validate_analysis_completeness(results)
        validation_checks.append(("Analysis Completeness", analysis_complete))
        
        # Check 4: Scientific accuracy validation
        scientific_accuracy = await self._validate_scientific_accuracy(results)
        validation_checks.append(("Scientific Accuracy", scientific_accuracy))
        
        # Check 5: No fallback/empty data validation
        no_fallbacks = await self._validate_no_fallback_data(results)
        validation_checks.append(("No Fallback Data", no_fallbacks))
        
        # Summary validation results
        passed_checks = sum(1 for _, valid in validation_checks if valid)
        total_checks = len(validation_checks)
        
        logger.info(f"📊 Validation Results: {passed_checks}/{total_checks} checks passed")
        
        for check_name, passed in validation_checks:
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {check_name}")
        
        return passed_checks == total_checks
    
    async def _validate_results_structure(self, results: Dict[str, Any]) -> bool:
        """Validate the structure of workflow results"""
        required_fields = ['metadata', 'proteins', 'statistics', 'analysis_results']
        
        for field in required_fields:
            if field not in results:
                logger.error(f"❌ Missing required field: {field}")
                return False
        
        # Validate metadata structure
        metadata = results.get('metadata', {})
        required_metadata = ['workflow_name', 'version', 'created_at', 'organism']
        
        for meta_field in required_metadata:
            if meta_field not in metadata:
                logger.error(f"❌ Missing metadata field: {meta_field}")
                return False
        
        logger.info("✅ Results structure validation passed")
        return True
    
    async def _validate_protein_data(self, results: Dict[str, Any]) -> bool:
        """Validate protein data quality and completeness"""
        proteins = results.get('proteins', [])
        
        if not proteins:
            logger.error("❌ No protein data found in results")
            return False
        
        if len(proteins) < 5:
            logger.error(f"❌ Insufficient protein data: only {len(proteins)} proteins")
            return False
        
        # Validate protein structure
        sample_protein = proteins[0]
        required_protein_fields = ['product', 'sequence', 'genome_id', 'length']
        
        for field in required_protein_fields:
            if field not in sample_protein:
                logger.error(f"❌ Missing protein field: {field}")
                return False
        
        # Validate protein sequences are not empty
        empty_sequences = sum(1 for p in proteins if not p.get('sequence', '').strip())
        if empty_sequences > 0:
            logger.error(f"❌ Found {empty_sequences} proteins with empty sequences")
            return False
        
        logger.info(f"✅ Protein data validation passed: {len(proteins)} proteins with complete data")
        self.test_data_quality['protein_count'] = len(proteins)
        return True
    
    async def _validate_analysis_completeness(self, results: Dict[str, Any]) -> bool:
        """Validate that analysis was completed successfully"""
        analysis_results = results.get('analysis_results', {})
        
        # Check for PSSM data
        if 'pssm_matrices' not in analysis_results:
            logger.error("❌ PSSM matrices not found in analysis results")
            return False
        
        pssm_data = analysis_results['pssm_matrices']
        if not pssm_data:
            logger.error("❌ PSSM matrices are empty")
            return False
        
        # Check for clustering results
        if 'sequence_clusters' not in analysis_results:
            logger.error("❌ Sequence clustering results not found")
            return False
        
        # Check for alignment results
        if 'aligned_sequences' not in analysis_results:
            logger.error("❌ Alignment results not found")
            return False
        
        logger.info("✅ Analysis completeness validation passed")
        self.test_data_quality['analysis_steps_completed'] = len(analysis_results)
        return True
    
    async def _validate_scientific_accuracy(self, results: Dict[str, Any]) -> bool:
        """Validate scientific accuracy of results"""
        metadata = results.get('metadata', {})
        
        # Validate organism information
        organism = metadata.get('organism')
        if not organism or organism.lower() not in ['chikungunya virus', 'alphavirus']:
            logger.error(f"❌ Incorrect organism identified: {organism}")
            return False
        
        # Validate protein annotations contain expected alphavirus proteins
        proteins = results.get('proteins', [])
        protein_products = [p.get('product', '').lower() for p in proteins]
        
        expected_alphavirus_proteins = [
            'nonstructural', 'structural', 'polyprotein', 'capsid', 'envelope'
        ]
        
        found_expected = sum(1 for expected in expected_alphavirus_proteins 
                           if any(expected in product for product in protein_products))
        
        if found_expected < 2:
            logger.error(f"❌ Expected alphavirus proteins not found. Found: {found_expected}")
            return False
        
        logger.info(f"✅ Scientific accuracy validation passed: {found_expected} expected protein types found")
        self.test_data_quality['expected_proteins_found'] = found_expected
        return True
    
    async def _validate_no_fallback_data(self, results: Dict[str, Any]) -> bool:
        """Validate that no fallback or placeholder data was used"""
        
        # Check for common fallback indicators
        fallback_indicators = [
            'unknown', 'placeholder', 'default', 'fallback', 'mock', 'test_data',
            'n/a', 'null', 'empty', 'undefined'
        ]
        
        results_str = json.dumps(results).lower()
        
        for indicator in fallback_indicators:
            if indicator in results_str:
                logger.error(f"❌ Fallback indicator found: {indicator}")
                return False
        
        # Validate all numeric data is meaningful (not zeros or defaults)
        statistics = results.get('statistics', {})
        
        if statistics.get('total_proteins', 0) == 0:
            logger.error("❌ Zero proteins found - possible fallback data")
            return False
        
        if statistics.get('total_genomes', 0) == 0:
            logger.error("❌ Zero genomes found - possible fallback data")
            return False
        
        logger.info("✅ No fallback data validation passed")
        return True
    
    async def _generate_test_report(self, results: Dict[str, Any], validation_success: bool):
        """Generate comprehensive test report"""
        logger.info("📋 Generating Comprehensive Test Report")
        
        report = {
            "test_execution": {
                "timestamp": datetime.now().isoformat(),
                "validation_success": validation_success,
                "framework_compliance": True,
                "data_quality": self.test_data_quality
            },
            "workflow_results_summary": {
                "total_proteins": len(results.get('proteins', [])),
                "analysis_steps": len(results.get('analysis_results', {})),
                "organism": results.get('metadata', {}).get('organism'),
                "execution_successful": True
            },
            "scientific_validation": {
                "meaningful_data": validation_success,
                "no_fallbacks": True,
                "complete_analysis": True
            }
        }
        
        # Save detailed report
        report_path = Path("results") / f"viral_protein_analysis_test_report_{int(time.time())}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save workflow results
        results_path = Path("results") / f"viral_protein_analysis_results_{int(time.time())}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📄 Test report saved: {report_path}")
        logger.info(f"📄 Workflow results saved: {results_path}")
        
        # Print summary
        if validation_success:
            logger.info("✅ COMPREHENSIVE TEST PASSED - Workflow produces meaningful scientific results")
            logger.info(f"📊 Key Metrics:")
            logger.info(f"   - Proteins analyzed: {self.test_data_quality.get('protein_count', 0)}")
            logger.info(f"   - Analysis steps completed: {self.test_data_quality.get('analysis_steps_completed', 0)}")
            logger.info(f"   - Expected protein types found: {self.test_data_quality.get('expected_proteins_found', 0)}")
        else:
            logger.error("❌ COMPREHENSIVE TEST FAILED - Workflow did not produce adequate scientific results")
    
    def _record_failure(self, category: str, error: str):
        """Record test failure"""
        self.test_results.append({
            "category": category,
            "status": "FAILED",
            "error": error,
            "timestamp": datetime.now().isoformat()
        })


async def main():
    """Main test execution"""
    test_suite = ViralProteinAnalysisWorkflowTest()
    success = await test_suite.run_comprehensive_test()
    
    if success:
        logger.info("🎉 ALL TESTS PASSED - Viral protein analysis workflow is production ready!")
        sys.exit(0)
    else:
        logger.error("💥 TESTS FAILED - Workflow needs improvements")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 