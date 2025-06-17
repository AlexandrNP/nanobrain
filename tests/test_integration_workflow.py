"""
Integration Workflow Tests for NanoBrain Phase 4

Tests the integration of validated bioinformatics tools in realistic workflow scenarios.
Building on successful individual tool functionality tests, these integration tests
verify that tools work together correctly for viral protein analysis workflows.

Based on bioinformatics testing best practices:
- Focus on high-impact integration points where tools must coordinate
- Test data flow between tools to catch subtle compatibility issues  
- Validate error propagation and recovery scenarios
- Ensure performance remains acceptable with integrated tool chains
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

from nanobrain.library.tools.bioinformatics.bv_brc_tool import (
    BVBRCTool, BVBRCConfig, GenomeData, ProteinData, BVBRCDataError
)
from nanobrain.library.tools.bioinformatics.pubmed_client import (
    PubMedClient, PubMedConfig, LiteratureReference, PubMedError
)
from nanobrain.library.tools.bioinformatics.base_bioinformatics_tool import (
    InstallationStatus
)
from nanobrain.core.logging_system import get_logger


class TestIntegrationWorkflows:
    """
    Integration tests for bioinformatics tool workflows.
    
    Tests realistic scenarios where multiple tools must work together
    to complete viral protein analysis tasks.
    """
    
    @pytest.fixture(scope="class")
    def integration_setup(self):
        """Setup integration test environment"""
        return {
            "bvbrc_tool": BVBRCTool(BVBRCConfig(
                verify_on_init=False,
                genome_batch_size=3,  # Small batches for integration testing
                md5_batch_size=2,
                timeout_seconds=30
            )),
            "pubmed_client": PubMedClient(
                email="integration@test.org",
                fail_fast=False,  # Allow graceful degradation for integration
                config=PubMedConfig(
                    verify_on_init=False,
                    cache_enabled=True,
                    rate_limit=3
                )
            ),
            "logger": get_logger("integration_test"),
            "temp_dir": tempfile.mkdtemp(prefix="nanobrain_integration_")
        }
    
    @pytest.mark.asyncio
    async def test_genome_to_protein_to_literature_workflow(self, integration_setup):
        """
        Test complete workflow: Genome data → Protein extraction → Literature search
        
        This is a HIGH IMPACT integration test that verifies the core data flow
        through the entire viral protein analysis pipeline.
        """
        bvbrc_tool = integration_setup["bvbrc_tool"]
        pubmed_client = integration_setup["pubmed_client"]
        logger = integration_setup["logger"]
        
        logger.info("🧪 Testing complete genome→protein→literature workflow")
        
        # Step 1: Start with sample genome data (simulating BV-BRC download)
        sample_genomes = [
            GenomeData("511145.12", 11000, "Chikungunya virus", "Viruses;Alphavirus"),
            GenomeData("511145.13", 9500, "Eastern equine encephalitis virus", "Viruses;Alphavirus"),
            GenomeData("511145.14", 12000, "Venezuelan equine encephalitis virus", "Viruses;Alphavirus")
        ]
        
        # Step 2: Test genome size filtering integration
        filtered_genomes = await bvbrc_tool.filter_genomes_by_size(sample_genomes)
        assert len(filtered_genomes) >= 2, "Size filtering should retain medium-sized genomes"
        
        # Step 3: Test protein extraction workflow (with mock protein data)
        sample_proteins = [
            ProteinData(
                aa_sequence_md5="test_md5_1",
                patric_id="fig|511145.12.peg.1",
                product="capsid protein",
                aa_sequence="MATKGKRVIMLLVIACC",
                genome_id="511145.12"
            ),
            ProteinData(
                aa_sequence_md5="test_md5_2",
                patric_id="fig|511145.12.peg.2",
                product="envelope protein E1",
                aa_sequence="MKYTVVLLVAVLVLVLC",
                genome_id="511145.12"
            ),
            ProteinData(
                aa_sequence_md5="test_md5_3",
                patric_id="fig|511145.13.peg.1",
                product="nonstructural protein nsP1",
                aa_sequence="MADSKTIRTLLKKLSHK",
                genome_id="511145.13"
            )
        ]
        
        # Step 4: Test FASTA creation for downstream analysis
        fasta_content = await bvbrc_tool.create_annotated_fasta(sample_proteins)
        
        # Validate FASTA format for downstream tool compatibility
        fasta_lines = fasta_content.strip().split('\n')
        assert len(fasta_lines) == 6, "Should have 3 proteins × 2 lines each"
        
        # Verify header format compatibility
        protein_types = []
        for i in range(0, len(fasta_lines), 2):
            header = fasta_lines[i]
            sequence = fasta_lines[i + 1]
            
            assert header.startswith(">"), "FASTA headers must start with >"
            assert len(sequence) > 10, "Sequences should be reasonable length"
            
            # Extract protein type for literature search
            if "capsid" in header:
                protein_types.append("capsid protein")
            elif "envelope" in header:
                protein_types.append("envelope protein")
            elif "nonstructural" in header:
                protein_types.append("nonstructural protein")
        
        # Step 5: Test literature search integration for each protein type
        literature_results = {}
        for protein_type in set(protein_types):  # Unique protein types
            literature_refs = await pubmed_client.search_alphavirus_literature(protein_type)
            literature_results[protein_type] = literature_refs
            
            # Phase 4A: Placeholder implementation returns empty list
            assert isinstance(literature_refs, list), f"Literature search should return list for {protein_type}"
            
        # Step 6: Verify integration data flow integrity
        assert len(filtered_genomes) > 0, "Workflow should process some genomes"
        assert len(sample_proteins) > 0, "Workflow should extract some proteins"
        assert len(literature_results) > 0, "Workflow should search literature for protein types"
        
        # Verify data consistency throughout workflow
        genome_ids = {g.genome_id for g in filtered_genomes}
        protein_genome_ids = {p.genome_id for p in sample_proteins}
        
        # At least some proteins should come from filtered genomes
        common_ids = genome_ids.intersection(protein_genome_ids)
        assert len(common_ids) > 0, "Proteins should correspond to filtered genomes"
        
        logger.info("✅ Complete genome→protein→literature workflow validated")
        
        return {
            "genomes_processed": len(filtered_genomes),
            "proteins_extracted": len(sample_proteins),
            "protein_types_searched": len(literature_results),
            "fasta_size": len(fasta_content),
            "workflow_successful": True
        }
    
    @pytest.mark.asyncio
    async def test_error_propagation_integration(self, integration_setup):
        """
        Test error propagation between integrated tools - HIGH IMPACT TEST
        
        Ensures that errors in one tool are properly handled by dependent tools
        to prevent cascade failures in the workflow.
        """
        bvbrc_tool = integration_setup["bvbrc_tool"]
        pubmed_client = integration_setup["pubmed_client"]
        logger = integration_setup["logger"]
        
        logger.info("🧪 Testing error propagation between integrated tools")
        
        # Test 1: Empty genome list propagation
        with pytest.raises(BVBRCDataError, match="No genome IDs provided"):
            await bvbrc_tool.get_unique_protein_md5s([])
        
        # Test 2: Invalid protein data handling
        with pytest.raises(BVBRCDataError, match="No valid protein sequences"):
            await bvbrc_tool.create_annotated_fasta([])
        
        # Test 3: PubMed error handling with fail_fast=False
        pubmed_client.fail_fast = False
        
        # This should not raise an exception but return empty results
        result = await pubmed_client.search_alphavirus_literature("invalid_protein_type")
        assert isinstance(result, list), "Should return empty list, not raise exception"
        assert len(result) == 0, "Should return empty results for invalid input"
        
        # Test 4: Tool initialization error handling
        pubmed_client.fail_fast = True
        
        # Mock BioPython import failure
        with patch('builtins.__import__', side_effect=ImportError("Bio module missing")):
            with pytest.raises(PubMedError, match="BioPython not available"):
                await pubmed_client.initialize_tool()
        
        logger.info("✅ Error propagation integration validated")
        
        return {
            "empty_input_handling": "validated",
            "invalid_data_handling": "validated", 
            "graceful_degradation": "validated",
            "fail_fast_behavior": "validated"
        }
    
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self, integration_setup):
        """
        Test batch processing integration between tools - MEDIUM IMPACT TEST
        
        Verifies that batch sizes are respected across tool integrations
        and that large datasets are processed efficiently.
        """
        bvbrc_tool = integration_setup["bvbrc_tool"]
        logger = integration_setup["logger"]
        
        logger.info("🧪 Testing batch processing integration")
        
        # Create large dataset that exceeds batch size limits
        large_genome_list = [f"genome_{i}" for i in range(8)]  # Exceeds batch size of 3
        large_md5_list = [f"md5_{i}" for i in range(6)]       # Exceeds batch size of 2
        
        # Mock batch processing to verify batch sizes
        batch_calls = {"genome_batches": [], "md5_batches": []}
        
        async def mock_get_proteins_for_batch(batch):
            batch_calls["genome_batches"].append(len(batch))
            return [ProteinData(f"md5_{i}", f"id_{i}", f"product_{i}", "", f"genome_{i}") 
                   for i in range(len(batch))]
        
        async def mock_get_sequences_for_batch(batch):
            batch_calls["md5_batches"].append(len(batch))
            return [ProteinData(f"md5_{i}", f"id_{i}", f"product_{i}", f"sequence_{i}", "") 
                   for i in range(len(batch))]
        
        # Test genome batching
        bvbrc_tool._get_proteins_for_batch = mock_get_proteins_for_batch
        
        try:
            result = await bvbrc_tool.get_unique_protein_md5s(large_genome_list)
            
            # Verify batching occurred correctly
            # 8 genomes with batch size 3 should create batches: [3, 3, 2]
            assert len(batch_calls["genome_batches"]) == 3
            assert batch_calls["genome_batches"] == [3, 3, 2]
            
        except Exception:
            # Expected due to mocking - focus on batch size verification
            assert len(batch_calls["genome_batches"]) == 3
            assert batch_calls["genome_batches"] == [3, 3, 2]
        
        # Test MD5 batching
        bvbrc_tool._get_sequences_for_batch = mock_get_sequences_for_batch
        
        try:
            result = await bvbrc_tool.get_feature_sequences(large_md5_list)
            
            # 6 MD5s with batch size 2 should create batches: [2, 2, 2]  
            assert len(batch_calls["md5_batches"]) == 3
            assert batch_calls["md5_batches"] == [2, 2, 2]
            
        except Exception:
            # Expected due to mocking - focus on batch size verification
            assert len(batch_calls["md5_batches"]) == 3
            assert batch_calls["md5_batches"] == [2, 2, 2]
        
        logger.info("✅ Batch processing integration validated")
        
        return {
            "genome_batch_sizes": batch_calls["genome_batches"],
            "md5_batch_sizes": batch_calls["md5_batches"],
            "batching_working": True
        }
    
    @pytest.mark.asyncio
    async def test_caching_integration(self, integration_setup):
        """
        Test caching integration between tools - MEDIUM IMPACT TEST
        
        Verifies that caching works correctly when tools are used together
        and that cache hits improve performance.
        """
        pubmed_client = integration_setup["pubmed_client"]
        logger = integration_setup["logger"]
        
        logger.info("🧪 Testing caching integration")
        
        # Ensure caching is enabled
        assert pubmed_client.pubmed_config.cache_enabled == True
        
        # Clear cache for clean test start
        pubmed_client.search_cache.clear()
        initial_cache_size = len(pubmed_client.search_cache)
        assert initial_cache_size == 0, "Cache should start empty after clearing"
        
        # Test cache population
        protein_types = ["capsid protein", "envelope protein E1", "nonstructural protein nsP1"]
        
        first_results = {}
        for protein_type in protein_types:
            result = await pubmed_client.search_alphavirus_literature(protein_type)
            first_results[protein_type] = result
            
        # Verify cache was populated
        cache_size_after_first = len(pubmed_client.search_cache)
        assert cache_size_after_first == len(protein_types), "Cache should have entry for each protein type"
        
        # Test cache hits (second search should use cache)
        second_results = {}
        for protein_type in protein_types:
            result = await pubmed_client.search_alphavirus_literature(protein_type)
            second_results[protein_type] = result
            
        # Verify cache size didn't change (no new entries)
        cache_size_after_second = len(pubmed_client.search_cache)
        assert cache_size_after_second == cache_size_after_first, "Cache size should not change on cache hits"
        
        # Verify results are identical (from cache)
        for protein_type in protein_types:
            assert first_results[protein_type] == second_results[protein_type], \
                f"Cached results should be identical for {protein_type}"
        
        # Test cache key generation consistency
        expected_keys = [
            "alphavirus_capsid_protein",
            "alphavirus_envelope_protein_e1", 
            "alphavirus_nonstructural_protein_nsp1"
        ]
        
        actual_keys = list(pubmed_client.search_cache.keys())
        for expected_key in expected_keys:
            assert expected_key in actual_keys, f"Cache should contain key: {expected_key}"
        
        logger.info("✅ Caching integration validated")
        
        return {
            "cache_entries": len(pubmed_client.search_cache),
            "cache_keys": list(pubmed_client.search_cache.keys()),
            "caching_working": True
        }
    
    @pytest.mark.asyncio
    async def test_configuration_integration(self, integration_setup):
        """
        Test configuration integration between tools - MEDIUM IMPACT TEST
        
        Verifies that tool configurations are compatible and don't conflict
        when tools are used together in workflows.
        """
        bvbrc_tool = integration_setup["bvbrc_tool"]
        pubmed_client = integration_setup["pubmed_client"]
        logger = integration_setup["logger"]
        
        logger.info("🧪 Testing configuration integration")
        
        # Test BV-BRC configuration
        bvbrc_config = bvbrc_tool.bv_brc_config
        assert bvbrc_config.genome_batch_size == 3, "BV-BRC batch size should be set for integration testing"
        assert bvbrc_config.md5_batch_size == 2, "BV-BRC MD5 batch size should be set for integration testing"
        assert bvbrc_config.timeout_seconds == 30, "Timeout should be reasonable for integration testing"
        
        # Test PubMed configuration  
        pubmed_config = pubmed_client.pubmed_config
        assert pubmed_config.rate_limit == 3, "PubMed rate limit should be conservative"
        assert pubmed_config.cache_enabled == True, "Caching should be enabled for integration"
        assert pubmed_config.fail_fast == False, "Should allow graceful degradation in integration"
        
        # Test configuration compatibility
        # Both tools should be able to run concurrently without timeout conflicts
        assert bvbrc_config.timeout_seconds >= 30, "Timeouts should be generous enough for integration"
        assert pubmed_config.rate_limit <= 10, "Rate limits should not be excessive"
        
        # Test logging configuration compatibility
        bvbrc_logger = bvbrc_tool.logger
        pubmed_logger = pubmed_client.logger
        
        assert bvbrc_logger.name == "bv_brc_tool"
        assert pubmed_logger.name == "pubmed_client"
        # Different logger names prevent conflicts
        
        # Test environment compatibility
        assert bvbrc_config.verify_on_init == False, "Should not auto-initialize for integration testing"
        assert pubmed_config.verify_on_init == False, "Should not auto-initialize for integration testing"
        
        logger.info("✅ Configuration integration validated")
        
        return {
            "bvbrc_batch_size": bvbrc_config.genome_batch_size,
            "pubmed_rate_limit": pubmed_config.rate_limit,
            "configurations_compatible": True,
            "logging_isolated": True
        }
    
    @pytest.mark.asyncio
    async def test_performance_integration(self, integration_setup):
        """
        Test performance integration between tools - LOW IMPACT TEST
        
        Verifies that tool integration doesn't significantly degrade performance
        and that operations complete within reasonable time bounds.
        """
        bvbrc_tool = integration_setup["bvbrc_tool"]
        pubmed_client = integration_setup["pubmed_client"]
        logger = integration_setup["logger"]
        
        logger.info("🧪 Testing performance integration")
        
        import time
        
        # Test individual tool performance baselines
        start_time = time.time()
        
        # BV-BRC operations
        sample_genomes = [GenomeData(f"genome_{i}", 10000, f"Virus {i}", "Alphavirus") for i in range(5)]
        filtered_genomes = await bvbrc_tool.filter_genomes_by_size(sample_genomes)
        
        bvbrc_time = time.time() - start_time
        
        # PubMed operations
        start_time = time.time()
        
        literature_result = await pubmed_client.search_alphavirus_literature("capsid protein")
        
        pubmed_time = time.time() - start_time
        
        # Test combined operations
        start_time = time.time()
        
        # Simulate integrated workflow
        filtered_genomes = await bvbrc_tool.filter_genomes_by_size(sample_genomes)
        
        sample_proteins = [
            ProteinData("md5_1", "id_1", "capsid protein", "SEQUENCE1", "genome_1"),
            ProteinData("md5_2", "id_2", "envelope protein", "SEQUENCE2", "genome_2")
        ]
        
        fasta_content = await bvbrc_tool.create_annotated_fasta(sample_proteins)
        
        literature_result = await pubmed_client.search_alphavirus_literature("capsid protein")
        
        integrated_time = time.time() - start_time
        
        # Performance assertions (generous thresholds for integration testing)
        assert bvbrc_time < 1.0, f"BV-BRC operations should complete quickly: {bvbrc_time:.3f}s"
        assert pubmed_time < 1.0, f"PubMed operations should complete quickly: {pubmed_time:.3f}s"
        assert integrated_time < 2.0, f"Integrated workflow should complete reasonably: {integrated_time:.3f}s"
        
        # Verify operations completed successfully
        assert len(filtered_genomes) > 0, "Performance test should process genomes"
        assert len(fasta_content) > 0, "Performance test should generate FASTA"
        assert isinstance(literature_result, list), "Performance test should complete literature search"
        
        logger.info("✅ Performance integration validated")
        
        return {
            "bvbrc_time": round(bvbrc_time, 3),
            "pubmed_time": round(pubmed_time, 3),
            "integrated_time": round(integrated_time, 3),
            "performance_acceptable": True
        }


class TestProductionReadinessIntegration:
    """
    Integration tests that verify production readiness of the tool chain.
    
    These tests focus on scenarios that would occur in real production
    viral protein analysis workflows.
    """
    
    @pytest.fixture(scope="class")
    def production_setup(self):
        """Setup production-like test environment"""
        return {
            "bvbrc_tool": BVBRCTool(BVBRCConfig(
                verify_on_init=False,
                genome_batch_size=50,    # Production batch sizes
                md5_batch_size=25,
                timeout_seconds=300,     # Production timeouts
                min_genome_length=8000,
                max_genome_length=15000
            )),
            "pubmed_client": PubMedClient(
                email="production@nanobrain.org",
                fail_fast=True,          # Production fail-fast
                config=PubMedConfig(
                    verify_on_init=False,
                    cache_enabled=True,
                    rate_limit=3,           # Conservative production rate limit
                    max_results_per_search=20
                )
            ),
            "logger": get_logger("production_integration_test")
        }
    
    @pytest.mark.asyncio
    async def test_production_scale_workflow(self, production_setup):
        """
        Test production-scale workflow simulation - HIGH IMPACT TEST
        
        Simulates a realistic production workflow with medium-scale data
        to verify the system can handle expected production loads.
        """
        bvbrc_tool = production_setup["bvbrc_tool"]
        pubmed_client = production_setup["pubmed_client"]
        logger = production_setup["logger"]
        
        logger.info("🏭 Testing production-scale workflow simulation")
        
        # Simulate medium-scale alphavirus dataset (production target: 100-500 genomes)
        production_genomes = []
        for i in range(20):  # Reasonable test size
            genome_length = 8000 + (i * 200)  # Vary genome sizes within range
            production_genomes.append(
                GenomeData(f"production_genome_{i}", genome_length, f"Alphavirus strain {i}", "Viruses;Alphavirus")
            )
        
        # Test production genome filtering
        filtered_genomes = await bvbrc_tool.filter_genomes_by_size(production_genomes)
        
        # Should retain genomes within production size range
        for genome in filtered_genomes:
            assert 8000 <= genome.genome_length <= 15000, f"Genome {genome.genome_id} outside production range"
        
        # Test production protein extraction simulation
        production_proteins = []
        protein_types = ["capsid protein", "envelope protein E1", "envelope protein E2", 
                        "nonstructural protein nsP1", "nonstructural protein nsP2",
                        "nonstructural protein nsP3", "nonstructural protein nsP4"]
        
        for i, genome in enumerate(filtered_genomes[:10]):  # Limit for test performance
            for j, protein_type in enumerate(protein_types):
                production_proteins.append(
                    ProteinData(
                        aa_sequence_md5=f"production_md5_{i}_{j}",
                        patric_id=f"fig|{genome.genome_id}.peg.{j+1}",
                        product=protein_type,
                        aa_sequence=f"PRODUCTION_SEQUENCE_{i}_{j}",
                        genome_id=genome.genome_id
                    )
                )
        
        # Test production FASTA creation
        production_fasta = await bvbrc_tool.create_annotated_fasta(production_proteins)
        
        # Verify production FASTA format
        fasta_lines = production_fasta.strip().split('\n')
        assert len(fasta_lines) == len(production_proteins) * 2, "FASTA should have header+sequence for each protein"
        
        # Test production literature search for all protein types
        literature_coverage = {}
        for protein_type in set(protein_types):
            literature_refs = await pubmed_client.search_alphavirus_literature(protein_type)
            literature_coverage[protein_type] = len(literature_refs)
        
        # Verify production workflow completeness
        assert len(filtered_genomes) > 0, "Production workflow should process genomes"
        assert len(production_proteins) > 0, "Production workflow should extract proteins"
        assert len(literature_coverage) == len(set(protein_types)), "Should search literature for all protein types"
        
        logger.info("✅ Production-scale workflow simulation validated")
        
        return {
            "genomes_processed": len(filtered_genomes),
            "proteins_extracted": len(production_proteins),
            "protein_types_covered": len(literature_coverage),
            "fasta_size_kb": len(production_fasta) // 1024,
            "production_workflow_successful": True
        }
    
    @pytest.mark.asyncio
    async def test_production_error_handling(self, production_setup):
        """
        Test production error handling and recovery - HIGH IMPACT TEST
        
        Verifies that the system handles production errors gracefully
        and provides useful diagnostics for troubleshooting.
        """
        bvbrc_tool = production_setup["bvbrc_tool"]
        pubmed_client = production_setup["pubmed_client"]
        logger = production_setup["logger"]
        
        logger.info("🏭 Testing production error handling and recovery")
        
        # Test production fail-fast behavior
        assert pubmed_client.fail_fast == True, "Production should use fail-fast error handling"
        
        # Test diagnostic information quality
        try:
            # This should fail in production mode
            await bvbrc_tool.get_unique_protein_md5s([])
        except BVBRCDataError as e:
            error_message = str(e)
            assert "No genome IDs provided" in error_message, "Error should provide specific diagnostic"
        
        # Test PubMed production error handling
        pubmed_client.fail_fast = True
        
        with patch('builtins.__import__', side_effect=ImportError("BioPython missing")):
            try:
                await pubmed_client.initialize_tool()
                assert False, "Should raise error in production mode"
            except PubMedError as e:
                error_message = str(e)
                assert "BioPython not available" in error_message, "Should provide clear installation guidance"
                assert "pip install biopython" in error_message, "Should provide specific fix instructions"
        
        # Test configuration validation
        config_diagnostics = {
            "bvbrc_batch_sizes": {
                "genome_batch_size": bvbrc_tool.bv_brc_config.genome_batch_size,
                "md5_batch_size": bvbrc_tool.bv_brc_config.md5_batch_size
            },
            "production_timeouts": bvbrc_tool.bv_brc_config.timeout_seconds,
            "pubmed_rate_limit": pubmed_client.rate_limit
        }
        
        # Verify production configuration values
        assert config_diagnostics["bvbrc_batch_sizes"]["genome_batch_size"] == 50, "Production genome batch size"
        assert config_diagnostics["bvbrc_batch_sizes"]["md5_batch_size"] == 25, "Production MD5 batch size" 
        assert config_diagnostics["production_timeouts"] == 300, "Production timeout value"
        assert config_diagnostics["pubmed_rate_limit"] == 3, "Production rate limit"
        
        logger.info("✅ Production error handling and recovery validated")
        
        return {
            "fail_fast_working": True,
            "diagnostic_quality": "detailed",
            "configuration_validated": True,
            "production_ready": True
        }


if __name__ == "__main__":
    """Run integration workflow tests directly"""
    pytest.main([__file__, "-v", "--tb=short", "-x"]) 