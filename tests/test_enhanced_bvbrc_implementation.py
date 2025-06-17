"""
Test Suite for Enhanced BV-BRC Implementation

Tests for the complete implementation of exact BV-BRC command sequence:
- Virus name resolution with fuzzy matching
- BV-BRC command pipeline execution
- Intelligent caching system
- Enhanced BVBRCTool integration

Author: NanoBrain Development Team
Date: December 2024
Version: 4.1.1
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCTool, BVBRCConfig
from nanobrain.library.workflows.viral_protein_analysis.virus_name_resolver import (
    VirusNameResolver, TaxonResolution, TaxonInfo
)
from nanobrain.library.workflows.viral_protein_analysis.bvbrc_command_pipeline import (
    BVBRCCommandPipeline, PipelineResult, PipelineFiles, CommandResult
)
from nanobrain.library.workflows.viral_protein_analysis.bvbrc_cache_manager import (
    BVBRCCacheManager, CacheEntry, CacheStats
)


class TestVirusNameResolver:
    """Test virus name resolution system"""
    
    @pytest.fixture
    async def resolver(self):
        """Create virus name resolver with test data"""
        # Create test CSV data
        test_csv_content = """genome_id,genome_name,taxon_lineage_names
11020.100,Barham forest virus,Viruses;Alphavirus
11021.200,Chikungunya virus strain 37997,Viruses;Alphavirus
11022.300,Eastern equine encephalitis virus,Viruses;Alphavirus
11023.400,Venezuelan equine encephalitis virus,Viruses;Alphavirus
11024.500,Western equine encephalitis virus,Viruses;Alphavirus"""
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(test_csv_content)
            test_csv_path = f.name
        
        resolver = VirusNameResolver(csv_path=test_csv_path)
        await resolver.initialize_virus_index()
        
        yield resolver
        
        # Cleanup
        Path(test_csv_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_exact_name_resolution(self, resolver):
        """Test exact virus name resolution"""
        resolution = await resolver.resolve_virus_name("Chikungunya virus strain 37997")
        
        assert resolution is not None
        assert resolution.taxon_id == "11021"
        assert resolution.confidence == 100
        assert resolution.match_type == "exact"
        assert "Chikungunya" in resolution.matched_name
    
    @pytest.mark.asyncio
    async def test_fuzzy_name_resolution(self, resolver):
        """Test fuzzy virus name resolution"""
        # Test common abbreviations
        resolution = await resolver.resolve_virus_name("CHIKV")
        assert resolution is not None
        assert resolution.taxon_id == "11021"
        assert resolution.confidence >= 80
        
        # Test partial names
        resolution = await resolver.resolve_virus_name("Eastern equine")
        assert resolution is not None
        assert resolution.taxon_id == "11022"
        assert resolution.confidence >= 80
    
    @pytest.mark.asyncio
    async def test_no_match_suggestions(self, resolver):
        """Test suggestion system for failed matches"""
        resolution = await resolver.resolve_virus_name("Unknown virus")
        assert resolution is None
        
        suggestions = await resolver.suggest_similar_names("Eastern", max_suggestions=3)
        assert len(suggestions) > 0
        assert any("Eastern equine" in name for name, _ in suggestions)
    
    @pytest.mark.asyncio
    async def test_available_taxa_listing(self, resolver):
        """Test listing available taxa"""
        taxa = await resolver.get_available_taxa()
        assert len(taxa) == 5  # 5 unique taxon IDs in test data
        
        # Check taxon info structure
        for taxon in taxa:
            assert isinstance(taxon, TaxonInfo)
            assert taxon.taxon_id
            assert taxon.primary_name
            assert taxon.genome_count > 0
    
    @pytest.mark.asyncio
    async def test_name_variations_generation(self, resolver):
        """Test virus name variation generation"""
        variations = resolver._generate_name_variations("Eastern equine encephalitis virus")
        
        expected_variations = [
            "Eastern equine encephalitis virus",
            "Eastern equine encephalitis",
            "EEEV",  # Acronym
            "Eastern eq encephalitis virus"  # Abbreviation
        ]
        
        for expected in expected_variations:
            assert any(expected.lower() in var.lower() for var in variations)


class TestBVBRCCommandPipeline:
    """Test BV-BRC command pipeline execution"""
    
    @pytest.fixture
    def mock_cli_path(self):
        """Create mock CLI path structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cli_dir = Path(temp_dir) / "bin"
            cli_dir.mkdir()
            
            # Create mock CLI tools
            for tool in ["p3-all-genomes", "p3-get-genome-features", "p3-get-feature-sequence"]:
                tool_path = cli_dir / tool
                tool_path.write_text("#!/bin/bash\necho 'mock output'")
                tool_path.chmod(0o755)
            
            yield str(cli_dir)
    
    @pytest.fixture
    def pipeline(self, mock_cli_path):
        """Create command pipeline with mocked CLI tools"""
        return BVBRCCommandPipeline(
            bvbrc_cli_path=mock_cli_path,
            timeout_seconds=30,
            preserve_files=True
        )
    
    @pytest.mark.asyncio
    async def test_cli_tools_verification(self, mock_cli_path):
        """Test CLI tools verification"""
        # Should pass with mock tools
        pipeline = BVBRCCommandPipeline(bvbrc_cli_path=mock_cli_path)
        assert pipeline.cli_path.exists()
        
        # Should fail with missing tools
        with pytest.raises(FileNotFoundError):
            BVBRCCommandPipeline(bvbrc_cli_path="/nonexistent/path")
    
    @pytest.mark.asyncio
    async def test_pipeline_execution_structure(self, pipeline):
        """Test pipeline execution structure (mocked)"""
        with patch.object(pipeline, '_execute_shell_command_with_output') as mock_execute:
            # Mock successful command execution
            mock_execute.return_value = CommandResult(
                success=True,
                stdout="mock\tdata\ngenome123\t12000\tTest virus\tViruses",
                stderr="",
                return_code=0,
                execution_time=0.1,
                command="mock_command"
            )
            
            # Mock file counting methods
            with patch.object(pipeline, '_count_data_lines', return_value=10):
                with patch.object(pipeline, '_count_fasta_sequences', return_value=5):
                    
                    result = await pipeline.execute_pipeline("11020")
                    
                    assert result.success
                    assert result.taxon_id == "11020"
                    assert result.genome_count == 10
                    assert result.sequence_count == 5
                    assert result.files is not None
                    assert len(result.commands_executed) == 4  # 4 steps
    
    @pytest.mark.asyncio
    async def test_pipeline_failure_handling(self, pipeline):
        """Test pipeline failure handling"""
        with patch.object(pipeline, '_execute_shell_command_with_output') as mock_execute:
            # Mock failed command execution
            mock_execute.return_value = CommandResult(
                success=False,
                stdout="",
                stderr="Command failed",
                return_code=1,
                execution_time=0.1,
                command="mock_command"
            )
            
            result = await pipeline.execute_pipeline("11020")
            
            assert not result.success
            assert "Step 1 failed" in result.error_message
            assert result.commands_executed is not None
            assert len(result.commands_executed) == 1  # Failed on first step
    
    @pytest.mark.asyncio
    async def test_command_timeout_handling(self, pipeline):
        """Test command timeout handling"""
        with patch.object(pipeline, '_execute_shell_command_with_output') as mock_execute:
            # Mock timeout
            async def timeout_side_effect(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate slow command
                return CommandResult(
                    success=False,
                    stdout="",
                    stderr="Command timed out after 30 seconds",
                    return_code=-1,
                    execution_time=30.0,
                    command="mock_command"
                )
            
            mock_execute.side_effect = timeout_side_effect
            
            result = await pipeline.execute_pipeline("11020")
            
            assert not result.success
            assert "timed out" in result.error_message.lower()


class TestBVBRCCacheManager:
    """Test BV-BRC cache management system"""
    
    @pytest.fixture
    async def cache_manager(self):
        """Create cache manager with temporary directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "test_cache"
            manager = BVBRCCacheManager(
                cache_dir=str(cache_dir),
                max_cache_size_mb=10,  # Small limit for testing
                default_expiry_hours=1
            )
            yield manager
    
    @pytest.fixture
    def mock_pipeline_result(self):
        """Create mock pipeline result for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir)
            
            # Create mock result files
            genomes_file = working_dir / "11020.tsv"
            features_file = working_dir / "11020.id_md5"
            unique_file = working_dir / "11020.uniqe.md5"
            sequences_file = working_dir / "11020.unique.seq"
            
            genomes_file.write_text("genome_id\tgenome_name\n11020.100\tTest virus")
            features_file.write_text("patric_id\taa_sequence_md5\nfig|123\tabc123")
            unique_file.write_text("abc123\ndef456")
            sequences_file.write_text(">fig|123|test protein\nMETQSTART")
            
            files = PipelineFiles(
                working_dir=working_dir,
                genomes_tsv=genomes_file,
                features_id_md5=features_file,
                unique_md5=unique_file,
                sequences_fasta=sequences_file
            )
            
            result = PipelineResult(
                success=True,
                taxon_id="11020",
                execution_time=10.5,
                files=files,
                commands_executed=[],
                genome_count=1,
                feature_count=1,
                unique_md5_count=2,
                sequence_count=1
            )
            
            yield result
    
    @pytest.mark.asyncio
    async def test_cache_storage_and_retrieval(self, cache_manager, mock_pipeline_result):
        """Test caching pipeline results"""
        # Store result
        cache_key = await cache_manager.store_result(mock_pipeline_result)
        assert cache_key
        assert "11020" in cache_key
        
        # Retrieve result
        cached_result = await cache_manager.get_cached_result("11020")
        assert cached_result is not None
        assert cached_result.success
        assert cached_result.taxon_id == "11020"
        assert cached_result.genome_count == 1
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache_manager, mock_pipeline_result):
        """Test cache expiration functionality"""
        # Store with very short expiry
        cache_key = await cache_manager.store_result(
            mock_pipeline_result, expiry_hours=0.001  # ~3.6 seconds
        )
        
        # Should be available immediately
        cached_result = await cache_manager.get_cached_result("11020")
        assert cached_result is not None
        
        # Wait for expiration and clean up
        await asyncio.sleep(0.1)  # Short wait
        expired_count = await cache_manager.clear_expired_entries()
        
        # Should be gone after cleanup
        cached_result = await cache_manager.get_cached_result("11020")
        assert cached_result is None
    
    @pytest.mark.asyncio
    async def test_cache_statistics(self, cache_manager, mock_pipeline_result):
        """Test cache statistics reporting"""
        # Initial stats
        stats = await cache_manager.get_cache_stats()
        assert stats.total_entries == 0
        assert stats.total_size_mb == 0
        
        # Add cache entry
        await cache_manager.store_result(mock_pipeline_result)
        
        # Updated stats
        stats = await cache_manager.get_cache_stats()
        assert stats.total_entries == 1
        assert stats.total_size_mb > 0
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, cache_manager):
        """Test cache key generation"""
        key1 = cache_manager._generate_cache_key("11020")
        key2 = cache_manager._generate_cache_key("11020")
        key3 = cache_manager._generate_cache_key("11021")
        
        # Same taxon should generate same key
        assert key1 == key2
        
        # Different taxon should generate different key
        assert key1 != key3
        
        # Should include taxon ID
        assert "11020" in key1
    
    @pytest.mark.asyncio
    async def test_cache_with_additional_params(self, cache_manager, mock_pipeline_result):
        """Test cache with additional parameters"""
        # Store with additional params
        cache_key = await cache_manager.store_result(
            mock_pipeline_result, {"virus_name": "CHIKV"}
        )
        
        # Retrieve with same params
        cached_result = await cache_manager.get_cached_result(
            "11020", {"virus_name": "CHIKV"}
        )
        assert cached_result is not None
        
        # Different params should not match
        cached_result = await cache_manager.get_cached_result(
            "11020", {"virus_name": "EEEV"}
        )
        assert cached_result is None


class TestEnhancedBVBRCTool:
    """Test enhanced BV-BRC tool integration"""
    
    @pytest.fixture
    async def bvbrc_tool(self):
        """Create enhanced BV-BRC tool with mocked components"""
        config = BVBRCConfig(verify_on_init=False)
        tool = BVBRCTool(config)
        
        # Mock initialization to avoid real BV-BRC requirement
        with patch.object(tool, 'detect_existing_installation') as mock_detect:
            mock_detect.return_value = MagicMock(
                found=True,
                executable_path="/mock/path",
                is_functional=True
            )
            
            with patch.object(tool, '_setup_cli_tools'):
                with patch.object(tool.virus_resolver, 'initialize_virus_index'):
                    await tool.initialize_tool()
        
        yield tool
    
    @pytest.mark.asyncio
    async def test_virus_processing_workflow(self, bvbrc_tool):
        """Test complete virus processing workflow"""
        with patch.object(bvbrc_tool.virus_resolver, 'resolve_virus_name') as mock_resolve:
            mock_resolve.return_value = TaxonResolution(
                taxon_id="11020",
                matched_name="Test virus",
                confidence=95,
                match_type="fuzzy",
                synonyms=["Test virus"]
            )
            
            with patch.object(bvbrc_tool.cache_manager, 'get_cached_result') as mock_cache_get:
                mock_cache_get.return_value = None  # No cache hit
                
                with patch.object(bvbrc_tool.command_pipeline, 'execute_pipeline') as mock_pipeline:
                    mock_pipeline.return_value = PipelineResult(
                        success=True,
                        taxon_id="11020",
                        execution_time=5.0,
                        files=None,
                        sequence_count=10
                    )
                    
                    with patch.object(bvbrc_tool.cache_manager, 'store_result') as mock_cache_store:
                        result = await bvbrc_tool.get_proteins_for_virus("CHIKV")
                        
                        assert result.success
                        assert result.taxon_id == "11020"
                        assert result.sequence_count == 10
                        
                        # Verify workflow steps
                        mock_resolve.assert_called_once_with("CHIKV", 80)
                        mock_cache_get.assert_called_once()
                        mock_pipeline.assert_called_once_with("11020")
                        mock_cache_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_virus_name_resolution_failure(self, bvbrc_tool):
        """Test handling of virus name resolution failures"""
        with patch.object(bvbrc_tool.virus_resolver, 'resolve_virus_name') as mock_resolve:
            mock_resolve.return_value = None  # No resolution
            
            with patch.object(bvbrc_tool.virus_resolver, 'suggest_similar_names') as mock_suggest:
                mock_suggest.return_value = [("Similar virus", 75)]
                
                with pytest.raises(Exception) as exc_info:
                    await bvbrc_tool.get_proteins_for_virus("Unknown virus")
                
                assert "Could not resolve virus name" in str(exc_info.value)
                assert "Similar virus" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_cache_hit_handling(self, bvbrc_tool):
        """Test cache hit handling"""
        mock_cached_result = PipelineResult(
            success=True,
            taxon_id="11020",
            execution_time=0.0,
            files=None,
            sequence_count=5
        )
        
        with patch.object(bvbrc_tool.virus_resolver, 'resolve_virus_name') as mock_resolve:
            mock_resolve.return_value = TaxonResolution(
                taxon_id="11020",
                matched_name="Test virus",
                confidence=100,
                match_type="exact"
            )
            
            with patch.object(bvbrc_tool.cache_manager, 'get_cached_result') as mock_cache_get:
                mock_cache_get.return_value = mock_cached_result
                
                result = await bvbrc_tool.get_proteins_for_virus("Test virus")
                
                assert result == mock_cached_result
                assert bvbrc_tool.cache_hit_count == 1
                
                # Pipeline should not be executed for cache hits
                assert not hasattr(bvbrc_tool.command_pipeline, 'execute_pipeline')
    
    @pytest.mark.asyncio
    async def test_available_viruses_listing(self, bvbrc_tool):
        """Test listing available viruses"""
        mock_taxa = [
            TaxonInfo("11020", "Virus A", ["Virus A"], 5, ["11020.1"]),
            TaxonInfo("11021", "Virus B", ["Virus B"], 3, ["11021.1"]),
        ]
        
        with patch.object(bvbrc_tool.virus_resolver, 'get_available_taxa') as mock_get_taxa:
            mock_get_taxa.return_value = mock_taxa
            
            available = await bvbrc_tool.list_available_viruses(limit=1)
            
            assert len(available) == 1
            assert available[0].taxon_id == "11020"
    
    @pytest.mark.asyncio
    async def test_cache_management_operations(self, bvbrc_tool):
        """Test cache management operations"""
        # Test cache statistics
        mock_stats = CacheStats(
            total_entries=5,
            total_size_mb=10.5,
            cache_hits=3,
            cache_misses=2,
            expired_entries=1,
            hit_rate=60.0
        )
        
        with patch.object(bvbrc_tool.cache_manager, 'get_cache_stats') as mock_stats_get:
            mock_stats_get.return_value = mock_stats
            
            stats = await bvbrc_tool.get_cache_statistics()
            assert stats.total_entries == 5
            assert stats.hit_rate == 60.0
        
        # Test cache clearing
        with patch.object(bvbrc_tool.cache_manager, 'clear_expired_entries') as mock_clear:
            mock_clear.return_value = 2
            
            cleared = await bvbrc_tool.clear_cache(expired_only=True)
            assert cleared == 2


@pytest.mark.integration
class TestIntegrationWithoutBVBRC:
    """Integration tests that don't require actual BV-BRC installation"""
    
    @pytest.mark.asyncio
    async def test_component_integration(self):
        """Test that all components integrate correctly"""
        # Create real instances (but don't initialize with real data)
        resolver = VirusNameResolver("nonexistent.csv")
        
        # Test that components can be created without errors
        cache_manager = BVBRCCacheManager()
        
        # Test configuration
        config = BVBRCConfig(verify_on_init=False)
        assert config.tool_name == "bv_brc"
        assert config.installation_path == "/Applications/BV-BRC.app"
        
        # Test tool creation
        tool = BVBRCTool(config)
        assert tool.bv_brc_config == config
        assert tool.virus_resolver is not None
        assert tool.cache_manager is not None
    
    @pytest.mark.asyncio
    async def test_error_handling_without_data(self):
        """Test error handling when data files are missing"""
        resolver = VirusNameResolver("nonexistent.csv")
        
        with pytest.raises(FileNotFoundError):
            await resolver.initialize_virus_index()
        
        # Should handle gracefully when not initialized
        resolution = await resolver.resolve_virus_name("test")
        assert resolution is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 