# PHASE 4 INTEGRATION TESTING PLAN: REAL API CALLS & PRODUCTION DEPLOYMENT

**Date**: December 2024  
**Status**: 📋 **PLANNING**  
**Version**: 4.1.0  
**Target**: Remove all mocks, enable real API calls, create production-ready alphavirus workflow  

## Executive Summary

This plan details the implementation steps required to:

1. **Enable Real API Integration**: Remove all mocks from Phase 4 testing and replace with actual BV-BRC and PubMed API calls
2. **Production Alphavirus Workflow**: Create a complete production-ready alphavirus workflow based on the detailed implementation plan
3. **End-to-End Integration Testing**: Ensure seamless integration between all phases with real data flows
4. **Production Deployment Ready**: Deliver a fully deployable system for production research environments

## 🎯 Objectives Overview

### **Primary Goals**
- ✅ Replace all mock implementations with real API calls
- ✅ Implement complete 14-step alphavirus workflow for production
- ✅ Ensure Phase 1-4 integration with real data flows
- ✅ Create production deployment infrastructure
- ✅ Validate performance with real data volumes

### **Success Criteria**
- All tests pass with real API calls (no mocks)
- Alphavirus workflow produces valid Viral_PSSM.json output
- Production deployment successfully processes real viral protein data
- System handles network failures and API rate limits gracefully
- Documentation complete for production deployment

---

## 📋 DETAILED IMPLEMENTATION PLAN

### **PHASE 4.1: REAL API INTEGRATION ENABLEMENT**

#### **Step 4.1.1: BV-BRC Real API Integration**
**Objective**: Enable actual BV-BRC CLI tool integration with real data downloads
**Duration**: 2-3 days
**Priority**: Critical

**Current State Analysis**:
```python
# Current Issue: Tests use mocked BV-BRC responses
mock_genomes_data = b"""genome_id\tgenome_length\tgenome_name\ttaxon_lineage_names
511145.12\t11000\tTest Alphavirus 1\tViruses;Alphavirus"""

# Need: Real BV-BRC API calls via CLI tools
```

**Implementation Steps**:

1. **BV-BRC Installation Verification**
   ```python
   # File: nanobrain/library/tools/bioinformatics/bv_brc_real_api_client.py
   class BVBRCRealAPIClient(BVBRCTool):
       """Real BV-BRC API client with production settings"""
       
       def __init__(self, production_mode: bool = True):
           super().__init__(BVBRCConfig(
               verify_on_init=production_mode,
               use_cache=True,
               enable_retry=True,
               max_retries=3,
               timeout_seconds=300
           ))
           
       async def verify_real_installation(self) -> Dict[str, Any]:
           """Verify actual BV-BRC installation with diagnostics"""
           verification_result = {
               "bv_brc_app_exists": False,
               "cli_tools_accessible": False,
               "test_query_successful": False,
               "installation_path": self.config.installation_path,
               "executable_path": self.config.executable_path,
               "diagnostics": []
           }
           
           # Check application bundle
           app_path = Path(self.config.installation_path)
           verification_result["bv_brc_app_exists"] = app_path.exists()
           
           if not app_path.exists():
               verification_result["diagnostics"].append(
                   f"BV-BRC app not found at {app_path}. Please install from https://www.bv-brc.org/"
               )
               return verification_result
           
           # Check CLI tools accessibility
           cli_path = Path(self.config.executable_path)
           p3_all_genomes = cli_path / "p3-all-genomes"
           verification_result["cli_tools_accessible"] = p3_all_genomes.exists()
           
           if not p3_all_genomes.exists():
               verification_result["diagnostics"].append(
                   f"CLI tools not found at {cli_path}. Check BV-BRC installation."
               )
               return verification_result
           
           # Test actual API call
           try:
               result = await self.execute_p3_command("p3-all-genomes", [
                   "--eq", "genome_id,511145.12",
                   "--attr", "genome_id,genome_name",
                   "--limit", "1"
               ])
               
               if result.success and result.stdout:
                   lines = result.stdout_text.strip().split('\n')
                   if len(lines) > 1:
                       verification_result["test_query_successful"] = True
                       verification_result["diagnostics"].append("✅ Real API call successful")
                   else:
                       verification_result["diagnostics"].append("⚠️ API call returned headers only")
               else:
                   verification_result["diagnostics"].append(
                       f"❌ API call failed: {result.stderr_text}"
                   )
                   
           except Exception as e:
               verification_result["diagnostics"].append(f"❌ API call exception: {e}")
           
           return verification_result
   ```

2. **Real Data Download Implementation**
   ```python
   async def download_real_alphavirus_genomes(self) -> List[GenomeData]:
       """Download actual Alphavirus genomes from BV-BRC"""
       
       self.logger.info("🔄 Starting real Alphavirus genome download from BV-BRC")
       
       # Real API call - no mocks
       command_args = [
           "--eq", "taxon_lineage_names,Alphavirus",
           "--attr", "genome_id,genome_length,genome_name,taxon_lineage_names,genome_status",
           "--limit", "1000"  # Reasonable limit for testing
       ]
       
       try:
           result = await self.execute_p3_command("p3-all-genomes", command_args)
           
           if not result.success:
               raise BVBRCDataError(f"Real API call failed: {result.stderr_text}")
           
           # Parse real data with validation
           genomes = await self._parse_real_genome_data(result.stdout)
           
           # Real data validation
           if len(genomes) < 5:  # Expect at least 5 Alphavirus genomes
               self.logger.warning(f"Only {len(genomes)} Alphavirus genomes found - unusually low")
           
           self.logger.info(f"✅ Successfully downloaded {len(genomes)} real Alphavirus genomes")
           return genomes
           
       except Exception as e:
           self.logger.error(f"❌ Real Alphavirus genome download failed: {e}")
           raise
   ```

3. **Error Handling for Real API Calls**
   ```python
   async def handle_real_api_errors(self, operation: str, error: Exception) -> Dict[str, Any]:
       """Handle real API errors with user guidance"""
       
       error_guidance = {
           "operation": operation,
           "error_type": type(error).__name__,
           "error_message": str(error),
           "suggested_actions": [],
           "retry_recommended": False
       }
       
       if "Network" in str(error) or "timeout" in str(error).lower():
           error_guidance["suggested_actions"].extend([
               "Check internet connectivity",
               "Verify BV-BRC service status at https://www.bv-brc.org/",
               "Try again in a few minutes"
           ])
           error_guidance["retry_recommended"] = True
           
       elif "permission" in str(error).lower() or "access" in str(error).lower():
           error_guidance["suggested_actions"].extend([
               "Check BV-BRC installation permissions",
               "Run with appropriate user privileges",
               "Verify BV-BRC CLI tools are executable"
           ])
           
       elif "not found" in str(error).lower():
           error_guidance["suggested_actions"].extend([
               "Install BV-BRC application from https://www.bv-brc.org/",
               "Verify installation path configuration",
               "Check that CLI tools are properly installed"
           ])
           
       return error_guidance
   ```

#### **Step 4.1.2: PubMed Real API Integration**
**Objective**: Enable actual PubMed API calls with proper rate limiting
**Duration**: 1-2 days
**Priority**: High

**Implementation Steps**:

1. **Real PubMed API Client**
   ```python
   # File: nanobrain/library/tools/bioinformatics/pubmed_real_api_client.py
   from Bio import Entrez
   import aiohttp
   import asyncio
   from typing import List, Dict, Any
   
   class PubMedRealAPIClient:
       """Real PubMed API client with rate limiting and caching"""
       
       def __init__(self, email: str, api_key: Optional[str] = None):
           self.email = email
           self.api_key = api_key
           Entrez.email = email
           if api_key:
               Entrez.api_key = api_key
               
           # Rate limiting (NCBI allows 3 requests/second without API key, 10/second with)
           self.rate_limit = 10 if api_key else 3
           self.last_request_time = 0
           
       async def search_real_alphavirus_literature(self, protein_type: str) -> List[Dict[str, Any]]:
           """Search real PubMed for Alphavirus literature"""
           
           await self._enforce_rate_limit()
           
           # Construct optimized search query
           search_terms = self._build_alphavirus_search_query(protein_type)
           
           try:
               # Real Entrez esearch call
               handle = Entrez.esearch(
                   db="pubmed",
                   term=search_terms,
                   retmax=20,
                   sort="relevance"
               )
               search_results = Entrez.read(handle)
               handle.close()
               
               pmids = search_results["IdList"]
               
               if not pmids:
                   self.logger.warning(f"No PubMed results for {protein_type} Alphavirus")
                   return []
               
               # Fetch article details
               await self._enforce_rate_limit()
               
               handle = Entrez.efetch(
                   db="pubmed",
                   id=",".join(pmids[:10]),  # Limit to top 10
                   rettype="medline",
                   retmode="xml"
               )
               articles = Entrez.read(handle)
               handle.close()
               
               # Process real article data
               processed_articles = []
               for article in articles["PubmedArticle"]:
                   processed_article = self._process_real_article(article, protein_type)
                   if processed_article:
                       processed_articles.append(processed_article)
               
               self.logger.info(f"✅ Found {len(processed_articles)} relevant articles for {protein_type}")
               return processed_articles
               
           except Exception as e:
               self.logger.error(f"❌ Real PubMed search failed for {protein_type}: {e}")
               return []
               
       async def _enforce_rate_limit(self):
           """Enforce NCBI rate limiting"""
           current_time = time.time()
           time_since_last = current_time - self.last_request_time
           
           min_interval = 1.0 / self.rate_limit
           if time_since_last < min_interval:
               await asyncio.sleep(min_interval - time_since_last)
               
           self.last_request_time = time.time()
   ```

### **PHASE 4.2: PRODUCTION ALPHAVIRUS WORKFLOW IMPLEMENTATION**

#### **Step 4.2.1: Complete 14-Step Production Workflow**
**Objective**: Implement the full 14-step Alphavirus workflow for production use
**Duration**: 4-5 days
**Priority**: Critical

**Implementation Architecture**:
```python
# File: nanobrain/library/workflows/viral_protein_analysis/production_alphavirus_workflow.py
class ProductionAlphavirusWorkflow:
    """
    Production-ready 14-step Alphavirus protein analysis workflow
    
    Implements complete workflow from PHASE2_IMPLEMENTATION_PLAN.md:
    Steps 1-7:  BV-BRC Data Acquisition
    Step 8:     Annotation Mapping with ICTV
    Steps 9-11: Sequence Curation
    Step 12:    MMseqs2 Clustering
    Step 13:    MUSCLE Multiple Sequence Alignment
    Step 14:    PSSM Analysis and Viral_PSSM.json Output
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_production_config(config_path)
        self.logger = get_logger("production_alphavirus_workflow")
        
        # Initialize real API clients (no mocks)
        self.bvbrc_client = BVBRCRealAPIClient(production_mode=True)
        self.pubmed_client = PubMedRealAPIClient(
            email=self.config.pubmed.email,
            api_key=self.config.pubmed.api_key
        )
        
        # Initialize bioinformatics tools
        self.mmseqs_tool = MMseqs2Tool(production_mode=True)
        self.muscle_tool = MUSCLETool(production_mode=True)
        self.pssm_generator = PSSMGeneratorTool(production_mode=True)
        
        # Workflow state
        self.workflow_data = ProductionWorkflowData()
        self.execution_metrics = ProductionExecutionMetrics()
        
    async def execute_complete_workflow(self, input_params: Dict[str, Any]) -> ProductionWorkflowResult:
        """Execute complete production workflow with real API calls"""
        
        self.logger.info("🚀 Starting production Alphavirus workflow")
        workflow_start_time = time.time()
        
        try:
            # STEPS 1-7: BV-BRC DATA ACQUISITION
            await self._update_progress("Starting BV-BRC data acquisition", 5)
            acquisition_result = await self._execute_bvbrc_acquisition()
            self.workflow_data.update_from_acquisition(acquisition_result)
            
            # STEP 8: ANNOTATION MAPPING
            await self._update_progress("Mapping annotations with ICTV standards", 20)
            mapping_result = await self._execute_annotation_mapping()
            self.workflow_data.update_from_mapping(mapping_result)
            
            # STEPS 9-11: SEQUENCE CURATION
            await self._update_progress("Curating protein sequences", 35)
            curation_result = await self._execute_sequence_curation()
            self.workflow_data.update_from_curation(curation_result)
            
            # STEP 12: CLUSTERING
            await self._update_progress("Clustering proteins with MMseqs2", 50)
            clustering_result = await self._execute_mmseqs_clustering()
            self.workflow_data.update_from_clustering(clustering_result)
            
            # STEP 13: ALIGNMENT
            await self._update_progress("Performing multiple sequence alignment", 70)
            alignment_result = await self._execute_muscle_alignment()
            self.workflow_data.update_from_alignment(alignment_result)
            
            # STEP 14: PSSM ANALYSIS
            await self._update_progress("Generating PSSM matrices and final report", 85)
            pssm_result = await self._execute_pssm_analysis()
            self.workflow_data.update_from_pssm(pssm_result)
            
            # GENERATE VIRAL_PSSM.JSON OUTPUT
            await self._update_progress("Creating Viral_PSSM.json output", 95)
            viral_pssm_json = await self._generate_viral_pssm_output()
            
            # FINALIZE RESULTS
            await self._update_progress("Workflow completed successfully", 100)
            
            total_execution_time = time.time() - workflow_start_time
            
            return ProductionWorkflowResult(
                success=True,
                execution_time=total_execution_time,
                workflow_data=self.workflow_data,
                viral_pssm_json=viral_pssm_json,
                metrics=self.execution_metrics.to_dict(),
                output_files=await self._collect_output_files()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Production workflow failed: {e}")
            return ProductionWorkflowResult(
                success=False,
                error=str(e),
                execution_time=time.time() - workflow_start_time,
                partial_data=self.workflow_data
            )
```

**Individual Step Implementations**:

1. **Steps 1-7: BV-BRC Data Acquisition**
   ```python
   async def _execute_bvbrc_acquisition(self) -> Dict[str, Any]:
       """Execute complete BV-BRC data acquisition (Steps 1-7)"""
       
       # Step 1: Download Alphavirus genomes
       genomes = await self.bvbrc_client.download_real_alphavirus_genomes()
       self.execution_metrics.record_step_completion("bvbrc_genome_download", len(genomes))
       
       # Step 2: Filter genomes by size
       filtered_genomes = await self.bvbrc_client.filter_genomes_by_size(genomes)
       self.execution_metrics.record_step_completion("genome_filtering", len(filtered_genomes))
       
       # Steps 3-4: Get unique protein MD5s
       genome_ids = [g.genome_id for g in filtered_genomes]
       unique_proteins = await self.bvbrc_client.get_unique_protein_md5s(genome_ids)
       self.execution_metrics.record_step_completion("protein_md5_extraction", len(unique_proteins))
       
       # Step 5: Get feature sequences
       md5_list = [p.aa_sequence_md5 for p in unique_proteins]
       sequences = await self.bvbrc_client.get_feature_sequences(md5_list)
       self.execution_metrics.record_step_completion("sequence_download", len(sequences))
       
       # Step 6: Get annotations
       annotations = await self.bvbrc_client.get_protein_annotations(md5_list)
       self.execution_metrics.record_step_completion("annotation_download", len(annotations))
       
       # Step 7: Create annotated FASTA
       annotated_fasta = await self.bvbrc_client.create_annotated_fasta(sequences, annotations)
       self.execution_metrics.record_step_completion("fasta_creation", len(annotated_fasta.split('\n')))
       
       return {
           "original_genomes": genomes,
           "filtered_genomes": filtered_genomes,
           "unique_proteins": unique_proteins,
           "sequences": sequences,
           "annotations": annotations,
           "annotated_fasta": annotated_fasta
       }
   ```

2. **Step 8: ICTV Annotation Mapping**
   ```python
   async def _execute_annotation_mapping(self) -> Dict[str, Any]:
       """Execute annotation mapping with ICTV standards (Step 8)"""
       
       # Load ICTV Alphavirus genome organization
       ictv_mapping = await self._load_ictv_alphavirus_mapping()
       
       # Standardize annotations using ICTV mapping
       standardized_annotations = []
       for annotation in self.workflow_data.annotations:
           standard_annotation = await self._map_to_ictv_standard(annotation, ictv_mapping)
           standardized_annotations.append(standard_annotation)
       
       # Generate genome schematics
       genome_schematics = await self._generate_ictv_genome_schematics(standardized_annotations)
       
       return {
           "standardized_annotations": standardized_annotations,
           "genome_schematics": genome_schematics,
           "ictv_mapping_applied": True
       }
       
   async def _load_ictv_alphavirus_mapping(self) -> Dict[str, Any]:
       """Load ICTV Alphavirus genome organization mapping"""
       return {
           "genome_organization": "5'-nsP1-nsP2-nsP3-nsP4-Capsid-E3-E2-6K-E1-3'",
           "non_structural": {
               "nsP1": ["nonstructural protein 1", "nsp1", "replicase"],
               "nsP2": ["nonstructural protein 2", "nsp2", "protease", "helicase"],
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
       }
   ```

#### **Step 4.2.2: Viral_PSSM.json Production Output**
**Objective**: Generate production-quality Viral_PSSM.json matching reference format
**Duration**: 2 days
**Priority**: High

**Implementation**:
```python
async def _generate_viral_pssm_output(self) -> Dict[str, Any]:
    """Generate production Viral_PSSM.json output with literature integration"""
    
    viral_pssm_output = {
        "metadata": {
            "organism": "Alphavirus",
            "analysis_date": datetime.now().isoformat(),
            "coordinate_system": "1-based",
            "method": "nanobrain_production_alphavirus_analysis",
            "version": "4.1.0",
            "data_source": "BV-BRC",
            "total_genomes_analyzed": len(self.workflow_data.filtered_genomes),
            "clustering_method": "MMseqs2",
            "clustering_parameters": self.config.clustering.__dict__,
            "alignment_method": "MUSCLE",
            "pssm_generation_method": "literature_guided_nanobrain",
            "literature_integration": True,
            "ictv_mapping_applied": True
        },
        "proteins": [],
        "analysis_summary": {
            "total_proteins": len(self.workflow_data.clusters),
            "structural_proteins": len([c for c in self.workflow_data.clusters if c.protein_class == "structural"]),
            "non_structural_proteins": len([c for c in self.workflow_data.clusters if c.protein_class == "non_structural"]),
            "literature_supported_proteins": len([c for c in self.workflow_data.clusters if c.literature_support]),
            "high_confidence_boundaries": len([c for c in self.workflow_data.clusters if c.boundary_confidence > 0.8])
        },
        "quality_metrics": {
            "clustering_effectiveness": self.workflow_data.clustering_analysis.effectiveness_score,
            "alignment_quality": np.mean([c.alignment_quality for c in self.workflow_data.aligned_clusters]),
            "literature_coverage": self.execution_metrics.literature_coverage_percentage,
            "overall_confidence": self.execution_metrics.overall_confidence_score
        }
    }
    
    # Generate protein entries with literature references
    for cluster in self.workflow_data.clusters:
        protein_entry = await self._create_protein_entry_with_literature(cluster)
        viral_pssm_output["proteins"].append(protein_entry)
    
    return viral_pssm_output

async def _create_protein_entry_with_literature(self, cluster: ClusterData) -> Dict[str, Any]:
    """Create protein entry with integrated literature references"""
    
    # Get literature support for this protein
    literature_refs = await self.pubmed_client.search_real_alphavirus_literature(
        cluster.protein_class
    )
    
    return {
        "id": cluster.cluster_id,
        "function": cluster.standardized_annotation.standard_name,
        "protein_class": cluster.protein_class,
        "ictv_classification": cluster.ictv_classification,
        "boundaries": {
            "start": cluster.predicted_boundaries.start,
            "end": cluster.predicted_boundaries.end,
            "confidence": cluster.boundary_confidence,
            "method": "literature_guided_pssm",
            "supporting_evidence": cluster.supporting_evidence
        },
        "pssm_profile": {
            "matrix": cluster.pssm_matrix.matrix.tolist(),
            "alphabet": cluster.pssm_matrix.alphabet,
            "length": cluster.pssm_matrix.length,
            "conservation_profile": cluster.pssm_matrix.conservation_profile.tolist(),
            "consensus_sequence": cluster.pssm_matrix.consensus_sequence
        },
        "cluster_info": {
            "member_count": len(cluster.members),
            "representative_sequence": cluster.representative.sequence,
            "consensus_score": cluster.consensus_score,
            "alignment_quality": cluster.alignment_quality.mean_conservation
        },
        "literature_references": [
            {
                "pmid": ref["pmid"],
                "title": ref["title"],
                "authors": ref["authors"],
                "journal": ref["journal"],
                "year": ref["year"],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{ref['pmid']}/",
                "relevance_score": ref["relevance_score"],
                "boundary_evidence": ref.get("boundary_evidence", [])
            }
            for ref in literature_refs[:5]  # Top 5 most relevant
        ],
        "confidence_metrics": {
            "annotation_confidence": cluster.annotation_confidence,
            "boundary_confidence": cluster.boundary_confidence,
            "literature_support_score": cluster.literature_support_score,
            "overall_confidence": cluster.overall_confidence
        }
    }
```

### **PHASE 4.3: INTEGRATION TESTING WITH REAL APIs**

#### **Step 4.3.1: Real API Integration Test Suite**
**Objective**: Create comprehensive test suite using only real API calls
**Duration**: 3 days
**Priority**: Critical

**Implementation**:
```python
# File: tests/test_real_api_integration.py
@pytest.mark.integration
@pytest.mark.real_api
class TestRealAPIIntegration:
    """Integration tests using real API calls only - no mocks"""
    
    @pytest.fixture(scope="session")
    async def real_api_setup(self):
        """Setup real API clients for testing"""
        # Verify all tools are available
        verifier = ExternalToolsVerifier()
        tool_status = await verifier.verify_all_tools()
        
        if not all(tool_status.values()):
            pytest.skip(f"Required tools not available: {tool_status}")
        
        # Setup real API clients
        bvbrc_client = BVBRCRealAPIClient(production_mode=False)  # Test mode
        pubmed_client = PubMedRealAPIClient(
            email="test@nanobrain.org",
            api_key=None  # Use rate-limited access for testing
        )
        
        return {
            "bvbrc": bvbrc_client,
            "pubmed": pubmed_client,
            "tools_verified": tool_status
        }
    
    @pytest.mark.asyncio
    async def test_real_bvbrc_alphavirus_download(self, real_api_setup):
        """Test real BV-BRC Alphavirus genome download"""
        bvbrc_client = real_api_setup["bvbrc"]
        
        # Real API call to download Alphavirus genomes
        genomes = await bvbrc_client.download_real_alphavirus_genomes()
        
        # Validate real data
        assert len(genomes) >= 5, f"Expected at least 5 Alphavirus genomes, got {len(genomes)}"
        
        for genome in genomes:
            assert isinstance(genome, GenomeData)
            assert genome.genome_id is not None
            assert genome.genome_length > 0
            assert "alphavirus" in genome.taxon_lineage.lower()
            
        self.logger.info(f"✅ Successfully downloaded {len(genomes)} real Alphavirus genomes")
    
    @pytest.mark.asyncio
    async def test_complete_real_alphavirus_workflow(self, real_api_setup):
        """Test complete Alphavirus workflow with real APIs"""
        
        # Initialize production workflow
        workflow = ProductionAlphavirusWorkflow()
        
        # Execute with real APIs (limited dataset for testing)
        input_params = {
            "target_genus": "Alphavirus",
            "limit_genomes": 10,  # Limit for testing
            "test_mode": True
        }
        
        result = await workflow.execute_complete_workflow(input_params)
        
        # Validate real workflow results
        assert result.success, f"Workflow failed: {result.error}"
        assert result.viral_pssm_json is not None
        assert len(result.viral_pssm_json["proteins"]) > 0
        
        # Validate Viral_PSSM.json structure
        viral_pssm = result.viral_pssm_json
        assert "metadata" in viral_pssm
        assert "proteins" in viral_pssm
        assert "analysis_summary" in viral_pssm
        
        # Validate protein entries have real data
        for protein in viral_pssm["proteins"]:
            assert "literature_references" in protein
            assert len(protein["literature_references"]) > 0  # Should have real literature
            assert "pssm_profile" in protein
            assert protein["pssm_profile"]["matrix"]  # Should have real PSSM data
            
        self.logger.info(f"✅ Complete real workflow succeeded in {result.execution_time:.2f} seconds")
```

### **PHASE 4.4: PRODUCTION DEPLOYMENT INFRASTRUCTURE**

#### **Step 4.4.1: Production Docker Configuration**
**Objective**: Create production-ready Docker deployment with real API access
**Duration**: 2 days
**Priority**: High

**Enhanced Docker Configuration**:
```yaml
# docker/docker-compose.production.yml
version: '3.8'

services:
  nanobrain-alphavirus-production:
    build:
      context: ..
      dockerfile: docker/Dockerfile.production
      target: production
    environment:
      - NANOBRAIN_ENV=production
      - ENABLE_REAL_APIS=true
      - BVBRC_INSTALLATION_PATH=/opt/bvbrc
      - PUBMED_EMAIL=${PUBMED_EMAIL}
      - PUBMED_API_KEY=${PUBMED_API_KEY}
    volumes:
      - bvbrc_data:/opt/bvbrc
      - alphavirus_output:/app/data/output
      - alphavirus_cache:/app/data/cache
    ports:
      - "8001:8001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    
  redis-production:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  postgres-production:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=nanobrain_alphavirus
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  bvbrc_data:
    driver: local
  alphavirus_output:
    driver: local
  alphavirus_cache:
    driver: local
  redis_data:
    driver: local
  postgres_data:
    driver: local
```

**Production Dockerfile with Real Tool Installation**:
```dockerfile
# docker/Dockerfile.production
FROM python:3.11-slim as base

# Install system dependencies for bioinformatics tools
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install MMseqs2
FROM base as mmseqs_installer
RUN wget https://github.com/soedinglab/MMseqs2/releases/latest/download/mmseqs-linux-avx2.tar.gz \
    && tar xvzf mmseqs-linux-avx2.tar.gz \
    && cp mmseqs/bin/* /usr/local/bin/ \
    && rm -rf mmseqs*

# Install MUSCLE
FROM mmseqs_installer as muscle_installer
RUN wget https://github.com/rcedgar/muscle/releases/latest/download/muscle5.1.linux_intel64 \
    && chmod +x muscle5.1.linux_intel64 \
    && mv muscle5.1.linux_intel64 /usr/local/bin/muscle

# Production stage
FROM muscle_installer as production

# Create application user
RUN useradd -m -s /bin/bash nanobrain

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=nanobrain:nanobrain nanobrain/ nanobrain/
COPY --chown=nanobrain:nanobrain requirements.txt .
COPY --chown=nanobrain:nanobrain setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# Create data directories
RUN mkdir -p /app/data/{output,cache,logs} \
    && chown -R nanobrain:nanobrain /app/data

# Switch to application user
USER nanobrain

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Start production server
CMD ["python", "-m", "nanobrain.library.workflows.viral_protein_analysis.production_server"]
```

---

## 🔄 INTEGRATION TESTING EXECUTION PLAN

### **Phase 4A: Infrastructure Setup (Days 1-2)**
1. **Day 1**: Set up real API client implementations
2. **Day 2**: Configure production environment and Docker

### **Phase 4B: Real API Integration (Days 3-6)**
3. **Day 3**: Implement BV-BRC real API calls and testing
4. **Day 4**: Implement PubMed real API calls and testing
5. **Day 5**: Integrate external tools (MMseqs2, MUSCLE) with real data
6. **Day 6**: End-to-end real API testing

### **Phase 4C: Production Workflow (Days 7-11)**
7. **Day 7-8**: Implement complete 14-step production workflow
8. **Day 9**: Implement Viral_PSSM.json production output
9. **Day 10**: Performance testing with real data volumes
10. **Day 11**: Production deployment testing

### **Phase 4D: Final Integration (Days 12-14)**
11. **Day 12**: Phase 1-4 integration testing
12. **Day 13**: Production documentation and deployment guides
13. **Day 14**: Final validation and production readiness assessment

---

## ❓ CLARIFICATION QUESTIONS

Before proceeding with implementation, I need clarification on:

### **1. BV-BRC Installation Requirements**
- **Question**: Do you have BV-BRC CLI tools installed locally at `/Applications/BV-BRC.app/`?
- **Impact**: Determines whether we need to provide installation instructions or use alternative access methods
- **Options**: 
  - A) Use existing local installation
  - B) Provide Docker-based BV-BRC installation
  - C) Use BV-BRC web API instead of CLI tools

### **2. PubMed API Access**
- **Question**: Do you have a PubMed API key for increased rate limits (10 req/sec vs 3 req/sec)?
- **Impact**: Affects literature search performance and testing speed
- **Recommendation**: Obtain API key from NCBI for production use

### **3. Expected Data Volumes**
- **Question**: What's the target data volume for production Alphavirus analysis?
- **Impact**: Affects performance testing thresholds and resource allocation
- **Options**:
  - A) Small scale: 10-50 genomes
  - B) Medium scale: 100-500 genomes  
  - C) Large scale: 1000+ genomes

### **4. Production Deployment Target**
- **Question**: What's the target deployment environment?
- **Impact**: Affects Docker configuration and resource planning
- **Options**:
  - A) Local development machine
  - B) Single production server
  - C) Cloud deployment (AWS/GCP/Azure)
  - D) HPC cluster environment

### **5. Literature Integration Scope**
- **Question**: Should literature integration be limited to PubMed or include other sources?
- **Impact**: Affects API integration complexity and data quality
- **Current Plan**: PubMed only for initial implementation

### **6. Error Handling Strategy**
- **Question**: How should the system handle real API failures (network issues, rate limits, etc.)?
- **Impact**: Affects user experience and system reliability
- **Options**:
  - A) Fail fast with detailed error messages
  - B) Retry with exponential backoff
  - C) Graceful degradation with cached data
  - D) User interaction for failure resolution

---

## 🎯 SUCCESS CRITERIA CHECKLIST

**Technical Requirements**:
- [ ] All tests pass with real API calls (zero mocks)
- [ ] Complete 14-step Alphavirus workflow implemented
- [ ] Viral_PSSM.json output matches reference format
- [ ] Production Docker deployment functional
- [ ] Performance meets requirements (<1 hour for full dataset)
- [ ] Literature integration provides relevant references

**Quality Requirements**:
- [ ] >90% test coverage maintained
- [ ] Error handling covers all API failure modes  
- [ ] Documentation complete for production deployment
- [ ] User interface provides clear progress feedback
- [ ] System handles network failures gracefully

**Production Readiness**:
- [ ] Docker deployment works end-to-end
- [ ] Configuration management supports multiple environments
- [ ] Monitoring and health checks functional
- [ ] Resource usage within acceptable limits
- [ ] Security best practices implemented

**Please provide clarification on the questions above so I can proceed with the most appropriate implementation approach!** 