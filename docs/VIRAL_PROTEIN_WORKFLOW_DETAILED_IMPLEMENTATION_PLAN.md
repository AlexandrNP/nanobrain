# VIRAL PROTEIN WORKFLOW - DETAILED IMPLEMENTATION PLAN (UPDATED)

## OVERVIEW

This document provides a comprehensive step-by-step implementation plan for the viral protein boundary identification workflow, incorporating all user requirements and clarifications:

### USER CLARIFICATIONS INCORPORATED ✅

1. **BV-BRC Authentication**: Anonymous access with optional authentication
2. **PubMed Caching**: Aggressive caching strategy implemented
3. **Web Framework**: Standalone implementation reusing NanoBrain components
4. **Reference Validation**: Prioritize shorter, well-conserved sequence clusters
5. **User Timeout**: 48 hours for production, 10 seconds for testing
6. **Literature Scope**: Focus on PubMed, placeholder for future expansion
7. **Confidence Weighting**: Prioritize boundaries supported by literature
8. **Error Recovery**: No cached fallback, notify user of failures
9. **Test Organism**: EEEV (Eastern Equine Encephalitis Virus)

### EEEV-SPECIFIC TESTING CONFIGURATION

**Target Organism**: Eastern Equine Encephalitis Virus (EEEV)
**Genome Size**: ~11.7 kb
**Key Structural Proteins**:
- Capsid protein (C)
- Envelope proteins (E1, E2, E3)
- 6K protein
**Expected Results**: 4-6 structural protein boundaries

## DETAILED STEP-BY-STEP IMPLEMENTATION PLAN

### PHASE 1: CORE FRAMEWORK SETUP

#### Step 1.1: Enhanced Email Configuration System
**Objective**: Create configurable email system with optional usage
**Files**: 
- `nanobrain/library/workflows/viral_protein_analysis/config/email_config.yml`
- `nanobrain/core/bioinformatics/email_manager.py`

**Implementation Steps**:
1. Create email configuration with optional usage flags
2. Implement EmailManager with conditional email provision
3. Add email validation and rate limiting
4. Create testing configuration for EEEV workflow

**Enhanced Email Configuration**:
```yaml
# config/email_config.yml
email_config:
  default_email: "onarykov@anl.gov"
  
  # Service-specific email usage
  service_usage:
    bvbrc_api:
      required: false
      use_email: false  # Anonymous access
      fallback_with_auth: true
    pubmed_api:
      required: true   # For rate limiting
      use_email: true
      respect_rate_limits: true
      
  # Rate limiting configuration
  rate_limiting:
    pubmed:
      max_requests_per_hour: 100
      aggressive_caching: true
      cache_duration_hours: 168  # 1 week
    bvbrc:
      max_requests_per_second: 10
      anonymous_limits: true
      
  # Testing configuration
  testing:
    timeout_seconds: 10
    use_cached_responses: true
    mock_api_calls: false
    
  # Production configuration  
  production:
    timeout_hours: 48
    use_cached_responses: true
    mock_api_calls: false
```

**Enhanced EmailManager Interface**:
```python
class EmailManager:
    """Enhanced email manager with conditional usage and caching"""
    
    def __init__(self, config_path: str, environment: str = "production"):
        self.config = self._load_config(config_path)
        self.environment = environment
        self.email = self.config.get("default_email", "onarykov@anl.gov")
        self.cache = CacheManager()
        
    def get_email_for_service(self, service: str) -> Optional[str]:
        """Return email only if required and configured for service"""
        service_config = self.config.get("service_usage", {}).get(service, {})
        
        if service_config.get("use_email", False):
            return self.email
        return None
        
    def should_authenticate(self, service: str) -> bool:
        """Check if service should use authentication"""
        service_config = self.config.get("service_usage", {}).get(service, {})
        return service_config.get("fallback_with_auth", False)
        
    def get_timeout_config(self) -> Dict[str, Any]:
        """Get timeout configuration based on environment"""
        if self.environment == "testing":
            return self.config.get("testing", {})
        return self.config.get("production", {})
```

#### Step 1.2: Aggressive Caching System
**Objective**: Implement comprehensive caching for API responses
**Files**:
- `nanobrain/core/bioinformatics/cache_manager.py`
- `nanobrain/library/workflows/viral_protein_analysis/config/cache_config.yml`

**Implementation Steps**:
1. Create multi-tier caching system (memory + disk)
2. Implement cache invalidation strategies
3. Add cache warming for common EEEV queries
4. Create cache statistics and monitoring

**Cache Configuration**:
```yaml
# config/cache_config.yml
cache_config:
  storage:
    memory_cache:
      max_size_mb: 512
      ttl_hours: 24
    disk_cache:
      directory: "data/cache"
      max_size_gb: 5
      ttl_days: 7
      
  strategies:
    pubmed_references:
      aggressive_caching: true
      cache_duration_days: 30
      pre_warm_common_queries: true
      deduplicate_similar: true
      
    bvbrc_data:
      cache_duration_hours: 168  # 1 week
      invalidate_on_version_change: true
      compress_large_responses: true
      
  # EEEV-specific cache warming
  eeev_preload:
    organisms:
      - "Eastern equine encephalitis virus"
      - "EEEV"
    protein_types:
      - "capsid"
      - "envelope"
      - "structural"
    common_searches:
      - "EEEV structural proteins"
      - "alphavirus capsid boundary"
      - "envelope protein cleavage"
```

**CacheManager Interface**:
```python
class CacheManager:
    """Multi-tier caching system with aggressive PubMed caching"""
    
    def __init__(self, config_path: str = "config/cache_config.yml"):
        self.config = self._load_config(config_path)
        self.memory_cache = TTLCache(maxsize=1000, ttl=3600)
        self.disk_cache = DiskCache(self.config["storage"]["disk_cache"]["directory"])
        
    async def get_cached_response(self, cache_key: str, service: str) -> Optional[Any]:
        """Get cached response with service-specific strategies"""
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
            
        # Check disk cache
        disk_result = await self.disk_cache.get(cache_key)
        if disk_result and self._is_cache_valid(disk_result, service):
            # Promote to memory cache
            self.memory_cache[cache_key] = disk_result["data"]
            return disk_result["data"]
            
        return None
        
    async def cache_response(self, cache_key: str, data: Any, service: str) -> None:
        """Cache response with service-specific TTL"""
        service_config = self.config["strategies"].get(service, {})
        
        # Store in memory cache
        self.memory_cache[cache_key] = data
        
        # Store in disk cache with metadata
        await self.disk_cache.set(cache_key, {
            "data": data,
            "service": service,
            "timestamp": time.time(),
            "ttl": self._get_ttl_for_service(service)
        })
        
    async def warm_eeev_cache(self) -> None:
        """Pre-warm cache with common EEEV queries"""
        eeev_config = self.config.get("eeev_preload", {})
        
        for organism in eeev_config.get("organisms", []):
            for protein_type in eeev_config.get("protein_types", []):
                cache_key = f"pubmed_{organism}_{protein_type}_boundary"
                if not await self.get_cached_response(cache_key, "pubmed_references"):
                    # This would trigger actual API call during initialization
                    pass
```

#### Step 1.3: Standalone Web Interface with NanoBrain Components
**Objective**: Create standalone interface reusing NanoBrain components
**Files**:
- `nanobrain/library/workflows/viral_protein_analysis/web/standalone_interface.py`
- `nanobrain/library/workflows/viral_protein_analysis/web/config/standalone_config.yml`

**Implementation Steps**:
1. Create standalone FastAPI application
2. Integrate selected NanoBrain components (logging, data units)
3. Implement custom WebSocket for progress tracking
4. Add EEEV-specific UI components

**Standalone Interface Configuration**:
```yaml
# config/standalone_config.yml
standalone_interface:
  server:
    host: "0.0.0.0"
    port: 8001
    title: "EEEV Protein Boundary Analysis"
    description: "Standalone interface for viral protein boundary identification"
    
  nanobrain_integration:
    reuse_components:
      - "logging_system"
      - "data_units"
      - "workflow_base"
    custom_components:
      - "websocket_manager"
      - "prompt_system"
      - "progress_tracker"
      
  eeev_specific:
    default_organism: "Eastern equine encephalitis virus"
    expected_proteins:
      - "capsid protein"
      - "envelope protein E1"
      - "envelope protein E2"
      - "6K protein"
    visualization:
      genome_size: 11700
      show_structural_organization: true
      
  user_interaction:
    production_timeout_hours: 48
    testing_timeout_seconds: 10
    auto_approve_threshold: 0.8
    require_confirmation_threshold: 0.5
```

**Standalone Interface Class**:
```python
class StandaloneViralProteinInterface:
    """Standalone interface reusing selected NanoBrain components"""
    
    def __init__(self, config_path: str = "config/standalone_config.yml"):
        self.config = self._load_config(config_path)
        
        # Reuse NanoBrain components
        self.logger = get_logger("viral_protein_interface")  # NanoBrain logging
        self.data_manager = DataUnitManager()  # NanoBrain data units
        
        # Custom components
        self.websocket_manager = CustomWebSocketManager()
        self.prompt_system = WebPromptSystem()
        self.progress_tracker = ProgressTracker()
        
        # EEEV-specific configuration
        self.eeev_config = self.config.get("eeev_specific", {})
        
    async def initialize(self) -> FastAPI:
        """Initialize standalone FastAPI application"""
        app = FastAPI(
            title=self.config["server"]["title"],
            description=self.config["server"]["description"]
        )
        
        # Add CORS middleware
        app.add_middleware(CORSMiddleware, allow_origins=["*"])
        
        # Setup routes
        await self._setup_routes(app)
        
        # Setup WebSocket
        await self._setup_websocket(app)
        
        return app
        
    async def _setup_routes(self, app: FastAPI) -> None:
        """Setup API routes with EEEV-specific defaults"""
        
        @app.post("/api/v1/analyze")
        async def analyze_eeev_proteins(request: EEEVAnalysisRequest):
            """Analyze EEEV proteins with literature support"""
            
            # Set EEEV defaults
            if not request.organism:
                request.organism = self.eeev_config["default_organism"]
                
            # Get timeout based on environment
            timeout_config = self._get_timeout_config()
            
            # Start analysis
            analysis_id = await self._start_analysis(request, timeout_config)
            
            return {
                "analysis_id": analysis_id,
                "status": "started",
                "organism": request.organism,
                "expected_proteins": self.eeev_config["expected_proteins"],
                "timeout_config": timeout_config
            }
```

### PHASE 2: API CLIENTS WITH ENHANCED FEATURES

#### Step 2.1: BV-BRC API Client with Anonymous/Authenticated Access
**Objective**: Implement BV-BRC API with flexible authentication
**File**: `nanobrain/library/tools/bioinformatics/bvbrc_api_client.py`

**Implementation Steps**:
1. Implement anonymous access as primary method
2. Add optional authentication fallback
3. Create EEEV-specific query optimizations
4. Add response validation and error handling

**BV-BRC API Client Interface**:
```python
class BVBRCAPIClient(BaseAPIClient):
    """BV-BRC API client with anonymous access and optional authentication"""
    
    def __init__(self, email_manager: EmailManager, cache_manager: CacheManager):
        super().__init__("https://www.bv-brc.org/api/", email_manager)
        self.cache_manager = cache_manager
        self.use_authentication = False
        self.authentication_token = None
        
    async def get_eeev_genomes(self) -> List[Dict[str, Any]]:
        """Get EEEV genomes with caching and validation"""
        cache_key = "bvbrc_eeev_genomes"
        
        # Check cache first
        cached_result = await self.cache_manager.get_cached_response(
            cache_key, "bvbrc_data"
        )
        if cached_result:
            return cached_result
            
        # Query BV-BRC API
        params = {
            "q": 'organism:"Eastern equine encephalitis virus"',
            "select": "genome_id,genome_name,genome_length,genome_status,completion_date",
            "limit": 100,
            "sort": "-completion_date"
        }
        
        try:
            # Try anonymous access first
            response = await self._make_anonymous_request("genome", params)
        except AuthenticationError:
            if self.email_manager.should_authenticate("bvbrc_api"):
                response = await self._make_authenticated_request("genome", params)
            else:
                raise
                
        # Validate EEEV-specific genome data
        validated_genomes = self._validate_eeev_genomes(response)
        
        # Cache the results
        await self.cache_manager.cache_response(cache_key, validated_genomes, "bvbrc_data")
        
        return validated_genomes
        
    def _validate_eeev_genomes(self, genomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate EEEV genome data with expected characteristics"""
        validated = []
        
        for genome in genomes:
            # EEEV genomes should be ~11.7kb
            genome_length = genome.get("genome_length", 0)
            if 10000 <= genome_length <= 13000:  # Allow some variation
                validated.append(genome)
            else:
                self.logger.warning(f"EEEV genome {genome.get('genome_id')} has unusual length: {genome_length}")
                
        return validated
        
    async def get_eeev_structural_proteins(self, genome_ids: List[str]) -> List[Dict[str, Any]]:
        """Get EEEV structural proteins with boundary information"""
        cache_key = f"bvbrc_eeev_proteins_{'_'.join(genome_ids[:3])}"
        
        cached_result = await self.cache_manager.get_cached_response(
            cache_key, "bvbrc_data"
        )
        if cached_result:
            return cached_result
            
        # Query for structural proteins
        structural_keywords = ["capsid", "envelope", "structural", "E1", "E2", "6K"]
        keyword_query = " OR ".join([f'product:*{kw}*' for kw in structural_keywords])
        
        params = {
            "q": f'genome_id:({" OR ".join(genome_ids)}) AND ({keyword_query})',
            "select": "patric_id,genome_id,start,end,strand,product,aa_sequence_md5,aa_length",
            "limit": 1000
        }
        
        try:
            response = await self._make_anonymous_request("genome_feature", params)
        except AuthenticationError:
            if self.email_manager.should_authenticate("bvbrc_api"):
                response = await self._make_authenticated_request("genome_feature", params)
            else:
                raise
                
        # Filter and validate structural proteins
        structural_proteins = self._filter_eeev_structural_proteins(response)
        
        await self.cache_manager.cache_response(cache_key, structural_proteins, "bvbrc_data")
        
        return structural_proteins
        
    def _filter_eeev_structural_proteins(self, proteins: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter for genuine EEEV structural proteins"""
        structural_proteins = []
        
        for protein in proteins:
            product = protein.get("product", "").lower()
            
            # Prioritize shorter, well-conserved proteins
            aa_length = protein.get("aa_length", 0)
            
            # EEEV structural protein characteristics
            if any(keyword in product for keyword in ["capsid", "envelope", "structural"]):
                # Assign priority based on length and conservation
                if aa_length < 300:  # Shorter proteins get higher priority
                    protein["priority_score"] = 1.0
                elif aa_length < 500:
                    protein["priority_score"] = 0.8
                else:
                    protein["priority_score"] = 0.6
                    
                structural_proteins.append(protein)
                
        # Sort by priority (shorter, well-conserved first)
        structural_proteins.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
        
        return structural_proteins
```

#### Step 2.2: Enhanced PubMed Client with Aggressive Caching
**Objective**: Implement PubMed client with literature-prioritized boundary detection
**File**: `nanobrain/library/tools/bioinformatics/pubmed_api_client.py`

**Implementation Steps**:
1. Implement aggressive caching with deduplication
2. Create EEEV-specific search strategies
3. Add literature-based boundary prioritization
4. Implement reference quality scoring

**Enhanced PubMed Client Interface**:
```python
class EnhancedPubMedAPIClient(BaseAPIClient):
    """PubMed client with aggressive caching and literature prioritization"""
    
    def __init__(self, email_manager: EmailManager, cache_manager: CacheManager):
        super().__init__("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/", email_manager)
        self.cache_manager = cache_manager
        self.eeev_search_terms = self._load_eeev_search_terms()
        
    def _load_eeev_search_terms(self) -> Dict[str, List[str]]:
        """Load EEEV-specific search terms for boundary detection"""
        return {
            "capsid": [
                "EEEV capsid protein boundary",
                "alphavirus capsid cleavage",
                "Eastern equine encephalitis capsid processing",
                "capsid protein start codon",
                "capsid protein terminus"
            ],
            "envelope": [
                "EEEV envelope protein boundary",
                "alphavirus E1 E2 cleavage",
                "envelope protein signal peptide",
                "glycoprotein processing site",
                "E1 E2 junction"
            ],
            "structural": [
                "EEEV structural polyprotein",
                "alphavirus structural protein processing",
                "polyprotein cleavage sites",
                "structural protein boundaries"
            ]
        }
        
    async def search_eeev_boundary_literature(self, protein_type: str) -> List[Dict[str, Any]]:
        """Search for EEEV-specific boundary literature with aggressive caching"""
        cache_key = f"pubmed_eeev_{protein_type}_boundary_literature"
        
        # Check cache with extended TTL for literature
        cached_result = await self.cache_manager.get_cached_response(
            cache_key, "pubmed_references"
        )
        if cached_result:
            return cached_result
            
        # Get protein-specific search terms
        search_terms = self.eeev_search_terms.get(protein_type, [])
        if not search_terms:
            return []
            
        all_references = []
        
        # Search with each term and deduplicate
        for term in search_terms:
            term_references = await self._search_with_term(term)
            all_references.extend(term_references)
            
        # Deduplicate by PMID
        unique_references = self._deduplicate_references(all_references)
        
        # Filter for boundary-specific content
        boundary_references = []
        for ref in unique_references:
            if self._validate_boundary_content(ref, protein_type):
                boundary_score = self._calculate_boundary_score(ref, protein_type)
                ref["boundary_score"] = boundary_score
                
                # Only include if it has genuine boundary information
                if boundary_score > 0.3:
                    boundary_references.append(ref)
                    
        # Sort by boundary relevance score
        boundary_references.sort(key=lambda x: x.get("boundary_score", 0), reverse=True)
        
        # Aggressive caching with long TTL
        await self.cache_manager.cache_response(
            cache_key, boundary_references, "pubmed_references"
        )
        
        return boundary_references
        
    def _validate_boundary_content(self, reference: Dict[str, Any], protein_type: str) -> bool:
        """Validate reference contains genuine boundary information"""
        abstract = reference.get("abstract", "").lower()
        title = reference.get("title", "").lower()
        
        # EEEV-specific validation
        has_eeev = any(term in abstract + title for term in [
            "eastern equine encephalitis", "eeev", "alphavirus"
        ])
        
        # Protein-specific validation
        has_protein = protein_type.lower() in abstract + title
        
        # Boundary-specific validation (stricter criteria)
        boundary_indicators = [
            "cleavage site", "processing site", "boundary",
            "amino acid position", "residue position",
            "start codon", "stop codon", "terminus",
            "signal peptide", "cleavage", "processing"
        ]
        
        has_boundary_info = any(indicator in abstract for indicator in boundary_indicators)
        
        # Additional validation for specific position mentions
        has_position_info = bool(re.search(r'position\s+\d+|residue\s+\d+|site\s+\d+', abstract))
        
        return has_eeev and has_protein and (has_boundary_info or has_position_info)
        
    def _calculate_boundary_score(self, reference: Dict[str, Any], protein_type: str) -> float:
        """Calculate literature-based boundary confidence score"""
        abstract = reference.get("abstract", "").lower()
        title = reference.get("title", "").lower()
        
        score = 0.0
        
        # Base score for EEEV mention
        if "eastern equine encephalitis" in abstract + title:
            score += 0.3
        elif "eeev" in abstract + title:
            score += 0.2
        elif "alphavirus" in abstract + title:
            score += 0.1
            
        # Protein-specific score
        if protein_type.lower() in title:
            score += 0.2
        elif protein_type.lower() in abstract:
            score += 0.1
            
        # Boundary information score
        boundary_terms = {
            "cleavage site": 0.3,
            "processing site": 0.3,
            "amino acid position": 0.25,
            "residue position": 0.25,
            "boundary": 0.2,
            "terminus": 0.15,
            "signal peptide": 0.15
        }
        
        for term, weight in boundary_terms.items():
            if term in abstract:
                score += weight
                
        # Experimental validation bonus
        experimental_terms = ["mutagenesis", "deletion", "truncation", "site-directed"]
        if any(term in abstract for term in experimental_terms):
            score += 0.2
            
        return min(score, 1.0)
```

### PHASE 3: WORKFLOW IMPLEMENTATION WITH LITERATURE PRIORITIZATION

#### Step 3.1: Enhanced Protein Annotation Step
**Objective**: Implement protein annotation with shorter sequence prioritization
**File**: `nanobrain/library/workflows/viral_protein_analysis/steps/protein_annotation_step.py`

**Implementation Steps**:
1. Implement BV-BRC data acquisition with EEEV optimization
2. Add sequence clustering with length-based prioritization
3. Integrate user timeout configuration (48h/10s)
4. Add comprehensive error handling without fallbacks

**Enhanced Protein Annotation Step**:
```python
class EnhancedProteinAnnotationStep(BioinformaticsStep):
    """Protein annotation with shorter sequence prioritization and EEEV optimization"""
    
    def __init__(self, config: ProteinAnnotationStepConfig):
        super().__init__(config)
        self.bvbrc_client = BVBRCAPIClient(self.email_manager, self.cache_manager)
        self.mmseqs_tool = MMseqs2Tool()
        self.timeout_config = self.email_manager.get_timeout_config()
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process EEEV protein annotation with literature prioritization"""
        
        try:
            # Step 1: Get EEEV genomes
            await self._update_progress("Retrieving EEEV genome data", 10)
            eeev_genomes = await self.bvbrc_client.get_eeev_genomes()
            
            if not eeev_genomes:
                raise WorkflowError("No EEEV genomes found", notify_user=True)
                
            # Step 2: Filter genomes by quality and completeness
            await self._update_progress("Filtering genome data", 20)
            quality_genomes = self._filter_quality_genomes(eeev_genomes)
            
            # Step 3: Get structural proteins
            await self._update_progress("Retrieving structural proteins", 40)
            genome_ids = [g["genome_id"] for g in quality_genomes]
            structural_proteins = await self.bvbrc_client.get_eeev_structural_proteins(genome_ids)
            
            if not structural_proteins:
                raise WorkflowError("No structural proteins found", notify_user=True)
                
            # Step 4: Prioritize shorter, well-conserved sequences
            await self._update_progress("Prioritizing sequences", 60)
            prioritized_proteins = self._prioritize_shorter_sequences(structural_proteins)
            
            # Step 5: Cluster sequences with conservation focus
            await self._update_progress("Clustering sequences", 80)
            clusters = await self._cluster_with_conservation_priority(prioritized_proteins)
            
            # Step 6: User validation with timeout
            await self._update_progress("Validating clusters", 90)
            validated_clusters = await self._validate_clusters_with_timeout(clusters)
            
            await self._update_progress("Annotation complete", 100)
            
            return {
                "clusters": validated_clusters,
                "genome_count": len(quality_genomes),
                "protein_count": len(structural_proteins),
                "prioritization_applied": True,
                "eeev_optimized": True
            }
            
        except Exception as e:
            # No fallback to cached data - notify user of failure
            await self._notify_user_of_failure(e)
            raise
            
    def _prioritize_shorter_sequences(self, proteins: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize shorter, well-conserved sequences as specified"""
        
        # Group by protein type for conservation analysis
        protein_groups = {}
        for protein in proteins:
            product = protein.get("product", "").lower()
            
            # Classify protein type
            if "capsid" in product:
                ptype = "capsid"
            elif "envelope" in product or "e1" in product or "e2" in product:
                ptype = "envelope"
            elif "6k" in product:
                ptype = "6k"
            else:
                ptype = "other"
                
            if ptype not in protein_groups:
                protein_groups[ptype] = []
            protein_groups[ptype].append(protein)
            
        prioritized_proteins = []
        
        for ptype, group_proteins in protein_groups.items():
            # Sort by length (shorter first) and conservation indicators
            group_proteins.sort(key=lambda x: (
                x.get("aa_length", 0),  # Shorter first
                -x.get("priority_score", 0)  # Higher conservation score first
            ))
            
            # Take top sequences prioritizing shorter ones
            max_sequences = min(len(group_proteins), 10)  # Limit per type
            selected = group_proteins[:max_sequences]
            
            # Mark as prioritized
            for protein in selected:
                protein["prioritized"] = True
                protein["protein_type"] = ptype
                
            prioritized_proteins.extend(selected)
            
        return prioritized_proteins
        
    async def _cluster_with_conservation_priority(self, proteins: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster sequences with focus on conservation and shorter sequences"""
        
        # Configure MMseqs2 for conservation-focused clustering
        clustering_config = {
            "similarity_threshold": 0.8,  # Higher threshold for better conservation
            "coverage_threshold": 0.7,
            "cluster_mode": "set-cover",  # Prioritize representative sequences
            "prioritize_shorter": True
        }
        
        clusters = await self.mmseqs_tool.cluster_sequences(proteins, clustering_config)
        
        # Post-process clusters to prioritize shorter, well-conserved sequences
        processed_clusters = []
        for cluster in clusters:
            # Calculate cluster conservation score
            conservation_score = self._calculate_cluster_conservation(cluster)
            
            # Prioritize clusters with shorter sequences
            avg_length = np.mean([p.get("aa_length", 0) for p in cluster["members"]])
            length_priority = 1.0 / (1.0 + avg_length / 200.0)  # Shorter = higher priority
            
            cluster["conservation_score"] = conservation_score
            cluster["length_priority"] = length_priority
            cluster["combined_score"] = conservation_score * 0.6 + length_priority * 0.4
            
            processed_clusters.append(cluster)
            
        # Sort by combined score (conservation + length priority)
        processed_clusters.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        return processed_clusters
        
    async def _validate_clusters_with_timeout(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate clusters with configurable timeout"""
        validated_clusters = []
        timeout_seconds = self.timeout_config.get("timeout_seconds", 
                                                  self.timeout_config.get("timeout_hours", 48) * 3600)
        
        for cluster in clusters:
            if cluster.get("combined_score", 0) < 0.5:
                # Request user validation with timeout
                try:
                    approved = await asyncio.wait_for(
                        self._prompt_cluster_validation(cluster),
                        timeout=timeout_seconds
                    )
                    
                    if approved:
                        validated_clusters.append(cluster)
                        
                except asyncio.TimeoutError:
                    self.logger.warning(f"User validation timeout for cluster {cluster.get('id')}")
                    # Skip cluster on timeout - no fallback
                    continue
                    
            else:
                # Auto-approve high-confidence clusters
                validated_clusters.append(cluster)
                
        return validated_clusters
        
    async def _notify_user_of_failure(self, error: Exception) -> None:
        """Notify user of failure without fallback options"""
        notification = {
            "type": "workflow_failure",
            "error": str(error),
            "step": "protein_annotation",
            "timestamp": time.time(),
            "recovery_options": [
                "Retry the analysis",
                "Check network connectivity",
                "Contact support if issue persists"
            ],
            "no_fallback_available": True
        }
        
        await self.websocket_manager.send_notification(notification)
```

#### Step 3.2: Literature-Prioritized Boundary Detection Step
**Objective**: Implement boundary detection prioritizing literature support
**File**: `nanobrain/library/workflows/viral_protein_analysis/steps/boundary_detection_step.py`

**Implementation Steps**:
1. Integrate literature-backed PSSM generation
2. Implement boundary scoring with literature prioritization
3. Add confidence assessment combining PSSM and literature evidence
4. Create user validation with timeout handling

**Literature-Prioritized Boundary Detection**:
```python
class LiteraturePrioritizedBoundaryDetectionStep(BioinformaticsStep):
    """Boundary detection prioritizing literature-supported boundaries"""
    
    def __init__(self, config: BoundaryDetectionStepConfig):
        super().__init__(config)
        self.pubmed_client = EnhancedPubMedAPIClient(self.email_manager, self.cache_manager)
        self.pssm_generator = LiteraturePSSMGenerator(self.pubmed_client)
        self.timeout_config = self.email_manager.get_timeout_config()
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process boundary detection with literature prioritization"""
        
        boundary_results = []
        total_clusters = len(input_data.get("clusters", []))
        
        try:
            for i, cluster in enumerate(input_data.get("clusters", [])):
                progress = int((i / total_clusters) * 90)  # Reserve 10% for final processing
                await self._update_progress(f"Processing {cluster.get('protein_type', 'protein')} cluster", progress)
                
                # Get literature support first (prioritize literature)
                literature_support = await self._get_literature_support(cluster)
                
                # Generate PSSM with literature guidance
                pssm_result = await self.pssm_generator.generate_literature_guided_pssm(
                    cluster, literature_support
                )
                
                # Combine evidence with literature priority
                boundary_prediction = await self._combine_evidence_with_literature_priority(
                    pssm_result, literature_support
                )
                
                # Validate if confidence is low
                if boundary_prediction["confidence"] < 0.7:
                    validated = await self._validate_boundary_with_timeout(
                        boundary_prediction, literature_support
                    )
                    if not validated:
                        continue
                        
                boundary_results.append(boundary_prediction)
                
            await self._update_progress("Finalizing results", 100)
            
            return {
                "boundary_results": boundary_results,
                "literature_prioritized": True,
                "total_proteins": len(boundary_results),
                "eeev_specific": True,
                "confidence_distribution": self._analyze_confidence_distribution(boundary_results)
            }
            
        except Exception as e:
            await self._notify_user_of_failure(e)
            raise
            
    async def _get_literature_support(self, cluster: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get literature support for cluster with aggressive caching"""
        protein_type = cluster.get("protein_type", "structural")
        
        # Search for EEEV-specific literature
        literature_refs = await self.pubmed_client.search_eeev_boundary_literature(protein_type)
        
        # Filter for high-quality boundary information
        high_quality_refs = []
        for ref in literature_refs:
            if ref.get("boundary_score", 0) > 0.5:
                # Extract specific boundary information
                boundary_info = self._extract_boundary_positions(ref, protein_type)
                if boundary_info:
                    ref["extracted_boundaries"] = boundary_info
                    high_quality_refs.append(ref)
                    
        return high_quality_refs
        
    async def _combine_evidence_with_literature_priority(self, 
                                                        pssm_result: Dict[str, Any], 
                                                        literature_support: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine PSSM and literature evidence with literature priority (70% weight)"""
        
        pssm_boundaries = pssm_result.get("boundary_predictions", [])
        literature_boundaries = []
        
        # Extract literature-supported boundaries
        for ref in literature_support:
            for boundary in ref.get("extracted_boundaries", []):
                literature_boundaries.append({
                    "position": boundary["position"],
                    "type": boundary["type"],
                    "literature_score": ref.get("boundary_score", 0),
                    "pmid": ref.get("pmid"),
                    "support_type": "literature"
                })
                
        # Combine and prioritize literature-supported boundaries
        combined_boundaries = []
        
        for pssm_boundary in pssm_boundaries:
            pssm_pos = pssm_boundary.get("position")
            pssm_score = pssm_boundary.get("score", 0)
            
            # Look for literature support within ±5 amino acids
            literature_match = None
            for lit_boundary in literature_boundaries:
                if abs(lit_boundary["position"] - pssm_pos) <= 5:
                    literature_match = lit_boundary
                    break
                    
            if literature_match:
                # Literature-supported boundary (high confidence)
                combined_score = (
                    literature_match["literature_score"] * 0.7 +  # Literature priority
                    pssm_score * 0.3
                )
                
                combined_boundaries.append({
                    "position": pssm_pos,
                    "type": pssm_boundary.get("type"),
                    "combined_score": combined_score,
                    "pssm_score": pssm_score,
                    "literature_score": literature_match["literature_score"],
                    "literature_supported": True,
                    "supporting_pmid": literature_match["pmid"],
                    "confidence": min(combined_score * 1.2, 1.0)  # Boost for literature support
                })
            else:
                # PSSM-only boundary (lower confidence)
                combined_boundaries.append({
                    "position": pssm_pos,
                    "type": pssm_boundary.get("type"),
                    "combined_score": pssm_score * 0.5,  # Penalize lack of literature support
                    "pssm_score": pssm_score,
                    "literature_score": 0.0,
                    "literature_supported": False,
                    "confidence": pssm_score * 0.5
                })
                
        # Sort by combined score (literature priority)
        combined_boundaries.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        # Calculate overall confidence
        if combined_boundaries:
            literature_supported_count = sum(1 for b in combined_boundaries if b["literature_supported"])
            literature_support_ratio = literature_supported_count / len(combined_boundaries)
            overall_confidence = np.mean([b["confidence"] for b in combined_boundaries])
            
            # Boost confidence if many boundaries are literature-supported
            if literature_support_ratio > 0.5:
                overall_confidence = min(overall_confidence * 1.3, 1.0)
        else:
            overall_confidence = 0.0
            
        return {
            "boundaries": combined_boundaries,
            "confidence": overall_confidence,
            "literature_support_ratio": literature_support_ratio,
            "pssm_data": pssm_result,
            "literature_references": literature_support
        }
        
    async def _validate_boundary_with_timeout(self, 
                                            boundary_prediction: Dict[str, Any], 
                                            literature_support: List[Dict[str, Any]]) -> bool:
        """Validate boundary prediction with user interaction and timeout"""
        
        timeout_seconds = self.timeout_config.get("timeout_seconds", 
                                                  self.timeout_config.get("timeout_hours", 48) * 3600)
        
        validation_data = {
            "boundaries": boundary_prediction["boundaries"],
            "confidence": boundary_prediction["confidence"],
            "literature_support": literature_support,
            "literature_support_ratio": boundary_prediction.get("literature_support_ratio", 0),
            "recommendation": self._generate_validation_recommendation(boundary_prediction)
        }
        
        try:
            approved = await asyncio.wait_for(
                self._prompt_boundary_validation(validation_data),
                timeout=timeout_seconds
            )
            return approved
            
        except asyncio.TimeoutError:
            self.logger.warning("Boundary validation timeout - skipping low-confidence prediction")
            return False
            
    def _generate_validation_recommendation(self, boundary_prediction: Dict[str, Any]) -> str:
        """Generate recommendation for user validation"""
        confidence = boundary_prediction.get("confidence", 0)
        lit_ratio = boundary_prediction.get("literature_support_ratio", 0)
        
        if confidence > 0.8 and lit_ratio > 0.5:
            return "HIGH CONFIDENCE: Strong literature support with good PSSM agreement"
        elif lit_ratio > 0.7:
            return "LITERATURE SUPPORTED: Multiple references support these boundaries"
        elif confidence > 0.7:
            return "PSSM SUPPORTED: Good computational prediction, limited literature"
        else:
            return "LOW CONFIDENCE: Manual review recommended"
```

### PHASE 4: TESTING AND VALIDATION FRAMEWORK

#### Step 4.1: EEEV-Specific Testing Suite
**Objective**: Create comprehensive testing framework for EEEV workflow
**Files**: 
- `tests/test_eeev_workflow.py`
- `tests/data/eeev_test_data.yml`

**EEEV Testing Configuration**:
```yaml
# tests/data/eeev_test_data.yml
eeev_test_data:
  expected_results:
    genome_count: 5-20
    structural_proteins:
      - name: "capsid protein"
        expected_length: 250-280
        expected_boundaries: 2
      - name: "envelope protein E1"
        expected_length: 400-450
        expected_boundaries: 2-3
      - name: "envelope protein E2"
        expected_length: 420-470
        expected_boundaries: 2-3
      - name: "6K protein"
        expected_length: 50-60
        expected_boundaries: 2
        
  literature_expectations:
    min_references_per_protein: 2
    min_boundary_score: 0.3
    expected_pmids:
      - "12345678"  # Example PMIDs for EEEV structural proteins
      - "87654321"
      
  performance_benchmarks:
    max_execution_time_seconds: 300
    cache_hit_ratio_min: 0.7
    api_call_limit: 100
    
  timeout_testing:
    production_timeout_hours: 48
    testing_timeout_seconds: 10
    mock_user_responses: true
```

## IMPLEMENTATION QUESTIONS FOR FINAL CLARIFICATION

### 1. EEEV Genome Selection Criteria
**Question**: Should we focus on complete EEEV genomes only, or include draft assemblies?
**Impact**: Affects data quality and boundary prediction accuracy

### 2. Literature Reference Threshold
**Question**: What minimum boundary score should we use to include literature references (currently 0.3)?
**Impact**: Affects reference quality vs. quantity balance

### 3. Cluster Size Limits
**Question**: Should we set maximum cluster sizes to prioritize shorter sequences more effectively?
**Impact**: Affects computational efficiency and result quality

### 4. Cache Warming Strategy
**Question**: Should we pre-warm the cache with common EEEV queries during system initialization?
**Impact**: Affects initial startup time vs. runtime performance

### 5. User Interface Complexity
**Question**: How detailed should the boundary validation interface be for user review?
**Impact**: Affects user experience and validation accuracy

### 6. Error Notification Detail
**Question**: How much technical detail should be included in user error notifications?
**Impact**: Affects user understanding vs. interface simplicity

## IMPLEMENTATION TIMELINE

**Phase 1** (Core Framework): 3-4 days
**Phase 2** (API Clients): 4-5 days  
**Phase 3** (Workflow Steps): 5-6 days
**Phase 4** (Testing): 2-3 days
**Phase 5** (Documentation): 1-2 days

**Total Estimated Time**: 15-20 days

## NEXT STEPS

Please review this enhanced implementation plan and provide clarification on the remaining questions. The plan now includes:

✅ Anonymous BV-BRC access with optional authentication
✅ Aggressive PubMed caching with deduplication
✅ Standalone interface reusing NanoBrain components  
✅ Shorter sequence prioritization in clustering
✅ Configurable timeouts (48h production, 10s testing)
✅ Literature-prioritized boundary detection (70% weight)
✅ No fallback strategies - user notification on failures
✅ EEEV-specific optimizations and testing

**Ready for implementation upon your approval!** 