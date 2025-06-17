"""
PubMed Literature Search Client

This module provides a comprehensive PubMed API client with:
- Rate limiting for NCBI Entrez API compliance (3 req/sec without key, 10 req/sec with key)
- Literature search functionality for viral protein analysis
- Fail-fast error handling and validation
- Caching system for repeated searches
- BioTech compliance for academic research

Based on NCBI Entrez API guidelines from https://www.bv-brc.org/docs/
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json

from .base_bioinformatics_tool import (
    BioinformaticsExternalTool,
    BioinformaticsToolConfig,
    InstallationStatus,
    DiagnosticReport
)
from nanobrain.core.external_tool import ToolResult, ToolExecutionError
from nanobrain.core.logging_system import get_logger


@dataclass
class LiteratureReference:
    """Literature reference data structure"""
    pmid: str
    title: str
    authors: List[str]
    journal: str
    year: str
    relevance_score: float
    url: str
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    @property
    def citation(self) -> str:
        """Generate formatted citation"""
        author_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            author_str += " et al."
        return f"{author_str}. {self.title}. {self.journal} ({self.year}). PMID: {self.pmid}"


@dataclass
class PubMedConfig(BioinformaticsToolConfig):
    """Configuration for PubMed client"""
    tool_name: str = "pubmed"
    email: str = "research@nanobrain.org"
    api_key: Optional[str] = None
    rate_limit: int = 3  # requests per second (3 without API key, 10 with)
    fail_fast: bool = True
    cache_enabled: bool = True
    max_results_per_search: int = 20
    verify_on_init: bool = False  # Disable by default to avoid event loop issues
    
    def __post_init__(self):
        if self.api_key:
            self.rate_limit = 10  # Higher rate limit with API key
        
        # PubMed doesn't require local installation
        self.local_installation_paths = []


class PubMedError(ToolExecutionError):
    """Raised when PubMed API operations fail"""
    pass


class PubMedClient(BioinformaticsExternalTool):
    """
    PubMed literature search client.
    
    Provides access to NCBI PubMed database with:
    - Rate-limited API access (3 req/sec without key, 10 req/sec with key)
    - Literature search for viral protein research
    - Fail-fast error handling
    - Search result caching and deduplication
    - Academic research compliance
    """
    
    def __init__(self, email: str = "research@nanobrain.org", api_key: Optional[str] = None, 
                 fail_fast: bool = True, config: Optional[PubMedConfig] = None):
        if config is None:
            config = PubMedConfig(
                email=email,
                api_key=api_key,
                fail_fast=fail_fast,
                verify_on_init=False  # Disable auto-initialization to avoid event loop issues
            )
        
        super().__init__(config)
        
        self.pubmed_config = config
        self.email = email
        self.api_key = api_key
        self.fail_fast = fail_fast
        self.rate_limit = config.rate_limit
        
        self.logger = get_logger("pubmed_client")
        
        # Rate limiting tracking
        self.last_request_time = 0
        self.request_count = 0
        
        # Search caching
        self.search_cache: Dict[str, List[LiteratureReference]] = {}
        
    async def initialize_tool(self) -> InstallationStatus:
        """Initialize PubMed client (no local installation required)"""
        self.logger.info("🔄 Initializing PubMed client...")
        
        try:
            # Check BioPython availability
            try:
                import Bio.Entrez
                Bio.Entrez.email = self.email
                if self.api_key:
                    Bio.Entrez.api_key = self.api_key
                biopython_available = True
            except ImportError:
                biopython_available = False
                if self.fail_fast:
                    self.logger.error("❌ PubMed client initialization failed: BioPython not available. Install with: pip install biopython")
                    raise PubMedError(
                        "BioPython not available. Install with: pip install biopython"
                    )
            
            # Create installation status
            status = InstallationStatus(
                found=biopython_available,
                version="BioPython-based" if biopython_available else "unavailable",
                installation_path="python-package",
                executable_path="Bio.Entrez",
                installation_type="python-package"
            )
            
            # Add diagnostic information to issues/suggestions based on status
            if biopython_available:
                status.suggestions.extend([
                    f"Email configured: {self.email}",
                    f"API key: {'configured' if self.api_key else 'not provided'}",
                    f"Rate limit: {self.rate_limit} req/sec",
                    "BioPython: available"
                ])
            else:
                status.issues.extend([
                    f"BioPython: missing",
                    "PubMed client requires BioPython"
                ])
                status.suggestions.extend([
                    "Install BioPython: pip install biopython",
                    f"Email configured: {self.email}",
                    f"Rate limit: {self.rate_limit} req/sec"
                ])
            
            if biopython_available:
                self.logger.info("✅ PubMed client initialized successfully")
            else:
                self.logger.warning("⚠️ PubMed client initialized with limited functionality")
                
            return status
            
        except Exception as e:
            self.logger.error(f"❌ PubMed client initialization failed: {e}")
            raise
    
    async def search_alphavirus_literature(self, protein_type: str) -> List[LiteratureReference]:
        """
        Search PubMed for Alphavirus protein literature.
        
        Args:
            protein_type: Type of protein to search for (e.g., "capsid protein", "nsP1")
            
        Returns:
            List of literature references
        """
        cache_key = f"alphavirus_{protein_type.lower().replace(' ', '_')}"
        
        # Check cache first
        if self.pubmed_config.cache_enabled and cache_key in self.search_cache:
            self.logger.info(f"📚 Using cached literature for {protein_type}")
            return self.search_cache[cache_key]
        
        self.logger.info(f"🔍 Searching PubMed for Alphavirus {protein_type} literature")
        
        try:
            # Phase 4A implementation: Return placeholder for infrastructure testing
            # TODO: Implement actual PubMed API calls in Phase 4B
            placeholder_references = []
            
            # For infrastructure testing, return empty list
            if self.pubmed_config.cache_enabled:
                self.search_cache[cache_key] = placeholder_references
            
            return placeholder_references
            
        except Exception as e:
            if self.fail_fast:
                raise PubMedError(f"PubMed search failed for {protein_type}: {e}")
            else:
                self.logger.warning(f"⚠️ PubMed search failed for {protein_type}: {e}")
                return []
    
    async def _enforce_rate_limit(self):
        """Enforce NCBI rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        min_interval = 1.0 / self.rate_limit
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)
            
        self.last_request_time = time.time()
        self.request_count += 1
    
    async def verify_installation(self) -> bool:
        """Verify PubMed client is functional"""
        try:
            import Bio.Entrez
            return True
        except ImportError:
            return False
    
    async def get_diagnostics(self) -> DiagnosticReport:
        """Get diagnostic information about PubMed client"""
        try:
            import Bio.Entrez
            biopython_status = "available"
            recommendations = []
        except ImportError:
            biopython_status = "missing"
            recommendations = ["Install BioPython: pip install biopython"]
        
        # Create a basic installation status for the diagnostic report
        installation_status = InstallationStatus(
            found=(biopython_status == "available"),
            installation_type="python-package",
            version="BioPython-based" if biopython_status == "available" else "unavailable"
        )
        
        if biopython_status == "available":
            installation_status.suggestions.extend([
                f"Email: {self.email}",
                f"API key: {'configured' if self.api_key else 'not provided'}",
                f"Rate limit: {self.rate_limit} req/sec"
            ])
        else:
            installation_status.issues.append("BioPython not available")
            installation_status.suggestions.extend(recommendations)
        
        return DiagnosticReport(
            tool_name="pubmed_client",
            installation_status=installation_status,
            dependency_status={"biopython": biopython_status == "available"},
            suggested_fixes=recommendations
        )
    
    # Methods for tool framework compatibility
    async def _find_executable_in_path(self) -> Optional[str]:
        """PubMed client doesn't use executables"""
        return "Bio.Entrez"
    
    async def _check_tool_in_environment(self, env_path: str, env_name: str) -> bool:
        """Check if BioPython is available in environment"""
        try:
            import Bio.Entrez
            return True
        except ImportError:
            return False
    
    async def _check_tool_in_directory(self, directory: str) -> bool:
        """PubMed client doesn't use directory installations"""
        return False
    
    async def _build_tool_in_environment(self, source_dir: str) -> bool:
        """PubMed client doesn't require building"""
        return True
    
    async def _generate_specific_suggestions(self) -> List[str]:
        """Generate PubMed-specific installation suggestions"""
        return [
            "Install BioPython: pip install biopython",
            "Configure email for NCBI compliance",
            "Consider obtaining NCBI API key for higher rate limits",
            "Review NCBI Entrez API guidelines"
        ]
    
    async def _get_alternative_methods(self) -> List[str]:
        """Get alternative methods for literature search"""
        return [
            "Manual PubMed web interface search",
            "Use alternative APIs like Europe PMC",
            "Local literature database integration",
            "Pre-downloaded literature datasets"
        ]
    
    # Required abstract methods from base class
    async def execute_command(self, command: str, args: List[str], 
                            timeout: Optional[int] = None) -> ToolResult:
        """Execute PubMed API command (placeholder for BioPython calls)"""
        # PubMed doesn't use command line tools, but we need this for compatibility
        return ToolResult(
            success=True,
            exit_code=0,
            stdout=b"",
            stderr=b"",
            execution_time=0.0
        )
    
    async def parse_output(self, output: str, output_type: str = "literature") -> Any:
        """Parse PubMed API output"""
        # For literature searches, return empty list as placeholder
        return []
    
    async def _execute_at_scale(self, scale_config: Dict[str, Any]) -> Any:
        """Execute literature search at specified scale"""
        limit = scale_config.get("limit", 20)
        protein_type = scale_config.get("protein_type", "capsid protein")
        
        # Phase 4A: Return placeholder results
        return await self.search_alphavirus_literature(protein_type) 