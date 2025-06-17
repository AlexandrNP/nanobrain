"""
Standalone Web Interface for Viral Protein Analysis Workflow
Phase 3 Implementation - Standalone Interface with NanoBrain Components

Provides a standalone FastAPI interface that reuses selected NanoBrain components
for EEEV protein boundary analysis with literature support.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import yaml
from dataclasses import dataclass

# NanoBrain component imports (using relative imports to avoid dependency issues)
try:
    from nanobrain.core.logger import get_logger
    from nanobrain.core.data_unit import DataUnitManager
    from nanobrain.core.resource_monitor import ResourceMonitor, ResourceMonitorConfig
    from nanobrain.core.bioinformatics.email_manager import EmailManager
    from nanobrain.core.bioinformatics.cache_manager import CacheManager
    from nanobrain.library.workflows.viral_protein_analysis.eeev_workflow import EEEVWorkflow
    from nanobrain.library.tools.bioinformatics.pubmed_api_client import EnhancedPubMedAPIClient
except ImportError:
    # Fallback for missing dependencies in testing
    pass

@dataclass
class AnalysisProgress:
    """Progress tracking for analysis."""
    analysis_id: str
    status: str  # "started", "running", "completed", "failed", "paused"
    current_step: str
    progress_percentage: float
    message: str
    timestamp: float
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class EEEVAnalysisRequest(BaseModel):
    """Request model for EEEV analysis."""
    organism: Optional[str] = "Eastern equine encephalitis virus"
    analysis_type: str = "boundary_detection"
    enable_literature_search: bool = True
    enable_caching: bool = True
    timeout_hours: Optional[float] = None
    include_protein_types: List[str] = ["capsid", "envelope", "structural", "6K"]
    output_format: str = "viral_pssm_json"

class StandaloneConfig(BaseModel):
    """Configuration for standalone interface."""
    server_host: str = "0.0.0.0"
    server_port: int = 8001
    title: str = "EEEV Protein Boundary Analysis"
    description: str = "Standalone interface for viral protein boundary identification"
    environment: str = "production"
    
    # NanoBrain component settings
    enable_logging: bool = True
    enable_resource_monitoring: bool = True
    enable_caching: bool = True
    
    # EEEV-specific settings
    default_organism: str = "Eastern equine encephalitis virus"
    expected_proteins: List[str] = ["capsid protein", "envelope protein E1", "envelope protein E2", "6K protein"]
    genome_size_kb: float = 11.7
    
    # Environment-specific timeout settings
    timeout_config: Dict[str, Dict[str, Any]] = {
        "production": {"timeout_hours": 48, "mock_api_calls": False},
        "testing": {"timeout_seconds": 10, "mock_api_calls": True},
        "development": {"timeout_seconds": 30, "mock_api_calls": False}
    }

class WebSocketManager:
    """WebSocket manager for progress tracking."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.logger = get_logger("websocket_manager") if 'get_logger' in globals() else None
        
    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        if self.logger:
            self.logger.debug(f"WebSocket connected. Total: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if self.logger:
            self.logger.debug(f"WebSocket disconnected. Total: {len(self.active_connections)}")
        
    async def send_progress_update(self, progress: AnalysisProgress):
        """Send progress update to all connected clients."""
        message = {
            "type": "progress_update",
            "data": {
                "analysis_id": progress.analysis_id,
                "status": progress.status,
                "current_step": progress.current_step,
                "progress_percentage": progress.progress_percentage,
                "message": progress.message,
                "timestamp": progress.timestamp,
                "results": progress.results,
                "error_message": progress.error_message
            }
        }
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to send WebSocket message: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

class StandaloneViralProteinInterface:
    """
    Standalone interface reusing selected NanoBrain components.
    
    Features:
    - FastAPI-based web interface
    - WebSocket for real-time progress tracking
    - NanoBrain logging and data management integration
    - Resource monitoring and user notifications
    - EEEV-specific configuration and defaults
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize standalone interface."""
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize logging if available
        try:
            self.logger = get_logger("viral_protein_interface")
        except:
            import logging
            self.logger = logging.getLogger("viral_protein_interface")
            logging.basicConfig(level=logging.INFO)
        
        # Initialize components
        self.websocket_manager = WebSocketManager()
        self.active_analyses: Dict[str, AnalysisProgress] = {}
        
        # Initialize NanoBrain components if available
        self._initialize_nanobrain_components()
        
        self.logger.info(f"Standalone interface initialized for {self.config.environment} environment")
        
    def _load_config(self, config_path: Optional[str]) -> StandaloneConfig:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                return StandaloneConfig(**config_data.get("standalone_interface", {}))
            except Exception as e:
                print(f"Failed to load config from {config_path}: {e}")
        
        return StandaloneConfig()
    
    def _initialize_nanobrain_components(self):
        """Initialize NanoBrain components if available."""
        try:
            # Initialize data manager
            self.data_manager = DataUnitManager()
            
            # Initialize email and cache managers
            self.email_manager = EmailManager(environment=self.config.environment)
            self.cache_manager = CacheManager()
            
            # Initialize resource monitoring
            if self.config.enable_resource_monitoring:
                resource_config = ResourceMonitorConfig(
                    disk_warning_gb=1.0,
                    disk_critical_gb=0.5,
                    monitoring_interval_seconds=30.0
                )
                self.resource_monitor = ResourceMonitor(resource_config)
            else:
                self.resource_monitor = None
            
            # Initialize PubMed client
            self.pubmed_client = EnhancedPubMedAPIClient(
                self.email_manager,
                self.cache_manager
            )
            
            self.logger.info("NanoBrain components initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize some NanoBrain components: {e}")
            # Set fallback components
            self.data_manager = None
            self.email_manager = None
            self.cache_manager = None
            self.resource_monitor = None
            self.pubmed_client = None
    
    async def initialize(self) -> FastAPI:
        """Initialize and configure FastAPI application."""
        app = FastAPI(
            title=self.config.title,
            description=self.config.description,
            version="3.0.0"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        await self._setup_routes(app)
        
        # Setup WebSocket endpoint
        await self._setup_websocket(app)
        
        # Start background services
        await self._start_background_services()
        
        self.logger.info("FastAPI application initialized")
        return app
        
    async def _setup_routes(self, app: FastAPI) -> None:
        """Setup API routes."""
        
        @app.get("/")
        async def root():
            """Root endpoint with interface information."""
            return {
                "title": self.config.title,
                "description": self.config.description,
                "version": "3.0.0",
                "environment": self.config.environment,
                "endpoints": {
                    "analyze": "/api/v1/analyze",
                    "status": "/api/v1/status/{analysis_id}",
                    "results": "/api/v1/results/{analysis_id}",
                    "websocket": "/ws",
                    "health": "/health"
                }
            }
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "environment": self.config.environment,
                "components": {
                    "email_manager": "ok" if self.email_manager else "unavailable",
                    "cache_manager": "ok" if self.cache_manager else "unavailable",
                    "pubmed_client": "ok" if self.pubmed_client else "unavailable"
                }
            }
            
            # Check PubMed API access if available
            if self.pubmed_client:
                try:
                    pubmed_ok = await self.pubmed_client.verify_installation()
                    health_status["components"]["pubmed_client"] = "ok" if pubmed_ok else "error"
                except Exception:
                    health_status["components"]["pubmed_client"] = "error"
            
            # Check resource status if available
            if self.resource_monitor:
                try:
                    disk_info = await self.resource_monitor.check_disk_space()
                    health_status["resources"] = {
                        "disk_free_gb": disk_info.free_gb,
                        "disk_warning": disk_info.warning_triggered,
                        "disk_critical": disk_info.critical_triggered
                    }
                except Exception as e:
                    health_status["resources"] = {"error": str(e)}
            
            return health_status
        
        @app.post("/api/v1/analyze")
        async def analyze_eeev_proteins(
            request: EEEVAnalysisRequest,
            background_tasks: BackgroundTasks
        ):
            """Start EEEV protein analysis."""
            
            # Generate analysis ID
            analysis_id = f"eeev_{int(time.time())}_{len(self.active_analyses)}"
            
            # Set defaults
            if not request.organism:
                request.organism = self.config.default_organism
            
            # Get timeout configuration
            timeout_config = self._get_timeout_config()
            if request.timeout_hours is not None:
                timeout_config["timeout_hours"] = request.timeout_hours
            
            # Initialize progress tracking
            progress = AnalysisProgress(
                analysis_id=analysis_id,
                status="started",
                current_step="initialization",
                progress_percentage=0.0,
                message="Analysis started",
                timestamp=time.time()
            )
            
            self.active_analyses[analysis_id] = progress
            
            # Start analysis in background
            background_tasks.add_task(
                self._run_analysis,
                analysis_id,
                request,
                timeout_config
            )
            
            # Send initial progress update
            await self.websocket_manager.send_progress_update(progress)
            
            return {
                "analysis_id": analysis_id,
                "status": "started",
                "organism": request.organism,
                "expected_proteins": self.config.expected_proteins,
                "timeout_config": timeout_config,
                "websocket_url": "/ws"
            }
        
        @app.get("/api/v1/status/{analysis_id}")
        async def get_analysis_status(analysis_id: str):
            """Get analysis status."""
            if analysis_id not in self.active_analyses:
                raise HTTPException(status_code=404, detail="Analysis not found")
            
            progress = self.active_analyses[analysis_id]
            return {
                "analysis_id": analysis_id,
                "status": progress.status,
                "current_step": progress.current_step,
                "progress_percentage": progress.progress_percentage,
                "message": progress.message,
                "timestamp": progress.timestamp,
                "error_message": progress.error_message
            }
        
        @app.get("/api/v1/results/{analysis_id}")
        async def get_analysis_results(analysis_id: str):
            """Get analysis results."""
            if analysis_id not in self.active_analyses:
                raise HTTPException(status_code=404, detail="Analysis not found")
            
            progress = self.active_analyses[analysis_id]
            
            if progress.status != "completed":
                raise HTTPException(
                    status_code=400, 
                    detail=f"Analysis not completed. Status: {progress.status}"
                )
            
            return {
                "analysis_id": analysis_id,
                "status": progress.status,
                "results": progress.results,
                "timestamp": progress.timestamp
            }
    
    async def _setup_websocket(self, app: FastAPI) -> None:
        """Setup WebSocket endpoint for real-time updates."""
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive and handle client messages
                    data = await websocket.receive_text()
                    # Echo back for connection testing
                    await websocket.send_text(f"Echo: {data}")
            except Exception as e:
                self.logger.debug(f"WebSocket connection closed: {e}")
            finally:
                self.websocket_manager.disconnect(websocket)
    
    async def _start_background_services(self) -> None:
        """Start background services."""
        # Start resource monitoring if available
        if self.resource_monitor:
            try:
                await self.resource_monitor.start_monitoring()
                self.logger.info("Resource monitoring started")
            except Exception as e:
                self.logger.warning(f"Failed to start resource monitoring: {e}")
        
        # Warm cache if available
        if self.cache_manager and self.config.enable_caching:
            try:
                # Warm cache for EEEV-specific data
                await self.cache_manager.warm_eeev_cache()
                self.logger.info("EEEV cache warmed")
            except Exception as e:
                self.logger.warning(f"Failed to warm cache: {e}")
        
        self.logger.info("Background services started")
    
    async def _run_analysis(
        self,
        analysis_id: str,
        request: EEEVAnalysisRequest,
        timeout_config: Dict[str, Any]
    ) -> None:
        """Run EEEV analysis in background."""
        progress = self.active_analyses[analysis_id]
        
        try:
            # Update progress
            progress.status = "running"
            progress.current_step = "workflow_initialization"
            progress.progress_percentage = 10.0
            progress.message = "Initializing EEEV workflow"
            await self.websocket_manager.send_progress_update(progress)
            
            # Execute workflow
            results = await self._execute_mock_workflow(request, progress, timeout_config)
            
            # Complete analysis
            progress.status = "completed"
            progress.current_step = "complete"
            progress.progress_percentage = 100.0
            progress.message = "Analysis completed successfully"
            progress.results = results
            await self.websocket_manager.send_progress_update(progress)
            
            self.logger.info(f"Analysis {analysis_id} completed successfully")
            
        except Exception as e:
            # Handle analysis failure
            progress.status = "failed"
            progress.error_message = str(e)
            progress.message = f"Analysis failed: {str(e)}"
            await self.websocket_manager.send_progress_update(progress)
            
            self.logger.error(f"Analysis {analysis_id} failed: {e}")
    
    async def _execute_mock_workflow(
        self,
        request: EEEVAnalysisRequest,
        progress: AnalysisProgress,
        timeout_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute mock workflow for demonstration."""
        
        # Mock workflow steps
        steps = [
            ("literature_search", "Searching literature for boundary information", 40.0),
            ("sequence_analysis", "Analyzing protein sequences", 60.0),
            ("boundary_detection", "Detecting protein boundaries", 80.0),
            ("result_formatting", "Formatting results", 90.0)
        ]
        
        results = {
            "organism": request.organism,
            "analysis_type": request.analysis_type,
            "proteins_analyzed": request.include_protein_types,
            "boundaries_detected": [],
            "literature_references": [],
            "viral_pssm_json": {},
            "analysis_metadata": {
                "environment": self.config.environment,
                "timestamp": time.time(),
                "version": "3.0.0",
                "timeout_config": timeout_config
            }
        }
        
        for step_name, step_message, step_progress in steps:
            progress.current_step = step_name
            progress.message = step_message
            progress.progress_percentage = step_progress
            await self.websocket_manager.send_progress_update(progress)
            
            # Simulate work based on environment
            if self.config.environment == "testing":
                await asyncio.sleep(0.5)
            else:
                await asyncio.sleep(2.0)
            
            # Add mock results for each step
            if step_name == "literature_search" and request.enable_literature_search:
                results["literature_references"] = [
                    {
                        "protein_type": protein_type,
                        "pmid": f"mock_pmid_{protein_type}",
                        "title": f"Mock study on {protein_type} boundary detection",
                        "boundary_score": 0.85,
                        "extracted_boundaries": [{"position": 150, "confidence": 0.9}]
                    }
                    for protein_type in request.include_protein_types
                ]
            
            elif step_name == "boundary_detection":
                results["boundaries_detected"] = [
                    {
                        "protein_type": protein_type,
                        "start_position": 100 * (i + 1),
                        "end_position": 100 * (i + 1) + 200,
                        "confidence": 0.85 + (i * 0.05)
                    }
                    for i, protein_type in enumerate(request.include_protein_types)
                ]
        
        return results
    
    def _get_timeout_config(self) -> Dict[str, Any]:
        """Get timeout configuration for current environment."""
        return self.config.timeout_config.get(
            self.config.environment,
            self.config.timeout_config["production"]
        )
    
    async def shutdown(self) -> None:
        """Shutdown interface and cleanup resources."""
        if self.resource_monitor:
            try:
                await self.resource_monitor.stop_monitoring()
            except Exception as e:
                self.logger.warning(f"Error stopping resource monitor: {e}")
        
        self.logger.info("Standalone interface shutdown")

# FastAPI app factory
async def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create and configure FastAPI application."""
    interface = StandaloneViralProteinInterface(config_path)
    app = await interface.initialize()
    
    # Store interface reference in app state
    app.state.interface = interface
    
    return app

# For running with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    async def main():
        app = await create_app()
        config = uvicorn.Config(app, host="0.0.0.0", port=8001, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    
    asyncio.run(main()) 