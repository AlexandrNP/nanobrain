#!/usr/bin/env python3
"""
Viral Annotation Backend Deployment Script

A fully deployment-ready script that sets up and runs the viral annotation
pipeline as a backend service with web interface integration.

This deployment script:
1. Sets up the viral annotation pipeline as a web service
2. Integrates with NanoBrain's existing web interface
3. Provides REST API endpoints for frontend integration
4. Includes Docker deployment configuration
5. Offers both development and production deployment modes

Author: NanoBrain Development Team
Date: December 2024
Version: 4.1.0
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from demo.viral_annotation_pipeline_demo import ViralAnnotationPipeline, PipelineResult
except ImportError:
    # Fallback import
    from viral_annotation_pipeline_demo import ViralAnnotationPipeline, PipelineResult

# Optional imports - deployment script can work without full NanoBrain
try:
    from nanobrain.library.interfaces.web.web_interface import WebInterface
except ImportError:
    WebInterface = None

try:
    from nanobrain.core.config.config_manager import ConfigManager
except ImportError:
    ConfigManager = None


# API Models
class ViralAnnotationRequest(BaseModel):
    """Request model for viral annotation pipeline"""
    target_virus: str = Field(default="Alphavirus", description="Target virus family")
    input_genomes: Optional[List[str]] = Field(default=None, description="Input genome sequences")
    limit: int = Field(default=10, description="Maximum genomes to process")
    output_format: str = Field(default="json", description="Output format (json, fasta, csv)")
    include_literature: bool = Field(default=True, description="Include literature references")


class ViralAnnotationResponse(BaseModel):
    """Response model for viral annotation pipeline"""
    job_id: str
    success: bool
    execution_time: Optional[float] = None
    total_genomes: Optional[int] = None
    total_proteins: Optional[int] = None
    total_clusters: Optional[int] = None
    output_files: Optional[Dict[str, str]] = None
    viral_pssm_url: Optional[str] = None
    error_message: Optional[str] = None


class JobStatus(BaseModel):
    """Job status model"""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: int  # 0-100
    message: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[ViralAnnotationResponse] = None


class ViralAnnotationBackend:
    """
    Backend service for viral annotation pipeline with web interface integration
    """
    
    def __init__(self, output_dir: Optional[Path] = None, development_mode: bool = True):
        """Initialize the backend service"""
        
        # Setup logging
        self.logger = logging.getLogger("viral_annotation_backend")
        self.logger.setLevel(logging.INFO)
        
        # Setup directories
        self.output_dir = output_dir or Path("./viral_annotation_output")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.development_mode = development_mode
        
        # Job management
        self.active_jobs: Dict[str, JobStatus] = {}
        self.pipelines: Dict[str, ViralAnnotationPipeline] = {}
        
        # WebSocket connections for real-time updates
        self.websocket_connections: List[WebSocket] = []
        
        # Initialize FastAPI app
        self.app = self._create_fastapi_app()
        
        self.logger.info("🚀 Viral Annotation Backend initialized")
        self.logger.info(f"   Development mode: {development_mode}")
        self.logger.info(f"   Output directory: {self.output_dir}")
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create and configure FastAPI application"""
        
        app = FastAPI(
            title="NanoBrain Viral Annotation Backend",
            description="Production-ready viral protein analysis pipeline backend",
            version="4.1.0",
            docs_url="/docs" if self.development_mode else None,
            redoc_url="/redoc" if self.development_mode else None
        )
        
        # CORS middleware for frontend integration
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if self.development_mode else ["http://localhost:3000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI):
        """Add API routes to FastAPI application"""
        
        @app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "service": "NanoBrain Viral Annotation Backend",
                "version": "4.1.0",
                "status": "running",
                "endpoints": {
                    "docs": "/docs" if self.development_mode else "disabled",
                    "health": "/health",
                    "annotate": "/api/v1/annotate",
                    "status": "/api/v1/jobs/{job_id}",
                    "download": "/api/v1/download/{job_id}/{file_type}",
                    "websocket": "/ws"
                }
            }
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "active_jobs": len(self.active_jobs),
                "output_directory": str(self.output_dir),
                "development_mode": self.development_mode
            }
        
        @app.post("/api/v1/annotate", response_model=ViralAnnotationResponse)
        async def start_annotation(
            request: ViralAnnotationRequest,
            background_tasks: BackgroundTasks
        ):
            """Start viral annotation pipeline"""
            
            job_id = str(uuid.uuid4())
            
            # Create job status
            job_status = JobStatus(
                job_id=job_id,
                status="pending",
                progress=0,
                message="Job queued for execution",
                started_at=datetime.now()
            )
            
            self.active_jobs[job_id] = job_status
            
            # Start pipeline in background
            background_tasks.add_task(
                self._execute_annotation_pipeline,
                job_id,
                request
            )
            
            # Notify WebSocket clients
            await self._notify_websocket_clients({
                "type": "job_started",
                "job_id": job_id,
                "status": job_status.dict()
            })
            
            return ViralAnnotationResponse(
                job_id=job_id,
                success=True
            )
        
        @app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
        async def get_job_status(job_id: str):
            """Get job status"""
            
            if job_id not in self.active_jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            return self.active_jobs[job_id]
        
        @app.get("/api/v1/jobs")
        async def list_jobs():
            """List all jobs"""
            
            return {
                "jobs": [job.dict() for job in self.active_jobs.values()],
                "total": len(self.active_jobs)
            }
        
        @app.get("/api/v1/download/{job_id}/{file_type}")
        async def download_file(job_id: str, file_type: str):
            """Download result files"""
            
            if job_id not in self.active_jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job_status = self.active_jobs[job_id]
            
            if job_status.status != "completed" or not job_status.result:
                raise HTTPException(status_code=400, detail="Job not completed")
            
            if not job_status.result.output_files:
                raise HTTPException(status_code=404, detail="No output files available")
            
            if file_type not in job_status.result.output_files:
                raise HTTPException(status_code=404, detail=f"File type '{file_type}' not found")
            
            file_path = Path(job_status.result.output_files[file_type])
            
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="File not found on disk")
            
            return FileResponse(
                path=file_path,
                filename=f"{job_id}_{file_type}.{file_path.suffix}",
                media_type="application/octet-stream"
            )
        
        @app.delete("/api/v1/jobs/{job_id}")
        async def delete_job(job_id: str):
            """Delete job and cleanup files"""
            
            if job_id not in self.active_jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            # Stop pipeline if running
            if job_id in self.pipelines:
                pipeline = self.pipelines[job_id]
                pipeline.cleanup()
                del self.pipelines[job_id]
            
            # Remove job status
            del self.active_jobs[job_id]
            
            return {"message": f"Job {job_id} deleted successfully"}
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
    
    async def _execute_annotation_pipeline(self, job_id: str, request: ViralAnnotationRequest):
        """Execute the annotation pipeline in background"""
        
        try:
            # Update job status
            job_status = self.active_jobs[job_id]
            job_status.status = "running"
            job_status.message = "Initializing pipeline"
            
            # Create pipeline instance
            pipeline_output_dir = self.output_dir / job_id
            pipeline = ViralAnnotationPipeline(output_dir=pipeline_output_dir)
            self.pipelines[job_id] = pipeline
            
            # Initialize tools
            await pipeline.initialize_tools()
            
            # Progress callback to update job status
            async def progress_callback(message: str, percentage: int):
                job_status.progress = percentage
                job_status.message = message
                
                # Notify WebSocket clients
                await self._notify_websocket_clients({
                    "type": "job_progress",
                    "job_id": job_id,
                    "progress": percentage,
                    "message": message
                })
            
            # Execute pipeline
            result = await pipeline.execute_pipeline(
                input_genomes=request.input_genomes,
                target_virus=request.target_virus,
                limit=request.limit,
                progress_callback=progress_callback
            )
            
            # Update job status with results
            if result.success:
                job_status.status = "completed"
                job_status.progress = 100
                job_status.message = "Pipeline completed successfully"
                job_status.completed_at = datetime.now()
                
                # Create response with file URLs
                base_url = f"/api/v1/download/{job_id}"
                
                job_status.result = ViralAnnotationResponse(
                    job_id=job_id,
                    success=True,
                    execution_time=result.execution_time,
                    total_genomes=len(result.viral_genomes),
                    total_proteins=len(result.protein_annotations),
                    total_clusters=len(result.clusters),
                    output_files=result.output_files,
                    viral_pssm_url=f"{base_url}/viral_pssm_json"
                )
            else:
                job_status.status = "failed"
                job_status.message = f"Pipeline failed: {result.error_message}"
                job_status.completed_at = datetime.now()
                
                job_status.result = ViralAnnotationResponse(
                    job_id=job_id,
                    success=False,
                    error_message=result.error_message
                )
            
            # Notify WebSocket clients of completion
            await self._notify_websocket_clients({
                "type": "job_completed",
                "job_id": job_id,
                "status": job_status.dict()
            })
            
        except Exception as e:
            # Handle pipeline execution errors
            self.logger.error(f"Pipeline execution failed for job {job_id}: {e}")
            
            job_status = self.active_jobs[job_id]
            job_status.status = "failed"
            job_status.message = f"Pipeline execution error: {str(e)}"
            job_status.completed_at = datetime.now()
            
            job_status.result = ViralAnnotationResponse(
                job_id=job_id,
                success=False,
                error_message=str(e)
            )
            
            # Cleanup pipeline
            if job_id in self.pipelines:
                self.pipelines[job_id].cleanup()
                del self.pipelines[job_id]
    
    async def _notify_websocket_clients(self, message: Dict[str, Any]):
        """Notify all connected WebSocket clients"""
        
        if not self.websocket_connections:
            return
        
        message_str = json.dumps(message)
        
        # Send to all connected clients
        disconnected_clients = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message_str)
            except Exception:
                disconnected_clients.append(websocket)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.websocket_connections.remove(client)
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8001):
        """Start the backend server"""
        
        self.logger.info(f"🌐 Starting Viral Annotation Backend server")
        self.logger.info(f"   Host: {host}")
        self.logger.info(f"   Port: {port}")
        self.logger.info(f"   API Documentation: http://{host}:{port}/docs")
        
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info" if self.development_mode else "warning",
            reload=self.development_mode
        )
        
        server = uvicorn.Server(config)
        await server.serve()


def create_docker_compose_config(port: int = 8001) -> str:
    """Create Docker Compose configuration for deployment"""
    
    return f"""version: '3.8'

services:
  viral-annotation-backend:
    build:
      context: .
      dockerfile: demo/Dockerfile.viral-annotation
    ports:
      - "{port}:{port}"
    environment:
      - NANOBRAIN_ENV=production
      - VIRAL_ANNOTATION_PORT={port}
      - VIRAL_ANNOTATION_OUTPUT_DIR=/app/data/output
    volumes:
      - viral_annotation_data:/app/data
      - ./demo_output:/app/output
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  viral_annotation_data:
    driver: local
"""


def create_dockerfile() -> str:
    """Create Dockerfile for viral annotation backend"""
    
    return """FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    wget \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY nanobrain/ nanobrain/
COPY demo/ demo/
COPY setup.py .

# Install NanoBrain
RUN pip install -e .

# Create data directories
RUN mkdir -p /app/data/output /app/data/temp

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8001/health || exit 1

# Start the server
CMD ["python", "demo/deploy_viral_annotation_backend.py", "--host", "0.0.0.0", "--port", "8001", "--production"]
"""


async def main():
    """Main deployment function"""
    
    parser = argparse.ArgumentParser(description="Deploy Viral Annotation Backend")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")
    parser.add_argument("--production", action="store_true", help="Run in production mode")
    parser.add_argument("--create-docker", action="store_true", help="Create Docker configuration files")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.WARNING if args.production else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("viral_annotation_deployment")
    
    # Create Docker configuration if requested
    if args.create_docker:
        logger.info("📦 Creating Docker configuration files...")
        
        # Create docker-compose.yml
        docker_compose_path = Path("docker-compose.viral-annotation.yml")
        with open(docker_compose_path, 'w') as f:
            f.write(create_docker_compose_config(args.port))
        logger.info(f"   Created: {docker_compose_path}")
        
        # Create Dockerfile
        dockerfile_path = Path("demo/Dockerfile.viral-annotation")
        dockerfile_path.parent.mkdir(exist_ok=True)
        with open(dockerfile_path, 'w') as f:
            f.write(create_dockerfile())
        logger.info(f"   Created: {dockerfile_path}")
        
        logger.info("🐳 Docker configuration created!")
        logger.info("   To build and run:")
        logger.info(f"     docker-compose -f {docker_compose_path} up --build")
        return
    
    # Initialize backend
    development_mode = not args.production
    backend = ViralAnnotationBackend(
        output_dir=args.output_dir,
        development_mode=development_mode
    )
    
    print("🧬 NanoBrain Viral Annotation Backend")
    print("=" * 60)
    print(f"Mode: {'Development' if development_mode else 'Production'}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Output: {backend.output_dir}")
    print("=" * 60)
    
    if development_mode:
        print(f"📖 API Documentation: http://{args.host}:{args.port}/docs")
        print(f"🏥 Health Check: http://{args.host}:{args.port}/health")
        print(f"🔗 WebSocket: ws://{args.host}:{args.port}/ws")
    
    print("\n🚀 Starting server...")
    
    try:
        await backend.start_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main())) 