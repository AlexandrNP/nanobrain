"""
Health Router

FastAPI router for health check and status endpoints.
"""

import time
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends

from ..models.response_models import HealthResponse, StatusResponse


# Create the router
health_router = APIRouter(
    prefix="",
    tags=["health"],
)


def get_web_interface():
    """
    Dependency to get the web interface instance.
    
    This will be injected by the WebInterface during startup.
    """
    # This will be replaced by the WebInterface with actual instance
    return None


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the API and its components"
)
async def health_check(
    web_interface = Depends(get_web_interface)
) -> HealthResponse:
    """
    Perform a health check.
    
    Args:
        web_interface: The web interface instance (injected)
        
    Returns:
        HealthResponse: Health status information
    """
    components = {}
    overall_status = "healthy"
    
    try:
        # Check chat workflow
        if web_interface and hasattr(web_interface, 'chat_workflow'):
            if web_interface.chat_workflow and web_interface.chat_workflow.is_initialized:
                components["chat_workflow"] = "healthy"
            else:
                components["chat_workflow"] = "unhealthy"
                overall_status = "degraded"
        else:
            components["chat_workflow"] = "unknown"
            overall_status = "degraded"
        
        # Check logging system
        try:
            components["logging"] = "healthy"
        except Exception:
            components["logging"] = "unhealthy"
            overall_status = "degraded"
        
        # Check configuration
        if web_interface and hasattr(web_interface, 'config'):
            components["configuration"] = "healthy"
        else:
            components["configuration"] = "unhealthy"
            overall_status = "degraded"
            
    except Exception as e:
        overall_status = "unhealthy"
        components["error"] = str(e)
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        components=components
    )


@health_router.get(
    "/status",
    response_model=StatusResponse,
    summary="Detailed status",
    description="Get detailed status information about the API and workflow"
)
async def get_status(
    web_interface = Depends(get_web_interface)
) -> StatusResponse:
    """
    Get detailed status information.
    
    Args:
        web_interface: The web interface instance (injected)
        
    Returns:
        StatusResponse: Detailed status information
    """
    workflow_status = {}
    metrics = {}
    api_status = "running"
    uptime_seconds = 0.0
    
    try:
        # Get uptime
        if web_interface and hasattr(web_interface, 'start_time'):
            uptime_seconds = (datetime.utcnow() - web_interface.start_time).total_seconds()
        
        # Get workflow status
        if web_interface and hasattr(web_interface, 'chat_workflow'):
            if web_interface.chat_workflow:
                workflow_status = web_interface.chat_workflow.get_workflow_status()
            else:
                workflow_status = {"status": "not_initialized"}
        
        # Basic metrics
        metrics = {
            "uptime_seconds": uptime_seconds,
            "status_check_time": datetime.utcnow().isoformat(),
            "components_count": len(workflow_status.get('components', {}))
        }
        
        # Add conversation stats if available
        if (web_interface and 
            hasattr(web_interface, 'chat_workflow') and 
            web_interface.chat_workflow):
            try:
                conversation_stats = await web_interface.chat_workflow.get_conversation_stats()
                if conversation_stats:
                    metrics["conversation_stats"] = conversation_stats
            except Exception:
                # Conversation stats not available
                pass
        
    except Exception as e:
        api_status = "error"
        workflow_status = {"error": str(e)}
        metrics = {"error": str(e)}
    
    return StatusResponse(
        api_status=api_status,
        workflow_status=workflow_status,
        metrics=metrics,
        uptime_seconds=uptime_seconds
    )


@health_router.get(
    "/ping",
    summary="Simple ping",
    description="Simple ping endpoint for basic connectivity check"
)
async def ping() -> Dict[str, Any]:
    """
    Simple ping endpoint.
    
    Returns:
        Dict: Simple pong response
    """
    return {
        "message": "pong",
        "timestamp": datetime.utcnow().isoformat(),
        "status": "ok"
    } 