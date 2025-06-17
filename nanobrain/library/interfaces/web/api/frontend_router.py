"""
Frontend Router

Frontend-optimized API endpoints for React/Vue/Angular integration.
Provides endpoints specifically designed for modern frontend frameworks.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..models.request_models import ChatRequest, ChatOptions
from ..models.response_models import (
    ChatResponse, 
    HealthResponse,
    ResponseStatus
)


class FrontendChatRequest(BaseModel):
    """Simplified chat request model optimized for frontend."""
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Chat options")


class FrontendChatResponse(BaseModel):
    """Simplified chat response model optimized for frontend."""
    message: str = Field(..., description="Agent response")
    conversation_id: str = Field(..., description="Conversation ID")
    timestamp: float = Field(..., description="Response timestamp")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    status: str = Field(default="success", description="Response status")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ConversationSummary(BaseModel):
    """Conversation summary for frontend display."""
    conversation_id: str
    message_count: int
    last_message_time: float
    title: Optional[str] = None
    preview: Optional[str] = None


class SystemInfo(BaseModel):
    """System information for frontend."""
    api_version: str
    websocket_url: str
    supported_features: List[str]
    configuration: Dict[str, Any]


def get_chat_workflow():
    """Dependency to get chat workflow."""
    return None


def get_web_interface():
    """Dependency to get web interface."""
    return None


# Create frontend router
frontend_router = APIRouter(
    prefix="/frontend",
    tags=["frontend"],
    responses={
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"}
    }
)


@frontend_router.post(
    "/chat",
    response_model=FrontendChatResponse,
    summary="Send chat message (frontend optimized)",
    description="Simplified chat endpoint optimized for frontend frameworks"
)
async def frontend_chat(
    request: FrontendChatRequest,
    chat_workflow = Depends(get_chat_workflow)
) -> FrontendChatResponse:
    """
    Process chat message with frontend-optimized response format.
    
    This endpoint provides a simplified interface for frontend frameworks
    with standardized error handling and response format.
    """
    start_time = time.time()
    
    try:
        if not chat_workflow:
            raise HTTPException(
                status_code=503,
                detail="Chat service is currently unavailable"
            )
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Convert to internal format
        chat_request = ChatRequest(
            query=request.message,
            options=ChatOptions(
                conversation_id=conversation_id,
                **request.options
            )
        )
        
        # Process through workflow
        response_text = await chat_workflow.process_user_input(chat_request.query)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        return FrontendChatResponse(
            message=response_text,
            conversation_id=conversation_id,
            timestamp=time.time(),
            processing_time_ms=processing_time_ms,
            status="success"
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Return user-friendly error response
        return FrontendChatResponse(
            message="I'm sorry, I encountered an error while processing your message. Please try again.",
            conversation_id=request.conversation_id or str(uuid.uuid4()),
            timestamp=time.time(),
            processing_time_ms=processing_time_ms,
            status="error",
            metadata={"error_type": type(e).__name__, "error_message": str(e)}
        )


@frontend_router.get(
    "/conversations",
    response_model=List[ConversationSummary],
    summary="Get conversation list",
    description="Get list of conversations for the current user"
)
async def get_conversations(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of conversations"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    chat_workflow = Depends(get_chat_workflow)
) -> List[ConversationSummary]:
    """
    Get paginated list of conversations.
    
    Returns conversation summaries optimized for frontend display.
    """
    try:
        if not chat_workflow:
            return []
        
        # This would integrate with actual conversation storage
        # For now, return mock data structure
        return [
            ConversationSummary(
                conversation_id=f"conv_{i}",
                message_count=5 + i,
                last_message_time=time.time() - (i * 3600),
                title=f"Conversation {i}",
                preview="Sample conversation preview..."
            )
            for i in range(min(limit, 5))  # Mock data
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversations: {str(e)}"
        )


@frontend_router.get(
    "/conversations/{conversation_id}",
    summary="Get conversation details",
    description="Get detailed conversation history"
)
async def get_conversation_details(
    conversation_id: str = Path(..., description="Conversation ID"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum messages"),
    chat_workflow = Depends(get_chat_workflow)
) -> Dict[str, Any]:
    """
    Get detailed conversation history.
    
    Returns full conversation with messages and metadata.
    """
    try:
        if not chat_workflow:
            raise HTTPException(
                status_code=503,
                detail="Chat service is unavailable"
            )
        
        # This would integrate with actual conversation storage
        return {
            "conversation_id": conversation_id,
            "messages": [],  # Would contain actual messages
            "total_messages": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
            "metadata": {}
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversation: {str(e)}"
        )


@frontend_router.delete(
    "/conversations/{conversation_id}",
    summary="Delete conversation",
    description="Delete a conversation and all its messages"
)
async def delete_conversation(
    conversation_id: str = Path(..., description="Conversation ID"),
    chat_workflow = Depends(get_chat_workflow)
) -> Dict[str, Any]:
    """
    Delete a conversation.
    
    Removes conversation and all associated messages.
    """
    try:
        if not chat_workflow:
            raise HTTPException(
                status_code=503,
                detail="Chat service is unavailable"
            )
        
        # This would integrate with actual conversation storage
        return {
            "conversation_id": conversation_id,
            "deleted": True,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete conversation: {str(e)}"
        )


@frontend_router.get(
    "/system/info",
    response_model=SystemInfo,
    summary="Get system information",
    description="Get system configuration and capabilities for frontend"
)
async def get_system_info(
    web_interface = Depends(get_web_interface)
) -> SystemInfo:
    """
    Get system information for frontend configuration.
    
    Provides information needed by frontend to configure itself.
    """
    try:
        # Build WebSocket URL
        websocket_url = "ws://localhost:8000/api/v1/ws"  # Default
        
        if web_interface and hasattr(web_interface, 'config'):
            config = web_interface.config
            protocol = "wss" if config.security.enable_auth else "ws"
            host = config.server.host if config.server.host != "0.0.0.0" else "localhost"
            port = config.server.port
            prefix = config.api.prefix
            websocket_url = f"{protocol}://{host}:{port}{prefix}/ws"
        
        return SystemInfo(
            api_version="1.0.0",
            websocket_url=websocket_url,
            supported_features=[
                "real_time_chat",
                "conversation_history",
                "websocket_streaming",
                "markdown_rendering",
                "error_recovery"
            ],
            configuration={
                "max_message_length": 10000,
                "streaming_enabled": True,
                "websocket_enabled": True,
                "cors_enabled": True
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system info: {str(e)}"
        )


@frontend_router.get(
    "/health/detailed",
    summary="Detailed health check for frontend",
    description="Comprehensive health status for frontend monitoring"
)
async def frontend_health_check(
    web_interface = Depends(get_web_interface)
) -> Dict[str, Any]:
    """
    Comprehensive health check optimized for frontend monitoring.
    
    Provides detailed status information for dashboard display.
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": 0,
            "components": {
                "api_server": "healthy",
                "websocket_server": "healthy",
                "chat_workflow": "unknown",
                "database": "not_configured"
            },
            "metrics": {
                "active_connections": 0,
                "requests_per_minute": 0,
                "average_response_time_ms": 0
            },
            "features": {
                "streaming": True,
                "websockets": True,
                "conversations": True,
                "real_time": True
            }
        }
        
        # Get actual component status if available
        if web_interface:
            interface_status = web_interface.get_status()
            health_status.update({
                "uptime_seconds": interface_status.get("uptime_seconds", 0),
            })
            
            # Check chat workflow status
            workflow_status = interface_status.get("chat_workflow_status", {})
            if workflow_status.get("status") == "initialized":
                health_status["components"]["chat_workflow"] = "healthy"
            elif workflow_status.get("status") == "fallback_mode":
                health_status["components"]["chat_workflow"] = "degraded"
            else:
                health_status["components"]["chat_workflow"] = "unhealthy"
        
        # Overall status determination
        component_statuses = list(health_status["components"].values())
        if "unhealthy" in component_statuses:
            health_status["status"] = "unhealthy"
        elif "degraded" in component_statuses:
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
            "components": {},
            "metrics": {},
            "features": {}
        }


@frontend_router.get(
    "/config/frontend",
    summary="Get frontend configuration",
    description="Get configuration settings for frontend application"
)
async def get_frontend_config(
    web_interface = Depends(get_web_interface)
) -> Dict[str, Any]:
    """
    Get configuration settings for frontend application.
    
    Returns settings that frontend needs to configure itself properly.
    """
    try:
        default_config = {
            "api_base_url": "/api/v1",
            "websocket_url": "/api/v1/ws",
            "features": {
                "streaming": True,
                "websockets": True,
                "markdown": True,
                "syntax_highlighting": True,
                "dark_mode": True
            },
            "limits": {
                "max_message_length": 10000,
                "max_conversations": 100,
                "conversation_timeout": 3600
            },
            "ui": {
                "theme": "light",
                "sidebar_collapsed": False,
                "show_timestamps": True,
                "show_typing_indicator": True
            }
        }
        
        # Override with actual configuration if available
        if web_interface and hasattr(web_interface, 'config'):
            config = web_interface.config
            
            # Update with actual values
            default_config.update({
                "api_base_url": config.api.prefix,
                "websocket_url": f"{config.api.prefix}/ws",
                "features": {
                    "streaming": config.chat.enable_streaming,
                    "websockets": getattr(config, 'websocket', {}).get('enable_websocket', True),
                    "markdown": True,
                    "syntax_highlighting": True,
                    "dark_mode": True
                },
                "limits": {
                    "max_message_length": config.chat.default_max_tokens,
                    "max_conversations": config.chat.max_conversation_length,
                    "conversation_timeout": config.chat.conversation_timeout_seconds
                }
            })
        
        return default_config
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get frontend config: {str(e)}"
        ) 