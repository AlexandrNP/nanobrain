"""
Chat Router

FastAPI router for chat-related endpoints.
"""

import time
import uuid
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse

from ..models.request_models import ChatRequest
from ..models.response_models import (
    ChatResponse, 
    ChatMetadata, 
    ErrorResponse, 
    ResponseStatus
)


# Create the router
chat_router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)


def get_chat_workflow():
    """
    Dependency to get the chat workflow instance.
    
    This will be injected by the WebInterface during startup.
    """
    # This will be replaced by the WebInterface with actual workflow
    return None


@chat_router.post(
    "/",
    response_model=ChatResponse,
    summary="Process a chat request",
    description="Send a message to the chat agent and receive a response"
)
async def process_chat(
    request: ChatRequest,
    chat_workflow = Depends(get_chat_workflow)
) -> ChatResponse:
    """
    Process a chat request through the NanoBrain workflow.
    
    Args:
        request: The chat request containing query and options
        chat_workflow: The chat workflow instance (injected)
        
    Returns:
        ChatResponse: The processed response with metadata
        
    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    
    try:
        if chat_workflow is None:
            raise HTTPException(
                status_code=500,
                detail="Chat workflow is not available"
            )
        
        # Generate conversation ID if not provided
        conversation_id = request.options.conversation_id or str(uuid.uuid4())
        
        # Process the request through the workflow
        response_text = await chat_workflow.process_user_input(request.query)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Get workflow status for additional metadata
        workflow_status = chat_workflow.get_workflow_status()
        
        # Create response metadata
        metadata = ChatMetadata(
            processing_time_ms=processing_time_ms,
            conversation_id=conversation_id,
            rag_enabled=request.options.use_rag,
            model_used=request.options.model,
            custom_metadata={
                "request_options": request.options.dict(),
                "workflow_status": workflow_status
            }
        )
        
        # Create and return response
        return ChatResponse(
            response=response_text,
            status=ResponseStatus.SUCCESS,
            conversation_id=conversation_id,
            request_id=request.request_id,
            metadata=metadata
        )
        
    except Exception as e:
        # Calculate processing time even for errors
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Log the error (this will be handled by middleware)
        error_message = f"Error processing chat request: {str(e)}"
        
        # Create error metadata
        metadata = ChatMetadata(
            processing_time_ms=processing_time_ms,
            conversation_id=request.options.conversation_id or str(uuid.uuid4()),
            rag_enabled=request.options.use_rag,
            custom_metadata={
                "error": str(e),
                "request_options": request.options.dict()
            }
        )
        
        # Return error response
        return ChatResponse(
            response=f"I apologize, but I encountered an error while processing your request: {str(e)}",
            status=ResponseStatus.ERROR,
            conversation_id=request.options.conversation_id or str(uuid.uuid4()),
            request_id=request.request_id,
            metadata=metadata,
            error_message=error_message
        )


@chat_router.get(
    "/conversations/{conversation_id}",
    summary="Get conversation history",
    description="Retrieve the history of a conversation"
)
async def get_conversation_history(
    conversation_id: str,
    limit: int = 50,
    chat_workflow = Depends(get_chat_workflow)
) -> Dict[str, Any]:
    """
    Get conversation history.
    
    Args:
        conversation_id: ID of the conversation
        limit: Maximum number of messages to return
        chat_workflow: The chat workflow instance (injected)
        
    Returns:
        Dict containing conversation history
        
    Raises:
        HTTPException: If conversation not found or error occurs
    """
    try:
        if chat_workflow is None:
            raise HTTPException(
                status_code=500,
                detail="Chat workflow is not available"
            )
        
        # Get conversation statistics
        conversation_stats = await chat_workflow.get_conversation_stats()
        
        return {
            "conversation_id": conversation_id,
            "statistics": conversation_stats,
            "limit": limit,
            "message": "Conversation history endpoint - implementation depends on workflow setup"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving conversation history: {str(e)}"
        )


@chat_router.delete(
    "/conversations/{conversation_id}",
    summary="Delete conversation",
    description="Delete a conversation and its history"
)
async def delete_conversation(
    conversation_id: str,
    chat_workflow = Depends(get_chat_workflow)
) -> Dict[str, Any]:
    """
    Delete a conversation.
    
    Args:
        conversation_id: ID of the conversation to delete
        chat_workflow: The chat workflow instance (injected)
        
    Returns:
        Dict confirming deletion
        
    Raises:
        HTTPException: If conversation not found or error occurs
    """
    try:
        if chat_workflow is None:
            raise HTTPException(
                status_code=500,
                detail="Chat workflow is not available"
            )
        
        # This would need to be implemented based on the workflow's capabilities
        return {
            "conversation_id": conversation_id,
            "status": "deleted",
            "message": "Conversation deletion endpoint - implementation depends on workflow setup"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting conversation: {str(e)}"
        ) 