"""
WebSocket Router

FastAPI WebSocket router for real-time frontend communication.
Provides real-time chat streaming, progress updates, and bidirectional communication.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.websockets import WebSocketState
from dataclasses import dataclass
from enum import Enum
import logging

# Get logger
logger = logging.getLogger("websocket_router")


class MessageType(str, Enum):
    """WebSocket message types."""
    # Client to server
    CHAT_REQUEST = "chat_request"
    PING = "ping"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    
    # Server to client
    CHAT_RESPONSE = "chat_response"
    CHAT_STREAM_CHUNK = "chat_stream_chunk"
    CHAT_STREAM_END = "chat_stream_end"
    PROGRESS_UPDATE = "progress_update"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    PONG = "pong"


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    type: MessageType
    data: Dict[str, Any]
    message_id: Optional[str] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "data": self.data,
            "message_id": self.message_id,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebSocketMessage':
        """Create from dictionary."""
        return cls(
            type=MessageType(data["type"]),
            data=data["data"],
            message_id=data.get("message_id"),
            timestamp=data.get("timestamp")
        )


class ConnectionManager:
    """Manages WebSocket connections for real-time communication."""
    
    def __init__(self):
        # Active connections
        self.connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Subscriptions for targeted messaging
        self.subscriptions: Dict[str, Set[str]] = {}  # topic -> connection_ids
        
        # Message queues for offline clients
        self.message_queues: Dict[str, List[WebSocketMessage]] = {}
        
    async def connect(self, websocket: WebSocket, connection_id: Optional[str] = None) -> str:
        """Accept new WebSocket connection."""
        await websocket.accept()
        
        if connection_id is None:
            connection_id = str(uuid.uuid4())
        
        self.connections[connection_id] = websocket
        self.connection_metadata[connection_id] = {
            "connected_at": time.time(),
            "last_ping": time.time(),
            "subscriptions": set()
        }
        
        logger.info(f"WebSocket connection established: {connection_id}")
        
        # Send connection confirmation
        welcome_message = WebSocketMessage(
            type=MessageType.STATUS_UPDATE,
            data={
                "status": "connected",
                "connection_id": connection_id,
                "server_time": time.time()
            }
        )
        await self._send_to_connection(connection_id, welcome_message)
        
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Remove WebSocket connection."""
        if connection_id in self.connections:
            # Remove from all subscriptions
            for topic, subscribers in self.subscriptions.items():
                subscribers.discard(connection_id)
            
            # Clean up empty subscriptions
            self.subscriptions = {
                topic: subscribers 
                for topic, subscribers in self.subscriptions.items() 
                if subscribers
            }
            
            # Remove connection
            del self.connections[connection_id]
            del self.connection_metadata[connection_id]
            
            logger.info(f"WebSocket connection closed: {connection_id}")
    
    async def send_to_connection(self, connection_id: str, message: WebSocketMessage) -> bool:
        """Send message to specific connection."""
        return await self._send_to_connection(connection_id, message)
    
    async def _send_to_connection(self, connection_id: str, message: WebSocketMessage) -> bool:
        """Internal method to send message to connection."""
        if connection_id not in self.connections:
            logger.warning(f"Connection not found: {connection_id}")
            return False
        
        websocket = self.connections[connection_id]
        
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps(message.to_dict()))
                return True
            else:
                logger.warning(f"Connection not in connected state: {connection_id}")
                return False
        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
            self.disconnect(connection_id)
            return False
    
    async def broadcast(self, message: WebSocketMessage, exclude: Optional[Set[str]] = None):
        """Broadcast message to all connections."""
        exclude = exclude or set()
        
        for connection_id in list(self.connections.keys()):
            if connection_id not in exclude:
                await self._send_to_connection(connection_id, message)
    
    async def send_to_topic(self, topic: str, message: WebSocketMessage):
        """Send message to all subscribers of a topic."""
        if topic in self.subscriptions:
            for connection_id in list(self.subscriptions[topic]):
                await self._send_to_connection(connection_id, message)
    
    def subscribe(self, connection_id: str, topic: str):
        """Subscribe connection to a topic."""
        if connection_id in self.connections:
            if topic not in self.subscriptions:
                self.subscriptions[topic] = set()
            
            self.subscriptions[topic].add(connection_id)
            self.connection_metadata[connection_id]["subscriptions"].add(topic)
            
            logger.info(f"Connection {connection_id} subscribed to {topic}")
    
    def unsubscribe(self, connection_id: str, topic: str):
        """Unsubscribe connection from a topic."""
        if topic in self.subscriptions:
            self.subscriptions[topic].discard(connection_id)
            
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["subscriptions"].discard(topic)
            
            logger.info(f"Connection {connection_id} unsubscribed from {topic}")
    
    def get_active_connections(self) -> List[str]:
        """Get list of active connection IDs."""
        return list(self.connections.keys())
    
    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.connections)
    
    async def ping_all_connections(self):
        """Send ping to all connections to check health."""
        ping_message = WebSocketMessage(
            type=MessageType.PONG,
            data={"server_time": time.time()}
        )
        
        for connection_id in list(self.connections.keys()):
            success = await self._send_to_connection(connection_id, ping_message)
            if success:
                self.connection_metadata[connection_id]["last_ping"] = time.time()


# Global connection manager
connection_manager = ConnectionManager()


def get_chat_workflow():
    """Dependency to get chat workflow (will be injected)."""
    return None


# Create WebSocket router
websocket_router = APIRouter(
    prefix="",
    tags=["websocket"]
)


@websocket_router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    connection_id: Optional[str] = None,
    chat_workflow = Depends(get_chat_workflow)
):
    """
    Main WebSocket endpoint for real-time communication.
    
    Supports:
    - Real-time chat with streaming responses
    - Progress updates for long-running operations
    - Bidirectional messaging
    - Topic-based subscriptions
    - Connection health monitoring
    """
    connection_id = await connection_manager.connect(websocket, connection_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                message = WebSocketMessage.from_dict(message_data)
                
                # Handle different message types
                await handle_websocket_message(connection_id, message, chat_workflow)
                
            except json.JSONDecodeError:
                error_message = WebSocketMessage(
                    type=MessageType.ERROR,
                    data={
                        "error": "invalid_json",
                        "message": "Invalid JSON format"
                    }
                )
                await connection_manager.send_to_connection(connection_id, error_message)
            
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                error_message = WebSocketMessage(
                    type=MessageType.ERROR,
                    data={
                        "error": "message_handling_error",
                        "message": str(e)
                    }
                )
                await connection_manager.send_to_connection(connection_id, error_message)
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
    finally:
        connection_manager.disconnect(connection_id)


async def handle_websocket_message(
    connection_id: str, 
    message: WebSocketMessage, 
    chat_workflow
):
    """Handle incoming WebSocket messages."""
    
    if message.type == MessageType.PING:
        # Respond to ping
        pong_message = WebSocketMessage(
            type=MessageType.PONG,
            data={"server_time": time.time()}
        )
        await connection_manager.send_to_connection(connection_id, pong_message)
    
    elif message.type == MessageType.CHAT_REQUEST:
        # Handle chat request with streaming
        await handle_chat_request(connection_id, message, chat_workflow)
    
    elif message.type == MessageType.SUBSCRIBE:
        # Subscribe to topic
        topic = message.data.get("topic")
        if topic:
            connection_manager.subscribe(connection_id, topic)
            
            status_message = WebSocketMessage(
                type=MessageType.STATUS_UPDATE,
                data={
                    "action": "subscribed",
                    "topic": topic
                }
            )
            await connection_manager.send_to_connection(connection_id, status_message)
    
    elif message.type == MessageType.UNSUBSCRIBE:
        # Unsubscribe from topic
        topic = message.data.get("topic")
        if topic:
            connection_manager.unsubscribe(connection_id, topic)
            
            status_message = WebSocketMessage(
                type=MessageType.STATUS_UPDATE,
                data={
                    "action": "unsubscribed",
                    "topic": topic
                }
            )
            await connection_manager.send_to_connection(connection_id, status_message)


async def handle_chat_request(connection_id: str, message: WebSocketMessage, chat_workflow):
    """Handle chat request with streaming support."""
    
    if not chat_workflow:
        error_message = WebSocketMessage(
            type=MessageType.ERROR,
            data={
                "error": "chat_workflow_unavailable",
                "message": "Chat workflow is not available"
            }
        )
        await connection_manager.send_to_connection(connection_id, error_message)
        return
    
    try:
        # Extract request data
        request_data = message.data
        query = request_data.get("query", "")
        options = request_data.get("options", {})
        conversation_id = request_data.get("conversation_id")
        request_id = message.message_id
        
        # Check if streaming is enabled
        enable_streaming = options.get("enable_streaming", False)
        
        if enable_streaming:
            # Stream response in chunks
            await stream_chat_response(
                connection_id, 
                query, 
                options, 
                conversation_id, 
                request_id, 
                chat_workflow
            )
        else:
            # Send complete response
            response_text = await chat_workflow.process_user_input(query)
            
            response_message = WebSocketMessage(
                type=MessageType.CHAT_RESPONSE,
                data={
                    "response": response_text,
                    "conversation_id": conversation_id,
                    "request_id": request_id,
                    "status": "success"
                }
            )
            await connection_manager.send_to_connection(connection_id, response_message)
    
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        error_message = WebSocketMessage(
            type=MessageType.ERROR,
            data={
                "error": "chat_processing_error",
                "message": str(e),
                "request_id": message.message_id
            }
        )
        await connection_manager.send_to_connection(connection_id, error_message)


async def stream_chat_response(
    connection_id: str,
    query: str,
    options: Dict[str, Any],
    conversation_id: Optional[str],
    request_id: str,
    chat_workflow
):
    """Stream chat response in chunks for real-time feedback."""
    
    try:
        # For now, simulate streaming by processing response and chunking
        # In a real implementation, this would integrate with streaming-capable LLMs
        
        response_text = await chat_workflow.process_user_input(query)
        
        # Split response into chunks (simulate streaming)
        words = response_text.split()
        chunk_size = 5  # Words per chunk
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            
            if i + chunk_size < len(words):
                chunk += " "  # Add space if not last chunk
            
            chunk_message = WebSocketMessage(
                type=MessageType.CHAT_STREAM_CHUNK,
                data={
                    "chunk": chunk,
                    "chunk_index": i // chunk_size,
                    "is_final": False,
                    "conversation_id": conversation_id,
                    "request_id": request_id
                }
            )
            
            await connection_manager.send_to_connection(connection_id, chunk_message)
            
            # Simulate streaming delay
            await asyncio.sleep(0.1)
        
        # Send end of stream
        end_message = WebSocketMessage(
            type=MessageType.CHAT_STREAM_END,
            data={
                "conversation_id": conversation_id,
                "request_id": request_id,
                "status": "completed",
                "total_chunks": (len(words) + chunk_size - 1) // chunk_size
            }
        )
        await connection_manager.send_to_connection(connection_id, end_message)
    
    except Exception as e:
        logger.error(f"Error streaming chat response: {e}")
        error_message = WebSocketMessage(
            type=MessageType.ERROR,
            data={
                "error": "streaming_error",
                "message": str(e),
                "request_id": request_id
            }
        )
        await connection_manager.send_to_connection(connection_id, error_message)


# Additional WebSocket endpoints for specific purposes

@websocket_router.websocket("/ws/workflow/{workflow_id}")
async def workflow_websocket(
    websocket: WebSocket,
    workflow_id: str,
    connection_id: Optional[str] = None
):
    """
    WebSocket endpoint for workflow-specific communication.
    
    Provides real-time updates for long-running workflows like viral protein analysis.
    """
    connection_id = await connection_manager.connect(websocket, connection_id)
    
    # Subscribe to workflow-specific topic
    workflow_topic = f"workflow_{workflow_id}"
    connection_manager.subscribe(connection_id, workflow_topic)
    
    try:
        while True:
            # Keep connection alive and handle workflow-specific messages
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                
                # Handle workflow-specific message types
                if message_data.get("type") == "workflow_command":
                    # Handle workflow control commands (pause, resume, cancel)
                    await handle_workflow_command(workflow_id, message_data, connection_id)
                
            except json.JSONDecodeError:
                pass  # Ignore invalid JSON
            except Exception as e:
                logger.error(f"Error in workflow WebSocket: {e}")
    
    except WebSocketDisconnect:
        logger.info(f"Workflow WebSocket disconnected: {connection_id}")
    finally:
        connection_manager.disconnect(connection_id)


async def handle_workflow_command(workflow_id: str, message_data: Dict[str, Any], connection_id: str):
    """Handle workflow control commands."""
    command = message_data.get("command")
    
    if command == "get_status":
        # Send workflow status
        status_message = WebSocketMessage(
            type=MessageType.STATUS_UPDATE,
            data={
                "workflow_id": workflow_id,
                "status": "running",  # This would come from actual workflow
                "progress": 45.0,
                "current_step": "data_processing"
            }
        )
        await connection_manager.send_to_connection(connection_id, status_message)


# Helper functions for external usage

async def broadcast_to_all(message_type: MessageType, data: Dict[str, Any]):
    """Broadcast message to all connected WebSocket clients."""
    message = WebSocketMessage(type=message_type, data=data)
    await connection_manager.broadcast(message)


async def send_to_topic(topic: str, message_type: MessageType, data: Dict[str, Any]):
    """Send message to all subscribers of a specific topic."""
    message = WebSocketMessage(type=message_type, data=data)
    await connection_manager.send_to_topic(topic, message)


async def send_progress_update(workflow_id: str, progress: float, message: str, step: str):
    """Send progress update for a specific workflow."""
    topic = f"workflow_{workflow_id}"
    await send_to_topic(
        topic,
        MessageType.PROGRESS_UPDATE,
        {
            "workflow_id": workflow_id,
            "progress": progress,
            "message": message,
            "current_step": step,
            "timestamp": time.time()
        }
    )


# Connection manager instance for external access
def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance."""
    return connection_manager 