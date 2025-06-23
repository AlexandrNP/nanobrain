"""
NanoBrain Library - Specialized Data Infrastructure

This module provides specialized data components that extend core functionality:
- ConversationHistoryUnit: Persistent conversation storage and search
- SessionManager: Session lifecycle management  
- ExportManager: Data import/export utilities
- Chat data structures: Domain-specific data models

IMPORTANT: Core data units (DataUnitMemory, DataUnitFile, etc.) have been 
moved to nanobrain.core.data_unit to eliminate duplication.
"""

# Specialized extensions only (NOT duplicates)
from .conversation_history import ConversationHistoryUnit, ConversationMessage
from .session_manager import SessionManager  
from .export_manager import ExportManager
from .chat_session_data import (
    ChatMessage, ChatSessionData, QueryClassificationData, 
    ConversationalResponseData, AnnotationJobData, MessageType, MessageRole
)

__all__ = [
    # Specialized data components (legitimate extensions)
    'ConversationHistoryUnit',
    'SessionManager', 
    'ExportManager',
    
    # Chat data structures
    'ConversationMessage',
    'ChatMessage',
    'ChatSessionData',
    'QueryClassificationData',
    'ConversationalResponseData', 
    'AnnotationJobData',
    'MessageType',
    'MessageRole'
] 