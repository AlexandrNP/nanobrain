"""
NanoBrain Library - Specialized Data Infrastructure

This module provides specialized data components that extend core functionality:
- ConversationHistoryUnit: Persistent conversation storage and search
- SessionManager: Session lifecycle management  
- ExportManager: Data import/export utilities
- CacheManager: Multi-tier caching system for framework-wide use
- EmailManager: Email configuration and service management
- Chat data structures: Domain-specific data models

IMPORTANT: Core data units (DataUnitMemory, DataUnitFile, etc.) have been 
moved to nanobrain.core.data_unit to eliminate duplication.
"""

# Specialized extensions only (NOT duplicates)
from .conversation_history import ConversationHistoryUnit, ConversationMessage
from .session_manager import SessionManager  
from .export_manager import ExportManager
from .cache_manager import CacheManager
from .email_manager import EmailManager
from .chat_session_data import (
    ChatMessage, ChatSessionData, QueryClassificationData, 
    ConversationalResponseData, AnnotationJobData, MessageType, MessageRole
)

__all__ = [
    # Specialized data components (legitimate extensions)
    'ConversationHistoryUnit',
    'SessionManager', 
    'ExportManager',
    'CacheManager',
    'EmailManager',
    
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