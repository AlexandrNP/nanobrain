"""
NanoBrain Library - Interfaces

User interfaces and external system integrations for the NanoBrain framework.

This module provides:
- Web interfaces (REST API, WebSocket)
- CLI interfaces and command processors
- External system integrations
- Protocol adapters and connectors

Separation of Concerns:
- Web interfaces provide HTTP/REST API access to workflows
- CLI interfaces provide command-line access
- Each interface type follows common patterns for consistency
- Interfaces are decoupled from core workflow logic
"""

from .web import WebInterface, WebInterfaceConfig

__all__ = [
    'WebInterface',
    'WebInterfaceConfig'
] 