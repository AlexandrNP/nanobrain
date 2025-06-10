"""
NanoBrain Library - Conversational Agents

Enhanced conversational agents with advanced capabilities like protocol support,
collaboration features, and performance tracking.

Available agents:
- EnhancedCollaborativeAgent: Multi-protocol agent with A2A and MCP support

Features:
- Agent-to-Agent (A2A) collaboration
- Model Context Protocol (MCP) tool integration
- Intelligent delegation based on configurable rules
- Performance tracking and metrics collection
- Enhanced conversation management
- Fallback mechanisms for robust operation
"""

from .enhanced_collaborative_agent import EnhancedCollaborativeAgent

__all__ = [
    'EnhancedCollaborativeAgent',
] 