"""
NanoBrain Library - Workflows

Complete workflow implementations with proper step interconnections.

Available workflows:
- ChatWorkflow: Enhanced chat workflow with modular step architecture
- ParslChatWorkflow: Distributed chat workflow using Parsl executor
"""

# Import available workflows
from .chat_workflow.chat_workflow import ChatWorkflow, create_chat_workflow

# Import Parsl chat workflow
from .chat_workflow_parsl.workflow import ParslChatWorkflow, create_parsl_chat_workflow

__all__ = [
    'ChatWorkflow',
    'create_chat_workflow',
    'ParslChatWorkflow',
    'create_parsl_chat_workflow',
] 