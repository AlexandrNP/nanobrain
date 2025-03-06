from typing import List, Dict, Any, Optional, Union
import os
import re
from difflib import SequenceMatcher

from src.Step import Step
from src.ExecutorBase import ExecutorBase


class StepContextSearch(Step):
    """
    Tool for searching the surrounding context for relevant information.
    
    Biological analogy: Associative memory retrieval.
    Justification: Like how associative memory retrieves related information
    based on cues, this tool searches context for relevant information based
    on search terms.
    """
    def __init__(self, executor: ExecutorBase, **kwargs):
        super().__init__(executor, **kwargs)
        
        # Tool-specific attributes
        self.top_results = kwargs.get('top_results', 5)
        self.min_similarity = kwargs.get('min_similarity', 0.3)
        self.search_documentation = kwargs.get('search_documentation', True)
        self.search_workflow = kwargs.get('search_workflow', True)
        self.search_chat_history = kwargs.get('search_chat_history', True)
    
    async def process(self, inputs: List[Any]) -> Any:
        """
        Search the surrounding context for relevant information.
        
        Args:
            inputs: List containing:
                - query: Search query string
                - context_provider: Object providing context (AgentWorkflowBuilder)
                - search_areas: List of areas to search (optional)
        
        Returns:
            Dictionary with search results
        """
        # Extract inputs
        if not inputs or len(inputs) < 2:
            return {
                "success": False,
                "error": "Missing required inputs: query and context_provider are required"
            }
        
        query = inputs[0]
        context_provider = inputs[1]
        search_areas = inputs[2] if len(inputs) > 2 else ["documentation", "workflow", "chat_history"]
        
        # Validate search_areas
        valid_areas = ["documentation", "workflow", "chat_history"]
        search_areas = [area for area in search_areas if area in valid_areas]
        
        if not search_areas:
            return {
                "success": False,
                "error": f"Invalid search areas: {inputs[2]}. Must be one or more of {valid_areas}"
            }
        
        # Search each requested area
        results = {}
        
        if "documentation" in search_areas and self.search_documentation:
            results["documentation"] = self._search_documentation(query, context_provider)
        
        if "workflow" in search_areas and self.search_workflow:
            results["workflow"] = self._search_workflow(query, context_provider)
        
        if "chat_history" in search_areas and self.search_chat_history:
            results["chat_history"] = self._search_chat_history(query, context_provider)
        
        # Merge results if multiple areas were searched
        all_results = []
        for area, area_results in results.items():
            for result in area_results:
                all_results.append({
                    "area": area,
                    **result
                })
        
        # Sort by relevance
        all_results.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Limit to top results
        all_results = all_results[:self.top_results]
        
        return {
            "success": True,
            "results": all_results
        }
    
    def _search_documentation(self, query: str, context_provider) -> List[Dict[str, Any]]:
        """Search the documentation context."""
        results = []
        
        # Get documentation context
        documentation_context = getattr(context_provider, 'documentation_context', {})
        
        for key, content in documentation_context.items():
            relevance = self._calculate_relevance(query, content)
            if relevance >= self.min_similarity:
                # Extract a snippet that contains the match
                snippet = self._extract_snippet(query, content)
                
                results.append({
                    "key": key,
                    "relevance": relevance,
                    "snippet": snippet
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        return results
    
    def _search_workflow(self, query: str, context_provider) -> List[Dict[str, Any]]:
        """Search the workflow context."""
        results = []
        
        # Get workflow context
        workflow_context = getattr(context_provider, 'workflow_context', {})
        
        for file_path, content in workflow_context.items():
            relevance = self._calculate_relevance(query, content)
            if relevance >= self.min_similarity:
                # Extract a snippet that contains the match
                snippet = self._extract_snippet(query, content)
                
                results.append({
                    "file_path": file_path,
                    "relevance": relevance,
                    "snippet": snippet
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        return results
    
    def _search_chat_history(self, query: str, context_provider) -> List[Dict[str, Any]]:
        """Search the chat history."""
        results = []
        
        # Get chat history
        chat_history = getattr(context_provider, 'memory', [])
        archived_history = getattr(context_provider, 'archive_context', {})
        
        # Search chat history
        for i, message in enumerate(chat_history):
            content = message.get('content', '')
            role = message.get('role', '')
            
            relevance = self._calculate_relevance(query, content)
            if relevance >= self.min_similarity:
                # Extract a snippet that contains the match
                snippet = self._extract_snippet(query, content)
                
                results.append({
                    "index": i,
                    "role": role,
                    "relevance": relevance,
                    "snippet": snippet
                })
        
        # Search archived history
        for key, content in archived_history.items():
            relevance = self._calculate_relevance(query, content)
            if relevance >= self.min_similarity:
                # Extract a snippet that contains the match
                snippet = self._extract_snippet(query, content)
                
                results.append({
                    "archive_key": key,
                    "relevance": relevance,
                    "snippet": snippet
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        return results
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate the relevance of content to the query."""
        if not content:
            return 0.0
        
        # Convert to string if necessary
        if not isinstance(content, str):
            try:
                content = str(content)
            except:
                return 0.0
        
        # Check for exact matches
        if query.lower() in content.lower():
            return 1.0
        
        # Calculate similarity
        return SequenceMatcher(None, query.lower(), content.lower()).ratio()
    
    def _extract_snippet(self, query: str, content: str, context_chars: int = 100) -> str:
        """Extract a snippet of content that contains the query."""
        if not content:
            return ""
        
        # Convert to string if necessary
        if not isinstance(content, str):
            try:
                content = str(content)
            except:
                return ""
        
        # Find the position of the query (case insensitive)
        match = re.search(query, content, re.IGNORECASE)
        if match:
            start_pos = match.start()
            end_pos = match.end()
            
            # Get the context around the match
            start = max(0, start_pos - context_chars)
            end = min(len(content), end_pos + context_chars)
            
            # Extract the snippet
            snippet = content[start:end]
            
            # Add ellipsis if needed
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."
            
            return snippet
        
        # If no exact match, return the beginning of the content
        return content[:min(len(content), context_chars * 2)] + "..."
    
    async def search_all(self, query: str, context_provider) -> Dict[str, Any]:
        """
        Search all context areas for the query.
        
        Args:
            query: Search query string
            context_provider: Object providing context (AgentWorkflowBuilder)
        
        Returns:
            Dictionary with search results
        """
        return await self.process([query, context_provider])
    
    async def search_documentation(self, query: str, context_provider) -> Dict[str, Any]:
        """
        Search only documentation for the query.
        
        Args:
            query: Search query string
            context_provider: Object providing context (AgentWorkflowBuilder)
        
        Returns:
            Dictionary with search results
        """
        return await self.process([query, context_provider, ["documentation"]])
    
    async def search_workflow(self, query: str, context_provider) -> Dict[str, Any]:
        """
        Search only the current workflow for the query.
        
        Args:
            query: Search query string
            context_provider: Object providing context (AgentWorkflowBuilder)
        
        Returns:
            Dictionary with search results
        """
        return await self.process([query, context_provider, ["workflow"]])
    
    async def search_chat_history(self, query: str, context_provider) -> Dict[str, Any]:
        """
        Search only chat history for the query.
        
        Args:
            query: Search query string
            context_provider: Object providing context (AgentWorkflowBuilder)
        
        Returns:
            Dictionary with search results
        """
        return await self.process([query, context_provider, ["chat_history"]]) 