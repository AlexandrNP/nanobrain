from typing import List, Dict, Any, Optional, Union
import os
import json
import requests
from datetime import datetime

from src.Step import Step
from src.ExecutorBase import ExecutorBase


class StepWebSearch(Step):
    """
    Tool for searching the web for information about APIs and best practices.
    
    Biological analogy: External information gathering.
    Justification: Like how humans seek external information to supplement
    their knowledge, this tool searches the web for additional information.
    """
    def __init__(self, executor: ExecutorBase, **kwargs):
        super().__init__(executor, **kwargs)
        
        # Tool-specific attributes
        self.search_engine = kwargs.get('search_engine', 'duckduckgo')
        self.num_results = kwargs.get('num_results', 5)
        self.timeout = kwargs.get('timeout', 10)  # seconds
        
        # API key management
        self.api_key = kwargs.get('api_key', os.environ.get(f"{self.search_engine.upper()}_API_KEY", ""))
    
    async def process(self, inputs: List[Any]) -> Any:
        """
        Search the web for information.
        
        Args:
            inputs: List containing:
                - query: Search query string
                - search_engine: Search engine to use (optional)
                - num_results: Number of results to return (optional)
        
        Returns:
            Dictionary with search results
        """
        # Extract inputs
        if not inputs or len(inputs) < 1:
            return {
                "success": False,
                "error": "Missing required input: query"
            }
        
        query = inputs[0]
        search_engine = inputs[1] if len(inputs) > 1 else self.search_engine
        num_results = inputs[2] if len(inputs) > 2 else self.num_results
        
        # Validate search_engine
        if search_engine not in ['duckduckgo', 'google', 'bing']:
            return {
                "success": False,
                "error": f"Invalid search engine: {search_engine}. Must be 'duckduckgo', 'google', or 'bing'"
            }
        
        # Get the API key for the search engine
        api_key = self.api_key
        if not api_key and search_engine != 'duckduckgo':
            return {
                "success": False,
                "error": f"No API key found for {search_engine}. Set the {search_engine.upper()}_API_KEY environment variable or pass it in the constructor."
            }
        
        try:
            # Perform the search using the appropriate method
            if search_engine == 'duckduckgo':
                results = self._search_duckduckgo(query, num_results)
            elif search_engine == 'google':
                results = self._search_google(query, num_results, api_key)
            elif search_engine == 'bing':
                results = self._search_bing(query, num_results, api_key)
            else:
                # This should never happen due to validation above
                results = []
            
            # Return the results
            return {
                "success": True,
                "query": query,
                "search_engine": search_engine,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to search the web: {e}"
            }
    
    def _search_duckduckgo(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Search using DuckDuckGo (no API key required)."""
        # Mock implementation for now
        # In a real implementation, this would use the DuckDuckGo API
        return [
            {
                "title": "Mocked DuckDuckGo Result 1",
                "link": "https://example.com/result1",
                "snippet": f"This is a mocked result for the query: {query}"
            },
            {
                "title": "Mocked DuckDuckGo Result 2",
                "link": "https://example.com/result2",
                "snippet": f"Another mocked result for the query: {query}"
            }
        ][:num_results]
    
    def _search_google(self, query: str, num_results: int, api_key: str) -> List[Dict[str, str]]:
        """Search using Google Custom Search API (requires API key)."""
        # In a real implementation, this would use the Google Custom Search API
        # For now, we'll use a simple mock implementation
        return [
            {
                "title": "Mocked Google Result 1",
                "link": "https://example.com/result1",
                "snippet": f"This is a mocked result for the query: {query}"
            },
            {
                "title": "Mocked Google Result 2",
                "link": "https://example.com/result2",
                "snippet": f"Another mocked result for the query: {query}"
            }
        ][:num_results]
    
    def _search_bing(self, query: str, num_results: int, api_key: str) -> List[Dict[str, str]]:
        """Search using Bing Search API (requires API key)."""
        # In a real implementation, this would use the Bing Search API
        # For now, we'll use a simple mock implementation
        return [
            {
                "title": "Mocked Bing Result 1",
                "link": "https://example.com/result1",
                "snippet": f"This is a mocked result for the query: {query}"
            },
            {
                "title": "Mocked Bing Result 2",
                "link": "https://example.com/result2",
                "snippet": f"Another mocked result for the query: {query}"
            }
        ][:num_results]
    
    async def search(self, query: str, search_engine: str = None, num_results: int = None) -> Dict[str, Any]:
        """
        Search the web for information.
        
        Args:
            query: Search query string
            search_engine: Search engine to use (optional)
            num_results: Number of results to return (optional)
        
        Returns:
            Dictionary with search results
        """
        inputs = [query]
        if search_engine:
            inputs.append(search_engine)
        if num_results:
            inputs.append(num_results)
        
        return await self.process(inputs)
    
    async def search_api(self, api_name: str) -> Dict[str, Any]:
        """
        Search for information about an API.
        
        Args:
            api_name: Name of the API to search for
        
        Returns:
            Dictionary with search results
        """
        query = f"{api_name} API documentation"
        return await self.process([query])
    
    async def search_best_practices(self, topic: str) -> Dict[str, Any]:
        """
        Search for best practices on a topic.
        
        Args:
            topic: Topic to search for best practices
        
        Returns:
            Dictionary with search results
        """
        query = f"{topic} best practices"
        return await self.process([query]) 