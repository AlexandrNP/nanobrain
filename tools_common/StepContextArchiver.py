from typing import List, Dict, Any, Optional, Union
import os
import re
from datetime import datetime

from src.Step import Step
from src.ExecutorBase import ExecutorBase


class StepContextArchiver(Step):
    """
    Tool for summarizing and archiving important context that's being pushed out of the immediate context window.
    
    Biological analogy: Long-term memory consolidation.
    Justification: Like how the brain consolidates important information from short-term to long-term memory,
    this tool summarizes and archives important context that would otherwise be lost.
    """
    def __init__(self, executor: ExecutorBase, **kwargs):
        super().__init__(executor, **kwargs)
        
        # Tool-specific attributes
        self.max_summary_length = kwargs.get('max_summary_length', 500)
        self.min_message_length = kwargs.get('min_message_length', 100)
        self.archive_directory = kwargs.get('archive_directory', 'context_archives')
        self.max_archive_entries = kwargs.get('max_archive_entries', 50)
        self.importance_keywords = kwargs.get('importance_keywords', [
            "important", "critical", "key", "essential", "crucial", "vital",
            "significant", "fundamental", "pivotal", "central", "core"
        ])
        
        # Create archive directory if it doesn't exist
        if not os.path.exists(self.archive_directory):
            os.makedirs(self.archive_directory)
    
    async def process(self, inputs: List[Any]) -> Any:
        """
        Summarize and archive important context.
        
        Args:
            inputs: List containing:
                - messages: List of messages to archive
                - context_provider: Object providing context (AgentWorkflowBuilder)
                - archive_key: Optional key for the archive (defaults to timestamp)
        
        Returns:
            Dictionary with archive results
        """
        # Extract inputs
        if not inputs or len(inputs) < 2:
            return {
                "success": False,
                "error": "Missing required inputs: messages and context_provider are required"
            }
        
        messages = inputs[0]
        context_provider = inputs[1]
        archive_key = inputs[2] if len(inputs) > 2 else f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate messages
        if not isinstance(messages, list):
            return {
                "success": False,
                "error": "Messages must be a list"
            }
        
        if not messages:
            return {
                "success": False,
                "error": "No messages to archive"
            }
        
        # Process and archive the messages
        try:
            # Extract important information from messages
            important_points = self._extract_important_points(messages)
            
            # Generate a summary
            summary = self._generate_summary(important_points)
            
            # Save to archive
            archive_path = self._save_to_archive(archive_key, summary, important_points, messages)
            
            # Update the context provider's archive context
            if hasattr(context_provider, 'archive_context'):
                context_provider.archive_context[archive_key] = summary
            
            return {
                "success": True,
                "archive_key": archive_key,
                "summary": summary,
                "important_points": important_points,
                "archive_path": archive_path
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error archiving context: {str(e)}"
            }
    
    def _extract_important_points(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract important points from messages."""
        important_points = []
        
        for i, message in enumerate(messages):
            content = message.get('content', '')
            role = message.get('role', '')
            
            # Skip short messages
            if len(content) < self.min_message_length:
                continue
            
            # Calculate importance score
            importance_score = self._calculate_importance(content)
            
            # Extract key sentences
            key_sentences = self._extract_key_sentences(content, importance_score)
            
            if key_sentences:
                important_points.append({
                    "index": i,
                    "role": role,
                    "importance": importance_score,
                    "key_sentences": key_sentences
                })
        
        # Sort by importance
        important_points.sort(key=lambda x: x["importance"], reverse=True)
        
        return important_points[:self.max_archive_entries]
    
    def _calculate_importance(self, content: str) -> float:
        """Calculate the importance of a message."""
        if not content:
            return 0.0
        
        # Base importance on length (longer messages tend to be more important)
        importance = min(1.0, len(content) / 2000)
        
        # Check for importance keywords
        keyword_count = sum(1 for keyword in self.importance_keywords if keyword.lower() in content.lower())
        importance += keyword_count * 0.1
        
        # Check for code blocks (often important)
        code_block_count = content.count("```")
        importance += (code_block_count / 2) * 0.2
        
        # Check for lists (often important)
        list_item_count = len(re.findall(r'^\s*[-*]\s+', content, re.MULTILINE))
        importance += list_item_count * 0.05
        
        # Check for questions (often important)
        question_count = content.count("?")
        importance += question_count * 0.05
        
        return min(1.0, importance)
    
    def _extract_key_sentences(self, content: str, importance_score: float) -> List[str]:
        """Extract key sentences from content based on importance."""
        if not content:
            return []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        # Calculate the number of sentences to extract based on importance
        num_sentences = max(1, min(5, int(len(sentences) * importance_score)))
        
        # Score each sentence
        sentence_scores = []
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence) < 20:
                continue
                
            # Calculate sentence importance
            score = 0.0
            
            # Check for importance keywords
            keyword_count = sum(1 for keyword in self.importance_keywords if keyword.lower() in sentence.lower())
            score += keyword_count * 0.2
            
            # Check for code snippets
            if "```" in sentence or "`" in sentence:
                score += 0.3
            
            # Check for lists
            if re.match(r'^\s*[-*]\s+', sentence):
                score += 0.2
            
            # Check for questions
            if "?" in sentence:
                score += 0.2
            
            sentence_scores.append((sentence, score))
        
        # Sort by score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top sentences
        return [sentence for sentence, _ in sentence_scores[:num_sentences]]
    
    def _generate_summary(self, important_points: List[Dict[str, Any]]) -> str:
        """Generate a summary from important points."""
        if not important_points:
            return "No important points found."
        
        summary_parts = ["# Context Archive Summary"]
        
        # Add the most important points
        for i, point in enumerate(important_points[:5]):
            role = point.get("role", "unknown")
            key_sentences = point.get("key_sentences", [])
            
            if key_sentences:
                summary_parts.append(f"\n## Point {i+1} ({role})")
                for sentence in key_sentences[:2]:
                    summary_parts.append(f"- {sentence}")
        
        # Add a count of additional points
        if len(important_points) > 5:
            summary_parts.append(f"\n... and {len(important_points) - 5} more important points.")
        
        # Join and truncate if needed
        summary = "\n".join(summary_parts)
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length] + "..."
        
        return summary
    
    def _save_to_archive(self, archive_key: str, summary: str, important_points: List[Dict[str, Any]], messages: List[Dict[str, Any]]) -> str:
        """Save the archive to disk."""
        # Create the archive file path
        archive_path = os.path.join(self.archive_directory, f"{archive_key}.md")
        
        # Prepare the archive content
        archive_content = [
            f"# Context Archive: {archive_key}",
            f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Summary",
            summary,
            "\n## Important Points"
        ]
        
        # Add important points
        for i, point in enumerate(important_points):
            role = point.get("role", "unknown")
            importance = point.get("importance", 0.0)
            key_sentences = point.get("key_sentences", [])
            
            archive_content.append(f"\n### Point {i+1} ({role}, importance: {importance:.2f})")
            for sentence in key_sentences:
                archive_content.append(f"- {sentence}")
        
        # Add original messages
        archive_content.append("\n## Original Messages")
        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            archive_content.append(f"\n### Message {i+1} ({role})")
            archive_content.append(f"```\n{content}\n```")
        
        # Write to file
        with open(archive_path, "w") as f:
            f.write("\n".join(archive_content))
        
        return archive_path
    
    async def archive_messages(self, messages: List[Dict[str, Any]], context_provider, archive_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Archive messages.
        
        Args:
            messages: List of messages to archive
            context_provider: Object providing context (AgentWorkflowBuilder)
            archive_key: Optional key for the archive (defaults to timestamp)
        
        Returns:
            Dictionary with archive results
        """
        return await self.process([messages, context_provider, archive_key])
    
    async def archive_chat_history(self, context_provider, archive_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Archive the chat history.
        
        Args:
            context_provider: Object providing context (AgentWorkflowBuilder)
            archive_key: Optional key for the archive (defaults to timestamp)
        
        Returns:
            Dictionary with archive results
        """
        # Get chat history
        chat_history = getattr(context_provider, 'memory', [])
        
        return await self.process([chat_history, context_provider, archive_key])
    
    async def summarize_context(self, context: str, context_provider, archive_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Summarize and archive a context string.
        
        Args:
            context: Context string to summarize and archive
            context_provider: Object providing context (AgentWorkflowBuilder)
            archive_key: Optional key for the archive (defaults to timestamp)
        
        Returns:
            Dictionary with archive results
        """
        # Convert context to a message
        messages = [{
            "role": "system",
            "content": context
        }]
        
        return await self.process([messages, context_provider, archive_key]) 