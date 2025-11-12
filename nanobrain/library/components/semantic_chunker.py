#!/usr/bin/env python3
"""
Semantic Chunker Component

Intelligently chunks research papers while preserving semantic meaning
and context for optimal RAG performance.
"""

import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from nanobrain.library.components.mmd_parser import PaperMetadata


@dataclass
class DocumentChunk:
    """A semantic chunk of a document."""
    chunk_id: str
    paper_doi: str
    paper_title: str
    section_name: str
    content: str
    chunk_index: int
    total_chunks: int
    char_start: int
    char_end: int
    metadata: Dict[str, Any]


class SemanticChunker:
    """
    Chunk research papers semantically while preserving context.
    
    This component creates optimal chunks for RAG by:
    - Respecting section boundaries
    - Maintaining semantic coherence
    - Adding contextual metadata
    - Handling overlaps intelligently
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "chunk_size": 1000,  # Target characters per chunk
            "overlap_size": 100,  # Overlap between chunks
            "min_chunk_size": 200,  # Minimum viable chunk size
            "preserve_sections": True,  # Don't split across sections
            "include_title_context": True  # Add paper title to each chunk
        }
        
        self.config = {**default_config, **(config or {})}

        self.chunk_size = self.config["chunk_size"]
        self.overlap_size = self.config["overlap_size"]
        self.min_chunk_size = self.config["min_chunk_size"]
        self.preserve_sections = self.config["preserve_sections"]
        self.include_title_context = self.config["include_title_context"]

        self.logger = logging.getLogger("SemanticChunker")
        self.chunks_created = 0
        self.papers_processed = 0

    async def initialize(self):
        """Initialize the semantic chunker."""
        self.logger.info(f"SemanticChunker initialized: chunk_size={self.chunk_size}, overlap={self.overlap_size}")
    
    def chunk_paper(self, paper: PaperMetadata) -> List[DocumentChunk]:
        """
        Chunk a research paper into semantic chunks.
        
        Args:
            paper: PaperMetadata object from MMD parser
            
        Returns:
            List of DocumentChunk objects
        """
        try:
            chunks = []
            
            # Process abstract separately if it exists
            if paper.abstract and len(paper.abstract) > self.min_chunk_size:
                abstract_chunks = self._chunk_text(
                    text=paper.abstract,
                    section_name="Abstract",
                    paper=paper
                )
                chunks.extend(abstract_chunks)
            
            # Process each section
            for section_name, section_content in paper.sections.items():
                if len(section_content.strip()) < self.min_chunk_size:
                    continue
                
                section_chunks = self._chunk_text(
                    text=section_content,
                    section_name=section_name,
                    paper=paper
                )
                chunks.extend(section_chunks)
            
            # Update chunk indices and metadata
            for i, chunk in enumerate(chunks):
                chunk.chunk_index = i
                chunk.total_chunks = len(chunks)
                chunk.chunk_id = f"{paper.doi}_chunk_{i:03d}"
            
            self.chunks_created += len(chunks)
            self.papers_processed += 1
            
            self.logger.debug(f"Created {len(chunks)} chunks for paper: {paper.title[:50]}...")
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to chunk paper {paper.doi}: {e}")
            return []
    
    def _chunk_text(self, text: str, section_name: str, paper: PaperMetadata) -> List[DocumentChunk]:
        """Chunk a section of text semantically."""
        chunks = []
        
        # Clean the text
        text = self._clean_text(text)
        
        if len(text) <= self.chunk_size:
            # Text fits in one chunk
            chunk = self._create_chunk(
                content=text,
                section_name=section_name,
                paper=paper,
                char_start=0,
                char_end=len(text)
            )
            chunks.append(chunk)
            return chunks
        
        # Split into sentences for better semantic boundaries
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk with current content
                chunk = self._create_chunk(
                    content=current_chunk.strip(),
                    section_name=section_name,
                    paper=paper,
                    char_start=current_start,
                    char_end=current_start + len(current_chunk)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.overlap_size)
                current_chunk = overlap_text + sentence
                current_start = current_start + len(current_chunk) - len(overlap_text) - len(sentence)
            else:
                current_chunk += sentence
        
        # Add final chunk if there's remaining content
        if current_chunk.strip():
            chunk = self._create_chunk(
                content=current_chunk.strip(),
                section_name=section_name,
                paper=paper,
                char_start=current_start,
                char_end=current_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, content: str, section_name: str, paper: PaperMetadata, 
                     char_start: int, char_end: int) -> DocumentChunk:
        """Create a DocumentChunk object."""
        
        # Add title context if enabled
        if self.include_title_context and section_name.lower() != "title":
            contextual_content = f"Paper: {paper.title}\nSection: {section_name}\n\n{content}"
        else:
            contextual_content = content
        
        metadata = {
            "authors": paper.authors,
            "file_path": paper.file_path,
            "section": section_name,
            "char_count": len(content),
            "has_title_context": self.include_title_context
        }
        
        return DocumentChunk(
            chunk_id="",  # Will be set later
            paper_doi=paper.doi,
            paper_title=paper.title,
            section_name=section_name,
            content=contextual_content,
            chunk_index=0,  # Will be set later
            total_chunks=0,  # Will be set later
            char_start=char_start,
            char_end=char_end,
            metadata=metadata
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better chunking."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove figure/table references that might break context
        text = re.sub(r'\[Figure \d+\]', '', text)
        text = re.sub(r'\[Table \d+\]', '', text)
        
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for semantic boundaries."""
        # Simple sentence splitting - could be enhanced with NLTK
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() + ' ' for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of current chunk."""
        if len(text) <= overlap_size:
            return text
        
        # Try to find a good sentence boundary for overlap
        overlap_text = text[-overlap_size:]
        sentence_start = overlap_text.find('. ')
        
        if sentence_start > 0:
            return overlap_text[sentence_start + 2:]
        
        return overlap_text
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chunking statistics."""
        return {
            "papers_processed": self.papers_processed,
            "chunks_created": self.chunks_created,
            "avg_chunks_per_paper": self.chunks_created / max(self.papers_processed, 1)
        }
