#!/usr/bin/env python3
"""
MMD Parser Component

Parses MMD (Markdown) files from APECx and extracts content and metadata
for RAG database creation.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class PaperMetadata:
    """Metadata extracted from MMD file."""
    file_path: str
    doi: str
    title: str
    authors: List[str]
    abstract: str
    sections: Dict[str, str]
    total_length: int


class MMDParser:
    """
    Parse MMD files and extract structured content and metadata.
    
    This component processes research papers in MMD format and extracts:
    - Paper metadata (title, authors, DOI)
    - Structured content (abstract, sections)
    - Full text for chunking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("MMDParser")
        self.parsed_count = 0
        self.error_count = 0

    async def initialize(self):
        """Initialize the MMD parser."""
        self.logger.info("MMDParser initialized")
    
    def parse_mmd_file(self, file_path: Path) -> Optional[PaperMetadata]:
        """
        Parse a single MMD file and extract metadata and content.
        
        Args:
            file_path: Path to the MMD file
            
        Returns:
            PaperMetadata object or None if parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract DOI from filename
            doi = self._extract_doi_from_filename(file_path.name)
            
            # Extract title (first line starting with #)
            title = self._extract_title(content)
            
            # Extract authors (line after title)
            authors = self._extract_authors(content)
            
            # Extract abstract
            abstract = self._extract_abstract(content)
            
            # Extract sections
            sections = self._extract_sections(content)
            
            metadata = PaperMetadata(
                file_path=str(file_path),
                doi=doi,
                title=title,
                authors=authors,
                abstract=abstract,
                sections=sections,
                total_length=len(content)
            )
            
            self.parsed_count += 1
            self.logger.debug(f"Parsed {file_path.name}: {title[:50]}...")
            
            return metadata
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Failed to parse {file_path}: {e}")
            return None
    
    def _extract_doi_from_filename(self, filename: str) -> str:
        """Extract DOI from MMD filename."""
        # Remove .mmd extension and convert underscores to dots/slashes
        doi = filename.replace('.mmd', '')
        
        # Convert common patterns
        if doi.startswith('10.'):
            # Replace first underscore after 10.xxxx with /
            parts = doi.split('_', 2)
            if len(parts) >= 2:
                doi = f"{parts[0]}_{parts[1]}/{parts[2]}" if len(parts) > 2 else f"{parts[0]}_{parts[1]}"
                doi = doi.replace('_', '.')
        
        return doi
    
    def _extract_title(self, content: str) -> str:
        """Extract paper title from content."""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# ') and not line.startswith('## '):
                return line[2:].strip()
        return "Unknown Title"
    
    def _extract_authors(self, content: str) -> List[str]:
        """Extract authors from content."""
        lines = content.split('\n')
        title_found = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('# ') and not line.startswith('## '):
                title_found = True
                continue
            
            if title_found and line and not line.startswith('#'):
                # This should be the authors line
                authors = [author.strip() for author in line.split(',')]
                return authors
        
        return ["Unknown Author"]
    
    def _extract_abstract(self, content: str) -> str:
        """Extract abstract from content."""
        # Look for # Abstract section
        abstract_match = re.search(r'# Abstract\s*\n\n(.*?)(?=\n## |\n# [^A]|\Z)', content, re.DOTALL)
        if abstract_match:
            return abstract_match.group(1).strip()
        
        # Fallback: look for abstract-like content after title/authors
        lines = content.split('\n')
        in_abstract = False
        abstract_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('## ') or (line.startswith('# ') and 'abstract' not in line.lower()):
                if in_abstract:
                    break
                continue
            
            if 'abstract' in line.lower() and line.startswith('#'):
                in_abstract = True
                continue
            
            if in_abstract and line:
                abstract_lines.append(line)
        
        return ' '.join(abstract_lines) if abstract_lines else "No abstract found"
    
    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract all sections from content."""
        sections = {}
        
        # Split by headers (## or #)
        section_pattern = r'^(#{1,3})\s+(.+?)$'
        lines = content.split('\n')
        
        current_section = None
        current_content = []
        
        for line in lines:
            header_match = re.match(section_pattern, line.strip())
            
            if header_match:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = header_match.group(2).strip()
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def get_stats(self) -> Dict[str, int]:
        """Get parsing statistics."""
        return {
            "parsed_count": self.parsed_count,
            "error_count": self.error_count,
            "success_rate": self.parsed_count / max(self.parsed_count + self.error_count, 1)
        }
