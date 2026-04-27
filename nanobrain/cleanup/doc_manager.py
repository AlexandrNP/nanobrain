"""
Documentation consolidation and management system.

This module provides functionality to identify, analyze, and consolidate
overlapping and redundant documentation files throughout the repository.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import difflib

from .models import CleanupResult, ValidationResult


@dataclass
class DocumentInfo:
    """Information about a documentation file."""
    path: Path
    size: int
    content_hash: str
    title: str
    sections: List[str]
    links: List[str]
    last_modified: float


@dataclass
class DocumentCluster:
    """A cluster of related documentation files."""
    primary_doc: DocumentInfo
    related_docs: List[DocumentInfo]
    similarity_scores: Dict[str, float]
    merge_strategy: str


class DocManager:
    """Manages documentation consolidation and organization."""
    
    def __init__(self, repo_root: Path):
        """Initialize the documentation manager.
        
        Args:
            repo_root: Root directory of the repository
        """
        self.repo_root = Path(repo_root)
        self.docs_cache: Dict[str, DocumentInfo] = {}
        
        # Patterns for different types of documentation
        self.doc_patterns = {
            'readme': re.compile(r'readme\.md$', re.IGNORECASE),
            'guide': re.compile(r'(guide|tutorial|howto)\.md$', re.IGNORECASE),
            'api': re.compile(r'(api|reference)\.md$', re.IGNORECASE),
            'plan': re.compile(r'(plan|proposal|design)\.md$', re.IGNORECASE),
            'analysis': re.compile(r'(analysis|assessment|review)\.md$', re.IGNORECASE),
            'summary': re.compile(r'(summary|report|status)\.md$', re.IGNORECASE),
            'implementation': re.compile(r'(implementation|completion).*\.md$', re.IGNORECASE)
        }
        
        # Files to preserve (never consolidate)
        self.preserve_files = {
            'README.md',  # Main repository README
            'INSTALLATION_GUIDE.md',
            'CHANGELOG.md',
            'CONTRIBUTING.md',
            'LICENSE.md'
        }
        
        # Directories to prioritize for documentation
        self.priority_dirs = ['docs/', 'demos/', 'nanobrain/']
    
    def scan_documentation(self) -> Dict[str, List[DocumentInfo]]:
        """Scan repository for all documentation files.
        
        Returns:
            Dictionary mapping document types to lists of DocumentInfo objects
        """
        doc_files = {}
        
        # Find all markdown files
        for md_file in self.repo_root.rglob('*.md'):
            if self._should_skip_file(md_file):
                continue
                
            doc_info = self._analyze_document(md_file)
            if doc_info:
                doc_type = self._classify_document(doc_info)
                if doc_type not in doc_files:
                    doc_files[doc_type] = []
                doc_files[doc_type].append(doc_info)
                
        return doc_files
    
    def identify_redundant_docs(self, doc_files: Dict[str, List[DocumentInfo]]) -> List[DocumentCluster]:
        """Identify overlapping and redundant documentation.
        
        Args:
            doc_files: Dictionary of document types and their files
            
        Returns:
            List of document clusters with redundancy information
        """
        clusters = []
        
        for doc_type, docs in doc_files.items():
            if len(docs) <= 1:
                continue
                
            # Find similar documents within each type
            type_clusters = self._cluster_similar_docs(docs, doc_type)
            clusters.extend(type_clusters)
            
        return clusters
    
    def consolidate_documentation(self, clusters: List[DocumentCluster]) -> CleanupResult:
        """Consolidate redundant documentation files.
        
        Args:
            clusters: List of document clusters to consolidate
            
        Returns:
            CleanupResult with consolidation details
        """
        consolidated_files = []
        removed_files = []
        errors = []
        
        for cluster in clusters:
            try:
                result = self._consolidate_cluster(cluster)
                if result['consolidated_file']:
                    consolidated_files.append(result['consolidated_file'])
                removed_files.extend(result['removed_files'])
                
            except Exception as e:
                errors.append(f"Failed to consolidate cluster {cluster.primary_doc.path}: {e}")
        
        return CleanupResult(
            success=len(errors) == 0,
            files_removed=removed_files,
            files_processed=len(consolidated_files) + len(removed_files),
            errors=errors,
            details={
                'consolidated_files': consolidated_files,
                'removed_files': removed_files,
                'clusters_processed': len(clusters)
            }
        )
    
    def update_main_readme(self) -> CleanupResult:
        """Update the main README.md with current framework status.
        
        Returns:
            CleanupResult with update details
        """
        readme_path = self.repo_root / 'README.md'
        
        try:
            # Read current README
            if readme_path.exists():
                current_content = readme_path.read_text(encoding='utf-8')
            else:
                current_content = ""
            
            # Generate new README content
            new_content = self._generate_main_readme()
            
            # Write updated README
            readme_path.write_text(new_content, encoding='utf-8')
            
            return CleanupResult(
                success=True,
                files_processed=1,
                details={
                    'updated_file': str(readme_path),
                    'content_length': len(new_content),
                    'sections_added': self._count_sections(new_content)
                }
            )
            
        except Exception as e:
            return CleanupResult(
                success=False,
                files_processed=1,
                errors=[f"Failed to update main README: {e}"]
            )
    
    def create_demo_entry_points(self) -> CleanupResult:
        """Create clear entry point documentation for each demo.
        
        Returns:
            CleanupResult with demo documentation details
        """
        demos_dir = self.repo_root / 'demos'
        if not demos_dir.exists():
            return CleanupResult(
                success=False,
                errors=["Demos directory not found"]
            )
        
        created_files = []
        updated_files = []
        errors = []
        
        # Process each demo directory
        for demo_dir in demos_dir.iterdir():
            if not demo_dir.is_dir() or demo_dir.name.startswith('.'):
                continue
                
            try:
                result = self._create_demo_readme(demo_dir)
                if result['created']:
                    created_files.append(result['file'])
                else:
                    updated_files.append(result['file'])
                    
            except Exception as e:
                errors.append(f"Failed to create demo README for {demo_dir.name}: {e}")
        
        return CleanupResult(
            success=len(errors) == 0,
            files_processed=len(created_files) + len(updated_files),
            errors=errors,
            details={
                'created_files': created_files,
                'updated_files': updated_files
            }
        )
    
    def validate_documentation_links(self) -> ValidationResult:
        """Validate all documentation links and references.
        
        Returns:
            ValidationResult with link validation details
        """
        broken_links = []
        valid_links = []
        errors = []
        
        # Scan all markdown files for links
        for md_file in self.repo_root.rglob('*.md'):
            if self._should_skip_file(md_file):
                continue
                
            try:
                content = md_file.read_text(encoding='utf-8')
                links = self._extract_links(content)
                
                for link in links:
                    if self._is_internal_link(link):
                        if self._validate_internal_link(md_file, link):
                            valid_links.append((str(md_file), link))
                        else:
                            broken_links.append((str(md_file), link))
                            
            except Exception as e:
                errors.append(f"Failed to validate links in {md_file}: {e}")
        
        is_valid = len(broken_links) == 0 and len(errors) == 0
        
        return ValidationResult(
            component="documentation_links",
            tests_passed=[f"Valid links: {len(valid_links)}"],
            tests_failed=[f"Broken links: {len(broken_links)}"] if broken_links else [],
            success=is_valid,
            error_details=str({
                'broken_links': broken_links,
                'valid_links_count': len(valid_links),
                'files_checked': len(list(self.repo_root.rglob('*.md'))),
                'errors': errors
            }) if not is_valid else None
        )
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if a file should be skipped during processing."""
        # Skip hidden files and directories
        if any(part.startswith('.') for part in file_path.parts):
            return True
            
        # Skip test results and temporary directories
        skip_dirs = {'test-results', '__pycache__', '.pytest_cache', 'node_modules'}
        if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
            return True
            
        return False
    
    def _analyze_document(self, file_path: Path) -> Optional[DocumentInfo]:
        """Analyze a documentation file and extract metadata."""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Extract title (first heading)
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            title = title_match.group(1) if title_match else file_path.stem
            
            # Extract sections (all headings)
            sections = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
            
            # Extract links
            links = self._extract_links(content)
            
            # Calculate content hash
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            return DocumentInfo(
                path=file_path,
                size=len(content),
                content_hash=content_hash,
                title=title,
                sections=sections,
                links=links,
                last_modified=file_path.stat().st_mtime
            )
            
        except Exception:
            return None
    
    def _classify_document(self, doc_info: DocumentInfo) -> str:
        """Classify a document based on its filename and content."""
        filename = doc_info.path.name.lower()
        
        # First check filename patterns
        for doc_type, pattern in self.doc_patterns.items():
            if pattern.search(filename):
                return doc_type
                
        # Then classify by content if filename doesn't match
        title_lower = doc_info.title.lower()
        
        # Check for API first (more specific) - prioritize API over guide
        if 'api' in title_lower or 'reference' in title_lower:
            return 'api'
        elif any(word in title_lower for word in ['guide', 'tutorial', 'howto']):
            return 'guide'
        elif any(word in title_lower for word in ['plan', 'proposal', 'design']):
            return 'plan'
        elif any(word in title_lower for word in ['analysis', 'assessment', 'review']):
            return 'analysis'
        elif any(word in title_lower for word in ['summary', 'report', 'status']):
            return 'summary'
        elif any(word in title_lower for word in ['implementation', 'completion']):
            return 'implementation'
        else:
            return 'other'
    
    def _cluster_similar_docs(self, docs: List[DocumentInfo], doc_type: str) -> List[DocumentCluster]:
        """Cluster similar documents together."""
        clusters = []
        processed = set()
        
        for i, doc1 in enumerate(docs):
            if doc1.path.name in processed:
                continue
                
            cluster_docs = []
            similarity_scores = {}
            
            for j, doc2 in enumerate(docs):
                if i == j or doc2.path.name in processed:
                    continue
                    
                similarity = self._calculate_similarity(doc1, doc2)
                if similarity > 0.7:  # High similarity threshold
                    cluster_docs.append(doc2)
                    similarity_scores[str(doc2.path)] = similarity
                    processed.add(doc2.path.name)
            
            if cluster_docs:
                # Choose primary document (most recent or in priority directory)
                primary = self._choose_primary_document([doc1] + cluster_docs)
                related = [d for d in [doc1] + cluster_docs if d != primary]
                
                clusters.append(DocumentCluster(
                    primary_doc=primary,
                    related_docs=related,
                    similarity_scores=similarity_scores,
                    merge_strategy=self._determine_merge_strategy(doc_type)
                ))
                
            processed.add(doc1.path.name)
        
        return clusters
    
    def _calculate_similarity(self, doc1: DocumentInfo, doc2: DocumentInfo) -> float:
        """Calculate similarity between two documents."""
        # Title similarity
        title_similarity = difflib.SequenceMatcher(None, doc1.title, doc2.title).ratio()
        
        # Section similarity
        sections1_str = ' '.join(doc1.sections)
        sections2_str = ' '.join(doc2.sections)
        section_similarity = difflib.SequenceMatcher(None, sections1_str, sections2_str).ratio()
        
        # Content hash similarity (exact match)
        hash_similarity = 1.0 if doc1.content_hash == doc2.content_hash else 0.0
        
        # Weighted average
        return (title_similarity * 0.3 + section_similarity * 0.4 + hash_similarity * 0.3)
    
    def _choose_primary_document(self, docs: List[DocumentInfo]) -> DocumentInfo:
        """Choose the primary document from a cluster."""
        # Prefer documents in priority directories
        for priority_dir in self.priority_dirs:
            for doc in docs:
                if str(doc.path).startswith(priority_dir):
                    return doc
        
        # Prefer more recent documents
        return max(docs, key=lambda d: d.last_modified)
    
    def _determine_merge_strategy(self, doc_type: str) -> str:
        """Determine the merge strategy for a document type."""
        if doc_type in ['summary', 'analysis', 'implementation']:
            return 'archive_old'
        elif doc_type in ['guide', 'api']:
            return 'merge_content'
        else:
            return 'keep_primary'
    
    def _consolidate_cluster(self, cluster: DocumentCluster) -> Dict:
        """Consolidate a cluster of documents."""
        if cluster.merge_strategy == 'archive_old':
            return self._archive_old_documents(cluster)
        elif cluster.merge_strategy == 'merge_content':
            return self._merge_document_content(cluster)
        else:
            return self._keep_primary_only(cluster)
    
    def _archive_old_documents(self, cluster: DocumentCluster) -> Dict:
        """Archive old documents, keeping only the primary."""
        archive_dir = self.repo_root / 'docs' / 'archive'
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        removed_files = []
        
        for doc in cluster.related_docs:
            # Move to archive
            archive_path = archive_dir / doc.path.name
            doc.path.rename(archive_path)
            removed_files.append(str(doc.path))
        
        return {
            'consolidated_file': str(cluster.primary_doc.path),
            'removed_files': removed_files
        }
    
    def _merge_document_content(self, cluster: DocumentCluster) -> Dict:
        """Merge content from related documents into primary."""
        primary_content = cluster.primary_doc.path.read_text(encoding='utf-8')
        
        # Add sections from related documents
        for doc in cluster.related_docs:
            related_content = doc.path.read_text(encoding='utf-8')
            # Simple merge - append unique sections
            primary_content += f"\n\n## From {doc.path.name}\n\n{related_content}"
            
            # Remove the related document
            doc.path.unlink()
        
        # Write merged content
        cluster.primary_doc.path.write_text(primary_content, encoding='utf-8')
        
        return {
            'consolidated_file': str(cluster.primary_doc.path),
            'removed_files': [str(doc.path) for doc in cluster.related_docs]
        }
    
    def _keep_primary_only(self, cluster: DocumentCluster) -> Dict:
        """Keep only the primary document, remove others."""
        removed_files = []
        
        for doc in cluster.related_docs:
            doc.path.unlink()
            removed_files.append(str(doc.path))
        
        return {
            'consolidated_file': str(cluster.primary_doc.path),
            'removed_files': removed_files
        }
    
    def _generate_main_readme(self) -> str:
        """Generate content for the main README.md file."""
        return """# Nanobrain Framework

A research-preview event-driven AI agent framework for distributed workflows on HPC systems.

## Overview

Nanobrain is an experimental framework designed to orchestrate AI agents across distributed computing environments, with particular focus on High Performance Computing (HPC) systems. The framework provides event-driven workflow orchestration, agent coordination, and distributed execution capabilities.

## Current Status

⚠️ **Research Preview**: This framework is in active development and cleanup phase. Many components are experimental and subject to change.

### Core Components

- **Core Framework** (`nanobrain/core/`): Event-driven agent orchestration
- **Library** (`nanobrain/library/`): Reusable workflow components and tools
- **Cleanup System** (`nanobrain/cleanup/`): Repository maintenance and organization tools

### Working Demos

- **viral_pssm_workflow**: Viral protein sequence analysis pipeline
- **rag_database_creation**: RAG (Retrieval-Augmented Generation) database setup
- **academylink_aurora_demo**: Academy integration demonstration
- **simple_demo**: Basic framework usage example

## Installation

See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for detailed installation instructions.

## Quick Start

```bash
# Install the framework
pip install -e .

# Run a simple demo
python demos/simple_demo/run_demo.py
```

## Documentation

- **Framework Architecture**: `docs/01_FRAMEWORK_CORE_ARCHITECTURE.md`
- **Workflow Orchestration**: `docs/02_WORKFLOW_ORCHESTRATION.md`
- **API Reference**: `docs/API_REFERENCE.md`
- **Demo Documentation**: See individual demo directories

## Development Status

This repository is currently undergoing comprehensive cleanup and reorganization. Many temporary files, redundant documentation, and experimental code are being consolidated or removed.

### Cleanup Progress

- ✅ Backup system implementation
- ✅ Temporary file removal
- ✅ Demo consolidation
- ✅ Import resolution
- ✅ Configuration standardization
- 🔄 Documentation consolidation (in progress)
- ⏳ Dependency management
- ⏳ Repository structure reorganization

## Contributing

This is a research project. Contributions are welcome but please note the experimental nature of the codebase.

## License

[Add license information]

## Contact

[Add contact information]
"""
    
    def _count_sections(self, content: str) -> int:
        """Count the number of sections in markdown content."""
        return len(re.findall(r'^#+\s+', content, re.MULTILINE))
    
    def _create_demo_readme(self, demo_dir: Path) -> Dict:
        """Create or update README for a demo directory."""
        readme_path = demo_dir / 'README.md'
        demo_name = demo_dir.name
        
        # Check if demo has configuration files
        config_files = list(demo_dir.glob('*.yml')) + list(demo_dir.glob('*.yaml'))
        python_files = list(demo_dir.glob('*.py'))
        
        # Generate README content
        content = f"""# {demo_name.replace('_', ' ').title()} Demo

## Overview

[Brief description of what this demo demonstrates]

## Files

"""
        
        # List Python files
        if python_files:
            content += "### Python Scripts\n\n"
            for py_file in python_files:
                content += f"- `{py_file.name}`: [Description]\n"
            content += "\n"
        
        # List configuration files
        if config_files:
            content += "### Configuration Files\n\n"
            for config_file in config_files:
                content += f"- `{config_file.name}`: [Description]\n"
            content += "\n"
        
        content += """## Usage

```bash
# Navigate to demo directory
cd demos/{demo_name}

# Run the demo
python [main_script].py
```

## Requirements

- Nanobrain framework installed
- [Additional requirements]

## Configuration

[Configuration instructions]

## Expected Output

[Description of expected results]

## Troubleshooting

[Common issues and solutions]
""".format(demo_name=demo_name)
        
        created = not readme_path.exists()
        readme_path.write_text(content, encoding='utf-8')
        
        return {
            'file': str(readme_path),
            'created': created
        }
    
    def _extract_links(self, content: str) -> List[str]:
        """Extract all links from markdown content."""
        # Markdown links: [text](url)
        md_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        
        # Reference links: [text]: url
        ref_links = re.findall(r'^\[([^\]]+)\]:\s*(.+)$', content, re.MULTILINE)
        
        links = []
        for text, url in md_links + ref_links:
            links.append(url.strip())
        
        return links
    
    def _is_internal_link(self, link: str) -> bool:
        """Check if a link is internal to the repository."""
        return not (link.startswith('http://') or link.startswith('https://') or link.startswith('mailto:'))
    
    def _validate_internal_link(self, source_file: Path, link: str) -> bool:
        """Validate an internal link."""
        # Remove anchor fragments
        link_path = link.split('#')[0]
        if not link_path:
            return True  # Anchor-only links are valid
        
        # Resolve relative path
        if link_path.startswith('/'):
            target_path = self.repo_root / link_path.lstrip('/')
        else:
            target_path = source_file.parent / link_path
        
        return target_path.exists()