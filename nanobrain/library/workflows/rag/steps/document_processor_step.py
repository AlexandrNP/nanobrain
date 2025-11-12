#!/usr/bin/env python3
"""
Document Processor Step for RAG systems.
Deterministic document parsing, text extraction, and chunking operations.
"""

import asyncio
import logging
import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.logging_system import get_logger

logger = get_logger(__name__)


class DocumentProcessorStep(BaseStep):
    """
    Deterministic document processing step.

    Handles file parsing, text extraction, and chunking without LLM processing.
    Uses specialized parsers and chunking algorithms for efficient processing.

    Key Features:
    - Multi-format document support (PDF, TXT, DOCX, HTML, Markdown)
    - Incremental processing with change detection
    - Batch processing with parallel execution
    - Comprehensive error handling and logging
    """

    COMPONENT_TYPE = "document_processor_step"

    def __init__(self, *args, **kwargs):
        """Prevent direct instantiation - use from_config instead"""
        raise RuntimeError(
            "Direct instantiation of DocumentProcessorStep is prohibited. "
            "ALL framework components must use DocumentProcessorStep.from_config() "
            "as per mandatory framework requirements."
        )

    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize DocumentProcessorStep from configuration."""
        # Call parent initialization
        super()._init_from_config(config, component_config, dependencies)

        # Processing configuration
        self.supported_formats = component_config.get(
            'supported_formats', ['pdf', 'txt', 'docx', 'html', 'md'])
        self.chunk_size = component_config.get('chunk_size', 1000)
        self.chunk_overlap = component_config.get('chunk_overlap', 200)
        self.enable_ocr = component_config.get('enable_ocr', False)
        self.preserve_metadata = component_config.get(
            'preserve_metadata', True)

        # File processing settings
        self.input_directory = Path(component_config.get(
            'input_directory', 'data/documents'))
        self.output_directory = Path(component_config.get(
            'output_directory', 'data/processed_chunks'))
        self.batch_size = component_config.get('batch_size', 10)
        self.parallel_processing = component_config.get(
            'parallel_processing', True)

        # Initialize parsers and chunker
        self.parsers = self._initialize_parsers()
        self.chunker = self._initialize_chunker()

        # Create output directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"📄 DocumentProcessorStep initialized for formats: {self.supported_formats}")

    def _initialize_parsers(self) -> Dict[str, Any]:
        """Initialize document parsers for supported formats."""
        parsers = {}

        # For now, use simple text extraction
        # In full implementation, would use specialized parsers
        for format_type in self.supported_formats:
            parsers[format_type] = self._create_simple_parser(format_type)

        return parsers

    def _create_simple_parser(self, format_type: str) -> Any:
        """Create simple parser for format type."""
        class SimpleParser:
            def __init__(self, format_type: str):
                self.format_type = format_type

            async def parse(self, file_path: Path) -> Dict[str, Any]:
                """Simple text extraction."""
                try:
                    if self.format_type == 'txt' or self.format_type == 'md':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    else:
                        # For other formats, read as text for now
                        # In full implementation, would use format-specific libraries
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()

                    metadata = {
                        'file_size': file_path.stat().st_size,
                        'parser_type': self.format_type,
                        'file_path': str(file_path)
                    }

                    return {
                        'text': text,
                        'metadata': metadata
                    }
                except Exception as e:
                    logger.error(f"Failed to parse {file_path}: {e}")
                    return {'text': '', 'metadata': {'error': str(e)}}

        return SimpleParser(format_type)

    def _initialize_chunker(self) -> Any:
        """Initialize text chunking strategy."""
        class SimpleChunker:
            def __init__(self, chunk_size: int, overlap: int):
                self.chunk_size = chunk_size
                self.overlap = overlap

            async def create_chunks(self, text: str) -> List[Dict[str, Any]]:
                """Create overlapping text chunks."""
                if not text.strip():
                    return []

                chunks = []
                start = 0
                chunk_index = 0

                while start < len(text):
                    end = start + self.chunk_size
                    chunk_text = text[start:end]

                    # Try to break at word boundary
                    if end < len(text) and not text[end].isspace():
                        last_space = chunk_text.rfind(' ')
                        if last_space > start + self.chunk_size // 2:
                            end = start + last_space
                            chunk_text = text[start:end]

                    chunks.append({
                        'text': chunk_text.strip(),
                        'start_char': start,
                        'end_char': end,
                        'chunk_index': chunk_index
                    })

                    # Move start position with overlap
                    start = max(start + self.chunk_size - self.overlap, end)
                    chunk_index += 1

                    if start >= len(text):
                        break

                return chunks

        return SimpleChunker(self.chunk_size, self.chunk_overlap)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process documents deterministically.

        Args:
            input_data: Contains document folder path or file list

        Returns:
            Dictionary with processed chunks and metadata
        """
        try:
            start_time = asyncio.get_event_loop().time()

            # Get documents to process
            documents = await self._get_documents_to_process(input_data)

            if not documents:
                logger.warning("⚠️ No documents found to process")
                return self._create_empty_result()

            logger.info(f"📄 Processing {len(documents)} documents")

            # Process documents in batches
            all_chunks = []
            processing_stats = {
                'total_documents': len(documents),
                'processed_documents': 0,
                'total_chunks': 0,
                'failed_documents': 0,
                'processing_time': 0
            }

            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                batch_chunks = await self._process_document_batch(batch)
                all_chunks.extend(batch_chunks)
                processing_stats['processed_documents'] += len(batch)

                logger.info(
                    f"📊 Processed batch {i//self.batch_size + 1}: {len(batch_chunks)} chunks")

            processing_stats['total_chunks'] = len(all_chunks)
            processing_stats['processing_time'] = asyncio.get_event_loop(
            ).time() - start_time

            # Save processed chunks if configured
            save_chunks = getattr(self.config, 'save_chunks', True)
            if save_chunks:
                await self._save_processed_chunks(all_chunks)

            result = {
                'chunks': all_chunks,
                'processing_stats': processing_stats,
                'chunk_metadata': self._generate_chunk_metadata(all_chunks)
            }

            logger.info(
                f"✅ Document processing completed: {len(all_chunks)} chunks in {processing_stats['processing_time']:.2f}s")

            return result

        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            return self._create_error_result(str(e))

    async def _get_documents_to_process(self, input_data: Dict[str, Any]) -> List[Path]:
        """Get list of documents to process."""
        documents = []

        # Check for explicit file list
        if 'file_paths' in input_data:
            for file_path in input_data['file_paths']:
                path = Path(file_path)
                if path.exists() and path.suffix[1:].lower() in self.supported_formats:
                    documents.append(path)

        # Check input directory
        elif self.input_directory.exists():
            for format_ext in self.supported_formats:
                documents.extend(self.input_directory.glob(f"*.{format_ext}"))
                documents.extend(
                    self.input_directory.glob(f"**/*.{format_ext}"))

        # Remove duplicates and sort
        documents = sorted(list(set(documents)))

        # Filter out already processed documents if incremental processing enabled
        incremental_processing = getattr(self.config, 'incremental_processing', True)
        if incremental_processing:
            documents = await self._filter_unprocessed_documents(documents)

        return documents

    async def _filter_unprocessed_documents(self, documents: List[Path]) -> List[Path]:
        """Filter out documents that have already been processed."""
        unprocessed = []

        for doc_path in documents:
            # Create hash of document for change detection
            doc_hash = self._calculate_document_hash(doc_path)
            hash_file = self.output_directory / \
                f"{doc_path.stem}_{doc_hash}.processed"

            if not hash_file.exists():
                unprocessed.append(doc_path)
            else:
                logger.debug(
                    f"⏭️ Skipping already processed document: {doc_path.name}")

        return unprocessed

    def _calculate_document_hash(self, doc_path: Path) -> str:
        """Calculate hash of document for change detection."""
        hasher = hashlib.md5()

        # Include file modification time and size
        stat = doc_path.stat()
        hasher.update(f"{stat.st_mtime}_{stat.st_size}".encode())

        return hasher.hexdigest()[:8]

    async def _process_document_batch(self, documents: List[Path]) -> List[Dict[str, Any]]:
        """Process a batch of documents."""
        if self.parallel_processing:
            # Process documents in parallel
            tasks = [self._process_single_document(doc) for doc in documents]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process documents sequentially
            results = []
            for doc in documents:
                result = await self._process_single_document(doc)
                results.append(result)

        # Flatten results and filter out errors
        chunks = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"❌ Document processing error: {result}")
                continue
            if isinstance(result, list):
                chunks.extend(result)

        return chunks

    async def _process_single_document(self, doc_path: Path) -> List[Dict[str, Any]]:
        """Process a single document into chunks."""
        try:
            # Get appropriate parser
            file_extension = doc_path.suffix[1:].lower()
            parser = self.parsers.get(file_extension)

            if not parser:
                raise ValueError(
                    f"No parser available for format: {file_extension}")

            # Extract text and metadata
            extracted_data = await parser.parse(doc_path)
            text_content = extracted_data['text']
            metadata = extracted_data['metadata']

            if not text_content.strip():
                logger.warning(
                    f"⚠️ No text content extracted from: {doc_path.name}")
                return []

            # Create chunks
            chunks = await self.chunker.create_chunks(text_content)

            # Add metadata to chunks
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'content': chunk['text'],
                    'metadata': {
                        'source_file': str(doc_path),
                        'file_name': doc_path.name,
                        'file_extension': file_extension,
                        'chunk_id': f"{doc_path.stem}_chunk_{i:04d}",
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk['text']),
                        'start_char': chunk.get('start_char', 0),
                        'end_char': chunk.get('end_char', len(chunk['text'])),
                        **metadata  # Include document metadata
                    }
                }
                processed_chunks.append(chunk_data)

            # Mark document as processed
            incremental_processing = getattr(self.config, 'incremental_processing', True)
            if incremental_processing:
                await self._mark_document_processed(doc_path)

            logger.debug(
                f"✅ Processed {doc_path.name}: {len(processed_chunks)} chunks")

            return processed_chunks

        except Exception as e:
            logger.error(f"❌ Failed to process {doc_path.name}: {e}")
            return []

    async def _mark_document_processed(self, doc_path: Path) -> None:
        """Mark document as processed to avoid reprocessing."""
        doc_hash = self._calculate_document_hash(doc_path)
        hash_file = self.output_directory / \
            f"{doc_path.stem}_{doc_hash}.processed"
        hash_file.touch()

    async def _save_processed_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Save processed chunks to output directory."""
        output_file = self.output_directory / "processed_chunks.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Saved {len(chunks)} chunks to {output_file}")

    def _generate_chunk_metadata(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate metadata about processed chunks."""
        if not chunks:
            return {}

        total_chars = sum(len(chunk['content']) for chunk in chunks)
        source_files = list(
            set(chunk['metadata']['source_file'] for chunk in chunks))

        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'average_chunk_size': total_chars // len(chunks) if chunks else 0,
            'source_files': source_files,
            'file_formats': list(set(chunk['metadata']['file_extension'] for chunk in chunks))
        }

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when no documents found."""
        return {
            'chunks': [],
            'processing_stats': {
                'total_documents': 0,
                'processed_documents': 0,
                'total_chunks': 0,
                'failed_documents': 0,
                'processing_time': 0
            },
            'chunk_metadata': {}
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result."""
        return {
            'chunks': [],
            'processing_stats': {
                'total_documents': 0,
                'processed_documents': 0,
                'total_chunks': 0,
                'failed_documents': 1,
                'processing_time': 0
            },
            'chunk_metadata': {},
            'error': error_message
        }
