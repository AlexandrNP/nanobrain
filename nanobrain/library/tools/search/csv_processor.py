"""
CSV Processing Utilities for NanoBrain Framework

This module provides utilities for processing CSV files including format detection,
data validation, and transformation for Elasticsearch indexing.
"""

import csv
import os
from typing import Dict, List, Any, Optional, Iterator
import logging

# Optional imports with fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class CSVProcessor:
    """
    CSV processing utility class following NanoBrain patterns.
    Provides format detection, validation, and data transformation capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CSV processor with configuration"""
        self.config = config or {}
        
        # Default configuration
        self.max_file_size_mb = self.config.get('max_file_size_mb', 500)
        self.max_rows_per_batch = self.config.get('max_rows_per_batch', 10000)
        self.supported_delimiters = self.config.get('supported_delimiters', [',', ';', '\t', '|', '^'])
        self.supported_encodings = self.config.get('supported_encodings', ['utf-8', 'latin-1', 'cp1252', 'utf-16'])
        self.auto_detect_delimiter = self.config.get('auto_detect_delimiter', True)
        self.auto_detect_encoding = self.config.get('auto_detect_encoding', True)
        self.chunk_size = self.config.get('chunk_size', 1000)
        
        logger.info("✅ CSV Processor initialized")
    
    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using chardet if available"""
        try:
            if not CHARDET_AVAILABLE:
                logger.warning("chardet not available, using utf-8 default")
                return 'utf-8'
            
            with open(file_path, 'rb') as f:
                sample = f.read(10240)  # Read first 10KB
            
            detected = chardet.detect(sample)
            confidence = detected.get('confidence', 0)
            encoding = detected.get('encoding', 'utf-8')
            
            if confidence > 0.7:
                logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                return encoding
            else:
                logger.warning(f"Low confidence encoding detection: {encoding} ({confidence:.2f}), using utf-8")
                return 'utf-8'
                
        except Exception as e:
            logger.error(f"Encoding detection failed: {e}, using utf-8")
            return 'utf-8'
    
    def detect_delimiter(self, file_path: str, encoding: str = 'utf-8') -> str:
        """Detect CSV delimiter by analyzing sample data"""
        try:
            if not self.auto_detect_delimiter:
                return ','
            
            with open(file_path, 'r', encoding=encoding) as f:
                sample_lines = [f.readline() for _ in range(5)]
                sample_lines = [line.strip() for line in sample_lines if line.strip()]
            
            if not sample_lines:
                return ','
            
            # Count occurrences of each delimiter
            delimiter_scores = {}
            for delimiter in self.supported_delimiters:
                scores = []
                for line in sample_lines:
                    count = line.count(delimiter)
                    scores.append(count)
                
                # Check consistency (similar counts across lines)
                if scores and max(scores) > 0:
                    avg_score = sum(scores) / len(scores)
                    consistency = 1 - (max(scores) - min(scores)) / (max(scores) + 1)
                    delimiter_scores[delimiter] = avg_score * consistency
            
            if delimiter_scores:
                best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
                logger.info(f"Detected delimiter: '{best_delimiter}'")
                return best_delimiter
            
            return ','
            
        except Exception as e:
            logger.error(f"Delimiter detection failed: {e}, using comma")
            return ','
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate CSV file basic properties"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_size_mb': 0,
            'readable': False
        }
        
        try:
            # Check file existence
            if not os.path.exists(file_path):
                validation_result['errors'].append('File does not exist')
                validation_result['valid'] = False
                return validation_result
            
            # Check file size
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            validation_result['file_size_mb'] = file_size_mb
            
            if file_size_mb > self.max_file_size_mb:
                validation_result['errors'].append(f'File size {file_size_mb:.1f}MB exceeds limit {self.max_file_size_mb}MB')
                validation_result['valid'] = False
            
            # Check readability
            try:
                with open(file_path, 'r') as f:
                    f.read(1)
                validation_result['readable'] = True
            except Exception as e:
                validation_result['errors'].append(f'File not readable: {e}')
                validation_result['valid'] = False
            
            return validation_result
            
        except Exception as e:
            validation_result['errors'].append(f'Validation failed: {e}')
            validation_result['valid'] = False
            return validation_result
    
    def analyze_structure(self, file_path: str) -> Dict[str, Any]:
        """Analyze CSV file structure and metadata"""
        try:
            # Validate file first
            validation = self.validate_file(file_path)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': 'File validation failed',
                    'validation': validation
                }
            
            # Detect encoding and delimiter
            encoding = self.detect_encoding(file_path)
            delimiter = self.detect_delimiter(file_path, encoding)
            
            # Analyze structure
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.reader(f, delimiter=delimiter)
                
                # Get headers
                headers = next(reader, [])
                if not headers:
                    return {
                        'success': False,
                        'error': 'No headers found in CSV file'
                    }
                
                # Sample data rows for type detection
                sample_rows = []
                row_count = 0
                for row in reader:
                    sample_rows.append(row)
                    row_count += 1
                    if len(sample_rows) >= 10:  # Sample first 10 rows
                        break
                
                # Continue counting remaining rows for total estimate
                for row in reader:
                    row_count += 1
                    if row_count >= 10000:  # Limit counting for very large files
                        break
            
            # Detect column types
            column_types = {}
            for i, header in enumerate(headers):
                column_values = [row[i] if i < len(row) else '' for row in sample_rows]
                column_types[header] = self._detect_column_type(column_values)
            
            structure = {
                'encoding': encoding,
                'delimiter': delimiter,
                'headers': headers,
                'column_count': len(headers),
                'estimated_rows': row_count,
                'column_types': column_types,
                'sample_data': sample_rows[:5]  # First 5 rows for preview
            }
            
            logger.info(f"✅ CSV structure analyzed: {len(headers)} columns, ~{row_count} rows")
            
            return {
                'success': True,
                'structure': structure,
                'validation': validation
            }
            
        except Exception as e:
            logger.error(f"❌ Structure analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _detect_column_type(self, values: List[str]) -> str:
        """Detect column data type based on sample values"""
        if not values:
            return 'text'
        
        # Remove empty values
        non_empty = [v.strip() for v in values if v.strip()]
        if not non_empty:
            return 'text'
        
        # Type detection counters
        integer_count = 0
        float_count = 0
        boolean_count = 0
        date_count = 0
        
        for value in non_empty:
            # Check boolean
            if value.lower() in ['true', 'false', 'yes', 'no', '1', '0']:
                boolean_count += 1
                continue
            
            # Check integer
            try:
                int(value)
                integer_count += 1
                continue
            except ValueError:
                pass
            
            # Check float
            try:
                float(value)
                float_count += 1
                continue
            except ValueError:
                pass
            
            # Check date patterns
            if self._looks_like_date(value):
                date_count += 1
        
        total = len(non_empty)
        
        # Determine type by majority (80% threshold)
        if boolean_count / total > 0.8:
            return 'boolean'
        elif integer_count / total > 0.8:
            return 'integer'
        elif (integer_count + float_count) / total > 0.8:
            return 'float'
        elif date_count / total > 0.6:
            return 'date'
        else:
            return 'text'
    
    def _looks_like_date(self, value: str) -> bool:
        """Check if value looks like a date"""
        # Simple date pattern detection
        date_separators = ['-', '/', '.', ' ']
        
        for sep in date_separators:
            if sep in value:
                parts = value.split(sep)
                if len(parts) >= 2:
                    # Check if parts look like date components
                    numeric_parts = 0
                    for part in parts:
                        if part.isdigit() and 1 <= len(part) <= 4:
                            numeric_parts += 1
                    
                    if numeric_parts >= 2:
                        return True
        
        return False
    
    def read_in_chunks(self, file_path: str, chunk_size: Optional[int] = None) -> Iterator[List[Dict[str, Any]]]:
        """Read CSV file in chunks for memory-efficient processing"""
        chunk_size = chunk_size or self.chunk_size
        
        # Get file structure
        structure_result = self.analyze_structure(file_path)
        if not structure_result['success']:
            raise ValueError(f"Failed to analyze CSV structure: {structure_result['error']}")
        
        structure = structure_result['structure']
        
        if PANDAS_AVAILABLE:
            # Use pandas for efficient chunked reading
            try:
                for chunk_df in pd.read_csv(
                    file_path,
                    chunksize=chunk_size,
                    delimiter=structure['delimiter'],
                    encoding=structure['encoding']
                ):
                    yield chunk_df.to_dict('records')
            except Exception as e:
                logger.error(f"Pandas chunked reading failed: {e}, falling back to manual processing")
                # Fall through to manual processing
        
        # Manual chunked reading
        with open(file_path, 'r', encoding=structure['encoding']) as f:
            reader = csv.DictReader(f, delimiter=structure['delimiter'])
            chunk = []
            
            for row in reader:
                chunk.append(row)
                
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
            
            # Yield remaining rows
            if chunk:
                yield chunk


# Factory function following NanoBrain patterns
def from_config(config: Dict[str, Any]) -> CSVProcessor:
    """Create CSVProcessor from configuration dictionary"""
    return CSVProcessor(config)
