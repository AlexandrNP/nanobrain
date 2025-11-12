"""
CSV Validation Utilities for NanoBrain Framework

This module provides comprehensive validation capabilities for CSV files including
format validation, data quality checks, and schema validation.
"""

import csv
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CSVValidationRule:
    """Base class for CSV validation rules"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def validate(self, data: Any) -> Dict[str, Any]:
        """Validate data and return result"""
        raise NotImplementedError


class HeaderValidationRule(CSVValidationRule):
    """Validate CSV headers"""
    
    def __init__(self):
        super().__init__("header_validation", "Validate CSV headers for duplicates and empty values")
    
    def validate(self, headers: List[str]) -> Dict[str, Any]:
        """Validate headers"""
        issues = []
        warnings = []
        
        # Check for empty headers
        empty_indices = [i for i, h in enumerate(headers) if not h.strip()]
        if empty_indices:
            issues.append(f"Empty headers at positions: {empty_indices}")
        
        # Check for duplicate headers
        seen = set()
        duplicates = set()
        for header in headers:
            if header in seen:
                duplicates.add(header)
            seen.add(header)
        
        if duplicates:
            warnings.append(f"Duplicate headers: {list(duplicates)}")
        
        # Check for very long headers
        long_headers = [h for h in headers if len(h) > 100]
        if long_headers:
            warnings.append(f"Very long headers (>100 chars): {len(long_headers)} found")
        
        # Check for special characters that might cause issues
        problematic_headers = []
        for header in headers:
            if re.search(r'[^\w\s\-_.]', header):
                problematic_headers.append(header)
        
        if problematic_headers:
            warnings.append(f"Headers with special characters: {len(problematic_headers)} found")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'header_count': len(headers),
            'unique_headers': len(set(headers))
        }


class DataConsistencyRule(CSVValidationRule):
    """Validate data consistency across rows"""
    
    def __init__(self):
        super().__init__("data_consistency", "Validate data consistency and format")
    
    def validate(self, rows: List[List[str]], headers: List[str]) -> Dict[str, Any]:
        """Validate data consistency"""
        issues = []
        warnings = []
        
        if not rows:
            return {
                'valid': True,
                'issues': [],
                'warnings': ['No data rows to validate'],
                'row_count': 0
            }
        
        expected_columns = len(headers)
        
        # Check row length consistency
        inconsistent_rows = []
        for i, row in enumerate(rows):
            if len(row) != expected_columns:
                inconsistent_rows.append({
                    'row_index': i,
                    'expected_columns': expected_columns,
                    'actual_columns': len(row)
                })
        
        if inconsistent_rows:
            if len(inconsistent_rows) > len(rows) * 0.1:  # More than 10% inconsistent
                issues.append(f"Many rows have inconsistent column count: {len(inconsistent_rows)}/{len(rows)}")
            else:
                warnings.append(f"Some rows have inconsistent column count: {len(inconsistent_rows)}/{len(rows)}")
        
        # Check for completely empty rows
        empty_rows = [i for i, row in enumerate(rows) if all(not cell.strip() for cell in row)]
        if empty_rows:
            warnings.append(f"Empty rows found: {len(empty_rows)}")
        
        # Check for rows with mostly empty cells
        mostly_empty_rows = []
        for i, row in enumerate(rows):
            non_empty_cells = sum(1 for cell in row if cell.strip())
            if non_empty_cells < len(row) * 0.3:  # Less than 30% filled
                mostly_empty_rows.append(i)
        
        if mostly_empty_rows:
            warnings.append(f"Rows with mostly empty cells: {len(mostly_empty_rows)}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'row_count': len(rows),
            'inconsistent_rows': len(inconsistent_rows),
            'empty_rows': len(empty_rows),
            'mostly_empty_rows': len(mostly_empty_rows)
        }


class DataTypeValidationRule(CSVValidationRule):
    """Validate data types within columns"""
    
    def __init__(self):
        super().__init__("data_type_validation", "Validate data types and formats within columns")
    
    def validate(self, rows: List[List[str]], headers: List[str], expected_types: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Validate data types"""
        issues = []
        warnings = []
        column_analysis = {}
        
        if not rows or not headers:
            return {
                'valid': True,
                'issues': [],
                'warnings': ['No data to validate'],
                'column_analysis': {}
            }
        
        # Analyze each column
        for col_idx, header in enumerate(headers):
            column_values = [row[col_idx] if col_idx < len(row) else '' for row in rows]
            analysis = self._analyze_column(column_values, header)
            column_analysis[header] = analysis
            
            # Check for type consistency if expected type is provided
            if expected_types and header in expected_types:
                expected_type = expected_types[header]
                detected_type = analysis['detected_type']
                
                if detected_type != expected_type:
                    warnings.append(f"Column '{header}': expected {expected_type}, detected {detected_type}")
            
            # Check for potential data quality issues
            if analysis['empty_percentage'] > 50:
                warnings.append(f"Column '{header}': high percentage of empty values ({analysis['empty_percentage']:.1f}%)")
            
            if analysis['unique_percentage'] < 5 and analysis['non_empty_count'] > 10:
                warnings.append(f"Column '{header}': low data diversity ({analysis['unique_percentage']:.1f}% unique)")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'column_analysis': column_analysis
        }
    
    def _analyze_column(self, values: List[str], column_name: str) -> Dict[str, Any]:
        """Analyze a single column"""
        total_count = len(values)
        non_empty_values = [v.strip() for v in values if v.strip()]
        non_empty_count = len(non_empty_values)
        unique_values = set(non_empty_values)
        
        # Detect data type
        detected_type = self._detect_type(non_empty_values)
        
        # Calculate statistics
        empty_percentage = ((total_count - non_empty_count) / total_count * 100) if total_count > 0 else 0
        unique_percentage = (len(unique_values) / non_empty_count * 100) if non_empty_count > 0 else 0
        
        # Find potential issues
        issues = []
        
        # Check for very long values
        long_values = [v for v in non_empty_values if len(v) > 1000]
        if long_values:
            issues.append(f"Very long values found: {len(long_values)}")
        
        # Check for potential encoding issues
        encoding_issues = [v for v in non_empty_values if any(ord(c) > 127 for c in v)]
        if encoding_issues and len(encoding_issues) > non_empty_count * 0.1:
            issues.append(f"Potential encoding issues: {len(encoding_issues)} values")
        
        return {
            'detected_type': detected_type,
            'total_count': total_count,
            'non_empty_count': non_empty_count,
            'unique_count': len(unique_values),
            'empty_percentage': empty_percentage,
            'unique_percentage': unique_percentage,
            'issues': issues,
            'sample_values': list(unique_values)[:5]  # First 5 unique values as sample
        }
    
    def _detect_type(self, values: List[str]) -> str:
        """Detect the most likely data type for a list of values"""
        if not values:
            return 'empty'
        
        # Type counters
        integer_count = 0
        float_count = 0
        boolean_count = 0
        date_count = 0
        email_count = 0
        url_count = 0
        
        for value in values:
            # Boolean check
            if value.lower() in ['true', 'false', 'yes', 'no', '1', '0', 'y', 'n']:
                boolean_count += 1
                continue
            
            # Integer check
            try:
                int(value)
                integer_count += 1
                continue
            except ValueError:
                pass
            
            # Float check
            try:
                float(value)
                float_count += 1
                continue
            except ValueError:
                pass
            
            # Email check
            if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
                email_count += 1
                continue
            
            # URL check
            if re.match(r'^https?://', value):
                url_count += 1
                continue
            
            # Date check (basic patterns)
            if re.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}', value) or \
               re.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}', value):
                date_count += 1
                continue
        
        total = len(values)
        
        # Determine type by majority (70% threshold for most types, 50% for specialized types)
        if boolean_count / total > 0.7:
            return 'boolean'
        elif integer_count / total > 0.7:
            return 'integer'
        elif (integer_count + float_count) / total > 0.7:
            return 'numeric'
        elif email_count / total > 0.5:
            return 'email'
        elif url_count / total > 0.5:
            return 'url'
        elif date_count / total > 0.5:
            return 'date'
        else:
            return 'text'


class CSVValidator:
    """
    Comprehensive CSV validator following NanoBrain patterns.
    Provides configurable validation rules and detailed reporting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize validator with configuration"""
        self.config = config or {}
        
        # Initialize validation rules
        self.rules = [
            HeaderValidationRule(),
            DataConsistencyRule(),
            DataTypeValidationRule()
        ]
        
        # Configuration options
        self.max_sample_rows = self.config.get('max_sample_rows', 1000)
        self.strict_mode = self.config.get('strict_mode', False)
        
        logger.info("✅ CSV Validator initialized")
    
    def validate_file(self, file_path: str, expected_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate entire CSV file"""
        try:
            logger.info(f"🔍 Validating CSV file: {file_path}")
            
            # Read file structure
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader, [])
                
                # Sample data rows for validation
                sample_rows = []
                for i, row in enumerate(reader):
                    sample_rows.append(row)
                    if i >= self.max_sample_rows:
                        break
            
            # Run validation rules
            validation_results = {}
            overall_valid = True
            all_issues = []
            all_warnings = []
            
            # Header validation
            header_result = self.rules[0].validate(headers)
            validation_results['headers'] = header_result
            if not header_result['valid']:
                overall_valid = False
            all_issues.extend(header_result['issues'])
            all_warnings.extend(header_result['warnings'])
            
            # Data consistency validation
            consistency_result = self.rules[1].validate(sample_rows, headers)
            validation_results['data_consistency'] = consistency_result
            if not consistency_result['valid']:
                overall_valid = False
            all_issues.extend(consistency_result['issues'])
            all_warnings.extend(consistency_result['warnings'])
            
            # Data type validation
            expected_types = expected_schema.get('column_types') if expected_schema else None
            type_result = self.rules[2].validate(sample_rows, headers, expected_types)
            validation_results['data_types'] = type_result
            if not type_result['valid']:
                overall_valid = False
            all_issues.extend(type_result['issues'])
            all_warnings.extend(type_result['warnings'])
            
            # Summary
            summary = {
                'overall_valid': overall_valid,
                'total_issues': len(all_issues),
                'total_warnings': len(all_warnings),
                'file_path': file_path,
                'headers_count': len(headers),
                'sample_rows_validated': len(sample_rows)
            }
            
            logger.info(f"✅ Validation complete: {'PASSED' if overall_valid else 'FAILED'} "
                       f"({len(all_issues)} issues, {len(all_warnings)} warnings)")
            
            return {
                'success': True,
                'summary': summary,
                'validation_results': validation_results,
                'all_issues': all_issues,
                'all_warnings': all_warnings
            }
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path
            }
    
    def validate_data_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate data quality for processed CSV data"""
        try:
            if not data:
                return {
                    'success': True,
                    'quality_score': 0,
                    'issues': ['No data to validate'],
                    'warnings': []
                }
            
            issues = []
            warnings = []
            
            # Check for missing values
            total_cells = len(data) * len(data[0]) if data else 0
            empty_cells = 0
            
            for row in data:
                for value in row.values():
                    if not str(value).strip():
                        empty_cells += 1
            
            empty_percentage = (empty_cells / total_cells * 100) if total_cells > 0 else 0
            
            if empty_percentage > 30:
                issues.append(f"High percentage of empty values: {empty_percentage:.1f}%")
            elif empty_percentage > 10:
                warnings.append(f"Moderate percentage of empty values: {empty_percentage:.1f}%")
            
            # Calculate quality score (0-100)
            quality_score = max(0, 100 - empty_percentage - len(issues) * 10 - len(warnings) * 5)
            
            return {
                'success': True,
                'quality_score': quality_score,
                'empty_percentage': empty_percentage,
                'total_rows': len(data),
                'total_cells': total_cells,
                'issues': issues,
                'warnings': warnings
            }
            
        except Exception as e:
            logger.error(f"❌ Data quality validation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Factory function following NanoBrain patterns
def from_config(config: Dict[str, Any]) -> CSVValidator:
    """Create CSVValidator from configuration dictionary"""
    return CSVValidator(config)
