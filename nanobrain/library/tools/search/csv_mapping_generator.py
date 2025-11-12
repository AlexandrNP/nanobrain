"""
CSV Mapping Generator for Elasticsearch

This module provides utilities for generating optimized Elasticsearch mappings
from CSV file structures, following NanoBrain architectural patterns.
"""

from typing import Dict, List, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)


class CSVMappingGenerator:
    """
    Generate Elasticsearch mappings optimized for CSV data.
    Supports automatic type detection and fuzzy search optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize mapping generator with configuration"""
        self.config = config or {}
        
        # Configuration options
        self.enable_fuzzy_fields = self.config.get('enable_fuzzy_fields', True)
        self.enable_keyword_fields = self.config.get('enable_keyword_fields', True)
        self.max_keyword_length = self.config.get('max_keyword_length', 256)
        self.default_analyzer = self.config.get('default_analyzer', 'standard')
        self.fuzzy_analyzer = self.config.get('fuzzy_analyzer', 'csv_fuzzy_analyzer')
        
        # Type mapping configuration
        self.type_mappings = self.config.get('type_mappings', {
            'integer': 'long',
            'float': 'double',
            'boolean': 'boolean',
            'date': 'date',
            'text': 'text',
            'email': 'text',
            'url': 'text'
        })
        
        logger.info("✅ CSV Mapping Generator initialized")
    
    def generate_mapping(self, csv_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete Elasticsearch mapping from CSV structure"""
        try:
            logger.info("🔧 Generating Elasticsearch mapping from CSV structure")
            
            headers = csv_structure.get('headers', [])
            column_types = csv_structure.get('column_types', {})
            
            if not headers:
                raise ValueError("No headers found in CSV structure")
            
            # Generate properties for each column
            properties = {}
            for header in headers:
                column_type = column_types.get(header, 'text')
                properties[header] = self._generate_field_mapping(header, column_type)
            
            # Add metadata fields
            properties.update(self._generate_metadata_fields())
            
            # Create complete mapping
            mapping = {
                "properties": properties,
                "dynamic": "strict"  # Prevent automatic field creation
            }
            
            # Add mapping metadata
            mapping["_meta"] = {
                "csv_columns": len(headers),
                "generated_by": "nanobrain_csv_mapping_generator",
                "column_types": column_types
            }
            
            logger.info(f"✅ Mapping generated for {len(headers)} columns")
            
            return mapping
            
        except Exception as e:
            logger.error(f"❌ Mapping generation failed: {e}")
            raise
    
    def _generate_field_mapping(self, field_name: str, field_type: str) -> Dict[str, Any]:
        """Generate mapping for a single field"""
        
        # Get Elasticsearch type
        es_type = self.type_mappings.get(field_type, 'text')
        
        if es_type == 'text':
            return self._generate_text_field_mapping(field_name, field_type)
        elif es_type == 'date':
            return self._generate_date_field_mapping()
        elif es_type in ['long', 'double']:
            return self._generate_numeric_field_mapping(es_type)
        elif es_type == 'boolean':
            return self._generate_boolean_field_mapping()
        else:
            # Default text mapping
            return self._generate_text_field_mapping(field_name, field_type)
    
    def _generate_text_field_mapping(self, field_name: str, field_type: str) -> Dict[str, Any]:
        """Generate mapping for text fields with fuzzy search support"""
        mapping = {
            "type": "text",
            "analyzer": self.default_analyzer
        }
        
        # Add multi-fields for different search strategies
        fields = {}
        
        # Keyword field for exact matching and aggregations
        if self.enable_keyword_fields:
            fields["keyword"] = {
                "type": "keyword",
                "ignore_above": self.max_keyword_length
            }
        
        # Fuzzy search field
        if self.enable_fuzzy_fields:
            fields["fuzzy"] = {
                "type": "text",
                "analyzer": self.fuzzy_analyzer
            }
        
        # Special handling for specific field types
        if field_type == 'email':
            fields["email"] = {
                "type": "text",
                "analyzer": "keyword"  # Don't tokenize emails
            }
        elif field_type == 'url':
            fields["url"] = {
                "type": "text",
                "analyzer": "keyword"  # Don't tokenize URLs
            }
        
        # Add field-specific optimizations based on field name
        if self._is_name_field(field_name):
            # Optimize for name searching
            fields["name_search"] = {
                "type": "text",
                "analyzer": "standard",
                "search_analyzer": "standard"
            }
        elif self._is_id_field(field_name):
            # Optimize for ID searching
            fields["id_search"] = {
                "type": "keyword",
                "normalizer": "lowercase"
            }
        
        if fields:
            mapping["fields"] = fields
        
        return mapping
    
    def _generate_date_field_mapping(self) -> Dict[str, Any]:
        """Generate mapping for date fields"""
        return {
            "type": "date",
            "format": "yyyy-MM-dd||yyyy/MM/dd||dd-MM-yyyy||dd/MM/yyyy||strict_date_optional_time||epoch_millis"
        }
    
    def _generate_numeric_field_mapping(self, es_type: str) -> Dict[str, Any]:
        """Generate mapping for numeric fields"""
        return {
            "type": es_type
        }
    
    def _generate_boolean_field_mapping(self) -> Dict[str, Any]:
        """Generate mapping for boolean fields"""
        return {
            "type": "boolean"
        }
    
    def _generate_metadata_fields(self) -> Dict[str, Any]:
        """Generate metadata fields for CSV tracking"""
        return {
            "_csv_import_timestamp": {
                "type": "date"
            },
            "_csv_row_number": {
                "type": "long"
            },
            "_csv_file_name": {
                "type": "keyword"
            },
            "_csv_file_hash": {
                "type": "keyword"
            }
        }
    
    def _is_name_field(self, field_name: str) -> bool:
        """Check if field is likely a name field"""
        name_patterns = [
            r'.*name.*',
            r'.*title.*',
            r'.*label.*',
            r'.*description.*'
        ]
        
        field_lower = field_name.lower()
        return any(re.match(pattern, field_lower) for pattern in name_patterns)
    
    def _is_id_field(self, field_name: str) -> bool:
        """Check if field is likely an ID field"""
        id_patterns = [
            r'.*id$',
            r'^id.*',
            r'.*_id$',
            r'.*code.*',
            r'.*sku.*',
            r'.*key.*'
        ]
        
        field_lower = field_name.lower()
        return any(re.match(pattern, field_lower) for pattern in id_patterns)
    
    def generate_index_settings(self, csv_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized index settings for CSV data"""
        estimated_rows = csv_structure.get('estimated_rows', 1000)
        column_count = csv_structure.get('column_count', 10)
        
        # Calculate optimal shard configuration
        if estimated_rows < 10000:
            shards = 1
        elif estimated_rows < 100000:
            shards = 2
        else:
            shards = min(5, max(1, estimated_rows // 50000))
        
        # Replica configuration
        replicas = 0 if estimated_rows < 50000 else 1
        
        settings = {
            "number_of_shards": shards,
            "number_of_replicas": replicas,
            "refresh_interval": "30s",  # Optimize for bulk indexing
            "analysis": {
                "analyzer": {
                    "csv_text_analyzer": {
                        "type": "standard",
                        "stopwords": "_english_"
                    },
                    "csv_fuzzy_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding", "stop"]
                    },
                    "csv_name_analyzer": {
                        "type": "custom",
                        "tokenizer": "keyword",
                        "filter": ["lowercase", "asciifolding"]
                    }
                },
                "normalizer": {
                    "lowercase": {
                        "type": "custom",
                        "filter": ["lowercase"]
                    }
                }
            }
        }
        
        return settings
    
    def optimize_mapping_for_search(self, mapping: Dict[str, Any], search_fields: List[str]) -> Dict[str, Any]:
        """Optimize mapping for specific search fields"""
        try:
            optimized_mapping = mapping.copy()
            properties = optimized_mapping.get("properties", {})
            
            for field_name in search_fields:
                if field_name in properties:
                    field_mapping = properties[field_name]
                    
                    # Add search-optimized analyzers for important fields
                    if field_mapping.get("type") == "text":
                        if "fields" not in field_mapping:
                            field_mapping["fields"] = {}
                        
                        # Add search-optimized field
                        field_mapping["fields"]["search"] = {
                            "type": "text",
                            "analyzer": "csv_text_analyzer",
                            "search_analyzer": "standard"
                        }
                        
                        # Boost important fields
                        field_mapping["boost"] = 2.0
            
            return optimized_mapping
            
        except Exception as e:
            logger.error(f"❌ Mapping optimization failed: {e}")
            return mapping


# Factory function following NanoBrain patterns
def from_config(config: Dict[str, Any]) -> CSVMappingGenerator:
    """Create CSVMappingGenerator from configuration dictionary"""
    return CSVMappingGenerator(config)
