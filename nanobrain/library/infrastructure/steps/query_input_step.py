"""
Query Input Step for NanoBrain Framework

This module provides a specialized step for processing and validating search queries,
including query parsing, enhancement, normalization, and parameter extraction.

**SINGLE RESPONSIBILITY**: Query processing and validation
**FRAMEWORK COMPLIANCE**: Full compliance with NanoBrain step patterns
**CONFIGURATION-DRIVEN**: All processing options via YAML configuration
**EVENT-DRIVEN**: Automatic processing on query data unit changes

This component follows NanoBrain framework patterns:
- Inherits from BaseStep for framework compliance
- Uses from_config pattern for component creation
- Provides comprehensive configuration validation
- Supports event-driven workflow orchestration

Usage:
    from nanobrain.library.infrastructure.steps import QueryInputStep
    
    # Create via from_config (framework pattern)
    step = QueryInputStep.from_config('config/query_input_step.yml')
    
    # Execute within workflow context
    await step.execute()
"""

import re
import time
from typing import Dict, Any, List

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.logging_system import OperationType


class ProcessingError(Exception):
    """Custom exception for processing errors"""
    pass


class QueryInputStepConfig(StepConfig):
    """Configuration schema for Query Input Step."""
    
    # Query Validation Configuration
    query_validation: Dict[str, Any] = {
        'min_length': 2,
        'max_length': 1000,
        'sanitization_enabled': True,
        'allow_empty': False,
        'trim_whitespace': True
    }
    
    # Query Enhancement Configuration
    query_enhancement: Dict[str, Any] = {
        'spell_check_enabled': True,
        'synonym_expansion': True,
        'stemming_enabled': True,
        'case_normalization': True,
        'special_char_handling': 'normalize'
    }
    
    # Search Parameter Extraction
    parameter_extraction: Dict[str, Any] = {
        'extract_field_hints': True,
        'extract_operators': True,
        'extract_quotes': True,
        'extract_wildcards': True,
        'default_search_fields': ['name', 'description']
    }
    
    # Query Type Detection
    type_detection: Dict[str, Any] = {
        'enable_auto_detection': True,
        'numeric_pattern_detection': True,
        'date_pattern_detection': True,
        'email_pattern_detection': True,
        'url_pattern_detection': True
    }


class QueryInputStep(BaseStep):
    """
    ✅ FRAMEWORK COMPLIANCE: Query Input Step for processing and validating search queries
    
    Provides comprehensive query processing capabilities including validation, enhancement,
    normalization, and parameter extraction. Prepares queries for optimal fuzzy search
    execution with intelligent type detection and field suggestions.
    
    **Single Responsibility**: Query processing and validation
    **Framework Integration**: Full compliance with NanoBrain step patterns
    **Configuration-Driven**: All processing options via YAML configuration
    **Event-Driven**: Automatic processing on query data unit changes
    
    **Input Data Units:**
        * user_query (DataUnitMemory): Raw user search query string or object
        * query_context (DataUnitMemory, optional): Additional query context and hints
    
    **Output Data Units:**
        * processed_query (DataUnitMemory): Enhanced and validated search query
        * query_metadata (DataUnitMemory): Query analysis and processing metadata
    
    **Configuration Options:**
        * query_validation: Length limits, sanitization, and validation rules
        * query_enhancement: Spell check, synonyms, stemming, and normalization
        * parameter_extraction: Field hints, operators, and search parameters
        * type_detection: Automatic query type and pattern detection
    
    **Processing Flow:**
        1. Validate query format and length constraints
        2. Sanitize and normalize query text
        3. Extract search parameters and field hints
        4. Detect query type and patterns
        5. Enhance query with synonyms and corrections
    
    **Usage Patterns:**
        ```python
        from nanobrain.library.infrastructure.steps import QueryInputStep
        
        # Create step from configuration
        query_step = QueryInputStep.from_config('config/query_input_step.yml')
        
        # Execute within workflow context
        await query_step.execute()
        
        # Step automatically handles:
        # - Query validation and sanitization
        # - Parameter extraction and field detection
        # - Query enhancement and normalization
        # - Type detection and pattern analysis
        ```
    
    Attributes:
        name (str): Step identifier for logging and debugging
        description (str): Human-readable step description
        config (QueryInputStepConfig): Step configuration instance
    
    See Also:
        * :class:`BaseStep`: Base framework step interface
        * :class:`QueryInputStepConfig`: Configuration schema
        * :class:`DataUnitMemory`: Memory-based data unit for inputs/outputs
        * :class:`CSVFuzzySearchStep`: Fuzzy search step for query execution
    """
    
    # MANDATORY COMPONENT METADATA
    COMPONENT_TYPE: str = "query_input_step"
    REQUIRED_CONFIG_FIELDS: List[str] = ['name']
    
    @classmethod
    def _get_config_class(cls):
        """Return QueryInputStepConfig for configuration validation"""
        return QueryInputStepConfig
    
    @classmethod
    def extract_component_config(cls, config: QueryInputStepConfig) -> Dict[str, Any]:
        """Extract QueryInputStep-specific configuration"""
        base_config = super().extract_component_config(config)
        return {
            **base_config,
            'query_validation': getattr(config, 'query_validation', {}),
            'query_enhancement': getattr(config, 'query_enhancement', {}),
            'parameter_extraction': getattr(config, 'parameter_extraction', {}),
            'type_detection': getattr(config, 'type_detection', {})
        }
    
    def _init_from_config(self, config: QueryInputStepConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize QueryInputStep with configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Store configuration
        self.query_validation = component_config.get('query_validation', {})
        self.query_enhancement = component_config.get('query_enhancement', {})
        self.parameter_extraction = component_config.get('parameter_extraction', {})
        self.type_detection = component_config.get('type_detection', {})
        
        # Processing statistics
        self.processing_stats = {
            'total_queries': 0,
            'validation_failures': 0,
            'enhancement_applied': 0,
            'average_processing_time': 0.0
        }
        
        self.nb_logger.info(f"Query Input Step {self.name} initialized",
                           validation_config=self.query_validation,
                           enhancement_config=self.query_enhancement)
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process and validate search query with comprehensive enhancement"""
        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE,
            f"{self.name}.process",
            input_keys=list(input_data.keys()),
            step_type="QueryInputStep"
        ) as context:
            
            try:
                processing_start_time = time.time()
                self.nb_logger.info(f"Starting query processing for step: {self.name}")
                
                # Extract and validate input data
                user_query_data = input_data.get('user_query')
                query_context_data = input_data.get('query_context', {})
                
                if not user_query_data:
                    if not self.query_validation.get('allow_empty', False):
                        raise ProcessingError("No user query provided")
                    user_query_data = ""
                
                # Extract query string
                if isinstance(user_query_data, dict):
                    raw_query = user_query_data.get('query', user_query_data.get('search_query', ''))
                else:
                    raw_query = str(user_query_data)
                
                # Step 1: Validate query
                self.nb_logger.debug(f"Validating query: '{raw_query}'")
                validation_result = await self._validate_query(raw_query)
                
                if not validation_result['valid']:
                    self.processing_stats['validation_failures'] += 1
                    raise ProcessingError(f"Query validation failed: {validation_result['error']}")
                
                # Step 2: Sanitize and normalize query
                sanitized_query = await self._sanitize_query(raw_query)
                
                # Step 3: Extract search parameters
                extracted_params = await self._extract_parameters(sanitized_query, query_context_data)
                
                # Step 4: Detect query type and patterns
                query_analysis = await self._analyze_query_type(sanitized_query)
                
                # Step 5: Enhance query
                enhanced_query = await self._enhance_query(sanitized_query, extracted_params, query_analysis)
                
                # Step 6: Generate final processed query
                processing_end_time = time.time()
                processing_duration = processing_end_time - processing_start_time
                
                self.processing_stats['total_queries'] += 1
                self.processing_stats['average_processing_time'] = (
                    (self.processing_stats['average_processing_time'] * (self.processing_stats['total_queries'] - 1) + processing_duration) /
                    self.processing_stats['total_queries']
                )
                
                if enhanced_query != sanitized_query:
                    self.processing_stats['enhancement_applied'] += 1
                
                processed_query = {
                    'original_query': raw_query,
                    'sanitized_query': sanitized_query,
                    'enhanced_query': enhanced_query,
                    'search_parameters': extracted_params,
                    'query_analysis': query_analysis,
                    'processing_timestamp': processing_end_time
                }
                
                query_metadata = {
                    'processing_duration': processing_duration,
                    'validation_result': validation_result,
                    'enhancement_applied': enhanced_query != sanitized_query,
                    'extracted_parameters': extracted_params,
                    'query_type': query_analysis.get('query_type', 'unknown'),
                    'confidence': query_analysis.get('confidence', 0.0),
                    'suggestions': query_analysis.get('suggestions', []),
                    'processing_stats': self.processing_stats.copy(),
                    'timestamp': processing_end_time
                }
                
                self.nb_logger.info("Query processing completed successfully",
                                   original_query=raw_query,
                                   enhanced_query=enhanced_query,
                                   query_type=query_analysis.get('query_type'),
                                   processing_duration=processing_duration)
                
                return {
                    'processed_query': processed_query,
                    'query_metadata': query_metadata
                }
                
            except Exception as e:
                self.nb_logger.error(f"Query processing failed for step {self.name}: {e}",
                                   error_type=type(e).__name__,
                                   user_query=input_data.get('user_query'))
                
                # Return error results
                return {
                    'processed_query': {
                        'original_query': input_data.get('user_query', ''),
                        'error': str(e),
                        'status': 'failed'
                    },
                    'query_metadata': {
                        'error': str(e),
                        'timestamp': time.time(),
                        'status': 'failed'
                    }
                }
    
    async def _validate_query(self, query: str) -> Dict[str, Any]:
        """Validate query against configuration rules"""
        try:
            validation_result = {
                'valid': True,
                'error': None,
                'warnings': []
            }
            
            # Check length constraints
            min_length = self.query_validation.get('min_length', 2)
            max_length = self.query_validation.get('max_length', 1000)
            
            if len(query) < min_length:
                validation_result['valid'] = False
                validation_result['error'] = f"Query too short (minimum {min_length} characters)"
                return validation_result
            
            if len(query) > max_length:
                validation_result['valid'] = False
                validation_result['error'] = f"Query too long (maximum {max_length} characters)"
                return validation_result
            
            # Check for empty query
            if not query.strip() and not self.query_validation.get('allow_empty', False):
                validation_result['valid'] = False
                validation_result['error'] = "Empty query not allowed"
                return validation_result
            
            # Check for potentially problematic patterns
            if re.search(r'[<>{}[\]\\]', query):
                validation_result['warnings'].append("Query contains special characters that may affect search")
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Validation error: {e}",
                'warnings': []
            }
    
    async def _sanitize_query(self, query: str) -> str:
        """Sanitize and normalize query text"""
        try:
            sanitized = query
            
            # Trim whitespace
            if self.query_validation.get('trim_whitespace', True):
                sanitized = sanitized.strip()
            
            # Normalize case
            if self.query_enhancement.get('case_normalization', True):
                sanitized = sanitized.lower()
            
            # Handle special characters
            special_char_handling = self.query_enhancement.get('special_char_handling', 'normalize')
            if special_char_handling == 'normalize':
                # Replace multiple spaces with single space
                sanitized = re.sub(r'\s+', ' ', sanitized)
                # Remove or normalize problematic characters
                sanitized = re.sub(r'[^\w\s\-\.\'"@]', ' ', sanitized)
                sanitized = re.sub(r'\s+', ' ', sanitized).strip()
            
            return sanitized
            
        except Exception as e:
            self.nb_logger.warning(f"Query sanitization failed: {e}")
            return query
    
    async def _extract_parameters(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract search parameters from query and context"""
        try:
            parameters = {
                'search_fields': self.parameter_extraction.get('default_search_fields', []),
                'operators': [],
                'quoted_phrases': [],
                'wildcards': [],
                'field_hints': []
            }
            
            # Extract quoted phrases
            if self.parameter_extraction.get('extract_quotes', True):
                quoted_matches = re.findall(r'"([^"]*)"', query)
                parameters['quoted_phrases'] = quoted_matches
            
            # Extract field hints (field:value patterns)
            if self.parameter_extraction.get('extract_field_hints', True):
                field_matches = re.findall(r'(\w+):(\w+)', query)
                parameters['field_hints'] = [{'field': field, 'value': value} for field, value in field_matches]
            
            # Extract operators
            if self.parameter_extraction.get('extract_operators', True):
                operators = re.findall(r'(AND|OR|NOT|\+|\-)', query, re.IGNORECASE)
                parameters['operators'] = operators
            
            # Extract wildcards
            if self.parameter_extraction.get('extract_wildcards', True):
                wildcards = re.findall(r'\w*[\*\?]\w*', query)
                parameters['wildcards'] = wildcards
            
            # Merge context parameters
            if context:
                if 'search_fields' in context:
                    parameters['search_fields'] = context['search_fields']
                if 'fuzzy_threshold' in context:
                    parameters['fuzzy_threshold'] = context['fuzzy_threshold']
                if 'max_results' in context:
                    parameters['max_results'] = context['max_results']
            
            return parameters
            
        except Exception as e:
            self.nb_logger.warning(f"Parameter extraction failed: {e}")
            return {'search_fields': self.parameter_extraction.get('default_search_fields', [])}
    
    async def _analyze_query_type(self, query: str) -> Dict[str, Any]:
        """Analyze query type and detect patterns"""
        try:
            analysis = {
                'query_type': 'text',
                'confidence': 0.8,
                'patterns': [],
                'suggestions': []
            }
            
            if not self.type_detection.get('enable_auto_detection', True):
                return analysis
            
            # Numeric pattern detection
            if self.type_detection.get('numeric_pattern_detection', True):
                if re.search(r'\d+', query):
                    analysis['patterns'].append('numeric')
                    if re.search(r'^\d+$', query.strip()):
                        analysis['query_type'] = 'numeric'
                        analysis['confidence'] = 0.9
            
            # Date pattern detection
            if self.type_detection.get('date_pattern_detection', True):
                date_patterns = [
                    r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
                    r'\d{1,2}[-/]\d{1,2}[-/]\d{4}'
                ]
                for pattern in date_patterns:
                    if re.search(pattern, query):
                        analysis['patterns'].append('date')
                        analysis['query_type'] = 'date'
                        analysis['confidence'] = 0.85
                        break
            
            # Email pattern detection
            if self.type_detection.get('email_pattern_detection', True):
                if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', query):
                    analysis['patterns'].append('email')
                    analysis['query_type'] = 'email'
                    analysis['confidence'] = 0.95
            
            # URL pattern detection
            if self.type_detection.get('url_pattern_detection', True):
                if re.search(r'https?://\S+', query):
                    analysis['patterns'].append('url')
                    analysis['query_type'] = 'url'
                    analysis['confidence'] = 0.95
            
            # Generate suggestions based on analysis
            if analysis['query_type'] == 'numeric':
                analysis['suggestions'].append("Consider using range queries for numeric searches")
            elif analysis['query_type'] == 'date':
                analysis['suggestions'].append("Date queries work best with YYYY-MM-DD format")
            
            return analysis
            
        except Exception as e:
            self.nb_logger.warning(f"Query type analysis failed: {e}")
            return {'query_type': 'text', 'confidence': 0.5, 'patterns': [], 'suggestions': []}
    
    async def _enhance_query(self, query: str, parameters: Dict[str, Any], 
                           analysis: Dict[str, Any]) -> str:
        """Enhance query with improvements and corrections"""
        try:
            enhanced = query
            
            # Apply stemming if enabled
            if self.query_enhancement.get('stemming_enabled', True):
                # Simple stemming - remove common suffixes
                words = enhanced.split()
                stemmed_words = []
                for word in words:
                    if len(word) > 4:
                        if word.endswith('ing'):
                            word = word[:-3]
                        elif word.endswith('ed'):
                            word = word[:-2]
                        elif word.endswith('s') and not word.endswith('ss'):
                            word = word[:-1]
                    stemmed_words.append(word)
                enhanced = ' '.join(stemmed_words)
            
            # Apply synonym expansion if enabled
            if self.query_enhancement.get('synonym_expansion', True):
                # Simple synonym mapping
                synonyms = {
                    'phone': 'smartphone mobile device',
                    'laptop': 'computer notebook',
                    'car': 'vehicle automobile'
                }
                
                words = enhanced.split()
                expanded_words = []
                for word in words:
                    expanded_words.append(word)
                    if word in synonyms:
                        expanded_words.extend(synonyms[word].split())
                
                if len(expanded_words) > len(words):
                    enhanced = ' '.join(expanded_words)
            
            return enhanced
            
        except Exception as e:
            self.nb_logger.warning(f"Query enhancement failed: {e}")
            return query
