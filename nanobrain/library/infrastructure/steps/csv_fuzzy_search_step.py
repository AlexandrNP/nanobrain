"""
CSV Fuzzy Search Step for NanoBrain Framework

This module provides a specialized step for executing fuzzy search operations
on CSV data indexed in Elasticsearch, leveraging Phase 2 fuzzy search capabilities.

**SINGLE RESPONSIBILITY**: Fuzzy search execution with confidence scoring
**FRAMEWORK COMPLIANCE**: Full compliance with NanoBrain step patterns
**CONFIGURATION-DRIVEN**: All search options via YAML configuration
**EVENT-DRIVEN**: Automatic search execution on query data unit changes

This component follows NanoBrain framework patterns:
- Inherits from BaseStep for framework compliance
- Uses from_config pattern for component creation
- Provides comprehensive configuration validation
- Supports event-driven workflow orchestration

Usage:
    from nanobrain.library.infrastructure.steps import CSVFuzzySearchStep
    
    # Create via from_config (framework pattern)
    step = CSVFuzzySearchStep.from_config('config/csv_fuzzy_search_step.yml')
    
    # Execute within workflow context
    await step.execute()
"""

import time
from typing import Dict, Any, List

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.logging_system import OperationType


class ProcessingError(Exception):
    """Custom exception for processing errors"""
    pass


class CSVFuzzySearchStepConfig(StepConfig):
    """Configuration schema for CSV Fuzzy Search Step."""
    
    # Search Configuration
    search_parameters: Dict[str, Any] = {
        'fuzzy_threshold': 0.7,
        'max_results': 50,
        'result_scoring': 'confidence',
        'field_weights': {
            'name': 2.0,
            'description': 1.5,
            'category': 1.0
        }
    }
    
    # Fuzzy Algorithm Configuration
    fuzzy_algorithms: Dict[str, Any] = {
        'primary': 'elasticsearch_fuzzy',
        'fallback': 'fuzzywuzzy',
        'phonetic_matching': True,
        'hybrid_mode': True
    }
    
    # Result Formatting Configuration
    result_formatting: Dict[str, Any] = {
        'include_confidence_scores': True,
        'include_field_highlights': True,
        'include_suggestions': True,
        'include_metadata': True
    }
    
    # Performance Configuration
    performance_config: Dict[str, Any] = {
        'cache_enabled': True,
        'cache_ttl': 3600,
        'timeout': 30,
        'concurrent_searches': 1
    }


class CSVFuzzySearchStep(BaseStep):
    """
    ✅ FRAMEWORK COMPLIANCE: CSV Fuzzy Search Step for executing search operations
    
    Provides comprehensive fuzzy search capabilities on CSV data with confidence scoring,
    suggestions, and advanced result formatting. Integrates with Phase 2 fuzzy search
    infrastructure and Elasticsearch MCP server capabilities.
    
    **Single Responsibility**: Fuzzy search execution with confidence scoring
    **Framework Integration**: Full compliance with NanoBrain step patterns
    **Configuration-Driven**: All search options via YAML configuration
    **Event-Driven**: Automatic search execution on query data unit changes
    
    **Input Data Units:**
        * search_query (DataUnitMemory): Search query string or object
        * indexed_data (DataUnitMemory): Reference to indexed CSV data
        * search_config (DataUnitMemory, optional): Runtime search configuration
    
    **Output Data Units:**
        * search_results (DataUnitMemory): Search results with confidence scores
        * search_metadata (DataUnitMemory): Search execution metadata and statistics
    
    **Configuration Options:**
        * search_parameters: Fuzzy threshold, result limits, field weights
        * fuzzy_algorithms: Algorithm selection and hybrid mode settings
        * result_formatting: Output format and metadata inclusion options
        * performance_config: Caching, timeouts, and concurrency settings
    
    **Processing Flow:**
        1. Validate search query and indexed data references
        2. Extract search parameters and field specifications
        3. Execute fuzzy search with confidence scoring
        4. Generate suggestions for query improvement
        5. Format results with highlights and metadata
    
    **Usage Patterns:**
        ```python
        from nanobrain.library.infrastructure.steps import CSVFuzzySearchStep
        
        # Create step from configuration
        search_step = CSVFuzzySearchStep.from_config('config/csv_fuzzy_search_step.yml')
        
        # Execute within workflow context
        await search_step.execute()
        
        # Step automatically handles:
        # - Query validation and enhancement
        # - Multi-algorithm fuzzy search execution
        # - Confidence scoring and result ranking
        # - Suggestion generation and formatting
        ```
    
    Attributes:
        name (str): Step identifier for logging and debugging
        description (str): Human-readable step description
        config (CSVFuzzySearchStepConfig): Step configuration instance
        mcp_client: MCP client for Elasticsearch operations
    
    See Also:
        * :class:`BaseStep`: Base framework step interface
        * :class:`CSVFuzzySearchStepConfig`: Configuration schema
        * :class:`DataUnitMemory`: Memory-based data unit for inputs/outputs
        * :class:`CSVImportStep`: CSV import step for data preparation
    """
    
    # MANDATORY COMPONENT METADATA
    COMPONENT_TYPE: str = "csv_fuzzy_search_step"
    REQUIRED_CONFIG_FIELDS: List[str] = ['name']
    
    @classmethod
    def _get_config_class(cls):
        """Return CSVFuzzySearchStepConfig for configuration validation"""
        return CSVFuzzySearchStepConfig
    
    @classmethod
    def extract_component_config(cls, config: CSVFuzzySearchStepConfig) -> Dict[str, Any]:
        """Extract CSVFuzzySearchStep-specific configuration"""
        base_config = super().extract_component_config(config)
        return {
            **base_config,
            'search_parameters': getattr(config, 'search_parameters', {}),
            'fuzzy_algorithms': getattr(config, 'fuzzy_algorithms', {}),
            'result_formatting': getattr(config, 'result_formatting', {}),
            'performance_config': getattr(config, 'performance_config', {})
        }
    
    def _init_from_config(self, config: CSVFuzzySearchStepConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize CSVFuzzySearchStep with configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Store configuration
        self.search_parameters = component_config.get('search_parameters', {})
        self.fuzzy_algorithms = component_config.get('fuzzy_algorithms', {})
        self.result_formatting = component_config.get('result_formatting', {})
        self.performance_config = component_config.get('performance_config', {})
        
        # Initialize MCP client (will be connected during execution)
        self.mcp_client = None
        
        # Search state
        self.search_stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'average_response_time': 0.0,
            'last_search_time': None
        }
        
        self.nb_logger.info(f"CSV Fuzzy Search Step {self.name} initialized",
                           search_params=self.search_parameters,
                           algorithms=self.fuzzy_algorithms)
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process fuzzy search with comprehensive result formatting"""
        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE,
            f"{self.name}.process",
            input_keys=list(input_data.keys()),
            step_type="CSVFuzzySearchStep"
        ) as context:
            
            try:
                search_start_time = time.time()
                self.nb_logger.info(f"Starting fuzzy search process for step: {self.name}")
                
                # Extract and validate input data
                search_query_data = input_data.get('search_query')
                indexed_data_info = input_data.get('indexed_data')
                search_config_data = input_data.get('search_config', {})
                
                if not search_query_data:
                    raise ProcessingError("No search query provided")
                
                if not indexed_data_info:
                    raise ProcessingError("No indexed data reference provided")
                
                # Extract search query string
                if isinstance(search_query_data, dict):
                    query_string = search_query_data.get('query', search_query_data.get('search_query', ''))
                else:
                    query_string = str(search_query_data)
                
                if not query_string.strip():
                    raise ProcessingError("Empty search query provided")
                
                # Extract index information
                if isinstance(indexed_data_info, dict):
                    index_name = indexed_data_info.get('index_name')
                    if not index_name:
                        raise ProcessingError("No index name found in indexed data")
                else:
                    raise ProcessingError("Invalid indexed data format")
                
                # Initialize MCP client connection
                await self._initialize_mcp_client()
                
                # Merge search configuration
                effective_search_config = {
                    **self.search_parameters,
                    **search_config_data
                }
                
                # Step 1: Execute fuzzy search
                self.nb_logger.info(f"Executing fuzzy search: '{query_string}' on index: {index_name}")
                
                search_result = await self._execute_fuzzy_search(
                    index_name, query_string, effective_search_config
                )
                
                if not search_result.get('success'):
                    raise ProcessingError(f"Fuzzy search failed: {search_result.get('error')}")
                
                # Step 2: Format and enhance results
                formatted_results = await self._format_search_results(
                    search_result, query_string, effective_search_config
                )
                
                # Step 3: Generate search metadata
                search_end_time = time.time()
                search_duration = search_end_time - search_start_time
                
                self.search_stats['total_searches'] += 1
                self.search_stats['last_search_time'] = search_end_time
                self.search_stats['average_response_time'] = (
                    (self.search_stats['average_response_time'] * (self.search_stats['total_searches'] - 1) + search_duration) /
                    self.search_stats['total_searches']
                )
                
                if search_result.get('from_cache'):
                    self.search_stats['cache_hits'] += 1
                
                search_metadata = {
                    'query': query_string,
                    'index_name': index_name,
                    'search_duration': search_duration,
                    'total_results': len(formatted_results.get('results', [])),
                    'from_cache': search_result.get('from_cache', False),
                    'search_config': effective_search_config,
                    'suggestions': formatted_results.get('suggestions', []),
                    'query_analysis': search_result.get('query_analysis', {}),
                    'timestamp': search_end_time,
                    'step_stats': self.search_stats.copy()
                }
                
                self.nb_logger.info("Fuzzy search completed successfully",
                                   query=query_string,
                                   results_count=len(formatted_results.get('results', [])),
                                   search_duration=search_duration,
                                   from_cache=search_result.get('from_cache', False))
                
                return {
                    'search_results': formatted_results,
                    'search_metadata': search_metadata
                }
                
            except Exception as e:
                self.nb_logger.error(f"Fuzzy search failed for step {self.name}: {e}",
                                   error_type=type(e).__name__,
                                   search_query=input_data.get('search_query'))
                
                # Return error results
                return {
                    'search_results': {
                        'results': [],
                        'error': str(e),
                        'status': 'failed'
                    },
                    'search_metadata': {
                        'query': input_data.get('search_query', ''),
                        'error': str(e),
                        'timestamp': time.time(),
                        'status': 'failed'
                    }
                }
    
    async def _initialize_mcp_client(self):
        """Initialize MCP client connection to Elasticsearch server"""
        try:
            # Import MCP client (lazy import to avoid circular dependencies)
            from nanobrain.library.tools.search.elasticsearch_mcp_server import ElasticsearchMCPServer
            from nanobrain.library.tools.search.elasticsearch_mcp_server import ElasticsearchMCPConfig
            
            # Create MCP server configuration
            mcp_config = ElasticsearchMCPConfig.from_config({
                'tool_name': 'elasticsearch_csv_server',
                'mcp_host': 'localhost',
                'mcp_port': 8080,
                'elasticsearch_host': 'localhost',
                'elasticsearch_port': 9200,
                'csv_processing_enabled': True,
                'csv_fuzzy_threshold': self.search_parameters.get('fuzzy_threshold', 0.7),
                'csv_max_search_results': self.search_parameters.get('max_results', 50),
                'csv_field_weights': self.search_parameters.get('field_weights', {}),
                'csv_cache_enabled': self.performance_config.get('cache_enabled', True),
                'csv_cache_ttl': self.performance_config.get('cache_ttl', 3600),
                **self.fuzzy_algorithms,
                **self.result_formatting
            })
            
            # Create MCP server instance
            self.mcp_client = ElasticsearchMCPServer(mcp_config)
            
            self.nb_logger.debug("MCP client initialized successfully")
            
        except Exception as e:
            raise ProcessingError(f"Failed to initialize MCP client: {e}")
    
    async def _execute_fuzzy_search(self, index_name: str, query: str, 
                                   search_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fuzzy search using MCP server"""
        try:
            return await self.mcp_client._fuzzy_search_csv(
                index_name=index_name,
                query=query,
                fuzzy_threshold=search_config.get('fuzzy_threshold', 0.7),
                max_results=search_config.get('max_results', 50),
                search_fields=search_config.get('search_fields')
            )
        except Exception as e:
            return {
                'success': False,
                'error': f"Fuzzy search execution failed: {e}"
            }
    
    async def _format_search_results(self, search_result: Dict[str, Any], 
                                   query: str, search_config: Dict[str, Any]) -> Dict[str, Any]:
        """Format and enhance search results"""
        try:
            results = search_result.get('results', [])
            
            # Apply result formatting based on configuration
            formatted_results = []
            
            for result in results:
                formatted_result = {
                    'document': result.get('document', {}),
                    'score': result.get('score', 0.0),
                    'confidence': result.get('confidence', 0.0)
                }
                
                # Include optional formatting elements
                if self.result_formatting.get('include_field_highlights', True):
                    formatted_result['highlights'] = result.get('highlights', {})
                    formatted_result['fuzzy_highlights'] = result.get('fuzzy_highlights', [])
                
                if self.result_formatting.get('include_metadata', True):
                    formatted_result['metadata'] = {
                        'document_id': result.get('document_id'),
                        'match_type': result.get('match_type', 'fuzzy'),
                        'fuzzy_algorithm': result.get('fuzzy_algorithm'),
                        'fuzzy_score': result.get('fuzzy_score'),
                        'combined_score': result.get('combined_score')
                    }
                
                formatted_results.append(formatted_result)
            
            # Sort by confidence score
            formatted_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            return {
                'results': formatted_results,
                'total_results': len(formatted_results),
                'query': query,
                'suggestions': search_result.get('suggestions', []),
                'search_fields': search_result.get('search_fields', []),
                'fuzzy_threshold': search_result.get('fuzzy_threshold', 0.7),
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'results': [],
                'error': f"Result formatting failed: {e}",
                'status': 'failed'
            }
