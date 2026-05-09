"""
CSV Import Step for NanoBrain Framework

This module provides a specialized step for importing CSV files and indexing them
in Elasticsearch using the enhanced MCP server capabilities from Phase 1 and 2.

**SINGLE RESPONSIBILITY**: CSV file import and Elasticsearch indexing
**FRAMEWORK COMPLIANCE**: Full compliance with NanoBrain step patterns
**CONFIGURATION-DRIVEN**: All processing options via YAML configuration
**EVENT-DRIVEN**: Automatic processing on file data unit changes

This component follows NanoBrain framework patterns:
- Inherits from BaseStep for framework compliance
- Uses from_config pattern for component creation
- Provides comprehensive configuration validation
- Supports event-driven workflow orchestration

Usage:
    from nanobrain.library.infrastructure.steps import CSVImportStep
    
    # Create via from_config (framework pattern)
    step = CSVImportStep.from_config('config/csv_import_step.yml')
    
    # Execute within workflow context
    await step.execute()
"""

import time
from typing import Dict, Any, List
from pathlib import Path

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.logging_system import OperationType


class ProcessingError(Exception):
    """Custom exception for processing errors"""
    pass


class CSVImportStepConfig(StepConfig):
    """Configuration schema for CSV Import Step."""
    
    # CSV Processing Configuration
    csv_processing: Dict[str, Any] = {
        'max_file_size_mb': 500,
        'auto_detect_delimiter': True,
        'auto_detect_encoding': True,
        'validation_enabled': True,
        'chunk_size': 1000,
        'parallel_processing': True
    }
    
    # Elasticsearch Configuration
    elasticsearch_config: Dict[str, Any] = {
        'index_prefix': 'csv_data_',
        'bulk_size': 1000,
        'refresh_interval': '1s',
        'create_index': True,
        'optimize_mapping': True
    }
    
    # MCP Server Configuration
    mcp_server_config: Dict[str, Any] = {
        'server_name': 'elasticsearch_csv_server',
        'connection_timeout': 30,
        'request_timeout': 300
    }


class CSVImportStep(BaseStep):
    """
    ✅ FRAMEWORK COMPLIANCE: CSV Import Step for processing and indexing CSV files
    
    Provides comprehensive CSV file import capabilities with Elasticsearch indexing.
    Integrates with Phase 1 CSV processing utilities and Phase 2 fuzzy search infrastructure.
    Supports automatic format detection, validation, and optimized bulk indexing.
    
    **Single Responsibility**: CSV file import and Elasticsearch indexing
    **Framework Integration**: Full compliance with NanoBrain step patterns
    **Configuration-Driven**: All processing options via YAML configuration
    **Event-Driven**: Automatic processing on file data unit changes
    
    **Input Data Units:**
        * csv_file (DataUnitFile): CSV file to import and process
        * import_config (DataUnitMemory, optional): Runtime import configuration
    
    **Output Data Units:**
        * indexed_data (DataUnitMemory): Import results and index information
        * import_status (DataUnitMemory): Detailed import status and metrics
    
    **Configuration Options:**
        * csv_processing: CSV format detection and validation settings
        * elasticsearch_config: Index creation and bulk import settings
        * mcp_server_config: MCP server connection and timeout settings
    
    **Processing Flow:**
        1. Validate CSV file and configuration
        2. Analyze CSV structure and format
        3. Create optimized Elasticsearch index
        4. Import data in batches with progress tracking
        5. Generate import results and status information
    
    **Usage Patterns:**
        ```python
        from nanobrain.library.infrastructure.steps import CSVImportStep
        
        # Create step from configuration
        import_step = CSVImportStep.from_config('config/csv_import_step.yml')
        
        # Execute within workflow context
        await import_step.execute()
        
        # Step automatically handles:
        # - CSV file validation and analysis
        # - Elasticsearch index creation
        # - Bulk data import with progress tracking
        # - Error handling and recovery
        ```
    
    Attributes:
        name (str): Step identifier for logging and debugging
        description (str): Human-readable step description
        config (CSVImportStepConfig): Step configuration instance
        mcp_client: MCP client for Elasticsearch operations
    
    See Also:
        * :class:`BaseStep`: Base framework step interface
        * :class:`CSVImportStepConfig`: Configuration schema
        * :class:`DataUnitFile`: File-based data unit for CSV input
        * :class:`DataUnitMemory`: Memory-based data unit for results
    """
    
    # MANDATORY COMPONENT METADATA
    COMPONENT_TYPE: str = "csv_import_step"
    REQUIRED_CONFIG_FIELDS: List[str] = ['name']
    
    @classmethod
    def _get_config_class(cls):
        """Return CSVImportStepConfig for configuration validation"""
        return CSVImportStepConfig
    
    @classmethod
    def extract_component_config(cls, config: CSVImportStepConfig) -> Dict[str, Any]:
        """Extract CSVImportStep-specific configuration"""
        base_config = super().extract_component_config(config)
        return {
            **base_config,
            'csv_processing': getattr(config, 'csv_processing', {}),
            'elasticsearch_config': getattr(config, 'elasticsearch_config', {}),
            'mcp_server_config': getattr(config, 'mcp_server_config', {})
        }
    
    def _init_from_config(self, config: CSVImportStepConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize CSVImportStep with configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Store configuration
        self.csv_processing_config = component_config.get('csv_processing', {})
        self.elasticsearch_config = component_config.get('elasticsearch_config', {})
        self.mcp_server_config = component_config.get('mcp_server_config', {})
        
        # Initialize MCP client (will be connected during execution)
        self.mcp_client = None
        
        # Processing state
        self.import_progress = {
            'total_rows': 0,
            'processed_rows': 0,
            'errors': [],
            'start_time': None,
            'end_time': None,
            'status': 'initialized'
        }
        
        self.nb_logger.info(f"CSV Import Step {self.name} initialized",
                           csv_config=self.csv_processing_config,
                           es_config=self.elasticsearch_config)
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process CSV file import with comprehensive error handling"""
        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE,
            f"{self.name}.process",
            input_keys=list(input_data.keys()),
            step_type="CSVImportStep"
        ) as context:
            
            try:
                self.nb_logger.info(f"Starting CSV import process for step: {self.name}")
                self.import_progress['status'] = 'processing'
                self.import_progress['start_time'] = time.time()
                
                # Extract and validate input data
                csv_file_data = input_data.get('csv_file')
                import_config_data = input_data.get('import_config', {})
                
                if not csv_file_data:
                    raise ProcessingError("No CSV file provided for import")
                
                # Get file path from data unit
                if isinstance(csv_file_data, dict) and 'file_path' in csv_file_data:
                    csv_file_path = csv_file_data['file_path']
                elif hasattr(csv_file_data, 'file_path'):
                    csv_file_path = str(csv_file_data.file_path)
                else:
                    csv_file_path = str(csv_file_data)
                
                # Validate file exists
                if not Path(csv_file_path).exists():
                    raise ProcessingError(f"CSV file not found: {csv_file_path}")
                
                # Initialize MCP client connection
                await self._initialize_mcp_client()
                
                # Step 1: Analyze CSV structure
                self.nb_logger.info("Analyzing CSV file structure")
                structure_result = await self._analyze_csv_structure(csv_file_path)
                
                if not structure_result.get('success'):
                    raise ProcessingError(f"CSV analysis failed: {structure_result.get('error')}")
                
                # Step 2: Create Elasticsearch index
                index_name = self._generate_index_name(csv_file_path, import_config_data)
                self.nb_logger.info(f"Creating Elasticsearch index: {index_name}")
                
                index_result = await self._create_elasticsearch_index(index_name, structure_result['structure'])
                
                if not index_result.get('success'):
                    raise ProcessingError(f"Index creation failed: {index_result.get('error')}")
                
                # Step 3: Import CSV data
                self.nb_logger.info("Starting CSV data import")
                import_result = await self._import_csv_data(csv_file_path, index_name)
                
                if not import_result.get('success'):
                    raise ProcessingError(f"CSV import failed: {import_result.get('error')}")
                
                # Step 4: Generate results
                self.import_progress['status'] = 'completed'
                self.import_progress['end_time'] = time.time()
                self.import_progress['total_rows'] = import_result.get('rows_imported', 0)
                self.import_progress['processed_rows'] = import_result.get('rows_imported', 0)
                
                processing_time = self.import_progress['end_time'] - self.import_progress['start_time']
                
                # Prepare output data
                indexed_data = {
                    'index_name': index_name,
                    'structure': structure_result['structure'],
                    'mapping': index_result.get('mapping', {}),
                    'rows_imported': import_result.get('rows_imported', 0),
                    'file_path': csv_file_path,
                    'import_timestamp': time.time()
                }
                
                import_status = {
                    'status': 'success',
                    'processing_time': processing_time,
                    'rows_per_second': import_result.get('rows_imported', 0) / max(processing_time, 1),
                    'progress': self.import_progress.copy(),
                    'elasticsearch_info': {
                        'index_name': index_name,
                        'document_count': import_result.get('rows_imported', 0),
                        'index_size': index_result.get('index_size', 'unknown')
                    }
                }
                
                self.nb_logger.info("CSV import completed successfully",
                                   rows_imported=import_result.get('rows_imported', 0),
                                   processing_time=processing_time,
                                   index_name=index_name)
                
                return {
                    'indexed_data': indexed_data,
                    'import_status': import_status
                }
                
            except Exception as e:
                self.import_progress['status'] = 'failed'
                self.import_progress['end_time'] = time.time()
                self.import_progress['errors'].append(str(e))
                
                self.nb_logger.error(f"CSV import failed for step {self.name}: {e}",
                                   error_type=type(e).__name__,
                                   progress=self.import_progress)
                
                # Return error status
                return {
                    'indexed_data': None,
                    'import_status': {
                        'status': 'failed',
                        'error': str(e),
                        'progress': self.import_progress.copy()
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
                'tool_name': self.mcp_server_config.get('server_name', 'elasticsearch_csv_server'),
                'mcp_host': 'localhost',
                'mcp_port': 8080,
                'elasticsearch_host': 'localhost',
                'elasticsearch_port': 9200,
                'csv_processing_enabled': True,
                **self.csv_processing_config,
                **self.elasticsearch_config
            })
            
            # Create MCP server instance
            self.mcp_client = ElasticsearchMCPServer(mcp_config)
            
            self.nb_logger.debug("MCP client initialized successfully")
            
        except Exception as e:
            raise ProcessingError(f"Failed to initialize MCP client: {e}")
    
    async def _analyze_csv_structure(self, csv_file_path: str) -> Dict[str, Any]:
        """Analyze CSV file structure using MCP server"""
        try:
            return await self.mcp_client._analyze_csv_structure(csv_file_path)
        except Exception as e:
            return {
                'success': False,
                'error': f"CSV structure analysis failed: {e}"
            }
    
    async def _create_elasticsearch_index(self, index_name: str, csv_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Create Elasticsearch index with optimized mapping"""
        try:
            return await self.mcp_client._create_csv_index(index_name, csv_structure)
        except Exception as e:
            return {
                'success': False,
                'error': f"Index creation failed: {e}"
            }
    
    async def _import_csv_data(self, csv_file_path: str, index_name: str) -> Dict[str, Any]:
        """Import CSV data into Elasticsearch index"""
        try:
            return await self.mcp_client._import_csv_file(csv_file_path, index_name)
        except Exception as e:
            return {
                'success': False,
                'error': f"CSV data import failed: {e}"
            }
    
    def _generate_index_name(self, csv_file_path: str, import_config: Dict[str, Any]) -> str:
        """Generate Elasticsearch index name"""
        # Use custom index name if provided
        if import_config.get('index_name'):
            return import_config['index_name']
        
        # Generate from file name and prefix
        file_name = Path(csv_file_path).stem
        prefix = self.elasticsearch_config.get('index_prefix', 'csv_data_')
        
        # Clean file name for Elasticsearch index naming rules
        clean_name = file_name.lower().replace(' ', '_').replace('-', '_')
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
        
        return f"{prefix}{clean_name}"
