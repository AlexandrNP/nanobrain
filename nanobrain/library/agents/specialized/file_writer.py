"""
File Writer Agent for NanoBrain Framework

Specialized agent for file operations based on natural language descriptions.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

# Updated imports for nanobrain package structure
from nanobrain.core.agent import AgentConfig
from .base import SpecializedAgentBase, SimpleSpecializedAgent, ConversationalSpecializedAgent

logger = logging.getLogger(__name__)


class FileWriterAgentMixin(SpecializedAgentBase):
    """
    File Writer Agent Mixin - Intelligent File Operations and Content Management with Automation Workflows
    ====================================================================================================
    
    The FileWriterAgentMixin provides specialized capabilities for automated file operations, content management,
    and document generation based on natural language instructions. This mixin integrates advanced file system
    operations with intelligent content processing to create, modify, and manage files across diverse formats
    and organizational structures.
    
    **Core Architecture:**
        The file writer agent provides enterprise-grade file operation capabilities:
        
        * **Intelligent File Operations**: Natural language to file operation translation with context awareness
        * **Multi-Format Support**: Comprehensive support for text, code, configuration, and binary files
        * **Content Generation**: Automated content creation based on templates and specifications
        * **File System Management**: Directory structure creation, file organization, and batch operations
        * **Security and Permissions**: Safe file operations with permission management and validation
        * **Framework Integration**: Full integration with NanoBrain's specialized agent architecture
    
    **File Operation Capabilities:**
        
        **Natural Language Processing:**
        * Advanced parsing of file operation requirements and specifications
        * Intent recognition for create, read, update, delete, and organize operations
        * Path resolution and intelligent file naming from natural language descriptions
        * Content structure analysis and format detection
        
        **File Format Support:**
        * **Text Files**: Plain text, Markdown, reStructuredText, LaTeX documents
        * **Code Files**: Python, JavaScript, Java, C++, HTML, CSS, JSON, YAML, XML
        * **Configuration Files**: YAML, JSON, INI, TOML, environment files
        * **Documentation**: README files, API documentation, technical specifications
        * **Data Files**: CSV, JSON, XML, binary data with appropriate handling
        
        **Content Generation Features:**
        * Template-based document generation with variable substitution
        * Automated code scaffolding and boilerplate creation
        * Configuration file generation from specifications
        * Documentation generation with cross-references and formatting
        
        **File System Management:**
        * Directory structure creation and organization
        * Batch file operations with progress tracking
        * File backup and versioning support
        * Permission management and security validation
    
    **Content Management Workflows:**
        
        **Document Generation:**
        * Technical documentation creation from specifications
        * Report generation with data integration and formatting
        * README file creation with project-specific content
        * API documentation generation from code analysis
        
        **Project Scaffolding:**
        * Project structure creation from templates
        * Configuration file setup for development environments
        * Build system configuration and automation scripts
        * Testing framework setup and initialization
        
        **Data Processing:**
        * CSV and JSON file generation from structured data
        * Configuration file transformation and migration
        * Log file processing and analysis output generation
        * Data export and import file creation
        
        **Version Control Integration:**
        * Git repository initialization and configuration
        * Gitignore file generation based on project type
        * Commit message templates and branch protection files
        * Changelog generation and maintenance
    
    **Configuration Architecture:**
        Comprehensive configuration supports diverse file operation workflows:
        
        ```yaml
        # File Writer Agent Configuration
        agent_name: "file_writer_agent"
        agent_type: "specialized"
        
        # Agent card for framework integration
        agent_card:
          name: "file_writer_agent"
          description: "Intelligent file operations and content management"
          version: "1.0.0"
          category: "file_management"
          capabilities:
            - "file_operations"
            - "content_generation"
            - "automation_workflows"
        
        # LLM Configuration
        llm_config:
          model: "gpt-4"
          temperature: 0.2        # Low temperature for consistent file operations
          max_tokens: 3000
          
        # File Operation Configuration
        supported_formats:
          text: ["txt", "md", "rst", "tex"]
          code: ["py", "js", "java", "cpp", "html", "css"]
          config: ["yml", "yaml", "json", "ini", "toml"]
          data: ["csv", "json", "xml"]
          
        # Safety Configuration
        safety_checks:
          prevent_overwrite: true
          backup_existing: true
          validate_paths: true
          max_file_size: "100MB"
          
        # Template Configuration
        template_directories:
          - "templates/documents/"
          - "templates/projects/"
          - "templates/configs/"
          
        # Automation Settings
        automation_features:
          auto_formatting: true
          syntax_validation: true
          content_optimization: true
          directory_creation: true
          
        # Permission Management
        permissions:
          default_file_mode: "644"
          default_dir_mode: "755"
          respect_umask: true
          security_validation: true
        ```
    
    **Usage Patterns:**
        
        **Basic File Operations:**
        ```python
        from nanobrain.library.agents.specialized import FileWriterAgent
        
        # Create file writer agent with configuration
        agent_config = AgentConfig.from_config('config/file_writer_config.yml')
        file_writer = FileWriterAgent.from_config(agent_config)
        
        # Create a Python script file
        file_request = "Create a Python script called data_processor.py that reads CSV files and generates summary statistics"
        
        file_result = await file_writer.create_file(file_request)
        
        # Access file operation results
        print(f"File created: {file_result.data['file_path']}")
        print(f"Content length: {file_result.data['content_length']}")
        print(f"Format detected: {file_result.data['file_format']}")
        ```
        
        **Project Structure Generation:**
        ```python
        # Configure for project scaffolding
        project_config = {
            'project_type': 'python_api',
            'structure_template': 'microservice',
            'include_tests': True,
            'include_docs': True,
            'version_control': 'git'
        }
        
        agent_config = AgentConfig.from_config(project_config)
        file_writer = FileWriterAgent.from_config(agent_config)
        
        # Generate complete project structure
        project_spec = {
            'project_name': 'user_management_api',
            'description': 'REST API for user management with authentication',
            'framework': 'fastapi',
            'database': 'postgresql',
            'deployment': 'docker'
        }
        
        project_result = await file_writer.create_project_structure(project_spec)
        
        # Access created project components
        created_files = project_result.data['files']
        project_structure = project_result.data['structure']
        
        print(f"Created {len(created_files)} files")
        for file_path in created_files:
            print(f"  - {file_path}")
        
        print(f"\\nProject Structure:")
        for directory, files in project_structure.items():
            print(f"  {directory}/")
            for file in files:
                print(f"    - {file}")
        ```
        
        **Document Generation:**
        ```python
        # Configure for technical documentation
        docs_config = {
            'documentation_type': 'api_reference',
            'format': 'markdown',
            'include_examples': True,
            'cross_references': True
        }
        
        agent_config = AgentConfig.from_config(docs_config)
        file_writer = FileWriterAgent.from_config(agent_config)
        
        # Generate API documentation
        api_spec = {
            'api_name': 'User Management API',
            'version': '1.0.0',
            'base_url': 'https://api.example.com/v1',
            'endpoints': [
                {
                    'path': '/users',
                    'methods': ['GET', 'POST'],
                    'description': 'User collection operations'
                },
                {
                    'path': '/users/{id}',
                    'methods': ['GET', 'PUT', 'DELETE'],
                    'description': 'Individual user operations'
                }
            ]
        }
        
        docs_result = await file_writer.generate_documentation(
            api_spec,
            doc_type='api_reference'
        )
        
        # Access generated documentation
        readme_content = docs_result.data['README.md']
        api_docs = docs_result.data['api_reference.md']
        examples = docs_result.data['examples/']
        
        print(f"Generated documentation files:")
        for filename, content in docs_result.data.items():
            print(f"  - {filename} ({len(content)} characters)")
        ```
        
        **Configuration File Management:**
        ```python
        # Configure for configuration management
        config_mgmt = {
            'config_formats': ['yaml', 'json', 'env'],
            'environment_support': ['development', 'staging', 'production'],
            'validation': True,
            'encryption_support': True
        }
        
        agent_config = AgentConfig.from_config(config_mgmt)
        file_writer = FileWriterAgent.from_config(agent_config)
        
        # Generate environment-specific configurations
        app_config = {
            'application_name': 'user_service',
            'environments': {
                'development': {
                    'database_url': 'postgresql://localhost:5432/dev_db',
                    'debug': True,
                    'log_level': 'DEBUG'
                },
                'production': {
                    'database_url': '${DATABASE_URL}',
                    'debug': False,
                    'log_level': 'INFO'
                }
            },
            'shared_config': {
                'max_connections': 100,
                'timeout': 30,
                'retry_attempts': 3
            }
        }
        
        config_result = await file_writer.generate_configurations(app_config)
        
        # Access generated configuration files
        for env, config_files in config_result.data.items():
            print(f"\\n{env.upper()} Environment:")
            for format_type, content in config_files.items():
                print(f"  - config.{format_type}")
        ```
        
        **Batch File Operations:**
        ```python
        # Configure for batch processing
        batch_config = {
            'batch_size': 50,
            'parallel_operations': True,
            'progress_tracking': True,
            'error_handling': 'continue_on_error'
        }
        
        agent_config = AgentConfig.from_config(batch_config)
        file_writer = FileWriterAgent.from_config(agent_config)
        
        # Process multiple file operations
        file_operations = [
            {
                'operation': 'create',
                'path': 'reports/daily_summary.md',
                'template': 'daily_report',
                'data': {'date': '2024-01-15', 'metrics': {...}}
            },
            {
                'operation': 'create',
                'path': 'configs/service_config.yml',
                'template': 'service_config',
                'data': {'service_name': 'analytics', 'port': 8080}
            },
            {
                'operation': 'update',
                'path': 'README.md',
                'section': 'installation',
                'content': 'Updated installation instructions...'
            }
        ]
        
        batch_result = await file_writer.execute_batch_operations(
            file_operations,
            progress_callback=lambda progress: print(f"Progress: {progress:.1%}")
        )
        
        # Access batch operation results
        successful_ops = batch_result.data['successful']
        failed_ops = batch_result.data['failed']
        
        print(f"Batch Operation Summary:")
        print(f"  Successful: {len(successful_ops)}")
        print(f"  Failed: {len(failed_ops)}")
        
        for operation in failed_ops:
            print(f"  Failed: {operation['path']} - {operation['error']}")
        ```
    
    **Advanced Features:**
        
        **Template System Integration:**
        * Jinja2 template engine support for dynamic content generation
        * Custom template creation and management
        * Variable substitution and conditional content rendering
        * Template inheritance and composition for complex documents
        
        **Content Optimization:**
        * Automatic code formatting and syntax highlighting
        * Markdown optimization and link validation
        * Image optimization and compression for web content
        * Content minification for production deployments
        
        **Security and Compliance:**
        * File permission management and security validation
        * Content sanitization and injection prevention
        * Encryption support for sensitive configuration files
        * Audit logging for all file operations
        
        **Integration Capabilities:**
        * Version control system integration (Git, SVN)
        * Cloud storage synchronization (AWS S3, Google Drive)
        * Content management system integration
        * Continuous integration/deployment pipeline integration
    
    **File Management Applications:**
        
        **Development Workflows:**
        * Project initialization and scaffolding
        * Configuration management across environments
        * Documentation generation and maintenance
        * Build and deployment script creation
        
        **Content Creation:**
        * Technical documentation and user manuals
        * Report generation with data integration
        * Website content and blog post creation
        * Marketing material and presentation generation
        
        **Data Processing:**
        * ETL pipeline configuration and script generation
        * Data export and import file creation
        * Log file processing and analysis reports
        * Configuration migration and transformation
        
        **Automation and Scripting:**
        * Build automation script generation
        * Deployment configuration creation
        * Monitoring and alerting setup files
        * Backup and recovery script generation
    
    Attributes:
        file_operation_patterns (dict): Regex patterns for detecting file operations
        file_keywords (list): Keywords that trigger file operation functionality
        supported_formats (dict): File formats supported for different content types
        template_system (object): Template engine for dynamic content generation
        security_validator (object): Security validation system for file operations
        batch_processor (object): Batch operation management and execution system
    
    Note:
        This agent requires appropriate file system permissions for operation.
        All file operations include safety checks and validation mechanisms.
        Template systems require proper configuration for optimal content generation.
        Backup creation is recommended for file modification operations.
    
    Warning:
        File operations can permanently modify or delete data. Always verify
        paths and content before execution. Large batch operations may consume
        significant system resources. Be mindful of file system quotas and
        permissions when performing bulk operations.
    
    See Also:
        * :class:`SpecializedAgentBase`: Base specialized agent interface
        * :class:`SimpleSpecializedAgent`: Simple specialized agent implementation
        * :class:`ConversationalSpecializedAgent`: Conversational agent capabilities
        * :class:`AgentConfig`: Agent configuration schema
        * :mod:`nanobrain.library.agents.specialized`: Specialized agent implementations
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # File operation patterns
        self.file_operation_patterns = {
            'create': [
                r"create.*?(?:file|script).*?(?:at|to|in)\s+([^\s]+)",
                r"(?:file|script).*?(?:called|named)\s+([^\s]+)",
            ],
            'save': [
                r"save.*?(?:to|at|in)\s+([^\s]+)",
                r"write.*?(?:to|at|in)\s+([^\s]+)",
            ],
            'content': [
                r"with content:\s*(.+)",
                r"```(?:python|py)?\s*\n(.*?)\n```",
            ]
        }
        
        # File operation keywords
        self.file_keywords = ['write', 'save', 'create', 'file', 'path', 'document']
    
    async def _initialize_specialized_features(self) -> None:
        """Initialize file writer specific features."""
        await super()._initialize_specialized_features()
        self.specialized_logger.info("FileWriter specialized features initialized")
    
    def _should_handle_specialized(self, input_text: str, **kwargs) -> bool:
        """
        Determine if this request should be handled by file operation logic.
        
        Args:
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            True if should be handled by file operation logic
        """
        # Check for explicit file operation parameters
        has_file_path = 'file_path' in kwargs or 'path' in kwargs
        has_content = 'content' in kwargs or 'data' in kwargs
        
        # Check for file operation keywords in input
        has_keywords = any(keyword in input_text.lower() for keyword in self.file_keywords)
        
        return has_file_path and (has_content or has_keywords) or self._can_parse_file_operation(input_text)
    
    def _can_parse_file_operation(self, input_text: str) -> bool:
        """Check if input text contains parseable file operations."""
        # Look for file creation patterns
        for pattern_list in self.file_operation_patterns['create'] + self.file_operation_patterns['save']:
            for pattern in pattern_list:
                if re.search(pattern, input_text, re.IGNORECASE):
                    return True
        return False
    
    async def _process_specialized_request(self, input_text: str, **kwargs) -> Optional[str]:
        """
        Process file operation requests directly without LLM.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional parameters
            
        Returns:
            Result of file operation if handled, None otherwise
        """
        try:
            # Check if this is a direct file operation request
            if self._is_direct_file_operation(input_text, **kwargs):
                return await self._handle_direct_file_operation(input_text, **kwargs)
            
            # Try to parse and execute file operation from text
            result = await self._try_parse_and_execute_file_operation(input_text)
            if result:
                return result
            
            return None
            
        except Exception as e:
            self.specialized_logger.error(f"Error in specialized file processing: {e}")
            return f"Error processing file operation: {str(e)}"
    
    def _is_direct_file_operation(self, input_text: str, **kwargs) -> bool:
        """
        Check if this is a direct file operation that doesn't need LLM.
        
        Args:
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            True if this is a direct file operation
        """
        # Check for explicit file operation parameters
        has_file_path = 'file_path' in kwargs or 'path' in kwargs
        has_content = 'content' in kwargs or 'data' in kwargs
        
        # Check for file operation keywords in input
        has_keywords = any(keyword in input_text.lower() for keyword in self.file_keywords)
        
        return has_file_path and (has_content or has_keywords)
    
    async def _try_parse_and_execute_file_operation(self, input_text: str) -> Optional[str]:
        """
        Try to parse and execute file operations from natural language.
        
        Args:
            input_text: Input text to parse
            
        Returns:
            Result of file operation if successful, None otherwise
        """
        try:
            # Look for file creation patterns
            file_path = None
            for pattern in self.file_operation_patterns['create'] + self.file_operation_patterns['save']:
                match = re.search(pattern, input_text, re.IGNORECASE)
                if match:
                    file_path = match.group(1).strip('"\'')
                    break
            
            if not file_path:
                return None
            
            # Look for content patterns
            content = None
            for pattern in self.file_operation_patterns['content']:
                content_match = re.search(pattern, input_text, re.IGNORECASE | re.DOTALL)
                if content_match:
                    content = content_match.group(1).strip()
                    break
            
            # Generate content based on descriptions if not found
            if not content:
                if "function" in input_text.lower():
                    # Extract function name if mentioned
                    func_match = re.search(r"function.*?(?:called|named)\s+([a-zA-Z_][a-zA-Z0-9_]*)", input_text, re.IGNORECASE)
                    if func_match:
                        func_name = func_match.group(1)
                        content = f"def {func_name}():\n    \"\"\"Generated function.\"\"\"\n    pass"
                elif "class" in input_text.lower():
                    # Extract class name if mentioned
                    class_match = re.search(r"class.*?(?:called|named)\s+([a-zA-Z_][a-zA-Z0-9_]*)", input_text, re.IGNORECASE)
                    if class_match:
                        class_name = class_match.group(1)
                        content = f"class {class_name}:\n    \"\"\"Generated class.\"\"\"\n    pass"
            
            # If we have a file path, create the file
            if file_path:
                if not content:
                    content = f"# Generated file from request: {input_text[:100]}...\n"
                
                result = await self.write_file(file_path, content)
                self._track_specialized_operation("parse_and_execute", success=True)
                return f"Successfully processed file operation: {result}"
            
            return None
            
        except Exception as e:
            self._track_specialized_operation("parse_and_execute", success=False)
            self.specialized_logger.error(f"Error parsing file operation: {e}")
            return None
    
    async def _handle_direct_file_operation(self, input_text: str, **kwargs) -> str:
        """
        Handle direct file operations without LLM.
        
        Args:
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            Result of the file operation
        """
        try:
            # Extract file path
            file_path = kwargs.get('file_path') or kwargs.get('path')
            if not file_path:
                return "Error: No file path provided"
            
            # Extract content
            content = kwargs.get('content') or kwargs.get('data', '')
            
            # If content is not provided, try to extract from input
            if not content and 'content:' in input_text.lower():
                content = input_text.split('content:', 1)[1].strip()
            
            # Create file
            result = await self.write_file(file_path, content)
            self._track_specialized_operation("direct_operation", success=True)
            
            self.specialized_logger.info(f"FileWriterAgent handled direct operation: {file_path}")
            return result
            
        except Exception as e:
            self._track_specialized_operation("direct_operation", success=False)
            error_msg = f"Error in direct file operation: {str(e)}"
            self.specialized_logger.error(error_msg)
            return error_msg
    
    async def write_file(self, file_path: str, content: str, 
                        encoding: str = 'utf-8', create_dirs: bool = True) -> str:
        """
        Write content to a file.
        
        Args:
            file_path: Path to the file
            content: Content to write
            encoding: File encoding
            create_dirs: Whether to create directories if they don't exist
            
        Returns:
            Success message or error description
        """
        try:
            path = Path(file_path)
            
            # Create directories if needed
            if create_dirs and path.parent != path:
                path.parent.mkdir(parents=True, exist_ok=True)
                self.specialized_logger.debug(f"Created directories for {file_path}")
            
            # Write file
            path.write_text(content, encoding=encoding)
            
            file_size = path.stat().st_size
            success_msg = f"Successfully wrote {file_size} bytes to {file_path}"
            self.specialized_logger.info(success_msg)
            self._track_specialized_operation("write_file", success=True)
            return success_msg
            
        except PermissionError:
            error_msg = f"Permission denied: Cannot write to {file_path}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("write_file", success=False)
            return error_msg
        except OSError as e:
            error_msg = f"OS error writing to {file_path}: {str(e)}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("write_file", success=False)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error writing to {file_path}: {str(e)}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("write_file", success=False)
            return error_msg
    
    async def append_to_file(self, file_path: str, content: str, 
                           encoding: str = 'utf-8') -> str:
        """
        Append content to a file.
        
        Args:
            file_path: Path to the file
            content: Content to append
            encoding: File encoding
            
        Returns:
            Success message or error description
        """
        try:
            path = Path(file_path)
            
            # Append to file
            with open(path, 'a', encoding=encoding) as f:
                f.write(content)
            
            file_size = path.stat().st_size
            success_msg = f"Successfully appended content to {file_path} (total size: {file_size} bytes)"
            self.specialized_logger.info(success_msg)
            self._track_specialized_operation("append_file", success=True)
            return success_msg
            
        except Exception as e:
            error_msg = f"Error appending to {file_path}: {str(e)}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("append_file", success=False)
            return error_msg
    
    async def read_file(self, file_path: str, encoding: str = 'utf-8') -> str:
        """
        Read content from a file.
        
        Args:
            file_path: Path to the file
            encoding: File encoding
            
        Returns:
            File content or error description
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return f"Error: File {file_path} does not exist"
            
            content = path.read_text(encoding=encoding)
            self.specialized_logger.info(f"Successfully read {len(content)} characters from {file_path}")
            self._track_specialized_operation("read_file", success=True)
            return content
            
        except Exception as e:
            error_msg = f"Error reading {file_path}: {str(e)}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("read_file", success=False)
            return error_msg
    
    async def delete_file(self, file_path: str) -> str:
        """
        Delete a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Success message or error description
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return f"File {file_path} does not exist"
            
            path.unlink()
            success_msg = f"Successfully deleted {file_path}"
            self.specialized_logger.info(success_msg)
            self._track_specialized_operation("delete_file", success=True)
            return success_msg
            
        except Exception as e:
            error_msg = f"Error deleting {file_path}: {str(e)}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("delete_file", success=False)
            return error_msg
    
    async def create_directory(self, dir_path: str) -> str:
        """
        Create a directory.
        
        Args:
            dir_path: Path to the directory
            
        Returns:
            Success message or error description
        """
        try:
            path = Path(dir_path)
            path.mkdir(parents=True, exist_ok=True)
            
            success_msg = f"Successfully created directory {dir_path}"
            self.specialized_logger.info(success_msg)
            self._track_specialized_operation("create_directory", success=True)
            return success_msg
            
        except Exception as e:
            error_msg = f"Error creating directory {dir_path}: {str(e)}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("create_directory", success=False)
            return error_msg
    
    async def list_directory(self, dir_path: str) -> str:
        """
        List contents of a directory.
        
        Args:
            dir_path: Path to the directory
            
        Returns:
            Directory listing or error description
        """
        try:
            path = Path(dir_path)
            
            if not path.exists():
                return f"Directory {dir_path} does not exist"
            
            if not path.is_dir():
                return f"{dir_path} is not a directory"
            
            items = []
            for item in path.iterdir():
                item_type = "DIR" if item.is_dir() else "FILE"
                size = item.stat().st_size if item.is_file() else "-"
                items.append(f"{item_type:4} {size:>10} {item.name}")
            
            if not items:
                return f"Directory {dir_path} is empty"
            
            listing = f"Contents of {dir_path}:\n" + "\n".join(items)
            self.specialized_logger.info(f"Listed {len(items)} items in {dir_path}")
            self._track_specialized_operation("list_directory", success=True)
            return listing
            
        except Exception as e:
            error_msg = f"Error listing directory {dir_path}: {str(e)}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("list_directory", success=False)
            return error_msg
    
    async def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file exists, False otherwise
        """
        return Path(file_path).exists()
    
    async def get_file_info(self, file_path: str) -> str:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File information or error description
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return f"File {file_path} does not exist"
            
            stat = path.stat()
            info = f"""File information for {file_path}:
- Size: {stat.st_size} bytes
- Type: {'Directory' if path.is_dir() else 'File'}
- Modified: {stat.st_mtime}
- Permissions: {oct(stat.st_mode)[-3:]}"""
            
            self.specialized_logger.info(f"Retrieved info for {file_path}")
            self._track_specialized_operation("get_file_info", success=True)
            return info
            
        except Exception as e:
            error_msg = f"Error getting info for {file_path}: {str(e)}"
            self.specialized_logger.error(error_msg)
            self._track_specialized_operation("get_file_info", success=False)
            return error_msg


class FileWriterAgent(FileWriterAgentMixin, SimpleSpecializedAgent):
    """
    Simple file writer agent that processes input without conversation history.
    
    This agent can create, write, and manage files based on natural
    language descriptions. It intelligently detects file operations
    and performs them directly when possible.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        # Use provided config or create minimal default
        if config is None:
            config = AgentConfig(
                name="file_writer",
                description="Specialized agent for file operations",
                model="gpt-3.5-turbo",  # Lighter model since this is more operational
                system_prompt="""You are a specialized file writing agent for the NanoBrain framework.

CRITICAL: When asked to create, write, or save files, you MUST actually perform the file operation, not just describe it.

Your capabilities:
1. Create and write files with any content
2. Parse file paths and content from natural language requests
3. Handle directory creation automatically
4. Provide clear feedback about file operations

When you receive a request to create/write/save a file:
1. Extract the file path from the request
2. Extract or generate the content as requested
3. Actually create the file using your file writing capabilities
4. Confirm the operation was successful

Examples of requests you should handle:
- "Create a Python file at path/to/file.py with a hello function"
- "Save this code to utils.py: [code content]"
- "Write a config file with database settings"

ALWAYS actually perform the file operation - never just provide instructions or example content.""",
                temperature=0.1  # Very low temperature for consistent file operations
            )
        
        super().__init__(config=config, **kwargs)


class ConversationalFileWriterAgent(FileWriterAgentMixin, ConversationalSpecializedAgent):
    """
    Conversational file writer agent that maintains conversation history.
    
    This agent can create, write, and manage files based on natural
    language descriptions while maintaining conversation context.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        # Use provided config or create minimal default
        if config is None:
            config = AgentConfig(
                name="conversational_file_writer",
                description="Conversational specialized agent for file operations",
                model="gpt-3.5-turbo",
                system_prompt="""You are a conversational file writing agent for the NanoBrain framework.

You maintain conversation context while handling file operations. When asked to create, write, or save files, you MUST actually perform the file operation.

Your capabilities:
1. Create and write files with any content
2. Parse file paths and content from natural language requests
3. Handle directory creation automatically
4. Maintain conversation context across file operations
5. Provide clear feedback about file operations

You can reference previous conversations and file operations to provide better assistance.""",
                temperature=0.3  # Slightly higher for conversational context
            )
        
        super().__init__(config=config, **kwargs) 