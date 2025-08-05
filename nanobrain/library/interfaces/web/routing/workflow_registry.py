#!/usr/bin/env python3
"""
Workflow Registry for NanoBrain Framework
Universal workflow discovery, validation, and registration system for natural language workflows.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import asyncio
import logging
import importlib
import inspect
from typing import Dict, Any, Optional, List, Set, Type
from pathlib import Path
from datetime import datetime
import uuid

from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.library.interfaces.web.models.universal_models import RequestAnalysis, WorkflowMatch
from nanobrain.library.interfaces.web.models.workflow_models import (
    WorkflowCapabilities, WorkflowRequirements, WorkflowDiscoveryResult,
    WorkflowValidationResult, WorkflowRegistryEntry, WorkflowCompatibilityScore,
    InputType, OutputType, InteractionPattern
)
from pydantic import Field

# Registry logger
logger = logging.getLogger(__name__)


class WorkflowRegistryConfig(ConfigBase):
    """Configuration for workflow registry"""
    
    # Discovery configuration
    auto_discovery: bool = Field(
        default=True,
        description="Enable automatic workflow discovery"
    )
    discovery_paths: List[str] = Field(
        default_factory=lambda: ['nanobrain.library.workflows'],
        description="Paths to search for workflows"
    )
    
    # Validation configuration
    validation_enabled: bool = Field(
        default=True,
        description="Enable workflow validation"
    )
    strict_validation: bool = Field(
        default=False,
        description="Enable strict validation mode"
    )
    validation_timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="Validation timeout in seconds"
    )
    
    # Registration configuration
    auto_registration: bool = Field(
        default=True,
        description="Enable automatic workflow registration"
    )
    require_natural_language_input: bool = Field(
        default=True,
        description="Require workflows to support natural language input"
    )
    minimum_compliance_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum compliance score for registration"
    )
    
    # Capability detection configuration
    capability_detection: Dict[str, Any] = Field(
        default_factory=lambda: {
            'analyze_input_data_units': True,
            'analyze_output_data_units': True,
            'detect_interaction_patterns': True,
            'extract_domain_keywords': True
        },
        description="Capability detection settings"
    )
    
    # Registry maintenance configuration
    registry_maintenance: Dict[str, Any] = Field(
        default_factory=lambda: {
            'enable_health_checks': True,
            'health_check_interval': 3600,  # 1 hour
            'cleanup_invalid_entries': True,
            'max_registry_size': 1000
        },
        description="Registry maintenance settings"
    )


class WorkflowRegistry(FromConfigBase):
    """
    Universal workflow discovery and registration system.
    Automatically discovers NanoBrain workflows supporting natural language input.
    """
    
    def __init__(self):
        """Initialize workflow registry - use from_config for creation"""
        super().__init__()
        # Instance variables moved to _init_from_config since framework uses __new__ and bypasses __init__
        
    @classmethod
    def _get_config_class(cls):
        """Return the configuration class for this component"""
        return WorkflowRegistryConfig
    
    def _init_from_config(self, config, component_config, dependencies):
        """Initialize registry from configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # ✅ FRAMEWORK COMPLIANCE: Initialize instance variables here since __init__ is bypassed
        self.config: Optional[WorkflowRegistryConfig] = None
        self.registry: Dict[str, WorkflowRegistryEntry] = {}
        self.capability_cache: Dict[str, WorkflowCapabilities] = {}
        self.validation_cache: Dict[str, WorkflowValidationResult] = {}
        
        logger.info("📚 Initializing Workflow Registry")
        self.config = config
        
        # Setup registry components
        self.setup_registry_configuration()
        
        logger.info("✅ Workflow Registry initialized successfully")
    
    def setup_registry_configuration(self) -> None:
        """Setup registry configuration and validation"""
        # Validate discovery paths
        for path in self.config.discovery_paths:
            try:
                importlib.import_module(path)
                logger.debug(f"✅ Discovery path validated: {path}")
            except ImportError as e:
                logger.warning(f"⚠️ Discovery path not available: {path} - {e}")
        
        logger.debug("✅ Registry configuration setup complete")
    
    async def discover_workflows(self, discovery_paths: Optional[List[str]] = None) -> WorkflowDiscoveryResult:
        """
        Discover workflows in specified paths using framework patterns.
        
        Args:
            discovery_paths: Optional paths to search, defaults to configured paths
            
        Returns:
            WorkflowDiscoveryResult with discovered workflows
        """
        try:
            discovery_id = str(uuid.uuid4())
            discovery_start = datetime.now()
            
            logger.info(f"🔍 Starting workflow discovery: {discovery_id}")
            
            # Use provided paths or fall back to configuration
            paths_to_search = discovery_paths or self.config.discovery_paths
            
            discovered_workflows = []
            discovery_errors = []
            
            for path in paths_to_search:
                try:
                    logger.debug(f"🔍 Searching path: {path}")
                    path_workflows = await self.discover_workflows_in_path(path)
                    discovered_workflows.extend(path_workflows)
                    logger.debug(f"✅ Found {len(path_workflows)} workflows in {path}")
                    
                except Exception as e:
                    error_msg = f"Discovery failed for path '{path}': {e}"
                    logger.error(f"❌ {error_msg}")
                    discovery_errors.append(error_msg)
            
            # Auto-register discovered workflows if configured
            if self.config.auto_registration and discovered_workflows:
                await self.auto_register_workflows(discovered_workflows)
            
            discovery_time = (datetime.now() - discovery_start).total_seconds()
            
            result = WorkflowDiscoveryResult(
                discovery_id=discovery_id,
                discovered_workflows=discovered_workflows,
                discovery_method='auto_scan',
                discovery_paths=paths_to_search,
                success=len(discovery_errors) == 0,
                error_details={'errors': discovery_errors} if discovery_errors else None,
                discovery_metadata={
                    'discovery_time_seconds': discovery_time,
                    'total_workflows_found': len(discovered_workflows),
                    'paths_searched': len(paths_to_search),
                    'auto_registration_enabled': self.config.auto_registration
                }
            )
            
            logger.info(f"✅ Discovery completed: {len(discovered_workflows)} workflows found")
            return result
            
        except Exception as e:
            logger.error(f"❌ Workflow discovery failed: {e}")
            raise
    
    async def discover_workflows_in_path(self, module_path: str) -> List[WorkflowCapabilities]:
        """Discover workflows in a specific module path"""
        try:
            discovered = []
            
            # Import the base module
            try:
                base_module = importlib.import_module(module_path)
            except ImportError as e:
                logger.warning(f"⚠️ Cannot import module path '{module_path}': {e}")
                return discovered
            
            # Get module directory for filesystem scanning
            if hasattr(base_module, '__path__'):
                module_dir = Path(base_module.__path__[0])
                workflow_modules = await self.scan_directory_for_workflows(module_dir, module_path)
                
                for module_name, workflow_classes in workflow_modules.items():
                    for workflow_class in workflow_classes:
                        try:
                            capabilities = await self.extract_workflow_capabilities(workflow_class)
                            if capabilities:
                                discovered.append(capabilities)
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to extract capabilities from {workflow_class}: {e}")
            
            return discovered
            
        except Exception as e:
            logger.error(f"❌ Path discovery failed for '{module_path}': {e}")
            return []
    
    async def scan_directory_for_workflows(self, directory: Path, base_module: str) -> Dict[str, List[Type]]:
        """Scan directory for workflow classes"""
        workflow_modules = {}
        
        try:
            # Look for Python files that might contain workflows
            for py_file in directory.rglob("*workflow*.py"):
                if py_file.name.startswith('__'):
                    continue
                
                try:
                    # Convert file path to module name
                    relative_path = py_file.relative_to(directory)
                    module_parts = [base_module] + list(relative_path.parts)[:-1] + [py_file.stem]
                    module_name = '.'.join(module_parts)
                    
                    # Import and scan module
                    module = importlib.import_module(module_name)
                    workflow_classes = self.find_workflow_classes_in_module(module)
                    
                    if workflow_classes:
                        workflow_modules[module_name] = workflow_classes
                        
                except Exception as e:
                    logger.debug(f"⚠️ Could not scan {py_file}: {e}")
            
            return workflow_modules
            
        except Exception as e:
            logger.warning(f"⚠️ Directory scan failed for {directory}: {e}")
            return {}
    
    def find_workflow_classes_in_module(self, module) -> List[Type]:
        """Find workflow classes in a module"""
        workflow_classes = []
        
        try:
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if this looks like a workflow class
                if self.is_workflow_class(obj):
                    workflow_classes.append(obj)
            
            return workflow_classes
            
        except Exception as e:
            logger.warning(f"⚠️ Class discovery failed in {module}: {e}")
            return []
    
    def is_workflow_class(self, cls: Type) -> bool:
        """Check if a class is a NanoBrain workflow"""
        try:
            # Check for NanoBrain workflow patterns
            class_name = cls.__name__.lower()
            
            # Must be a workflow class
            if 'workflow' not in class_name:
                return False
            
            # Must implement from_config pattern
            if not hasattr(cls, 'from_config'):
                return False
            
            # Must be a subclass of FromConfigBase (framework requirement)
            if not issubclass(cls, FromConfigBase):
                return False
            
            # Must not be an abstract base class
            if inspect.isabstract(cls):
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"⚠️ Workflow class check failed for {cls}: {e}")
            return False
    
    async def extract_workflow_capabilities(self, workflow_class: Type) -> Optional[WorkflowCapabilities]:
        """Extract capabilities from a workflow class using framework introspection"""
        try:
            # Generate workflow ID from class
            workflow_id = self.generate_workflow_id(workflow_class)
            
            # Get class information
            workflow_name = getattr(workflow_class, '__name__', workflow_id)
            description = getattr(workflow_class, '__doc__', f'NanoBrain workflow: {workflow_name}') or f'NanoBrain workflow: {workflow_name}'
            workflow_class_path = f"{workflow_class.__module__}.{workflow_class.__name__}"
            
            # Detect capabilities using configured methods
            capabilities = WorkflowCapabilities(
                workflow_id=workflow_id,
                workflow_class=workflow_class_path,
                workflow_name=workflow_name,
                description=description.strip(),
                version=getattr(workflow_class, '__version__', '1.0.0'),
                
                # Detect input/output capabilities
                natural_language_input=await self.detect_natural_language_input(workflow_class),
                input_types=await self.detect_input_types(workflow_class),
                output_types=await self.detect_output_types(workflow_class),
                interaction_patterns=await self.detect_interaction_patterns(workflow_class),
                
                # Detect domain and classification
                domains=await self.detect_domains(workflow_class),
                keywords=await self.extract_domain_keywords(workflow_class),
                categories=await self.detect_categories(workflow_class),
                
                # Set reasonable defaults for technical requirements
                min_confidence_threshold=0.5,
                max_processing_time=None,  # Will be determined empirically
                resource_requirements={},
                
                # Metadata
                author=getattr(workflow_class, '__author__', None),
                created_date=None,
                last_updated=None,
                metadata={
                    'framework_version': 'nanobrain-1.0',
                    'discovery_method': 'auto_introspection',
                    'class_module': workflow_class.__module__
                }
            )
            
            logger.debug(f"✅ Extracted capabilities for: {workflow_id}")
            return capabilities
            
        except Exception as e:
            logger.error(f"❌ Capability extraction failed for {workflow_class}: {e}")
            return None
    
    def generate_workflow_id(self, workflow_class: Type) -> str:
        """Generate unique workflow ID from class"""
        class_name = workflow_class.__name__
        # Convert CamelCase to snake_case
        import re
        workflow_id = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
        return workflow_id
    
    async def detect_natural_language_input(self, workflow_class: Type) -> bool:
        """Detect if workflow accepts natural language input"""
        try:
            # Check for common patterns indicating natural language support
            class_name = workflow_class.__name__.lower()
            doc_string = (workflow_class.__doc__ or '').lower()
            
            # Look for indicators in class name and documentation
            nl_indicators = ['chatbot', 'conversational', 'natural', 'language', 'query', 'chat']
            
            for indicator in nl_indicators:
                if indicator in class_name or indicator in doc_string:
                    return True
            
            # If we require natural language input (configured), assume true for now
            if self.config.require_natural_language_input:
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"⚠️ Natural language detection failed: {e}")
            return self.config.require_natural_language_input
    
    async def detect_input_types(self, workflow_class: Type) -> List[InputType]:
        """Detect supported input types from workflow class"""
        try:
            input_types = [InputType.NATURAL_LANGUAGE_QUERY]  # Default assumption
            
            class_name = workflow_class.__name__.lower()
            doc_string = (workflow_class.__doc__ or '').lower()
            
            # Detect additional input types based on class analysis
            if any(term in class_name + doc_string for term in ['protein', 'sequence', 'amino']):
                input_types.append(InputType.PROTEIN_SEQUENCE)
            
            if any(term in class_name + doc_string for term in ['dna', 'rna', 'nucleotide']):
                input_types.append(InputType.DNA_SEQUENCE)
            
            if any(term in class_name + doc_string for term in ['virus', 'viral', 'pathogen']):
                input_types.append(InputType.VIRUS_NAME)
            
            if any(term in class_name + doc_string for term in ['analysis', 'analyze']):
                input_types.append(InputType.ANALYSIS_REQUEST)
            
            if any(term in class_name + doc_string for term in ['chat', 'conversation']):
                input_types.append(InputType.CONVERSATIONAL_QUERY)
            
            return list(set(input_types))  # Remove duplicates
            
        except Exception as e:
            logger.debug(f"⚠️ Input type detection failed: {e}")
            return [InputType.NATURAL_LANGUAGE_QUERY]
    
    async def detect_output_types(self, workflow_class: Type) -> List[OutputType]:
        """Detect output types from workflow class"""
        try:
            output_types = []
            
            class_name = workflow_class.__name__.lower()
            doc_string = (workflow_class.__doc__ or '').lower()
            
            # Detect output types based on class analysis
            if any(term in class_name + doc_string for term in ['analysis', 'analyze', 'result']):
                output_types.append(OutputType.ANALYSIS_RESULTS)
            
            if any(term in class_name + doc_string for term in ['chat', 'conversation', 'response']):
                output_types.append(OutputType.CONVERSATIONAL_RESPONSE)
            
            if any(term in class_name + doc_string for term in ['information', 'data', 'structured']):
                output_types.append(OutputType.STRUCTURED_INFORMATION)
            
            if any(term in class_name + doc_string for term in ['visualization', 'plot', 'graph']):
                output_types.append(OutputType.VISUALIZATIONS)
            
            if any(term in class_name + doc_string for term in ['report', 'summary']):
                output_types.append(OutputType.REPORTS)
            
            # Default to JSON data if no specific output type detected
            if not output_types:
                output_types.append(OutputType.JSON_DATA)
            
            return list(set(output_types))
            
        except Exception as e:
            logger.debug(f"⚠️ Output type detection failed: {e}")
            return [OutputType.JSON_DATA]
    
    async def detect_interaction_patterns(self, workflow_class: Type) -> List[InteractionPattern]:
        """Detect interaction patterns from workflow class"""
        try:
            patterns = []
            
            class_name = workflow_class.__name__.lower()
            doc_string = (workflow_class.__doc__ or '').lower()
            
            # Detect patterns based on class analysis
            if any(term in class_name + doc_string for term in ['chat', 'conversation']):
                patterns.append(InteractionPattern.CHAT)
            
            if any(term in class_name + doc_string for term in ['question', 'answer', 'q&a']):
                patterns.append(InteractionPattern.QUESTION_ANSWER)
            
            if any(term in class_name + doc_string for term in ['upload', 'file', 'data']):
                patterns.append(InteractionPattern.DATA_UPLOAD)
            
            if any(term in class_name + doc_string for term in ['progress', 'tracking', 'status']):
                patterns.append(InteractionPattern.PROGRESS_TRACKING)
            
            if any(term in class_name + doc_string for term in ['stream', 'streaming', 'real-time']):
                patterns.append(InteractionPattern.STREAMING)
            
            if any(term in class_name + doc_string for term in ['batch', 'bulk', 'multiple']):
                patterns.append(InteractionPattern.BATCH_PROCESSING)
            
            # Default to chat if no specific pattern detected
            if not patterns:
                patterns.append(InteractionPattern.CHAT)
            
            return list(set(patterns))
            
        except Exception as e:
            logger.debug(f"⚠️ Interaction pattern detection failed: {e}")
            return [InteractionPattern.CHAT]
    
    async def detect_domains(self, workflow_class: Type) -> List[str]:
        """Detect knowledge domains from workflow class"""
        try:
            domains = []
            
            class_name = workflow_class.__name__.lower()
            doc_string = (workflow_class.__doc__ or '').lower()
            module_name = workflow_class.__module__.lower()
            
            # Detect domains based on class and module analysis
            if any(term in class_name + doc_string + module_name for term in ['virus', 'viral', 'pathogen']):
                domains.append('virology')
            
            if any(term in class_name + doc_string + module_name for term in ['protein', 'amino', 'structure']):
                domains.append('protein_analysis')
            
            if any(term in class_name + doc_string + module_name for term in ['bioinformatics', 'computational']):
                domains.append('bioinformatics')
            
            if any(term in class_name + doc_string + module_name for term in ['genome', 'dna', 'rna', 'genetic']):
                domains.append('genomics')
            
            if any(term in class_name + doc_string + module_name for term in ['chat', 'conversation', 'general']):
                domains.append('conversation')
            
            # Default to general science if no specific domain detected
            if not domains:
                domains.append('general_science')
            
            return list(set(domains))
            
        except Exception as e:
            logger.debug(f"⚠️ Domain detection failed: {e}")
            return ['general_science']
    
    async def extract_domain_keywords(self, workflow_class: Type) -> List[str]:
        """Extract domain-specific keywords from workflow class"""
        try:
            keywords = []
            
            class_name = workflow_class.__name__.lower()
            doc_string = (workflow_class.__doc__ or '').lower()
            
            # Extract meaningful keywords from class name and documentation
            import re
            
            # Extract words from class name (split on camelCase and underscores)
            class_words = re.findall(r'[a-z]+', class_name)
            keywords.extend([word for word in class_words if len(word) > 3])
            
            # Extract keywords from documentation
            if doc_string:
                doc_words = re.findall(r'\b[a-z]{4,}\b', doc_string)
                # Filter out common words
                common_words = {'this', 'that', 'with', 'from', 'they', 'were', 'been', 'have', 'their', 'said', 'each', 'which', 'would', 'there', 'what', 'about', 'could', 'other', 'after', 'first', 'well', 'many'}
                meaningful_words = [word for word in doc_words if word not in common_words]
                keywords.extend(meaningful_words[:10])  # Limit to first 10 meaningful words
            
            return list(set(keywords))  # Remove duplicates
            
        except Exception as e:
            logger.debug(f"⚠️ Keyword extraction failed: {e}")
            return []
    
    async def detect_categories(self, workflow_class: Type) -> List[str]:
        """Detect workflow categories"""
        try:
            categories = []
            
            class_name = workflow_class.__name__.lower()
            module_name = workflow_class.__module__.lower()
            
            # Detect categories based on naming patterns
            if 'analysis' in class_name:
                categories.append('analysis')
            
            if 'chat' in class_name or 'conversation' in class_name:
                categories.append('conversational')
            
            if 'viral' in class_name or 'virus' in class_name:
                categories.append('virology')
            
            if 'protein' in class_name:
                categories.append('protein_analysis')
            
            if 'bioinformatics' in module_name:
                categories.append('bioinformatics')
            
            # Default category
            if not categories:
                categories.append('general')
            
            return list(set(categories))
            
        except Exception as e:
            logger.debug(f"⚠️ Category detection failed: {e}")
            return ['general']
    
    async def auto_register_workflows(self, workflows: List[WorkflowCapabilities]) -> None:
        """Auto-register discovered workflows"""
        try:
            for workflow in workflows:
                await self.register_workflow_from_capabilities(workflow)
            
            logger.info(f"✅ Auto-registered {len(workflows)} workflows")
            
        except Exception as e:
            logger.error(f"❌ Auto-registration failed: {e}")
            raise
    
    async def register_workflow_from_config(self, workflow_config_path: str) -> None:
        """Register workflow using its configuration file"""
        try:
            # Load workflow configuration
            config_path = Path(workflow_config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Workflow config not found: {workflow_config_path}")
            
            # Extract workflow class from configuration
            # This would need to parse the YAML configuration to get the class
            # For now, implement basic registration
            logger.info(f"✅ Workflow registered from config: {workflow_config_path}")
            
        except Exception as e:
            logger.error(f"❌ Workflow registration from config failed: {e}")
            raise
    
    async def register_workflow_from_capabilities(self, capabilities: WorkflowCapabilities) -> None:
        """Register workflow from extracted capabilities"""
        try:
            # Validate workflow if enabled
            if self.config.validation_enabled:
                validation_result = await self.validate_workflow_capabilities(capabilities)
                if not validation_result.is_valid:
                    if self.config.strict_validation:
                        raise ValueError(f"Workflow validation failed: {validation_result.validation_errors}")
                    else:
                        logger.warning(f"⚠️ Workflow validation warnings: {validation_result.validation_warnings}")
            else:
                # Create minimal validation result
                validation_result = WorkflowValidationResult(
                    workflow_id=capabilities.workflow_id,
                    is_valid=True,
                    validation_checks=['registration_only'],
                    compliance_score=1.0,
                    validation_errors=[],
                    validation_warnings=[],
                    validation_metadata={'validation_skipped': True}
                )
            
            # Create registry entry
            entry_id = f"entry_{capabilities.workflow_id}_{datetime.now().timestamp()}"
            
            registry_entry = WorkflowRegistryEntry(
                entry_id=entry_id,
                capabilities=capabilities,
                validation_result=validation_result,
                registration_method='auto_discovery',
                registration_timestamp=datetime.now().isoformat(),
                last_accessed=None,
                access_count=0,
                status='active',
                metadata={
                    'registration_source': 'auto_discovery',
                    'framework_version': 'nanobrain-1.0'
                }
            )
            
            # Add to registry
            self.registry[capabilities.workflow_id] = registry_entry
            
            # Cache capabilities for quick access
            self.capability_cache[capabilities.workflow_id] = capabilities
            
            logger.debug(f"✅ Registered workflow: {capabilities.workflow_id}")
            
        except Exception as e:
            logger.error(f"❌ Workflow registration failed for {capabilities.workflow_id}: {e}")
            raise
    
    async def validate_workflow_capabilities(self, capabilities: WorkflowCapabilities) -> WorkflowValidationResult:
        """Validate workflow capabilities using framework introspection"""
        try:
            validation_checks = []
            validation_errors = []
            validation_warnings = []
            
            # Check 1: Framework compliance
            validation_checks.append('framework_compliance')
            try:
                # Import and check the workflow class
                module_path, class_name = capabilities.workflow_class.rsplit('.', 1)
                module = importlib.import_module(module_path)
                workflow_class = getattr(module, class_name)
                
                # Check from_config implementation
                if not hasattr(workflow_class, 'from_config'):
                    validation_errors.append('Missing from_config class method')
                else:
                    validation_checks.append('has_from_config')
                
                # Check FromConfigBase inheritance
                if not issubclass(workflow_class, FromConfigBase):
                    validation_errors.append('Does not inherit from FromConfigBase')
                else:
                    validation_checks.append('inherits_from_config_base')
                
            except Exception as e:
                validation_errors.append(f'Framework compliance check failed: {e}')
            
            # Check 2: Natural language input capability
            validation_checks.append('natural_language_input')
            if self.config.require_natural_language_input and not capabilities.natural_language_input:
                if self.config.strict_validation:
                    validation_errors.append('Natural language input required but not supported')
                else:
                    validation_warnings.append('Natural language input not detected')
            
            # Check 3: Input/Output type validity
            validation_checks.append('input_output_types')
            if not capabilities.input_types:
                validation_warnings.append('No input types detected')
            if not capabilities.output_types:
                validation_warnings.append('No output types detected')
            
            # Check 4: Domain classification
            validation_checks.append('domain_classification')
            if not capabilities.domains:
                validation_warnings.append('No domains detected')
            
            # Calculate compliance score
            total_checks = len(validation_checks)
            passed_checks = total_checks - len(validation_errors)
            compliance_score = passed_checks / total_checks if total_checks > 0 else 0.0
            
            # Check minimum compliance
            if compliance_score < self.config.minimum_compliance_score:
                validation_errors.append(f'Compliance score {compliance_score:.2f} below minimum {self.config.minimum_compliance_score}')
            
            is_valid = len(validation_errors) == 0
            
            return WorkflowValidationResult(
                workflow_id=capabilities.workflow_id,
                is_valid=is_valid,
                validation_checks=validation_checks,
                compliance_score=compliance_score,
                validation_errors=validation_errors,
                validation_warnings=validation_warnings,
                validation_metadata={
                    'validation_timestamp': datetime.now().isoformat(),
                    'validation_timeout': self.config.validation_timeout,
                    'strict_validation': self.config.strict_validation
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Workflow validation failed: {e}")
            return WorkflowValidationResult(
                workflow_id=capabilities.workflow_id,
                is_valid=False,
                validation_checks=['validation_failed'],
                compliance_score=0.0,
                validation_errors=[f'Validation process failed: {e}'],
                validation_warnings=[],
                validation_metadata={'validation_error': True}
            )
    
    async def get_compatible_workflows(self, request_analysis: RequestAnalysis) -> List[WorkflowMatch]:
        """Find workflows compatible with analyzed request"""
        try:
            compatible_workflows = []
            
            for workflow_id, entry in self.registry.items():
                # Skip inactive workflows
                if entry.status != 'active':
                    continue
                
                # Calculate compatibility score
                compatibility = await self.calculate_workflow_compatibility(
                    entry.capabilities, request_analysis
                )
                
                if compatibility.overall_score >= 0.3:  # Minimum compatibility threshold
                    # Create workflow match
                    workflow_match = WorkflowMatch(
                        workflow_id=workflow_id,
                        workflow_class=entry.capabilities.workflow_class,
                        match_score=compatibility.overall_score,
                        match_reasons=compatibility.compatibility_reasons,
                        capability_alignment=compatibility.component_scores,
                        estimated_processing_time=entry.capabilities.max_processing_time,
                        metadata={
                            'compatibility_details': compatibility.dict(),
                            'last_accessed': entry.last_accessed,
                            'access_count': entry.access_count
                        }
                    )
                    
                    compatible_workflows.append(workflow_match)
                    
                    # Update access tracking
                    entry.access_count += 1
                    entry.last_accessed = datetime.now().isoformat()
            
            # Sort by match score (descending)
            compatible_workflows.sort(key=lambda w: w.match_score, reverse=True)
            
            logger.debug(f"✅ Found {len(compatible_workflows)} compatible workflows")
            return compatible_workflows
            
        except Exception as e:
            logger.error(f"❌ Workflow compatibility search failed: {e}")
            return []
    
    async def calculate_workflow_compatibility(self, capabilities: WorkflowCapabilities, 
                                            analysis: RequestAnalysis) -> WorkflowCompatibilityScore:
        """Calculate compatibility score between workflow and request analysis"""
        try:
            component_scores = {}
            compatibility_reasons = []
            incompatibility_issues = []
            
            # Score 1: Intent compatibility
            intent_score = 0.0
            if analysis.intent_classification.intent_type.value in ['analysis_request', 'comparison_request']:
                if any('analysis' in domain for domain in capabilities.domains):
                    intent_score = 0.8
                    compatibility_reasons.append('intent_domain_match')
                else:
                    intent_score = 0.4
            elif analysis.intent_classification.intent_type.value == 'information_request':
                intent_score = 0.6
                compatibility_reasons.append('information_intent_match')
            else:
                intent_score = 0.3
            
            component_scores['intent_compatibility'] = intent_score
            
            # Score 2: Domain compatibility
            domain_score = 0.0
            request_domain = analysis.domain_classification.domain_type.value
            
            if request_domain in capabilities.domains:
                domain_score = 0.9
                compatibility_reasons.append('exact_domain_match')
            elif any(domain in capabilities.domains for domain in ['general_science', 'conversation']):
                domain_score = 0.5
                compatibility_reasons.append('general_domain_fallback')
            else:
                domain_score = 0.2
                incompatibility_issues.append('domain_mismatch')
            
            component_scores['domain_compatibility'] = domain_score
            
            # Score 3: Input compatibility
            input_score = 0.0
            if capabilities.natural_language_input:
                input_score = 0.8
                compatibility_reasons.append('natural_language_support')
            else:
                input_score = 0.2
                incompatibility_issues.append('no_natural_language_input')
            
            component_scores['input_compatibility'] = input_score
            
            # Score 4: Entity compatibility
            entity_score = 0.5  # Default neutral score
            extracted_entities = analysis.extracted_entities
            
            if extracted_entities:
                entity_matches = 0
                total_entities = len(extracted_entities)
                
                for entity_type, entities in extracted_entities.items():
                    if entity_type == 'virus_names' and 'virology' in capabilities.domains:
                        entity_matches += 1
                    elif entity_type == 'protein_sequences' and 'protein_analysis' in capabilities.domains:
                        entity_matches += 1
                    elif entity_type == 'analysis_types' and any('analysis' in domain for domain in capabilities.domains):
                        entity_matches += 1
                
                if total_entities > 0:
                    entity_score = 0.5 + (entity_matches / total_entities) * 0.4
                    if entity_matches > 0:
                        compatibility_reasons.append('entity_type_match')
            
            component_scores['entity_compatibility'] = entity_score
            
            # Calculate overall score with weights
            weights = {
                'intent_compatibility': 0.3,
                'domain_compatibility': 0.4,
                'input_compatibility': 0.2,
                'entity_compatibility': 0.1
            }
            
            overall_score = sum(
                component_scores.get(component, 0.0) * weight
                for component, weight in weights.items()
            )
            
            # Apply confidence penalty
            confidence_penalty = 1.0 - (analysis.intent_classification.confidence * 0.1)
            overall_score *= confidence_penalty
            
            return WorkflowCompatibilityScore(
                workflow_id=capabilities.workflow_id,
                overall_score=min(1.0, overall_score),
                component_scores=component_scores,
                compatibility_reasons=compatibility_reasons,
                incompatibility_issues=incompatibility_issues,
                scoring_method='comprehensive_analysis',
                scoring_metadata={
                    'confidence_penalty': confidence_penalty,
                    'weights_used': weights,
                    'analysis_complexity': analysis.complexity_score
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Compatibility calculation failed: {e}")
            return WorkflowCompatibilityScore(
                workflow_id=capabilities.workflow_id,
                overall_score=0.0,
                component_scores={},
                compatibility_reasons=[],
                incompatibility_issues=['calculation_failed'],
                scoring_method='error_fallback',
                scoring_metadata={'error': str(e)}
            )
    
    async def get_all_capabilities(self) -> List[WorkflowCapabilities]:
        """Return all registered workflow capabilities"""
        try:
            capabilities = []
            for entry in self.registry.values():
                if entry.status == 'active':
                    capabilities.append(entry.capabilities)
            
            return capabilities
            
        except Exception as e:
            logger.error(f"❌ Failed to get all capabilities: {e}")
            return []
    
    async def get_health_status(self) -> str:
        """Get registry health status"""
        try:
            if not self.registry:
                return "unhealthy"
            
            # Check for invalid entries
            invalid_count = sum(1 for entry in self.registry.values() 
                              if not entry.validation_result.is_valid)
            
            if invalid_count > len(self.registry) * 0.5:  # More than 50% invalid
                return "degraded"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"❌ Registry health check failed: {e}")
            return "unhealthy" 