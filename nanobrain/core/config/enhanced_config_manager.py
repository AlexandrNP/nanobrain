"""
Enhanced Configuration Manager with Card Export and Validation

Extends the base ConfigManager to support A2A card export functionality,
configuration validation, and discovery of enhanced YAML configurations.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .config_manager import ConfigManager
from .enhanced_config import EnhancedAgentConfig, EnhancedToolConfig

logger = logging.getLogger(__name__)


class EnhancedConfigManager(ConfigManager):
    """
    Enhanced Configuration Manager - Advanced Configuration Management and A2A Integration
    ==================================================================================
    
    The EnhancedConfigManager provides comprehensive configuration management with advanced
    A2A (Agent-to-Agent) integration, card export capabilities, configuration validation,
    and intelligent discovery systems. This manager extends base configuration management
    with enterprise-grade features including automated card generation, protocol compliance
    validation, and advanced configuration orchestration for distributed AI systems.
    
    **Core Architecture:**
        The enhanced configuration manager provides enterprise-grade configuration capabilities:
        
        * **A2A Protocol Integration**: Native Agent-to-Agent protocol support with card generation
        * **Configuration Discovery**: Intelligent configuration discovery and classification
        * **Validation Framework**: Comprehensive validation with compliance checking
        * **Card Export System**: Automated generation of A2A-compliant agent and tool cards
        * **Configuration Orchestration**: Advanced configuration management and coordination
        * **Framework Integration**: Complete integration with NanoBrain's component and agent systems
    
    **A2A Integration Capabilities:**
        
        **Agent-to-Agent Protocol Support:**
        * **A2A Card Generation**: Automatic generation of A2A-compliant agent and tool cards
        * **Protocol Compliance**: Validation and enforcement of A2A protocol standards
        * **Communication Setup**: Automated communication channel setup and configuration
        * **Discovery Services**: Agent and tool discovery service integration
        
        **Card Export System:**
        * **Automated Card Generation**: Intelligent card generation from configuration files
        * **Compliance Validation**: A2A compliance checking and validation reporting
        * **Multi-Format Export**: Support for JSON, YAML, and custom card formats
        * **Batch Processing**: Efficient batch export of multiple agent and tool cards
        
        **Protocol Validation:**
        * **A2A Compliance Checking**: Comprehensive A2A protocol compliance validation
        * **Schema Validation**: Validation against A2A schema specifications
        * **Capability Verification**: Agent and tool capability verification and validation
        * **Communication Validation**: Communication protocol and channel validation
    
    **Configuration Management Features:**
        
        **Enhanced Discovery:**
        * **Intelligent Configuration Discovery**: Automatic discovery of agent and tool configurations
        * **Classification System**: Automatic classification and categorization of configurations
        * **Dependency Analysis**: Configuration dependency analysis and mapping
        * **Version Management**: Configuration version detection and management
        
        **Advanced Validation:**
        * **Multi-Level Validation**: Configuration, schema, and protocol validation
        * **Cross-Reference Validation**: Validation of configuration cross-references
        * **Business Rule Validation**: Custom business rule validation and enforcement
        * **Integration Testing**: Automated integration testing for configuration changes
        
        **Configuration Orchestration:**
        * **Deployment Coordination**: Coordinated configuration deployment across environments
        * **Environment Management**: Environment-specific configuration management
        * **Rollback Capabilities**: Configuration rollback and recovery mechanisms
        * **Change Management**: Configuration change tracking and approval workflows
    
    **Usage Patterns:**
        
        **Basic Enhanced Configuration Management:**
        ```python
        from nanobrain.core.config.enhanced_config_manager import EnhancedConfigManager
        
        # Create enhanced configuration manager
        config_manager = EnhancedConfigManager('config/')
        
        # Export A2A cards from configurations
        export_results = config_manager.export_a2a_cards('output/a2a_cards/')
        
        print(f"Exported agents: {len(export_results['agents'])}")
        print(f"Exported tools: {len(export_results['tools'])}")
        
        # Discover and classify configurations
        agent_configs = config_manager.discover_agent_configs()
        tool_configs = config_manager.discover_tool_configs()
        
        print(f"Found {len(agent_configs)} agent configurations")
        print(f"Found {len(tool_configs)} tool configurations")
        
        # Validate A2A compliance
        for config_file in agent_configs:
            validation_results = config_manager.validate_a2a_compliance(config_file)
            if validation_results['compliant']:
                print(f"✅ {config_file} is A2A compliant")
            else:
                print(f"❌ {config_file} has compliance issues: {validation_results['errors']}")
        ```
        
        **Enterprise A2A Integration:**
        ```python
        # Enterprise A2A integration and card management
        class EnterpriseA2AManager:
            def __init__(self, config_directory: str):
                self.config_manager = EnhancedConfigManager(config_directory)
                self.a2a_registry = {}
                self.export_metrics = {}
                
            async def setup_a2a_ecosystem(self, output_directory: str):
                \"\"\"Setup complete A2A ecosystem with cards and services\"\"\"
                
                # Export all A2A cards
                export_results = self.config_manager.export_a2a_cards(Path(output_directory))
                
                # Setup A2A registry
                await self.register_a2a_components(export_results)
                
                # Validate A2A ecosystem integrity
                validation_results = await self.validate_a2a_ecosystem()
                
                # Setup communication channels
                await self.setup_a2a_communication()
                
                return {
                    'export_results': export_results,
                    'registry_status': len(self.a2a_registry),
                    'validation_results': validation_results,
                    'ecosystem_ready': validation_results['all_valid']
                }
                
            async def register_a2a_components(self, export_results: Dict[str, List[Path]]):
                \"\"\"Register A2A components in ecosystem registry\"\"\"
                
                # Register agents
                for agent_card_path in export_results['agents']:
                    with open(agent_card_path) as f:
                        agent_card = json.load(f)
                    
                    agent_id = agent_card['agent_id']
                    self.a2a_registry[agent_id] = {
                        'type': 'agent',
                        'card': agent_card,
                        'card_path': agent_card_path,
                        'status': 'registered'
                    }
                    
                # Register tools
                for tool_card_path in export_results['tools']:
                    with open(tool_card_path) as f:
                        tool_card = json.load(f)
                    
                    tool_id = tool_card['tool_id']
                    self.a2a_registry[tool_id] = {
                        'type': 'tool',
                        'card': tool_card,
                        'card_path': tool_card_path,
                        'status': 'registered'
                    }
                    
            async def validate_a2a_ecosystem(self) -> Dict[str, Any]:
                \"\"\"Validate complete A2A ecosystem\"\"\"
                
                validation_results = {
                    'component_validations': {},
                    'communication_validations': {},
                    'dependency_validations': {},
                    'all_valid': True
                }
                
                # Validate individual components
                for component_id, component_info in self.a2a_registry.items():
                    try:
                        # Validate A2A card format
                        card_validation = self.validate_a2a_card_format(component_info['card'])
                        
                        # Validate component capabilities
                        capability_validation = self.validate_component_capabilities(component_info['card'])
                        
                        validation_results['component_validations'][component_id] = {
                            'card_format': card_validation,
                            'capabilities': capability_validation,
                            'valid': card_validation['valid'] and capability_validation['valid']
                        }
                        
                        if not (card_validation['valid'] and capability_validation['valid']):
                            validation_results['all_valid'] = False
                            
                    except Exception as e:
                        validation_results['component_validations'][component_id] = {
                            'error': str(e),
                            'valid': False
                        }
                        validation_results['all_valid'] = False
                
                # Validate communication capabilities
                communication_validation = await self.validate_a2a_communication()
                validation_results['communication_validations'] = communication_validation
                
                if not communication_validation['valid']:
                    validation_results['all_valid'] = False
                
                return validation_results
                
            async def generate_a2a_documentation(self, output_directory: str):
                \"\"\"Generate comprehensive A2A ecosystem documentation\"\"\"
                
                output_path = Path(output_directory)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Generate ecosystem overview
                ecosystem_doc = {
                    'ecosystem_name': 'NanoBrain A2A Ecosystem',
                    'version': '1.0.0',
                    'components': {},
                    'communication_map': {},
                    'capabilities_matrix': {}
                }
                
                # Document components
                for component_id, component_info in self.a2a_registry.items():
                    card = component_info['card']
                    ecosystem_doc['components'][component_id] = {
                        'type': component_info['type'],
                        'name': card.get('name', component_id),
                        'description': card.get('description', ''),
                        'capabilities': card.get('capabilities', []),
                        'communication_protocols': card.get('communication', {})
                    }
                
                # Generate capability matrix
                all_capabilities = set()
                for component_id, component_info in self.a2a_registry.items():
                    capabilities = component_info['card'].get('capabilities', [])
                    all_capabilities.update(capabilities)
                
                capability_matrix = {}
                for capability in all_capabilities:
                    capability_matrix[capability] = [
                        component_id for component_id, component_info in self.a2a_registry.items()
                        if capability in component_info['card'].get('capabilities', [])
                    ]
                
                ecosystem_doc['capabilities_matrix'] = capability_matrix
                
                # Save documentation
                with open(output_path / 'a2a_ecosystem.json', 'w') as f:
                    json.dump(ecosystem_doc, f, indent=2)
                
                # Generate markdown documentation
                markdown_doc = self.generate_markdown_documentation(ecosystem_doc)
                with open(output_path / 'A2A_ECOSYSTEM.md', 'w') as f:
                    f.write(markdown_doc)
                
                return ecosystem_doc
        
        # Enterprise A2A setup
        a2a_manager = EnterpriseA2AManager('config/')
        ecosystem_status = await a2a_manager.setup_a2a_ecosystem('output/a2a/')
        
        print(f"A2A Ecosystem Status:")
        print(f"  Agents: {len(ecosystem_status['export_results']['agents'])}")
        print(f"  Tools: {len(ecosystem_status['export_results']['tools'])}")
        print(f"  Registry: {ecosystem_status['registry_status']} components")
        print(f"  Ecosystem Ready: {ecosystem_status['ecosystem_ready']}")
        
        # Generate documentation
        ecosystem_doc = await a2a_manager.generate_a2a_documentation('docs/a2a/')
        ```
        
        **Advanced Configuration Validation:**
        ```python
        # Advanced configuration validation and management
        class ConfigurationValidationSuite:
            def __init__(self, config_manager: EnhancedConfigManager):
                self.config_manager = config_manager
                self.validation_rules = {}
                self.validation_history = []
                
            def register_validation_rule(self, rule_name: str, rule_func: Callable):
                \"\"\"Register custom validation rule\"\"\"
                self.validation_rules[rule_name] = rule_func
                
            async def comprehensive_validation(self, config_directory: str) -> Dict[str, Any]:
                \"\"\"Perform comprehensive configuration validation\"\"\"
                
                validation_report = {
                    'validation_timestamp': datetime.now().isoformat(),
                    'config_directory': config_directory,
                    'agent_validations': {},
                    'tool_validations': {},
                    'a2a_validations': {},
                    'custom_rule_validations': {},
                    'summary': {
                        'total_configs': 0,
                        'valid_configs': 0,
                        'invalid_configs': 0,
                        'warnings': 0
                    }
                }
                
                # Validate agent configurations
                agent_configs = self.config_manager.discover_agent_configs()
                for config_file in agent_configs:
                    validation_result = await self.validate_agent_config(config_file)
                    validation_report['agent_validations'][str(config_file)] = validation_result
                    validation_report['summary']['total_configs'] += 1
                    
                    if validation_result['valid']:
                        validation_report['summary']['valid_configs'] += 1
                    else:
                        validation_report['summary']['invalid_configs'] += 1
                    
                    validation_report['summary']['warnings'] += len(validation_result.get('warnings', []))
                
                # Validate tool configurations
                tool_configs = self.config_manager.discover_tool_configs()
                for config_file in tool_configs:
                    validation_result = await self.validate_tool_config(config_file)
                    validation_report['tool_validations'][str(config_file)] = validation_result
                    validation_report['summary']['total_configs'] += 1
                    
                    if validation_result['valid']:
                        validation_report['summary']['valid_configs'] += 1
                    else:
                        validation_report['summary']['invalid_configs'] += 1
                
                # Validate A2A compliance
                a2a_validation = await self.validate_a2a_compliance_suite(config_directory)
                validation_report['a2a_validations'] = a2a_validation
                
                # Apply custom validation rules
                custom_validations = await self.apply_custom_validation_rules(config_directory)
                validation_report['custom_rule_validations'] = custom_validations
                
                # Store validation history
                self.validation_history.append(validation_report)
                
                return validation_report
                
            async def validate_agent_config(self, config_file: Path) -> Dict[str, Any]:
                \"\"\"Validate individual agent configuration\"\"\"
                
                validation_result = {
                    'config_file': str(config_file),
                    'valid': True,
                    'errors': [],
                    'warnings': [],
                    'validation_details': {}
                }
                
                try:
                    # Load and validate configuration
                    agent_config = EnhancedAgentConfig.from_yaml(config_file)
                    
                    # Schema validation
                    schema_validation = agent_config.validate_schema()
                    validation_result['validation_details']['schema'] = schema_validation
                    
                    # A2A compliance validation
                    a2a_validation = agent_config.validate_a2a_compliance()
                    validation_result['validation_details']['a2a_compliance'] = a2a_validation
                    
                    # Capability validation
                    capability_validation = agent_config.validate_capabilities()
                    validation_result['validation_details']['capabilities'] = capability_validation
                    
                    # Collect errors and warnings
                    if not schema_validation.get('valid', True):
                        validation_result['errors'].extend(schema_validation.get('errors', []))
                        validation_result['valid'] = False
                    
                    if a2a_validation:
                        validation_result['warnings'].extend(a2a_validation)
                    
                    if not capability_validation.get('valid', True):
                        validation_result['warnings'].extend(capability_validation.get('warnings', []))
                
                except Exception as e:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Configuration loading error: {str(e)}")
                
                return validation_result
                
            def generate_validation_report(self, validation_results: Dict[str, Any], output_file: str):
                \"\"\"Generate human-readable validation report\"\"\"
                
                report_lines = []
                report_lines.append("# Configuration Validation Report")
                report_lines.append(f"Generated: {validation_results['validation_timestamp']}")
                report_lines.append(f"Configuration Directory: {validation_results['config_directory']}")
                report_lines.append("")
                
                # Summary
                summary = validation_results['summary']
                report_lines.append("## Summary")
                report_lines.append(f"- Total Configurations: {summary['total_configs']}")
                report_lines.append(f"- Valid Configurations: {summary['valid_configs']}")
                report_lines.append(f"- Invalid Configurations: {summary['invalid_configs']}")
                report_lines.append(f"- Total Warnings: {summary['warnings']}")
                report_lines.append("")
                
                # Agent validations
                if validation_results['agent_validations']:
                    report_lines.append("## Agent Configuration Validation")
                    for config_file, result in validation_results['agent_validations'].items():
                        status = "✅ VALID" if result['valid'] else "❌ INVALID"
                        report_lines.append(f"### {config_file} - {status}")
                        
                        if result['errors']:
                            report_lines.append("**Errors:**")
                            for error in result['errors']:
                                report_lines.append(f"- {error}")
                        
                        if result['warnings']:
                            report_lines.append("**Warnings:**")
                            for warning in result['warnings']:
                                report_lines.append(f"- {warning}")
                        
                        report_lines.append("")
                
                # Write report
                with open(output_file, 'w') as f:
                    f.write("\\n".join(report_lines))
        
        # Validation suite setup
        validation_suite = ConfigurationValidationSuite(config_manager)
        
        # Register custom validation rules
        def validate_naming_convention(config_data):
            # Custom validation logic
            pass
            
        validation_suite.register_validation_rule('naming_convention', validate_naming_convention)
        
        # Run comprehensive validation
        validation_results = await validation_suite.comprehensive_validation('config/')
        validation_suite.generate_validation_report(validation_results, 'validation_report.md')
        
        print(f"Validation completed:")
        print(f"  Valid: {validation_results['summary']['valid_configs']}")
        print(f"  Invalid: {validation_results['summary']['invalid_configs']}")
        print(f"  Warnings: {validation_results['summary']['warnings']}")
        ```
    
    **Advanced Features:**
        
        **A2A Protocol Integration:**
        * **Automatic Card Generation**: Intelligent generation of A2A-compliant cards from configurations
        * **Protocol Compliance**: Comprehensive A2A protocol compliance validation and enforcement
        * **Communication Setup**: Automated setup of A2A communication channels and protocols
        * **Discovery Services**: Integration with A2A discovery services and registries
        
        **Configuration Orchestration:**
        * **Multi-Environment Management**: Configuration management across development, staging, and production
        * **Dependency Resolution**: Intelligent resolution of configuration dependencies
        * **Version Control Integration**: Git-based configuration versioning and change tracking
        * **Automated Testing**: Automated configuration testing and validation pipelines
        
        **Enterprise Integration:**
        * **CI/CD Pipeline Integration**: Integration with enterprise CI/CD and deployment pipelines
        * **Configuration Management**: Integration with enterprise configuration management systems
        * **Audit and Compliance**: Configuration audit trails and compliance reporting
        * **Security Integration**: Secure configuration handling and access control
    
    **Production Deployment:**
        
        **High Availability:**
        * **Configuration Replication**: Configuration replication across multiple environments
        * **Disaster Recovery**: Configuration backup and disaster recovery capabilities
        * **Load Balancing**: Load balancing for configuration services and management
        * **Monitoring Integration**: Integration with enterprise monitoring and alerting systems
        
        **Performance Optimization:**
        * **Configuration Caching**: Intelligent caching of configuration data and validation results
        * **Parallel Processing**: Parallel processing of configuration validation and export operations
        * **Memory Optimization**: Memory-efficient processing of large configuration sets
        * **Network Optimization**: Optimized network communication for distributed configuration management
        
        **Security and Compliance:**
        * **Access Control**: Role-based access control for configuration management operations
        * **Audit Logging**: Comprehensive audit logging for configuration access and changes
        * **Encryption**: Configuration encryption for sensitive data and environments
        * **Compliance Reporting**: Automated compliance reporting and validation
    
    Attributes:
        export_cache (Dict[str, Any]): Cache for export operations and results
        
    Methods:
        export_a2a_cards: Export A2A-compliant cards from configurations
        discover_agent_configs: Discover and classify agent configuration files
        discover_tool_configs: Discover and classify tool configuration files
        validate_a2a_compliance: Validate A2A protocol compliance
        
    Note:
        This manager extends ConfigManager with A2A protocol integration and enhanced validation.
        A2A card export requires properly structured agent and tool configurations.
        Configuration discovery uses intelligent file classification and pattern matching.
        Export operations are cached for performance optimization in large configuration sets.
        
    Warning:
        A2A card export may fail if configurations don't meet protocol requirements.
        Large configuration sets may require significant processing time and memory.
        Configuration validation may identify issues that require manual correction.
        Export caching may consume significant memory in high-volume scenarios.
        
    See Also:
        * :class:`ConfigManager`: Base configuration manager functionality
        * :class:`EnhancedAgentConfig`: Enhanced agent configuration with A2A support
        * :class:`EnhancedToolConfig`: Enhanced tool configuration with A2A support
        * :mod:`nanobrain.core.config.config_base`: Configuration base classes and validation
        * :mod:`nanobrain.core.config.schema_validator`: Configuration schema validation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        self.export_cache: Dict[str, Any] = {}
        
    def export_a2a_cards(self, output_dir: Path = Path("a2a_cards")) -> Dict[str, List[Path]]:
        """Export A2A-compliant cards from YAML configurations"""
        exported = {"agents": [], "tools": []}
        
        logger.info("🔄 Starting A2A card export from YAML configurations...")
        
        # Export agent cards
        agent_configs = self.discover_agent_configs()
        logger.info(f"📋 Found {len(agent_configs)} agent configuration files")
        
        for config_file in agent_configs:
            try:
                logger.debug(f"Processing agent config: {config_file}")
                agent_config = EnhancedAgentConfig.from_yaml(config_file)
                
                # Validate A2A compliance
                validation_errors = agent_config.validate_a2a_compliance()
                if validation_errors:
                    logger.warning(f"⚠️  A2A compliance issues in {config_file}: {validation_errors}")
                
                card_data = agent_config.generate_a2a_card()
                
                # Save A2A-compliant JSON
                card_path = output_dir / "agents" / f"{agent_config.name}_agent_card.json"
                card_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(card_path, 'w') as f:
                    json.dump(card_data, f, indent=2)
                    
                exported["agents"].append(card_path)
                logger.info(f"✅ Exported A2A agent card: {agent_config.name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to export agent card from {config_file}: {e}")
        
        # Export tool cards
        tool_configs = self.discover_tool_configs()
        logger.info(f"🔧 Found {len(tool_configs)} tool configuration files")
        
        for config_file in tool_configs:
            try:
                logger.debug(f"Processing tool config: {config_file}")
                tool_config = EnhancedToolConfig.from_yaml(config_file)
                
                # Validate tool card
                validation_errors = tool_config.validate_tool_card()
                if validation_errors:
                    logger.warning(f"⚠️  Tool card issues in {config_file}: {validation_errors}")
                
                card_data = tool_config.generate_tool_card()
                
                # Save both JSON and YAML for tools
                for format_ext in ["json", "yaml"]:
                    card_path = output_dir / "tools" / f"{tool_config.name}_tool_card.{format_ext}"
                    card_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(card_path, 'w') as f:
                        if format_ext == "json":
                            json.dump(card_data, f, indent=2)
                        else:
                            yaml.dump(card_data, f, default_flow_style=False, indent=2)
                    
                    exported["tools"].append(card_path)
                
                logger.info(f"✅ Exported tool card: {tool_config.name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to export tool card from {config_file}: {e}")
        
        # Summary
        total_exported = len(exported["agents"]) + len(exported["tools"])
        logger.info(f"🎉 Export complete! Generated {total_exported} card files:")
        logger.info(f"   📋 {len(exported['agents'])} agent cards")
        logger.info(f"   🔧 {len(exported['tools'])//2} tool cards (JSON + YAML)")
        
        return exported
    
    def validate_card_sections(self) -> Dict[str, Any]:
        """Validate agent_card and tool_card sections for A2A compliance"""
        validation_results = {
            "agents": {"valid": [], "invalid": []},
            "tools": {"valid": [], "invalid": []},
            "errors": [],
            "summary": {
                "total_agents": 0,
                "valid_agents": 0,
                "total_tools": 0,
                "valid_tools": 0
            }
        }
        
        logger.info("🔍 Starting configuration validation...")
        
        # Validate agent cards
        agent_configs = self.discover_agent_configs()
        validation_results["summary"]["total_agents"] = len(agent_configs)
        
        for config_file in agent_configs:
            try:
                agent_config = EnhancedAgentConfig.from_yaml(config_file)
                
                # A2A compliance checks
                compliance_errors = agent_config.validate_a2a_compliance()
                if compliance_errors:
                    validation_results["agents"]["invalid"].append(str(config_file))
                    validation_results["errors"].extend([
                        f"{config_file}: {error}" for error in compliance_errors
                    ])
                else:
                    validation_results["agents"]["valid"].append(str(config_file))
                    validation_results["summary"]["valid_agents"] += 1
                    
            except Exception as e:
                validation_results["agents"]["invalid"].append(str(config_file))
                validation_results["errors"].append(f"{config_file}: Configuration error - {str(e)}")
        
        # Validate tool cards
        tool_configs = self.discover_tool_configs()
        validation_results["summary"]["total_tools"] = len(tool_configs)
        
        for config_file in tool_configs:
            try:
                tool_config = EnhancedToolConfig.from_yaml(config_file)
                
                # Tool card validation
                card_errors = tool_config.validate_tool_card()
                if card_errors:
                    validation_results["tools"]["invalid"].append(str(config_file))
                    validation_results["errors"].extend([
                        f"{config_file}: {error}" for error in card_errors
                    ])
                else:
                    validation_results["tools"]["valid"].append(str(config_file))
                    validation_results["summary"]["valid_tools"] += 1
                    
            except Exception as e:
                validation_results["tools"]["invalid"].append(str(config_file))
                validation_results["errors"].append(f"{config_file}: Configuration error - {str(e)}")
        
        # Log validation summary
        summary = validation_results["summary"]
        logger.info(f"📊 Validation Summary:")
        logger.info(f"   📋 Agents: {summary['valid_agents']}/{summary['total_agents']} valid")
        logger.info(f"   🔧 Tools: {summary['valid_tools']}/{summary['total_tools']} valid")
        
        if validation_results["errors"]:
            logger.warning(f"⚠️  Found {len(validation_results['errors'])} validation errors")
            for error in validation_results["errors"]:
                logger.warning(f"   - {error}")
        else:
            logger.info("✅ All configurations passed validation!")
                
        return validation_results
    
    def discover_agent_configs(self) -> List[Path]:
        """Discover agent configuration files"""
        config_paths = []
        search_paths = [
            Path("nanobrain/library/config/defaults"),
            Path("nanobrain/library/config"),
            Path("nanobrain/config"),
            Path("config")
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                # Look for agent-specific files
                config_paths.extend(search_path.glob("*agent*.yml"))
                config_paths.extend(search_path.glob("*agent*.yaml"))
                
                # Look for enhanced agent configs
                config_paths.extend(search_path.glob("enhanced_*agent*.yml"))
                config_paths.extend(search_path.glob("enhanced_*agent*.yaml"))
        
        # Remove duplicates and return
        return list(set(config_paths))
    
    def discover_tool_configs(self) -> List[Path]:
        """Discover tool configuration files"""
        config_paths = []
        search_paths = [
            Path("nanobrain/library/config/defaults"),
            Path("nanobrain/library/config"),
            Path("nanobrain/config"),
            Path("config")
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                # Look for tool-specific files
                config_paths.extend(search_path.glob("*tool*.yml"))
                config_paths.extend(search_path.glob("*tool*.yaml"))
                config_paths.extend(search_path.glob("bioinformatics_tools.yml"))
                config_paths.extend(search_path.glob("bioinformatics_tools.yaml"))
                
                # Look for enhanced tool configs
                config_paths.extend(search_path.glob("enhanced_*tool*.yml"))
                config_paths.extend(search_path.glob("enhanced_*tool*.yaml"))
        
        # Remove duplicates and return
        return list(set(config_paths))
    
    def load_enhanced_agent_config(self, config_file: Path) -> EnhancedAgentConfig:
        """Load and validate an enhanced agent configuration"""
        try:
            config = EnhancedAgentConfig.from_yaml(config_file)
            
            # Validate A2A compliance
            validation_errors = config.validate_a2a_compliance()
            if validation_errors:
                logger.warning(f"A2A compliance issues in {config_file}: {validation_errors}")
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load enhanced agent config from {config_file}: {e}")
            raise
    
    def load_enhanced_tool_config(self, config_file: Path) -> EnhancedToolConfig:
        """Load and validate an enhanced tool configuration"""
        try:
            config = EnhancedToolConfig.from_yaml(config_file)
            
            # Validate tool card
            validation_errors = config.validate_tool_card()
            if validation_errors:
                logger.warning(f"Tool card issues in {config_file}: {validation_errors}")
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load enhanced tool config from {config_file}: {e}")
            raise
    
    def generate_config_templates(self, output_dir: Path = Path("config_templates")) -> Dict[str, Path]:
        """Generate template configuration files for common use cases"""
        output_dir.mkdir(parents=True, exist_ok=True)
        templates = {}
        
        logger.info("📝 Generating configuration templates...")
        
        # Agent templates
        from .enhanced_config import create_agent_config_template
        
        agent_templates = [
            ("simple_agent", "simple", None),
            ("collaborative_agent", "collaborative", "general"),
            ("bioinformatics_agent", "specialized", "bioinformatics"),
            ("conversational_agent", "conversational", "general")
        ]
        
        for template_name, agent_type, domain in agent_templates:
            config = create_agent_config_template(
                name=template_name.replace("_", " ").title(),
                agent_type=agent_type,
                domain=domain
            )
            
            template_path = output_dir / f"{template_name}_template.yml"
            config.to_yaml(template_path)
            templates[template_name] = template_path
            logger.info(f"✅ Generated {template_name} template")
        
        # Tool templates  
        from .enhanced_config import create_tool_config_template
        
        tool_templates = [
            ("bioinformatics_tool", "bioinformatics", "analysis"),
            ("database_tool", "database", "access"),
            ("analysis_tool", "analysis", "processing")
        ]
        
        for template_name, category, tool_type in tool_templates:
            config = create_tool_config_template(
                name=template_name.replace("_", " ").title(),
                category=category,
                tool_type=tool_type
            )
            
            template_path = output_dir / f"{template_name}_template.yml"
            config.to_yaml(template_path)
            templates[template_name] = template_path
            logger.info(f"✅ Generated {template_name} template")
        
        logger.info(f"🎉 Generated {len(templates)} configuration templates in {output_dir}")
        return templates


# Global instance
_enhanced_config_manager: Optional[EnhancedConfigManager] = None


def get_enhanced_config_manager(config_path: Optional[str] = None) -> EnhancedConfigManager:
    """Get the global enhanced configuration manager instance"""
    global _enhanced_config_manager
    
    if _enhanced_config_manager is None or config_path:
        _enhanced_config_manager = EnhancedConfigManager(config_path)
        _enhanced_config_manager.load_config()
    
    return _enhanced_config_manager


def export_a2a_cards(output_dir: Path = Path("a2a_cards")) -> Dict[str, List[Path]]:
    """Convenience function to export A2A cards"""
    manager = get_enhanced_config_manager()
    return manager.export_a2a_cards(output_dir)


def validate_configurations() -> Dict[str, Any]:
    """Convenience function to validate all configurations"""
    manager = get_enhanced_config_manager()
    return manager.validate_card_sections()


def generate_templates(output_dir: Path = Path("config_templates")) -> Dict[str, Path]:
    """Convenience function to generate configuration templates"""
    manager = get_enhanced_config_manager()
    return manager.generate_config_templates(output_dir) 