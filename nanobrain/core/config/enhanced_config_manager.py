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
    """Enhanced configuration manager with card export capabilities"""
    
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