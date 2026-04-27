#!/usr/bin/env python3
"""
A2A Card Validation Script for NanoBrain Framework

Validates that all Tool Cards and Agent Cards comply with A2A protocol requirements.
"""

import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.core.card_schemas import (
    ToolCard, AgentCard, get_card_manager
)


class A2ACardValidator:
    """Validates A2A protocol compliance for Tool Cards and Agent Cards."""
    
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_tool_card(self, tool_card_path: Path) -> Dict[str, Any]:
        """Validate a Tool Card for A2A compliance."""
        print(f"  🔍 Validating Tool Card: {tool_card_path.name}")
        
        errors = []
        warnings = []
        
        try:
            # Load the card
            if tool_card_path.suffix == ".json":
                with open(tool_card_path, 'r') as f:
                    card_data = json.load(f)
            elif tool_card_path.suffix in [".yaml", ".yml"]:
                with open(tool_card_path, 'r') as f:
                    card_data = yaml.safe_load(f)
            else:
                errors.append(f"Unsupported file format: {tool_card_path.suffix}")
                return {"errors": errors, "warnings": warnings}
            
            # Required fields validation
            required_fields = [
                "name", "version", "description", "purpose", "category",
                "input_modes", "output_modes", "a2a_compatible", "a2a_version"
            ]
            
            for field in required_fields:
                if field not in card_data:
                    errors.append(f"Missing required field: {field}")
                elif not card_data[field]:
                    warnings.append(f"Empty required field: {field}")
            
            # A2A specific validations
            if card_data.get("a2a_compatible") is not True:
                errors.append("Tool must be A2A compatible")
            
            if not card_data.get("a2a_version"):
                errors.append("A2A version must be specified")
            
            # Input/Output modes validation
            if not card_data.get("input_modes"):
                warnings.append("No input modes specified")
            
            if not card_data.get("output_modes"):
                warnings.append("No output modes specified")
            
            # Usage examples validation
            if not card_data.get("usage_examples"):
                warnings.append("No usage examples provided")
            elif len(card_data["usage_examples"]) == 0:
                warnings.append("Empty usage examples list")
            
            # Schema validation
            if card_data.get("input_schema"):
                schema_errors = self._validate_io_schema(card_data["input_schema"], "input")
                errors.extend(schema_errors)
            
            if card_data.get("output_schema"):
                schema_errors = self._validate_io_schema(card_data["output_schema"], "output")
                errors.extend(schema_errors)
            
            # Performance metrics validation
            if not card_data.get("performance"):
                warnings.append("No performance metrics provided")
            
            # Documentation validation
            if not card_data.get("documentation_url"):
                warnings.append("No documentation URL provided")
            
            print(f"    ✅ Tool Card validation complete")
            if errors:
                print(f"    ❌ {len(errors)} errors found")
            if warnings:
                print(f"    ⚠️  {len(warnings)} warnings found")
            
        except Exception as e:
            errors.append(f"Failed to validate tool card: {str(e)}")
            print(f"    ❌ Validation failed: {e}")
        
        return {"errors": errors, "warnings": warnings}
    
    def validate_agent_card(self, agent_card_path: Path) -> Dict[str, Any]:
        """Validate an Agent Card for A2A compliance."""
        print(f"  🔍 Validating Agent Card: {agent_card_path.name}")
        
        errors = []
        warnings = []
        
        try:
            # Load the card
            with open(agent_card_path, 'r') as f:
                card_data = json.load(f)
            
            # Required A2A fields validation
            required_fields = [
                "name", "version", "description", "purpose", "url",
                "agent_type", "capabilities", "skills", "defaultInputModes",
                "defaultOutputModes", "authentication"
            ]
            
            for field in required_fields:
                if field not in card_data:
                    errors.append(f"Missing required A2A field: {field}")
                elif not card_data[field] and field != "skills":  # skills can be empty array
                    warnings.append(f"Empty required field: {field}")
            
            # A2A specific validations
            if not card_data.get("url"):
                errors.append("Agent URL must be specified for A2A protocol")
            
            # Agent type validation
            valid_agent_types = ["conversational", "specialized", "collaborative", "task_oriented"]
            if card_data.get("agent_type") not in valid_agent_types:
                warnings.append(f"Agent type '{card_data.get('agent_type')}' not in recommended types: {valid_agent_types}")
            
            # Capabilities validation
            capabilities = card_data.get("capabilities", {})
            expected_capabilities = [
                "streaming", "push_notifications", "state_transition_history",
                "multi_turn_conversation", "context_retention", "tool_usage",
                "delegation", "collaboration"
            ]
            
            for cap in expected_capabilities:
                if cap not in capabilities:
                    warnings.append(f"Missing capability declaration: {cap}")
            
            # Skills validation
            skills = card_data.get("skills", [])
            if not skills:
                warnings.append("No skills declared - consider adding skill declarations")
            else:
                for i, skill in enumerate(skills):
                    skill_errors = self._validate_skill(skill, i)
                    errors.extend(skill_errors)
            
            # Input/Output modes validation
            if not card_data.get("defaultInputModes"):
                errors.append("Default input modes must be specified")
            
            if not card_data.get("defaultOutputModes"):
                errors.append("Default output modes must be specified")
            
            # Authentication validation
            auth = card_data.get("authentication", {})
            if not auth.get("schemes"):
                warnings.append("No authentication schemes specified")
            
            # Usage examples validation
            if not card_data.get("usage_examples"):
                warnings.append("No usage examples provided")
            
            # Provider information validation
            if not card_data.get("provider"):
                warnings.append("No provider information specified")
            
            print(f"    ✅ Agent Card validation complete")
            if errors:
                print(f"    ❌ {len(errors)} errors found")
            if warnings:
                print(f"    ⚠️  {len(warnings)} warnings found")
            
        except Exception as e:
            errors.append(f"Failed to validate agent card: {str(e)}")
            print(f"    ❌ Validation failed: {e}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_io_schema(self, schema: Dict[str, Any], schema_type: str) -> List[str]:
        """Validate input/output schema structure."""
        errors = []
        
        required_schema_fields = ["format", "description"]
        for field in required_schema_fields:
            if field not in schema:
                errors.append(f"Missing required {schema_type} schema field: {field}")
        
        # Validate parameters if present
        if "parameters" in schema:
            for i, param in enumerate(schema["parameters"]):
                param_errors = self._validate_parameter(param, i, schema_type)
                errors.extend(param_errors)
        
        return errors
    
    def _validate_parameter(self, param: Dict[str, Any], index: int, schema_type: str) -> List[str]:
        """Validate parameter schema."""
        errors = []
        
        required_param_fields = ["name", "type", "description"]
        for field in required_param_fields:
            if field not in param:
                errors.append(f"Missing required field in {schema_type} parameter {index}: {field}")
        
        return errors
    
    def _validate_skill(self, skill: Dict[str, Any], index: int) -> List[str]:
        """Validate skill declaration."""
        errors = []
        
        required_skill_fields = ["id", "name", "description"]
        for field in required_skill_fields:
            if field not in skill:
                errors.append(f"Missing required field in skill {index}: {field}")
        
        return errors
    
    def validate_all_cards(self, cards_directory: str = "cards") -> Dict[str, Any]:
        """Validate all Tool Cards and Agent Cards."""
        print("🔍 Starting A2A Card Validation...")
        
        cards_path = Path(cards_directory)
        if not cards_path.exists():
            return {"error": f"Cards directory not found: {cards_directory}"}
        
        tool_results = []
        agent_results = []
        
        # Validate Tool Cards
        tools_dir = cards_path / "tools"
        if tools_dir.exists():
            print("\n📋 Validating Tool Cards:")
            for tool_card_file in tools_dir.glob("*_tool_card.*"):
                result = self.validate_tool_card(tool_card_file)
                result["file"] = tool_card_file.name
                tool_results.append(result)
        
        # Validate Agent Cards
        agents_dir = cards_path / "agents"
        if agents_dir.exists():
            print("\n🤖 Validating Agent Cards:")
            for agent_card_file in agents_dir.glob("*_agent_card.json"):
                result = self.validate_agent_card(agent_card_file)
                result["file"] = agent_card_file.name
                agent_results.append(result)
        
        # Summary
        total_tool_errors = sum(len(r["errors"]) for r in tool_results)
        total_tool_warnings = sum(len(r["warnings"]) for r in tool_results)
        total_agent_errors = sum(len(r["errors"]) for r in agent_results)
        total_agent_warnings = sum(len(r["warnings"]) for r in agent_results)
        
        print(f"\n📊 Validation Summary:")
        print(f"  📋 Tool Cards: {len(tool_results)} validated")
        print(f"     ❌ Errors: {total_tool_errors}")
        print(f"     ⚠️  Warnings: {total_tool_warnings}")
        print(f"  🤖 Agent Cards: {len(agent_results)} validated")
        print(f"     ❌ Errors: {total_agent_errors}")
        print(f"     ⚠️  Warnings: {total_agent_warnings}")
        
        total_errors = total_tool_errors + total_agent_errors
        total_warnings = total_tool_warnings + total_agent_warnings
        
        if total_errors == 0:
            print(f"\n🎉 All cards are A2A protocol compliant!")
            if total_warnings > 0:
                print(f"   ⚠️  Consider addressing {total_warnings} warnings for improved compliance")
        else:
            print(f"\n❌ Found {total_errors} critical errors that need to be fixed")
        
        return {
            "tool_cards": tool_results,
            "agent_cards": agent_results,
            "summary": {
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "tool_errors": total_tool_errors,
                "tool_warnings": total_tool_warnings,
                "agent_errors": total_agent_errors,
                "agent_warnings": total_agent_warnings
            }
        }


def main():
    """Run A2A card validation."""
    validator = A2ACardValidator()
    results = validator.validate_all_cards()
    
    # Exit with error code if validation failed
    if results.get("summary", {}).get("total_errors", 0) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()