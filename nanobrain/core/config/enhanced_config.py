"""
Enhanced Configuration System with Integrated Card Metadata

Implements YAML-based configuration with mandatory agent_card and tool_card sections
for A2A protocol compliance and unified framework architecture.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path
import json
import yaml
import logging

from .yaml_config import YAMLConfig

logger = logging.getLogger(__name__)


class InputOutputFormat(BaseModel):
    """Structured input/output format specification"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    primary_mode: str
    supported_modes: List[str]
    content_types: List[str]
    format_schema: Dict[str, Any]


class A2ACapabilities(BaseModel):
    """A2A capability declarations for agents"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = False
    multi_turn_conversation: bool = True
    context_retention: bool = True
    tool_usage: bool = False
    delegation: bool = False
    collaboration: bool = False


class A2ASkill(BaseModel):
    """A2A skill definition for agents"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    id: str
    name: str
    description: str
    complexity: str = "intermediate"  # beginner, intermediate, advanced, expert
    input_modes: List[str] = Field(default_factory=list)
    output_modes: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)


class PerformanceSpec(BaseModel):
    """Performance characteristics specification"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    typical_response_time: str
    max_response_time: Optional[str] = None
    memory_usage: Optional[str] = None
    cpu_requirements: Optional[str] = None
    concurrency_support: bool = False
    max_concurrent_sessions: Optional[int] = None
    concurrent_limit: Optional[int] = None
    rate_limit: Optional[str] = None
    scaling_characteristics: Optional[str] = None


class UsageExample(BaseModel):
    """Detailed usage example with input/output"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str
    description: str
    context: Optional[str] = None
    input_example: Dict[str, Any]
    expected_output: Dict[str, Any]


class AgentCardSection(BaseModel):
    """Agent card metadata section for A2A protocol compliance"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    version: str = "1.0.0"
    purpose: str
    detailed_description: str
    url: Optional[str] = None
    domain: Optional[str] = None
    expertise_level: str = "intermediate"
    
    # Input/Output specifications
    input_format: InputOutputFormat
    output_format: InputOutputFormat
    
    # A2A Protocol requirements
    capabilities: A2ACapabilities = Field(default_factory=A2ACapabilities)
    skills: List[A2ASkill] = Field(default_factory=list)
    
    # Performance and usage
    performance: PerformanceSpec
    usage_examples: List[UsageExample] = Field(default_factory=list)


class ToolCapabilities(BaseModel):
    """Tool-specific capabilities"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    batch_processing: bool = False
    streaming: bool = False
    caching: bool = False
    rate_limiting: bool = False
    authentication_required: bool = False
    concurrent_requests: int = 1


class ToolCardSection(BaseModel):
    """Tool card metadata section"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    version: str = "1.0.0"
    purpose: str
    detailed_description: str
    category: str
    subcategory: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # Input/Output specifications
    input_format: InputOutputFormat
    output_format: InputOutputFormat
    
    # Tool characteristics
    capabilities: ToolCapabilities = Field(default_factory=ToolCapabilities)
    performance: PerformanceSpec
    usage_examples: List[UsageExample] = Field(default_factory=list)


class EnhancedAgentConfig(YAMLConfig):
    """Enhanced agent configuration with integrated agent_card section"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    # Existing fields
    name: str
    description: str = ""
    agent_type: str = "simple"
    llm_config: Dict[str, Any] = Field(default_factory=dict)
    memory_config: Dict[str, Any] = Field(default_factory=dict)
    tools_config: Dict[str, Any] = Field(default_factory=dict)
    
    # MANDATORY: Agent card section for A2A compliance
    agent_card: AgentCardSection
    
    # Framework metadata
    framework_metadata: Dict[str, Any] = Field(default_factory=lambda: {
        "config_type": "agent",
        "framework_version": "1.0.0"
    })
    
    def generate_a2a_card(self) -> Dict[str, Any]:
        """Generate A2A compliant agent card from configuration"""
        return {
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "a2a_version": "1.0.0",
            "a2a_compatible": True,
            **self.agent_card.model_dump()
        }
    
    def validate_a2a_compliance(self) -> List[str]:
        """Validate A2A protocol compliance"""
        errors = []
        
        if not self.agent_card.capabilities:
            errors.append("Missing A2A capabilities declaration")
        
        if not self.agent_card.skills:
            errors.append("Missing agent skills definition")
        
        if not self.agent_card.input_format:
            errors.append("Missing input format specification")
            
        if not self.agent_card.output_format:
            errors.append("Missing output format specification")
            
        if not self.agent_card.purpose.strip():
            errors.append("Missing agent purpose description")
            
        if not self.agent_card.detailed_description.strip():
            errors.append("Missing detailed agent description")
        
        return errors


class EnhancedToolConfig(YAMLConfig):
    """Enhanced tool configuration with integrated tool_card section"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    # Existing fields
    name: str
    description: str = ""
    tool_type: str = "analysis"
    timeout: float = 600.0
    max_retries: int = 3
    
    # MANDATORY: Tool card section
    tool_card: ToolCardSection
    
    # Framework metadata
    framework_metadata: Dict[str, Any] = Field(default_factory=lambda: {
        "config_type": "tool",
        "framework_version": "1.0.0"
    })
    
    def generate_tool_card(self) -> Dict[str, Any]:
        """Generate tool card from configuration"""
        return {
            "name": self.name,
            "description": self.description,
            "tool_type": self.tool_type,
            **self.tool_card.model_dump()
        }
    
    def validate_tool_card(self) -> List[str]:
        """Validate tool card completeness"""
        errors = []
        
        if not self.tool_card.input_format:
            errors.append("Missing input format specification")
            
        if not self.tool_card.output_format:
            errors.append("Missing output format specification")
            
        if not self.tool_card.purpose.strip():
            errors.append("Missing tool purpose description")
            
        if not self.tool_card.detailed_description.strip():
            errors.append("Missing detailed tool description")
            
        if not self.tool_card.category.strip():
            errors.append("Missing tool category")
        
        return errors


def create_agent_config_template(
    name: str,
    agent_type: str = "simple",
    domain: Optional[str] = None
) -> EnhancedAgentConfig:
    """Create a template agent configuration with required sections"""
    
    # Basic input/output format for agents
    input_format = InputOutputFormat(
        primary_mode="json",
        supported_modes=["text", "json"],
        content_types=["application/json", "text/plain"],
        format_schema={
            "type": "object",
            "required_fields": {
                "message": {
                    "type": "string",
                    "description": "Natural language instruction or query"
                }
            },
            "optional_fields": {
                "context": {
                    "type": "object",
                    "description": "Conversation context"
                }
            }
        }
    )
    
    output_format = InputOutputFormat(
        primary_mode="json",
        supported_modes=["json", "text"],
        content_types=["application/json", "text/plain"],
        format_schema={
            "type": "object",
            "guaranteed_fields": {
                "response": {
                    "type": "string",
                    "description": "Natural language response"
                },
                "metadata": {
                    "type": "object",
                    "description": "Execution metadata"
                }
            }
        }
    )
    
    # Basic capabilities based on agent type
    capabilities = A2ACapabilities(
        multi_turn_conversation=True,
        context_retention=True,
        tool_usage=(agent_type in ["collaborative", "specialized"]),
        delegation=(agent_type == "collaborative"),
        collaboration=(agent_type == "collaborative")
    )
    
    # Basic skill
    basic_skill = A2ASkill(
        id="natural_language_processing",
        name="Natural Language Processing",
        description="Basic natural language understanding and generation",
        complexity="intermediate",
        input_modes=["text", "json"],
        output_modes=["text", "json"],
        examples=["Process natural language queries", "Generate coherent responses"]
    )
    
    performance = PerformanceSpec(
        typical_response_time="1-5 seconds",
        concurrency_support=True,
        memory_usage="50-200 MB",
        cpu_requirements="Low to Medium"
    )
    
    agent_card = AgentCardSection(
        purpose=f"A {agent_type} agent for {domain or 'general'} tasks",
        detailed_description=f"This {agent_type} agent provides capabilities for {domain or 'general purpose'} processing and interaction.",
        domain=domain,
        input_format=input_format,
        output_format=output_format,
        capabilities=capabilities,
        skills=[basic_skill],
        performance=performance
    )
    
    return EnhancedAgentConfig(
        name=name,
        description=f"{agent_type.title()} agent for {domain or 'general'} tasks",
        agent_type=agent_type,
        agent_card=agent_card
    )


def create_tool_config_template(
    name: str,
    category: str,
    tool_type: str = "analysis"
) -> EnhancedToolConfig:
    """Create a template tool configuration with required sections"""
    
    # Basic input/output format for tools
    input_format = InputOutputFormat(
        primary_mode="json",
        supported_modes=["json", "data"],
        content_types=["application/json"],
        format_schema={
            "type": "object",
            "required_fields": {
                "query_type": {
                    "type": "string",
                    "description": "Type of operation to perform"
                },
                "parameters": {
                    "type": "object",
                    "description": "Operation parameters"
                }
            }
        }
    )
    
    output_format = InputOutputFormat(
        primary_mode="json",
        supported_modes=["json", "structured"],
        content_types=["application/json"],
        format_schema={
            "type": "object",
            "guaranteed_fields": {
                "status": {
                    "type": "string",
                    "description": "Operation status"
                },
                "data": {
                    "type": "object",
                    "description": "Result data"
                },
                "metadata": {
                    "type": "object",
                    "description": "Execution metadata"
                }
            }
        }
    )
    
    capabilities = ToolCapabilities(
        batch_processing=True,
        caching=True
    )
    
    performance = PerformanceSpec(
        typical_response_time="1-10 seconds",
        memory_usage="50-200 MB",
        cpu_requirements="Low"
    )
    
    tool_card = ToolCardSection(
        purpose=f"A {category} tool for {tool_type} operations",
        detailed_description=f"This tool provides {category} capabilities for {tool_type} operations.",
        category=category,
        input_format=input_format,
        output_format=output_format,
        capabilities=capabilities,
        performance=performance
    )
    
    return EnhancedToolConfig(
        name=name,
        description=f"{category.title()} {tool_type} tool",
        tool_type=tool_type,
        tool_card=tool_card
    ) 