"""
Card Schema System for NanoBrain Framework

Implements mandatory Tool Cards and Agent Cards as required by the A2A (Agent-to-Agent) protocol.
Provides comprehensive metadata about tool and agent capabilities for discoverability and interoperability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import json
import yaml
from pathlib import Path

from .logging_system import get_logger

logger = get_logger(__name__)


class InputMode(Enum):
    """Supported input modes for tools and agents."""
    TEXT = "text"
    DATA = "data"
    FILE = "file"
    STREAM = "stream"
    JSON = "json"
    BINARY = "binary"
    MULTIPART = "multipart"


class OutputMode(Enum):
    """Supported output modes for tools and agents."""
    TEXT = "text"
    DATA = "data"
    FILE = "file"
    STREAM = "stream"
    JSON = "json"
    BINARY = "binary"
    STRUCTURED = "structured"


class ParameterType(Enum):
    """Parameter types for tool and agent inputs."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    FILE_PATH = "file_path"
    ENUM = "enum"


@dataclass
class ParameterSchema:
    """Schema definition for a parameter."""
    name: str
    type: ParameterType
    description: str
    required: bool = False
    default: Optional[Any] = None
    enum_values: Optional[List[str]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    example: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "required": self.required
        }
        
        if self.default is not None:
            result["default"] = self.default
        if self.enum_values:
            result["enum"] = self.enum_values
        if self.min_value is not None:
            result["minimum"] = self.min_value
        if self.max_value is not None:
            result["maximum"] = self.max_value
        if self.pattern:
            result["pattern"] = self.pattern
        if self.example is not None:
            result["example"] = self.example
            
        return result


@dataclass
class IOSchema:
    """Input/Output schema definition."""
    format: str  # JSON Schema format
    description: str
    parameters: List[ParameterSchema] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    content_type: Optional[str] = None
    encoding: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "format": self.format,
            "description": self.description,
            "parameters": [param.to_dict() for param in self.parameters],
            "examples": self.examples,
            "content_type": self.content_type,
            "encoding": self.encoding
        }


@dataclass
class UsageExample:
    """Usage example for tools and agents."""
    name: str
    description: str
    input: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    context: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "input": self.input,
            "expected_output": self.expected_output,
            "context": self.context,
            "tags": self.tags
        }


@dataclass
class PerformanceMetrics:
    """Performance characteristics of tools and agents."""
    typical_response_time: Optional[str] = None  # e.g., "< 1 second", "1-5 minutes"
    memory_usage: Optional[str] = None
    cpu_requirements: Optional[str] = None
    concurrency_limit: Optional[int] = None
    rate_limit: Optional[str] = None
    scaling_characteristics: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "typical_response_time": self.typical_response_time,
            "memory_usage": self.memory_usage,
            "cpu_requirements": self.cpu_requirements,
            "concurrency_limit": self.concurrency_limit,
            "rate_limit": self.rate_limit,
            "scaling_characteristics": self.scaling_characteristics
        }


@dataclass
class ToolCard:
    """
    Mandatory Tool Card for external tools as required by A2A protocol.
    
    Provides comprehensive metadata about tool capabilities, inputs, outputs,
    and usage patterns for discoverability and interoperability.
    """
    # Basic identification
    name: str
    version: str
    description: str
    purpose: str  # Detailed explanation of what the tool does
    
    # Tool classification
    category: str  # e.g., "bioinformatics", "data_processing", "analysis"
    subcategory: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Input/Output specifications
    input_modes: List[InputMode] = field(default_factory=list)
    output_modes: List[OutputMode] = field(default_factory=list)
    input_schema: Optional[IOSchema] = None
    output_schema: Optional[IOSchema] = None
    
    # Usage information
    usage_examples: List[UsageExample] = field(default_factory=list)
    common_use_cases: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    # Technical specifications
    dependencies: List[str] = field(default_factory=list)
    system_requirements: List[str] = field(default_factory=list)
    supported_formats: List[str] = field(default_factory=list)
    performance: Optional[PerformanceMetrics] = None
    
    # A2A compatibility
    a2a_compatible: bool = True
    a2a_version: str = "1.0.0"
    authentication_required: bool = False
    authentication_schemes: List[str] = field(default_factory=list)
    
    # Metadata
    provider: Optional[str] = None
    license: Optional[str] = None
    documentation_url: Optional[str] = None
    source_url: Optional[str] = None
    created_date: Optional[str] = None
    last_updated: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ToolCard to dictionary representation."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "purpose": self.purpose,
            "category": self.category,
            "subcategory": self.subcategory,
            "tags": self.tags,
            "input_modes": [mode.value for mode in self.input_modes],
            "output_modes": [mode.value for mode in self.output_modes],
            "input_schema": self.input_schema.to_dict() if self.input_schema else None,
            "output_schema": self.output_schema.to_dict() if self.output_schema else None,
            "usage_examples": [example.to_dict() for example in self.usage_examples],
            "common_use_cases": self.common_use_cases,
            "limitations": self.limitations,
            "dependencies": self.dependencies,
            "system_requirements": self.system_requirements,
            "supported_formats": self.supported_formats,
            "performance": self.performance.to_dict() if self.performance else None,
            "a2a_compatible": self.a2a_compatible,
            "a2a_version": self.a2a_version,
            "authentication_required": self.authentication_required,
            "authentication_schemes": self.authentication_schemes,
            "provider": self.provider,
            "license": self.license,
            "documentation_url": self.documentation_url,
            "source_url": self.source_url,
            "created_date": self.created_date,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCard':
        """Create ToolCard from dictionary."""
        # Handle enums
        input_modes = [InputMode(mode) for mode in data.get("input_modes", [])]
        output_modes = [OutputMode(mode) for mode in data.get("output_modes", [])]
        
        # Handle nested objects
        input_schema = None
        if data.get("input_schema"):
            input_schema = IOSchema(
                format=data["input_schema"]["format"],
                description=data["input_schema"]["description"],
                parameters=[
                    ParameterSchema(
                        name=p["name"],
                        type=ParameterType(p["type"]),
                        description=p["description"],
                        required=p.get("required", False),
                        default=p.get("default"),
                        enum_values=p.get("enum"),
                        min_value=p.get("minimum"),
                        max_value=p.get("maximum"),
                        pattern=p.get("pattern"),
                        example=p.get("example")
                    ) for p in data["input_schema"].get("parameters", [])
                ],
                examples=data["input_schema"].get("examples", []),
                content_type=data["input_schema"].get("content_type"),
                encoding=data["input_schema"].get("encoding")
            )
        
        output_schema = None
        if data.get("output_schema"):
            output_schema = IOSchema(
                format=data["output_schema"]["format"],
                description=data["output_schema"]["description"],
                parameters=[
                    ParameterSchema(
                        name=p["name"],
                        type=ParameterType(p["type"]),
                        description=p["description"],
                        required=p.get("required", False),
                        default=p.get("default"),
                        enum_values=p.get("enum"),
                        min_value=p.get("minimum"),
                        max_value=p.get("maximum"),
                        pattern=p.get("pattern"),
                        example=p.get("example")
                    ) for p in data["output_schema"].get("parameters", [])
                ],
                examples=data["output_schema"].get("examples", []),
                content_type=data["output_schema"].get("content_type"),
                encoding=data["output_schema"].get("encoding")
            )
        
        usage_examples = [
            UsageExample(
                name=ex["name"],
                description=ex["description"],
                input=ex["input"],
                expected_output=ex.get("expected_output"),
                context=ex.get("context"),
                tags=ex.get("tags", [])
            ) for ex in data.get("usage_examples", [])
        ]
        
        performance = None
        if data.get("performance"):
            performance = PerformanceMetrics(**data["performance"])
        
        return cls(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            purpose=data["purpose"],
            category=data["category"],
            subcategory=data.get("subcategory"),
            tags=data.get("tags", []),
            input_modes=input_modes,
            output_modes=output_modes,
            input_schema=input_schema,
            output_schema=output_schema,
            usage_examples=usage_examples,
            common_use_cases=data.get("common_use_cases", []),
            limitations=data.get("limitations", []),
            dependencies=data.get("dependencies", []),
            system_requirements=data.get("system_requirements", []),
            supported_formats=data.get("supported_formats", []),
            performance=performance,
            a2a_compatible=data.get("a2a_compatible", True),
            a2a_version=data.get("a2a_version", "1.0.0"),
            authentication_required=data.get("authentication_required", False),
            authentication_schemes=data.get("authentication_schemes", []),
            provider=data.get("provider"),
            license=data.get("license"),
            documentation_url=data.get("documentation_url"),
            source_url=data.get("source_url"),
            created_date=data.get("created_date"),
            last_updated=data.get("last_updated")
        )


@dataclass
class Skill:
    """A2A skill definition for agents."""
    id: str
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    input_modes: List[InputMode] = field(default_factory=list)
    output_modes: List[OutputMode] = field(default_factory=list)
    complexity: Optional[str] = None  # "simple", "intermediate", "advanced"
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "examples": self.examples,
            "input_modes": [mode.value for mode in self.input_modes],
            "output_modes": [mode.value for mode in self.output_modes],
            "complexity": self.complexity,
            "dependencies": self.dependencies
        }


@dataclass
class AgentCapabilities:
    """Capabilities of an agent."""
    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = False
    multi_turn_conversation: bool = True
    context_retention: bool = True
    tool_usage: bool = False
    delegation: bool = False
    collaboration: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "streaming": self.streaming,
            "push_notifications": self.push_notifications,
            "state_transition_history": self.state_transition_history,
            "multi_turn_conversation": self.multi_turn_conversation,
            "context_retention": self.context_retention,
            "tool_usage": self.tool_usage,
            "delegation": self.delegation,
            "collaboration": self.collaboration
        }


@dataclass
class AgentCard:
    """
    Mandatory Agent Card as required by A2A protocol.
    
    Provides comprehensive metadata about agent capabilities, skills,
    and interaction patterns for agent discovery and collaboration.
    """
    # Basic identification
    name: str
    version: str
    description: str
    purpose: str
    url: str
    
    # Agent classification
    agent_type: str  # e.g., "conversational", "specialized", "collaborative"
    domain: Optional[str] = None  # e.g., "bioinformatics", "coding", "general"
    expertise_level: str = "intermediate"  # "beginner", "intermediate", "advanced", "expert"
    
    # Capabilities and skills
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)
    skills: List[Skill] = field(default_factory=list)
    supported_languages: List[str] = field(default_factory=lambda: ["en"])
    
    # Input/Output specifications
    default_input_modes: List[InputMode] = field(default_factory=lambda: [InputMode.TEXT])
    default_output_modes: List[OutputMode] = field(default_factory=lambda: [OutputMode.TEXT])
    input_schema: Optional[IOSchema] = None
    output_schema: Optional[IOSchema] = None
    
    # Interaction patterns
    conversation_patterns: List[str] = field(default_factory=list)
    response_styles: List[str] = field(default_factory=list)
    interaction_modes: List[str] = field(default_factory=lambda: ["chat"])
    
    # Usage information
    usage_examples: List[UsageExample] = field(default_factory=list)
    common_use_cases: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    # Technical specifications
    max_context_length: Optional[int] = None
    typical_response_time: Optional[str] = None
    concurrency_support: bool = False
    session_management: bool = True
    
    # A2A protocol requirements
    a2a_version: str = "1.0.0"
    authentication: Dict[str, Any] = field(default_factory=lambda: {"schemes": ["none"]})
    provider: Optional[Dict[str, str]] = None
    documentation_url: Optional[str] = None
    
    # Metadata
    created_date: Optional[str] = None
    last_updated: Optional[str] = None
    license: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AgentCard to dictionary representation for A2A protocol."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "purpose": self.purpose,
            "url": self.url,
            "agent_type": self.agent_type,
            "domain": self.domain,
            "expertise_level": self.expertise_level,
            "capabilities": self.capabilities.to_dict(),
            "skills": [skill.to_dict() for skill in self.skills],
            "supported_languages": self.supported_languages,
            "defaultInputModes": [mode.value for mode in self.default_input_modes],
            "defaultOutputModes": [mode.value for mode in self.default_output_modes],
            "input_schema": self.input_schema.to_dict() if self.input_schema else None,
            "output_schema": self.output_schema.to_dict() if self.output_schema else None,
            "conversation_patterns": self.conversation_patterns,
            "response_styles": self.response_styles,
            "interaction_modes": self.interaction_modes,
            "usage_examples": [example.to_dict() for example in self.usage_examples],
            "common_use_cases": self.common_use_cases,
            "limitations": self.limitations,
            "max_context_length": self.max_context_length,
            "typical_response_time": self.typical_response_time,
            "concurrency_support": self.concurrency_support,
            "session_management": self.session_management,
            "authentication": self.authentication,
            "provider": self.provider,
            "documentationUrl": self.documentation_url,
            "created_date": self.created_date,
            "last_updated": self.last_updated,
            "license": self.license
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCard':
        """Create AgentCard from dictionary."""
        # Handle enums
        default_input_modes = [InputMode(mode) for mode in data.get("defaultInputModes", ["text"])]
        default_output_modes = [OutputMode(mode) for mode in data.get("defaultOutputModes", ["text"])]
        
        # Handle capabilities
        capabilities_data = data.get("capabilities", {})
        capabilities = AgentCapabilities(**capabilities_data)
        
        # Handle skills
        skills = [
            Skill(
                id=skill["id"],
                name=skill["name"],
                description=skill["description"],
                tags=skill.get("tags", []),
                examples=skill.get("examples", []),
                input_modes=[InputMode(mode) for mode in skill.get("input_modes", [])],
                output_modes=[OutputMode(mode) for mode in skill.get("output_modes", [])],
                complexity=skill.get("complexity"),
                dependencies=skill.get("dependencies", [])
            ) for skill in data.get("skills", [])
        ]
        
        # Handle nested schemas (similar to ToolCard)
        input_schema = None
        if data.get("input_schema"):
            # Implementation similar to ToolCard.from_dict()
            pass
            
        output_schema = None
        if data.get("output_schema"):
            # Implementation similar to ToolCard.from_dict()
            pass
        
        # Handle usage examples
        usage_examples = [
            UsageExample(
                name=ex["name"],
                description=ex["description"],
                input=ex["input"],
                expected_output=ex.get("expected_output"),
                context=ex.get("context"),
                tags=ex.get("tags", [])
            ) for ex in data.get("usage_examples", [])
        ]
        
        return cls(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            purpose=data["purpose"],
            url=data["url"],
            agent_type=data.get("agent_type", "conversational"),
            domain=data.get("domain"),
            expertise_level=data.get("expertise_level", "intermediate"),
            capabilities=capabilities,
            skills=skills,
            supported_languages=data.get("supported_languages", ["en"]),
            default_input_modes=default_input_modes,
            default_output_modes=default_output_modes,
            input_schema=input_schema,
            output_schema=output_schema,
            conversation_patterns=data.get("conversation_patterns", []),
            response_styles=data.get("response_styles", []),
            interaction_modes=data.get("interaction_modes", ["chat"]),
            usage_examples=usage_examples,
            common_use_cases=data.get("common_use_cases", []),
            limitations=data.get("limitations", []),
            max_context_length=data.get("max_context_length"),
            typical_response_time=data.get("typical_response_time"),
            concurrency_support=data.get("concurrency_support", False),
            session_management=data.get("session_management", True),
            authentication=data.get("authentication", {"schemes": ["none"]}),
            provider=data.get("provider"),
            documentation_url=data.get("documentationUrl"),
            created_date=data.get("created_date"),
            last_updated=data.get("last_updated"),
            license=data.get("license")
        )


class CardManager:
    """Manager for Tool Cards and Agent Cards."""
    
    def __init__(self, cards_directory: Optional[str] = None):
        self.cards_directory = Path(cards_directory) if cards_directory else Path("cards")
        self.tool_cards: Dict[str, ToolCard] = {}
        self.agent_cards: Dict[str, AgentCard] = {}
        self.logger = get_logger(f"{self.__class__.__name__}")
    
    def register_tool_card(self, tool_card: ToolCard) -> None:
        """Register a tool card."""
        self.tool_cards[tool_card.name] = tool_card
        self.logger.info(f"Registered tool card: {tool_card.name}")
    
    def register_agent_card(self, agent_card: AgentCard) -> None:
        """Register an agent card."""
        self.agent_cards[agent_card.name] = agent_card
        self.logger.info(f"Registered agent card: {agent_card.name}")
    
    def get_tool_card(self, name: str) -> Optional[ToolCard]:
        """Get a tool card by name."""
        return self.tool_cards.get(name)
    
    def get_agent_card(self, name: str) -> Optional[AgentCard]:
        """Get an agent card by name."""
        return self.agent_cards.get(name)
    
    def save_tool_card(self, tool_card: ToolCard, format: str = "yaml") -> Path:
        """Save a tool card to file."""
        self.cards_directory.mkdir(parents=True, exist_ok=True)
        
        filename = f"{tool_card.name}_tool_card.{format}"
        file_path = self.cards_directory / "tools" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "yaml":
            with open(file_path, 'w') as f:
                yaml.dump(tool_card.to_dict(), f, default_flow_style=False, indent=2)
        elif format == "json":
            with open(file_path, 'w') as f:
                json.dump(tool_card.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Saved tool card to: {file_path}")
        return file_path
    
    def save_agent_card(self, agent_card: AgentCard, format: str = "json") -> Path:
        """Save an agent card to file (JSON for A2A compatibility)."""
        self.cards_directory.mkdir(parents=True, exist_ok=True)
        
        filename = f"{agent_card.name}_agent_card.{format}"
        file_path = self.cards_directory / "agents" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(file_path, 'w') as f:
                json.dump(agent_card.to_dict(), f, indent=2)
        elif format == "yaml":
            with open(file_path, 'w') as f:
                yaml.dump(agent_card.to_dict(), f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Saved agent card to: {file_path}")
        return file_path
    
    def load_tool_card(self, file_path: Union[str, Path]) -> ToolCard:
        """Load a tool card from file."""
        file_path = Path(file_path)
        
        if file_path.suffix == ".yaml" or file_path.suffix == ".yml":
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
        elif file_path.suffix == ".json":
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        tool_card = ToolCard.from_dict(data)
        self.register_tool_card(tool_card)
        return tool_card
    
    def load_agent_card(self, file_path: Union[str, Path]) -> AgentCard:
        """Load an agent card from file."""
        file_path = Path(file_path)
        
        if file_path.suffix == ".json":
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif file_path.suffix == ".yaml" or file_path.suffix == ".yml":
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        agent_card = AgentCard.from_dict(data)
        self.register_agent_card(agent_card)
        return agent_card
    
    def discover_cards(self) -> None:
        """Discover and load all cards from the cards directory."""
        if not self.cards_directory.exists():
            return
        
        # Load tool cards
        tools_dir = self.cards_directory / "tools"
        if tools_dir.exists():
            for card_file in tools_dir.glob("*_tool_card.*"):
                try:
                    self.load_tool_card(card_file)
                except Exception as e:
                    self.logger.warning(f"Failed to load tool card {card_file}: {e}")
        
        # Load agent cards
        agents_dir = self.cards_directory / "agents"
        if agents_dir.exists():
            for card_file in agents_dir.glob("*_agent_card.*"):
                try:
                    self.load_agent_card(card_file)
                except Exception as e:
                    self.logger.warning(f"Failed to load agent card {card_file}: {e}")
    
    def list_tool_cards(self) -> List[str]:
        """List all registered tool card names."""
        return list(self.tool_cards.keys())
    
    def list_agent_cards(self) -> List[str]:
        """List all registered agent card names."""
        return list(self.agent_cards.keys())
    
    def search_tools(self, category: Optional[str] = None, 
                    tags: Optional[List[str]] = None) -> List[ToolCard]:
        """Search tool cards by category and tags."""
        results = []
        
        for tool_card in self.tool_cards.values():
            match = True
            
            if category and tool_card.category != category:
                match = False
            
            if tags:
                tool_tags = set(tool_card.tags)
                search_tags = set(tags)
                if not search_tags.intersection(tool_tags):
                    match = False
            
            if match:
                results.append(tool_card)
        
        return results
    
    def search_agents(self, agent_type: Optional[str] = None,
                     domain: Optional[str] = None,
                     skills: Optional[List[str]] = None) -> List[AgentCard]:
        """Search agent cards by type, domain, and skills."""
        results = []
        
        for agent_card in self.agent_cards.values():
            match = True
            
            if agent_type and agent_card.agent_type != agent_type:
                match = False
            
            if domain and agent_card.domain != domain:
                match = False
            
            if skills:
                agent_skills = {skill.id for skill in agent_card.skills}
                search_skills = set(skills)
                if not search_skills.intersection(agent_skills):
                    match = False
            
            if match:
                results.append(agent_card)
        
        return results


# Global card manager instance
_card_manager = None

def get_card_manager(cards_directory: Optional[str] = None) -> CardManager:
    """Get the global card manager instance."""
    global _card_manager
    if _card_manager is None:
        _card_manager = CardManager(cards_directory)
    return _card_manager