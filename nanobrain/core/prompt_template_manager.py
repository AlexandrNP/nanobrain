"""
Prompt Template Manager for NanoBrain Framework

Core utility for managing prompt templates across all NanoBrain components.
Provides loading, validation, formatting, and caching of prompt templates.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import yaml
import logging
from string import Template
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


class PromptTemplate(BaseModel):
    """Model for a single prompt template."""
    
    model_config = ConfigDict(extra="allow")
    
    template: str
    description: Optional[str] = None
    required_params: List[str] = Field(default_factory=list)
    optional_params: List[str] = Field(default_factory=list)
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    

class PromptTemplateConfig(BaseModel):
    """Configuration for prompt templates."""
    
    model_config = ConfigDict(extra="allow")
    
    prompts: Dict[str, PromptTemplate] = Field(default_factory=dict)
    contexts: Dict[str, PromptTemplate] = Field(default_factory=dict)
    version: str = "1.0.0"
    description: Optional[str] = None


class PromptTemplateManager:
    """
    Core utility for managing prompt templates across NanoBrain.
    
    This manager provides:
    - Loading templates from YAML files or dictionaries
    - Template validation and parameter checking
    - Safe template formatting with parameter substitution
    - Template versioning and updates
    - Context template support
    """
    
    def __init__(self, 
                 template_source: Optional[Union[str, Path, Dict[str, Any]]] = None,
                 enable_validation: bool = True):
        """
        Initialize the prompt template manager.
        
        Args:
            template_source: Path to YAML file, dict, or None
            enable_validation: Whether to validate templates on load
        """
        self.templates: PromptTemplateConfig = PromptTemplateConfig()
        self.enable_validation = enable_validation
        self._template_cache: Dict[str, Template] = {}
        
        if template_source:
            self.load_templates(template_source)
    
    def load_templates(self, source: Union[str, Path, Dict[str, Any]]) -> None:
        """
        Load templates from various sources.
        
        Args:
            source: YAML file path, dictionary, or YAML string
        """
        if isinstance(source, dict):
            # Load from dictionary
            self.templates = PromptTemplateConfig(**source)
        elif isinstance(source, (str, Path)):
            # First try to parse as YAML string (if it's a string and contains newlines or colons)
            if isinstance(source, str) and ('\n' in source or (': ' in source and not source.endswith('.yml') and not source.endswith('.yaml'))):
                try:
                    data = yaml.safe_load(source)
                    if isinstance(data, dict):
                        self.templates = PromptTemplateConfig(**data)
                    else:
                        raise ValueError(f"Invalid YAML content: expected dict, got {type(data)}")
                except yaml.YAMLError as e:
                    raise ValueError(f"Invalid template source - not valid YAML: {e}")
            else:
                # Try as file path
                try:
                    path = Path(source)
                    if path.exists() and path.is_file():
                        # Load from file
                        with open(path, 'r') as f:
                            data = yaml.safe_load(f)
                            self.templates = PromptTemplateConfig(**data)
                        logger.info(f"Loaded prompt templates from {path}")
                    else:
                        # Last attempt: parse as YAML string
                        try:
                            data = yaml.safe_load(str(source))
                            if isinstance(data, dict):
                                self.templates = PromptTemplateConfig(**data)
                            else:
                                raise ValueError(f"Invalid YAML content: expected dict, got {type(data)}")
                        except yaml.YAMLError:
                            raise ValueError(f"Template source not found as file and not valid YAML: {source}")
                except (OSError, ValueError) as e:
                    # If path operations fail, try as YAML string
                    try:
                        data = yaml.safe_load(str(source))
                        if isinstance(data, dict):
                            self.templates = PromptTemplateConfig(**data)
                        else:
                            raise ValueError(f"Invalid YAML content: expected dict, got {type(data)}")
                    except yaml.YAMLError:
                        raise ValueError(f"Invalid template source: {source}")
        else:
            raise ValueError(f"Invalid template source type: {type(source)}")
        
        if self.enable_validation:
            self.validate_templates()
        
        # Clear cache when loading new templates
        self._template_cache.clear()
    
    def get_prompt(self, 
                  prompt_name: str, 
                  params: Optional[Dict[str, Any]] = None,
                  include_contexts: Optional[List[str]] = None) -> str:
        """
        Get a formatted prompt by name.
        
        Args:
            prompt_name: Name of the prompt template
            params: Parameters for template substitution
            include_contexts: List of context templates to prepend
            
        Returns:
            Formatted prompt string
        """
        if prompt_name not in self.templates.prompts:
            raise KeyError(f"Prompt '{prompt_name}' not found")
        
        prompt_template = self.templates.prompts[prompt_name]
        params = params or {}
        
        # Add contexts if specified
        full_template = ""
        if include_contexts:
            for context_name in include_contexts:
                if context_name in self.templates.contexts:
                    context_template = self.templates.contexts[context_name]
                    full_template += context_template.template + "\n\n"
        
        full_template += prompt_template.template
        
        # Use cached Template object for performance
        cache_key = f"{prompt_name}:{','.join(include_contexts or [])}"
        if cache_key not in self._template_cache:
            self._template_cache[cache_key] = Template(full_template)
        
        template_obj = self._template_cache[cache_key]
        
        # Safe substitution (missing params won't raise errors)
        try:
            return template_obj.safe_substitute(**params)
        except Exception as e:
            logger.error(f"Error formatting prompt '{prompt_name}': {e}")
            raise
    
    def validate_templates(self) -> List[str]:
        """
        Validate all loaded templates.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate each prompt
        for name, prompt in self.templates.prompts.items():
            if not prompt.template:
                errors.append(f"Prompt '{name}' has empty template")
            
            # Check if template is valid
            try:
                Template(prompt.template)
            except Exception as e:
                errors.append(f"Prompt '{name}' has invalid template: {e}")
        
        # Validate contexts
        for name, context in self.templates.contexts.items():
            if not context.template:
                errors.append(f"Context '{name}' has empty template")
        
        if errors:
            logger.warning(f"Template validation found {len(errors)} errors")
        
        return errors
    
    def list_prompts(self) -> List[str]:
        """Get list of available prompt names."""
        return list(self.templates.prompts.keys())
    
    def list_contexts(self) -> List[str]:
        """Get list of available context names."""
        return list(self.templates.contexts.keys())
    
    def get_prompt_info(self, prompt_name: str) -> Dict[str, Any]:
        """Get detailed information about a prompt."""
        if prompt_name not in self.templates.prompts:
            raise KeyError(f"Prompt '{prompt_name}' not found")
        
        prompt = self.templates.prompts[prompt_name]
        return {
            "description": prompt.description,
            "required_params": prompt.required_params,
            "optional_params": prompt.optional_params,
            "examples": prompt.examples,
            "template_preview": prompt.template[:200] + "..." if len(prompt.template) > 200 else prompt.template
        }
    
    def update_prompt(self, prompt_name: str, template: str, **kwargs) -> None:
        """Update or create a prompt template."""
        if prompt_name not in self.templates.prompts:
            self.templates.prompts[prompt_name] = PromptTemplate(template=template)
        else:
            self.templates.prompts[prompt_name].template = template
        
        # Update other fields if provided
        for key, value in kwargs.items():
            if hasattr(self.templates.prompts[prompt_name], key):
                setattr(self.templates.prompts[prompt_name], key, value)
        
        # Clear cache for this prompt
        self._clear_cache_for_prompt(prompt_name)
    
    def save_templates(self, file_path: Union[str, Path]) -> None:
        """Save current templates to YAML file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.templates.model_dump(exclude_none=True)
        
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved prompt templates to {file_path}")
    
    def _clear_cache_for_prompt(self, prompt_name: str) -> None:
        """Clear template cache entries for a specific prompt."""
        keys_to_remove = [k for k in self._template_cache.keys() if k.startswith(f"{prompt_name}:")]
        for key in keys_to_remove:
            del self._template_cache[key]
    
    def merge_templates(self, other: Union['PromptTemplateManager', Dict[str, Any]]) -> None:
        """Merge templates from another manager or dictionary."""
        if isinstance(other, PromptTemplateManager):
            other_data = other.templates.model_dump()
        else:
            other_data = other
        
        # Merge prompts
        if 'prompts' in other_data:
            for name, prompt_data in other_data['prompts'].items():
                self.templates.prompts[name] = PromptTemplate(**prompt_data)
        
        # Merge contexts
        if 'contexts' in other_data:
            for name, context_data in other_data['contexts'].items():
                self.templates.contexts[name] = PromptTemplate(**context_data)
        
        # Clear cache after merge
        self._template_cache.clear()
    
    def extract_template_params(self, prompt_name: str) -> Dict[str, List[str]]:
        """
        Extract parameter placeholders from a template.
        
        Returns:
            Dict with 'found' and 'missing' parameter lists
        """
        if prompt_name not in self.templates.prompts:
            raise KeyError(f"Prompt '{prompt_name}' not found")
        
        prompt = self.templates.prompts[prompt_name]
        template_obj = Template(prompt.template)
        
        # Extract all placeholders
        import re
        pattern = re.compile(r'\$\{([^}]+)\}|\$([a-zA-Z_][a-zA-Z0-9_]*)')
        found_params = set()
        
        for match in pattern.finditer(prompt.template):
            param_name = match.group(1) or match.group(2)
            found_params.add(param_name)
        
        # Compare with declared params
        declared_params = set(prompt.required_params + prompt.optional_params)
        
        return {
            'found': list(found_params),
            'missing': list(found_params - declared_params),
            'unused': list(declared_params - found_params)
        } 