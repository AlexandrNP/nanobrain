"""
Prompt Template Manager for NanoBrain Framework

Provides dynamic prompt loading and template variable substitution for agents.

G14 — extended 2026-05-09 with the typed-template contract per
``apecx-mcp-integration/docs/llm_prompt_contracts.md §3`` and
``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G14``. New fields
are additive: existing PromptTemplate consumers continue to work
unchanged. G14-compliant templates declare ``template_id``,
``content_hash``, ``model_constraint``, ``holes``, ``system_prompt``,
``user_template``, ``output_schema_ref``, and (optional)
``regression_fixtures``.
"""

import hashlib
import json
import re
import yaml
import logging
from typing import Dict, Any, Literal, Optional, List, Union
from pathlib import Path
from string import Template
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from .logging_system import get_logger
# Import new ConfigBase for constructor prohibition
from .config.config_base import ConfigBase

logger = logging.getLogger(__name__)


# G14 — recognized hole types. Subset of JSON Schema's type vocabulary;
# matches the skeleton hole grammar from agent_workflow_authoring.md §4.1.
PromptHoleType = Literal["string", "integer", "number", "boolean", "array", "object", "any"]


class PromptHole(BaseModel):
    """G14 — a typed parameter slot in a prompt template.

    Per ``llm_prompt_contracts.md §3``: each hole declares its type,
    whether it's required, an optional default for optional holes, and
    a human description. The PromptTemplateManager substitutes hole
    values at render time and FAIL-FASTs on missing required holes.
    """
    model_config = ConfigDict(extra="forbid")

    type: PromptHoleType = "string"
    required: bool = True
    default: Any = None
    description: str = ""


# Regex matching template_id grammar: <family>.<name>@<semver>.
# Examples:
#   phase0_planning.default@1.4.0
#   skeleton_selection.exhaustive@2.0.1-rc.1
_TEMPLATE_ID_RE = re.compile(
    r"^(?P<family>[a-z][a-z0-9_]*)\.(?P<name>[a-z][a-z0-9_]*)@"
    r"(?P<semver>\d+\.\d+\.\d+(?:-[0-9A-Za-z\-.]+)?)$"
)


def compute_template_content_hash(
    *,
    system_prompt: str = "",
    user_template: str = "",
    holes: Optional[Dict[str, Any]] = None,
    output_schema_ref: Optional[Dict[str, Any]] = None,
) -> str:
    """G14 — compute the canonical content hash for a prompt template.

    The hash covers the template body fields that determine the LLM
    output: system_prompt, user_template, declared holes (sorted keys),
    and output_schema_ref. NON-content fields (description,
    model_constraint, regression_fixtures) are excluded — they describe
    the template, not its semantics.

    The same template body produces the same hash regardless of YAML
    ordering, comment placement, or quoting style. Used for provenance:
    every LLM call records the (template_id, content_hash) pair so a
    post-hoc audit can detect template tampering.
    """
    holes = holes or {}
    canonical = json.dumps(
        {
            "system_prompt": system_prompt,
            "user_template": user_template,
            # Sort holes by key for deterministic hashing.
            "holes": {k: holes[k] for k in sorted(holes.keys())},
            "output_schema_ref": output_schema_ref,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PromptTemplate(BaseModel):
    """A single prompt template.

    Carries two contracts side-by-side:

    1. **Legacy v1 contract**: ``template`` (single string),
       ``required_params``, ``optional_params``, ``examples``. Existing
       consumers continue to work; the PromptTemplateManager.get_prompt
       path is unchanged.
    2. **G14 contract**: ``template_id``, ``content_hash``,
       ``model_constraint``, ``holes``, ``system_prompt``,
       ``user_template``, ``output_schema_ref``, ``regression_fixtures``.
       Per ``llm_prompt_contracts.md §3``: this is the
       provenance-anchored, content-hash-pinned, gate-bindable shape
       that production prompts SHOULD migrate to.

    A template is **G14-compliant** when ``template_id`` is set AND at
    least one of ``system_prompt`` / ``user_template`` is set. Compliant
    templates auto-compute their ``content_hash`` if the field is empty
    or set to the literal sentinel ``"<computed-at-load>"``.
    """
    model_config = ConfigDict(extra="allow")

    # Legacy v1 fields (preserved for backward compat).
    template: str = ""
    description: Optional[str] = None
    required_params: List[str] = Field(default_factory=list)
    optional_params: List[str] = Field(default_factory=list)
    examples: List[Dict[str, Any]] = Field(default_factory=list)

    # G14 fields (all optional so legacy templates load unchanged).
    template_id: Optional[str] = Field(
        default=None,
        description="G14 — semver-pinned name '<family>.<name>@<semver>'. "
                    "Required for G14 compliance.",
    )
    content_hash: Optional[str] = Field(
        default=None,
        description="G14 — SHA-256 of the canonical template body. "
                    "Auto-computed at load time when empty or set to "
                    "the sentinel '<computed-at-load>'.",
    )
    model_constraint: List[str] = Field(
        default_factory=list,
        description="G14 — list of acceptable model identifiers. Empty "
                    "list = any model (not recommended for production).",
    )
    holes: Dict[str, PromptHole] = Field(
        default_factory=dict,
        description="G14 — typed parameter slots. Each hole has type, "
                    "required flag, optional default, and description.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="G14 — system-role message body. The agent's "
                    "system_prompt: field is hydrated from this when set.",
    )
    user_template: Optional[str] = Field(
        default=None,
        description="G14 — user-role message template, with $hole "
                    "substitution placeholders (string.Template grammar).",
    )
    output_schema_ref: Optional[Dict[str, Any]] = Field(
        default=None,
        description="G14 — reference to a G6 SchemaRef shape, e.g. "
                    "{class: 'pkg.mod.OutputModel'} or {json_schema: {...}}. "
                    "The downstream validator step uses this to enforce "
                    "the LLM's output shape.",
    )
    regression_fixtures: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="G14 — list of {input: {...}, expected_output_contract: {...}} "
                    "fixtures used by the prompt's regression test suite. "
                    "Each fixture exercises one expected behavior.",
    )

    @field_validator("template_id")
    @classmethod
    def _validate_template_id(cls, v: Optional[str]) -> Optional[str]:
        """When set, template_id MUST match the canonical grammar.
        Empty / None passes through (legacy templates have no id)."""
        if v is None:
            return v
        if not _TEMPLATE_ID_RE.match(v):
            raise ValueError(
                f"FAIL-FAST: template_id={v!r} must match "
                f"'<family>.<name>@<semver>' (e.g. 'phase0_planning.default@1.4.0'); "
                f"family + name lowercase snake_case, semver per semver.org"
            )
        return v

    @model_validator(mode="after")
    def _resolve_content_hash(self) -> "PromptTemplate":
        """Auto-compute content_hash when set to the sentinel or empty
        AND the template is G14-compliant. Legacy templates without
        template_id leave content_hash alone."""
        if not self.template_id:
            return self  # legacy template — skip hash computation
        sentinels = (None, "", "<computed-at-load>")
        if self.content_hash in sentinels:
            self.content_hash = compute_template_content_hash(
                system_prompt=self.system_prompt or "",
                user_template=self.user_template or self.template or "",
                holes={k: v.model_dump() for k, v in self.holes.items()},
                output_schema_ref=self.output_schema_ref,
            )
        return self

    @property
    def is_g14_compliant(self) -> bool:
        """A template is G14-compliant when it has both template_id and
        at least one body field (system_prompt or user_template)."""
        return bool(
            self.template_id
            and (self.system_prompt or self.user_template)
        )

    @property
    def template_family(self) -> Optional[str]:
        """Extract the family segment from template_id, or None if legacy."""
        if not self.template_id:
            return None
        m = _TEMPLATE_ID_RE.match(self.template_id)
        return m.group("family") if m else None

    @property
    def template_semver(self) -> Optional[str]:
        """Extract the semver segment from template_id, or None if legacy."""
        if not self.template_id:
            return None
        m = _TEMPLATE_ID_RE.match(self.template_id)
        return m.group("semver") if m else None

    def render(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """G14 — render system_prompt and user_template with $hole substitution.

        Returns a dict with two keys: ``system`` (rendered system_prompt
        or empty) and ``user`` (rendered user_template or empty).

        Required holes that are absent raise ValueError. Optional holes
        with defaults substitute the default; optional holes without
        defaults substitute an empty string.

        Legacy templates (no template_id) raise ValueError; legacy
        consumers should keep using PromptTemplateManager.get_prompt
        instead.
        """
        if not self.is_g14_compliant:
            raise ValueError(
                "PromptTemplate.render() is the G14 path; legacy templates "
                "(no template_id) must use PromptTemplateManager.get_prompt"
            )

        params = dict(params or {})

        # Apply defaults for absent optional holes; FAIL-FAST on absent required.
        for name, hole in self.holes.items():
            if name in params:
                continue
            if hole.required:
                raise ValueError(
                    f"FAIL-FAST: PromptTemplate {self.template_id!r} render "
                    f"missing required hole {name!r}"
                )
            params[name] = hole.default if hole.default is not None else ""

        rendered_system = ""
        rendered_user = ""

        if self.system_prompt:
            rendered_system = Template(self.system_prompt).safe_substitute(**params)
        if self.user_template:
            rendered_user = Template(self.user_template).safe_substitute(**params)

        return {"system": rendered_system, "user": rendered_user}
    

class PromptTemplateConfig(ConfigBase):
    """
    Configuration for prompt templates - INHERITS constructor prohibition.
    
    ❌ FORBIDDEN: PromptTemplateConfig(prompts={...}, ...)
    ✅ REQUIRED: PromptTemplateConfig.from_config('path/to/config.yml')
    """
    
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
        # FIXED: Don't instantiate empty PromptTemplateConfig - violates framework rules
        self.templates: Optional[PromptTemplateConfig] = None
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
            # Load from dictionary - store directly as we can't use from_config with dict
            self.templates = PromptTemplateConfig.model_validate(source)
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists() and path.is_file():
                # Load from file path using from_config (framework requirement)
                self.templates = PromptTemplateConfig.from_config(str(path))
                logger.info(f"Loaded prompt templates from {path}")
            else:
                raise ValueError(f"Template file not found: {source}")
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