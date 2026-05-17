"""PromptTemplate — G14 framework primitive (2026-05-17).

A first-class prompt object that:
  * Loads from YAML config (system_prompt OR system_prompt_file;
    user_template OR user_template_file).
  * Supports variable substitution via Python ``str.format(**params)``.
  * Carries ``regression_fixtures`` for the G25 PromptRegressionHarness.
  * Exposes ``content_hash`` (SHA-256 of system + user templates) so
    snapshot caches invalidate when the template body changes.
  * Compatible with the G25 harness's expected shape:
    ``template_id`` + ``render(params) -> {"system": ..., "user": ...}``.

Why this exists
===============

Pre-G14 the workspace used hand-rolled prompt files (e.g.,
``apecx-mcp-integration/composition/composer_prompts/system.md``)
loaded by ad-hoc ``Path(...).read_text()`` calls in each Step's
``_init_from_config``. The workaround had three problems:

  1. **No versioning**. A prompt edit silently changed step behavior
     with no trace in test snapshots.
  2. **No regression fixtures**. The G25 harness was designed against
     a PromptTemplate that didn't exist; tests had to construct ad-hoc
     mock objects with the right duck-typed shape.
  3. **No reuse**. Each Step re-implemented prompt loading + parameter
     substitution + file-path resolution.

This primitive closes all three. After G14 ships, the workaround in
``apecx-mcp-integration/docs/WORKAROUND_INVENTORY.md`` Phase 2 G14
row can be retired.

Design choices
==============

* **Lenient on shape**: either ``system_prompt`` OR ``user_template``
  may be omitted. Allows system-only prompts (the common case for
  code-gen) and user-only prompts (rare but valid for completions).
  At least ONE must be present.
* **str.format substitution**: chosen over Jinja2 for zero runtime
  deps. Variables use ``{var_name}`` syntax. Operators wanting Jinja
  can wrap.
* **Missing variables fail loudly**: ``render({})`` with a template
  containing ``{var}`` raises KeyError with the missing name. No
  silent empty substitution.
* **regression_fixtures field on the config**, not a method —
  declarative + serializable + supports YAML authoring of test cases
  alongside the template body.

Honest scope limitations
========================

* No conditional logic in templates (no ``{% if %}`` blocks). For
  multi-shape prompts, author multiple PromptTemplates + select in
  the calling Step.
* No partial rendering (no leaving placeholders unfilled). All
  ``{var}`` must be in the params dict.
* The content_hash is over the RAW template strings (post-file-load,
  pre-substitution). Two templates that render identically but have
  different raw strings will have different hashes. Trade-off:
  hashes are cheap + deterministic; rendering each variant is more
  expensive.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import ConfigDict, Field, model_validator

from .component_base import ComponentConfigurationError, FromConfigBase
from .config.config_base import ConfigBase


class PromptTemplateConfig(ConfigBase):
    """Configuration for ``PromptTemplate``.

    Either ``system_prompt`` OR ``system_prompt_file`` may be set
    (not both). Same for ``user_template`` / ``user_template_file``.
    At least one of system/user (in either form) must resolve to
    a non-empty string.

    ``extra='forbid'`` enforces the workspace rule: YAML typos
    fail at config load rather than silently using defaults.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=False)

    # Framework tracking attribute set by ConfigBase.from_config.
    source_path: Optional[str] = Field(default=None)

    template_id: str = Field(
        ...,
        description=(
            "Stable identifier for this template. Used by the G25 "
            "regression harness for snapshot file naming. Should be "
            "a short, filesystem-safe string (e.g., 'code_writer_v1')."
        ),
    )

    system_prompt: Optional[str] = Field(
        default=None,
        description="Inline system prompt string. Mutually exclusive with system_prompt_file.",
    )

    system_prompt_file: Optional[str] = Field(
        default=None,
        description=(
            "Path to a file containing the system prompt. Relative "
            "paths resolve against this YAML's directory (G40 helper)."
        ),
    )

    user_template: Optional[str] = Field(
        default=None,
        description=(
            "Inline user-message template string. Supports ``{var}`` "
            "substitution via str.format(**params). Mutually exclusive "
            "with user_template_file."
        ),
    )

    user_template_file: Optional[str] = Field(
        default=None,
        description=(
            "Path to a file containing the user template. Relative "
            "paths resolve against this YAML's directory."
        ),
    )

    regression_fixtures: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of fixtures for the G25 PromptRegressionHarness. "
            "Each fixture: {input: dict, contract: dict}. The harness "
            "renders the template with input, invokes the LLM, and "
            "validates the response against contract."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _strip_framework_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data.pop("class", None)
        return data

    @model_validator(mode="after")
    def _validate_one_source_per_role(self) -> "PromptTemplateConfig":
        """Enforce 'system_prompt OR system_prompt_file, not both'
        on each role; and ensure at least one of system/user is set."""
        if self.system_prompt is not None and self.system_prompt_file is not None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: PromptTemplate {self.template_id!r}: set EITHER "
                f"system_prompt OR system_prompt_file, not both."
            )
        if self.user_template is not None and self.user_template_file is not None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: PromptTemplate {self.template_id!r}: set EITHER "
                f"user_template OR user_template_file, not both."
            )
        has_system = bool(self.system_prompt) or bool(self.system_prompt_file)
        has_user = bool(self.user_template) or bool(self.user_template_file)
        if not has_system and not has_user:
            raise ComponentConfigurationError(
                f"FAIL-FAST: PromptTemplate {self.template_id!r}: at least one "
                f"of (system_prompt, system_prompt_file, user_template, "
                f"user_template_file) must be non-empty."
            )
        return self


class PromptTemplate(FromConfigBase):
    """A first-class prompt object with substitution + regression fixtures.

    Usage::

        template = PromptTemplate.from_config("my_prompt.yml")
        rendered = template.render({"function_name": "fib"})
        # rendered == {"system": "...", "user": "Write a function fib..."}

        # G25 harness integration:
        from nanobrain.library.testing import PromptRegressionHarness
        harness = PromptRegressionHarness(template=template, llm_callable=...)
        report = await harness.run()

    The ``content_hash`` property is the SHA-256 of (system, user)
    template strings — change either and the hash changes, which
    invalidates any G25 snapshot cache automatically.
    """

    COMPONENT_TYPE: ClassVar[str] = "prompt_template"
    REQUIRED_CONFIG_FIELDS: ClassVar[List[str]] = ["template_id"]

    @classmethod
    def _get_config_class(cls):
        return PromptTemplateConfig

    @classmethod
    def extract_component_config(cls, config: PromptTemplateConfig) -> Dict[str, Any]:
        return {
            "template_id": config.template_id,
            "system_prompt": config.system_prompt,
            "system_prompt_file": config.system_prompt_file,
            "user_template": config.user_template,
            "user_template_file": config.user_template_file,
            "regression_fixtures": list(config.regression_fixtures),
            "source_path": getattr(config, "source_path", None),
        }

    @classmethod
    def resolve_dependencies(
        cls, component_config: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        return {}

    def _init_from_config(
        self,
        config: PromptTemplateConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        self._template_id: str = component_config["template_id"]
        self._regression_fixtures: List[Dict[str, Any]] = list(
            component_config.get("regression_fixtures") or []
        )

        # Resolve system + user template strings (load files if needed).
        source_path = component_config.get("source_path")
        self._system_prompt: str = self._resolve_template_source(
            inline=component_config.get("system_prompt"),
            file_path=component_config.get("system_prompt_file"),
            source_path=source_path,
            role="system",
        )
        self._user_template: str = self._resolve_template_source(
            inline=component_config.get("user_template"),
            file_path=component_config.get("user_template_file"),
            source_path=source_path,
            role="user",
        )

        # Pre-compute content hash for G25 snapshot invalidation.
        self._content_hash: str = self._compute_content_hash()

    def _resolve_template_source(
        self,
        *,
        inline: Optional[str],
        file_path: Optional[str],
        source_path: Optional[str],
        role: str,
    ) -> str:
        """Return the template string — either the inline value or
        the contents of the referenced file. Empty string when both
        are None (one of system/user may be missing, validated by
        the config's _validate_one_source_per_role)."""
        if inline is not None:
            return inline
        if file_path is None:
            return ""
        p = Path(file_path)
        if not p.is_absolute() and source_path:
            p = (Path(source_path).resolve().parent / p).resolve()
        elif not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        try:
            content = p.read_text(encoding="utf-8")
        except OSError as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: PromptTemplate {self._template_id!r}: failed to "
                f"read {role} template at {p}: {e}"
            ) from e
        if not content.strip():
            raise ComponentConfigurationError(
                f"FAIL-FAST: PromptTemplate {self._template_id!r}: {role} "
                f"template at {p} is empty"
            )
        return content

    def _compute_content_hash(self) -> str:
        """SHA-256 of (system + user) raw template strings. Prefixed
        with 'sha256:' for forward-compatibility with future hash
        algos."""
        h = hashlib.sha256()
        h.update(self._system_prompt.encode("utf-8"))
        h.update(b"\x00")
        h.update(self._user_template.encode("utf-8"))
        return "sha256:" + h.hexdigest()

    # ---- G25 harness compatibility surface ----

    @property
    def template_id(self) -> str:
        """G25 harness reads this for snapshot file naming."""
        return self._template_id

    @property
    def content_hash(self) -> str:
        """G25 harness reads this for snapshot invalidation."""
        return self._content_hash

    @property
    def regression_fixtures(self) -> List[Dict[str, Any]]:
        """G25 harness iterates over these for regression runs."""
        return list(self._regression_fixtures)

    @property
    def system_prompt(self) -> str:
        """Raw system template string (post-file-load, pre-substitution).
        Exposed for callers that don't need variable rendering."""
        return self._system_prompt

    @property
    def user_template(self) -> str:
        """Raw user template string (post-file-load, pre-substitution)."""
        return self._user_template

    def render(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Render the template with the given parameters.

        Returns ``{"system": rendered_system, "user": rendered_user}``.
        Either field may be an empty string if the corresponding
        template was not set.

        Variables use ``{var}`` syntax (str.format). Missing
        variables raise ``KeyError`` with the missing name — NO
        silent empty substitution.

        Args:
            params: dict of variables to substitute. Pass an empty
                dict for templates with no variables.

        Returns:
            dict with two keys: ``"system"`` and ``"user"``.

        Raises:
            KeyError: when a template references ``{var}`` and
                ``var`` is not in params.
        """
        if not isinstance(params, dict):
            raise TypeError(
                f"PromptTemplate {self._template_id!r}: render params must "
                f"be a dict, got {type(params).__name__}"
            )
        try:
            rendered_system = (
                self._system_prompt.format(**params) if self._system_prompt else ""
            )
            rendered_user = (
                self._user_template.format(**params) if self._user_template else ""
            )
        except KeyError as e:
            raise KeyError(
                f"PromptTemplate {self._template_id!r}: missing parameter "
                f"{e.args[0]!r} for template rendering"
            ) from e
        return {"system": rendered_system, "user": rendered_user}


__all__ = ["PromptTemplate", "PromptTemplateConfig"]
