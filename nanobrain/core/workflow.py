"""
Workflow System for NanoBrain Framework

Provides graph-based workflow orchestration extending the Step system.
Workflows can contain multiple steps connected by links, with support for
hierarchical configuration, various execution strategies, and comprehensive
progress reporting with persistent checkpoints.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Literal, Optional, Union, Set, Tuple
from pathlib import Path

from pydantic import Field, model_validator
import yaml
import json
from datetime import datetime

from .step import BaseStep, Step, StepConfig
from .data_unit import DataUnitBase
from .link import LinkBase
from .executor import LocalExecutor
from .logging_system import get_logger, OperationType
from .workflow_progress import ProgressReporter
from .workflow_graph import WorkflowGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# G7 Step 2 — auto_transfer deprecation WARNING (added 2026-05-09)
# ---------------------------------------------------------------------------
#
# Per `apecx-mcp-integration/docs/nanobrain_capability_gaps.md G7`: the
# `LinkBase.auto_transfer` field defaults to False, meaning a `DirectLink`
# (or subclass) without an explicit `auto_transfer: true` in YAML silently
# no-ops on every trigger fire. The workflow loads cleanly, every step
# runs, no exception is raised — but no data ever transfers. This is the
# dominant silent-failure shape documented in `architecture.md §13`
# brutal-truth #3.
#
# Step 2 of the G7 migration plan is to emit a deprecation WARNING at
# workflow load when a v1 workflow (the current default) has a Link config
# that omits `auto_transfer`. Authors silence the warning by being
# explicit (`auto_transfer: true` if you want transfer, `auto_transfer: false`
# if you genuinely want to suppress it).
#
# Step 3 (land v2 semantics) and Step 4 (flip new-workflow default to v2)
# are deferred — they need cross-repo coordination per the gap doc's
# Migration section.
# ---------------------------------------------------------------------------

# Tokens we look for in `class:` strings to decide whether a link config
# is one of the link subclasses subject to the auto_transfer warning.
# Prefix-matched against the class path; case-sensitive.
_LINK_CLASSES_NEEDING_AUTO_TRANSFER = (
    "DirectLink",
    "TransformLink",
    "ConditionalLink",
    "FileLink",
    "QueueLink",
)

# G10 Step 2 — link classes whose `gate_semantics` field is meaningful.
# Today only ConditionalLink reads gate_semantics on the link side, but
# the propagation logic checks against this list to stay forward-
# compatible with future gated link types.
_LINK_CLASSES_NEEDING_GATE_SEMANTICS = (
    "ConditionalLink",
)

# G10 Step 2 — trigger classes whose `gate_semantics` field is meaningful.
# AllDataReceivedTrigger is the consumer that observes sentinel-bearing
# data units; ManualTrigger / TimerTrigger / DataUnitChangeTrigger /
# EventTrigger don't gate, so propagation skips them.
_TRIGGER_CLASSES_NEEDING_GATE_SEMANTICS = (
    "AllDataReceivedTrigger",
)


def _link_class_needs_auto_transfer_check(class_path: str) -> bool:
    """Return True if `class_path` names a link class whose auto_transfer
    semantics matter. Conservative: returns True only when the trailing
    name in the dotted path matches one of the known link subclasses.
    Returns False for AcademyLink and any unknown classes.
    """
    if not isinstance(class_path, str):
        return False
    trailing = class_path.rsplit(".", 1)[-1]
    return trailing in _LINK_CLASSES_NEEDING_AUTO_TRANSFER


def _link_class_needs_gate_semantics_check(class_path: str) -> bool:
    """G10 Step 2 — does this link class read gate_semantics?"""
    if not isinstance(class_path, str):
        return False
    trailing = class_path.rsplit(".", 1)[-1]
    return trailing in _LINK_CLASSES_NEEDING_GATE_SEMANTICS


def _trigger_class_needs_gate_semantics_check(class_path: str) -> bool:
    """G10 Step 2 — does this trigger class read gate_semantics?"""
    if not isinstance(class_path, str):
        return False
    trailing = class_path.rsplit(".", 1)[-1]
    return trailing in _TRIGGER_CLASSES_NEEDING_GATE_SEMANTICS


def _set_inline_default_in_entry(
    entry: Dict[str, Any],
    field: str,
    value: Any,
    rewrite_path_reference: bool = False,
    workflow_directory: Optional[str] = None,
) -> bool:
    """G10 Step 2 / G7 Step 3 / G7 Step 4 helper. Mutate ``entry`` in
    place to set ``field=value`` if absent. Handles three shapes:

    - Nested: ``{class: ..., config: {...}}`` — mutate ``config`` dict.
    - Flat:   ``{class: ..., <field-keys>: ...}`` — mutate ``entry`` itself.
    - Path-reference: ``{class: ..., config: "external.yml"}`` — when
      ``rewrite_path_reference=True``, load the external YAML, inject
      the field if absent, and rewrite the entry to NESTED inline form
      (``config: <loaded-dict>``) so downstream consumers see a single
      uniform shape. The original file is NOT modified — the rewrite
      happens in-memory on the parent WorkflowConfig dict only.

    Args:
        entry: Link/trigger entry dict to mutate.
        field: Field name to set (e.g., ``auto_transfer``, ``gate_semantics``).
        value: Value to set when the field is absent.
        rewrite_path_reference: When True (G7 Step 4 caller), path-
            reference configs are loaded and rewritten. When False
            (legacy G7 Step 3 / G10 Step 2 path), they are skipped.
        workflow_directory: Base directory for resolving relative paths
            in ``config:`` strings. When None, paths resolve relative
            to CWD (matches ConfigBase.from_config search order).

    Returns True if the field was set (including the path-reference
    rewrite + set case); False if it was already present or path-
    reference was skipped.
    """
    if not isinstance(entry, dict):
        return False
    inner = entry.get("config")
    if isinstance(inner, dict):
        if field in inner:
            return False
        inner[field] = value
        return True
    if isinstance(inner, str):
        if not rewrite_path_reference:
            # Legacy behavior — path-reference; cannot inject without
            # loading the YAML.
            return False
        # G7 Step 4 — load + inject + rewrite in-memory.
        loaded = _load_path_reference_config(inner, workflow_directory)
        if not isinstance(loaded, dict):
            raise ValueError(
                f"FAIL-FAST: path-reference config {inner!r} loaded as "
                f"{type(loaded).__name__}, expected dict"
            )
        if field not in loaded:
            loaded[field] = value
            mutated = True
        else:
            mutated = False
        entry["config"] = loaded
        return mutated
    # Flat shape: entry itself is the config dict
    if field in entry:
        return False
    entry[field] = value
    return True


def _load_path_reference_config(
    path_str: str, workflow_directory: Optional[str],
) -> Any:
    """Load a path-reference config string to its YAML body.

    Resolution order matches the framework's standard:
        1. absolute path
        2. relative to ``workflow_directory`` when set
        3. relative to current working directory

    Lazy import of yaml + pathlib to avoid import-time cycles.
    """
    import yaml as _yaml
    from pathlib import Path as _Path

    candidates: List["_Path"] = []
    p = _Path(path_str)
    if p.is_absolute():
        candidates.append(p)
    else:
        if workflow_directory:
            candidates.append(_Path(workflow_directory) / p)
        candidates.append(_Path.cwd() / p)
        candidates.append(p)  # last-ditch: pass through

    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as f:
                return _yaml.safe_load(f)

    raise FileNotFoundError(
        f"FAIL-FAST: path-reference config {path_str!r} not found in any "
        f"of: {[str(c) for c in candidates]}"
    )


def _link_inline_config_omits_auto_transfer(link_entry: Any) -> Optional[bool]:
    """Return:
        True  — link entry is an inline config dict and `auto_transfer` is missing
        False — link entry has explicit `auto_transfer` key (any value)
        None  — cannot statically determine (path-reference config, or already
                a resolved LinkBase instance, or shape we don't recognize)

    Two YAML shapes for inline configs:
        1. Top-level keys directly on the link entry (legacy):
            {class: "...", source: "...", target: "...", auto_transfer: true}
        2. Nested under config: (canonical):
            {class: "...", config: {source: "...", auto_transfer: true}}
    Both shapes are inspected.
    """
    if not isinstance(link_entry, dict):
        return None  # already-resolved LinkBase or unknown shape

    inner_config = link_entry.get("config")
    if isinstance(inner_config, dict):
        return "auto_transfer" not in inner_config
    if isinstance(inner_config, str):
        return None  # path-reference; can't tell without loading the file

    # Legacy shape — top-level keys ARE the config
    return "auto_transfer" not in link_entry


def _warn_on_implicit_auto_transfer(
    workflow_name: str,
    links_config: Dict[str, Any],
    config_version: int,
) -> None:
    """Emit one WARNING per link that omits `auto_transfer`. v2 workflows
    suppress the warning because the v2 default (G7 Step 3, shipped
    2026-05-09) flips auto_transfer to True for inline link configs at
    WorkflowConfig validation time — see ``_apply_v2_link_defaults``.

    Path-reference link configs cannot be statically inspected; we emit a
    softer DEBUG-level note for those so authors who care can find them.
    Path-reference configs are NOT mutated by the v2 default — that is
    Step 4 scope (workspace-wide default flip + external YAML rewriting).
    """
    if config_version >= 2:
        return  # v2 default already flipped auto_transfer for inline configs

    if not isinstance(links_config, dict) or not links_config:
        return

    implicit_links: List[str] = []
    indeterminate_links: List[str] = []

    for link_name, link_entry in links_config.items():
        if not isinstance(link_entry, dict):
            continue  # already-resolved LinkBase; skip
        class_path = link_entry.get("class", "")
        if not _link_class_needs_auto_transfer_check(class_path):
            continue  # AcademyLink (auto_transfer is set explicitly there) or unknown

        omits = _link_inline_config_omits_auto_transfer(link_entry)
        if omits is True:
            implicit_links.append(f"{link_name} ({class_path})")
        elif omits is None:
            indeterminate_links.append(f"{link_name} ({class_path})")

    if implicit_links:
        logger.warning(
            "Workflow %r: %d link(s) omit `auto_transfer` and will silently no-op "
            "on trigger fire (links: %s). Add `auto_transfer: true` explicitly "
            "to every DirectLink/TransformLink/ConditionalLink/FileLink/QueueLink "
            "in your YAML, OR set `config_version: 2` on this workflow once "
            "G7 v2 semantics ship to opt into the new default. "
            "See nanobrain_capability_gaps.md G7 + architecture.md §13 brutal-truth #3.",
            workflow_name,
            len(implicit_links),
            ", ".join(implicit_links),
        )

    if indeterminate_links:
        logger.debug(
            "Workflow %r: %d link config(s) use path-reference form, so the "
            "auto_transfer field cannot be statically verified at workflow load. "
            "Audit each linked YAML to ensure `auto_transfer: true` is set "
            "(links: %s).",
            workflow_name,
            len(indeterminate_links),
            ", ".join(indeterminate_links),
        )


def derive_nested_namespace(
    *,
    parent_namespace: str,
    child_workflow_name: str,
    strategy: str = "scoped",
) -> str:
    """G31 — derive a nested workflow's namespace from its parent.

    Args:
        parent_namespace: The outer ``WorkflowRunContext.proxystore_namespace``
            value (e.g., ``"run_abc"``).
        child_workflow_name: The nested workflow's ``name`` field;
            used as the suffix component under ``scoped`` strategy.
        strategy: ``"scoped"`` (default) or ``"inherit"``.

    Returns:
        The derived namespace string for the nested workflow:
          * scoped: ``"<parent_namespace>.<child_workflow_name>"``
          * inherit: ``parent_namespace`` verbatim

    Raises:
        ValueError: on unknown strategy or empty child_workflow_name
            under ``scoped`` (would produce an ambiguous namespace).

    P4+a decision: ``scoped`` is the default because silent-namespace-
    collision is a worse failure than over-isolation.
    """
    if strategy not in ("scoped", "inherit"):
        raise ValueError(
            f"FAIL-FAST: derive_nested_namespace strategy must be "
            f"'scoped' or 'inherit'; got {strategy!r}"
        )
    if strategy == "inherit":
        return parent_namespace
    # scoped
    if not child_workflow_name:
        raise ValueError(
            "FAIL-FAST: derive_nested_namespace 'scoped' strategy "
            "requires a non-empty child_workflow_name; got empty."
        )
    if not parent_namespace:
        # No parent namespace to nest under — the child becomes the
        # full namespace verbatim. Equivalent to a top-level child
        # but preserves the workflow's name as the namespace.
        return child_workflow_name
    return f"{parent_namespace}.{child_workflow_name}"


class WorkflowConfig(StepConfig):
    """
    Enhanced Configuration for workflows extending StepConfig - INHERITS constructor prohibition.

    ✅ FRAMEWORK COMPLIANCE:
    - Supports class+config patterns for steps, links, and triggers
    - ConfigBase._resolve_nested_objects() automatically instantiates components
    - Complete validation through ConfigBase schemas
    - Pure configuration-driven workflow creation

    ❌ FORBIDDEN: WorkflowConfig(name="test", steps=...)
    ✅ REQUIRED: WorkflowConfig.from_config('path/to/config.yml')
    """

    # G7 — config_version field. v1 (current) preserves the historical
    # default of auto_transfer=False on links. As of G7 Step 5
    # (2026-05-09), the LinkConfig.auto_transfer FIELD-LEVEL default is
    # True, so v1 and v2 BOTH get auto_transfer=True automatically;
    # explicit ``auto_transfer: false`` at the link level is the only
    # way to get the no-op behavior. The v2 mutator
    # (``_apply_v2_link_defaults``) is now redundant for the True case
    # but is kept as a safety net for path-reference rewriting and
    # future field-level changes. The deprecation WARNING for v1 is
    # also redundant under the new field default (Pydantic supplies
    # True for omitted keys before our warning runs); kept for
    # documentation purposes.
    config_version: Literal[1, 2] = Field(
        default=2,
        description="Workflow config schema version. As of G7 Step 5, "
                    "BOTH v1 and v2 produce auto_transfer=True by default "
                    "(the field-level default was flipped). v2 also "
                    "rewrites path-reference link configs in-memory and "
                    "propagates workflow-level defaults (gate_semantics, "
                    "etc.) into nested links/triggers. Declare "
                    "auto_transfer: false on a link explicitly when you "
                    "want a no-op link."
    )

    # G31 — namespace strategy for nested workflows (workflow-as-substep).
    # When this Workflow is loaded as a CHILD step of an outer workflow,
    # the strategy controls how its data-units / proxystore namespace
    # is derived from the outer run context. Two strategies:
    #
    #   ``scoped`` (default, recommended): the nested workflow gets its
    #     own namespace under the parent's (e.g., parent="run_abc",
    #     child workflow named "subflow_x" -> child namespace
    #     "run_abc.subflow_x"). Multi-tenant isolation is preserved
    #     ACROSS levels, not just at the top.
    #
    #   ``inherit``: the nested workflow shares the parent's namespace
    #     verbatim. Use when the nested workflow is logically part of
    #     the parent's tenant scope and namespace collisions are
    #     impossible by construction.
    #
    # Top-level workflows (no parent) ignore this field — the run
    # context's own namespace_template is used directly. The field
    # only takes effect when the workflow is nested.
    #
    # P4+a decision (eval_03 §8.9): scoped is the default because
    # silent-namespace-collision is a worse failure than over-isolation.
    namespace_strategy: Literal["scoped", "inherit"] = Field(
        default="scoped",
        description="G31 — how this workflow's namespace derives from "
                    "an outer parent run context when loaded as a "
                    "nested step. 'scoped' (default) appends the "
                    "workflow name to the parent namespace; 'inherit' "
                    "shares the parent namespace verbatim. Top-level "
                    "(non-nested) workflows ignore this field.",
    )

    # G10 Step 2 — workflow-level gate_semantics that propagates to every
    # inline ConditionalLink AND every inline AllDataReceivedTrigger that
    # omits its own per-link/per-trigger setting. ``setdefault`` semantics:
    # explicit per-component values are NEVER overridden. When unset
    # (the default), the framework leaves each component on its own
    # default (``publish_empty`` for legacy compat).
    #
    # See ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G10``.
    gate_semantics: Optional[Literal["publish_empty", "gate_to_bottom"]] = Field(
        default=None,
        description="Workflow-level default gate_semantics. When set, this "
                    "value is injected into every inline ConditionalLink "
                    "and AllDataReceivedTrigger that omits its own "
                    "gate_semantics field. Explicit per-component values "
                    "always win. Recommended: set to 'gate_to_bottom' on "
                    "any workflow that uses gating to avoid the "
                    "publish-empty deadlock failure shape."
    )

    # Enhanced workflow configuration supporting class+config patterns
    steps: Dict[str, Any] = Field(
        default_factory=dict,
        description="Step definitions with class+config patterns or step_id configurations"
    )
    links: Dict[str, Any] = Field(
        default_factory=dict,
        description="Link definitions with class+config patterns"
    )
    agents: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent definitions with class+config patterns"
    )
    # ✅ UNIFIED RESOLUTION: Inherit list-based triggers from StepConfig (workflows ARE steps)
    # triggers: List[Union[Dict[str, Any], 'TriggerBase']] inherited from StepConfig

    # Data-driven workflows don't need execution strategies
    # Steps execute automatically via triggers when data is available

    enable_monitoring: bool = True
    workflow_directory: Optional[str] = None

    # Execution configuration
    max_parallel_steps: int = 10
    step_timeout: float = 300.0  # 5 minutes
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # Validation configuration
    validate_graph: bool = True
    allow_cycles: bool = False
    require_connected_graph: bool = True

    # Progress reporting configuration
    enable_progress_reporting: bool = True

    # Executor configuration for workflow-level executor
    executor_config: Optional[str] = Field(
        default=None,
        description="Path to executor configuration file for workflow-level executor"
    )
    progress_batch_interval: float = 3.0
    progress_collapsed_by_default: bool = True
    progress_show_technical_errors: bool = True
    progress_preserve_session_history: bool = True

    # Resolved components storage (populated by ConfigBase)
    resolved_agents: Dict[str, Any] = Field(
        default_factory=dict,
        description="Instantiated agent objects from configuration"
    )

    # G7 Step 3 — when config_version >= 2, inject auto_transfer=True into
    # every inline link config that omits the field. Path-reference link
    # configs (link_entry.config is a string path to an external YAML)
    # are NOT mutated — that requires loading the external YAML, which is
    # out-of-scope for Step 3 and deferred to Step 4 along with the
    # workspace-wide default flip.
    @model_validator(mode='after')
    def _apply_v2_link_defaults(self) -> 'WorkflowConfig':
        """Mutate self.links inline-config dicts to set auto_transfer=True
        when config_version >= 2 and the field is absent.

        Mutation is in-place on the same dict objects that LinkBase.from_config
        will subsequently consume — see workflow_graph.add_link / loader code.
        Explicit values (True or False) are NEVER overridden; only absent keys.

        G7 Step 4 — path-reference link configs (``config: "external.yml"``)
        are also rewritten in v2: the external YAML is loaded, the
        field is injected, and the entry is rewritten to nested-inline
        form. Path resolution honors ``self.workflow_directory`` when
        set; otherwise CWD-relative.
        """
        if self.config_version < 2:
            return self
        if not self.links or not isinstance(self.links, dict):
            return self

        for link_name, link_entry in self.links.items():
            if not isinstance(link_entry, dict):
                continue
            class_path = link_entry.get('class', '')
            if not _link_class_needs_auto_transfer_check(class_path):
                continue
            _set_inline_default_in_entry(
                link_entry, 'auto_transfer', True,
                rewrite_path_reference=True,
                workflow_directory=self.workflow_directory,
            )

        return self

    # G10 Step 2 — propagate workflow-level gate_semantics into every
    # inline ConditionalLink and AllDataReceivedTrigger that omits its
    # own per-component setting. Path-reference configs are NOT mutated
    # (parity with G7 Step 3; loading external YAML is Step 3 scope of
    # this gap and tracked separately).
    @model_validator(mode='after')
    def _propagate_gate_semantics(self) -> 'WorkflowConfig':
        """Inject workflow.gate_semantics into child links/triggers.

        Walks:
          - self.links[<name>]                 — every link entry
          - self.steps[<step_name>].triggers[] — list-shape triggers under
                                                 each step (inline configs)
          - self.triggers[]                    — workflow-level triggers
                                                 (inherited from StepConfig)

        For each entry whose class is in the gate-semantics-aware
        whitelist, ``setdefault('gate_semantics', <workflow-default>)`` is
        called. Explicit per-component values are preserved.

        When ``self.gate_semantics`` is None, this validator is a no-op.
        """
        if self.gate_semantics is None:
            return self

        gate_value = self.gate_semantics
        rewrite_path_ref = self.config_version >= 2  # G7 Step 4

        # 1. Links
        if isinstance(self.links, dict):
            for link_entry in self.links.values():
                if not isinstance(link_entry, dict):
                    continue
                class_path = link_entry.get('class', '')
                if not _link_class_needs_gate_semantics_check(class_path):
                    continue
                _set_inline_default_in_entry(
                    link_entry, 'gate_semantics', gate_value,
                    rewrite_path_reference=rewrite_path_ref,
                    workflow_directory=self.workflow_directory,
                )

        # 2. Per-step triggers
        if isinstance(self.steps, dict):
            for step_entry in self.steps.values():
                self._propagate_gate_semantics_into_step(
                    step_entry, gate_value, rewrite_path_ref,
                )

        # 3. Workflow-level triggers (inherited from StepConfig)
        own_triggers = getattr(self, 'triggers', None)
        if isinstance(own_triggers, list):
            for trig_entry in own_triggers:
                self._maybe_set_trigger_gate_semantics(
                    trig_entry, gate_value, rewrite_path_ref,
                    self.workflow_directory,
                )

        return self

    def _propagate_gate_semantics_into_step(
        self, step_entry: Any, gate_value: str, rewrite_path_ref: bool,
    ) -> None:
        """Helper for ``_propagate_gate_semantics``: inspect a single
        step entry, find its inline triggers, and stamp gate_semantics
        on the gate-aware ones."""
        if not isinstance(step_entry, dict):
            return  # already-resolved BaseStep instance; skip

        # Triggers can live at the top level of the step entry OR nested
        # under a 'config' dict (mirrors the link two-shape pattern).
        for container in (step_entry, step_entry.get('config') or {}):
            if not isinstance(container, dict):
                continue
            triggers = container.get('triggers')
            if not isinstance(triggers, list):
                continue
            for trig_entry in triggers:
                self._maybe_set_trigger_gate_semantics(
                    trig_entry, gate_value, rewrite_path_ref,
                    self.workflow_directory,
                )

    @staticmethod
    def _maybe_set_trigger_gate_semantics(
        trig_entry: Any, gate_value: str,
        rewrite_path_ref: bool = False,
        workflow_directory: Optional[str] = None,
    ) -> None:
        """Stamp gate_semantics on a trigger config dict if its class is
        gate-aware AND the field is absent. No-op for resolved instances
        or non-gate-aware trigger classes. Path-reference configs are
        loaded + rewritten when ``rewrite_path_ref=True`` (G7 Step 4)."""
        if not isinstance(trig_entry, dict):
            return
        class_path = trig_entry.get('class', '')
        if not _trigger_class_needs_gate_semantics_check(class_path):
            return
        _set_inline_default_in_entry(
            trig_entry, 'gate_semantics', gate_value,
            rewrite_path_reference=rewrite_path_ref,
            workflow_directory=workflow_directory,
        )


# WorkflowGraph imported from workflow_graph.py




    def remove_step(self, step_id: str) -> None:
        """Remove a step and all its connections from the graph."""
        if step_id not in self.nodes:
            raise ValueError(f"Step {step_id} not found in workflow graph")

        # Remove all edges involving this step
        edges_to_remove = []
        for link_id, link_info in self.edges.items():
            if link_info['source_id'] == step_id or link_info['target_id'] == step_id:
                edges_to_remove.append(link_id)

        for link_id in edges_to_remove:
            self.remove_link(link_id)

        # Remove from adjacency lists
        for connected_id in self.adjacency[step_id]:
            self.reverse_adjacency[connected_id].discard(step_id)

        for predecessor_id in self.reverse_adjacency[step_id]:
            self.adjacency[predecessor_id].discard(step_id)

        # Remove the step
        del self.nodes[step_id]
        del self.adjacency[step_id]
        del self.reverse_adjacency[step_id]

        self._invalidate_cache()
        self.logger.debug(f"Removed step from workflow graph: {step_id}")

    def remove_link(self, link_id: str) -> None:
        """Remove a link from the graph."""
        if link_id not in self.edges:
            raise ValueError(f"Link {link_id} not found in workflow graph")

        link_info = self.edges[link_id]

        # Get source and target step IDs from stored info
        source_id = link_info['source_id']
        target_id = link_info['target_id']

        if source_id and target_id:
            self.adjacency[source_id].discard(target_id)
            self.reverse_adjacency[target_id].discard(source_id)

        del self.edges[link_id]
        self._invalidate_cache()
        self.logger.debug(f"Removed link from workflow graph: {link_id}")

    def get_step(self, step_id: str) -> Optional[BaseStep]:
        """Get a step by ID."""
        return self.nodes.get(step_id)

    def get_link(self, link_id: str) -> Optional[LinkBase]:
        """Get a link by ID."""
        link_info = self.edges.get(link_id)
        return link_info['link'] if link_info else None

    def get_step_dependencies(self, step_id: str) -> Set[str]:
        """Get all steps that must execute before the given step."""
        if step_id not in self.nodes:
            raise ValueError(f"Step {step_id} not found in workflow graph")
        return self.reverse_adjacency[step_id].copy()

    def get_step_dependents(self, step_id: str) -> Set[str]:
        """Get all steps that depend on the given step."""
        if step_id not in self.nodes:
            raise ValueError(f"Step {step_id} not found in workflow graph")
        return self.adjacency[step_id].copy()

    def has_cycles(self) -> bool:
        """Check if the graph contains cycles using DFS."""
        color = {step_id: 0 for step_id in self.nodes}  # 0: white, 1: gray, 2: black

        def dfs(step_id: str) -> bool:
            if color[step_id] == 1:  # Back edge found - cycle detected
                return True
            if color[step_id] == 2:  # Already processed
                return False

            color[step_id] = 1  # Mark as being processed

            for neighbor in self.adjacency[step_id]:
                if dfs(neighbor):
                    return True

            color[step_id] = 2  # Mark as completely processed
            return False

        for step_id in self.nodes:
            if color[step_id] == 0:
                if dfs(step_id):
                    return True

        return False

    def get_execution_order(self) -> List[str]:
        """Get topological execution order using Kahn's algorithm."""
        if self._execution_order is not None:
            return self._execution_order.copy()

        # Kahn's algorithm for topological sorting
        in_degree = {step_id: len(
            self.reverse_adjacency[step_id]) for step_id in self.nodes}
        queue = [step_id for step_id, degree in in_degree.items()
                 if degree == 0]
        execution_order = []

        while queue:
            current = queue.pop(0)
            execution_order.append(current)

            for neighbor in self.adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(execution_order) != len(self.nodes):
            raise ValueError(
                "Workflow graph contains cycles - cannot determine execution order")

        self._execution_order = execution_order
        return execution_order.copy()

    def get_parallel_execution_levels(self) -> List[List[str]]:
        """Get steps grouped by execution level for parallel execution."""
        execution_order = self.get_execution_order()
        levels = []
        processed = set()

        while processed != set(self.nodes.keys()):
            current_level = []

            for step_id in execution_order:
                if step_id in processed:
                    continue

                # Check if all dependencies are satisfied
                dependencies = self.get_step_dependencies(step_id)
                if dependencies.issubset(processed):
                    current_level.append(step_id)

            if not current_level:
                raise ValueError(
                    "Cannot determine parallel execution levels - possible circular dependency")

            levels.append(current_level)
            processed.update(current_level)

        return levels

    def validate_graph(self, allow_cycles: bool = False, require_connected: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate the workflow graph structure.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        warnings = []

        # Check for empty graph
        if not self.nodes:
            errors.append("Workflow graph is empty - no steps defined")

        # ✅ USER GUIDANCE: Check for cycles but only warn if cycles are not allowed
        if self.has_cycles():
            if not allow_cycles:
                cycle_warning = (
                    "⚠️  WORKFLOW CYCLES DETECTED: This workflow contains cycles. "
                    "Ensure that appropriate resolution mechanisms are in place:\n"
                    "   • Data convergence logic to prevent infinite loops\n"
                    "   • Conditional triggers to break cycles when appropriate\n"
                    "   • Timeout mechanisms for long-running cycles\n"
                    "   • Clear termination conditions\n"
                    f"   Steps involved in cycles: {self._get_cycles_info()}"
                )
                warnings.append(cycle_warning)
                self.logger.warning(cycle_warning)

                # ✅ FRAMEWORK COMPLIANCE: Provide guidance on how to allow cycles
                self.logger.info(
                    "💡 To suppress cycle warnings, set 'allow_cycles: true' in workflow configuration "
                    "if you have confirmed appropriate resolution mechanisms are in place."
                )
            else:
                # Cycles are allowed - just log at debug level for troubleshooting
                self.logger.debug(
                    f"🔄 Workflow cycles detected but allowed by configuration. "
                    f"Steps involved: {self._get_cycles_info()}"
                )

        # Check for disconnected components if required
        if require_connected and len(self.nodes) > 1:
            if not self._is_weakly_connected():
                errors.append(
                    "Workflow graph is not connected - contains isolated components")

        # Check for orphaned steps (no inputs or outputs)
        orphaned_steps = []
        for step_id in self.nodes:
            has_input = len(self.reverse_adjacency[step_id]) > 0
            has_output = len(self.adjacency[step_id]) > 0

            if not has_input and not has_output and len(self.nodes) > 1:
                orphaned_steps.append(step_id)

        if orphaned_steps:
            errors.append(
                f"Orphaned steps found (no connections): {orphaned_steps}")

        # Validate that all links have valid source and target steps
        for link_id, link_info in self.edges.items():
            source_id = link_info['source_id']
            target_id = link_info['target_id']

            if source_id not in self.nodes:
                errors.append(
                    f"Link {link_id} has invalid source step: {source_id}")

            if target_id not in self.nodes:
                errors.append(
                    f"Link {link_id} has invalid target step: {target_id}")

            # ✅ CRITICAL: Check for self-referencing links (illegal in workflow architecture)
            if source_id == target_id:
                errors.append(
                    f"❌ ILLEGAL SELF-REFERENCING LINK: {link_id} connects step '{source_id}' to itself. "
                    f"Self-referencing links are prohibited in the workflow architecture as they create "
                    f"infinite trigger loops and prevent proper workflow execution.")

        # ✅ CYCLES ARE NOT ERRORS: Only fail on true structural problems
        is_valid = len(errors) == 0
        self._is_valid = is_valid

        # Log warnings separately
        if warnings:
            for warning in warnings:
                self.logger.warning(warning)

        return is_valid, errors

    def _get_cycles_info(self) -> str:
        """Get information about cycles in the graph for user guidance."""
        # Simple cycle detection for informational purposes
        try:
            strongly_connected = self._find_strongly_connected_components()
            cycle_components = [
                comp for comp in strongly_connected if len(comp) > 1]
            if cycle_components:
                return f"Strongly connected components: {cycle_components}"
            else:
                return "Self-referencing steps detected"
        except:
            return "Multiple interconnected steps"

    def _find_strongly_connected_components(self) -> List[List[str]]:
        """Find strongly connected components using Tarjan's algorithm."""
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        result = []

        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack[node] = True

            for neighbor in self.adjacency[node]:
                if neighbor not in index:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif on_stack[neighbor]:
                    lowlinks[node] = min(lowlinks[node], index[neighbor])

            if lowlinks[node] == index[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == node:
                        break
                result.append(component)

        for node in self.nodes:
            if node not in index:
                strongconnect(node)

        return result

    def _is_weakly_connected(self) -> bool:
        """Check if the graph is weakly connected (ignoring edge direction)."""
        if not self.nodes:
            return True

        visited = set()
        start_node = next(iter(self.nodes))
        stack = [start_node]

        while stack:
            current = stack.pop()
            if current in visited:
                continue

            visited.add(current)

            # Add both successors and predecessors (treat as undirected)
            stack.extend(self.adjacency[current] - visited)
            stack.extend(self.reverse_adjacency[current] - visited)

        return len(visited) == len(self.nodes)

    def _invalidate_cache(self) -> None:
        """Invalidate cached computations when graph structure changes."""
        self._execution_order = None
        self._strongly_connected_components = None
        self._is_valid = False

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "num_steps": len(self.nodes),
            "num_links": len(self.edges),
            "has_cycles": self.has_cycles(),
            "is_connected": self._is_weakly_connected(),
            "max_depth": self._calculate_max_depth(),
            "avg_branching_factor": self._calculate_avg_branching_factor()
        }

    def _calculate_max_depth(self) -> int:
        """Calculate maximum depth of the graph."""
        if not self.nodes:
            return 0

        try:
            levels = self.get_parallel_execution_levels()
            return len(levels)
        except ValueError:
            # Graph has cycles, return -1
            return -1

    def _calculate_avg_branching_factor(self) -> float:
        """Calculate average branching factor."""
        if not self.nodes:
            return 0.0

        total_edges = sum(len(neighbors)
                          for neighbors in self.adjacency.values())
        return total_edges / len(self.nodes)


class Workflow(Step):
    """
    Base Workflow Class - Multi-Step Orchestration with Event-Driven Execution
    =========================================================================

    The Workflow class is the primary orchestration component for creating complex,
    multi-step data processing pipelines within the NanoBrain framework. Workflows
    compose multiple steps, agents, and tools into sophisticated, event-driven
    processing systems with advanced error handling, monitoring, and execution strategies.

    **Core Architecture:**
        Workflows represent intelligent orchestration systems that:

        * **Orchestrate Components**: Coordinate multiple steps, agents, and tools
        * **Manage Data Flow**: Control data movement through configurable links
        * **Execute Strategies**: Support sequential, parallel, graph-based, and event-driven execution
        * **Handle Errors**: Comprehensive error handling with retry, rollback, and recovery
        * **Monitor Progress**: Real-time progress tracking with checkpoints and resumption
        * **Scale Execution**: Support local, distributed, and high-performance computing environments

    **Biological Analogy:**
        Like neural circuit complexes that contain multiple interconnected circuits working
        together in coordination, workflows are composed of steps working together through
        defined connections and data flow patterns. Neural circuits coordinate sensory input,
        processing, decision making, and motor output through sophisticated signaling networks -
        exactly how workflows coordinate data ingestion, processing, analysis, and output
        through configurable step networks and event-driven triggers.

    **Workflow Orchestration Architecture:**

        **Multi-Step Composition:**
        * Hierarchical step organization with nested workflows
        * Dynamic step creation and configuration from YAML
        * Step dependency tracking and resolution
        * Conditional step execution based on data and results

        **Data Flow Management:**
        * Configurable links for data transfer between steps
        * Multiple link types (direct, transform, conditional, queue)
        * Data validation and type checking across step boundaries
        * Streaming data support for real-time processing

        **Execution Strategies:**
        * **Sequential**: Steps execute in defined order
        * **Parallel**: Independent steps execute concurrently
        * **Graph-Based**: Dependency-aware execution with optimization
        * **Event-Driven**: Steps triggered by data availability and conditions

        **Error Handling and Recovery:**
        * Comprehensive error detection and classification
        * Retry mechanisms with exponential backoff
        * Rollback capabilities for data consistency
        * Graceful degradation and alternative path execution

    **Framework Integration:**
        Workflows seamlessly integrate with all framework components:

        * **Agent Integration**: Embed AI agents for intelligent processing
        * **Tool Orchestration**: Coordinate multiple tools across processing stages
        * **Executor Support**: Run on local, threaded, process, and distributed backends
        * **Monitoring Integration**: Comprehensive logging, metrics, and progress tracking
        * **Configuration Management**: Complete YAML-driven workflow definition
        * **Event System**: Integration with trigger system for event-driven execution

    **Execution Strategy Types:**
        The framework supports various execution strategies:

        * **Sequential Execution**: Traditional step-by-step processing
            - Predictable execution order
            - Resource-efficient for linear workflows
            - Simple error handling and debugging

        * **Parallel Execution**: Concurrent processing of independent steps
            - Maximum throughput for parallelizable workloads
            - Resource optimization through load balancing
            - Reduced overall execution time

        * **Graph-Based Execution**: Dependency-aware optimization
            - Automatic execution order determination
            - Optimal resource allocation
            - Dynamic parallelization based on dependencies

        * **Event-Driven Execution**: Reactive processing model
            - Real-time response to data availability
            - Efficient resource utilization
            - Complex conditional execution patterns

    **Configuration Architecture:**
        Workflows follow the framework's configuration-first design:

        ```yaml
        # Basic workflow configuration
        name: "data_processing_workflow"
        description: "Multi-stage data processing with AI analysis"
        # Data-driven execution via triggers
        error_handling: "retry"

        # Step definitions with class+config patterns
        steps:
          data_ingestion:
            class: "nanobrain.library.steps.DataIngestionStep"
            config:
              source_type: "file"
              file_path: "data/input.json"
              validation_schema: "schemas/input.json"

          ai_analysis:
            class: "nanobrain.library.steps.AgentStep"
            config:
              agent:
                class: "nanobrain.core.agent.ConversationalAgent"
                config: "config/analysis_agent.yml"
              processing_prompt: "Analyze the provided data for patterns"

          result_storage:
            class: "nanobrain.library.steps.DataOutputStep"
            config:
              output_format: "json"
              destination: "results/analysis.json"

        # Data flow links
        links:
          data_to_analysis:
            class: "nanobrain.core.link.DirectLink"
            config:
              source: "data_ingestion.output_data"
              target: "ai_analysis.input_data"

          analysis_to_storage:
            class: "nanobrain.core.link.TransformLink"
            config:
              source: "ai_analysis.results"
              target: "result_storage.input_data"
              transform_function: "format_analysis_results"

        # Event triggers
        triggers:
          - class: "nanobrain.core.trigger.DataUnitChangeTrigger"
            config:
              watch_data_units: ["input_data"]
              step_targets: ["data_ingestion"]

        # Execution configuration
        executor:
          class: "nanobrain.core.executor.ParslExecutor"
          config: "config/hpc_executor.yml"

        # Monitoring and progress
        monitoring:
          enable_progress_tracking: true
          checkpoint_interval: 30
          metrics_collection: true
          real_time_updates: true
        ```

    **Usage Patterns:**

        **Basic Workflow Execution:**
        ```python
        from nanobrain.core import Workflow

        # Create workflow from configuration
        workflow = Workflow.from_config('config/data_workflow.yml')

        # Execute workflow
        results = await workflow.execute()
        print(f"Workflow completed: {results}")

        # Access step results
        for step_name, result in results.items():
            print(f"Step {step_name}: {result}")
        ```

        **Event-Driven Workflow:**
        ```python
        # Event-driven workflow responds to data changes
        workflow = Workflow.from_config('config/realtime_workflow.yml')

        # Start workflow in monitoring mode
        await workflow.start_monitoring()

        # Workflow automatically processes new data as it arrives
        # Steps are triggered by data availability events
        ```

        **Distributed Workflow Execution:**
        ```python
        # High-performance distributed execution
        workflow = Workflow.from_config('config/hpc_workflow.yml')

        # Execute on distributed cluster
        with workflow.distributed_context():
            results = await workflow.execute()

        # Results automatically collected from all compute nodes
        ```

        **Nested Workflow Composition:**
        ```python
        # Workflows can contain other workflows
        main_workflow = Workflow.from_config('config/main_workflow.yml')

        # Sub-workflows execute as steps within main workflow
        # Full isolation and independent configuration
        results = await main_workflow.execute()
        ```

    **Advanced Features:**

        **Progress Tracking and Monitoring:**
        * Real-time progress updates with percentage completion
        * Step-by-step status tracking and timing information
        * Checkpoint creation for resumable execution
        * Performance metrics and optimization recommendations

        **Error Handling and Recovery:**
        * Automatic retry with configurable strategies
        * Rollback mechanisms for data consistency
        * Alternative execution paths for failure scenarios
        * Comprehensive error logging and diagnostic information

        **Dynamic Configuration:**
        * Runtime parameter updates and reconfiguration
        * Conditional step execution based on results
        * Dynamic workflow modification and extension
        * Template-based workflow generation

        **Performance Optimization:**
        * Automatic parallelization of independent steps
        * Resource allocation and load balancing
        * Caching of intermediate results
        * Memory management and cleanup

    **Execution Lifecycle:**
        Workflows follow a well-defined execution lifecycle:

        1. **Configuration Loading**: Parse and validate workflow configuration
        2. **Component Resolution**: Create steps, links, triggers, and executors
        3. **Dependency Analysis**: Build execution graph and determine order
        4. **Resource Allocation**: Setup execution backends and resource pools
        5. **Trigger Registration**: Setup event listeners and activation conditions
        6. **Execution Initialization**: Prepare all components for execution
        7. **Step Orchestration**: Execute steps according to strategy
        8. **Progress Monitoring**: Track progress and handle events
        9. **Result Collection**: Gather results and update data units
        10. **Cleanup and Finalization**: Release resources and persist state

    **Integration Patterns:**

        **Agent-Driven Workflows:**
        * Embed AI agents for intelligent decision making
        * Multi-agent collaboration within workflow steps
        * Agent-to-agent communication and coordination
        * Dynamic workflow adaptation based on agent insights

        **Tool-Intensive Workflows:**
        * Coordinate multiple specialized tools
        * Tool chaining and result passing
        * Parallel tool execution for performance
        * Tool failure handling and alternatives

        **Data-Centric Workflows:**
        * Large dataset processing with streaming
        * Data validation and quality assurance
        * Multi-format data transformation
        * Data lineage tracking and auditing

        **Real-Time Workflows:**
        * Event-driven processing for streaming data
        * Low-latency response to external events
        * Continuous monitoring and adaptation
        * Real-time analytics and alerting

    **Performance and Scalability:**

        **Execution Optimization:**
        * Automatic parallelization of independent operations
        * Resource pooling and reuse for efficiency
        * Intelligent scheduling and load balancing
        * Memory management and garbage collection

        **Scalability Features:**
        * Horizontal scaling across multiple compute nodes
        * Vertical scaling with resource allocation
        * Elastic scaling based on workload demands
        * Integration with cloud and HPC environments

        **Monitoring and Analytics:**
        * Real-time performance metrics and dashboards
        * Resource utilization tracking and optimization
        * Bottleneck identification and resolution recommendations
        * Historical performance analysis and trending

    **Error Handling and Reliability:**

        **Comprehensive Error Management:**
        * Exception handling with detailed diagnostics
        * Automatic retry mechanisms with intelligent backoff
        * Graceful degradation for partial failures
        * Alternative execution paths for resilience

        **Data Consistency:**
        * Transactional execution with rollback capabilities
        * Data validation at step boundaries
        * Conflict resolution for concurrent operations
        * Audit trails for debugging and compliance

        **Fault Tolerance:**
        * Checkpoint creation for resumable execution
        * State recovery after system failures
        * Redundancy and failover mechanisms
        * Health monitoring and automatic recovery

    **Development and Testing:**

        **Testing Support:**
        * Mock step implementations for testing
        * Workflow simulation and validation
        * Performance benchmarking and profiling
        * Unit and integration testing frameworks

        **Debugging Features:**
        * Step-by-step execution tracing
        * Data flow visualization and inspection
        * Interactive debugging and breakpoints
        * Comprehensive logging with structured output

        **Development Tools:**
        * Workflow validation and linting
        * Configuration templates and generators
        * Performance profiling and optimization tools
        * Visual workflow design and editing

    Attributes:
        name (str): Workflow identifier for logging and monitoring
        description (str): Human-readable workflow description and purpose
        steps (Dict[str, BaseStep]): Collection of workflow steps with identifiers
        links (List[LinkBase]): Data flow links connecting steps
        triggers (List[TriggerBase]): Event triggers for step activation
        # Data-driven workflows execute automatically via triggers
        executor (ExecutorBase): Execution backend for workflow operations
        progress (WorkflowProgress): Real-time progress tracking and status
        graph (WorkflowGraph): Execution graph with dependencies and optimization
        monitoring_enabled (bool): Whether comprehensive monitoring is active
        performance_metrics (Dict): Real-time performance and resource usage metrics

    Note:
        Workflows extend the Step class and can be used as steps within larger workflows,
        enabling hierarchical composition and modular design. All workflows must be
        created using the from_config pattern with proper configuration files following
        the framework's event-driven architecture patterns.

    Warning:
        Workflows may consume significant computational resources depending on complexity,
        execution strategy, and the number of steps. Monitor resource usage and implement
        appropriate limits, timeouts, and cleanup mechanisms. Ensure proper error handling
        for long-running or distributed workflows.

    See Also:
        * :class:`Step`: Base step class that workflows extend
        * :class:`WorkflowConfig`: Workflow configuration schema and validation
        * :class:`WorkflowGraph`: Execution graph management and optimization
        * Data-driven execution via triggers and links
        * :class:`LinkBase`: Data flow connection management
        * :class:`TriggerBase`: Event trigger system for workflow activation
        * :mod:`nanobrain.library.workflows`: Specialized workflow implementations
    """

    COMPONENT_TYPE = "workflow"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': '',
        'steps': [],
        'links': [],

        'error_handling': 'stop',  # FAIL-FAST: Changed from 'continue' to 'stop'
        'enable_monitoring': True,
        'auto_initialize': True,
        'debug_mode': False
    }

    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return WorkflowConfig - ONLY method that differs from other components"""
        return WorkflowConfig

    @classmethod
    def from_config(cls, config_path: Union[str, Path], **context) -> 'Workflow':
        """
        Enhanced workflow loading with automatic component instantiation

        ✅ FRAMEWORK COMPLIANCE:
        - Leverages ConfigBase._resolve_nested_objects() for automatic component instantiation
        - Steps, links, and triggers created via class+config patterns
        - No manual factory functions or redundant creation logic
        - Complete validation through ConfigBase schemas

        Args:
            config_path: Path to workflow configuration file
            **context: Additional context

        Returns:
            Fully initialized workflow instance

        Example Configuration:
        ```yaml
        name: "enhanced_workflow"
        description: "Workflow with automatic component instantiation"

        # Steps created via class+config patterns
        steps:
          data_acquisition:
            class: "nanobrain.library.steps.bv_brc_data_acquisition_step.BVBRCDataAcquisitionStep"
            config: "config/steps/BVBRCDataAcquisitionStep.yml"

          analysis:
            class: "nanobrain.library.steps.analysis_step.AnalysisStep"
            config:
              name: "protein_analysis"
              analysis_type: "protein_structure"

        # Links created via class+config patterns
        links:
          data_flow:
            class: "nanobrain.core.link.DirectLink"
            config: "config/links/DataFlowLink.yml"

        # Triggers created via class+config patterns
        triggers:
          data_updated:
            class: "nanobrain.core.trigger.DataUnitChangeTrigger"
            config:
              data_unit_name: "protein_data"
              threshold: 10
        ```

        ✅ FRAMEWORK COMPLIANCE:
        - ConfigBase._resolve_nested_objects() automatically instantiates all components
        - Components validated through their respective ConfigBase schemas
        - No manual component creation or factory dependencies
        - Complete configuration-driven workflow creation
        """

        # ACADEMY INTEGRATION: Detect and setup Academy manager if needed
        if cls._requires_academy_integration(config_path):
            # Setup the SINGLETON Academy manager (but don't store it in context!)
            cls._setup_academy_manager()
            # Only store a flag indicating Academy integration is enabled
            context['_academy_integration_enabled'] = True

        # Use enhanced WorkflowConfig.from_config() method - automatically resolves class+config patterns
        workflow_config = WorkflowConfig.from_config(config_path, **context)

        # ConfigBase._resolve_nested_objects() has already instantiated all components
        # Extract resolved components from the configuration
        resolved_components = cls._extract_resolved_components(workflow_config)

        # Create workflow instance from resolved configuration
        workflow = cls._create_from_resolved_config(
            workflow_config, resolved_components, **context)

        # CRITICAL: Store config path on workflow instance for ParslExecutor integration
        # This enables steps to access their parent workflow's config path for sub-workflow creation
        if hasattr(workflow_config, 'source_path'):
            workflow._config_path = workflow_config.source_path
        else:
            # Fallback - use the original config_path parameter
            workflow._config_path = str(config_path)

        return workflow

    @classmethod
    def from_skeleton(
        cls,
        skeleton: Union[str, Path, "Skeleton", Dict[str, Any]],
        bindings: Optional[Dict[str, Any]] = None,
    ) -> "Workflow":
        """G9-completion (2026-05-09) — ergonomic skeleton-based workflow loader.

        Pre-G9-completion the framework shipped a Skeleton + SkeletonRegistry
        primitive (G9 v1) but the *ergonomic* loader the gap proposal named
        as G9's whole point ("let an agent pick a skeleton and bind holes
        without authoring a full YAML") was deferred. Agents authoring
        workflows had to hand-assemble PlanLoweringStep + SkeletonLoaderStep
        YAML to get from skeleton to runnable workflow — defeating the
        skeleton ergonomic.

        ``Workflow.from_skeleton(skeleton, bindings)`` collapses that dance:
        one call, one Workflow instance.

        Args:
            skeleton: Skeleton input. Accepts:
              * ``str`` / ``Path`` — file path to a Skeleton YAML config
              * ``Skeleton`` — pre-loaded Skeleton instance
              * ``dict`` — inline Skeleton config (programmatic / test path)
            bindings: Mapping of hole_name → value. Required holes that
                are not in this dict (and have no default) FAIL-FAST.
                Extra keys not in skeleton.holes FAIL-FAST.

        Returns:
            Fully-initialized ``Workflow`` instance, identical to what
            ``Workflow.from_config(<lowered_yaml_path>)`` would produce.

        Failure modes (all FAIL-FAST):
            - skeleton fails to load (wrong path / invalid YAML / schema)
            - bindings missing a required hole
            - bindings include a name not declared in skeleton.holes
            - lowered YAML fails ``Workflow.from_config`` validation

        Cross-reference:
            - ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
              Round 2 G9-completion
            - ``apecx-mcp-integration/docs/development_roadmap.md`` 8.7
            - ``nanobrain/library/orchestration/skeleton.py`` (Skeleton primitive)
            - ``nanobrain/library/orchestration/plan_lowering_step.py``
              (the multi-step path that ``from_skeleton`` collapses)

        Example::

            wf = Workflow.from_skeleton(
                'configs/skeletons/multi_source_discovery.yml',
                bindings={'min_evidence': 3, 'corpus': 'pubmed_2025_q1'},
            )
            result = await wf.run({'query': 'EEEV vaccines'})
        """
        import json
        import re
        import tempfile
        from pathlib import Path as _Path

        from nanobrain.library.orchestration.skeleton import (
            Skeleton,
            SkeletonHole,
        )
        from nanobrain.core.component_base import ComponentConfigurationError

        # 1. Resolve the input to a Skeleton instance.
        if isinstance(skeleton, Skeleton):
            sk = skeleton
        elif isinstance(skeleton, (str, _Path)):
            # Read the file and route through the dict path of
            # Skeleton.from_config — that path pre-builds SkeletonHole
            # nested instances via the framework backdoor. The file
            # branch of Skeleton.from_config delegates to
            # super().from_config which trips Pydantic's
            # FromConfigBase guard on nested SkeletonHole construction.
            import yaml as _yaml

            sk_dict = _yaml.safe_load(_Path(skeleton).read_text())
            if not isinstance(sk_dict, dict):
                raise ComponentConfigurationError(
                    f"FAIL-FAST: Workflow.from_skeleton skeleton file "
                    f"{skeleton!r} did not parse to a YAML mapping; got "
                    f"{type(sk_dict).__name__}"
                )
            sk = Skeleton.from_config(sk_dict)
        elif isinstance(skeleton, dict):
            sk = Skeleton.from_config(skeleton)
        else:
            raise ComponentConfigurationError(
                f"FAIL-FAST: Workflow.from_skeleton expected str / Path / "
                f"Skeleton / dict, got {type(skeleton).__name__}"
            )

        bindings = dict(bindings or {})

        # 2. Binding validation — Gate 3 of agent_workflow_authoring.md §6.
        declared = sk.holes
        provided = set(bindings.keys())
        declared_names = set(declared.keys())
        missing_required = sorted(
            name
            for name, hole in declared.items()
            if hole.required and name not in provided
        )
        if missing_required:
            raise ComponentConfigurationError(
                f"FAIL-FAST: Workflow.from_skeleton(skeleton_id="
                f"{sk.skeleton_id!r}) missing required holes "
                f"{missing_required}. Declared holes: "
                f"{sorted(declared_names)}; provided bindings: "
                f"{sorted(provided)}."
            )
        extras = sorted(provided - declared_names)
        if extras:
            raise ComponentConfigurationError(
                f"FAIL-FAST: Workflow.from_skeleton(skeleton_id="
                f"{sk.skeleton_id!r}) extra binding keys not declared "
                f"in skeleton.holes: {extras}. Declared holes: "
                f"{sorted(declared_names)}."
            )

        # 3. Build the binding summary (apply defaults for unprovided
        # optional holes).
        binding_summary: Dict[str, Any] = {}
        for name, hole in declared.items():
            if name in provided:
                binding_summary[name] = bindings[name]
            elif hole.default is not None:
                binding_summary[name] = hole.default
            elif not hole.required:
                binding_summary[name] = None
            # required+missing was already caught in step 2

        # 4. Hole substitution. Mirrors PlanLoweringStep._substitute_holes
        # behavior — same regex, same YAML-canonical scalar form.
        token_pattern = re.compile(
            r"\{\{\s*(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*"
            r"(?P<type>string|integer|number|boolean|array|object|any|"
            r"tool_descriptor_ref)"
            r"(?:\s*\|\s*default\s*=\s*[^}]+?)?\s*\}\}"
        )

        def _yaml_scalar(value: Any) -> str:
            if value is None:
                return "null"
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, (int, float)):
                return json.dumps(value)
            if isinstance(value, str):
                return json.dumps(value)
            return json.dumps(
                value, sort_keys=isinstance(value, dict), default=str
            )

        def _replace(match: "re.Match[str]") -> str:
            name = match.group("name")
            if name not in binding_summary:
                # Token in body but not in holes — skeleton.validate_against_schema
                # should have caught it; defensively FAIL-FAST.
                raise ComponentConfigurationError(
                    f"FAIL-FAST: Workflow.from_skeleton skeleton "
                    f"{sk.skeleton_id!r} body has token "
                    f"{{{{ {name}: <type> }}}} not declared in skeleton.holes; "
                    f"declared: {sorted(declared_names)}."
                )
            return _yaml_scalar(binding_summary[name])

        lowered_yaml = token_pattern.sub(_replace, sk.body)

        # 5. Materialize the lowered YAML to disk so the existing
        # path-based ``from_config`` resolver can do its job (path
        # resolution for nested ``config: "<file>.yml"`` references is
        # tied to the YAML file's own directory).
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yml",
            prefix=f"skeleton_{sk.skeleton_id}_",
            delete=False,
        ) as tmp:
            tmp.write(lowered_yaml)
            tmp_path = _Path(tmp.name)

        return cls.from_config(tmp_path)

    @classmethod
    def _requires_academy_integration(cls, config_path: Union[str, Path]) -> bool:
        """
        Check if workflow configuration contains Academy links

        Args:
            config_path: Path to workflow configuration file

        Returns:
            True if Academy integration is required, False otherwise
        """
        from nanobrain.core.academy_integration import AcademyIntegration
        return AcademyIntegration.requires_academy_integration(config_path)

    @classmethod
    def _setup_academy_manager(cls):
        """
        Setup Academy manager for distributed processing

        Returns:
            Academy Manager instance for agent communication

        Raises:
            ImportError: If Academy framework is not available
            RuntimeError: If Academy manager setup fails
        """
        from nanobrain.core.academy_integration import AcademyIntegration
        return AcademyIntegration.setup_academy_manager()

    @classmethod
    def _extract_resolved_components(cls, workflow_config: WorkflowConfig) -> Dict[str, Any]:
        """
        Extract instantiated components from resolved workflow configuration

        ConfigBase._resolve_nested_objects() has already instantiated all components
        specified with class+config patterns. This method extracts and validates them.

        Args:
            workflow_config: Resolved workflow configuration

        Returns:
            Dictionary containing categorized instantiated components

        ✅ FRAMEWORK COMPLIANCE:
        - Components already instantiated via ConfigBase._resolve_nested_objects()
        - No manual component creation or factory logic
        - Complete validation through ConfigBase schemas
        - Components immediately available for workflow use
        """
        resolved_components = {
            'steps': {},
            'links': {},
            'triggers': {},
            'data_units': {}
        }

        # Extract resolved steps
        steps_config = getattr(workflow_config, 'steps', {})
        for step_id, step_instance in steps_config.items():
            # Validate that it's a proper step instance
            if hasattr(step_instance, 'execute') or hasattr(step_instance, '__class__'):
                # Check if it's an instantiated object (not a dict)
                if not isinstance(step_instance, dict):
                    resolved_components['steps'][step_id] = step_instance
                    logger.debug(
                        f"✅ Extracted resolved step: {step_id} ({step_instance.__class__.__name__})")
                else:
                    # If still a dict, it means it's a legacy configuration that needs manual handling
                    logger.warning(
                        f"⚠️ Step '{step_id}' not resolved via class+config - requires legacy handling")
            else:
                logger.warning(
                    f"⚠️ Skipping invalid step instance: {step_id} (missing execute method)")

        # Extract resolved links
        links_config = getattr(workflow_config, 'links', {})
        for link_id, link_instance in links_config.items():
            # Validate that it's a proper link instance
            if hasattr(link_instance, 'transfer') or hasattr(link_instance, '__class__'):
                # Check if it's an instantiated object (not a dict)
                if not isinstance(link_instance, dict):
                    resolved_components['links'][link_id] = link_instance
                    logger.debug(
                        f"✅ Extracted resolved link: {link_id} ({link_instance.__class__.__name__})")
                else:
                    logger.warning(
                        f"⚠️ Link '{link_id}' not resolved via class+config - requires legacy handling")
            else:
                logger.warning(
                    f"⚠️ Skipping invalid link instance: {link_id} (missing transfer method)")

        # Extract resolved triggers - ✅ UNIFIED RESOLUTION: Handle list format (workflows ARE steps)
        triggers_config = getattr(workflow_config, 'triggers', [])

        # Handle both legacy dict format and unified list format
        if isinstance(triggers_config, dict):
            # Legacy dictionary format (backward compatibility)
            for trigger_id, trigger_instance in triggers_config.items():
                # ✅ FRAMEWORK COMPLIANCE: Check for correct trigger methods (bind_action, not start)
                if hasattr(trigger_instance, 'bind_action') or hasattr(trigger_instance, '__class__'):
                    if not isinstance(trigger_instance, dict):
                        resolved_components['triggers'][trigger_id] = trigger_instance
                        logger.debug(
                            f"✅ Extracted resolved trigger: {trigger_id} ({trigger_instance.__class__.__name__})")
                    else:
                        logger.warning(
                            f"⚠️ Trigger '{trigger_id}' not resolved via class+config - requires legacy handling")
                else:
                    logger.warning(
                        f"⚠️ Skipping invalid trigger instance: {trigger_id} (missing bind_action method)")
        elif isinstance(triggers_config, list):
            # ✅ UNIFIED LIST FORMAT: Process resolved trigger instances from list
            for i, trigger_instance in enumerate(triggers_config):
                # Extract trigger_id from instance attributes or generate one
                trigger_id = getattr(
                    trigger_instance, 'trigger_id', f'trigger_{i}')

                # Validate that it's a proper trigger instance
                if hasattr(trigger_instance, 'bind_action') or hasattr(trigger_instance, '__class__'):
                    # Check if it's an instantiated object (not a dict)
                    if not isinstance(trigger_instance, dict):
                        resolved_components['triggers'][trigger_id] = trigger_instance
                        logger.debug(
                            f"✅ Extracted unified trigger: {trigger_id} ({trigger_instance.__class__.__name__})")
                    else:
                        logger.warning(
                            f"⚠️ Trigger '{trigger_id}' not resolved via unified format - still a dict")
                else:
                    logger.warning(
                        f"⚠️ Skipping invalid unified trigger: {trigger_id} (missing bind_action method)")

        logger.info(f"✅ Extracted resolved components: {len(resolved_components['steps'])} steps, "
                    f"{len(resolved_components['links'])} links, {len(resolved_components['triggers'])} triggers")

        return resolved_components

    @classmethod
    def _create_from_resolved_config(cls, workflow_config: WorkflowConfig, resolved_components: Dict[str, Any], **context) -> 'Workflow':
        """
        Create workflow instance from resolved configuration and instantiated components

        This method assembles the workflow using components that have already been
        instantiated by ConfigBase._resolve_nested_objects().

        Args:
            workflow_config: Resolved workflow configuration
            resolved_components: Dictionary of instantiated components
            **context: Additional context

        Returns:
            Fully initialized workflow instance

        ✅ FRAMEWORK COMPLIANCE:
        - Uses pre-instantiated components from ConfigBase resolution
        - No manual component creation or factory dependencies
        - Complete configuration-driven workflow assembly
        - Validates component compatibility and integration
        """
        # Create executor if specified in context
        executor = context.get('executor')

        # Create workflow instance using standard component creation pattern
        component_config = cls.extract_component_config(workflow_config)
        dependencies = cls.resolve_dependencies(component_config, **context)
        workflow = cls.create_instance(
            workflow_config, component_config, dependencies)

        # Integrate resolved components into workflow
        workflow._integrate_resolved_components(resolved_components)

        # Store resolved components for workflow operation
        workflow._resolved_components = resolved_components

        # Validate integrated components
        workflow._validate_integrated_components(resolved_components)

        logger.info(
            f"✅ Created workflow from resolved config: {workflow_config.name}")

        return workflow

    def _integrate_resolved_components(self, resolved_components: Dict[str, Any]) -> None:
        """Integrate resolved components - trust framework resolution"""

        # ✅ ARCHITECTURAL FIX: Don't add workflow as step node to its own graph
        # Workflow-level data units will be handled specially in link resolution

        # ✅ FRAMEWORK COMPLIANCE: Register all steps before link resolution
        logger.info(
            f"🔧 STEP REGISTRATION: Starting for workflow {getattr(self, 'name', 'Unknown')}. Total resolved steps: {len(resolved_components['steps'])}")

        for step_id, step_instance in resolved_components['steps'].items():
            self.child_steps[step_id] = step_instance
            self.workflow_graph.add_step(step_id, step_instance)

            # Set step integration properties
            if hasattr(step_instance, 'step_id'):
                step_instance.step_id = step_id
            if hasattr(step_instance, 'executor') and not step_instance.executor:
                step_instance.executor = self.executor

            logger.info(
                f"✅ REGISTERED STEP: '{step_id}' (type: {type(step_instance).__name__})")

        # ✅ CRITICAL DEBUGGING: Verify all steps are registered before link resolution
        logger.info(f"📋 FINAL CHILD_STEPS: {list(self.child_steps.keys())}")
        logger.info(
            f"🔧 LINK RESOLUTION: Starting for {len(resolved_components['links'])} links")

        # Resolve and integrate links with proper data unit resolution
        for link_id, link_instance in resolved_components['links'].items():
            try:
                # Get string references from link config
                if hasattr(link_instance, 'config') and hasattr(link_instance.config, 'source') and hasattr(link_instance.config, 'target'):
                    source_ref = link_instance.config.source
                    target_ref = link_instance.config.target

                    if source_ref and target_ref:
                        # ✅ ARCHITECTURAL FIX: Extract step IDs directly from references first
                        logger.info(
                            f"🔍 PROCESSING LINK: '{link_id}' | {source_ref} -> {target_ref}")
                        logger.info(
                            f"🔧 AVAILABLE CHILD_STEPS: {list(self.child_steps.keys())}")
                        source_step_id = self._extract_step_id_from_reference(
                            source_ref)
                        target_step_id = self._extract_step_id_from_reference(
                            target_ref)
                        logger.info(
                            f"📋 EXTRACTED STEP IDs: {source_step_id} -> {target_step_id}")
                        if not source_step_id and '.' in source_ref:
                            logger.error(
                                f"❌ STEP ID EXTRACTION FAILED for source: '{source_ref}'")
                        if not target_step_id and '.' in target_ref:
                            logger.error(
                                f"❌ STEP ID EXTRACTION FAILED for target: '{target_ref}'")

                        # Resolve string references to actual data unit objects
                        source_data_unit = self._resolve_data_unit_reference(
                            source_ref)
                        target_data_unit = self._resolve_data_unit_reference(
                            target_ref)

                        # Set resolved objects on link instance (using property setters for proper name updates)
                        link_instance.source = source_data_unit
                        link_instance.target = target_data_unit

                        # Add to workflow structures
                        self.step_links[link_id] = link_instance

                        # ✅ ARCHITECTURAL COMPLIANCE: Only add step-to-step connections to graph
                        # Workflow maintains pure orchestrator role with no virtual processing nodes
                        if source_step_id and target_step_id:
                            # Step-to-step: add direct connection - proper dataflow orchestration
                            self.workflow_graph.add_link(
                                link_id, link_instance, source_step_id, target_step_id)
                            logger.debug(
                                f"✅ Integrated step-to-step link: {link_id} ({source_step_id} -> {target_step_id})")
                        elif source_step_id and not target_step_id:
                            # Step-to-workflow: step produces output, step participates in workflow
                            logger.debug(
                                f"✅ Integrated step-to-workflow link: {link_id} ({source_step_id} -> workflow)")
                        elif not source_step_id and target_step_id:
                            # Workflow-to-step: workflow input flows to step, step participates in workflow
                            logger.debug(
                                f"✅ Integrated workflow-to-step link: {link_id} (workflow -> {target_step_id})")
                        else:
                            # Workflow-level link: pure data flow without step processing
                            logger.debug(
                                f"✅ Integrated workflow-level link: {link_id} (workflow internal)")
                    else:
                        logger.warning(
                            f"⚠️ Link {link_id} missing source/target references")
                        self.step_links[link_id] = link_instance
                else:
                    logger.warning(
                        f"⚠️ Link {link_id} missing config or source/target attributes")
                    self.step_links[link_id] = link_instance

            except Exception as e:
                logger.error(
                    f"❌ Failed to resolve link {link_id}: {e}", exc_info=True)
                # Still add the link even if resolution fails
                self.step_links[link_id] = link_instance

        # ✅ WORKFLOW SCOPE: Only manage step-to-step connections (links)
        # Steps handle their own internal data units and triggers independently

        # Store workflow-level triggers (should only reference workflow-level data units)
        self._workflow_triggers = resolved_components['triggers']

        # ✅ ARCHITECTURAL COMPLIANCE: No cross-scope trigger resolution
        # Each step handles its own trigger resolution during step.initialize()
        # Workflow only manages links between step data units

        logger.info(f"✅ Workflow integration complete: "
                    f"{len(self.child_steps)} steps, {len(self.step_links)} links")

    def _resolve_data_unit_reference(self, reference: str) -> Any:
        """
        Resolve string reference to actual data unit object

        Formats:
        - "workflow_input" -> workflow-level data unit
        - "user_query"/"chatbot_response" -> workflow-level data units
        - "step_id.data_unit_name" -> step-level data unit

        Args:
            reference: String reference to resolve

        Returns:
            Actual DataUnit instance

        Raises:
            ValueError: If reference cannot be resolved
        """
        if '.' not in reference:
            # ✅ FRAMEWORK COMPLIANCE: Workflow-level data units are stored in step_input_data_units/step_output_data_units
            # Check workflow's own input data units first
            if hasattr(self, 'step_input_data_units') and self.step_input_data_units and reference in self.step_input_data_units:
                return self.step_input_data_units[reference]

            # Check workflow's own output data units
            elif hasattr(self, 'step_output_data_units') and self.step_output_data_units and reference in self.step_output_data_units:
                return self.step_output_data_units[reference]

            # ✅ LEGACY SUPPORT: Keep backward compatibility for workflow_input/output references
            elif reference == 'workflow_input' and hasattr(self, 'input_data_unit'):
                return self.input_data_unit
            elif reference == 'workflow_output' and hasattr(self, 'output_data_unit'):
                return self.output_data_unit
            else:
                # ✅ ENHANCED ERROR REPORTING: Show available workflow-level data units
                available_input = list(
                    getattr(self, 'step_input_data_units', {}).keys())
                available_output = list(
                    getattr(self, 'step_output_data_units', {}).keys())
                available_units = available_input + available_output
                raise ValueError(
                    f"Workflow-level data unit '{reference}' not found. Available: {available_units}")
        else:
            # Step-level data unit
            step_id, data_unit_name = reference.split('.', 1)

            if step_id not in self.child_steps:
                raise ValueError(f"Step '{step_id}' not found in workflow")

            step = self.child_steps[step_id]

            # Check output first, then input data units
            if hasattr(step, 'step_output_data_units') and step.step_output_data_units and data_unit_name in step.step_output_data_units:
                return step.step_output_data_units[data_unit_name]
            elif hasattr(step, 'step_input_data_units') and step.step_input_data_units and data_unit_name in step.step_input_data_units:
                return step.step_input_data_units[data_unit_name]
            else:
                # Try hierarchical component registry lookup as fallback
                from nanobrain.core.logging_system import get_system_log_manager
                system_manager = get_system_log_manager()
                scoped_name = f"{step_id}.{data_unit_name}"
                component_id = f"data_units_{scoped_name}"

                if component_id in system_manager.component_registry:
                    component_info = system_manager.component_registry[component_id]
                    return component_info['instance']

                # ✅ ENHANCED ERROR REPORTING: Show available step data units
                available_input = list(
                    getattr(step, 'step_input_data_units', {}).keys())
                available_output = list(
                    getattr(step, 'step_output_data_units', {}).keys())
                available_units = available_input + available_output
                raise ValueError(
                    f"Data unit '{data_unit_name}' not found in step '{step_id}'. Available: {available_units}")

    def _extract_step_id_from_reference(self, reference: str) -> Optional[str]:
        """
        ✅ FRAMEWORK COMPLIANCE: Extract step ID directly from data unit reference

        Handles both patterns:
        - "step.data_unit" -> returns "step"
        - "data_unit" -> returns None (workflow-level)

        Args:
            reference: Data unit reference string

        Returns:
            Step ID if step-level reference, None if workflow-level
        """
        if not reference or not isinstance(reference, str):
            return None

        # Handle step.data_unit notation
        if '.' in reference:
            step_id, data_unit_name = reference.split('.', 1)
            step_id = step_id.strip()

            # ✅ ENHANCED DEBUGGING: Validate that step exists in workflow
            if step_id in self.child_steps:
                logger.debug(
                    f"✅ Step ID '{step_id}' found for reference '{reference}'")
                return step_id
            else:
                logger.error(
                    f"❌ Step '{step_id}' referenced in '{reference}' not found in workflow")
                logger.error(
                    f"📋 Available steps in child_steps: {list(self.child_steps.keys())}")
                logger.error(
                    f"🔍 Step lookup failed for reference pattern: {reference}")
                return None
        else:
            # Workflow-level data unit (no step prefix)
            return None

    # REMOVED: _ensure_virtual_workflow_node - violated pure orchestrator architecture
    # Workflows are orchestrators, not processors - no virtual processing nodes allowed

    def _get_step_id_for_data_unit(self, data_unit: Any) -> Optional[str]:
        """
        Get the step ID for a data unit, handling both workflow-level and step-level data units

        Args:
            data_unit: The data unit to find the step ID for

        Returns:
            Step ID if found, None if workflow-level data unit
        """
        data_unit_name = getattr(data_unit, 'name', None)

        if not data_unit_name:
            return None

        # ✅ ARCHITECTURAL FIX: Don't return 'workflow' as step ID
        # Workflow-level data units should return None
        if data_unit_name in ['workflow_input', 'workflow_output', 'user_query', 'chatbot_response']:
            return None

        # 🔥 BRUTAL FIX: Handle step.data_unit naming pattern
        # If data unit name is like "query_processor.output", extract step and local names
        if '.' in data_unit_name:
            potential_step_id, local_data_unit_name = data_unit_name.split('.', 1)

            # Check if this step exists and has the local data unit
            if potential_step_id in self.child_steps:
                step_instance = self.child_steps[potential_step_id]

                # Check step's output data units for local name
                if hasattr(step_instance, 'step_output_data_units') and step_instance.step_output_data_units:
                    if local_data_unit_name in step_instance.step_output_data_units:
                        logger.debug(f"✅ Found step '{potential_step_id}' for data unit '{data_unit_name}' (local: '{local_data_unit_name}')")
                        return potential_step_id

                # Check step's input data units for local name
                if hasattr(step_instance, 'step_input_data_units') and step_instance.step_input_data_units:
                    if local_data_unit_name in step_instance.step_input_data_units:
                        logger.debug(f"✅ Found step '{potential_step_id}' for data unit '{data_unit_name}' (local: '{local_data_unit_name}')")
                        return potential_step_id

        # Fallback: For step-level data units, find which step owns this data unit by full name
        for step_id, step_instance in self.child_steps.items():
            # Check step's output data units
            if hasattr(step_instance, 'step_output_data_units') and step_instance.step_output_data_units:
                if data_unit_name in step_instance.step_output_data_units:
                    logger.debug(f"✅ Found step '{step_id}' for data unit '{data_unit_name}' (full name match)")
                    return step_id

            # Check step's input data units
            if hasattr(step_instance, 'step_input_data_units') and step_instance.step_input_data_units:
                if data_unit_name in step_instance.step_input_data_units:
                    logger.debug(f"✅ Found step '{step_id}' for data unit '{data_unit_name}' (full name match)")
                    return step_id

        # 🔥 ENHANCED ERROR REPORTING: Show what data units are actually available
        logger.warning(f"⚠️ Could not find step ID for data unit: {data_unit_name}")
        logger.debug("🔍 Available step data units:")
        for step_id, step_instance in self.child_steps.items():
            input_units = list(getattr(step_instance, 'step_input_data_units', {}).keys())
            output_units = list(getattr(step_instance, 'step_output_data_units', {}).keys())
            logger.debug(f"   Step '{step_id}': inputs={input_units}, outputs={output_units}")

        return None

    def _validate_integrated_components(self, resolved_components: Dict[str, Any]) -> None:
        """
        Validate that integrated components are compatible and properly configured

        Args:
            resolved_components: Dictionary of instantiated components

        Raises:
            ValueError: If component integration validation fails
        """
        # Validate steps
        for step_id, step_instance in resolved_components['steps'].items():
            if not hasattr(step_instance, 'execute'):
                raise ValueError(
                    f"❌ Invalid step: {step_id} missing execute method")

            # Validate step has required configuration
            if not hasattr(step_instance, 'config') or not hasattr(step_instance, 'name'):
                logger.warning(
                    f"⚠️ Step {step_id} missing standard configuration attributes")

        # Validate links reference existing steps and check for self-referencing links
        for link_id, link_instance in resolved_components['links'].items():
            if hasattr(link_instance, 'source') and hasattr(link_instance, 'target') and link_instance.source and link_instance.target:
                # ✅ CRITICAL: Check for self-referencing data unit links during configuration validation
                # Only check object identity, not name equality (different objects can have same name)
                if link_instance.source is link_instance.target:
                    source_name = getattr(link_instance.source, 'name', str(link_instance.source))
                    error_msg = (
                        f"❌ ILLEGAL SELF-REFERENCING DATA UNIT LINK DETECTED IN CONFIGURATION: "
                        f"Link '{link_id}' connects data unit '{source_name}' to itself. "
                        f"Self-referencing links are prohibited in the workflow architecture as they "
                        f"create infinite trigger loops and prevent proper workflow execution. "
                        f"Please remove this link from your configuration.")
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Get step IDs for source and target data units
                source_step_id = self._get_step_id_for_data_unit(
                    link_instance.source)
                target_step_id = self._get_step_id_for_data_unit(
                    link_instance.target)

                # Check if source and target step IDs exist in workflow graph
                source_found = source_step_id in self.workflow_graph.nodes if source_step_id else False
                target_found = target_step_id in self.workflow_graph.nodes if target_step_id else False

                if not source_found:
                    logger.warning(
                        f"⚠️ Link {link_id} source step '{source_step_id}' not found in workflow graph")
                if not target_found:
                    logger.warning(
                        f"⚠️ Link {link_id} target step '{target_step_id}' not found in workflow graph")

        # ✅ FRAMEWORK COMPLIANCE: Validate triggers have required framework methods
        for trigger_id, trigger_instance in resolved_components['triggers'].items():
            if not hasattr(trigger_instance, 'bind_action'):
                raise ValueError(
                    f"❌ Invalid trigger: {trigger_id} missing bind_action method")

        logger.info("✅ All integrated components validated successfully")

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve Workflow dependencies including custom executor configuration"""
        # Check for workflow-level executor configuration
        executor = kwargs.get('executor')
        if executor is None:
            # Import here to avoid circular imports
            from .executor import ExecutorConfig, ParslExecutor

            # Try to use workflow-level executor_config
            executor_config_path = component_config.get('executor_config')
            if executor_config_path:
                try:
                    # Load executor configuration from file
                    executor_config = ExecutorConfig.from_config(executor_config_path)

                    # Create appropriate executor based on type
                    if executor_config.executor_type.value == 'parsl':
                        executor = ParslExecutor.from_config(executor_config)
                        logger.info(f"✅ Created ParslExecutor from workflow config: {executor_config_path}")
                    elif executor_config.executor_type.value == 'thread':
                        from .executor import ThreadExecutor
                        executor = ThreadExecutor.from_config(executor_config)
                        logger.info(f"✅ Created ThreadExecutor from workflow config: {executor_config_path}")
                    elif executor_config.executor_type.value == 'process':
                        from .executor import ProcessExecutor
                        executor = ProcessExecutor.from_config(executor_config)
                        logger.info(f"✅ Created ProcessExecutor from workflow config: {executor_config_path}")
                    else:
                        # Default to LocalExecutor
                        executor = LocalExecutor.from_config(executor_config)
                        logger.info(f"✅ Created LocalExecutor from workflow config: {executor_config_path}")

                except Exception as e:
                    logger.warning(f"⚠️ Failed to load workflow executor config {executor_config_path}: {e}")
                    logger.info("🔄 Falling back to default LocalExecutor")
                    # Fall back to default behavior
                    executor = None

            # If no executor config or loading failed, use default from parent
            if executor is None:
                base_deps = super().resolve_dependencies(component_config, **kwargs)
                executor = base_deps['executor']

        return {
            'executor': executor
        }

    @classmethod
    def extract_component_config(cls, config: WorkflowConfig) -> Dict[str, Any]:
        """Extract Workflow configuration"""
        base_config = super().extract_component_config(config)
        return {
            **base_config,
            'steps': getattr(config, 'steps', []),
            'links': getattr(config, 'links', []),
            # Data-driven workflows don't need execution strategies
            'enable_monitoring': getattr(config, 'enable_monitoring', True),
            'workflow_directory': getattr(config, 'workflow_directory', None),
            'max_parallel_steps': getattr(config, 'max_parallel_steps', 10),
            'step_timeout': getattr(config, 'step_timeout', 300.0),
            'retry_attempts': getattr(config, 'retry_attempts', 3),
            'retry_delay': getattr(config, 'retry_delay', 1.0),
            'validate_graph': getattr(config, 'validate_graph', True),
            'allow_cycles': getattr(config, 'allow_cycles', False),
            'require_connected_graph': getattr(config, 'require_connected_graph', True),
            'enable_progress_reporting': getattr(config, 'enable_progress_reporting', True),
            'progress_batch_interval': getattr(config, 'progress_batch_interval', 3.0),
            'progress_collapsed_by_default': getattr(config, 'progress_collapsed_by_default', True),
            'progress_show_technical_errors': getattr(config, 'progress_show_technical_errors', True),
            'progress_preserve_session_history': getattr(config, 'progress_preserve_session_history', True),
            'executor_config': getattr(config, 'executor_config', None)
        }

    def _init_from_config(self, config: WorkflowConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Enhanced workflow initialization with automatic data unit creation"""
        super()._init_from_config(config, component_config, dependencies)

        # G7 Step 2 — emit deprecation WARNING for v1 workflows with
        # DirectLinks (or subclasses) whose YAML omits auto_transfer.
        # Without auto_transfer: true the link silently no-ops on every
        # trigger fire — the dominant silent-failure shape in the codebase
        # (architecture.md §13 brutal-truth #3).
        # See `apecx-mcp-integration/docs/nanobrain_capability_gaps.md G7`.
        _warn_on_implicit_auto_transfer(
            workflow_name=getattr(config, 'name', '<unnamed>'),
            links_config=getattr(config, 'links', {}) or {},
            config_version=getattr(config, 'config_version', 1),
        )

        # Workflow-specific configuration
        self.workflow_config = config

        # Core workflow components
        self.workflow_graph = WorkflowGraph()

        # Resolve workflow directory properly
        workflow_dir = component_config.get('workflow_directory') or "."
        if not Path(workflow_dir).is_absolute():
            # If relative path, search for it in common locations
            possible_paths = [
                Path(workflow_dir),  # Relative to current directory
                Path.cwd() / workflow_dir,  # Relative to current working directory
                # Relative to nanobrain root
                Path(__file__).parent.parent / workflow_dir,
                # One level up from nanobrain/core/
                Path(__file__).parent.parent.parent / workflow_dir,
            ]

            for possible_path in possible_paths:
                if possible_path.exists():
                    # Use absolute path
                    workflow_dir = str(possible_path.resolve())
                    break
            else:
                # If none found, try to find nanobrain package root more systematically
                current = Path(__file__).parent  # nanobrain/core/
                while current.parent != current:  # Go up until filesystem root
                    candidate = current / workflow_dir
                    if candidate.exists():
                        workflow_dir = str(candidate.resolve())
                        break
                    current = current.parent

        # Step and link management
        self.child_steps: Dict[str, BaseStep] = {}
        self.step_links: Dict[str, LinkBase] = {}

        # Execution state
        self.execution_order: List[str] = []
        self.current_step_index: int = 0
        self.is_workflow_complete: bool = False
        self.failed_steps: Set[str] = set()
        self.completed_steps: Set[str] = set()

        # Performance tracking
        self.step_execution_times: Dict[str, float] = {}
        self.workflow_start_time: Optional[float] = None
        self.workflow_end_time: Optional[float] = None

        # Progress reporting
        self.progress_reporter: Optional[ProgressReporter] = None
        if component_config.get('enable_progress_reporting', True):
            session_id = dependencies.get('session_id')
            self.progress_reporter = ProgressReporter(
                workflow_id=f"{self.name}_{int(time.time())}",
                workflow_name=self.name,
                session_id=session_id
            )
            self.progress_reporter.workflow_progress.batch_interval = component_config.get(
                'progress_batch_interval', 3.0)
            self.progress_reporter.workflow_progress.collapsed_by_default = component_config.get(
                'progress_collapsed_by_default', True)
            self.progress_reporter.workflow_progress.show_technical_errors = component_config.get(
                'progress_show_technical_errors', True)
            self.progress_reporter.workflow_progress.preserve_session_history = component_config.get(
                'progress_preserve_session_history', True)

        # Workflow-specific logger
        self.workflow_logger = get_logger(
            f"workflow.{self.name}", debug_mode=component_config.get('debug_mode', False))

        self.workflow_logger.info(f"Initialized workflow: {self.name}")

    # Workflow inherits FromConfigBase.__init__ which prevents direct instantiation
    # Use Workflow.from_config() to create instances



    async def initialize(self) -> None:
        """Initialize workflow: load steps, create links, build graph."""
        self.workflow_logger.info(f"🔥 BRUTAL TRUTH: Workflow.initialize() called for {self.name}")
        if self._is_initialized:
            self.workflow_logger.info("🔥 BRUTAL TRUTH: Workflow already initialized, skipping")
            return

        self.workflow_logger.info("🔥 BRUTAL TRUTH: Starting workflow initialization")
        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE,
            f"{self.name}.initialize_workflow"
        ) as context:

            # Initialize as Step first
            await super().initialize()

            # Load workflow configuration
            await self._load_workflow_configuration()

            # FAIL-FAST: Validate workflow integrity before initialization
            # SKIP for divergence-enabled workflows (imperative execution doesn't use links)
            divergence_enabled = getattr(self.workflow_config, 'divergence_enabled', False)
            if divergence_enabled:
                self.workflow_logger.info("🔥 DIVERGENCE MODE: Skipping link-based validation")
            else:
                self._validate_workflow_integrity()

            # Initialize child steps
            await self._initialize_child_steps()

            # Create step links
            await self._create_step_links()

            # Build and validate workflow graph
            await self._build_workflow_graph()

            # Skip graph validation for divergence-enabled workflows
            if not divergence_enabled:
                await self._validate_workflow()
            else:
                self.workflow_logger.info("🔥 DIVERGENCE MODE: Skipping graph validation")

            # Data-driven workflows don't need predetermined execution order

            # Initialize progress reporting
            if self.progress_reporter:
                self.progress_reporter.initialize_steps(
                    self.workflow_config.steps)
                await self.progress_reporter.update_progress(
                    'workflow_init', 100, 'completed',
                    message="Workflow initialized successfully"
                )

            context.metadata['num_steps'] = len(self.child_steps)
            context.metadata['num_links'] = len(self.step_links)
            context.metadata['execution_strategy'] = 'data_driven'

        self.workflow_logger.info(
            f"Workflow {self.name} initialized successfully",
            num_steps=len(self.child_steps),
            num_links=len(self.step_links),
            execution_strategy='data_driven'
        )

    async def run(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        *,
        await_cascade: bool = True,
        timeout: float = 60.0,
        settle_ms: int = 50,
        raise_on_cascade_timeout: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """G8 — canonical synchronous entry point for a workflow run.

        ``Workflow.process()`` is intentionally fire-and-forget: it
        deposits ``input_data`` into the first step's input data unit and
        returns immediately while triggers fire in background tasks. That
        shape is correct for trigger-driven event flow but is the
        load-bearing source of the "the workflow loaded; nothing happened;
        no error" silent-failure shape (architecture.md §13 brutal-truth #1).

        ``run()`` is the canonical synchronous wrapper: invoke ``process()``,
        await the cascade until quiet, then collect the workflow-level
        output data units into a dict and return them. Use ``run()`` from
        any caller that wants "give me the answer" semantics; reserve
        ``process()`` for trigger-injection scenarios where the caller
        intentionally yields control to the trigger cascade.

        Args:
            input_data: Mapping of workflow-level input data unit name to
                value. Defaults to empty dict (for workflows whose first
                step has no inputs — rare but legal).
            await_cascade: When True (default), block until the trigger
                cascade drains. When False, return immediately after
                ``process()`` returns — equivalent to calling ``process()``
                directly.
            timeout: Maximum seconds to wait for the cascade to drain.
                Default 60s — appropriate for typical LLM-bound workflows.
                For short pure-compute workflows, lower (5-10s) is faster
                to diagnose hung steps.
            settle_ms: Milliseconds the trigger executor must remain idle
                before the cascade is considered drained. Default 50ms —
                small enough to be responsive, large enough to absorb
                routine post-step bookkeeping.
            raise_on_cascade_timeout: When True, a cascade timeout raises
                ``TimeoutError``. When False (default), return a result
                dict with ``"status": "cascade_timeout"`` so the caller
                can inspect partial output. Operators should default to
                False in production (partial output is more diagnosable
                than an exception); tests may prefer True for fail-fast.

        Returns:
            A dict with one key per workflow-level output data unit, plus
            a ``"status"`` field. Possible status values:
              * ``"completed"`` — cascade drained cleanly
              * ``"completed_no_await"`` — ``await_cascade=False``; outputs
                may not yet reflect the cascade's effect
              * ``"cascade_timeout"`` — cascade did not drain within
                ``timeout`` (when ``raise_on_cascade_timeout=False``)
              * ``"no_first_step"`` — workflow has no executable first step

        Example::

            wf = Workflow.from_config('my_workflow.yml')
            result = await wf.run({'query': 'find candidates'})
            assert result['status'] == 'completed'
            answer = result['final_answer']  # whatever the workflow's
                                              # output data unit is named

        Cross-reference: ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G8``.
        """
        if input_data is None:
            input_data = {}

        process_result = await self.process(input_data, **kwargs)

        if not await_cascade:
            # Caller explicitly opted out of waiting. Pass through
            # process()'s return value (typically a status dict) and tag
            # it so the consumer knows outputs may be stale.
            outputs = await self._collect_workflow_output_data_units()
            outputs["status"] = "completed_no_await"
            outputs["_process_return"] = process_result
            return outputs

        # Special-case the no-first-step shape from process(): no cascade
        # to wait for, just echo the status and return empty outputs.
        if isinstance(process_result, dict) and process_result.get("status") == "no_first_step":
            outputs = {"status": "no_first_step", "workflow": self.name}
            return outputs

        cascade_drained = await self.wait_for_cascade(
            timeout=timeout, settle_ms=settle_ms
        )

        outputs = await self._collect_workflow_output_data_units()

        if not cascade_drained:
            if raise_on_cascade_timeout:
                raise TimeoutError(
                    f"Workflow {self.name!r} cascade did not drain within "
                    f"{timeout}s (settle_ms={settle_ms}). Partial outputs: "
                    f"{list(outputs.keys())}"
                )
            outputs["status"] = "cascade_timeout"
            outputs["_timeout_seconds"] = timeout
        else:
            outputs["status"] = "completed"

        return outputs

    def aggregate_resource_envelope(self) -> "ResourceEnvelope":
        """G12 — aggregate per-step ResourceEnvelopes into one
        workflow-level envelope.

        Walks every child step (and recurses into nested workflows),
        collects each step's declared ``resource_envelope``, and
        applies the per-field aggregation rules from
        ``aggregate_resource_envelopes`` (sum walltime + cost; max
        cpu + memory; union capability_tokens).

        Returns a ResourceEnvelope. Steps without a declared envelope
        contribute nothing — they don't penalize the rollup, but they
        also don't bound it. Operators who want a fully-bounded
        workflow MUST declare envelopes on every step.

        Used by:
        - HPC bundle exporter (sizes the PBS request).
        - HITL cost gate (checks workflow envelope against per-user
          threshold before executing).

        See ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G12``.
        """
        from .step import ResourceEnvelope, aggregate_resource_envelopes

        envelopes: List[ResourceEnvelope] = []

        # Walk children. self.child_steps is the standard container
        # populated by _init_from_config.
        children = getattr(self, "child_steps", {}) or {}
        for step_id, step in children.items():
            # Recurse into nested Workflows (Workflow IS a Step, so a
            # nested Workflow's aggregate is its own subtree's roll-up).
            if isinstance(step, Workflow):
                envelopes.append(step.aggregate_resource_envelope())
                continue

            cfg = getattr(step, "config", None)
            if cfg is None:
                continue
            raw = getattr(cfg, "resource_envelope", None)
            if raw is None:
                continue
            if isinstance(raw, ResourceEnvelope):
                envelopes.append(raw)
            elif isinstance(raw, dict):
                ResourceEnvelope._allow_direct_instantiation = True
                try:
                    envelopes.append(ResourceEnvelope(**raw))
                finally:
                    ResourceEnvelope._allow_direct_instantiation = False
            # Other types are silently skipped — type-mismatch should
            # have FAIL-FASTed at config load.

        return aggregate_resource_envelopes(envelopes)

    async def _collect_workflow_output_data_units(self) -> Dict[str, Any]:
        """G8 helper — read every workflow-level output data unit's
        current value into a dict. Workflow-level outputs live on
        ``self.step_output_data_units`` (per the framework's "workflow IS
        a step" model). Each data unit's ``.get()`` may itself await
        (e.g., DataUnitProxyRef.get() round-trips through the store).
        """
        outputs: Dict[str, Any] = {}
        owned = getattr(self, "step_output_data_units", {}) or {}
        for unit_name, data_unit in owned.items():
            try:
                outputs[unit_name] = await data_unit.get()
            except Exception as e:
                # Don't let one bad data unit poison the whole result;
                # record the error inline so the caller can diagnose.
                outputs[unit_name] = None
                outputs.setdefault("_errors", {})[unit_name] = repr(e)
        return outputs

    async def wait_for_cascade(
        self,
        timeout: float = 30.0,
        settle_ms: int = 50,
    ) -> bool:
        """Block until the trigger cascade drains.

        ``Workflow.process(input_data)`` in data-driven mode is
        fire-and-forget: it writes to the first step's input data unit
        and returns immediately while the trigger cascade fires in
        background tasks. Tests and synchronous callers need a way to
        wait until the cascade has fully propagated to the workflow's
        output data units.

        This method delegates to
        :meth:`AsyncTriggerExecutor.wait_for_all_tasks`, which loops
        over the executor's background-task set and handles the
        transitive case (task A spawning task B during its execution).

        Returns ``True`` when the cascade is quiet for ``settle_ms``
        milliseconds; ``False`` on timeout.

        Example::

            wf = Workflow.from_config('my_workflow.yml')
            await wf.process({'query': 'EEEV vaccines'})
            ok = await wf.wait_for_cascade(timeout=60.0)
            assert ok, 'cascade did not drain in time'
            output = await wf.last_output_data_unit().get()
        """
        from .trigger import AsyncTriggerExecutor

        executor = await AsyncTriggerExecutor.get_instance()
        return await executor.wait_for_all_tasks(
            timeout=timeout,
            settle_ms=settle_ms,
        )

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """
        Workflow processing with support for both data-driven and divergence-aware execution.

        Two execution modes:
        1. Data-driven (default): Populate first step, let triggers handle flow
        2. Imperative (divergence_enabled=True): Execute steps sequentially, check for divergence
        """
        # Check if divergence is enabled
        divergence_enabled = getattr(self.config, 'divergence_enabled', False)

        if divergence_enabled:
            # IMPERATIVE MODE: Execute steps and check for divergence
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(
                    f"🔥 DIVERGENCE MODE: Executing workflow {self.name} with divergence support")
            return await self._process_with_divergence(input_data, **kwargs)
        else:
            # DATA-DRIVEN MODE: Original behavior
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(
                    f"🚀 Initiating data flow for workflow: {self.name}")

            # Find the first step in the workflow
            first_step = self._get_first_step()
            if not first_step:
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.warning(
                        "⚠️ No first step found - no data flow initiated")
                return {"status": "no_first_step", "workflow": self.name}

            # Populate input data units of the FIRST STEP only
            populated_units = 0
            if hasattr(first_step, 'step_input_data_units'):
                for unit_name, data_unit in first_step.step_input_data_units.items():
                    if unit_name in input_data:
                        await data_unit.set(input_data[unit_name])
                        populated_units += 1
                        if hasattr(self, 'nb_logger') and self.nb_logger:
                            self.nb_logger.info(
                                f"📥 Populated {unit_name} in first step: {first_step.name}")

            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(
                    f"✅ Data flow initiated - populated {populated_units} data units in first step")

            return {
                "status": "data_flow_initiated",
                "workflow": self.name,
                "first_step": first_step.name,
                "populated_units": populated_units
            }

    async def _process_with_divergence(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """
        Imperative workflow execution with divergence support.

        Executes steps sequentially, checks for divergence_spec in results,
        and spawns parallel subworkflows as needed.
        """
        # Get execution order (topologically sorted steps)
        execution_order = self._get_execution_order()

        if not execution_order:
            self.nb_logger.warning("⚠️ No execution order found")
            return {"status": "no_steps", "workflow": self.name}

        self.nb_logger.info(f"🔥 Execution order: {execution_order}")

        current_data = input_data

        for i, step_name in enumerate(execution_order):
            step = self.child_steps.get(step_name)
            if not step:
                raise RuntimeError(f"Step '{step_name}' not found in child_steps")

            self.nb_logger.info(f"🔥 Executing step {i+1}/{len(execution_order)}: {step_name}")

            # Execute step and wait for result
            step_result = await self._execute_step_imperative(step, current_data)

            # CHECK FOR DIVERGENCE
            if isinstance(step_result, dict) and 'divergence_spec' in step_result:
                self.nb_logger.info(f"🔥 DIVERGENCE DETECTED at step {step_name}")

                # Get remaining steps after this one
                remaining_steps = execution_order[i + 1:]

                if remaining_steps:
                    # Spawn parallel subworkflows with remaining steps
                    self.nb_logger.info(f"🔥 Spawning subworkflows with remaining steps: {remaining_steps}")

                    diverged_results = await self._spawn_diverged_subworkflows(
                        step_result['divergence_spec'],
                        remaining_steps,
                        step_result.get('result', {})
                    )

                    # NEW: Check for local aggregation (per-level aggregation before continuing)
                    local_agg_config = step_result['divergence_spec'].get('local_aggregation', {})

                    if local_agg_config.get('enabled', False):
                        # Local aggregation: aggregate N children → 1 result at THIS level
                        self.nb_logger.info(f"🔄 Local aggregation enabled: aggregating {len(diverged_results)} results at this level")

                        aggregated_result = self._aggregate_diverged_results(
                            diverged_results,
                            method=local_agg_config.get('method', 'reduce')
                        )

                        self.nb_logger.info(f"✅ Local aggregation complete: {len(diverged_results)} → 1 result")

                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result
                        # IMPORTANT: Return aggregated result immediately
                        # The subworkflows already completed ALL remaining steps
                        # Don't re-execute those steps at this level
                        return aggregated_result

                    # Original: Check if global aggregation is needed
                    aggregation_config = step_result['divergence_spec'].get('aggregation', {})

                    if aggregation_config.get('enabled', True):
                        # Aggregate results
                        return self._aggregate_diverged_results(
                            diverged_results,
                            method=aggregation_config.get('method', 'list')
                        )
                    else:
                        # Flow through without aggregation
                        self.nb_logger.info("🔥 Skipping aggregation - flowing results through")
                        return {
                            'diverged_results': diverged_results,
                            'flow_through': True
                        }
                else:
                    # Last step diverged - just return results
                    self.nb_logger.info("🔥 Last step diverged - no remaining steps")
                    return step_result.get('result', step_result)
            else:
                # No divergence - continue with next step
                current_data = step_result

        # All steps completed without divergence
        return current_data

    def _get_execution_order(self) -> List[str]:
        """Get topologically sorted list of steps for execution."""
        # Simple implementation - just return child_steps keys in order
        # TODO: Use workflow graph for proper topological sort
        return list(self.child_steps.keys())

    async def _execute_step_imperative(self, step, input_data: Dict[str, Any]) -> Any:
        """
        Execute step imperatively while respecting data-driven architecture.

        CRITICAL: Steps expect data in their DATA UNITS, not as direct parameters.
        This method bridges imperative execution with data-driven architecture by:
        1. Populating step's input data units with current_data
        2. Reading from data units to build process() input dict
        3. Calling step.process() with properly formatted input
        4. Writing result to step's output data units
        """
        self.nb_logger.info(f"🔥 Imperatively executing step: {step.name}")

        # 1. Populate step's input data units with current_data
        # Map input_data to step's expected input data unit keys
        if not hasattr(step, 'step_input_data_units'):
            self.nb_logger.warning(f"Step {step.name} has no input data units")
            step.step_input_data_units = {}

        for unit_name, data_unit in step.step_input_data_units.items():
            # For single input data unit, pass entire input_data
            if len(step.step_input_data_units) == 1:
                self.nb_logger.info(f"🔥 Writing entire input_data to single data unit '{unit_name}'")
                await data_unit.set(input_data)
                break
            else:
                # For multiple inputs, try to match keys
                if unit_name in input_data:
                    self.nb_logger.info(f"🔥 Writing input_data['{unit_name}'] to data unit")
                    await data_unit.set(input_data[unit_name])

        # 2. Prepare input_data dict for process() with correct keys
        # Read from populated data units to build process() input
        process_input = {}
        for unit_name, data_unit in step.step_input_data_units.items():
            data = await data_unit.get()
            process_input[unit_name] = data
            self.nb_logger.info(f"🔥 Read data from unit '{unit_name}': {type(data)}")

        # 3. Call step.process() with properly formatted input
        self.nb_logger.info(f"🔥 Calling {step.name}.process() with keys: {list(process_input.keys())}")
        result = await step.process(process_input)
        self.nb_logger.info(f"🔥 Step {step.name} returned result type: {type(result)}")

        # 4. Write result to step's output data units
        if not hasattr(step, 'step_output_data_units'):
            step.step_output_data_units = {}

        if isinstance(result, dict):
            for unit_name, data_unit in step.step_output_data_units.items():
                if unit_name in result:
                    self.nb_logger.info(f"🔥 Writing result['{unit_name}'] to output data unit")
                    await data_unit.set(result[unit_name])
                elif len(step.step_output_data_units) == 1:
                    # Single output - write entire result
                    self.nb_logger.info(f"🔥 Writing entire result to single output data unit '{unit_name}'")
                    await data_unit.set(result)
                    break

        return result

    def _get_first_step(self):
        """
        Get the first step in the workflow for data flow initiation.

        Returns the first step found in child_steps.
        In a proper data-driven workflow, this should be determined by
        configuration or dependency analysis.
        """
        if hasattr(self, 'child_steps') and self.child_steps:
            # Return the first step (in a real implementation, this would be
            # determined by workflow configuration or dependency analysis)
            return next(iter(self.child_steps.values()))

        # Fallback: check resolved components
        if hasattr(self, '_resolved_components'):
            resolved_steps = self._resolved_components.get('steps', {})
            if resolved_steps:
                return next(iter(resolved_steps.values()))

        return None

    # ==================== DIVERGENCE SUPPORT ====================

    async def _spawn_diverged_subworkflows(
        self,
        divergence_spec: Dict[str, Any],
        remaining_steps: List[str],
        base_result: Any
    ) -> List[Any]:
        """
        Spawn parallel subworkflows with strictness control.

        Args:
            divergence_spec: Specification for divergence (parallel_tasks, task_params, etc.)
            remaining_steps: List of step names to include in subworkflows
            base_result: Base result to merge with task-specific parameters

        Returns:
            List of results from parallel subworkflows
        """
        import uuid

        parallel_tasks = divergence_spec['parallel_tasks']
        task_params = divergence_spec['task_params']
        resource_spec = divergence_spec.get('resource_spec', {})

        # Extract strictness configuration
        strictness = divergence_spec.get('strictness', {
            'fail_fast': True,  # DEFAULT: Fail on first error
            'required_success_count': None,
            'continue_on_error': False
        })

        self.nb_logger.info(f"🔥 Spawning {parallel_tasks} subworkflows")
        self.nb_logger.info(f"🔥 Strictness: fail_fast={strictness['fail_fast']}")
        self.nb_logger.info(f"🔥 Remaining steps: {remaining_steps}")

        futures = []

        for i, params in enumerate(task_params):
            # Generate subworkflow config
            subworkflow_config_path = await self._generate_subworkflow_config(
                remaining_steps=remaining_steps,
                task_index=i,
                divergence_id=str(uuid.uuid4().hex[:8])
            )

            # Prepare input data for this subworkflow
            subworkflow_input = {
                **base_result,  # Include result from diverged step
                **params        # Add task-specific parameters
            }

            self.nb_logger.info(f"🔥 Creating task for subworkflow {i}")

            # Create async task for subworkflow execution
            # CRITICAL: ParslExecutor will detect subworkflow context and use WorkQueue
            async def create_subworkflow_task(config_path, input_data):
                return await self._execute_subworkflow_from_config(
                    config_path=config_path,
                    input_data=input_data
                )

            # Store task coroutine for parallel execution
            task = create_subworkflow_task(subworkflow_config_path, subworkflow_input)
            futures.append((i, task))

        # Collect results with strictness enforcement
        import asyncio

        results = []
        errors = []

        # Execute all tasks in parallel
        self.nb_logger.info(f"🔥 Executing {len(futures)} subworkflows in parallel")

        # Execute with proper error handling based on strictness
        if strictness['fail_fast']:
            # Fail on first error
            try:
                task_coroutines = [task for _, task in futures]
                completed_results = await asyncio.gather(*task_coroutines, return_exceptions=False)

                # All succeeded
                for i, result in enumerate(completed_results):
                    results.append({
                        'task_index': i,
                        'status': 'success',
                        'result': result
                    })
            except Exception as e:
                self.nb_logger.error(f"🔥 FAIL FAST: Task failed: {e}")
                raise RuntimeError(f"Subworkflow failed (fail_fast=True): {e}") from e
        else:
            # Collect all results, including errors
            task_coroutines = [task for _, task in futures]
            completed_results = await asyncio.gather(*task_coroutines, return_exceptions=True)

            for i, result in enumerate(completed_results):
                if isinstance(result, Exception):
                    self.nb_logger.error(f"❌ Task {i} failed: {result}")
                    errors.append({
                        'task_index': i,
                        'error': str(result)
                    })

                    if strictness['continue_on_error']:
                        results.append({
                            'task_index': i,
                            'status': 'failed',
                            'error': str(result)
                        })
                else:
                    results.append({
                        'task_index': i,
                        'status': 'success',
                        'result': result
                    })

        # Check if we met required success count
        if strictness['required_success_count'] is not None:
            success_count = sum(1 for r in results if r['status'] == 'success')
            if success_count < strictness['required_success_count']:
                raise RuntimeError(
                    f"Only {success_count}/{strictness['required_success_count']} "
                    f"tasks succeeded (required minimum not met)"
                )

        # If we have errors and not continuing on error, fail now
        if errors and not strictness['continue_on_error']:
            raise RuntimeError(
                f"{len(errors)}/{len(results)} tasks failed: {errors}"
            )

        self.nb_logger.info(f"🔥 All subworkflows completed: {len(results)} total, {len([r for r in results if r['status'] == 'success'])} succeeded")

        return results

    def _get_subworkflow_executor_config(self) -> Dict[str, Any]:
        """
        Get executor config for subworkflows by reading parent workflow's YAML.

        This is the CORRECT configuration-based approach:
        - Load parent workflow's YAML file
        - Extract executor section as-is
        - Return it for subworkflow config generation

        Subworkflows inherit the parent executor. For ParslExecutor, this allows
        task distribution across nodes while maintaining ONE PBS job (Parsl submits
        one PBS job and distributes tasks, not multiple independent jobs).

        Returns:
            Dict with executor class and config path (e.g., {'class': '...', 'config': '...'})
        """

        # Read parent workflow's config file to get executor section
        if hasattr(self, '_config_path') and self._config_path:
            try:
                with open(self._config_path, 'r') as f:
                    parent_config = yaml.safe_load(f)

                if 'executor' in parent_config and isinstance(parent_config['executor'], dict):
                    executor_config = parent_config['executor']
                    self.nb_logger.info(
                        f"🔄 Subworkflow inheriting executor from parent config: "
                        f"{executor_config.get('class', 'unknown')} with config {executor_config.get('config', 'unknown')}"
                    )
                    return executor_config
                else:
                    self.nb_logger.warning(
                        f"⚠️ Parent config {self._config_path} has no 'executor' section, using fallback"
                    )
            except Exception as e:
                self.nb_logger.error(f"❌ Failed to read parent config {self._config_path}: {e}")
        else:
            self.nb_logger.warning("⚠️ Workflow has no _config_path, cannot read parent executor config")

        # Fallback: construct from runtime executor instance
        self.nb_logger.info(f"🔄 Fallback: constructing executor config from runtime instance {self.executor.__class__.__name__}")
        return {
            'class': f"{self.executor.__class__.__module__}.{self.executor.__class__.__name__}",
            'config': 'test_divergence/local_executor.yml'  # Last resort default
        }

    async def _generate_subworkflow_config(
        self,
        remaining_steps: List[str],
        task_index: int,
        divergence_id: str
    ) -> str:
        """
        Generate config file for subworkflow - KEEP for accountability.

        Args:
            remaining_steps: Steps to include in subworkflow
            task_index: Index of this parallel task
            divergence_id: Unique ID for this divergence

        Returns:
            Path to generated config file
        """
        from pathlib import Path
        import yaml

        # Store in persistent location for accountability
        config_base_dir = Path("executed_workflows") / "diverged_subworkflows"

        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workflow_name_clean = self.config.name.replace('/', '_').replace(' ', '_')
        divergence_dir = config_base_dir / f"{workflow_name_clean}_{timestamp}_{divergence_id}"

        config_dir = divergence_dir / f"task_{task_index}"
        config_dir.mkdir(parents=True, exist_ok=True)

        self.nb_logger.info(f"📝 Creating subworkflow config in: {config_dir}")

        # Build subworkflow config with ONLY remaining steps
        subworkflow_config = {
            'name': f"{self.config.name}_subworkflow_{task_index}",
            'description': f"Subworkflow for parallel task {task_index}",
            'class': 'nanobrain.core.workflow.Workflow',
            'divergence_enabled': True,  # Enable recursive divergence

            # CRITICAL: Include ONLY remaining steps
            'steps': {},

            # CRITICAL: Use LocalExecutor for subworkflows when parent uses ParslExecutor
            # This prevents submitting multiple PBS jobs (only ONE job allowed!)
            'executor': self._get_subworkflow_executor_config(),

            # Input/output data units (simplified)
            'input_data_units': {},
            'output_data_units': {}
        }

        # Add remaining steps to config
        # CRITICAL: Each step entry must specify executor to override step config's executor
        executor_config = subworkflow_config['executor']  # Use workflow-level executor config

        for step_name in remaining_steps:
            if step_name in self.child_steps:
                step = self.child_steps[step_name]
                # Reference to original step config
                if hasattr(step, '_config_path'):
                    subworkflow_config['steps'][step_name] = {
                        'class': f"{step.__class__.__module__}.{step.__class__.__name__}",
                        'config': str(step._config_path),
                        # Add executor config to override step config's executor
                        # This ensures steps inherit subworkflow's executor (e.g., ParslExecutor)
                        'executor': executor_config
                    }

        # Write config to YAML file
        config_path = config_dir / "subworkflow_config.yml"
        with open(config_path, 'w') as f:
            # CRITICAL: sort_keys=False to preserve execution order!
            yaml.dump(subworkflow_config, f, default_flow_style=False, sort_keys=False)

        # ALSO write metadata for accountability
        metadata = {
            'parent_workflow': self.config.name,
            'parent_config': str(self._config_path) if hasattr(self, '_config_path') else None,
            'divergence_point': remaining_steps[0] if remaining_steps else 'final',
            'task_index': task_index,
            'divergence_id': divergence_id,
            'created_at': timestamp,
            'remaining_steps': remaining_steps
        }

        metadata_path = config_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.nb_logger.info(f"📝 Generated subworkflow config (KEPT): {config_path}")

        return str(config_path)

    async def _execute_subworkflow_from_config(
        self,
        config_path: str,
        input_data: Dict[str, Any],
        resource_specification: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute a subworkflow from config file.

        This is called via executor.submit(), so it needs to be a simple function.
        """
        self.nb_logger.info(f"🔥 Loading subworkflow from: {config_path}")

        # Load and execute subworkflow
        subworkflow = Workflow.from_config(config_path)

        # CRITICAL: Reuse parent's executor instance to avoid submitting multiple PBS jobs
        # All subworkflows share the same executor (e.g., same ParslExecutor = same PBS job)
        subworkflow.executor = self.executor
        self.nb_logger.info(f"🔄 Subworkflow inherits parent's executor: {type(self.executor).__name__}")

        await subworkflow.initialize()

        self.nb_logger.info(f"🔥 Executing subworkflow: {subworkflow.name}")
        result = await subworkflow.process(input_data)

        self.nb_logger.info(f"🔥 Subworkflow completed: {subworkflow.name}")

        return result

    def _aggregate_diverged_results(
        self,
        results: List[Dict[str, Any]],
        method: str = 'list'
    ) -> Any:
        """
        Aggregate results from parallel subworkflows.

        Args:
            results: List of result dictionaries with status and result
            method: Aggregation method ('list', 'merge', 'reduce', 'custom')

        Returns:
            Aggregated result
        """
        if method == 'list':
            # Simple list aggregation
            return {
                'parallel_results': [r['result'] for r in results if r['status'] == 'success'],
                'result_count': len(results),
                'success_count': sum(1 for r in results if r['status'] == 'success'),
                'failure_count': sum(1 for r in results if r['status'] == 'failed'),
                'aggregation_method': 'list'
            }
        elif method == 'merge':
            # Merge dictionaries
            merged = {}
            for r in results:
                if r['status'] == 'success' and isinstance(r['result'], dict):
                    merged.update(r['result'])
            return merged
        elif method == 'reduce':
            # Reduce N results → 1 aggregated result (for local aggregation)
            # Extract all successful results
            successful_results = [r['result'] for r in results if r['status'] == 'success']

            # If results contain 'final_numbers', aggregate them
            all_numbers = []
            for result in successful_results:
                if isinstance(result, dict):
                    if 'final_numbers' in result:
                        all_numbers.extend(result['final_numbers'])
                    elif 'output_numbers' in result:
                        all_numbers.extend(result['output_numbers'])

            if all_numbers:
                # Return aggregated numbers
                return {
                    'final_numbers': all_numbers,
                    'aggregated_from': len(successful_results),
                    'total_numbers': len(all_numbers)
                }
            else:
                # Generic reduction: just collect all results
                return {
                    'aggregated_results': successful_results,
                    'aggregated_from': len(successful_results)
                }
        else:
            # Default: just return list
            return [r['result'] for r in results if r['status'] == 'success']

    # ==================== END DIVERGENCE SUPPORT ====================

    async def _load_workflow_configuration(self) -> None:
        """Load step configurations from workflow configuration."""
        self.workflow_logger.debug("Loading workflow step configurations")

        # Workflow steps are already specified in the config
        # This method can be extended to load additional configuration
        pass

    async def _initialize_child_steps(self) -> None:
        """
        Initialize child steps from resolved components

        Steps are already instantiated by ConfigBase._resolve_nested_objects()
        and integrated into the workflow via _integrate_resolved_components().
        This method initializes the pre-instantiated steps.

        ✅ FRAMEWORK COMPLIANCE:
        - Uses pre-instantiated steps from ConfigBase resolution
        - No manual step creation or factory logic
        - Steps already validated through ConfigBase schemas
        - Immediate availability for workflow execution
        """
        self.workflow_logger.info("🔥 BRUTAL TRUTH: _initialize_child_steps() called")
        if not hasattr(self, '_resolved_components'):
            self.workflow_logger.warning(
                "⚠️ No resolved components found - workflow may not be fully configured")
            return

        resolved_steps = self._resolved_components.get('steps', {})
        self.workflow_logger.info(
            f"🔥 BRUTAL TRUTH: Found {len(resolved_steps)} resolved steps: {list(resolved_steps.keys())}")
        self.workflow_logger.info(
            f"Initializing {len(resolved_steps)} pre-instantiated child steps")

        # Initialize each resolved step
        for step_id, step_instance in resolved_steps.items():
            try:
                # Initialize the step if not already initialized
                if hasattr(step_instance, 'initialize') and hasattr(step_instance, '_is_initialized'):
                    if not step_instance._is_initialized:
                        self.workflow_logger.info(
                            f"🔥 BRUTAL TRUTH: About to initialize step: {step_id}")
                        await step_instance.initialize()
                        self.workflow_logger.info(
                            f"🔥 BRUTAL TRUTH: Successfully initialized step: {step_id}")
                    else:
                        self.workflow_logger.info(
                            f"🔥 BRUTAL TRUTH: Step already initialized, skipping: {step_id}")
                elif hasattr(step_instance, 'initialize'):
                    # Initialize even if _is_initialized attribute is not present
                    self.workflow_logger.info(
                        f"🔥 BRUTAL TRUTH: About to initialize step (no _is_initialized): {step_id}")
                    await step_instance.initialize()
                    self.workflow_logger.info(
                        f"🔥 BRUTAL TRUTH: Successfully initialized step (no _is_initialized): {step_id}")
                else:
                    self.workflow_logger.info(
                        f"🔥 BRUTAL TRUTH: Step does not require initialization: {step_id}")

                # Ensure step has required workflow integration properties
                if not hasattr(step_instance, 'step_id'):
                    step_instance.step_id = step_id

                # Set executor if not already set
                if hasattr(step_instance, 'executor') and not step_instance.executor:
                    step_instance.executor = self.executor

                # Set workflow directory context
                if hasattr(step_instance, 'workflow_directory') and not step_instance.workflow_directory:
                    step_instance.workflow_directory = self.workflow_config.workflow_directory

                # CRITICAL: Add parent workflow reference for ParslExecutor integration
                # This enables steps to access their parent workflow for divergence point creation
                step_instance.parent_workflow = self

                # Store the workflow config path for sub-workflow creation
                if hasattr(self, '_config_path'):
                    step_instance.parent_workflow_config_path = self._config_path

            except Exception as e:
                self.workflow_logger.error(
                    f"❌ Failed to initialize resolved step {step_id}: {e}", exc_info=True)
                raise ValueError(
                    f"Step initialization failed: {step_id} - {str(e)}") from e

        self.workflow_logger.info(
            f"✅ Initialized {len(resolved_steps)} resolved child steps")

    async def _create_step_links(self) -> None:
        """
        Initialize step links from resolved components

        Links are already instantiated by ConfigBase._resolve_nested_objects()
        and integrated into the workflow via _integrate_resolved_components().
        This method starts the pre-instantiated links.

        ✅ FRAMEWORK COMPLIANCE:
        - Uses pre-instantiated links from ConfigBase resolution
        - No manual link creation or factory logic
        - Links already validated through ConfigBase schemas
        - Immediate availability for workflow execution
        """
        if not hasattr(self, '_resolved_components'):
            self.workflow_logger.warning(
                "⚠️ No resolved components found - workflow may not be fully configured")
            return

        resolved_links = self._resolved_components.get('links', {})
        self.workflow_logger.info(
            f"Initializing {len(resolved_links)} pre-instantiated step links")

        # Initialize each resolved link
        for link_id, link_instance in resolved_links.items():
            try:
                # FRAMEWORK FIX: Ensure source and target data units are properly resolved
                if hasattr(link_instance, 'config'):
                    source_ref = getattr(link_instance.config, 'source', None)
                    target_ref = getattr(link_instance.config, 'target', None)

                    if source_ref and target_ref:
                        # Resolve actual data unit references
                        source_unit = self._resolve_data_unit_reference(
                            source_ref)
                        target_unit = self._resolve_data_unit_reference(
                            target_ref)

                        if source_unit and target_unit:
                            link_instance.source = source_unit
                            link_instance.target = target_unit
                            self.workflow_logger.debug(
                                f"🔗 Resolved data units for link {link_id}: {source_ref} -> {target_ref}")
                        else:
                            self.workflow_logger.warning(
                                f"⚠️ Could not resolve data units for link {link_id}: {source_ref} -> {target_ref}")

                # Ensure link has required workflow integration properties
                if not hasattr(link_instance, 'name') or not link_instance.name:
                    if hasattr(link_instance, 'name'):
                        link_instance.name = link_id

                # FRAMEWORK FIX: Setup automatic transfer after data unit resolution
                if hasattr(link_instance, '_setup_automatic_transfer_if_possible'):
                    link_instance._setup_automatic_transfer_if_possible()
                    self.workflow_logger.debug(
                        f"🔄 Setup automatic transfer for link {link_id}")

                # Start the link if it has a start method
                if hasattr(link_instance, 'start'):
                    await link_instance.start()
                    self.workflow_logger.debug(
                        f"✅ Started resolved link: {link_id}")
                else:
                    self.workflow_logger.debug(
                        f"✅ Link does not require starting: {link_id}")

                # Validate link has source and target
                if not (hasattr(link_instance, 'source') and hasattr(link_instance, 'target')):
                    self.workflow_logger.warning(
                        f"⚠️ Link {link_id} missing source/target properties")
                elif not (link_instance.source and link_instance.target):
                    self.workflow_logger.warning(
                        f"⚠️ Link {link_id} has None source/target after resolution")
                else:
                    # ✅ CRITICAL: Check for self-referencing data unit links
                    source_name = getattr(
                        link_instance.source, 'name', str(link_instance.source))
                    target_name = getattr(
                        link_instance.target, 'name', str(link_instance.target))

                    # Only check object identity, not name equality (different objects can have same name)
                    if link_instance.source is link_instance.target:
                        error_msg = (
                            f"❌ ILLEGAL SELF-REFERENCING DATA UNIT LINK: {link_id} connects "
                            f"data unit '{source_name}' to itself. Self-referencing links are "
                            f"prohibited in the workflow architecture as they create infinite "
                            f"trigger loops and prevent proper workflow execution.")
                        self.workflow_logger.error(error_msg)
                        raise ValueError(error_msg)

            except Exception as e:
                self.workflow_logger.error(
                    f"❌ Failed to initialize resolved link {link_id}: {e}", exc_info=True)
                raise ValueError(
                    f"Link initialization failed: {link_id} - {str(e)}") from e

        self.workflow_logger.info(
            f"✅ Initialized {len(resolved_links)} resolved step links")

    async def _build_workflow_graph(self) -> None:
        """Build the internal workflow graph representation."""
        # Graph is built incrementally in _initialize_child_steps and _create_step_links
        self.workflow_logger.debug("Workflow graph built successfully")

    async def _validate_workflow(self) -> None:
        """Validate the workflow graph structure."""
        if not self.workflow_config.validate_graph:
            self.workflow_logger.debug("Graph validation disabled")
            return

        is_valid, errors = self.workflow_graph.validate_graph(
            allow_cycles=self.workflow_config.allow_cycles,
            require_connected=self.workflow_config.require_connected_graph
        )

        if not is_valid:
            error_msg = "Workflow graph validation failed:\n" + \
                "\n".join(f"  - {error}" for error in errors)
            self.workflow_logger.error(error_msg)
            raise ValueError(error_msg)

        self.workflow_logger.info("Workflow graph validation passed")

    def _is_link_source_match(self, link: Any, source_data_unit_name: str) -> bool:
        """Check if a link has the specified data unit as its source."""
        if hasattr(link, 'config') and hasattr(link.config, 'source'):
            return link.config.source == source_data_unit_name
        if hasattr(link, 'source_ref'):
            return link.source_ref == source_data_unit_name
        if hasattr(link, 'source') and hasattr(link.source, 'name'):
            return link.source.name == source_data_unit_name
        return False

    # Enhancement 4: Event-Driven Completion Helper Methods
    def _get_completion_strategy(self) -> str:
        """Get the configured completion detection strategy."""
        if hasattr(self.workflow_config, 'completion_detection') and hasattr(self.workflow_config.completion_detection, 'strategy'):
            return self.workflow_config.completion_detection.strategy
        return 'event_driven'  # Data-driven workflows are always event-driven

    async def _wait_for_workflow_completion(self, output_unit_name: str) -> Any:
        """Wait for workflow completion via output data unit monitoring."""
        output_data_unit = self.step_output_data_units[output_unit_name]

        # Get timeout from configuration
        timeout = self._get_completion_timeout()

        # Setup completion detection
        completion_event = asyncio.Event()
        result_data = {'output': None, 'error': None}

        def on_completion(change_event):
            try:
                data = change_event.get('data')
                if data is not None:
                    result_data['output'] = data
                    completion_event.set()
            except Exception as e:
                result_data['error'] = e
                completion_event.set()

        # Register completion listener
        output_data_unit.register_change_listener(on_completion)

        try:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(
                    f"⏱️ Waiting for workflow completion on '{output_unit_name}' (timeout: {timeout}s)")

            # Wait for completion or timeout
            await asyncio.wait_for(completion_event.wait(), timeout=timeout)

            # Check for errors
            if result_data['error']:
                raise result_data['error']

            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info("✅ Workflow completed successfully")

            return result_data['output']

        except asyncio.TimeoutError:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"⏰ Workflow timed out after {timeout}s")
            raise TimeoutError(
                f"Workflow execution timed out after {timeout} seconds")

        finally:
            # Always cleanup the listener
            try:
                if hasattr(output_data_unit, 'unregister_change_listener'):
                    output_data_unit.unregister_change_listener(on_completion)
            except Exception as e:
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.warning(
                        f"⚠️ Failed to cleanup completion listener: {e}", exc_info=True)

    def _get_completion_timeout(self) -> float:
        """Get the configured completion timeout."""
        if hasattr(self.workflow_config, 'completion_detection') and hasattr(self.workflow_config.completion_detection, 'default_timeout_seconds'):
            return self.workflow_config.completion_detection.default_timeout_seconds
        return getattr(self.workflow_config, 'execution_timeout_seconds', 300)

    # Enhancement 3: Configuration Data Unit Creation Helper Method
    def _create_workflow_data_unit(self, unit_name: str, unit_config: Dict[str, Any], unit_type: str) -> DataUnitBase:
        """
        Create a workflow data unit from configuration using the standard from_config pattern.

        Reuses existing framework patterns for consistent data unit creation.
        """
        import importlib

        if not isinstance(unit_config, dict):
            raise ValueError(
                f"Data unit configuration for '{unit_name}' must be a dictionary")

        # Get the class path with default fallback
        class_path = unit_config.get(
            'class', 'nanobrain.core.data_unit.DataUnitMemory')

        # Dynamic import resolution (same pattern as existing framework code)
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            data_unit_class = getattr(module, class_name)

            # Validate it's a proper DataUnit class
            if not issubclass(data_unit_class, DataUnitBase):
                raise ValueError(
                    f"Class {class_path} is not a valid DataUnit class")

            # Prepare configuration dict
            config_dict = unit_config.copy()
            config_dict['name'] = unit_name

            # Create data unit using standard from_config pattern
            # Pass parent scope for proper component registration
            data_unit = data_unit_class.from_config(
                config_dict, parent_scope=self.name)

            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(
                    f"🏭 Created {unit_type} data unit '{unit_name}' using {class_name}")

            return data_unit

        except ImportError as e:
            raise ValueError(
                f"Failed to import data unit class '{class_path}': {e}")
        except AttributeError as e:
            raise ValueError(
                f"Class '{class_name}' not found in module '{module_path}': {e}")
        except Exception as e:
            logger = get_logger(f"workflow.{self.name}.data_units")
            logger.error(
                f"Failed to create data unit '{unit_name}' from config: {e}", exc_info=True)
            raise ValueError(
                f"Failed to create data unit '{unit_name}' from config: {e}")

    async def _update_step_progress(self, step_id: str, progress: int, message: str = None) -> None:
        """Update step progress during execution."""
        if self.progress_reporter:
            await self.progress_reporter.update_progress(
                step_id, progress, message=message
            )

    async def _propagate_step_data(self, step_id: str, step_result: Any) -> None:
        """Propagate step result data through connected links."""
        # Find all links originating from this step
        outgoing_links = [
            link for link in self.step_links.values()
            if hasattr(link, 'source') and getattr(link.source, 'name', None) == step_id
        ]

        # Transfer data through each link
        for link in outgoing_links:
            try:
                await link.transfer(step_result)
            except Exception as e:
                self.workflow_logger.error(
                    f"Data propagation failed for link {link.name}: {e}", exc_info=True)

    async def _handle_step_error(self, step_id: str, error: Exception) -> None:
        """Handle step execution errors according to error handling strategy."""
        error_strategy = self.workflow_config.error_handling

        self.workflow_logger.error(
            f"Step {step_id} failed with error: {error}")

        if error_strategy == ErrorHandlingStrategy.STOP:
            raise error
        elif error_strategy == ErrorHandlingStrategy.CONTINUE:
            # Continue with next steps
            self.workflow_logger.warning(
                f"Continuing workflow despite step {step_id} failure")
        elif error_strategy == ErrorHandlingStrategy.RETRY:
            # Implement retry logic
            await self._retry_step(step_id, error)
        elif error_strategy == ErrorHandlingStrategy.ROLLBACK:
            # Implement rollback logic
            await self._rollback_workflow(step_id, error)

    async def _retry_step(self, step_id: str, original_error: Exception) -> None:
        """Retry failed step execution."""
        max_retries = self.workflow_config.retry_attempts
        retry_delay = self.workflow_config.retry_delay

        for attempt in range(max_retries):
            self.workflow_logger.info(
                f"Retrying step {step_id}, attempt {attempt + 1}/{max_retries}")

            try:
                await asyncio.sleep(retry_delay)
                step = self.child_steps[step_id]
                result = await step.execute()

                # Remove from failed steps if successful
                self.failed_steps.discard(step_id)
                self.completed_steps.add(step_id)

                await self._propagate_step_data(step_id, result)

                self.workflow_logger.info(
                    f"Step {step_id} succeeded on retry attempt {attempt + 1}")
                return

            except Exception as e:
                self.workflow_logger.warning(
                    f"Step {step_id} retry attempt {attempt + 1} failed: {e}", exc_info=True)
                if attempt == max_retries - 1:
                    # Final attempt failed
                    raise original_error

    async def _rollback_workflow(self, failed_step_id: str, error: Exception) -> None:
        """Rollback workflow state after step failure."""
        self.workflow_logger.warning(
            f"Rolling back workflow due to step {failed_step_id} failure")

        # This is a placeholder for rollback logic
        # In a full implementation, this would:
        # 1. Undo changes made by completed steps
        # 2. Reset data units to previous states
        # 3. Clean up resources

        raise error

    async def shutdown(self) -> None:
        """Shutdown the workflow and cleanup resources."""
        self.workflow_logger.info(f"Shutting down workflow: {self.name}")

        # Shutdown all child steps
        for step_id, step in self.child_steps.items():
            try:
                await step.shutdown()
            except Exception as e:
                self.workflow_logger.error(
                    f"Error shutting down step {step_id}: {e}")

        # Stop all links
        for link_id, link in self.step_links.items():
            try:
                await link.stop()
            except Exception as e:
                self.workflow_logger.error(
                    f"Error stopping link {link_id}: {e}")

        # Configuration cache is managed by the framework automatically

        # Shutdown as Step
        await super().shutdown()

        self.workflow_logger.info(f"Workflow {self.name} shutdown complete")

    async def execute_distributed(
        self,
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute workflow using distributed execution if available.

        This method automatically detects if the configured executor supports
        distributed execution and uses it when available, falling back to
        standard execution otherwise.

        Args:
            input_data: Input data for workflow execution
            **kwargs: Additional execution parameters

        Returns:
            Workflow execution result
        """
        # Import here to avoid circular imports
        from nanobrain.core.executor import ParslExecutor

        # Check if executor supports distributed execution
        if (hasattr(self.executor, 'execute_workflow_distributed') and
            isinstance(self.executor, ParslExecutor)):

            # Get workflow configuration path
            config_path = getattr(self.config, 'source_path', None)
            if not config_path:
                # Fall back to standard execution if no config path available
                self.workflow_logger.warning(
                    "No source configuration path available for distributed execution, falling back to standard execution"
                )
                return await self.process(input_data, **kwargs)

            # Execute using distributed method
            self.workflow_logger.info(f"Executing workflow {self.name} using distributed execution")
            return await self.executor.execute_workflow_distributed(config_path, input_data)
        else:
            # Fall back to standard execution
            self.workflow_logger.debug("Executor does not support distributed execution, using standard execution")
            return await self.process(input_data, **kwargs)

    def _validate_workflow_integrity(self) -> None:
        """
        FAIL-FAST: Validate workflow integrity before execution.

        Checks for:
        - Broken step chains (steps with no data sources)
        - Circular dependencies
        - Unreachable steps
        - Invalid step configurations
        """
        from .component_base import ComponentConfigurationError

        try:
            self.workflow_logger.info("🔍 FAIL-FAST: Validating workflow integrity...")

            # Check 1: Validate all steps have proper data flow
            self._validate_step_data_flow()

            # Check 2: Detect circular dependencies
            self._validate_no_circular_dependencies()

            # Check 3: Find unreachable steps
            self._validate_no_unreachable_steps()

            # Check 4: Validate step configurations
            self._validate_step_configurations()

            self.workflow_logger.info("✅ FAIL-FAST: Workflow integrity validation passed")

        except Exception as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: Workflow integrity validation failed: {e}. "
                f"Fix the workflow structure before execution."
            ) from e

    def _validate_step_data_flow(self) -> None:
        """Validate that all steps have proper data sources and consumers."""
        issues = []

        for step_name, step_config in self.workflow_config.steps.items():
            # Check if step has input data units that need sources
            if hasattr(step_config, 'input_data_units'):
                for input_name, input_config in step_config.input_data_units.items():
                    if not self._has_data_source_for_step_input(step_name, input_name):
                        issues.append(
                            f"Step '{step_name}' input '{input_name}' has no data source. "
                            f"Add a link from another step's output or provide initial data."
                        )

            # Check if step has output data units that are consumed
            if hasattr(step_config, 'output_data_units'):
                for output_name, output_config in step_config.output_data_units.items():
                    if not self._has_data_consumer_for_step_output(step_name, output_name):
                        issues.append(
                            f"Step '{step_name}' output '{output_name}' has no consumer. "
                            f"Add a link to another step's input or mark as final output."
                        )

        if issues:
            raise ValueError("Data flow issues found:\n" + "\n".join(f"  - {issue}" for issue in issues))

    def _validate_no_circular_dependencies(self) -> None:
        """Detect circular dependencies in the workflow graph."""
        # Build dependency graph
        dependencies = {}
        for step_name in self.workflow_config.steps.keys():
            dependencies[step_name] = self._get_step_dependencies(step_name)

        # Detect cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for neighbor in dependencies.get(node, []):
                if has_cycle(neighbor):
                    return True

            rec_stack.remove(node)
            return False

        for step_name in dependencies:
            if step_name not in visited:
                if has_cycle(step_name):
                    raise ValueError(
                        f"Circular dependency detected involving step '{step_name}'. "
                        f"This would cause infinite loops. Review step dependencies."
                    )

    def _validate_no_unreachable_steps(self) -> None:
        """Find steps that can never be executed."""
        # Find entry points (steps with no dependencies or external inputs)
        entry_points = []
        for step_name in self.workflow_config.steps.keys():
            dependencies = self._get_step_dependencies(step_name)
            if not dependencies:
                entry_points.append(step_name)

        if not entry_points:
            raise ValueError(
                "No entry points found. At least one step must have external input or no dependencies."
            )

        # Find reachable steps from entry points
        reachable = set()

        def mark_reachable(step_name):
            if step_name in reachable:
                return
            reachable.add(step_name)
            for dependent in self._get_step_dependents(step_name):
                mark_reachable(dependent)

        for entry_point in entry_points:
            mark_reachable(entry_point)

        # Find unreachable steps
        all_steps = set(self.workflow_config.steps.keys())
        unreachable = all_steps - reachable

        if unreachable:
            raise ValueError(
                f"Unreachable steps found: {list(unreachable)}. "
                f"These steps will never execute. Fix the workflow graph."
            )

    def _validate_step_configurations(self) -> None:
        """Validate individual step configurations."""
        for step_name, step_config in self.workflow_config.steps.items():
            # Check if step class exists
            if hasattr(step_config, 'class'):
                try:
                    from .component_base import import_class_from_path
                    step_class = import_class_from_path(getattr(step_config, 'class'))
                except ImportError as e:
                    raise ValueError(
                        f"Step '{step_name}' class '{getattr(step_config, 'class')}' not found: {e}"
                    )

                # Check if step class has required methods
                if not hasattr(step_class, 'process'):
                    raise ValueError(
                        f"Step '{step_name}' class '{getattr(step_config, 'class')}' missing 'process' method"
                    )

    def _has_data_source_for_step_input(self, step_name: str, input_name: str) -> bool:
        """Check if a step input has a data source."""
        target_reference = f"{step_name}.{input_name}"

        # Check resolved links for this input
        if hasattr(self, '_resolved_components') and 'links' in self._resolved_components:
            for link_id, link_instance in self._resolved_components['links'].items():
                if hasattr(link_instance, 'target'):
                    # Get target name from the link instance
                    target_name = getattr(link_instance.target, 'name', str(link_instance.target))
                    if target_name == target_reference:
                        return True

        # Fallback: Check workflow config links
        for link_id, link_config in self.workflow_config.links.items():
            if hasattr(link_config, 'config') and hasattr(link_config.config, 'target'):
                if link_config.config.target == target_reference:
                    return True

        return False

    def _has_data_consumer_for_step_output(self, step_name: str, output_name: str) -> bool:
        """Check if a step output has a consumer."""
        source_reference = f"{step_name}.{output_name}"

        # Check resolved links for this output
        if hasattr(self, '_resolved_components') and 'links' in self._resolved_components:
            for link_id, link_instance in self._resolved_components['links'].items():
                if hasattr(link_instance, 'source'):
                    # Get source name from the link instance
                    source_name = getattr(link_instance.source, 'name', str(link_instance.source))
                    if source_name == source_reference:
                        return True

        # Fallback: Check workflow config links
        for link_id, link_config in self.workflow_config.links.items():
            if hasattr(link_config, 'config') and hasattr(link_config.config, 'source'):
                if link_config.config.source == source_reference:
                    return True

        return False

    def _get_step_dependencies(self, step_name: str) -> List[str]:
        """Get list of steps that this step depends on."""
        dependencies = []
        for link_config in self.workflow_config.links:
            if (hasattr(link_config, 'target') and
                link_config.target.startswith(f"{step_name}.")):
                source_step = link_config.source.split('.')[0]
                if source_step != step_name:  # Avoid self-dependencies
                    dependencies.append(source_step)
        return dependencies

    def _get_step_dependents(self, step_name: str) -> List[str]:
        """Get list of steps that depend on this step."""
        dependents = []
        for link_config in self.workflow_config.links:
            if (hasattr(link_config, 'source') and
                link_config.source.startswith(f"{step_name}.")):
                target_step = link_config.target.split('.')[0]
                if target_step != step_name:  # Avoid self-dependencies
                    dependents.append(target_step)
        return dependents


# Utility functions for workflow creation


async def create_workflow(config: Union[WorkflowConfig, Dict[str, Any], str], **kwargs) -> Workflow:
    """
    Create and initialize a workflow.

    Args:
        config: Workflow configuration (WorkflowConfig, dict, or path to YAML file)
        **kwargs: Additional arguments passed to workflow initialization

    Returns:
        Initialized Workflow instance
    """
    if isinstance(config, str):
        # Load from file path
        import tempfile
        import yaml

        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Create temporary config file for from_config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_config_path = f.name
        workflow = Workflow.from_config(temp_config_path)
        import os
        os.unlink(temp_config_path)
    else:
        workflow = Workflow.from_config(config)

    await workflow.initialize()

    return workflow


async def create_workflow_from_config(config: Union[str, Dict, WorkflowConfig]) -> 'Workflow':
    """
    Create and initialize a workflow from configuration.

    Args:
        config: Workflow configuration (WorkflowConfig, dict, or path to YAML file)

    Returns:
        Initialized Workflow instance
    """
    if isinstance(config, str):
        # Load from file path
        import tempfile
        import yaml

        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Create temporary config file for from_config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_config_path = f.name
        workflow = Workflow.from_config(temp_config_path)
        import os
        os.unlink(temp_config_path)
    elif isinstance(config, dict):
        # Dict config - need to save to file first
        import tempfile
        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config, f)
            temp_config_path = f.name
        workflow = Workflow.from_config(temp_config_path)
        import os
        os.unlink(temp_config_path)
    elif isinstance(config, WorkflowConfig):
        # Config object - need to save to file first
        import tempfile
        import yaml
        config_dict = config.to_dict() if hasattr(
            config, 'to_dict') else config.__dict__
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_config_path = f.name
        workflow = Workflow.from_config(temp_config_path)
        import os
        os.unlink(temp_config_path)
    else:
        workflow = Workflow.from_config(config)

    await workflow.initialize()

    return workflow


# Factory function for creating workflows
async def create_workflow(config: Union[WorkflowConfig, Dict[str, Any], str], **kwargs) -> Workflow:
    """
    Create and initialize a workflow.

    Args:
        config: Workflow configuration (WorkflowConfig, dict, or path to YAML file)
        **kwargs: Additional arguments passed to Workflow constructor

    Returns:
        Initialized Workflow instance
    """
    if isinstance(config, str):
        # Already a file path, use directly
        workflow = Workflow.from_config(config)
    elif isinstance(config, dict):
        # Create temporary YAML file for dict config
        import tempfile
        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config, f)
            temp_config_path = f.name
        workflow = Workflow.from_config(temp_config_path)
        import os
        os.unlink(temp_config_path)
    elif isinstance(config, WorkflowConfig):
        # Config object - need to save to file first
        import tempfile
        import yaml
        config_dict = config.to_dict() if hasattr(
            config, 'to_dict') else config.__dict__
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_config_path = f.name
        workflow = Workflow.from_config(temp_config_path)
        import os
        os.unlink(temp_config_path)
    else:
        workflow = Workflow.from_config(config)

    await workflow.initialize()

    return workflow
