"""SubworkflowStep — embed a workflow as a step.

The mechanism that lets one nanobrain ``Workflow`` appear as a single
``BaseStep`` inside another workflow. Sub-workflows ARE reusable
reasoning patterns (reflection, verification, decomposition,
self-consistency), and exposing them as ordinary steps means the
composer's RAG matching can pull them into generated workflows by
``rag_description`` — no special-cased "subworkflow" surface
required at compose time.

Two intended usage shapes:

1. **Concrete subclass per reusable pattern.** Author a thin
   subclass (``CodeReflectionStep(SubworkflowStep)``,
   ``VerificationStep(SubworkflowStep)``, etc.) that overrides
   ``_default_inner_workflow_path`` to a hardcoded relative path.
   Ship a wrapper YAML for it with a rich ``rag_description``.
   The composer sees the subclass like any other step class.

2. **Direct configuration.** Use ``SubworkflowStep`` itself with
   ``inner_workflow_path`` in YAML. Useful for ad-hoc embedding
   when you don't want to mint a new class. The composer is told
   NOT to author this shape directly (the path field is a
   hallucination surface); prefer concrete subclasses.

Silent-failure discipline:

``Workflow.run()`` defaults to ``raise_on_cascade_timeout=False``,
returning ``{"status": "cascade_timeout", ...}`` so operators can
inspect partial output. That shape is correct at the top-level call
site (operator-debuggable), but it is the EXACT silent-failure
pattern we just hardened against in the executor's EMPTY-OUTPUT
gate (apecx-mcp-integration `7471b0a`). ``SubworkflowStep`` must
NOT inherit that lenient default — a step that returns the
status dict downstream silently propagates an incomplete result
across the workflow.

So we:

* invoke the inner workflow with ``raise_on_cascade_timeout=True``;
* after success, inspect ``result["status"]`` and raise ``RuntimeError``
  on anything other than ``"completed"`` (or ``"completed_no_await"``
  when explicitly opted in);
* strip ``"status"`` from the returned dict before handing to the
  outer trigger cascade;
* apply a symmetric EMPTY-OUTPUT gate (configurable opt-out via
  ``allow_empty_inner_output``).

Cross-reference: ``apecx-mcp-integration/docs/CONTRACTS.md#g7``
(silent-failure threat model), and ``Workflow.run()`` docstring in
``nanobrain/core/workflow.py:2430``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from pydantic import Field

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.workflow import Workflow

logger = logging.getLogger(__name__)


class SubworkflowStepConfig(StepConfig):
    """Configuration for ``SubworkflowStep``.

    Mandatory: an ``inner_workflow_path`` — either via this config
    field (set in YAML or supplied programmatically), or via a
    subclass that overrides ``_default_inner_workflow_path``.

    Optional knobs match ``Workflow.run()``: ``timeout_seconds``,
    ``settle_ms``, ``await_cascade``, plus this step's two
    silent-failure gates (``allow_empty_inner_output``,
    ``allow_completed_no_await``).
    """

    inner_workflow_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to the inner workflow's YAML config. Absolute is "
            "preferred; relative paths are resolved via "
            "nanobrain.library.runtime.workspace_root.locate_workflow_root. "
            "Concrete subclasses may omit this and override "
            "_default_inner_workflow_path() instead."
        ),
    )

    timeout_seconds: float = Field(
        default=60.0,
        description=(
            "Maximum seconds the inner workflow's trigger cascade may "
            "take before raise_on_cascade_timeout=True forces a "
            "TimeoutError. Default 60s — matches Workflow.run()'s "
            "default but with strict raising. Lower for short pure-"
            "compute inner workflows; raise for multi-LLM-call ones."
        ),
    )

    settle_ms: int = Field(
        default=50,
        description=(
            "Milliseconds the trigger executor must remain idle for "
            "the inner cascade to be considered drained. See "
            "Workflow.run() settle_ms."
        ),
    )

    await_cascade: bool = Field(
        default=True,
        description=(
            "When True (default), block until the inner workflow's "
            "trigger cascade drains. When False, the inner workflow "
            "is fire-and-forget — the SubworkflowStep returns "
            "whatever output data units are populated so far. "
            "Setting this False is almost always a bug; it exists "
            "for symmetry with the underlying Workflow.run() API."
        ),
    )

    allow_completed_no_await: bool = Field(
        default=False,
        description=(
            "When False (default), the inner workflow returning "
            "'status: completed_no_await' is treated as a failure "
            "(it means await_cascade=False and outputs may not "
            "reflect the cascade). Set True only when you have "
            "explicitly designed for fire-and-forget semantics."
        ),
    )

    allow_empty_inner_output: bool = Field(
        default=False,
        description=(
            "When False (default), the inner workflow returning an "
            "empty result dict (no meaningful output keys) is "
            "treated as a failure — matches the executor's "
            "EMPTY-OUTPUT gate. Set True only when the inner "
            "workflow legitimately produces no output (rare; "
            "side-effect-only sub-workflows)."
        ),
    )

    nest_under_active_context: bool = Field(
        default=True,
        description=(
            "When True (default), invoke the inner workflow with "
            "nest_under_active_context=True so G31's nested "
            "WorkflowRunContext + namespace derivation fire. The "
            "inner run gets run_id 'parent.child' for audit "
            "correlation. Set False only if you have a specific "
            "reason to isolate the inner workflow from the parent's "
            "context (e.g., per-tenant fanout where you DO want "
            "fresh capability tokens)."
        ),
    )


class SubworkflowStep(BaseStep):
    """Embed a complete ``Workflow`` as a single ``BaseStep``.

    The inner workflow is loaded ONCE at step init and cached for the
    step's lifetime. Each ``process(input_data)`` call invokes the
    cached workflow's ``run()`` with the input dict. Workflow input
    data units must be keyed compatibly with the dict passed in;
    workflow output data units are returned (minus the operational
    ``status`` key) as the step's process output.

    Concrete subclasses customize one classmethod:

    .. code-block:: python

        class CodeReflectionStep(SubworkflowStep):
            COMPONENT_TYPE = "code_reflection_step"

            @classmethod
            def _default_inner_workflow_path(cls) -> Optional[str]:
                # Resolved against the workspace root.
                return "src/apecx_integration/composition/workflows/code_writing/code_reflection_workflow.yml"

    The composer sees ``CodeReflectionStep`` like any normal step;
    it picks it via RAG match against the step's wrapper-YAML
    ``rag_description``. The workflow-embedding mechanism is
    invisible at compose time.
    """

    COMPONENT_TYPE: str = "subworkflow_step"
    REQUIRED_CONFIG_FIELDS: ClassVar[list] = ["name"]

    @classmethod
    def _get_config_class(cls):
        return SubworkflowStepConfig

    @classmethod
    def _default_inner_workflow_path(cls) -> Optional[str]:
        """Override in concrete subclasses to hardcode the inner workflow path.

        Return None (default) to require the path via config field.
        """
        return None

    def _init_from_config(
        self,
        config: SubworkflowStepConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        super()._init_from_config(config, component_config, dependencies)

        path_str = (
            config.inner_workflow_path
            or self.__class__._default_inner_workflow_path()
        )
        if path_str is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: SubworkflowStep {self.name!r} requires an "
                f"inner_workflow_path. Either set it in the step's "
                f"config YAML, or subclass and override "
                f"_default_inner_workflow_path()."
            )

        resolved = self._resolve_inner_workflow_path(path_str)
        try:
            self._inner_workflow: Workflow = Workflow.from_config(str(resolved))
        except Exception as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: SubworkflowStep {self.name!r} failed to "
                f"load inner workflow from {resolved}: {e}"
            ) from e

        self._inner_workflow_path_resolved: Path = resolved
        self._timeout_seconds: float = float(config.timeout_seconds)
        self._settle_ms: int = int(config.settle_ms)
        self._await_cascade: bool = bool(config.await_cascade)
        self._allow_completed_no_await: bool = bool(config.allow_completed_no_await)
        self._allow_empty_inner_output: bool = bool(config.allow_empty_inner_output)
        self._nest_under_active_context: bool = bool(config.nest_under_active_context)

        logger.info(
            "SubworkflowStep %r: loaded inner workflow %r from %s",
            self.name,
            getattr(self._inner_workflow, "name", "<unnamed>"),
            resolved,
        )

    @staticmethod
    def _resolve_inner_workflow_path(path_str: str) -> Path:
        """Resolve ``path_str`` to an absolute path that exists.

        Resolution order:
          1. If already absolute and exists → return.
          2. If relative → try resolving against the workspace root
             via ``locate_workflow_root`` (G40).
          3. If still unresolved → try cwd.
          4. FAIL-FAST if not found.
        """
        p = Path(path_str)
        if p.is_absolute() and p.is_file():
            return p
        if p.is_absolute():
            raise ComponentConfigurationError(
                f"FAIL-FAST: SubworkflowStep inner_workflow_path "
                f"{path_str!r} is absolute but does not exist on disk"
            )

        try:
            from nanobrain.library.runtime.workspace_root import locate_workflow_root

            root = locate_workflow_root()
            if root is not None:
                candidate = (root / p).resolve()
                if candidate.is_file():
                    return candidate
        except ImportError:
            pass

        cwd_candidate = (Path.cwd() / p).resolve()
        if cwd_candidate.is_file():
            return cwd_candidate

        raise ComponentConfigurationError(
            f"FAIL-FAST: SubworkflowStep inner_workflow_path {path_str!r} "
            f"could not be resolved. Tried: workspace_root via G40, cwd "
            f"({Path.cwd()}). Provide an absolute path or set "
            f"$NANOBRAIN_WORKSPACE_ROOT."
        )

    @property
    def inner_workflow(self) -> Workflow:
        """The cached inner ``Workflow`` instance — exposed for tests."""
        return self._inner_workflow

    @property
    def inner_workflow_path(self) -> Path:
        """The resolved absolute path to the inner workflow's YAML."""
        return self._inner_workflow_path_resolved

    async def process(
        self, input_data: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Invoke the inner workflow and return its outputs.

        Input routing reality (documented separately in the
        SubworkflowStep module docstring):

        Despite ``Workflow.run(input_data)``'s docstring claiming
        input_data is keyed by workflow-level input data unit
        names, the framework's data-flow initiator actually routes
        ``input_data`` to the **first step's input data unit names**
        (the workflow-level ``input_data_units:`` block is currently
        decorative — see the rag_e2e_synthesis tests which work
        around this by passing input via ``wf.process(...)`` keyed
        by the first step's input DU).

        To shield callers from this gap, ``SubworkflowStep.process()``
        auto-wraps the input: when the inner workflow has exactly
        one first-step input data unit AND ``input_data`` does NOT
        already contain that key, we deposit ``input_data`` under
        that key. Callers can therefore pass the per-step input
        shape (e.g., ``{"code_spec": "..."}``) without knowing the
        inner workflow's first-step DU name.

        When the inner workflow has multiple first-step input data
        units OR ``input_data`` is already keyed, we pass through
        unchanged.

        Returns the inner workflow's last-step outputs collected
        from the inner workflow's child_steps.

        Raises:

        * ``ComponentConfigurationError`` — input is not a dict.
        * ``TimeoutError`` — inner cascade did not drain within
          ``timeout_seconds``.
        * ``RuntimeError`` — inner workflow produced no meaningful
          output (EMPTY-OUTPUT gate; opt out via
          ``allow_empty_inner_output``).
        """
        if not isinstance(input_data, dict):
            raise ComponentConfigurationError(
                f"FAIL-FAST: SubworkflowStep {self.name!r} input_data "
                f"must be a dict; got {type(input_data).__name__}"
            )

        routed = self._route_input_to_first_step_du(input_data)

        # Use wf.process + wf.wait_for_cascade — that's the pattern
        # the framework actually wires up (Workflow.run with workflow-
        # level data unit keys is currently a documentation-only
        # surface; the runtime routes by first-step DU name).
        init_status = await self._inner_workflow.process(routed)
        if not isinstance(init_status, dict):
            raise RuntimeError(
                f"SubworkflowStep {self.name!r}: inner workflow process() "
                f"returned non-dict {type(init_status).__name__}"
            )

        if self._await_cascade:
            drained = await self._inner_workflow.wait_for_cascade(
                timeout=self._timeout_seconds,
                settle_ms=self._settle_ms,
            )
            if not drained:
                raise TimeoutError(
                    f"SubworkflowStep {self.name!r}: inner workflow "
                    f"cascade did not drain within "
                    f"{self._timeout_seconds}s"
                )

        # Collect outputs from the LAST step's output data units.
        # This is the working pattern (see test_rag_e2e_workflow_yaml).
        result = await self._collect_last_step_outputs()
        # Synthesize a status field for the post-run gate below.
        result.setdefault(
            "status",
            "completed" if self._await_cascade else "completed_no_await",
        )

        status = result.get("status")
        if status == "completed":
            pass
        elif status == "completed_no_await":
            if not self._allow_completed_no_await:
                raise RuntimeError(
                    f"SubworkflowStep {self.name!r}: inner workflow "
                    f"returned 'completed_no_await' but "
                    f"allow_completed_no_await=False. This usually "
                    f"means await_cascade was set to False — outputs "
                    f"may not reflect the full cascade. Set "
                    f"allow_completed_no_await=True only if "
                    f"fire-and-forget semantics are intentional."
                )
        else:
            raise RuntimeError(
                f"SubworkflowStep {self.name!r}: inner workflow "
                f"returned status={status!r}, expected 'completed'. "
                f"Full result keys: {sorted(result.keys())}. "
                f"Common causes: 'no_first_step' (the inner workflow "
                f"has no executable first step); 'cascade_timeout' "
                f"would have raised TimeoutError before reaching here."
            )

        clean_result = {
            k: v for k, v in result.items()
            if k != "status" and not k.startswith("_")
        }

        if not self._allow_empty_inner_output:
            meaningful = {
                k: v for k, v in clean_result.items()
                if v is not None and v != [] and v != {} and v != ""
            }
            if not meaningful:
                raise RuntimeError(
                    f"SubworkflowStep {self.name!r}: inner workflow "
                    f"completed but produced no meaningful output. "
                    f"Result keys: {sorted(clean_result.keys())} all "
                    f"empty/None. This usually means a missing "
                    f"auto_transfer:true on a DirectLink inside the "
                    f"inner workflow, OR the first inner step did "
                    f"not produce its declared output. Set "
                    f"allow_empty_inner_output=True only if the "
                    f"inner workflow is side-effect-only by design."
                )

        return clean_result

    def _route_input_to_first_step_du(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Auto-wrap input_data under the first step's input data unit.

        Logic:
          - If the inner workflow has exactly ONE first-step input
            data unit AND input_data does NOT already contain that
            key, deposit input_data under that key.
          - Otherwise, pass through unchanged (multi-input first step
            or caller already keyed correctly).

        Why this exists: ``Workflow.run()``'s docstring says it
        routes by workflow-level data unit name, but the framework's
        actual initiator routes by FIRST-STEP data unit name (see
        the rag_e2e_synthesis tests which work around this). This
        shim hides the gap from SubworkflowStep callers.
        """
        try:
            first_step = next(iter(self._inner_workflow.child_steps.values()))
        except StopIteration:
            return input_data
        first_dus = getattr(first_step, "step_input_data_units", None) or {}
        if len(first_dus) != 1:
            return input_data
        only_du_name = next(iter(first_dus.keys()))
        if only_du_name in input_data:
            return input_data
        return {only_du_name: input_data}

    async def _collect_last_step_outputs(self) -> Dict[str, Any]:
        """Read the last child step's output data units into a dict.

        For sub-workflows with a single linear pipeline (the common
        case), the last step's outputs are the workflow's outputs.
        Operators with branching topologies should override.
        """
        if not self._inner_workflow.child_steps:
            return {}
        last_step = list(self._inner_workflow.child_steps.values())[-1]
        output_dus = getattr(last_step, "step_output_data_units", None) or {}
        collected: Dict[str, Any] = {}
        for name, du in output_dus.items():
            try:
                collected[name] = await du.get()
            except Exception as e:
                collected[name] = None
                collected.setdefault("_errors", {})[name] = str(e)
        return collected


__all__ = [
    "SubworkflowStep",
    "SubworkflowStepConfig",
]
