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
from nanobrain.core.step_events import StepEvent, subscribe_to_step_events
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
            "_default_inner_workflow_path() instead. Mutually exclusive "
            "with inner_workflow_builder."
        ),
    )

    inner_workflow_builder: Optional[str] = Field(
        default=None,
        description=(
            "Dotted-path string ('pkg.mod.func' or 'pkg.mod:func') to a "
            "NO-ARG callable that returns a fully-loaded Workflow "
            "instance. This is the seam for embedding a workflow that "
            "exists only as a programmatic builder (e.g. the lightweight "
            "WorkflowBuilder's `build_*` catalog entry-points) rather "
            "than as a static YAML on disk. The callable is resolved + "
            "invoked ONCE at step init and the resulting Workflow is "
            "cached for the step's lifetime — identical lifecycle to the "
            "inner_workflow_path branch. Mutually exclusive with "
            "inner_workflow_path. Concrete subclasses may omit this and "
            "override _default_inner_workflow_builder() instead."
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

    @classmethod
    def _default_inner_workflow_builder(cls) -> Optional[str]:
        """Override in concrete subclasses to hardcode a builder dotted-path.

        Return None (default) to require the builder via config field
        (or to use the path branch instead). Symmetric with
        ``_default_inner_workflow_path``.
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
        builder_str = (
            config.inner_workflow_builder
            or self.__class__._default_inner_workflow_builder()
        )

        if path_str is not None and builder_str is not None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: SubworkflowStep {self.name!r} declares BOTH "
                f"inner_workflow_path ({path_str!r}) and "
                f"inner_workflow_builder ({builder_str!r}). They are "
                f"mutually exclusive — the inner workflow comes from "
                f"exactly one source. Pick one."
            )
        if path_str is None and builder_str is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: SubworkflowStep {self.name!r} requires an "
                f"inner_workflow_path OR an inner_workflow_builder. "
                f"Either set one in the step's config YAML, or subclass "
                f"and override _default_inner_workflow_path() / "
                f"_default_inner_workflow_builder()."
            )

        if builder_str is not None:
            self._inner_workflow: Workflow = self._build_inner_workflow(builder_str)
            self._inner_workflow_path_resolved: Optional[Path] = None
            logger.info(
                "SubworkflowStep %r: built inner workflow %r via builder %s",
                self.name,
                getattr(self._inner_workflow, "name", "<unnamed>"),
                builder_str,
            )
            self._init_runtime_knobs(config)
            return

        resolved = self._resolve_inner_workflow_path(path_str)
        try:
            self._inner_workflow = Workflow.from_config(str(resolved))
        except Exception as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: SubworkflowStep {self.name!r} failed to "
                f"load inner workflow from {resolved}: {e}"
            ) from e

        self._inner_workflow_path_resolved = resolved
        logger.info(
            "SubworkflowStep %r: loaded inner workflow %r from %s",
            self.name,
            getattr(self._inner_workflow, "name", "<unnamed>"),
            resolved,
        )
        self._init_runtime_knobs(config)

    def _init_runtime_knobs(self, config: SubworkflowStepConfig) -> None:
        """Bind the run()-passthrough + silent-failure-gate knobs.

        Shared by both the path and builder init branches so the
        EMPTY-OUTPUT / status / await-cascade discipline is byte-for-byte
        identical regardless of where the inner workflow came from.
        """
        self._timeout_seconds: float = float(config.timeout_seconds)
        self._settle_ms: int = int(config.settle_ms)
        self._await_cascade: bool = bool(config.await_cascade)
        self._allow_completed_no_await: bool = bool(config.allow_completed_no_await)
        self._allow_empty_inner_output: bool = bool(config.allow_empty_inner_output)
        self._nest_under_active_context: bool = bool(config.nest_under_active_context)

    def _build_inner_workflow(self, builder_spec: str) -> Workflow:
        """Resolve a dotted-path no-arg builder callable + invoke it.

        Mirrors the framework's existing dotted-path resolution
        convention (G22 ``target_workflow`` in
        ``library/runtime/entry_triggers.py``). The callable MUST take
        no required arguments and return a ``Workflow`` instance — the
        same object ``Workflow.from_config`` would yield, so every
        downstream silent-failure gate is preserved unchanged.

        FAIL-FAST on: non-dotted spec, unimportable module, missing
        attribute, a class (from_config discipline forbids ad-hoc class
        construction here), a non-callable, a builder that raises, or a
        builder that returns something other than a Workflow.
        """
        import importlib

        if not isinstance(builder_spec, str) or "." not in builder_spec:
            raise ComponentConfigurationError(
                f"FAIL-FAST: SubworkflowStep {self.name!r} "
                f"inner_workflow_builder must be a dotted-path string "
                f"like 'pkg.mod.build_func' or 'pkg.mod:build_func'; got "
                f"{builder_spec!r}"
            )
        module_path, _, attr_path = builder_spec.partition(":")
        if not attr_path:
            module_path, _, attr_path = builder_spec.rpartition(".")
        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise ComponentConfigurationError(
                f"FAIL-FAST: SubworkflowStep {self.name!r} "
                f"inner_workflow_builder module {module_path!r} not "
                f"importable: {exc}"
            ) from exc
        obj: Any = mod
        for part in attr_path.split("."):
            try:
                obj = getattr(obj, part)
            except AttributeError as exc:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: SubworkflowStep {self.name!r} "
                    f"inner_workflow_builder attribute {attr_path!r} not "
                    f"found on module {module_path!r}: {exc}"
                ) from exc

        if isinstance(obj, type):
            raise ComponentConfigurationError(
                f"FAIL-FAST: SubworkflowStep {self.name!r} "
                f"inner_workflow_builder {builder_spec!r} resolved to a "
                f"class ({obj.__name__}); expected a no-arg callable that "
                f"RETURNS a Workflow instance, not a class. The "
                f"from_config discipline forbids ad-hoc class "
                f"instantiation here."
            )
        if not callable(obj):
            raise ComponentConfigurationError(
                f"FAIL-FAST: SubworkflowStep {self.name!r} "
                f"inner_workflow_builder {builder_spec!r} resolved to "
                f"{type(obj).__name__}, which is not callable. Expected a "
                f"no-arg callable returning a Workflow."
            )

        try:
            inner = obj()
        except Exception as exc:
            raise ComponentConfigurationError(
                f"FAIL-FAST: SubworkflowStep {self.name!r} "
                f"inner_workflow_builder {builder_spec!r} raised when "
                f"invoked: {exc}"
            ) from exc

        if not isinstance(inner, Workflow):
            raise ComponentConfigurationError(
                f"FAIL-FAST: SubworkflowStep {self.name!r} "
                f"inner_workflow_builder {builder_spec!r} returned "
                f"{type(inner).__name__}, expected a Workflow instance. "
                f"A lightweight builder must return `builder.load()` (a "
                f"Workflow), not the WorkflowBuilder itself or a config "
                f"dict."
            )
        return inner

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
    def inner_workflow_path(self) -> Optional[Path]:
        """The resolved absolute path to the inner workflow's YAML.

        ``None`` when the inner workflow came from an
        ``inner_workflow_builder`` callable rather than a YAML path.
        """
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

        # CACHED-RE-RUN STALENESS FIX (2026-06-13). The inner workflow is built ONCE
        # and reused across process() calls. The poll loop below waits until the last
        # step's output DU is "populated" — but on a RE-RUN that DU still holds the
        # PRIOR run's output, so the poll returns INSTANTLY with stale data before the
        # new cascade re-populates it. In a long-lived process (e.g. the MCP server,
        # which caches workflows per-process), the 2nd+ call therefore silently returns
        # the FIRST call's result for a DIFFERENT input — a severe silent correctness
        # bug. Clear the last-step output DUs first so "populated" means THIS run.
        # (Detected: viral_epitope_evidence_review returned influenza's sequence
        # conservation for a subsequent HIV query in the same process.)
        await self._clear_inner_last_step_outputs()

        routed = self._route_input_to_first_step_du(input_data)

        # FAST inner-failure detection (G37). When an inner step RAISES, the
        # trigger executor SWALLOWS the exception (G127 — Workflow.run does not
        # propagate it), so the inner output data unit never populates and the
        # poll loop below would otherwise wait the FULL timeout_seconds before
        # giving up — an N-minute hang for a failure that happened in seconds.
        # Instead, subscribe to the inner cascade's step_failed events: the
        # inner step tasks are create_task-spawned transitively within THIS
        # task's context, so they inherit the contextvar-based subscriber and
        # publish_step_event reaches `_capture_inner_failure`. The poll loop
        # checks the capture each iteration and re-raises the inner step's REAL
        # exception immediately, so the caller (e.g. a degrade-loud outer step)
        # sees "inner step X failed: <real reason>" in seconds, not a generic
        # timeout. This is a general nested-failure robustness win, not specific
        # to any one inner workflow. Source: 2026-06-13 BUG A.
        inner_step_names = set(self._inner_workflow.child_steps.keys())
        inner_failures: list[StepEvent] = []

        def _capture_inner_failure(event: StepEvent) -> None:
            if (
                event.event_type == "step_failed"
                and event.step_name in inner_step_names
            ):
                inner_failures.append(event)

        with subscribe_to_step_events(_capture_inner_failure):
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

            # A first step that already failed during process()'s settle is
            # surfaced immediately (before we even enter the poll loop).
            self._raise_if_inner_step_failed(inner_failures)

            if self._await_cascade:
                # Cannot use inner_workflow.wait_for_cascade here when this
                # step is itself running inside another cascade: the
                # framework's AsyncTriggerExecutor is a process-wide
                # singleton (see workflow.py:wait_for_cascade), so the
                # inner drain-detection sees our OWN task in the queue
                # and waits indefinitely for itself to finish. Classic
                # shared-executor re-entrance deadlock.
                #
                # Workaround: poll the inner workflow's last step's output
                # data units until they're populated. The asyncio.sleep
                # yields control so the inner cascade's tasks can actually
                # run. When fan-out lands (inner cascade with multiple last
                # steps), this loop needs the polling-of-each-output
                # extension; for the linear case this is sufficient.
                #
                # Source: 2026-05-12 nested-cascade deadlock investigation.
                await self._poll_inner_workflow_until_drained(inner_failures)

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

        Two shapes the caller might use:
          1. **Direct payload** — ``{"code_spec": "...", "function_name": "..."}``
             (test or programmatic caller).
          2. **Framework-wrapped** — the cascade-driven path. When the
             outer workflow's framework invokes ``SubworkflowStep.process``,
             it constructs ``input_data`` as ``{<my_own_input_du_name>:
             <payload>}``. We must unwrap that BEFORE re-wrapping
             for the inner workflow's first-step input DU — otherwise
             we double-wrap and the inner step receives ``{<my_du>:
             <real_payload>}`` instead of just ``<real_payload>``.

        Algorithm:
          a. If input_data has exactly one key AND that key is the name
             of one of THIS step's input data units, unwrap.
          b. Then, if the inner workflow has exactly one first-step
             input DU AND input_data does NOT already contain that
             key, wrap under that DU.
        """
        # Step (a): unwrap the framework's outer wrapping.
        my_input_dus = getattr(self, "step_input_data_units", None) or {}
        if (
            isinstance(input_data, dict)
            and len(input_data) == 1
            and len(my_input_dus) > 0
        ):
            sole_key = next(iter(input_data.keys()))
            if sole_key in my_input_dus and isinstance(
                input_data[sole_key], dict
            ):
                input_data = input_data[sole_key]

        # Step (b): wrap for the inner's first-step input DU.
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

    def _raise_if_inner_step_failed(
        self, inner_failures: Optional[list[StepEvent]]
    ) -> None:
        """Re-raise the inner cascade's FIRST captured step failure, fast.

        When an inner step raised, the trigger executor swallowed the
        exception (G127), so the only signal is the ``step_failed`` event
        captured by ``process()``'s subscriber. Surface it as a
        ``RuntimeError`` carrying the inner step's real type + message so the
        caller degrades with the actual reason instead of a generic timeout.
        """
        if not inner_failures:
            return
        ev = inner_failures[0]
        exc = ev.payload.get("exception", {}) if isinstance(ev.payload, dict) else {}
        raise RuntimeError(
            f"SubworkflowStep {self.name!r}: inner workflow step "
            f"{ev.step_name!r} failed "
            f"({exc.get('type', 'Exception')}: {exc.get('message', '')}). "
            f"Surfaced via step_failed event without waiting the "
            f"{self._timeout_seconds}s inner-cascade timeout."
        )

    async def _clear_inner_last_step_outputs(self) -> None:
        """Reset the inner workflow's last-step output data units to None before a
        (re-)run, so the poll loop waits for the NEW cascade to populate them rather
        than reading the PRIOR run's stale value. Idempotent + best-effort (a clear
        failure must not break the run). See the cached-re-run note in ``process()``."""
        if not self._inner_workflow.child_steps:
            return
        last_step = list(self._inner_workflow.child_steps.values())[-1]
        for du in (getattr(last_step, "step_output_data_units", None) or {}).values():
            try:
                if hasattr(du, "clear"):
                    await du.clear()
                elif hasattr(du, "set"):
                    await du.set(None)
            except Exception:
                pass

    async def _poll_inner_workflow_until_drained(
        self, inner_failures: Optional[list[StepEvent]] = None
    ) -> None:
        """Poll the inner workflow's last step's output data units
        until they're populated, OR raise TimeoutError.

        See the comment in ``process()`` for why this can't use
        ``wait_for_cascade`` (singleton-executor deadlock under
        nested cascades).

        ``inner_failures`` is the live list populated by ``process()``'s
        step-event subscriber; each iteration checks it FIRST so a raising
        inner step short-circuits the wait (BUG A — fast inner-failure
        detection) instead of stalling until ``timeout_seconds``.
        """
        if not self._inner_workflow.child_steps:
            return  # Nothing to wait on; let collect step return {}.

        last_step = list(self._inner_workflow.child_steps.values())[-1]
        output_dus = getattr(last_step, "step_output_data_units", None) or {}
        if not output_dus:
            return

        deadline = asyncio.get_event_loop().time() + self._timeout_seconds
        poll_interval = max(self._settle_ms / 1000.0, 0.05)
        last_du = next(iter(output_dus.values()))
        while True:
            # FAST-FAIL: an inner step raised → surface it now, don't wait.
            self._raise_if_inner_step_failed(inner_failures)
            try:
                value = await last_du.get()
            except Exception:
                value = None
            if value is not None:
                # Last step has emitted; give the cascade one more
                # settle cycle to propagate transitive updates.
                await asyncio.sleep(poll_interval)
                return
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(
                    f"SubworkflowStep {self.name!r}: inner workflow's "
                    f"last step output not populated within "
                    f"{self._timeout_seconds}s. Inner workflow may "
                    f"have an internal step-process failure that was "
                    f"swallowed by the trigger executor."
                )
            await asyncio.sleep(poll_interval)
            # Re-check after the yield: the inner step task may have raised
            # while we slept; surface it before looping back to poll the DU.
            self._raise_if_inner_step_failed(inner_failures)

    async def _collect_last_step_outputs(self) -> Dict[str, Any]:
        """Collect the inner workflow's RESULT — preferring workflow-
        level output data units, falling back to the last step's
        outputs.

        Inner workflows commonly route MULTIPLE step outputs into
        workflow-level data units via DirectLinks. The last step alone
        wouldn't see all of them. Workflow-level output_data_units see
        EVERY link target — so prefer those.

        Falls back to the last step's outputs when the workflow
        declares no output_data_units (legacy / single-output
        workflows).

        **2026-05-12 nested-cascade fix**: when the inner workflow
        declares exactly ONE workflow-level output AND its value is
        a dict (the common case for "wrap a workflow as a step"), we
        FLATTEN — return that dict directly rather than ``{<single
        output name>: <dict>}``. Otherwise the wrapper key
        accumulates as cascades nest, eventually defeating the
        downstream step's single-level unwrap logic and producing
        spurious ``input_data["code_source"] == None`` errors.

        For multi-output inner workflows, we keep the dict-of-outputs
        shape — downstream readers either unwrap by name OR pull the
        full bundle. Operators chaining a multi-output sub-workflow
        into a step that expects a flat shape should compose an
        intermediate adapter step.
        """
        # Prefer workflow-level outputs.
        wf_outputs = getattr(self._inner_workflow, "step_output_data_units", None)
        if wf_outputs:
            collected: Dict[str, Any] = {}
            for name, du in wf_outputs.items():
                try:
                    collected[name] = await du.get()
                except Exception as e:
                    collected[name] = None
                    collected.setdefault("_errors", {})[name] = str(e)
            # Single-output flatten: if exactly one workflow-level
            # output AND its value is a dict, return the dict
            # directly. Otherwise return the dict-of-outputs.
            real_keys = [k for k in collected if k != "_errors"]
            if len(real_keys) == 1:
                sole_value = collected[real_keys[0]]
                if isinstance(sole_value, dict):
                    return sole_value
            return collected

        if not self._inner_workflow.child_steps:
            return {}
        last_step = list(self._inner_workflow.child_steps.values())[-1]
        output_dus = getattr(last_step, "step_output_data_units", None) or {}
        collected = {}
        for name, du in output_dus.items():
            try:
                collected[name] = await du.get()
            except Exception as e:
                collected[name] = None
                collected.setdefault("_errors", {})[name] = str(e)
        # Same single-output flatten for the last-step fallback path.
        real_keys = [k for k in collected if k != "_errors"]
        if len(real_keys) == 1:
            sole_value = collected[real_keys[0]]
            if isinstance(sole_value, dict):
                return sole_value
        return collected


__all__ = [
    "SubworkflowStep",
    "SubworkflowStepConfig",
]
