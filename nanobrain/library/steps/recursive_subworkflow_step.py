"""RecursiveSubworkflowStep — dynamic recursive workflow self-reference.

G110 (2026-05-17). Companion to ``SubworkflowStep`` that enables a
workflow to invoke ITSELF as an inner step. Loads the inner workflow
LAZILY at process() time (vs SubworkflowStep's config-time load),
so a self-reference doesn't trigger infinite recursion at workflow
construction.

The motivating use case (per apecx-mcp-integration/docs/hd_rss_pattern_2026-05-17.md):
true dynamic-recursive workflow self-reference for HD-RSS-style
patterns. HD-RSS judges atomicity → decomposes into subgoals →
recursively solves each subgoal (this IS the framework primitive that
was missing) → composes bottom-up. Today HD-RSS's recursion lives
in pure Python; with this primitive, it could be expressed as a
nanobrain workflow whose body contains a RecursiveSubworkflowStep
pointing back to itself.

Mechanism
---------
* **Lazy inner-workflow load.** Unlike SubworkflowStep which loads
  + caches the inner workflow at ``_init_from_config`` time, this
  Step holds only the PATH at init time. The actual
  ``Workflow.from_config(path)`` call happens in process(), per
  invocation. This means a self-referencing YAML doesn't crash
  the workflow loader.
* **Depth threading.** Each process() call accepts an envelope with
  a ``_recursion_depth`` field (default 0). The Step increments it
  + injects into the recursive call's input. At ``max_recursion_depth``,
  the Step emits a terminal envelope ``{'_recursion_terminated':
  True, ...}`` rather than recursing — the consumer decides what
  to do with the unsolvable subproblem.
* **Fresh inner instance per call.** Building a fresh Workflow
  instance per recursive call costs ~1-2s of from_config overhead.
  This is the price of state isolation — alternative caching by
  depth would leak iteration-counter state across recursion levels.
  Mitigation: cap depth at 3 by default (worst-case ~6s overhead).

Honest scope limitations
------------------------
* The TARGET workflow is the responsibility of the author. This
  Step is the recursion MECHANISM; the workflow body's domain
  logic (decompose, atomic-judge, compose) is separate.
* No memoization of recursive subproblems. Two calls with identical
  inputs at the same depth will re-execute. Adding a content-hash
  cache is straightforward but adds complexity.
* The depth cap is per-Step instance. If a workflow contains TWO
  RecursiveSubworkflowStep instances at the same level, each has
  its own depth counter. Author with care.
* When the depth cap fires, the Step does NOT raise — it emits a
  terminal envelope so the workflow can route via ConditionalLink
  to a "give up gracefully" path. This is the same discipline as
  LoopController's loop_exhausted.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.workflow import Workflow

logger = logging.getLogger(__name__)


class RecursiveSubworkflowStepConfig(StepConfig):
    """Configuration for ``RecursiveSubworkflowStep``.

    Mandatory: ``inner_workflow_path`` — the YAML to load per-call.
    For SELF-reference, set this to the same workflow YAML that
    contains this step.

    Optional knobs:

    * ``max_recursion_depth`` — hard cap. Reaching this emits a
      terminal envelope rather than recursing. Default 3 (matches
      HD-RSS's _MAX_RECURSION_DEPTH).
    * ``depth_field_name`` — the envelope field carrying the current
      depth. Default ``_recursion_depth``. Underscore-prefixed by
      convention (framework-reserved key, not user payload).
    * ``terminal_marker_field`` — the field set to True in the
      terminal envelope. Default ``_recursion_terminated``.
    * ``input_data_unit_name`` — the inner workflow's first-step
      input data unit name. Used to thread the envelope into the
      inner ``Workflow.run({<name>: envelope})``.
    * ``output_data_unit_name`` — the workflow-level output to
      extract from the inner workflow's run() result.
    * ``timeout_seconds`` — per-call inner-workflow timeout.
    """

    inner_workflow_path: str = Field(
        ...,
        description=(
            "Path to the YAML of the workflow to recurse into. May "
            "be the same as the OUTER workflow's path for true "
            "self-reference. Resolved at process() time, not at "
            "config-load time (this is what makes self-reference "
            "safe)."
        ),
    )

    max_recursion_depth: int = Field(
        default=3,
        ge=1,
        description=(
            "Hard cap on recursion depth. The Step emits a terminal "
            "envelope (rather than recursing) when the incoming "
            "depth equals this cap. Matches HD-RSS's default of 3."
        ),
    )

    depth_field_name: str = Field(
        default="_recursion_depth",
        description=(
            "Envelope field carrying the current depth. The Step "
            "reads it (default 0 if missing) + writes depth+1 into "
            "the recursive call's envelope. Underscore-prefixed by "
            "convention to mark as framework-reserved."
        ),
    )

    terminal_marker_field: str = Field(
        default="_recursion_terminated",
        description=(
            "Envelope field set to True in the terminal envelope (at "
            "depth cap). Consumers route on this to handle the "
            "'recursion gave up' case via ConditionalLink predicate."
        ),
    )

    input_data_unit_name: str = Field(
        ...,
        description=(
            "Name of the inner workflow's first-step input data unit. "
            "Used to call ``inner_workflow.run({<name>: envelope})``."
        ),
    )

    output_data_unit_name: str = Field(
        ...,
        description=(
            "Name of the workflow-level output data unit on the "
            "inner workflow. Used to extract the recursion result "
            "from the inner workflow's run() output dict."
        ),
    )

    timeout_seconds: float = Field(
        default=300.0,
        gt=0.0,
        description=(
            "Per-call inner workflow run timeout. Should accommodate "
            "the recursion's worst case (max_depth levels of nesting "
            "× per-level work)."
        ),
    )


class RecursiveSubworkflowStep(BaseStep):
    """Bounded-recursion sub-workflow invocation.

    Expected ``process()`` input shape::

        {
            "<envelope fields>": ...,
            "_recursion_depth": N,  # optional, default 0
        }

    Behavior:

    * If depth >= max_recursion_depth → emit terminal envelope::

        {**input, "_recursion_terminated": True}

      Caller routes via ConditionalLink to handle the unsolved
      subproblem.

    * Otherwise → load a FRESH inner workflow instance, inject
      depth+1, run, return the inner workflow's output unit value.

    Termination guarantees:

    * Hard depth cap (config field) prevents infinite recursion.
    * Per-call workflow load isolates state (no LoopController
      counter leakage across depths).
    """

    COMPONENT_TYPE: str = "recursive_subworkflow_step"
    REQUIRED_CONFIG_FIELDS: list[str] = [
        "name", "inner_workflow_path",
        "input_data_unit_name", "output_data_unit_name",
    ]

    @classmethod
    def _get_config_class(cls):
        return RecursiveSubworkflowStepConfig

    @classmethod
    def extract_component_config(
        cls, config: RecursiveSubworkflowStepConfig
    ) -> Dict[str, Any]:
        base = super().extract_component_config(config)
        return {
            **base,
            "inner_workflow_path": config.inner_workflow_path,
            "max_recursion_depth": config.max_recursion_depth,
            "depth_field_name": config.depth_field_name,
            "terminal_marker_field": config.terminal_marker_field,
            "input_data_unit_name": config.input_data_unit_name,
            "output_data_unit_name": config.output_data_unit_name,
            "timeout_seconds": config.timeout_seconds,
        }

    def _init_from_config(
        self,
        config: RecursiveSubworkflowStepConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        super()._init_from_config(config, component_config, dependencies)

        # Resolve the inner path to absolute at init time — fails
        # fast if the path doesn't exist. We DON'T load the workflow
        # here; that happens per process() call to support
        # self-reference.
        self._inner_workflow_path: Path = self._resolve_path(
            component_config["inner_workflow_path"]
        )
        self._max_depth: int = int(component_config["max_recursion_depth"])
        self._depth_field: str = str(component_config["depth_field_name"])
        self._terminal_field: str = str(component_config["terminal_marker_field"])
        self._input_du_name: str = str(component_config["input_data_unit_name"])
        self._output_du_name: str = str(component_config["output_data_unit_name"])
        self._timeout_seconds: float = float(component_config["timeout_seconds"])

        logger.info(
            "RecursiveSubworkflowStep %r initialized: inner=%s, max_depth=%d",
            self.name, self._inner_workflow_path, self._max_depth,
        )

    @staticmethod
    def _resolve_path(path_str: str) -> Path:
        """Resolve to an absolute path that exists. FAIL-FAST if not.
        Reuses the same resolution pattern as SubworkflowStep."""
        p = Path(path_str)
        if p.is_absolute() and p.is_file():
            return p
        if p.is_absolute():
            raise ComponentConfigurationError(
                f"FAIL-FAST: RecursiveSubworkflowStep inner_workflow_path "
                f"{path_str!r} is absolute but does not exist on disk"
            )
        # Try workspace root via G40 helper.
        try:
            from nanobrain.library.runtime.workspace_root import locate_workflow_root

            root = locate_workflow_root()
            if root is not None:
                candidate = (root / p).resolve()
                if candidate.is_file():
                    return candidate
        except ImportError:
            pass
        # Try cwd.
        cwd_candidate = (Path.cwd() / p).resolve()
        if cwd_candidate.is_file():
            return cwd_candidate
        raise ComponentConfigurationError(
            f"FAIL-FAST: RecursiveSubworkflowStep inner_workflow_path "
            f"{path_str!r} could not be resolved. Tried: workspace_root, "
            f"cwd ({Path.cwd()}). Provide an absolute path or set "
            f"$NANOBRAIN_WORKSPACE_ROOT."
        )

    async def process(
        self, input_data: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Recurse — or terminate when at depth cap."""
        if not isinstance(input_data, dict):
            raise ValueError(
                f"RecursiveSubworkflowStep {self.name!r}: input_data must be "
                f"a dict, got {type(input_data).__name__}"
            )

        depth = int(input_data.get(self._depth_field, 0))

        if depth >= self._max_depth:
            logger.info(
                "RecursiveSubworkflowStep %r: depth cap %d reached; "
                "emitting terminal envelope (no further recursion)",
                self.name, self._max_depth,
            )
            return {
                **input_data,
                self._terminal_field: True,
            }

        # Build the recursive-call envelope: increment depth +
        # carry forward all input fields. Strip the terminal marker
        # if present (defensive — should never be present on input).
        next_envelope = {
            **input_data,
            self._depth_field: depth + 1,
        }
        next_envelope.pop(self._terminal_field, None)

        # Load a FRESH inner workflow instance per call. The
        # from_config + initialize cost is the price of state
        # isolation across recursion levels.
        inner_workflow = Workflow.from_config(str(self._inner_workflow_path))

        logger.info(
            "RecursiveSubworkflowStep %r: invoking inner workflow at depth %d → %d",
            self.name, depth, depth + 1,
        )
        outputs = await inner_workflow.run(
            {self._input_du_name: next_envelope},
            timeout=self._timeout_seconds,
            settle_ms=200,
        )

        if outputs is None:
            raise RuntimeError(
                f"RecursiveSubworkflowStep {self.name!r}: inner workflow "
                f"returned None at depth {depth + 1}"
            )

        # Return the workflow-level output as the result of this
        # recursive call. Strip status / errors fields from the run()
        # output dict — those are observability, not the actual
        # recursion result.
        result = outputs.get(self._output_du_name)
        if result is None:
            raise RuntimeError(
                f"RecursiveSubworkflowStep {self.name!r}: inner workflow "
                f"did not produce output {self._output_du_name!r} at depth "
                f"{depth + 1}. Available keys: {list(outputs.keys())}"
            )

        return result if isinstance(result, dict) else {"value": result}
