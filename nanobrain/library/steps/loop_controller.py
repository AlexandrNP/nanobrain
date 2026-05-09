"""LoopController — bounded-cycle relaxation primitive for nanobrain workflows.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G18``: a Step
that owns the iteration counter for a back-edge through the workflow DAG.
Without an iteration cap, a workflow with a back-edge becomes an infinite
loop. The framework's existing cycle detector rejects all cycles by default;
``LoopController`` is the framework-blessed exception that allows DECLARED
back-edges with a hard iteration cap.

Use case
--------
The motivating use case is the agent-authored workflow repair loop
(``agent_workflow_authoring.md §7``): when validation rejects an
``ExecutionPlan``, the orchestrator routes the rejection back to
``Phase0PlanningStep`` for repair. The repair loop has a hard cap of
**two repair attempts** (per the spec); after the cap, the agent
escalates rather than looping forever.

Behavior
--------
On each invocation, the controller:

1. Reads the current iteration count from its persistent state.
2. If ``count < max_iterations``, increments the counter, writes it back,
   and returns a payload that the downstream ``ConditionalLink`` interprets
   as "continue the loop" (the controller's output carries the original
   payload plus an ``allow_continue: true`` marker).
3. If ``count >= max_iterations``, returns a payload with
   ``allow_continue: false`` and ``loop_exhausted: true``. The downstream
   ConditionalLink routes this to the escalation path.

The downstream ConditionalLink uses G1's declarative predicate DSL:

    predicate: {op: eq, field: allow_continue, value: true}

State persistence
-----------------
The iteration counter is held on the controller instance for the lifetime
of the workflow run. Across process restarts, persistent state requires
G5 (CheckpointStep / ResumeStep) — the controller's counter survives
resume because the counter lives in the controller's data unit, which is
captured by the checkpoint manifest.

What this primitive does NOT do
-------------------------------
- Does NOT relax the workflow integrity validator's cycle detection.
  G18 Step 2 (workflow validator extension allowing declared back-edges
  through a LoopController) is a SEPARATE task. Today, a workflow with
  a back-edge through this controller still trips the cycle detector
  unless ``allow_cycles: true`` is set on the workflow. The controller
  is a runtime primitive; the validator extension is a load-time
  primitive. Both are needed for the full G18 contract; this commit
  ships the runtime piece.
- Does NOT track iteration history. Only the count is recorded. If the
  agent needs to remember "what did I try in iteration 1", it must
  thread that state through the back-edge payload itself.

Cross-references:
- ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G18`` — gap proposal
- ``apecx-mcp-integration/docs/agent_workflow_authoring.md §7`` — the repair loop
- ``apecx-mcp-integration/docs/reasoning_patterns_library.md P7`` — retry-with-feedback
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from pydantic import Field

from nanobrain.core.step import BaseStep, StepConfig

logger = logging.getLogger(__name__)


class LoopControllerConfig(StepConfig):
    """Configuration for LoopController.

    Extends ``StepConfig`` with a single required field:

    - ``max_iterations``: positive int. Hard cap on iterations. Reaching
      this count causes the controller to emit ``loop_exhausted: true``;
      the downstream ConditionalLink routes to escalation.

    Optional fields:

    - ``initial_count``: int (default 0). The starting count, in case a
      caller wants to seed the controller (e.g., re-entering after a
      checkpoint resume).
    - ``payload_passthrough_key``: str (default ``"payload"``). The output
      dict's key holding the upstream payload pass-through, useful when
      downstream consumers want to forward the original input alongside
      the loop-control flag.
    """
    max_iterations: int = Field(
        ...,
        ge=1,
        description="Hard cap on iterations. The controller emits "
                    "loop_exhausted=true on the (max_iterations+1)th call.",
    )
    initial_count: int = Field(
        default=0,
        ge=0,
        description="Starting iteration count (default 0). Useful when "
                    "reseeding after a checkpoint resume.",
    )
    payload_passthrough_key: str = Field(
        default="payload",
        description="Key under which the upstream input is echoed in the "
                    "controller's output dict.",
    )


class LoopController(BaseStep):
    """Bounded-cycle iteration counter (G18).

    Output shape (always):

    ```python
    {
        "allow_continue": bool,        # downstream gate predicate fires on this
        "iteration": int,              # the count BEFORE the gate decision
        "max_iterations": int,         # echo of the configured cap
        "loop_exhausted": bool,        # True iff allow_continue is False
        "<payload_passthrough_key>": <whatever was passed in>,
    }
    ```

    Downstream ConditionalLink predicates (G1 DSL):

    ```yaml
    # Continue path (back to the work step):
    predicate: {op: eq, field: allow_continue, value: true}

    # Escalation path (forward to the escalation step):
    predicate: {op: eq, field: loop_exhausted, value: true}
    ```

    The two predicates are mutually exclusive — exactly one of the two
    downstream ConditionalLinks fires per controller invocation.
    """

    COMPONENT_TYPE: str = "loop_controller"
    REQUIRED_CONFIG_FIELDS = ["name", "max_iterations"]

    @classmethod
    def _get_config_class(cls):
        """Tell the framework's from_config loader which Pydantic schema
        to use for YAML validation. Without this override, the loader
        defaults to base StepConfig and silently drops our extra fields."""
        return LoopControllerConfig

    def _init_from_config(
        self,
        config: LoopControllerConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        super()._init_from_config(config, component_config, dependencies)

        # Cache the typed config for runtime access. We don't mutate the
        # original config object; the counter is a separate per-step state.
        self._loop_config: LoopControllerConfig = config

        # Iteration counter — incremented on each process() call up to the
        # configured cap. Lives on the instance for the workflow run's
        # lifetime; G5 checkpoint capture preserves this across restarts.
        self._iteration_count: int = config.initial_count

    async def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """One invocation of the loop gate.

        On every call:

        1. Compare current count to ``max_iterations``.
        2. If under the cap: increment count, emit ``allow_continue=True``.
        3. If at/over the cap: leave count untouched (idempotent on
           repeated calls after exhaustion), emit ``allow_continue=False``
           and ``loop_exhausted=True``.

        The ``input_data`` payload is echoed under ``payload_passthrough_key``
        so downstream steps can route it without losing it.
        """
        cap = self._loop_config.max_iterations
        passthrough_key = self._loop_config.payload_passthrough_key
        pre_count = self._iteration_count

        if pre_count < cap:
            self._iteration_count += 1
            allow = True
            exhausted = False
            logger.debug(
                "LoopController %s: iteration %d/%d — continuing",
                self.name, self._iteration_count, cap,
            )
        else:
            # At or above cap. Don't increment further.
            allow = False
            exhausted = True
            logger.info(
                "LoopController %s: cap reached (%d/%d) — emitting "
                "loop_exhausted; downstream escalation should fire",
                self.name, pre_count, cap,
            )

        return {
            "allow_continue": allow,
            "iteration": pre_count,
            "max_iterations": cap,
            "loop_exhausted": exhausted,
            passthrough_key: input_data,
        }

    def reset(self) -> None:
        """Manually reset the iteration counter to the configured initial
        value. Used by tests and by orchestrators that want to reuse a
        controller instance across multiple distinct loop runs.
        """
        self._iteration_count = self._loop_config.initial_count

    @property
    def iteration_count(self) -> int:
        """Current iteration count. Useful for tests and inspection."""
        return self._iteration_count

    @property
    def is_exhausted(self) -> bool:
        """True iff a call to ``process()`` would emit ``loop_exhausted=True``."""
        return self._iteration_count >= self._loop_config.max_iterations
