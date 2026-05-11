"""Orchestration primitives for agent-authored workflows.

Per ``apecx-mcp-integration/docs/CONTRACTS.md#alignment-§4.2`` and
``apecx-mcp-integration/docs/CONTRACTS.md#workflow-lowering``: the orchestrator
that composes analytical workflows is itself a nanobrain workflow. This
package ships the framework-side primitives the orchestrator uses:

- :class:`ExecutionPlanConfig` (G16) — the typed Phase-0 output schema
- :class:`ExecutionPlanDataUnit` (G16) — the carrier that ferries the
  plan between orchestrator steps

Future modules (deferred to follow-up tasks):

- ``skeleton.py`` (G9) — Skeleton primitive with hole grammar
- ``skeleton_loader_step.py`` (G17) — resolves skeleton_id+version
- ``plan_lowering_step.py`` (G17) — applies the 7 lowering steps
"""

from .execution_plan import (
    ExecutionPlanConfig,
    ExecutionPlanDataUnit,
    ExecutionPlanLayer,
    ExecutionPlanResourceEnvelope,
    ExecutionPlanProvenanceSeed,
    ExecutionPlanToolInvocation,
    ExecutionPlanInterSkeletonLink,
    ExecutionPlanSkeletonRef,
    ExecutionPlanStrategy,
)
from .run_context import (
    WorkflowRunContext,
    WorkflowRunContextConfig,
    current_run_context,
)
from .skeleton import (
    Skeleton,
    SkeletonHole,
    SkeletonHoleType,
    SkeletonRegistry,
    compute_skeleton_body_hash,
)
from .skeleton_loader_step import (
    SkeletonLoaderStep,
    SkeletonLoaderStepConfig,
)
from .plan_lowering_step import (
    PlanLoweringStep,
    PlanLoweringStepConfig,
)

__all__ = [
    "ExecutionPlanConfig",
    "ExecutionPlanDataUnit",
    "ExecutionPlanLayer",
    "ExecutionPlanResourceEnvelope",
    "ExecutionPlanProvenanceSeed",
    "ExecutionPlanToolInvocation",
    "ExecutionPlanInterSkeletonLink",
    "ExecutionPlanSkeletonRef",
    "ExecutionPlanStrategy",
    "WorkflowRunContext",
    "WorkflowRunContextConfig",
    "current_run_context",
    "Skeleton",
    "SkeletonHole",
    "SkeletonHoleType",
    "SkeletonRegistry",
    "compute_skeleton_body_hash",
    "SkeletonLoaderStep",
    "SkeletonLoaderStepConfig",
    "PlanLoweringStep",
    "PlanLoweringStepConfig",
]
