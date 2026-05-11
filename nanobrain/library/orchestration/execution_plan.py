"""ExecutionPlanConfig + ExecutionPlanDataUnit (G16).

Per ``apecx-mcp-integration/docs/CONTRACTS.md#g16`` and
``apecx-mcp-integration/docs/CONTRACTS.md#workflow-execution-plan``: the
typed Phase-0 output schema. The orchestrator agent's sole authoring
artifact is an ExecutionPlan; this module defines the schema as a
nanobrain primitive (Pydantic ConfigBase with extra='forbid') and the
DataUnit carrier that ferries it between orchestrator steps.

Three workspace constraints honored:

1. **`extra='forbid'` everywhere** (workspace memory `pydantic_extra_forbid_rule`).
   A YAML typo silently using a default is exactly the silent-failure
   shape the design package was built to eliminate.
2. **Cross-references the design doc, not the framework** for the
   semantic spec. This module owns the on-the-wire shape; ownership of
   *what each field means* lives in `CONTRACTS.md#workflow-execution-plan`.
3. **Backward compatibility**: this module is purely additive. Apecx-mcp
   code that uses a hand-rolled ExecutionPlan dict continues to work
   until the consumer migrates to the typed primitive.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import ConfigDict, Field, field_validator

from nanobrain.core.config.config_base import ConfigBase
from nanobrain.core.data_unit import DataUnitMemory


# Strategy literal — matches CONTRACTS.md#workflow-strategies (A=skeleton
# selection, B=composition, C=synthesis; D is forbidden by construction).
ExecutionPlanStrategy = Literal["A", "B", "C"]

# Layer types — matches CONTRACTS.md#output-layer-types.
ExecutionPlanLayerType = Literal[
    "sequence", "structural", "functional", "evidence", "cross_source", "design"
]

# Executor names — matches the framework's ExecutorBase subclass surface.
ExecutionPlanExecutorKind = Literal[
    "LocalExecutor", "ThreadExecutor", "ProcessExecutor", "ParslExecutor"
]


class ExecutionPlanLayer(ConfigBase):
    """One layer in a plan's `layers` list.

    Per `CONTRACTS.md#output-layer`. A layer pairs a layer_type
    with the data sources to consult and a brief expected_contribution
    string.
    """
    model_config = ConfigDict(extra="forbid")

    layer_id: str
    layer_type: ExecutionPlanLayerType
    data_sources: List[str] = Field(default_factory=list)
    expected_contribution: str = ""
    depends_on: List[str] = Field(default_factory=list)


class ExecutionPlanSkeletonRef(ConfigBase):
    """A composed skeleton reference (Strategy B only).

    The plan's primary `skeleton_id` is the first composed; additional
    skeletons are listed here with an `alias` used by inter_skeleton_links.
    """
    model_config = ConfigDict(extra="forbid")

    skeleton_id: str
    skeleton_version: str
    alias: str


class ExecutionPlanToolInvocation(ConfigBase):
    """A tool slot to fill in the lowered workflow.

    `tool_descriptor_ref` resolves to a UTD in the catalog at lowering
    time. `input_bindings` map the tool's input schema to expressions
    referencing upstream layer outputs (Jinja-like templating).
    """
    model_config = ConfigDict(extra="forbid")

    slot_id: str
    tool_descriptor_ref: str
    input_bindings: Dict[str, Any] = Field(default_factory=dict)


class ExecutionPlanInterSkeletonLink(ConfigBase):
    """A typed boundary link between two composed skeletons (Strategy B).

    `from_` and `to` are `<alias>.<data_unit_name>` references. The
    `link_class` chooses between DirectLink and ConditionalLink (G1
    predicates supported via the `predicate` field).
    """
    model_config = ConfigDict(extra="forbid")

    # Use Field alias so YAML can spell it `from:` despite Python keyword.
    from_: str = Field(alias="from")
    to: str
    link_class: Literal["DirectLink", "ConditionalLink"]
    predicate: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Required when link_class='ConditionalLink'. Shape "
                    "matches the G1 PredicateConfig schema in "
                    "nanobrain.core.link.PredicateConfig."
    )


class ExecutionPlanResourceEnvelope(ConfigBase):
    """Per `CONTRACTS.md#workflow-execution-plan`. Pinned at plan emission so
    the cost/walltime gates (HITL §3.4 / §3.5) have an honest target."""
    model_config = ConfigDict(extra="forbid")

    executor: ExecutionPlanExecutorKind
    walltime_minutes: int = Field(ge=1, le=1440)
    estimated_cost_usd: float = Field(ge=0.0)
    hpc_eligible: bool


class ExecutionPlanProvenanceSeed(ConfigBase):
    """Provenance threading per `CONTRACTS.md#workflow-execution-plan`. Every
    step in the lowered workflow inherits this seed via the lowering
    pipeline's Step 6 (per `CONTRACTS.md#workflow-lowering`)."""
    model_config = ConfigDict(extra="forbid")

    session_id: str
    user_id: str
    intent: str
    phase0_evidence_refs: List[str] = Field(default_factory=list)


_SKELETON_VERSION_RE = re.compile(r"^[a-f0-9]{12}$")


class ExecutionPlanConfig(ConfigBase):
    """G16 — typed Phase-0 ExecutionPlan.

    The orchestrator agent's authoring output. The full semantic spec
    lives in `apecx-mcp-integration/docs/CONTRACTS.md#workflow-execution-plan`;
    this class is the on-the-wire schema.

    Workspace discipline:
    - `model_config = {"extra": "forbid"}` — typos are FAIL-FAST, not
      silent default-application.
    - `plan_version` is a Literal["1"] today; future versions bump this
      to "2", "3", … so consumers can branch on schema version.
    - `skeleton_version` accepts either a 12-character hex digest prefix
      (per the gap proposal's content-addressing) OR a semver tag
      ("1.2.3-beta.1"). The lowering pipeline's Gate-2 resolves either
      to a specific digest at load time.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    plan_version: Literal["1"] = "1"
    strategy: ExecutionPlanStrategy
    skeleton_id: str
    skeleton_version: str
    skeleton_refs: Optional[List[ExecutionPlanSkeletonRef]] = Field(
        default=None,
        description="Strategy B only — additional skeletons composed with "
                    "the primary skeleton_id. None or [] for Strategy A/C.",
    )
    active_layers: List[ExecutionPlanLayerType] = Field(default_factory=list)
    parameter_bindings: Dict[str, Any] = Field(default_factory=dict)
    tool_invocations: List[ExecutionPlanToolInvocation] = Field(
        default_factory=list
    )
    resource_envelope: ExecutionPlanResourceEnvelope
    inter_skeleton_links: Optional[List[ExecutionPlanInterSkeletonLink]] = Field(
        default=None,
        description="Strategy B only — typed boundary links between "
                    "composed skeletons. None or [] for Strategy A/C.",
    )
    provenance_seed: ExecutionPlanProvenanceSeed

    @field_validator("skeleton_version")
    @classmethod
    def _validate_skeleton_version(cls, v: str) -> str:
        """Accept either a 12-char hex prefix OR a semver-ish tag.

        The lowering pipeline's Gate-2 resolves the value to a specific
        registry digest at load time. We do NOT require strict semver
        because the registry may use looser tags (e.g., "v1.2.3-rc.1");
        we just reject obviously-wrong shapes (empty, whitespace, control
        characters).
        """
        if not v or not v.strip() or any(c.isspace() for c in v):
            raise ValueError(
                "FAIL-FAST: ExecutionPlanConfig.skeleton_version must be a "
                "non-empty whitespace-free string (12-char hex digest "
                "prefix or registry semver tag)"
            )
        return v


class ExecutionPlanDataUnit(DataUnitMemory):
    """G16 — DataUnitMemory subclass that carries an ExecutionPlanConfig.

    The orchestrator's `Phase0PlanningStep` writes its output to a
    DataUnit of this class; the downstream `SkeletonSelectorStep`,
    `PlanLoweringStep`, and validation gates all read from it.

    Why a typed subclass rather than a plain DataUnitMemory:
    1. **Type signaling**: a workflow YAML referencing this class makes
       it clear to the reader (and to the validator) that the data unit
       carries a structured ExecutionPlan, not arbitrary bytes.
    2. **Validation hook**: future framework-side enhancements can
       validate the payload against ExecutionPlanConfig at .set() time
       (analogous to G6's validate_on_set for ProxyRef). Today we
       inherit DataUnitMemory's untyped storage; the validation lives
       in the producer (Phase0PlanningStep validates before .set()).
    3. **Discoverability**: nanobrain.lightweight.WorkflowBuilder can
       enumerate ExecutionPlanDataUnit as a known data-unit class for
       orchestrator-shaped workflows.

    The class adds NO behavior beyond DataUnitMemory in v1. All custom
    semantics (typed-payload validation at .set() time) are deferred to
    a follow-up task once the validator surface is stable.
    """

    pass
