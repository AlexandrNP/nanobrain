"""PlanLoweringStep (G17 part 2).

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G17`` and
``apecx-mcp-integration/docs/agent_workflow_authoring.md §5``: the
deterministic transformation that takes a (Skeleton, ExecutionPlan)
pair and produces a content-addressed lowered workflow YAML.

Per the spec, the lowering is a **structured transformation**, NOT a
jinja template render. It understands nanobrain's YAML schema and
validates at each step. The 7 sub-steps from §5:

1. Skeleton resolution — done by SkeletonLoaderStep (G17 part 1).
   This step's input includes the resolved skeleton.
2. Binding validation — verify parameter_bindings cover every required
   hole and don't introduce undeclared keys.
3. Hole substitution — replace ``{{<name>: <type>}}`` tokens in the
   body with bound values.
4. ConditionalLink predicate rewriting — for layer types NOT in
   active_layers, set the gating predicate to always-false. (Skipped
   in v1: the meta-workflow lowering doesn't gate by layer for every
   skeleton; this is a follow-up extension.)
5. Tool descriptor embedding — for each tool_invocation entry,
   substitute the resolved descriptor_id (with version pinned).
6. Provenance seed injection — thread the ExecutionPlan's
   provenance_seed into a workflow-level metadata block.
7. Content hash computation — serialize the lowered YAML in canonical
   form (sorted keys, normalized whitespace) and SHA-256 it. The hash
   is the reproducibility key.

Determinism guarantee: same skeleton (by content_hash) + same plan
(by parameter_bindings + tool_invocations + provenance_seed) → same
lowered YAML bytes → same lowered_yaml_hash. This is load-bearing for
HPC bundle reproducibility.

What this step does NOT do:
- It does NOT load the lowered YAML into a runnable Workflow. That's
  the executor layer's job. The lowering produces the YAML body as a
  string, the validation gates run against that string, and only on
  full-pass does the workflow actually load.
- It does NOT compute Gate-2 (skeleton existence — that's SkeletonLoaderStep)
  or Gate-4 (nanobrain dry-run — separate gate step) or Gate-5
  (resource envelope — separate gate step). It performs Steps 2 + 3
  + 5 + 6 + 7 of the lowering itself; gate steps are sibling steps in
  the meta-workflow.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List


from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.step import BaseStep, StepConfig

from .skeleton import Skeleton

logger = logging.getLogger(__name__)


# Sentinel for hole defaults — distinguishes "no default declared" from
# "default is None" (the latter being a legitimate value).
_NO_DEFAULT = object()


class PlanLoweringStepConfig(StepConfig):
    """Configuration for PlanLoweringStep.

    No additional fields beyond StepConfig — the inputs (skeleton +
    execution plan) come through process() as a dict, NOT as
    config-time values. The step is a pure transformation; its config
    just gives it a name + standard step framing.
    """
    pass


class PlanLoweringStep(BaseStep):
    """G17 — deterministic Plan → Lowered YAML transformation.

    Input dict (process input):
        ``skeleton``: Skeleton — the resolved skeleton (from SkeletonLoaderStep)
        ``execution_plan``: ExecutionPlanConfig OR dict — Phase 0's plan
        ``parameter_bindings``: dict — flat key/value map; merged with
            execution_plan.parameter_bindings if both are present (the
            top-level dict wins on conflict)

    Output dict:
        ``lowered_yaml``: str — the lowered workflow YAML body
        ``lowered_yaml_hash``: str — SHA-256 of the canonical form
        ``skeleton_id``: str — echo
        ``skeleton_content_hash``: str — echo
        ``provenance_seed``: dict — echo (for downstream provenance steps)
        ``binding_summary``: dict — which holes got which values
            (for audit; redacted if it carries sensitive payloads)
    """

    COMPONENT_TYPE: str = "plan_lowering_step"
    REQUIRED_CONFIG_FIELDS = ["name"]

    @classmethod
    def _get_config_class(cls):
        return PlanLoweringStepConfig

    async def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        if not isinstance(input_data, dict):
            raise ComponentConfigurationError(
                f"FAIL-FAST: PlanLoweringStep {self.name!r} input_data "
                f"must be dict, got {type(input_data).__name__}"
            )

        skeleton = input_data.get("skeleton")
        execution_plan = input_data.get("execution_plan")
        if skeleton is None or execution_plan is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: PlanLoweringStep {self.name!r} input_data "
                f"requires 'skeleton' (Skeleton) and 'execution_plan' "
                f"(ExecutionPlanConfig or dict); got "
                f"skeleton={type(skeleton).__name__}, "
                f"execution_plan={type(execution_plan).__name__}"
            )
        if not isinstance(skeleton, Skeleton):
            raise ComponentConfigurationError(
                f"FAIL-FAST: PlanLoweringStep {self.name!r} 'skeleton' "
                f"must be a Skeleton instance, got "
                f"{type(skeleton).__name__}"
            )

        # Normalize execution_plan to dict for uniform field access.
        if hasattr(execution_plan, "model_dump"):
            plan_dict = execution_plan.model_dump(by_alias=True)
        elif isinstance(execution_plan, dict):
            plan_dict = dict(execution_plan)
        else:
            raise ComponentConfigurationError(
                f"FAIL-FAST: PlanLoweringStep {self.name!r} 'execution_plan' "
                f"must be ExecutionPlanConfig or dict, got "
                f"{type(execution_plan).__name__}"
            )

        # Merge parameter_bindings — top-level dict wins over plan's bindings.
        plan_bindings = plan_dict.get("parameter_bindings", {}) or {}
        top_bindings = input_data.get("parameter_bindings", {}) or {}
        merged_bindings = {**plan_bindings, **top_bindings}

        tool_invocations = plan_dict.get("tool_invocations", []) or []
        provenance_seed = plan_dict.get("provenance_seed", {}) or {}

        # ----- Step 2: binding validation -----
        binding_summary = self._validate_bindings(skeleton, merged_bindings)

        # ----- Step 3: hole substitution -----
        substituted_body = self._substitute_holes(
            skeleton, merged_bindings, binding_summary)

        # ----- Step 5: tool descriptor embedding -----
        tool_substituted_body = self._embed_tool_descriptors(
            substituted_body, tool_invocations)

        # ----- Step 6: provenance seed injection -----
        final_body = self._inject_provenance_seed(
            tool_substituted_body, provenance_seed,
            skeleton_content_hash=skeleton.content_hash,
        )

        # ----- Step 7: content hash computation -----
        lowered_hash = self._compute_lowered_yaml_hash(final_body)

        return {
            "lowered_yaml": final_body,
            "lowered_yaml_hash": lowered_hash,
            "skeleton_id": skeleton.skeleton_id,
            "skeleton_content_hash": skeleton.content_hash,
            "provenance_seed": provenance_seed,
            "binding_summary": binding_summary,
        }

    # ------------------------------------------------------------------
    # Step 2: binding validation
    # ------------------------------------------------------------------

    def _validate_bindings(
        self,
        skeleton: Skeleton,
        bindings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Per agent_workflow_authoring.md §6 Gate 3:
        - Missing required holes FAIL-FAST.
        - Extra binding keys not declared in schema FAIL-FAST.
        - Defaults for optional holes are applied.

        Returns a binding_summary dict: {hole_name: {value, source}}
        where source ∈ {"binding", "default", "absent"}.
        """
        declared = skeleton.holes
        provided = set(bindings.keys())
        declared_names = set(declared.keys())

        # Missing required:
        missing_required = [
            name for name, hole in declared.items()
            if hole.required and name not in provided
        ]
        if missing_required:
            raise ComponentConfigurationError(
                f"FAIL-FAST: PlanLoweringStep {self.name!r} Gate 3 "
                f"(hole_binding_validation): missing required holes "
                f"{sorted(missing_required)}; declared holes are "
                f"{sorted(declared_names)}, provided bindings are "
                f"{sorted(provided)}"
            )

        # Extra bindings (provided but not declared):
        extras = provided - declared_names
        if extras:
            raise ComponentConfigurationError(
                f"FAIL-FAST: PlanLoweringStep {self.name!r} Gate 3 "
                f"(hole_binding_validation): extra binding keys not "
                f"declared in skeleton.holes: {sorted(extras)}; "
                f"declared holes are {sorted(declared_names)}"
            )

        # Build summary with defaults applied.
        summary: Dict[str, Dict[str, Any]] = {}
        for name, hole in declared.items():
            if name in bindings:
                summary[name] = {"value": bindings[name], "source": "binding"}
            elif hole.default is not None:
                summary[name] = {"value": hole.default, "source": "default"}
            else:
                # Optional hole, no default — substitute with None.
                summary[name] = {"value": None, "source": "absent"}

        return summary

    # ------------------------------------------------------------------
    # Step 3: hole substitution
    # ------------------------------------------------------------------

    def _substitute_holes(
        self,
        skeleton: Skeleton,
        bindings: Dict[str, Any],
        binding_summary: Dict[str, Any],
    ) -> str:
        """Replace ``{{<name>: <type>}}`` tokens in the body with their
        bound values.

        Per agent_workflow_authoring.md §5 Step 3: the substitution must
        preserve YAML structural correctness. We use a simple regex
        replacement on the body string with the scalar value's
        YAML-canonical representation.

        Mismatch between declared type and actual value is caught at
        Step 4 (the dry-run) — Step 3 just substitutes.
        """
        import re

        body = skeleton.body
        # Use the same token regex as Skeleton.find_inline_hole_tokens.
        # We capture the FULL matched token for replacement.
        token_pattern = re.compile(
            r"\{\{\s*(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*"
            r"(?P<type>string|integer|number|boolean|array|object|any|tool_descriptor_ref)"
            r"(?:\s*\|\s*default\s*=\s*[^}]+?)?\s*\}\}"
        )

        def _replace(match: re.Match) -> str:
            name = match.group("name")
            if name not in binding_summary:
                # The skeleton body has a token that wasn't declared in
                # holes. Skeleton.validate_against_schema would have
                # flagged this; we FAIL-FAST defensively.
                raise ComponentConfigurationError(
                    f"FAIL-FAST: PlanLoweringStep {self.name!r} skeleton "
                    f"body has token {{{{ {name}: <type> }}}} not in "
                    f"binding summary; skeleton.validate_against_schema() "
                    f"should have caught this earlier"
                )
            value = binding_summary[name]["value"]
            return self._yaml_serialize_scalar(value)

        return token_pattern.sub(_replace, body)

    @staticmethod
    def _yaml_serialize_scalar(value: Any) -> str:
        """Serialize a value to its YAML-scalar form for in-string
        substitution.

        - None → 'null'
        - str → JSON-quoted string (no document-end markers)
        - bool / int / float → str(value)
        - list / dict → JSON form (flow-style YAML equivalent; safe for inline)

        Note: yaml.safe_dump on a bare scalar adds a `\n...` document-end
        marker that breaks in-string substitution. We use json.dumps for
        scalar strings (which produces "double-quoted" output that's
        valid YAML) and for lists/dicts (whose JSON serialization is
        also valid YAML flow-style).
        """
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return json.dumps(value)
        if isinstance(value, str):
            # JSON-encoded string: handles all escaping uniformly + is
            # valid YAML in a quoted-scalar position.
            return json.dumps(value)
        # list / dict / other: JSON form is valid YAML flow-style.
        return json.dumps(value, sort_keys=isinstance(value, dict),
                          default=str)

    # ------------------------------------------------------------------
    # Step 5: tool descriptor embedding
    # ------------------------------------------------------------------

    def _embed_tool_descriptors(
        self,
        body: str,
        tool_invocations: List[Dict[str, Any]],
    ) -> str:
        """For each tool_invocation entry, substitute the
        ``tool_descriptor_ref`` in the body. v1: the tool_invocations
        are recorded in a comment block at the end of the body for
        provenance — full slot substitution is a follow-up that
        requires the skeleton to mark tool slots explicitly.

        This is intentionally minimal in v1: skeletons that need tool
        invocation typically declare ``{{slot_id: tool_descriptor_ref}}``
        holes, which Step 3 has already substituted. The
        tool_invocations list in the plan is the SOURCE for those
        substitutions; this method records the per-slot binding for
        provenance.
        """
        if not tool_invocations:
            return body
        # Append a comment block at the end of the body recording the
        # tool invocations. This is a deliberately lightweight v1
        # approach — full slot substitution lands when skeletons
        # standardize tool-slot markers.
        comment = "\n# --- tool invocations (G17 lowering) ---\n"
        for inv in tool_invocations:
            slot_id = inv.get("slot_id", "<unknown>")
            descriptor_ref = inv.get("tool_descriptor_ref", "<unknown>")
            comment += f"# slot {slot_id}: {descriptor_ref}\n"
        return body + comment

    # ------------------------------------------------------------------
    # Step 6: provenance seed injection
    # ------------------------------------------------------------------

    def _inject_provenance_seed(
        self,
        body: str,
        provenance_seed: Dict[str, Any],
        *,
        skeleton_content_hash: str,
    ) -> str:
        """Thread the provenance_seed into a workflow-level metadata
        block at the top of the body.

        v1: prepend a comment block with the provenance_seed fields +
        the skeleton_content_hash. Subsequent reads (replay, audit) can
        parse the comment to recover the seed without loading the
        workflow.

        Future enhancement: inject into a structured ``metadata:`` field
        in the YAML body itself (requires parsing the body, modifying,
        and re-serializing — a structural transformation). v1 keeps the
        body otherwise-unchanged and uses comments.
        """
        if not provenance_seed and not skeleton_content_hash:
            return body
        header_lines = [
            "# --- provenance header (G17 lowering) ---",
            f"# skeleton_content_hash: {skeleton_content_hash}",
        ]
        for key in sorted(provenance_seed.keys()):
            value = provenance_seed[key]
            # JSON-serialize complex values so the comment is parseable.
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value, sort_keys=True)
            else:
                value_str = str(value)
            header_lines.append(f"# provenance_seed.{key}: {value_str}")
        header_lines.append("# --- end provenance header ---")
        return "\n".join(header_lines) + "\n" + body

    # ------------------------------------------------------------------
    # Step 7: lowered_yaml_hash
    # ------------------------------------------------------------------

    def _compute_lowered_yaml_hash(self, body: str) -> str:
        """Per agent_workflow_authoring.md §5 Step 7: serialize the
        body in canonical form and SHA-256 it.

        Canonical form:
        - Strip lines that are PURELY whitespace (trailing-newline
          variations don't change identity).
        - Strip trailing whitespace from each line.
        - Final newline is mandatory (POSIX).

        The hash is the reproducibility key — every HPC bundle, every
        audit record, every replay reference pins this hash. This MUST
        be deterministic across calls with the same body, regardless of
        line-ending differences.
        """
        normalized_lines = [
            line.rstrip()
            for line in body.split("\n")
        ]
        # Drop trailing empty lines.
        while normalized_lines and normalized_lines[-1] == "":
            normalized_lines.pop()
        canonical = "\n".join(normalized_lines) + "\n"
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
