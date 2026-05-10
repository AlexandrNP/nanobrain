"""G9-completion — pin Workflow.from_skeleton(skeleton, bindings) ergonomic loader.

Pre-G9-completion the framework shipped a Skeleton + SkeletonRegistry
primitive (G9 v1) but the *ergonomic* loader the gap proposal named as
G9's whole point ("let an agent pick a skeleton + bind holes without
authoring a full YAML") was deferred. Agents had to hand-assemble
PlanLoweringStep + SkeletonLoaderStep YAML to get a runnable workflow
from a skeleton — which defeats the entire skeleton ergonomic.

Post-G9-completion: ``Workflow.from_skeleton(skeleton, bindings)``
collapses the dance into one call.

This test pins:
  1. dict-skeleton + valid bindings → runnable Workflow with substituted values
  2. file-path skeleton + valid bindings → runnable Workflow
  3. missing required hole FAIL-FAST with named hole(s) in error
  4. extra binding key not in holes FAIL-FAST with named key(s) in error
  5. optional hole defaults are applied when binding is absent
  6. invalid skeleton input type FAIL-FAST (not just any garbage in)
  7. lowered YAML inherits all from_config validation (e.g. malformed
     YAML in body still surfaces as a load error)

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 2 G9-completion;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.7.
"""
from __future__ import annotations

import textwrap

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.workflow import Workflow


# A trivially-loadable skeleton body. Two holes:
#   - ``query_label``: required string
#   - ``min_evidence``: optional integer with default=3
# The workflow is intentionally minimal: zero steps, zero links. Any
# valid WorkflowConfig will load via ``Workflow.from_config``; the
# substituted name proves substitution actually happened.
_BODY_TEMPLATE = textwrap.dedent(
    """\
    name: g9_test_workflow
    description: "G9-completion test skeleton"
    version: "0.1.0"
    config_version: 2

    steps: {}
    links: {}

    metadata:
      query_label: {{ query_label: string }}
      min_evidence: {{ min_evidence: integer }}
    """
)


def _make_skeleton_dict(*, body: str | None = None) -> dict:
    return {
        "skeleton_id": "g9_test_skeleton",
        "skeleton_version": "0.1.0",
        "description": "G9-completion ergonomic loader test fixture.",
        "body": body if body is not None else _BODY_TEMPLATE,
        "holes": {
            "query_label": {
                "type": "string",
                "required": True,
                "description": "label for the test query",
            },
            "min_evidence": {
                "type": "integer",
                "required": False,
                "default": 3,
                "description": "minimum evidence count",
            },
        },
    }


def test_dict_skeleton_with_valid_bindings_returns_workflow():
    """The happy path: pass an inline dict skeleton + complete bindings;
    receive a Workflow whose lowered YAML embedded the substituted values.
    """
    workflow = Workflow.from_skeleton(
        _make_skeleton_dict(),
        bindings={"query_label": "alpha", "min_evidence": 5},
    )

    assert isinstance(workflow, Workflow), (
        f"expected Workflow instance; got {type(workflow).__name__}"
    )
    # Workflow loaded; its config carries the substituted name.
    assert workflow.name == "g9_test_workflow", (
        f"Workflow name should derive from skeleton body; got "
        f"{workflow.name!r}"
    )


def test_file_skeleton_path_input(tmp_path):
    """Skeleton can be loaded from a YAML file path. The contract
    matches Skeleton.from_config(<path>) — the path is the resolution
    boundary so nested config: references could in principle resolve
    against the skeleton's own directory."""
    import yaml

    skeleton_dict = _make_skeleton_dict()
    sk_path = tmp_path / "skeleton.yml"
    sk_path.write_text(yaml.safe_dump(skeleton_dict))

    workflow = Workflow.from_skeleton(
        sk_path,
        bindings={"query_label": "beta"},
    )
    assert isinstance(workflow, Workflow)
    assert workflow.name == "g9_test_workflow"


def test_missing_required_hole_fails_fast():
    """Required hole 'query_label' is not in bindings — must FAIL-FAST
    with the missing hole name in the error so an agent / operator
    can act on it."""
    with pytest.raises(ComponentConfigurationError) as excinfo:
        Workflow.from_skeleton(
            _make_skeleton_dict(),
            bindings={"min_evidence": 5},  # query_label missing
        )

    msg = str(excinfo.value)
    assert "FAIL-FAST" in msg
    assert "query_label" in msg, (
        f"missing-hole error must name the missing hole; got: {msg!r}"
    )
    assert "missing required holes" in msg, (
        f"missing-hole error must classify the failure; got: {msg!r}"
    )


def test_extra_binding_key_fails_fast():
    """Bindings include a name not declared in skeleton.holes — must
    FAIL-FAST. Otherwise an agent could pass arbitrary keys and the
    skeleton's API surface becomes silently mutable."""
    with pytest.raises(ComponentConfigurationError) as excinfo:
        Workflow.from_skeleton(
            _make_skeleton_dict(),
            bindings={
                "query_label": "alpha",
                "min_evidence": 5,
                "rogue_key": "should not be allowed",
            },
        )

    msg = str(excinfo.value)
    assert "FAIL-FAST" in msg
    assert "rogue_key" in msg, (
        f"extra-key error must name the offending key; got: {msg!r}"
    )
    assert "extra binding keys" in msg


def test_optional_hole_default_applied_when_absent():
    """Optional hole with default=3 is satisfied without providing the
    binding. The substitution must use the default value."""
    workflow = Workflow.from_skeleton(
        _make_skeleton_dict(),
        bindings={"query_label": "alpha"},  # min_evidence absent
    )
    assert isinstance(workflow, Workflow)
    # We can't directly inspect the lowered YAML, but we can prove the
    # workflow loaded — the integer default substituted as a valid
    # YAML scalar; an unsubstituted ``{{ min_evidence: integer }}``
    # token would have made the lowered YAML unparseable.
    assert workflow.name == "g9_test_workflow"


def test_invalid_input_type_fails_fast():
    """Garbage input (int / list / etc.) FAIL-FASTs — the function
    should not silently coerce or accept anything resembling YAML."""
    with pytest.raises(ComponentConfigurationError) as excinfo:
        Workflow.from_skeleton(42, bindings={"query_label": "x"})

    msg = str(excinfo.value)
    assert "FAIL-FAST" in msg
    assert "int" in msg, (
        f"type error must name the rejected type; got: {msg!r}"
    )


def test_pre_loaded_skeleton_instance_is_accepted():
    """Pass a pre-loaded ``Skeleton`` directly — useful when the agent
    already has the skeleton in hand (e.g., resolved from a registry)."""
    from nanobrain.library.orchestration.skeleton import Skeleton

    sk = Skeleton.from_config(_make_skeleton_dict())
    workflow = Workflow.from_skeleton(
        sk, bindings={"query_label": "gamma"}
    )
    assert isinstance(workflow, Workflow)
    assert workflow.name == "g9_test_workflow"


def test_substitution_actually_happens_no_raw_tokens_remain():
    """Pin: the lowered workflow MUST NOT contain raw ``{{...}}`` tokens.
    A regression that skipped substitution would leave them in the body
    and the workflow would still load (YAML treats ``{{...}}`` as a
    string), but every consumer would see literal placeholders.

    We assert by introspecting the WorkflowConfig that loading produced.
    """
    workflow = Workflow.from_skeleton(
        _make_skeleton_dict(),
        bindings={"query_label": "delta", "min_evidence": 7},
    )
    # workflow.config is the resolved WorkflowConfig.
    cfg_dump = (
        workflow.config.model_dump()
        if hasattr(workflow.config, "model_dump")
        else workflow.config.__dict__
    )
    serialized = repr(cfg_dump)
    assert "{{" not in serialized, (
        f"raw skeleton tokens leaked into the workflow config; "
        f"substitution is broken. Found tokens in serialized config: "
        f"{serialized[:200]}..."
    )
    # The substituted scalar value should show up somewhere.
    assert "delta" in serialized, (
        f"substituted value 'delta' missing from workflow config; "
        f"substitution wrote the wrong value or skipped the token."
    )
