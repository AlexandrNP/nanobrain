# Workflow Authoring Paths in Nanobrain

**Audience:** developers + LLM agents authoring nanobrain workflows.
**Purpose:** show the three legitimate ways to construct a runnable
``Workflow`` and when to pick each.

All three paths converge on the same runtime: a ``Workflow`` instance
produced via ``from_config`` or its specialized loaders. The path you
pick affects authoring ergonomics, NOT runtime behavior.

---

## 1. Hand-written YAML — `Workflow.from_config(path)`

The canonical, file-first authoring surface. Use when:
- The workflow shape is stable and well-documented.
- You want the YAML to be the source of truth for review + audit.
- The workflow ships in a package and is loaded from disk.

```yaml
# my_workflow.yml
name: example_pipeline
description: "Two-step example"
version: "0.1.0"
config_version: 2  # G7 + G39 — explicit pin

input_data_units:
  query:
    class: "nanobrain.core.data_unit.DataUnitMemory"
    name: query
    persistent: false

output_data_units:
  answer:
    class: "nanobrain.core.data_unit.DataUnitMemory"
    name: answer
    persistent: false

steps:
  retrieve:
    class: "my_pkg.steps.RetrieveStep"
    config: "steps/retrieve.yml"
  synthesize:
    class: "my_pkg.steps.SynthesizeStep"
    config: "steps/synthesize.yml"

links:
  q_to_retrieve:
    class: "nanobrain.core.link.DirectLink"
    config:
      link_type: direct
      source: "query"
      target: "retrieve.query"
      auto_transfer: true        # G7/G39 — explicit always
  retrieve_to_synth:
    class: "nanobrain.core.link.DirectLink"
    config:
      link_type: direct
      source: "retrieve.docs"
      target: "synthesize.context"
      auto_transfer: true
  synth_to_output:
    class: "nanobrain.core.link.DirectLink"
    config:
      link_type: direct
      source: "synthesize.answer"
      target: "answer"
      auto_transfer: true
```

```python
from nanobrain.core.workflow import Workflow

wf = Workflow.from_config("my_workflow.yml")
result = await wf.run({"query": "EEEV vaccines"})
print(result["answer"])
```

**Trade-offs**: most explicit; verbose for repetitive shapes; the
canonical PR-reviewable form.

---

## 2. Skeleton + bindings — `Workflow.from_skeleton(skeleton, bindings)` (G9-completion)

A *parameterized* template. The skeleton declares typed holes; the
caller binds them; the framework lowers + loads. Use when:
- The same workflow shape runs against many parameter sets.
- An LLM agent picks a skeleton from a registry and binds N typed
  holes (the original meta-workflow design's whole point).
- You want one canonical workflow body and many bindings.

```yaml
# rag_skeleton.yml — a Skeleton config, not a Workflow config
skeleton_id: rag_pipeline
skeleton_version: "0.1.0"
description: "RAG pipeline with corpus + query parameters"
holes:
  corpus:
    type: string
    required: true
    description: "name of the indexed corpus to retrieve from"
  min_evidence:
    type: integer
    required: false
    default: 3
body: |
  name: rag_workflow
  version: "0.1.0"
  config_version: 2

  steps:
    retrieve:
      class: "my_pkg.steps.RetrieveStep"
      config:
        corpus: {{ corpus: string }}
        min_evidence: {{ min_evidence: integer }}
    synthesize:
      class: "my_pkg.steps.SynthesizeStep"
      config: "steps/synthesize.yml"

  links:
    # ... auto_transfer: true on every DirectLink
```

```python
from nanobrain.core.workflow import Workflow

wf = Workflow.from_skeleton(
    "rag_skeleton.yml",
    bindings={"corpus": "pubmed_2025_q1", "min_evidence": 5},
)
result = await wf.run({"query": "alphavirus vaccines"})
```

**Trade-offs**: trades verbosity for typed parameterization; missing
required holes FAIL-FAST with a named error; extra binding keys
FAIL-FAST too (no silent typo absorption).

The skeleton can also be passed as a `Skeleton` instance or an
inline dict — useful for tests + programmatic skeleton construction:

```python
sk = {
    "skeleton_id": "rag",
    "skeleton_version": "0.1.0",
    "body": "...",
    "holes": {...},
}
wf = Workflow.from_skeleton(sk, bindings={...})
```

---

## 3. Lightweight WorkflowBuilder — programmatic API

A Python builder that emits a fully-resolved workflow without
writing YAML. Use when:
- Prototyping a workflow shape interactively.
- The DAG shape is computed at runtime (e.g., one step per element
  in a discovered list).
- You're generating many similar workflows from code (parameter
  sweeps, agent-authored fan-out).

```python
from nanobrain.lightweight import WorkflowBuilder

builder = WorkflowBuilder("dynamic_pipeline", description="Built at runtime")

builder.add_step(
    "retrieve",
    component_class="my_pkg.steps.RetrieveStep",
    corpus="pubmed_2025_q1",
)
builder.add_step(
    "synthesize",
    component_class="my_pkg.steps.SynthesizeStep",
    model="mistral-nemo",
)

builder.add_link(
    source="retrieve.docs",
    target="synthesize.context",
    link_type="direct",
    auto_transfer=True,
)

builder.add_trigger(
    step_name="retrieve",
    trigger_class="nanobrain.core.trigger.DataUnitChangeTrigger",
    data_unit="query",
)

# Materialize via the framework's standard loader path.
wf = builder.load()
result = await wf.run({"query": "..."})
```

**Trade-offs**: no YAML on disk; harder to audit / diff in PR;
useful for one-off + computed DAGs; converges on the same
``Workflow.from_config`` runtime so all framework guarantees apply.

---

## Picking a path

| Scenario | Recommended path |
|---|---|
| New workflow that ships in a package, reviewed in PR | (1) Hand-written YAML |
| Workflow run with many parameter sets; one canonical shape | (2) Skeleton + bindings |
| Agent picks a workflow from a typed registry (meta-workflow) | (2) Skeleton + bindings |
| Computed-at-runtime DAG (parameter sweep, dynamic fan-out) | (3) WorkflowBuilder |
| Interactive prototyping in a notebook | (3) WorkflowBuilder |
| HPC bundle replay (deterministic reproduction) | (1) Hand-written YAML — auditable |

---

## Common contracts (apply to all three paths)

- ``config_version: 2`` — explicit pin recommended (lint script in
  apecx-mcp-integration enforces it on integration YAMLs).
- ``auto_transfer: true`` on every ``DirectLink`` — explicit even
  though G7 Step 5 made it the field default. Documentation-as-code.
- Steps own their data units + triggers; workflows own links.
- ``Workflow.run(input_data, timeout, settle_ms)`` is the canonical
  entry point. ``process()`` is fire-and-forget (G35).
- Workflow runs inside a ``WorkflowRunContext`` get per-run namespace
  isolation (G13) and capability-token enforcement (G28).

## Cross-references

- ``nanobrain/CLAUDE.md`` — framework-level rules + recent additions
- ``apecx-mcp-integration/.claude/skills/nanobrain-workflow-authoring/``
  — workflow YAML authoring SKILL doc
- ``apecx-mcp-integration/.claude/skills/nanobrain-lightweight/``
  — WorkflowBuilder ergonomic path SKILL doc
- ``apecx-mcp-integration/docs/development_roadmap.md`` §8.7 G9-completion
  — the post-G9 ergonomic loader rationale
