# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Nanobrain is an event-driven AI agent framework for distributed workflows. It's currently in research preview and has dependencies on HPC systems and external frameworks. The framework uses a mandatory configuration-driven architecture where ALL components are created through the `from_config()` pattern.

## Recent additions (2026-05-11 — eval_03 arc finalization: adversarial probes + G27 Option B + Rec 4 migration)

This chain finalizes the eval_03 arc opened on 2026-05-09. The
recommendations from the prior session's status report were:
Rec 1 (deploy + collect Option B data — operator action; shipped
framework-native equivalent as an evaluation TOOL), Rec 2 (G27
Option B framework-side wiring), Rec 4 (migrate apecx-mcp's
``_workspace.py`` to nanobrain's G40 helper), plus adversarial
probes for critical bugs.

**Adversarial probes**: 3 silent-failure bugs found + fixed:

- ``CostTracker.record`` accepted NaN — NaN slipped past ``< 0``
  AND ``> cap`` checks (every NaN comparison returns False), silently
  corrupting the ledger. Fixed: explicit ``math.isnan`` + ``math.isinf``
  guards raise ``ValueError`` before any state mutation. Source:
  ``nanobrain/core/cost_envelope.py``.
- ``_subscriber_stack`` contextvar default was a mutable list — a
  tripwire for cross-context mutation bugs. Fixed: default is now
  ``()``; mutation attempts raise ``AttributeError`` immediately.
  Source: ``nanobrain/core/step_events.py``.
- ``WorkflowRunner.resume_suspended`` had a concurrent-call race —
  two callers passing the status check could spawn duplicate asyncio
  tasks for the same task_id. Fixed: guard via ``self._tasks`` dict
  in-flight detection. Source:
  ``nanobrain/library/runtime/workflow_runner.py``.

All three fixes ship with regression tests in
``tests/unit/test_g_arc_adversarial_probes.py`` (9 tests).

**G27 Option B** — opt-in G5 checkpoint integration on
``DeferredHITLStep``:

- ``DeferredHITLStep`` config gains optional ``checkpoint_dir``.
  When set, ``process()`` writes a G5-compatible filesystem manifest
  of ``input_data`` to ``<checkpoint_dir>/<approval_id>.manifest.json``
  before raising ``ApprovalPendingError``.
- ``ApprovalPendingError`` gains ``checkpoint_manifest_handle:
  Optional[str]``. The handle propagates through
  ``suspension_info["checkpoint_manifest_handle"]`` and into the
  resumed workflow's payload via reserved key
  ``__resume_checkpoint_handle__``.
- Workflow authors thread the handle into ``ResumeStep`` to
  short-circuit pre-HITL work — Option B becomes genuine composition
  with the already-shipped G5 primitives. The framework does NOT
  introspect the opaque ``workflow_callable`` to auto-skip steps.
- Behavior preservation: when ``checkpoint_dir`` is absent (default),
  Option A's behavior is bit-for-bit identical — no manifest written,
  handle is None, no reserved key in resumed payload.
- 8 new tests in ``tests/unit/test_g27_option_b_checkpoint.py``,
  including a load-bearing round-trip test against ``ResumeStep``.

**Rec 1 framework-native equivalent — G27 Option B evaluation tool**:

- ``nanobrain/scripts/g27_option_b_eval.py``. Operator-facing CLI tool
  that reads SQLite/Postgres TaskStore and emits a recommendation
  (``stick_with_A`` / ``inconclusive`` / ``promote_B``) based on
  ``max_resume_count × pre_hitl_step_seconds`` vs. configurable
  thresholds. Supports ``--json`` for machine-readable output. 9
  tests in ``tests/unit/test_g27_option_b_eval_tool.py``.

**Rec 4 — apecx-mcp migration to nanobrain G40 helper**:

- ``apecx-mcp-integration/src/apecx_integration/_workspace.py`` now
  delegates to ``nanobrain.library.runtime.workspace_root.locate_workflow_root``,
  retiring G40-WA-1 from the workaround inventory. The
  ``$APECX_WORKSPACE_ROOT`` env var is forwarded to G40's
  ``env_var`` parameter for parity with the prior behavior.

**Net regression status this chain**: 8 new framework-side tests
(Option B) + 9 new framework-side tests (adversarial probes) + 9 new
framework-side tests (Option B eval tool); ``939 passed, 7 skipped,
0 regressions`` for nanobrain unit suite; ``51 passed, 4 skipped``
for nanobrain integration suite.

## Recent additions (2026-05-10 — eval_03 Tier 4 + G31 runner-side wiring + G27↔G21 design)

This chain landed eval_03 Tier 4 (the final 4 deferred items) plus the
G31 runner-side wiring follow-up (auto-install nested
``WorkflowRunContext``) plus a recorded design doc for the G27↔G21
runner-side integration (deferred pending operator decision on
resume-from-start vs. resume-from-step semantics).

**Tier 4 (4 items, all shipped)**:

- **G34** — ``ConfigBase.model_config`` already had
  ``str_strip_whitespace=False`` at ``config_base.py:684``; this chain
  added a regression-pin test
  (``tests/unit/test_g34_strip_whitespace_off.py``, 9 tests) so a
  future refactor that flips it back to True surfaces immediately.
  Paired with apecx-mcp-integration retiring the named-format-enum
  workaround in ``composition/steps/file_readers.py``.
- **G36** — Documented the two-stage whitelist defense (apecx-mcp
  AST scanner = Stage 1 / pre-emit; nanobrain G20 ``import_whitelist``
  = Stage 2 / YAML class-path load). New
  ``apecx-mcp-integration/docs/whitelist_layering.md`` + cross-
  references in both source files. "Folding into one is explicitly
  out of scope" — different stages, different bypass classes.
- **G38** — ``HTTPBackendAdapter`` (``BACKEND_NAME="http"``) at
  ``library/tools/http_backend_adapter.py``. Generic httpx-backed
  adapter for ToolExecutionStep. POST/GET/PUT, default + per-call
  headers, run_context_namespace propagates as
  ``X-Nanobrain-Run-Namespace`` header, JSON / non-JSON / error
  parsing, owned vs. injected client lifecycle. 14 integration tests
  against real httpx ASGITransport.
- **G40** — ``locate_workflow_root`` + ``require_workflow_root`` at
  ``library/runtime/workspace_root.py``. Walks upward looking for
  default markers (``pyproject.toml``, ``.git``, ``setup.py``,
  ``CLAUDE.md``, ``apecx-mcp-integration``); ``$NANOBRAIN_WORKSPACE_ROOT``
  env-var override; returns CLOSEST matching ancestor (deepest, not
  highest); custom marker lists supported. Replaces brittle
  ``Path(__file__).parents[N]`` patterns. 10 unit tests.

**G31 runner-side (follow-up to the primitive shipped earlier)**:

- ``Workflow.run(..., nest_under_active_context=True)`` — explicit
  kwarg (caller-explicit, no implicit auto-detect). When True AND an
  outer ``WorkflowRunContext`` is active, derives the nested namespace
  via ``derive_nested_namespace`` using this workflow's
  ``namespace_strategy``, builds a nested context with
  ``run_id=f"{parent}.{child}"`` for audit correlation, inherits
  parent's ``capability_tokens`` (no privilege drop), activates for
  the run, restores outer on exit. When True without outer context:
  warns + falls through. When False (default): no behavior change.
  7 unit tests pin every branch.

**G27↔G21 wiring (design doc only, deferred)**:

- ``nanobrain/docs/g27_g21_wiring_design.md`` records the choice
  surface for runner-side suspend/resume on ``ApprovalPendingError``.
  Two options analyzed (resume-from-start = deterministic re-run vs.
  resume-from-step = G5 checkpoint integration); recommendation is
  Option A v1 (operators compose with CheckpointStep manually for
  Option B semantics). Solo-implementing one without operator review
  would over-commit the project.

**Files added this chain (register for future Claude sessions)**:
- ``nanobrain/library/tools/http_backend_adapter.py`` — G38
- ``nanobrain/library/runtime/workspace_root.py`` — G40
- ``nanobrain/docs/g27_g21_wiring_design.md`` — G27 wiring design
- ``apecx-mcp-integration/docs/whitelist_layering.md`` — G36 layering

**SKILL files updated** (LLM-guidance for the new primitives):
``nanobrain-workflow-authoring``, ``nanobrain-step-authoring``,
``nanobrain-agents-tools``, ``nanobrain-data-units-triggers-links``,
``nanobrain-config-yaml`` (5 of 9 skill files). Future Claude sessions
loading any of these via the Skill tool see the new primitives.

**Net regression status**: 40 new framework-side tests this chain
(G34: 9, G38: 14, G40: 10, G31-runner: 7); 65+19+23+91 = 198 existing
tests verified non-regressing.

## Recent additions (2026-05-09 — eval_03 Tier 0-3 chain: G4/G9/G11/G24-G28/G31/G33/G35/G37/G39/G43-G45)

This chain landed the entire Tier 0-3 ship-out from
``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md`` plus the
G4/G9/G11 partial-shipment completions that the gap doc explicitly
flagged as deferred. **One developer-session, 16 commits across 2
repos, ~141 new framework-side tests**, all green together.

**Tier 0 (silent-failure closure)**:
- **G33** — ``nanobrain/core/async_logging.py`` and ``logging_system.py``
  ``_default_writable_log_dir()`` now resolves
  ``$NANOBRAIN_LOG_DIR`` -> ``~/.cache/nanobrain/logs/`` -> tempdir.
  Cwd-relative ``Path("logs")`` retired; the apecx-mcp bootstrap
  ``os.chdir(~/.apecx)`` workaround retired.
- **G35** — ``apecx-mcp-integration/control_plane/executors/local.py``
  swaps ``workflow.process({})`` for ``workflow.run({}, timeout=...,
  settle_ms=...)`` so multi-step composed workflows actually drain
  cascades + persist real outputs (not the trigger-init status dict).
  ``cascade_timeout`` / ``no_first_step`` are now terminal failures.
- **G44** — ``DataUnitProxyRef.namespace()`` no longer silently
  returns ``""`` when no ``WorkflowRunContext`` is active; emits a
  rate-limited WARNING per instance, OR raises
  ``ComponentConfigurationError`` under ``NANOBRAIN_STRICT_NAMESPACE=1``.
- **G43** — 14 ``print(f"DEBUG: ...")`` lines stripped from
  ``core/mcp_support.py:_register_mcp_tool``.

**Tier 1 (G4/G9/G11 completions + integration migration)**:
- **G4-completion** — ``BaseStep._execute_process`` now wraps every
  ``process()`` call: resolves ``current_provenance_context()``,
  records inputs / outputs / exception / timing on success AND on
  raise. Recorder failures are swallowed so a buggy recorder cannot
  mask the step's real exception.
- **G9-completion** — ``Workflow.from_skeleton(skeleton, bindings)``
  classmethod. Loads a Skeleton (path / dict / Skeleton), validates
  bindings (FAIL-FAST on missing required / extras), substitutes
  ``{{name: type}}`` tokens, materializes the lowered YAML to a temp
  file, and delegates to existing ``from_config``. Collapses the
  PlanLoweringStep + SkeletonLoaderStep dance into one call —
  unblocks Track B's agent-authored-workflows arc.
- **G11-completion** — ``LocalParslAdapter`` at
  ``library/tools/local_parsl_adapter.py``. ``ToolBackendAdapter``
  with ``BACKEND_NAME="local_parsl"`` that dispatches Python callables
  via Parsl. P0++b decision: ``executor_kind="thread"`` default
  (ThreadPoolExecutor — lowest-overhead, fork-safe, no cluster
  prereqs); ``"process"`` / ``"htex"`` available; bring-your-own
  ``parsl_config`` for HPC. Lazy parsl.load with class-level lock;
  scope() context for clean teardown. 9 integration tests against
  REAL Parsl 2026.5.4.
- **G39** — apecx-mcp-integration migration: every workflow YAML
  declares ``config_version: 2``; every previously-implicit DirectLink
  has explicit ``auto_transfer: true``; ``scripts/lint_workflow_yamls.py``
  pre-commit hook gates future regressions.
- **G45** — ``nanobrain/library/workflows/`` audit: 7 workflow YAMLs
  pinned to ``config_version: 2``; 11 inline DirectLinks across 3
  files got explicit ``auto_transfer: true``.

**Tier 2 (autonomy preconditions)**:
- **G27** — ``DeferredHITLStep`` (``library/steps/deferred_hitl_step.py``)
  + ``ApprovalStore`` protocol with ``InMemoryApprovalStore`` +
  ``FileApprovalStore`` (``library/runtime/approval_store.py``).
  P6+a decision: ``approval_id_strategy="deterministic"`` default
  (SHA-256 of run_id + step_name + prompt) so workflow retries don't
  emit duplicate approvals. Step is idempotent + stateless;
  suspension is signaled by ``ApprovalPendingError``; rejection by
  ``ApprovalRejectedError``. 22 tests parameterized over both backends.
- **G26** — ``CostEnvelope`` + ``CostTracker`` at
  ``core/cost_envelope.py``. P6+b: per-step + per-workflow caps;
  both declarative; both optional. ``record(kind, amount)`` is
  thread-safe + failure-atomic; ``CostEnvelopeBreach`` workflow-
  terminal exception. 15 tests including 8-thread × 1000-record
  contention test that proves the lock prevents bypass.
- **G24** — ``DataSourceRegistry`` at
  ``library/runtime/data_source_registry.py``. P6+c: YAML manifest
  format. ``DataSourceEntry`` with ``content_hash`` (sha256:...) +
  ``compute_content_hash`` helper that hashes files OR directories
  order-stably. ``ContentHashMismatch`` raises on drift. Unknown
  entry keys FAIL-FAST (typo protection).
- **G25** — ``PromptRegressionHarness`` at
  ``library/testing/prompt_regression.py``. Backend-neutral: caller
  plugs in ``async def llm(*, system, user) -> str``. Reads
  ``regression_fixtures`` off a G14 ``PromptTemplate``, validates
  responses against per-fixture contracts (contains / not_contains /
  regex / json_schema / equals). Snapshot mode content-addresses
  by ``(template_id, fixture_index, content_hash)`` so a template
  body change auto-invalidates the snapshot.

**Tier 3 (meta-workflow preconditions)**:
- **G31** — ``WorkflowConfig.namespace_strategy`` field
  (``Literal["scoped","inherit"]``; default ``"scoped"``) +
  ``derive_nested_namespace`` pure helper. P4+a: scoped is the
  default — silent-namespace-collision is a worse failure than
  over-isolation. Multi-tenant isolation now propagates ACROSS
  nesting levels.
- **G28** — ``verify_capability(required, target_name=...)`` at
  ``core/capabilities.py``. P4+b: ``WorkflowRunContext`` is the
  SINGLE source of truth for ``capability_tokens``.
  ``CapabilityNotGranted`` workflow-terminal exception. Strict
  default: no run context = no granted tokens.
  ``ToolExecutionStep.process()`` now checks ``utd.requires_capability``
  BEFORE the adapter is touched.
- **G37** — ``StepEvent`` + ``subscribe_to_step_events`` at
  ``core/step_events.py``. P4+c: v1 schema FROZEN with
  ``event_schema_version: int`` on every event. Three event types:
  ``step_start`` (before process), ``step_complete`` (after success),
  ``step_failed`` (on exception). Subscriber failures swallowed —
  observability never replaces correctness. Composes cleanly with
  G4-completion (different concern: provenance is durable audit;
  step events are live publish stream).

**Net regression status:**
141/141 new framework-side tests pass together; 51+33+77+91+17+10+12+22+1014+15+9+22+11+14 = 358 existing tests verified non-regressing across the chain (where overlapping scopes hit the same test files).

**Cross-repo commits**:
- nanobrain `academy-integration` branch: 12 commits (368cae3, b7a0280,
  305b516, 10d2551, c3b4b86, 005dc87, a128550, b3cf87f handed off
  paired apecx-mcp commits, plus G27/G26/G24/G25/G28/G37/G31).
- apecx-mcp-integration `main` branch: 5 commits (7bd9d2d, 72b3d8d,
  b3cf87f, ff69ac8, 2d1cb1b).

Open questions answered with documented defaults; operators can
override later via the documented surfaces (executor_kind for G11,
approval_id_strategy for G27, namespace_strategy for G31, env-var
NANOBRAIN_STRICT_NAMESPACE for G44).

## Recent additions (2026-05-09 — auto_transfer flip + apecx-setup orchestrator)

- **G7 Step 5 — `LinkConfig.auto_transfer` field default flipped to True.**
  The user's brutal-truth pushback was correct: the False default was
  the original sin that required FOUR migration steps (G7 Step 1-4)
  to clean up. Step 5 is the simpler answer — flip the field default;
  authors who genuinely want a no-op link declare `auto_transfer:
  false` explicitly. The four-step migration becomes a redundant
  safety net (the v2 mutator's `setdefault` is now a no-op for
  omitted keys but stays in place for path-reference rewriting).
  Verified: 773 passed + 5 skipped (0 regressions); adversarial
  probe loop still hits 300/300 zero-bug stop criterion.
- **`apecx-setup` orchestrator** at
  `apecx-mcp-integration/src/apecx_integration/cli/setup.py`. Single
  entry that subsumes data-download + Docker container bring-up
  (Postgres + Redis) + Ollama model pull + FAISS index build +
  verification. Every subcommand idempotent. `apecx-setup verify`
  prints a per-component health table. Container names prefixed
  `apecx-` so they don't collide with test containers. We do NOT
  install Docker/Ollama/gh ourselves — we tell the user exactly
  what's missing and how to install.
- Total this chain: **2 commits across nanobrain + apecx-mcp-integration.**
  Full regression: 773 passed + 5 skipped, 0 regressions.

## Recent additions (2026-05-09 — T-RH-03 + ToolBase.from_python_callable + adversarial harness)

- **T-RH-03 minimum: RheaMCPDispatcher** at
  `nanobrain/library/tools/rhea_mcp_dispatcher.py`. ToolBase subclass
  that materializes from a UTD pointing at it, manages an MCP HTTP
  session against a Rhea worker, and dispatches tool calls via
  JSON-RPC. End-to-end validated against a live Rhea container (12
  tests pass). Two real `from_descriptor` bugs surfaced + fixed:
  dict-input nested-class admittance and inline-config dict→YAML-file
  materialization. **Closes the deferred dispatch loop** opened by
  T-RH-02 in the previous chain.
- **`ToolBase.from_python_callable`** at
  `nanobrain/library/tools/python_callable_dispatcher.py` (lazy-
  imported via the classmethod on `core/tool.py`). Wraps an in-process
  Python callable as a ToolBase via auto-derived UTD. Sync callables
  run in `asyncio.to_thread`; missing/extra payload keys FAIL-FAST.
  15 unit tests.
- **Adversarial probe harness** at
  `tests/adversarial/probe_harness.py`. 14 categories generate
  parameterized probes targeting UTD, ToolBase, WorkflowRunner,
  CheckpointStep/ResumeStep, gate_semantics, lightweight builder,
  TimerTrigger replay, EventTrigger filtering, AllDataReceivedTrigger
  predicate, ConditionalLink predicates, RheaMCPDispatcher SSE parsing,
  FileEntryStateStore CRUD. **Stop criterion satisfied: 300/300
  consecutive zero-bug probes on TWO different seed offsets.** The
  harness's first run surfaced 28 false-positives in MY probe-
  generator (used `gpu_light` instead of `gpu_single|gpu_multi`); the
  framework correctly FAIL-FAST'd; fixed the probe.
- **E2E real-data validation** for the rag_e2e_synthesis pipeline:
  31/32 pass against real Ollama + real FAISS + real VIOLIN/BV-BRC
  CSVs. The single failure is LLM-output quality (synthesizer's
  strict-citation gate refused an mistral-nemo response without
  inline `[N]` markers); NOT a framework bug.
- Total this chain: **1 commit, ~27 new tests, 1 critical
  from_descriptor bug fixed (would have blocked anyone passing a
  dict-form UTD with nested classes).** Full nanobrain regression with
  Postgres + Redis + Rhea up: **777 passed, 1 skipped, 0 regressions.**

## Recent additions (2026-05-09 — Academy/Rhea/tool-wrapping chain)

- **Academy lifecycle bug fixed** at
  `nanobrain/core/academy_integration.py`. ``AcademyAgentHandle.__call__``
  was entering the Manager BEFORE checking whether a real handle was
  registered, leaving a partially-initialized Manager singleton on
  the placeholder fail-fast raise. Subsequent tests in the same
  process deadlocked. Fix: reorder checks so demo-mode + placeholder
  paths fail-fast WITHOUT touching the Manager. Result: full 6-test
  Academy suite passes cleanly in 1.38s.
- **Spawn / scrape / utilize patterns validated** for Academy in
  workflows. New `tests/integration/test_academy_in_workflow.py` (11
  tests): dynamic agent registration from inside workflow code;
  Python-side action introspection (`_scrape_agent_actions`); dispatch
  via attribute + `__call__` syntax; end-to-end orchestrator step
  that spawns + scrapes + utilizes in one `process()` call.
- **`UnifiedToolDescriptor.from_python_callable` factory** at
  `nanobrain/core/unified_tool_descriptor.py`. Mirrors Rhea/FastMCP's
  auto-generation: introspects `inspect.signature` + docstring +
  type annotations to derive a UTD from a Python function. Author
  overrides any field via kwargs. 18 unit tests at
  `tests/unit/test_utd_from_python_callable.py`.
- **Rhea MCP server brought up** in Docker with minimum-viable
  dependency set (just Redis on port 6379). MCP HTTP transport
  responds at `http://localhost:3001/mcp/` with proper session
  handshake. Postgres + MinIO + embedding services are NOT required
  for tool-host functionality.
- **Cross-framework end-to-end integration test** at
  `tests/integration/test_cross_framework_deployment.py` (4 tests):
  live MCP discovery against the Rhea worker; Rhea→UTD wire-format
  conversion validated by nanobrain's UTD validator; ONE workflow
  via WorkflowRunner.run_detached doing both Rhea discovery AND
  Academy dispatch. Gated on `RHEA_MCP_URL` env var.
- **Rhea-side T-RH-02 minimum** at
  `apecx-cowork/rhea/rhea/extensions/apecx_utd_extension/utd_producer.py`
  (commit `23c876b` on apecx-integration branch). Pure-Python wire-
  format converter; does NOT import nanobrain (cross-framework
  contract is the dict shape). FastMCP `Tool` → UTD dict consumable
  by `UnifiedToolDescriptor.from_dict`.
- Total this chain: **2 commits across nanobrain + rhea, ~33 new
  tests, 1 critical lifecycle bug fixed.** Full nanobrain regression
  with Postgres + Redis + Rhea up: **750 passed, 1 skipped, 0
  regressions.**

## Recent additions (2026-05-09 — deployment-validation chain)

- **Ruff lint promoted to gating.** Auto-fix sweep applied 1023 safe
  fixes across 218 files; ``[tool.ruff]`` config in ``pyproject.toml``
  declares 12 documented category-level ignores (each with a deferred-
  cleanup note); ``[tool.ruff.lint.per-file-ignores]`` carves out
  legacy modules (``cleanup/**``, ``demos/**``, ``examples.py``);
  ``.github/workflows/tests.yml`` lint job is now gating
  (continue-on-error removed). Two real bugs surfaced + fixed:
  G15 UTD re-export block in ``core/tool.py`` (test imported names
  ruff dropped as "unused"); ``import sys`` in
  ``elasticsearch_mcp_server.py`` (used by ``__main__`` block).
- **Postgres validated end-to-end.** Spun up Postgres 16 in Docker;
  all 15 G21 Step 4 tests passed including the 2 integration tests
  (``test_full_lifecycle_against_real_postgres``,
  ``test_runner_with_postgres_backend_via_from_config``) that were
  previously skipped without ``POSTGRES_TEST_DSN``.
- **Docker sandbox runtime validated.** Real bug surfaced + fixed in
  apecx-mcp-integration (``--security-opt seccomp=default`` is not
  a Docker keyword; Docker Desktop on Mac treats it as a file path).
  After fix, all 27 sandbox tests pass — including the 4 runtime
  integration tests gated on ``APECX_T13B_SANDBOX_EXECUTE=1``.
- **Academy integration verified in isolation.** Single Academy test
  passes in 0.66s against a real local Academy agent. Honest
  caveat: the full 6-test suite has cross-test contamination — the
  ``test_unregistered_agent_raises_not_implemented`` placeholder
  hangs the Academy Manager singleton. This is a test-fixture bug
  in apecx-mcp-integration, NOT a framework-side bug.
- **Rhea apecx-integration scaffold validates inside container.**
  Rhea Docker image builds cleanly with the apecx fork; ``import
  rhea.extensions.apecx_utd_extension`` works at runtime in the
  built container. Full Rhea production stack bring-up requires
  ``.env.docker`` + 4 backing services + multi-GB embedding model
  pull (operator scope; documented in Rhea's deploy README).
- **Mixed-deployment integration suite (5 tests, 0.92s).** Validates
  G5 + G7 + G10 + G21 + lightweight WorkflowBuilder all interacting
  in one workflow shape: prep → checkpoint → resume → consume; same
  pipeline driven by run_detached; cross-runner SQLite durability;
  WorkflowBuilder generating v2-compatible workflow with mutators
  fired; long-running detached workflow with heartbeat + checkpoint.
  Pre-this-chain, NO test exercised all five primitives together.
- Total this chain: **6 commits across nanobrain + apecx-mcp-integration,
  ~10 new tests, 1 sandbox real-bug fix.** Full nanobrain regression
  with both Postgres + Redis up: **717 passed, 1 skipped, 0
  regressions ever.**

## Recent additions (2026-05-09 — infra chain)

- **Infrastructure-validated.** PostgresTaskStore (G21 Step 4) and
  RedisKey rebuild (G5 Step 2) are now end-to-end tested against
  real Postgres 16 and Redis 7 containers. The skipped-without-env-var
  contract is preserved; CI runs the same path automatically.
- **G21 Step 5 — BaseStep automatic PauseSignal cooperation.**
  ``BaseStep._execute_process`` now consults ``current_pause_signal()``
  before each ``process()`` call. User step code does NOT need to
  consult the contextvar manually for pause to work — the framework
  cooperates automatically at step boundaries. Closes the last G21
  deferral. Layering compliance: lazy + cached import of the runtime
  module from core/step.py (core MUST NOT depend on library statically;
  the helper degrades to no-op when the runtime module is unavailable).
- **G5 Step 2 RedisKey rebuild.** ProxyStore Redis-backed checkpoints
  now round-trip cross-process the same way FileKey does. Manifest
  carries connector hints (redis_host + redis_port); fresh ResumeStep
  re-registers the Store from hints. ``proxystore_connector_kind``
  expanded from ``Literal['file']`` to ``Literal['file', 'redis']``.
- **GitHub Actions CI.** New ``.github/workflows/tests.yml`` runs the
  full unit suite on push + PR for Python 3.12 with Postgres 16 +
  Redis 7 service containers. The "0 regressions" claim is now
  CI-enforced. New ``[test]`` extra in ``pyproject.toml`` declares
  the canonical test stack (pytest + pytest-asyncio + pytest-cov +
  proxystore + psycopg[binary]).
- **Lightweight WorkflowBuilder hardening.** Audit surfaced three
  silent-failure shapes: dead ``version: '2.0'`` field (framework
  reads ``config_version``); discovery only finding DirectLink and
  zero triggers; no ``add_trigger`` API at all. All fixed. New
  ``add_link()`` (full link-type discrimination), ``add_trigger()``
  (workflow-level + step-level), ``load()`` (closes the loop via
  ``Workflow.from_config``). Framework class paths resolved via a
  static map; discovery is the fallback for user-defined classes.
- Total this chain: **5 commits + 1 CI workflow + ~37 new unit
  tests**. Full nanobrain regression: **712 passed with both Postgres
  + Redis up**, 1 skipped (the pre-existing framework skip), 0
  regressions.

## Recent additions (2026-05-09)

- **All 22 nanobrain capability gaps + every documented Step 2-4
  follow-up now shipped.** Combined diff this date covers:
  - **G5 Step 2** — ProxyStore Key cross-process serialization
    (NamedTuple `_asdict` + Store re-registration from manifest hints
    in `nanobrain/library/steps/checkpoint_resume.py`).
  - **G5 Step 3** — `ResumeStep on_missing='rebuild'` resolves a
    dotted-path callable, regenerates the data, writes a fresh
    manifest so subsequent resumes hit the cache.
  - **G7 Step 4** — workspace-wide v2 default + path-reference YAML
    rewriting. `WorkflowConfig.config_version` default flipped from
    v1 → v2; `_apply_v2_link_defaults` now loads + injects + rewrites
    `config: "external.yml"` link entries.
  - **G10 Step 2** — workflow-level `gate_semantics` propagation to
    every inline ConditionalLink and AllDataReceivedTrigger via a
    `model_validator`. Path-reference configs honored under v2.
  - **G21 Step 2** — cooperative pause via `PauseSignal` contextvar
    + `WorkflowRunner.pause` / `.resume` / `.is_paused`. Bonus:
    `await_completion(timeout=...)` fixed to wrap in `asyncio.shield`
    so the timeout no longer silently cancels in-flight tasks.
  - **G21 Step 3** — heartbeat watchdog + stale-task reaper on
    `WorkflowRunner` (lazy-started; `heartbeat_interval_seconds=0`
    disables; race-condition handler keeps the watchdog's `failed`
    verdict when CancelledError fires after).
  - **G21 Step 4** — Postgres durability backend. `_TaskStore` →
    public `TaskStore`; `PostgresTaskStore` ships with psycopg 3 as
    a lazy/optional import. Integration tests gated on
    `POSTGRES_TEST_DSN` env var.
  - **G22 Step 2** — `target_workflow` dotted-path resolution (callable
    OR `.run`-bearing instance; classes deliberately rejected).
  - **G22 Step 3** — missed-schedule policy (`skip|catch_up|merge`)
    on TimerTrigger + WorkflowEntryTrigger. Off-by-one fix via
    integer-millisecond arithmetic.
  - **G22 Step 4** — durable inner-trigger → launch binding via
    `EntryStateStore` (in-memory + file-backed). Persists last-fire
    timestamp; `recover_from_durable_state()` drives `replay_missed_fires`
    at startup.
  - Total this date: **8 commits, 11 new test files, ~98 new unit
    tests**. Full nanobrain regression: 673 passed, 3 skipped, 0
    regressions ever across 5 chains.
  - See `apecx-mcp-integration/docs/WORKAROUND_INVENTORY.md §8` for
    the per-gap shipping log and the (now empty) deferred-follow-up
    list.

- **G22 v1 — EventTrigger + WorkflowEntryTrigger** shipped:
  - `EventTrigger` at `nanobrain/core/trigger.py` (transport-agnostic;
    callers invoke `fire_event(body)` from a webhook handler or
    message-bus consumer). Optional G1 `event_filter` predicate dict
    on `TriggerConfig.event_filter` gates fires. `TriggerType.EVENT`
    enum value added.
  - `WorkflowEntryTrigger` at `nanobrain/library/runtime/entry_triggers.py`
    wraps any inner TriggerBase to launch a detached workflow run via
    G21's WorkflowRunner. Auto-generates task IDs, optionally calls
    a dotted-path-resolved `payload_factory`, forwards `autonomy_level`
    and `cost_envelope_template` into payload metadata as
    `__autonomy_level__` / `__cost_envelope_template__`. Optional
    `on_launch(handle)` callback for caller-side task tracking.
  - 21 unit tests across two new files; full regression 557 passed,
    1 skipped, 0 regressions. All 22 capability gaps now shipped.

- **G21 v1 — WorkflowRunner.run_detached** shipped at
  `nanobrain/library/runtime/workflow_runner.py` (new subpackage).
  `run_detached(workflow_callable, task_id, payload)` schedules an
  asyncio task and returns the queued `DetachedTaskHandle` immediately.
  Two task store backends: `in_memory` (default) and `sqlite` (stdlib
  `sqlite3`). Lifecycle states: queued → running → (completed |
  cancelled | failed). `cancel(task_id)` cooperatively cancels the
  underlying asyncio task. `await_completion(task_id, timeout)` is a
  test/sync convenience join-point. `pause` is reserved for Step 2
  (raises NotImplementedError; needs step-level cancellation hook in
  BaseStep). Cross-process resume + heartbeat watchdog deferred to
  Steps 3-4. 18 unit tests at `tests/unit/test_workflow_runner.py`,
  full regression 536 passed, 1 skipped, 0 regressions.

- **G7 Step 3 — v2 auto_transfer default flip (active)** shipped at
  `nanobrain/core/workflow.py`: `WorkflowConfig._apply_v2_link_defaults`
  Pydantic `model_validator(mode='after')` mutates inline link configs
  to set `auto_transfer: True` when `config_version >= 2`. Both flat
  (`{class, source, target, ...}`) and nested (`{class, config: {...}}`)
  shapes are handled. Explicit values (True OR False) are preserved
  via `setdefault`. Path-reference configs (where `config:` is a string
  YAML path) are NOT mutated — that is Step 4 scope. AcademyLink and
  unknown link classes are skipped (whitelist via
  `_link_class_needs_auto_transfer_check`). 11 new tests at
  `tests/unit/test_workflow_auto_transfer_warning.py` (40 total in
  that file); full regression 518 passed, 1 skipped, 0 regressions.

- **G10 Step 1 — gate-to-bottom semantics (mechanism)** shipped at
  `nanobrain/core/link.py` + `nanobrain/core/trigger.py`:
  `ConditionalLink.GATED_OFF_SENTINEL = "__nanobrain_gated_off__"`,
  `LinkConfig.gate_semantics` and `TriggerConfig.gate_semantics`
  fields (default `"publish_empty"` for legacy compat). Under
  `gate_to_bottom`, a False ConditionalLink writes the sentinel to
  the target; AllDataReceivedTrigger's new `_is_satisfied(payload)`
  predicate counts the sentinel as satisfied AND excludes it from
  the trigger payload — fan-in proceeds with N-1 keys instead of
  deadlocking. User `process()` never sees the magic string. Step 2
  (workflow-default propagation) and Step 3 (default-flip in v2)
  remain pending. 16 unit tests at
  `tests/unit/test_gate_to_bottom.py`, all green; G1+G2 regression
  re-run: 73 passed, 1 skipped, 0 regressions.

- **G5 — CheckpointStep + ResumeStep primitives** shipped at
  `nanobrain/library/steps/checkpoint_resume.py`: content-addressed
  filesystem snapshots (default backend) + manifest with code-identity
  capture + atomic write (write-tmp-then-rename). `ResumeStep` validates
  manifest version, SHA-256 content hash on read, and supports
  `on_missing: fail|skip|rebuild` (rebuild is `NotImplementedError` until
  G5 Step 3). Streams (async iterators) are FAIL-FAST'd at capture time.
  ProxyStore backend exists but is in-process-only for v1 (Key
  serialization deferred). 20 unit tests at
  `tests/unit/test_checkpoint_resume.py`, all green.

- **G1 — Declarative ConditionalLink predicate DSL** shipped at
  `nanobrain/core/link.py`: `PredicateConfig` (Pydantic, `extra: forbid`,
  fixed op vocabulary `eq | ne | in | contains | exists | all | any | not`),
  `evaluate_predicate` (FAIL-FASTs on dotted-path miss when `op != "exists"`),
  `get_nested_value_strict` (uses `_PATH_MISS` sentinel to distinguish
  legitimate `None` payloads from missing fields). Backwards-compatible:
  legacy `field/operator/value` dicts and bare-string conditions still work
  with deprecation warnings. Tests at `tests/unit/test_link_predicates.py`
  (48 tests, all green). See `apecx-mcp-integration/docs/nanobrain_capability_gaps.md G1`
  for the proposal.

## ⚠️ Dominant silent-failure shape

**`DirectLink` (and other `LinkBase` subclasses) defaults `auto_transfer=False`.**
Without an explicit `auto_transfer: true` in YAML, the link silently no-ops:
the workflow loads, every step runs, no exception, but no data ever
transfers. Gap **G7** (`apecx-mcp-integration/docs/nanobrain_capability_gaps.md`)
proposes flipping the default in `config_version: 2`. Until then, every
hand-authored link YAML must include the flag. The
`.claude/skills/nanobrain-data-units-triggers-links/SKILL.md` carries the
full warning.

## Core Architecture Principles

### The `from_config()` Pattern (MANDATORY)

**ALL components in Nanobrain MUST be created using `from_config()` - direct instantiation is explicitly forbidden.**

```python
# ✅ REQUIRED - Configuration-based creation
agent = ConversationalAgent.from_config('config/agent.yml')
step = MyStep.from_config('config/step.yml')
workflow = Workflow.from_config('config/workflow.yml')

# ❌ FORBIDDEN - Direct instantiation (will raise runtime errors)
agent = ConversationalAgent(name="test")  # NEVER DO THIS
step = MyStep()  # NEVER DO THIS
```

All components inherit from `FromConfigBase` (`nanobrain/core/component_base.py`), which enforces this pattern through:
- Constructor prohibition using `__new__` override
- YAML-first configuration loading
- Recursive dependency resolution
- Schema validation via Pydantic

### Component Ownership Rules

**Both workflows and steps can define their own data units. Steps own their data units and triggers. Workflows manage links between steps and can define workflow-level data units.**

```yaml
# Step configuration (config/data_preparation_step.yml)
# Real example from demos/academylink_aurora_demo/
name: data_preparation_step
description: "Local data preparation step"

# Step OWNS these data units
input_data_units:
  raw_input:
    class: "nanobrain.core.data_unit.DataUnitMemory"
    name: "raw_input"
    description: "Raw input data for preparation"
    persistent: false

output_data_units:
  prepared_data:
    class: "nanobrain.core.data_unit.DataUnitMemory"
    name: "prepared_data"
    description: "Prepared data for heavy computation"
    persistent: false

# Executor configuration
executor:
  class: nanobrain.core.executor.LocalExecutor
  config: local_step_executor.yml
```

```yaml
# Workflow configuration (config/mixed_execution_workflow_aurora.yml)
# Real example from demos/academylink_aurora_demo/
name: academylink_aurora_workflow
version: "2.0"

# Workflow-level input/output data units
input_data_units:
  raw_input:
    class: "nanobrain.core.data_unit.DataUnitMemory"
    name: "raw_input"
    description: "Raw input data for the workflow"
    persistent: false

output_data_units:
  final_results:
    class: "nanobrain.core.data_unit.DataUnitMemory"
    name: "final_results"
    description: "Final aggregated results from the workflow"
    persistent: false

# Workflow defines executors for different steps
executors:
  local_executor:
    executor_type: local
    name: local_executor
    max_workers: 2
    timeout: 60

  aurora_executor:
    executor_type: parsl
    name: aurora_executor
    parsl_config_file: ../aurora_parsl_executor.yml
    timeout: 600

# Steps reference their own configs and specify which executor to use
steps:
  data_preparation:
    class: demos.academylink_aurora_demo.steps.DataPreparationStep
    config: config/data_preparation_step.yml
    executor: local_executor

  aurora_computation:
    class: demos.academylink_aurora_demo.steps.AuroraComputationStep
    config: config/aurora_computation_step.yml
    executor: aurora_executor

  result_aggregation:
    class: demos.academylink_aurora_demo.steps.ResultAggregationStep
    config: config/result_aggregation_step.yml
    executor: local_executor

# Workflow's responsibility: LINKS between steps
links:
  # Academy computation link (for distributed execution)
  aurora_computation_link:
    class: "nanobrain.academy_integration.academy_link.AcademyLink"
    config: "config/aurora_computation_link.yml"

  aurora_results_link:
    class: "nanobrain.academy_integration.academy_link.AcademyLink"
    config: "config/aurora_results_link.yml"

execution:
  timeout: 600
  retry_attempts: 2
  parallel_execution: false
```

### Academy Link Configuration (for Distributed Execution)

**Academy links enable data transfer between local and distributed (HPC) components:**

```yaml
# Link configuration (config/aurora_computation_link.yml)
# Real example from demos/academylink_aurora_demo/
class: "nanobrain.academy_integration.academy_link.AcademyLink"
name: aurora_computation_link
link_type: academy
academy_agent_handle: aurora_computation_agent
action_name: process
source: "data_preparation.prepared_data"  # From local step
target: "aurora_computation.aurora_input"  # To HPC step
timeout_seconds: 300
retry_attempts: 3
auto_transfer: true
proxystore_enabled: true
proxystore_store_dir: "/home/onarykov/proxystore_academylink_aurora"
proxystore_store_name: "academylink-aurora-workflow"
proxystore_connector_type: "file"
```

### Event-Driven Data Flow

Data flows through the system via this mandatory pattern:

```
1. Data deposited → Input DataUnit
2. Trigger activates → Step executes
3. Step processes → Output DataUnit
4. Link transfers → Next Input DataUnit
5. Repeat until workflow complete
```

Steps never call each other directly - all communication happens through DataUnits, Triggers, and Links.

### Method Responsibility Matrix

**Steps must implement `process()` with business logic and should NOT override `execute()`:**

| Method | Responsibility | Subclass Should |
|--------|---------------|-----------------|
| `execute()` | Infrastructure (environment setup, data collection, executor delegation) | ❌ NOT override (except extraordinary circumstances) |
| `process()` | Business logic (processing, algorithms, transformations) | ✅ ALWAYS implement |
| `_execute_process()` | Internal bridge (delegates to process) | ❌ NEVER override |

```python
# ✅ CORRECT
class MyStep(BaseStep):
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """ALL business logic goes here"""
        # Process data, apply transformations, etc.
        return processed_result

# ❌ WRONG
class BadStep(BaseStep):
    async def execute(self, input_data):  # DON'T override execute()
        return self._do_processing(input_data)
```

## Key Architectural Components

### Core Abstractions
- **Agents** (`core/agent.py`): AI entities with LLM integration and tool calling
- **Steps** (`core/step.py`): Data processing units, base class for workflows
- **Workflows** (`core/workflow.py`): Specialized steps that orchestrate multiple steps via DAG
- **Data Units** (`core/data_unit.py`): Type-safe, event-driven data containers
- **Triggers** (`core/trigger.py`): Event activation mechanisms (data changes, conditions, timers)
- **Links** (`core/link.py`): Define data flow between components
- **Tools** (`core/tool.py`): Capability extensions for agents/steps
- **Executors** (`core/executor.py`): Execution backends (local, thread, process, Parsl)

### Framework Integration
- **WorkflowGraph** (`core/workflow_graph.py`): DAG management, cycle detection, topological sort
- **WorkflowValidator** (`core/workflow_validation.py`): Structural validation
- **AsyncTriggerExecutor** (`core/trigger.py`): Non-blocking trigger execution, deadlock prevention
- **A2A Protocol** (`core/a2a_support.py`): Agent-to-Agent collaboration (Google spec)
- **MCP Support** (`core/mcp_support.py`): Model Context Protocol integration
- **Academy Integration** (`academy_integration/`): Integration with Academy distributed framework
- **Parsl Support** (`core/distributed/workflow_execution.py`): HPC distributed execution

### Directory Structure
- `nanobrain/core/` - Core framework abstractions (mandatory from_config pattern)
- `nanobrain/library/` - Reusable implementations (agents, workflows, tools, steps)
- `nanobrain/academy_integration/` - Academy distributed computing integration
- `nanobrain/config/` - Configuration system and templates
- `nanobrain/lightweight/` - Minimal framework for constrained environments
- `config/` - Configuration files (YAML)
- `demos/` - Demo implementations and examples
- `tests/` - Test suite (unit, integration, performance, playwright)

## Development Commands

### Installation
```bash
# Development installation
pip install -e .[dev]

# With LLM support (requires API keys)
pip install -e .[llm]

# With distributed computing (requires HPC)
pip install -e .[distributed]

# Everything
pip install -e .[all]
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/core/          # Core framework tests
pytest tests/performance/   # Performance tests

# Run with markers
pytest -m "not slow"        # Skip slow tests
pytest -m integration       # Only integration tests
pytest -m unit              # Only unit tests
```

### Code Quality
```bash
# Format code
black nanobrain/

# Lint code
flake8 nanobrain/

# Type checking
mypy nanobrain/

# Alternative (using ruff)
ruff check nanobrain/
ruff format nanobrain/
```

### Building Documentation
```bash
cd docs
sphinx-build -b html source build/html
```

## Critical Rules for Code Generation

When working with this codebase, you MUST follow these rules (see `docs/LLM_CODE_GENERATION_RULES.md` for complete details):

1. **ALL objects from YAML configuration files ONLY** - No direct instantiation
2. **Steps own their data units and triggers** - Workflows only manage links
3. **Workflows are steps with links** - They inherit from Step
4. **Store all prompts in configuration files** - No hardcoded prompts in code
5. **Minimize complexity while following framework rules** - Reuse existing components
6. **Everything is configurable** - No hardcoded values

### Forbidden Patterns
```python
# ❌ NEVER create objects directly
config = DataUnitConfig(name="test")
data_unit = DataUnitMemory()
agent = Agent(model="gpt-4")

# ❌ NEVER use ComponentFactory (it was removed)
create_component(...)  # This no longer exists

# ❌ NEVER hardcode configurations
step = Step.from_config({'name': 'processor'})  # Should be YAML file

# ❌ NEVER mix responsibilities
# Workflow managing step's data units - WRONG
# Step managing its own data units - CORRECT
```

### Required Patterns
```python
# ✅ ALWAYS use from_config with YAML files
component = ComponentClass.from_config('config/component.yml')

# ✅ ALWAYS implement process() in steps, not execute()
class MyStep(BaseStep):
    async def process(self, input_data, **kwargs):
        # Business logic here
        return result

# ✅ ALWAYS define triggers for event-driven execution
triggers:
  - trigger_type: "data_updated"
    data_unit: "input_data"
```

## Configuration System

### Configuration File Path Resolution
Paths are resolved in this order:
1. Absolute paths
2. Relative to calling class's directory
3. Relative to class parent directory
4. Relative to current working directory
5. Relative to workflow base directory

### Recursive Component References
```yaml
# Use class + config pattern for nested components
agent:
  class: "nanobrain.core.agent.ConversationalAgent"
  config: "config/agent.yml"

tools:
  - class: "nanobrain.library.tools.WebSearchTool"
    config: "config/web_search.yml"
```

### Environment Variable Interpolation
```yaml
model: "${MODEL_NAME:-gpt-3.5-turbo}"
api_key: "${OPENAI_API_KEY}"
debug: "${DEBUG_MODE:-false}"
```

## Important Context

### Research Framework Status
This is a research framework in active development with:
- Hardcoded paths and environment-specific configurations
- External dependencies on HPC systems (Parsl, Academy)
- Mock implementations for many distributed features
- Breaking changes expected in future versions

### Known Issues
- Hardcoded paths throughout codebase
- Missing `__init__.py` files in some directories
- Circular imports in some modules
- No proper error handling for missing dependencies
- Configuration files not packaged properly

### HPC and Distributed Execution
Many features require:
- Aurora supercomputer access (for some demos)
- Academy framework installation (proprietary)
- Parsl configuration for distributed execution
- Specific conda environment setup

When working on distributed features, be aware these may not work in all environments.

## Testing Philosophy

- Unit tests in `tests/unit/`
- Integration tests in `tests/integration/`
- Core framework tests in `tests/core/`
- Performance tests in `tests/performance/`
- Web interface tests using Playwright in `tests/playwright/`

Tests should follow pytest conventions and use async where appropriate (`pytest-asyncio`).

## Key Files to Understand

- `nanobrain/core/component_base.py` (867 lines) - FromConfigBase, mandatory pattern enforcement
- `nanobrain/core/workflow.py` (~600 lines) - Workflow orchestration
- `nanobrain/core/agent.py` (~500 lines) - Agent with LLM integration
- `nanobrain/core/step.py` (~400 lines) - Step processing
- `nanobrain/core/data_unit.py` (~500 lines) - Data unit system
- `nanobrain/core/trigger.py` (~400 lines) - Event-driven triggers
- `nanobrain/core/config/config_base.py` (~400 lines) - Configuration loading
- `docs/LLM_CODE_GENERATION_RULES.md` (570 lines) - Mandatory code generation rules

## Working with the Codebase

1. **Study the functioning example** - See `demos/academylink_aurora_demo/config/mixed_execution_workflow_aurora.yml` for a complete, working workflow configuration
2. **Read `docs/LLM_CODE_GENERATION_RULES.md` first** - Contains mandatory patterns
3. **Always create YAML configs** - Never hardcode configurations
4. **Follow the from_config pattern** - No exceptions
5. **Respect component ownership** - Both workflows and steps can define data units; steps own their triggers, workflows manage links
6. **Implement process(), not execute()** - Keep business logic separate from infrastructure
7. **Use event-driven data flow** - No direct step-to-step calls
8. **Test with pytest** - Write tests following existing patterns
9. **Consider HPC context** - Some features only work in specific environments

## Reference Implementation

The most complete and functioning workflow example is located at:
- **Workflow**: `demos/academylink_aurora_demo/config/mixed_execution_workflow_aurora.yml`
- **Step configs**: `demos/academylink_aurora_demo/config/data_preparation_step.yml` and related files
- **Link configs**: `demos/academylink_aurora_demo/config/aurora_computation_link.yml` and related files

This demonstrates:
- Mixed execution (local + distributed HPC via Parsl)
- Workflow-level and step-level data units
- Multiple executors (local and Aurora HPC)
- Academy links for distributed data transfer
- ProxyStore integration for large data handling
