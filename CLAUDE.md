# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Nanobrain is an event-driven AI agent framework for distributed workflows. It's currently in research preview and has dependencies on HPC systems and external frameworks. The framework uses a mandatory configuration-driven architecture where ALL components are created through the `from_config()` pattern.

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
