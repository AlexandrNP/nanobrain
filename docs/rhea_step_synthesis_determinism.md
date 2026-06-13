# Rhea step synthesis + determinism wire (E2-R)

Proposed capability-gap id: **G130** (cite this when registering the gap in
`apecx-mcp-integration/docs/nanobrain_capability_gaps.md`).

This document records the design shipped under task E2-R: turning a Rhea/Galaxy
tool *name* into a from_config-ready nanobrain `Step` with **honest, deterministic
provenance**, and surfacing the determinism pins Rhea already carries but the base
MCP wire dropped.

## What was rigid before

1. Nothing turned "a Galaxy tool name" into a workflow-usable Step. `RheaMCPDiscovery`
   produced UTD dicts and `ToolExecutionStep`/`RheaFileToolStep` could run them, but
   the glue (discover → pick UTD → branch file-vs-JSON → emit a step config) did not exist.
2. The two tool-shape paths were disjoint: `ToolExecutionStep` (JSON inputs, `RheaAdapter`)
   vs. `RheaFileToolStep` (file inputs, ProxyStore) — with no synthesizer choosing between them.
3. Every determinism pin was dropped at the Rhea MCP wire. `RheaMCPDiscovery` hardcoded
   `version=1.0.0`, `determinism=R3`, `side_effects=network` for **every** tool and never
   set `container_image_digest`. A safe-but-false default that lies about provenance.

## The synthesizer seam (Priority 1)

`nanobrain.library.tools.rhea_step_synthesizer.synthesize_rhea_step(tool_name, ...)`
(async) returns a `RheaStepSpec`:

- discovers the worker catalog (`RheaMCPDiscovery`), optionally calling `find_tools(query)`
  first because Rhea's catalog is **dynamic** (a tool is absent from `tools/list` until a
  semantic query surfaces it into the session);
- selects the UTD matching the name (FAIL LOUD with the available names otherwise);
- **branches file-vs-JSON on the authoritative `file_input_args` discriminator** (see below);
- emits either a `ToolExecutionStep` config (`tool_descriptor=<utd>`, backend=`rhea`) or a
  `RheaFileToolStep` config (`tool_name`/`find_tools_query`/`file_input_arg`/…).

The sync DAG-assembly half is `WorkflowBuilder.add_rhea_step(step_name, spec, **overrides)`.
Discovery/synthesis (network I/O) stays async and separate from DAG assembly (sync) — a
clean seam that does not force async into the otherwise-sync builder.

### Why file-vs-JSON cannot be a heuristic

A Galaxy `<param type="data">` (a FILE input) serializes into the MCP `inputSchema` as an
indistinguishable `{"type": "string"}`. There is **no marker** in the base inputSchema that
says "this string is a file path / redis key". A heuristic on the inputSchema would
*guess*, and a wrong guess does real damage (a JSON config passes a raw string where the
tool needs a staged redis_key, or vice-versa). So the discriminator is the worker's
**authoritative** `file_input_args` list (the Galaxy `type="data"` param names), surfaced
via the determinism wire below. When the worker did not surface it (an old worker) and the
caller gave no explicit `file_input_args=` override, synthesis **FAILS LOUD** — it never guesses.

## The determinism wire (Priority 2)

### Rhea side (`rhea` repo)

`rhea/rhea/extensions/apecx_utd_extension/provenance_annotations.py`:
`build_apecx_provenance(galaxy_tool) -> dict` extracts the determinism-pinning metadata the
Galaxy `Tool` already carries — `version`, `requirements` (versioned), `containers` (image
refs/digests), `version_command`, and `file_input_args` — plus an explicit `stochastic` flag
(default False; Galaxy XML has no determinism field, and bioinformatics Galaxy tools are
overwhelmingly deterministic algorithms — a wrapper for a known-stochastic tool sets this
True). The block rides on the MCP `ToolAnnotations` field (which is `extra="allow"`, so the
MCP schema is not broken) under the key `apecx_provenance`. Wired into `create_tool`
(`rhea/server/utils.py`) so `tools/list` carries it. Best-effort: a malformed Tool degrades
to a title-only annotation rather than breaking tool creation.

### nanobrain side (`nanobrain` repo)

`RheaMCPDiscovery._mcp_tool_to_utd` reads `annotations.apecx_provenance` and builds an HONEST UTD:

| field | honest rule |
|---|---|
| descriptor version | real worker version (sanitized to the UTD grammar); empty → `@unpinned` (NOT a false `@1.0.0`) |
| `container_image_digest` | set **only** when the container value is a real OCI digest (`@sha256:`); a mutable `image:tag` is recorded as `mcp_support.container_image_ref`, never masqueraded as a digest |
| `determinism` | `stochastic` flag → R3; else versioned **and** containerized → R2 (reproducible up to FP — the strongest honest claim from Galaxy metadata; R1 bit-exactness is not asserted); else → R3 (unpinned/unknown) |
| `side_effects` | containerized tool → `filesystem_write`; no container (a pure MCP function: find_tools, a search API) → `network` |
| `file_input_args` | stashed in `provenance_pin.mcp_support` for the synthesizer; absent → synthesizer FAILS LOUD on the file-vs-JSON branch |

An old worker (no `apecx_provenance`) yields an honest `@unpinned` / R3 / network UTD — and
no `file_input_args`, so the synthesizer FAILS LOUD rather than guessing.

## Priority 3 — designed, NOT built

### Content-addressed file staging (G24 `content_hash` into the Rhea ProxyStore path)

Today `RheaFileToolStep` stages input bytes via `RheaFileProxy.from_buffer(name, bytes, redis)`
which yields a redis_key derived from the logical file name. Two different byte payloads with
the same name collide; an identical payload re-staged gets a fresh key (no dedup, no cache).

**Design**: compute `content_hash = sha256(bytes)` (reuse `nanobrain.library.runtime.data_source_registry.compute_content_hash`, the G24 helper) and incorporate it into the ProxyStore key
(`rhea-input/<content_hash>`). Effects: (a) identical inputs dedup to one staged object — a
content-addressed cache that makes a re-run of the same alignment a no-op stage; (b) the
provenance record can pin the exact input bytes by hash, closing the audit loop with the UTD's
`container_image_digest`. This needs a small Rhea-side change to `RheaFileProxy` (accept/honor a
caller-supplied key) plus a nanobrain-side hash compute; deferred because it touches the Rhea
ProxyStore key contract and warrants its own real-data verification against a live worker.

### Seed surface for inherently-stochastic tools

The determinism wire honestly classifies a `stochastic: true` tool as R3, but offers no way to
make such a tool *reproducible* by pinning its random seed. **Design**: add an optional
`seed_param: str | None` to `apecx_provenance` (the Galaxy param name that controls the tool's
RNG seed). When present, the synthesizer (a) records it in the UTD's `mcp_support`, and (b) a
future `ToolExecutionStep`/`RheaFileToolStep` option `pin_seed: int` injects it into the tool
args, upgrading the effective determinism from R3 to R2 for that invocation (the UTD stays R3 —
the tool is *inherently* stochastic; only the *pinned* invocation is reproducible). Deferred:
needs a per-tool seed-param inventory from Rhea and a real stochastic tool to verify against.

## Real-data status

Authored on a laptop with **no reachable Rhea worker** (`$RHEA_MCP_URL` unset, docker daemon
down). Therefore:

- **Unit-verified** (fake MCP via httpx MockTransport — wire-shape only, per the mocks carve-out):
  the Rhea provenance helper, the discovery determinism-read, the synthesizer's file-vs-JSON
  branch + FAIL-LOUD paths, the `WorkflowBuilder` integration, and a **real `Workflow.run`**
  cascade driving a synthesized `ToolExecutionStep` to a concrete output value (G127-safe).
- **Live-gated** (`tests/integration/test_rhea_synthesize_step_live.py`, skipped unless
  `$RHEA_MCP_URL` is set): synthesize from a REAL Galaxy tool, assert REAL determinism pins,
  run end-to-end. This is the real-data proof that has NOT been executed here — the gap is
  explicit, not hidden.
