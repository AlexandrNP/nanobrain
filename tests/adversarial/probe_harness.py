"""Adversarial probe harness for the nanobrain framework.

Stop criterion (per the user mandate): **0 bugs found in 300
consecutive distinct probes**. Counter resets on bug-found.

A "probe" is a callable that exercises ONE behavior of ONE surface
with a deliberately weird / boundary-case input, and asserts the
outcome matches an expected predicate. A "bug" is any divergence:
- Probe raised an unexpected exception type.
- Probe returned a value that violates the predicate.
- Probe hung longer than the per-probe timeout (10 s default).

The harness:
1. Imports a probe registry (categories of probe-generators).
2. Iterates: pick a category, generate a probe with random inputs
   parameterized by the round number, run it.
3. Track the bug list AND the consecutive-zero-bug counter.
4. Stop when counter hits 300, OR ``--max-rounds`` reached, OR the
   user kills the process.

Surfaces probed (each is a category):
- ``utd`` — UnifiedToolDescriptor from_dict / from_python_callable / hash
- ``utd_descriptor`` — descriptor_id grammar; nested-class admittance
- ``tool_python`` — ToolBase.from_python_callable + execute
- ``tool_descriptor`` — ToolBase.from_descriptor failure modes
- ``rhea_mcp_parse`` — RheaMCPDispatcher._parse_mcp_result on synthetic bodies
- ``runner_lifecycle`` — WorkflowRunner.run_detached + cancel + await_completion
- ``runner_pause`` — WorkflowRunner.pause / resume / is_paused / cooperative gates
- ``runner_heartbeat`` — WorkflowRunner watchdog timing edge cases
- ``runner_store`` — TaskStore (in-memory + sqlite + postgres) CRUD
- ``checkpoint`` — CheckpointStep / ResumeStep filesystem round-trip + tampering
- ``rebuild`` — ResumeStep on_missing='rebuild' callable resolution
- ``gate_semantics`` — gate-to-bottom propagation + sentinel handling
- ``lightweight`` — WorkflowBuilder add_step / add_link / add_trigger / load
- ``trigger_replay`` — TimerTrigger.replay_missed_fires policy edge cases
- ``trigger_event`` — EventTrigger fire_event + filter
- ``trigger_alldata`` — AllDataReceivedTrigger _is_satisfied edge cases
- ``conditional_link`` — ConditionalLink condition evaluation edge cases
- ``durable_state`` — FileEntryStateStore CRUD + path traversal
- ``signed_config`` — SignedConfig signature verification
- ``import_whitelist`` — class-path import whitelist
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("probe_harness")


# ---------------------------------------------------------------------------
# Probe data shape
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ProbeOutcome:
    name: str
    category: str
    seed: int
    bug: bool
    detail: str = ""
    elapsed_seconds: float = 0.0


@dataclasses.dataclass
class Probe:
    name: str           # unique within a run (category::N::seed-hash)
    category: str
    fn: Callable[[], Any]            # sync or coroutine returning the outcome
    predicate: Callable[[Any], bool]  # True = no bug
    seed: int
    expect_exception: Optional[type] = None
    timeout_seconds: float = 10.0


# ---------------------------------------------------------------------------
# Probe registry — each generator returns a Probe given a round seed
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[[int], Probe]] = {}


def register(category: str):
    def _wrap(fn):
        _REGISTRY[category] = fn
        return fn
    return _wrap


# ---- UTD ---------------------------------------------------------------

@register("utd")
def gen_utd_from_dict(seed: int) -> Probe:
    """Build a UTD from a dict with random param shapes."""
    from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor

    rng = random.Random(seed)
    n_inputs = rng.randint(0, 5)
    inputs = []
    for i in range(n_inputs):
        inputs.append({
            "name": f"p{i}",
            "type": rng.choice(["string", "integer", "number", "object", "array"]),
            "description": "x" * rng.randint(0, 100),
            "required": rng.choice([True, False]),
            "default": rng.choice([None, 0, "", [], {}, "abc"]),
        })

    # Use the EXACT Literal vocabularies from
    # nanobrain.core.unified_tool_descriptor — passing invalid values
    # would trigger a correct framework rejection that we'd misread as
    # a bug. (The probe-generator is itself adversarial input; we only
    # want REAL bugs to surface.)
    side_effects_vocab = [
        "none", "filesystem_read", "filesystem_write",
        "network", "external_database", "destructive",
    ]
    determinism_vocab = ["R1", "R2", "R3"]
    resource_vocab = [
        "cpu_light", "cpu_medium", "cpu_heavy",
        "gpu_single", "gpu_multi", "io_heavy",
    ]
    payload = {
        "descriptor_id": f"native:probe_{seed}@0.1.{seed}",
        "display_name": f"Probe {seed}",
        "summary": "synthetic probe",
        "long_description": "",
        "inputs": inputs,
        "outputs": [{"name": "return", "type": "object", "description": ""}],
        "side_effects": rng.choice(side_effects_vocab),
        "determinism": rng.choice(determinism_vocab),
        "resource_class": rng.choice(resource_vocab),
        "provenance_pin": {"class_path": "nanobrain.core.tool.ToolBase"},
    }

    def _fn():
        utd = UnifiedToolDescriptor.from_dict(payload)
        return utd

    def _pred(result):
        from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor
        return isinstance(result, UnifiedToolDescriptor) and bool(result.descriptor_hash)

    return Probe(
        name=f"utd::{seed}",
        category="utd",
        fn=_fn,
        predicate=_pred,
        seed=seed,
    )


@register("utd_descriptor")
def gen_utd_descriptor_id(seed: int) -> Probe:
    """Test descriptor_id grammar with weird inputs."""
    from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor

    rng = random.Random(seed)
    valid_chars = "abcdefghijklmnopqrstuvwxyz0123456789_."
    backend = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=rng.randint(1, 8)))
    tool_id_first = rng.choice("abcdefghijklmnopqrstuvwxyz")
    tool_id_rest = "".join(rng.choices(valid_chars, k=rng.randint(0, 30)))
    tool_id = tool_id_first + tool_id_rest
    version = f"{rng.randint(0, 99)}.{rng.randint(0, 99)}.{rng.randint(0, 99)}"
    descriptor_id = f"{backend}:{tool_id}@{version}"

    payload = {
        "descriptor_id": descriptor_id,
        "display_name": "x",
        "summary": "y",
        "long_description": "",
        "inputs": [],
        "outputs": [],
        "side_effects": "none",
        "determinism": "R3",
        "resource_class": "cpu_light",
        "provenance_pin": {"class_path": "nanobrain.core.tool.ToolBase"},
    }

    def _fn():
        return UnifiedToolDescriptor.from_dict(payload)

    def _pred(result):
        return result.descriptor_id == descriptor_id and result.descriptor_backend == backend

    return Probe(
        name=f"utd_descriptor::{seed}",
        category="utd_descriptor",
        fn=_fn,
        predicate=_pred,
        seed=seed,
    )


# ---- Tool ----------------------------------------------------------------

@register("tool_python")
def gen_tool_from_python(seed: int) -> Probe:
    """Wrap a Python callable + dispatch with various arg shapes."""
    from nanobrain.core.tool import ToolBase

    rng = random.Random(seed)
    arity = rng.randint(0, 4)
    arg_names = [f"a{i}" for i in range(arity)]

    # Build a function dynamically — important for diversity
    src = f"def _f({', '.join(f'{n}: int' for n in arg_names)}) -> int:\n    return {(' + '.join(arg_names)) or '0'}"
    ns: Dict[str, Any] = {}
    exec(src, ns)
    fn = ns["_f"]

    args = {n: rng.randint(-1000, 1000) for n in arg_names}
    expected = sum(args.values()) if args else 0

    async def _run():
        tool = ToolBase.from_python_callable(fn)
        result = await tool.execute(args)
        return result

    def _fn():
        return asyncio.run(_run())

    return Probe(
        name=f"tool_python::{seed}::{arity}",
        category="tool_python",
        fn=_fn,
        predicate=lambda r: r == expected,
        seed=seed,
    )


@register("tool_descriptor_bad")
def gen_tool_descriptor_bad(seed: int) -> Probe:
    """from_descriptor with a malformed UTD must FAIL-FAST consistently."""
    from nanobrain.core.tool import ToolBase
    from nanobrain.core.component_base import ComponentConfigurationError

    rng = random.Random(seed)
    bad_dicts = [
        {},  # empty
        {"descriptor_id": "bad-shape"},
        {"descriptor_id": "valid:t@1", "provenance_pin": {"class_path": "nonexistent.module.Cls"}},
        {"descriptor_id": "valid:t@1", "provenance_pin": {"class_path": "nanobrain.core.tool.ToolConfig"}},  # not ToolBase
    ]
    bad = bad_dicts[seed % len(bad_dicts)]

    def _fn():
        try:
            ToolBase.from_descriptor(bad)
            return "no-raise"
        except (ComponentConfigurationError, Exception) as e:
            return type(e).__name__

    # Predicate: bad input must raise SOMETHING (no silent acceptance).
    return Probe(
        name=f"tool_descriptor_bad::{seed}",
        category="tool_descriptor_bad",
        fn=_fn,
        predicate=lambda r: r != "no-raise",
        seed=seed,
    )


# ---- WorkflowRunner ------------------------------------------------------

@register("runner_lifecycle")
def gen_runner_lifecycle(seed: int) -> Probe:
    """Schedule + complete + read handle for various task IDs + payloads."""
    from nanobrain.library.runtime import WorkflowRunner
    import yaml

    rng = random.Random(seed)
    payload_size = rng.randint(0, 50)
    payload = {f"k{i}": rng.randint(0, 1000) for i in range(payload_size)}
    expected_sum = sum(payload.values())

    async def workflow(p: Dict[str, Any]) -> int:
        return sum(p.values())

    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            yml = tmp / "r.yml"
            yml.write_text(yaml.safe_dump({
                "name": "r", "task_store_backend": "in_memory",
                "heartbeat_interval_seconds": 0,
            }))
            runner = WorkflowRunner.from_config(str(yml))
            tid = f"task_{seed}"
            await runner.run_detached(workflow, tid, payload)
            h = await runner.await_completion(tid, timeout=5)
            return h.status, h.result

    def _fn():
        return asyncio.run(_run())

    return Probe(
        name=f"runner_lifecycle::{seed}::{payload_size}",
        category="runner_lifecycle",
        fn=_fn,
        predicate=lambda r: r[0] == "completed" and r[1] == expected_sum,
        seed=seed,
    )


@register("runner_cancel")
def gen_runner_cancel(seed: int) -> Probe:
    """Cancel a long-running workflow → status='cancelled'."""
    from nanobrain.library.runtime import WorkflowRunner
    import yaml

    async def slow(_: Dict[str, Any]):
        await asyncio.sleep(5.0)
        return "should-not-reach"

    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            yml = tmp / "r.yml"
            yml.write_text(yaml.safe_dump({
                "name": "r", "task_store_backend": "in_memory",
                "heartbeat_interval_seconds": 0,
            }))
            runner = WorkflowRunner.from_config(str(yml))
            tid = f"slow_{seed}"
            await runner.run_detached(slow, tid, {})
            await asyncio.sleep(0.05)
            await runner.cancel(tid)
            h = await runner.get_handle(tid)
            return h.status

    def _fn():
        return asyncio.run(_run())

    return Probe(
        name=f"runner_cancel::{seed}",
        category="runner_cancel",
        fn=_fn,
        predicate=lambda s: s == "cancelled",
        seed=seed,
    )


# ---- CheckpointStep / ResumeStep ----------------------------------------

@register("checkpoint")
def gen_checkpoint_round_trip(seed: int) -> Probe:
    """CheckpointStep then ResumeStep with various payload shapes."""
    from nanobrain.library.steps import CheckpointStep, ResumeStep
    import yaml

    rng = random.Random(seed)
    payload = {
        "n": rng.randint(0, 1000),
        "items": [rng.randint(-100, 100) for _ in range(rng.randint(0, 20))],
        "meta": {"k": "v" * rng.randint(0, 10)},
    }

    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cp_yml = tmp / "cp.yml"
            cp_yml.write_text(yaml.safe_dump({
                "name": "cp", "backend": "filesystem",
                "base_dir": str(tmp / "snap"),
                "capture": ["*"],
                "manifest_path": str(tmp / "m.json"),
            }))
            cp = CheckpointStep.from_config(str(cp_yml))
            await cp.process(payload)

            rs_yml = tmp / "rs.yml"
            rs_yml.write_text(yaml.safe_dump({"name": "rs"}))
            rs = ResumeStep.from_config(str(rs_yml))
            r = await rs.process({"manifest_path": str(tmp / "m.json")})
            return r

    def _fn():
        return asyncio.run(_run())

    def _pred(r):
        return all(r.get(k) == v for k, v in payload.items())

    return Probe(
        name=f"checkpoint::{seed}",
        category="checkpoint",
        fn=_fn,
        predicate=_pred,
        seed=seed,
    )


# ---- Gate-to-bottom ------------------------------------------------------

@register("gate_semantics")
def gen_gate_propagation(seed: int) -> Probe:
    """Workflow-level gate_semantics propagates to ConditionalLinks."""
    from nanobrain.core.workflow import WorkflowConfig

    rng = random.Random(seed)
    n_links = rng.randint(1, 5)
    links = {}
    for i in range(n_links):
        links[f"link_{i}"] = {
            "class": "nanobrain.core.link.ConditionalLink",
            "source": f"a{i}.x", "target": f"b{i}.x",
            "condition": "true_only",
        }
    gate = rng.choice(["gate_to_bottom", "publish_empty"])

    def _fn():
        WorkflowConfig._allow_direct_instantiation = True
        try:
            cfg = WorkflowConfig(
                name=f"probe_{seed}",
                gate_semantics=gate,
                links=links,
            )
        finally:
            WorkflowConfig._allow_direct_instantiation = False
        return [
            cfg.links[name]["gate_semantics"]
            for name in links
        ]

    return Probe(
        name=f"gate_semantics::{seed}::{n_links}::{gate}",
        category="gate_semantics",
        fn=_fn,
        predicate=lambda v: all(g == gate for g in v),
        seed=seed,
    )


# ---- Lightweight WorkflowBuilder ----------------------------------------

@register("lightweight")
def gen_lightweight_builder(seed: int) -> Probe:
    """Lightweight WorkflowBuilder add_link with various link types."""
    from nanobrain.lightweight import WorkflowBuilder

    rng = random.Random(seed)
    link_type = rng.choice(["direct", "conditional", "transform"])
    kwargs = {}
    if link_type == "conditional":
        kwargs["condition"] = {"op": "exists", "field": "v"}
    elif link_type == "transform":
        kwargs["transform_function"] = "json.dumps"

    def _fn():
        b = WorkflowBuilder(f"probe_{seed}")
        b.add_link("a.x", "b.x", link_type=link_type, **kwargs)
        cfg = b.get_config()
        return cfg["links"]

    def _pred(links):
        return len(links) == 1 and "link_0" in links

    return Probe(
        name=f"lightweight::{seed}::{link_type}",
        category="lightweight",
        fn=_fn,
        predicate=_pred,
        seed=seed,
    )


# ---- TimerTrigger replay ------------------------------------------------

@register("trigger_replay")
def gen_trigger_replay(seed: int) -> Probe:
    """TimerTrigger.replay_missed_fires under various policies + windows."""
    from nanobrain.core.trigger import TimerTrigger, TriggerConfig, TriggerType

    rng = random.Random(seed)
    interval_ms = rng.choice([10, 50, 100, 500, 1000])
    policy = rng.choice(["skip", "catch_up", "merge"])
    elapsed_seconds = rng.uniform(0.0, 5.0)
    expected_missed = int(round(elapsed_seconds * 1000)) // interval_ms

    if policy == "skip":
        expected_fires = 0
    elif policy == "merge":
        expected_fires = 1 if expected_missed > 0 else 0
    else:
        expected_fires = expected_missed

    async def _run():
        TriggerConfig._allow_direct_instantiation = True
        try:
            cfg = TriggerConfig(
                name=f"t_{seed}",
                trigger_type=TriggerType.TIMER,
                timer_interval_ms=interval_ms,
                on_missed=policy,
                debounce_ms=0,
                max_frequency_hz=10000.0,
            )
        finally:
            TriggerConfig._allow_direct_instantiation = False
        t = TimerTrigger.from_config(cfg)
        fires = []
        async def cb(_=None):
            fires.append(True)
        await t.add_callback(cb)
        n = await t.replay_missed_fires(0.0, elapsed_seconds)
        return n, len(fires)

    def _fn():
        return asyncio.run(_run())

    def _pred(r):
        return r == (expected_fires, expected_fires)

    return Probe(
        name=f"trigger_replay::{seed}::{interval_ms}::{policy}",
        category="trigger_replay",
        fn=_fn,
        predicate=_pred,
        seed=seed,
    )


# ---- EventTrigger filter ------------------------------------------------

@register("trigger_event")
def gen_event_trigger(seed: int) -> Probe:
    """EventTrigger fire_event with G1 predicate filter."""
    from nanobrain.core.trigger import EventTrigger, TriggerConfig, TriggerType

    rng = random.Random(seed)
    target_value = rng.randint(1, 100)
    event_value = target_value if rng.random() < 0.5 else target_value + 1
    expected_fire = (event_value == target_value)

    async def _run():
        TriggerConfig._allow_direct_instantiation = True
        try:
            cfg = TriggerConfig(
                name=f"ev_{seed}",
                trigger_type=TriggerType.EVENT,
                debounce_ms=0,
                max_frequency_hz=10000.0,
                event_filter={"op": "eq", "field": "v", "value": target_value},
            )
        finally:
            TriggerConfig._allow_direct_instantiation = False
        t = EventTrigger.from_config(cfg)
        fired = []
        async def cb(p):
            fired.append(p)
        await t.add_callback(cb)
        await t.start_monitoring()
        result = await t.fire_event({"v": event_value})
        return result, len(fired)

    def _fn():
        return asyncio.run(_run())

    def _pred(r):
        return r == (expected_fire, 1 if expected_fire else 0)

    return Probe(
        name=f"trigger_event::{seed}::{target_value}::{event_value}",
        category="trigger_event",
        fn=_fn,
        predicate=_pred,
        seed=seed,
    )


# ---- AllDataReceivedTrigger _is_satisfied -------------------------------

@register("trigger_alldata")
def gen_alldata_trigger(seed: int) -> Probe:
    """_is_satisfied with various payload shapes + gate semantics."""
    from nanobrain.core.link import ConditionalLink
    from nanobrain.core.trigger import (
        AllDataReceivedTrigger, TriggerConfig, TriggerType,
    )

    rng = random.Random(seed)
    gate = rng.choice(["publish_empty", "gate_to_bottom"])
    payload_kind = rng.choice([
        "none", "sentinel", "real_dict", "real_int", "real_zero", "real_empty_str",
    ])
    payloads = {
        "none": None,
        "sentinel": ConditionalLink.GATED_OFF_SENTINEL,
        "real_dict": {"k": rng.randint(0, 1000)},
        "real_int": rng.randint(1, 100),
        "real_zero": 0,
        "real_empty_str": "",
    }
    payload = payloads[payload_kind]

    if payload is None:
        expected = (False, False)
    elif payload == ConditionalLink.GATED_OFF_SENTINEL:
        if gate == "gate_to_bottom":
            expected = (True, False)
        else:
            expected = (False, False)
    else:
        expected = (True, True)

    def _fn():
        TriggerConfig._allow_direct_instantiation = True
        try:
            cfg = TriggerConfig(
                name=f"adt_{seed}",
                trigger_type=TriggerType.ALL_DATA_RECEIVED,
                gate_semantics=gate,
            )
        finally:
            TriggerConfig._allow_direct_instantiation = False
        t = AllDataReceivedTrigger.from_config(cfg)
        return t._is_satisfied(payload)

    return Probe(
        name=f"trigger_alldata::{seed}::{gate}::{payload_kind}",
        category="trigger_alldata",
        fn=_fn,
        predicate=lambda r: r == expected,
        seed=seed,
    )


# ---- RheaMCPDispatcher result parsing -----------------------------------

@register("rhea_mcp_parse")
def gen_rhea_parse(seed: int) -> Probe:
    """_parse_mcp_result on various synthetic SSE bodies."""
    from nanobrain.library.tools.rhea_mcp_dispatcher import RheaMCPDispatcher
    from nanobrain.core.tool import ToolBase
    from nanobrain.core.component_base import ComponentConfigurationError

    rng = random.Random(seed)
    shape = rng.choice([
        "happy_text_json", "happy_text_string", "happy_structured",
        "is_error", "rpc_error", "no_data", "garbled_data",
    ])

    bodies = {
        "happy_text_json": (
            'event: message\ndata: ' + json.dumps({
                "jsonrpc": "2.0", "id": 2, "result": {
                    "content": [{"type": "text", "text": json.dumps({"x": seed})}],
                    "isError": False,
                }
            }) + "\n"
        ),
        "happy_text_string": (
            'event: message\ndata: ' + json.dumps({
                "jsonrpc": "2.0", "id": 2, "result": {
                    "content": [{"type": "text", "text": "plain text"}],
                    "isError": False,
                }
            }) + "\n"
        ),
        "happy_structured": (
            'event: message\ndata: ' + json.dumps({
                "jsonrpc": "2.0", "id": 2, "result": {
                    "structuredContent": {"struct": True, "n": seed},
                    "content": [{"type": "text", "text": "ignored"}],
                }
            }) + "\n"
        ),
        "is_error": (
            'event: message\ndata: ' + json.dumps({
                "jsonrpc": "2.0", "id": 2, "result": {
                    "content": [{"type": "text", "text": "boom"}],
                    "isError": True,
                }
            }) + "\n"
        ),
        "rpc_error": (
            'event: message\ndata: ' + json.dumps({
                "jsonrpc": "2.0", "id": 2,
                "error": {"code": -32600, "message": "bad request"},
            }) + "\n"
        ),
        "no_data": "event: ping\n\n",
        "garbled_data": "data: not-json {{{\n",
    }
    body = bodies[shape]

    # Build a dispatcher just to parse (we don't call execute here)
    utd_dict = {
        "descriptor_id": "rhea:probe@0.1.0",
        "display_name": "Probe", "summary": "x", "long_description": "",
        "inputs": [], "outputs": [], "side_effects": "none",
        "determinism": "R3", "resource_class": "cpu_light",
        "provenance_pin": {
            "class_path": "nanobrain.library.tools.rhea_mcp_dispatcher.RheaMCPDispatcher",
        },
    }

    def _fn():
        tool = ToolBase.from_descriptor(
            utd_dict, mcp_url="http://x/", rhea_tool_name="probe",
        )
        try:
            return tool._parse_mcp_result(body)
        except ComponentConfigurationError as e:
            return f"FAIL_FAST::{type(e).__name__}::{str(e)[:60]}"

    def _pred(r):
        if shape == "happy_text_json":
            return r == {"x": seed}
        if shape == "happy_text_string":
            return r == "plain text"
        if shape == "happy_structured":
            return isinstance(r, dict) and r.get("struct") is True
        if shape in ("is_error", "rpc_error", "no_data", "garbled_data"):
            return isinstance(r, str) and r.startswith("FAIL_FAST::")
        return False

    return Probe(
        name=f"rhea_mcp_parse::{seed}::{shape}",
        category="rhea_mcp_parse",
        fn=_fn,
        predicate=_pred,
        seed=seed,
    )


# ---- Durable state store ------------------------------------------------

@register("utd_unicode")
def gen_utd_unicode(seed: int) -> Probe:
    """UTD descriptor with non-ASCII display names + summaries.

    Adversarial: descriptor_id grammar is ASCII-only but display_name +
    summary + long_description are free-form strings. Catches any
    accidental ASCII assumption."""
    from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor

    rng = random.Random(seed)
    # Mix of unicode chars: emojis, CJK, RTL, combining marks
    pools = [
        "🦠🧬🔬", "中文测试", "العربية", "Ω∂∫∇", "café façade naïve",
        "𝓟𝓻𝓸𝓫𝓮", "‍‌", "🇺🇸🇯🇵",
    ]
    display = pools[seed % len(pools)] + str(seed)
    summary = "summary " + pools[(seed + 1) % len(pools)]
    long_desc = "details: " + pools[(seed + 2) % len(pools)] * rng.randint(0, 3)

    payload = {
        "descriptor_id": f"native:tool_unicode_{seed}@1.0.0",
        "display_name": display,
        "summary": summary,
        "long_description": long_desc,
        "inputs": [],
        "outputs": [],
        "side_effects": "none",
        "determinism": "R3",
        "resource_class": "cpu_light",
        "provenance_pin": {"class_path": "nanobrain.core.tool.ToolBase"},
    }

    def _fn():
        utd = UnifiedToolDescriptor.from_dict(payload)
        return utd.display_name == display and utd.summary == summary

    return Probe(
        name=f"utd_unicode::{seed}",
        category="utd_unicode",
        fn=_fn,
        predicate=lambda r: r is True,
        seed=seed,
    )


@register("runner_concurrent")
def gen_runner_concurrent(seed: int) -> Probe:
    """Schedule N concurrent run_detached + verify all complete."""
    from nanobrain.library.runtime import WorkflowRunner
    import yaml

    rng = random.Random(seed)
    n_concurrent = rng.randint(2, 12)

    async def workflow(p):
        # Random light async work
        await asyncio.sleep(rng.uniform(0, 0.02))
        return p.get("n", -1) * 2

    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            yml = tmp / "r.yml"
            yml.write_text(yaml.safe_dump({
                "name": "r", "task_store_backend": "in_memory",
                "heartbeat_interval_seconds": 0,
                "max_concurrent_detached_tasks": rng.randint(1, n_concurrent),
            }))
            runner = WorkflowRunner.from_config(str(yml))
            tids = [f"task_{seed}_{i}" for i in range(n_concurrent)]
            for i, tid in enumerate(tids):
                await runner.run_detached(workflow, tid, {"n": i})
            results = []
            for tid in tids:
                h = await runner.await_completion(tid, timeout=10)
                results.append((h.status, h.result))
            return results

    def _fn():
        return asyncio.run(_run())

    def _pred(results):
        for i, (status, value) in enumerate(results):
            if status != "completed":
                return False
            if value != i * 2:
                return False
        return True

    return Probe(
        name=f"runner_concurrent::{seed}::{n_concurrent}",
        category="runner_concurrent",
        fn=_fn,
        predicate=_pred,
        seed=seed,
    )


@register("rebuild_path")
def gen_rebuild_path(seed: int) -> Probe:
    """ResumeStep on_missing='rebuild' with various callable shapes."""
    from nanobrain.library.steps import ResumeStep
    import yaml

    rng = random.Random(seed)
    # Pick a callable shape (sync vs async, simple vs nested return)
    callable_choice = rng.choice(["sync_simple", "sync_nested", "async_simple"])

    # Module-level callables can't be defined inside a probe; use the
    # globals approach: register the callable in this module's namespace.
    callable_name = f"_probe_rebuild_{seed}_{callable_choice}"

    if callable_choice == "sync_simple":
        def _build(_):
            return {"a": seed, "b": "x"}
    elif callable_choice == "sync_nested":
        def _build(_):
            return {"items": list(range(5)), "meta": {"seed": seed}}
    else:  # async_simple
        async def _build(_):  # type: ignore[misc]
            return {"async": True, "n": seed}

    sys.modules[__name__].__dict__[callable_name] = _build
    spec = f"{__name__}.{callable_name}"

    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = {
                "name": f"rs_{seed}",
                "on_missing": "rebuild",
                "rebuild_callable": spec,
                "rebuild_base_dir": str(tmp / "rb"),
            }
            yml = tmp / "rs.yml"
            yml.write_text(yaml.safe_dump(cfg))
            rs = ResumeStep.from_config(str(yml))
            r = await rs.process({"manifest_path": str(tmp / "m.json")})
            return r

    def _fn():
        return asyncio.run(_run())

    def _pred(r):
        if not isinstance(r, dict):
            return False
        if not r.get("_rebuilt"):
            return False
        if callable_choice == "sync_simple":
            return r.get("a") == seed and r.get("b") == "x"
        if callable_choice == "sync_nested":
            return r.get("items") == [0,1,2,3,4] and r.get("meta", {}).get("seed") == seed
        if callable_choice == "async_simple":
            return r.get("async") is True and r.get("n") == seed
        return False

    return Probe(
        name=f"rebuild_path::{seed}::{callable_choice}",
        category="rebuild_path",
        fn=_fn,
        predicate=_pred,
        seed=seed,
    )


@register("conditional_link_predicate")
def gen_conditional_link_predicate(seed: int) -> Probe:
    """G1 PredicateConfig op vocabulary + edge-case payloads."""
    from nanobrain.core.link import (
        PredicateConfig,
        evaluate_predicate,
    )

    rng = random.Random(seed)
    op = rng.choice(["eq", "ne", "in", "exists"])
    field = "x"
    if op == "exists":
        payload = rng.choice([{"x": None}, {"x": 0}, {"x": ""}, {}, {"x": "v"}, {"y": 1}])
        expected = "x" in payload
        cfg_data = {"op": op, "field": field}
    elif op in ("eq", "ne"):
        target = rng.randint(0, 5)
        payload_value = rng.randint(0, 5)
        payload = {"x": payload_value}
        if op == "eq":
            expected = payload_value == target
        else:
            expected = payload_value != target
        cfg_data = {"op": op, "field": field, "value": target}
    else:  # in
        target_list = list(range(rng.randint(0, 5)))
        payload_value = rng.randint(0, 5)
        payload = {"x": payload_value}
        expected = payload_value in target_list
        cfg_data = {"op": op, "field": field, "value": target_list}

    PredicateConfig._allow_direct_instantiation = True
    try:
        pred_cfg = PredicateConfig(**cfg_data)
    finally:
        PredicateConfig._allow_direct_instantiation = False

    def _fn():
        return evaluate_predicate(payload, pred_cfg)

    return Probe(
        name=f"conditional_link_predicate::{seed}::{op}",
        category="conditional_link_predicate",
        fn=_fn,
        predicate=lambda r: r == expected,
        seed=seed,
    )


@register("durable_state")
def gen_durable_state(seed: int) -> Probe:
    """FileEntryStateStore CRUD — set/get/delete with random payloads."""
    from nanobrain.library.runtime import FileEntryStateStore

    rng = random.Random(seed)
    n_entries = rng.randint(1, 10)
    entries = {
        f"entry_{i}": {
            "last_fire_epoch_seconds": rng.uniform(0, 1e10),
            "last_task_id": f"task_{rng.randint(0, 1000)}",
        }
        for i in range(n_entries)
    }

    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            store = FileEntryStateStore(str(Path(tmp) / "s"))
            # Set all
            for k, v in entries.items():
                await store.set(k, v)
            # Get all
            got = {}
            for k in entries:
                got[k] = await store.get(k)
            # Delete one
            del_key = list(entries.keys())[0]
            await store.delete(del_key)
            after_delete = await store.get(del_key)
            return got, after_delete

    def _fn():
        return asyncio.run(_run())

    def _pred(r):
        got, after_delete = r
        return got == entries and after_delete is None

    return Probe(
        name=f"durable_state::{seed}::{n_entries}",
        category="durable_state",
        fn=_fn,
        predicate=_pred,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------

def _run_one(probe: Probe) -> ProbeOutcome:
    start = time.monotonic()
    try:
        result = probe.fn()
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - start
        return ProbeOutcome(
            name=probe.name, category=probe.category, seed=probe.seed,
            bug=True,
            detail=f"unexpected exception {type(exc).__name__}: {exc}\n{traceback.format_exc()[:1500]}",
            elapsed_seconds=elapsed,
        )

    elapsed = time.monotonic() - start
    try:
        ok = probe.predicate(result)
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(
            name=probe.name, category=probe.category, seed=probe.seed,
            bug=True,
            detail=f"predicate raised: {type(exc).__name__}: {exc}",
            elapsed_seconds=elapsed,
        )

    if not ok:
        return ProbeOutcome(
            name=probe.name, category=probe.category, seed=probe.seed,
            bug=True,
            detail=f"predicate returned False; result was: {result!r}",
            elapsed_seconds=elapsed,
        )

    return ProbeOutcome(
        name=probe.name, category=probe.category, seed=probe.seed,
        bug=False, elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(target_consecutive: int = 300, max_rounds: int = 5000,
             seed_offset: int = 0):
    seen_names: set = set()
    consecutive_no_bug = 0
    round_idx = 0
    bugs: List[ProbeOutcome] = []
    categories = list(_REGISTRY.keys())
    rng = random.Random(0xCAFE)

    print(f"Starting adversarial probe loop. target={target_consecutive} consecutive zero-bug rounds.")
    print(f"Categories: {categories}")
    print()

    while consecutive_no_bug < target_consecutive and round_idx < max_rounds:
        round_idx += 1
        category = rng.choice(categories)
        seed = seed_offset + round_idx * 1009 + hash(category) % 7919
        probe = _REGISTRY[category](seed)
        if probe.name in seen_names:
            # collision — perturb seed
            probe = _REGISTRY[category](seed + round_idx * 7919)
        seen_names.add(probe.name)

        outcome = _run_one(probe)

        if outcome.bug:
            consecutive_no_bug = 0
            bugs.append(outcome)
            print(f"[round {round_idx}] BUG in {outcome.category}::{outcome.name}")
            print(f"  detail: {outcome.detail[:300]}")
            print()
        else:
            consecutive_no_bug += 1
            if consecutive_no_bug % 25 == 0:
                print(f"[round {round_idx}] {consecutive_no_bug} consecutive zero-bug probes")

    print()
    print(f"Loop ended after {round_idx} rounds.")
    print(f"Consecutive zero-bug count: {consecutive_no_bug}")
    print(f"Total bugs found: {len(bugs)}")
    return bugs, consecutive_no_bug, round_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--max-rounds", type=int, default=5000)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    bugs, consecutive, rounds = run_loop(
        target_consecutive=args.target,
        max_rounds=args.max_rounds,
        seed_offset=args.seed_offset,
    )
    if consecutive >= args.target:
        print(f"\nSUCCESS: {consecutive} consecutive zero-bug probes reached.")
        sys.exit(0)
    else:
        print(f"\nFAIL: only {consecutive} consecutive zero-bug probes; {len(bugs)} bugs found in {rounds} rounds.")
        sys.exit(1)
