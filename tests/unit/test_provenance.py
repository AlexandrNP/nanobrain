"""Tests for G4 — step-level provenance threading + redact vocabulary.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G4``.

Tests cover:
1. ProvenanceContext from_config + lifecycle
2. JsonlSink — write + flush + buffering
3. Each redaction primitive in isolation
4. Default redaction resolution
5. Path-redaction primitive (path:<dotted>)
6. apply_redactions ordering + multiple primitives
7. activate() context manager + current_provenance_context()
8. record_step_invocation end-to-end
9. Disabled context is a no-op
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from nanobrain.core.provenance import (
    JsonlSink,
    ProvenanceContext,
    apply_redactions,
    current_provenance_context,
    normalize_redactions,
)


# ---------------------------------------------------------------------------
# 1. ProvenanceContext.from_config
# ---------------------------------------------------------------------------

class TestProvenanceContextFromConfig:

    def test_default_enabled_with_default_redactions(self):
        ctx = ProvenanceContext.from_config()
        assert ctx.enabled is True
        # Default redactions are ['prompts', 'executor_env']
        assert ctx.redactions == ["prompts", "executor_env"]
        assert ctx.sink is None

    def test_explicit_redactions_preserved(self):
        ctx = ProvenanceContext.from_config({
            "enabled": True,
            "redact": ["payload", "tool_args"],
        })
        assert ctx.redactions == ["payload", "tool_args"]

    def test_empty_redact_list_means_no_redactions(self):
        """Operator opt-out: explicit empty list = no redactions."""
        ctx = ProvenanceContext.from_config({
            "redact": [],
        })
        assert ctx.redactions == []

    def test_disabled_context(self):
        ctx = ProvenanceContext.from_config({"enabled": False})
        assert ctx.enabled is False

    def test_sink_path_auto_builds_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink_path = Path(tmp) / "prov.jsonl"
            ctx = ProvenanceContext.from_config({
                "sink_path": str(sink_path),
            })
            assert isinstance(ctx.sink, JsonlSink)

    def test_extra_field_rejected(self):
        with pytest.raises(Exception):
            ProvenanceContext.from_config({
                "enabled": True,
                "ghost_field": "bad",
            })


# ---------------------------------------------------------------------------
# 2. JsonlSink — write + flush + buffer
# ---------------------------------------------------------------------------

class TestJsonlSink:

    def test_write_and_flush(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                sink_path = Path(tmp) / "prov.jsonl"
                sink = JsonlSink(sink_path, flush_every=1)
                await sink.write_record({"key": "value", "num": 42})
                # flush_every=1 means write immediately.
                lines = sink_path.read_text().splitlines()
                assert len(lines) == 1
                rec = json.loads(lines[0])
                assert rec["key"] == "value"
                assert rec["num"] == 42
        asyncio.run(run())

    def test_buffering_behavior(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                sink_path = Path(tmp) / "prov.jsonl"
                sink = JsonlSink(sink_path, flush_every=3)
                # Write 2 records — file should not yet exist (buffered):
                await sink.write_record({"i": 1})
                await sink.write_record({"i": 2})
                assert not sink_path.exists()
                # Third record triggers flush:
                await sink.write_record({"i": 3})
                lines = sink_path.read_text().splitlines()
                assert len(lines) == 3
        asyncio.run(run())

    def test_flush_on_demand(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                sink_path = Path(tmp) / "prov.jsonl"
                sink = JsonlSink(sink_path, flush_every=10)
                await sink.write_record({"i": 1})
                # Explicit flush even though buffer has 1 < 10:
                await sink.flush()
                lines = sink_path.read_text().splitlines()
                assert len(lines) == 1
        asyncio.run(run())

    def test_creates_parent_directory(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                sink_path = Path(tmp) / "nested" / "deeper" / "prov.jsonl"
                sink = JsonlSink(sink_path, flush_every=1)
                await sink.write_record({"i": 1})
                assert sink_path.is_file()
        asyncio.run(run())

    def test_records_are_json_serializable_via_default(self):
        """Records that contain non-JSON-native types (datetime, etc)
        coerce via default=str."""
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                sink_path = Path(tmp) / "prov.jsonl"
                sink = JsonlSink(sink_path, flush_every=1)
                from datetime import datetime, timezone
                ts = datetime.now(timezone.utc)
                await sink.write_record({"when": ts})
                lines = sink_path.read_text().splitlines()
                # Should parse without error:
                rec = json.loads(lines[0])
                assert "when" in rec
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 3-4. Redaction primitives
# ---------------------------------------------------------------------------

class TestRedactionPrimitives:

    def test_payload_redaction_strips_value(self):
        record = {
            "inputs": {"q": {"value": "find this", "size_bytes": 10}},
            "outputs": {"a": {"value": "found"}},
        }
        result = apply_redactions(record, ["payload"])
        assert result["inputs"]["q"]["value"]["redacted"] == "payload"
        assert "size_bytes" in result["inputs"]["q"]["value"]
        assert "hash" in result["inputs"]["q"]["value"]
        assert result["outputs"]["a"]["value"]["redacted"] == "payload"

    def test_payload_redaction_preserves_size_bytes_field(self):
        """The original 'size_bytes' alongside 'value' is preserved
        (only the inner 'value' gets replaced, not sibling fields)."""
        record = {
            "inputs": {"q": {"value": "x", "size_bytes": 1, "metadata": {"k": "v"}}}
        }
        result = apply_redactions(record, ["payload"])
        assert result["inputs"]["q"]["size_bytes"] == 1
        assert result["inputs"]["q"]["metadata"] == {"k": "v"}

    def test_tool_args_redaction_strips_args(self):
        record = {
            "tool_calls": [{
                "tool_id": "rhea:muscle.align@5.1.0",
                "descriptor_hash": "abc123",
                "args": {"sequences": "ATCG", "alphabet": "DNA"},
            }],
        }
        result = apply_redactions(record, ["tool_args"])
        tc = result["tool_calls"][0]
        assert tc["tool_id"] == "rhea:muscle.align@5.1.0"  # preserved
        assert tc["descriptor_hash"] == "abc123"  # preserved
        assert tc["args"]["redacted"] == "tool_args"
        assert "schema_hash" in tc["args"]

    def test_prompts_redaction_strips_prompt_text(self):
        record = {
            "llm_calls": [{
                "prompt_text": "system: be terse",
                "prompt_template_id": "phase0.default@1.0.0",
                "template_version": "abc123",
                "param_hash": "def456",
            }]
        }
        result = apply_redactions(record, ["prompts"])
        llm = result["llm_calls"][0]
        assert "prompt_text" not in llm
        assert llm["redacted_prompt"]["redacted"] == "prompts"
        assert llm["redacted_prompt"]["template_id"] == "phase0.default@1.0.0"

    def test_llm_completions_redaction(self):
        record = {
            "llm_calls": [{
                "completion_text": "the answer is 42",
                "token_count": 5,
            }]
        }
        result = apply_redactions(record, ["llm_completions"])
        llm = result["llm_calls"][0]
        assert llm["completion_text"]["redacted"] == "llm_completions"
        assert llm["completion_text"]["char_count"] == 16
        assert llm["completion_text"]["token_count"] == 5
        assert "hash" in llm["completion_text"]

    def test_executor_env_redaction_keeps_names(self):
        record = {
            "executor_metadata": {
                "host": "node-001",
                "env": {"API_KEY": "secret-xyz", "DEBUG": "1"},
            }
        }
        result = apply_redactions(record, ["executor_env"])
        em = result["executor_metadata"]
        assert em["host"] == "node-001"  # preserved
        assert em["env"]["redacted"] == "executor_env"
        assert sorted(em["env"]["names"]) == ["API_KEY", "DEBUG"]
        # Names ARE preserved (intentional — operator can audit which
        # env vars the step touched). VALUES are gone — verify by
        # checking the secret value isn't present anywhere:
        assert "secret-xyz" not in str(em["env"])
        assert "DEBUG" in str(em["env"])  # name kept
        # The redacted dict has only 'redacted' and 'names' keys:
        assert set(em["env"].keys()) == {"redacted", "names"}

    def test_path_redaction_targets_dotted_path(self):
        record = {
            "executor_metadata": {
                "host": "node-001",
                "secret": {"token": "very-secret"},
            }
        }
        result = apply_redactions(record, ["path:executor_metadata.secret.token"])
        secret_token = result["executor_metadata"]["secret"]["token"]
        assert secret_token["redacted"] == "path"
        assert secret_token["path"] == "executor_metadata.secret.token"

    def test_path_redaction_missing_path_silently_skipped(self):
        """Path redaction tolerates missing paths (no error). Less
        strict than G1 predicate FAIL-FAST because redaction is
        defensive — better to no-op a missing path than to fail the
        whole record."""
        record = {"x": 1}
        result = apply_redactions(record, ["path:nonexistent.path"])
        assert result == {"x": 1}

    def test_unknown_primitive_logs_warning_no_op(self, caplog):
        record = {"x": 1}
        import logging
        with caplog.at_level(logging.WARNING):
            result = apply_redactions(record, ["bogus_primitive"])
        assert result == {"x": 1}
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("Unknown redaction" in r.getMessage() for r in warnings)


# ---------------------------------------------------------------------------
# 4. Default-redaction resolution
# ---------------------------------------------------------------------------

class TestNormalizeRedactions:

    def test_none_returns_defaults(self):
        result = normalize_redactions(None)
        assert result == ["prompts", "executor_env"]

    def test_empty_list_returns_empty(self):
        assert normalize_redactions([]) == []

    def test_explicit_list_passes_through(self):
        assert normalize_redactions(["payload", "tool_args"]) == ["payload", "tool_args"]

    def test_returns_copy(self):
        wl = ["payload"]
        result = normalize_redactions(wl)
        result.append("tool_args")
        assert wl == ["payload"]  # original unchanged


# ---------------------------------------------------------------------------
# 5. apply_redactions ordering
# ---------------------------------------------------------------------------

class TestApplyRedactionsOrdering:

    def test_multiple_primitives_all_apply(self):
        record = {
            "inputs": {"q": {"value": "x"}},
            "llm_calls": [{
                "prompt_text": "system: be terse",
                "completion_text": "ok",
            }],
            "executor_metadata": {"env": {"K": "v"}},
        }
        result = apply_redactions(record, [
            "payload", "prompts", "llm_completions", "executor_env",
        ])
        # All four redactions visible in result:
        assert result["inputs"]["q"]["value"]["redacted"] == "payload"
        assert result["llm_calls"][0]["redacted_prompt"]["redacted"] == "prompts"
        assert result["llm_calls"][0]["completion_text"]["redacted"] == "llm_completions"
        assert result["executor_metadata"]["env"]["redacted"] == "executor_env"

    def test_does_not_mutate_input_record(self):
        record = {"inputs": {"q": {"value": "secret"}}}
        original_value = record["inputs"]["q"]["value"]
        apply_redactions(record, ["payload"])
        # Original record is untouched:
        assert record["inputs"]["q"]["value"] == original_value


# ---------------------------------------------------------------------------
# 6. activate() context manager
# ---------------------------------------------------------------------------

class TestActivateContextManager:

    def test_activate_installs_context(self):
        ctx = ProvenanceContext.from_config()
        assert current_provenance_context() is None
        with ctx.activate():
            assert current_provenance_context() is ctx
        assert current_provenance_context() is None

    def test_activate_restores_on_exception(self):
        ctx = ProvenanceContext.from_config()
        with pytest.raises(ValueError):
            with ctx.activate():
                raise ValueError("intentional")
        assert current_provenance_context() is None

    def test_concurrent_tasks_isolated(self):
        async def run():
            results = []

            async def task_a():
                ctx = ProvenanceContext.from_config()
                with ctx.activate():
                    await asyncio.sleep(0)
                    results.append(("a", current_provenance_context() is ctx))

            async def task_b():
                ctx = ProvenanceContext.from_config()
                with ctx.activate():
                    await asyncio.sleep(0)
                    results.append(("b", current_provenance_context() is ctx))

            await asyncio.gather(task_a(), task_b())
            assert all(is_self for _, is_self in results)

        asyncio.run(run())


# ---------------------------------------------------------------------------
# 7. record_step_invocation end-to-end
# ---------------------------------------------------------------------------

class TestRecordStepInvocation:

    def test_minimal_record(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                sink_path = Path(tmp) / "prov.jsonl"
                ctx = ProvenanceContext.from_config({
                    "sink_path": str(sink_path),
                    "flush_every": 1,
                })
                await ctx.record_step_invocation(step_name="my_step")
                lines = sink_path.read_text().splitlines()
                assert len(lines) == 1
                rec = json.loads(lines[0])
                assert rec["step_name"] == "my_step"
                assert "ts" in rec  # auto-set
        asyncio.run(run())

    def test_full_record_with_redactions_applied(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                sink_path = Path(tmp) / "prov.jsonl"
                ctx = ProvenanceContext.from_config({
                    "sink_path": str(sink_path),
                    "flush_every": 1,
                    # Use default redactions: prompts + executor_env
                })
                await ctx.record_step_invocation(
                    step_name="full_step",
                    inputs={"q": {"value": "sensitive query"}},
                    llm_calls=[{
                        "prompt_text": "system: secret prompt body",
                        "completion_text": "answer",
                    }],
                    executor_metadata={
                        "host": "node-001",
                        "env": {"API_KEY": "secret-xyz"},
                    },
                )
                rec = json.loads(sink_path.read_text())
                # Default redactions: prompts + executor_env applied:
                assert "prompt_text" not in rec["llm_calls"][0]
                assert rec["llm_calls"][0]["redacted_prompt"]["redacted"] == "prompts"
                assert rec["executor_metadata"]["env"]["redacted"] == "executor_env"
                # NOT in defaults: payload + llm_completions pass through:
                assert rec["inputs"]["q"]["value"] == "sensitive query"
                assert rec["llm_calls"][0]["completion_text"] == "answer"
        asyncio.run(run())

    def test_disabled_context_no_op(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                sink_path = Path(tmp) / "prov.jsonl"
                ctx = ProvenanceContext.from_config({
                    "enabled": False,
                    "sink_path": str(sink_path),
                    "flush_every": 1,
                })
                await ctx.record_step_invocation(step_name="x")
                # Disabled — sink file should not exist:
                assert not sink_path.exists()
        asyncio.run(run())

    def test_no_sink_no_op(self):
        """Context with no sink configured — record_step_invocation
        runs through redaction logic but writes nowhere. No error."""
        async def run():
            ctx = ProvenanceContext.from_config({"enabled": True})
            assert ctx.sink is None
            # Should not raise:
            await ctx.record_step_invocation(step_name="x")
        asyncio.run(run())

    def test_exception_field_recorded(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                sink_path = Path(tmp) / "prov.jsonl"
                ctx = ProvenanceContext.from_config({
                    "sink_path": str(sink_path),
                    "flush_every": 1,
                })
                await ctx.record_step_invocation(
                    step_name="failing_step",
                    exception={"type": "ValueError", "message": "boom"},
                )
                rec = json.loads(sink_path.read_text())
                assert rec["exception"]["type"] == "ValueError"
        asyncio.run(run())
