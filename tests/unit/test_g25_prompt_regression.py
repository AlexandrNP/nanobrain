"""G25 — pin the PromptRegressionHarness contract.

eval_03 Round 3 G25: G14 (PromptTemplate) was shipped, but the
harness that catches AC1-breaking regressions did not. Two AC1
regressions on 2026-04-22 from prose-level edits to system.md
demonstrated G14 was an incomplete cure without G25.

This test pins:
  1. all-pass run reports total/passed/failed correctly
  2. contains check fails when substring is missing
  3. not_contains check fails when forbidden substring appears
  4. matches_regex check fails when regex doesn't match
  5. json_schema (required keys) check
  6. json_schema (property type) check
  7. equals check
  8. fixture-level errors localize to one CaseResult, not the whole run
  9. snapshot mode records on first run, replays on second
 10. snapshot is invalidated when content_hash changes
 11. failures_summary() builds an actionable string
 12. empty regression_fixtures returns an all-pass report (zero cases)
 13. legacy template (no render() method) falls back to system_prompt /
     user_template
 14. fixture render error localizes to that CaseResult

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G25;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.8.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from nanobrain.library.testing.prompt_regression import (
    CaseResult,
    HarnessReport,
    PromptRegressionHarness,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakeTemplate:
    """A minimal stand-in for G14 PromptTemplate. The harness only
    reads template_id, content_hash, regression_fixtures, and either
    .render(params) OR .system_prompt / .user_template — so a dataclass
    is sufficient."""

    template_id: str = "test.regression@0.1.0"
    content_hash: str = "sha256:" + "0" * 64
    regression_fixtures: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt: str = "You are a test assistant."
    user_template: str = "Process: $payload"

    def render(self, params: Dict[str, Any]) -> Dict[str, str]:
        # Light token substitution mimicking PromptTemplate.render.
        user = self.user_template.replace(
            "$payload", str(params.get("payload", ""))
        )
        return {"system": self.system_prompt, "user": user}


def _llm_returning(static_response: str):
    async def _llm(*, system: str, user: str) -> str:
        return static_response

    return _llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_pass_run_reports_aggregate_correctly():
    fixtures = [
        {
            "input": {"payload": "alpha"},
            "expected_output_contract": {"contains": ["ok"]},
        },
        {
            "input": {"payload": "beta"},
            "expected_output_contract": {"contains": ["ok"]},
        },
    ]
    harness = PromptRegressionHarness(
        template=_FakeTemplate(regression_fixtures=fixtures),
        llm_callable=_llm_returning("response: ok"),
    )
    report = asyncio.run(harness.run())
    assert isinstance(report, HarnessReport)
    assert report.total == 2
    assert report.passed == 2
    assert report.failed == 0
    assert report.all_passed is True
    assert report.failures_summary() == ""


def test_contains_check_fails_when_substring_missing():
    fixtures = [
        {
            "input": {},
            "expected_output_contract": {"contains": ["MISSING"]},
        }
    ]
    harness = PromptRegressionHarness(
        template=_FakeTemplate(regression_fixtures=fixtures),
        llm_callable=_llm_returning("response without the needle"),
    )
    report = asyncio.run(harness.run())
    assert report.failed == 1
    assert "missing 'MISSING'" in report.cases[0].failures[0]


def test_not_contains_check_fails_when_forbidden_substring_appears():
    fixtures = [
        {
            "input": {},
            "expected_output_contract": {"not_contains": ["FORBIDDEN"]},
        }
    ]
    harness = PromptRegressionHarness(
        template=_FakeTemplate(regression_fixtures=fixtures),
        llm_callable=_llm_returning("contains FORBIDDEN keyword"),
    )
    report = asyncio.run(harness.run())
    assert report.failed == 1
    assert "FORBIDDEN" in report.cases[0].failures[0]


def test_regex_check_fails_when_pattern_does_not_match():
    fixtures = [
        {
            "input": {},
            "expected_output_contract": {
                "matches_regex": [r"^citation:\s*\[\d+\]"],
            },
        }
    ]
    harness = PromptRegressionHarness(
        template=_FakeTemplate(regression_fixtures=fixtures),
        llm_callable=_llm_returning("no citation marker here"),
    )
    report = asyncio.run(harness.run())
    assert report.failed == 1
    assert "does not match" in report.cases[0].failures[0]


def test_json_schema_required_keys_check():
    fixtures = [
        {
            "input": {},
            "expected_output_contract": {
                "json_schema": {
                    "required": ["title", "summary"],
                }
            },
        }
    ]
    harness = PromptRegressionHarness(
        template=_FakeTemplate(regression_fixtures=fixtures),
        llm_callable=_llm_returning(json.dumps({"title": "x"})),  # missing 'summary'
    )
    report = asyncio.run(harness.run())
    assert report.failed == 1
    assert "summary" in report.cases[0].failures[0]


def test_json_schema_property_type_check():
    fixtures = [
        {
            "input": {},
            "expected_output_contract": {
                "json_schema": {
                    "properties": {
                        "count": {"type": "integer"},
                    }
                }
            },
        }
    ]
    harness = PromptRegressionHarness(
        template=_FakeTemplate(regression_fixtures=fixtures),
        llm_callable=_llm_returning(
            json.dumps({"count": "not-an-integer"})
        ),
    )
    report = asyncio.run(harness.run())
    assert report.failed == 1
    assert "type mismatch" in report.cases[0].failures[0]
    assert "count" in report.cases[0].failures[0]


def test_equals_check():
    fixtures = [
        {
            "input": {},
            "expected_output_contract": {"equals": "exact-match"},
        }
    ]
    harness = PromptRegressionHarness(
        template=_FakeTemplate(regression_fixtures=fixtures),
        llm_callable=_llm_returning("not-the-expected-value"),
    )
    report = asyncio.run(harness.run())
    assert report.failed == 1
    assert "equals check" in report.cases[0].failures[0]


def test_fixture_error_localizes_to_one_case():
    """Mid-run llm error or render error must NOT abort other
    fixtures. Each case carries its own pass/fail outcome."""
    fixtures = [
        {
            "input": {},
            "expected_output_contract": {"contains": ["ok"]},
        },
        {
            "input": {},
            "expected_output_contract": {"contains": ["ok"]},
        },
    ]

    call_count = {"n": 0}

    async def _llm_raising_on_second(*, system: str, user: str) -> str:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("transient llm error")
        return "ok"

    harness = PromptRegressionHarness(
        template=_FakeTemplate(regression_fixtures=fixtures),
        llm_callable=_llm_raising_on_second,
    )
    report = asyncio.run(harness.run())
    assert report.passed == 1
    assert report.failed == 1
    assert report.cases[0].passed
    assert not report.cases[1].passed
    assert "transient llm error" in report.cases[1].failures[0]


def test_snapshot_mode_records_then_replays(tmp_path):
    fixtures = [
        {
            "input": {"payload": "x"},
            "expected_output_contract": {"contains": ["snapshot-result"]},
        }
    ]
    template = _FakeTemplate(regression_fixtures=fixtures)
    snapshots = tmp_path / "snapshots"

    # First run records.
    harness1 = PromptRegressionHarness(
        template=template,
        llm_callable=_llm_returning("snapshot-result-from-llm"),
        snapshot_path=snapshots,
    )
    report1 = asyncio.run(harness1.run())
    assert report1.all_passed
    assert report1.cases[0].snapshot_action == "recorded"
    assert any(snapshots.iterdir()), "snapshot file was not written"

    # Second run replays — even with a different llm_callable that
    # would fail the contract, the cached snapshot is what's compared.
    harness2 = PromptRegressionHarness(
        template=template,
        llm_callable=_llm_returning("totally-different-response"),
        snapshot_path=snapshots,
    )
    report2 = asyncio.run(harness2.run())
    assert report2.cases[0].snapshot_action == "replayed"
    # Replayed response is from snapshot, so contains check passes.
    assert report2.all_passed


def test_snapshot_invalidated_by_content_hash_change(tmp_path):
    """Snapshot path is content-addressed by content_hash; a change
    to the template body produces a NEW path, forcing a re-record."""
    snapshots = tmp_path / "snapshots"
    fixtures = [
        {
            "input": {},
            "expected_output_contract": {"contains": ["x"]},
        }
    ]
    t1 = _FakeTemplate(
        content_hash="sha256:" + "1" * 64,
        regression_fixtures=fixtures,
    )
    asyncio.run(
        PromptRegressionHarness(
            template=t1,
            llm_callable=_llm_returning("x"),
            snapshot_path=snapshots,
        ).run()
    )

    # Change content_hash → different snapshot path → re-record.
    t2 = _FakeTemplate(
        content_hash="sha256:" + "2" * 64,
        regression_fixtures=fixtures,
    )
    report = asyncio.run(
        PromptRegressionHarness(
            template=t2,
            llm_callable=_llm_returning("x"),
            snapshot_path=snapshots,
        ).run()
    )
    assert report.cases[0].snapshot_action == "recorded"


def test_failures_summary_is_actionable():
    fixtures = [
        {
            "input": {},
            "expected_output_contract": {"contains": ["alpha"]},
        }
    ]
    harness = PromptRegressionHarness(
        template=_FakeTemplate(
            template_id="test.regression@0.5.0",
            regression_fixtures=fixtures,
        ),
        llm_callable=_llm_returning("response missing the keyword"),
    )
    report = asyncio.run(harness.run())
    summary = report.failures_summary()
    assert "FAILED" in summary
    assert "test.regression@0.5.0" in summary
    assert "alpha" in summary
    assert "fixture[0]" in summary


def test_empty_regression_fixtures_returns_zero_cases():
    harness = PromptRegressionHarness(
        template=_FakeTemplate(regression_fixtures=[]),
        llm_callable=_llm_returning("never invoked"),
    )
    report = asyncio.run(harness.run())
    assert report.total == 0
    assert report.all_passed is True


def test_legacy_template_without_render_method_falls_back():
    """A template without ``render()`` method (legacy v1 contract)
    still works — harness falls back to system_prompt + user_template
    verbatim."""

    class _LegacyTemplate:
        template_id = "legacy.test@0.1.0"
        content_hash = None
        system_prompt = "system body"
        user_template = "user body"
        regression_fixtures = [
            {
                "input": {},
                "expected_output_contract": {"contains": ["ok"]},
            }
        ]

    harness = PromptRegressionHarness(
        template=_LegacyTemplate(),
        llm_callable=_llm_returning("ok"),
    )
    report = asyncio.run(harness.run())
    assert report.all_passed


def test_render_error_localizes_to_one_case():
    """If the template's render() raises (e.g., missing required hole),
    that's a per-case failure, not a harness-level abort."""

    class _BrokenRenderTemplate:
        template_id = "broken.render@0.1.0"
        content_hash = None
        regression_fixtures = [
            {
                "input": {},
                "expected_output_contract": {"contains": ["never-reached"]},
            }
        ]
        system_prompt = "."
        user_template = "."

        def render(self, params):
            raise ValueError("missing required hole 'foo'")

    harness = PromptRegressionHarness(
        template=_BrokenRenderTemplate(),
        llm_callable=_llm_returning("never-invoked"),
    )
    report = asyncio.run(harness.run())
    assert report.failed == 1
    assert "render error" in report.cases[0].failures[0]
    assert "missing required hole" in report.cases[0].failures[0]
