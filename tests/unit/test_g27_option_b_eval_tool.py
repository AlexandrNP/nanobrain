"""Tests for the G27 Option B evaluation tool.

Pins the decision logic the operator-facing script uses to recommend
Option A vs. Option B. The tool itself reads from a TaskStore +
prints a verdict; this test exercises the pure ``evaluate()``
function with synthetic TaskRow inputs.

Source: nanobrain/scripts/g27_option_b_eval.py; design doc
``nanobrain/docs/g27_g21_wiring_design.md`` "Option B evaluation
framework".
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

# The script lives under nanobrain/scripts/; expose it as a
# module-importable path for the test runner.
_SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2] / "scripts"
)
sys.path.insert(0, str(_SCRIPTS_DIR))

import g27_option_b_eval as eval_tool  # noqa: E402


def _row(resume_count: int = 0, status: str = "completed") -> eval_tool.TaskRow:
    return eval_tool.TaskRow(
        task_id=f"t-{resume_count}-{status}",
        status=status,
        resume_count=resume_count,
        cost_actual={},
        suspension_info=None,
    )


# ---------------------------------------------------------------------------
# evaluate() — recommendation logic
# ---------------------------------------------------------------------------


def test_no_resumes_recommends_stick_with_a():
    """Zero resumes across all tasks → Option A is fine."""
    verdict = eval_tool.evaluate(
        [_row(0), _row(0), _row(0)],
        pre_hitl_step_seconds=10.0,
        upgrade_threshold_seconds=300.0,
        stick_threshold_seconds=30.0,
    )
    assert verdict.recommendation == "stick_with_A"
    assert verdict.tasks_with_resumes == 0
    assert verdict.max_resume_count == 0


def test_low_resume_cost_recommends_stick_with_a():
    """max_resume_count × pre_hitl = 2 × 10s = 20s, below stick
    threshold (30s) → Option A."""
    verdict = eval_tool.evaluate(
        [_row(0), _row(1), _row(2)],
        pre_hitl_step_seconds=10.0,
        upgrade_threshold_seconds=300.0,
        stick_threshold_seconds=30.0,
    )
    assert verdict.recommendation == "stick_with_A"
    assert verdict.max_resume_count == 2


def test_high_resume_cost_recommends_promote_b():
    """max_resume_count × pre_hitl = 5 × 120s = 600s, above upgrade
    threshold (300s) → Option B."""
    verdict = eval_tool.evaluate(
        [_row(0), _row(2), _row(5)],
        pre_hitl_step_seconds=120.0,
        upgrade_threshold_seconds=300.0,
        stick_threshold_seconds=30.0,
    )
    assert verdict.recommendation == "promote_B"
    assert "600.0s" in verdict.rationale or "600" in verdict.rationale


def test_mid_range_returns_inconclusive():
    """max_resume_count × pre_hitl falls between thresholds →
    inconclusive, operator decides."""
    verdict = eval_tool.evaluate(
        [_row(0), _row(2)],
        pre_hitl_step_seconds=50.0,  # 2 × 50 = 100s; between 30s and 300s
        upgrade_threshold_seconds=300.0,
        stick_threshold_seconds=30.0,
    )
    assert verdict.recommendation == "inconclusive"
    assert "Operator judgment" in verdict.rationale


def test_no_pre_hitl_estimate_returns_inconclusive():
    """When --pre-hitl-step-seconds is not provided, the tool
    cannot compute a recommendation. Surfaces the distribution
    only."""
    verdict = eval_tool.evaluate(
        [_row(0), _row(3)],
        pre_hitl_step_seconds=None,
        upgrade_threshold_seconds=300.0,
        stick_threshold_seconds=30.0,
    )
    assert verdict.recommendation == "inconclusive"
    assert "No --pre-hitl-step-seconds" in verdict.rationale
    # Distribution still computed.
    assert verdict.max_resume_count == 3
    assert verdict.tasks_with_resumes == 1


def test_empty_rows_safe():
    """No tasks in the store → all-zero stats, recommendation
    follows the threshold logic against 0 cost."""
    verdict = eval_tool.evaluate(
        [],
        pre_hitl_step_seconds=10.0,
        upgrade_threshold_seconds=300.0,
        stick_threshold_seconds=30.0,
    )
    assert verdict.total_tasks == 0
    assert verdict.recommendation == "stick_with_A"  # 0 cost <= stick


def test_mean_resume_count_only_counts_resumed_tasks():
    """The mean is over tasks WITH at least one resume — zeros
    don't dilute the signal."""
    # 3 resumed: 2, 4, 6 → mean 4.0. Plus 7 non-resumed.
    rows = [_row(2), _row(4), _row(6)] + [_row(0) for _ in range(7)]
    verdict = eval_tool.evaluate(
        rows,
        pre_hitl_step_seconds=1.0,
        upgrade_threshold_seconds=300.0,
        stick_threshold_seconds=30.0,
    )
    assert verdict.tasks_with_resumes == 3
    assert verdict.mean_resume_count_among_resumed == 4.0


# ---------------------------------------------------------------------------
# SQLite end-to-end smoke test
# ---------------------------------------------------------------------------


def test_sqlite_round_trip(tmp_path):
    """End-to-end: build a SQLite TaskStore, insert a few rows,
    run the eval tool's main(), assert exit code 0 + recommendation
    text in output."""
    db_path = tmp_path / "tasks.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    # Mimic the SqliteTaskStore schema.
    conn.execute(
        """
        CREATE TABLE detached_tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_heartbeat_at TEXT,
            completed_at TEXT,
            result_json TEXT,
            error TEXT,
            cost_actual_json TEXT,
            suspension_info_json TEXT,
            resume_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "INSERT INTO detached_tasks VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("t1", "completed", "2026-05-11T00:00:00Z", None, None,
         None, None, None, None, 0),
    )
    conn.execute(
        "INSERT INTO detached_tasks VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("t2", "completed", "2026-05-11T00:00:00Z", None, None,
         None, None, None, None, 4),
    )
    conn.close()

    # Read back through the tool.
    rows = eval_tool._rows_from_sqlite(str(db_path))
    assert len(rows) == 2
    assert {r.task_id for r in rows} == {"t1", "t2"}
    assert max(r.resume_count for r in rows) == 4


def test_main_emits_json_when_requested(tmp_path, capsys):
    db_path = tmp_path / "tasks.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute(
        """
        CREATE TABLE detached_tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_heartbeat_at TEXT,
            completed_at TEXT,
            result_json TEXT,
            error TEXT,
            cost_actual_json TEXT,
            suspension_info_json TEXT,
            resume_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "INSERT INTO detached_tasks VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("t1", "completed", "2026-05-11T00:00:00Z", None, None,
         None, None, None, None, 1),
    )
    conn.close()

    rc = eval_tool.main([
        "--sqlite", str(db_path),
        "--json",
        "--pre-hitl-step-seconds", "1.0",
    ])
    out = capsys.readouterr().out

    import json as _json
    payload = _json.loads(out)
    assert payload["total_tasks"] == 1
    assert payload["max_resume_count"] == 1
    assert payload["recommendation"] in {"stick_with_A", "inconclusive", "promote_B"}
    assert rc == 0
