"""G27 Option B evaluation tool — should you implement no-re-run resume?

Operator-facing CLI script. Queries a deployed nanobrain TaskStore
(SQLite or Postgres) and computes the re-run cost vs. total cost
ratio across resumed tasks. Outputs an Option-A-vs-Option-B
recommendation per the rule of thumb in
``nanobrain/docs/g27_g21_wiring_design.md``.

## Why this tool exists

Per the G27 wiring design doc, Option A (re-run-from-start) is the
shipped default. Option B (G5 checkpoint integration for no-re-run
resume) is more complex and should be implemented ONLY if real
deployment data shows the re-run cost is meaningful. This tool
computes the deciding metric so operators don't have to roll their
own SQL.

## Usage

    # SQLite backend
    python -m nanobrain.scripts.g27_option_b_eval \\
        --sqlite /var/nanobrain/tasks.db

    # Postgres backend (DSN format documented in psycopg)
    python -m nanobrain.scripts.g27_option_b_eval \\
        --postgres "postgresql://nb:secret@localhost/nb"

    # Custom thresholds (override design-doc defaults)
    python -m nanobrain.scripts.g27_option_b_eval \\
        --sqlite /var/nanobrain/tasks.db \\
        --upgrade-threshold-seconds 60 \\
        --stick-threshold-seconds 5

## What the tool DOES NOT do

- It does NOT modify the TaskStore. Read-only queries.
- It does NOT include in-flight tasks (only those whose status is
  terminal: completed / cancelled / failed; PLUS suspended tasks
  for partial-trajectory visibility).
- It does NOT compute pre-HITL step COST automatically — for v1 we
  assume the operator estimates pre-HITL cost per task externally;
  the tool surfaces ``resume_count`` and ``cost_actual`` raw so
  operators can multiply themselves.

Decision rule of thumb (from design doc):
  - resume_count × pre_hitl_step_seconds < 30s per task: stick with A
  - resume_count × pre_hitl_step_seconds > 5 minutes per task OR
    multi-resume cycles common: promote B

Source: nanobrain/docs/g27_g21_wiring_design.md "Option B
evaluation framework" section.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class TaskRow:
    """One row from the TaskStore, normalized for analysis."""

    task_id: str
    status: str
    resume_count: int
    cost_actual: dict
    suspension_info: Optional[dict]


@dataclass
class Verdict:
    """Per-deployment Option-A-vs-B recommendation."""

    total_tasks: int
    tasks_with_resumes: int
    max_resume_count: int
    mean_resume_count_among_resumed: float
    recommendation: str  # one of "stick_with_A", "promote_B", "inconclusive"
    rationale: str

    def to_dict(self) -> dict:
        return {
            "total_tasks": self.total_tasks,
            "tasks_with_resumes": self.tasks_with_resumes,
            "max_resume_count": self.max_resume_count,
            "mean_resume_count_among_resumed": round(
                self.mean_resume_count_among_resumed, 3
            ),
            "recommendation": self.recommendation,
            "rationale": self.rationale,
        }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G27 Option B evaluation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument(
        "--sqlite",
        type=str,
        help="Path to a SQLite TaskStore DB.",
    )
    backend.add_argument(
        "--postgres",
        type=str,
        help="psycopg DSN for a Postgres TaskStore (e.g. "
        "'postgresql://user:pass@host/db').",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default=None,
        help=(
            "Postgres table name override (default: "
            "'nanobrain_detached_tasks'). Ignored for SQLite (the "
            "table name is fixed at 'detached_tasks')."
        ),
    )
    parser.add_argument(
        "--upgrade-threshold-seconds",
        type=float,
        default=300.0,
        help=(
            "Per-task re-run cost threshold above which Option B is "
            "recommended. Default: 300 (5 minutes). Pair with "
            "--pre-hitl-step-seconds; the tool checks "
            "max_resume_count × pre_hitl_step_seconds against this."
        ),
    )
    parser.add_argument(
        "--stick-threshold-seconds",
        type=float,
        default=30.0,
        help=(
            "Per-task re-run cost threshold below which Option A is "
            "safe. Default: 30 seconds."
        ),
    )
    parser.add_argument(
        "--pre-hitl-step-seconds",
        type=float,
        default=None,
        help=(
            "Estimated seconds spent on pre-HITL steps per workflow "
            "invocation. Required for the recommendation; operators "
            "estimate this externally (sum of pre-HITL step "
            "durations from their workflow's telemetry). When "
            "omitted, the tool prints raw resume_count distribution "
            "without a recommendation."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text.",
    )
    return parser.parse_args(argv)


def _rows_from_sqlite(path: str) -> List[TaskRow]:
    """Read rows from a SQLite TaskStore. Returns ALL rows so the
    caller can filter (the script reports both terminal + suspended)."""
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            "SELECT task_id, status, resume_count, cost_actual_json, "
            "suspension_info_json "
            "FROM detached_tasks"
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return [_normalize_row(r) for r in rows]


def _rows_from_postgres(dsn: str, table_name: str) -> List[TaskRow]:
    """Read rows from a Postgres TaskStore. Synchronous (uses
    psycopg's blocking API; the eval tool is not on the hot path)."""
    try:
        import psycopg
    except ImportError as exc:
        raise ImportError(
            "Postgres mode requires psycopg. Install: pip install "
            "'psycopg[binary]'"
        ) from exc
    conn = psycopg.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT task_id, status, resume_count, "
                f"cost_actual_json, suspension_info_json "
                f"FROM {table_name}"
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [_normalize_row(r) for r in rows]


def _normalize_row(r: Sequence) -> TaskRow:
    """Convert a DB row tuple into a TaskRow. Tolerates pre-migration
    shapes where resume_count or suspension_info_json may be NULL."""
    task_id, status, resume_count, cost_json, susp_json = (
        r[0], r[1], r[2] or 0, r[3], r[4]
    )
    return TaskRow(
        task_id=task_id,
        status=status,
        resume_count=int(resume_count),
        cost_actual=json.loads(cost_json) if cost_json else {},
        suspension_info=json.loads(susp_json) if susp_json else None,
    )


def evaluate(
    rows: Iterable[TaskRow],
    *,
    pre_hitl_step_seconds: Optional[float],
    upgrade_threshold_seconds: float,
    stick_threshold_seconds: float,
) -> Verdict:
    """Compute the verdict from a sequence of TaskRow."""
    rows_list = list(rows)
    total = len(rows_list)
    resumed = [r for r in rows_list if r.resume_count > 0]
    max_rc = max((r.resume_count for r in rows_list), default=0)
    mean_rc = (
        sum(r.resume_count for r in resumed) / len(resumed)
        if resumed
        else 0.0
    )

    if pre_hitl_step_seconds is None:
        # No cost estimate provided; we cannot produce a hard
        # recommendation. Surface the distribution; operator decides.
        return Verdict(
            total_tasks=total,
            tasks_with_resumes=len(resumed),
            max_resume_count=max_rc,
            mean_resume_count_among_resumed=mean_rc,
            recommendation="inconclusive",
            rationale=(
                "No --pre-hitl-step-seconds provided. Provide an "
                "estimated pre-HITL step duration to compute "
                "per-task re-run cost and produce a recommendation."
            ),
        )

    # Compute upper-bound re-run cost: max_resume_count × pre_hitl
    # is the worst-case per-task cost paid because of Option A.
    max_rerun_cost = max_rc * pre_hitl_step_seconds
    if max_rerun_cost >= upgrade_threshold_seconds:
        return Verdict(
            total_tasks=total,
            tasks_with_resumes=len(resumed),
            max_resume_count=max_rc,
            mean_resume_count_among_resumed=mean_rc,
            recommendation="promote_B",
            rationale=(
                f"Worst-case re-run cost = "
                f"{max_rerun_cost:.1f}s "
                f"(max_resume_count={max_rc} × pre_hitl_step_seconds="
                f"{pre_hitl_step_seconds:.1f}) exceeds the upgrade "
                f"threshold of {upgrade_threshold_seconds:.1f}s. "
                f"Implementing Option B's no-re-run resume semantic "
                f"will save substantial wasted compute / cost."
            ),
        )
    if max_rerun_cost <= stick_threshold_seconds:
        return Verdict(
            total_tasks=total,
            tasks_with_resumes=len(resumed),
            max_resume_count=max_rc,
            mean_resume_count_among_resumed=mean_rc,
            recommendation="stick_with_A",
            rationale=(
                f"Worst-case re-run cost = "
                f"{max_rerun_cost:.1f}s is at or below the stick "
                f"threshold of {stick_threshold_seconds:.1f}s. "
                f"Option A's re-run-from-start semantic is "
                f"acceptable; Option B's complexity is not justified."
            ),
        )
    return Verdict(
        total_tasks=total,
        tasks_with_resumes=len(resumed),
        max_resume_count=max_rc,
        mean_resume_count_among_resumed=mean_rc,
        recommendation="inconclusive",
        rationale=(
            f"Worst-case re-run cost = "
            f"{max_rerun_cost:.1f}s falls between the stick threshold "
            f"({stick_threshold_seconds:.1f}s) and upgrade threshold "
            f"({upgrade_threshold_seconds:.1f}s). Operator judgment "
            f"required: factor in deployment scale + cost-per-second "
            f"+ user experience cost of wait-for-resume."
        ),
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.sqlite:
        rows = _rows_from_sqlite(args.sqlite)
    else:
        table_name = args.table_name or "nanobrain_detached_tasks"
        rows = _rows_from_postgres(args.postgres, table_name)

    verdict = evaluate(
        rows,
        pre_hitl_step_seconds=args.pre_hitl_step_seconds,
        upgrade_threshold_seconds=args.upgrade_threshold_seconds,
        stick_threshold_seconds=args.stick_threshold_seconds,
    )

    if args.json:
        print(json.dumps(verdict.to_dict(), indent=2))
        return 0

    # Human-readable summary.
    print("G27 Option B evaluation")
    print("=" * 60)
    print(f"Total tasks:                       {verdict.total_tasks}")
    print(f"Tasks with at least one resume:    {verdict.tasks_with_resumes}")
    print(f"Maximum resume_count per task:     {verdict.max_resume_count}")
    print(
        f"Mean resume_count (resumed only):  "
        f"{verdict.mean_resume_count_among_resumed:.2f}"
    )
    print(f"Recommendation:                    {verdict.recommendation}")
    print()
    print("Rationale:")
    print(f"  {verdict.rationale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
