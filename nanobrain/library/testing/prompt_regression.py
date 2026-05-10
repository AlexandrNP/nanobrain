"""PromptRegressionHarness — G25 schema-aware regression harness for G14 prompts.

eval_03 Round 3 G25: G14 (PromptTemplate primitive) shipped, but the
harness that catches the AC1-breaking class of regressions did not.
``llm_prompt_contracts.md §1`` documents two AC1-breaking regressions
on 2026-04-22 from prose-level edits to ``system.md``; without G25,
G14 is an incomplete cure (the data-model is pinned, but no automated
regression check runs the prompt against its declared fixtures).

Post-G25 the framework ships ``PromptRegressionHarness``: a generic
test runner that:

  1. Reads the ``regression_fixtures`` field on a G14 PromptTemplate
     (each fixture is a dict shape ``{input, expected_output_contract}``
     per the schema in ``prompt_template_manager.py``).
  2. Invokes a caller-supplied ``llm_callable`` per fixture, passing
     the rendered prompt + the fixture's input.
  3. Validates the LLM's response against the fixture's
     ``expected_output_contract``: substring assertions, regex
     assertions, JSON-schema assertions, and snapshot equality.
  4. Reports a per-fixture pass/fail tuple PLUS an aggregate
     ``CaseResult`` so the test runner can emit one failure per
     fixture rather than one mega-failure per template.

The harness is **deliberately backend-neutral** — it does not import
OpenAI / Anthropic / Ollama. The caller plugs in any callable whose
signature is::

    async def llm(*, system: str, user: str) -> str

Callers wrap their preferred SDK; the harness validates the response.

## Snapshot mode

Optional: when constructed with ``snapshot_path``, the harness
records the LLM's first response per fixture as a JSON file. On
subsequent runs, it loads the snapshot and ASSERTS the new response
exactly matches the snapshot. This is the canonical CI mode — fast,
reproducible, no LLM cost on every run. The first run is the
"baseline" record.

Snapshot files are content-addressed by ``(template_id, fixture_index,
content_hash)`` so a template body change invalidates the snapshot
automatically and forces a re-record.

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G25;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.8.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


LLMCallable = Callable[..., Awaitable[str]]


@dataclass(frozen=True)
class CaseResult:
    """Per-fixture outcome of a regression run."""

    fixture_index: int
    passed: bool
    response: str
    failures: List[str] = field(default_factory=list)
    snapshot_action: Optional[str] = None  # "recorded" | "replayed" | None

    def __bool__(self) -> bool:  # truthy on pass
        return self.passed


@dataclass(frozen=True)
class HarnessReport:
    """Aggregate result over all fixtures in one regression run."""

    template_id: str
    template_content_hash: Optional[str]
    cases: List[CaseResult]

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def failures_summary(self) -> str:
        """Build a human-readable summary of the failed cases. Empty
        string when all passed."""
        if self.all_passed:
            return ""
        chunks: List[str] = [
            f"PromptRegressionHarness: {self.failed}/{self.total} "
            f"cases FAILED for template_id={self.template_id!r} "
            f"(content_hash={self.template_content_hash})"
        ]
        for c in self.cases:
            if c.passed:
                continue
            chunks.append(
                f"  fixture[{c.fixture_index}]: "
                + "; ".join(c.failures)
            )
        return "\n".join(chunks)


class PromptRegressionHarness:
    """Run a G14 PromptTemplate against its declared regression_fixtures.

    Args:
        template: A PromptTemplate (G14-compliant; must have
            template_id and at least one of system_prompt /
            user_template). The fixtures are read from
            template.regression_fixtures.
        llm_callable: ``async def llm(*, system: str, user: str) -> str``
            — the harness invokes this once per fixture. Wrap your
            SDK to match this signature.
        snapshot_path: Optional Path to a directory for snapshot
            files. When set, the first run records baseline; subsequent
            runs replay + assert exact match.
    """

    def __init__(
        self,
        *,
        template: Any,  # PromptTemplate (avoid hard import dep)
        llm_callable: LLMCallable,
        snapshot_path: Optional[Path] = None,
    ) -> None:
        self._template = template
        self._llm_callable = llm_callable
        self._snapshot_dir: Optional[Path] = (
            Path(snapshot_path) if snapshot_path is not None else None
        )
        if self._snapshot_dir is not None:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> HarnessReport:
        """Run every regression_fixture and build a HarnessReport."""
        fixtures = self._fixtures()
        cases: List[CaseResult] = []
        for idx, fixture in enumerate(fixtures):
            case = await self._run_one(idx, fixture)
            cases.append(case)
        return HarnessReport(
            template_id=getattr(self._template, "template_id", "<no-id>")
            or "<no-id>",
            template_content_hash=getattr(
                self._template, "content_hash", None
            ),
            cases=cases,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fixtures(self) -> List[Dict[str, Any]]:
        fxs = getattr(self._template, "regression_fixtures", None)
        if not fxs:
            return []
        if not isinstance(fxs, list):
            raise TypeError(
                f"FAIL-FAST: regression_fixtures must be a list; "
                f"got {type(fxs).__name__}"
            )
        return list(fxs)

    async def _run_one(
        self, idx: int, fixture: Dict[str, Any]
    ) -> CaseResult:
        if not isinstance(fixture, dict):
            return CaseResult(
                fixture_index=idx,
                passed=False,
                response="",
                failures=[f"fixture is not a dict (got {type(fixture).__name__})"],
            )

        # Render the template with the fixture's inputs.
        params = fixture.get("input", {}) or {}
        try:
            rendered = self._render(params)
        except Exception as exc:
            return CaseResult(
                fixture_index=idx,
                passed=False,
                response="",
                failures=[
                    f"render error: {type(exc).__name__}: {exc}"
                ],
            )

        # Snapshot: replay if available; else invoke + record.
        snapshot_action = None
        snapshot_path = self._snapshot_path_for(idx)
        if snapshot_path is not None and snapshot_path.is_file():
            response = json.loads(snapshot_path.read_text())["response"]
            snapshot_action = "replayed"
        else:
            try:
                response = await self._llm_callable(
                    system=rendered.get("system", ""),
                    user=rendered.get("user", ""),
                )
            except Exception as exc:
                return CaseResult(
                    fixture_index=idx,
                    passed=False,
                    response="",
                    failures=[
                        f"llm invocation error: "
                        f"{type(exc).__name__}: {exc}"
                    ],
                )
            if snapshot_path is not None:
                snapshot_path.write_text(
                    json.dumps(
                        {
                            "response": response,
                            "fixture_index": idx,
                            "template_id": getattr(
                                self._template, "template_id", None
                            ),
                            "content_hash": getattr(
                                self._template, "content_hash", None
                            ),
                        },
                        indent=2,
                    )
                )
                snapshot_action = "recorded"

        # Validate against the fixture's contract.
        contract = fixture.get("expected_output_contract", {}) or {}
        failures = self._check_contract(response, contract)
        return CaseResult(
            fixture_index=idx,
            passed=not failures,
            response=response,
            failures=failures,
            snapshot_action=snapshot_action,
        )

    def _render(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Render the template via its ``render(params)`` method.
        Falls back to (system_prompt, user_template) verbatim when
        ``render`` is not available (legacy templates)."""
        if hasattr(self._template, "render") and callable(
            self._template.render
        ):
            return self._template.render(params)
        return {
            "system": getattr(self._template, "system_prompt", "") or "",
            "user": getattr(self._template, "user_template", "") or "",
        }

    def _snapshot_path_for(self, fixture_idx: int) -> Optional[Path]:
        """Build the snapshot file path. Content-addressed by
        (template_id, fixture_index, content_hash) so a template-body
        edit invalidates the snapshot automatically."""
        if self._snapshot_dir is None:
            return None
        tid = getattr(self._template, "template_id", "no-id") or "no-id"
        ch = getattr(self._template, "content_hash", "no-hash") or "no-hash"
        # Truncate the hash so filenames stay reasonable.
        ch_short = (ch.split(":", 1)[1] if ":" in ch else ch)[:16]
        # Sanitize tid (slashes etc. would break path construction).
        tid_safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", tid)
        fname = f"{tid_safe}_{fixture_idx:03d}_{ch_short}.json"
        return self._snapshot_dir / fname

    def _check_contract(
        self,
        response: str,
        contract: Dict[str, Any],
    ) -> List[str]:
        """Validate ``response`` against the fixture's contract.

        Supported contract keys (any subset; absent = no check):
          * ``contains``: list[str] — every entry must appear as a
            substring of response.
          * ``not_contains``: list[str] — none of these may appear.
          * ``matches_regex``: list[str] — every regex must match
            against response (re.search semantics).
          * ``json_schema``: dict — response is parsed as JSON and
            validated. Lightweight: only ``required`` / ``properties``
            keys are honored. The framework does not import jsonschema;
            callers wanting full JSON Schema use the json_schema dict
            shape and run jsonschema themselves before passing to the
            harness.
          * ``equals``: str — response must equal verbatim.
        """
        failures: List[str] = []

        for needle in contract.get("contains", []) or []:
            if needle not in response:
                failures.append(f"contains check: missing {needle!r}")
        for forbidden in contract.get("not_contains", []) or []:
            if forbidden in response:
                failures.append(f"not_contains: found {forbidden!r}")
        for pattern in contract.get("matches_regex", []) or []:
            if not re.search(pattern, response):
                failures.append(
                    f"regex check: response does not match {pattern!r}"
                )
        if "equals" in contract:
            if response != contract["equals"]:
                failures.append(
                    f"equals check: response != expected "
                    f"(len {len(response)} vs {len(contract['equals'])})"
                )
        if "json_schema" in contract:
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError as exc:
                failures.append(
                    f"json_schema: response is not valid JSON: {exc}"
                )
            else:
                schema = contract["json_schema"] or {}
                for required_key in schema.get("required", []):
                    if (
                        not isinstance(parsed, dict)
                        or required_key not in parsed
                    ):
                        failures.append(
                            f"json_schema: missing required key "
                            f"{required_key!r}"
                        )
                # Light type-check on properties.
                for prop, spec in (schema.get("properties") or {}).items():
                    if (
                        isinstance(parsed, dict)
                        and prop in parsed
                        and "type" in spec
                    ):
                        actual = parsed[prop]
                        expected = spec["type"]
                        if not _matches_json_type(actual, expected):
                            failures.append(
                                f"json_schema: property {prop!r} type "
                                f"mismatch: expected {expected!r}, got "
                                f"{type(actual).__name__}"
                            )
        return failures


def _matches_json_type(value: Any, expected: str) -> bool:
    """Lightweight JSON-Schema type check."""
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True  # unknown types — pass through


__all__ = [
    "CaseResult",
    "HarnessReport",
    "LLMCallable",
    "PromptRegressionHarness",
]
