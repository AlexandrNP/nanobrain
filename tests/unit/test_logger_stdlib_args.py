"""NanoBrainLogger accepts stdlib %-format positional args.

Source: 2026-05-11 audit found 4 latent call sites passing
stdlib-style positional args to NanoBrainLogger.info/.warning
(``core/trigger.py:192,217`` + ``pubmed_client.py:643,655``).
Pre-fix, the method signature was ``(self, message: str, **kwargs)``
— so any positional arg beyond the message crashed with
``TypeError: ... takes 2 positional arguments but N were given``
ON THE EXACT BRANCH the diagnostic was meant to clarify (timeout,
0-results, error path). The bug shape: framework's own error path
crashes louder than the real failure, masking it.

This test pins three properties:

1. The stdlib-style %-format calling convention now works:
   ``logger.warning("got %d for %r", n, q)`` formats correctly.
2. The canonical f-string form still works:
   ``logger.warning(f"got {n} for {q!r}")`` — no regression.
3. Bad format strings do NOT crash the caller (logging primitives
   must be best-effort, not load-bearing). A TypeError on
   ``"got %d" % ("string",)`` falls back to a safe concat.
"""
from __future__ import annotations

import logging

import pytest

from nanobrain.core.logging_system import (
    NanoBrainLogger,
    get_logger,
)


@pytest.fixture
def logger() -> NanoBrainLogger:
    """A fresh NanoBrainLogger for each test (no shared state)."""
    return get_logger("test_logger_stdlib_args")


# ---------------------------------------------------------------------------
# Property 1 — stdlib-style %-args supported
# ---------------------------------------------------------------------------


def test_warning_accepts_single_positional_arg(logger, caplog):
    """``logger.warning("template %d", n)`` — the exact shape that
    crashed ``core/trigger.py:217`` before the fix."""
    with caplog.at_level(logging.WARNING):
        logger.warning("Timeout waiting for %d background tasks", 5)
    # The %-formatted message lands in the log record.
    assert any(
        "Timeout waiting for 5 background tasks" in rec.message
        for rec in caplog.records
    )


def test_info_accepts_multiple_positional_args(logger, caplog):
    """``logger.info("got %d for %r", n, q)`` — the shape from
    ``pubmed_client.py:655``."""
    with caplog.at_level(logging.INFO):
        logger.info(
            "PubMed returned %d references for protein_type=%r",
            7,
            "envelope",
        )
    assert any(
        "PubMed returned 7 references for protein_type='envelope'"
        in rec.message
        for rec in caplog.records
    )


def test_all_visible_levels_accept_positional_args(logger, caplog):
    """Every visible level method — info, warning, error, critical —
    must accept the same calling convention. (DEBUG/TRACE filtering
    is governed by NanoBrainLogger's own LogLevel enum, not caplog
    — they're out of scope for this signature-contract test.)"""
    with caplog.at_level(logging.INFO):
        logger.info("i=%d", 2)
        logger.warning("w=%d", 3)
        logger.error("e=%d", 4)
        logger.critical("c=%d", 5)

    rendered = "\n".join(r.message for r in caplog.records)
    assert "i=2" in rendered
    assert "w=3" in rendered
    assert "e=4" in rendered
    assert "c=5" in rendered


def test_trace_and_debug_accept_positional_args_without_crash(logger):
    """trace() and debug() ALSO accept positional args; their
    visibility is governed by NanoBrainLogger's own level filter,
    but the signature MUST accept the stdlib calling convention so
    development-time enablement of those levels doesn't crash the
    caller. We assert no exception — visibility is a separate concern."""
    logger.trace("t=%d", 1)
    logger.debug("d=%d", 1)


# ---------------------------------------------------------------------------
# Property 2 — canonical f-string form preserved (no regression)
# ---------------------------------------------------------------------------


def test_warning_fstring_path_unchanged(logger, caplog):
    """Existing f-string callers MUST keep working bit-for-bit."""
    with caplog.at_level(logging.WARNING):
        n = 42
        logger.warning(f"already-formatted message with {n}")
    assert any(
        "already-formatted message with 42" in r.message
        for r in caplog.records
    )


def test_warning_with_kwargs_still_works(logger, caplog):
    """Kwargs (the canonical structured-logging path) still flow
    through, even when positional args are present."""
    with caplog.at_level(logging.WARNING):
        logger.warning("event %s happened", "X", request_id="req-123")
    # The formatted message lands.
    rendered = "\n".join(r.message for r in caplog.records)
    assert "event X happened" in rendered
    # The kwargs propagate into the structured payload (we don't
    # assert on the exact JSON shape — that's an implementation
    # detail — but the kwargs path MUST not crash when combined
    # with positional args, which is the point).


# ---------------------------------------------------------------------------
# Property 3 — bad format strings fall back, do not crash
# ---------------------------------------------------------------------------


def test_warning_mismatched_format_does_not_crash(logger, caplog):
    """``"%d" % ("string",)`` raises TypeError in Python. The
    logger MUST NOT propagate that — logging is best-effort, never
    load-bearing. The fallback surfaces both template + args so
    the operator can debug."""
    with caplog.at_level(logging.WARNING):
        logger.warning("count is %d", "not-an-int")  # type: ignore[arg-type]
    rendered = "\n".join(r.message for r in caplog.records)
    # Either the fallback or the formatted-with-coercion path is
    # acceptable; what we forbid is the call CRASHING.
    assert "not-an-int" in rendered or "unformattable-args" in rendered


def test_warning_too_few_format_args_does_not_crash(logger, caplog):
    """``"got %d for %s" % (5,)`` — Python raises TypeError. Same
    fallback contract."""
    with caplog.at_level(logging.WARNING):
        logger.warning("got %d for %s", 5)
    rendered = "\n".join(r.message for r in caplog.records)
    assert (
        "unformattable-args=(5,)" in rendered
        or "got 5 for" in rendered  # if the impl tolerantly handles
    )


def test_warning_extra_format_args_does_not_crash(logger, caplog):
    """``"%d" % (5, 7)`` — extra args raise TypeError. Same fallback."""
    with caplog.at_level(logging.WARNING):
        logger.warning("got %d", 5, 7)
    rendered = "\n".join(r.message for r in caplog.records)
    # Either fallback OR Python's %-format tolerates this implicitly;
    # the only forbidden outcome is a propagating exception.
    assert "5" in rendered
