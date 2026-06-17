"""Step events — G37 cascade-aware step-level event hook.

eval_03 Round 3 G37: G4-completion (sibling commit) wraps
``BaseStep._execute_process`` for *audit-trail* recording — the
provenance recorder writes one record per process() call. G37
exposes the *live* publish stream of those same events so external
consumers (the integration's provenance/recorder.py, dashboard
ticker, log shipper) can subscribe.

Pre-G37 the integration's recorder was workflow-run-granular only —
step-internal events from G4-completion's recorder were on disk but
not visible to the consumer's process. The integration had to
either (a) tail the JSONL sink file (fragile, race-prone) or (b)
monkey-patch _execute_process (brittle).

Post-G37 ``subscribe_to_step_events(subscriber)`` is the supported
hook. The subscriber sees the same events G4-completion records,
synchronously, in the calling task's context.

## P4+c decision (open question §8.9)

**v1 event schema frozen** with explicit ``event_schema_version: int``
on every emitted event. Consumers store the schema version in their
durable records so a future v2 schema can be parsed alongside v1
without ambiguity. The framework will not break v1 — additive
fields will bump the version; semantic changes will keep both
versions for at least one minor release.

## Schema

Three event types:

  * ``step_start`` — emitted before ``process()`` is called. Carries
    ``inputs`` (the dict passed to process).
  * ``step_complete`` — emitted after ``process()`` returns. Carries
    ``outputs`` (the dict returned by process; or wrapped under
    ``_result`` for non-dict returns) and ``duration_seconds``.
  * ``step_failed`` — emitted when ``process()`` raises. Carries
    ``exception`` (type, message, optional truncated traceback) and
    ``duration_seconds``.

All events carry: ``event_schema_version`` (=1), ``event_type``,
``step_name``, ``run_id`` (when available), ``timestamp_iso``.

## Subscriber failure isolation

If a subscriber raises during ``on_step_event``, the framework
SWALLOWS the exception and logs a warning. Subscriber bugs MUST NOT
break the step they observe. Mirrors the G4-completion recorder-
failure-non-fatal contract.

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G37;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.9 (P4+c).
"""
from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional

logger = logging.getLogger(__name__)


EVENT_SCHEMA_VERSION = 1
# step_progress is ADDITIVE to the frozen v1 schema: the StepEvent FIELDS are unchanged, so
# event_schema_version stays 1. A subscriber that switches on event_type and doesn't handle
# step_progress simply ignores it (backward-compatible). It carries incremental, mid-process()
# progress so a long step is not silent between step_start and step_complete.
StepEventType = Literal["step_start", "step_complete", "step_failed", "step_progress"]


@dataclass(frozen=True)
class StepEvent:
    """One step-lifecycle event. v1 schema (frozen)."""

    event_type: StepEventType
    step_name: str
    run_id: Optional[str]
    timestamp_iso: str
    payload: Dict[str, Any] = field(default_factory=dict)
    event_schema_version: int = EVENT_SCHEMA_VERSION


StepEventSubscriber = Callable[[StepEvent], None]


# Contextvar holding the active subscriber stack. Each subscribe()
# call appends; the context manager pops on exit. Concurrent asyncio
# tasks see their own stack (PEP 567).
#
# Adversarial-probe note (2026-05-11): the default is an EMPTY TUPLE,
# not an empty list. Mutable defaults on contextvars are a tripwire —
# if any code path mutated ``_subscriber_stack.get()`` in place
# (rather than copying via ``list(current) + [sub]``), the mutation
# would be shared across every context that hadn't explicitly
# ``.set()``-ed. The tuple default makes that bug class
# impossible — mutation raises ``AttributeError`` immediately.
_subscriber_stack: contextvars.ContextVar[
    "tuple[StepEventSubscriber, ...]"
] = contextvars.ContextVar("step_event_subscribers", default=())


@contextmanager
def subscribe_to_step_events(
    subscriber: StepEventSubscriber,
) -> Iterator[StepEventSubscriber]:
    """Install ``subscriber`` for the duration of the with-block.

    Multiple nested subscriptions stack — every active subscriber
    gets every event.

    Subscriber callable shape::

        def on_event(event: StepEvent) -> None:
            ...

    The subscriber executes synchronously in the calling task's
    context. Long-running work should be deferred (queue + worker
    pattern) — the framework will not wait for the subscriber.
    """
    current = _subscriber_stack.get()
    # Build an immutable tuple — see _subscriber_stack docstring on
    # why the default is a tuple, not a list.
    new_stack = tuple(current) + (subscriber,)
    token = _subscriber_stack.set(new_stack)
    try:
        yield subscriber
    finally:
        _subscriber_stack.reset(token)


def publish_step_event(event: StepEvent) -> None:
    """Internal API — invoked by ``BaseStep._execute_process`` to
    emit one event to every active subscriber.

    Subscriber exceptions are logged + swallowed so a buggy
    subscriber cannot break the step it observes.
    """
    subscribers = _subscriber_stack.get()
    if not subscribers:
        return
    for sub in subscribers:
        try:
            sub(event)
        except Exception as exc:
            logger.warning(
                "Step-event subscriber %r raised %s: %s — swallowed "
                "to preserve step's actual return path.",
                getattr(sub, "__qualname__", sub),
                type(exc).__name__,
                exc,
            )


def _now_iso() -> str:
    """Helper: current time in UTC ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat()


def _make_step_start_event(
    *,
    step_name: str,
    run_id: Optional[str],
    inputs: Dict[str, Any],
) -> StepEvent:
    return StepEvent(
        event_type="step_start",
        step_name=step_name,
        run_id=run_id,
        timestamp_iso=_now_iso(),
        payload={"inputs": inputs},
    )


def _make_step_complete_event(
    *,
    step_name: str,
    run_id: Optional[str],
    outputs: Any,
    duration_seconds: float,
) -> StepEvent:
    if isinstance(outputs, dict):
        outputs_payload = outputs
    else:
        outputs_payload = {"_result": outputs}
    return StepEvent(
        event_type="step_complete",
        step_name=step_name,
        run_id=run_id,
        timestamp_iso=_now_iso(),
        payload={
            "outputs": outputs_payload,
            "duration_seconds": duration_seconds,
        },
    )


def _make_step_failed_event(
    *,
    step_name: str,
    run_id: Optional[str],
    exception_type: str,
    exception_message: str,
    duration_seconds: float,
    traceback_text: Optional[str] = None,
) -> StepEvent:
    payload: Dict[str, Any] = {
        "exception": {
            "type": exception_type,
            "message": exception_message,
        },
        "duration_seconds": duration_seconds,
    }
    if traceback_text is not None:
        payload["exception"]["traceback"] = traceback_text
    return StepEvent(
        event_type="step_failed",
        step_name=step_name,
        run_id=run_id,
        timestamp_iso=_now_iso(),
        payload=payload,
    )


def _make_step_progress_event(
    *,
    step_name: str,
    run_id: Optional[str],
    message: str,
    data: Optional[Dict[str, Any]] = None,
    fraction: Optional[float] = None,
) -> StepEvent:
    """Build an incremental progress event emitted mid-process() via BaseStep.emit_progress.

    ``message`` is a short human line ("fetched 5000/13000 records"); ``data`` is optional
    structured detail; ``fraction`` (clamped to [0,1]) is optional completion.
    """
    payload: Dict[str, Any] = {"message": message}
    if data is not None:
        payload["data"] = data
    if fraction is not None:
        payload["fraction"] = max(0.0, min(1.0, float(fraction)))
    return StepEvent(
        event_type="step_progress",
        step_name=step_name,
        run_id=run_id,
        timestamp_iso=_now_iso(),
        payload=payload,
    )


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "StepEvent",
    "StepEventSubscriber",
    "StepEventType",
    "_make_step_complete_event",
    "_make_step_failed_event",
    "_make_step_progress_event",
    "_make_step_start_event",
    "publish_step_event",
    "subscribe_to_step_events",
]
