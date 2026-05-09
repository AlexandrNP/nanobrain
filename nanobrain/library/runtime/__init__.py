"""nanobrain.library.runtime — long-running workflow lifecycle primitives.

Currently exports:

- ``WorkflowRunner`` — G21 v1: schedules detached workflow runs, owns
  the per-task asyncio task registry, and persists status to an
  in-memory or SQLite task store.
- ``DetachedTaskHandle`` — the dataclass returned by ``run_detached``.
- ``WorkflowRunnerConfig`` — Pydantic config for the runner.
"""

from .workflow_runner import (
    DetachedTaskHandle,
    InMemoryTaskStore,
    PauseSignal,
    PostgresTaskStore,
    SqliteTaskStore,
    TaskStore,
    WorkflowRunner,
    WorkflowRunnerConfig,
    current_pause_signal,
)
from .entry_triggers import (
    EntryStateStore,
    FileEntryStateStore,
    InMemoryEntryStateStore,
    WorkflowEntryTrigger,
    WorkflowEntryTriggerConfig,
)

__all__ = [
    "DetachedTaskHandle",
    "InMemoryTaskStore",
    "PauseSignal",
    "PostgresTaskStore",
    "SqliteTaskStore",
    "TaskStore",
    "WorkflowRunner",
    "WorkflowRunnerConfig",
    "current_pause_signal",
    "EntryStateStore",
    "FileEntryStateStore",
    "InMemoryEntryStateStore",
    "WorkflowEntryTrigger",
    "WorkflowEntryTriggerConfig",
]
