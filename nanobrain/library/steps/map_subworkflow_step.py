"""MapSubworkflowStep — run an inner workflow once PER item of a list, concurrently.

The map-over-collection primitive. ``SubworkflowStep`` embeds ONE inner workflow
(cached, run once per call); ``RecursiveSubworkflowStep`` runs a FRESH inner per
call along a recursion depth. Neither runs an inner workflow once per element of
an input list and collects the results — that gap is what this step fills.

Given an input dict carrying a list under ``item_list_key`` (plus any
``static_params_keys`` shared by every item), it builds a FRESH inner workflow
instance per item and runs them CONCURRENTLY (bounded by ``max_concurrency``),
collecting each inner run's output into a list under ``output_list_key``.

Why fresh-instance-per-item (load-bearing): ``Workflow.run``/``process`` mutate the
workflow's data units, and ``Workflow.run`` holds a per-instance ``asyncio.Lock``
(nanobrain CLAUDE.md 2026-06-14) — overlapping runs on ONE instance serialize. So
to get real ``asyncio.gather`` concurrency the map MUST give each item its own
Workflow instance, mirroring ``RecursiveSubworkflowStep``'s per-call build
(recursive_subworkflow_step.py:376). Concurrent drives on DISTINCT instances are
safe post-G115/G125 (the ``_active_workflow_id`` ContextVar is copied per gather
task; each instance owns its own state).

Each item is driven through the SAME clear→route→process→poll→collect→gate path as
``SubworkflowStep`` (reused via the parameterized ``_drive_inner(workflow, input)``
seam), so the G37 fast-fail + cached-re-run-clear + single-output flatten all apply
per item.

Silent-failure discipline (degrade-loud PER ITEM): one item's inner run raising
must NOT fail the whole map. Each failing item becomes a NAMED note
``{"_map_item_error": "<type>: <msg>", "_item_index": i}`` in the output list, and
the indices→messages map is surfaced under ``_map_errors``. A caller that needs
strict all-or-nothing checks the emptiness of ``_map_errors``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import Field

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.workflow import Workflow
from nanobrain.library.steps.subworkflow_step import (
    SubworkflowStep,
    SubworkflowStepConfig,
)

logger = logging.getLogger(__name__)


class MapSubworkflowStepConfig(SubworkflowStepConfig):
    """Configuration for ``MapSubworkflowStep`` (extends ``SubworkflowStepConfig``)."""

    item_list_key: str = Field(
        ...,
        description=(
            "Key in this step's input dict holding the LIST to map over. Each "
            "element becomes one inner-workflow run."
        ),
    )
    item_param_key: str = Field(
        ...,
        description=(
            "Key under which each list element is injected into the per-item "
            "input passed to the inner workflow."
        ),
    )
    static_params_keys: List[str] = Field(
        default_factory=list,
        description=(
            "Keys copied UNCHANGED from this step's input into every item's "
            "inner input (e.g. a shared resolution `plan`)."
        ),
    )
    step_input_data_unit_name: Optional[str] = Field(
        default=None,
        description=(
            "Name of THIS step's own input data unit, used to unwrap the trigger "
            "envelope ({<name>: payload}) before reading the item list."
        ),
    )
    max_concurrency: int = Field(
        default=4,
        ge=1,
        description="Max inner workflows running at once (asyncio.Semaphore cap).",
    )
    output_list_key: str = Field(
        default="items",
        description="Key under which the collected per-item results land.",
    )


class MapSubworkflowStep(SubworkflowStep):
    """Run the inner workflow once per list item, concurrently, and collect results."""

    COMPONENT_TYPE: str = "map_subworkflow_step"
    REQUIRED_CONFIG_FIELDS: ClassVar[list] = ["name", "item_list_key", "item_param_key"]

    @classmethod
    def _get_config_class(cls):
        return MapSubworkflowStepConfig

    def _init_from_config(
        self,
        config: MapSubworkflowStepConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        # super() resolves + validates the inner workflow source (builds one cached
        # instance for early FAIL-FAST) and binds the timeout/settle/nest/gate knobs.
        super()._init_from_config(config, component_config, dependencies)

        # Capture the raw source spec so we can build a FRESH instance per item.
        self._map_builder_spec: Optional[str] = (
            config.inner_workflow_builder
            or self.__class__._default_inner_workflow_builder()
        )
        self._map_path = self._inner_workflow_path_resolved  # Path | None

        self._item_list_key: str = config.item_list_key
        self._item_param_key: str = config.item_param_key
        self._static_params_keys: List[str] = list(config.static_params_keys or [])
        self._step_input_du: Optional[str] = config.step_input_data_unit_name
        self._max_concurrency: int = int(config.max_concurrency)
        self._output_list_key: str = config.output_list_key

    def _make_fresh_inner(self) -> Workflow:
        """Build a brand-new inner Workflow instance (state isolation per item)."""
        if self._map_builder_spec is not None:
            return self._build_inner_workflow(self._map_builder_spec)
        return Workflow.from_config(str(self._map_path))

    async def _run_one(self, index: int, item: Any, statics: Dict[str, Any]) -> Dict[str, Any]:
        """Run the inner workflow on one item via the shared drive; degrade-loud."""
        per_item_input = {self._item_param_key: item, **statics}
        try:
            inner = self._make_fresh_inner()
            return await self._drive_inner(inner, per_item_input)
        except Exception as exc:  # noqa: BLE001 — per-item degrade-loud is the contract
            logger.warning(
                "MapSubworkflowStep %s: item %d failed: %s: %s",
                self.name,
                index,
                type(exc).__name__,
                exc,
            )
            return {"_map_item_error": f"{type(exc).__name__}: {exc}", "_item_index": index}

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if not isinstance(input_data, dict):
            raise ComponentConfigurationError(
                f"FAIL-FAST: MapSubworkflowStep {self.name!r} input_data must be a "
                f"dict; got {type(input_data).__name__}"
            )
        # Unwrap the trigger envelope ({<my_input_du>: payload}).
        if (
            self._step_input_du
            and len(input_data) == 1
            and self._step_input_du in input_data
            and isinstance(input_data[self._step_input_du], dict)
        ):
            input_data = input_data[self._step_input_du]

        items = input_data.get(self._item_list_key)
        if not isinstance(items, list):
            raise ComponentConfigurationError(
                f"FAIL-FAST: MapSubworkflowStep {self.name!r} input has no list under "
                f"{self._item_list_key!r} (got {type(items).__name__}). Available keys: "
                f"{sorted(input_data.keys())}"
            )
        statics = {k: input_data[k] for k in self._static_params_keys if k in input_data}

        if not items:
            return {self._output_list_key: [], "_map_errors": {}}

        sem = asyncio.Semaphore(self._max_concurrency)

        async def _guarded(i: int, it: Any) -> Dict[str, Any]:
            async with sem:
                return await self._run_one(i, it, statics)

        results = await asyncio.gather(*(_guarded(i, it) for i, it in enumerate(items)))
        errors = {
            r["_item_index"]: r["_map_item_error"]
            for r in results
            if isinstance(r, dict) and "_map_item_error" in r
        }
        logger.info(
            "MapSubworkflowStep %s: mapped %d item(s), %d error(s)",
            self.name,
            len(items),
            len(errors),
        )
        return {self._output_list_key: results, "_map_errors": errors}


__all__ = ["MapSubworkflowStep", "MapSubworkflowStepConfig"]
