"""A trivial real nanobrain step used as a dispatch-test fixture.

``GlobusComputeExecutor`` uses dispatch approach (B): the remote endpoint
reconstructs the step via ``step_class.from_config(<path>)`` and calls
``step.process(input_data)``. To prove that reconstruction works WITHOUT a
live Globus endpoint, the unit test runs the module-level
``_run_step_on_endpoint`` worker function IN-PROCESS against this real step
+ its YAML config.

It is a genuine ``BaseStep`` (``from_config`` only, ``async def process``,
``self.nb_logger``) — deliberately the smallest possible real step so the
test exercises real framework reconstruction, not a mock.
"""

from __future__ import annotations

from typing import Any, Dict

from nanobrain.core.step import BaseStep, StepConfig


class TrivialEchoStep(BaseStep):
    """Echoes its input back under the ``echoed`` key, uppercasing strings."""

    COMPONENT_TYPE: str = "trivial_echo_step"

    @classmethod
    def _get_config_class(cls):
        return StepConfig

    async def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        self.nb_logger.info("TrivialEchoStep %r: echoing input", self.name)
        payload = input_data
        if isinstance(payload, dict) and len(payload) == 1:
            # Unwrap a single-key trigger envelope if present.
            (only_value,) = payload.values()
            payload = only_value
        if isinstance(payload, str):
            payload = payload.upper()
        return {"echoed": payload}


__all__ = ["TrivialEchoStep"]
