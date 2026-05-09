"""SkeletonLoaderStep (G17 part 1).

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G17`` and
``apecx-mcp-integration/docs/agent_workflow_authoring.md §5``: resolves
``skeleton_id + skeleton_version`` against a SkeletonRegistry and emits
the resolved Skeleton's body + holes for the downstream PlanLoweringStep.

This is Step 1 of the lowering pipeline (per agent_workflow_authoring.md §5):
"Skeleton resolution. Resolve skeleton_id + skeleton_version against the
content-addressed registry. Fetch skeleton.yml and skeleton.schema.json.
If the version is a semver tag, resolve it to a digest now and record
the resolved digest in the lowered YAML's provenance header."

The step is intentionally narrow: input = a dict containing
``{skeleton_id, skeleton_version}`` (typically extracted from an
ExecutionPlanConfig); output = the resolved Skeleton instance + its
content_hash for the lowering pipeline's provenance header.

Registry source:
- The registry is passed as a dependency at ``from_config`` time
  (via ``kwargs['skeleton_registry']``).
- A workflow that uses SkeletonLoaderStep MUST inject its registry
  via the orchestrator's loader. v1 doesn't have a "default global
  registry" — the registry is application-side state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict


from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.step import BaseStep, StepConfig

from .skeleton import SkeletonRegistry

logger = logging.getLogger(__name__)


class SkeletonLoaderStepConfig(StepConfig):
    """Configuration for SkeletonLoaderStep.

    No skeleton-specific config beyond the standard StepConfig — the
    registry is passed at from_config time via kwargs (it's a
    dependency, not a configurable parameter).
    """
    pass


class SkeletonLoaderStep(BaseStep):
    """G17 — Step 1 of the lowering pipeline.

    Input dict (process input):
        ``skeleton_id``: str — the skeleton handle
        ``skeleton_version``: str — semver tag OR content hash (12 or 64 char)

    Output dict:
        ``skeleton_id``: str — echo of input
        ``skeleton_version``: str — the AS-RESOLVED hash (always the
            content_hash, even if input was a semver tag)
        ``skeleton``: Skeleton — the resolved instance
        ``content_hash``: str — same as skeleton_version above
        ``body``: str — convenience echo of the skeleton's YAML body
        ``holes``: dict — convenience echo of the skeleton's holes
    """

    COMPONENT_TYPE: str = "skeleton_loader_step"
    REQUIRED_CONFIG_FIELDS = ["name"]

    @classmethod
    def _get_config_class(cls):
        return SkeletonLoaderStepConfig

    @classmethod
    def resolve_dependencies(cls, component_config, **kwargs):
        deps = super().resolve_dependencies(component_config, **kwargs)
        registry = kwargs.get("skeleton_registry")
        if registry is None:
            raise ComponentConfigurationError(
                "FAIL-FAST: SkeletonLoaderStep requires a skeleton_registry "
                "dependency. Pass via from_config(path, skeleton_registry=...). "
                "v1 has no default global registry — apecx-mcp-side code "
                "owns the registry."
            )
        if not isinstance(registry, SkeletonRegistry):
            raise ComponentConfigurationError(
                f"FAIL-FAST: SkeletonLoaderStep skeleton_registry must be a "
                f"SkeletonRegistry instance, got {type(registry).__name__}"
            )
        deps["skeleton_registry"] = registry
        return deps

    def _init_from_config(
        self,
        config: SkeletonLoaderStepConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        super()._init_from_config(config, component_config, dependencies)
        self._registry: SkeletonRegistry = dependencies["skeleton_registry"]

    @property
    def registry(self) -> SkeletonRegistry:
        return self._registry

    async def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        if not isinstance(input_data, dict):
            raise ComponentConfigurationError(
                f"FAIL-FAST: SkeletonLoaderStep {self.name!r} input_data "
                f"must be a dict with skeleton_id + skeleton_version, got "
                f"{type(input_data).__name__}"
            )
        skeleton_id = input_data.get("skeleton_id")
        skeleton_version = input_data.get("skeleton_version")
        if not skeleton_id or not skeleton_version:
            raise ComponentConfigurationError(
                f"FAIL-FAST: SkeletonLoaderStep {self.name!r} input_data "
                f"missing required keys: skeleton_id={skeleton_id!r}, "
                f"skeleton_version={skeleton_version!r}"
            )

        try:
            skeleton = self._registry.lookup(skeleton_id, skeleton_version)
        except KeyError as e:
            # Re-raise as ComponentConfigurationError to keep the
            # FAIL-FAST contract uniform across the lowering pipeline.
            raise ComponentConfigurationError(str(e)) from e

        return {
            "skeleton_id": skeleton.skeleton_id,
            # Always emit the AS-RESOLVED content_hash, even if the
            # caller passed a semver tag. The lowering pipeline records
            # this in the lowered YAML's provenance header.
            "skeleton_version": skeleton.content_hash,
            "skeleton": skeleton,
            "content_hash": skeleton.content_hash,
            "body": skeleton.body,
            "holes": skeleton.holes,
        }
