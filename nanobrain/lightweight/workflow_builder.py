"""
Lightweight Workflow Builder
============================

Programmatic workflow construction that generates the same dict shape
the framework's ``WorkflowConfig.from_config`` consumes. Designed for
rapid prototyping, demos, and tests where YAML-first authoring is
heavyweight; the generated config is validated by the framework's
canonical Pydantic schema (G7 v2 defaults apply automatically).

Two ways to consume the builder's output:

1. ``builder.get_config()`` returns the dict; pass to
   ``WorkflowConfig.from_config(...)`` or save to YAML for inspection.
2. ``builder.load()`` is the convenience that builds + validates a
   real ``Workflow`` instance via ``Workflow.from_config(get_config())``.
   This is the canonical seam — equivalent to writing the YAML by hand
   and loading it. Same Pydantic validation, same v2 default-flip
   (auto_transfer-True, gate-aware propagation, path-reference
   rewriting), same FAIL-FAST surface.

What the builder does NOT do:

- Recompose nested step / link / trigger configs from per-component
  YAML files. The dict is in-memory inline; the framework's v2
  path-reference rewriting (G7 Step 4) does not apply because there
  are no path references to rewrite. This is fine — inline is the
  preferred shape for programmatic construction.
- Discover external workflow YAML directories. Use the canonical
  ``Workflow.from_config('path/to/workflow.yml')`` path for that.
"""

from typing import Dict, List, Any, Optional


# ---------------------------------------------------------------------------
# Framework-known import paths
# ---------------------------------------------------------------------------
#
# The lightweight discovery layer (``MinimalConfigDiscovery``) scans
# example YAML files for ``class:`` references — which means framework
# primitives that have NO example YAML in the repo are invisible to the
# builder. That's brittle: most TriggerBase subclasses + most non-Direct
# link classes have no inline YAML example, so the discovery's view of
# the framework is severely undercounted.
#
# We sidestep the brittleness by maintaining a small map of well-known
# framework class paths here. The builder consults this map FIRST; if a
# user names a custom class not in the map, it falls back to discovery.
# This keeps the lightweight builder useful for the framework's OWN
# primitives without depending on the discovery layer's coverage.
#
# Add new framework classes here when they ship in core/ or library/.

_FRAMEWORK_LINK_CLASS_PATHS: Dict[str, str] = {
    "DirectLink": "nanobrain.core.link.DirectLink",
    "ConditionalLink": "nanobrain.core.link.ConditionalLink",
    "TransformLink": "nanobrain.core.link.TransformLink",
    "FileLink": "nanobrain.core.link.FileLink",
    "QueueLink": "nanobrain.core.link.QueueLink",
}

_FRAMEWORK_TRIGGER_CLASS_PATHS: Dict[str, str] = {
    "DataUnitChangeTrigger": "nanobrain.core.trigger.DataUnitChangeTrigger",
    "AllDataReceivedTrigger": "nanobrain.core.trigger.AllDataReceivedTrigger",
    "TimerTrigger": "nanobrain.core.trigger.TimerTrigger",
    "ManualTrigger": "nanobrain.core.trigger.ManualTrigger",
    "EventTrigger": "nanobrain.core.trigger.EventTrigger",
}

# Import discovery - handle both relative and absolute imports
try:
    from .discovery_minimal import MinimalConfigDiscovery
except ImportError:
    # Fallback for isolated testing
    import sys
    from pathlib import Path
    discovery_path = Path(__file__).parent / "discovery_minimal.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("discovery_minimal", discovery_path)
    discovery_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(discovery_module)
    MinimalConfigDiscovery = discovery_module.MinimalConfigDiscovery


class WorkflowBuilder:
    """
    Lightweight workflow builder that generates proper configuration.
    
    BRUTAL TRUTH: This is iteration 1 - basic functionality only.
    No fancy features, just core workflow building with discovered components.
    """
    
    def __init__(self, name: str, description: str = ""):
        """Initialize workflow builder."""
        
        self.name = name
        self.description = description
        
        # Initialize discovery system
        self.discovery = MinimalConfigDiscovery()
        self.discovered_classes = self.discovery.discover_from_files()
        
        # Workflow configuration structure.
        #
        # config_version: 2 is set explicitly so the v2 mutators
        # (G7 Step 3+4 — auto_transfer default-True, path-reference
        # rewriting; G10 Step 2 — gate_semantics propagation) fire
        # uniformly. Even though v2 is the workspace default as of
        # G7 Step 4 (2026-05-09), declaring it explicitly keeps the
        # generated dict self-contained — it survives a hypothetical
        # future config_version default change without semantic drift.
        #
        # The legacy ``version: "2.0"`` field this used to carry was
        # DEAD — the framework loader reads ``config_version``, not
        # ``version``. Removed to eliminate the silent-confusion shape.
        self.workflow_config = {
            "name": name,
            "description": description,
            "config_version": 2,
            "steps": {},
            "links": {},
            "triggers": [],
            "input_data_units": {},
            "output_data_units": {},
        }

        self._step_counter = 0
    
    def add_step(self, step_name: str, component_class: str, **kwargs) -> 'WorkflowBuilder':
        """
        Add a step to the workflow using a discovered component.
        
        Args:
            step_name: Name for this step
            component_class: Class name (e.g., "EnhancedCollaborativeAgent")
            **kwargs: Additional configuration parameters
        
        Returns:
            Self for method chaining
        """
        
        # Validate component exists
        if component_class not in self.discovered_classes:
            available = list(self.discovered_classes.keys())
            raise ValueError(f"Unknown component '{component_class}'. Available: {available}")
        
        # Get component info
        component_info = self.discovered_classes[component_class]
        
        # Build step configuration
        step_config = {
            "name": step_name,
            "class": component_info["class_path"],
            "description": kwargs.get("description", f"Step using {component_class}")
        }
        
        # Add user-provided parameters
        for param, value in kwargs.items():
            if param != "description":  # Already handled above
                step_config[param] = value
        
        # Add to workflow
        self.workflow_config["steps"][step_name] = step_config
        self._step_counter += 1
        
        return self
    
    def add_input(self, name: str, data_unit_type: str = "DataUnitMemory") -> 'WorkflowBuilder':
        """Add input data unit to workflow."""
        
        # Find data unit class
        data_unit_classes = self.discovery.get_classes_by_category("data_unit")
        
        if data_unit_type not in data_unit_classes:
            raise ValueError(f"Unknown data unit type '{data_unit_type}'. Available: {data_unit_classes}")
        
        data_unit_info = self.discovered_classes[data_unit_type]
        
        input_config = {
            "name": name,
            "class": data_unit_info["class_path"],
            "description": f"Input data unit: {name}"
        }
        
        self.workflow_config["input_data_units"][name] = input_config
        return self
    
    def add_output(self, name: str, data_unit_type: str = "DataUnitMemory") -> 'WorkflowBuilder':
        """Add output data unit to workflow."""
        
        # Find data unit class
        data_unit_classes = self.discovery.get_classes_by_category("data_unit")
        
        if data_unit_type not in data_unit_classes:
            raise ValueError(f"Unknown data unit type '{data_unit_type}'. Available: {data_unit_classes}")
        
        data_unit_info = self.discovered_classes[data_unit_type]
        
        output_config = {
            "name": name,
            "class": data_unit_info["class_path"],
            "description": f"Output data unit: {name}"
        }
        
        self.workflow_config["output_data_units"][name] = output_config
        return self
    
    def connect(self, source: str, target: str) -> 'WorkflowBuilder':
        """Connect two components with a default DirectLink.

        Convenience over :meth:`add_link` — equivalent to
        ``add_link(source, target, link_type='direct')``. The generated
        link inherits the workflow's ``config_version: 2`` default, so
        ``auto_transfer: true`` is injected automatically by the
        framework's v2 model_validator (G7 Step 3) when the workflow
        loads. Use :meth:`add_link` directly for ConditionalLink,
        TransformLink, or per-link gate_semantics overrides.
        """
        return self.add_link(source, target, link_type="direct")

    def add_link(
        self,
        source: str,
        target: str,
        link_type: str = "direct",
        condition: Optional[Dict[str, Any]] = None,
        gate_semantics: Optional[str] = None,
        transform_function: Optional[str] = None,
        link_name: Optional[str] = None,
        **kwargs: Any,
    ) -> 'WorkflowBuilder':
        """Add a link to the workflow with full type discrimination.

        Args:
            source: ``"step_name.unit"`` reference for the source.
            target: ``"step_name.unit"`` reference for the target.
            link_type: ``"direct"`` (default), ``"conditional"``,
                ``"transform"``, or ``"file"``. Maps to the
                corresponding ``LinkBase`` subclass discovered by the
                discovery system.
            condition: Required for ``link_type='conditional'``. A G1
                predicate dict, e.g.
                ``{"op": "exists", "field": "value"}``.
            gate_semantics: Optional per-link override
                (``'publish_empty'`` or ``'gate_to_bottom'``). When
                unset, the workflow-level default applies.
            transform_function: Required for ``link_type='transform'``.
                Dotted-path string resolving to a callable.
            link_name: Optional explicit name. Defaults to
                ``link_<N>``.
            **kwargs: Passed through to the generated link config dict.

        Returns ``self`` for chaining.

        Raises ``ValueError`` if the requested link class is not
        discovered, or if a required field for the chosen link_type
        is missing.
        """
        type_to_class = {
            "direct": "DirectLink",
            "conditional": "ConditionalLink",
            "transform": "TransformLink",
            "file": "FileLink",
            "queue": "QueueLink",
        }
        if link_type not in type_to_class:
            raise ValueError(
                f"FAIL-FAST: unknown link_type {link_type!r}. Choose "
                f"from: {sorted(type_to_class.keys())}"
            )
        link_class_name = type_to_class[link_type]
        link_class_path = self._resolve_class_path(
            link_class_name, _FRAMEWORK_LINK_CLASS_PATHS, "link",
        )

        # Per-link-type required-field checks. FAIL-FAST at builder time
        # rather than at workflow-load time — much better authoring UX.
        if link_type == "conditional" and condition is None:
            raise ValueError(
                "FAIL-FAST: ConditionalLink requires a `condition` "
                "predicate dict (G1 declarative form, e.g. "
                "{'op': 'exists', 'field': 'x'})"
            )
        if link_type == "transform" and not transform_function:
            raise ValueError(
                "FAIL-FAST: TransformLink requires a "
                "`transform_function` dotted-path string"
            )

        name = link_name or f"link_{len(self.workflow_config['links'])}"

        link_config: Dict[str, Any] = {
            "name": name,
            "class": link_class_path,
            "source": source,
            "target": target,
        }
        if condition is not None:
            link_config["condition"] = condition
        if gate_semantics is not None:
            link_config["gate_semantics"] = gate_semantics
        if transform_function is not None:
            link_config["transform_function"] = transform_function
        link_config.update(kwargs)

        self.workflow_config["links"][name] = link_config
        return self

    def add_trigger(
        self,
        step_name: Optional[str] = None,
        trigger_type: str = "data_updated",
        trigger_name: Optional[str] = None,
        **kwargs: Any,
    ) -> 'WorkflowBuilder':
        """Add a trigger to the workflow.

        A workflow with NO triggers has no entry point and silently
        does nothing when run — the dominant authoring confusion that
        this method exists to solve.

        Args:
            step_name: When set, the trigger is added to that step's
                inline ``triggers:`` list. When unset, the trigger is
                added to the workflow-level ``triggers:`` list (which
                is inherited from StepConfig because workflows ARE
                steps in nanobrain's model).
            trigger_type: ``"data_updated"`` (default; G1 / built-in),
                ``"all_data_received"``, ``"timer"``, ``"manual"``,
                ``"event"`` (G22).
            trigger_name: Optional name for the trigger; defaults to
                ``trigger_<N>``.
            **kwargs: Passed through to the trigger config (e.g.
                ``timer_interval_ms``, ``event_filter``, etc.).

        Returns ``self`` for chaining.

        Raises ``ValueError`` when ``step_name`` is set but unknown,
        or when ``trigger_type`` is not in the supported set.
        """
        type_to_class = {
            "data_updated": "DataUnitChangeTrigger",
            "all_data_received": "AllDataReceivedTrigger",
            "timer": "TimerTrigger",
            "manual": "ManualTrigger",
            "event": "EventTrigger",
        }
        if trigger_type not in type_to_class:
            raise ValueError(
                f"FAIL-FAST: unknown trigger_type {trigger_type!r}. "
                f"Choose from: {sorted(type_to_class.keys())}"
            )
        cls_name = type_to_class[trigger_type]
        cls_path = self._resolve_class_path(
            cls_name, _FRAMEWORK_TRIGGER_CLASS_PATHS, "trigger",
        )
        name = trigger_name or f"trigger_{self._next_trigger_index()}"

        trigger_config: Dict[str, Any] = {
            "name": name,
            "class": cls_path,
            "trigger_type": trigger_type,
        }
        trigger_config.update(kwargs)

        if step_name is None:
            self.workflow_config["triggers"].append(trigger_config)
        else:
            if step_name not in self.workflow_config["steps"]:
                raise ValueError(
                    f"FAIL-FAST: cannot attach trigger to unknown step "
                    f"{step_name!r}. Add the step first via add_step()."
                )
            step_entry = self.workflow_config["steps"][step_name]
            step_entry.setdefault("triggers", []).append(trigger_config)
        return self

    def _resolve_class_path(
        self, class_name: str, framework_map: Dict[str, str], category: str,
    ) -> str:
        """Resolve a class name to its full dotted import path.

        Resolution order:
          1. Framework-known map (``_FRAMEWORK_LINK_CLASS_PATHS`` /
             ``_FRAMEWORK_TRIGGER_CLASS_PATHS``). Stable, complete,
             does not depend on discovery's YAML-scan coverage.
          2. Discovery system. For user-defined classes that exist in
             at least one example YAML.

        FAIL-FAST when neither path resolves the name — surfaces an
        explicit error AT BUILDER TIME rather than at workflow load.
        """
        if class_name in framework_map:
            return framework_map[class_name]
        if class_name in self.discovered_classes:
            return self.discovered_classes[class_name]["class_path"]
        framework_known = sorted(framework_map.keys())
        discovered_in_category = (
            self.discovery.get_classes_by_category(category) or []
        )
        raise ValueError(
            f"FAIL-FAST: {category} class {class_name!r} is neither in "
            f"the framework-known map (known: {framework_known}) nor "
            f"discovered (discovered: {sorted(discovered_in_category)}). "
            f"For a custom class, ensure at least one YAML example "
            f"references it OR add the import path to the WorkflowBuilder "
            f"framework map."
        )

    def _next_trigger_index(self) -> int:
        """Walk the workflow + step trigger lists to compute the next
        free trigger index for default naming."""
        count = len(self.workflow_config["triggers"])
        for step_entry in self.workflow_config["steps"].values():
            count += len(step_entry.get("triggers", []) or [])
        return count

    def load(self):
        """Build a real ``Workflow`` instance from the generated config.

        Convenience that closes the loop: the same dict you'd get from
        ``get_config()`` is passed through ``Workflow.from_config()``,
        which exercises the framework's canonical Pydantic validation
        and v2 mutators (G7 Step 3+4 auto_transfer injection,
        G10 Step 2 gate_semantics propagation, etc.).

        Returns:
            A constructed ``Workflow`` ready to ``run()``.

        Raises:
            Whatever ``Workflow.from_config`` raises on validation
            failure — typically ``ComponentConfigurationError`` or
            ``ValueError`` with a ``FAIL-FAST:`` prefix.
        """
        # Lazy import to avoid loading the heavy core stack at module
        # import time (the lightweight builder should stay light).
        from nanobrain.core.workflow import Workflow

        return Workflow.from_config(self.workflow_config)
    
    def get_config(self) -> Dict[str, Any]:
        """Get the generated workflow configuration."""
        return self.workflow_config.copy()
    
    def save_config(self, file_path: str) -> str:
        """Save workflow configuration to file."""
        
        import json
        
        with open(file_path, 'w') as f:
            json.dump(self.workflow_config, f, indent=2)
        
        return file_path
    
    def list_available_components(self, category: Optional[str] = None) -> List[str]:
        """List available components by category."""
        
        if category:
            return self.discovery.get_classes_by_category(category)
        else:
            return list(self.discovered_classes.keys())
    
    def get_component_info(self, component_class: str) -> Optional[Dict[str, Any]]:
        """Get information about a component."""
        
        return self.discovered_classes.get(component_class)
