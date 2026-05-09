"""Tests for ``UnifiedToolDescriptor.from_python_callable``.

This factory mirrors Rhea / FastMCP's auto-generation pattern:
the author writes a regular Python function with type hints + docstring,
the framework introspects the signature to produce a UTD without
hand-authoring the 30+ fields.

Coverage:
1. Basic happy path: typed callable → UTD with input/output specs.
2. Optional parameters: defaults are recorded; required=False.
3. *args / **kwargs are skipped (UTD has no variadic concept).
4. Untyped parameters get ``Any`` type.
5. Docstring → summary + long_description.
6. Method (``self``) is filtered out.
7. Override mechanism — author-supplied fields trump auto-derived.
8. descriptor_id sanitization for module names like ``__main__``.
9. provenance_pin defaults to module.qualname; override available.
10. Round-trip through descriptor_hash — same callable, same hash.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from nanobrain.core.unified_tool_descriptor import (
    UnifiedToolDescriptor,
    UTDCostEstimate,
)


# Module-level callables used as fixtures (so descriptor_id contains
# this test module's name and stays stable across runs).

def _typed_search(query: str, max_results: int = 10) -> dict:
    """Search the catalogue for a query string.

    Returns up to max_results matches as a dict.
    """
    return {}


def _untyped_fn(x, y=5):
    """Untyped function for the Any-fallback test."""
    return x + y


def _variadic_fn(*args, **kwargs) -> Any:
    """Function with *args/**kwargs — UTD has no variadic concept."""
    return args


class _FakeToolInstance:
    """Method-on-instance fixture for the self-filtering test."""

    def my_method(self, x: int) -> int:
        """A method with a self parameter."""
        return x * 2


# ---------------------------------------------------------------------------
# 1-2. Happy path + defaults
# ---------------------------------------------------------------------------

class TestHappyPath:

    def test_typed_callable_inputs_outputs(self):
        utd = UnifiedToolDescriptor.from_python_callable(
            _typed_search, backend="native", version="0.1.0",
        )
        # Inputs: query (required, str, no default) + max_results (optional, int, 10)
        names = [i.name for i in utd.inputs]
        assert names == ["query", "max_results"]
        types = [i.type for i in utd.inputs]
        assert types == ["str", "int"]
        required = [i.required for i in utd.inputs]
        assert required == [True, False]
        defaults = [i.default for i in utd.inputs]
        assert defaults == [None, 10]
        # Output
        assert len(utd.outputs) == 1
        assert utd.outputs[0].name == "return"
        assert utd.outputs[0].type == "dict"

    def test_summary_and_long_description_from_docstring(self):
        utd = UnifiedToolDescriptor.from_python_callable(_typed_search)
        assert utd.summary == "Search the catalogue for a query string."
        assert "Returns up to max_results matches" in utd.long_description

    def test_display_name_is_qualname(self):
        utd = UnifiedToolDescriptor.from_python_callable(_typed_search)
        assert utd.display_name == "_typed_search"


# ---------------------------------------------------------------------------
# 3-4. Edge cases on signature
# ---------------------------------------------------------------------------

class TestSignatureEdgeCases:

    def test_variadic_args_kwargs_skipped(self):
        utd = UnifiedToolDescriptor.from_python_callable(_variadic_fn)
        # No inputs because *args + **kwargs are deliberately skipped
        assert len(utd.inputs) == 0

    def test_untyped_parameters_become_any(self):
        utd = UnifiedToolDescriptor.from_python_callable(_untyped_fn)
        types = [i.type for i in utd.inputs]
        assert types == ["Any", "Any"]
        # Defaults preserved
        defaults = [i.default for i in utd.inputs]
        assert defaults == [None, 5]

    def test_self_param_filtered_for_methods(self):
        # Bind the method on an instance so self is the first param
        instance = _FakeToolInstance()
        utd = UnifiedToolDescriptor.from_python_callable(
            _FakeToolInstance.my_method,  # unbound method
        )
        names = [i.name for i in utd.inputs]
        assert "self" not in names
        assert names == ["x"]


# ---------------------------------------------------------------------------
# 5-6. Override mechanism
# ---------------------------------------------------------------------------

class TestOverrides:

    def test_overrides_replace_auto_derived(self):
        utd = UnifiedToolDescriptor.from_python_callable(
            _typed_search,
            display_name="Custom Display",
            summary="Custom Summary",
        )
        assert utd.display_name == "Custom Display"
        assert utd.summary == "Custom Summary"
        # Long description still auto-derived from docstring
        assert "Returns up to max_results matches" in utd.long_description

    def test_cost_estimate_override(self):
        utd = UnifiedToolDescriptor.from_python_callable(
            _typed_search,
            cost_estimate={
                "estimated_seconds": 5.0,
                "estimated_usd": 0.01,
                "confidence": "high",
            },
        )
        assert utd.cost_estimate is not None
        assert utd.cost_estimate.estimated_seconds == 5.0
        assert utd.cost_estimate.confidence == "high"

    def test_determinism_override(self):
        # Valid DeterminismClass values are R1, R2, R3.
        utd = UnifiedToolDescriptor.from_python_callable(
            _typed_search, determinism="R1",
        )
        assert utd.determinism == "R1"


# ---------------------------------------------------------------------------
# 7. descriptor_id grammar handling
# ---------------------------------------------------------------------------

class TestDescriptorIdGrammar:

    def test_descriptor_id_default_includes_module_and_qualname(self):
        utd = UnifiedToolDescriptor.from_python_callable(
            _typed_search, backend="native", version="0.1.0",
        )
        # backend:tool_id@version
        assert utd.descriptor_id.startswith("native:")
        assert utd.descriptor_id.endswith("@0.1.0")
        assert "_typed_search" in utd.descriptor_id

    def test_descriptor_id_explicit_override(self):
        utd = UnifiedToolDescriptor.from_python_callable(
            _typed_search,
            descriptor_id="native:my.custom.tool@2.0.0",
        )
        assert utd.descriptor_id == "native:my.custom.tool@2.0.0"

    def test_descriptor_id_handles_underscore_module(self):
        """A function defined in __main__ (or any module starting with
        non-letter) must produce a valid descriptor_id; the grammar
        requires the FIRST char of tool_id to be [a-z]."""
        # Simulate by setting __module__ on a local function:
        def fake_fn() -> None:
            return None
        fake_fn.__module__ = "__main__"

        utd = UnifiedToolDescriptor.from_python_callable(fake_fn)
        # Sanitization: leading non-letters get an "fn_" prefix
        assert utd.descriptor_id.startswith("native:fn_")


# ---------------------------------------------------------------------------
# 8. Provenance pin
# ---------------------------------------------------------------------------

class TestProvenancePin:

    def test_provenance_class_path_default(self):
        utd = UnifiedToolDescriptor.from_python_callable(_typed_search)
        # module.qualname
        assert utd.provenance_pin.class_path.endswith("._typed_search")

    def test_provenance_class_path_override(self):
        utd = UnifiedToolDescriptor.from_python_callable(
            _typed_search,
            provenance_class_path="custom.module.path",
        )
        assert utd.provenance_pin.class_path == "custom.module.path"


# ---------------------------------------------------------------------------
# 9. Determinism / hash consistency
# ---------------------------------------------------------------------------

class TestHashConsistency:

    def test_same_callable_same_hash(self):
        u1 = UnifiedToolDescriptor.from_python_callable(_typed_search)
        u2 = UnifiedToolDescriptor.from_python_callable(_typed_search)
        assert u1.descriptor_hash == u2.descriptor_hash

    def test_different_signatures_different_hashes(self):
        def alt(query: str, n: int = 5) -> dict:
            """Alt search."""
            return {}
        u1 = UnifiedToolDescriptor.from_python_callable(_typed_search)
        u2 = UnifiedToolDescriptor.from_python_callable(alt)
        # Different param names → different inputs → different hashes
        assert u1.descriptor_hash != u2.descriptor_hash


# ---------------------------------------------------------------------------
# 10. Brutal-truth limits documented in docstring
# ---------------------------------------------------------------------------

class TestDocumentedLimits:

    def test_subscripted_type_preserved_as_string(self):
        """Subscripted type hints (Dict[str, int]) are preserved as
        their str() repr (with the 'typing.' prefix stripped). This
        is more graceful than collapsing to the bare type name —
        downstream consumers that care about the parameterization
        get to see it."""
        from typing import Dict as TypingDict

        def parameterized(x: TypingDict[str, int]) -> dict:
            """Param dict."""
            return {}
        utd = UnifiedToolDescriptor.from_python_callable(parameterized)
        # Type preserves the parameterization (the typing.Dict alias
        # name comes through whatever the runtime picks):
        assert "Dict" in utd.inputs[0].type
        assert "[str, int]" in utd.inputs[0].type

    def test_default_determinism_is_r3(self):
        """Default is the LEAST assertive (safest) classification.
        Authors must opt-in to stronger guarantees."""
        utd = UnifiedToolDescriptor.from_python_callable(_typed_search)
        assert utd.determinism == "R3"
