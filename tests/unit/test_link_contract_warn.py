"""Project A Step 1b: the WARN-only link contract checker (Workflow._check_link_contracts).

Tested via the return value (the method logs AND returns the reason, deterministically).
Non-binding: incompatible/malformed -> a warning reason, NEVER a raise. Gradual: an
undeclared side is skipped. No mocks. (self is unused by the method -> call with None.)
"""

from __future__ import annotations

from nanobrain.core.workflow import Workflow


class _Cfg:
    def __init__(self, contract):
        self.contract = contract


class _DU:
    def __init__(self, contract, name="du"):
        self.name = name
        self.config = _Cfg(contract)


def _check(src_contract, tgt_contract):
    return Workflow._check_link_contracts(None, "L1", _DU(src_contract), _DU(tgt_contract))


def test_incompatible_record_warns():
    # consumer needs 'b' the producer doesn't guarantee -> mismatch warning.
    r = _check({"kind": "record", "required_keys": ["a"]}, {"kind": "record", "required_keys": ["a", "b"]})
    assert r is not None and "MISMATCH" in r


def test_kind_mismatch_warns():
    r = _check({"kind": "record", "required_keys": ["a"]}, {"kind": "file"})
    assert r is not None and "MISMATCH" in r


def test_compatible_no_warn():
    # producer guarantees a superset -> compatible -> no warning.
    assert _check({"kind": "record", "required_keys": ["a", "b"]}, {"kind": "record", "required_keys": ["a"]}) is None


def test_undeclared_side_skipped():
    assert _check({"kind": "record", "required_keys": ["a"]}, None) is None
    assert _check(None, {"kind": "file"}) is None
    assert _check(None, None) is None


def test_malformed_contract_warns_not_raises():
    # bad kind -> parse_contract raises internally -> caught -> "skipped" warning, NOT a raise.
    r = _check({"kind": "bogus"}, {"kind": "bogus"})
    assert r is not None and "skipped" in r
