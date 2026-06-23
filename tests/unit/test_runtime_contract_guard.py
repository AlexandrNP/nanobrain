"""Project A Step 2: the runtime DataUnitMemory.set() contract guard.

config_version>=3 (active via the _active_config_version ContextVar) makes a contract
violation RAISE; <3 WARNs (non-binding). None/no-contract are skipped. Real DataUnitMemory
via from_config, real set()/get(). No mocks. (asyncio.run copies the current context, so
the ContextVar set in the test is visible inside set().)
"""

from __future__ import annotations

import asyncio

import pytest

from nanobrain.core.data_contract import ContractViolationError, _active_config_version
from nanobrain.core.data_unit import DataUnitMemory


def _du(contract=None):
    cfg = {"class": "nanobrain.core.data_unit.DataUnitMemory", "name": "d"}
    if contract:
        cfg["contract"] = contract
    return DataUnitMemory.from_config(cfg)


def _with_version(v, coro_fn):
    tok = _active_config_version.set(v)
    try:
        return asyncio.run(coro_fn())
    finally:
        _active_config_version.reset(tok)


def test_v3_violation_raises():
    du = _du({"kind": "record", "required_keys": ["a"]})
    with pytest.raises(ContractViolationError):
        _with_version(3, lambda: du.set({"wrong": 1}))  # missing required 'a'


def test_v3_valid_value_stored():
    du = _du({"kind": "record", "required_keys": ["a"]})
    _with_version(3, lambda: du.set({"a": 1}))
    assert asyncio.run(du.get()) == {"a": 1}


def test_v2_violation_warns_not_raises():
    du = _du({"kind": "record", "required_keys": ["a"]})
    # Non-binding under v2: no raise, value still stored.
    _with_version(2, lambda: du.set({"wrong": 1}))
    assert asyncio.run(du.get()) == {"wrong": 1}


def test_none_skipped_at_v3():
    du = _du({"kind": "record", "required_keys": ["a"]})
    # None is a framework control write, not data -> skipped even at v3 (no raise).
    _with_version(3, lambda: du.set(None))


def test_no_contract_never_checked_at_v3():
    du = _du()  # no contract declared
    _with_version(3, lambda: du.set({"anything": 1}))  # no contract -> no check, no raise
    assert asyncio.run(du.get()) == {"anything": 1}
