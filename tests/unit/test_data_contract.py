"""Project A Step 1: the pure data-unit contract relation. No I/O, no mocks."""

from __future__ import annotations

import pytest

from nanobrain.core.data_contract import Contract, compatible, parse_contract


def _ok(p, c):
    return compatible(p, c)[0]


def test_kind_mismatch_incompatible():
    assert not _ok(Contract("record"), Contract("file"))
    assert not _ok(Contract("text"), Contract("collection"))


def test_text_always_compatible():
    assert _ok(Contract("text"), Contract("text"))


def test_record_width_producer_superset_ok():
    p = parse_contract({"kind": "record", "required_keys": ["a", "b", "c"]})
    c = parse_contract({"kind": "record", "required_keys": ["a", "b"]})
    assert _ok(p, c)  # producer guarantees a superset -> ok


def test_record_missing_required_key_breaks():
    p = parse_contract({"kind": "record", "required_keys": ["a"]})
    c = parse_contract({"kind": "record", "required_keys": ["a", "b"]})
    assert not _ok(p, c)  # consumer needs 'b', producer doesn't guarantee it


def test_record_additive_evolution():
    base = parse_contract({"kind": "record", "required_keys": ["a"]})
    added = parse_contract({"kind": "record", "required_keys": ["a", "b"]})
    # Producer ADDS a key -> still compatible with a consumer needing only 'a'.
    assert _ok(added, base)
    # Producer REMOVES a key the consumer needs -> breaks.
    assert not _ok(base, added)


def test_record_depth_nested_and_scalar():
    p = parse_contract(
        {"kind": "record", "required": {"x": {"kind": "text"}, "y": {"kind": "record", "required_keys": ["z"]}}}
    )
    c_ok = parse_contract(
        {"kind": "record", "required": {"x": {"kind": "text"}, "y": {"kind": "record", "required_keys": ["z"]}}}
    )
    assert _ok(p, c_ok)
    # Nested kind mismatch on a required key -> incompatible.
    c_bad = parse_contract({"kind": "record", "required": {"x": {"kind": "file"}}})
    assert not _ok(p, c_bad)


def test_undeclared_value_kind_is_any():
    # required_keys (kinds = any) producer vs a consumer requiring a typed value -> any side
    # undeclared makes the depth check pass.
    p = parse_contract({"kind": "record", "required_keys": ["x"]})  # x: any
    c = parse_contract({"kind": "record", "required": {"x": {"kind": "text"}}})
    assert _ok(p, c)


def test_collection_element():
    p = parse_contract({"kind": "collection", "element": {"kind": "text"}})
    assert _ok(p, parse_contract({"kind": "collection", "element": {"kind": "text"}}))
    assert not _ok(p, parse_contract({"kind": "collection", "element": {"kind": "file"}}))
    # undeclared element -> any -> compatible
    assert _ok(p, parse_contract({"kind": "collection"}))


def test_file_extensions():
    p = parse_contract({"kind": "file", "extensions": ["fasta"]})
    assert _ok(p, parse_contract({"kind": "file", "extensions": ["fasta", "fa"]}))  # subset ok
    assert not _ok(p, parse_contract({"kind": "file", "extensions": ["pdb"]}))  # not accepted
    assert _ok(p, parse_contract({"kind": "file"}))  # consumer undeclared -> any


def test_parse_fail_loud_on_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        parse_contract({"kind": "blob"})
    with pytest.raises(ValueError):
        parse_contract("not a dict")
