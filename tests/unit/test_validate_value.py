"""Project A Step 2: validate_value — does an ACTUAL value satisfy a Contract? No mocks."""

from __future__ import annotations

from nanobrain.core.data_contract import parse_contract, validate_value


def _ok(spec, value):
    return validate_value(parse_contract(spec), value)[0]


def test_record_required_present_extra_ok():
    assert _ok({"kind": "record", "required_keys": ["a", "b"]}, {"a": 1, "b": 2, "c": 3})


def test_record_missing_key():
    assert not _ok({"kind": "record", "required_keys": ["a", "b"]}, {"a": 1})


def test_record_not_a_dict():
    assert not _ok({"kind": "record"}, [1, 2])


def test_record_nested_value_kind():
    spec = {"kind": "record", "required": {"x": {"kind": "text"}}}
    assert _ok(spec, {"x": "hi"})
    assert not _ok(spec, {"x": 123})  # x must be text


def test_collection_list_ok_str_and_dict_rejected():
    assert _ok({"kind": "collection"}, [1, 2])
    assert _ok({"kind": "collection"}, (1, 2))
    assert not _ok({"kind": "collection"}, "abc")  # str is not a collection
    assert not _ok({"kind": "collection"}, {"a": 1})


def test_collection_element_recurse():
    spec = {"kind": "collection", "element": {"kind": "text"}}
    assert _ok(spec, ["a", "b"])
    assert not _ok(spec, ["a", 1])  # element 1 is not text


def test_file_extensions():
    assert _ok({"kind": "file", "extensions": ["fasta"]}, "x.fasta")
    assert not _ok({"kind": "file", "extensions": ["fasta"]}, "x.pdb")
    assert _ok({"kind": "file"}, "anything.txt")  # no extension constraint
    assert not _ok({"kind": "file"}, 123)  # not a path/str


def test_text():
    assert _ok({"kind": "text"}, "hi")
    assert not _ok({"kind": "text"}, 5)


def test_handle_presence():
    assert _ok({"kind": "handle"}, object())
    assert not _ok({"kind": "handle"}, None)
