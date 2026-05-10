"""G24 — pin the DataSourceRegistry contract.

eval_03 Round 3 G24: pre-G24 every consumer of an external data source
rolled their own version-pin policy; reproducibility was per-consumer
rather than per-workflow. R2/R3 reproducibility is non-negotiable for
audit-bound deployments.

This test pins:
  1. from_yaml() loads a well-formed manifest
  2. from_yaml() FAIL-FASTs on missing file
  3. from_dict() accepts both top-level and bare-map shapes
  4. unknown entry keys FAIL-FAST (typo protection)
  5. missing required fields (version, location) FAIL-FAST
  6. get(missing) FAIL-FASTs with available-name list
  7. content_hash verification: matching hash returns True
  8. content_hash verification: mismatched hash raises ContentHashMismatch
  9. content_hash absent -> verify_content_hash returns True (opt-out)
 10. content_hash with non-sha256 prefix FAIL-FASTs
 11. compute_content_hash on a directory is order-stable
 12. compute_content_hash on missing path raises FileNotFoundError
 13. list_entries / list_names / __contains__ / __len__ work
 14. refresh_cadence_seconds is parsed as float when given as int

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G24;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.8 (P6+c).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nanobrain.library.runtime.data_source_registry import (
    ContentHashMismatch,
    DataSourceRegistry,
    compute_content_hash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_data_file(tmp_path: Path, content: bytes = b"hello world") -> Path:
    """Write a small data file and return its path."""
    p = tmp_path / "data.bin"
    p.write_bytes(content)
    return p


def _manifest_for(tmp_path: Path, **overrides):
    """Build a one-entry manifest dict, with overrideable fields."""
    p = _write_data_file(tmp_path)
    base = {
        "viper_v3": {
            "version": "3.0.1",
            "location": str(p),
            "description": "VIPER alphavirus catalog",
            "refresh_cadence_seconds": 86400,
            "refresh_cadence_policy": "manual",
            "metadata": {"curator": "EEEV-team"},
        }
    }
    base["viper_v3"].update(overrides)
    return {"data_sources": base}


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


def test_from_yaml_loads_well_formed_manifest(tmp_path):
    manifest = _manifest_for(tmp_path)
    yaml_path = tmp_path / "data_sources.yml"
    yaml_path.write_text(yaml.safe_dump(manifest))
    reg = DataSourceRegistry.from_yaml(yaml_path)
    assert "viper_v3" in reg
    entry = reg.get("viper_v3")
    assert entry.version == "3.0.1"
    assert entry.refresh_cadence_seconds == 86400.0
    assert entry.metadata == {"curator": "EEEV-team"}


def test_from_yaml_missing_file_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="FAIL-FAST"):
        DataSourceRegistry.from_yaml(tmp_path / "nonexistent.yml")


def test_from_dict_accepts_bare_map(tmp_path):
    """The from_dict path tolerates both shapes:
    - {"data_sources": {name: entry}}
    - {name: entry}     (bare map, no top-level wrapper)
    """
    manifest = _manifest_for(tmp_path)
    bare = manifest["data_sources"]  # strip the data_sources wrapper
    reg = DataSourceRegistry.from_dict(bare)
    assert "viper_v3" in reg


def test_unknown_entry_key_fails_fast(tmp_path):
    """Catches YAML typos at load — without G24 a misnamed field
    silently lands as ignored metadata."""
    manifest = _manifest_for(tmp_path)
    manifest["data_sources"]["viper_v3"]["versionn"] = "typo"  # extra
    with pytest.raises(ValueError, match="unknown keys"):
        DataSourceRegistry.from_dict(manifest)


def test_missing_required_fields_fail_fast(tmp_path):
    """Each entry MUST declare version + location. Anything else is
    optional — those two are the identity-bearing fields."""
    p = _write_data_file(tmp_path)
    bad_manifest = {
        "data_sources": {
            "viper_v3": {"location": str(p)},  # missing 'version'
        }
    }
    with pytest.raises(ValueError, match="missing required field 'version'"):
        DataSourceRegistry.from_dict(bad_manifest)


# ---------------------------------------------------------------------------
# Lookup tests
# ---------------------------------------------------------------------------


def test_get_missing_fails_fast_with_available_names(tmp_path):
    reg = DataSourceRegistry.from_dict(_manifest_for(tmp_path))
    with pytest.raises(KeyError) as excinfo:
        reg.get("viper_v9999")
    msg = str(excinfo.value)
    assert "viper_v9999" in msg
    assert "viper_v3" in msg, (
        f"missing-name error must list available names; got: {msg}"
    )


def test_list_entries_and_names(tmp_path):
    reg = DataSourceRegistry.from_dict(_manifest_for(tmp_path))
    entries = reg.list_entries()
    assert len(entries) == 1
    assert reg.list_names() == ["viper_v3"]
    assert len(reg) == 1
    assert "viper_v3" in reg


def test_refresh_cadence_int_parsed_as_float(tmp_path):
    """YAML int -> Python int normally; the entry stores float so
    arithmetic with it is uniform regardless of YAML quoting."""
    reg = DataSourceRegistry.from_dict(_manifest_for(tmp_path))
    entry = reg.get("viper_v3")
    assert isinstance(entry.refresh_cadence_seconds, float)


# ---------------------------------------------------------------------------
# Content hash verification
# ---------------------------------------------------------------------------


def test_verify_content_hash_match_returns_true(tmp_path):
    # Use a distinct filename so _manifest_for's default file write
    # does not overwrite ours.
    p = tmp_path / "verifiable.bin"
    p.write_bytes(b"reproducible-bytes")
    actual_hex = compute_content_hash(p)
    manifest = _manifest_for(
        tmp_path,
        content_hash=f"sha256:{actual_hex}",
    )
    # Point the entry at our purpose-written file.
    manifest["data_sources"]["viper_v3"]["location"] = str(p)
    reg = DataSourceRegistry.from_dict(manifest)
    entry = reg.get("viper_v3")
    assert entry.verify_content_hash() is True


def test_verify_content_hash_mismatch_raises(tmp_path):
    p = _write_data_file(tmp_path, b"on-disk-content")
    manifest = _manifest_for(
        tmp_path,
        content_hash="sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    )
    manifest["data_sources"]["viper_v3"]["location"] = str(p)
    reg = DataSourceRegistry.from_dict(manifest)
    entry = reg.get("viper_v3")
    with pytest.raises(ContentHashMismatch) as excinfo:
        entry.verify_content_hash()
    msg = str(excinfo.value)
    assert "viper_v3" in msg
    assert "drift" in msg
    assert "Refusing to proceed" in msg


def test_verify_content_hash_absent_returns_true_opt_out(tmp_path):
    """An entry without content_hash is doc-only — verify is a
    no-op (returns True). This is the documented opt-out."""
    p = _write_data_file(tmp_path)
    manifest = _manifest_for(tmp_path)
    manifest["data_sources"]["viper_v3"]["location"] = str(p)
    # Strip content_hash to exercise opt-out.
    manifest["data_sources"]["viper_v3"].pop("content_hash", None)
    reg = DataSourceRegistry.from_dict(manifest)
    entry = reg.get("viper_v3")
    assert entry.verify_content_hash() is True


def test_verify_content_hash_non_sha256_prefix_fails_fast(tmp_path):
    p = _write_data_file(tmp_path)
    manifest = _manifest_for(
        tmp_path,
        content_hash="md5:abcdef",  # non-sha256 declared
    )
    manifest["data_sources"]["viper_v3"]["location"] = str(p)
    reg = DataSourceRegistry.from_dict(manifest)
    entry = reg.get("viper_v3")
    with pytest.raises(ValueError, match="must start with 'sha256:'"):
        entry.verify_content_hash()


# ---------------------------------------------------------------------------
# compute_content_hash tests
# ---------------------------------------------------------------------------


def test_compute_content_hash_directory_is_order_stable(tmp_path):
    """Hashing the same directory twice produces the same digest."""
    d = tmp_path / "src"
    d.mkdir()
    (d / "a.txt").write_bytes(b"alpha")
    (d / "b.txt").write_bytes(b"bravo")
    (d / "sub").mkdir()
    (d / "sub" / "c.txt").write_bytes(b"charlie")
    hash1 = compute_content_hash(d)
    hash2 = compute_content_hash(d)
    assert hash1 == hash2
    # File contents matter — modifying any file changes the hash.
    (d / "a.txt").write_bytes(b"alpha-modified")
    hash3 = compute_content_hash(d)
    assert hash1 != hash3


def test_compute_content_hash_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="neither file nor directory"):
        compute_content_hash(tmp_path / "does-not-exist")
