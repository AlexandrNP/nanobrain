"""ComponentIndex.is_stale — the silent-staleness guard.

A built FAISS index is a snapshot of a corpus. If the corpus changes (a
component added / description edited) but the index is not rebuilt, retrieval
silently runs over the OLD corpus and misses the new components. ``is_stale``
detects that cheaply (parse + hash, no embeddings) so a caller can rebuild or
degrade rather than serve stale hits.

The index is built ONCE (real embeddings, module-scoped) for a real index_hash;
each staleness assertion points is_stale at a different manifest — is_stale never
re-embeds, so this is fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nanobrain.lightweight.component_index import ComponentIndex


def _write_manifest(path: Path, components: list[dict]) -> Path:
    path.write_text(yaml.safe_dump({"components": components}), encoding="utf-8")
    return path


_BASE_COMPONENTS = [
    {"step_id": "alpha", "step_name": "alpha", "rag_description": "the alpha component"},
    {"step_id": "beta", "step_name": "beta", "rag_description": "the beta component"},
]


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[ComponentIndex, Path]:
    d = tmp_path_factory.mktemp("ci_stale")
    m = _write_manifest(d / "manifest.yml", _BASE_COMPONENTS)
    idx = ComponentIndex()
    idx.rebuild(manifest_paths=[m], library_version="0.1.0")
    return idx, d


def test_not_stale_when_corpus_unchanged(built):
    idx, d = built
    same = _write_manifest(d / "same.yml", _BASE_COMPONENTS)
    assert idx.is_stale([same], library_version="0.1.0") is False


def test_stale_when_component_added(built):
    idx, d = built
    more = _write_manifest(
        d / "more.yml",
        _BASE_COMPONENTS + [{"step_id": "gamma", "rag_description": "a new component"}],
    )
    assert idx.is_stale([more], library_version="0.1.0") is True


def test_stale_when_description_changed(built):
    idx, d = built
    changed = _write_manifest(
        d / "changed.yml",
        [
            {"step_id": "alpha", "step_name": "alpha", "rag_description": "EDITED alpha"},
            {"step_id": "beta", "step_name": "beta", "rag_description": "the beta component"},
        ],
    )
    assert idx.is_stale([changed], library_version="0.1.0") is True


def test_stale_on_library_version_bump(built):
    idx, d = built
    same = _write_manifest(d / "v.yml", _BASE_COMPONENTS)
    # Same corpus, bumped library_version => the embedding/index is logically
    # stale (the version is part of the hash).
    assert idx.is_stale([same], library_version="0.2.0") is True
