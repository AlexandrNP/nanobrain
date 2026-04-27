# ruff: noqa: I001, E402
# Import order in this file is load-bearing — see the module docstring
# and the pre-import comment below ``from __future__``. Do not let an
# import-sort fix run over it; the segfault it prevents is silent.
"""Component RAG index for the LLM composer.

Embeds each library component's ``rag_description`` + ``rag_examples``
via ``sentence-transformers/all-mpnet-base-v2``, stores a FAISS
``IndexFlatIP`` (inner product on L2-normalized vectors = cosine),
and exposes ``search(query, k)`` for the composer's retrieval tier.

Role in the pipeline
--------------------
T-COMP Phase 2 shipped with a linear-scan ``ComponentCatalog`` that
substring-matches queries against descriptions. Phase 4 replaces
that with ``ComponentIndex.search``. Until Phase 4 wires it, the
linear-scan path remains the production default; this module is
the drop-in replacement.

Design deviation from the original T03 spec
--------------------------------------------
The spec (implementation_plan.md §T03 step 3) says to walk the
``library/domain/`` tree. That tree does not exist in the current
repo — the only curated corpus with ``rag_description`` /
``rag_examples`` fields lives in T02 manifests at
``apecx_integration/composition/workflows/*/manifest.yml``. Walking
``nanobrain/library/*`` directly would produce garbage embeddings
because most classes have no curated description.

This implementation walks the T02 manifest set instead — the same
set ``ComponentCatalog.from_manifests`` consumes, so the retrieval
corpora stay in parity with the composer.

Determinism
-----------
``rebuild()`` produces a stable ``index_hash`` given the same
``(model_name, library_version, sorted component ids + descriptions)``
input. The FAISS binary itself may differ byte-for-byte across
runs (float32 ordering under BLAS), which is why the hash is
computed over the SOURCE text, not the index bytes. That satisfies
AC1's "deterministic index hash given the same library version"
without making us fight BLAS nondeterminism.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# IMPORTANT: ``sentence_transformers`` must be imported before
# ``faiss``. On macOS ARM (and some Linux configurations), importing
# faiss-cpu first links in an OpenMP runtime that conflicts with the
# one ``torch`` / ``sentence-transformers`` brings in, and the
# process silently segfaults during ``SentenceTransformer.encode``.
# Diagnosed 2026-04-23 while bootstrapping T03. Do not "tidy" this
# by alphabetizing the import block — the order is load-bearing.
from sentence_transformers import SentenceTransformer  # noqa: I001

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402


_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
_EMBEDDING_DIM: int = 768
# Default to CPU: on Apple Silicon machines without a fresh PyTorch
# build, the "mps" backend can silently crash during model load
# (observed 2026-04-23 while bootstrapping T03). CPU is ~10x slower
# but survives the one-shot embedding of 10 components in <2s.
# Operators with working CUDA / MPS override via ``device=...`` in
# the ``ComponentIndex`` constructor.
_DEFAULT_DEVICE: str = "cpu"


@dataclass(frozen=True, kw_only=True)
class ComponentMatch:
    """One hit from ``ComponentIndex.search``.

    ``similarity`` is cosine, clamped to [0, 1] (inner-product on
    L2-normalized vectors can occasionally drift slightly above 1
    from float32 rounding; clamping keeps the range honest for
    downstream consumers that assume a probability-shaped score).
    """

    id: str
    name: str
    class_path: str
    description: str
    similarity: float
    manifest_path: str
    yaml_path: str | None = None
    examples: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ComponentRecord:
    """One catalog row, pre-embedding."""

    id: str
    name: str
    class_path: str
    yaml_path: str | None
    description: str
    examples: tuple[str, ...]
    manifest_path: str

    @property
    def text_for_embedding(self) -> str:
        body = self.description.strip()
        if self.examples:
            body = body + "\n\n" + "\n".join(self.examples)
        return body


class ComponentIndex:
    """FAISS-backed RAG index over T02 manifest components.

    Lifecycle:
        idx = ComponentIndex()
        idx.rebuild(manifest_paths=[...], library_version="0.1.0")
        idx.save(Path("./rag_index"))
        # later...
        idx2 = ComponentIndex.load(Path("./rag_index"))
        hits = idx2.search("find genes related to EEEV", k=5)
    """

    def __init__(
        self,
        *,
        model_name: str = _MODEL_NAME,
        device: str = _DEFAULT_DEVICE,
    ) -> None:
        self._model_name: str = model_name
        self._device: str = device
        self._model: SentenceTransformer | None = None
        self._faiss_index: faiss.Index | None = None
        self._records: tuple[ComponentRecord, ...] = ()
        self._library_version: str = ""
        self._index_hash: str = ""

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def rebuild(
        self,
        *,
        manifest_paths: list[Path],
        library_version: str,
    ) -> None:
        self._library_version = library_version
        records = self._collect_records(manifest_paths)
        if not records:
            raise ValueError(
                "no components with rag_description found in the "
                "supplied manifests — ComponentIndex cannot be built "
                "from an empty corpus"
            )
        self._records = tuple(records)
        embeddings = self._encode([r.text_for_embedding for r in records])
        index = faiss.IndexFlatIP(_EMBEDDING_DIM)
        index.add(embeddings)
        self._faiss_index = index
        self._index_hash = self._compute_hash(
            records=self._records,
            library_version=self._library_version,
            model_name=self._model_name,
        )

    @property
    def index_hash(self) -> str:
        return self._index_hash

    @property
    def library_version(self) -> str:
        return self._library_version

    @property
    def model_name(self) -> str:
        return self._model_name

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_dir: Path) -> None:
        if self._faiss_index is None:
            raise RuntimeError("call rebuild() before save()")
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(
            self._faiss_index, str(index_dir / "faiss.bin")
        )
        payload: dict[str, Any] = {
            "model": self._model_name,
            "library_version": self._library_version,
            "index_hash": self._index_hash,
            "records": [
                {
                    "id": r.id,
                    "name": r.name,
                    "class_path": r.class_path,
                    "yaml_path": r.yaml_path,
                    "description": r.description,
                    "examples": list(r.examples),
                    "manifest_path": r.manifest_path,
                }
                for r in self._records
            ],
        }
        (index_dir / "metadata.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path) -> ComponentIndex:
        meta = json.loads(
            (index_dir / "metadata.json").read_text(encoding="utf-8")
        )
        obj = cls(model_name=meta["model"])
        obj._records = tuple(
            ComponentRecord(
                id=r["id"],
                name=r["name"],
                class_path=r["class_path"],
                yaml_path=r["yaml_path"],
                description=r["description"],
                examples=tuple(r["examples"]),
                manifest_path=r["manifest_path"],
            )
            for r in meta["records"]
        )
        obj._library_version = meta["library_version"]
        obj._index_hash = meta["index_hash"]
        obj._faiss_index = faiss.read_index(
            str(index_dir / "faiss.bin")
        )
        return obj

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[ComponentMatch]:
        if self._faiss_index is None:
            raise RuntimeError(
                "call rebuild() or load() before search()"
            )
        if not query or not query.strip():
            return []
        k_effective = min(k, len(self._records))
        q = self._encode([query])
        sims, idxs = self._faiss_index.search(q, k_effective)
        out: list[ComponentMatch] = []
        for sim, idx in zip(sims[0], idxs[0], strict=True):
            if idx < 0:
                continue
            r = self._records[idx]
            clamped = float(max(0.0, min(1.0, sim)))
            out.append(
                ComponentMatch(
                    id=r.id,
                    name=r.name,
                    class_path=r.class_path,
                    description=r.description,
                    similarity=clamped,
                    manifest_path=r.manifest_path,
                    yaml_path=r.yaml_path,
                    examples=r.examples,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self._model_name, device=self._device
            )
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        arr = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return arr.astype("float32")

    @staticmethod
    def _collect_records(
        manifest_paths: list[Path],
    ) -> list[ComponentRecord]:
        records: list[ComponentRecord] = []
        for manifest_path in manifest_paths:
            doc = yaml.safe_load(
                manifest_path.read_text(encoding="utf-8")
            )
            workflow_slug = manifest_path.parent.name
            for c in doc.get("components") or []:
                rag_desc = c.get("rag_description") or c.get(
                    "description", ""
                )
                if not rag_desc:
                    continue
                examples = tuple(c.get("rag_examples") or ())
                step_id = str(c.get("step_id", c.get("name", "")))
                name = (
                    c.get("step_name")
                    or c.get("name")
                    or step_id
                )
                records.append(
                    ComponentRecord(
                        id=f"{workflow_slug}/{name}:{step_id}",
                        name=name,
                        class_path=c.get("class", ""),
                        yaml_path=c.get("yaml"),
                        description=rag_desc.strip(),
                        examples=tuple(e.strip() for e in examples),
                        manifest_path=str(manifest_path),
                    )
                )
        return records

    @staticmethod
    def _compute_hash(
        *,
        records: tuple[ComponentRecord, ...],
        library_version: str,
        model_name: str,
    ) -> str:
        h = hashlib.sha256()
        h.update(model_name.encode("utf-8"))
        h.update(b"\n")
        h.update(library_version.encode("utf-8"))
        h.update(b"\n")
        for r in sorted(records, key=lambda x: x.id):
            h.update(r.id.encode("utf-8"))
            h.update(b"|")
            h.update(r.description.encode("utf-8"))
            h.update(b"|")
            for ex in r.examples:
                h.update(ex.encode("utf-8"))
                h.update(b"~")
            h.update(b"\n")
        return h.hexdigest()
