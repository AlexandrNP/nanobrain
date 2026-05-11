"""DataSourceRegistry — G24 versioned-data-source manifest primitive.

eval_03 Round 3 G24: pre-G24 every consumer of an external data source
(VIOLIN snapshots, BV-BRC pulls, FAISS indices, taxdumps, RAG corpora)
rolled their own version-pin policy. ``CONTRACTS.md#data-version-pin``
describes 14 such sources for the integration; without a primitive,
each consumer's pin policy drifted independently and reproducibility
was per-consumer rather than per-workflow. R2/R3 reproducibility is
non-negotiable for an audit-bound deployment.

P6+c decision (open question CONTRACTS.md#decision-p6+a (decisions section)):
  * **YAML format** — matches every other nanobrain config; one file
    per registry; entries are dict-shaped.

DataSourceRegistry is the *primitive*. APECx contributes the *catalog
content* (the per-source manifest entries with real versions, real
refresh cadences, real content hashes). The primitive does NOT know
about VIOLIN, FAISS, etc. — it knows about manifest entries and how
to verify them.

## Manifest schema

```yaml
data_sources:
  viper_v3:
    version: "3.0.1"
    description: "VIPER alphavirus catalog v3"
    location: "/path/to/viper_v3.parquet"   # or s3://, https://, etc
    content_hash: "sha256:abc123..."        # Optional; verified at load if present
    refresh_cadence_seconds: 86400          # 24 hours
    refresh_cadence_policy: "manual"        # manual | scheduled | event
    metadata:
      curator: "EEEV-team"
      last_refreshed: "2026-04-15T00:00:00Z"
```

## Verification model

A consumer that needs to load ``viper_v3`` calls:

  reg = DataSourceRegistry.from_yaml('configs/data_sources.yml')
  entry = reg.get('viper_v3')
  entry.verify_content_hash()   # raises ContentHashMismatch on drift

Without G24, the consumer either skipped verification (silent staleness
on disk) OR rolled their own — both common failure modes.

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G24;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.8 (P6+c).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml

logger = logging.getLogger(__name__)


RefreshCadencePolicy = Literal["manual", "scheduled", "event"]


@dataclass(frozen=True)
class DataSourceEntry:
    """One entry in the data-source manifest.

    Frozen + hashable so two entries hashing to the same value are
    interchangeable for caching / equality purposes.
    """

    name: str
    version: str
    location: str
    description: str = ""
    content_hash: Optional[str] = None  # "sha256:<hex>" form
    refresh_cadence_seconds: Optional[float] = None
    refresh_cadence_policy: RefreshCadencePolicy = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- Hashing helpers -------------------------------------------------

    def verify_content_hash(self) -> bool:
        """Verify the on-disk content hash matches the declared hash.

        Returns:
            True if the entry has no declared hash (nothing to verify
            against — a doc-only entry).
            True if hash matches.

        Raises:
            ContentHashMismatch: when the declared hash and the
                computed hash disagree.
            FileNotFoundError: when ``location`` is a local path that
                does not exist.
            ValueError: when ``location`` is a non-local URL (s3://,
                https://) and verification cannot be performed without
                downloading; callers must download first.
        """
        if self.content_hash is None:
            return True  # nothing to verify — opt-out by omission
        path = Path(self.location)
        if not path.is_absolute() and not path.exists():
            raise ValueError(
                f"FAIL-FAST: DataSourceEntry {self.name!r} location "
                f"{self.location!r} is not a local path; cannot verify "
                f"content_hash without downloading first. Download to "
                f"a local path then re-verify."
            )
        if not path.is_file() and not path.is_dir():
            raise FileNotFoundError(
                f"FAIL-FAST: DataSourceEntry {self.name!r} location "
                f"{self.location!r} does not exist on disk."
            )

        actual = compute_content_hash(path)
        declared = self.content_hash.lower().strip()
        if not declared.startswith("sha256:"):
            raise ValueError(
                f"FAIL-FAST: DataSourceEntry {self.name!r} content_hash "
                f"{declared!r} must start with 'sha256:' prefix."
            )
        declared_hex = declared.split(":", 1)[1]
        if actual != declared_hex:
            raise ContentHashMismatch(
                name=self.name,
                location=self.location,
                declared=declared_hex,
                actual=actual,
            )
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "location": self.location,
            "description": self.description,
            "content_hash": self.content_hash,
            "refresh_cadence_seconds": self.refresh_cadence_seconds,
            "refresh_cadence_policy": self.refresh_cadence_policy,
            "metadata": dict(self.metadata),
        }


class ContentHashMismatch(Exception):
    """Raised when a verified data source's on-disk content does not
    match the manifest's declared hash. Workflow-terminal: continuing
    risks polluting downstream artifacts with stale data."""

    def __init__(
        self,
        *,
        name: str,
        location: str,
        declared: str,
        actual: str,
    ) -> None:
        super().__init__(
            f"ContentHashMismatch: data source {name!r} at "
            f"{location!r} content_hash drift; "
            f"declared=sha256:{declared}, actual=sha256:{actual}. "
            f"Refusing to proceed — re-fetch the source OR update "
            f"the manifest with the new hash + bumped version."
        )
        self.name = name
        self.location = location
        self.declared = declared
        self.actual = actual


def compute_content_hash(path: Path) -> str:
    """Compute SHA-256 over a file's bytes, OR over the concatenated
    bytes of all files in a directory (sorted by relative path).
    Returns the hex digest (no prefix).

    For directories, the hash is order-stable: sort the relative
    paths, concat ``relpath || NUL || file_bytes || NUL`` for each.
    Empty directories hash to a stable digest (sha256 of "").
    """
    h = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    if path.is_dir():
        for entry in sorted(path.rglob("*")):
            if not entry.is_file():
                continue
            rel = entry.relative_to(path).as_posix().encode("utf-8")
            h.update(rel)
            h.update(b"\x00")
            with entry.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            h.update(b"\x00")
        return h.hexdigest()
    raise FileNotFoundError(
        f"FAIL-FAST: compute_content_hash: {path} is neither file "
        f"nor directory."
    )


class DataSourceRegistry:
    """Versioned data-source manifest. Load from YAML; lookup by name.

    Use::

        reg = DataSourceRegistry.from_yaml('configs/data_sources.yml')
        entry = reg.get('viper_v3')
        entry.verify_content_hash()  # raises on drift

        for entry in reg.list_entries():
            ...  # iterate
    """

    def __init__(self, entries: Dict[str, DataSourceEntry]) -> None:
        self._entries: Dict[str, DataSourceEntry] = dict(entries)

    # ---- Loaders --------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataSourceRegistry":
        """Load a registry from a YAML manifest file.

        The YAML structure:

            data_sources:
              <name>:
                version: ...
                location: ...
                ...

        Bare-dict input is also accepted (already-parsed YAML) — see
        ``from_dict``.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"FAIL-FAST: DataSourceRegistry.from_yaml: {path} "
                f"does not exist."
            )
        return cls.from_dict(yaml.safe_load(path.read_text()) or {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSourceRegistry":
        """Build a registry from a parsed dict. The dict may either
        be the top-level manifest (with ``data_sources`` key) or just
        the ``data_sources`` map directly."""
        if "data_sources" in data:
            sources = data["data_sources"]
        else:
            sources = data
        if not isinstance(sources, dict):
            raise ValueError(
                f"FAIL-FAST: DataSourceRegistry expected a dict of "
                f"name -> entry; got {type(sources).__name__}"
            )

        entries: Dict[str, DataSourceEntry] = {}
        for name, entry_dict in sources.items():
            if not isinstance(entry_dict, dict):
                raise ValueError(
                    f"FAIL-FAST: DataSourceRegistry entry {name!r} "
                    f"must be a dict; got {type(entry_dict).__name__}"
                )
            # Reject unknown top-level keys to catch typos early.
            known = {
                "version", "location", "description", "content_hash",
                "refresh_cadence_seconds", "refresh_cadence_policy",
                "metadata",
            }
            extras = set(entry_dict.keys()) - known
            if extras:
                raise ValueError(
                    f"FAIL-FAST: DataSourceRegistry entry {name!r} "
                    f"has unknown keys {sorted(extras)}; known: "
                    f"{sorted(known)}"
                )
            for required in ("version", "location"):
                if required not in entry_dict:
                    raise ValueError(
                        f"FAIL-FAST: DataSourceRegistry entry {name!r} "
                        f"missing required field {required!r}"
                    )
            entries[name] = DataSourceEntry(
                name=name,
                version=str(entry_dict["version"]),
                location=str(entry_dict["location"]),
                description=str(entry_dict.get("description", "")),
                content_hash=entry_dict.get("content_hash"),
                refresh_cadence_seconds=(
                    float(entry_dict["refresh_cadence_seconds"])
                    if entry_dict.get("refresh_cadence_seconds") is not None
                    else None
                ),
                refresh_cadence_policy=entry_dict.get(
                    "refresh_cadence_policy", "manual"
                ),
                metadata=dict(entry_dict.get("metadata") or {}),
            )
        return cls(entries)

    # ---- Lookup --------------------------------------------------------

    def get(self, name: str) -> DataSourceEntry:
        """Look up an entry by name. FAIL-FAST when missing."""
        if name not in self._entries:
            available = sorted(self._entries.keys())
            raise KeyError(
                f"FAIL-FAST: DataSourceRegistry has no entry "
                f"{name!r}; available: {available}"
            )
        return self._entries[name]

    def has(self, name: str) -> bool:
        return name in self._entries

    def list_entries(self) -> List[DataSourceEntry]:
        return list(self._entries.values())

    def list_names(self) -> List[str]:
        return sorted(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries


__all__ = [
    "ContentHashMismatch",
    "DataSourceEntry",
    "DataSourceRegistry",
    "RefreshCadencePolicy",
    "compute_content_hash",
]
