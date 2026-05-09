"""SignedConfig loader (G19).

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G19`` and
``apecx-mcp-integration/docs/security_threat_model.md §6.1``: detached
ed25519 signature verification for YAML configs. HPC bundles ship signed
``.yml.sig`` files so a replay can verify the YAML hasn't been tampered
with between export and replay.

Threat model:
- HPC bundle exporter signs ``workflow.yml`` with the operator's
  private key, producing ``workflow.yml.sig``.
- Replay-side loader (this module) verifies the signature against the
  operator's public key BEFORE loading the YAML.
- Tampered YAML (signature won't verify) FAIL-FASTs.
- Missing signature when ``--require-signed`` is set FAIL-FASTs.
- Missing signature when ``--require-signed`` is NOT set is allowed
  (the legacy un-signed path).

Implementation notes:

- Uses ``cryptography`` library (already a transitive dep via proxystore).
- ``ed25519`` per the ``apecx-mcp-integration/docs/tool_descriptor_contract.md §10.2``
  signing protocol — small keys, fast, post-quantum-friendly.
- Signature is over the canonical bytes of the file (the EXACT bytes
  on disk, not a re-serialization). This means trailing newlines and
  whitespace matter — the bundle exporter writes the file once and
  signs the bytes it just wrote.
- Public key is loaded from a deployment-configured trust root path.
  G20 will tighten this with explicit class-path whitelisting; G19
  here just validates signatures against a single operator key.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from .component_base import ComponentConfigurationError

logger = logging.getLogger(__name__)


# Lazy import marker — populated on first use of cryptography.
_CRYPTO_IMPORT_ERROR: Optional[ImportError] = None


def _import_cryptography():
    """Lazy import of ed25519 primitives. Returns the (Ed25519PublicKey,
    InvalidSignature) tuple. Raises ComponentConfigurationError with
    install hint if cryptography is unavailable."""
    global _CRYPTO_IMPORT_ERROR
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.serialization import (
            load_pem_public_key,
        )
    except ImportError as e:
        _CRYPTO_IMPORT_ERROR = e
        raise ComponentConfigurationError(
            f"FAIL-FAST: SignedConfig requires the cryptography package. "
            f"Install with: pip install cryptography. "
            f"Original ImportError: {e}"
        ) from e
    return Ed25519PublicKey, InvalidSignature, load_pem_public_key


def load_public_key_from_pem(public_key_path: Union[str, Path]):
    """Load an ed25519 public key from a PEM file.

    Returns an ``Ed25519PublicKey`` instance suitable for passing to
    :func:`verify_detached_signature`.

    FAIL-FASTs on missing file, malformed PEM, or non-ed25519 key.
    """
    path = Path(public_key_path)
    if not path.is_file():
        raise ComponentConfigurationError(
            f"FAIL-FAST: SignedConfig public key not found at {path}"
        )

    Ed25519PublicKey, _, load_pem_public_key = _import_cryptography()

    try:
        pub_key = load_pem_public_key(path.read_bytes())
    except Exception as e:
        raise ComponentConfigurationError(
            f"FAIL-FAST: SignedConfig public key at {path} failed to "
            f"parse as PEM: {e}"
        ) from e

    if not isinstance(pub_key, Ed25519PublicKey):
        raise ComponentConfigurationError(
            f"FAIL-FAST: SignedConfig public key at {path} is "
            f"{type(pub_key).__name__}, expected Ed25519PublicKey"
        )

    return pub_key


def verify_detached_signature(
    config_path: Union[str, Path],
    signature_path: Union[str, Path],
    public_key,
) -> None:
    """Verify a detached ed25519 signature.

    Args:
        config_path: Path to the config YAML file (the bytes that were
            signed).
        signature_path: Path to the detached signature file (raw bytes
            of the signature, NOT base64).
        public_key: An Ed25519PublicKey instance from
            :func:`load_public_key_from_pem`.

    Raises:
        ComponentConfigurationError on missing file, missing signature,
            wrong-shape signature, or signature mismatch.
    """
    _, InvalidSignature, _ = _import_cryptography()

    cfg_path = Path(config_path)
    sig_path = Path(signature_path)

    if not cfg_path.is_file():
        raise ComponentConfigurationError(
            f"FAIL-FAST: SignedConfig config file not found at {cfg_path}"
        )
    if not sig_path.is_file():
        raise ComponentConfigurationError(
            f"FAIL-FAST: SignedConfig signature file not found at {sig_path}"
        )

    config_bytes = cfg_path.read_bytes()
    sig_bytes = sig_path.read_bytes()

    try:
        public_key.verify(sig_bytes, config_bytes)
    except InvalidSignature as e:
        raise ComponentConfigurationError(
            f"FAIL-FAST: SignedConfig signature verification FAILED for "
            f"{cfg_path}. The config bytes do not match the signature; "
            f"the file may have been tampered with, OR the signature "
            f"was produced by a different key than {public_key!r}."
        ) from e
    except Exception as e:
        raise ComponentConfigurationError(
            f"FAIL-FAST: SignedConfig signature verification raised "
            f"unexpected error for {cfg_path}: {e}"
        ) from e


def load_signed_config(
    config_path: Union[str, Path],
    public_key_path: Union[str, Path],
    *,
    signature_suffix: str = ".sig",
    require_signed: bool = True,
) -> bytes:
    """One-shot: verify a config's detached signature and return the
    config bytes.

    Args:
        config_path: Path to the config file to load.
        public_key_path: Path to the operator's PEM-encoded ed25519
            public key.
        signature_suffix: Suffix to append to ``config_path`` to find
            the detached signature. Default ``.sig`` — so for
            ``workflow.yml`` we look for ``workflow.yml.sig``.
        require_signed: When True (default), missing signature FAIL-FASTs.
            When False, missing signature is allowed (legacy un-signed
            path) — emits a WARNING log entry instead.

    Returns:
        The raw bytes of the config file.

    Use the returned bytes with your config-loader of choice (yaml.safe_load,
    Pydantic .model_validate_json, etc).
    """
    cfg_path = Path(config_path)
    sig_path = cfg_path.with_suffix(cfg_path.suffix + signature_suffix)

    if not sig_path.is_file():
        if require_signed:
            raise ComponentConfigurationError(
                f"FAIL-FAST: SignedConfig require_signed=True but no "
                f"signature found at {sig_path} for {cfg_path}"
            )
        else:
            logger.warning(
                "SignedConfig: no signature at %s; loading %s unsigned "
                "(require_signed=False — legacy path)",
                sig_path, cfg_path,
            )
            return cfg_path.read_bytes()

    pub_key = load_public_key_from_pem(public_key_path)
    verify_detached_signature(cfg_path, sig_path, pub_key)
    logger.debug(
        "SignedConfig: signature verified for %s (sig at %s)",
        cfg_path, sig_path,
    )
    return cfg_path.read_bytes()
