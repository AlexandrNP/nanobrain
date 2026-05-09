"""Tests for G19 — SignedConfig loader.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G19``:
detached ed25519 signature verification for YAML configs.

Tests cover:
1. load_public_key_from_pem — happy path + error paths
2. verify_detached_signature — happy path + tamper detection
3. load_signed_config — full sign+verify+load pipeline
4. require_signed=True/False semantics
5. Tampered config FAIL-FASTs
6. Wrong key FAIL-FASTs
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.signed_config import (
    load_public_key_from_pem,
    load_signed_config,
    verify_detached_signature,
)


# Lazy import — these tests REQUIRE cryptography (which is installed
# transitively via proxystore). Tests fail loud, not silently skip.
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption,
)


# ---------------------------------------------------------------------------
# Helpers — make a fresh keypair + sign a config file in a tmpdir
# ---------------------------------------------------------------------------

def _generate_keypair():
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    pub_pem = pub.public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    return priv, pub_pem


def _setup_signed(tmp: Path, content: bytes = b"name: test\n"):
    """Write a config + matching sig + pub key to tmp; return paths."""
    priv, pub_pem = _generate_keypair()

    cfg_path = tmp / "workflow.yml"
    cfg_path.write_bytes(content)

    sig_path = tmp / "workflow.yml.sig"
    sig_path.write_bytes(priv.sign(content))

    pub_path = tmp / "operator.pub"
    pub_path.write_bytes(pub_pem)

    return cfg_path, sig_path, pub_path, priv


# ---------------------------------------------------------------------------
# 1. load_public_key_from_pem
# ---------------------------------------------------------------------------

class TestLoadPublicKey:

    def test_valid_ed25519_pem(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, pub_pem = _generate_keypair()
            pub_path = tmp / "op.pub"
            pub_path.write_bytes(pub_pem)
            pub = load_public_key_from_pem(pub_path)
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            assert isinstance(pub, Ed25519PublicKey)

    def test_missing_file_fails_fast(self):
        with pytest.raises(ComponentConfigurationError) as exc_info:
            load_public_key_from_pem("/nonexistent/op.pub")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_malformed_pem_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bad_path = tmp / "bad.pub"
            bad_path.write_bytes(b"not a real PEM file")
            with pytest.raises(ComponentConfigurationError) as exc_info:
                load_public_key_from_pem(bad_path)
            assert "FAIL-FAST" in str(exc_info.value)
            assert "failed to parse as PEM" in str(exc_info.value)

    def test_non_ed25519_key_rejected(self):
        """An RSA key (or any non-ed25519) is rejected — the framework
        commits to ed25519 specifically per the gap proposal."""
        from cryptography.hazmat.primitives.asymmetric.rsa import (
            generate_private_key,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rsa_priv = generate_private_key(public_exponent=65537, key_size=2048)
            rsa_pub_pem = rsa_priv.public_key().public_bytes(
                Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
            pub_path = tmp / "rsa.pub"
            pub_path.write_bytes(rsa_pub_pem)
            with pytest.raises(ComponentConfigurationError) as exc_info:
                load_public_key_from_pem(pub_path)
            assert "FAIL-FAST" in str(exc_info.value)
            assert "Ed25519PublicKey" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. verify_detached_signature
# ---------------------------------------------------------------------------

class TestVerifyDetachedSignature:

    def test_valid_signature_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg, sig, pub_path, _ = _setup_signed(tmp)
            pub_key = load_public_key_from_pem(pub_path)
            # Should not raise:
            verify_detached_signature(cfg, sig, pub_key)

    def test_tampered_config_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg, sig, pub_path, _ = _setup_signed(tmp)
            # Tamper with the config AFTER signing:
            cfg.write_bytes(b"name: tampered\n")
            pub_key = load_public_key_from_pem(pub_path)
            with pytest.raises(ComponentConfigurationError) as exc_info:
                verify_detached_signature(cfg, sig, pub_key)
            assert "FAIL-FAST" in str(exc_info.value)
            assert "verification FAILED" in str(exc_info.value)
            assert "tampered" in str(exc_info.value)

    def test_wrong_key_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg, sig, _, _ = _setup_signed(tmp)
            # Build a SECOND keypair and try to verify with it:
            _, wrong_pub_pem = _generate_keypair()
            wrong_pub_path = tmp / "wrong.pub"
            wrong_pub_path.write_bytes(wrong_pub_pem)
            wrong_pub_key = load_public_key_from_pem(wrong_pub_path)
            with pytest.raises(ComponentConfigurationError) as exc_info:
                verify_detached_signature(cfg, sig, wrong_pub_key)
            assert "FAIL-FAST" in str(exc_info.value)
            assert "verification FAILED" in str(exc_info.value)

    def test_missing_config_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, pub_path, _ = _setup_signed(tmp)
            pub_key = load_public_key_from_pem(pub_path)
            with pytest.raises(ComponentConfigurationError) as exc_info:
                verify_detached_signature(
                    tmp / "ghost.yml",
                    tmp / "workflow.yml.sig",
                    pub_key,
                )
            assert "FAIL-FAST" in str(exc_info.value)
            assert "config file not found" in str(exc_info.value)

    def test_missing_signature_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg, _, pub_path, _ = _setup_signed(tmp)
            pub_key = load_public_key_from_pem(pub_path)
            with pytest.raises(ComponentConfigurationError) as exc_info:
                verify_detached_signature(
                    cfg, tmp / "ghost.sig", pub_key,
                )
            assert "FAIL-FAST" in str(exc_info.value)
            assert "signature file not found" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. load_signed_config — full pipeline
# ---------------------------------------------------------------------------

class TestLoadSignedConfig:

    def test_full_pipeline_returns_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg, _, pub_path, _ = _setup_signed(
                tmp, content=b"name: my_workflow\nversion: 1\n")
            body = load_signed_config(cfg, pub_path, require_signed=True)
            assert body == b"name: my_workflow\nversion: 1\n"

    def test_missing_signature_with_require_signed_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, pub_path, _ = _setup_signed(tmp)
            cfg_no_sig = tmp / "unsigned.yml"
            cfg_no_sig.write_bytes(b"x: 1")
            with pytest.raises(ComponentConfigurationError) as exc_info:
                load_signed_config(cfg_no_sig, pub_path, require_signed=True)
            assert "FAIL-FAST" in str(exc_info.value)
            assert "no signature found" in str(exc_info.value)

    def test_missing_signature_with_require_signed_false_warns(self, caplog):
        """Legacy un-signed path: warning, not error."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, _, pub_path, _ = _setup_signed(tmp)
            cfg_no_sig = tmp / "unsigned.yml"
            cfg_no_sig.write_bytes(b"name: legacy\n")
            with caplog.at_level(logging.WARNING):
                body = load_signed_config(
                    cfg_no_sig, pub_path, require_signed=False)
            assert body == b"name: legacy\n"
            warnings = [r for r in caplog.records if r.levelname == "WARNING"]
            assert any("no signature" in r.getMessage() for r in warnings)

    def test_tampered_config_fails_fast_via_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg, _, pub_path, _ = _setup_signed(tmp)
            cfg.write_bytes(b"tampered: yes\n")
            with pytest.raises(ComponentConfigurationError):
                load_signed_config(cfg, pub_path, require_signed=True)

    def test_custom_signature_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            priv, pub_pem = _generate_keypair()
            cfg = tmp / "x.yml"
            cfg.write_bytes(b"x: 1\n")
            # Use ".pgp" instead of default ".sig":
            sig = tmp / "x.yml.pgp"
            sig.write_bytes(priv.sign(cfg.read_bytes()))
            pub_path = tmp / "op.pub"
            pub_path.write_bytes(pub_pem)
            body = load_signed_config(
                cfg, pub_path, signature_suffix=".pgp",
            )
            assert body == b"x: 1\n"
