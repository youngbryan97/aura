"""core/security/tls_local.py
──────────────────────────
Local self-signed TLS for the LAN surface.

Why: browsers require a secure context for getUserMedia — phone
microphone capture over plain HTTP is impossible no matter what the
server allows. A locally generated self-signed certificate (accepted
once on the phone) makes https://<lan-ip>:8000 a secure context, which
combined with the owner-granted per-device voice scope opens the phone
voice lane.

The certificate is generated once, keyed 0600, SANs covering localhost
and the host's current LAN addresses, 2-year validity, regenerated when
expired or when the LAN address set changes. Enabled only when
AURA_ENABLE_TLS=1 — the desktop app's plain-HTTP loopback default is
untouched otherwise.
"""
from __future__ import annotations

import datetime
import ipaddress
import logging
import socket
from pathlib import Path

from core.config import get_config
from core.runtime.errors import record_degradation

logger = logging.getLogger("Security.TLSLocal")

_TLS_ERRORS = (ImportError, OSError, RuntimeError, TypeError, ValueError)


def tls_enabled() -> bool:
    from core.runtime.flags import FlagKind, declare

    return bool(
        declare(
            "AURA_ENABLE_TLS",
            kind=FlagKind.BOOL,
            default=False,
            description="Serve the loopback interface over local TLS instead of plain HTTP",
            owner="core.security.tls_local",
        ).value()
    )


def tls_dir() -> Path:
    return Path(get_config().paths.data_dir) / "security" / "tls"


def _lan_ips() -> list[str]:
    addresses: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))  # route-selection only
            primary = probe.getsockname()[0]
            if primary:
                addresses.append(primary)
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            candidate = info[4][0]
            if candidate not in addresses:
                addresses.append(candidate)
    except OSError:
        pass
    return addresses


def ensure_local_certificate() -> tuple[Path, Path] | None:
    """Create (or reuse) the local self-signed cert. Returns
    (cert_path, key_path), or None when generation is impossible."""
    try:
        from cryptography import x509
        from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID

        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import (
            FileWriteBatchEntry,
            get_file_write_gateway,
        )

        directory = tls_dir()
        cert_path, key_path = directory / "aura_local.crt", directory / "aura_local.key"
        wanted_ips = sorted(set(_lan_ips()) | {"127.0.0.1"})

        def _matching_pair(certificate, private_key) -> bool:
            encoding = serialization.Encoding.DER
            public_format = serialization.PublicFormat.SubjectPublicKeyInfo
            return certificate.public_key().public_bytes(
                encoding,
                public_format,
            ) == private_key.public_key().public_bytes(encoding, public_format)

        def _load_pair() -> tuple[object, object]:
            certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
            private_key = serialization.load_pem_private_key(
                key_path.read_bytes(),
                password=None,
            )
            if not _matching_pair(certificate, private_key):
                raise ValueError("local TLS certificate and private key do not match")
            public_key = certificate.public_key()
            if not isinstance(public_key, ec.EllipticCurvePublicKey) or not isinstance(
                public_key.curve,
                ec.SECP256R1,
            ):
                raise ValueError("local TLS certificate must use ECDSA P-256")
            if certificate.subject != certificate.issuer:
                raise ValueError("local TLS certificate is not self-issued")
            public_key.verify(
                certificate.signature,
                certificate.tbs_certificate_bytes,
                ec.ECDSA(certificate.signature_hash_algorithm),
            )
            constraints = certificate.extensions.get_extension_for_class(
                x509.BasicConstraints
            ).value
            if constraints.ca:
                raise ValueError("local TLS leaf certificate cannot be a CA")
            return certificate, private_key

        if cert_path.exists() and key_path.exists():
            try:
                certificate, _private_key = _load_pair()
                not_after = certificate.not_valid_after_utc
                current_ips = {
                    str(ip)
                    for ip in certificate.extensions.get_extension_for_class(
                        x509.SubjectAlternativeName
                    ).value.get_values_for_type(x509.IPAddress)
                }
                current_dns = set(
                    certificate.extensions.get_extension_for_class(
                        x509.SubjectAlternativeName
                    ).value.get_values_for_type(x509.DNSName)
                )
                now = datetime.datetime.now(datetime.UTC)
                fresh = (
                    certificate.not_valid_before_utc <= now
                    and not_after > now + datetime.timedelta(days=30)
                )
                if fresh and set(wanted_ips) <= current_ips and "localhost" in current_dns:
                    return cert_path, key_path
                logger.info("Regenerating local TLS cert (expiring or LAN set changed)")
            except (
                InvalidSignature,
                OSError,
                TypeError,
                UnsupportedAlgorithm,
                ValueError,
                x509.ExtensionNotFound,
            ) as exc:
                logger.warning("Regenerating invalid local TLS material: %s", exc)
        elif cert_path.exists() or key_path.exists():
            logger.warning("Regenerating incomplete local TLS material")

        try:
            key = ec.generate_private_key(ec.SECP256R1())
        except UnsupportedAlgorithm as exc:
            raise RuntimeError("ECDSA P-256 is unavailable for local TLS") from exc
        name = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "Aura Local"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Aura"),
        ])
        san = x509.SubjectAlternativeName(
            [x509.DNSName("localhost")]
            + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in wanted_ips]
        )
        now = datetime.datetime.now(datetime.UTC)
        try:
            certificate = (
                x509.CertificateBuilder()
                .subject_name(name)
                .issuer_name(name)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now - datetime.timedelta(minutes=5))
                .not_valid_after(now + datetime.timedelta(days=730))
                .add_extension(san, critical=False)
                .add_extension(
                    x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .sign(key, hashes.SHA256())
            )
        except UnsupportedAlgorithm as exc:
            raise RuntimeError("SHA-256 signing is unavailable for local TLS") from exc
        key_bytes = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        cert_bytes = certificate.public_bytes(serialization.Encoding.PEM)
        gateway = get_file_write_gateway()
        with local_internal_governed_scope(
            "security.tls_local.ensure_certificate",
            domain="file_write",
        ):
            gateway.ensure_directory(
                directory,
                source="core.security.tls_local.ensure_certificate",
            )
            gateway.write_bytes_batch(
                (
                    FileWriteBatchEntry(key_path, key_bytes, mode=0o600),
                    FileWriteBatchEntry(cert_path, cert_bytes, mode=0o644),
                ),
                source="core.security.tls_local.ensure_certificate",
            )

        try:
            persisted_certificate, _persisted_key = _load_pair()
        except (InvalidSignature, UnsupportedAlgorithm) as exc:
            raise RuntimeError("persisted local TLS material is not verifiable") from exc
        if persisted_certificate.fingerprint(hashes.SHA256()) != certificate.fingerprint(
            hashes.SHA256()
        ):
            raise RuntimeError("persisted local TLS certificate failed verification")
        logger.info("Local TLS certificate ready (SANs: %s)", ", ".join(wanted_ips))
        return cert_path, key_path
    except _TLS_ERRORS as exc:
        record_degradation("security.tls_local", exc)
        logger.error("Local TLS certificate unavailable: %s", exc)
        return None
