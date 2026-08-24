"""Identity rules for a bounded-capability package, so a verdict cannot migrate.

CP566 adjudicated a `BOUNDED_WOW_SIGNAL` on a Qwen2.5-32B fuse and CP568 sealed
it into a serving package. The checkpoint underneath has been replaced. The
package noticed -- `active_model_mismatch,resident_manifest_drift`, with zero
drifted source files -- and that is the binding doing its job.

The tempting repair is to re-seal the existing package against the new
checkpoint. Every hash would recompute, the alarm would go green, and a claim
measured on one model would be serving on another. So the rules here make that
specific move impossible rather than discouraged:

* a package id is derived from the descriptor it was measured against, so the
  27B cannot produce the CP568 id no matter what is put in the payload;
* every evidence file a package cites must name the same descriptor, and a
  citation naming another one is a refusal rather than a warning;
* the evidence namespace is part of the identity, so a fresh campaign cannot
  write into the directory holding the old verdict.

None of this decides whether the 27B is any good. It decides that whatever it
turns out to be will be said about the 27B.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

RECOVERY_PACKAGE_SCHEMA: Final = "aura.rlc.bounded_recovery_package.v1"
RECOVERY_CERTIFICATE_SCHEMA: Final = "aura.rlc.bounded_recovery_certificate.v1"

#: The 32B package this one must never be mistaken for.
LEGACY_PACKAGE_ID: Final = "cp568-resident-semantic-neural-active-r1"
LEGACY_EVIDENCE_ROOT: Final = "artifacts/closeout/latent_cortex"

#: A fresh campaign writes here and nowhere else. Sharing a root with the old
#: verdict is how an adjudicator ends up reading one campaign's file while
#: believing it read another's.
RECOVERY_EVIDENCE_ROOT: Final = "artifacts/migration/27b/recovery"


class RecoveryPackageError(RuntimeError):
    """The package does not describe the checkpoint it claims to."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class DescriptorIdentity:
    """The part of a checkpoint that a claim is allowed to be about."""

    path: str
    config_sha256: str
    weights_index_sha256: str
    tokenizer_sha256: str
    model_type: str
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    full_attention_layers: int
    linear_attention_layers: int

    def fingerprint(self) -> str:
        """Stable across a move, sensitive to every byte that decides behaviour.

        The path is deliberately excluded. A checkpoint copied to another
        directory is the same checkpoint, and a claim that broke when somebody
        reorganised a models folder would teach people to re-seal packages.
        """
        return canonical_sha256(
            {
                "config_sha256": self.config_sha256,
                "weights_index_sha256": self.weights_index_sha256,
                "tokenizer_sha256": self.tokenizer_sha256,
                "model_type": self.model_type,
                "num_hidden_layers": self.num_hidden_layers,
                "hidden_size": self.hidden_size,
                "vocab_size": self.vocab_size,
                "full_attention_layers": self.full_attention_layers,
                "linear_attention_layers": self.linear_attention_layers,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "config_sha256": self.config_sha256,
            "weights_index_sha256": self.weights_index_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "model_type": self.model_type,
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "full_attention_layers": self.full_attention_layers,
            "linear_attention_layers": self.linear_attention_layers,
            "fingerprint": self.fingerprint(),
        }


def package_id(descriptor: DescriptorIdentity, *, campaign: str) -> str:
    """Derive the id from the checkpoint, so it cannot be chosen.

    A hand-written id is a field somebody can copy from the old package. This
    one changes the moment any behaviour-deciding byte does.
    """
    campaign = str(campaign or "").strip().lower()
    if not campaign or not campaign.replace("-", "").replace("_", "").isalnum():
        raise RecoveryPackageError("campaign label must be a plain slug")
    return f"{campaign}-{descriptor.fingerprint()[:20]}"


def _behavior_file_sha256(behavior: dict[str, Any], name: str) -> str:
    files = behavior.get("files")
    if not isinstance(files, list):
        raise RecoveryPackageError("descriptor behavior identity carries no files")
    matches = [
        record
        for record in files
        if isinstance(record, dict) and record.get("path") == name
    ]
    if len(matches) != 1:
        raise RecoveryPackageError(
            f"descriptor behavior identity must name {name} exactly once"
        )
    digest = matches[0].get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RecoveryPackageError(
            f"descriptor behavior identity has an invalid {name} digest"
        )
    return digest


def descriptor_from_manifest(manifest: dict[str, Any]) -> DescriptorIdentity:
    descriptor = manifest.get("artifact_descriptor")
    if not isinstance(descriptor, dict):
        raise RecoveryPackageError("manifest carries no artifact_descriptor")
    profile = descriptor.get("artifact_profile")
    if not isinstance(profile, dict):
        raise RecoveryPackageError("descriptor carries no artifact_profile")
    behaviour = descriptor.get("behavior_identity")
    if not isinstance(behaviour, dict):
        raise RecoveryPackageError("descriptor carries no behavior_identity")
    return DescriptorIdentity(
        path=str(profile.get("path") or manifest.get("active_model_path") or ""),
        config_sha256=_behavior_file_sha256(behaviour, "config.json"),
        weights_index_sha256=_behavior_file_sha256(
            behaviour,
            "model.safetensors.index.json",
        ),
        tokenizer_sha256=_behavior_file_sha256(behaviour, "tokenizer.json"),
        model_type=str(profile.get("model_type") or ""),
        num_hidden_layers=int(profile.get("num_hidden_layers") or 0),
        hidden_size=int(profile.get("hidden_size") or 0),
        vocab_size=int(profile.get("vocab_size") or 0),
        full_attention_layers=int(profile.get("full_attention_layers") or 0),
        linear_attention_layers=int(profile.get("linear_attention_layers") or 0),
    )


def evidence_namespace_errors(paths: list[str]) -> list[str]:
    """Refuse a package that writes into, or reads a verdict from, the old root."""
    errors: list[str] = []
    for path in paths:
        normalized = str(path).replace("\\", "/")
        if normalized.startswith(LEGACY_EVIDENCE_ROOT):
            errors.append(f"evidence_in_legacy_namespace:{normalized}")
        elif not normalized.startswith(RECOVERY_EVIDENCE_ROOT):
            errors.append(f"evidence_outside_recovery_namespace:{normalized}")
    return errors


def inherited_verdict_errors(package: dict[str, Any]) -> list[str]:
    """Refuse any route by which the 32B verdict becomes this package's."""
    errors: list[str] = []
    if package.get("package_id") == LEGACY_PACKAGE_ID:
        errors.append("package_reuses_the_legacy_id")
    legacy = package.get("legacy_claim")
    if not isinstance(legacy, dict):
        errors.append("package_does_not_name_the_legacy_claim_it_supersedes")
    elif legacy.get("authority_over_this_package") != "none":
        errors.append("legacy_claim_is_granted_authority")
    verdict = package.get("verdict")
    evidence = package.get("evidence") or {}
    if verdict and not evidence.get("measured_on_this_checkpoint"):
        errors.append("verdict_without_evidence_measured_on_this_checkpoint")
    return errors


def certificate_errors(
    certificate: dict[str, Any], descriptor: DescriptorIdentity
) -> list[str]:
    """Independent verification: does this certificate describe THIS model?"""
    errors: list[str] = []
    if certificate.get("schema") != RECOVERY_CERTIFICATE_SCHEMA:
        return ["certificate_schema_unrecognised"]

    body = {k: v for k, v in certificate.items() if k != "certificate_sha256"}
    if canonical_sha256(body) != certificate.get("certificate_sha256"):
        errors.append("certificate_digest_does_not_cover_the_certificate")

    identity = certificate.get("descriptor_identity")
    if not isinstance(identity, dict):
        errors.append("certificate_names_no_descriptor")
    elif identity.get("fingerprint") != descriptor.fingerprint():
        errors.append("certificate_describes_a_different_checkpoint")

    errors.extend(inherited_verdict_errors(certificate))
    errors.extend(
        evidence_namespace_errors(list(certificate.get("evidence_paths") or []))
    )
    return errors


def build_package(
    descriptor: DescriptorIdentity,
    *,
    campaign: str,
    evidence_paths: list[str] | None = None,
    verdict: str | None = None,
    measured_on_this_checkpoint: bool = False,
) -> dict[str, Any]:
    """A package that has not measured anything yet carries no verdict."""
    body: dict[str, Any] = {
        "schema": RECOVERY_PACKAGE_SCHEMA,
        "package_id": package_id(descriptor, campaign=campaign),
        "descriptor_identity": descriptor.as_dict(),
        "evidence_root": RECOVERY_EVIDENCE_ROOT,
        "evidence_paths": sorted(evidence_paths or []),
        "verdict": verdict,
        "evidence": {"measured_on_this_checkpoint": bool(measured_on_this_checkpoint)},
        "legacy_claim": {
            "package_id": LEGACY_PACKAGE_ID,
            "checkpoint": "Aura-32B-crsm-closeout-jul1-20260701-215118",
            "verdict": "BOUNDED_WOW_SIGNAL",
            "authority_over_this_package": "none",
            "disposition": "retained as 32B history; stays inactive on this checkpoint",
        },
        "authorizes": [],
        "never_authorizes": [
            "ordinary_chat_authorized",
            "arbitrary_reasoning_authorized",
            "global runtime promotion",
            "static weight fusion",
            "frontier performance",
        ],
    }
    problems = inherited_verdict_errors(body) + evidence_namespace_errors(
        body["evidence_paths"]
    )
    if problems:
        raise RecoveryPackageError("; ".join(problems))
    return {**body, "package_sha256": canonical_sha256(body)}


def load_manifest(install_root: Path) -> dict[str, Any]:
    path = Path(install_root) / "training/fused-model/active.json"
    if not path.exists():
        raise RecoveryPackageError(f"no active cortex manifest at {path}")
    return json.loads(path.read_text())
