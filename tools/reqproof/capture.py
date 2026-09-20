#!/usr/bin/env python3
"""Run checked proof commands and bind successful receipts to acceptance cells.

This is deliberately narrower than a generic command recorder. A proof must
be declared in the checked, content-hashed specification registry; execute from
an exactly pushed clean source commit; and leave tracked source unchanged.
Only then may its immutable receipt become requirement evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402
from tools.reqproof.evidence import (  # noqa: E402
    COMMAND_RECEIPT_SCHEMA_V2,
    EvidenceLedgerError,
    add_entry,
    expand_source_selectors,
    load_evidence_ledger,
    sha256_file,
    validate_source_selectors,
    write_evidence_ledger_atomic,
)
from tools.reqproof.schema import SHA256_RE, Registry, load_registry  # noqa: E402

SPEC_SCHEMA_VERSION = 1
PROOF_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
DEFAULT_SPEC_PATH = ROOT / "config" / "requirement_proof_specs.json"
DEFAULT_REGISTRY_PATH = ROOT / "config" / "requirement_registry.json"
DEFAULT_LEDGER_PATH = ROOT / "config" / "requirement_evidence_ledger.json"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "reqproof" / "evidence"
_BLOCKED_ENV_PREFIXES = ("AURA_", "PYTEST_", "COVERAGE_")
_OUTPUT_LIMIT_MAX = 8 * 1024 * 1024
_FAILURE_DIAGNOSTIC_BYTES = 4096
_FORBIDDEN_COMMANDS = frozenset({"bash", "dash", "fish", "osascript", "sh", "zsh"})
_BEARER_SECRET_RE = re.compile(r"(?i)(\bauthorization\s*:\s*bearer\s+)[^\s,;]+")
_NAMED_SECRET_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)"
    r"\b\s*[=:]\s*)[^\s,;]+"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [^-\n]*PRIVATE KEY-----.*?-----END [^-\n]*PRIVATE KEY-----",
    re.DOTALL,
)


class ProofCaptureError(RuntimeError):
    """A proof specification or capture violated the evidence contract."""


class Gateway(Protocol):
    def run(self, argv: Sequence[str], **kwargs: Any) -> Any: ...


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProofCaptureError(message)


def _string(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{name} must be a non-empty string")
    return cast(str, value)


def _canonical_sha256(body: dict[str, Any]) -> str:
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_relative_path(value: Any, name: str) -> str:
    text = _string(value, name)
    path = PurePosixPath(text)
    _require("\\" not in text, f"{name} must use POSIX separators")
    _require(not path.is_absolute(), f"{name} must be repo-relative")
    _require(
        bool(path.parts) and all(part not in {"", ".", ".."} for part in path.parts),
        f"{name} contains an unsafe path component",
    )
    _require(path.as_posix() == text, f"{name} is not canonical")
    return text


@dataclass(frozen=True)
class EvidenceTarget:
    requirement_id: str
    evidence_class: str
    acceptance_ids: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Any, name: str) -> EvidenceTarget:
        _require(isinstance(data, dict), f"{name} must be an object")
        allowed = {"requirement_id", "evidence_class", "acceptance_ids"}
        _require(set(data) == allowed, f"{name} fields must be exactly {sorted(allowed)}")
        acceptance = data["acceptance_ids"]
        _require(
            isinstance(acceptance, list) and bool(acceptance),
            f"{name}.acceptance_ids must be non-empty",
        )
        acceptance_ids = tuple(
            _string(item, f"{name}.acceptance_ids") for item in acceptance
        )
        _require(
            all(re.fullmatch(r"A[1-9][0-9]*", item) for item in acceptance_ids),
            f"{name}.acceptance_ids contains an invalid ID",
        )
        _require(
            acceptance_ids
            == tuple(sorted(set(acceptance_ids), key=lambda value: int(value[1:]))),
            f"{name}.acceptance_ids must be sorted unique A-number values",
        )
        return cls(
            requirement_id=_string(data["requirement_id"], f"{name}.requirement_id"),
            evidence_class=_string(data["evidence_class"], f"{name}.evidence_class"),
            acceptance_ids=acceptance_ids,
        )


@dataclass(frozen=True)
class ProofSpec:
    proof_id: str
    command: tuple[str, ...]
    cwd: str
    timeout_seconds: int
    max_output_bytes: int
    source_paths: tuple[str, ...]
    source_globs: tuple[str, ...]
    evidence_targets: tuple[EvidenceTarget, ...]

    @classmethod
    def from_dict(cls, data: Any, name: str) -> ProofSpec:
        _require(isinstance(data, dict), f"{name} must be an object")
        allowed = {
            "id",
            "command",
            "cwd",
            "timeout_seconds",
            "max_output_bytes",
            "source_paths",
            "source_globs",
            "evidence_targets",
        }
        _require(set(data) == allowed, f"{name} fields must be exactly {sorted(allowed)}")
        proof_id = _string(data["id"], f"{name}.id")
        _require(bool(PROOF_ID_RE.fullmatch(proof_id)), f"{name}.id is not canonical")
        command = data["command"]
        _require(
            isinstance(command, list) and bool(command),
            f"{name}.command must be non-empty",
        )
        argv = tuple(_string(item, f"{name}.command") for item in command)
        _require(
            all("{" not in item and "}" not in item for item in argv if item != "{python}"),
            f"{name}.command contains an unsupported placeholder",
        )
        argv0 = Path(argv[0]).name
        _require(
            argv0 not in _FORBIDDEN_COMMANDS,
            f"{name}.command may not invoke a shell or script interpreter",
        )
        _require(
            not (argv[0] == "{python}" and "-c" in argv[1:]),
            f"{name}.command may not execute inline Python",
        )
        timeout = data["timeout_seconds"]
        output_limit = data["max_output_bytes"]
        _require(
            isinstance(timeout, int) and 1 <= timeout <= 3600,
            f"{name}.timeout_seconds out of range",
        )
        _require(
            isinstance(output_limit, int) and 1024 <= output_limit <= _OUTPUT_LIMIT_MAX,
            f"{name}.max_output_bytes out of range",
        )
        try:
            selectors = validate_source_selectors(
                {"paths": data["source_paths"], "globs": data["source_globs"]}
            )
        except EvidenceLedgerError as exc:
            raise ProofCaptureError(f"{name} has invalid source selectors: {exc}") from exc
        targets_raw = data["evidence_targets"]
        _require(
            isinstance(targets_raw, list) and bool(targets_raw),
            f"{name}.evidence_targets must be non-empty",
        )
        targets = tuple(
            EvidenceTarget.from_dict(item, f"{name}.evidence_targets[{index}]")
            for index, item in enumerate(targets_raw)
        )
        target_keys = [
            (item.requirement_id, item.evidence_class, item.acceptance_ids)
            for item in targets
        ]
        _require(
            target_keys == sorted(set(target_keys)),
            f"{name}.evidence_targets must be sorted and unique",
        )
        cwd = data["cwd"]
        if cwd != ".":
            cwd = _safe_relative_path(cwd, f"{name}.cwd")
        return cls(
            proof_id=proof_id,
            command=argv,
            cwd=cwd,
            timeout_seconds=timeout,
            max_output_bytes=output_limit,
            source_paths=selectors["paths"],
            source_globs=selectors["globs"],
            evidence_targets=targets,
        )


@dataclass(frozen=True)
class ProofSpecRegistry:
    specs: tuple[ProofSpec, ...]
    content_sha256: str

    @classmethod
    def from_dict(cls, data: Any) -> ProofSpecRegistry:
        _require(isinstance(data, dict), "proof spec registry must be an object")
        _require(
            set(data) == {"schema_version", "specs", "content_sha256"},
            "proof spec registry fields are not exact",
        )
        _require(
            data["schema_version"] == SPEC_SCHEMA_VERSION,
            f"proof spec schema_version must be {SPEC_SCHEMA_VERSION}",
        )
        specs_raw = data["specs"]
        _require(isinstance(specs_raw, list), "proof spec registry specs must be a list")
        specs = tuple(
            ProofSpec.from_dict(item, f"specs[{index}]")
            for index, item in enumerate(specs_raw)
        )
        ids = [item.proof_id for item in specs]
        _require(ids == sorted(set(ids)), "proof specs must be sorted and unique by ID")
        recorded = _string(data["content_sha256"], "content_sha256")
        _require(bool(SHA256_RE.fullmatch(recorded)), "content_sha256 is not sha256")
        body = {"schema_version": SPEC_SCHEMA_VERSION, "specs": specs_raw}
        actual = _canonical_sha256(body)
        _require(
            recorded == actual,
            f"proof spec content hash mismatch: recorded {recorded[:12]}, actual {actual[:12]}",
        )
        return cls(specs=specs, content_sha256=recorded)

    def by_id(self) -> dict[str, ProofSpec]:
        return {item.proof_id: item for item in self.specs}


def load_proof_specs(path: Path) -> ProofSpecRegistry:
    try:
        return ProofSpecRegistry.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProofCaptureError(f"cannot load proof specs {path}: {exc}") from exc


def validate_spec_targets(specs: ProofSpecRegistry, registry: Registry) -> None:
    known = registry.by_id()
    for spec in specs.specs:
        for target in spec.evidence_targets:
            _require(
                target.requirement_id in known,
                f"{spec.proof_id} targets unknown requirement {target.requirement_id}",
            )
            requirement = known[target.requirement_id]
            _require(
                target.evidence_class in requirement.evidence_required,
                f"{spec.proof_id} targets unrequired class {target.evidence_class}",
            )
            valid = {
                f"A{index}" for index in range(1, len(requirement.acceptance) + 1)
            }
            _require(
                set(target.acceptance_ids) <= valid,
                f"{spec.proof_id} targets unknown acceptance units",
            )
            invalid_modalities = [
                acceptance_id
                for acceptance_id in target.acceptance_ids
                if target.evidence_class
                not in requirement.required_evidence_for(acceptance_id)
            ]
            _require(
                not invalid_modalities,
                f"{spec.proof_id} targets class {target.evidence_class} for "
                f"acceptance units that do not require it: {invalid_modalities}",
            )


def _git(gateway: Gateway, root: Path, *args: str) -> str:
    result = gateway.run(
        # Read-only, and bounded at thirty seconds below. The fsmonitor
        # daemon turns `status --untracked-files=all` on this checkout from
        # under a second into more than that, and a capture that times out
        # here refuses a proof for the clock rather than for the evidence.
        ["git", "-c", "core.fsmonitor=false", *args],
        cwd=root,
        timeout=30,
        read_only=True,
        source="reqproof_capture_git_probe",
        accelerator_capability="none",
    )
    _require(
        result.returncode == 0,
        f"git {' '.join(args)} failed: {result.stderr.strip()}",
    )
    return cast(str, result.stdout.strip())


def assert_pushed_clean_source(gateway: Gateway, root: Path) -> str:
    head = _git(gateway, root, "rev-parse", "HEAD")
    remote = _git(gateway, root, "rev-parse", "origin/main")
    _require(
        head == remote,
        f"proof source is not exact pushed main: HEAD={head[:12]} origin/main={remote[:12]}",
    )
    _require(
        not _git(gateway, root, "status", "--porcelain", "--untracked-files=all"),
        "proof source tree is dirty",
    )
    return head


def _resolve_sources(root: Path, paths: tuple[str, ...]) -> list[dict[str, Any]]:
    root_resolved = root.resolve()
    manifest: list[dict[str, Any]] = []
    for ref in paths:
        target = root.joinpath(*PurePosixPath(ref).parts)
        current = root
        for part in PurePosixPath(ref).parts:
            current = current / part
            _require(not current.is_symlink(), f"source path traverses a symlink: {ref}")
        _require(target.is_file(), f"source path is not a regular file: {ref}")
        _require(
            target.resolve().is_relative_to(root_resolved),
            f"source path escapes root: {ref}",
        )
        manifest.append(
            {
                "path": ref,
                "sha256": sha256_file(target),
                "size_bytes": target.stat().st_size,
            }
        )
    return manifest


def _proof_environment(log_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not any(key.startswith(prefix) for prefix in _BLOCKED_ENV_PREFIXES)
    }
    overrides = {
        "AURA_LOG_DIR": str(log_dir),
        "AURA_TEST_MODE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.update(overrides)
    inherited_names = sorted(key for key in env if key not in overrides)
    policy = {
        "name": "host-sanitized-v1",
        "removed_prefixes": list(_BLOCKED_ENV_PREFIXES),
        "override_names": sorted(overrides),
        "inherited_names_sha256": hashlib.sha256(
            "\n".join(inherited_names).encode("utf-8")
        ).hexdigest(),
    }
    return env, policy


def _redact_diagnostic(text: str) -> str:
    redacted = _BEARER_SECRET_RE.sub(r"\1<redacted>", text)
    redacted = _NAMED_SECRET_RE.sub(r"\1<redacted>", redacted)
    return _PRIVATE_KEY_RE.sub("<redacted-private-key>", redacted)


def _utf8_tail(text: str, budget: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= budget:
        return text
    marker = b"...[truncated]...\n"
    tail_budget = max(0, budget - len(marker))
    return (marker + encoded[-tail_budget:]).decode("utf-8", errors="replace")


def _failure_diagnostic(stdout: str, stderr: str) -> str:
    streams = [
        (label, _redact_diagnostic(value))
        for label, value in (("stdout", stdout), ("stderr", stderr))
        if value
    ]
    if not streams:
        return "<no captured output>"
    label_budget = sum(len(label.encode("utf-8")) + len(" tail:\n") for label, _ in streams)
    separator_budget = max(0, len(streams) - 1)
    content_budget = max(
        256,
        _FAILURE_DIAGNOSTIC_BYTES - label_budget - separator_budget,
    )
    per_stream_budget = max(128, content_budget // len(streams))
    return "\n".join(
        f"{label} tail:\n{_utf8_tail(value, per_stream_budget)}"
        for label, value in streams
    )


def _atomic_write_new(path: Path, payload: str) -> None:
    _require(not path.exists(), f"refusing to overwrite proof receipt {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, 0o644)
        temp.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)


def capture_proof(
    *,
    root: Path,
    spec_registry_path: Path,
    registry_path: Path,
    ledger_path: Path,
    artifact_root: Path,
    proof_id: str,
    record: bool,
    gateway: Gateway | None = None,
) -> Path:
    gateway = gateway or get_subprocess_gateway()
    specs = load_proof_specs(spec_registry_path)
    registry = load_registry(registry_path)
    validate_spec_targets(specs, registry)
    _require(proof_id in specs.by_id(), f"unknown proof ID {proof_id}")
    spec = specs.by_id()[proof_id]
    source_commit = assert_pushed_clean_source(gateway, root)
    source_selectors = {
        "paths": spec.source_paths,
        "globs": spec.source_globs,
    }
    expanded_sources = expand_source_selectors(root, source_selectors)
    source_manifest = _resolve_sources(root, expanded_sources)
    command = tuple(
        sys.executable if item == "{python}" else item for item in spec.command
    )
    cwd = root if spec.cwd == "." else root.joinpath(*PurePosixPath(spec.cwd).parts)
    _require(
        cwd.is_dir() and cwd.resolve().is_relative_to(root.resolve()),
        "proof cwd is invalid",
    )
    canonical_artifact_root = root / "artifacts" / "reqproof" / "evidence"
    _require(
        artifact_root.resolve() == canonical_artifact_root.resolve(),
        "artifact root must be the canonical repository evidence directory",
    )
    receipt_ref = PurePosixPath(
        "artifacts", "reqproof", "evidence", proof_id, f"{source_commit}.json"
    )
    receipt_path = root.joinpath(*receipt_ref.parts)

    started = datetime.now(UTC)
    monotonic_started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=f"aura-reqproof-{proof_id}-"
    ) as log_dir_text:
        env, env_policy = _proof_environment(Path(log_dir_text))
        try:
            result = gateway.run(
                command,
                cwd=cwd,
                env=env,
                timeout=spec.timeout_seconds,
                capture_output=True,
                text=True,
                stdin_devnull=True,
                offline_tooling=True,
                source=f"proof_tooling:reqproof_capture:{proof_id}",
                accelerator_capability="none",
            )
        except (TimeoutError, subprocess.TimeoutExpired) as exc:
            raise ProofCaptureError(
                f"proof {proof_id} exceeded {spec.timeout_seconds}s"
            ) from exc
    duration = time.monotonic() - monotonic_started
    finished = datetime.now(UTC)
    stdout = str(result.stdout or "")
    stderr = str(result.stderr or "")
    output_size = len(stdout.encode("utf-8")) + len(stderr.encode("utf-8"))
    _require(
        result.returncode == 0,
        f"proof {proof_id} failed with exit {result.returncode}:\n"
        f"{_failure_diagnostic(stdout, stderr)}",
    )
    _require(
        output_size <= spec.max_output_bytes,
        f"proof output {output_size} bytes exceeds {spec.max_output_bytes}-byte contract",
    )
    _require(
        _git(gateway, root, "rev-parse", "HEAD") == source_commit,
        "HEAD changed while proof was running",
    )
    _require(
        not _git(gateway, root, "status", "--porcelain", "--untracked-files=all"),
        "proof command mutated the source tree",
    )

    receipt = {
        "schema": COMMAND_RECEIPT_SCHEMA_V2,
        "proof_id": proof_id,
        "verdict": "pass",
        "source_commit": source_commit,
        "origin_main_commit": source_commit,
        "spec_registry": {
            "path": spec_registry_path.resolve().relative_to(root.resolve()).as_posix(),
            "content_sha256": specs.content_sha256,
        },
        "command": list(command),
        "cwd": spec.cwd,
        "timeout_seconds": spec.timeout_seconds,
        "returncode": result.returncode,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": round(duration, 6),
        "environment_policy": env_policy,
        "source_selectors": {
            "paths": list(spec.source_paths),
            "globs": list(spec.source_globs),
        },
        "source_manifest": source_manifest,
        "evidence_targets": [
            {
                "requirement_id": target.requirement_id,
                "evidence_class": target.evidence_class,
                "acceptance_ids": list(target.acceptance_ids),
            }
            for target in spec.evidence_targets
        ],
        "stdout": stdout,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr": stderr,
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "output_bytes": output_size,
        "non_claims": [
            "unlisted evidence classes or acceptance units",
            "live runtime behavior",
            "release readiness",
            "soak reliability",
        ],
    }
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    _atomic_write_new(receipt_path, payload)
    if not record:
        return receipt_path

    try:
        ledger = load_evidence_ledger(ledger_path)
        recorded_at = finished.date().isoformat()
        for target in spec.evidence_targets:
            ledger = add_entry(
                ledger,
                registry,
                requirement_id=target.requirement_id,
                evidence_class=target.evidence_class,
                acceptance_ids=target.acceptance_ids,
                ref=receipt_ref.as_posix(),
                commit=source_commit,
                recorded_at=recorded_at,
                root=root,
            )
        write_evidence_ledger_atomic(ledger, ledger_path)
    except (EvidenceLedgerError, OSError, ValueError):
        receipt_path.unlink(missing_ok=True)
        raise
    return receipt_path


def capture_proofs(
    *,
    root: Path,
    spec_registry_path: Path,
    registry_path: Path,
    ledger_path: Path,
    artifact_root: Path,
    proof_ids: Sequence[str],
    record: bool,
    gateway: Gateway | None = None,
) -> tuple[Path, ...]:
    """Capture several proofs against one clean pushed source snapshot.

    Each proof still runs through the single-proof contract. Its provisional
    receipt is removed before the next proof starts so generated custody files
    cannot make the source checkout appear dirty. Receipts and ledger entries
    become visible only after every proof has passed.
    """
    ordered_ids = tuple(proof_ids)
    _require(bool(ordered_ids), "at least one proof ID is required")
    _require(
        len(set(ordered_ids)) == len(ordered_ids),
        "batch proof IDs must be unique",
    )

    gateway = gateway or get_subprocess_gateway()
    pending: list[tuple[Path, str, dict[str, Any]]] = []
    try:
        for proof_id in ordered_ids:
            receipt_path = capture_proof(
                root=root,
                spec_registry_path=spec_registry_path,
                registry_path=registry_path,
                ledger_path=ledger_path,
                artifact_root=artifact_root,
                proof_id=proof_id,
                record=False,
                gateway=gateway,
            )
            payload = receipt_path.read_text(encoding="utf-8")
            receipt = json.loads(payload)
            receipt_path.unlink()
            pending.append((receipt_path, payload, receipt))
    except Exception:
        for receipt_path, _, _ in pending:
            receipt_path.unlink(missing_ok=True)
        raise

    written: list[Path] = []
    try:
        for receipt_path, payload, _ in pending:
            _atomic_write_new(receipt_path, payload)
            written.append(receipt_path)
        if not record:
            return tuple(written)

        registry = load_registry(registry_path)
        ledger = load_evidence_ledger(ledger_path)
        for receipt_path, _, receipt in pending:
            receipt_ref = receipt_path.relative_to(root).as_posix()
            recorded_at = datetime.fromisoformat(receipt["finished_at"]).date().isoformat()
            for target in receipt["evidence_targets"]:
                ledger = add_entry(
                    ledger,
                    registry,
                    requirement_id=target["requirement_id"],
                    evidence_class=target["evidence_class"],
                    acceptance_ids=target["acceptance_ids"],
                    ref=receipt_ref,
                    commit=receipt["source_commit"],
                    recorded_at=recorded_at,
                    root=root,
                )
        write_evidence_ledger_atomic(ledger, ledger_path)
    except (EvidenceLedgerError, OSError, ValueError):
        for receipt_path in written:
            receipt_path.unlink(missing_ok=True)
        raise
    return tuple(written)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        help="checked proof specification ID; repeat for an atomic batch",
    )
    parser.add_argument("--spec-registry", default=str(DEFAULT_SPEC_PATH))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument(
        "--record",
        action="store_true",
        help="atomically add the receipt to the evidence ledger",
    )
    args = parser.parse_args()
    try:
        receipts = capture_proofs(
            root=ROOT,
            spec_registry_path=Path(args.spec_registry),
            registry_path=Path(args.registry),
            ledger_path=Path(args.ledger),
            artifact_root=DEFAULT_ARTIFACT_ROOT,
            proof_ids=args.spec,
            record=args.record,
        )
    except (ProofCaptureError, EvidenceLedgerError, ValueError, OSError) as exc:
        print(f"reqproof capture failed: {exc}", file=sys.stderr)
        return 2
    result: dict[str, Any] = {
        "proof_ids": args.spec,
        "receipts": [str(receipt) for receipt in receipts],
        "recorded": args.record,
    }
    if len(receipts) == 1:
        result.update(proof_id=args.spec[0], receipt=str(receipts[0]))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
