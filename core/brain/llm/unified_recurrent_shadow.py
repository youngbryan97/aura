"""Fail-closed runtime loader for certified unified-recurrence shadow tissue.

The controller remains a separate neural module. Loading it must not mutate the
resident model or grant response-serving authority; promotion is a later gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Never

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from core.brain.llm.unified_recurrent_shadow_battery import (
    UnifiedRecurrentShadowBatteryError,
    validate_shadow_canary_battery,
)
from core.brain.llm.unified_recurrent_shadow_contract import (
    LOAD_SCHEMA,
    seal_shadow_load_receipt,
)
from core.brain.llm.unified_recurrent_shadow_migration import (
    shadow_source_migration_errors,
)
from core.brain.llm.unified_recurrent_shadow_probe_contract import (
    RECEIPT_SCHEMA as PROBE_RECEIPT_SCHEMA,
)
from core.brain.llm.unified_recurrent_shadow_probe_contract import (
    seal_shadow_probe_receipt,
    shadow_probe_request_errors,
    token_sequence_sha256,
)
from core.learning.intrinsic_recurrence import make_recurrent_caches
from core.learning.recurrent_answer_emission import (
    RecurrentAnswerEmissionContract,
    tokenizer_answer_emission_contract,
)
from core.learning.recurrent_literal_grounding import LiteralObservationContract
from core.learning.recurrent_opcode_grounding import (
    OpcodeObservationContract,
    tokenizer_opcode_contract,
)
from core.learning.unified_intrinsic_objective import UnifiedIntrinsicTrainingSpec
from core.learning.unified_intrinsic_recurrence import (
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
    unified_recurrent_logits,
)

PACKAGE_SCHEMA: Final = "aura.unified_intrinsic.shadow_package.v2"
COMPLETE_SCHEMA: Final = "aura.unified_intrinsic.shadow_package_complete.v2"
_MAX_JSON_BYTES: Final = 64 * 1024 * 1024
_MAX_CONTROLLER_BYTES: Final = 2 * 1024 * 1024 * 1024
_PACKAGE_ID: Final = re.compile(r"[a-z0-9][a-z0-9._-]{0,119}")


class UnifiedRecurrentShadowError(RuntimeError):
    """A shadow package cannot be proven compatible with this worker."""


def _fail(message: str) -> Never:
    raise UnifiedRecurrentShadowError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _stable_identity(path: Path, *, maximum: int) -> dict[str, Any]:
    if path.is_symlink():
        _fail("unified shadow artifact is a symlink")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o400
                or not 0 < before.st_size <= maximum
            ):
                _fail("unified shadow artifact custody differs")
            digest = hashlib.sha256()
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise UnifiedRecurrentShadowError("unified shadow artifact is unavailable") from exc
    if remaining or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        _fail("unified shadow artifact changed while reading")
    return {"sha256": digest.hexdigest(), "size_bytes": before.st_size}


def _stable_bytes(path: Path, *, maximum: int) -> bytes:
    identity = _stable_identity(path, maximum=maximum)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise UnifiedRecurrentShadowError("unified shadow artifact is unavailable") from exc
    if len(raw) != identity["size_bytes"] or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
        _fail("unified shadow artifact changed before canonical decode")
    return raw


def _load_controller_tensors(
    path: Path,
    binding: dict[str, Any],
) -> dict[str, Any]:
    """Load through one verified descriptor and close it before returning."""

    if path.is_symlink():
        _fail("unified shadow controller is a symlink")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            before = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o400
                or not 0 < before.st_size <= _MAX_CONTROLLER_BYTES
            ):
                _fail("unified shadow controller custody differs")
            digest = hashlib.sha256()
            remaining = before.st_size
            while remaining:
                chunk = handle.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
            if (
                remaining
                or before.st_size != binding.get("size_bytes")
                or digest.hexdigest() != binding.get("sha256")
            ):
                _fail("unified shadow controller bytes differ")
            handle.seek(0)
            tensors = mx.load(handle, format="safetensors")
            if not isinstance(tensors, dict):
                _fail("unified shadow controller payload differs")
            if tensors:
                mx.eval(*tensors.values())
            after = os.fstat(handle.fileno())
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                _fail("unified shadow controller changed while loading")
            return tensors
    except OSError as exc:
        raise UnifiedRecurrentShadowError(
            "unified shadow controller is unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _document(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _stable_bytes(path, maximum=_MAX_JSON_BYTES)
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UnifiedRecurrentShadowError("unified shadow document is invalid") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        _fail("unified shadow document is not canonical")
    return value, raw


def _decoded_bound_document(raw: bytes | None, *, role: str) -> dict[str, Any]:
    if raw is None:  # pragma: no cover - controller is the only undecoded role
        _fail(f"unified shadow {role} bytes are unavailable")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UnifiedRecurrentShadowError(
            f"unified shadow {role} document is invalid"
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        _fail(f"unified shadow {role} document is not canonical")
    return value


def _package_root(path: Path) -> Path:
    lexical = path.expanduser().absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            observed = current.lstat()
        except OSError as exc:
            raise UnifiedRecurrentShadowError("unified shadow package is unavailable") from exc
        if stat.S_ISLNK(observed.st_mode):
            _fail("unified shadow package path contains a symlink")
    root = lexical.resolve(strict=True)
    observed = root.stat()
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        _fail("unified shadow package custody differs")
    return root


def _binding(
    root: Path,
    value: Any,
    *,
    role: str,
    decode: bool = True,
) -> tuple[Path, bytes | None]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
        _fail(f"unified shadow {role} binding differs")
    name = value.get("path")
    if not isinstance(name, str) or Path(name).name != name:
        _fail(f"unified shadow {role} path differs")
    maximum = _MAX_CONTROLLER_BYTES if role == "controller" else _MAX_JSON_BYTES
    path = root / name
    identity = _stable_identity(path, maximum=maximum)
    if identity["size_bytes"] != value.get("size_bytes") or identity["sha256"] != value.get(
        "sha256"
    ):
        _fail(f"unified shadow {role} bytes differ")
    return path, (_stable_bytes(path, maximum=maximum) if decode else None)


def inspect_shadow_package(path: Path) -> dict[str, Any]:
    """Verify package custody, inventory, commitments, and non-authority."""

    root = _package_root(path)
    manifest, manifest_raw = _document(root / "manifest.json")
    completion, _completion_raw = _document(root / "PACKAGE_COMPLETE.json")
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    completion_body = {key: value for key, value in completion.items() if key != "complete_sha256"}
    domain = manifest.get("domain_contract")
    unsupported = manifest.get("claims_not_supported")
    package_id = manifest.get("package_id")
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or not isinstance(package_id, str)
        or _PACKAGE_ID.fullmatch(package_id) is None
        or manifest.get("manifest_sha256") != _canonical_sha256(manifest_body)
        or manifest.get("mode") != "shadow_only"
        or manifest.get("serving_authority") is not False
        or not isinstance(domain, dict)
        or domain.get("qualification") != "generator_and_grammar_bound"
        or not isinstance(domain.get("families"), list)
        or not domain["families"]
        or any(
            not isinstance(family, str) or not family
            for family in domain["families"]
        )
        or not isinstance(domain.get("task_depths"), list)
        or not domain["task_depths"]
        or any(type(depth) is not int or depth < 1 for depth in domain["task_depths"])
        or type(domain.get("recurrence_depth")) is not int
        or domain["recurrence_depth"] < 2
        or domain.get("ordinary_chat_authorized") is not False
        or domain.get("arbitrary_reasoning_authorized") is not False
        or not isinstance(unsupported, list)
        or unsupported.count("global_activation") != 1
        or unsupported.count("static_weight_fusion") != 1
    ):
        _fail("unified shadow manifest authority or identity differs")
    if (
        completion.get("schema") != COMPLETE_SCHEMA
        or completion.get("package_id") != manifest.get("package_id")
        or completion.get("manifest_sha256") != manifest.get("manifest_sha256")
        or completion.get("manifest_file_sha256") != hashlib.sha256(manifest_raw).hexdigest()
        or completion.get("mode") != "shadow_only"
        or completion.get("serving_authority") is not False
        or completion.get("complete_sha256") != _canonical_sha256(completion_body)
    ):
        _fail("unified shadow completion receipt differs")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        _fail("unified shadow artifact inventory is unavailable")
    bound: dict[str, tuple[Path, bytes | None]] = {}
    for role in (
        "controller",
        "checkpoint",
        "campaign_completion",
        "replication_plan",
        "replication_verdict",
        "canary_battery",
    ):
        bound[role] = _binding(
            root,
            artifacts.get(role),
            role=role,
            decode=role != "controller",
        )
    migration_binding = artifacts.get("source_migration")
    migration_sha256 = manifest.get("source_migration_sha256")
    if (migration_binding is None) != (migration_sha256 is None):
        _fail("unified shadow source migration inventory differs")
    if migration_binding is not None:
        bound["source_migration"] = _binding(
            root,
            migration_binding,
            role="source_migration",
        )
    reports = artifacts.get("replication_reports")
    if not isinstance(reports, list) or not reports:
        _fail("unified shadow replication reports are unavailable")
    for index, report in enumerate(reports, start=1):
        bound[f"replication_report_{index}"] = _binding(
            root,
            report,
            role=f"replication_report_{index}",
        )
    expected = {"manifest.json", "PACKAGE_COMPLETE.json"}
    expected.update(file_path.name for file_path, _raw in bound.values())
    if {entry.name for entry in root.iterdir()} != expected or completion.get(
        "bound_artifact_count"
    ) != len(bound) + 1:
        _fail("unified shadow artifact inventory differs")

    documents = {
        role: _decoded_bound_document(raw, role=role)
        for role, (_path, raw) in bound.items()
        if role != "controller"
    }
    checkpoint = documents["checkpoint"]
    completion = documents["campaign_completion"]
    plan = documents["replication_plan"]
    verdict = documents["replication_verdict"]
    canary_battery = documents["canary_battery"]
    try:
        validate_shadow_canary_battery(canary_battery)
    except UnifiedRecurrentShadowBatteryError as exc:
        raise UnifiedRecurrentShadowError(
            "unified shadow canary battery differs"
        ) from exc
    report_rows = verdict.get("reports")
    if (
        verdict.get("supported") is not True
        or verdict.get("verdict_sha256") != manifest.get("replication_verdict_sha256")
        or verdict.get("plan_sha256") != manifest.get("replication_plan_sha256")
        or plan.get("plan_sha256") != manifest.get("replication_plan_sha256")
        or verdict.get("checkpoint_sha256") != manifest.get("checkpoint_sha256")
        or checkpoint.get("checkpoint_sha256") != manifest.get("checkpoint_sha256")
        or not isinstance(checkpoint.get("identity"), dict)
        or checkpoint["identity"].get("identity_sha256")
        != manifest.get("checkpoint_identity_sha256")
        or canary_battery.get("battery_sha256")
        != manifest.get("canary_battery_sha256")
        or canary_battery.get("replication_plan_sha256")
        != manifest.get("replication_plan_sha256")
        or canary_battery.get("replication_verdict_sha256")
        != manifest.get("replication_verdict_sha256")
        or not isinstance(completion.get("checkpoint"), dict)
        or completion["checkpoint"].get("checkpoint_sha256")
        != manifest.get("checkpoint_sha256")
        or not isinstance(report_rows, list)
        or len(report_rows) != len(reports)
    ):
        _fail("unified shadow scientific evidence differs")
    if "source_migration" in documents:
        migration_errors = shadow_source_migration_errors(
            documents["source_migration"],
            manifest=manifest,
            checkpoint=checkpoint,
        )
        if migration_errors:
            _fail(
                "unified shadow source migration differs:"
                + ",".join(migration_errors)
            )
    report_commitments: dict[int, str] = {}
    for row in report_rows:
        if (
            not isinstance(row, dict)
            or type(row.get("seed")) is not int
            or not isinstance(row.get("report_sha256"), str)
            or row["seed"] in report_commitments
        ):
            _fail("unified shadow replication verdict report inventory differs")
        report_commitments[row["seed"]] = row["report_sha256"]
    observed_reports: dict[int, str] = {}
    for index in range(1, len(reports) + 1):
        report = documents[f"replication_report_{index}"]
        report_body = {
            key: value for key, value in report.items() if key != "report_sha256"
        }
        seed = report.get("evaluation_seed", report.get("seed"))
        if (
            type(seed) is not int
            or seed in observed_reports
            or report.get("report_sha256") != _canonical_sha256(report_body)
        ):
            _fail("unified shadow replication report identity differs")
        observed_reports[seed] = report["report_sha256"]
    if observed_reports != report_commitments:
        _fail("unified shadow replication report commitments differ")
    return {
        "root": root,
        "manifest": manifest,
        "checkpoint": checkpoint,
        "canary_battery": canary_battery,
        "controller_path": bound["controller"][0],
        "controller_binding": dict(artifacts["controller"]),
    }


def _contract_from_identity(
    identity: dict[str, Any],
) -> tuple[
    LiteralObservationContract,
    OpcodeObservationContract,
    RecurrentAnswerEmissionContract,
]:
    literal_value = identity.get("literal_observation_contract")
    opcode_value = identity.get("opcode_observation_contract")
    answer_value = identity.get("answer_emission_contract")
    if not all(isinstance(value, dict) for value in (literal_value, opcode_value, answer_value)):
        _fail("unified shadow tokenizer contracts are unavailable")
    literal = LiteralObservationContract(
        tuple(literal_value.get("digit_token_ids", ())),
        max_value=literal_value.get("max_value"),
        schema=literal_value.get("schema"),
    )
    opcode = OpcodeObservationContract(
        tuple(
            (row.get("opcode"), tuple(row.get("token_ids", ())))
            for row in opcode_value.get("patterns", ())
            if isinstance(row, dict)
        ),
        tuple(
            (row.get("name"), tuple(row.get("token_ids", ())))
            for row in opcode_value.get("contexts", ())
            if isinstance(row, dict)
        ),
        schema=opcode_value.get("schema"),
    )
    answer = RecurrentAnswerEmissionContract(
        digit_token_ids=tuple(answer_value.get("digit_token_ids", ())),
        eos_token_id=answer_value.get("eos_token_id"),
        family_markers=tuple(
            (row.get("family"), tuple(row.get("token_ids", ())))
            for row in answer_value.get("family_markers", ())
            if isinstance(row, dict)
        ),
        syntax=tuple(
            (row.get("name"), tuple(row.get("token_ids", ())))
            for row in answer_value.get("syntax", ())
            if isinstance(row, dict)
        ),
        schema=answer_value.get("schema"),
    )
    if (
        literal.contract_sha256 != literal_value.get("contract_sha256")
        or opcode.contract_sha256 != opcode_value.get("contract_sha256")
        or answer.contract_sha256 != answer_value.get("contract_sha256")
    ):
        _fail("unified shadow tokenizer contract commitments differ")
    return literal, opcode, answer


def _model_extent_matches(model_path: Path, identity: dict[str, Any]) -> bool:
    model_identity = identity.get("model")
    if not isinstance(model_identity, dict):
        return False
    try:
        model_root = model_path.resolve(strict=True)
        expected = Path(str(model_identity["canonical_path"])).resolve(strict=True)
        if model_root != expected:
            return False
        config = model_root / "config.json"
        if hashlib.sha256(config.read_bytes()).hexdigest() != model_identity.get("config_sha256"):
            return False
        weights = model_identity.get("weights")
        if not isinstance(weights, list) or not weights:
            return False
        for row in weights:
            if not isinstance(row, dict):
                return False
            candidate = model_root / str(row.get("name") or "")
            if (
                candidate.is_symlink()
                or candidate.parent != model_root
                or candidate.stat().st_size != row.get("size")
            ):
                return False
    # not a failure: a file whose stat does not match its row is not the file that
    # was recorded, and False is the refusing direction.
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _source_mechanics_match(identity: dict[str, Any]) -> bool:
    source = identity.get("source_sha256s")
    if not isinstance(source, dict):
        return False
    root = Path(__file__).resolve().parents[3]
    selected = {
        name: digest
        for name, digest in source.items()
        if isinstance(name, str) and name.startswith("core/learning/")
    }
    if not selected:
        return False
    try:
        return all(
            isinstance(digest, str)
            and hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
            for name, digest in selected.items()
        )
    # not a failure: a file that cannot be read cannot match its digest, and False is
    # the refusing direction.
    except OSError:
        return False


@dataclass(frozen=True, slots=True)
class LoadedUnifiedRecurrentShadow:
    controller: UnifiedRecurrentController
    spec: UnifiedIntrinsicTrainingSpec
    answer_contract: RecurrentAnswerEmissionContract
    receipt: dict[str, Any]
    canary_battery: dict[str, Any] | None = None
    literal_contract: LiteralObservationContract | None = None
    opcode_contract: OpcodeObservationContract | None = None

    def supports(self, public_tokens: list[int] | tuple[int, ...]) -> bool:
        return self.answer_contract.family(public_tokens) in set(self.receipt["families"])

    def decode_general_recurrent_tokens(
        self,
        model: Any,
        public_tokens: Sequence[int],
        *,
        max_tokens: int,
        recurrence_depth: int | None = None,
        controller: UnifiedRecurrentController | None = None,
        completion_check: Callable[[tuple[int, ...]], bool] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        activity: Callable[[], None] | None = None,
    ) -> tuple[tuple[int, ...], bool, int]:
        """Decode through recurrent hidden tissue without typed answer helpers.

        This is an experiment surface, not serving authority.  It deliberately
        omits state-slot insertion, the terminal grammar, and the digit pointer
        so broader evaluations measure only the controller's shared hidden-state
        correction and transport operators under the frozen language-model head.
        """

        if not public_tokens or max_tokens < 1:
            raise UnifiedRecurrentShadowError(
                "general recurrent decode dimensions are invalid"
            )
        selected = self.controller if controller is None else controller
        if not isinstance(selected, UnifiedRecurrentController):
            raise UnifiedRecurrentShadowError(
                "general recurrent decode controller is invalid"
            )
        depth = (
            int(self.receipt["recurrence_depth"])
            if recurrence_depth is None
            else recurrence_depth
        )
        if type(depth) is not int or depth < 1:
            raise UnifiedRecurrentShadowError(
                "general recurrent decode depth is invalid"
            )
        row = tuple(int(value) for value in public_tokens)
        tokens = mx.array([row], dtype=mx.int32)
        generated: list[int] = []
        stopped = False
        plan = self.spec.plan_at(depth)
        caches = make_recurrent_caches(model, plan)
        eos_token_id = self.answer_contract.eos_token_id
        started = time.perf_counter()
        from core.brain.llm.latent_cortex.recurrence_adapter import (
            recurrence_adapter_scope,
        )

        with recurrence_adapter_scope(start=None, stop=None):
            next_tokens = tokens
            for _index in range(max_tokens):
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("unified_general_recurrent_decode_cancelled")
                if activity is not None:
                    activity()
                output, _telemetry = unified_recurrent_logits(
                    model,
                    next_tokens,
                    plan,
                    selected,
                    answer_digit_pointer_enabled=False,
                    caches=caches,
                )
                logits = (
                    output.logits
                    if hasattr(output, "logits")
                    else output[0]
                    if isinstance(output, tuple)
                    else output
                )
                token_id = int(mx.argmax(logits[0, -1]).item())
                generated.append(token_id)
                if eos_token_id is not None and token_id == eos_token_id:
                    stopped = True
                    break
                if completion_check is not None and completion_check(tuple(generated)):
                    stopped = True
                    break
                next_tokens = mx.array([[token_id]], dtype=tokens.dtype)
        elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000.0)))
        return tuple(generated), stopped, elapsed_ms

    def decode_recurrent_tokens(
        self,
        model: Any,
        public_tokens: Sequence[int],
        *,
        max_tokens: int,
        cancel_check: Callable[[], bool] | None = None,
        activity: Callable[[], None] | None = None,
        progress: Callable[[Mapping[str, int | str]], None] | None = None,
    ) -> tuple[tuple[int, ...], bool, int]:
        """Run the one canonical recurrent token loop used by every live arm.

        The progress contract reports only counts and stages. Token identities
        remain inside the worker until an independently authorized caller
        validates the typed answer.
        """

        if not public_tokens or max_tokens < 1:
            raise UnifiedRecurrentShadowError(
                "unified recurrent decode dimensions are invalid"
            )
        row = tuple(int(value) for value in public_tokens)
        tokens = mx.array([row], dtype=mx.int32)
        generated: list[int] = []
        stopped = False
        plan = self.spec.plan_at(int(self.receipt["recurrence_depth"]))
        started = time.perf_counter()
        if progress is not None:
            progress(
                {
                    "stage": "recurrent_decode_started",
                    "generated_token_count": 0,
                    "maximum_token_count": max_tokens,
                }
            )
        from core.brain.llm.latent_cortex.recurrence_adapter import (
            recurrence_adapter_scope,
        )

        with recurrence_adapter_scope(start=None, stop=None):
            for _index in range(max_tokens):
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("unified_recurrent_decode_cancelled")
                if activity is not None:
                    activity()
                output, _telemetry = unified_recurrent_logits(
                    model,
                    tokens,
                    plan,
                    self.controller,
                    state_slot_start=len(row),
                    answer_emission_contract=self.answer_contract,
                    answer_digit_pointer_enabled=True,
                )
                logits = (
                    output.logits
                    if hasattr(output, "logits")
                    else output[0]
                    if isinstance(output, tuple)
                    else output
                )
                token_id = int(mx.argmax(logits[0, -1]).item())
                generated.append(token_id)
                if progress is not None:
                    progress(
                        {
                            "stage": "recurrent_token_generated",
                            "generated_token_count": len(generated),
                            "maximum_token_count": max_tokens,
                        }
                    )
                if token_id == self.answer_contract.eos_token_id:
                    stopped = True
                    break
                tokens = mx.concatenate(
                    [tokens, mx.array([[token_id]], dtype=tokens.dtype)],
                    axis=1,
                )
        elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000.0)))
        if progress is not None:
            progress(
                {
                    "stage": "recurrent_decode_completed",
                    "generated_token_count": len(generated),
                    "maximum_token_count": max_tokens,
                }
            )
        return tuple(generated), stopped, elapsed_ms

    def probe(
        self,
        model: Any,
        request: dict[str, Any],
        *,
        cancel_check: Callable[[], bool] | None = None,
        activity: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Run matched base/recurrent decodes without exposing either output."""

        errors = shadow_probe_request_errors(request)
        if errors:
            raise UnifiedRecurrentShadowError(",".join(errors))
        public_tokens = tuple(int(value) for value in request["public_token_ids"])
        expected_tokens = tuple(int(value) for value in request["expected_token_ids"])
        max_tokens = int(request["max_tokens"])
        request_sha256 = str(request["request_sha256"])
        family = self.answer_contract.family(public_tokens)
        if family not in set(self.receipt["families"]):
            return self._empty_probe_receipt(
                request_sha256=request_sha256,
                status="abstained",
                reason="unsupported_public_token_family",
                input_token_count=len(public_tokens),
                expected_token_count=len(expected_tokens),
                max_tokens=max_tokens,
            )
        if expected_tokens[-1] != self.answer_contract.eos_token_id:
            raise UnifiedRecurrentShadowError(
                "unified recurrent shadow expected sequence lacks exact EOS"
            )

        prompt = mx.array([public_tokens], dtype=mx.int32)

        def logits(value: Any) -> Any:
            if hasattr(value, "logits"):
                return value.logits
            if isinstance(value, tuple):
                return value[0]
            return value

        def decode(operation: Callable[[Any], Any]) -> tuple[tuple[int, ...], bool, int]:
            started = time.perf_counter()
            tokens = prompt
            generated: list[int] = []
            stopped = False
            for _index in range(max_tokens):
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("unified_recurrent_shadow_probe_cancelled")
                if activity is not None:
                    activity()
                next_logits = operation(tokens)
                token_id = int(mx.argmax(next_logits[0, -1]).item())
                generated.append(token_id)
                if token_id == self.answer_contract.eos_token_id:
                    stopped = True
                    break
                tokens = mx.concatenate(
                    [tokens, mx.array([[token_id]], dtype=tokens.dtype)],
                    axis=1,
                )
            elapsed_ms = max(0, int(round((time.perf_counter() - started) * 1000.0)))
            return tuple(generated), stopped, elapsed_ms

        base_tokens, base_stopped, base_latency_ms = decode(
            lambda tokens: logits(model(tokens))
        )
        shadow_tokens, shadow_stopped, shadow_latency_ms = (
            self.decode_recurrent_tokens(
                model,
                public_tokens,
                max_tokens=max_tokens,
                cancel_check=cancel_check,
                activity=activity,
            )
        )
        output_commitment_key = secrets.token_bytes(32)
        body = {
            "schema": PROBE_RECEIPT_SCHEMA,
            "request_sha256": request_sha256,
            "status": "completed",
            "reason": "matched_shadow_probe_completed",
            "package_id": self.receipt["package_id"],
            "controller_sha256": self.receipt["controller_sha256"],
            "family": family,
            "recurrence_depth": int(self.receipt["recurrence_depth"]),
            "input_token_count": len(public_tokens),
            "expected_token_count": len(expected_tokens),
            "max_tokens": max_tokens,
            "base_token_count": len(base_tokens),
            "base_output_sha256": token_sequence_sha256(
                base_tokens,
                key=output_commitment_key,
            ),
            "base_exact_match": base_tokens == expected_tokens,
            "base_stopped_on_eos": base_stopped,
            "base_latency_ms": base_latency_ms,
            "shadow_token_count": len(shadow_tokens),
            "shadow_output_sha256": token_sequence_sha256(
                shadow_tokens,
                key=output_commitment_key,
            ),
            "shadow_exact_match": shadow_tokens == expected_tokens,
            "shadow_stopped_on_eos": shadow_stopped,
            "shadow_latency_ms": shadow_latency_ms,
            "outputs_equal": shadow_tokens == base_tokens,
            "output_exposed": False,
            "serving_authority": False,
        }
        return seal_shadow_probe_receipt(body)

    def _empty_probe_receipt(
        self,
        *,
        request_sha256: str,
        status: str,
        reason: str,
        input_token_count: int,
        expected_token_count: int,
        max_tokens: int,
    ) -> dict[str, Any]:
        return seal_shadow_probe_receipt(
            {
                "schema": PROBE_RECEIPT_SCHEMA,
                "request_sha256": request_sha256,
                "status": status,
                "reason": reason,
                "package_id": self.receipt["package_id"],
                "controller_sha256": self.receipt["controller_sha256"],
                "family": "",
                "recurrence_depth": int(self.receipt["recurrence_depth"]),
                "input_token_count": input_token_count,
                "expected_token_count": expected_token_count,
                "max_tokens": max_tokens,
                "base_token_count": 0,
                "base_output_sha256": "",
                "base_exact_match": False,
                "base_stopped_on_eos": False,
                "base_latency_ms": 0,
                "shadow_token_count": 0,
                "shadow_output_sha256": "",
                "shadow_exact_match": False,
                "shadow_stopped_on_eos": False,
                "shadow_latency_ms": 0,
                "outputs_equal": False,
                "output_exposed": False,
                "serving_authority": False,
            }
        )


def load_unified_recurrent_shadow(
    package: Path,
    *,
    model: Any,
    tokenizer: Any,
    model_path: Path,
) -> LoadedUnifiedRecurrentShadow:
    """Load compatible controller weights without altering the resident model."""

    verified = inspect_shadow_package(package)
    manifest = verified["manifest"]
    checkpoint = verified["checkpoint"]
    identity = checkpoint.get("identity")
    if not isinstance(identity, dict):
        _fail("unified shadow checkpoint identity is unavailable")
    identity_body = {key: value for key, value in identity.items() if key != "identity_sha256"}
    if identity.get("identity_sha256") != _canonical_sha256(identity_body):
        _fail("unified shadow checkpoint identity differs")
    spec_identity = identity.get("spec")
    if not isinstance(spec_identity, dict):
        _fail("unified shadow recurrence specification is unavailable")
    if any(
        type(identity.get(key)) is not int
        for key in ("controller_rank", "depth_basis_size", "init_seed")
    ) or any(
        int(identity[key]) < minimum
        for key, minimum in (
            ("controller_rank", 1),
            ("depth_basis_size", 1),
            ("init_seed", 0),
        )
    ):
        _fail("unified shadow controller construction identity differs")
    expected_wiring = {
        "window_tissue_mode": "controller_only",
        "window": [spec_identity.get("prelude_end"), spec_identity.get("coda_start")],
        "adapted_sites": [],
        "adapted_projection_count": 0,
        "continuous_depth_operator_count": 0,
        "continuous_depth_basis_size": 0,
        "coda_adapted": False,
        "readout_adapted": False,
        "ordinary_inference_requires_scope": False,
        "recurrence_phase_trains_shared_state_bridge": False,
        "state_bridge": "typed_recurrent_controller_only",
    }
    if (
        identity.get("window_tissue_mode") != "controller_only"
        or identity.get("wiring") != expected_wiring
        or not _source_mechanics_match(identity)
        or not _model_extent_matches(model_path, identity)
    ):
        _fail("unified shadow mechanics or resident model binding differs")
    literal, opcode, answer = _contract_from_identity(identity)
    if (
        tokenizer_opcode_contract(tokenizer) != opcode
        or tokenizer_answer_emission_contract(tokenizer, opcode) != answer
        or tuple(literal.digit_token_ids) != tuple(answer.digit_token_ids)
    ):
        _fail("unified shadow live tokenizer differs")

    spec_value = dict(spec_identity)
    spec_value["train_depths"] = tuple(spec_value["train_depths"])
    spec_value["heldout_depths"] = tuple(spec_value["heldout_depths"])
    spec = UnifiedIntrinsicTrainingSpec(**spec_value)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers or not 0 <= spec.prelude_end < spec.coda_start <= len(layers):
        _fail("unified shadow recurrent window differs from resident model")
    hidden_size = int(layers[0].input_layernorm.weight.shape[0])
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=hidden_size,
            correction_rank=int(identity["controller_rank"]),
            depth_basis_size=int(identity["depth_basis_size"]),
            minimum_iterations=1,
            initialization_seed=int(identity["init_seed"]),
            literal_digit_token_ids=literal.digit_token_ids,
            opcode_token_patterns=opcode.patterns,
            opcode_context_patterns=opcode.contexts,
        )
    )
    tensors = _load_controller_tensors(
        verified["controller_path"],
        verified["controller_binding"],
    )
    prefix = "bundle.controller."
    loaded = {
        name.removeprefix(prefix): value
        for name, value in tensors.items()
        if name.startswith(prefix)
    }
    expected = dict(tree_flatten(controller.trainable_parameters()))
    if set(tensors) != {f"{prefix}{name}" for name in expected} or set(loaded) != set(expected):
        _fail("unified shadow controller tensor inventory differs")
    if any(tuple(loaded[name].shape) != tuple(expected[name].shape) for name in expected):
        _fail("unified shadow controller tensor shape differs")
    controller.update(tree_unflatten(list(loaded.items())))
    mx.eval(controller.parameters())
    controller_sha256 = controller.parameter_sha256()
    migrated_controller_sha256 = identity.get("source_migration_controller_sha256")
    if (
        migrated_controller_sha256 is not None
        and migrated_controller_sha256 != controller_sha256
    ):
        _fail("unified shadow migrated controller identity differs")
    body = {
        "schema": LOAD_SCHEMA,
        "configured": True,
        "loaded": True,
        "reason": "certified_shadow_package_loaded",
        "package_id": manifest["package_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "controller_sha256": controller_sha256,
        "families": list(manifest["domain_contract"]["families"]),
        "task_depths": list(manifest["domain_contract"]["task_depths"]),
        "recurrence_depth": int(manifest["domain_contract"]["recurrence_depth"]),
        "model_identity_strength": "config_behavior_hash_and_weight_extent",
        "mode": "shadow_only",
        "serving_authority": False,
    }
    receipt = seal_shadow_load_receipt(body)
    return LoadedUnifiedRecurrentShadow(
        controller,
        spec,
        answer,
        receipt,
        verified["canary_battery"],
        literal,
        opcode,
    )


__all__ = [
    "COMPLETE_SCHEMA",
    "LOAD_SCHEMA",
    "PACKAGE_SCHEMA",
    "LoadedUnifiedRecurrentShadow",
    "UnifiedRecurrentShadowError",
    "inspect_shadow_package",
    "load_unified_recurrent_shadow",
]
