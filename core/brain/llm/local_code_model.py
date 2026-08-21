"""Provenance-bound local code generation on Aura's in-process model lane.

The code lane deliberately bypasses persona steering because steering can
corrupt symbolic output.  Bypass does not mean ungoverned: model identity,
resource ownership, request bounds, cancellation, output completeness, and
cleanup are all explicit and receipted here.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import hashlib
import json
import logging
import math
import os
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
    canonical_json_bytes,
    full_weight_checkpoint_identity,
    model_behavior_bundle_identity,
)
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_REL = "training/fused-model/Aura-32B-20260510-151144"
_TRUST_SCHEMA = "aura.local_code_model.trust.v1"
_MAX_CODE_TOKENS = 2048


def max_code_tokens() -> int:
    """The largest generation this lane will accept.

    Published so callers ask instead of guessing. build_app defaulted to 9000
    against this 2048 and raised local_code_model_max_tokens_out_of_policy on
    every single call — a skill whose own default made it impossible to run.
    """
    return _MAX_CODE_TOKENS
_MAX_PROMPT_BYTES = 2 * 1024 * 1024
_MAX_CONTEXT_TOKENS = 131_072
_DEFAULT_TIMEOUT_S = 600.0
_MAX_TIMEOUT_S = 1800.0
_CLEANUP_TIMEOUT_S = 60.0
_SUPPORTED_ARCHITECTURES = frozenset({"Qwen2ForCausalLM", "Qwen3ForCausalLM"})
_REQUIRED_MODEL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")

_load_lock = threading.RLock()
_generation_state_lock = threading.Lock()
_model: Any = None
_tokenizer: Any = None
_loaded_path: str | None = None
_loaded_identity: TrustedModelIdentity | None = None
_lane_lease: Any = None
_active_generations = 0
_eviction_in_progress = False


class LocalCodeModelError(RuntimeError):
    """Stable failure envelope for callers that need to select a fallback."""


class ReadinessState(StrEnum):
    ABSENT = "absent"
    UNVERIFIED = "unverified"
    CONFIGURED = "configured"
    VALIDATING = "validating"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TrustedModelIdentity:
    real_path: str
    checkpoint_fingerprint: str
    checkpoint_files: int
    behavior_bundle_sha256: str
    architecture: str
    max_context_tokens: int
    trust_manifest_sha256: str

    @property
    def privacy_safe_id(self) -> str:
        return self.checkpoint_fingerprint[:16]


@dataclass(frozen=True, slots=True)
class ReadinessReceipt:
    state: ReadinessState
    reason: str
    model_id: str = ""
    checked_at: float = 0.0


@dataclass(frozen=True, slots=True)
class DeviceReleaseReceipt:
    references_cleared: bool
    backend_synchronized: bool
    cache_cleared: bool
    active_memory_before: int | None
    active_memory_after: int | None
    verified: bool


@dataclass(frozen=True, slots=True)
class UnloadReceipt:
    requested_path_matches: bool
    references_cleared: bool
    lease_released: bool
    device_release: DeviceReleaseReceipt
    reason: str

    def __bool__(self) -> bool:
        return self.references_cleared or self.lease_released


@dataclass(frozen=True, slots=True)
class CodeGenerationReceipt:
    request_id: str
    model_id: str
    checkpoint_fingerprint: str
    behavior_bundle_sha256: str
    trust_manifest_sha256: str
    lane_receipt_id: str
    lane_fencing_token_sha256: str
    route: str
    steering_hooks: tuple[str, ...]
    seed: int
    temperature: float
    max_tokens: int
    input_tokens: int
    output_tokens: int
    termination: str
    validation_status: str
    output_sha256: str
    elapsed_s: float


@dataclass(frozen=True, slots=True)
class CodeGenerationResult:
    text: str
    receipt: CodeGenerationReceipt


class _FairAsyncGate:
    """FIFO cross-event-loop gate with cancellation-safe waiter removal."""

    def __init__(self) -> None:
        self._lock = checked_lock("local_code_model")
        self._waiters: deque[tuple[object, threading.Event]] = deque()
        self._owner: object | None = None

    def _grant_next_locked(self) -> None:
        if self._owner is None and self._waiters:
            token, event = self._waiters[0]
            self._owner = token
            event.set()

    async def acquire(self, *, deadline: float) -> tuple[object, int]:
        token = object()
        event = threading.Event()
        with self._lock:
            position = len(self._waiters)
            self._waiters.append((token, event))
            self._grant_next_locked()
        try:
            remaining = _remaining(deadline, "lifecycle_gate")
            granted = await asyncio.to_thread(event.wait, remaining)
            if not granted:
                raise TimeoutError("local_code_model_deadline:lifecycle_gate")
            return token, position
        except BaseException:
            with self._lock:
                was_owner = self._owner is token
                self._waiters = deque(item for item in self._waiters if item[0] is not token)
                if was_owner:
                    self._owner = None
                self._grant_next_locked()
            event.set()
            raise

    def release(self, token: object) -> None:
        with self._lock:
            if self._owner is not token:
                raise RuntimeError("local_code_model_lifecycle_owner_mismatch")
            if not self._waiters or self._waiters[0][0] is not token:
                raise RuntimeError("local_code_model_lifecycle_queue_corrupt")
            self._waiters.popleft()
            self._owner = None
            self._grant_next_locked()


_lifecycle_gate = _FairAsyncGate()


def _remaining(deadline: float, stage: str) -> float:
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0.0:
        raise TimeoutError(f"local_code_model_deadline:{stage}")
    return remaining


def _deadline(timeout_s: Any) -> float:
    try:
        timeout = float(timeout_s)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("local_code_model_timeout_invalid") from exc
    if not math.isfinite(timeout) or not 0.1 <= timeout <= _MAX_TIMEOUT_S:
        raise ValueError("local_code_model_timeout_out_of_policy")
    return time.monotonic() + timeout


@contextlib.asynccontextmanager
async def _lifecycle_context(*, deadline: float) -> AsyncIterator[int]:
    token, position = await _lifecycle_gate.acquire(deadline=deadline)
    try:
        yield position
    finally:
        _lifecycle_gate.release(token)


def _resolve_model_path() -> str:
    configured = os.environ.get("AURA_CODE_MODEL_PATH") or os.environ.get("AURA_MODEL_PATH")
    if configured:
        return str(Path(configured).expanduser())
    root = Path(__file__).resolve().parents[3]
    return str(root / _DEFAULT_MODEL_REL)


def _trust_manifest_path() -> Path:
    configured = os.environ.get("AURA_CODE_MODEL_TRUST_MANIFEST")
    if configured:
        return Path(configured).expanduser()
    # state_root(), not Path.home()/".aura". Resolving the live root here meant
    # a TEST run wrote its code-model trust manifest into the real instance's
    # state — deciding which model the live Aura trusts to write code. That is
    # exactly the class core/runtime/state_ownership.py exists to prevent, and
    # asking state_root() is how a non-live profile gets its own root.
    from core.runtime.state_ownership import state_root

    return state_root() / "trust" / "local_code_model.json"


def _canonical_path(path: str) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def build_model_trust_manifest(model_path: str | Path) -> dict[str, Any]:
    """Build trust material for explicit, out-of-band operator enrollment."""
    root = Path(model_path).expanduser().resolve(strict=True)
    checkpoint = full_weight_checkpoint_identity(root)
    behavior = model_behavior_bundle_identity(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    architecture = str(architectures[0]) if architectures else ""
    payload: dict[str, Any] = {
        "schema": _TRUST_SCHEMA,
        "model_real_path": str(root),
        "checkpoint": checkpoint,
        "behavior": behavior,
        "architecture": architecture,
        "created_at": time.time(),
    }
    payload["manifest_sha256"] = _manifest_digest(payload)
    return payload


def write_model_trust_manifest(model_path: str | Path, destination: str | Path) -> Path:
    """Write an operator-requested trust root outside the enrolled model tree."""
    root = Path(model_path).expanduser().resolve(strict=True)
    target = Path(destination).expanduser().resolve(strict=False)
    if target == root or root in target.parents:
        raise ValueError("trust_manifest_must_be_external_to_model")
    atomic_write_text(
        target,
        json.dumps(build_model_trust_manifest(root), sort_keys=True, separators=(",", ":")),
        durable=True,
        mode=0o600,
    )
    return target


def _load_trust_manifest(model_root: Path) -> tuple[dict[str, Any], Path]:
    configured_path = _trust_manifest_path()
    if configured_path.is_symlink():
        raise LocalCodeModelError("local_code_model_trust_root_symlink_rejected")
    manifest_path = configured_path.resolve(strict=True)
    if manifest_path == model_root or model_root in manifest_path.parents:
        raise LocalCodeModelError("local_code_model_trust_root_not_external")
    stat = manifest_path.stat()
    if stat.st_uid != os.getuid() or stat.st_mode & 0o022:
        raise LocalCodeModelError("local_code_model_trust_root_permissions_invalid")
    if stat.st_size > 4 * 1024 * 1024:
        raise LocalCodeModelError("local_code_model_trust_manifest_too_large")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != _TRUST_SCHEMA:
        raise LocalCodeModelError("local_code_model_trust_manifest_invalid")
    if payload.get("manifest_sha256") != _manifest_digest(payload):
        raise LocalCodeModelError("local_code_model_trust_manifest_digest_mismatch")
    if payload.get("model_real_path") != str(model_root):
        raise LocalCodeModelError("local_code_model_trust_path_mismatch")
    return payload, manifest_path


def _validate_model_trust(model_path: str) -> TrustedModelIdentity:
    requested = Path(model_path).expanduser()
    if requested.is_symlink():
        raise LocalCodeModelError("local_code_model_symlink_rejected")
    root = requested.resolve(strict=True)
    if not root.is_dir():
        raise LocalCodeModelError("local_code_model_not_directory")
    for name in _REQUIRED_MODEL_FILES:
        candidate = root / name
        if not candidate.is_file() or candidate.is_symlink():
            raise LocalCodeModelError(f"local_code_model_required_file_invalid:{name}")
    payload, _manifest_path = _load_trust_manifest(root)
    checkpoint = full_weight_checkpoint_identity(root)
    behavior = model_behavior_bundle_identity(root)
    if payload.get("checkpoint") != checkpoint:
        raise LocalCodeModelError("local_code_model_checkpoint_identity_mismatch")
    if payload.get("behavior") != behavior:
        raise LocalCodeModelError("local_code_model_behavior_identity_mismatch")

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    architecture = str(architectures[0]) if architectures else ""
    if architecture not in _SUPPORTED_ARCHITECTURES or payload.get("architecture") != architecture:
        raise LocalCodeModelError("local_code_model_architecture_unsupported")
    max_context = int(config.get("max_position_embeddings") or 0)
    if not 256 <= max_context <= _MAX_CONTEXT_TOKENS:
        raise LocalCodeModelError("local_code_model_context_window_invalid")
    return TrustedModelIdentity(
        real_path=str(root),
        checkpoint_fingerprint=str(checkpoint["fingerprint"]),
        checkpoint_files=int(checkpoint["files"]),
        behavior_bundle_sha256=str(behavior["bundle_sha256"]),
        architecture=architecture,
        max_context_tokens=max_context,
        trust_manifest_sha256=str(payload["manifest_sha256"]),
    )


def _reuse_or_validate_model_trust(model_path: str) -> TrustedModelIdentity:
    """Reuse identity only while its exact, fenced model remains resident."""
    try:
        requested = str(Path(model_path).expanduser().resolve(strict=True))
    except OSError:
        requested = ""
    with _load_lock:
        if (
            requested
            and _model is not None
            and _lane_lease is not None
            and _loaded_identity is not None
            and _loaded_identity.real_path == requested
        ):
            return _loaded_identity
    return _validate_model_trust(model_path)


def _ensure_loaded(model_path: str, identity: TrustedModelIdentity) -> None:
    global _loaded_identity, _loaded_path, _model, _tokenizer
    if _model is not None and _loaded_identity == identity:
        return
    with _load_lock:
        if _model is not None and _loaded_identity == identity:
            return
        from mlx_lm import load

        logger.info("[CODE] Loading verified local code model id=%s", identity.privacy_safe_id)
        loaded_model, loaded_tokenizer = load(identity.real_path)
        if loaded_model is None or loaded_tokenizer is None:
            raise LocalCodeModelError("local_code_model_load_returned_empty_components")
        _model, _tokenizer = loaded_model, loaded_tokenizer
        _loaded_path = identity.real_path
        _loaded_identity = identity
        logger.info("[CODE] Verified unsteered code model ready id=%s", identity.privacy_safe_id)


def _clear_loaded_model() -> DeviceReleaseReceipt:
    global _loaded_identity, _loaded_path, _model, _tokenizer
    with _load_lock:
        had_references = _model is not None or _tokenizer is not None
        active_before: int | None = None
        active_after: int | None = None
        synchronized = False
        cache_cleared = False
        backend_present = False
        try:
            import mlx.core as mx

            backend_present = True
            mx.synchronize()
            synchronized = True
            getter = getattr(mx, "get_active_memory", None)
            if callable(getter):
                active_before = int(getter())
            _model = None
            _tokenizer = None
            _loaded_path = None
            _loaded_identity = None
            gc.collect()
            clear_cache = getattr(mx, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()
                cache_cleared = True
            mx.synchronize()
            if callable(getter):
                active_after = int(getter())
        except ImportError:
            _model = None
            _tokenizer = None
            _loaded_path = None
            _loaded_identity = None
            gc.collect()
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _model = None
            _tokenizer = None
            _loaded_path = None
            _loaded_identity = None
            gc.collect()
            record_degradation("local_code_model.cleanup", exc, severity="critical")
        verified = (not backend_present) or (
            synchronized
            and cache_cleared
            and (active_after is None or active_before is None or active_after <= active_before)
        )
        return DeviceReleaseReceipt(
            references_cleared=had_references,
            backend_synchronized=synchronized,
            cache_cleared=cache_cleared,
            active_memory_before=active_before,
            active_memory_after=active_after,
            verified=verified,
        )


async def _cleanup_loaded_model(*, timeout_s: float = _CLEANUP_TIMEOUT_S) -> DeviceReleaseReceipt:
    from core.runtime.model_lane_control import run_owned_model_thread_call

    return await run_owned_model_thread_call(
        _clear_loaded_model,
        operation_name="local-code-model-cleanup",
        timeout_s=timeout_s,
    )


async def unload_local_code_model(
    *,
    reason: str = "local_code_model_unloaded",
    expected_path: str | None = None,
    timeout_s: float = _CLEANUP_TIMEOUT_S,
) -> UnloadReceipt:
    global _lane_lease
    deadline = _deadline(timeout_s)
    expected_real = (
        _canonical_path(expected_path)
        if expected_path is not None
        else None
    )
    async with _lifecycle_context(deadline=deadline):
        path_matches = expected_real is None or _loaded_path in {None, expected_real}
        if not path_matches:
            return UnloadReceipt(
                requested_path_matches=False,
                references_cleared=False,
                lease_released=False,
                device_release=DeviceReleaseReceipt(False, False, False, None, None, False),
                reason="loaded_model_owned_by_different_path",
            )
        lease, _lane_lease = _lane_lease, None
        release = await _cleanup_loaded_model(timeout_s=_remaining(deadline, "cleanup"))
        released = False
        if lease is not None:
            released = bool(await asyncio.wait_for(
                lease.release(reason=reason), timeout=_remaining(deadline, "lease_release")
            ))
        if not release.verified:
            logger.critical("Local code model device release was not measurable: %s", release)
        return UnloadReceipt(path_matches, release.references_cleared, released, release, reason)


async def _evict_local_code_model(owner: Any, reason: str) -> bool:
    global _eviction_in_progress
    with _generation_state_lock:
        if _active_generations > 0 or _eviction_in_progress:
            return False
        _eviction_in_progress = True
    try:
        expected = str(getattr(owner, "model_path", "") or "") or None
        receipt = await unload_local_code_model(
            reason=f"lane_eviction:{reason}", expected_path=expected
        )
        return receipt.requested_path_matches and receipt.device_release.verified
    finally:
        with _generation_state_lock:
            _eviction_in_progress = False


async def _compensate_local_code_model(owner: Any, _reason: str) -> bool:
    path = str(getattr(owner, "model_path", "") or _resolve_model_path())
    deadline = _deadline(_DEFAULT_TIMEOUT_S)
    async with _lifecycle_context(deadline=deadline):
        identity = await asyncio.wait_for(
            asyncio.to_thread(_validate_model_trust, path),
            timeout=_remaining(deadline, "trust_validation"),
        )
        await _ensure_loaded_with_lane(path, identity, deadline=deadline)
    return bool(_model is not None and _loaded_identity == identity and _lane_lease is not None)


async def _ensure_loaded_with_lane(
    model_path: str,
    identity: TrustedModelIdentity,
    *,
    deadline: float,
) -> None:
    global _lane_lease
    if _model is not None and _loaded_identity == identity and _lane_lease is not None:
        return
    if _active_generations > 1:
        raise LocalCodeModelError("local_code_model_path_replacement_during_generation")
    prior_lease, _lane_lease = _lane_lease, None
    if _model is not None:
        release = await _cleanup_loaded_model(timeout_s=_remaining(deadline, "replacement_cleanup"))
        if not release.verified:
            raise LocalCodeModelError("local_code_model_replacement_cleanup_unverified")
    if prior_lease is not None:
        await asyncio.wait_for(
            prior_lease.release(reason="local_code_model_path_replaced"),
            timeout=_remaining(deadline, "prior_lease_release"),
        )

    from core.runtime.model_lane_control import (
        acquire_in_process_model_lane,
        run_owned_model_thread_call,
    )

    lease = await asyncio.wait_for(
        acquire_in_process_model_lane(
            owner_id="local-code-model",
            model_path=model_path,
            purpose="serve",
            priority=50,
            preemptible=False,
            evict=_evict_local_code_model,
            compensate=_compensate_local_code_model,
            metadata={
                "provider": "local_code_model",
                "unsteered": True,
                "model_id": identity.privacy_safe_id,
                "trust_manifest_sha256": identity.trust_manifest_sha256,
                "activation_state": "loading",
            },
        ),
        timeout=_remaining(deadline, "lane_admission"),
    )
    try:
        await run_owned_model_thread_call(
            lambda: _ensure_loaded(model_path, identity),
            operation_name="local-code-model-load",
            timeout_s=_remaining(deadline, "model_load"),
        )
    except BaseException:
        with contextlib.suppress(Exception):
            await _cleanup_loaded_model(timeout_s=_CLEANUP_TIMEOUT_S)
        await lease.release(reason="local_code_model_load_rolled_back")
        raise
    if not await asyncio.wait_for(
        lease.set_preemptible(True), timeout=_remaining(deadline, "activation_fence")
    ):
        await _cleanup_loaded_model(timeout_s=_CLEANUP_TIMEOUT_S)
        await lease.release(reason="local_code_model_activation_fence_lost")
        raise LocalCodeModelError("local_code_model_activation_fence_lost")
    _lane_lease = lease


def _bounded_int(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"local_code_model_{name}_invalid")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"local_code_model_{name}_invalid") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"local_code_model_{name}_out_of_policy")
    return result


def _bounded_temperature(value: Any) -> float:
    try:
        temperature = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("local_code_model_temperature_invalid") from exc
    if not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
        raise ValueError("local_code_model_temperature_out_of_policy")
    return temperature


def _encode_tokens(tokenizer: Any, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        encoded = encode(text)
    elif callable(tokenizer):
        encoded = tokenizer(text)
        if isinstance(encoded, Mapping):
            encoded = encoded.get("input_ids")
    else:
        raise LocalCodeModelError("local_code_model_tokenizer_encode_unavailable")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, Sequence) and encoded and isinstance(encoded[0], Sequence):
        encoded = encoded[0]
    if not isinstance(encoded, Sequence):
        raise LocalCodeModelError("local_code_model_tokenizer_result_invalid")
    return [int(item) for item in encoded]


def _lane_identity(lease: Any) -> tuple[str, str]:
    decision = getattr(lease, "decision", None)
    fencing_token = str(getattr(decision, "fencing_token", "") or "")
    return (
        str(getattr(decision, "receipt_id", "") or "unavailable"),
        hashlib.sha256(fencing_token.encode("utf-8")).hexdigest()
        if fencing_token
        else "unavailable",
    )


class LocalCodeModel:
    """Raw local code model with trusted identity and bounded execution."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or _resolve_model_path()
        self._readiness = ReadinessReceipt(ReadinessState.UNVERIFIED, "not_probed")
        self._last_result: CodeGenerationResult | None = None
        self._state_lock = checked_lock("local_code_model")

    def readiness(self) -> ReadinessReceipt:
        with self._state_lock:
            return self._readiness

    def _set_readiness(self, state: ReadinessState, reason: str, model_id: str = "") -> None:
        with self._state_lock:
            self._readiness = ReadinessReceipt(state, reason, model_id, time.time())

    def is_configured(self) -> bool:
        try:
            path = Path(self.model_path).expanduser()
            manifest = _trust_manifest_path()
            return path.is_dir() and manifest.is_file()
        except OSError:
            return False

    def is_available(self) -> bool:
        """Return true only after this exact instance has completed a load probe."""
        return self.readiness().state is ReadinessState.READY

    async def probe_readiness(self, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> ReadinessReceipt:
        deadline = _deadline(timeout_s)
        self._set_readiness(ReadinessState.VALIDATING, "trust_validation_started")
        try:
            async with _lifecycle_context(deadline=deadline):
                identity = await asyncio.wait_for(
                    asyncio.to_thread(_reuse_or_validate_model_trust, self.model_path),
                    timeout=_remaining(deadline, "trust_validation"),
                )
                self._set_readiness(ReadinessState.LOADING, "bounded_load_probe", identity.privacy_safe_id)
                await _ensure_loaded_with_lane(self.model_path, identity, deadline=deadline)
            self._set_readiness(ReadinessState.READY, "verified_loaded", identity.privacy_safe_id)
        except FileNotFoundError:
            self._set_readiness(ReadinessState.ABSENT, "model_or_trust_root_absent")
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            self._set_readiness(ReadinessState.FAILED, f"{type(exc).__name__}:{exc}")
        return self.readiness()

    def _prepare_prompt_sync(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        identity: TrustedModelIdentity,
    ) -> tuple[str, int]:
        with _load_lock:
            if _model is None or _tokenizer is None or _loaded_identity != identity:
                raise LocalCodeModelError("local_code_model_not_loaded")
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            full_prompt = str(
                _tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            )
            input_tokens = len(_encode_tokens(_tokenizer, full_prompt))
            if input_tokens + max_tokens > identity.max_context_tokens:
                raise LocalCodeModelError(
                    "local_code_model_context_budget_exceeded:"
                    f"input={input_tokens}:output={max_tokens}:window={identity.max_context_tokens}"
                )
            return full_prompt, input_tokens

    def _generate_sync(
        self,
        full_prompt: str,
        max_tokens: int,
        temperature: float,
        seed: int,
        identity: TrustedModelIdentity,
    ) -> tuple[str, int]:
        with _load_lock:
            if _model is None or _tokenizer is None or _loaded_identity != identity:
                raise LocalCodeModelError("local_code_model_not_loaded")
            from mlx_lm import generate as mlx_generate
            from mlx_lm.sample_utils import make_sampler

            with contextlib.suppress(ImportError, AttributeError):
                import mlx.core as mx

                mx.random.seed(seed)
            sampler = make_sampler(temp=temperature)
            text = str(
                mlx_generate(
                    _model,
                    _tokenizer,
                    prompt=full_prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
            )
            return text, len(_encode_tokens(_tokenizer, text))

    async def think_with_receipt(
        self,
        prompt: str,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> CodeGenerationResult:
        global _active_generations
        started = time.monotonic()
        prompt = str(prompt or "")
        system_prompt = str(system_prompt or "")
        if not prompt.strip():
            raise ValueError("local_code_model_prompt_empty")
        if len(prompt.encode("utf-8")) + len(system_prompt.encode("utf-8")) > _MAX_PROMPT_BYTES:
            raise ValueError("local_code_model_prompt_bytes_exceeded")
        max_tokens = _bounded_int(
            kwargs.get("max_tokens", _MAX_CODE_TOKENS),
            name="max_tokens",
            minimum=1,
            maximum=_MAX_CODE_TOKENS,
        )
        temperature = _bounded_temperature(kwargs.get("temperature", 0.0))
        seed = _bounded_int(
            kwargs.get("seed", 0), name="seed", minimum=0, maximum=2**32 - 1
        )
        deadline = _deadline(kwargs.get("timeout_s", _DEFAULT_TIMEOUT_S))
        request_id = hashlib.sha256(
            f"{time.time_ns()}:{id(self)}:{seed}".encode("ascii")
        ).hexdigest()[:24]

        async with _lifecycle_context(deadline=deadline):
            with _generation_state_lock:
                if _eviction_in_progress:
                    raise LocalCodeModelError("local_code_model_eviction_in_progress")
                _active_generations += 1
            try:
                self._set_readiness(ReadinessState.VALIDATING, "trust_validation_started")
                identity = await asyncio.wait_for(
                    asyncio.to_thread(_reuse_or_validate_model_trust, self.model_path),
                    timeout=_remaining(deadline, "trust_validation"),
                )
                self._set_readiness(ReadinessState.LOADING, "load_or_reuse", identity.privacy_safe_id)
                await _ensure_loaded_with_lane(self.model_path, identity, deadline=deadline)
                self._set_readiness(ReadinessState.READY, "verified_loaded", identity.privacy_safe_id)

                from core.runtime.model_lane_control import run_owned_model_thread_call

                full_prompt, input_tokens = await run_owned_model_thread_call(
                    lambda: self._prepare_prompt_sync(
                        prompt, system_prompt, max_tokens, identity
                    ),
                    operation_name="local-code-model-context-admission",
                    timeout_s=_remaining(deadline, "context_admission"),
                )
                text, output_tokens = await run_owned_model_thread_call(
                    lambda: self._generate_sync(
                        full_prompt, max_tokens, temperature, seed, identity
                    ),
                    operation_name="local-code-model-generate",
                    timeout_s=_remaining(deadline, "generation"),
                )
                stripped = text.strip()
                termination = "token_limit" if output_tokens >= max_tokens else "completed"
                validation = "complete_nonempty" if stripped and termination == "completed" else "invalid"
                if not stripped:
                    raise LocalCodeModelError("local_code_model_empty_output")
                if termination != "completed":
                    raise LocalCodeModelError(
                        f"local_code_model_output_incomplete:tokens={output_tokens}:cap={max_tokens}"
                    )
                lane_receipt, fencing_token_sha256 = _lane_identity(_lane_lease)
                receipt = CodeGenerationReceipt(
                    request_id=request_id,
                    model_id=identity.privacy_safe_id,
                    checkpoint_fingerprint=identity.checkpoint_fingerprint,
                    behavior_bundle_sha256=identity.behavior_bundle_sha256,
                    trust_manifest_sha256=identity.trust_manifest_sha256,
                    lane_receipt_id=lane_receipt,
                    lane_fencing_token_sha256=fencing_token_sha256,
                    route="local_mlx_unsteered_no_cloud",
                    steering_hooks=(),
                    seed=seed,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    termination=termination,
                    validation_status=validation,
                    output_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    elapsed_s=time.monotonic() - started,
                )
                result = CodeGenerationResult(text=text, receipt=receipt)
                with self._state_lock:
                    self._last_result = result
                return result
            except asyncio.CancelledError:
                raise
            except (OSError, RuntimeError, ValueError, TimeoutError):
                self._set_readiness(ReadinessState.FAILED, "generation_failed")
                raise
            finally:
                with _generation_state_lock:
                    _active_generations = max(0, _active_generations - 1)

    async def think(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        return (await self.think_with_receipt(prompt, system_prompt, **kwargs)).text

    async def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        return await self.think(prompt, system_prompt=system_prompt, **kwargs)

    def last_receipt(self) -> dict[str, Any] | None:
        with self._state_lock:
            return asdict(self._last_result.receipt) if self._last_result is not None else None

    async def close(self) -> UnloadReceipt:
        receipt = await unload_local_code_model(
            reason="local_code_model_closed", expected_path=self.model_path
        )
        if receipt.requested_path_matches:
            self._set_readiness(ReadinessState.CONFIGURED, "unloaded")
        return receipt


_singleton: LocalCodeModel | None = None
_singleton_lock = threading.Lock()


def get_local_code_model() -> LocalCodeModel | None:
    """Return configured code lane; readiness is established by an async probe/use."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            candidate = LocalCodeModel()
            if not candidate.is_configured():
                logger.warning(
                    "LocalCodeModel is not configured: model=%s trust_root=%s",
                    candidate.model_path,
                    _trust_manifest_path(),
                )
                return None
            candidate._set_readiness(ReadinessState.CONFIGURED, "trust_root_present")
            _singleton = candidate
        return _singleton


__all__ = [
    "CodeGenerationReceipt",
    "CodeGenerationResult",
    "DeviceReleaseReceipt",
    "LocalCodeModel",
    "LocalCodeModelError",
    "ReadinessReceipt",
    "ReadinessState",
    "TrustedModelIdentity",
    "UnloadReceipt",
    "build_model_trust_manifest",
    "get_local_code_model",
    "max_code_tokens",
    "unload_local_code_model",
    "write_model_trust_manifest",
]
