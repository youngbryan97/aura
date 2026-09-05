"""Model-lane-owned Auto-AVSR visual-only decoder.

The upstream Auto-AVSR code and checkpoint live in Aura's model store rather
than this repository. Both are integrity-checked before import/load. PyTorch's
``weights_only`` loader is mandatory. The 250M-parameter model is held under an
in-process model-lane heartbeat and every inference drains before cancellation
can release or zero its ephemeral mouth-crop input.
"""
from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.perception.visual_speech import BackendPrediction
from core.runtime.state_ownership import shared_asset_root, state_root

logger = logging.getLogger("Aura.VisualSpeech.AutoAVSR")

_DEFAULT_ROOT = shared_asset_root() / "models" / "visual_speech" / "auto_avsr"
_CHECKPOINT_SHA256 = "fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb"
_RUNTIME_SHA256 = "16c00029964c56771bb3e7bf511c152204dfcb50f0837022575b8b97b93bceab"
_RUNTIME_COMMIT = "182b62837773ab01052d4ac21ef1d2203ea7d267"
_CHECKPOINT_BYTES = 1_001_892_616


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    byte_count = 0
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or ".git" in path.parts
            or "__pycache__" in path.parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        relative = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        count += 1
        byte_count += len(data)
    return digest.hexdigest(), count, byte_count


def _require_runtime_symbol(symbol: Any, runtime_root: Path) -> None:
    module = sys.modules.get(str(getattr(symbol, "__module__", "")))
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError("Auto-AVSR imported symbol has no source provenance")
    try:
        Path(module_file).resolve().relative_to(runtime_root.resolve())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Auto-AVSR imported symbol escaped verified runtime") from exc


@dataclass(frozen=True)
class AutoAVSRConfig:
    model_root: Path = _DEFAULT_ROOT
    checkpoint_name: str = "vsr_trlrs2lrs3vox2avsp_base.pth"
    device: str = "cpu"
    beam_size: int = 10
    ctc_weight: float = 0.1
    request_gb: float = 4.0
    inference_timeout_s: float = 180.0
    torch_threads: int = 4

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "mps"}:
            raise ValueError("Auto-AVSR device must be cpu or mps")
        if not 1 <= self.beam_size <= 40:
            raise ValueError("Auto-AVSR beam size must be between 1 and 40")
        if not 0.0 <= self.ctc_weight <= 1.0:
            raise ValueError("Auto-AVSR CTC weight must be between 0 and 1")
        if not 1.0 <= self.request_gb <= 16.0:
            raise ValueError("Auto-AVSR memory request must be between 1 and 16 GiB")
        if not 1 <= self.torch_threads <= 32:
            raise ValueError("Auto-AVSR torch thread count must be between 1 and 32")

    @property
    def checkpoint_path(self) -> Path:
        return self.model_root / self.checkpoint_name

    @property
    def runtime_path(self) -> Path:
        return self.model_root / "runtime"

    @property
    def manifest_path(self) -> Path:
        return self.model_root / "manifest.json"


class _AutoAVSRRuntime:
    def __init__(self, config: AutoAVSRConfig) -> None:
        self.config = config
        self.model: Any = None
        self.beam_search: Any = None
        self.token_list: list[str] = []
        self.torch: Any = None
        self.loaded = False
        self.model_id = f"auto-avsr-vsr-20.3wer:{_CHECKPOINT_SHA256[:16]}"

    def load(self) -> None:
        if self.loaded:
            return
        self._verify_integrity()
        runtime_path = str(self.config.runtime_path)
        added_runtime_path = runtime_path not in sys.path
        if added_runtime_path:
            sys.path.insert(0, runtime_path)
        try:
            import torch
            from espnet.nets.batch_beam_search import BatchBeamSearch
            from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
            from espnet.nets.scorers.length_bonus import LengthBonus
        finally:
            if added_runtime_path:
                try:
                    sys.path.remove(runtime_path)
                except ValueError:
                    pass
        for symbol in (BatchBeamSearch, E2E, LengthBonus):
            _require_runtime_symbol(symbol, self.config.runtime_path)

        torch.set_num_threads(self.config.torch_threads)
        units_path = self.config.runtime_path / "spm" / "unigram" / "unigram5000_units.txt"
        units = units_path.read_text(encoding="utf-8").splitlines()
        self.token_list = ["<blank>", *[line.split()[0] for line in units], "<eos>"]
        state = torch.load(
            self.config.checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(state, dict) or not state:
            raise RuntimeError("Auto-AVSR checkpoint did not contain a state dictionary")
        model = E2E(
            len(self.token_list),
            "video",
            ctc_weight=self.config.ctc_weight,
        )
        model.load_state_dict(state, strict=True)
        del state
        model.eval()
        model.to(self.config.device)
        scorers = model.scorers()
        scorers["lm"] = None
        scorers["length_bonus"] = LengthBonus(len(self.token_list))
        weights = {
            "decoder": 1.0 - self.config.ctc_weight,
            "ctc": self.config.ctc_weight,
            "lm": 0.0,
            "length_bonus": 0.0,
        }
        self.beam_search = BatchBeamSearch(
            beam_size=self.config.beam_size,
            vocab_size=len(self.token_list),
            weights=weights,
            scorers=scorers,
            sos=model.odim - 1,
            eos=model.odim - 1,
            token_list=self.token_list,
            pre_beam_score_key=None,
        )
        self.model = model
        self.torch = torch
        self.loaded = True

    def _verify_integrity(self) -> None:
        config = self.config
        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise RuntimeError("Auto-AVSR manifest is malformed")
        expected_manifest = {
            "checkpoint_sha256": _CHECKPOINT_SHA256,
            "checkpoint_bytes": _CHECKPOINT_BYTES,
            "code_commit": _RUNTIME_COMMIT,
            "code_source_sha256": _RUNTIME_SHA256,
        }
        for key, value in expected_manifest.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"Auto-AVSR manifest mismatch:{key}")
        if config.checkpoint_path.stat().st_size != _CHECKPOINT_BYTES:
            raise RuntimeError("Auto-AVSR checkpoint size mismatch")
        if _sha256_file(config.checkpoint_path) != _CHECKPOINT_SHA256:
            raise RuntimeError("Auto-AVSR checkpoint digest mismatch")
        runtime_sha, runtime_files, runtime_bytes = _runtime_digest(config.runtime_path)
        if runtime_sha != _RUNTIME_SHA256:
            raise RuntimeError("Auto-AVSR runtime source digest mismatch")
        if manifest.get("code_source_files") != runtime_files:
            raise RuntimeError("Auto-AVSR runtime file-count mismatch")
        if manifest.get("code_source_bytes") != runtime_bytes:
            raise RuntimeError("Auto-AVSR runtime byte-count mismatch")

    def infer(self, mouth_crops: NDArray[np.uint8]) -> BackendPrediction:
        if not self.loaded or self.model is None or self.beam_search is None:
            raise RuntimeError("Auto-AVSR runtime is not loaded")
        if mouth_crops.dtype != np.uint8 or mouth_crops.ndim != 4:
            raise ValueError("Auto-AVSR input must be uint8 [T,H,W,C]")
        if not 1 <= mouth_crops.shape[0] <= 500:
            raise ValueError("Auto-AVSR frame count is outside bounds")
        if mouth_crops.shape[1:3] != (96, 96) or mouth_crops.shape[3] not in (1, 3):
            raise ValueError("Auto-AVSR mouth crops must be 96x96 with one or three channels")

        torch = self.torch
        tensor = torch.from_numpy(mouth_crops).to(dtype=torch.float32) / 255.0
        if mouth_crops.shape[3] == 3:
            tensor = (
                0.2989 * tensor[..., 0]
                + 0.5870 * tensor[..., 1]
                + 0.1140 * tensor[..., 2]
            ).unsqueeze(1)
        else:
            tensor = tensor.permute(0, 3, 1, 2)
        tensor = tensor[:, :, 4:92, 4:92]
        tensor = (tensor - 0.421) / 0.165
        tensor = tensor.to(self.config.device)

        with torch.inference_mode():
            encoded = self.model.frontend(tensor.unsqueeze(0))
            encoded = self.model.proj_encoder(encoded)
            encoded, _mask = self.model.encoder(encoded, None)
            hypotheses = self.beam_search(encoded.squeeze(0))
        if not hypotheses:
            return BackendPrediction(
                transcript="",
                confidence=None,
                calibrated=False,
                backend="auto_avsr",
                model_id=self.model_id,
            )
        alternatives: list[tuple[str, float]] = []
        for hypothesis in hypotheses[:3]:
            token_ids = [int(value) for value in hypothesis.yseq[1:].tolist()]
            transcript = self._tokens_to_text(token_ids)
            score = float(hypothesis.score)
            alternatives.append((transcript, score))
        transcript, score = alternatives[0]
        return BackendPrediction(
            transcript=transcript,
            confidence=None,
            calibrated=False,
            backend="auto_avsr",
            model_id=self.model_id,
            score=score if math.isfinite(score) else None,
            alternatives=tuple(alternatives),
        )

    def _tokens_to_text(self, token_ids: list[int]) -> str:
        pieces = [
            self.token_list[token_id]
            for token_id in token_ids
            if 0 <= token_id < len(self.token_list)
        ]
        return " ".join(
            "".join(pieces).replace("<eos>", "").replace("\u2581", " ").split()
        )

    def clear(self) -> None:
        self.model = None
        self.beam_search = None
        self.token_list = []
        self.torch = None
        self.loaded = False
        gc.collect()


class AutoAVSRBackend:
    """Visual-only Auto-AVSR backend with model-lane ownership."""

    def __init__(self, config: AutoAVSRConfig | None = None) -> None:
        self.config = config or AutoAVSRConfig()
        self._runtime: _AutoAVSRRuntime | None = None
        self._lane_lease: Any = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._last_error = ""
        self._loads = 0
        self._inferences = 0

    def available(self) -> tuple[bool, str]:
        config = self.config
        try:
            if not config.manifest_path.is_file():
                return False, "manifest_missing"
            manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                return False, "manifest_malformed"
            for key, expected in (
                ("checkpoint_sha256", _CHECKPOINT_SHA256),
                ("checkpoint_bytes", _CHECKPOINT_BYTES),
                ("code_commit", _RUNTIME_COMMIT),
                ("code_source_sha256", _RUNTIME_SHA256),
            ):
                if manifest.get(key) != expected:
                    return False, f"manifest_mismatch:{key}"
            if not config.checkpoint_path.is_file():
                return False, "checkpoint_missing"
            if config.checkpoint_path.stat().st_size != _CHECKPOINT_BYTES:
                return False, "checkpoint_size_mismatch"
            if not config.runtime_path.is_dir():
                return False, "runtime_missing"
            return True, "ready"
        except (OSError, TypeError, ValueError) as exc:
            return False, f"filesystem_error:{type(exc).__name__}"

    async def infer(
        self,
        mouth_crops: NDArray[np.uint8],
        *,
        fps: float,
    ) -> BackendPrediction:
        if not math.isfinite(fps) or not 10.0 <= fps <= 60.0:
            raise ValueError("Auto-AVSR input fps must be between 10 and 60")
        await self._ensure_loaded()
        async with self._inference_lock:
            runtime = self._runtime
            lease = self._lane_lease
            if runtime is None or lease is None:
                raise RuntimeError("Auto-AVSR model lane is unavailable")
            if not await lease.set_preemptible(False):
                raise RuntimeError("Auto-AVSR model lane fencing was lost")
            try:
                from core.runtime.model_lane_control import run_owned_model_thread_call

                prediction = await run_owned_model_thread_call(
                    lambda: runtime.infer(mouth_crops),
                    operation_name="auto-avsr-visual-speech-inference",
                    timeout_s=self.config.inference_timeout_s,
                )
                if not isinstance(prediction, BackendPrediction):
                    raise RuntimeError("Auto-AVSR decoder returned an invalid prediction")
                self._inferences += 1
                return prediction
            finally:
                if not await lease.set_preemptible(True):
                    self._last_error = "model_lane_preemptibility_restore_failed"

    async def _ensure_loaded(self) -> None:
        if (
            self._runtime is not None
            and self._runtime.loaded
            and self._lane_lease is not None
        ):
            return
        async with self._load_lock:
            if (
                self._runtime is not None
                and self._runtime.loaded
                and self._lane_lease is not None
            ):
                return
            available, reason = self.available()
            if not available:
                raise RuntimeError(f"Auto-AVSR unavailable:{reason}")
            from core.runtime.model_lane_control import (
                acquire_in_process_model_lane,
                run_owned_model_thread_call,
            )

            lease = await acquire_in_process_model_lane(
                owner_id="auto-avsr-visual-speech",
                model_path=str(self.config.checkpoint_path),
                purpose="serve",
                request_gb=self.config.request_gb,
                priority=60,
                preemptible=False,
                evict=self._evict_for_lane,
                compensate=self._compensate_lane,
                metadata={
                    "provider": "auto_avsr",
                    "modality": "visual_speech",
                    "video_only": True,
                    "activation_state": "loading",
                },
            )
            runtime = _AutoAVSRRuntime(self.config)
            try:
                await run_owned_model_thread_call(
                    runtime.load,
                    operation_name="auto-avsr-visual-speech-load",
                    timeout_s=self.config.inference_timeout_s,
                )
                if not await lease.set_preemptible(True):
                    raise RuntimeError("Auto-AVSR model lane activation fence was lost")
            except asyncio.CancelledError:
                await run_owned_model_thread_call(
                    runtime.clear,
                    operation_name="auto-avsr-cancelled-load-clear",
                )
                await lease.release(reason="auto_avsr_load_cancelled")
                raise
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                self._last_error = f"{type(exc).__name__}:{str(exc)[:240]}"
                await run_owned_model_thread_call(
                    runtime.clear,
                    operation_name="auto-avsr-failed-load-clear",
                )
                await lease.release(reason="auto_avsr_load_failed")
                raise
            self._runtime = runtime
            self._lane_lease = lease
            self._loads += 1
            self._last_error = ""

    async def unload(self, *, reason: str = "auto_avsr_unloaded") -> bool:
        async with self._load_lock:
            async with self._inference_lock:
                runtime, self._runtime = self._runtime, None
                lease, self._lane_lease = self._lane_lease, None
                if runtime is not None:
                    from core.runtime.model_lane_control import run_owned_model_thread_call

                    await run_owned_model_thread_call(
                        runtime.clear,
                        operation_name="auto-avsr-unload-clear",
                    )
                if lease is not None:
                    await lease.release(reason=reason)
                return runtime is not None or lease is not None

    async def _evict_for_lane(self, _owner: Any, reason: str) -> bool:
        if self._inference_lock.locked():
            return False
        return await self.unload(reason=f"lane_eviction:{reason}")

    async def _compensate_lane(self, _owner: Any, _reason: str) -> bool:
        try:
            await self._ensure_loaded()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            return False
        return self._runtime is not None and self._runtime.loaded and self._lane_lease is not None

    def get_status(self) -> dict[str, object]:
        available, reason = self.available()
        runtime = self._runtime
        return {
            "available": available,
            "availability_reason": reason,
            "loaded": bool(runtime is not None and runtime.loaded),
            "integrity_verified": bool(runtime is not None and runtime.loaded),
            "lane_owned": self._lane_lease is not None,
            "loads": self._loads,
            "inferences": self._inferences,
            "last_error": self._last_error,
            "backend": "auto_avsr",
            "modality": "video_only",
            "checkpoint_sha256": _CHECKPOINT_SHA256,
            "runtime_sha256": _RUNTIME_SHA256,
            "upstream_commit": _RUNTIME_COMMIT,
            "reported_lrs3_wer_percent": 20.3,
            "confidence_calibrated": False,
        }


_BACKEND: AutoAVSRBackend | None = None


def get_auto_avsr_backend() -> AutoAVSRBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = AutoAVSRBackend()
    return _BACKEND


__all__ = ["AutoAVSRBackend", "AutoAVSRConfig", "get_auto_avsr_backend"]
