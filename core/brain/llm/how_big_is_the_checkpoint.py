"""How many gigabytes a checkpoint needs, and how much the host has.

Lifted out of ``mlx_client`` whole. Admission asks two questions before it
loads anything — how big is this artifact, and is there room — and the answers
came from nine functions and four constants scattered through a module that is
otherwise about talking to a worker process.

The rules they encode are the interesting part, and each of them was measured
rather than assumed: only weight files count toward a footprint, because
tokenizer caches and training logs are not loaded; the walk stops at two levels
and 512 files, because a deep tree once made it unbounded; and a measured
artifact beats an environment override, because the override is a guess and
the bytes on disk are not.

The two names it reads back out of ``mlx_client`` are imported inside the
functions that use them. That module imports this one, so a top-level import
would be a cycle, and the call-time form is also what keeps a test's patch of
them visible from here.
"""

from __future__ import annotations

import logging
import math
import os
import pathlib
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger("LLM.MLX")

__all__ = [
    "_MAX_ARTIFACT_FILES_SCANNED",
    "_MAX_ARTIFACT_SCAN_DEPTH",
    "_PATH_SIZE_CACHE",
    "_WEIGHT_FILE_SUFFIXES",
    "_env_projected_footprint_gb",
    "_measured_model_footprint_gb",
    "_model_load_min_available_gb",
    "_path_size_gb",
    "_projected_footprint_from_artifact_gb",
    "_projected_model_footprint_gb",
    "_weight_files",
]


def _model_load_min_available_gb(model_path: str) -> float:
    from core.brain.llm.mlx_client import (
        _finite_env_float,
        _measured_model_footprint_gb,
        _model_matches_class,
    )

    def _env_float(name: str, default: float) -> float:
        return _finite_env_float(name, default, minimum=0.0)

    try:
        total_gb = float(psutil.virtual_memory().total) / float(1024**3)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, psutil.Error) as exc:
        logger.debug("Total system memory unreadable, reporting 0GB: %s", exc)
        total_gb = 0.0
    if _model_matches_class(model_path, ("72b", "solver")):
        default = 52.0 if 0.0 < total_gb < 96.0 else 34.0
        return _env_float("AURA_MLX_72B_LOAD_MIN_AVAILABLE_GB", default)
    if _model_matches_class(model_path, ("32b", "cortex", "zenith")):
        # Derive the requirement from the model actually on disk rather than a
        # constant that happens to be wrong for it. Measured 2026-07-25: the
        # resident 32B is 17.2GB on disk and the flat 24.0 gate refused it on a
        # host sitting at 20.4GB available — a 6.8GB margin over true need, and
        # the cortex starved through six deaths in one run because of it.
        #
        # weights x 1.20 + 1GB covers KV cache and activations for a normal
        # context with room to spare. The flat default remains the CEILING, so
        # this can only ever relax toward the real footprint, never tighten
        # past a deliberate operator setting — and it floors at 16GB so a
        # mis-sized or unreadable model directory cannot wave a load through.
        default = 24.0 if total_gb >= 60.0 else 22.0
        measured = _measured_model_footprint_gb(model_path)
        if measured is not None:
            derived = measured * 1.20 + 1.0
            default = max(16.0, min(default, derived))
        return _env_float("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", default)
    return _env_float("AURA_MLX_LOAD_MIN_AVAILABLE_GB", 8.0)


def _measured_model_footprint_gb(model_path: Any) -> float | None:
    """Total size of the model directory in GB, or None if unreadable.

    Returns None on anything surprising — a missing directory, a permission
    error, an implausible size — so the caller keeps its conservative default.
    """
    text = str(model_path or "").strip()
    if not text:
        return None  # Path("") is the CWD, which is a real directory
    try:
        root = pathlib.Path(text)
        if not root.is_dir():
            return None
        total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Model footprint unreadable: %s", exc)
        return None
    gb = total / float(1024**3)
    if not (1.0 < gb < 200.0):
        return None
    return gb


def _env_projected_footprint_gb(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in {"", "auto", "detect", "detected"}:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        logger.debug("Projected footprint override is not a number, ignoring it: %s", exc)
        return None
    # A zero/negative override makes a real multi-GB worker appear free and
    # NaN/inf poisons every downstream admission sum — ignore such overrides.
    if not math.isfinite(value) or value <= 0.0:
        logger.warning("Ignoring invalid projected-footprint override %s=%r.", name, raw)
        return None
    return value


# Model artifacts are immutable while the runtime holds them (fusion
# publishes a NEW directory), so their size is computed once per
# (path, mtime) and reused. Uncached, this rglob+stat walk ran on the
# EVENT LOOP inside model-load admission while 20GB of safetensors reads
# saturated the disk — the 5.5-8.6s loop stalls captured in
# data/error_logs/stalls/stall_1784673149 / stall_1784675621 bottom out
# exactly here (pathlib stat under _projected_footprint_from_artifact_gb).
#: Extensions that hold model weights. Everything else in a checkpoint
#: directory — tokenizer caches, logs, receipts, adapters, temp files — is not
#: what gets loaded into memory, so it is not part of the footprint that RAM
#: admission is computed from (CP126 50d8ed03).
_WEIGHT_FILE_SUFFIXES = frozenset(
    {".safetensors", ".bin", ".gguf", ".npz", ".pt", ".pth"}
)
#: Depth and count ceilings for the artifact scan. A checkpoint's weights sit
#: at the top level or one directory below it; a scan that follows an arbitrary
#: tree is unbounded work on an admission path.
_MAX_ARTIFACT_SCAN_DEPTH = 2
_MAX_ARTIFACT_FILES_SCANNED = 512


def _weight_files(root: Path):
    """Weight files within the artifact, bounded in depth."""
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir():
                    if depth + 1 < _MAX_ARTIFACT_SCAN_DEPTH:
                        stack.append((entry, depth + 1))
                    continue
                if entry.suffix.lower() in _WEIGHT_FILE_SUFFIXES:
                    yield entry
            except OSError:
                continue


_PATH_SIZE_CACHE: dict[tuple[str, int], float] = {}


def _path_size_gb(model_path: str) -> float:
    path = Path(str(model_path or "")).expanduser()
    try:
        if path.is_file():
            return float(path.stat().st_size) / float(1024**3)
        if not path.is_dir():
            return 0.0
        cache_key = (str(path), path.stat().st_mtime_ns)
        cached = _PATH_SIZE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        # CP126 50d8ed03: this walked EVERY descendant and counted every file,
        # so tokenizer caches, training logs, adapters, receipts and temporary
        # artifacts inflated the "model footprint" that RAM admission is
        # computed from — and a directory with a deep subtree made the walk
        # unbounded. What the footprint means is the weights that get loaded,
        # so only weight files count, only the top two levels are walked, and
        # the walk stops at a file ceiling rather than running as long as the
        # tree is deep.
        total = 0
        scanned = 0
        for child in _weight_files(path):
            try:
                total += child.stat().st_size
            except OSError:
                continue
            scanned += 1
            if scanned >= _MAX_ARTIFACT_FILES_SCANNED:
                logger.debug(
                    "Artifact size scan for %s stopped at %d files.",
                    path,
                    scanned,
                )
                break
        size_gb = float(total) / float(1024**3)
        if len(_PATH_SIZE_CACHE) > 64:
            _PATH_SIZE_CACHE.clear()
        _PATH_SIZE_CACHE[cache_key] = size_gb
        return size_gb
    except OSError as exc:
        logger.debug("Path size unreadable, reporting 0GB: %s", exc)
        return 0.0


def _projected_footprint_from_artifact_gb(model_path: str, *, fallback_gb: float) -> float:
    """Estimate live model footprint from the local artifact when possible.

    The launcher previously used one static 32B projection for every artifact.
    That is too blunt for Aura: the active fused 4-bit model is materially
    smaller than the old 8-bit base artifact, while a genuine 8-bit path should
    still be treated as too expensive for a tight desktop process cap.
    """
    from core.brain.llm.mlx_client import _model_matches_class, _path_size_gb

    size_gb = _path_size_gb(model_path)
    if size_gb <= 0.0:
        return fallback_gb
    if _model_matches_class(model_path, ("72b", "solver")):
        overhead = max(4.0, size_gb * 0.14)
    elif _model_matches_class(model_path, ("32b", "cortex", "zenith", "aura-32b")):
        overhead = max(3.0, size_gb * 0.30)
    else:
        overhead = max(1.0, size_gb * 0.20)
    return max(1.0, size_gb + overhead)


def _projected_model_footprint_gb(model_path: str) -> float:
    from core.brain.llm.mlx_client import (
        _env_projected_footprint_gb,
        _finite_env_float,
        _model_is_quantized,
        _model_matches_class,
        _projected_footprint_from_artifact_gb,
    )

    def _env_float(name: str, default: float) -> float:
        return _finite_env_float(name, default, minimum=0.0)

    if _model_matches_class(model_path, ("72b", "solver")):
        override = _env_projected_footprint_gb("AURA_MLX_72B_PROJECTED_FOOTPRINT_GB")
        if override is not None:
            return override
        return _projected_footprint_from_artifact_gb(model_path, fallback_gb=41.0)
    if _model_matches_class(model_path, ("32b", "cortex", "zenith")):
        override = _env_projected_footprint_gb("AURA_MLX_32B_PROJECTED_FOOTPRINT_GB")
        if override is not None:
            return override
        # Quantization changes the footprint by more than a third, and it is
        # a property of the checkpoint, not of its directory name. Ask the
        # artifact; fall back to the name only when it cannot be read.
        quantized = _model_is_quantized(model_path)
        default = 20.0 if quantized else 35.0
        return _projected_footprint_from_artifact_gb(model_path, fallback_gb=default)
    if _model_matches_class(model_path, ("14b",)):
        return _env_float("AURA_MLX_14B_PROJECTED_FOOTPRINT_GB", 10.0)
    if _model_matches_class(model_path, ("7b",)):
        return _env_float("AURA_MLX_7B_PROJECTED_FOOTPRINT_GB", 5.0)
    return _env_float("AURA_MLX_PROJECTED_FOOTPRINT_GB", 4.0)
