"""Refuse known model-backend imports inside a classical-only Python probe.

This process-local boundary complements model-worker admission, CPU vision
requests and a measured software browser driver. It cannot police native
code, external services or arbitrary child interpreters. Install it before
starting probe work; a backend already loaded makes admission fail.
"""
from __future__ import annotations

import importlib.abc
import sys
from collections import deque
from contextlib import contextmanager

from core.verify import invariant

MODEL_BACKENDS = frozenset({'mlx', 'mlx_lm', 'mlx_vlm', 'torch', 'tensorflow', 'jax', 'jaxlib',
                            'transformers', 'sentence_transformers', 'llama_cpp', 'ctranslate2',
                            'onnxruntime', 'coremltools', 'openvino', 'tensorrt', 'cupy', 'vllm'})


def _blocked_root(name: str) -> str | None:
    root = name.partition('.')[0]
    return root if root in MODEL_BACKENDS else None


def _loaded_backends() -> tuple[str, ...]:
    return tuple(sorted({root for name in tuple(sys.modules) if (root := _blocked_root(name)) is not None}))


class ClassicalImportBoundary(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self.attempts: deque[str] = deque(maxlen=64)

    def find_spec(self, fullname: str, path=None, target=None):
        if _blocked_root(fullname) is not None:
            self.attempts.append(fullname[:256])
            raise ImportError(f'model backend import refused by classical compute boundary: {fullname}')
        return None

    @property
    def active(self) -> bool:
        return any(finder is self for finder in sys.meta_path)

    def report(self) -> dict:
        return {'scope': 'current_python_interpreter', 'active': self.active,
                'blocked_backends': sorted(MODEL_BACKENDS), 'loaded_backends': list(_loaded_backends()),
                'refused_imports': list(self.attempts)}

    def require_intact(self) -> None:
        if not self.active:
            raise RuntimeError('classical compute import boundary is not installed')
        loaded = _loaded_backends()
        if loaded:
            raise RuntimeError('model backends already loaded in classical probe: ' + ', '.join(loaded))


def current_classical_boundary() -> ClassicalImportBoundary | None:
    return next((finder for finder in sys.meta_path if isinstance(finder, ClassicalImportBoundary)), None)


@contextmanager
def classical_compute_boundary():
    """Scope a probe's import policy without changing another running process."""
    loaded = _loaded_backends()
    if loaded:
        raise RuntimeError('cannot admit a classical probe with loaded model backends: ' + ', '.join(loaded))
    boundary = ClassicalImportBoundary()
    sys.meta_path.insert(0, boundary)
    try:
        boundary.require_intact()
        yield boundary
    finally:
        try:
            boundary.require_intact()
        finally:
            sys.meta_path[:] = [finder for finder in sys.meta_path if finder is not boundary]


@invariant('runtime.classical_probe_recognizes_backend_import_roots', scope='runtime',
           owner='core/runtime/classical_compute_boundary.py', observational=False)
def _classical_import_invariant() -> tuple:
    assert _blocked_root('mlx.core') == 'mlx' and _blocked_root('torch.nn') == 'torch'
    assert _blocked_root('numpy.linalg') is None and _blocked_root('torch_support') is None
    assert ClassicalImportBoundary().report()['scope'] == 'current_python_interpreter'
    return ()
