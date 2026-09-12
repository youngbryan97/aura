"""The MLX client and worker source, wherever it now lives.

`core/brain/llm/mlx_worker.py` was one 12,500-line module, so a test that
wanted to assert a call site exists could read that one file. It is now
several: the surface quality gate and the surface repairs moved out when the
module went back under the size gate's ceiling, and more will follow.

A test that keeps reading the one file does not fail honestly. It reads a
shorter file, finds nothing, and reports a missing call site — which is
indistinguishable from the call site having been deleted. Four assertions in
this suite did exactly that.

So the source is asked for by what it is rather than by filename.
"""
from __future__ import annotations

import pathlib


def worker_modules() -> tuple[pathlib.Path, ...]:
    """Every module the worker process is built from, mlx_worker.py first."""
    here = pathlib.Path(__file__).resolve().parent.parent / "core" / "brain" / "llm"
    main = here / "mlx_worker.py"
    rest = sorted(p for p in here.glob("mlx_worker_*.py") if p.name != main.name)
    return (main, *rest)


def worker_source() -> str:
    """Their text, concatenated. The order is stable so slices stay stable."""
    return "\n".join(path.read_text(encoding="utf-8") for path in worker_modules())


def client_modules() -> tuple[pathlib.Path, ...]:
    """Every module MLXLocalClient is built from, mlx_client.py first.

    Named rather than globbed: `mlx_vision_client.py` and `mlx_vision_worker.py`
    sit in the same directory and are a different client. A glob that swept them
    in would make these assertions pass on text from the wrong lane.
    """
    here = pathlib.Path(__file__).resolve().parent.parent / "core" / "brain" / "llm"
    names = (
        "mlx_client.py",
        "mlx_client_worker_identity.py",
        "mlx_latent_reasoning.py",
        "mlx_unified_recurrent.py",
        "mlx_warmup_and_adapters.py",
    )
    return tuple(here / name for name in names)


def client_source() -> str:
    """Their text, concatenated. The order is stable so slices stay stable."""
    return "\n".join(path.read_text(encoding="utf-8") for path in client_modules())
