"""Either half of the worker loads first, and the markers belong to neither.

``mlx_worker`` imports the surface-quality helpers at module scope and
``mlx_worker_surface_quality`` imported one marker back the same way, so
the pair only resolved when the worker was imported first. On its own:

    ImportError: cannot import name '_BACKEND_SYMBOLIC_SURFACE_MARKERS'
    from partially initialized module

Anything reaching for the surface-quality module alone hit that, and the
way through was to import the worker first and leave a comment saying
why — an ordering that reads as style and is load-bearing.
"""

from __future__ import annotations

import subprocess
import sys

PYTHON = sys.executable


def _cold_import(module: str) -> subprocess.CompletedProcess:
    """Import in a fresh interpreter, where the cycle actually bites."""

    return subprocess.run(
        [PYTHON, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_the_surface_quality_half_imports_first():
    done = _cold_import("core.brain.llm.mlx_worker_surface_quality")
    assert done.returncode == 0, done.stderr[-2000:]


def test_the_worker_half_imports_first():
    done = _cold_import("core.brain.llm.mlx_worker")
    assert done.returncode == 0, done.stderr[-2000:]


def test_the_markers_module_depends_on_neither():
    import ast
    import inspect

    from core.brain.llm import mlx_worker_surface_markers

    tree = ast.parse(inspect.getsource(mlx_worker_surface_markers))
    reached = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            reached.add(node.module)
        elif isinstance(node, ast.Import):
            reached.update(a.name for a in node.names)
    assert not any("mlx_worker" in name for name in reached), (
        f"the shared markers import back into the pair they exist to separate: {reached}"
    )


def test_neither_half_defines_a_marker_the_other_reads():
    import inspect

    from core.brain.llm import mlx_worker, mlx_worker_surface_quality

    for module in (mlx_worker, mlx_worker_surface_quality):
        source = inspect.getsource(module)
        for marker in ("_CORRUPT_LANGUAGE_MARKERS", "_BACKEND_SYMBOLIC_SURFACE_MARKERS"):
            assert f"{marker} = re.compile" not in source, (
                f"{module.__name__} defines {marker}; two definitions is how the "
                "cycle comes back"
            )


def test_both_halves_see_the_same_marker_objects():
    from core.brain.llm import (
        mlx_worker,
        mlx_worker_surface_markers,
        mlx_worker_surface_quality,
    )

    # The worker never read the corrupt-language pattern itself — it takes
    # _contains_corrupted_language from the other half — so it does not
    # import it at all now.
    assert (
        mlx_worker_surface_quality._CORRUPT_LANGUAGE_MARKERS
        is mlx_worker_surface_markers._CORRUPT_LANGUAGE_MARKERS
    )
    assert (
        mlx_worker._BACKEND_SYMBOLIC_SURFACE_MARKERS
        is mlx_worker_surface_markers._BACKEND_SYMBOLIC_SURFACE_MARKERS
    )


def test_the_fusion_identity_is_still_read_inside_the_function():
    """It is mutable module state, so it must NOT be hoisted.

    ``_FUSION_MODEL_IDENTITY`` is reassigned at runtime and monkeypatched
    by tests/test_fusion_gate_in_worker.py. Binding it at import would
    freeze the value and every later patch would stop being seen — which
    is why it stayed on the worker while the constants moved.
    """

    import inspect

    from core.brain.llm import mlx_worker_surface_quality

    source = inspect.getsource(mlx_worker_surface_quality)
    module_scope = source[: source.index("\ndef ")]
    assert "_FUSION_MODEL_IDENTITY" not in module_scope
    assert "from .mlx_worker import _FUSION_MODEL_IDENTITY" in source
