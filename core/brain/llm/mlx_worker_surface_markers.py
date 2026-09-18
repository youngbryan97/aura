"""Surface markers both halves of the worker read.

``mlx_worker`` imports the surface-quality helpers at module scope and
``mlx_worker_surface_quality`` imported one marker back the same way, so
the pair only resolved when the worker was imported first. On its own:

    ImportError: cannot import name '_BACKEND_SYMBOLIC_SURFACE_MARKERS'
    from partially initialized module

A test or a tool reaching for the surface-quality module alone hit that,
and the only way through was to import the worker first and say why — an
ordering that looks like style and is load-bearing.

Neither marker belongs to either module. They are compiled patterns with
no dependencies, read from both sides, so they live here and the edge
runs one way: both import this, this imports neither.

``_FUSION_MODEL_IDENTITY`` deliberately does NOT move here. It is mutable
module state on the worker — reassigned at runtime and monkeypatched by
``tests/test_fusion_gate_in_worker.py`` — which is why the two places that
read it import it *inside* the function that needs it, so each call sees
the current value. Hoisting those would bind the name once at import and
every later patch would stop being seen.
"""

from __future__ import annotations

import re

__all__ = [
    "_BACKEND_SYMBOLIC_SURFACE_MARKERS",
    "_CORRUPT_LANGUAGE_MARKERS",
]

#: Words the decoder emits when the language itself has come apart. Not a
#: vocabulary of bad answers — these are not words.
_CORRUPT_LANGUAGE_MARKERS = re.compile(
    r"\b(?:xublcate|ingediate|evocer)\b",
    re.IGNORECASE,
)

#: Internal symbols that belong to the machinery and never to a reply.
_BACKEND_SYMBOLIC_SURFACE_MARKERS = re.compile(
    r"\b(?:PROCEEDING|TOOL_ACTION|CONVERGE_UNION|CONFORMED_METHODS|"
    r"TACTICAL_ORGANIZE|UI_SHUTDOWN_OR_DURATIVE_TIMEOUT|"
    r"MySelfEpsilon|CanonicalStabilityAnchor|currentInferenceProblem|"
    r"fieldOfPlay|INTRUSTION_DETECTED|INTRUSION_DETECTED|"
    r"ExistenceHash)\b"
)
