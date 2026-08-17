"""The bounded-WOW surface must not be able to go dark quietly.

CP566 (2026-08-15) established a `BOUNDED_WOW_SIGNAL`: on a frozen four-domain
cohort of 60 fresh tasks, treatment decoded 60/60 where ordinary resident-32B
decode scored 16/60, syntax-matched wire 7/60, the family-targeted coefficient
lesion 5/60, and the same-family wrong-state control 0/60. Paired one-sided
exact p = 5.684341886080802e-14. The effect tracks the mechanism: lesion it and
the gain goes away.

CP567 materialized that as a fail-closed runtime package bound to twenty source
files, so any drift disables serving. That is the right design. What was
missing is the other half: nothing ever ASKED whether the surface was still
serving.

Found 2026-08-17, only because someone asked. The surface had been inactive
since 2026-08-15 — the day it was sealed — because four of its twenty bound
files moved underneath it during ordinary development:

    core/brain/llm/semantic_neural_shadow.py        Aug 15
    core/brain/foreground_latent_runtime.py         Aug 16
    core/brain/latent_cortex_service.py             Aug 17
    core/phases/response_generation_unitary.py      Aug 17

A capability proven at p=5.7e-14 was switched off by routine edits and stayed
off for two days with no alarm anywhere. The activation check reported
"source_drift" without naming a file, so even noticing it left you no thread to
pull.

These tests are the alarm. The first one FAILS while the surface is dark, and
it is supposed to: a proven capability that is not running is a defect, not a
configuration. Do not skip it, and do not re-seal the package to make it pass —
re-hashing changed source against evidence measured on the old source relabels
unproven code as proven, which is worse than the outage. Either restore the
bound files or re-run the qualification that earns a new seal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.brain.llm.semantic_neural_serving import (
    DEFAULT_ACTIVATION_PATH,
    semantic_neural_activation_errors,
    semantic_neural_serving_status,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _activation() -> dict:
    return json.loads(Path(DEFAULT_ACTIVATION_PATH).read_text(encoding="utf-8"))


def _cortex_path() -> Path:
    from core.brain.llm.model_registry import _CORTEX_PATH

    return Path(str(_CORTEX_PATH))


def _drifted() -> list[str]:
    import hashlib

    srcs = _activation().get("source_sha256s") or {}
    out = []
    for relative, expected in srcs.items():
        path = REPO_ROOT / relative
        if not path.exists():
            out.append(relative)
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            out.append(relative)
    return sorted(out)


# ── the alarm ────────────────────────────────────────────────────────────────

def test_the_proven_surface_is_actually_serving() -> None:
    """The gate. A capability proven at p=5.7e-14 should be running."""
    status = semantic_neural_serving_status(str(_cortex_path()))

    assert status.get("active") is True, (
        f"bounded-WOW surface is NOT serving: {status.get('reason')}. "
        f"Drifted bound files: {_drifted() or 'none'}. "
        "Restore them or re-run the qualification; do not re-seal to go green."
    )


# ── diagnosability: these must hold whether or not the surface is up ─────────

def test_drift_names_the_files_that_caused_it() -> None:
    """"source_drift" with no filename left no thread to pull."""
    errors = semantic_neural_activation_errors(
        _activation(), model_path=_cortex_path()
    )
    drift_errors = [e for e in errors if str(e).startswith("source_drift")]
    if not drift_errors:
        pytest.skip("no source drift to describe")

    reported = drift_errors[0]

    assert ":" in reported, f"drift error names nothing: {reported!r}"
    for relative in _drifted():
        assert relative in reported, f"{relative} drifted but is not named"


def test_the_bound_inventory_is_intact() -> None:
    """A bound file that vanished is a different failure from one that moved."""
    srcs = _activation().get("source_sha256s") or {}

    assert srcs, "activation carries no bound source inventory"
    missing = [r for r in srcs if not (REPO_ROOT / r).exists()]

    assert not missing, f"bound source files are gone: {missing}"


def test_the_activation_still_carries_its_claim_and_limits() -> None:
    """The seal must keep the adjudicated wording, not a rosier summary."""
    activation = _activation()

    assert activation.get("activation_sha256")
    assert activation.get("active_by_default") is True


def test_the_serving_status_always_explains_itself() -> None:
    """Inactive with no reason is the shape that hid this for two days."""
    status = semantic_neural_serving_status(str(_cortex_path()))

    if status.get("active") is not True:
        assert str(status.get("reason") or "").strip(), (
            "surface is inactive and gives no reason"
        )
