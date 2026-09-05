"""The production steering vectors must actually load.

CP947 bound steering vectors to exact Cortex geometry by requiring a
``model_descriptor_sha256`` field on each ``.npz``. The vector files carried
``model_config_sha256`` instead, and the loader defaults the missing key to the
empty string before comparing it against a 64-hex digest. Every one of the
fifty production vectors was therefore skipped, unconditionally, for every
model, and ``load_production_vectors`` returned an empty dict.

Nothing noticed. An empty dict is a valid return, an injector built from it
installs hooks that inject nothing, and a campaign run against it reports a
steered condition that was never steered. The failure is silent by
construction, which is why it needs a test rather than a reader.

This is the test that would have failed the moment CP947 landed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.evaluation.steering_injection import load_production_vectors

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "training" / "vectors"

#: Dimensions the runtime asks for. Each must resolve to at least one layer.
PRODUCTION_DIMENSIONS = (
    "valence_positive",
    "arousal",
    "curiosity",
    "frustration",
    "energy",
)


def _stamped_digests() -> set[str]:
    """Every descriptor digest the installed vectors claim to be bound to."""
    digests: set[str] = set()
    for path in VECTORS.glob("*.npz"):
        try:
            with np.load(path, allow_pickle=True) as z:
                if "model_descriptor_sha256" in z:
                    digests.add(str(z["model_descriptor_sha256"].item()))
        except (OSError, ValueError, KeyError):
            continue
    return digests


@pytest.mark.skipif(not VECTORS.is_dir(), reason="no production vectors installed")
def test_installed_vectors_carry_a_descriptor_binding():
    """Refutes: a vector may sit in the production directory unloadable.

    A file the loader silently skips is worse than a missing one, because the
    directory looks populated.
    """
    stamped = _stamped_digests()
    assert stamped, (
        "no installed vector carries model_descriptor_sha256; "
        "load_production_vectors will return {} for every model. "
        "Re-extract with training/extract_steering_vectors.py."
    )


@pytest.mark.skipif(not VECTORS.is_dir(), reason="no production vectors installed")
def test_every_production_dimension_resolves_to_layers():
    """Refutes: the loader may return an empty dict and be believed.

    An injector built from {} installs hooks that inject nothing, and the arm
    still reports itself as steered.
    """
    stamped = _stamped_digests()
    if not stamped:
        pytest.skip("no stamped vectors; covered by the binding test")
    digest = sorted(stamped)[0]

    empty = []
    for dimension in PRODUCTION_DIMENSIONS:
        vectors = load_production_vectors(
            VECTORS, dimensions=(dimension,), model_descriptor_sha256=digest
        )
        if not vectors:
            empty.append(dimension)
    assert not empty, (
        f"dimensions resolve to no layers: {empty}. A steered arm built from "
        "these injects nothing while reporting that it steered."
    )


@pytest.mark.skipif(not VECTORS.is_dir(), reason="no production vectors installed")
def test_a_foreign_descriptor_loads_nothing():
    """Refutes: the binding is decorative.

    Vectors from one model inhabit an activation basis another model does not
    share, so a digest that matches nothing must return nothing. Without this
    the test above would pass on a loader that ignored the digest entirely.
    """
    foreign = "0" * 64
    vectors = load_production_vectors(
        VECTORS, dimensions=("valence_positive",), model_descriptor_sha256=foreign
    )
    assert vectors == {}


def test_a_malformed_digest_is_refused():
    """Refutes: an unidentified model may be steered.

    The identity is the whole point of the binding; accepting a blank one
    reintroduces the bug in a new place.
    """
    with pytest.raises(ValueError):
        load_production_vectors(VECTORS, model_descriptor_sha256="")


@pytest.mark.skipif(not VECTORS.is_dir(), reason="no production vectors installed")
def test_the_empty_result_is_reported(caplog):
    """Refutes: a load that matched nothing may pass in silence.

    Three days of every campaign injecting nothing produced no output at all.
    """
    import logging

    with caplog.at_level(logging.WARNING):
        load_production_vectors(
            VECTORS, dimensions=("valence_positive",),
            model_descriptor_sha256="1" * 64,
        )
    assert any("matched descriptor" in r.message or "No steering vectors" in r.message
               for r in caplog.records), "an empty load said nothing"
