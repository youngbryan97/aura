"""A bundle's contract has to describe the capture that made it.

The extraction contract is hashed and the migration authority binds the vectors
to that hash, so a contract naming the wrong capture is a false record however
correct every digest in it is. It said
`difference_of_means_last_token_hidden_state` and
`tools/capture_27b_steering_vectors.py` as constants, which were true of the one
capture that existed when the tool was written and stopped being true the day a
second one did.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _seal_module():
    spec = importlib.util.spec_from_file_location(
        "seal_caa_generation", ROOT / "tools/seal_caa_generation.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["seal_caa_generation"] = module
    spec.loader.exec_module(module)
    return module


DESCRIPTOR = "a" * 64


@pytest.fixture
def generation(tmp_path):
    """Two vector files and a plan, which is the least a seal needs."""
    directory = tmp_path / "vectors"
    directory.mkdir()
    for layer in (27, 31):
        np.savez(
            directory / f"valence_positive_layer{layer}.npz",
            v=np.ones(8, dtype=np.float32),
            model_descriptor_sha256=DESCRIPTOR,
        )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "model_path": "/models/checkpoint",
                "hidden_size": 8,
                "target_layers": [
                    {"index": 27, "kind": "full_attention"},
                    {"index": 31, "kind": "full_attention"},
                ],
            }
        )
    )
    return plan, directory


def _seal(monkeypatch, module, plan, directory) -> dict:
    """Run the seal against a stubbed cortex identity.

    The attribute is patched on the real registry rather than the module being
    replaced in `sys.modules`: the tool imports it inside `main`, so the
    per-call binding sees the patch, and the session teardown still finds the
    registry's own cache reset where it expects it.
    """
    from types import SimpleNamespace

    from core.brain.llm import model_registry

    monkeypatch.setattr(
        model_registry,
        "get_active_cortex_spec",
        lambda **_: SimpleNamespace(descriptor_sha256=DESCRIPTOR),
    )
    assert module.main(["--plan", str(plan), "--vectors", str(directory)]) == 0
    return json.loads((directory / "metadata.json").read_text())


def test_a_capture_record_is_what_the_contract_says(monkeypatch, generation):
    plan, directory = generation
    (directory / "capture.json").write_text(
        json.dumps(
            {
                "schema": "aura.caa.vector_capture.v1",
                "method": "contrastive_activation_addition",
                "statistic": "difference_of_means_generation_position_hidden_state",
                "position": "last token of the assistant opening",
                "extraction_carriers": ["Tell me where the work stands."],
                "captured_by": "tools/derive_generation_position_steering_vectors.py",
            }
        )
    )
    contract = _seal(monkeypatch, _seal_module(), plan, directory)["extraction_contract"]
    assert contract["statistic"] == (
        "difference_of_means_generation_position_hidden_state"
    )
    assert contract["captured_by"] == (
        "tools/derive_generation_position_steering_vectors.py"
    )
    assert contract["position"] == "last token of the assistant opening"
    assert contract["extraction_carriers"] == ["Tell me where the work stands."]
    # The schema key belongs to the capture record, not to the contract the
    # authority hashes.
    assert "schema" not in contract


def test_a_generation_with_no_capture_record_keeps_the_historical_values(
    monkeypatch, generation
):
    """The only capture those constants were ever true of."""
    plan, directory = generation
    contract = _seal(monkeypatch, _seal_module(), plan, directory)["extraction_contract"]
    assert contract["statistic"] == "difference_of_means_last_token_hidden_state"
    assert contract["captured_by"] == "tools/capture_27b_steering_vectors.py"


def test_two_captures_do_not_hash_to_the_same_contract(monkeypatch, generation):
    """The hash is what the authority binds to, so it has to separate them."""
    plan, directory = generation
    without = _seal(monkeypatch, _seal_module(), plan, directory)
    (directory / "capture.json").write_text(
        json.dumps(
            {
                "schema": "aura.caa.vector_capture.v1",
                "method": "contrastive_activation_addition",
                "statistic": "difference_of_means_generation_position_hidden_state",
                "captured_by": "tools/derive_generation_position_steering_vectors.py",
            }
        )
    )
    with_record = _seal(monkeypatch, _seal_module(), plan, directory)
    assert (
        without["extraction_contract"]["extraction_contract_sha256"]
        != with_record["extraction_contract"]["extraction_contract_sha256"]
    )
    assert without["generation_sha256"] != with_record["generation_sha256"]
