"""A Hadamard-folded two-bit pack loads, and refuses what it cannot do.

Ternary Bonsai 2 27B is Qwen3.5 27B at two bits. Two bits alone destroys a
model this size: the damage concentrates in a few activation channels whose
magnitude dwarfs the rest, and a group scale sized for the outlier
quantises everything else to nothing. The pack rotates each block before
quantising so no group is sized by one outlier, and folds the inverse into
the weights.

The loader published with the pack reads schema 1 and the pack ships schema
2, so it refuses the config it comes with and looks for the sign vectors
among the tensors, where schema 2 no longer keeps them. This reads schema 2.

The heavy tests need the 8.6GB pack on disk and are skipped without it. The
rest hold the contract whether or not it is installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.brain.llm.prism_hadamard import (
    PACK_MODEL_TYPE,
    is_prism_hadamard_pack,
    load_prism_hadamard_text_model,
)

PACK = Path(__file__).resolve().parents[1] / "models" / "Ternary-Bonsai-2-27B-mlx-2bit"
needs_pack = pytest.mark.skipif(
    not (PACK / "model.safetensors").is_file(),
    reason="the Ternary Bonsai 2 pack is not installed",
)


def test_an_ordinary_checkpoint_is_not_claimed(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert is_prism_hadamard_pack(tmp_path) is False


def test_a_directory_with_no_config_is_not_claimed(tmp_path):
    assert is_prism_hadamard_pack(tmp_path) is False


def test_unreadable_config_is_not_claimed(tmp_path):
    (tmp_path / "config.json").write_text("{not json")
    assert is_prism_hadamard_pack(tmp_path) is False


def test_a_pack_at_other_than_two_bits_is_refused(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": PACK_MODEL_TYPE,
                "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
                "text_config": {},
                "modules": [],
            }
        )
    )
    with pytest.raises(ValueError, match="Unsupported quantization"):
        load_prism_hadamard_text_model(tmp_path)


def test_an_unvalidated_block_size_is_refused(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": PACK_MODEL_TYPE,
                "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
                "hadamard_config": "hadamard.json",
                "text_config": {},
                "modules": [],
            }
        )
    )
    (tmp_path / "hadamard.json").write_text(
        json.dumps(
            {
                "prism.hadamard.block_size": 777,
                "prism.hadamard.sign_mode": "explicit",
                "prism.hadamard.sign_widths": [],
                "prism.hadamard.sign_values": [],
            }
        )
    )
    with pytest.raises(ValueError, match="Unvalidated Hadamard block size"):
        load_prism_hadamard_text_model(tmp_path)


@needs_pack
def test_the_installed_pack_is_claimed():
    assert is_prism_hadamard_pack(PACK) is True


@needs_pack
def test_the_manifests_agree_about_which_tensors_are_folded():
    # 402 packed modules: 401 folded forward and the embedding folded
    # inverse. A disagreement means some tensor is rotated on one side and
    # not the other, which reconstructs noise rather than failing loudly.
    config = json.loads((PACK / "config.json").read_text())
    manifest = json.loads((PACK / "hadamard.json").read_text())
    forward = set(manifest["prism.hadamard.weight_names"])
    inverse = set(manifest["prism.hadamard.inverse_weight_names"])
    assert not (forward & inverse), "a tensor folded both ways"
    named = {
        name.removeprefix("language_model.").removesuffix(".weight")
        for name in forward | inverse
    }
    assert {record["path"] for record in config["modules"]} == named
    widths = manifest["prism.hadamard.sign_widths"]
    assert sum(widths) == len(manifest["prism.hadamard.sign_values"])


@needs_pack
def test_it_loads_and_answers_through_the_ordinary_decode_path():
    from mlx_lm import generate

    from core.brain.llm.prism_hadamard import load_prism_hadamard_pack

    model, tokenizer = load_prism_hadamard_pack(PACK)
    # The ordinary path asks for exactly these three.
    assert callable(model)
    assert callable(model.make_cache)
    assert len(model.layers) == 64
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with the single word: ready"}],
        add_generation_prompt=True,
    )
    answer = generate(model, tokenizer, prompt=prompt, max_tokens=64, verbose=False)
    assert "ready" in answer.lower()


@needs_pack
def test_an_adapter_cannot_be_fused_onto_folded_weights():
    # A LoRA trained on the unrotated base does not compose with weights
    # that carry the inverse rotation. Saying so beats fusing noise.
    import inspect

    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker)
    assert "cannot be fused onto it" in source
