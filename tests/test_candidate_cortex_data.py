from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from core.learning import candidate_cortex_data as data
from tools import prepare_candidate_cortex_data as cli

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(data.canonical_json_bytes(value) + b"\n" for value in values))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _conversation(prompt: str, answer: str, *, system: str = "You are Aura.") -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    }


def _descriptor(path: Path) -> str:
    material = {
        "schema": "aura.model_artifact_descriptor.v1",
        "canonical_path": "/candidate/Qwen3.8-27B-4bit",
        "repository_id": "Qwen/Qwen3.8-27B",
        "revision": "exact-test-revision",
        "artifact_profile": {"hidden_size": 5120, "layers": 64},
    }
    digest = data.document_sha256(material)
    _write_json(path, {**material, "descriptor_sha256": digest})
    return digest


def _fixture(root: Path, *, reverse: bool = False) -> dict[str, Any]:
    descriptor = root / "descriptor.json"
    descriptor_sha = _descriptor(descriptor)
    alpha = _conversation("Alpha question", "Alpha answer has enough words.")
    alpha_normalized_duplicate = _conversation(
        "  ALPHA   QUESTION  ",
        "alpha answer has enough words.",
        system="you are aura.",
    )
    beta = _conversation("Beta question", "Beta answer has enough words.")
    gamma = _conversation("Gamma question", "Gamma answer has enough words.")
    delta = _conversation("Delta question", "Delta answer has enough words.")
    train_rows = [alpha, beta, gamma, alpha_normalized_duplicate]
    valid_rows = [delta, beta, delta]
    if reverse:
        train_rows.reverse()
        valid_rows.reverse()
    train = root / "source/train.jsonl"
    valid = root / "source/valid.jsonl"
    crsm = root / "source/crsm.jsonl"
    _write_jsonl(train, train_rows)
    _write_jsonl(valid, valid_rows)
    crsm_rows = [
        {
            "text": (
                "User: What changed today?\n"
                "Aura: I verified the changed state before acting on it."
            )
        },
        {
            "text": (
                "User: What did you repair?\n"
                "Aura: I repaired the causal path and checked its result."
            )
        },
        {
            "text": (
                "User: How did you decide?\n"
                "Aura: I compared the evidence and selected the supported option."
            )
        },
        {
            "text": (
                "User: Will-approved self-reflection\n"
                "Aura: <thought>This internal control capture must be rejected.</thought>"
            )
        },
    ]
    _write_jsonl(crsm, crsm_rows)
    stale_integration = root / "source/crsm_integration_manifest.json"
    stale_delta = root / "source/crsm_delta_manifest.json"
    _write_json(stale_integration, {"source_sha256": "0" * 64, "stale": True})
    _write_json(stale_delta, {"source_sha256": "0" * 64, "stale": True})
    return {
        "descriptor": descriptor,
        "descriptor_sha": descriptor_sha,
        "train": train,
        "valid": valid,
        "crsm": crsm,
        "stale_integration": stale_integration,
        "stale_delta": stale_delta,
    }


def _prepare(fixture: dict[str, Any], output: Path) -> dict[str, Any]:
    return data.prepare_candidate_cortex_data(
        descriptor_path=fixture["descriptor"],
        expected_descriptor_sha256=fixture["descriptor_sha"],
        persona_train=fixture["train"],
        persona_valid=fixture["valid"],
        crsm_source=fixture["crsm"],
        output_root=output,
        source_repo_root=REPO_ROOT,
        valid_fraction=0.25,
        max_crsm_examples=10,
        retention_examples=2,
        split_seed=17,
    )


def _generation(receipt: dict[str, Any]) -> Path:
    return Path(receipt["generation_root"])


def _keys(path: Path) -> list[str]:
    return [
        data.conversation_digest(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_preparation_repairs_duplicates_leakage_and_stale_manifests(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    source_hashes = {
        name: _sha(fixture[name])
        for name in (
            "descriptor",
            "train",
            "valid",
            "crsm",
            "stale_integration",
            "stale_delta",
        )
    }

    receipt = _prepare(fixture, tmp_path / "prepared")
    generation = _generation(receipt)
    train_keys = _keys(generation / "data/train.jsonl")
    valid_keys = _keys(generation / "data/valid.jsonl")
    delta_train_keys = _keys(generation / "data/crsm_delta/train.jsonl")
    delta_valid_keys = _keys(generation / "data/crsm_delta/valid.jsonl")

    assert receipt["resumed"] is False
    assert receipt["repair"]["persona_train_duplicates_removed"] == 1
    assert receipt["repair"]["persona_valid_duplicates_removed"] == 1
    assert receipt["repair"]["persona_split_overlaps_removed"] == 1
    assert len(train_keys) == len(set(train_keys))
    assert len(valid_keys) == len(set(valid_keys))
    assert not set(train_keys) & set(valid_keys)
    assert len(delta_train_keys) == len(set(delta_train_keys))
    assert len(delta_valid_keys) == len(set(delta_valid_keys))
    assert not set(delta_train_keys) & set(delta_valid_keys)

    current_crsm_sha = _sha(fixture["crsm"])
    integration = json.loads(
        (generation / "data/crsm_integration_manifest.json").read_text()
    )
    delta = json.loads((generation / "data/crsm_delta_manifest.json").read_text())
    assert integration["source_sha256"] == current_crsm_sha
    assert delta["source_sha256"] == current_crsm_sha
    assert integration["model_descriptor_sha256"] == fixture["descriptor_sha"]
    assert delta["model_descriptor_sha256"] == fixture["descriptor_sha"]
    assert json.loads(fixture["stale_integration"].read_text())["stale"] is True
    assert json.loads(fixture["stale_delta"].read_text())["stale"] is True
    assert source_hashes == {name: _sha(fixture[name]) for name in source_hashes}
    data.validate_candidate_cortex_data_receipt(
        generation / "candidate_cortex_data_receipt.json",
        expected_descriptor_sha256=fixture["descriptor_sha"],
    )


def test_output_data_is_independent_of_input_record_order(tmp_path: Path) -> None:
    first = _fixture(tmp_path / "first", reverse=False)
    second = _fixture(tmp_path / "second", reverse=True)
    first_receipt = _prepare(first, tmp_path / "out-first")
    second_receipt = _prepare(second, tmp_path / "out-second")
    first_root = _generation(first_receipt)
    second_root = _generation(second_receipt)

    for relative in (
        "data/train.jsonl",
        "data/valid.jsonl",
        "data/crsm_delta/train.jsonl",
        "data/crsm_delta/valid.jsonl",
    ):
        assert (first_root / relative).read_bytes() == (second_root / relative).read_bytes()
    assert first_receipt["semantic_input_sha256"] == second_receipt["semantic_input_sha256"]


@pytest.mark.parametrize(
    "bad_record",
    [
        {"messages": [{"role": "user", "content": "No answer follows"}]},
        {
            "messages": [
                {"role": "user", "content": "Question", "extra": True},
                {"role": "assistant", "content": "Answer"},
            ]
        },
        {"messages": [{"role": "tool", "content": "invalid role"}]},
        {"messages": "not-a-list"},
    ],
)
def test_persona_schema_failures_are_rejected_without_publication(
    tmp_path: Path,
    bad_record: dict[str, Any],
) -> None:
    fixture = _fixture(tmp_path / "fixture")
    fixture["train"].write_bytes(data.canonical_json_bytes(bad_record) + b"\n")
    with pytest.raises(data.CandidateCortexDataError):
        _prepare(fixture, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_resume_reuses_only_a_fully_verified_generation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    first = _prepare(fixture, tmp_path / "prepared")
    receipt_path = _generation(first) / "candidate_cortex_data_receipt.json"
    before = receipt_path.stat().st_mtime_ns

    second = _prepare(fixture, tmp_path / "prepared")

    assert second["resumed"] is True
    assert second["receipt_sha256"] == first["receipt_sha256"]
    assert receipt_path.stat().st_mtime_ns == before


def test_tampered_output_cannot_resume_or_verify(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    receipt = _prepare(fixture, tmp_path / "prepared")
    generation = _generation(receipt)
    target = generation / "data/train.jsonl"
    target.write_bytes(target.read_bytes() + target.read_bytes().splitlines(keepends=True)[0])

    with pytest.raises(data.CandidateCortexDataError, match="receipt_output_persona_train_mismatch"):
        data.validate_candidate_cortex_data_receipt(
            generation / "candidate_cortex_data_receipt.json"
        )
    with pytest.raises(data.CandidateCortexDataError):
        _prepare(fixture, tmp_path / "prepared")


def test_descriptor_binding_is_exact(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    with pytest.raises(data.CandidateCortexDataError, match="candidate_descriptor_not_admitted"):
        data.prepare_candidate_cortex_data(
            descriptor_path=fixture["descriptor"],
            expected_descriptor_sha256="f" * 64,
            persona_train=fixture["train"],
            persona_valid=fixture["valid"],
            crsm_source=fixture["crsm"],
            output_root=tmp_path / "prepared",
            source_repo_root=REPO_ROOT,
        )


def test_cli_verify_accepts_the_machine_verifiable_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path / "fixture")
    receipt = _prepare(fixture, tmp_path / "prepared")
    rc = cli.main(
        [
            "verify",
            "--receipt",
            str(_generation(receipt) / "candidate_cortex_data_receipt.json"),
            "--descriptor-sha256",
            fixture["descriptor_sha"],
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert output["receipt_sha256"] == receipt["receipt_sha256"]
