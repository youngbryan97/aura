from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.learning import candidate_cortex_kernel as kernel
from tools import build_candidate_cortex_kernel as cli


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(kernel.canonical_json_bytes(value) + b"\n")


def _descriptor(path: Path) -> str:
    material = {
        "schema": "aura.model_artifact_descriptor.v1",
        "canonical_path": "/candidate/Qwen3.8-27B-4bit",
        "repository_id": "Qwen/Qwen3.8-27B",
        "revision": "exact-test-revision",
        "artifact_profile": {"hidden_size": 5120, "layers": 64},
    }
    digest = kernel.document_sha256(material)
    _write_json(path, {**material, "descriptor_sha256": digest})
    return digest


def _record(
    domain: str,
    user: str,
    assistant: str,
    *,
    source_index: int,
    source_key: str | None = None,
) -> kernel.SourceRecord:
    return kernel.SourceRecord(
        domain=domain,
        messages=(("user", user), ("assistant", assistant)),
        binding_key=domain,
        source_key=source_key or f"fixture/{domain}.json#records",
        source_index=source_index,
    )


def _fixture(root: Path, *, reverse: bool = False, include_crsm: bool = True) -> dict[str, Any]:
    descriptor = root / "descriptor.json"
    descriptor_sha = _descriptor(descriptor)
    domains = sorted(kernel.CORE_DOMAINS | ({"crsm"} if include_crsm else set()))
    records: list[kernel.SourceRecord] = []
    bindings: dict[str, dict[str, Any]] = {}
    for domain in domains:
        source = root / "sources" / f"{domain}.json"
        _write_json(source, {"domain": domain, "revision": 1})
        bindings[domain] = kernel.file_binding(source)
        for index in range(3):
            records.append(
                _record(
                    domain,
                    f"{domain} request {index}",
                    f"A substantive {domain} answer number {index}.",
                    source_index=index,
                )
            )
    # This stale statement must be excluded, never rewritten.
    records.append(
        _record(
            "architecture",
            "What model are you?",
            "I am a Qwen2.5 32-billion parameter model.",
            source_index=50,
        )
    )
    # Full normalized duplicate and same-request variant exercise both identity
    # deduplication and connected-component grouping.
    records.append(
        _record(
            "personality",
            "  PERSONALITY   REQUEST 0 ",
            "a substantive personality answer number 0.",
            source_index=51,
        )
    )
    records.append(
        _record(
            "character_voice",
            "personality request 1",
            "A distinct answer to the shared request.",
            source_index=52,
            source_key="fixture/character_voice.json#variant",
        )
    )
    if reverse:
        records.reverse()
    return {
        "descriptor": descriptor,
        "descriptor_sha": descriptor_sha,
        "bundle": kernel.SourceBundle(tuple(records), bindings, "injected"),
        "source_paths": [Path(binding["path"]) for binding in bindings.values()],
    }


def _build(fixture: dict[str, Any], output: Path) -> dict[str, Any]:
    return kernel.build_candidate_cortex_kernel(
        descriptor_path=fixture["descriptor"],
        expected_descriptor_sha256=fixture["descriptor_sha"],
        output_root=output,
        source_repo_root=Path(__file__).resolve().parents[1],
        valid_fraction=0.25,
        split_seed=17,
        source_bundle=fixture["bundle"],
    )


def _root(receipt: dict[str, Any]) -> Path:
    return Path(receipt["generation_root"])


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rewrite_receipt(root: Path, mutate: Any) -> None:
    path = root / "candidate_cortex_kernel_receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt.pop("receipt_sha256")
    mutate(receipt)
    receipt["receipt_sha256"] = kernel.document_sha256(receipt)
    _write_json(path, receipt)


def _refresh_output_binding(root: Path, receipt: dict[str, Any], name: str) -> None:
    path = root / receipt["outputs"][name]["path"]
    binding = kernel.file_binding(path)
    binding.pop("path")
    receipt["outputs"][name] = {
        "path": path.relative_to(root).as_posix(),
        **binding,
    }


def test_builds_compact_system_free_stratified_kernel(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    receipt = _build(fixture, tmp_path / "out")
    root = _root(receipt)
    train = _jsonl(root / "data/train.jsonl")
    valid = _jsonl(root / "data/valid.jsonl")
    provenance = _jsonl(root / "data/provenance.jsonl")

    assert receipt["resumed"] is False
    assert receipt["direct_quotes_mode"] == "injected"
    assert receipt["filters"]["stale_model_records_filtered"] == 1
    assert receipt["counts"]["unique_conversations"] < receipt["counts"]["source_records"]
    assert len(train) == receipt["counts"]["train"]
    assert len(valid) == receipt["counts"]["valid"]
    assert all(message["role"] != "system" for row in train + valid for message in row["messages"])
    assert not any("qwen2.5" in json.dumps(row).casefold() for row in train + valid)
    assert not any("32-billion" in json.dumps(row).casefold() for row in train + valid)
    assert {row["split"] for row in provenance} == {"train", "valid"}

    group_splits: dict[str, set[str]] = {}
    for row in provenance:
        group_splits.setdefault(row["group_key"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())
    for domain, stats in receipt["domains"].items():
        assert stats["train"] > 0, domain
        if stats["groups"] >= 2:
            assert stats["valid"] > 0, domain
    kernel.verify_candidate_cortex_kernel(
        root / "candidate_cortex_kernel_receipt.json",
        expected_descriptor_sha256=fixture["descriptor_sha"],
    )


def test_record_order_does_not_change_content_address_or_data(tmp_path: Path) -> None:
    first = _fixture(tmp_path / "first", reverse=False)
    second = _fixture(tmp_path / "second", reverse=True)
    one = _build(first, tmp_path / "out-one")
    two = _build(second, tmp_path / "out-two")

    assert one["content_sha256"] == two["content_sha256"]
    for relative in ("data/train.jsonl", "data/valid.jsonl", "data/provenance.jsonl"):
        assert (_root(one) / relative).read_bytes() == (_root(two) / relative).read_bytes()


def test_complete_generation_resumes_only_after_full_verification(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    first = _build(fixture, tmp_path / "out")
    second = _build(fixture, tmp_path / "out")
    assert second["resumed"] is True
    assert second["receipt_sha256"] == first["receipt_sha256"]


def test_tampered_output_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    receipt = _build(fixture, tmp_path / "out")
    root = _root(receipt)
    train = root / "data/train.jsonl"
    train.write_bytes(train.read_bytes() + train.read_bytes().splitlines(keepends=True)[0])
    with pytest.raises(kernel.CandidateCortexKernelError, match="receipt_output_mismatch:train"):
        kernel.verify_candidate_cortex_kernel(root / "candidate_cortex_kernel_receipt.json")


def test_candidate_mismatch_and_stale_source_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    receipt = _build(fixture, tmp_path / "out")
    receipt_path = _root(receipt) / "candidate_cortex_kernel_receipt.json"
    with pytest.raises(kernel.CandidateCortexKernelError, match="receipt_candidate_mismatch"):
        kernel.verify_candidate_cortex_kernel(
            receipt_path,
            expected_descriptor_sha256="f" * 64,
        )
    fixture["source_paths"][0].write_text("changed", encoding="utf-8")
    with pytest.raises(kernel.CandidateCortexKernelError, match="receipt_input_stale"):
        kernel.verify_candidate_cortex_kernel(receipt_path)


@pytest.mark.parametrize(
    ("replacement", "expected"),
    [
        (
            {
                "messages": [
                    {"role": "system", "content": "You are Aura."},
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            },
            "system_message_forbidden",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "I run on Qwen2.5 with 32B parameters."},
                ]
            },
            "stale_model_claim_present",
        ),
    ],
)
def test_semantic_verifier_rejects_forbidden_messages_after_rebinding(
    tmp_path: Path,
    replacement: dict[str, Any],
    expected: str,
) -> None:
    fixture = _fixture(tmp_path / "fixture")
    receipt = _build(fixture, tmp_path / "out")
    root = _root(receipt)
    train_path = root / "data/train.jsonl"
    rows = _jsonl(train_path)
    rows[0] = replacement
    train_path.write_bytes(b"".join(kernel.canonical_json_bytes(row) + b"\n" for row in rows))

    def mutate(value: dict[str, Any]) -> None:
        _refresh_output_binding(root, value, "train")

    _rewrite_receipt(root, mutate)
    with pytest.raises(kernel.CandidateCortexKernelError, match=expected):
        kernel.verify_candidate_cortex_kernel(
            root / "candidate_cortex_kernel_receipt.json",
            verify_inputs=False,
        )


def test_recomputed_grouping_rejects_forged_split_provenance(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    receipt = _build(fixture, tmp_path / "out")
    root = _root(receipt)
    provenance_path = root / "data/provenance.jsonl"
    rows = _jsonl(provenance_path)
    rows[0]["group_key"] = "0" * 64
    provenance_path.write_bytes(
        b"".join(kernel.canonical_json_bytes(row) + b"\n" for row in rows)
    )

    def mutate(value: dict[str, Any]) -> None:
        _refresh_output_binding(root, value, "provenance")

    _rewrite_receipt(root, mutate)
    with pytest.raises(kernel.CandidateCortexKernelError, match="provenance_group_mismatch"):
        kernel.verify_candidate_cortex_kernel(
            root / "candidate_cortex_kernel_receipt.json",
            verify_inputs=False,
        )


def test_missing_domain_claim_is_rejected_even_with_valid_receipt_digest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    receipt = _build(fixture, tmp_path / "out")
    root = _root(receipt)

    def mutate(value: dict[str, Any]) -> None:
        value["domains"].pop("theory")

    _rewrite_receipt(root, mutate)
    with pytest.raises(kernel.CandidateCortexKernelError, match="required_domains_missing"):
        kernel.verify_candidate_cortex_kernel(
            root / "candidate_cortex_kernel_receipt.json",
            verify_inputs=False,
        )


def test_cli_verify_returns_machine_readable_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path / "fixture")
    receipt = _build(fixture, tmp_path / "out")
    rc = cli.main(
        [
            "verify",
            "--receipt",
            str(_root(receipt) / "candidate_cortex_kernel_receipt.json"),
            "--descriptor-sha256",
            fixture["descriptor_sha"],
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert output["receipt_sha256"] == receipt["receipt_sha256"]


def test_builder_rejects_missing_domain_before_publication(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "fixture")
    records = tuple(record for record in fixture["bundle"].records if record.domain != "theory")
    bundle = kernel.SourceBundle(records, fixture["bundle"].bindings, "injected")
    with pytest.raises(kernel.CandidateCortexKernelError, match="required_domains_missing:theory"):
        kernel.build_candidate_cortex_kernel(
            descriptor_path=fixture["descriptor"],
            expected_descriptor_sha256=fixture["descriptor_sha"],
            output_root=tmp_path / "out",
            source_repo_root=Path(__file__).resolve().parents[1],
            source_bundle=bundle,
        )
    assert not (tmp_path / "out").exists()


def test_crsm_filter_rejects_internal_rows_and_deduplicates_safe_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "crsm.jsonl"
    rows = (
        {"text": "User: Will-approved self-reflection\nAura: <thought>private</thought>"},
        {"text": "User: What changed?\nAura: I repaired the cache ownership boundary."},
        {"text": "User: What changed?\nAura: I repaired the cache ownership boundary."},
    )
    source.write_bytes(b"".join(kernel.canonical_json_bytes(row) + b"\n" for row in rows))

    records, bindings, counts = kernel._load_crsm(source)  # noqa: SLF001

    assert len(records) == 1
    assert bindings["crsm"]["sha256"] == kernel.file_binding(source)["sha256"]
    assert counts == {
        "crsm_accepted": 1,
        "crsm_duplicate": 1,
        "crsm_internal_capture": 1,
    }
