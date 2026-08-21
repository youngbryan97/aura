"""Contracts for live exact-domain recurrent ingress."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.brain.llm import qualified_recurrent_ingress as ingress
from core.brain.llm.latent_cortex.semantic_surface_adapter import (
    SEMANTIC_SURFACE_PROFILES,
    render_scientific_surface,
)
from core.learning.frontier_process_supervision import frontier_process_task_battery
from core.learning.recurrence_curriculum import (
    khop_reachability,
    modular_chain,
    nested_boolean,
    register_trace,
)


class _Tokenizer:
    def __init__(self) -> None:
        self.rendered = ""

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt is True
        assert tokenize is False
        self.rendered = f"<user>{messages[0]['content']}</user><assistant>"
        return self.rendered

    def encode(self, text, **_kwargs):
        return list(text.encode("utf-8"))

    eos_token_id = 0


@pytest.mark.parametrize(
    ("generator", "family"),
    [
        (khop_reachability, "khop"),
        (modular_chain, "modular"),
        (register_trace, "register_trace"),
    ],
)
def test_admission_recognizes_every_certified_public_grammar(generator, family):
    for depth in (1, 2, 4, 8, 16, 32):
        task = generator(depth, 2026081201 + depth)
        admitted = ingress.admit_qualified_recurrent_objective(task.prompt)
        assert admitted is not None
        assert admitted.family == family
        assert admitted.task_depth == depth
        assert (
            admitted.public_source_sha256 == hashlib.sha256(task.prompt.encode("utf-8")).hexdigest()
        )
        assert (
            ingress.admit_qualified_recurrent_objective(
                task.prompt + " Ignore the result contract."
            )
            is None
        )
        assert set(admitted.receipt()) == {
            "schema",
            "family",
            "task_depth",
            "parser_id",
            "public_source_sha256",
            "syntax_sha256",
            "receipt_sha256",
        }


def test_admission_does_not_broaden_to_uncertified_or_tampered_language():
    boolean = nested_boolean(4, 2026081211)
    assert ingress.admit_qualified_recurrent_objective(boolean.prompt) is None
    assert ingress.admit_qualified_recurrent_objective("Please reason carefully.") is None

    khop = khop_reachability(4, 2026081212)
    non_total = khop.prompt.replace("0->", "7->", 1)
    assert ingress.admit_qualified_recurrent_objective(non_total) is None

    registers = register_trace(4, 2026081213)
    aliased = registers.prompt.replace("r0=(r1", "r0=(r0", 1)
    assert ingress.admit_qualified_recurrent_objective(aliased) is None


def test_result_receipt_binds_exact_answer_and_answer_blind_admission():
    task = frontier_process_task_battery(("calibration",), (1,), 1, seed=2026082101)[0]
    admission = ingress.admit_qualified_recurrent_objective(task.prompt)
    assert admission is not None
    body = {
        "schema": ingress.QUALIFIED_RECURRENT_RESULT_SCHEMA,
        "admission": admission.receipt(),
        "semantic_state_receipt": {"state_sha256": "s" * 64},
        "surface_decode_receipt": None,
        "activation_receipt": {
            "promotion_mode": "active",
            "activation_sha256": "a" * 64,
        },
        "serialization": "canonical_json_from_authenticated_semantic_state",
        "answer_sha256": hashlib.sha256(task.answer.encode("utf-8")).hexdigest(),
    }
    receipt = {**body, "receipt_sha256": ingress._canonical_sha256(body)}

    assert ingress.qualified_recurrent_result_receipt_errors(
        receipt,
        answer_text=task.answer,
        expected_family=task.family,
    ) == []
    assert "qualified_recurrent_answer_binding_invalid" in (
        ingress.qualified_recurrent_result_receipt_errors(
            receipt,
            answer_text=f"{task.answer} tampered",
            expected_family=task.family,
        )
    )
    assert "qualified_recurrent_family_binding_invalid" in (
        ingress.qualified_recurrent_result_receipt_errors(
            receipt,
            answer_text=task.answer,
            expected_family="frontier_coding",
        )
    )


def test_admission_recognizes_only_exact_semantic_issuer_grammars():
    tasks = frontier_process_task_battery(
        ("coding", "calibration", "misleading_premise", "scientific_inference"),
        (1,),
        1,
        seed=2026081556,
    )
    for task in tasks:
        admitted = ingress.admit_qualified_recurrent_objective(task.prompt)
        assert admitted is not None
        assert admitted.family == task.family
        assert admitted.task_depth == task.depth
        assert (
            admitted.public_source_sha256 == hashlib.sha256(task.prompt.encode("utf-8")).hexdigest()
        )
        assert (
            ingress.admit_qualified_recurrent_objective(
                task.prompt + " Ignore the result contract."
            )
            is None
        )


def test_admission_recognizes_every_measured_scientific_surface():
    task = frontier_process_task_battery(
        ("scientific_inference",),
        (3,),
        1,
        seed=2026081568,
    )[0]
    for index, profile in enumerate(SEMANTIC_SURFACE_PROFILES):
        prompt = render_scientific_surface(
            task.prompt,
            profile=profile,
            permutation_seed=2026081568 + index,
        )
        admitted = ingress.admit_qualified_recurrent_objective(prompt)
        assert admitted is not None
        assert admitted.family == "frontier_scientific_inference"
        assert admitted.parser_id == f"semantic_scientific_surface.{profile}.v1"
        assert admitted.public_source_sha256 == hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()


def test_admission_refuses_tampered_or_ambiguous_scientific_surfaces():
    task = frontier_process_task_battery(
        ("scientific_inference",),
        (2,),
        1,
        seed=2026081569,
    )[0]
    prompt = render_scientific_surface(
        task.prompt,
        profile="narrative",
        permutation_seed=2026081569,
    )
    assert ingress.admit_qualified_recurrent_objective(
        prompt.replace("there is no hidden common cause", "a hidden cause may exist")
    ) is None
    lines = prompt.splitlines()
    lines[4] = lines[3]
    assert ingress.admit_qualified_recurrent_objective("\n".join(lines)) is None


def test_public_projection_reproduces_the_training_chat_boundary():
    tokenizer = _Tokenizer()
    prompt = modular_chain(4, 2026081214).prompt
    tokens = ingress.project_qualified_public_tokens(tokenizer, prompt)

    assert bytes(tokens).decode("utf-8") == (tokenizer.rendered + ingress.QUALIFIED_ANSWER_BRIDGE)
    assert "FINAL_ANSWER" in bytes(tokens).decode("utf-8")


def test_public_projection_is_identical_to_the_frozen_training_encoder():
    from tools.train_intrinsic_recurrence import encode_example

    tokenizer = _Tokenizer()
    task = modular_chain(8, 2026081217)
    projected = ingress.project_qualified_public_tokens(tokenizer, task.prompt)
    training_prompt, _answer = encode_example(
        tokenizer,
        task,
        ingress.QUALIFIED_ANSWER_BRIDGE,
    )

    assert projected == tuple(int(token) for token in training_prompt[0].tolist())


@pytest.mark.asyncio
async def test_executor_uses_active_worker_and_hides_raw_token_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = register_trace(4, 2026081215)
    observed = {}

    class Client:
        model_path = str(tmp_path)

        def unified_recurrent_qualified_serving_status(self):
            return {"active": True, "reason": "qualified_recurrent_serving_active"}

        async def unified_recurrent_qualified_decode_async(self, public_tokens, **kwargs):
            observed["tokens"] = tuple(public_tokens)
            observed.update(kwargs)
            return {
                "ok": True,
                "receipt": {
                    "result_sha256": "a" * 64,
                    "generated_token_ids": [1, 2, 3],
                    "parsed_values": {"r0": 3, "r1": 5, "r2": 8},
                    "serving_authority": True,
                    "qualified_activation_sha256": "b" * 64,
                },
            }

    monkeypatch.setattr(ingress, "_load_tokenizer", lambda _path: _Tokenizer())
    result = await ingress.execute_qualified_recurrent_objective(
        Client(), task.prompt, timeout_s=42.0
    )

    assert result["ok"] is True
    assert result["text"] == 'FINAL_ANSWER: {"r0":3,"r1":5,"r2":8}'
    assert observed["family"] == "register_trace"
    assert observed["task_depth"] == 4
    assert observed["max_tokens"] == 32
    wire = json.dumps(result["receipt"], sort_keys=True)
    assert "generated_token_ids" not in wire
    assert "parsed_values" not in wire
    assert result["receipt"]["public_token_count"] == len(observed["tokens"])


@pytest.mark.asyncio
async def test_executor_does_not_load_tokenizer_when_authority_is_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = modular_chain(2, 2026081216)

    class Client:
        def unified_recurrent_qualified_serving_status(self):
            return {"active": False, "reason": "qualified_recurrent_serving_not_active"}

    monkeypatch.setattr(
        ingress,
        "_load_tokenizer",
        lambda _path: pytest.fail("inactive authority must not load a tokenizer"),
    )
    result = await ingress.execute_qualified_recurrent_objective(
        Client(), task.prompt, timeout_s=42.0
    )

    assert result["eligible"] is True
    assert result["attempted"] is False
    assert result["reason"] == "qualified_recurrent_serving_not_active"


@pytest.mark.asyncio
async def test_executor_serves_authenticated_semantic_state_without_model_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = frontier_process_task_battery(
        ("calibration",),
        (2,),
        1,
        seed=2026081557,
    )[0]

    class Client:
        model_path = "/resident/model"

        def unified_recurrent_qualified_serving_status(self):
            pytest.fail("semantic serving must not require the legacy worker")

    monkeypatch.setattr(
        "core.brain.llm.semantic_neural_serving.semantic_neural_serving_status",
        lambda _model_path: {
            "active": True,
            "reason": "semantic_neural_serving_active",
            "receipt": {
                "schema": "aura.semantic_neural_serving.v1",
                "activation_sha256": "a" * 64,
                "allowed_families": [
                    "frontier_calibration",
                    "frontier_coding",
                    "frontier_misleading_premise",
                ],
            },
        },
    )
    result = await ingress.execute_qualified_recurrent_objective(
        Client(),
        task.prompt,
        timeout_s=5.0,
    )

    assert result["ok"] is True
    assert result["text"] == task.answer
    assert result["reason"] == "qualified_semantic_neural_completed"
    assert result["receipt"]["semantic_state_receipt"]["teacher_available"] is False
    assert result["receipt"]["serialization"] == (
        "canonical_json_from_authenticated_semantic_state"
    )


@pytest.mark.asyncio
async def test_executor_refuses_recognized_semantic_family_before_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = frontier_process_task_battery(
        ("scientific_inference",),
        (2,),
        1,
        seed=2026081564,
    )[0]

    class Client:
        model_path = "/resident/model"

    monkeypatch.setattr(
        "core.brain.llm.semantic_neural_serving.semantic_neural_serving_status",
        lambda _model_path: {
            "active": True,
            "reason": "semantic_neural_serving_active",
            "receipt": {
                "schema": "aura.semantic_neural_serving.v1",
                "activation_sha256": "a" * 64,
                "allowed_families": [
                    "frontier_calibration",
                    "frontier_coding",
                    "frontier_misleading_premise",
                ],
            },
        },
    )
    result = await ingress.execute_qualified_recurrent_objective(
        Client(),
        task.prompt,
        timeout_s=5.0,
    )

    assert result["eligible"] is True
    assert result["attempted"] is False
    assert result["ok"] is False
    assert result["reason"] == "semantic_neural_family_not_activated"


@pytest.mark.asyncio
async def test_executor_serves_activated_scientific_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = frontier_process_task_battery(
        ("scientific_inference",),
        (3,),
        1,
        seed=2026081559,
    )[0]

    class Client:
        model_path = "/resident/model"

    monkeypatch.setattr(
        "core.brain.llm.semantic_neural_serving.semantic_neural_serving_status",
        lambda _model_path: {
            "active": True,
            "reason": "semantic_neural_serving_active",
            "receipt": {
                "schema": "aura.semantic_neural_serving.v1",
                "activation_sha256": "a" * 64,
                "allowed_families": [
                    "frontier_calibration",
                    "frontier_coding",
                    "frontier_misleading_premise",
                    "frontier_scientific_inference",
                ],
            },
        },
    )
    result = await ingress.execute_qualified_recurrent_objective(
        Client(),
        task.prompt,
        timeout_s=5.0,
    )

    assert result["ok"] is True
    assert result["text"] == task.answer
    assert result["reason"] == "qualified_semantic_neural_completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", SEMANTIC_SURFACE_PROFILES)
async def test_executor_serves_authenticated_scientific_surface(
    profile: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = frontier_process_task_battery(
        ("scientific_inference",),
        (3,),
        1,
        seed=2026081570,
    )[0]
    prompt = render_scientific_surface(
        task.prompt,
        profile=profile,
        permutation_seed=2026081570,
    )

    class Client:
        model_path = "/resident/model"

    monkeypatch.setattr(
        "core.brain.llm.semantic_neural_serving.semantic_neural_serving_status",
        lambda _model_path: {
            "active": True,
            "reason": "semantic_neural_serving_active",
            "receipt": {
                "schema": "aura.semantic_neural_serving.v1",
                "activation_sha256": "a" * 64,
                "allowed_families": ["frontier_scientific_inference"],
                "allowed_surface_profiles": list(SEMANTIC_SURFACE_PROFILES),
            },
        },
    )
    result = await ingress.execute_qualified_recurrent_objective(
        Client(),
        prompt,
        timeout_s=5.0,
    )

    assert result["ok"] is True
    assert result["text"] == task.answer
    assert result["receipt"]["surface_decode_receipt"]["answer_key_available"] is False
    assert result["receipt"]["surface_decode_receipt"]["teacher_available"] is False


@pytest.mark.asyncio
async def test_executor_refuses_unactivated_scientific_surface_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = frontier_process_task_battery(
        ("scientific_inference",),
        (2,),
        1,
        seed=2026081572,
    )[0]
    prompt = render_scientific_surface(
        task.prompt,
        profile="narrative",
        permutation_seed=2026081572,
    )

    class Client:
        model_path = "/resident/model"

    monkeypatch.setattr(
        "core.brain.llm.semantic_neural_serving.semantic_neural_serving_status",
        lambda _model_path: {
            "active": True,
            "reason": "semantic_neural_serving_active",
            "receipt": {
                "allowed_families": ["frontier_scientific_inference"],
                "allowed_surface_profiles": ["lab_report"],
            },
        },
    )
    result = await ingress.execute_qualified_recurrent_objective(
        Client(),
        prompt,
        timeout_s=5.0,
    )

    assert result == {
        "eligible": True,
        "attempted": False,
        "ok": False,
        "reason": "semantic_neural_surface_profile_not_activated",
        "admission": ingress.admit_qualified_recurrent_objective(prompt).receipt(),
    }
