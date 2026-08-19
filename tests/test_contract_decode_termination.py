"""Contract tests: contract-aware decode termination (CP180b).

The engine must stop decoding with termination ``contract_complete`` the
moment a single FINAL_ANSWER JSON object completes (config-gated, default
off), the completion must count as a COMPLETE answer in the engine's own
failure gate and the service receipt contract, and the campaign's vanilla
path must stop streaming at the same uniform rule without consuming the
rest of the stream.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.resource_accounting import (  # noqa: E402
    build_information_receipt,
)
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    ComputeBudget,
    CortexConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)


def _eos_is_suppressed(eos_logit: float) -> bool:
    """Whether the decode is holding the end-of-text token down.

    Read from the engine's own floor rather than written here as a number.
    The floor moved to -1e4 so it stays finite in float16, and these tests
    still compared against -1e8: every masked logit read as unmasked, the
    doubles returned EOS on the first token, and a working contract decode
    looked like one that generates nothing. A test that hardcodes a constant
    it does not own drifts silently the day that constant is right to change.
    """
    from core.brain.llm.latent_cortex.engine import _FINITE_LOGIT_FLOOR

    return eos_logit <= _FINITE_LOGIT_FLOOR + 1.0


def _standing_in_for_sample(double) -> None:
    """Refuse a double the real sampler's callers could not call.

    A double that has drifted from the signature it replaces makes the engine
    look broken while it is correct: the decode fails with a TypeError about
    an unexpected keyword, and the test reports a contract failure that never
    happened. Checking the substitution at the point of substitution says
    which one is actually wrong.
    """
    import inspect

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine

    real = inspect.signature(LatentCortexEngine._sample)
    accepted = inspect.signature(double).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        return
    missing = [
        name
        for name, param in real.parameters.items()
        if name not in {"self", "logits"}
        and param.kind is inspect.Parameter.KEYWORD_ONLY
        and name not in accepted
    ]
    assert not missing, f"this double cannot be called the way _sample is: missing {missing}"


PROMPT_TOKENS = [5, 9, 17, 3, 42, 7, 11, 23]
CONTRACT_TEXT = 'FINAL_ANSWER: {"node": 6}'


def _profiled_stub_model(n_layers: int):
    args = SimpleNamespace(
        model_type="qwen2",
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        head_dim=16,
    )
    return SimpleNamespace(
        args=args,
        model=SimpleNamespace(args=args, layers=[object()] * n_layers),
    )


class _AccountingEngine:
    def __init__(self, token_count: int) -> None:
        self._tokens = [1] * token_count

    def _encode(self, *_args):
        return list(self._tokens)

    def _information_receipt(
        self,
        *,
        encoded_tokens,
        token_count,
        context_items,
        policy_evidence,
        verifier,
    ):
        del context_items, policy_evidence, verifier
        return build_information_receipt(
            sources=[
                {
                    "source_id": "rendered_model_input",
                    "kind": "model_input_tokens",
                    "content_sha256": hashlib.sha256(encoded_tokens).hexdigest(),
                    "byte_count": len(encoded_tokens),
                    "token_count": token_count,
                }
            ],
            policies={"fixture": "a" * 64},
        )


def _tiny_model():
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=8,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


class _ContractAtFive:
    """Decoded text becomes a complete contract answer at the fifth token."""

    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) % 127 + 1 for c in str(text)][:16] or [5]

    def decode(self, ids):
        if len(ids) == 1:
            return "}"  # every piece can close an object: cheap gate fires
        return CONTRACT_TEXT if len(ids) >= 5 else "still reasoning"


def _config(decode_contract: str, **overrides) -> CortexConfig:
    values = {
        "decode_max_tokens": 48,
        "decode_contract": decode_contract,
    }
    values.update(overrides)
    return CortexConfig(
        workspace=WorkspaceConfig(n_slots=4, seed=3),
        recurrence=RecurrenceConfig(max_steps=2, min_steps=1),
        branches=BranchConfig(n_branches=1),
        **values,
    )


def test_engine_stops_at_contract_completion_and_counts_complete():
    engine = LatentCortexEngine(
        _tiny_model(), _ContractAtFive(), config=_config("final_answer_v1")
    )
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())
    assert result.receipt.decode_termination == "contract_complete"
    assert result.receipt.decode_generated_tokens == 5
    assert result.ok, result.reason  # completion, not decode_incomplete


def test_engine_contract_off_preserves_historical_behavior():
    engine = LatentCortexEngine(
        _tiny_model(), _ContractAtFive(), config=_config("none")
    )
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())
    assert result.receipt.decode_termination != "contract_complete"


class _CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(char) for char in str(text)[:16]] or [5]

    def decode(self, ids):
        return "".join(chr(int(token)) for token in ids if int(token) != 0)


class _IrrecoverableContractTokenizer(_CharacterTokenizer):
    def decode(self, ids):
        return "FINAL_ANSWER: not-json" if ids else ""


def test_contract_masks_early_eos_and_completes_inside_bounded_grace():
    text = 'FINAL_ANSWER: {"node":6}'
    engine = LatentCortexEngine(
        _tiny_model(),
        _CharacterTokenizer(),
        config=_config(
            "final_answer_v1",
            decode_max_tokens=8,
            decode_contract_grace_tokens=len(text),
        ),
    )
    remaining = iter(ord(char) for char in text)
    observed_eos_logits: list[float] = []

    def eos_pressured_sample(
        logits,
        _temperature,
        _top_p,
        *,
        budget=None,
        random_key=None,
        operation="decode_sampling",
    ):
        del budget, random_key, operation
        eos_logit = float(logits[0].item())
        observed_eos_logits.append(eos_logit)
        if not _eos_is_suppressed(eos_logit):
            return 0
        return next(remaining)

    _standing_in_for_sample(eos_pressured_sample)
    engine._sample = eos_pressured_sample
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())

    assert result.ok, result.reason
    assert result.text == text
    assert result.receipt.decode_termination == "contract_complete"
    assert result.receipt.decode_contract_required is True
    assert result.receipt.decode_contract_satisfied is True
    assert result.receipt.decode_generated_tokens == len(text)
    assert result.receipt.decode_contract_grace_used_tokens == len(text) - 8
    assert observed_eos_logits
    assert all(_eos_is_suppressed(value) for value in observed_eos_logits)


def test_contract_incomplete_exhaustion_is_bounded_and_receipted():
    engine = LatentCortexEngine(
        _tiny_model(),
        _CharacterTokenizer(),
        config=_config(
            "final_answer_v1",
            decode_max_tokens=4,
            decode_contract_grace_tokens=3,
        ),
    )

    def never_complete(
        logits,
        _temperature,
        _top_p,
        *,
        budget=None,
        random_key=None,
        operation="decode_sampling",
    ):
        del budget, random_key, operation
        return ord("x") if _eos_is_suppressed(float(logits[0].item())) else 0

    _standing_in_for_sample(never_complete)
    engine._sample = never_complete
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())

    assert result.ok, result.reason
    assert result.text == "x" * 7
    assert result.receipt.decode_termination == "token_limit_contract_incomplete"
    assert result.receipt.decode_contract_required is True
    assert result.receipt.decode_contract_satisfied is False
    assert result.receipt.decode_contract_grace_tokens == 3
    assert result.receipt.decode_contract_grace_used_tokens == 3


def test_engine_stops_immediately_on_irrecoverable_contract_prefix():
    engine = LatentCortexEngine(
        _tiny_model(),
        _IrrecoverableContractTokenizer(),
        config=_config(
            "final_answer_v1",
            decode_max_tokens=48,
            decode_contract_grace_tokens=48,
        ),
    )
    sampled = {"count": 0}

    def sample_once(logits, _temperature, _top_p, *, budget=None, random_key=None, operation="decode_sampling"):
        del logits, budget, random_key, operation
        sampled["count"] += 1
        return 17

    _standing_in_for_sample(sample_once)
    engine._sample = sample_once
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())

    assert result.ok is False
    assert result.reason == "decode_incomplete:contract_irrecoverable"
    assert result.receipt.decode_termination == "contract_irrecoverable"
    assert result.receipt.decode_generated_tokens == 1
    assert result.receipt.decode_contract_satisfied is False
    assert result.text == "FINAL_ANSWER: not-json"
    assert sampled["count"] == 1


def test_service_receipt_contract_accepts_contract_complete():
    from core.brain.latent_cortex_service import LatentCortexService

    receipt = {
        "episode_id": "ep",
        "checkpoint_fingerprint": "a" * 64,
        "checkpoint_fingerprint_method": "sha256",
        "checkpoint_file_count": 4,
        "params_unchanged": True,
        "schedule_hash": "b" * 64,
        "steps_taken": 2,
        "n_slots": 4,
        "n_branches": 1,
        "budget": {
            "max_layer_apps": 1000,
            "spent_layer_apps": 10,
            "exhausted": False,
        },
        "decode_requested_tokens": 48,
        "decode_generated_tokens": 5,
        "decode_termination": "contract_complete",
        "decode_contract_required": True,
        "decode_contract_satisfied": True,
        "decode_contract_grace_tokens": 48,
        "decode_contract_grace_used_tokens": 0,
        "honest_flags": [],
    }
    config = {
        "n_slots": 4,
        "n_branches": 1,
        "decode_contract": "final_answer_v1",
        "decode_contract_grace_tokens": 48,
    }
    errors = LatentCortexService._receipt_contract_errors(receipt, config)
    assert "decode_incomplete" not in errors
    assert "decode_contract_unsatisfied" not in errors


def test_service_rejects_forged_or_incomplete_contract_completion():
    from core.brain.latent_cortex_service import LatentCortexService

    receipt = {
        "decode_requested_tokens": 48,
        "decode_generated_tokens": 52,
        "decode_termination": "contract_complete",
        "decode_contract_required": True,
        "decode_contract_satisfied": False,
        "decode_contract_grace_tokens": 8,
        "decode_contract_grace_used_tokens": 2,
    }
    config = {
        "n_slots": 4,
        "n_branches": 1,
        "decode_contract": "final_answer_v1",
        "decode_contract_grace_tokens": 8,
    }

    errors = LatentCortexService._receipt_contract_errors(receipt, config)

    assert "decode_contract_unsatisfied" in errors
    assert "decode_contract_grace_accounting_invalid" in errors


def test_service_rejects_irrecoverable_contract_output_as_product_incomplete():
    from core.brain.latent_cortex_service import LatentCortexService

    receipt = {
        "decode_requested_tokens": 48,
        "decode_generated_tokens": 1,
        "decode_termination": "contract_irrecoverable",
        "decode_contract_required": True,
        "decode_contract_satisfied": False,
        "decode_contract_grace_tokens": 48,
        "decode_contract_grace_used_tokens": 0,
    }
    config = {
        "n_slots": 4,
        "n_branches": 1,
        "decode_max_tokens": 48,
        "decode_contract": "final_answer_v1",
        "decode_contract_grace_tokens": 48,
    }

    errors = LatentCortexService._receipt_contract_errors(receipt, config)

    assert "decode_contract_unsatisfied" in errors
    assert "decode_contract_termination_mismatch" in errors
    assert "decode_incomplete" in errors


def test_campaign_vanilla_stream_stops_at_contract(monkeypatch):
    import mlx_lm

    from tools import run_latent_cortex_paired_campaign as runner

    consumed = {"count": 0}
    segments = [
        SimpleNamespace(text="I think ", generation_tokens=2),
        SimpleNamespace(text='FINAL_ANSWER: {"node', generation_tokens=7),
        SimpleNamespace(text='": 6}', generation_tokens=9),
        SimpleNamespace(text=" babble that must never stream", generation_tokens=15),
    ]

    def scripted_stream(model, tokenizer, prompt, max_tokens, **kwargs):
        for segment in segments:
            consumed["count"] += 1
            yield segment

    monkeypatch.setattr(mlx_lm, "stream_generate", scripted_stream)

    tokenizer = SimpleNamespace(encode=lambda text, **kw: [1] * 11)
    model = _profiled_stub_model(8)
    task = SimpleNamespace(
        prompt="What node?",
        domain="mathematics",
        response_contract=None,
    )
    text, layer_apps, *_ = runner._vanilla_once(
        model,
        tokenizer,
        task,
        max_tokens=256,
        accounting_engine=_AccountingEngine(11),
    )
    assert text == 'I think FINAL_ANSWER: {"node": 6}'
    assert consumed["count"] == 3  # the babble segment never streamed
    assert layer_apps == (11 + 8) * 8  # prefill emits token 1; eight decode forwards


def test_campaign_vanilla_without_contract_consumes_whole_stream(monkeypatch):
    import mlx_lm

    from tools import run_latent_cortex_paired_campaign as runner

    segments = [
        SimpleNamespace(text="no marker here ", generation_tokens=3),
        SimpleNamespace(text="just prose", generation_tokens=6),
    ]

    def scripted_stream(model, tokenizer, prompt, max_tokens, **kwargs):
        yield from segments

    monkeypatch.setattr(mlx_lm, "stream_generate", scripted_stream)
    tokenizer = SimpleNamespace(encode=lambda text, **kw: [1] * 4)
    model = _profiled_stub_model(8)
    task = SimpleNamespace(
        prompt="Say something",
        domain="calibration",
        response_contract=None,
    )
    text, layer_apps, *_ = runner._vanilla_once(
        model,
        tokenizer,
        task,
        max_tokens=256,
        accounting_engine=_AccountingEngine(4),
    )
    assert text == "no marker here just prose"
    assert layer_apps == (4 + 5) * 8


def test_campaign_stream_stops_when_contract_is_irrecoverably_invalid(monkeypatch):
    import mlx_lm

    from tools import run_latent_cortex_paired_campaign as runner

    consumed = {"count": 0}
    segments = [
        SimpleNamespace(text="FINAL_ANSWER: not-json", generation_tokens=4),
        SimpleNamespace(text='\nFINAL_ANSWER: {"node":6}', generation_tokens=10),
        SimpleNamespace(text=" runaway tail", generation_tokens=13),
    ]

    def scripted_stream(model, tokenizer, prompt, max_tokens, **kwargs):
        for segment in segments:
            consumed["count"] += 1
            yield segment

    monkeypatch.setattr(mlx_lm, "stream_generate", scripted_stream)
    tokenizer = SimpleNamespace(encode=lambda text, **kw: [1] * 4)
    model = _profiled_stub_model(8)
    task = SimpleNamespace(
        prompt="What node?",
        domain="mathematics",
        response_contract=None,
    )

    text, layer_apps, *_ = runner._vanilla_once(
        model,
        tokenizer,
        task,
        max_tokens=256,
        accounting_engine=_AccountingEngine(4),
    )

    assert text == "FINAL_ANSWER: not-json"
    assert consumed["count"] == 1
    assert layer_apps == (4 + 3) * 8


def test_branch_selection_receipts_contract_verdicts():
    """Every scored branch's probe leaves an auditable contract verdict."""
    engine = LatentCortexEngine(
        _tiny_model(),
        _ContractAtFive(),
        config=CortexConfig(
            workspace=WorkspaceConfig(n_slots=4, seed=3),
            recurrence=RecurrenceConfig(max_steps=2, min_steps=1),
            branches=BranchConfig(n_branches=2, exchange_interval=2),
            decode_max_tokens=48,
            decode_contract="final_answer_v1",
        ),
    )
    result = engine.reason(
        token_ids=PROMPT_TOKENS,
        budget=ComputeBudget(),
        verifier=lambda text: 1.0,
    )
    rows = result.receipt.branch_contract
    assert [row["branch"] for row in rows] == [0, 1]
    for row in rows:
        # `response_contract_valid` is a real verdict the engine now emits
        # per branch: the probe text is checked against the response contract
        # as well as the decode contract, and a branch that fails the second
        # is marked invalid with reason "response_contract_invalid". Kept as
        # an exact set rather than a superset so a junk field still fails —
        # the point of this assertion is that a receipt row carries the
        # verdicts and nothing else.
        assert set(row) == {
            "branch",
            "marker_count",
            "complete",
            "valid",
            "reason",
            "response_contract_valid",
        }
        assert row["response_contract_valid"] in (True, False, None)
        # _ContractAtFive probes decode to the complete contract text.
        assert row["complete"] is True and row["valid"] is True
    assert "branch_contract" in result.receipt.to_dict()


def test_branch_contract_empty_without_verifier():
    engine = LatentCortexEngine(
        _tiny_model(), _ContractAtFive(), config=_config("final_answer_v1")
    )
    result = engine.reason(token_ids=PROMPT_TOKENS, budget=ComputeBudget())
    assert result.receipt.branch_contract == []
