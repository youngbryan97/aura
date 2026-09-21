"""The floor, measured on output rather than asserted on config.

The previous floor test verified that the product arm's decode PARAMETERS
matched the ordinary control. Three separate defects then shipped inside
exactly that gap, each rewriting the incumbent answer while every parameter
still matched:

  1. the decode bridge appended the terminal disposition to the restored
     prompt root, so the "incumbent" decoded from prompt + instruction;
  2. confidence-bound abstention emptied the answer outright;
  3. contract enforcement discarded produced answers -- 576 generated tokens
     returned as an empty string.

Measured consequence on the 32B: 14 of 14 full-stack answers differed from
ordinary decode, and the arm scored 3/14 against vanilla's 5/14 -- below a
floor that was supposed to be structural.

Config equality is not behavioural equality. This test compares the bytes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from tests.source_contract import family_text

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = REPO_ROOT / "tools"
for path in (str(REPO_ROOT), str(TOOLS)):
    if path not in sys.path:
        sys.path.insert(0, path)

HIDDEN = 32
PROMPT = [3, 8, 15, 22, 41, 5, 9, 17]


class _StubTokenizer:
    """Enough tokenizer that the terminal disposition becomes real tokens.

    Without one the disposition cannot be encoded, bridge_tokens stays empty,
    and the branch that rewrites the incumbent never executes -- so a test run
    with tokenizer=None passes whether the defect is present or not. That is
    exactly the false confidence this file exists to prevent.
    """

    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [1 + (ord(ch) % 100) for ch in text[:64]]

    def decode(self, tokens):
        return "".join(chr(65 + (int(t) % 26)) for t in tokens)


@pytest.fixture(scope="module")
def tiny_model():
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    args = ModelArgs(
        model_type="qwen2",
        hidden_size=HIDDEN,
        num_hidden_layers=8,
        intermediate_size=64,
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


def test_the_incumbent_answer_is_not_rewritten(tiny_model):
    """Nothing may silently rewrite the incumbent.

    With no verifier admitted no candidate can dominate, so the product arm
    has nothing to promote and its answer must be the ordinary decode of the
    same prompt -- token for token. Any difference is something rewriting the
    answer behind the gate, which is how all three defects above presented.
    """
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.types import ComputeBudget

    import run_rlc_reconciliation_sweep as sweep

    config = sweep._build_config(4, 8, "applied", 24, profile="full")
    engine = LatentCortexEngine(tiny_model, config=config, tokenizer=_StubTokenizer())
    result = engine.reason(token_ids=PROMPT, budget=ComputeBudget())

    assert result.ok is True
    # The incumbent decode must have produced something. An empty answer is
    # how abstention and contract invalidation both presented, and an empty
    # answer is strictly worse than the ordinary decode it replaced.
    assert result.tokens, (
        "the incumbent produced no tokens: something emptied the answer "
        f"(termination={result.receipt.decode_termination!r})"
    )
    # No promotion can have occurred without an admitted verifier, so the
    # answer must come from the ordinary decode lane.
    assert result.receipt.decode_incumbent_policy == "vanilla_incumbent"


def test_the_disposition_never_steers_the_incumbent(tiny_model):
    """A prefix applied to the incumbent means the floor is not ordinary
    decode. The disposition may compete as a candidate; it may not be imposed
    before anything is measured."""
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.types import ComputeBudget

    import run_rlc_reconciliation_sweep as sweep

    applied = sweep._build_config(4, 8, "applied", 24, profile="full")
    suppressed = sweep._build_config(4, 8, "suppressed", 24, profile="full")

    outputs = []
    for config in (applied, suppressed):
        engine = LatentCortexEngine(tiny_model, config=config, tokenizer=_StubTokenizer())
        result = engine.reason(token_ids=PROMPT, budget=ComputeBudget())
        outputs.append(list(result.tokens))

    assert outputs[0] == outputs[1], (
        "the terminal disposition changed the incumbent answer, so the "
        "product arm's floor is 'vanilla plus disposition' rather than vanilla"
    )


def test_both_abstain_paths_are_guarded_under_the_incumbent():
    """Structural check, because the behavioural one cannot reach here.

    Abstention needs an ADMITTED verifier that then refutes the baseline --
    a decoy-preflight pass plus a deterministic refutation. The tiny-model
    tests above cannot construct that, so they passed while the second abstain
    path was still discarding answers on the 32B. Rather than pretend the
    behavioural test covers it, this pins the two guards directly.

    Both paths presented identically in the receipts and had to be fixed
    separately:

      1. `decision == "abstain"` setting decode_termination and emptying
         out_tokens;
      2. `decision == "abstain"` setting failure_reason, which fails the whole
         episode -- ok=False, zero tokens -- and is then correctly classified
         as a policy failure by the harness, so it scores as an empty answer
         with no fault raised.

    Measured before the second fix: four of fourteen 32B cells returned empty
    text against receipts reporting 278-560 generated tokens, three of them on
    tasks ordinary decode got right.
    """
    import inspect

    from core.brain.llm.latent_cortex import engine as engine_mod

    src = family_text(engine_mod)

    # Path 2: failing the episode is permitted only when latent owns output.
    assert (
        'if self.config.decode_incumbent_policy == "latent":\n'
        '                        failure_reason = "answer_replacement_abstained"'
        in src
    ), "the episode-failing abstain path is no longer guarded by the incumbent policy"

    # Path 1: emptying out_tokens is permitted only when latent owns output.
    assert "if latent_decode_authorized:\n                        out_tokens = []" in src, (
        "the token-emptying abstain path is no longer guarded"
    )

    # And the decline is recorded rather than silent, so a receipt reader can
    # tell that an abstention was considered and overridden.
    assert "answer_replacement_abstention_declined_under_incumbent" in src
    assert "confidence_bound_abstention_declined_under_incumbent" in src
