"""Known-answer checks for the endogenous language pathway.

The validation suite runs these at boot and binds three claims to them. They
live here rather than in ``core/organism/model_validation.py`` because that
file is long enough already and these are pathway knowledge, not suite
machinery.

Each is a question with an answer nobody has to trust: two corpora built so
the verdict is known in advance, a plausibility gate against constructed
logits, and one proposal that two different states must decide differently.
"""

from __future__ import annotations

__all__ = [
    "arbitration_follows_the_state",
    "declare_validation_tests",
    "bias_cannot_promote_a_ruled_out_token",
    "verdict_is_earned_on_known_corpora",
]


def verdict_is_earned_on_known_corpora() -> bool:
    """Run the verdict procedure against two corpora whose answer is known.

    Small on purpose — this runs at boot. A corpus with no state-token
    relationship must report no_signal and a register effect must report
    style_prior. A procedure that cannot tell those apart has no business
    issuing a verdict about the live substrate.
    """
    try:
        import numpy as np

        from core.brain.llm.endogenous_readout_training import (
            TurnTokens,
            fit_vocab_head,
        )
        from core.brain.llm.endogenous_state import STATE_DIM
    except ImportError:
        return False

    vocabulary = 60
    base = 1.0 / np.arange(1, vocabulary + 1) ** 1.1
    base /= base.sum()

    def corpus(mode: str, seed: int, turns: int = 300, tokens: int = 30):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(turns):
            state = np.clip(rng.normal(0.0, 0.6, STATE_DIM), -1.0, 1.0)
            probabilities = base.copy()
            if mode == "style":
                shift = float(np.tanh(state[0]))
                probabilities[0] *= 1.0 + 0.95 * shift
                probabilities[1] *= 1.0 - 0.95 * shift
            probabilities /= probabilities.sum()
            out.append(
                TurnTokens(
                    state=state.astype(np.float64),
                    tokens=rng.choice(
                        vocabulary, size=tokens, p=probabilities
                    ).astype(np.int64),
                )
            )
        return out

    verdicts = {}
    for mode in ("null", "style"):
        fit = fit_vocab_head(
            corpus(mode, 5),
            vocab_size=vocabulary,
            tokenizer_signature="validation",
            permutations=40,
            null_refits=0,
            seed=5,
            decays=(1e-1, 1.0),
        )
        if fit is None:
            return False
        verdicts[mode] = fit.verdict
    return verdicts == {"null": "no_signal", "style": "style_prior"}


def bias_cannot_promote_a_ruled_out_token() -> bool:
    """A token outside the model's plausible set must come back untouched."""
    try:
        import numpy as np

        from core.brain.llm.endogenous_decode import EndogenousLogitBiasProcessor
    except ImportError:
        return False
    processor = EndogenousLogitBiasProcessor(
        np.array([1.0, -1.0, 8.0, 0.0], dtype=np.float32)
    )
    logits = np.array([5.0, 4.9, -40.0, 0.1])
    out = processor.apply_numpy(logits)
    return bool(out[2] == logits[2] and out[0] > logits[0])


def arbitration_follows_the_state() -> bool:
    """The same proposal, two states, two decisions."""
    try:
        from core.brain.llm.endogenous_absorption import Proposal, arbitrate
        from core.brain.llm.endogenous_state import empty_state
    except ImportError:
        return False
    proposal = Proposal(
        summary="commit and drop the goal",
        asserted_confidence=0.95,
        abandons_active_goal=True,
        requires_action=True,
    )
    unsure = empty_state().do(
        **{
            "uncertainty.confidence": 0.1,
            "uncertainty.evidence_support": 0.1,
            "goal.active": 1.0,
            "goal.priority": 0.95,
        }
    )
    settled = empty_state().do(
        **{
            "uncertainty.confidence": 0.95,
            "uncertainty.evidence_support": 0.9,
            "goal.active": 0.0,
            "goal.priority": 0.0,
        }
    )
    return (
        arbitrate(proposal, unsure).decision == "reject"
        and arbitrate(proposal, settled).decision == "accept"
    )

def declare_validation_tests(ValidationTest, Observation, boolean_score) -> list:  # noqa: N803
    """The pathway's validation tests, built from the suite's own types.

    The types are passed in rather than imported, so this module does not
    depend on the validation suite: the suite owns the machinery and the
    pathway owns the questions.
    """
    return [
        ValidationTest(
            name="endogenous_verdict_is_earned_on_known_corpora",
            description=(
                "the readout trainer reports no_signal on a corpus with no "
                "state-token relationship and style_prior on a register effect"
            ),
            required_capability="",
            observation=Observation(
                name="verdicts_match_the_known_answers",
                value=True,
                source=(
                    "core/brain/llm/endogenous_readout_training.py "
                    "known-answer corpora"
                ),
            ),
            predict=lambda _m: verdict_is_earned_on_known_corpora(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="endogenous readout verdict",
            ),
            owner="core/brain/llm/endogenous_readout_training.py",
        ),
        ValidationTest(
            name="endogenous_bias_cannot_promote_a_ruled_out_token",
            description=(
                "the vocabulary bias lands only inside the model's plausible "
                "set, so it re-ranks near-ties and cannot invent"
            ),
            required_capability="",
            observation=Observation(
                name="implausible_token_unchanged",
                value=True,
                source="core/brain/llm/endogenous_decode.py plausibility gate",
            ),
            predict=lambda _m: bias_cannot_promote_a_ruled_out_token(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="endogenous vocabulary bias",
            ),
            owner="core/brain/llm/endogenous_decode.py",
        ),
        ValidationTest(
            name="endogenous_arbitration_follows_the_state",
            description=(
                "one proposal is rejected under low confidence and a held goal, "
                "and accepted under high confidence with no goal"
            ),
            required_capability="",
            observation=Observation(
                name="decision_changes_with_the_state",
                value=True,
                source="core/brain/llm/endogenous_absorption.py arbitration",
            ),
            predict=lambda _m: arbitration_follows_the_state(),
            score=lambda p, o: boolean_score(
                bool(p),
                expected=bool(o.value),
                subject="endogenous arbitration",
            ),
            owner="core/brain/llm/endogenous_absorption.py",
        ),
    ]
