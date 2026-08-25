"""Fitting the endogenous head, and refusing to overstate what the fit means.

The claim under test is narrow and checkable: **does Aura's cognitive state
carry information about which words she uses, beyond how often those words are
used at all?** Everything here exists to answer that without flattering it.

Five decisions do the work.

**The baseline is the unigram, not zero.** A head that only learned "the" is
common would score well against nothing and mean nothing. The bias vector
starts at the log unigram frequency of the training split and the weight
matrix starts at zero, so at initialisation the model *is* the unigram model.
Any held-out gain after that came from the state, because the state is the
only thing the weights can see.

**The split is by turn, not by token.** All tokens from one reply share one
state vector. Splitting by token would put the same z on both sides of the
line and the held-out score would measure memorisation. Turns go whole into
train, validation, or holdout.

**The fit stops when it stops helping.** Fourteen thousand parameters against
a few thousand tokens will fit noise given enough epochs, and an early version
of this file did: at sixty epochs every regime scored worse than the unigram,
and at a higher learning rate a corpus built with NO state-token relationship
was reported as content-bearing. Epoch selection now runs on an inner
validation split, and the nulls get the same treatment.

**The null is the correspondence, not the capacity.** The main test permutes
which state goes with which held-out turn and rescores the fitted head. If the
head is using the state-token correspondence, the true pairing must beat the
shuffled ones; if it learned a spurious mapping, both score the same. Two
hundred shuffles cost nothing because no refitting is involved, which is what
makes a real p-value affordable. A smaller set of permuted-state refits runs
alongside as the capacity control.

**Style and content are scored apart, against their own nulls.** A head that
learns "unhappy state → unhappy words" is a style adapter. A head that helps
on rare, content-carrying tokens is something stronger. Comparing a rare-token
gain against an overall null is how the first gets reported as the second, so
each is tested against the null computed for that same quantity.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.brain.llm.endogenous_pair_recorder import RecordedPair
from core.brain.llm.endogenous_state import STATE_DIM, layout_digest, semantics_digest
from core.brain.llm.endogenous_vocab_head import EndogenousVocabHead

logger = logging.getLogger("Aura.EndogenousTraining")

#: Below this many turns there is nothing to hold out. The refusal is the
#: point: a matrix fitted on forty turns would carry a report claiming it was
#: measured.
MIN_TURNS = 60

#: Distinct REPLIES the holdout must contain before a verdict means anything
#: general. Not a turn count: the same reply repeated four hundred times is one
#: observation of the state-to-words relationship, however many rows it fills.
#: Twenty is the point where the rare-token bucket stops being dominated by a
#: single repeated string on this corpus.
MIN_HOLDOUT_REPLIES = 20

#: Distinct replies the head must be FITTED on. A grouped split protects the
#: holdout from memorisation and can starve the training side doing it: the
#: first live corpus left nine replies to learn from, which no null detects
#: because the nulls are computed on the holdout.
MIN_TRAIN_REPLIES = 20

#: A token must appear this often before the head gets a row for it. Rows for
#: tokens seen twice are noise with a coefficient.
MIN_TOKEN_COUNT = 5

#: Shares held out. Fixed so a run cannot be tuned by choosing a friendlier
#: split. The validation share comes out of what remains after the holdout, so
#: the number the verdict rests on is never the number the fit was stopped on.
HOLDOUT_FRACTION = 0.25
VALIDATION_FRACTION = 0.2

#: Shuffles of the held-out state-turn correspondence. This is the p-value's
#: resolution: 200 shuffles can resolve p down to 0.005.
PERMUTATIONS = 200

#: Refits on permuted training states. Expensive, and a different control: it
#: answers what this much capacity can achieve with no real correspondence.
NULL_REFITS = 3

#: Below this many refits the capacity control is reported but does not
#: gate. A maximum over one sample is not a control, and letting it veto a
#: properly permutation-tested result was measured doing exactly that.
MIN_REFITS_TO_GATE = 3

#: Significance the improvement must reach before anything is claimed,
#: Bonferroni-split across the two tests a verdict can rest on (overall
#: and rare). Three tests are reported; two of them decide.
ALPHA = 0.05
DECIDING_TESTS = 2

#: Tokens ranked inside this share of total occurrence mass are the frequent,
#: function-carrying end of the vocabulary. Taken from the corpus rather than
#: from a word list, so it needs no English-specific authorship.
FUNCTION_MASS_SHARE = 0.5

#: Epoch budget and patience for the early-stopping loop.
MAX_EPOCHS = 120
PATIENCE = 12

#: Regularisation strengths the fit chooses between on the validation
#: split. A single hand-picked value would be a constant nobody measured.
DECAY_GRID: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


@dataclass(frozen=True)
class TurnTokens:
    """One turn: the state that held, and the tokens that came out of it.

    ``group`` is the identity of the REPLY, not of the turn. Turns that said
    the same thing share one, and the split keeps a group whole. See
    :func:`_three_way_split` for why that is not a refinement.
    """

    state: np.ndarray
    tokens: np.ndarray
    group: str = ""
    #: Which dimensions were reachable for this turn. Carried so variance can
    #: be measured over the turns where a dimension was actually READ. Without
    #: it, a dimension pinned at one value but occasionally absent shows
    #: variance that belongs to the presence mask, not to the dimension.
    present: np.ndarray | None = None


@dataclass(frozen=True)
class Scores:
    """What a model scored on the held-out turns."""

    log_likelihood: float
    top_k: float
    log_likelihood_frequent: float
    log_likelihood_rare: float

    def as_dict(self) -> dict[str, float]:
        return {
            "log_likelihood": round(self.log_likelihood, 6),
            "top_k_rate": round(self.top_k, 6),
            "log_likelihood_frequent": round(self.log_likelihood_frequent, 6),
            "log_likelihood_rare": round(self.log_likelihood_rare, 6),
        }


@dataclass(frozen=True)
class PermutationTest:
    """One quantity, its observed value, and where it sits among the shuffles."""

    observed: float
    null_mean: float
    null_std: float
    null_max: float
    p_value: float
    permutations: int

    @property
    def significant(self) -> bool:
        return self.observed > 0.0 and self.p_value <= ALPHA / DECIDING_TESTS

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed": round(self.observed, 6),
            "null_mean": round(self.null_mean, 6),
            "null_std": round(self.null_std, 6),
            "null_max": round(self.null_max, 6),
            "p_value": round(self.p_value, 6),
            "permutations": self.permutations,
            "threshold": round(ALPHA / DECIDING_TESTS, 6),
            "significant": self.significant,
        }


def _permutation_test(observed: float, nulls: Sequence[float]) -> PermutationTest:
    array = np.asarray(list(nulls), dtype=np.float64)
    if array.size == 0:
        return PermutationTest(observed, 0.0, 0.0, 0.0, 1.0, 0)
    # The +1 on both sides is the standard correction: a p-value of exactly
    # zero would claim more resolution than the number of shuffles bought.
    p = float((np.sum(array >= observed) + 1) / (array.size + 1))
    return PermutationTest(
        observed=float(observed),
        null_mean=float(np.mean(array)),
        null_std=float(np.std(array)),
        null_max=float(np.max(array)),
        p_value=p,
        permutations=int(array.size),
    )


@dataclass
class VocabFit:
    """A fitted head, its baselines, its nulls, and the verdict they support."""

    weights: np.ndarray
    bias: np.ndarray
    active_tokens: np.ndarray
    vocab_size: int
    n_turns_train: int
    n_turns_validation: int
    n_turns_holdout: int
    n_tokens_train: int
    n_tokens_holdout: int
    epochs_used: int
    decay_used: float
    trained: Scores
    unigram: Scores
    random_projection: Scores
    overall_test: PermutationTest
    rare_test: PermutationTest
    frequent_test: PermutationTest
    refit_null_scores: tuple[Scores, ...]
    top_k: int
    layout: str
    tokenizer: str
    #: The DERIVATIONS this was fitted against, not just the feature names. A
    #: head is fitted to what the numbers meant.
    semantics: str = ""
    #: What actually moved in the corpus this was fitted on. A verdict can
    #: only be a claim about the part of the state that varied.
    state_variance: dict[str, Any] = field(default_factory=dict)
    #: How many DIFFERENT replies the holdout scored on, and how many the whole
    #: corpus held. A repeated reply is one observation of the state-to-words
    #: relationship however many rows it fills.
    n_replies_holdout: int = 0
    n_replies_train: int = 0
    n_replies_total: int = 0
    trained_at: float = field(default_factory=time.time)

    @property
    def improvement_nats(self) -> float:
        """Held-out gain per token over the unigram model, in nats."""
        return float(self.trained.log_likelihood - self.unigram.log_likelihood)

    @property
    def rare_improvement_nats(self) -> float:
        return float(self.trained.log_likelihood_rare - self.unigram.log_likelihood_rare)

    @property
    def frequent_improvement_nats(self) -> float:
        return float(
            self.trained.log_likelihood_frequent - self.unigram.log_likelihood_frequent
        )

    @property
    def refit_null_improvements(self) -> tuple[float, ...]:
        return tuple(
            s.log_likelihood - self.unigram.log_likelihood for s in self.refit_null_scores
        )

    @property
    def refit_null_improvements_rare(self) -> tuple[float, ...]:
        return tuple(
            s.log_likelihood_rare - self.unigram.log_likelihood_rare
            for s in self.refit_null_scores
        )

    @property
    def _refit_gate_applies(self) -> bool:
        return len(self.refit_null_scores) >= MIN_REFITS_TO_GATE

    @property
    def refit_null_ceiling(self) -> float:
        values = self.refit_null_improvements
        return float(max(values)) if values else 0.0

    @property
    def refit_null_ceiling_rare(self) -> float:
        values = self.refit_null_improvements_rare
        return float(max(values)) if values else 0.0

    @property
    def overall_signal(self) -> bool:
        return self.overall_test.significant and (
            not self._refit_gate_applies
            or self.improvement_nats > self.refit_null_ceiling
        )

    @property
    def rare_signal(self) -> bool:
        """Whether the gain on rare tokens stands on its own.

        Tested against the rare null and the rare refit ceiling, never against
        the overall ones. A content effect lives in a small share of the
        tokens, so it can be unmissable on rare tokens and invisible in the
        average — an earlier version of this gate required overall
        significance first and reported exactly that case as no signal.
        """
        return self.rare_test.significant and (
            not self._refit_gate_applies
            or self.rare_improvement_nats > self.refit_null_ceiling_rare
        )

    @property
    def beats_refit_nulls(self) -> bool:
        return self.overall_signal or self.rare_signal

    @property
    def only_affect_varied(self) -> bool:
        """Whether affect was the ONLY channel that moved in this corpus.

        The distinction the whole verdict scheme exists for. A state whose
        only variance is affect can produce a real, significant gain over
        rare tokens and still be nothing more than a learned style adapter —
        cheerful state, cheerful vocabulary — because there is no other
        channel the effect could have come from. The gain is not downgraded
        here, because it is real and downgrading it would hide a measurement;
        it is NAMED, so a reader cannot take content_bearing to mean the goal
        system or the memory system contributed when neither one moved.
        """
        channels = set(self.state_variance.get("by_channel") or {})
        return channels == {"affect"}

    @property
    def corpus_is_too_repetitive(self) -> bool:
        """Whether the held-out side has enough DIFFERENT replies to score on.

        A gain measured over a handful of distinct replies is a claim about
        those replies. Aura's first live corpus held 116 turns and 39 replies,
        two of which — "ready" and a bare comma — accounted for 78 of them;
        grouping the split fixed the leakage, and this is what stops the
        remainder being read as a general result.
        """
        return (
            self.n_replies_holdout < MIN_HOLDOUT_REPLIES
            or self.n_replies_train < MIN_TRAIN_REPLIES
        )

    @property
    def verdict(self) -> str:
        """What this fit earns the right to claim, and nothing beyond it."""
        if self.overall_test.permutations == 0:
            return "no_verdict_no_null"
        if self.corpus_is_too_repetitive:
            return "no_verdict_corpus_too_repetitive"
        if self.rare_signal:
            return "content_bearing"
        if self.overall_signal:
            return "style_prior"
        return "no_signal"

    @property
    def usable(self) -> bool:
        """Whether this head may be attached to a decode loop at all."""
        return self.verdict in {"style_prior", "content_bearing"}

    def as_report(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "semantics": self.semantics,
            "tokenizer": self.tokenizer,
            "state_variance": self.state_variance,
            "only_affect_varied": self.only_affect_varied,
            "state_dim": int(STATE_DIM),
            "vocab_size": int(self.vocab_size),
            "active_tokens": int(self.active_tokens.size),
            "turns": {
                "train": self.n_turns_train,
                "validation": self.n_turns_validation,
                "holdout": self.n_turns_holdout,
                "distinct_replies_holdout": self.n_replies_holdout,
                "distinct_replies_train": self.n_replies_train,
                "distinct_replies_total": self.n_replies_total,
            },
            "tokens": {"train": self.n_tokens_train, "holdout": self.n_tokens_holdout},
            "epochs_used": self.epochs_used,
            "weight_decay_selected": self.decay_used,
            "top_k": int(self.top_k),
            "held_out": {
                "trained": self.trained.as_dict(),
                "unigram_baseline": self.unigram.as_dict(),
                "random_projection_baseline": self.random_projection.as_dict(),
            },
            "improvement_nats": round(self.improvement_nats, 6),
            "improvement_nats_frequent": round(self.frequent_improvement_nats, 6),
            "improvement_nats_rare": round(self.rare_improvement_nats, 6),
            "permutation_tests": {
                "overall": self.overall_test.as_dict(),
                "frequent": self.frequent_test.as_dict(),
                "rare": self.rare_test.as_dict(),
            },
            "refit_nulls": {
                "overall": [round(v, 6) for v in self.refit_null_improvements],
                "rare": [round(v, 6) for v in self.refit_null_improvements_rare],
            },
            "refit_null_ceiling": round(self.refit_null_ceiling, 6),
            "refit_null_ceiling_rare": round(self.refit_null_ceiling_rare, 6),
            "refit_gate_applied": self._refit_gate_applies,
            "overall_signal": self.overall_signal,
            "rare_signal": self.rare_signal,
            "alpha": ALPHA,
            "verdict": self.verdict,
            "usable": self.usable,
            "trained_at": self.trained_at,
            "what_this_means": (
                VERDICT_MEANING[self.verdict]
                + (
                    " — but affect was the only channel that varied in this "
                    "corpus, so nothing here distinguishes the effect from a "
                    "learned style adapter"
                    if self.only_affect_varied
                    else ""
                )
            ),
        }

    def to_head(self) -> EndogenousVocabHead:
        """Scatter the fitted rows into a full-vocabulary head.

        Tokens the corpus never contained keep a zero row, so the head is
        silent about words it has no evidence for. The unigram term stays
        behind: it is a constant per token rather than a function of state,
        and a frequency prior stapled onto a transformer that already has a
        better one would only add noise.
        """
        weights = np.zeros((self.vocab_size, STATE_DIM), dtype=np.float32)
        weights[self.active_tokens] = self.weights.astype(np.float32)
        return EndogenousVocabHead(
            weights=weights,
            bias=np.zeros(self.vocab_size, dtype=np.float32),
            vocab_size=int(self.vocab_size),
            semantics=self.semantics,
            layout=self.layout,
            tokenizer=self.tokenizer,
            trained=bool(self.usable),
            report=self.as_report(),
            trained_at=self.trained_at,
        )


VERDICT_MEANING = {
    "no_verdict_no_null": (
        "no permutation test completed, so nothing here is a measurement"
    ),
    "no_signal": (
        "the fit did not beat shuffles of its own held-out correspondence; the "
        "state carries no usable information about word choice in this corpus"
    ),
    "style_prior": (
        "the state shifts frequent, function-carrying words — register, "
        "hedging, directness. A learned style adapter, and not evidence of "
        "propositional content in the substrate"
    ),
    "no_verdict_corpus_too_repetitive": (
        "the held-out replies were too few and too alike to support any claim "
        "about the state-to-words relationship. Record more varied turns"
    ),
    "content_bearing": (
        "the gain survives on rare, content-carrying tokens, which a register "
        "shift cannot explain. The state carries information about what is "
        "being said, not only how"
    ),
}


def tokenize_pairs(
    pairs: Iterable[RecordedPair],
    tokenizer: Any,
    *,
    max_tokens_per_turn: int = 256,
) -> list[TurnTokens]:
    """Turn recorded (state, text) into (state, token-ids) per turn.

    Tokenising at fit time rather than at record time is what lets one corpus
    be fitted against whichever model is resident. It also means the ids here
    are the ids the head will bias, by construction.
    """
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise ValueError("tokenizer has no encode()")
    out: list[TurnTokens] = []
    for pair in pairs:
        try:
            ids = encode(pair.text)
        except (TypeError, ValueError) as exc:
            logger.debug("turn not tokenised: %s", exc)
            continue
        ids = np.asarray(list(ids)[:max_tokens_per_turn], dtype=np.int64)
        if ids.size == 0:
            continue
        state = np.where(pair.present, pair.values, 0.0).astype(np.float64)
        if state.shape != (STATE_DIM,):
            continue
        out.append(
            TurnTokens(
                state=state,
                tokens=ids,
                # The reply itself is the group. Two turns that produced the
                # same words are one unit for splitting, whatever else differed
                # about them.
                group=hashlib.sha256(pair.text.encode("utf-8")).hexdigest()[:16],
                present=np.asarray(pair.present, dtype=bool),
            )
        )
    return out


def varying_dimensions(turns: Sequence[TurnTokens]) -> dict[str, Any]:
    """Which named dimensions actually MOVED in the corpus that was fitted.

    A verdict is a claim about the state, and it can only be a claim about the
    part of the state that varied. A dimension pinned at one value across
    every turn is worse than an absent one: it reads as live, it pads the
    coverage figure, and it cannot carry information however strong the
    verdict above it.

    Measured on the live corpus, 2026-08-25: 5 of the 6 goal dimensions were
    constant across 1,629 turns, along with attention.load and
    recurrence.budget_used, both pinned at their ceiling. Reporting this
    beside the verdict is what stops a gain carried entirely by affect being
    read as evidence about goals, memory or recurrence.
    """
    if not turns:
        return {
            "varying": [],
            "constant": [],
            "never_present": [],
            "by_channel": {},
            "collinear_pairs": [],
            "cross_channel_collinear": [],
        }

    from core.brain.llm.endogenous_state import FEATURES

    states = np.stack([turn.state for turn in turns])
    if any(turn.present is not None for turn in turns):
        present = np.stack(
            [
                turn.present
                if turn.present is not None
                else np.ones(states.shape[1], dtype=bool)
                for turn in turns
            ]
        )
    else:
        present = np.ones_like(states, dtype=bool)

    varying: list[str] = []
    constant: list[str] = []
    absent: list[str] = []
    by_channel: dict[str, int] = {}
    spread = np.zeros(states.shape[1], dtype=float)
    for index, feature in enumerate(FEATURES):
        if index >= states.shape[1]:
            break
        # Variance among the turns where the dimension was READ. Measured over
        # the masked column instead, a dimension pinned at one value but
        # unreachable on 3% of turns reads as varying — and that variance
        # belongs to the presence mask, not to the dimension.
        seen = states[present[:, index], index]
        if seen.size == 0:
            absent.append(feature.name)
            continue
        spread[index] = float(seen.std())
        if spread[index] > 0.0:
            varying.append(feature.name)
            by_channel[feature.channel] = by_channel.get(feature.channel, 0) + 1
        else:
            constant.append(feature.name)
    # Two dimensions carrying one value are not two dimensions. Collinear
    # columns inflate the coverage figure, give the head two gradient paths to
    # the same signal, and — the part that matters for the causal work —
    # make an ablation of one channel silently an ablation of part of another.
    #
    # Measured on the live corpus, 2026-08-25: `temporal.future` was exactly
    # `goal.priority` and `temporal.past` exactly `memory.recall_hits`,
    # because the temporal channel is DERIVED from those two and the goal was
    # continuously active. That derivation is deliberate and documented; the
    # collinearity it produces is reported rather than hidden, so a channel
    # influence map built on this corpus is read with it in view.
    duplicates: list[tuple[str, str]] = []
    for left in range(min(len(FEATURES), states.shape[1])):
        if float(spread[left]) <= 0.0:
            continue
        for right in range(left + 1, min(len(FEATURES), states.shape[1])):
            if float(spread[right]) <= 0.0:
                continue
            both = present[:, left] & present[:, right]
            if both.sum() < 2:
                continue
            if np.allclose(states[both, left], states[both, right]):
                duplicates.append((FEATURES[left].name, FEATURES[right].name))
    return {
        "varying": varying,
        "constant": constant,
        "never_present": absent,
        "by_channel": dict(sorted(by_channel.items())),
        "collinear_pairs": [list(pair) for pair in duplicates],
        "cross_channel_collinear": [
            list(pair)
            for pair in duplicates
            if pair[0].split(".", 1)[0] != pair[1].split(".", 1)[0]
        ],
    }


def _distinct_replies(turns: Sequence[TurnTokens]) -> int:
    """How many different replies these turns contain.

    An unlabelled turn is its own reply, matching what the splitter does with
    it. Counting only labelled ones read every constructed corpus as having
    zero distinct replies, so the repetition gate refused corpora built with a
    known answer — a gate that fires on everything is the same as no gate.
    """
    return len(
        {turn.group or f"__turn_{index}" for index, turn in enumerate(turns)}
    )


def _forward_in_time_split(
    turns: Sequence[TurnTokens],
) -> tuple[list[TurnTokens], list[TurnTokens], list[TurnTokens]]:
    """Fit on the past, score on the future.

    The control a random split cannot give. State drifts slowly and topics
    cluster in time, so a held-out turn surrounded by training turns can share
    both its state and its words with its neighbours for reasons that have
    nothing to do with one causing the other. Scoring only on turns recorded
    AFTER everything the head saw removes that route.

    A group that straddles the boundary goes to the past side, so a reply is
    still never scored on after being fitted on.
    """
    total = len(turns)
    cut = total - max(1, int(total * HOLDOUT_FRACTION))
    seen_before_cut = {turn.group for turn in turns[:cut] if turn.group}
    past = list(turns[:cut])
    future = []
    for turn in turns[cut:]:
        if turn.group and turn.group in seen_before_cut:
            past.append(turn)
        else:
            future.append(turn)
    validation_size = max(1, int(len(past) * VALIDATION_FRACTION))
    # The validation slice is the END of the past, so early stopping is also
    # judged on turns later than the ones it fitted.
    validation = past[-validation_size:]
    train = past[:-validation_size]
    return train, validation, future


def _three_way_split(
    turns: Sequence[TurnTokens], *, seed: int
) -> tuple[list[TurnTokens], list[TurnTokens], list[TurnTokens]]:
    """Split by REPLY, never by turn.

    Splitting by turn measures memorisation whenever a reply repeats. Measured
    on Aura's own corpus, 2026-08-25: 116 recorded turns held 39 distinct
    replies, 41 of them the single word "ready" and 37 a bare comma. Those
    turns landed on both sides of a turn-wise split, the head learned the
    region of state space that says "ready", and scoring it on held-out
    "ready" turns returned a gain on a RARE token — which the trainer read as
    propositional content and reported `content_bearing`, its strongest
    verdict.

    Every null in this module endorsed it, and correctly: permuting the
    state-to-turn correspondence destroys the mapping, so the null sits near
    zero and the observed gain towers over it. A matched null answers "is this
    gain bigger than chance"; it cannot answer "was the answer in the training
    set". Only the split can.

    So a group is whole or absent. The holdout takes whole groups until it has
    its share of TURNS, which keeps the token counts comparable while making
    the held-out replies ones the head has never been fitted on.
    """
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[TurnTokens]] = {}
    for index, turn in enumerate(turns):
        # A turn with no group identity is its own group, so an unlabelled
        # corpus behaves exactly as it did before rather than collapsing into
        # one giant group.
        grouped.setdefault(turn.group or f"__turn_{index}", []).append(turn)

    keys = list(grouped)
    order = rng.permutation(len(keys))
    wanted = max(1, int(len(turns) * HOLDOUT_FRACTION))

    holdout: list[TurnTokens] = []
    rest: list[TurnTokens] = []
    for position in order:
        members = grouped[keys[position]]
        # A group that would OVERSHOOT the target is passed over rather than
        # taken. One reply can be most of a corpus — "ready" was 41 of 116
        # turns here — and taking it whole put 55% of the turns in the holdout
        # and left nine distinct replies to fit on. Skipping it keeps the
        # holdout near its share and the training side intact.
        if len(holdout) < wanted and (
            not holdout or len(holdout) + len(members) <= wanted
        ):
            holdout.extend(members)
        else:
            rest.extend(members)
    if not rest:
        # One group cannot be split. Say nothing rather than measure the
        # holdout against itself.
        return [], [], holdout

    validation_size = max(1, int(len(rest) * VALIDATION_FRACTION))
    validation = rest[:validation_size]
    train = rest[validation_size:]
    return train, validation, holdout


def _flatten(
    turns: Sequence[TurnTokens], remap: dict[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    labels: list[int] = []
    for turn in turns:
        for token in turn.tokens:
            index = remap.get(int(token))
            if index is None:
                continue
            states.append(turn.state)
            labels.append(index)
    if not labels:
        return np.zeros((0, STATE_DIM)), np.zeros(0, dtype=np.int64)
    return np.asarray(states, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(
        np.clip(np.sum(np.exp(shifted), axis=1, keepdims=True), 1e-300, None)
    )


def _balanced_log_likelihood(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    frequent: np.ndarray,
) -> float:
    """Validation criterion: the two ends of the vocabulary, weighted equally.

    Selecting the epoch on the plain mean log-likelihood stops the fit before
    anything rare has been learned, because a handful of function words carry
    most of the tokens. Measured on a corpus built so that one state dimension
    decides which of several RARE words appears: selection on the plain mean
    stopped at epoch 0-2 and reported no signal, while the structure was
    there. Frequent and rare are averaged so a gain at either end can stop the
    fit.
    """
    if x.shape[0] == 0:
        return 0.0
    truth = _log_softmax(x @ weights.T + bias)[np.arange(len(y)), y]
    is_frequent = frequent[y]
    parts = []
    if np.any(is_frequent):
        parts.append(float(np.mean(truth[is_frequent])))
    if np.any(~is_frequent):
        parts.append(float(np.mean(truth[~is_frequent])))
    return float(np.mean(parts)) if parts else 0.0


def _standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean and scale of the training states.

    The channels of z_Aura are on wildly different scales — a pooled substrate
    band sits near zero while a confidence reading fills [0, 1]. One shared
    learning rate across all of them leaves the small-variance channels
    effectively frozen. Standardising for the fit and folding the transform
    back into the weights afterwards costs nothing and removes the problem.
    """
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _fit_weights(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    log_prior: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray],
    frequent: np.ndarray,
    batch_size: int,
    learning_rate: float,
    l2: float,
    seed: int,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Multinomial logistic regression over the state, with the prior held fixed.

    Linear on purpose. The claim being supported is that the state carries
    information about word choice, and a linear head is the weakest model that
    can demonstrate it. Something deeper would leave open whether the capacity
    or the state did the work.

    **The bias does not move.** It is set to the training unigram and stays
    there, so the whole held-out gain is attributable to the state term and
    nothing else. The shipped head discards the bias anyway.

    **Momentum, not a per-parameter optimiser.** Adam was tried here and is
    actively wrong for this shape of problem: it normalises every parameter's
    step to roughly the learning rate, so fourteen thousand coefficients whose
    true value is zero move as fast as the handful that carry signal. Measured
    on a corpus with a real, strong style effect, validation degraded from the
    first epoch and the weight matrix reached 0.22 before anything had been
    learned. Standardising the inputs gives the comparable step sizes that
    motivated reaching for Adam, without the noise amplification.

    The epoch count is chosen, not set. Without that this overfits into a
    negative result on every regime, including the ones built with real signal.
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    bias = log_prior.astype(np.float64).copy()
    if n == 0:
        return np.zeros((n_classes, STATE_DIM)), bias, 0

    mean, scale = _standardise(x)
    xs = (x - mean) / scale
    val_x, val_y = validation
    val_xs = (val_x - mean) / scale if val_x.shape[0] else val_x

    weights = np.zeros((n_classes, STATE_DIM), dtype=np.float64)
    velocity = np.zeros_like(weights)
    momentum = 0.9

    best_weights = weights.copy()
    best_score = _balanced_log_likelihood(
        val_xs, val_y, weights, bias, frequent=frequent
    )
    best_epoch = 0
    since_improvement = 0

    for epoch in range(1, max(1, max_epochs) + 1):
        order = rng.permutation(n)
        for begin in range(0, n, batch_size):
            index = order[begin:begin + batch_size]
            xb, yb = xs[index], y[index]
            probabilities = np.exp(_log_softmax(xb @ weights.T + bias))
            probabilities[np.arange(len(yb)), yb] -= 1.0
            probabilities /= len(yb)
            gradient = probabilities.T @ xb + l2 * weights
            velocity = momentum * velocity + gradient
            weights -= learning_rate * velocity
        score = _balanced_log_likelihood(
            val_xs, val_y, weights, bias, frequent=frequent
        )
        if score > best_score:
            best_score = score
            best_weights = weights.copy()
            best_epoch = epoch
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= patience:
                break

    # Fold the standardisation back in, so the returned head reads raw z.
    raw_weights = best_weights / scale
    return raw_weights, bias - raw_weights @ mean, best_epoch


def _fit_with_selected_decay(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    log_prior: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray],
    frequent: np.ndarray,
    batch_size: int,
    learning_rate: float,
    decays: Sequence[float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Fit at each decay and keep the one the validation split prefers.

    A single hand-picked regularisation strength is a constant nobody
    measured. The grid is small and the selection uses the same balanced
    criterion the epoch does, so the holdout the verdict rests on is still
    untouched by either choice.
    """
    mean, scale = _standardise(x)
    val_x, val_y = validation
    val_xs = (val_x - mean) / scale if val_x.shape[0] else val_x
    best: tuple[np.ndarray, np.ndarray, int, float] | None = None
    best_score = -float("inf")
    for decay in decays:
        weights, bias, epochs = _fit_weights(
            x,
            y,
            n_classes=n_classes,
            log_prior=log_prior,
            validation=validation,
            frequent=frequent,
            batch_size=batch_size,
            learning_rate=learning_rate,
            l2=float(decay),
            seed=seed,
        )
        score = _balanced_log_likelihood(
            val_xs,
            val_y,
            weights * scale,
            bias + (weights @ mean),
            frequent=frequent,
        )
        if score > best_score:
            best_score = score
            best = (weights, bias, epochs, float(decay))
    if best is None:
        return (
            np.zeros((n_classes, STATE_DIM)),
            log_prior.astype(np.float64).copy(),
            0,
            float(decays[0]) if decays else 0.0,
        )
    return best


def _score(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    top_k: int,
    frequent: np.ndarray,
) -> Scores:
    if x.shape[0] == 0:
        return Scores(0.0, 0.0, 0.0, 0.0)
    logits = x @ weights.T + bias
    log_probabilities = _log_softmax(logits)
    truth = log_probabilities[np.arange(len(y)), y]
    k = max(1, min(int(top_k), logits.shape[1]))
    ranked = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    hits = float(np.mean(np.any(ranked == y[:, None], axis=1)))
    is_frequent = frequent[y]
    frequent_ll = float(np.mean(truth[is_frequent])) if np.any(is_frequent) else 0.0
    rare_ll = float(np.mean(truth[~is_frequent])) if np.any(~is_frequent) else 0.0
    return Scores(
        log_likelihood=float(np.mean(truth)),
        top_k=float(hits),
        log_likelihood_frequent=frequent_ll,
        log_likelihood_rare=rare_ll,
    )


def _likelihoods(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    bias: np.ndarray,
    *,
    frequent: np.ndarray,
) -> tuple[float, float, float]:
    """Overall, frequent-token and rare-token mean log-likelihood.

    Split out from ``_score`` because the permutation loop runs it four
    hundred times and has no use for the top-k ranking, which is the
    expensive half.
    """
    if x.shape[0] == 0:
        return 0.0, 0.0, 0.0
    logits = (x @ weights.T + bias) if weights is not None else np.broadcast_to(
        bias, (x.shape[0], bias.shape[0])
    )
    truth = _log_softmax(np.asarray(logits))[np.arange(len(y)), y]
    is_frequent = frequent[y]
    return (
        float(np.mean(truth)),
        float(np.mean(truth[is_frequent])) if np.any(is_frequent) else 0.0,
        float(np.mean(truth[~is_frequent])) if np.any(~is_frequent) else 0.0,
    )


def _permute_turn_states(
    turns: Sequence[TurnTokens], *, seed: int
) -> list[TurnTokens]:
    """Same turns, same states, wrong correspondence."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(turns))
    return [
        TurnTokens(state=turns[order[i]].state, tokens=turns[i].tokens)
        for i in range(len(turns))
    ]


def fit_vocab_head(
    turns: Sequence[TurnTokens],
    *,
    vocab_size: int,
    tokenizer_signature: str,
    batch_size: int = 2048,
    learning_rate: float = 0.01,
    decays: Sequence[float] = DECAY_GRID,
    top_k: int = 20,
    seed: int = 913,
    permutations: int = PERMUTATIONS,
    null_refits: int = NULL_REFITS,
    temporal: bool = False,
) -> VocabFit | None:
    """Fit the head, test it against its own shuffles, and return the verdict.

    ``temporal=True`` holds out the END of the corpus instead of a random
    sample of it. A random split lets a held-out reply sit between training
    replies IN TIME, and endogenous state drifts slowly while topics cluster —
    so a neighbouring turn can carry both a similar state and similar words
    without either causing the other. Grouping by reply stops the same string
    being memorised; only ordering stops the same afternoon being memorised.
    Turns must already be in recording order.
    """
    if len(turns) < MIN_TURNS:
        logger.warning(
            "Refusing to fit an endogenous head on %d turns (minimum %d).",
            len(turns),
            MIN_TURNS,
        )
        return None

    if temporal:
        train_turns, validation_turns, holdout_turns = _forward_in_time_split(turns)
    else:
        train_turns, validation_turns, holdout_turns = _three_way_split(turns, seed=seed)
    if not holdout_turns or not validation_turns or not train_turns:
        return None

    counts: dict[int, int] = {}
    for turn in train_turns:
        for token in turn.tokens:
            counts[int(token)] = counts.get(int(token), 0) + 1
    active = np.asarray(
        sorted(t for t, c in counts.items() if c >= MIN_TOKEN_COUNT), dtype=np.int64
    )
    if active.size < 2:
        logger.warning(
            "Refusing to fit: only %d tokens cleared the occurrence floor.", active.size
        )
        return None
    remap = {int(token): i for i, token in enumerate(active)}

    train_x, train_y = _flatten(train_turns, remap)
    val_x, val_y = _flatten(validation_turns, remap)
    test_x, test_y = _flatten(holdout_turns, remap)
    if train_x.shape[0] == 0 or test_x.shape[0] == 0 or val_x.shape[0] == 0:
        return None

    occurrence = np.asarray([counts[int(t)] for t in active], dtype=np.float64)
    log_prior = np.log(occurrence / occurrence.sum())

    # The frequent end of the vocabulary, taken from the corpus itself: the
    # smallest set of token types accounting for FUNCTION_MASS_SHARE of all
    # occurrences. In any natural corpus that is the function words, and it
    # gets there without anyone writing an English stop list.
    order = np.argsort(-occurrence)
    cumulative = np.cumsum(occurrence[order]) / occurrence.sum()
    frequent = np.zeros(active.size, dtype=bool)
    frequent[order[: int(np.searchsorted(cumulative, FUNCTION_MASS_SHARE)) + 1]] = True

    weights, bias, epochs_used, decay_used = _fit_with_selected_decay(
        train_x,
        train_y,
        n_classes=active.size,
        log_prior=log_prior,
        validation=(val_x, val_y),
        frequent=frequent,
        batch_size=batch_size,
        learning_rate=learning_rate,
        decays=decays,
        seed=seed,
    )
    trained_scores = _score(test_x, test_y, weights, bias, top_k=top_k, frequent=frequent)
    unigram_scores = _score(
        test_x, test_y, np.zeros_like(weights), log_prior, top_k=top_k, frequent=frequent
    )

    # The random projection, built exactly as SubstrateTokenGenerator builds
    # it — the thing this head replaces, not an idealised version of it.
    baseline_rng = np.random.default_rng(913 + STATE_DIM * 31 + active.size)
    projection = baseline_rng.standard_normal((active.size, STATE_DIM)) / math.sqrt(
        STATE_DIM
    )
    projection_scores = _score(
        test_x, test_y, projection, np.zeros(active.size), top_k=top_k, frequent=frequent
    )

    # The main null: the same fitted head, scored against shuffles of which
    # held-out state belongs to which held-out turn.
    overall_nulls: list[float] = []
    rare_nulls: list[float] = []
    frequent_nulls: list[float] = []
    for shuffle in range(max(0, int(permutations))):
        shuffled = _permute_turn_states(holdout_turns, seed=seed + 7919 + shuffle)
        sx, sy = _flatten(shuffled, remap)
        if sx.shape[0] == 0:
            continue
        scores = _likelihoods(sx, sy, weights, bias, frequent=frequent)
        base = _likelihoods(sx, sy, None, log_prior, frequent=frequent)
        overall_nulls.append(scores[0] - base[0])
        frequent_nulls.append(scores[1] - base[1])
        rare_nulls.append(scores[2] - base[2])

    # The capacity control: refit from scratch on permuted training states.
    refit_nulls: list[Scores] = []
    for refit in range(max(0, int(null_refits))):
        permuted_train = _permute_turn_states(train_turns, seed=seed + 101 + refit)
        permuted_val = _permute_turn_states(validation_turns, seed=seed + 211 + refit)
        px, py = _flatten(permuted_train, remap)
        pvx, pvy = _flatten(permuted_val, remap)
        if px.shape[0] == 0 or pvx.shape[0] == 0:
            continue
        null_w, null_b, _, _ = _fit_with_selected_decay(
            px,
            py,
            n_classes=active.size,
            log_prior=log_prior,
            validation=(pvx, pvy),
            frequent=frequent,
            batch_size=batch_size,
            learning_rate=learning_rate,
            decays=decays,
            seed=seed + 101 + refit,
        )
        null_scores = _score(
            test_x, test_y, null_w, null_b, top_k=top_k, frequent=frequent
        )
        refit_nulls.append(null_scores)

    fit = VocabFit(
        weights=weights,
        bias=bias,
        active_tokens=active,
        vocab_size=int(vocab_size),
        n_turns_train=len(train_turns),
        n_turns_validation=len(validation_turns),
        n_turns_holdout=len(holdout_turns),
        state_variance=varying_dimensions(turns),
        n_replies_holdout=_distinct_replies(holdout_turns),
        n_replies_train=_distinct_replies(train_turns),
        n_replies_total=_distinct_replies(turns),
        n_tokens_train=int(train_x.shape[0]),
        n_tokens_holdout=int(test_x.shape[0]),
        epochs_used=int(epochs_used),
        decay_used=float(decay_used),
        trained=trained_scores,
        unigram=unigram_scores,
        random_projection=projection_scores,
        overall_test=_permutation_test(
            trained_scores.log_likelihood - unigram_scores.log_likelihood, overall_nulls
        ),
        rare_test=_permutation_test(
            trained_scores.log_likelihood_rare - unigram_scores.log_likelihood_rare,
            rare_nulls,
        ),
        frequent_test=_permutation_test(
            trained_scores.log_likelihood_frequent
            - unigram_scores.log_likelihood_frequent,
            frequent_nulls,
        ),
        refit_null_scores=tuple(refit_nulls),
        top_k=int(max(1, min(int(top_k), active.size))),
        layout=layout_digest(),
        semantics=semantics_digest(),
        tokenizer=str(tokenizer_signature),
    )
    logger.info(
        "Endogenous head fitted on %d turns / %d tokens (%d epochs): %+.4f nats "
        "over unigram, p=%.4f; rare gain %+.4f, p=%.4f — verdict %s.",
        fit.n_turns_train,
        fit.n_tokens_train,
        fit.epochs_used,
        fit.improvement_nats,
        fit.overall_test.p_value,
        fit.rare_improvement_nats,
        fit.rare_test.p_value,
        fit.verdict,
    )
    return fit


__all__ = [
    "ALPHA",
    "DECIDING_TESTS",
    "FUNCTION_MASS_SHARE",
    "HOLDOUT_FRACTION",
    "MAX_EPOCHS",
    "MIN_TOKEN_COUNT",
    "MIN_TURNS",
    "MIN_HOLDOUT_REPLIES",
    "MIN_TRAIN_REPLIES",
    "varying_dimensions",
    "DECAY_GRID",
    "MIN_REFITS_TO_GATE",
    "NULL_REFITS",
    "PERMUTATIONS",
    "VALIDATION_FRACTION",
    "VERDICT_MEANING",
    "PermutationTest",
    "Scores",
    "TurnTokens",
    "VocabFit",
    "fit_vocab_head",
    "tokenize_pairs",
]
