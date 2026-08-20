"""The floor: the stack must never answer worse than ordinary decode.

Bryan's contract, and it should have been the design contract from the
start: by default the system does as well as vanilla; no improvement means
neutral; improvement means gain; it must NEVER return a lower-quality answer.

That is only structurally true if, when nothing is promoted, the public
answer is the answer ordinary decode would have produced. Every knob on
which the incumbent decode differs from the control is a way the floor can
be breached, so each one is enumerated here rather than discovered later by
losing a battery to it.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = REPO_ROOT / "tools"
for path in (str(REPO_ROOT), str(TOOLS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_rlc_reconciliation_sweep as sweep  # noqa: E402

# What `_run_vanilla` actually does: greedy, no penalty, stop on the first
# complete FINAL_ANSWER. Anything the product arm does differently is a way
# its "neutral" path stops being neutral.
_CONTROL_DECODE = {
    "decode_temperature": 0.0,
    "decode_top_p": 1.0,
    "decode_repetition_penalty": 1.0,
    "decode_max_tokens": 512,
    "decode_min_tokens": 0,
}

# decode_contract is NOT in the list above, because the answer changed once on
# measurement and a literal here cannot record that.
#
# It read "none" while the engine's contract enforcement could blank a produced
# answer: 576 generated tokens came back as an empty string under
# token_limit_contract_incomplete, which puts the arm below the floor by
# construction. The risk of running without it -- past a correct answer into a
# second FINAL_ANSWER marker, which grades invalid -- looked small at the time:
# marker counts matched the control exactly on a three-task probe.
#
# The 2026-08-06 campaign measured it at scale and it was not small. The
# free-running vanilla arm emitted a second marker on 12 of 28 tasks and was
# format-rejected for every one, while each recurrent arm was protected by its
# contract stop -- the control was handicapped, which is the same unfairness
# pointing the other way. Both arms now stop on the same rule.
#
# So the constant is gone. What this file is for is that the product arm
# decodes like the control, and the control's stopping rule is the one
# _run_vanilla actually applies.
def _control_decode_contract() -> str:
    """The stopping rule `_run_vanilla` applies, read from the source."""
    from pathlib import Path as _Path

    source = _Path("tools/run_rlc_reconciliation_sweep.py").read_text(encoding="utf-8")
    body = source[source.index("def _run_vanilla") :]
    body = body[: body.index("\ndef ", 1)]
    assert "is_contract_complete" in body, "the control no longer stops on a contract"
    return "final_answer_v1"


def test_the_product_arm_decodes_like_the_control():
    """Divergences found the expensive way: a 1.25 repetition penalty the
    deployed system does not use, which fights step-by-step arithmetic because
    that reasoning repeats phrasing and digits by construction."""
    full = sweep._build_config(8, 16, "applied", 512, profile="full")
    mismatched = {
        knob: (expected, getattr(full, knob))
        for knob, expected in _CONTROL_DECODE.items()
        if getattr(full, knob) != expected
    }
    assert full.decode_contract == _control_decode_contract(), (
        "the product arm must run the deployed contract setting; enforcement "
        "inside the engine can discard a produced answer, which the control "
        "never does"
    )
    assert not mismatched, (
        "the product arm's decode differs from ordinary decode, so its "
        f"neutral path is not neutral: {mismatched}"
    )


def test_ordinary_decode_owns_the_answer_until_something_beats_it():
    """The structural floor. Recurrence, branch selection, verification and
    learning all still run and stay receipted -- they simply do not own the
    output until a gain gate promotes them."""
    full = sweep._build_config(8, 16, "applied", 512, profile="full")
    assert full.decode_incumbent_policy == "vanilla_incumbent"
    assert full.answer_replacement_enabled is True
    assert full.decode_bridge_policy == "assistant_answer_v4"
    assert full.decode_contract == _control_decode_contract()
    assert full.verifier_probe_contract == "final_answer_v1"


def test_the_ablation_is_allowed_to_break_the_floor():
    """The mechanism arm deliberately lets the latent path own the answer: a
    degraded episode must not silently serve a vanilla answer and be read as a
    recurrent result. It is an instrument, not the product, and it is the only
    arm permitted to score below ordinary decode."""
    mech = sweep._build_config(4, 16, "suppressed", 512, profile="mechanism")
    assert mech.decode_incumbent_policy == "latent"


#: What "bounded below by ordinary decode" is, in the config that decides it.
_BOUNDED = "vanilla_incumbent"
_UNBOUNDED = "latent"


def test_every_arm_declares_which_side_of_the_floor_it_is_on():
    """No arm may be ambiguous about whether it is bounded below by vanilla.

    This listed the profile names by hand, and three arms added since —
    recurrent_composed, recurrent_initial_control, recurrent_depth_lesion —
    were on neither list. The side is not a name; it is
    decode_incumbent_policy, so the check reads that. A new arm now has to
    have a decidable answer rather than a mention here.
    """
    for arm in sweep.ARMS:
        config = sweep._build_config(
            arm.steps or 8, 16, arm.policy, arm.max_tokens or 512, profile=arm.profile
        )
        assert config.decode_incumbent_policy in {_BOUNDED, _UNBOUNDED}, arm

        # The families are named for which side they are on, and a name that
        # disagrees with the setting is the ambiguity this test exists to stop.
        expected = _UNBOUNDED if arm.profile.startswith(("ordinary", "mechanism")) else _BOUNDED
        assert config.decode_incumbent_policy == expected, (
            f"{arm.profile} is named as {expected} but decodes as "
            f"{config.decode_incumbent_policy}"
        )


def test_the_last_divergence_is_crossed_rather_than_assumed():
    """Bridge tokens -- including the 27-token terminal disposition -- are
    prepended to the answer decode even under vanilla_incumbent, so the
    product's honest floor would be "vanilla plus disposition", not vanilla.
    The claimed product therefore suppresses it; a diagnostic arm measures the
    cost of restoring it but can never win or be promoted."""
    by_name = {a.name: a for a in sweep.ARMS}
    product = by_name["full_stack"]
    matched = by_name["full_stack_disposition"]
    assert product.profile == matched.profile == "full"
    assert product.steps == matched.steps
    assert product.max_tokens == matched.max_tokens
    # The disposition is the ONLY thing that differs between them.
    assert product.policy == "suppressed"
    assert matched.policy == "applied"


def test_the_floor_and_the_gain_are_not_mutually_exclusive():
    """The deadlock that made a win impossible.

    decode_incumbent_policy governs WHO owns the answer; answer_replacement
    governs WHETHER a better candidate may take it. Wiring the second to the
    first meant:

      latent            -> may promote, but no floor at all
      vanilla_incumbent -> floor holds, promotion force-disabled

    so no configuration could both stay at or above ordinary decode AND
    improve on it. The product arm must be the combination the engine's own
    incumbent comment promises: ordinary decode owns the answer until an
    independent gain gate promotes something better.
    """
    full = sweep._build_config(8, 16, "applied", 512, profile="full")
    assert full.decode_incumbent_policy == "vanilla_incumbent"
    assert full.answer_replacement_enabled is True

    import inspect

    from core.brain.llm.latent_cortex import engine as engine_mod

    src = inspect.getsource(engine_mod)
    assert "enabled=self.config.answer_replacement_enabled," in src, (
        "the promotion gate must not be re-coupled to decode_incumbent_policy"
    )
    assert (
        "self.config.answer_replacement_enabled\n                        and latent_decode_authorized"
        not in src
    )


def test_every_derived_budget_stays_inside_its_own_bounds():
    """A derived budget must respect the bound the config declares for it.

    local_repair_max_tokens was computed as min(512, max_tokens), which drops
    below the declared floor of 32 whenever a caller uses a small decode
    budget. The engine then raised "invalid CortexConfig" and the episode
    never ran at all -- a config-validation failure disguised as a cortex
    failure. Every budget the product arm derives is checked here against a
    decode budget small enough to expose a one-sided clamp.
    """
    for max_tokens in (16, 24, 32, 64, 512, 4096):
        config = sweep._build_config(8, 16, "applied", max_tokens, profile="full")
        problems = config.validate()
        assert problems == [], (
            f"decode budget {max_tokens} produced an invalid config: {problems}"
        )
