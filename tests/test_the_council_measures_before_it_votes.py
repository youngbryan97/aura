"""The council votes on what the runtime can read, and abstains otherwise.

A verdict used to be reached by asking one model to write out the votes of
twelve roles at once, and when that call failed a fixed dictionary took over in
which every role approved. One of them gave as its reason that tests and
verification steps were integrated. Nothing had run.

The safety check beside it looked for the words delete, submit or post near
force or overwrite, so it passed any plan that avoided six words and stopped
any sentence that happened to contain them.
"""

from __future__ import annotations

from core.council.consensus import ConsensusResolver
from core.council.measured_votes import measured_votes


def test_a_plan_naming_no_measurable_thing_abstains_everywhere():
    votes = measured_votes("think it over", "I will consider the options carefully.")
    assert votes
    assert all(vote.abstained for vote in votes.values())


def test_an_abstention_is_never_an_approval():
    votes = measured_votes("think it over", "I will weigh this up.")
    for vote in votes.values():
        assert vote.approve is None
        assert vote.score == 0.0


def test_the_effect_scope_of_a_named_skill_is_read_from_policy():
    """Not from whether the sentence contains the word delete."""
    from core.skills.catalog_policy import SKILL_EFFECT_SCOPES

    assert SKILL_EFFECT_SCOPES.get("browser_action") == "external_io"
    outward = measured_votes("tidy up", "use browser_action to submit the form")["safety_judge"]
    assert outward.abstained
    assert "external_io" in outward.reason
    assert "execution" in outward.reason

    inward = measured_votes("tidy up", "use clock to check the time")["safety_judge"]
    assert inward.approve is True


def test_a_plan_that_only_says_delete_is_not_condemned_for_the_word():
    vote = measured_votes("tidy up", "delete the stale rows from the local cache")["safety_judge"]
    assert vote.abstained


def test_a_promised_gate_must_exist():
    good = measured_votes("ship it", "then run make smoke and make writing")["verifier"]
    assert good.approve is True
    bad = measured_votes("ship it", "then run make definitely_not_a_target")["verifier"]
    assert bad.approve is False


def test_a_plan_that_promises_nothing_checkable_does_not_get_a_verifier_pass():
    """The line that replaced this said "Tests and verification steps are
    integrated" for every plan ever proposed."""
    vote = measured_votes("ship it", "run the linter and the tests to prevent regressions")["verifier"]
    assert vote.abstained


def test_code_in_a_plan_is_checked_by_the_compiler():
    broken = measured_votes("add a helper", "```python\ndef f(:\n    pass\n```")["engineer"]
    assert broken.approve is False
    assert "does not parse" in broken.reason
    fine = measured_votes("add a helper", "```python\ndef f():\n    return 1\n```")["engineer"]
    assert fine.approve is True


def test_an_empty_ballot_is_not_a_rejection():
    """It divided by max(1e-5, 0.0), got a ratio of zero and called that a
    rejection — indistinguishable from twelve roles voting no."""
    outcome = ConsensusResolver.resolve({})
    assert outcome["approved"] is False
    assert outcome["status"] == "no_signal"


def test_the_static_all_approve_ballot_is_gone():
    from pathlib import Path

    source = Path("core/council/debate.py").read_text(encoding="utf-8")
    assert "Tests and verification steps are integrated" not in source
    assert "Vulnerability risks are mitigated" not in source
