"""Properties that only appear over many steps.

A counterfactual asks what one appraisal does; an ablation asks whether
one faculty matters. Neither can see whether a bond decays the way a bond
should, whether an anniversary still lands after a quiet year, or whether
someone who started complying can get their standing back.

Those are the claims most likely to be wrong, because they are the ones
nobody exercises by hand.
"""

from __future__ import annotations

import pytest

from core.interiority.faculties import load_all
from core.interiority.longitudinal import EPISODES, run_episodes


@pytest.fixture(scope="module", autouse=True)
def _loaded() -> None:
    load_all()


def test_every_long_running_property_holds() -> None:
    failures = [r for r in run_episodes() if not r.held]
    assert not failures, "\n".join(
        f"{r.name}: {r.question}\n    {r.detail}" for r in failures
    )


def test_the_episodes_ask_a_question_each() -> None:
    for episode in EPISODES:
        assert episode.question.endswith("?"), episode.name
        assert len(episode.question) > 30, episode.name
