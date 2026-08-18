"""Work that finished is not work that is outstanding.

LIVE, 2026-08-18: "Swarm internal monologue step completed" sat in
`active_goal_details`, alongside "Reconcile executive failure: sync_approved".

Every obligation in that list holds the autonomy gates shut — the same
mechanism that, earlier the same day, meant she had never once spoken on her
own. A sentence whose own subject is its completion cannot be a thing still to
do. It is a receipt, and receipts were being counted as debts.
"""

from __future__ import annotations

import pytest

from core.autonomy.research_goal_filter import (
    is_completion_record,
    is_stale_or_prompt_scaffold_goal,
)


@pytest.mark.parametrize(
    "record",
    [
        "Swarm internal monologue step completed",
        "Reconcile executive failure: sync_approved",
        "Reconcile executive failure: approved",
        "memory consolidation finished",
        "the migration completed",
        "verification pass done",
        "consolidation cycle finished",
    ],
)
def test_a_finished_thing_is_not_an_obligation(record):
    assert is_completion_record(record), record
    assert is_stale_or_prompt_scaffold_goal(record), record


@pytest.mark.parametrize(
    "goal",
    [
        "Complete the migration",
        "Finish the report",
        "Close the open incident",
        "Resolve the naming conflict",
        "Approve the pending change",
    ],
)
def test_an_imperative_keeps_its_place(goal):
    """"Complete the migration" is a thing to do; "migration completed" is not.

    The discriminator is grammatical: the verb leads an instruction and trails
    a record.
    """
    assert not is_completion_record(goal), goal
    assert not is_stale_or_prompt_scaffold_goal(goal), goal


@pytest.mark.parametrize(
    "goal",
    [
        "Find out who wrote the novel Solaris",
        "Think deeply about distributed systems consensus",
        "Investigate why the disk keeps filling up",
    ],
)
def test_real_goals_survive(goal):
    assert not is_stale_or_prompt_scaffold_goal(goal), goal
