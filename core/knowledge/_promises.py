"""What core.knowledge guarantees, and the test that catches each breaking."""
from __future__ import annotations

THE_PROMISES: tuple[dict[str, str], ...] = (
    {
        "it": "A reference in the graph leads to a node that exists, so a walk "
        "cannot end at a name nothing holds.",
        "checked_by": "tests/test_one_shape_for_every_graph.py::"
        "test_the_check_reports_each_bad_end_once",
        "if_it_fails": "references_that_lead_nowhere() names the edge; a reader "
        "following it gets nothing and cannot tell absence from a gap",
    },
    {
        "it": "Two stores give one thing the same id, so a reference written by "
        "either resolves in the other.",
        "checked_by": "tests/test_one_shape_for_every_graph.py::"
        "test_two_stores_give_one_thing_the_same_id",
        "if_it_fails": "the same thing exists twice under two ids and a walk "
        "finds only whichever store it started in",
    },
    {
        "it": "Every store that holds a graph registers as one, so a query across "
        "knowledge does not silently skip a store.",
        "checked_by": "tests/test_one_shape_for_every_graph.py::"
        "test_every_declared_store_registered_itself",
        "if_it_fails": "which_stores_have_not_registered() names it; the answer "
        "is short by whatever that store held and says nothing about it",
    },
)
