"""An answer she gave before, given again when asked for anew, is hers, and a turn with no answer is worse."""
from __future__ import annotations

import inspect


def test_a_draft_that_says_again_what_she_said_is_not_asked_for_cooler():
    """Cooled, it is the same words once more. LIVE 25 November 2026, 02:00: the same 500 tokens came twice."""
    from core.brain.llm.mlx_worker_surface_repair import _prepare_clean_retry_kwargs

    again, broken = {"temperature": 0.7, "top_p": 0.9}, {"temperature": 0.7, "top_p": 0.9}
    _prepare_clean_retry_kwargs(again, said_before=True)
    _prepare_clean_retry_kwargs(broken)
    assert again["temperature"] >= 0.7 and again["top_p"] == 0.9 and broken["temperature"] <= 0.35


def test_where_it_comes_again_it_is_served():
    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker)
    served = source.index('"the_same_answer_asked_twice"')
    assert source.rindex('if set(rejection_reasons) == {"repeated_earlier_reply"}:', 0, served) < served < source.index("salvaged = _mlx_worker_loop_part_19(")
    assert 'said_before="repeated_earlier_reply" in rejection_reasons' in source
