"""A percentage or a multiple she derived from what her run returned is worked out again from it (U07-08).

LIVE 18 November 2026: every file, line count and function count in her answer matched her run, and the same answer
said the gap from third to fourth place was "a comfortable ~75%". Her run had returned 10,397 and 10,215: 1.8%.
"""
from __future__ import annotations

import pytest

from core.epistemics.what_her_numbers_come_to import derived_figures, figures_her_run_does_not_give, numbers_in

pytestmark = pytest.mark.unit

RUN = """core/brain/llm/mlx_client.py 12589 324
core/brain/inference_gate.py 11227 199
core/brain/llm/latent_cortex/engine.py 10397 133
core/brain/llm/mlx_worker.py 10215 188
core/brain/cognitive_engine.py 3982 106"""
RAN = [{"tool": "code_repl", "ok": True, "as_made": {"arguments": {"code": "..."}, "returned": RUN}}]
TABLE = ("| `core/brain/llm/mlx_client.py` | 12,589 | **324** |\n| `core/brain/inference_gate.py` | 11,227 | 199 |\n"
         "| `core/brain/llm/latent_cortex/engine.py` | 10,397 | 133 |\n")
LIVE = ("For context, the next ones down were `llm/mlx_worker.py` (10,215 lines, 188 functions) and top-level "
        "`cognitive_engine.py` (3,982 lines, 106), so the gap between #3 and #4 is a comfortable ~75% in line count.")


def _judged(sentence: str, ran=RAN) -> list:
    return figures_her_run_does_not_give([sentence], ran, f"{TABLE}{LIVE} {sentence}")


def test_numbers_are_read_as_written():
    assert numbers_in("12,589 lines, 324 functions, 1.8% of it") == [12589.0, 324.0, 1.8]
    assert [(f.value, f.kind, f.tolerance) for f in derived_figures("~75% more, 2.6x faster, a 1920x1080 screen")] == [
        (75.0, "percent", 0.5), (2.6, "times", 0.05)]
    assert derived_figures("about 70% of it")[0].tolerance == 5.0, "a round figure is rounded to its tens"


def test_the_live_figure_is_refuted_and_the_record_names_what_it_was_worked_from():
    refuted = _judged(LIVE)
    assert [claim.sentence for claim in refuted] == [LIVE]
    assert refuted[0].said().startswith("No comparison of the figures my run returned that I gave (12,589, 324,")
    assert refuted[0].said().endswith("comes to 75%.")


@pytest.mark.parametrize("sentence", [
    "The gap between #3 and #4 is about 1.8% in line count.",
    "mlx_client.py has about 63% more functions than inference_gate.py.",
    "mlx_client.py holds 37% of the three files' lines.",
    "mlx_client.py has 2.4x the functions of engine.py.",
    "That makes mlx_client.py 12% larger than inference_gate.py.",
    "cognitive_engine.py is 39% of mlx_worker.py's length.",
])
def test_a_figure_some_comparison_of_hers_gives_stands(sentence):
    assert _judged(sentence) == []


def test_a_false_multiple_is_refuted():
    assert _judged("mlx_client.py is 5x the size of inference_gate.py.")


def test_a_figure_her_run_printed_stands():
    ran = [{"tool": "code_repl", "ok": True, "as_made": {"returned": RUN + "\nshare of the largest: 75%"}}]
    assert _judged("The largest holds 75% of it.", ran) == []


def test_nothing_is_judged_without_two_of_her_run_figures_or_a_run_that_came_back():
    assert figures_her_run_does_not_give(["It grew 75% since then."], RAN, "It grew 75% since then.") == []
    failed = [{"tool": "code_repl", "ok": False, "observed_content": RUN}]
    assert _judged(LIVE, failed) == []


def test_her_record_check_takes_it_out_and_carries_the_record_to_her_turn_asked_again(monkeypatch):
    import core.epistemics.what_her_record_refutes as record
    from core.brain.her_turn_as_asked import a_turn_is_asked, what_the_record_says
    from interface.routes.chat_turn_evidence import _correct_unevidenced_action_claims

    reply = f"{TABLE}\n{LIVE}"
    monkeypatch.setattr(record, "_sentences_of", lambda text: [TABLE.strip(), LIVE])
    monkeypatch.setattr(record, "what_her_record_refutes", lambda *a, **k: [])
    refuted = record.sentences_these_calls_refute(reply, RAN, "Find the three largest files and count them.")
    assert [claim.sentence for claim in refuted] == [LIVE]
    with a_turn_is_asked():
        kept = str(_correct_unevidenced_action_claims(reply, "Find the three largest files.", refuted))
        assert LIVE not in kept and "12,589" in kept
        assert any("comes to 75%" in fact for fact in what_the_record_says())
