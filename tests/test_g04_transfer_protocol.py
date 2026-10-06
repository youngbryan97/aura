"""G04's protocol freezes its effect from evidence and its runner grades without the models."""

from __future__ import annotations

import pytest

from core.learning.procedure_induction import Instruction, Program
from tools import g04_transfer_protocol as protocol
from tools import run_g04_transfer as runner


def test_the_planned_effect_is_the_lower_bound_of_the_composition_evidence() -> None:
    discordance, win_share = protocol.planned_effect()
    assert 0.25 < discordance < 34 / 96
    assert 0.9 < win_share < 1.0
    tasks = protocol.tasks_needed(alpha=0.05 / 4, target=0.9, discordance=discordance,
                                  win_share=win_share)
    assert tasks == 62


def test_a_decoder_that_executes_a_program_is_caught() -> None:
    program = Program(n_inputs=2, instructions=(Instruction("add", (0, 1)),))

    class Executes:
        def decode(self, **_kwargs):
            return program.run((2, 3))

    class Item:
        class ir:
            source_token_ids = ()
            source_text_sha256 = "x"
            model_basis_receipt_sha256 = "y"

        hidden_states = None
        public_inputs = (2, 3)

    outcome, failure = runner.decode(Executes(), Item())
    assert outcome is None
    assert failure.startswith("execution_during_decode")
    assert program.run((2, 3)) == 5, "execution is available again once the decode is done"


def test_a_runtime_failure_counts_against_the_arm() -> None:
    class Breaks:
        def decode(self, **_kwargs):
            raise KeyError("lost")

    class Item:
        class ir:
            source_token_ids = ()
            source_text_sha256 = "x"
            model_basis_receipt_sha256 = "y"

        hidden_states = None
        public_inputs = ()

    outcome, failure = runner.decode(Breaks(), Item())
    assert outcome is None and failure.startswith("runtime_failure:KeyError")


def test_the_exact_test_counts_only_disagreements() -> None:
    def row(candidate: bool, incumbent: bool) -> dict:
        return {"candidate": {"equivalent": candidate}, "incumbent": {"equivalent": incumbent}}

    rows = {f"t{i}": row(True, False) for i in range(9)}
    rows |= {"same1": row(True, True), "same2": row(False, False), "lost": row(False, True)}
    result = runner.mcnemar(rows, "candidate", "incumbent")
    assert (result["treatment_wins"], result["control_wins"]) == (9, 1)
    assert result["exact_one_sided_p"] == pytest.approx(11 / 1024)


def test_a_signature_found_in_a_consumed_request_fails_the_novelty_check() -> None:
    from core.learning.semantic_g04_transfer_corpus import build_g04_transfer_corpus

    strata = build_g04_transfer_corpus(seed=3, tasks_per_stratum=6)
    clean = {"examples": [{"source_text": "First, add 2 and 3.", "source_sha256": "0" * 64}],
             "max_depth": 5, "sequence_operations_with_computed_arguments": 0}
    assert protocol.novelty_report(strata, clean, lambda _n: 7)["failures"] == []
    leaked = dict(clean, examples=[{"source_text": "Step one: add 2 and 3.", "source_sha256": "1" * 64}])
    assert protocol.novelty_report(strata, leaked, lambda _n: 7)["failures"] == ["construction"]
    deeper = dict(clean, max_depth=7)
    assert "depth" in protocol.novelty_report(strata, deeper, lambda _n: 7)["failures"]
