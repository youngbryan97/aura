"""A cortex campaign cannot recover from a wrong answer once it has started.

The offline organism answers with a deterministic stub, which is right for
paired causal measurement and removes every path that runs through real
language generation. A whole-Aura claim needs a second campaign through the
real cortex, and that one has requirements that are not visible from inside it:
a sampler left free, a chat template that changed between the arms, a fallback
model serving one arm and the resident weights the other.

So they are checked before the run and without loading anything. Every answer
is ready, blocked with the reason, or unknown — and unknown is not ready,
because a campaign that cannot say what served it cannot claim what it served.

It found one immediately. The resident weights ship a generation config of
temperature 1.0, top_p 0.95 and top_k 20. That sampler is larger than any
displacement the experiment makes, so a cortex campaign has to fix it
explicitly rather than inherit it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.cortex_campaign_preflight import DETERMINISTIC, _chat_template, _check, preflight

pytestmark = pytest.mark.unit


def test_unknown_is_not_ready() -> None:
    assert _check("x", None, "")["status"] == "unknown"
    assert _check("x", False, "")["status"] == "blocked"
    assert _check("x", True, "")["status"] == "ready"


def test_an_unknown_check_keeps_the_scope_at_substrate_only() -> None:
    """A campaign that cannot say what served it cannot claim what it served."""
    from tools import cortex_campaign_preflight as pre

    source = Path(pre.__file__).read_text(encoding="utf-8")
    assert '"cortex_inclusive" if not blocked and not unknown else "substrate_only"' in source


def test_greedy_is_what_a_paired_arm_needs() -> None:
    assert DETERMINISTIC["temperature"] == 0.0
    assert DETERMINISTIC["top_k"] == 1


def test_the_template_is_found_in_either_place_a_model_carries_one(tmp_path) -> None:
    """Looking only in the tokenizer config reported a blocker on a model that has one."""

    class Spec:
        model_path = tmp_path

    (tmp_path / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    beside = _chat_template(Spec())
    assert beside["status"] == "ready"
    assert beside["source"] == "chat_template.jinja"

    (tmp_path / "chat_template.jinja").unlink()
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    inline = _chat_template(Spec())
    assert inline["status"] == "ready"
    assert inline["source"] == "tokenizer_config.json"
    assert inline["sha256"] == beside["sha256"]


def test_no_template_anywhere_is_a_blocker(tmp_path) -> None:
    class Spec:
        model_path = tmp_path

    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert _chat_template(Spec())["status"] == "blocked"


def test_the_preflight_runs_without_loading_a_model() -> None:
    """It plans a second cortex beside the live one; it does not start one."""
    report = preflight()
    assert set(report) >= {"checks", "blocked", "unknown", "scope_a_run_could_claim"}
    assert report["scope_a_run_could_claim"] in {"substrate_only", "cortex_inclusive"}


def test_every_requirement_a_run_cannot_recover_from_is_checked() -> None:
    names = {row["check"] for row in preflight()["checks"]}
    for wanted in (
        "active pointer", "tokenizer", "chat template",
        "deterministic decoding", "no fallback lane", "recurrent state restore",
    ):
        assert wanted in names
