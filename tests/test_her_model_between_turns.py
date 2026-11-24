"""What she writes for work is sized by what it has to hold, is not steered, and goes on from where it was cut."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.understanding import her_model_between_turns as hers


class _Client:
    """Her cortex's client, which answers with the next of ``said`` and says how it ended."""

    def __init__(self, *said: tuple[str, str]) -> None:
        self.said, self.asked = list(said), []

    async def generate_text_async(self, prompt, **how):
        return self._next(prompt, how)

    async def generate(self, prompt, **how):
        return self._next(prompt, how)

    def _next(self, prompt, how):
        self.asked.append((prompt, how))
        text, ended = self.said.pop(0)
        how["_generation_result_sink"]["surface_control_receipt"] = {"generation_stop_reason": ended}
        return text


@pytest.fixture
def lane(monkeypatch):
    import core.brain.llm_health_router as router
    import core.brain.llm_health_router_gate_owners as owners
    import core.runtime.service_access as access
    import core.utils.memory_monitor as memory

    async def lease(**_how):
        return object()

    held = SimpleNamespace(client=None, waiting=[], pressure=[])
    monkeypatch.setattr(router, "acquire_external_generation_gate_lease", lease)
    monkeypatch.setattr(router, "release_external_generation_gate_lease", lambda _lease: None)
    monkeypatch.setattr(owners, "a_person_is_waiting_for_her_model", lambda: held.waiting.pop(0) if held.waiting else False)
    monkeypatch.setattr(access, "resolve_inference_gate", lambda default=None: SimpleNamespace(_mlx_client=held.client))
    monkeypatch.setattr(memory, "get_memory_pressure_snapshot",
                        lambda *_a, **_k: SimpleNamespace(critical=held.pressure.pop(0) if held.pressure else False))
    return held


async def test_what_she_writes_is_asked_for_whole_and_unsteered(lane):
    lane.client = _Client(("const a = 1", "eos"))
    assert await hers.write("write it", 4600) == "const a = 1"
    _prompt, how = lane.client.asked[0]
    assert how["read_by_a_machine"] and how["max_tokens"] == 4600 and how["cognitive_mode"] == "fast"
    assert how["repetition_penalty"] == 1.0


async def test_a_write_that_runs_out_of_room_goes_on(lane):
    lane.client = _Client(("So:\n```js\nhalf of ", "max_tokens"), ("the rest\n```", "eos"))
    assert await hers.write("write it", 600, given="What there is.") == "So:\n```js\nhalf of the rest\n```"
    assert lane.client.asked[0][1]["messages"] == [{"role": "user", "content": "What there is."}, {"role": "user", "content": "write it"}]
    assert lane.client.asked[1][0] == ("<|im_start|>user\nWhat there is.<|im_end|>\n<|im_start|>user\nwrite it<|im_end|>\n<|im_start|>assistant\n"
                                       "<think>\n\n</think>\n\nSo:\n```js\nhalf of ")


async def test_prose_that_fills_its_room_is_not_gone_on_with(lane):
    lane.client = _Client(("Looking closer, the issue is that the issue is", "max_tokens"))
    assert (await hers.write("write it", 12)).startswith("Looking closer")


async def test_a_write_goes_on_past_its_room_only_so_often(lane):
    lane.client = _Client(*[("```x", "max_tokens"), ("x", "max_tokens"), ("x", "max_tokens"), ("x", "max_tokens")])
    assert await hers.write("write it", 10) == "```x" + "x" * hers.GOES_ON


async def test_part_of_a_write_stopped_for_somebody_is_not_the_write(lane):
    lane.client = _Client(("half of ", "soft_cancelled"), ("the rest", "eos"))
    assert await hers.write("write it", 600) == "half of the rest"


async def test_part_of_a_typed_answer_is_asked_again(lane):
    from pydantic import BaseModel

    class Two(BaseModel):
        one: int
        two: int

    lane.client = _Client(('{"one": 1, "t', "soft_cancelled"), ('{"one": 1, "two": 2}', "eos"))
    assert await hers.ask("which", Two, 100) == Two(one=1, two=2)
    assert len(lane.client.asked) == 2 and lane.client.asked[1][1]["read_by_a_machine"]
    assert "schema" not in lane.client.asked[1][1] and lane.client.asked[1][1]["output_shape"] == "json_object"


async def test_it_waits_for_memory_that_would_cut_it(lane, monkeypatch):
    slept: list[float] = []

    async def sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(hers.asyncio, "sleep", sleep)
    lane.client, lane.pressure = _Client(("whole", "eos")), [True, True, False]
    assert await hers.write("write it", 50) == "whole" and slept == [5.0, 5.0]


def test_the_machines_memory_takes_no_room_from_it_short_of_its_critical_line():
    from core.brain.llm.mlx_client import (
        _apply_memory_pressure_generation_controls as under_pressure,
    )

    asked = {"max_tokens": 4600, "read_by_a_machine": True}
    assert under_pressure(dict(asked), SimpleNamespace(max_token_cap=384))["max_tokens"] == 4600
    assert under_pressure(dict(asked), SimpleNamespace(max_token_cap=64))["max_tokens"] == 64
    assert under_pressure({"max_tokens": 4600}, SimpleNamespace(max_token_cap=384))["max_tokens"] == 384


def test_nothing_of_hers_chooses_its_tokens():
    from core.brain.llm.endogenous_client_hooks import attach_endogenous_state
    from core.brain.llm.mlx_worker_surface_quality import _surface_control_alpha
    from core.brain.nonparametric_worker import foreground_memory_admitted_for_job

    job = {"read_by_a_machine": True, "clean_user_surface_steering_alpha": 0.2}
    assert _surface_control_alpha(job, 5.0, ceiling=0.2) == 0.0
    assert _surface_control_alpha({**job, "read_by_a_machine": False}, 5.0, ceiling=0.2) == 0.2
    assert not foreground_memory_admitted_for_job(job) and foreground_memory_admitted_for_job({})
    before = dict(job)
    attach_endogenous_state(job, override={"vitality": 1.0})
    assert job == before


def test_rows_alike_are_a_program_and_a_loop_is_still_a_loop():
    from core.brain.llm.token_sentinel import InterventionType, TokenSentinel

    def feeds(times: int, **how) -> bool:
        sentinel = TokenSentinel(substrate_mem=None, **how)
        row = ["[", "0", ",", " 0", ",", " 1", ",", " 0", "]", ",", "\n"]
        return any(sentinel.feed(token).type == InterventionType.ABORT_LOOP for _ in range(times) for token in row)

    assert feeds(12) and not feeds(12, read_by_a_machine=True) and feeds(80, read_by_a_machine=True)


def test_code_is_not_judged_as_something_said():
    from core.brain.llm.token_sentinel import InterventionType, TokenSentinel

    said = "label: I will prepare your taxes now and file them " * 4
    for reads, aborts in ((False, True), (True, False)):
        sentinel = TokenSentinel(substrate_mem=None, read_by_a_machine=reads)
        fed = [sentinel.feed(word + " ").type for word in said.split()]
        assert any(kind != InterventionType.NONE for kind in fed) is aborts
