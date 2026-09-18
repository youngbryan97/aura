"""She carries an encyclopedia, and a machine with no network costs her nothing.

Asked what something IS she read her own shelf; asked how something is DONE
she went to the web, which is going outside for context that is in the
building — and is the one thing on a local system that cannot be relied on.

So the shelf is read for every question, and the web is an extra she takes
only where there is one. A machine with no network is something she knows
rather than something she finds out by waiting.
"""

from __future__ import annotations

import pytest

from core.agency import task_knowledge as knowing


class _Shelf:
    """Her offline reference, as the capability engine exposes it."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def execute(self, name, params, _context=None):
        if name != "local_reference_search":
            raise AssertionError(f"she went somewhere else: {name}")
        self.asked.append(str(params.get("query") or ""))
        return {
            "results": [
                {"title": "2048 (video game)", "snippet": "Tiles with the same number merge",
                 "source": "wikipedia", "score": 0.9}
            ]
        }


@pytest.fixture(autouse=True)
def _nothing_remembered():
    knowing.forget_everything()
    yield
    knowing.forget_everything()


@pytest.mark.asyncio
async def test_the_shelf_is_read_for_a_question_about_doing_something(monkeypatch):
    shelf = _Shelf()
    went_out: list[str] = []

    async def the_web(question, *, engine=None):
        went_out.append(question)
        return [], question

    monkeypatch.setattr(knowing, "_from_search", the_web)
    monkeypatch.setattr(knowing, "_search_results_for", lambda *a, **k: _none())
    monkeypatch.setattr(knowing, "_there_is_a_network", lambda: False)
    known = await knowing.learn_about(
        "play 2048 until the 2048 tile", engine=shelf, remember=False, because_stuck=True,
    )
    assert shelf.asked, "she never read her own shelf"
    assert went_out == [], "she went out with no network"
    assert any("merge" in finding.says for finding in known.findings)


@pytest.mark.asyncio
async def test_with_a_network_the_web_is_still_an_extra_after_the_shelf(monkeypatch):
    shelf = _Shelf()
    order: list[str] = []

    async def the_web(question, *, engine=None):
        order.append("web")
        return [], question

    async def nothing_else(*_a, **_k):
        return []

    real_shelf = knowing._from_her_own_shelf

    async def watched(question, *, engine=None):
        order.append("shelf")
        return await real_shelf(question, engine=engine)

    monkeypatch.setattr(knowing, "_from_her_own_shelf", watched)
    monkeypatch.setattr(knowing, "_from_search", the_web)
    monkeypatch.setattr(knowing, "_search_results_for", nothing_else)
    monkeypatch.setattr(knowing, "_read_the_best_answer", nothing_else)
    monkeypatch.setattr(knowing, "_there_is_a_network", lambda: True)
    await knowing.learn_about(
        "how do I open the 2048 app", engine=shelf, remember=False, because_stuck=True,
    )
    assert order and order[0] == "shelf"


def test_no_network_is_something_she_knows_rather_than_waits_for(monkeypatch):
    class _Report:
        has_network = False

    class _Discovery:
        def get_report(self):
            return _Report()

    monkeypatch.setattr(
        "core.capabilities.capability_discovery.get_capability_discovery", lambda: _Discovery()
    )
    assert knowing._what_her_machine_says_about_a_network() is False


def test_nothing_known_about_the_network_leaves_her_free_to_try(monkeypatch):
    class _Report:
        has_network = None

    class _Discovery:
        def get_report(self):
            return _Report()

    monkeypatch.setattr(
        "core.capabilities.capability_discovery.get_capability_discovery", lambda: _Discovery()
    )
    assert knowing._what_her_machine_says_about_a_network() is True


async def _none():
    return []
