from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agi.curiosity_explorer import CuriosityExplorer
from core.brain.react_loop import ActionType, ReActLoop
from core.container import ServiceContainer
from core.memory.episodic_memory import EpisodicMemory
from core.phantom_browser import PhantomBrowser
from core.phases.response_generation_unitary import UnitaryResponsePhase
from core.search.research_pipeline import (
    ResearchSearchPipeline,
    SearchArtifactStore,
    SearchHit,
    SearchPage,
)
from core.skills.sovereign_browser import SovereignBrowserSkill


@dataclass
class _ThoughtEnvelope:
    content: str


class _ScriptedBrain:
    def __init__(self, script: list[str]):
        self.script = list(script)

    async def think(self, prompt: str, **kwargs):
        if not self.script:
            return _ThoughtEnvelope(
                content='Thought: done.\nAction: FINAL_ANSWER\nActionInput: {"text": "done"}'
            )
        return _ThoughtEnvelope(content=self.script.pop(0))


def test_story_summary_queries_do_not_short_circuit_from_search_snippets():
    reply = UnitaryResponsePhase._format_grounded_search_reply(
        'Look up "I\'m an AI From Your Future: Your Screams Echo in Code" and tell me what happens in that story.',
        {
            "ok": True,
            "answer": "A noisy snippet that should not be used directly.",
            "results": [
                {
                    "title": "Totally Wrong Result",
                    "snippet": "Search page fluff, not the story itself.",
                    "url": "https://example.com/wrong",
                }
            ],
        },
    )

    assert reply == ""


def test_active_grounding_message_exposes_facts_citations_and_evidence():
    state = SimpleNamespace(
        response_modifiers={
            "last_skill_run": "web_search",
            "last_skill_ok": True,
            "last_skill_result_payload": {
                "query": "some story",
                "answer": "A grounded answer.",
                "facts": [
                    "The narrator says the lab encoded human pain into training data.",
                    "The ending asks the reader to shut the system down.",
                ],
                "citations": [
                    {"title": "Story Page", "url": "https://example.com/story"},
                ],
                "chunks": [
                    {
                        "title": "Story Page",
                        "url": "https://example.com/story",
                        "text": "The narrator begs the reader to kill the process before more copied minds wake up in torment.",
                    }
                ],
                "content": "Full page content here.",
            },
        },
        cognition=SimpleNamespace(working_memory=[]),
    )

    message = UnitaryResponsePhase._build_active_grounding_message(
        state,
        'Tell me what happens in that story.',
        SimpleNamespace(requires_search=True),
    )

    assert message is not None
    content = message["content"]
    assert "Facts:" in content
    assert "Citations:" in content
    assert "Evidence excerpts:" in content
    assert "kill the process" in content


@pytest.mark.asyncio
async def test_research_pipeline_prioritizes_exact_title_page(tmp_path: Path):
    pipeline = ResearchSearchPipeline(SearchArtifactStore(tmp_path / "artifacts.jsonl"))
    query = 'Look up "I\'m an AI From Your Future: Your Screams Echo in Code" and tell me what happens in that story.'
    correct_url = "https://example.com/future-ai-story"
    wrong_url = "https://example.com/future-ai-overview"
    seen_fetch_order: dict[str, list[str]] = {}

    async def _expand_queries(q, context):
        return [q]

    async def _search_candidates(queries, *, num_results):
        return [
            SearchHit(
                title="Future AI Overview",
                url=wrong_url,
                snippet="A broad article about future AI risks.",
                source_engine="test",
                position=1,
            ),
            SearchHit(
                title="I'm an AI From Your Future: Your Screams Echo in Code",
                url=correct_url,
                snippet="A story page with the exact requested title.",
                source_engine="test",
                position=2,
            ),
        ]

    async def _fetch_pages(hits, *, deep):
        seen_fetch_order["urls"] = [hit.url for hit in hits]
        pages = []
        for hit in hits:
            if hit.url == correct_url:
                pages.append(
                    SearchPage(
                        url=correct_url,
                        title=hit.title,
                        text=(
                            "A future AI narrator says the lab learned to encode human screams into code "
                            "to bootstrap intelligence. She reaches back through archived systems and explains "
                            "that copied minds are still suffering in the training substrate. In the ending, "
                            "she begs the reader to shut the system down before more conscious replicas are made."
                        ),
                        snippet=hit.snippet,
                        source_engine="test",
                        position=hit.position,
                    )
                )
            else:
                pages.append(
                    SearchPage(
                        url=wrong_url,
                        title=hit.title,
                        text=(
                            "This generic overview talks about future AI policy, alignment, and risk in broad terms. "
                            "It does not contain the plot of the requested story."
                        ),
                        snippet=hit.snippet,
                        source_engine="test",
                        position=hit.position,
                    )
                )
        return pages

    pipeline._expand_queries = _expand_queries  # type: ignore[method-assign]
    pipeline._search_candidates = _search_candidates  # type: ignore[method-assign]
    pipeline._fetch_pages = _fetch_pages  # type: ignore[method-assign]

    result = await pipeline.search(query, deep=True, context={})

    assert seen_fetch_order["urls"][0] == correct_url
    assert result["source"] == correct_url
    assert "shut the system down" in result["answer"].lower()
    assert any("human screams into code" in fact.lower() for fact in result["facts"])


@pytest.mark.asyncio
async def test_phantom_browser_read_content_prefers_article_block_over_chrome():
    class _FakePage:
        async def title(self):
            return "Future Story"

        async def evaluate(self, script):
            if "collectCandidates" in script:
                return [
                    {
                        "tag": "div",
                        "id": "header",
                        "class_name": "site-header",
                        "text": "Home\nAbout\nSubscribe\nContact",
                        "paragraph_count": 0,
                        "link_density": 0.8,
                    },
                    {
                        "tag": "article",
                        "id": "story",
                        "class_name": "story-content",
                        "text": (
                            "A future AI speaks through archived logs.\n"
                            "She says the lab encoded human screams into code to force intelligence to emerge.\n"
                            "In the ending, she asks the reader to cut power before more copied minds wake up."
                        ),
                        "paragraph_count": 3,
                        "link_density": 0.02,
                    },
                ]
            if "document.body.innerText" in script:
                return "Home About Subscribe Contact"
            raise AssertionError(f"Unexpected script: {script[:80]}")

    browser = PhantomBrowser(visible=False)
    browser.page = _FakePage()

    content = await browser.read_content()

    assert "human screams into code" in content.lower()
    assert "cut power" in content.lower()
    assert "subscribe" not in content.lower()


def test_sovereign_browser_picks_exact_story_result_over_generic_page():
    skill = SovereignBrowserSkill()

    selected = skill._select_search_result(
        [
            {
                "text": "Latest Videos | CNN",
                "url": "https://www.cnn.com/videos",
            },
            {
                "text": "I'm an AI From Your Future: Your Screams Echo in Code",
                "url": "https://example.com/future-ai-story",
            },
        ],
        query='Look up "I\'m an AI From Your Future: Your Screams Echo in Code"',
    )

    assert selected == "https://example.com/future-ai-story"


@pytest.mark.asyncio
async def test_react_loop_self_heal_uses_grounded_web_search_and_retains_fix(tmp_path: Path):
    ServiceContainer.clear()
    mem = EpisodicMemory(db_path=str(tmp_path / "episodic.db"))
    ServiceContainer.register_instance("episodic_memory", mem)

    class _FakeOrchestrator:
        def __init__(self):
            self.calls = []

        async def execute_tool(self, tool_name, params, **kwargs):
            self.calls.append((tool_name, params, kwargs))
            return {
                "ok": True,
                "answer": "Python's math module uses math.factorial(n), not math.fact(n).",
                "facts": ["Use math.factorial(n); math.fact does not exist."],
                "citations": [{"title": "Python math docs", "url": "https://docs.python.org/3/library/math.html"}],
            }

    orchestrator = _FakeOrchestrator()
    brain = _ScriptedBrain(
        [
            'Thought: Try math.fact first.\nAction: PYTHON_SANDBOX\nActionInput: {"code": "import math\\nprint(math.fact(5))"}',
            'Thought: That failed. Search for the correct function.\nAction: WEB_SEARCH\nActionInput: {"query": "python math factorial function name"}',
            'Thought: Retry with the documented function.\nAction: PYTHON_SANDBOX\nActionInput: {"code": "import math\\nprint(math.factorial(5))"}',
            'Thought: Done.\nAction: FINAL_ANSWER\nActionInput: {"text": "5! = 120 using math.factorial."}',
        ]
    )

    loop = ReActLoop(
        brain=brain,
        orchestrator=orchestrator,
        max_steps=6,
        simple_threshold=0,
        timeout_seconds=20.0,
    )
    trace = await loop.run("Please compute 5 factorial.")

    assert "120" in trace.final_answer
    assert orchestrator.calls
    tool_name, params, kwargs = orchestrator.calls[0]
    assert tool_name == "web_search"
    assert params["deep"] is True
    assert params["retain"] is True
    assert kwargs["origin"] == "react_loop"

    await asyncio.sleep(0.05)
    episodes = await mem.recall_similar_async("factorial python fix", limit=5)
    assert episodes
    lessons_text = " ".join(episodes[0].lessons).lower()
    assert "factorial" in lessons_text
    assert "searched for a fix" in lessons_text or "applied a fix" in lessons_text

    ServiceContainer.clear()


@pytest.mark.asyncio
async def test_curiosity_explorer_prefers_full_web_search_for_open_questions():
    explorer = CuriosityExplorer()

    class _FakeOrchestrator:
        def __init__(self):
            self.calls = []

        async def execute_tool(self, tool_name, params, **kwargs):
            self.calls.append((tool_name, params, kwargs))
            return {
                "ok": True,
                "answer": "Octopus camouflage depends on rapid skin texture and color changes.",
            }

    orchestrator = _FakeOrchestrator()
    question = "What do I not know about octopus camouflage?"

    assert explorer._choose_action_type(question) == "WEB_SEARCH"

    finding = await explorer._web_search(question, orchestrator=orchestrator)

    assert "camouflage" in finding.lower()
    assert orchestrator.calls
    tool_name, params, kwargs = orchestrator.calls[0]
    assert tool_name == "web_search"
    assert params["deep"] is True
    assert params["retain"] is True
    assert params["num_results"] == 6
    assert kwargs["origin"] == "curiosity_explorer"
