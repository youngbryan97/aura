"""The planner's menu and the executor must be the same list.

task_decomposer advertises an AVAILABLE PRIMITIVES block to the planning
model; mission_state._execute_node is what can actually run. Six entries were
on the menu with no handler — including extract_article and
summarize_sources, which is precisely what "find 3 articles, read them, and
write a synthesis" decomposes into. The plan was valid, the model was told
those primitives existed, and the step died as "Unknown action".

Alongside that: {{generated_content}} was emitted by the decomposer in three
places and resolved nowhere, so a heuristic plan wrote that literal string
into the user's PDF; and TaskNode.timeout_s was declared and serialised but
never enforced, so a hung fetch stalled a mission with nothing to show.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from core.planning.mission_state import MissionState
from core.planning.task_graph import TaskGraph, TaskNode


def _advertised_primitives() -> list[str]:
    source = Path("core/planning/task_decomposer.py").read_text(encoding="utf-8")
    block = source.split("AVAILABLE PRIMITIVES:")[1].split("\n\n")[0]
    return re.findall(r"^- (\w+):", block, re.M)


def _implemented_actions() -> set[str]:
    source = Path("core/planning/mission_state.py").read_text(encoding="utf-8")
    actions: set[str] = set()
    for match in re.finditer(r'action (?:==|in) (\("[^)]*"\)|"[^"]+")', source):
        actions |= set(re.findall(r'"(\w+)"', match.group(1)))
    return actions


def test_every_advertised_primitive_has_an_executor_handler():
    """A primitive on the planner's menu that cannot run is a trap."""
    advertised = _advertised_primitives()
    assert advertised, "could not parse the AVAILABLE PRIMITIVES block"
    missing = sorted(set(advertised) - _implemented_actions())
    assert not missing, (
        "task_decomposer offers the planner primitives mission_state cannot "
        f"execute: {missing}"
    )


# ── Placeholders resolve from what earlier steps produced ────────────────


def _graph_with(action: str, result: dict) -> TaskGraph:
    graph = TaskGraph("m", "test")
    node = TaskNode(task_id="t1", action=action)
    graph.add_node(node)
    graph.mark_running("t1")
    graph.mark_succeeded("t1", result=result)
    return graph


def test_generated_content_resolves_from_a_synthesis_step():
    state = MissionState.__new__(MissionState)
    graph = _graph_with(
        "summarize_sources",
        {"success": True, "result": {"text": "Orcas are cultural.", "sources": []}},
    )
    node = TaskNode(
        task_id="t2", action="create_pdf",
        params={"path": "/tmp/x.pdf", "title": "T", "body": "{{generated_content}}"},
    )

    resolved = state._resolve_params(node, graph)

    assert resolved["body"] == "Orcas are cultural."
    assert "{{" not in resolved["body"]


def test_sources_reach_the_pdf_as_structured_citations():
    """A synthesis PDF must be able to show where it got its material."""
    state = MissionState.__new__(MissionState)
    citations = [{"title": "Orca culture", "url": "https://example.org/a"}]
    graph = _graph_with(
        "summarize_sources",
        {"success": True, "result": {"text": "Body.", "sources": citations}},
    )
    node = TaskNode(task_id="t2", action="create_pdf",
                    params={"path": "/tmp/x.pdf", "body": "{{generated_content}}"})

    resolved = state._resolve_params(node, graph)

    assert resolved["sources"] == citations, "citations were dropped before the PDF"


def test_summarize_sources_collects_the_articles_that_were_read():
    """extract_article results feed the synthesis without the planner wiring them."""
    state = MissionState.__new__(MissionState)
    graph = TaskGraph("m", "test")
    for i in range(3):
        tid = f"a{i}"
        graph.add_node(TaskNode(task_id=tid, action="extract_article"))
        graph.mark_running(tid)
        graph.mark_succeeded(tid, result={
            "success": True,
            "result": {"url": f"https://example.org/{i}", "title": f"T{i}",
                       "body": f"Article body {i}"},
        })

    node = TaskNode(task_id="s", action="summarize_sources", params={})
    resolved = state._resolve_params(node, graph)

    assert len(resolved["sources"]) == 3
    assert {s["title"] for s in resolved["sources"]} == {"T0", "T1", "T2"}


def test_an_unresolvable_placeholder_is_left_visible_not_blanked():
    """A literal {{...}} in an artifact is a legible bug; an empty body is not."""
    state = MissionState.__new__(MissionState)
    graph = TaskGraph("m", "test")
    node = TaskNode(task_id="t", action="create_pdf",
                    params={"body": "{{generated_content}}"})

    resolved = state._resolve_params(node, graph)

    assert resolved["body"] == "{{generated_content}}"


# ── The PDF step refuses to claim success over nothing ───────────────────


def test_create_pdf_refuses_an_empty_body(monkeypatch):
    """Writing a blank PDF and reporting success is the failure that hides."""
    state = MissionState.__new__(MissionState)
    node = TaskNode(task_id="t", action="create_pdf",
                    params={"path": "/tmp/none.pdf", "title": "T", "body": "   "})

    result = asyncio.run(state._execute_node(node, TaskGraph("m", "g")))

    assert result["success"] is False
    assert "body" in result["error"].lower()


# ── Declared step budgets are actually enforced ──────────────────────────


def test_node_timeout_is_enforced_and_floored_per_action():
    state = MissionState.__new__(MissionState)

    # The 30s TaskNode default cannot bound a 32B synthesis over three
    # articles; the floor wins.
    slow = TaskNode(task_id="s", action="summarize_sources")
    assert state._node_timeout_s(slow) >= 300.0

    # A planner that asks for longer than the floor keeps its budget.
    generous = TaskNode(task_id="g", action="extract_article", timeout_s=600.0)
    assert state._node_timeout_s(generous) == 600.0

    # An ordinary step keeps its declared budget, never zero.
    ordinary = TaskNode(task_id="o", action="focus_app", timeout_s=30.0)
    assert state._node_timeout_s(ordinary) == 30.0
    assert state._node_timeout_s(TaskNode(task_id="z", action="wait", timeout_s=0.0)) >= 5.0


def test_a_hung_step_fails_the_node_instead_of_stalling_the_mission():
    """Before this, node.timeout_s was declared, serialised, and ignored."""
    from core.planning.mission_state import MissionStatus

    state = MissionState.__new__(MissionState)
    state._active_missions = {}
    state._advancing = set()
    state._persist_mission = lambda mission: None
    state._complete_mission = _noop_async
    state._try_recovery = _fail_recovery
    state._verify_node = _true_async

    graph = TaskGraph("m", "g")
    graph.add_node(TaskNode(task_id="hang", action="wait",
                            params={"seconds": 30}, timeout_s=0.1))

    class _M:
        pass

    mission = _M()
    mission.mission_id = "m1"
    mission.graph = graph
    mission.status = MissionStatus.ACTIVE
    mission.narration_log = []
    state._active_missions["m1"] = mission

    node = asyncio.run(state.advance_mission("m1"))

    assert node is not None
    assert graph.nodes["hang"].status.value in ("failed", "FAILED")
    assert any("budget" in line for line in mission.narration_log)


def test_the_orca_demo_mission_runs_end_to_end(tmp_path, monkeypatch):
    """The exact shape of the demo: folder, read 3 articles, synthesise, PDF.

    Every step here was broken before: extract_article and summarize_sources
    were "Unknown action", {{generated_content}} resolved to nothing, and the
    citations never reached the renderer.
    """
    pypdf = pytest.importorskip("pypdf")
    pytest.importorskip("reportlab")
    import types

    from core.container import ServiceContainer
    from core.planning.mission_state import MissionStatus

    articles = {
        "https://ex.org/1": ("Orca menopause", "Grandmothers lead pods to salmon."),
        "https://ex.org/2": ("Salish Sea decline", "Chinook scarcity and vessel noise."),
        "https://ex.org/3": ("Iberian contact", "Juveniles rub rudders off Iberia."),
    }

    class _Browser:
        async def extract_article_text(self, url):
            title, body = articles[url]
            return types.SimpleNamespace(
                url=url, title=title, author="", date="", body=body,
                source_domain="ex.org", word_count=len(body.split()),
            )

    seen: dict[str, str] = {}

    class _Router:
        async def think(self, prompt, **kwargs):
            seen["prompt"] = prompt
            return (
                "## Agreement\n\nOrca society is cultural [1][2].\n\n"
                "## My view\n\nThe menopause finding is load-bearing."
            )

    fakes = {"browser_controller": _Browser(), "llm_router": _Router()}
    real_get = ServiceContainer.get

    def _get(name, *args, **kwargs):
        if name in fakes:
            return fakes[name]
        return real_get(name, *args, **kwargs)

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(_get))

    out = tmp_path / "Orca Demo"
    state = MissionState.__new__(MissionState)
    state._active_missions = {}
    state._advancing = set()
    state._persist_mission = lambda mission: None
    state._complete_mission = _complete
    state._try_recovery = _fail_recovery
    state._verify_node = _true_async

    graph = TaskGraph("orca", "Orca Demo")
    graph.add_node(TaskNode(task_id="folder", action="create_folder",
                            params={"path": str(out)}))
    for i, url in enumerate(articles, 1):
        graph.add_node(TaskNode(task_id=f"read{i}", action="extract_article",
                                params={"url": url}, preconditions=["folder"]))
    graph.add_node(TaskNode(task_id="syn", action="summarize_sources",
                            params={"instruction": "Synthesise these articles."},
                            preconditions=["read1", "read2", "read3"]))
    graph.add_node(TaskNode(task_id="pdf", action="create_pdf",
                            params={"path": str(out / "orcas_summary.pdf"),
                                    "title": "Orcas: A Synthesis",
                                    "body": "{{generated_content}}"},
                            preconditions=["syn"]))

    mission = types.SimpleNamespace(
        mission_id="m", graph=graph, status=MissionStatus.ACTIVE, narration_log=[]
    )
    state._active_missions["m"] = mission

    async def drive():
        for _ in range(12):
            if await state.advance_mission("m") is None:
                break

    asyncio.run(drive())

    # All three articles reached the synthesis, and an opinion was asked for.
    assert "SOURCE 1" in seen["prompt"] and "SOURCE 3" in seen["prompt"]
    assert "My view" in seen["prompt"]

    pdf = out / "orcas_summary.pdf"
    assert pdf.exists(), f"no PDF; narration: {mission.narration_log}"
    text = "\n".join(page.extract_text() for page in pypdf.PdfReader(str(pdf)).pages)
    assert "{{" not in text, "an unresolved placeholder reached the user's PDF"
    assert "My view" in text
    assert "Sources" in text and "Orca menopause" in text and "ex.org/3" in text
    assert not [line for line in mission.narration_log if line.startswith("✗")]


def test_a_tilde_path_lands_in_the_home_directory(tmp_path, monkeypatch):
    """"in my Documents folder" invites ~; Path() would make a literal "~" dir."""
    state = MissionState.__new__(MissionState)
    monkeypatch.setenv("HOME", str(tmp_path))
    cwd_before = set(Path.cwd().iterdir())

    node = TaskNode(task_id="f", action="create_folder",
                    params={"path": "~/Documents/Orca Demo"})
    result = asyncio.run(state._execute_node(node, TaskGraph("m", "g")))

    assert result["success"] is True
    assert (tmp_path / "Documents" / "Orca Demo").is_dir()
    assert not Path("~").exists(), 'created a directory literally named "~"'
    assert set(Path.cwd().iterdir()) == cwd_before, "stray path created in the cwd"


def test_the_verifier_expands_the_same_paths_the_executor_writes(tmp_path, monkeypatch):
    """A file written to ~/x must not then fail verification for ~/x."""
    from core.capabilities.post_action_verifier import get_post_action_verifier

    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / "Documents" / "Orca Demo"
    target.mkdir(parents=True)
    (target / "orcas_summary.pdf").write_bytes(b"%PDF-1.4\n%stub\n")

    verifier = get_post_action_verifier()
    folder = asyncio.run(
        verifier.verify("folder_exists", {"path": "~/Documents/Orca Demo"})
    )
    assert folder.success, folder.evidence

    found = asyncio.run(
        verifier.verify("file_exists",
                        {"path": "~/Documents/Orca Demo/orcas_summary.pdf"})
    )
    assert found.success, found.evidence


async def _complete(mission):
    from core.planning.mission_state import MissionStatus

    mission.status = MissionStatus.COMPLETED


async def _noop_async(*args, **kwargs):
    return None


async def _true_async(*args, **kwargs):
    return True


async def _fail_recovery(*args, **kwargs):
    return False


# ── The renderer keeps the citations it is handed ────────────────────────


def test_pdf_renders_the_sources_it_is_given(tmp_path):
    """reportlab is the live path here (fpdf2 is not installed) and it used
    to accept `sources` and silently drop them, returning True."""
    pypdf = pytest.importorskip("pypdf")
    pytest.importorskip("reportlab")
    from core.capabilities.document_service import DocumentService

    out = tmp_path / "synthesis.pdf"
    ok = asyncio.run(
        DocumentService().create_pdf(
            str(out), "Orcas", "## Finding\n\nThey are cultural.",
            sources=[{"title": "Orca culture", "url": "https://example.org/a"},
                     {"title": "Pod decline", "url": "https://example.org/b"}],
        )
    )
    assert ok is True

    text = "\n".join(p.extract_text() for p in pypdf.PdfReader(str(out)).pages)
    assert "Sources" in text
    assert "Orca culture" in text
    assert "https://example.org/b" in text
