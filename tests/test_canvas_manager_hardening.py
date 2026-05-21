from types import SimpleNamespace

import pytest

from core.agency.canvas_manager import CanvasManager
from core.container import ServiceContainer


@pytest.fixture(autouse=True)
def clear_services():
    ServiceContainer.clear()
    yield
    ServiceContainer.clear()


class RecordingEngine:
    def __init__(self, response: str = "# Canvas\n\n## Decision\nAccepted."):
        self.response = response
        self.prompts = []

    async def think(self, *, objective, mode, priority):
        self.prompts.append((objective, mode, priority))
        return SimpleNamespace(content=self.response)


@pytest.mark.asyncio
async def test_canvas_update_uses_atomic_safe_path_and_returns_receipt(tmp_path):
    engine = RecordingEngine("# Canvas\n\n## Decision\nCare needs boundaries.")
    ServiceContainer.register_instance("cognitive_engine", engine, required=False)
    manager = CanvasManager(root_dir=str(tmp_path), think_timeout_s=1.0)

    result = await manager.autonomous_update(
        "../../Project:Alpha",
        "Decision",
        "Care needs explicit boundaries.",
    )

    assert result["ok"] is True
    assert result["project"] == "Project_Alpha"
    assert (tmp_path / "Project_Alpha.md").read_text(encoding="utf-8").startswith("# Canvas")
    assert ".." not in result["path"]
    assert str(tmp_path.resolve()) in result["path"]
    assert "Care needs explicit boundaries." in engine.prompts[0][0]


@pytest.mark.asyncio
async def test_canvas_update_fails_closed_without_engine(tmp_path):
    manager = CanvasManager(root_dir=str(tmp_path), think_timeout_s=1.0)

    result = await manager.autonomous_update("Project", "Topic", "Insight")

    assert result == {"ok": False, "reason": "cognitive_engine_unavailable"}
    assert not list(tmp_path.glob("*.md"))


@pytest.mark.asyncio
async def test_canvas_prune_keeps_tail_with_atomic_write(tmp_path):
    manager = CanvasManager(
        root_dir=str(tmp_path),
        max_canvas_bytes=20,
        keep_tail_lines=2,
    )
    canvas = tmp_path / "Project.md"
    canvas.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")

    pruned = await manager._prune_if_needed(canvas)

    assert pruned is True
    assert canvas.read_text(encoding="utf-8") == "line3\nline4\n"
