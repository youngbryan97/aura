import json

from core.data.simulation_well import SimulationDataset, SimulationWellRegistry
from core.media.temporal_atlas import TemporalAtlas
from core.runtime.activation_audit import DEFAULT_SPECS, ActivationAuditor
from core.runtime.capability_tokens import get_capability_token_store, reset_capability_token_store
from core.runtime.receipts import ReceiptStore
from core.runtime.service_manifest import SERVICE_MANIFEST, critical_violations, verify_manifest
from core.workspace.aura_workspace import AuraWorkspace
from core.workspace.markdown_workspace import MarkdownWorkspace


def test_markdown_workspace_commits_search_and_revert(tmp_path):
    ws = MarkdownWorkspace(tmp_path / "workspace.json")

    ws.write_file("docs/readme.md", "# Aura\n\nFlagship notes", user="aura")
    first = ws.commit("initial")
    ws.write_file("docs/readme.md", "# Aura\n\nUpdated notes", user="aura")

    status = ws.status()
    assert status["modified"] == ["docs/readme.md"]
    assert ws.grep("Updated")[0]["path"] == "docs/readme.md"

    ws.revert(first.change_id)
    assert "Flagship" in ws.read_file("docs/readme.md")
    assert ws.find("readme.md") == ["docs/readme.md"]


def test_markdown_workspace_nonblocking_conflict_merge(tmp_path):
    ws = MarkdownWorkspace(tmp_path / "workspace.json")
    ws.write_file("docs/plan.md", "base\n")
    base = ws.commit("base")
    ws.create_bookmark("feature", base.change_id)

    ws.write_file("docs/plan.md", "main change\n")
    ws.commit("main change", bookmark="main")

    ws.revert(base.change_id)
    ws.write_file("docs/plan.md", "feature change\n")
    ws.commit("feature change", bookmark="feature")

    result = ws.merge(target="main", source="feature")

    assert result.merge_type == "merge_commit"
    assert result.conflicted
    assert result.conflicts[0].path == "docs/plan.md"
    assert ws.read_file("docs/plan.md") == "main change\n"
    assert ws.status()["conflicts"] == ["docs/plan.md"]


def test_markdown_workspace_permissions(tmp_path):
    ws = MarkdownWorkspace(tmp_path / "workspace.json")
    ws.write_file("private/notes.md", "secret", user="aura", mode=0o600)

    try:
        ws.read_file("private/notes.md", user="guest")
    except PermissionError:
        pass
    else:
        raise AssertionError("guest should not read 0600 notes")

    ws.chmod("private/notes.md", 0o644, user="aura")
    assert ws.read_file("private/notes.md", user="guest") == "secret"


def test_aura_workspace_gates_writes_and_commits_receipted_artifacts(tmp_path):
    raw = MarkdownWorkspace(tmp_path / "workspace.json")
    workspace = AuraWorkspace(store=raw, receipt_store=ReceiptStore(tmp_path / "receipts"))

    result = workspace.write_artifact(
        "/aura/decisions/approved/ship-workspace.md",
        "# Decision\n\nShip the governed workspace path.",
        actor="test",
        purpose="unit_test",
        authority_receipt="gov-test-1",
    )

    assert result.path == "aura/decisions/approved/ship-workspace.md"
    assert result.content_hash
    assert result.receipt_id.startswith("memory_write-")
    assert result.commit_id
    assert "Ship the governed workspace" in raw.read_file(result.path)
    assert raw.log()[0].metadata["authority_receipt_id"] == "gov-test-1"


def test_aura_workspace_accepts_directory_scoped_capability_for_video_evidence(tmp_path):
    reset_capability_token_store()
    raw = MarkdownWorkspace(tmp_path / "workspace.json")
    workspace = AuraWorkspace(store=raw, receipt_store=ReceiptStore(tmp_path / "receipts"))
    token = get_capability_token_store().issue(
        capability="agent_workspace.write",
        scope="/aura/observations/video",
        issuer="test",
        receipt_id="cap-receipt-1",
    )
    atlas = TemporalAtlas(120.0)
    evidence = [
        atlas.add_evidence(13.2, 16.5, "Operator notices the failure light", confidence=0.82),
    ]

    result = workspace.write_video_evidence(
        "screening-001",
        evidence,
        actor="temporal_atlas",
        purpose="answer_video_question",
        capability_token=token.token_id,
    )

    written = raw.read_file(result.path)
    assert result.path == "aura/observations/video/screening-001/evidence.md"
    assert "`00:00:13.200-00:00:16.500`" in written
    assert "Operator notices" in written
    assert get_capability_token_store().get(token.token_id).status.value == "used"


def test_temporal_atlas_expands_marks_dead_and_tracks_evidence():
    atlas = TemporalAtlas(6400.0, grid_size=8, max_depth=2)

    assert len(atlas.root.cells) == 64
    child = atlas.expand("root", 0)
    assert round(child.cells[0].duration_s, 4) == round(6400.0 / 64 / 64, 4)

    atlas.mark_dead(0, 100)
    atlas.mark_promising("root", [1, 2])
    evidence = atlas.add_evidence(100, 120, "title card", confidence=0.8)

    assert atlas.node("root").cells[0].dead is True
    assert atlas.node("root").cells[1].promising is True
    assert evidence in atlas.scratchpad()
    assert atlas.coverage()["dead_zone_ratio"] > 0
    assert atlas.next_frontier(limit=1)[0].promising is True


def test_simulation_well_plans_and_streams_local_records(tmp_path):
    data_dir = tmp_path / "active"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        json.dumps({"step": 1, "value": 0.1}) + "\n" + json.dumps({"step": 2, "value": 0.2}) + "\n",
        encoding="utf-8",
    )
    registry = SimulationWellRegistry(tmp_path / "manifest.json")
    registry.register(
        SimulationDataset(
            name="active_matter",
            domain="biophysics",
            local_path=str(data_dir),
            size_gb=0.25,
            tags=("physics",),
        )
    )

    shards = registry.plan_shards(["active_matter"], max_total_gb=1.0)
    records = list(registry.stream_records("active_matter", limit=1))

    assert shards[0].uri.endswith("train")
    assert records == [{"step": 1, "value": 0.1}]
    assert registry.list(domain="biophysics")[0].name == "active_matter"


def test_agent_workspace_is_architecture_manifest_role():
    role = SERVICE_MANIFEST["agent_workspace"]
    workspace = AuraWorkspace
    snapshot = {
        role.canonical_owner: workspace,
        "agent_workspace": workspace,
    }

    assert role.critical is True
    assert role.canonical_owner == "aura_workspace"
    assert critical_violations(verify_manifest(snapshot, manifest={"agent_workspace": role})) == []


def test_agent_workspace_activation_spec_is_required_and_autostarted():
    spec = next(item for item in DEFAULT_SPECS if item.name == "agent_workspace")

    assert spec.required is True
    assert spec.auto_start is True
    assert spec.starter is not None
    assert spec.service_keys == ("aura_workspace", "agent_workspace")
    assert ActivationAuditor((spec,))
