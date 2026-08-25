"""The drawings have to leave the machine, and nothing else may leave with them.

A design is only delivered when a person can look at it and download it, so
these pin the surface: the sheets are served as SVG a browser renders
directly, the files come back under their own names with the right content
type, and a path with ``..`` in it reaches nothing.

The route module is checked by reading it rather than by booting a server.
The live instance owns port 8000 and a resident model, and neither should be
disturbed to check that a path is contained.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interface.routes import engineering as routes


def test_the_design_root_is_resolved_once_and_not_recomputed():
    assert routes.DESIGN_ROOT.is_absolute()
    assert routes.DESIGN_ROOT.name == "live_designs"


@pytest.mark.parametrize(
    "design_id,filename",
    [
        ("../../..", "CLAUDE.md"),
        ("thing", "../../../CLAUDE.md"),
        ("..", ""),
        ("thing/../../..", "pyproject.toml"),
    ],
)
def test_a_path_that_climbs_out_reaches_nothing(design_id, filename):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        routes._resolved(design_id, filename)
    # Not found rather than forbidden: saying which tells a prober where the
    # boundary is.
    assert caught.value.status_code == 404


def test_an_absent_design_is_not_found(tmp_path, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(routes, "DESIGN_ROOT", tmp_path.resolve())
    with pytest.raises(HTTPException) as caught:
        routes._resolved("nothing_here")
    assert caught.value.status_code == 404


def test_every_written_format_has_a_content_type():
    from core.engineering.export import FORMATS

    for kind in FORMATS:
        assert f".{kind}" in routes._MEDIA_TYPES, f"{kind} would download as octet-stream"


def test_a_summary_reads_the_model_file_rather_than_guessing(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "DESIGN_ROOT", tmp_path.resolve())
    folder = tmp_path / "widget"
    folder.mkdir()
    (folder / "widget.json").write_text(json.dumps({
        "design": {"name": "Widget", "purpose": "Do a thing.", "parts": [{}, {}],
                   "fingerprint": "abc123"},
        "verification": {"ok": True, "buildable": True, "grounded": 7, "blocking": 0,
                         "plain": "Everything checks out."},
        "sheets": [{"kind": "assembly"}, {"kind": "schematic"}],
    }))
    summary = routes._design_summary(folder)
    assert summary["name"] == "Widget"
    assert summary["parts"] == 2
    assert summary["findings"] == 7
    assert summary["sheets"] == ["assembly", "schematic"]
    assert summary["ok"] is True


def test_a_broken_model_file_still_lists_rather_than_raising(tmp_path, monkeypatch):
    """A design half-written by a crash must not take the whole list down."""
    monkeypatch.setattr(routes, "DESIGN_ROOT", tmp_path.resolve())
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "broken.json").write_text("{not json")
    summary = routes._design_summary(folder)
    assert summary["id"] == "broken"
    assert "name" not in summary


def test_drawing_a_new_design_is_owner_only():
    """Reading is for any paired surface; spending compute is not."""
    import inspect

    source = inspect.getsource(routes.draw_design)
    assert "_require_owner(request)" in source
    assert inspect.getsource(routes.delete_design).count("_require_owner") == 1
    # Reading must NOT be owner-gated: watching a drawing is part of the
    # conversation, and the phone is a paired conversation surface.
    for reader in (routes.list_designs, routes.design_sheet, routes.design_file):
        assert "_require_owner" not in inspect.getsource(reader)


def test_the_model_json_carries_what_the_panel_needs_to_render():
    """The first version carried neither the narrative nor the sheet list."""
    from core.engineering.export import build_bundle
    from core.engineering.model import design_from_brief
    from core.engineering.verify import verify_design

    design = design_from_brief({
        "name": "Shelf bracket",
        "purpose": "Hold a shelf off a wall.",
        "parts": [{
            "name": "Bracket", "function": "Carries the load into the wall",
            "solid": {"kind": "plate", "width": "120 mm", "depth": "80 mm",
                      "thickness": "6 mm"},
            "material": "al_6061_t6",
        }],
    })
    verdict = verify_design(design, (), check_validation=False)
    bundle = build_bundle(
        design, (), verdict, sheets=(), plan=None,
        narrative="It holds the shelf up.", formats=("json",),
    )
    entry = bundle.named(f"{bundle.slug}.json")
    assert entry is not None
    payload = json.loads(entry.text)
    for key in ("design", "findings", "verification", "narrative", "sheets", "bundle"):
        assert key in payload, f"the model file has no {key}"
    assert payload["narrative"] == "It holds the shelf up."


def test_the_bundle_listing_names_every_file_including_the_late_ones():
    """The page and the model file are written last and were being left out."""
    from core.engineering.export import build_bundle
    from core.engineering.model import design_from_brief
    from core.engineering.verify import verify_design

    design = design_from_brief({
        "name": "Shelf bracket",
        "parts": [{
            "name": "Bracket", "function": "Carries the load",
            "solid": {"kind": "plate", "width": "120 mm", "depth": "80 mm",
                      "thickness": "6 mm"},
            "material": "al_6061_t6",
        }],
    })
    verdict = verify_design(design, (), check_validation=False)
    bundle = build_bundle(design, (), verdict, sheets=(), plan=None, narrative="")
    listed = {
        entry["name"]
        for entry in json.loads(bundle.named(f"{bundle.slug}.json").text)["bundle"]["files"]
    }
    assert {entry.name for entry in bundle.files} == listed


def test_the_router_is_mounted_on_the_server():
    import inspect

    from interface import server

    source = inspect.getsource(server)
    assert "engineering_routes.router" in source
    assert 'prefix="/api"' in source
