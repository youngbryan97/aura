"""interface/routes/engineering.py
────────────────────────────────
HTTP surface for the designs Aura has drawn.

Reading a design — listing them, fetching a sheet, downloading a file — is
available to the owner and to paired conversation devices, because looking
at a drawing is part of the conversation. Drawing a NEW one spends real
compute and writes to disk, so it is owner-only.

Sheets are served as SVG rather than as JSON carrying SVG, so the panel can
put one straight into an ``img`` tag and a browser can print it. Downloads
are served with the right content type and a filename, so the browser saves
them under the name the bundle gave them rather than as the route path.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from interface.routes.devices import _owner_authenticated

logger = logging.getLogger("Aura.Server.Engineering")

router = APIRouter()

#: Where finished designs live. Resolved once so no request can walk out of it.
DESIGN_ROOT = (Path(__file__).resolve().parents[2] / "artifacts" / "live_designs").resolve()

#: What to send each file as. A browser given the wrong type on a mesh file
#: offers to display it rather than to save it.
_MEDIA_TYPES: dict[str, str] = {
    ".svg": "image/svg+xml",
    ".html": "text/html; charset=utf-8",
    ".json": "application/json",
    ".csv": "text/csv; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".stl": "model/stl",
    ".3mf": "model/3mf",
    ".obj": "model/obj",
    ".dxf": "image/vnd.dxf",
    ".scad": "text/plain; charset=utf-8",
}


def _require_owner(request: Request) -> None:
    if not _owner_authenticated(request):
        raise HTTPException(status_code=403, detail="Drawing a design is owner-only")


def _resolved(design_id: str, filename: str = "") -> Path:
    """A path inside the design root, or a 404.

    Resolution happens before the containment check, so ``..`` in either
    part cannot climb out. A traversal attempt is reported as not found
    rather than as forbidden, since saying which it is tells a prober where
    the boundary is.
    """
    candidate = (DESIGN_ROOT / design_id / filename).resolve() if filename else (
        DESIGN_ROOT / design_id
    ).resolve()
    try:
        candidate.relative_to(DESIGN_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="No such design") from exc
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="No such design")
    return candidate


def _design_summary(folder: Path) -> dict[str, Any]:
    """What a design looks like in a list, read from what was written."""
    import json

    entry: dict[str, Any] = {
        "id": folder.name,
        "files": sorted(p.name for p in folder.iterdir() if p.is_file()),
        "modified": folder.stat().st_mtime,
    }
    model = folder / f"{folder.name}.json"
    if model.exists():
        try:
            payload = json.loads(model.read_text())
        except (ValueError, OSError):
            return entry
        design = payload.get("design", {})
        verification = payload.get("verification", {})
        entry.update({
            "name": design.get("name", folder.name),
            "purpose": design.get("purpose", ""),
            "parts": len(design.get("parts", [])),
            "fingerprint": design.get("fingerprint", ""),
            "ok": verification.get("ok"),
            "buildable": verification.get("buildable"),
            "findings": verification.get("grounded"),
            "blocking": verification.get("blocking"),
            "verdict": verification.get("plain", ""),
            "sheets": [
                sheet.get("kind") for sheet in payload.get("sheets", [])
            ] or [
                p.stem.rsplit("_", 1)[-1]
                for p in folder.glob(f"{folder.name}_*.svg")
            ],
        })
    return entry


@router.get("/engineering/designs")
async def list_designs() -> dict[str, Any]:
    """Every design that has been drawn, newest first."""
    if not DESIGN_ROOT.exists():
        return {"ok": True, "designs": []}
    folders = [p for p in DESIGN_ROOT.iterdir() if p.is_dir()]
    designs = sorted(
        (_design_summary(folder) for folder in folders),
        key=lambda entry: entry.get("modified", 0.0),
        reverse=True,
    )
    return {"ok": True, "designs": designs}


@router.get("/engineering/designs/{design_id}")
async def inspect_design(design_id: str) -> dict[str, Any]:
    """The whole model, its findings and its verification, as written."""
    import json

    folder = _resolved(design_id)
    model = folder / f"{design_id}.json"
    if not model.exists():
        raise HTTPException(status_code=404, detail="This design has no model file")
    try:
        payload = json.loads(model.read_text())
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=500, detail="The model file is unreadable") from exc
    return {"ok": True, **payload}


@router.get("/engineering/designs/{design_id}/sheet/{kind}")
async def design_sheet(design_id: str, kind: str) -> Response:
    """One drawing, as SVG a browser can render or print directly."""
    folder = _resolved(design_id)
    sheet = folder / f"{design_id}_{kind}.svg"
    if not sheet.exists():
        raise HTTPException(status_code=404, detail=f"No {kind} sheet for this design")
    return Response(
        content=sheet.read_text(),
        media_type="image/svg+xml",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/engineering/designs/{design_id}/file/{filename}")
async def design_file(design_id: str, filename: str) -> FileResponse:
    """Download one file from a design, under the name the bundle gave it."""
    path = _resolved(design_id, filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No such file")
    suffix = path.suffix.lower()
    return FileResponse(
        path,
        media_type=_MEDIA_TYPES.get(suffix, "application/octet-stream"),
        filename=path.name,
    )


@router.get("/engineering/capability")
async def engineering_capability() -> dict[str, Any]:
    """What this can do, measured rather than claimed.

    Open to any paired surface, because a person asking what she can do
    should not need to be the owner to be told.
    """
    from core.engineering.faculty import capability_statement, engineering_report

    report = engineering_report()
    return {"ok": True, "statement": capability_statement(), **report}


@router.get("/engineering/reference")
async def engineering_reference() -> dict[str, Any]:
    """The catalogue behind the drawings: analyses, materials, symbols, terms.

    A panel that lists what is available needs this, and so does anybody
    asking what a term on a drawing means.
    """
    from core.engineering.analysis import ANALYSES
    from core.engineering.domains import DOMAINS
    from core.engineering.draw.sheet import SHEET_KINDS
    from core.engineering.draw.symbols import STANDARDS, SYMBOLS
    from core.engineering.explain import GLOSSARY
    from core.engineering.export import FORMATS
    from core.engineering.materials import FLUIDS, MATERIALS

    return {
        "ok": True,
        "analyses": [entry.to_dict() for entry in ANALYSES.values()],
        "materials": [entry.to_dict() for entry in MATERIALS.values()],
        "fluids": [entry.to_dict() for entry in FLUIDS.values()],
        "domains": [entry.to_dict() for entry in DOMAINS.values()],
        "symbols": [entry.to_dict() for entry in SYMBOLS.values()],
        "standards": STANDARDS,
        "sheet_kinds": SHEET_KINDS,
        "formats": FORMATS,
        "glossary": GLOSSARY,
    }


@router.get("/engineering/validation")
async def engineering_validation() -> dict[str, Any]:
    """Every published-answer problem, and whether this engine reproduces it.

    The panel shows this because it is the reason to believe anything else
    on a drawing. A failing case here means every number that used that
    formula is suspect, and saying so is the point.
    """
    from core.engineering.validation import run_validation

    return {"ok": True, **run_validation().to_dict()}


@router.post("/engineering/design")
async def draw_design(request: Request) -> JSONResponse:
    """Draw a design from a brief posted as JSON.

    Owner-only: it spends real compute and writes a directory of files. The
    body is the same brief the skill takes, so anything that can call the
    skill can call this and get the identical result.
    """
    _require_owner(request)
    try:
        brief = await request.json()
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="The body must be JSON") from exc
    if not isinstance(brief, dict) or not brief.get("parts"):
        return JSONResponse(
            {"ok": False, "error": "A brief needs at least one part."},
            status_code=400,
        )

    from core.engineering.studio import design_from_async

    options = {
        key: brief.pop(key)
        for key in ("theme", "size", "kinds", "formats", "out_dir")
        if key in brief
    }
    if isinstance(options.get("kinds"), list):
        options["kinds"] = tuple(options["kinds"])
    if isinstance(options.get("formats"), list):
        options["formats"] = tuple(options["formats"])
    try:
        result = await design_from_async(brief, **options)
    except (KeyError, ValueError, TypeError) as exc:
        # A brief naming a shape or a material nothing knows fails with the
        # name in the message, which is repairable. A drawing built on a
        # guessed substitute would not be.
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=422)

    payload = result.to_dict()
    payload["id"] = result.bundle.slug if result.bundle is not None else ""
    return JSONResponse({"ok": result.ok, **payload})


@router.delete("/engineering/designs/{design_id}")
async def delete_design(design_id: str, request: Request) -> dict[str, Any]:
    """Remove a design and everything written with it. Owner-only."""
    _require_owner(request)
    import shutil

    folder = _resolved(design_id)
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail="No such design")
    try:
        shutil.rmtree(folder)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Could not remove it") from exc
    return {"ok": True, "removed": design_id}
