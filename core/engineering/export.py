"""The bundle: everything about a design, in the formats it has to arrive in.

A design that exists only inside a program has not been delivered. This
writes the whole set — the drawings, the model, the parts list, the
calculations with their working, the build plan, and the geometry in the
formats a printer, a machine shop and a CAD package each want — and returns
what it wrote so the reply can say where the files are.

The HTML bundle is the one to open. It is a single self-contained file with
every sheet in it, the findings with their formulas, the parts list and the
build steps, and it needs nothing from the network. It prints to PDF from
any browser, which is what most people actually want a PDF for.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

__all__ = [
    "ExportedFile",
    "ExportBundle",
    "build_bundle",
    "write_bundle",
    "write_bundle_async",
    "FORMATS",
]

#: Every format the bundle can produce, and why somebody would want it.
FORMATS: dict[str, str] = {
    "html": "One self-contained page with every drawing, calculation and step. Start here.",
    "svg": "Each drawing on its own, as vector art that prints at any size.",
    "json": "The whole design as data, so another program can read it.",
    "csv": "The parts list as a spreadsheet, ready to price or order from.",
    "md": "The specification and calculations as text, for a document or a repository.",
    "stl": "Triangle mesh in millimetres, for a 3D printer.",
    "3mf": "Mesh with units, part names and colours, for a modern slicer.",
    "obj": "Mesh with named groups, for rendering.",
    "dxf": "Flat drawing on layers, for a cutter or a CAD package.",
    "scad": "Editable OpenSCAD source, so the dimensions can be changed.",
}


@dataclass(frozen=True, slots=True)
class ExportedFile:
    """One file in the bundle."""

    name: str
    kind: str
    text: str | None = None
    data: bytes | None = None
    description: str = ""

    @property
    def size(self) -> int:
        if self.data is not None:
            return len(self.data)
        return len((self.text or "").encode("utf-8"))

    def payload(self) -> bytes:
        if self.data is not None:
            return self.data
        return (self.text or "").encode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "bytes": self.size,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class ExportBundle:
    """Every file produced for one design."""

    slug: str
    files: tuple[ExportedFile, ...] = ()
    written: tuple[str, ...] = ()

    def named(self, name: str) -> ExportedFile | None:
        for entry in self.files:
            if entry.name == name:
                return entry
        return None

    def of_kind(self, kind: str) -> tuple[ExportedFile, ...]:
        return tuple(entry for entry in self.files if entry.kind == kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "files": [entry.to_dict() for entry in self.files],
            "written": list(self.written),
            "total_bytes": sum(entry.size for entry in self.files),
        }


def _bom_csv(design, plan) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "item", "part", "plain name", "quantity", "material", "mass",
        "shape", "how obtained", "specification", "standard", "supplier class",
        "unit cost", "lead time", "subsystem", "function",
    ])
    for part in design.parts:
        mass = part.mass()
        writer.writerow([
            part.balloon,
            part.name,
            part.lay_name or part.name,
            part.quantity,
            part.material.name if part.material else "",
            mass.text() if mass else "",
            part.solid.describe() if part.solid else "",
            part.sourcing.method,
            part.sourcing.specification,
            part.sourcing.standard,
            part.sourcing.supplier_class,
            part.sourcing.unit_cost.text() if part.sourcing.unit_cost else "",
            part.sourcing.lead_time.text() if part.sourcing.lead_time else "",
            part.subsystem,
            part.function,
        ])
    return output.getvalue()


def _spec_markdown(design, findings, verdict, plan, narrative: str) -> str:
    lines = [f"# {design.name}", ""]
    if design.purpose:
        lines += [design.purpose, ""]
    lines += [
        f"Revision {design.revision}. Model fingerprint `{design.fingerprint()}`.",
        f"Drawn to {design.standard}.",
        "",
        "## What it is",
        "",
        narrative or "No narrative was generated.",
        "",
        "## Whether it holds up",
        "",
        verdict.plain(),
        "",
        f"Checks run: {', '.join(verdict.checks_run)}.",
        "",
    ]
    if verdict.validation_note:
        lines += [verdict.validation_note, ""]

    if design.requirements:
        lines += ["## Requirements", "",
                  "| Requirement | Target | Verified by | Verdict |",
                  "| --- | --- | --- | --- |"]
        for requirement in design.requirements:
            lines.append(
                f"| {requirement.statement} | "
                f"{requirement.target.text() if requirement.target else '-'} | "
                f"`{requirement.check or '-'}` | {requirement.verdict} |"
            )
        lines.append("")

    lines += ["## Parts", "",
              "| No | Part | Qty | Material | Mass | How obtained |",
              "| --- | --- | --- | --- | --- | --- |"]
    for part in design.parts:
        mass = part.mass()
        lines.append(
            f"| {part.balloon} | {part.lay_name or part.name} | {part.quantity} | "
            f"{part.material.name if part.material else '-'} | "
            f"{mass.text() if mass else '-'} | "
            f"{part.sourcing.specification or part.sourcing.method} |"
        )
    lines.append("")

    lines += ["## Calculations", "",
              "Every figure below was computed from the model. The formula and the "
              "inputs are given so the arithmetic can be redone by hand.", ""]
    for finding in findings:
        lines += [
            f"### {finding.name} — {finding.value.text()}",
            "",
            finding.plain,
            "",
            f"- Formula: `{finding.formula}`",
            f"- With: {finding.substituted()}",
            f"- Method: {finding.method}",
        ]
        if finding.assumptions:
            lines.append(f"- Assumes: {', '.join(finding.assumptions)}")
        if finding.advice:
            lines.append(f"- To fix: {finding.advice}")
        lines.append("")

    if plan is not None and plan.steps:
        lines += ["## Building it", "", plan.plain(), ""]
        if plan.buy:
            lines += ["### To buy", "", "| Qty | Item | Specification | Unit cost |",
                      "| --- | --- | --- | --- |"]
            for item in plan.buy:
                lines.append(
                    f"| {item.quantity} | {item.name} | {item.specification} | "
                    f"{item.unit_cost.text() if item.unit_cost else '-'} |"
                )
            lines.append("")
        if plan.make:
            lines += ["### To make", "", "| Qty | Part | Process | Stock | Tolerance |",
                      "| --- | --- | --- | --- | --- |"]
            for item in plan.make:
                lines.append(
                    f"| {item.quantity} | {item.name} | {item.process} | {item.stock} | "
                    f"{item.tolerance.text() if item.tolerance else '-'} |"
                )
            lines.append("")
        lines += ["### Order of assembly", ""]
        for step in plan.steps:
            lines.append(f"{step.number}. **{step.action}.** {step.detail}")
            if step.check:
                lines.append(f"   - Check: {step.check}")
        lines.append("")

    if verdict.problems:
        lines += ["## Open items", ""]
        for problem in verdict.problems:
            lines.append(f"- **{problem.severity}** — {problem.message} {problem.advice}".rstrip())
        lines.append("")
    return "\n".join(lines)


_HTML_STYLE = """
:root{--bg:#f4f2ec;--paper:#fff;--ink:#1f2320;--soft:#6b6f68;--line:#ddd9d0;
--accent:#b3402a;--pass:#2f6b45;--warn:#b3762a;--fail:#a32b1e;}
:root:not([data-theme=light]) @media (prefers-color-scheme:dark){}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0e1512;
--paper:#131b18;--ink:#dde5e0;--soft:#8b9a92;--line:#25322c;--accent:#e0503a;
--pass:#5fd096;--warn:#e0a03a;--fail:#e0503a;}}
:root[data-theme=dark]{--bg:#0e1512;--paper:#131b18;--ink:#dde5e0;--soft:#8b9a92;
--line:#25322c;--accent:#e0503a;--pass:#5fd096;--warn:#e0a03a;--fail:#e0503a;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif}
header{padding:28px 32px 18px;border-bottom:1px solid var(--line)}
h1{margin:0 0 6px;font-size:26px;letter-spacing:.01em;color:var(--accent)}
h2{font-size:19px;margin:34px 0 10px}
h3{font-size:15px;margin:20px 0 6px}
.sub{color:var(--soft);margin:0}
.tiles{display:flex;flex-wrap:wrap;gap:22px;margin-top:16px}
.tile b{display:block;font-size:20px;font-variant-numeric:tabular-nums}
.tile span{color:var(--soft);font-size:11px;text-transform:uppercase;letter-spacing:.07em}
nav{display:flex;flex-wrap:wrap;gap:6px;padding:12px 32px;border-bottom:1px solid var(--line);
position:sticky;top:0;background:var(--bg);z-index:5}
nav button{font:inherit;font-size:13px;padding:6px 13px;border:1px solid var(--line);
border-radius:20px;background:var(--paper);color:var(--ink);cursor:pointer}
nav button[aria-selected=true]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
main{padding:0 32px 60px;max-width:1500px}
section[hidden]{display:none}
.sheet{background:var(--paper);border:1px solid var(--line);border-radius:6px;
margin:18px 0;overflow:hidden}
.sheet svg{width:100%;height:auto;display:block}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0 20px}
th{text-align:left;color:var(--soft);font-weight:600;font-size:11px;
text-transform:uppercase;letter-spacing:.06em;padding:7px 10px;border-bottom:1px solid var(--line)}
td{padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
td.num{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace}
.pass{color:var(--pass)}.warn{color:var(--warn)}.fail{color:var(--fail)}
.finding{border-left:3px solid var(--line);padding:2px 0 2px 14px;margin:0 0 18px}
.finding.fail{border-left-color:var(--fail)}
.finding.watch{border-left-color:var(--warn)}
.finding.pass{border-left-color:var(--pass)}
.finding .value{font-family:ui-monospace,Menlo,monospace;font-size:17px;font-weight:600}
.work{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--soft);
background:color-mix(in srgb,var(--ink) 5%,transparent);padding:6px 9px;
border-radius:4px;margin-top:6px;overflow-x:auto}
.step{display:flex;gap:14px;padding:11px 0;border-bottom:1px solid var(--line)}
.step .n{flex:0 0 30px;font-family:ui-monospace,Menlo,monospace;color:var(--soft)}
.check{color:var(--soft);font-size:13px;margin-top:3px}
.files{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0 24px}
.file{border:1px solid var(--line);border-radius:6px;padding:11px 14px;background:var(--paper);
min-width:190px}
.file b{display:block;font-family:ui-monospace,Menlo,monospace;font-size:13px}
.file span{color:var(--soft);font-size:12px}
.note{color:var(--soft);font-size:13px}
@media print{nav{display:none}section[hidden]{display:block!important}
.sheet{break-inside:avoid;page-break-inside:avoid}}
@media (max-width:700px){main,header,nav{padding-left:16px;padding-right:16px}}
"""

_HTML_SCRIPT = """
(function(){
  var tabs=[].slice.call(document.querySelectorAll('nav button'));
  function show(id){
    tabs.forEach(function(t){t.setAttribute('aria-selected', String(t.dataset.tab===id));});
    [].forEach.call(document.querySelectorAll('main > section'), function(s){
      s.hidden = s.id !== id;
    });
  }
  tabs.forEach(function(t){t.addEventListener('click', function(){show(t.dataset.tab);});});
  if(tabs.length){show(tabs[0].dataset.tab);}
})();
"""


def _html_bundle(design, findings, verdict, plan, sheets, narrative: str, files) -> str:
    from core.engineering.explain import annotate_terms, explain_part

    def tile(label: str, value: str, klass: str = "") -> str:
        return (
            f'<div class="tile"><span>{escape(label)}</span>'
            f'<b class="{klass}">{escape(value)}</b></div>'
        )

    by_id = {f.id: f for f in findings}
    tiles = []
    mass = by_id.get("assurance.mass_growth") or by_id.get("mass.total")
    if mass:
        tiles.append(tile("Mass", mass.value.text(), mass.verdict))
    power = by_id.get("electrical.total_draw")
    if power:
        tiles.append(tile("Power", power.value.text()))
    cost = design.total_cost()
    if cost:
        tiles.append(tile("Parts cost", f"{float(cost.value):,.0f}"))
    failures = [f for f in findings if f.verdict == "fail"]
    tiles.append(tile("Checks failed", str(len(failures)), "fail" if failures else "pass"))
    tiles.append(tile("Results computed", str(len(findings))))

    parts_rows = "".join(
        f"<tr><td class=num>{p.balloon}</td><td><b>{escape(p.lay_name or p.name)}</b>"
        f"<div class=note>{escape(p.function)}</div></td>"
        f"<td class=num>{p.quantity}</td>"
        f"<td>{escape(p.material.name if p.material else '-')}</td>"
        f"<td class=num>{escape(p.mass().text() if p.mass() else '-')}</td>"
        f"<td>{escape(p.sourcing.specification or p.sourcing.method)}</td></tr>"
        for p in design.parts
    )

    findings_html = "".join(
        f'<div class="finding {escape(f.verdict or "")}">'
        f"<div><b>{escape(f.name)}</b> "
        f'<span class="value {escape(f.verdict or "")}">{escape(f.value.text())}</span></div>'
        f"<div>{escape(f.plain)}</div>"
        f'<div class="work">{escape(f.substituted())}<br>{escape(f.method)}</div>'
        + (f'<div class="check">To fix: {escape(f.advice)}</div>' if f.advice else "")
        + "</div>"
        for f in findings
    )

    steps_html = ""
    files_html = ""
    if plan is not None:
        steps_html = "".join(
            f'<div class="step"><div class="n">{s.number}</div><div>'
            f"<b>{escape(s.action)}</b><div>{escape(s.detail)}</div>"
            + (f'<div class="check">Check: {escape(s.check)}</div>' if s.check else "")
            + (
                f'<div class="check">Tools: {escape(", ".join(s.tools))}</div>'
                if s.tools else ""
            )
            + "</div></div>"
            for s in plan.steps
        )
        buy_rows = "".join(
            f"<tr><td class=num>{i.quantity}</td><td>{escape(i.name)}</td>"
            f"<td>{escape(i.specification)}</td>"
            f"<td class=num>{escape(i.unit_cost.text() if i.unit_cost else '-')}</td></tr>"
            for i in plan.buy
        )
        make_rows = "".join(
            f"<tr><td class=num>{i.quantity}</td><td>{escape(i.name)}</td>"
            f"<td>{escape(i.process)}</td><td>{escape(i.stock)}</td>"
            f"<td class=num>{escape(i.tolerance.text() if i.tolerance else '-')}</td></tr>"
            for i in plan.make
        )
    else:
        buy_rows = make_rows = ""

    files_html = "".join(
        f'<div class="file"><b>{escape(entry.name)}</b>'
        f"<span>{escape(FORMATS.get(entry.kind, entry.description))}</span></div>"
        for entry in files
    )

    glossary = annotate_terms(
        " ".join([narrative] + [f.plain for f in findings]), limit=10
    )
    glossary_html = "".join(
        f"<tr><td><b>{escape(term)}</b></td><td>{escape(meaning)}</td></tr>"
        for term, meaning in glossary
    )

    problems_html = "".join(
        f'<tr><td class="{escape("fail" if p.blocking else "warn")}">{escape(p.severity)}</td>'
        f"<td>{escape(p.subject)}</td><td>{escape(p.message)} {escape(p.advice)}</td></tr>"
        for p in verdict.problems
    )

    sheet_sections = "".join(
        f'<section id="sheet-{index}" hidden><h2>{escape(sheet.kind.title())}</h2>'
        f'<p class="note">{escape(SHEET_CAPTIONS.get(sheet.kind, ""))} '
        f"Scale {escape(sheet.scale_text)}.</p>"
        f'<div class="sheet">{sheet.svg}</div></section>'
        for index, sheet in enumerate(sheets)
    )
    sheet_tabs = "".join(
        f'<button data-tab="sheet-{index}">{escape(sheet.kind.title())}</button>'
        for index, sheet in enumerate(sheets)
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(design.name)}</title>
<style>{_HTML_STYLE}</style></head><body>
<header>
<h1>{escape(design.name)}</h1>
<p class="sub">{escape(design.purpose)}</p>
<p class="sub">Revision {escape(design.revision)} &middot; drawn to {escape(design.standard)}
&middot; model {escape(design.fingerprint())}</p>
<div class="tiles">{''.join(tiles)}</div>
</header>
<nav>{sheet_tabs}
<button data-tab="how">How it works</button>
<button data-tab="checks">Calculations</button>
<button data-tab="parts">Parts</button>
<button data-tab="build">Building it</button>
<button data-tab="terms">Terms</button>
<button data-tab="files">Files</button>
</nav>
<main>
{sheet_sections}
<section id="how" hidden><h2>How it works</h2>
<p>{escape(narrative)}</p>
<h3>Part by part</h3>
{''.join(f'<p><b>{escape(p.lay_name or p.name)}.</b> {escape(explain_part(p))}</p>' for p in design.parts)}
</section>
<section id="checks" hidden><h2>Calculations</h2>
<p class="note">{escape(verdict.plain())}</p>
<p class="note">Checks run: {escape(', '.join(verdict.checks_run))}.</p>
{findings_html}
{'<h3>Open items</h3><table><tr><th>Severity</th><th>Where</th><th>What</th></tr>' + problems_html + '</table>' if problems_html else ''}
</section>
<section id="parts" hidden><h2>Parts</h2>
<table><tr><th>No</th><th>Part</th><th>Qty</th><th>Material</th><th>Mass</th>
<th>How obtained</th></tr>{parts_rows}</table></section>
<section id="build" hidden><h2>Building it</h2>
<p>{escape(plan.plain()) if plan else 'No build plan was produced.'}</p>
{'<h3>To buy</h3><table><tr><th>Qty</th><th>Item</th><th>Specification</th><th>Unit cost</th></tr>' + buy_rows + '</table>' if buy_rows else ''}
{'<h3>To make</h3><table><tr><th>Qty</th><th>Part</th><th>Process</th><th>Stock</th><th>Tolerance</th></tr>' + make_rows + '</table>' if make_rows else ''}
<h3>Order of assembly</h3>{steps_html}
</section>
<section id="terms" hidden><h2>Terms used here</h2>
<table><tr><th>Term</th><th>What it means</th></tr>{glossary_html}</table></section>
<section id="files" hidden><h2>Files</h2>
<p class="note">Everything written alongside this page.</p>
<div class="files">{files_html}</div></section>
</main>
<script>{_HTML_SCRIPT}</script>
</body></html>"""


SHEET_CAPTIONS: dict[str, str] = {
    "assembly": "The whole thing put together, labelled part by part.",
    "exploded": "Pulled apart in the order it comes off, numbered to the parts list.",
    "section": "Cut through, so the inside can be seen.",
    "orthographic": "Squared-on views with dimensions, for making from.",
    "schematic": "What connects to what, rather than what it looks like.",
}


def build_bundle(
    design,
    findings: tuple,
    verdict,
    *,
    sheets: tuple = (),
    plan=None,
    narrative: str = "",
    formats: tuple[str, ...] = (),
) -> ExportBundle:
    """Produce every requested format for one design, in memory."""
    from core.engineering.export_mesh import (
        MESH_FORMATS,
        dxf_text,
        obj_text,
        openscad_text,
        stl_binary,
        three_mf_bytes,
    )
    from core.engineering.model import slug as make_slug

    wanted = set(formats) if formats else set(FORMATS)
    stem = make_slug(design.name)
    files: list[ExportedFile] = []

    if "svg" in wanted:
        for sheet in sheets:
            files.append(ExportedFile(
                f"{stem}_{sheet.kind}.svg", "svg", text=sheet.svg,
                description=SHEET_CAPTIONS.get(sheet.kind, ""),
            ))
    if "csv" in wanted:
        files.append(ExportedFile(
            f"{stem}_parts.csv", "csv", text=_bom_csv(design, plan),
            description=FORMATS["csv"],
        ))
    if "md" in wanted:
        files.append(ExportedFile(
            f"{stem}.md", "md",
            text=_spec_markdown(design, findings, verdict, plan, narrative),
            description=FORMATS["md"],
        ))
    has_geometry = any(part.solid is not None for part in design.parts)
    if has_geometry:
        if "stl" in wanted:
            files.append(ExportedFile(
                f"{stem}.stl", "stl", data=stl_binary(design),
                description=MESH_FORMATS["stl"][2],
            ))
        if "3mf" in wanted:
            files.append(ExportedFile(
                f"{stem}.3mf", "3mf", data=three_mf_bytes(design),
                description=MESH_FORMATS["3mf"][2],
            ))
        if "obj" in wanted:
            files.append(ExportedFile(
                f"{stem}.obj", "obj", text=obj_text(design),
                description=MESH_FORMATS["obj"][2],
            ))
        if "dxf" in wanted:
            files.append(ExportedFile(
                f"{stem}.dxf", "dxf", text=dxf_text(design),
                description=MESH_FORMATS["dxf"][2],
            ))
        if "scad" in wanted:
            files.append(ExportedFile(
                f"{stem}.scad", "scad", text=openscad_text(design),
                description=MESH_FORMATS["scad"][2],
            ))
    if "json" in wanted:
        # Written after the geometry so its file list is the real one. The
        # first version carried neither the narrative nor the sheet list, so
        # the panel reading it back could show neither.
        payload = {
            "design": design.to_dict(),
            "findings": [f.to_dict() for f in findings],
            "verification": verdict.to_dict(),
            "narrative": narrative,
            "sheets": [sheet.to_dict() for sheet in sheets],
        }
        if plan is not None:
            payload["build"] = plan.to_dict()
        # The two files that do not exist yet when this list is built are
        # this one and the page, so both are named rather than left out.
        listed = [entry.to_dict() for entry in files]
        listed.append({"name": f"{stem}.json", "kind": "json",
                       "bytes": 0, "description": FORMATS["json"]})
        if "html" in wanted:
            listed.insert(0, {"name": f"{stem}.html", "kind": "html",
                              "bytes": 0, "description": FORMATS["html"]})
        payload["bundle"] = {"slug": stem, "files": listed}
        files.append(ExportedFile(
            f"{stem}.json", "json",
            text=json.dumps(payload, indent=2, sort_keys=True, default=str),
            description=FORMATS["json"],
        ))
    if "html" in wanted:
        # Written last so its file list can name everything else.
        files.insert(0, ExportedFile(
            f"{stem}.html", "html",
            text=_html_bundle(design, findings, verdict, plan, sheets, narrative, files),
            description=FORMATS["html"],
        ))
    return ExportBundle(slug=stem, files=tuple(files))


def _target_directory(out_dir: str | None) -> Path:
    root = (
        Path(__file__).resolve().parents[2] / "artifacts" / "live_designs"
    ).resolve()
    if not out_dir:
        return root
    from core.runtime.payload_values import payload_path

    resolved = payload_path({"out_dir": out_dir}, "out_dir", root=root, default=root)
    return Path(resolved or root)


def _bundle_target(bundle: ExportBundle, out_dir: str | None) -> Path:
    """Return the confined target after validating every caller-held name."""
    from core.engineering.model import slug as make_slug

    if bundle.slug != make_slug(bundle.slug):
        raise ValueError("engineering export bundle slug is not canonical")
    names: set[str] = set()
    for entry in bundle.files:
        name = str(entry.name)
        if not name or Path(name).name != name or Path(name).is_absolute():
            raise ValueError("engineering export filename must be one plain name")
        if name in names:
            raise ValueError(f"engineering export filename is duplicated: {name}")
        names.add(name)
    return _target_directory(out_dir) / bundle.slug


def write_bundle(bundle: ExportBundle, out_dir: str | None = None) -> ExportBundle:
    """Write the bundle to disk through the governed write gateway."""
    from core.runtime.file_write_gateway import get_file_write_gateway

    target = _bundle_target(bundle, out_dir)
    gateway = get_file_write_gateway()
    gateway.ensure_directory(str(target), source="engineering_export")
    written: list[str] = []
    for entry in bundle.files:
        path = target / entry.name
        if entry.data is not None:
            gateway.write_bytes(str(path), entry.data, source="engineering_export")
        else:
            gateway.write_text(str(path), entry.text or "", source="engineering_export")
        written.append(str(path))
    return ExportBundle(bundle.slug, bundle.files, tuple(written))


async def write_bundle_async(bundle: ExportBundle, out_dir: str | None = None) -> ExportBundle:
    """The same, from async code, so no fsync lands on the event loop."""
    from core.runtime.file_write_gateway import get_file_write_gateway

    target = _bundle_target(bundle, out_dir)
    gateway = get_file_write_gateway()
    await gateway.ensure_directory_async(str(target), source="engineering_export")
    written: list[str] = []
    for entry in bundle.files:
        path = target / entry.name
        if entry.data is not None:
            await gateway.write_bytes_async(str(path), entry.data, source="engineering_export")
        else:
            await gateway.write_text_async(str(path), entry.text or "", source="engineering_export")
        written.append(str(path))
    return ExportBundle(bundle.slug, bundle.files, tuple(written))


async def delete_design_bundle_async(
    design_id: str,
    out_dir: str | None = None,
) -> bool:
    """Delete one canonical design bundle from the engineering-owned tree.

    The HTTP surface supplies an identifier, never a path. Keeping path
    construction here makes export the single lifecycle owner for the fixed
    ``artifacts/live_designs`` namespace it already creates.
    """
    from core.engineering.model import slug as make_slug
    from core.runtime.file_write_gateway import get_file_write_gateway

    identity = str(design_id or "").strip()
    if not identity or identity != make_slug(identity):
        raise ValueError("engineering design id is not canonical")
    target = _target_directory(out_dir) / identity
    return await get_file_write_gateway().delete_path_async(
        target,
        recursive=True,
        source="engineering_export.delete_design_bundle",
    )
