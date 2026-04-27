#!/usr/bin/env python3
"""Generate a live-source Aura architecture report as HTML and PDF.

The report is intentionally source-grounded: it reads the repository's current
documentation and file tree, summarizes key architecture areas, and optionally
prints a PDF through a locally installed Chrome/Chromium binary.
"""

from core.runtime.atomic_writer import atomic_write_text
from __future__ import annotations

import argparse
import html
import os
import re
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path.home() / "Desktop"


@dataclass(frozen=True)
class SourceAnchor:
    title: str
    path: str
    rationale: str


SOURCE_ANCHORS = [
    SourceAnchor("Architecture Whitepaper", "ARCHITECTURE.md", "Technical specification of the tick model, IIT math, affective steering, substrate, memory, and limitations."),
    SourceAnchor("Plain-English Architecture", "HOW_IT_WORKS.md", "Narrative explanation of the same system for a non-specialist audience."),
    SourceAnchor("Repository Overview", "README.md", "Top-level framing, operating assumptions, deployment notes, and benchmark context."),
    SourceAnchor("Runtime System Health API", "interface/routes/system.py", "Live status, bootstrap, privacy, and desktop-access readiness surface."),
    SourceAnchor("Permission Guard", "core/security/permission_guard.py", "macOS privacy/permission preflight checks for screen, accessibility, and automation."),
    SourceAnchor("Computer Use Skill", "core/skills/computer_use.py", "Desktop-control and screen-text capability with permission-aware gating."),
    SourceAnchor("OS Manipulation Skill", "core/skills/os_manipulation.py", "Keyboard and mouse control path for direct UI interaction."),
    SourceAnchor("Capability Engine", "core/capability_engine.py", "Skill catalog, routing, execution policy, and runtime availability."),
    SourceAnchor("Consciousness Stack", "core/consciousness", "Implementation surface for the integrated-information, substrate, and executive modules."),
    SourceAnchor("Proof Kernel", "proof_kernel", "Auxiliary proof-oriented modules and report assets supporting the architecture narrative."),
]


PATENT_THEMES = [
    (
        "Inference-time affective control",
        "Emotion state is not only described in prompt text; it also drives residual-stream steering and sampling-time modulation during generation."
    ),
    (
        "Atomic tick-based cognition",
        "Foreground and background cognition share a locked, event-sourced tick pipeline that can preempt internal work for user-priority turns without partial commits."
    ),
    (
        "Persistent substrate + dream repair",
        "The architecture combines continuous state, long-lived memory, and off-cycle dream/repair phases to maintain identity and reduce drift."
    ),
    (
        "Governed desktop agency",
        "Desktop control is exposed as a capability layer with explicit macOS permission introspection, not a blind automation hook."
    ),
    (
        "Theory-linked observability",
        "The system exposes architectural evidence from consciousness, executive, security, and resilience layers through the same runtime that drives behavior."
    ),
]


def parse_args() -> argparse.Namespace:
    today = datetime.now().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--basename",
        default=f"Aura_Live_Source_Architecture_Report_{today}",
        help="Base filename without extension.",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Write only the HTML report and skip PDF export.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_paragraphs(markdown_text: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        if line.startswith(("#", "-", "*", "|", "```")):
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current).strip())
    return paragraphs


def first_meaningful_paragraph(path: Path) -> str:
    paragraphs = normalize_paragraphs(read_text(path))
    for paragraph in paragraphs:
        if len(paragraph) >= 60:
            return paragraph
    return paragraphs[0] if paragraphs else ""


def markdown_headings(path: Path, *, limit: int = 8) -> list[str]:
    headings: list[str] = []
    for line in read_text(path).splitlines():
        match = re.match(r"^(#{1,3})\s+(.*)$", line.strip())
        if not match:
            continue
        heading = match.group(2).strip()
        if heading:
            headings.append(heading)
        if len(headings) >= limit:
            break
    return headings


def git_value(args: Iterable[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def git_snapshot() -> dict[str, str | int]:
    short_commit = git_value(["rev-parse", "--short", "HEAD"])
    branch = git_value(["branch", "--show-current"])
    status_lines = git_value(["status", "--short"]).splitlines()
    return {
        "commit": short_commit or "unknown",
        "branch": branch or "unknown",
        "dirty_files": len([line for line in status_lines if line.strip()]),
    }


def count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)
    except Exception:
        return 0


def source_metrics() -> OrderedDict[str, dict[str, int]]:
    buckets = OrderedDict(
        {
            "core": {"files": 0, "python_files": 0, "python_lines": 0},
            "interface": {"files": 0, "python_files": 0, "python_lines": 0},
            "skills": {"files": 0, "python_files": 0, "python_lines": 0},
            "proof_kernel": {"files": 0, "python_files": 0, "python_lines": 0},
        }
    )

    for bucket_name in buckets:
        bucket_root = PROJECT_ROOT / bucket_name
        if not bucket_root.exists():
            continue
        for path in bucket_root.rglob("*"):
            if not path.is_file():
                continue
            buckets[bucket_name]["files"] += 1
            if path.suffix == ".py":
                buckets[bucket_name]["python_files"] += 1
                buckets[bucket_name]["python_lines"] += count_lines(path)
    return buckets


def chrome_binary() -> str:
    candidates = [
        os.environ.get("AURA_REPORT_CHROME", ""),
        "google-chrome",
        "chromium",
        "chromium-browser",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isabs(candidate):
            if Path(candidate).exists():
                return candidate
            continue
        resolved = shutil_which(candidate)
        if resolved:
            return resolved
    return ""


def shutil_which(binary: str) -> str:
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        if not folder:
            continue
        candidate = Path(folder) / binary
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return ""


def render_html() -> str:
    generated_at = datetime.now().astimezone()
    git = git_snapshot()
    metrics = source_metrics()

    doc_summaries = []
    for rel_path in ("README.md", "HOW_IT_WORKS.md", "ARCHITECTURE.md"):
        path = PROJECT_ROOT / rel_path
        doc_summaries.append(
            {
                "path": rel_path,
                "paragraph": first_meaningful_paragraph(path),
                "headings": markdown_headings(path),
            }
        )

    anchor_rows = []
    for anchor in SOURCE_ANCHORS:
        path = PROJECT_ROOT / anchor.path
        exists = path.exists()
        anchor_rows.append(
            {
                "title": anchor.title,
                "path": anchor.path,
                "exists": exists,
                "rationale": anchor.rationale,
                "lines": count_lines(path) if exists and path.is_file() else 0,
            }
        )

    metric_rows = []
    total_files = 0
    total_python_files = 0
    total_python_lines = 0
    for name, data in metrics.items():
        total_files += data["files"]
        total_python_files += data["python_files"]
        total_python_lines += data["python_lines"]
        metric_rows.append(
            f"""
            <tr>
              <td>{html.escape(name)}</td>
              <td>{data['files']}</td>
              <td>{data['python_files']}</td>
              <td>{data['python_lines']:,}</td>
            </tr>
            """
        )

    doc_blocks = []
    for summary in doc_summaries:
        heading_list = "".join(f"<li>{html.escape(item)}</li>" for item in summary["headings"])
        doc_blocks.append(
            f"""
            <section class="doc-card">
              <div class="doc-path">{html.escape(summary['path'])}</div>
              <p>{html.escape(summary['paragraph'])}</p>
              <ul>{heading_list}</ul>
            </section>
            """
        )

    theme_blocks = "".join(
        f"""
        <div class="theme-card">
          <h3>{html.escape(title)}</h3>
          <p>{html.escape(body)}</p>
        </div>
        """
        for title, body in PATENT_THEMES
    )

    anchor_table_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(row['title'])}</td>
          <td><code>{html.escape(row['path'])}</code></td>
          <td>{'yes' if row['exists'] else 'no'}</td>
          <td>{row['lines']:,}</td>
          <td>{html.escape(row['rationale'])}</td>
        </tr>
        """
        for row in anchor_rows
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Aura Live-Source Architecture Report</title>
  <style>
    :root {{
      --ink: #17212b;
      --muted: #5d6b79;
      --line: #d8e0e8;
      --panel: #f6f8fb;
      --accent: #0d6b8a;
      --accent-soft: #d9eef4;
      --warn: #8f5f00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background: #eef3f7;
      line-height: 1.45;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 36px 40px 56px;
      background: white;
    }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{
      font-size: 34px;
      letter-spacing: -0.03em;
      margin-bottom: 10px;
    }}
    h2 {{
      font-size: 20px;
      margin: 32px 0 14px;
      padding-bottom: 8px;
      border-bottom: 2px solid var(--line);
    }}
    h3 {{
      font-size: 15px;
      margin-bottom: 6px;
    }}
    p {{ margin: 0 0 12px; }}
    .cover {{
      border: 1px solid var(--line);
      background: linear-gradient(135deg, #f7fbfd, #eef5f9);
      padding: 26px 28px;
      border-radius: 18px;
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-size: 11px;
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .subtitle {{
      font-size: 16px;
      color: var(--muted);
      max-width: 780px;
      margin-bottom: 16px;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .meta-card {{
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
    }}
    .meta-label {{
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .meta-value {{
      font-size: 16px;
      font-weight: 700;
    }}
    .summary-grid, .theme-grid, .doc-grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .summary-card, .theme-card, .doc-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      background: var(--panel);
    }}
    .doc-path {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      color: var(--accent);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    ul {{
      margin: 8px 0 0 18px;
      padding: 0;
    }}
    li {{
      margin-bottom: 4px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 10px 12px;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{
      text-align: left;
      background: var(--accent-soft);
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }}
    .note {{
      border-left: 4px solid var(--warn);
      background: #fff8e8;
      padding: 12px 14px;
      color: #6c5400;
      margin-top: 12px;
    }}
    .footer {{
      margin-top: 28px;
      color: var(--muted);
      font-size: 12px;
    }}
    @media print {{
      body {{ background: white; }}
      main {{ max-width: none; margin: 0; padding: 24px 28px 40px; }}
      .summary-card, .theme-card, .doc-card, .cover {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="cover">
      <div class="eyebrow">Aura Live-Source Report</div>
      <h1>Aura Architecture Report</h1>
      <p class="subtitle">This report was generated from the repository’s current live source tree and documentation. It is tuned for technical review, diligence, and patent-oriented architecture discussion rather than marketing copy.</p>
      <div class="meta-grid">
        <div class="meta-card">
          <div class="meta-label">Generated</div>
          <div class="meta-value">{html.escape(generated_at.strftime("%Y-%m-%d %H:%M:%S %Z"))}</div>
        </div>
        <div class="meta-card">
          <div class="meta-label">Git Branch</div>
          <div class="meta-value">{html.escape(str(git['branch']))}</div>
        </div>
        <div class="meta-card">
          <div class="meta-label">Commit</div>
          <div class="meta-value">{html.escape(str(git['commit']))}</div>
        </div>
        <div class="meta-card">
          <div class="meta-label">Dirty Files</div>
          <div class="meta-value">{git['dirty_files']}</div>
        </div>
      </div>
    </section>

    <h2>Executive Snapshot</h2>
    <div class="summary-grid">
      <div class="summary-card">
        <h3>System Identity</h3>
        <p>Aura presents itself as a sovereign cognitive architecture running locally on a Mac, with a locked tick pipeline, local-first inference, persistent state, and a large theory-driven consciousness stack.</p>
      </div>
      <div class="summary-card">
        <h3>Operational Model</h3>
        <p>The repository describes a dual-loop runtime: foreground user-triggered cognition and background heartbeat cognition, both grounded in atomic state transition and event-sourced persistence.</p>
      </div>
      <div class="summary-card">
        <h3>Embodied Interface</h3>
        <p>The interface layer combines FastAPI, WebSocket streaming, voice, telemetry, privacy controls, and desktop access readiness, giving the cognitive stack an inspectable and actionable shell.</p>
      </div>
      <div class="summary-card">
        <h3>Novelty Orientation</h3>
        <p>The strongest architecture differentiators in the live source are residual-stream affective steering, theory-linked observability, dream-style repair, and permission-governed desktop agency.</p>
      </div>
    </div>

    <h2>Repository Metrics</h2>
    <table>
      <thead>
        <tr>
          <th>Area</th>
          <th>Files</th>
          <th>Python Files</th>
          <th>Python Lines</th>
        </tr>
      </thead>
      <tbody>
        {''.join(metric_rows)}
        <tr>
          <th>Total</th>
          <th>{total_files}</th>
          <th>{total_python_files}</th>
          <th>{total_python_lines:,}</th>
        </tr>
      </tbody>
    </table>

    <h2>Source-Grounded Document Summary</h2>
    <div class="doc-grid">
      {''.join(doc_blocks)}
    </div>

    <h2>Architecture Anchors</h2>
    <table>
      <thead>
        <tr>
          <th>Anchor</th>
          <th>Path</th>
          <th>Present</th>
          <th>Lines</th>
          <th>Why It Matters</th>
        </tr>
      </thead>
      <tbody>
        {anchor_table_rows}
      </tbody>
    </table>

    <h2>Potential Patent-Relevant Themes</h2>
    <div class="theme-grid">
      {theme_blocks}
    </div>
    <div class="note">These themes are technical observations derived from the current source tree. They are not legal conclusions or a patentability opinion.</div>

    <h2>Implementation Reading of the Current Codebase</h2>
    <p>The current repository organizes itself around a layered model: a local-first cognitive core, a large consciousness and affect substrate, an execution/skill plane, and an interface/observability shell. The documentation repeatedly emphasizes that affect, identity, and consciousness are meant to be causal parts of runtime behavior rather than narrative overlays.</p>
    <p>The live source also shows a pragmatic operating envelope around that theory. There are explicit routes for health, privacy, desktop permission readiness, and tool availability; there are local skill implementations for desktop control and screen-text access; and there are compatibility shims that keep older import surfaces aligned with the canonical core implementations.</p>
    <p>From a diligence perspective, the architecture is notable because the repo does not only declare high-level ideas. It exposes concrete files for permission gating, tool routing, continuous memory, system health, and executive governance, which makes the overall claim set more inspectable than a purely conceptual whitepaper would be.</p>

    <div class="footer">
      <p>Generated from <code>{html.escape(str(PROJECT_ROOT))}</code>.</p>
      <p>Artifacts are suitable for review packets, internal diligence, and architecture discussions.</p>
    </div>
  </main>
</body>
</html>
"""


def write_html(output_path: Path) -> None:
    atomic_write_text(output_path, render_html(), encoding="utf-8")


def export_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = chrome_binary()
    if not chrome:
        raise RuntimeError("No Chrome/Chromium binary found for PDF export.")

    html_uri = html_path.resolve().as_uri()
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--allow-file-access-from-files",
        f"--print-to-pdf={pdf_path}",
        html_uri,
    ]
    subprocess.run(cmd, check=True, timeout=180)


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / f"{args.basename}.html"
    pdf_path = output_dir / f"{args.basename}.pdf"

    write_html(html_path)
    print(f"HTML report written to: {html_path}")

    if args.html_only:
        return 0

    export_pdf(html_path, pdf_path)
    print(f"PDF report written to: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
