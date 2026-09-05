"""Deterministic, always-openable artifact builder.

Bryan's concern: when Aura wants to *show* something ("something like this?"),
she should not depend on Excel or any specific app being installed. This
builds real, openable artifacts in portable formats that work on any machine:

- **table** → CSV (opens in Excel/Numbers/Sheets/any editor) plus a styled
  HTML view (opens in any browser) — no spreadsheet app required.
- **doc** → HTML (rich, universal) with an optional Markdown source.
- **program** → a self-contained runnable file (Python or a single-file HTML
  app) she can point the user at.

All writes go through the governed file-write gateway. The output is a real
path on disk, ready to open — the deterministic backbone under the
``demonstrate_artifact`` affordance (the autonomous task engine remains the
richer, creative path; this is the floor that always succeeds).

Hardening (CP126): caller stems are sanitized and containment-checked so a
stem cannot escape the artifact directory; CSV cells that begin with a formula
trigger are neutralized (CSV injection); table/doc/program size is bounded;
default names carry a uuid so concurrent builds don't collide; multi-file
artifacts clean up on partial failure; success requires the file to actually
exist; and open_artifact only opens files inside the artifact directory.
"""
from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.ArtifactBuilder")

_MAX_ROWS = 100_000
_MAX_COLS = 1_000
_MAX_CELL_CHARS = 32_768
_MAX_BODY_CHARS = 5_000_000
_MAX_SOURCE_CHARS = 5_000_000
_MAX_STEM_CHARS = 96

# Cells beginning with one of these are interpreted as formulas by spreadsheet
# apps — a classic CSV-injection vector. They are neutralized with a leading '.
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

_STEM_SAFE = re.compile(r"[^A-Za-z0-9._-]")


@dataclass(frozen=True)
class BuiltArtifact:
    ok: bool
    kind: str
    paths: list[str]
    primary: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "kind": self.kind,
            "paths": self.paths,
            "primary": self.primary,
            "detail": self.detail,
        }


def _output_dir() -> Path:
    try:
        from core.config import config

        base = Path(config.paths.data_dir)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        base = state_root() / "data"
    out = base / "generated_artifacts"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_stem(stem: str | None, *, prefix: str) -> str:
    """Sanitize a caller stem into a single safe path component.

    A default stem carries a uuid so concurrent builds of the same kind cannot
    collide on integer wall time; a caller stem is stripped of path separators,
    dot-segments, and control characters so it can never escape the artifact
    directory.
    """
    if not stem or not str(stem).strip():
        return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    cleaned = _STEM_SAFE.sub("_", str(stem).strip())
    cleaned = cleaned.strip("._") or prefix
    return cleaned[:_MAX_STEM_CHARS]


def _artifact_path(out: Path, stem: str, ext: str) -> Path:
    """Resolve a stem+ext under ``out`` and assert it stays contained."""
    path = (out / f"{stem}.{ext}").resolve()
    root = out.resolve()
    if path != root and not str(path).startswith(str(root) + os.sep):
        raise ValueError(f"artifact path escapes the output directory: {path}")
    return path


def _write(path: Path, text: str) -> bool:
    """Governed write — falls back to a direct atomic write off the live runtime.

    Returns True only when the bytes are durably on disk (write receipt).
    """
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway
    except ImportError as exc:
        # The gateway is genuinely absent — this module is being used outside
        # a runtime. That is the ONLY case the fallback is for.
        logger.debug("Governed write unavailable, using atomic fallback: %s", exc)
        try:
            from core.runtime.atomic_writer import atomic_write_text

            atomic_write_text(path, text, encoding="utf-8", durable=True)
        except (OSError, ImportError, ValueError) as fexc:
            logger.warning("Artifact write failed for %s: %s", path, fexc)
            return False
    else:
        # The gateway exists, so its answer is the answer. The previous
        # version caught every exception from this block — including a
        # governance REFUSAL — and then performed the same write directly
        # through `atomic_write_text`. Any denial, taint check or scope
        # violation was converted into an ungoverned write to the identical
        # target, which is not a fallback but a bypass with a log line.
        try:
            with local_internal_governed_scope("artifact_builder.write", domain="file_write"):
                get_file_write_gateway().write_text(
                    path, text, source="artifact_builder", durable=True
                )
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            from core.runtime.errors import record_degradation

            record_degradation(
                "artifact_builder",
                exc,
                severity="warning",
                action="refused an artifact write because the governed write failed",
                extra={"path": str(path)},
            )
            logger.warning("Governed artifact write refused for %s: %s", path, exc)
            return False
    # Receipt: the builder must not claim success from control flow alone.
    try:
        return path.exists() and path.stat().st_size >= 0
    except OSError:
        return False


def _cleanup(paths: list[Path]) -> None:
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass


def _csv_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    if len(text) > _MAX_CELL_CHARS:
        text = text[:_MAX_CELL_CHARS]
    # Neutralize formula-injection triggers before quoting.
    if text[:1] in _CSV_FORMULA_PREFIXES:
        text = "'" + text
    if any(c in text for c in (",", '"', "\n")):
        return '"' + text.replace('"', '""') + '"'
    return text


def _bounded_rows(all_rows: list[Sequence[Any]]) -> tuple[list[Sequence[Any]] | None, str]:
    if len(all_rows) > _MAX_ROWS:
        return None, f"too many rows (>{_MAX_ROWS})"
    for row in all_rows:
        if len(row) > _MAX_COLS:
            return None, f"too many columns (>{_MAX_COLS})"
    return all_rows, ""


def build_table(
    rows: Sequence[Sequence[Any]],
    *,
    headers: Sequence[str] | None = None,
    title: str = "Table",
    stem: str | None = None,
) -> BuiltArtifact:
    """Build a table as CSV (universal) + styled HTML (any browser)."""
    stem = _safe_stem(stem, prefix="table")
    out = _output_dir()
    all_rows: list[Sequence[Any]] = []
    if headers:
        all_rows.append(list(headers))
    all_rows.extend(rows)
    if not all_rows:
        return BuiltArtifact(False, "table", [], "", "no rows provided")
    bounded, bound_err = _bounded_rows(all_rows)
    if bounded is None:
        return BuiltArtifact(False, "table", [], "", bound_err)

    csv_text = "\n".join(",".join(_csv_cell(c) for c in row) for row in all_rows) + "\n"

    def _row_html(cells: Sequence[Any], tag: str) -> str:
        return "<tr>" + "".join(f"<{tag}>{html.escape(str(c)[:_MAX_CELL_CHARS])}</{tag}>" for c in cells) + "</tr>"

    data_rows = rows if headers else all_rows
    header_html = _row_html(headers, "th") if headers else ""
    body_rows = "".join(_row_html(row, "td") for row in data_rows)
    html_text = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
 body{{font-family:-apple-system,system-ui,sans-serif;margin:2rem;color:#1a1a1a}}
 h1{{font-size:1.2rem}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{border:1px solid #d0d0d0;padding:.5rem .75rem;text-align:left}}
 th{{background:#f3f4f6;font-weight:600}}
 tr:nth-child(even) td{{background:#fafafa}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<table><thead>{header_html}</thead><tbody>{body_rows}</tbody></table>
</body></html>"""

    try:
        csv_path = _artifact_path(out, stem, "csv")
        html_path = _artifact_path(out, stem, "html")
    except ValueError as exc:
        return BuiltArtifact(False, "table", [], "", str(exc))

    # Multi-file set: if either write fails, clean up so we never leave half a
    # table on disk while claiming success.
    if not _write(csv_path, csv_text):
        return BuiltArtifact(False, "table", [], "", "csv write failed")
    if not _write(html_path, html_text):
        _cleanup([csv_path])
        return BuiltArtifact(False, "table", [], "", "html write failed; csv rolled back")

    return BuiltArtifact(
        True, "table",
        [str(csv_path), str(html_path)],
        str(csv_path),
        f"{len(data_rows)} data row(s); CSV opens in any spreadsheet, HTML in any browser",
    )


def build_doc(body_markdown: str, *, title: str = "Document", stem: str | None = None) -> BuiltArtifact:
    """Build a portable document as HTML (with the Markdown source alongside)."""
    if not isinstance(body_markdown, str):
        return BuiltArtifact(False, "doc", [], "", "body must be a string")
    if len(body_markdown) > _MAX_BODY_CHARS:
        return BuiltArtifact(False, "doc", [], "", f"document too large (>{_MAX_BODY_CHARS} chars)")
    stem = _safe_stem(stem, prefix="doc")
    out = _output_dir()

    lines_html = []
    for line in body_markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("### "):
            lines_html.append(f"<h3>{html.escape(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            lines_html.append(f"<h2>{html.escape(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            lines_html.append(f"<h1>{html.escape(stripped[2:])}</h1>")
        elif stripped.startswith(("- ", "* ")):
            lines_html.append(f"<li>{html.escape(stripped[2:])}</li>")
        else:
            lines_html.append(f"<p>{html.escape(stripped)}</p>")
    html_text = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>body{{font-family:-apple-system,system-ui,sans-serif;max-width:44rem;margin:2rem auto;line-height:1.6;color:#1a1a1a}}</style>
</head><body><h1>{html.escape(title)}</h1>{''.join(lines_html)}</body></html>"""

    try:
        md_path = _artifact_path(out, stem, "md")
        html_path = _artifact_path(out, stem, "html")
    except ValueError as exc:
        return BuiltArtifact(False, "doc", [], "", str(exc))

    if not _write(md_path, f"# {title}\n\n{body_markdown}\n"):
        return BuiltArtifact(False, "doc", [], "", "markdown write failed")
    if not _write(html_path, html_text):
        _cleanup([md_path])
        return BuiltArtifact(False, "doc", [], "", "html write failed; markdown rolled back")
    return BuiltArtifact(True, "doc", [str(html_path), str(md_path)], str(html_path),
                         "HTML opens in any browser; Markdown source alongside")


def build_program(source: str, *, language: str = "python", stem: str | None = None) -> BuiltArtifact:
    """Write a self-contained runnable program she can point the user at."""
    if not isinstance(source, str):
        return BuiltArtifact(False, "program", [], "", "source must be a string")
    if len(source) > _MAX_SOURCE_CHARS:
        return BuiltArtifact(False, "program", [], "", f"program too large (>{_MAX_SOURCE_CHARS} chars)")
    stem = _safe_stem(stem, prefix="program")
    out = _output_dir()
    ext = {"python": "py", "html": "html", "javascript": "js", "shell": "sh"}.get(language.lower(), "txt")

    # Provenance banner: this is Aura-generated code that has NOT been executed
    # or verified — mark it so a reader/opener treats it as untrusted.
    banner = _provenance_banner(language)
    body = source if source.endswith("\n") else source + "\n"
    text = f"{banner}{body}" if banner else body

    try:
        path = _artifact_path(out, stem, ext)
    except ValueError as exc:
        return BuiltArtifact(False, "program", [], "", str(exc))
    if not _write(path, text):
        return BuiltArtifact(False, "program", [], "", "program write failed")
    return BuiltArtifact(True, "program", [str(path)], str(path), f"self-contained {language} file (unverified)")


def _provenance_banner(language: str) -> str:
    tag = "Aura-generated, unverified — review before running."
    lang = language.lower()
    if lang in ("python", "shell"):
        return f"# {tag}\n"
    if lang == "javascript":
        return f"// {tag}\n"
    if lang == "html":
        return f"<!-- {tag} -->\n"
    return ""


def _resolve_open_target(path: str) -> Path | None:
    """Return the resolved artifact path only if it is a file inside the dir."""
    if not isinstance(path, str) or "://" in path:
        logger.warning("Refused to open non-artifact target: %r", path)
        return None
    try:
        resolved = Path(path).resolve()
        root = _output_dir().resolve()
        if resolved != root and not str(resolved).startswith(str(root) + os.sep):
            logger.warning("Refused to open path outside the artifact directory: %s", resolved)
            return None
        if not resolved.is_file():
            logger.warning("Refused to open non-existent artifact: %s", resolved)
            return None
        return resolved
    except (OSError, ValueError) as exc:
        logger.debug("Artifact open validation failed: %s", exc)
        return None


async def open_artifact(path: str) -> bool:
    """Open a BUILT artifact with the OS default handler (best-effort).

    Only files inside the artifact output directory are opened — this actuator
    will not open an arbitrary local path or a URL-like argument.
    """
    resolved = _resolve_open_target(path)
    if resolved is None:
        return False

    def _run_open() -> int:
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        result = get_subprocess_gateway().run(
            ["open", str(resolved)],
            timeout=10.0,
            offline_tooling=True,
            check=False,
            source="maintenance_tooling:artifact_builder_open",
            accelerator_capability="none",
        )
        return getattr(result, "returncode", 1)

    try:
        # Offload the blocking subprocess run so the coroutine never blocks the loop.
        return await asyncio.to_thread(_run_open) == 0
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        logger.debug("Artifact open skipped: %s", exc)
        return False
