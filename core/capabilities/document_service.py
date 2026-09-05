"""core/capabilities/document_service.py — Programmatic Document Creation
==========================================================================
Creates text, Markdown, PDF, and simple DOCX files directly WITHOUT
depending on any UI application.

This is the fallback layer: if app-based export fails (e.g., Notes can't
export PDF), Aura can still create the artifact programmatically and
explain the fallback honestly.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.DocumentService")


def _escape_xml(text: str) -> str:
    """Escape for reportlab's mini-markup, which parses paragraphs as XML."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class DocumentService:
    """Programmatic document creation — does NOT depend on UI apps.

    Usage:
        svc = get_document_service()
        success = await svc.create_pdf("/path/to/doc.pdf", "Title", "Body text")
    """

    def __init__(self) -> None:
        self._started = False
        self._created_count = 0

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("document_service", self, required=False)
        self._started = True
        logger.info("DocumentService ONLINE")

    async def create_text(self, path: str, content: str) -> bool:
        """Create a plain text file atomically."""
        try:
            p = Path(path)
            await get_file_write_gateway().write_text_async(
                p,
                content,
                encoding="utf-8",
                source="document_service.create_text",
            )
            self._created_count += 1
            logger.info("Created text file: %s (%d bytes)", p.name, len(content))
            return True
        except (OSError, RuntimeError) as e:
            record_degradation("document_service.text", e)
            return False

    async def create_markdown(self, path: str, content: str, title: str = "",
                               metadata: Optional[Dict[str, str]] = None) -> bool:
        """Create a Markdown file with optional YAML frontmatter."""
        try:
            parts = []
            if title or metadata:
                parts.append("---")
                if title:
                    parts.append(f"title: \"{title}\"")
                parts.append(f"date: {time.strftime('%Y-%m-%d %H:%M')}")
                if metadata:
                    for k, v in metadata.items():
                        parts.append(f"{k}: \"{v}\"")
                parts.append("---\n")

            if title and not content.startswith("#"):
                parts.append(f"# {title}\n")

            parts.append(content)
            full_content = "\n".join(parts)
            return await self.create_text(path, full_content)
        except (OSError, RuntimeError) as e:
            record_degradation("document_service.markdown", e)
            return False

    async def create_pdf(self, path: str, title: str, body: str,
                          sources: Optional[List[Dict[str, str]]] = None) -> bool:
        """Create a PDF document.

        Tries (in order):
        1. fpdf2 (lightweight, pure Python)
        2. reportlab (full-featured)
        3. Markdown → HTML → wkhtmltopdf (external tool)
        4. Text file fallback with .pdf.txt extension
        """
        p = Path(path)

        # Method 1: fpdf2
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # Title
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, title, ln=True, align="C")
            pdf.ln(5)

            # Date
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 6, f"Generated: {time.strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
            pdf.ln(10)

            # Body
            pdf.set_font("Helvetica", "", 11)
            # Handle unicode by encoding to latin-1 with replacement
            safe_body = body.encode("latin-1", errors="replace").decode("latin-1")
            for paragraph in safe_body.split("\n\n"):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                # Check if it's a heading
                if paragraph.startswith("#"):
                    level = min(len(paragraph) - len(paragraph.lstrip("#")), 4)
                    heading_text = paragraph.lstrip("# ").strip()
                    pdf.set_font("Helvetica", "B", max(11, 16 - level * 2))
                    pdf.cell(0, 8, heading_text, ln=True)
                    pdf.set_font("Helvetica", "", 11)
                else:
                    pdf.multi_cell(0, 6, paragraph)
                pdf.ln(3)

            # Sources
            if sources:
                pdf.ln(5)
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "Sources", ln=True)
                pdf.set_font("Helvetica", "", 10)
                for i, src in enumerate(sources, 1):
                    source_title = src.get("title", "Untitled")
                    source_url = src.get("url", "")
                    safe_title = source_title.encode("latin-1", errors="replace").decode("latin-1")
                    safe_url = source_url.encode("latin-1", errors="replace").decode("latin-1")
                    pdf.cell(0, 5, f"[{i}] {safe_title}", ln=True)
                    if source_url:
                        pdf.set_text_color(0, 0, 200)
                        pdf.cell(0, 5, f"    {safe_url}", ln=True)
                        pdf.set_text_color(0, 0, 0)

            # File hash in metadata
            pdf.set_font("Helvetica", "I", 8)
            pdf.ln(5)
            content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
            pdf.cell(0, 4, f"Content hash: {content_hash}", ln=True)

            payload = pdf.output(dest="S")
            if isinstance(payload, str):
                pdf_bytes = payload.encode("latin-1", errors="replace")
            else:
                pdf_bytes = bytes(payload)
            await get_file_write_gateway().write_bytes_async(
                p,
                pdf_bytes,
                source="document_service.create_pdf.fpdf2",
            )
            self._created_count += 1
            logger.info("Created PDF (fpdf2): %s", p.name)
            return True

        except ImportError:
            logger.debug("fpdf2 not available, trying reportlab")
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
            record_degradation("document_service.pdf_fpdf2", e)
            logger.debug("fpdf2 failed: %s, trying reportlab", e)

        # Method 2: reportlab
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            story.append(Paragraph(title, styles["Title"]))
            story.append(Spacer(1, 12))
            story.append(Paragraph(
                f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
                styles["Italic"],
            ))
            story.append(Spacer(1, 20))

            for para in body.split("\n\n"):
                para = para.strip()
                if para:
                    # Basic markdown heading detection
                    if para.startswith("#"):
                        heading = para.lstrip("# ").strip()
                        story.append(Paragraph(heading, styles["Heading2"]))
                    else:
                        # Escape XML special chars
                        safe = _escape_xml(para)
                        story.append(Paragraph(safe, styles["Normal"]))
                    story.append(Spacer(1, 6))

            # Sources. The fpdf2 branch above has always rendered these; this
            # one silently dropped them, so on any host without fpdf2 a cited
            # synthesis produced a PDF with no citations — and create_pdf still
            # returned True, so nothing upstream could tell. A synthesis that
            # cannot show its sources is a different document than the one that
            # was asked for.
            if sources:
                story.append(Spacer(1, 12))
                story.append(Paragraph("Sources", styles["Heading2"]))
                for i, src in enumerate(sources, 1):
                    source_title = _escape_xml(str(src.get("title", "") or "Untitled"))
                    source_url = _escape_xml(str(src.get("url", "") or ""))
                    entry = f"[{i}] {source_title}"
                    if source_url:
                        entry += f'<br/><font color="blue">{source_url}</font>'
                    story.append(Paragraph(entry, styles["Normal"]))
                    story.append(Spacer(1, 4))

            story.append(Spacer(1, 12))
            content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
            story.append(Paragraph(f"Content hash: {content_hash}", styles["Italic"]))

            doc.build(story)
            await get_file_write_gateway().write_bytes_async(
                p,
                buffer.getvalue(),
                source="document_service.create_pdf.reportlab",
            )
            self._created_count += 1
            logger.info("Created PDF (reportlab): %s", p.name)
            return True

        except ImportError:
            logger.debug("reportlab not available, trying text fallback")
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
            record_degradation("document_service.pdf_reportlab", e)
            logger.debug("reportlab failed: %s", e)

        # Method 3: Text-based PDF fallback
        # Write the content as a text file that can be converted later
        try:
            txt_path = p.with_suffix(".txt")
            content = f"{title}\n{'=' * len(title)}\n\nGenerated: {time.strftime('%Y-%m-%d %H:%M')}\n\n{body}"
            if sources:
                content += "\n\nSources:\n"
                for i, src in enumerate(sources, 1):
                    content += f"[{i}] {src.get('title', '')} — {src.get('url', '')}\n"
            await get_file_write_gateway().write_text_async(
                txt_path,
                content,
                encoding="utf-8",
                source="document_service.create_pdf.text_fallback",
            )
            self._created_count += 1
            logger.info("Created text fallback (PDF rendering unavailable): %s", txt_path.name)
            return True
        except OSError as e:
            record_degradation("document_service.pdf_fallback", e)
            return False

    async def verify(self, path: str) -> Dict[str, Any]:
        """Verify a document file exists and is valid."""
        p = Path(path)
        if not p.exists():
            return {"valid": False, "error": "File not found"}

        size = p.stat().st_size
        if size == 0:
            return {"valid": False, "error": "File is empty"}

        file_hash = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        is_pdf = False
        if p.suffix.lower() == ".pdf":
            with open(p, "rb") as f:
                is_pdf = f.read(5) == b"%PDF-"

        return {
            "valid": True,
            "path": str(p),
            "size_bytes": size,
            "hash": file_hash,
            "is_pdf": is_pdf,
            "suffix": p.suffix,
        }

    async def open_preview(self, path: str) -> bool:
        """Open a document for preview using macOS 'open' command."""
        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["open", path],
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                source="document_service.open_preview",
                accelerator_capability="none",
            )
            await asyncio.wait_for(proc.wait(), timeout=5.0)
            return proc.returncode == 0
        except (OSError, asyncio.TimeoutError):
            return False

    def get_status(self) -> Dict[str, Any]:
        return {"created_count": self._created_count}


_instance: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    global _instance
    if _instance is None:
        _instance = DocumentService()
    return _instance


__all__ = ["DocumentService", "get_document_service"]
