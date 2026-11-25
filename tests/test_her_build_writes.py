"""What she builds is kept by writes that are there: no test of this puts another function in their place."""
from __future__ import annotations

from pathlib import Path


def test_every_governed_write_that_is_asked_for_is_there():
    """LIVE 24 November 2026: "no governed write named 'write_bytes'", so no picture she painted was ever kept. Every test of
    painting put another function in its place."""
    import ast
    import inspect
    import re

    from core.rebuilding import her_build_writes

    root = Path(her_build_writes.__file__).resolve().parents[2]
    texts = [path.read_text(encoding="utf-8") for path in (*root.glob("core/**/*.py"), *root.glob("interface/**/*.py"))]
    asked = {name for text in texts if "her_build_writes import" in text for name in re.findall(r"\bgoverned\(?,?\s*[\"'](\w+)[\"']", text)}
    there = {node.value for node in ast.walk(ast.parse(inspect.getsource(her_build_writes.governed))) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert {"write_text", "write_bytes", "ensure_directory", "delete_file"} <= there and asked <= there, sorted(asked - there)


def test_a_picture_is_kept_by_the_write_it_is_really_kept_by(tmp_path):
    from core.rebuilding.her_build_writes import governed

    governed("ensure_directory", tmp_path / "pictures")
    governed("write_bytes", tmp_path / "pictures" / "icon.png", b"\x89PNG")
    assert (tmp_path / "pictures" / "icon.png").read_bytes() == b"\x89PNG"
