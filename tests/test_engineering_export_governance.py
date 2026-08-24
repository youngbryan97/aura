from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.engineering.export import ExportBundle, ExportedFile

pytestmark = pytest.mark.unit


class _Gateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bytes | str | None]] = []

    def ensure_directory(self, path: str, *, source: str) -> None:
        self.calls.append(("directory", path, source))

    def write_bytes(self, path: str, payload: bytes, *, source: str) -> None:
        self.calls.append(("bytes", path, payload))

    def write_text(self, path: str, payload: str, *, source: str) -> None:
        self.calls.append(("text", path, payload))

    async def ensure_directory_async(self, path: str, *, source: str) -> None:
        self.ensure_directory(path, source=source)

    async def write_bytes_async(
        self, path: str, payload: bytes, *, source: str
    ) -> None:
        self.write_bytes(path, payload, source=source)

    async def write_text_async(
        self, path: str, payload: str, *, source: str
    ) -> None:
        self.write_text(path, payload, source=source)


def _bundle(*files: ExportedFile, slug: str = "safe_design") -> ExportBundle:
    return ExportBundle(slug=slug, files=files)


@pytest.mark.parametrize(
    ("slug", "name"),
    [
        ("../escape", "report.txt"),
        ("/tmp/escape", "report.txt"),
        ("safe_design", "../report.txt"),
        ("safe_design", "nested/report.txt"),
        ("safe_design", "/tmp/report.txt"),
    ],
)
def test_writer_rejects_noncanonical_names_before_gateway_access(
    monkeypatch: pytest.MonkeyPatch, slug: str, name: str
) -> None:
    import core.runtime.file_write_gateway as gateway_module
    from core.engineering.export import write_bundle

    monkeypatch.setattr(
        gateway_module,
        "get_file_write_gateway",
        lambda: pytest.fail("invalid names reached the write gateway"),
    )
    with pytest.raises(ValueError):
        write_bundle(_bundle(ExportedFile(name, "txt", text="x"), slug=slug))


def test_writer_rejects_duplicate_targets_before_gateway_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.runtime.file_write_gateway as gateway_module
    from core.engineering.export import write_bundle

    monkeypatch.setattr(
        gateway_module,
        "get_file_write_gateway",
        lambda: pytest.fail("duplicate names reached the write gateway"),
    )
    with pytest.raises(ValueError, match="duplicated"):
        write_bundle(
            _bundle(
                ExportedFile("same.txt", "txt", text="one"),
                ExportedFile("same.txt", "txt", text="two"),
            )
        )


def test_sync_writer_confines_destination_and_routes_every_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.runtime.file_write_gateway as gateway_module
    from core.engineering.export import write_bundle

    gateway = _Gateway()
    monkeypatch.setattr(gateway_module, "get_file_write_gateway", lambda: gateway)
    bundle = _bundle(
        ExportedFile("report.txt", "txt", text="answer"),
        ExportedFile("mesh.bin", "bin", data=b"mesh"),
    )

    written = write_bundle(bundle, "/tmp/outside-aura")

    expected_root = (
        Path(__file__).resolve().parents[1] / "artifacts" / "live_designs" / bundle.slug
    ).resolve()
    assert gateway.calls == [
        ("directory", str(expected_root), "engineering_export"),
        ("text", str(expected_root / "report.txt"), "answer"),
        ("bytes", str(expected_root / "mesh.bin"), b"mesh"),
    ]
    assert written.written == (
        str(expected_root / "report.txt"),
        str(expected_root / "mesh.bin"),
    )


def test_async_writer_uses_the_same_validation_and_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.runtime.file_write_gateway as gateway_module
    from core.engineering.export import write_bundle_async

    gateway = _Gateway()
    monkeypatch.setattr(gateway_module, "get_file_write_gateway", lambda: gateway)
    bundle = _bundle(ExportedFile("report.txt", "txt", text="answer"))

    written = asyncio.run(write_bundle_async(bundle, "requested/subdirectory"))

    expected = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "live_designs"
        / "requested"
        / "subdirectory"
        / bundle.slug
    ).resolve()
    assert gateway.calls == [
        ("directory", str(expected), "engineering_export"),
        ("text", str(expected / "report.txt"), "answer"),
    ]
    assert written.written == (str(expected / "report.txt"),)
