import asyncio
from importlib import metadata
from pathlib import Path

from infrastructure.operations import DepOps, FileOps, WebOps


def test_dependency_check_reports_missing_without_install() -> None:
    install_calls: list[tuple[str, ...]] = []

    def version_lookup(package: str) -> str:
        if package == "present-package":
            return "1.0"
        raise metadata.PackageNotFoundError(package)

    def installer(packages: tuple[str, ...]) -> int:
        install_calls.append(packages)
        return 0

    result = DepOps.check_and_install(
        required_packages=("present-package", "absent-package"),
        auto_install=False,
        installer=installer,
        version_lookup=version_lookup,
    )

    assert result == {
        "ok": False,
        "installed": ["present-package"],
        "missing": ["absent-package"],
        "attempted_install": False,
        "install_exit_code": None,
    }
    assert install_calls == []


def test_dependency_check_runs_installer_when_explicitly_allowed() -> None:
    install_calls: list[tuple[str, ...]] = []
    version_checks: list[str] = []

    def version_lookup(package: str) -> str:
        version_checks.append(package)
        raise metadata.PackageNotFoundError(package)

    def installer(packages) -> int:
        install_calls.append(tuple(packages))
        return 0

    result = DepOps.check_and_install(
        required_packages=("absent-one", "absent-two"),
        auto_install=True,
        installer=installer,
        version_lookup=version_lookup,
    )

    assert result["ok"] is True
    assert result["attempted_install"] is True
    assert result["install_exit_code"] == 0
    assert version_checks == ["absent-one", "absent-two"]
    assert install_calls == [("absent-one", "absent-two")]


def test_web_ops_fetches_with_injected_fetcher() -> None:
    calls: list[tuple[str, float, int]] = []

    def fetcher(url: str, timeout_seconds: float, max_bytes: int) -> str:
        calls.append((url, timeout_seconds, max_bytes))
        return "<html>Aura</html>"

    result = asyncio.run(
        WebOps.fetch_page_text(
            "https://example.com",
            timeout_seconds=2,
            max_bytes=1024,
            fetcher=fetcher,
        )
    )

    assert result == "<html>Aura</html>"
    assert calls == [("https://example.com", 2.0, 1024)]


def test_web_ops_rejects_non_http_url() -> None:
    result = asyncio.run(WebOps.fetch_page_text("file:///etc/passwd"))

    assert result is None


def test_file_ops_load_file_respects_byte_limit(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("abcdef", encoding="utf-8")

    assert FileOps.load_file(path, max_bytes=3) == "abc"


def test_file_ops_mtime_edit_requires_permission(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("aura", encoding="utf-8")
    before = path.stat().st_mtime

    assert FileOps.timestomp(path, timestamp=1_700_000_000, allow=False) is False
    assert path.stat().st_mtime == before

    assert FileOps.timestomp(path, timestamp=1_700_000_000, allow=True) is True
    assert int(path.stat().st_mtime) == 1_700_000_000
