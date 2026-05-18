import core.security.permission_guard as permission_guard_module
from core.security.permission_guard import PermissionGuard


def test_automation_probe_is_not_applicable_off_macos(monkeypatch) -> None:
    monkeypatch.setattr(permission_guard_module.sys, "platform", "linux")

    result = PermissionGuard()._automation_preflight_probe()

    assert result == {"granted": True, "status": "not_applicable", "guidance": ""}


def test_automation_probe_reports_denied_native_error(monkeypatch) -> None:
    calls: list[str] = []

    def load_application():
        calls.append("load")
        raise RuntimeError("Not authorized to send Apple events (-1743)")

    monkeypatch.setattr(permission_guard_module.sys, "platform", "darwin")
    monkeypatch.setattr(permission_guard_module, "_load_scripting_bridge_application", load_application)

    result = PermissionGuard()._automation_preflight_probe()

    assert calls == ["load"]
    assert result["granted"] is False
    assert result["status"] == "denied"
    assert "Automation" in result["guidance"]


def test_automation_probe_reads_frontmost_process(monkeypatch) -> None:
    class DemoProcess:
        def __init__(self, name: str, frontmost: bool) -> None:
            self._name = name
            self._frontmost = frontmost

        def frontmost(self) -> bool:
            return self._frontmost

        def name(self) -> str:
            return self._name

    class DemoApplication:
        def processes(self):
            return [DemoProcess("Background", False), DemoProcess("Codex", True)]

    def application_with_bundle_identifier(_bundle_id: str):
        return DemoApplication()

    DemoApplication.applicationWithBundleIdentifier_ = staticmethod(application_with_bundle_identifier)
    monkeypatch.setattr(permission_guard_module.sys, "platform", "darwin")
    monkeypatch.setattr(permission_guard_module, "_load_scripting_bridge_application", lambda: DemoApplication)

    result = PermissionGuard()._automation_preflight_probe()

    assert result["granted"] is True
    assert result["status"] == "active"
    assert result["detail"] == "Codex"
