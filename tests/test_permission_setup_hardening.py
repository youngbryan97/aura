import asyncio

import core.security.permission_setup as permission_setup


def test_open_settings_pane_uses_injected_opener(monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(permission_setup.platform, "system", lambda: "Darwin")

    result = permission_setup.open_settings_pane("ACCESSIBILITY", opener=lambda url: opened.append(url) is None)

    assert result is True
    assert opened == [
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    ]


def test_open_settings_pane_rejects_unknown_permission(monkeypatch) -> None:
    monkeypatch.setattr(permission_setup.platform, "system", lambda: "Darwin")

    assert permission_setup.open_settings_pane("UNKNOWN", opener=lambda _url: True) is False


def test_check_all_permissions_records_typed_probe_failure(monkeypatch) -> None:
    class DemoGuard:
        async def check_permission(self, ptype, force=False):
            if ptype.name == "MIC":
                raise RuntimeError("mic backend unavailable")
            return {"granted": True, "available": True, "detail": f"{ptype.name}:{force}"}

        def get_guidance(self, ptype):
            return f"guidance:{ptype.name}"

    monkeypatch.setattr(permission_setup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(permission_setup, "get_permission_guard", lambda: DemoGuard())

    report = asyncio.run(permission_setup.check_all_permissions(refresh=True))

    mic_status = next(status for status in report.statuses if status.name == "MIC")
    assert report.supported is True
    assert report.all_granted is False
    assert "MIC" not in report.missing
    assert mic_status.granted is False
    assert mic_status.available is False
    assert mic_status.detail == "check_failed: RuntimeError"
