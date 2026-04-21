import builtins


def test_configure_inprocess_mlx_runtime_skips_import_when_metal_is_disabled(monkeypatch):
    from core.runtime import desktop_boot_safety as safety

    real_import = builtins.__import__
    attempted = []

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mlx.core" or (name == "mlx" and fromlist and "core" in fromlist):
            attempted.append(name)
            raise AssertionError("mlx should not be imported when in-process metal is disabled")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(
        safety,
        "inprocess_mlx_metal_enabled",
        lambda *args, **kwargs: (False, "macos26_guard"),
    )
    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    result = safety.configure_inprocess_mlx_runtime(force=True)

    assert attempted == []
    assert result["device"] == "cpu"
    assert result["reason"] == "macos26_guard"
