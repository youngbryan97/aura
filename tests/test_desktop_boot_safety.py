class _FakeMLX:
    cpu = object()
    gpu = object()

    def __init__(self, *, obeys_set: bool = True) -> None:
        self.current = self.gpu
        self.obeys_set = obeys_set
        self.set_calls = []

    def default_device(self):
        return self.current

    def set_default_device(self, device):
        self.set_calls.append(device)
        if self.obeys_set:
            self.current = device


def test_configure_inprocess_mlx_runtime_actually_pins_cpu(monkeypatch):
    from core.runtime import desktop_boot_safety as safety

    mx = _FakeMLX()

    monkeypatch.setattr(
        safety,
        "inprocess_mlx_metal_enabled",
        lambda *args, **kwargs: (False, "macos26_guard"),
    )
    monkeypatch.setattr(safety.importlib, "import_module", lambda _name: mx)

    result = safety.configure_inprocess_mlx_runtime(force=True)

    assert mx.set_calls == [mx.cpu]
    assert mx.default_device() is mx.cpu
    assert result["device"] == "cpu"
    assert result["reason"] == "macos26_guard"
    assert result["verified"] is True
    assert safety.mlx_process_uses_metal() is False


def test_mlx_device_contract_refuses_to_claim_a_setting_that_did_not_take(monkeypatch):
    from core.runtime import desktop_boot_safety as safety

    mx = _FakeMLX(obeys_set=False)
    monkeypatch.setattr(safety.importlib, "import_module", lambda _name: mx)

    result = safety.configure_mlx_process_device(
        "cpu",
        reason="test",
        force=True,
    )

    assert result["device"] == "unavailable"
    assert result["verified"] is False
    assert "device_configuration_failed" in result["reason"]


def test_model_process_can_own_a_separate_verified_metal_default(monkeypatch):
    from core.runtime import desktop_boot_safety as safety

    mx = _FakeMLX()
    monkeypatch.setattr(safety.importlib, "import_module", lambda _name: mx)

    result = safety.configure_mlx_process_device(
        "metal",
        reason="model_worker",
        force=True,
    )

    assert mx.set_calls == [mx.gpu]
    assert result == {
        "configured": True,
        "device": "metal",
        "reason": "model_worker",
        "verified": True,
    }
    assert safety.mlx_process_uses_metal() is True


def test_cpu_owned_parent_vram_purge_does_not_mutate_worker_metal_cache(monkeypatch):
    from core.managers import vram_manager
    from core.runtime import desktop_boot_safety as safety

    cache_clears = []

    class _Metal:
        @staticmethod
        def clear_cache():
            cache_clears.append("clear")

    monkeypatch.setattr(vram_manager, "MLX_AVAILABLE", True)
    monkeypatch.setattr(vram_manager, "mx", type("MX", (), {"metal": _Metal})())
    monkeypatch.setattr(safety, "mlx_process_uses_metal", lambda: False)

    vram_manager.VRAMManager().purge()

    assert cache_clears == []
