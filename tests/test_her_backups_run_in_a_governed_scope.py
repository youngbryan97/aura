"""Her state backup writes an archive, so it runs inside a governed scope.

Outside one the live runtime refused it ("subprocess_gateway.run:ops.backup.create called outside governed
context"): her last state backup is dated 13 September 2026, and every attempt from then to 10 November failed,
19 times in the last log alone.
"""
from __future__ import annotations

from types import SimpleNamespace

from core.governance_context import get_active_governance
from core.ops.backup import BackupManager


def test_the_archive_is_made_in_a_governed_scope_and_verified_read_only(tmp_path, monkeypatch):
    import core.runtime.subprocess_gateway as gateway

    calls = []

    class Gateway:
        def run(self, argv, **kwargs):
            calls.append((argv[2], kwargs.get("read_only", False), get_active_governance()))
            if argv[2] == "create":
                (tmp_path / "aura_state_20261110_120000-1.tar.gz").write_bytes(b"archive")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway, "get_subprocess_gateway", lambda: Gateway())
    manager = BackupManager()
    manager.backup_dir = tmp_path
    made = manager._run_verified_backup()
    assert made == tmp_path / "aura_state_20261110_120000-1.tar.gz"
    (create, create_read_only, create_token), (verify, verify_read_only, _token) = calls
    assert create == "create" and not create_read_only and create_token is not None
    assert verify == "verify" and verify_read_only
