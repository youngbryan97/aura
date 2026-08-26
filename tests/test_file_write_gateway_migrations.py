from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import sys

import pytest


class RecordingFileGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def write_text(self, path, text, *, source, encoding="utf-8") -> None:
        self.calls.append((str(path), encoding, source))
        path.write_text(text, encoding=encoding)

    # Async lane delegators: production code now calls *_async; fakes
    # must mirror the gateway surface or every governed write breaks.
    async def write_text_async(self, *args, **kwargs):
        return self.write_text(*args, **kwargs)


def test_feature_flags_save_uses_file_write_gateway(monkeypatch, tmp_path) -> None:
    import core.governance.feature_flags as feature_flags

    gateway = RecordingFileGateway()
    target = tmp_path / "feature_flags.json"
    monkeypatch.setattr(feature_flags, "get_file_write_gateway", lambda: gateway)

    flags = feature_flags.FeatureFlags(config_path=target)
    flags.set_flag("memory_dedup_on_write", False, reason="test")
    flags.save()

    assert json.loads(target.read_text(encoding="utf-8"))["memory_dedup_on_write"] is False
    assert gateway.calls == [(str(target), "utf-8", "core.governance.feature_flags.save")]


def test_outcome_ledger_save_uses_file_write_gateway(monkeypatch, tmp_path) -> None:
    import core.environment.outcome.ledger as outcome_ledger

    gateway = RecordingFileGateway()
    target = tmp_path / "outcomes.json"
    monkeypatch.setattr(outcome_ledger, "get_file_write_gateway", lambda: gateway)

    ledger = outcome_ledger.OutcomeLedger(target)
    ledger.record_outcome("inspect", "env", "ctx", True, 1.0, ["door_opened"])
    ledger.save()

    assert "env::ctx::inspect" in json.loads(target.read_text(encoding="utf-8"))
    assert gateway.calls == [(str(target), "utf-8", "core.environment.outcome.ledger.save")]


def test_belief_graph_save_uses_file_write_gateway(monkeypatch, tmp_path) -> None:
    import core.environment.belief_graph as belief_graph

    gateway = RecordingFileGateway()
    target = tmp_path / "belief.json"
    monkeypatch.setattr(belief_graph, "get_file_write_gateway", lambda: gateway)

    graph = belief_graph.EnvironmentBeliefGraph()
    graph.save(target)

    assert "nodes" in json.loads(target.read_text(encoding="utf-8"))
    assert gateway.calls == [(str(target), "utf-8", "core.environment.belief_graph.save")]


def test_semiotic_network_save_uses_file_write_gateway(monkeypatch, tmp_path) -> None:
    import core.grounding.semiotic_network as semiotic_network

    gateway = RecordingFileGateway()
    target = tmp_path / "semiotic.json"
    monkeypatch.setattr(semiotic_network, "get_file_write_gateway", lambda: gateway)

    network = semiotic_network.SemioticNetwork(target)
    network.save()

    assert "methods" in json.loads(target.read_text(encoding="utf-8"))
    assert gateway.calls == [(str(target), "utf-8", "core.grounding.semiotic_network.save")]


def test_unity_receipts_artifact_uses_file_write_gateway(monkeypatch, tmp_path) -> None:
    import core.unity.unity_receipts as unity_receipts

    gateway = RecordingFileGateway()
    target = tmp_path / "unity.json"
    monkeypatch.setattr(unity_receipts, "get_file_write_gateway", lambda: gateway)

    assert unity_receipts.write_unity_results_artifact(target, {"ok": True}) == target

    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
    assert gateway.calls == [
        (str(target), "utf-8", "core.unity.unity_receipts.write_results_artifact")
    ]


def test_proof_obligations_bytecode_check_uses_file_write_gateway(monkeypatch) -> None:
    import core.learning.proof_obligations as proof_obligations

    gateway = RecordingFileGateway()
    monkeypatch.setattr(proof_obligations, "get_file_write_gateway", lambda: gateway)

    ok, diagnostics = proof_obligations.ProofObligationEngine._bytecode_compiles(
        "x = 1\n",
        "candidate.py",
    )

    assert ok is True
    assert diagnostics == {"ok": True}
    assert gateway.calls
    assert gateway.calls[0][2] == "core.learning.proof_obligations.bytecode_compiles"


def test_plugin_allowlist_save_uses_file_write_gateway(monkeypatch, tmp_path) -> None:
    import core.security.plugin_allowlist as plugin_allowlist

    gateway = RecordingFileGateway()
    allowlist_path = tmp_path / "allow.json"
    plugin_path = tmp_path / "plugin.py"
    plugin_path.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(plugin_allowlist, "get_file_write_gateway", lambda: gateway)

    allowlist = plugin_allowlist.PluginAllowlist(allowlist_path)
    allowlist.record(plugin_path, approved_by="test", reason="unit")

    assert "entries" in json.loads(allowlist_path.read_text(encoding="utf-8"))
    assert gateway.calls == [(str(allowlist_path), "utf-8", "core.security.plugin_allowlist.save")]


def test_reddit_session_save_uses_file_write_gateway(monkeypatch, tmp_path) -> None:
    import core.skills.reddit_adapter as reddit_adapter

    gateway = RecordingFileGateway()
    target = tmp_path / "reddit_state.json"
    monkeypatch.setattr(reddit_adapter, "_STORAGE_STATE_FILE", target)
    monkeypatch.setattr(reddit_adapter, "get_file_write_gateway", lambda: gateway)

    class Context:
        async def cookies(self):
            return [{"name": "session", "value": "abc"}]

    class Browser:
        context = Context()

    skill = reddit_adapter.RedditAdapterSkill()
    asyncio.run(skill._save_session(Browser()))

    assert json.loads(target.read_text(encoding="utf-8"))["cookies"][0]["name"] == "session"
    assert gateway.calls == [(str(target), "utf-8", "core.skills.reddit_adapter.save_session")]


def test_file_write_gateway_drain_text_atomically_removes_drained_file(tmp_path) -> None:
    from core.runtime.file_write_gateway import FileWriteGateway

    gateway = FileWriteGateway()
    target = tmp_path / "queue.jsonl"

    gateway.append_text(target, '{"one": 1}\n', source="unit.append")
    gateway.append_text(target, '{"two": 2}\n', source="unit.append")

    drained = gateway.drain_text(target, source="unit.drain")

    assert drained.splitlines() == ['{"one": 1}', '{"two": 2}']
    assert not target.exists()
    assert gateway.drain_text(target, source="unit.drain") == ""


@pytest.mark.asyncio
async def test_file_write_gateway_atomically_replaces_directory_symlink(tmp_path) -> None:
    from core.runtime.file_write_gateway import FileWriteGateway

    gateway = FileWriteGateway()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    link = tmp_path / "active"

    gateway.replace_symlink(link, first, source="unit.symlink")
    assert link.is_symlink()
    assert link.resolve() == first.resolve()

    await gateway.replace_symlink_async(link, second, source="unit.symlink")
    assert link.resolve() == second.resolve()
    assert list(tmp_path.glob(".*.symlink.tmp")) == []
    assert gateway.delete_file(link, source="unit.delete_symlink") is True
    assert not link.is_symlink()


def test_file_write_gateway_refuses_to_replace_real_directory(tmp_path) -> None:
    from core.runtime.file_write_gateway import FileWriteGateway

    gateway = FileWriteGateway()
    target = tmp_path / "target"
    link = tmp_path / "active"
    target.mkdir()
    link.mkdir()

    with pytest.raises(IsADirectoryError, match="refusing to replace directory"):
        gateway.replace_symlink(link, target, source="unit.symlink")


def test_file_write_gateway_batch_commits_private_files_with_receipt(tmp_path) -> None:
    from core.runtime.file_write_gateway import FileWriteBatchEntry, FileWriteGateway

    gateway = FileWriteGateway()
    key_path = tmp_path / "server.key"
    cert_path = tmp_path / "server.crt"

    receipt = gateway.write_bytes_batch(
        (
            FileWriteBatchEntry(key_path, b"private-key", mode=0o600),
            FileWriteBatchEntry(cert_path, b"certificate", mode=0o644),
        ),
        source="unit.batch",
    )

    assert key_path.read_bytes() == b"private-key"
    assert cert_path.read_bytes() == b"certificate"
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(cert_path.stat().st_mode) == 0o644
    assert receipt.paths == (str(key_path), str(cert_path))
    assert dict(receipt.sha256) == {
        str(key_path): hashlib.sha256(b"private-key").hexdigest(),
        str(cert_path): hashlib.sha256(b"certificate").hexdigest(),
    }
    assert len(receipt.transaction_id) == 32


def test_file_write_gateway_batch_restores_prior_targets_on_failure(
    monkeypatch,
    tmp_path,
) -> None:
    import core.runtime.file_write_gateway as file_write_gateway

    gateway = file_write_gateway.FileWriteGateway()
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"old-first")
    second.write_bytes(b"old-second")
    first.chmod(0o640)
    second.chmod(0o600)
    real_atomic_write = file_write_gateway.atomic_write_bytes
    calls = 0

    def fail_second_write(path, payload, *, durable=True, mode=0o600):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second replacement failure")
        return real_atomic_write(path, payload, durable=durable, mode=mode)

    monkeypatch.setattr(file_write_gateway, "atomic_write_bytes", fail_second_write)

    with pytest.raises(file_write_gateway.FileWriteTransactionError) as exc_info:
        gateway.write_bytes_batch(
            (
                file_write_gateway.FileWriteBatchEntry(first, b"new-first"),
                file_write_gateway.FileWriteBatchEntry(second, b"new-second"),
            ),
            source="unit.batch.failure",
        )

    assert "prior targets restored" in str(exc_info.value)
    assert first.read_bytes() == b"old-first"
    assert second.read_bytes() == b"old-second"
    assert stat.S_IMODE(first.stat().st_mode) == 0o640
    assert stat.S_IMODE(second.stat().st_mode) == 0o600


def test_file_write_gateway_batch_rejects_ambiguous_targets(tmp_path) -> None:
    from core.runtime.file_write_gateway import (
        FileWriteBatchEntry,
        FileWriteGateway,
        FileWriteTransactionError,
    )

    gateway = FileWriteGateway()
    first = tmp_path / "first" / "value.bin"
    second = tmp_path / "second" / "value.bin"

    with pytest.raises(ValueError, match="share one directory"):
        gateway.write_bytes_batch(
            (FileWriteBatchEntry(first, b"one"), FileWriteBatchEntry(second, b"two")),
            source="unit.batch.cross_directory",
        )

    target = tmp_path / "target.bin"
    with pytest.raises(ValueError, match="duplicate batch target"):
        gateway.write_bytes_batch(
            (FileWriteBatchEntry(target, b"one"), FileWriteBatchEntry(target, b"two")),
            source="unit.batch.duplicate",
        )

    backing = tmp_path / "backing.bin"
    backing.write_bytes(b"unchanged")
    link = tmp_path / "link.bin"
    link.symlink_to(backing)
    with pytest.raises(FileWriteTransactionError, match="symlink"):
        gateway.write_bytes_batch(
            (FileWriteBatchEntry(link, b"replacement"),),
            source="unit.batch.symlink",
        )
    assert backing.read_bytes() == b"unchanged"

    lock_backing = tmp_path / "lock-backing.bin"
    lock_backing.write_bytes(b"do-not-follow")
    lock_path = tmp_path / ".aura_file_write_batch.lock"
    lock_path.symlink_to(lock_backing)
    with pytest.raises(OSError):
        gateway.write_bytes_batch(
            (FileWriteBatchEntry(target, b"replacement"),),
            source="unit.batch.lock_symlink",
        )
    assert lock_backing.read_bytes() == b"do-not-follow"


def test_directory_relative_batch_commits_exact_generation(tmp_path) -> None:
    from core.runtime.file_write_gateway import (
        DirectoryFileWriteBatchEntry,
        FileWriteGateway,
    )

    gateway = FileWriteGateway()
    receipt = gateway.write_bytes_batch_in_directory(
        tmp_path,
        (
            DirectoryFileWriteBatchEntry("data.bin", b"payload", 0o640),
            DirectoryFileWriteBatchEntry("manifest.json", b"{}", 0o600),
        ),
        allowed_existing_names={"data.bin", "manifest.json"},
        commit_marker="manifest.json",
        source="unit.directory_batch",
    )

    assert (tmp_path / "data.bin").read_bytes() == b"payload"
    assert (tmp_path / "manifest.json").read_bytes() == b"{}"
    assert stat.S_IMODE((tmp_path / "data.bin").stat().st_mode) == 0o640
    assert set(receipt.paths) == {
        str(tmp_path / "data.bin"),
        str(tmp_path / "manifest.json"),
    }
    assert {
        path.name for path in tmp_path.iterdir()
    } == {"data.bin", "manifest.json", ".aura_file_write_batch.lock"}


def test_directory_relative_batch_rejects_unexpected_and_symlink_entries(
    tmp_path,
) -> None:
    from core.runtime.file_write_gateway import (
        DirectoryFileWriteBatchEntry,
        FileWriteGateway,
        FileWriteTransactionError,
    )

    gateway = FileWriteGateway()
    directory = tmp_path / "package"
    directory.mkdir(mode=0o700)
    (directory / "unexpected.txt").write_text("foreign")
    with pytest.raises(FileWriteTransactionError, match="unexpected entries"):
        gateway.write_bytes_batch_in_directory(
            directory,
            (DirectoryFileWriteBatchEntry("manifest.json", b"{}"),),
            allowed_existing_names={"manifest.json"},
            commit_marker="manifest.json",
            source="unit.directory_batch.unexpected",
        )

    (directory / "unexpected.txt").unlink()
    backing = tmp_path / "backing"
    backing.write_bytes(b"unchanged")
    (directory / "manifest.json").symlink_to(backing)
    with pytest.raises(FileWriteTransactionError, match="non-regular"):
        gateway.write_bytes_batch_in_directory(
            directory,
            (DirectoryFileWriteBatchEntry("manifest.json", b"{}"),),
            allowed_existing_names={"manifest.json"},
            commit_marker="manifest.json",
            source="unit.directory_batch.symlink",
        )
    assert backing.read_bytes() == b"unchanged"


def test_directory_relative_batch_rolls_back_second_replace_failure(
    monkeypatch,
    tmp_path,
) -> None:
    import core.runtime.file_write_gateway as gateway_module

    gateway = gateway_module.FileWriteGateway()
    (tmp_path / "a.bin").write_bytes(b"old-a")
    (tmp_path / "manifest.json").write_bytes(b"old-manifest")
    real_replace = gateway_module.os.replace
    failed = False

    def fail_manifest_once(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        nonlocal failed
        if dst == "manifest.json" and not failed:
            failed = True
            raise OSError("injected marker replacement failure")
        return real_replace(
            src,
            dst,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(gateway_module.os, "replace", fail_manifest_once)
    with pytest.raises(
        gateway_module.FileWriteTransactionError,
        match="originals restored",
    ):
        gateway.write_bytes_batch_in_directory(
            tmp_path,
            (
                gateway_module.DirectoryFileWriteBatchEntry(
                    "a.bin",
                    b"new-a",
                ),
                gateway_module.DirectoryFileWriteBatchEntry(
                    "manifest.json",
                    b"new-manifest",
                ),
            ),
            allowed_existing_names={"a.bin", "manifest.json"},
            commit_marker="manifest.json",
            source="unit.directory_batch.rollback",
        )
    assert (tmp_path / "a.bin").read_bytes() == b"old-a"
    assert (tmp_path / "manifest.json").read_bytes() == b"old-manifest"


def test_directory_relative_batch_cleans_staging_failure_and_retries(
    monkeypatch,
    tmp_path,
) -> None:
    import core.runtime.file_write_gateway as gateway_module
    import core.runtime.file_write_primitives as primitives_module

    gateway = gateway_module.FileWriteGateway()
    real_stage = primitives_module.stage_bytes_at
    failed = False

    def fail_after_stage(directory_fd, name, payload, mode):
        nonlocal failed
        real_stage(directory_fd, name, payload, mode)
        if name.endswith("-0.tmp") and not failed:
            failed = True
            raise OSError("injected staging fsync failure")

    monkeypatch.setattr(
        primitives_module,
        "stage_bytes_at",
        fail_after_stage,
    )
    with pytest.raises(
        gateway_module.FileWriteTransactionError,
        match="originals restored",
    ):
        gateway.write_bytes_batch_in_directory(
            tmp_path,
            (
                gateway_module.DirectoryFileWriteBatchEntry(
                    "data.bin",
                    b"data",
                ),
                gateway_module.DirectoryFileWriteBatchEntry(
                    "manifest.json",
                    b"manifest",
                ),
            ),
            allowed_existing_names={"data.bin", "manifest.json"},
            commit_marker="manifest.json",
            source="unit.directory_batch.stage_failure",
        )
    assert {
        path.name for path in tmp_path.iterdir()
    } == {".aura_file_write_batch.lock"}

    monkeypatch.setattr(primitives_module, "stage_bytes_at", real_stage)
    gateway.write_bytes_batch_in_directory(
        tmp_path,
        (
            gateway_module.DirectoryFileWriteBatchEntry(
                "data.bin",
                b"data",
            ),
            gateway_module.DirectoryFileWriteBatchEntry(
                "manifest.json",
                b"manifest",
            ),
        ),
        allowed_existing_names={"data.bin", "manifest.json"},
        commit_marker="manifest.json",
        source="unit.directory_batch.stage_retry",
    )
    assert (tmp_path / "data.bin").read_bytes() == b"data"


def test_directory_relative_batch_recovers_process_death_before_retry(
    monkeypatch,
    tmp_path,
) -> None:
    import core.runtime.file_write_gateway as gateway_module
    import core.runtime.file_write_primitives as primitives_module
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    (tmp_path / "data.bin").write_bytes(b"old-data")
    (tmp_path / "manifest.json").write_bytes(b"old-manifest")
    script = """
import os
import sys
import core.runtime.file_write_gateway as gateway_module

target = sys.argv[1]
real_replace = gateway_module.os.replace

def crash_after_first_target(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
    result = real_replace(
        src,
        dst,
        src_dir_fd=src_dir_fd,
        dst_dir_fd=dst_dir_fd,
    )
    if dst == "data.bin":
        os._exit(91)
    return result

gateway_module.os.replace = crash_after_first_target
gateway_module.FileWriteGateway().write_bytes_batch_in_directory(
    target,
    (
        gateway_module.DirectoryFileWriteBatchEntry("data.bin", b"new-data"),
        gateway_module.DirectoryFileWriteBatchEntry(
            "manifest.json",
            b"new-manifest",
        ),
    ),
    allowed_existing_names={"data.bin", "manifest.json"},
    commit_marker="manifest.json",
    source="unit.directory_batch.process_death",
)
"""
    crashed = get_subprocess_gateway().run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=False,
        capture_output=True,
        offline_tooling=True,
        source="certification_tooling:directory_batch_process_death",
        accelerator_capability="none",
    )
    assert crashed.returncode == 91
    assert (tmp_path / ".aura_file_write_batch.journal").exists()

    real_stage = primitives_module.stage_bytes_at

    def stop_after_recovery(directory_fd, name, payload, mode):
        if "-recover-" in name:
            return real_stage(directory_fd, name, payload, mode)
        raise OSError("stop after recovery")

    monkeypatch.setattr(
        primitives_module,
        "stage_bytes_at",
        stop_after_recovery,
    )
    with pytest.raises(gateway_module.FileWriteTransactionError):
        gateway_module.FileWriteGateway().write_bytes_batch_in_directory(
            tmp_path,
            (
                gateway_module.DirectoryFileWriteBatchEntry(
                    "data.bin",
                    b"final-data",
                ),
                gateway_module.DirectoryFileWriteBatchEntry(
                    "manifest.json",
                    b"final-manifest",
                ),
            ),
            allowed_existing_names={"data.bin", "manifest.json"},
            commit_marker="manifest.json",
            source="unit.directory_batch.recovery_probe",
        )
    assert (tmp_path / "data.bin").read_bytes() == b"old-data"
    assert (tmp_path / "manifest.json").read_bytes() == b"old-manifest"
    assert not (tmp_path / ".aura_file_write_batch.journal").exists()

    monkeypatch.setattr(primitives_module, "stage_bytes_at", real_stage)
    gateway_module.FileWriteGateway().write_bytes_batch_in_directory(
        tmp_path,
        (
            gateway_module.DirectoryFileWriteBatchEntry(
                "data.bin",
                b"final-data",
            ),
            gateway_module.DirectoryFileWriteBatchEntry(
                "manifest.json",
                b"final-manifest",
            ),
        ),
        allowed_existing_names={"data.bin", "manifest.json"},
        commit_marker="manifest.json",
        source="unit.directory_batch.recovery_commit",
    )
    assert (tmp_path / "data.bin").read_bytes() == b"final-data"


def test_directory_relative_batch_commit_cleanup_survives_process_death(
    monkeypatch,
    tmp_path,
) -> None:
    import core.runtime.file_write_gateway as gateway_module
    import core.runtime.file_write_primitives as primitives_module
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    (tmp_path / "data.bin").write_bytes(b"old-data")
    (tmp_path / "manifest.json").write_bytes(b"old-manifest")
    script = """
import os
import sys
import core.runtime.file_write_gateway as gateway_module
import core.runtime.file_write_primitives as primitives_module

target = sys.argv[1]
real_unlink = primitives_module.unlink_private_regular_at

def crash_on_disposable_backup(directory_fd, name):
    if name.endswith(".bak"):
        os._exit(92)
    return real_unlink(directory_fd, name)

primitives_module.unlink_private_regular_at = crash_on_disposable_backup
gateway_module.FileWriteGateway().write_bytes_batch_in_directory(
    target,
    (
        gateway_module.DirectoryFileWriteBatchEntry("data.bin", b"new-data"),
        gateway_module.DirectoryFileWriteBatchEntry(
            "manifest.json", b"new-manifest"
        ),
    ),
    allowed_existing_names={"data.bin", "manifest.json"},
    commit_marker="manifest.json",
    source="unit.directory_batch.commit_cleanup_death",
)
"""
    crashed = get_subprocess_gateway().run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=False,
        capture_output=True,
        offline_tooling=True,
        source="certification_tooling:directory_batch_commit_cleanup_death",
        accelerator_capability="none",
    )
    assert crashed.returncode == 92
    assert (tmp_path / "data.bin").read_bytes() == b"new-data"
    assert not (tmp_path / ".aura_file_write_batch.journal").exists()
    assert any(path.name.endswith(".bak") for path in tmp_path.iterdir())

    real_stage = primitives_module.stage_bytes_at

    def stop_new_transaction(*_args, **_kwargs):
        raise OSError("stop after abandoned-backup cleanup")

    monkeypatch.setattr(
        primitives_module,
        "stage_bytes_at",
        stop_new_transaction,
    )
    with pytest.raises(gateway_module.FileWriteTransactionError):
        gateway_module.FileWriteGateway().write_bytes_batch_in_directory(
            tmp_path,
            (
                gateway_module.DirectoryFileWriteBatchEntry(
                    "data.bin", b"final-data"
                ),
                gateway_module.DirectoryFileWriteBatchEntry(
                    "manifest.json", b"final-manifest"
                ),
            ),
            allowed_existing_names={"data.bin", "manifest.json"},
            commit_marker="manifest.json",
            source="unit.directory_batch.commit_cleanup_probe",
        )
    assert (tmp_path / "data.bin").read_bytes() == b"new-data"
    assert not any(path.name.endswith(".bak") for path in tmp_path.iterdir())

    monkeypatch.setattr(primitives_module, "stage_bytes_at", real_stage)


def test_directory_relative_batch_recovery_cleanup_survives_process_death(
    tmp_path,
) -> None:
    import core.runtime.file_write_gateway as gateway_module
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    (tmp_path / "data.bin").write_bytes(b"old-data")
    (tmp_path / "manifest.json").write_bytes(b"old-manifest")
    crash_replace = """
import os
import sys
import core.runtime.file_write_gateway as gateway_module

target = sys.argv[1]
real_replace = gateway_module.os.replace
def crash(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
    result = real_replace(
        src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd
    )
    if dst == "data.bin":
        os._exit(93)
    return result
gateway_module.os.replace = crash
gateway_module.FileWriteGateway().write_bytes_batch_in_directory(
    target,
    (
        gateway_module.DirectoryFileWriteBatchEntry("data.bin", b"new-data"),
        gateway_module.DirectoryFileWriteBatchEntry(
            "manifest.json", b"new-manifest"
        ),
    ),
    allowed_existing_names={"data.bin", "manifest.json"},
    commit_marker="manifest.json",
    source="unit.directory_batch.recovery_cleanup_setup",
)
"""
    first = get_subprocess_gateway().run(
        [sys.executable, "-c", crash_replace, str(tmp_path)],
        check=False,
        capture_output=True,
        offline_tooling=True,
        source="certification_tooling:directory_batch_recovery_cleanup_setup",
        accelerator_capability="none",
    )
    assert first.returncode == 93
    assert (tmp_path / ".aura_file_write_batch.journal").exists()

    crash_cleanup = """
import os
import sys
import core.runtime.file_write_gateway as gateway_module
import core.runtime.file_write_primitives as primitives_module

target = sys.argv[1]
real_unlink = primitives_module.unlink_private_regular_at
def crash(directory_fd, name):
    if name.endswith(".bak"):
        os._exit(94)
    return real_unlink(directory_fd, name)
primitives_module.unlink_private_regular_at = crash
gateway_module.FileWriteGateway().write_bytes_batch_in_directory(
    target,
    (
        gateway_module.DirectoryFileWriteBatchEntry("data.bin", b"final-data"),
        gateway_module.DirectoryFileWriteBatchEntry(
            "manifest.json", b"final-manifest"
        ),
    ),
    allowed_existing_names={"data.bin", "manifest.json"},
    commit_marker="manifest.json",
    source="unit.directory_batch.recovery_cleanup_death",
)
"""
    second = get_subprocess_gateway().run(
        [sys.executable, "-c", crash_cleanup, str(tmp_path)],
        check=False,
        capture_output=True,
        offline_tooling=True,
        source="certification_tooling:directory_batch_recovery_cleanup_death",
        accelerator_capability="none",
    )
    assert second.returncode == 94
    assert (tmp_path / "data.bin").read_bytes() == b"old-data"
    assert (tmp_path / "manifest.json").read_bytes() == b"old-manifest"
    assert not (tmp_path / ".aura_file_write_batch.journal").exists()

    gateway_module.FileWriteGateway().write_bytes_batch_in_directory(
        tmp_path,
        (
            gateway_module.DirectoryFileWriteBatchEntry(
                "data.bin", b"final-data"
            ),
            gateway_module.DirectoryFileWriteBatchEntry(
                "manifest.json", b"final-manifest"
            ),
        ),
        allowed_existing_names={"data.bin", "manifest.json"},
        commit_marker="manifest.json",
        source="unit.directory_batch.recovery_cleanup_commit",
    )
    assert (tmp_path / "data.bin").read_bytes() == b"final-data"


def test_directory_relative_batch_refuses_unrecoverable_journal_size(
    monkeypatch,
    tmp_path,
) -> None:
    import core.runtime.file_write_batch_journal as journal_module
    import core.runtime.file_write_gateway as gateway_module

    (tmp_path / "data.bin").write_bytes(b"old-data")
    monkeypatch.setattr(
        journal_module,
        "MAX_DIRECTORY_BATCH_JOURNAL_BYTES",
        128,
    )
    with pytest.raises(
        gateway_module.FileWriteTransactionError,
        match="originals restored",
    ):
        gateway_module.FileWriteGateway().write_bytes_batch_in_directory(
            tmp_path,
            (gateway_module.DirectoryFileWriteBatchEntry("data.bin", b"new"),),
            allowed_existing_names={"data.bin"},
            commit_marker="data.bin",
            source="unit.directory_batch.journal_bound",
        )
    assert (tmp_path / "data.bin").read_bytes() == b"old-data"
    assert not (tmp_path / ".aura_file_write_batch.journal").exists()
    assert not any(
        path.name.startswith(".aura-batch-") for path in tmp_path.iterdir()
    )


def test_directory_relative_batch_reports_durable_committed_state(
    monkeypatch,
    tmp_path,
) -> None:
    import core.runtime.file_write_gateway as gateway_module

    (tmp_path / "data.bin").write_bytes(b"old-data")
    real_write_journal = gateway_module._write_directory_batch_journal

    def fail_after_committed_journal(*args, state, **kwargs):
        real_write_journal(*args, state=state, **kwargs)
        if state == "committed":
            raise OSError("failure after durable committed journal")

    monkeypatch.setattr(
        gateway_module,
        "_write_directory_batch_journal",
        fail_after_committed_journal,
    )
    with pytest.raises(
        gateway_module.FileWriteTransactionError,
        match="committed but durable cleanup",
    ):
        gateway_module.FileWriteGateway().write_bytes_batch_in_directory(
            tmp_path,
            (gateway_module.DirectoryFileWriteBatchEntry("data.bin", b"new"),),
            allowed_existing_names={"data.bin"},
            commit_marker="data.bin",
            source="unit.directory_batch.committed_state",
        )
    assert (tmp_path / "data.bin").read_bytes() == b"new"
    assert not (tmp_path / ".aura_file_write_batch.journal").exists()


def test_directory_relative_batch_classifies_persistent_committed_cleanup_failure(
    monkeypatch,
    tmp_path,
) -> None:
    import core.runtime.file_write_gateway as gateway_module
    import core.runtime.file_write_primitives as primitives_module

    (tmp_path / "data.bin").write_bytes(b"old-data")
    real_write_journal = gateway_module._write_directory_batch_journal
    real_unlink = primitives_module.unlink_private_regular_at

    def fail_after_committed_journal(*args, state, **kwargs):
        real_write_journal(*args, state=state, **kwargs)
        if state == "committed":
            raise OSError("failure after durable committed journal")

    def refuse_committed_journal_cleanup(directory_fd, name):
        if name == primitives_module.DIRECTORY_BATCH_JOURNAL_FILE:
            raise OSError("persistent committed cleanup failure")
        return real_unlink(directory_fd, name)

    monkeypatch.setattr(
        gateway_module,
        "_write_directory_batch_journal",
        fail_after_committed_journal,
    )
    monkeypatch.setattr(
        primitives_module,
        "unlink_private_regular_at",
        refuse_committed_journal_cleanup,
    )
    with pytest.raises(
        gateway_module.FileWriteTransactionError,
        match="committed but durable cleanup is incomplete",
    ):
        gateway_module.FileWriteGateway().write_bytes_batch_in_directory(
            tmp_path,
            (gateway_module.DirectoryFileWriteBatchEntry("data.bin", b"new"),),
            allowed_existing_names={"data.bin"},
            commit_marker="data.bin",
            source="unit.directory_batch.persistent_committed_cleanup",
        )
    assert (tmp_path / "data.bin").read_bytes() == b"new"
    assert (tmp_path / ".aura_file_write_batch.journal").exists()
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert gateway_module._load_directory_batch_journal(directory_fd)[
            "state"
        ] == "committed"
    finally:
        os.close(directory_fd)


def test_directory_relative_batch_rejects_hardlinked_lock_without_chmod(
    tmp_path,
) -> None:
    import fcntl

    import core.runtime.file_write_gateway as gateway_module

    backing = tmp_path.parent / f"{tmp_path.name}-lock-backing"
    backing.write_bytes(b"lock")
    backing.chmod(0o640)
    os.link(backing, tmp_path / ".aura_file_write_batch.lock")
    descriptor = os.open(backing, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        with pytest.raises(
            gateway_module.FileWriteTransactionError,
            match="lock path no longer binds",
        ):
            gateway_module.FileWriteGateway().write_bytes_batch_in_directory(
                tmp_path,
                (
                    gateway_module.DirectoryFileWriteBatchEntry(
                        "manifest.json", b"{}"
                    ),
                ),
                allowed_existing_names={"manifest.json"},
                commit_marker="manifest.json",
                source="unit.directory_batch.hardlink_lock",
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert stat.S_IMODE(backing.stat().st_mode) == 0o640


def test_directory_relative_batch_never_writes_to_swapped_path(
    monkeypatch,
    tmp_path,
) -> None:
    import core.runtime.file_write_gateway as gateway_module

    directory = tmp_path / "bound"
    moved = tmp_path / "moved"
    directory.mkdir(mode=0o700)
    (directory / "data.bin").write_bytes(b"old-data")
    (directory / "manifest.json").write_bytes(b"old-manifest")
    gateway = gateway_module.FileWriteGateway()
    real_replace = gateway_module.os.replace
    swapped = False

    def swap_path_then_replace(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        nonlocal swapped
        if not swapped:
            swapped = True
            directory.rename(moved)
            directory.mkdir(mode=0o700)
        return real_replace(
            src,
            dst,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(gateway_module.os, "replace", swap_path_then_replace)
    with pytest.raises(
        gateway_module.FileWriteTransactionError,
        match="originals restored",
    ):
        gateway.write_bytes_batch_in_directory(
            directory,
            (
                gateway_module.DirectoryFileWriteBatchEntry(
                    "data.bin",
                    b"new-data",
                ),
                gateway_module.DirectoryFileWriteBatchEntry(
                    "manifest.json",
                    b"new-manifest",
                ),
            ),
            allowed_existing_names={"data.bin", "manifest.json"},
            commit_marker="manifest.json",
            source="unit.directory_batch.path_swap",
        )
    assert not any(directory.iterdir())
    assert (moved / "data.bin").read_bytes() == b"old-data"
    assert (moved / "manifest.json").read_bytes() == b"old-manifest"


def test_file_write_gateway_owned_binary_is_narrow_private_and_no_follow(tmp_path) -> None:
    from core.runtime.file_write_gateway import FileWriteGateway

    gateway = FileWriteGateway()
    target = tmp_path / "ring.bin"
    with gateway.open_owned_binary(
        target,
        mode="w+b",
        permissions=0o640,
        source="unit.owned_binary",
    ) as handle:
        handle.write(b"ring")
        handle.flush()
        os.fsync(handle.fileno())

    assert target.read_bytes() == b"ring"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    with pytest.raises(ValueError, match="unsupported owned binary mode"):
        gateway.open_owned_binary(target, mode="wb", source="unit.invalid_mode")

    backing = tmp_path / "backing.bin"
    backing.write_bytes(b"unchanged")
    link = tmp_path / "ring-link.bin"
    link.symlink_to(backing)
    with pytest.raises(OSError, match="symlink"):
        gateway.open_owned_binary(link, mode="r+b", source="unit.owned_symlink")


def test_file_write_gateway_replace_file_durably_moves_source(tmp_path) -> None:
    from core.runtime.file_write_gateway import FileWriteGateway

    gateway = FileWriteGateway()
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")

    assert gateway.replace_file(source, destination, source="unit.replace") == str(
        destination
    )
    assert not source.exists()
    assert destination.read_bytes() == b"new"

    new_source = tmp_path / "new-source.bin"
    backing = tmp_path / "backing.bin"
    link = tmp_path / "destination-link.bin"
    new_source.write_bytes(b"replacement")
    backing.write_bytes(b"unchanged")
    link.symlink_to(backing)
    with pytest.raises(OSError, match="symlink"):
        gateway.replace_file(new_source, link, source="unit.replace_symlink")
    assert new_source.read_bytes() == b"replacement"
    assert backing.read_bytes() == b"unchanged"


def test_file_write_gateway_owns_binary_adapter_close_and_durable_flush(tmp_path) -> None:
    from core.runtime.file_write_gateway import FileWriteGateway

    gateway = FileWriteGateway()
    target = tmp_path / "stream.bin"
    closed: list[bool] = []

    class _Adapter:
        def __init__(self, handle, *, prefix: bytes) -> None:
            self.handle = handle
            self.handle.write(prefix)

        def write(self, payload: bytes) -> None:
            self.handle.write(payload)

        def close(self) -> None:
            closed.append(True)

    with gateway.open_owned_binary_adapter(
        target,
        mode="w+b",
        adapter=_Adapter,
        adapter_kwargs={"prefix": b"header:"},
        source="unit.binary_adapter",
    ) as adapter:
        adapter.write(b"payload")

    assert closed == [True]
    assert target.read_bytes() == b"header:payload"


def test_file_write_gateway_rejects_noncallable_binary_adapter(tmp_path) -> None:
    from core.runtime.file_write_gateway import FileWriteGateway

    gateway = FileWriteGateway()
    with pytest.raises(TypeError, match="binary adapter must be callable"):
        with gateway.open_owned_binary_adapter(
            tmp_path / "stream.bin",
            mode="w+b",
            adapter=None,
            source="unit.invalid_binary_adapter",
        ):
            pass


def test_write_bytes_if_absent_applies_requested_private_mode(tmp_path) -> None:
    from core.runtime.file_write_gateway import FileWriteGateway

    target = tmp_path / "identity.key"
    gateway = FileWriteGateway()

    assert gateway.write_bytes_if_absent(
        target,
        b"first",
        mode=0o640,
        source="unit.write_once_mode",
    ) is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert gateway.write_bytes_if_absent(
        target,
        b"second",
        mode=0o600,
        source="unit.write_once_mode_existing",
    ) is False
    assert target.read_bytes() == b"first"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_provision_private_bytes_rejects_unsafe_existing_identity(tmp_path) -> None:
    from core.runtime.file_write_gateway import (
        FileWriteGateway,
        FileWriteTransactionError,
    )

    gateway = FileWriteGateway()
    target = tmp_path / "identity.key"
    winner = b"w" * 32

    assert gateway.provision_private_bytes(
        target,
        winner,
        expected_size=32,
        source="unit.private_identity",
    ) == winner
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert gateway.provision_private_bytes(
        target,
        b"l" * 32,
        expected_size=32,
        source="unit.private_identity_loser",
    ) == winner

    target.chmod(0o644)
    with pytest.raises(FileWriteTransactionError, match="permissions"):
        gateway.provision_private_bytes(
            target,
            b"x" * 32,
            expected_size=32,
            source="unit.private_identity_world_readable",
        )

    target.unlink()
    backing = tmp_path / "backing.key"
    backing.write_bytes(b"b" * 32)
    backing.chmod(0o600)
    target.symlink_to(backing)
    with pytest.raises((OSError, FileWriteTransactionError)):
        gateway.provision_private_bytes(
            target,
            b"x" * 32,
            expected_size=32,
            source="unit.private_identity_symlink",
        )
