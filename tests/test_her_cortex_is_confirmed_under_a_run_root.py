"""A run with its own state root still confirms the cortex it loaded.

The whole report run of 23 September loaded her persona model and never
confirmed it: the key that signs the installation's promotion evidence was
looked up under the run's state root, where there is none, so the registry
called the active pointer invalid, the model had no identity, and her affective
steering never attached. Every turn said the substrate was not modulating
inference. A reports ground read off that run would have tested words cut off
from her feelings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.learning import cortex_migration_authority as authority
from tools import whole_environment

pytestmark = pytest.mark.unit


def test_without_a_pin_the_key_is_under_the_state_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.delenv(authority.AUTHORITY_KEY_FILE_ENV, raising=False)
    assert authority.default_authority_key_path() == tmp_path / "state" / "private/cortex-upgrade/migration-authority.key"


def test_a_pinned_key_is_read_where_it_was_pinned(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / "state"))
    installation = tmp_path / "installation" / "migration-authority.key"
    monkeypatch.setenv(authority.AUTHORITY_KEY_FILE_ENV, str(installation))
    assert authority.default_authority_key_path() == installation


def test_a_pinned_key_passes_the_same_custody_checks(monkeypatch, tmp_path: Path) -> None:
    """The pin moves where the key is read, not what a key has to be."""
    key = tmp_path / "migration-authority.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o644)
    monkeypatch.setenv(authority.AUTHORITY_KEY_FILE_ENV, str(key))
    with pytest.raises(authority.CortexMigrationAuthorityError, match="custody_invalid"):
        authority._authority_key(None)
    key.chmod(0o600)
    assert authority._authority_key(None) == b"k" * 32


def test_a_whole_run_pins_the_key_the_desktop_would_read(monkeypatch, tmp_path: Path) -> None:
    assert whole_environment._CUSTODY == {authority.AUTHORITY_KEY_FILE_ENV: "str(a.default_authority_key_path())"}
    assert authority.AUTHORITY_KEY_FILE_ENV not in whole_environment._ISOLATION
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / "desktop"))
    monkeypatch.delenv(authority.AUTHORITY_KEY_FILE_ENV, raising=False)
    resolved = eval(whole_environment._CUSTODY[authority.AUTHORITY_KEY_FILE_ENV], {"a": authority})  # noqa: S307
    assert resolved == str(tmp_path / "desktop" / "private/cortex-upgrade/migration-authority.key")


def test_the_pin_reaches_a_model_worker(monkeypatch, tmp_path: Path) -> None:
    """A worker's environment is scrubbed of anything that looks like a secret.

    The variable names a path, and "AUTHORITY" contains the marker "auth", so it
    was scrubbed: the worker confirmed nothing, and steering stayed detached.
    """
    import os

    from core.runtime import subprocess_gateway

    seen: dict[str, str] = {}
    pinned = str(tmp_path / "migration-authority.key")
    monkeypatch.setattr(
        os, "environ", {authority.AUTHORITY_KEY_FILE_ENV: pinned, "AURA_API_TOKEN": "x"}
    )
    subprocess_gateway._python_process_entrypoint(
        lambda: seen.update(os.environ), (), {}, {}, True
    )
    assert seen.get(authority.AUTHORITY_KEY_FILE_ENV) == pinned
    assert "AURA_API_TOKEN" not in seen
