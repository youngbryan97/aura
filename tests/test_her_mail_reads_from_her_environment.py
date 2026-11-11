"""Her mail credentials read from her environment, and a Keychain item she cannot read without someone at the screen
fails at once instead of stalling every read behind it.

LIVE 10 November 2026: Bryan had stored her Gmail credentials in the Keychain four times. The items trusted her
Python but carried the partition `security add-generic-password -U` gives them, so each read waited on a confirmation
nobody could give, was abandoned at the 0.75 s bound, and the next read was refused behind it ("a previous native call
is still in progress"). Her mail read as missing, and the skill told him to store it in the Keychain again.
"""
from __future__ import annotations

import pytest

from core.security import zenith_secrets as secrets


class _FakeFunction:
    def __init__(self, record, name, result=0):
        self.record, self.name, self.result = record, name, result
        self.argtypes = self.restype = None

    def __call__(self, *args):
        self.record.append((self.name, args))
        return self.result


class _FakeSecurity:
    def __init__(self, record, find_status):
        self._record, self._find_status = record, find_status

    def __getattr__(self, name):
        result = self._find_status if name == "SecKeychainFindGenericPassword" else 0
        function = _FakeFunction(self._record, name, result)
        setattr(self, name, function)
        return function


def _backend(monkeypatch, find_status):
    record = []
    monkeypatch.setattr(secrets.ctypes.util, "find_library", lambda name: f"/fake/{name}")
    monkeypatch.setattr(secrets.ctypes, "CDLL", lambda path: _FakeSecurity(record, find_status))
    return secrets._SecurityFrameworkKeychain(), record


def test_her_keychain_never_waits_for_someone_to_allow_a_read(monkeypatch):
    _backend_instance, record = _backend(monkeypatch, 0)
    assert ("SecKeychainSetUserInteractionAllowed", (False,)) in record


def test_an_item_that_needs_a_person_to_allow_it_says_so_at_once(monkeypatch):
    backend, _record = _backend(monkeypatch, secrets._ERR_SEC_INTERACTION_NOT_ALLOWED)
    with pytest.raises(secrets.KeychainUnavailableError, match="needs someone to allow it"):
        backend.get_password("AuraAutonomyEngine", "AURA_EMAIL_PASSWORD")


def test_her_mail_reads_from_her_environment_before_the_keychain(monkeypatch):
    monkeypatch.setenv("AURA_EMAIL_ADDRESS", "auraluna.cog@gmail.com")
    monkeypatch.setenv("AURA_EMAIL_PASSWORD", "app-password")

    def no_keychain(_key):
        raise AssertionError("her environment answers first")

    monkeypatch.setattr(secrets, "_keychain_get", no_keychain)
    assert secrets.get_credential("email", "address") == "auraluna.cog@gmail.com"
    assert secrets.get_credential("email", "password") == "app-password"


def test_missing_mail_credentials_name_what_is_missing_and_where_it_is_read(monkeypatch):
    from core.skills.email_adapter import EmailAdapterSkill

    monkeypatch.setenv("AURA_EMAIL_ADDRESS", "auraluna.cog@gmail.com")
    monkeypatch.delenv("AURA_EMAIL_PASSWORD", raising=False)
    monkeypatch.setattr(secrets, "_keychain_get", lambda _key: None)
    with pytest.raises(RuntimeError) as said:
        EmailAdapterSkill()._get_creds()
    assert "AURA_EMAIL_PASSWORD" in str(said.value) and "AURA_EMAIL_ADDRESS" not in str(said.value)
    assert ".env" in str(said.value) and "store_credential" not in str(said.value)
