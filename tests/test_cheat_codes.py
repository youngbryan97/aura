from types import SimpleNamespace

from core.security.cheat_codes import activate_cheat_code, resolve_cheat_code
from core.security.trust_engine import get_trust_engine
from core.security.user_recognizer import get_user_recognizer
from interface.server import (
    CHEAT_CODE_COOKIE_NAME,
    _decode_owner_session_cookie,
    _encode_owner_session_cookie,
    _restore_owner_session_from_request,
)


def _reset_security_state():
    get_trust_engine().reset_session()
    get_user_recognizer().reset_session()


def test_sovereign_cheat_code_activates_owner_session():
    _reset_security_state()

    result = activate_cheat_code("8emeraldS!", silent=True, source="test")

    assert result["ok"] is True
    assert result["trust_level"] == "sovereign"
    assert get_trust_engine().get_status()["level"] == "sovereign"
    assert get_user_recognizer().recognize("hello from bryan").passphrase_verified is True


def test_cheat_code_alias_resolves_known_easter_egg():
    entry = resolve_cheat_code("mega jump")

    assert entry is not None
    assert entry.effect == "mega_jump"
    assert entry.source_game.startswith("Sly 2")


def test_owner_session_cookie_restores_sovereign_without_raw_code():
    _reset_security_state()

    token = _encode_owner_session_cookie()
    payload = _decode_owner_session_cookie(token)
    restored = _restore_owner_session_from_request(
        SimpleNamespace(cookies={CHEAT_CODE_COOKIE_NAME: token})
    )

    assert payload is not None
    assert payload["scope"] == "sovereign_owner"
    assert restored is True
    assert get_trust_engine().get_status()["level"] == "sovereign"
