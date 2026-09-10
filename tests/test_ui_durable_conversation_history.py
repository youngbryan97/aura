import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from starlette.requests import Request

from core.container import ServiceContainer
from core.conversation.persistence import ConversationPersistence
from interface.routes import chat_history, chat_memory_state
from interface.routes.chat_common import _CHAT_REQUEST_PRINCIPAL, _CHAT_REQUEST_SURFACE


@pytest.fixture
def history_store(monkeypatch, tmp_path):
    store = ConversationPersistence(tmp_path / "history.db")
    original = ServiceContainer.get
    monkeypatch.setattr(ServiceContainer, "get", staticmethod(
        lambda key, default=None: store if key == "persistence" else original(key, default=default)
    ))
    monkeypatch.setattr(chat_memory_state, "_conversation_log", [])
    monkeypatch.setattr(chat_history.chat_delivery, "_authenticated_chat_principal", lambda request: "owner-a")
    return store


def owner_request():
    return Request({"type": "http", "scheme": "http", "path": "/api/ui/bootstrap",
                    "query_string": b"", "client": ("127.0.0.1", 1234),
                    "server": ("127.0.0.1", 8000), "headers": [(b"host", b"127.0.0.1:8000")]})


def record(store, cid, user="same question", answer="same answer", *, principal="owner-a", surface="owner", session="before-reboot"):
    store.record_exchange(user, answer, session_id=session, cid=cid,
                          principal_id=principal, principal_surface=surface)


@pytest.mark.asyncio
async def test_bootstrap_restores_full_exchange_and_original_identity(history_store):
    answer = "A complete answer.\n\n" + "Still part of the answer. " * 100
    record(history_store, "saved", answer=answer)
    record(history_store, "other-owner", principal="owner-b", session="other-owner-session")
    previous = (_CHAT_REQUEST_PRINCIPAL.get(), _CHAT_REQUEST_SURFACE.get())
    rows = await chat_history.recent_ui_conversation(owner_request())
    assert len(rows) == 1
    assert rows[0]["id"] == "saved"
    assert rows[0]["aura"] == answer.strip()
    assert rows[0]["timestamp"]
    assert rows[0]["status"] == "complete"
    assert "aura_runtime_stamp" not in rows[0]
    assert "action_episode" not in rows[0]
    assert (_CHAT_REQUEST_PRINCIPAL.get(), _CHAT_REQUEST_SURFACE.get()) == previous


@pytest.mark.asyncio
async def test_identical_text_in_distinct_turns_survives_but_live_ids_do_not_duplicate(history_store):
    record(history_store, "first")
    record(history_store, "second")
    chat_memory_state._conversation_log.extend([
        {"id": "second", "user": "same question", "aura": "same answer", "status": "complete"},
        {"id": "third", "user": "pending question", "aura": "", "status": "running"},
    ])
    rows = await chat_history.recent_ui_conversation(owner_request())
    assert [row["id"] for row in rows] == ["first", "second", "third"]
    assert rows[-1]["status"] == "running"
    again = await chat_history.recent_ui_conversation(owner_request())
    assert again == rows


@pytest.mark.asyncio
async def test_paired_restore_never_reads_another_principal_or_session(history_store, monkeypatch):
    monkeypatch.setattr(chat_history, "request_access_profile", lambda request: {
        "surface": "paired_device", "conversation_only": True,
    })
    monkeypatch.setattr(chat_history, "paired_device_session_id", lambda request: "paired-a")
    for cid, principal, surface, session in [
        ("allowed", "owner-a", "paired_device", "paired-a"),
        ("different-principal", "owner-b", "paired_device", "other-principal-session"),
        ("owner-surface", "owner-a", "owner", "owner-session"),
        ("different-session", "owner-a", "paired_device", "paired-b"),
    ]:
        record(history_store, cid, principal=principal, surface=surface, session=session)
    chat_memory_state._conversation_log.extend([
        {"id": "wrong-live-principal", "session_id": "paired-a", "principal_id": "owner-b", "principal_surface": "paired_device", "user": "private"},
        {"id": "wrong-live-surface", "session_id": "paired-a", "principal_id": "owner-a", "principal_surface": "owner", "user": "private"},
    ])
    rows = await chat_history.recent_ui_conversation(owner_request())
    assert [row["id"] for row in rows] == ["allowed"]


@pytest.mark.asyncio
async def test_completion_during_disk_read_uses_newest_live_state(history_store, monkeypatch):
    chat_memory_state._conversation_log.append({"id": "active", "user": "question", "aura": "", "status": "running"})

    async def read(**kwargs):
        await asyncio.sleep(0)
        chat_memory_state._conversation_log[0].update(aura="complete answer", status="complete")
        return [{"exchange_id": "active", "user": "question", "aura": "old answer"}]

    monkeypatch.setattr(chat_memory_state, "_load_durable_conversation_exchanges", read)
    rows = await chat_history.recent_ui_conversation(owner_request())
    assert len(rows) == 1
    assert rows[0]["aura"] == "complete answer"


@pytest.mark.asyncio
async def test_full_live_window_needs_no_disk_read(history_store, monkeypatch):
    chat_memory_state._conversation_log.extend({"id": str(i), "user": "question", "status": "running"} for i in range(45))

    async def forbidden(**kwargs):
        pytest.fail("a full live window should not read older disk history")

    monkeypatch.setattr(chat_memory_state, "_load_durable_conversation_exchanges", forbidden)
    rows = await chat_history.recent_ui_conversation(owner_request())
    assert [row["id"] for row in rows] == [str(i) for i in range(5, 45)]


@pytest.mark.asyncio
async def test_cancellation_restores_request_identity(history_store, monkeypatch):
    previous = (_CHAT_REQUEST_PRINCIPAL.get(), _CHAT_REQUEST_SURFACE.get())

    async def interrupted(**kwargs):
        assert _CHAT_REQUEST_PRINCIPAL.get() == "owner-a"
        raise asyncio.CancelledError

    monkeypatch.setattr(chat_memory_state, "_load_durable_conversation_exchanges", interrupted)
    with pytest.raises(asyncio.CancelledError):
        await chat_history.recent_ui_conversation(owner_request())
    assert (_CHAT_REQUEST_PRINCIPAL.get(), _CHAT_REQUEST_SURFACE.get()) == previous


@pytest.mark.asyncio
async def test_real_bootstrap_route_uses_the_durable_reader(history_store, service_container):
    from interface.routes.system import api_ui_bootstrap

    record(history_store, "route-saved", user="before restart", answer="still here")
    response = await api_ui_bootstrap(request=owner_request())
    rows = json.loads(response.body)["conversation"]["recent"]
    assert [(row["id"], row["aura"]) for row in rows] == [("route-saved", "still here")]


@pytest.mark.asyncio
@pytest.mark.parametrize("paired", [False, True])
async def test_history_restores_scoped_rows_while_default_executor_is_occupied(
    history_store, monkeypatch, paired,
):
    surface = "paired_device" if paired else "owner"
    session = "paired-a" if paired else "before-reboot"
    if paired:
        monkeypatch.setattr(chat_history, "request_access_profile", lambda request: {
            "surface": surface, "conversation_only": True,
        })
        monkeypatch.setattr(chat_history, "paired_device_session_id", lambda request: session)
    record(history_store, "visible", surface=surface, session=session)
    record(history_store, "private", principal="owner-b", session="private-session")
    loop = asyncio.get_running_loop()
    original_pool = loop._default_executor
    pool = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    started = threading.Event()

    def occupy_default_worker():
        started.set()
        release.wait(5)

    loop.set_default_executor(pool)
    pending = loop.run_in_executor(None, occupy_default_worker)
    try:
        while not started.is_set():
            await asyncio.sleep(0)
        rows = await chat_history.recent_ui_conversation(owner_request())
        assert [row["id"] for row in rows] == ["visible"]
        assert not release.is_set()
    finally:
        release.set()
        await pending
        loop._default_executor = original_pool
        pool.shutdown(wait=True, cancel_futures=True)
