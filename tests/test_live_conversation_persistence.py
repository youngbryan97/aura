import pytest

from core.utils.injected_blocks import is_stamped_runtime_payload

from core.conversation.persistence import ConversationPersistence
from interface.routes import chat as chat_routes


def _without_stamp(exchange: dict) -> dict:
    """The exchange minus the per-process proof, for value comparison."""
    return {key: value for key, value in exchange.items() if key != "aura_runtime_stamp"}


class _PersistenceFixture:
    def __init__(self):
        self.session_id = "session-live"
        self.rows = []

    def record_turn(
        self,
        role,
        content,
        origin="",
        cid=None,
        session_id=None,
        metadata=None,
        **scope,
    ):
        """The real store's signature, including what the caller actually sends.

        `chat_preflight` passes `metadata=` and the principal-scope kwargs.
        This fixture accepted neither, so every persisted turn raised
        TypeError inside the commit — which the route caught as a degradation
        and logged, leaving the assertion to fail on a missing row rather
        than on the reason. A stand-in narrower than the thing it stands in
        for tests a contract nobody has.
        """
        self.rows.append(
            {
                "role": role,
                "content": content,
                "origin": origin,
                "cid": cid,
                "session_id": session_id or self.session_id,
                "metadata": metadata,
                "scope": dict(scope),
                "created_at": float(len(self.rows) + 1),
            }
        )
        return f"turn-{len(self.rows)}"

    def get_recent_sessions(self, limit=10):
        return [{"id": self.session_id, "last_active": 1.0}][:limit]

    def get_session_history(self, session_id=None, limit=100):
        assert session_id == self.session_id
        return self.rows[-limit:]


@pytest.mark.asyncio
async def test_completed_live_exchange_survives_process_memory_clear(monkeypatch):
    persistence = _PersistenceFixture()
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence if name == "persistence" else default
        ),
    )

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    exchange_id = await chat_routes._begin_logged_exchange(
        "The continuity codeword is restart-echo-742."
    )
    await chat_routes._complete_logged_exchange(
        exchange_id,
        "The continuity codeword is restart-echo-742.",
        "I will retain restart-echo-742 across a process restart.",
        record_experience=False,
    )

    assert [row["role"] for row in persistence.rows] == ["user", "aura"]
    assert all(row["origin"] == "desktop_ui" for row in persistence.rows)

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    exchanges = await chat_routes._recent_completed_conversation_exchanges(
        current_user_message="What was the continuity codeword?",
        limit=6,
    )

    # Every reconstructed exchange carries this process's stamp. The stamp is
    # what separates a row this runtime rebuilt from a dict a caller handed
    # in, so assert it is there rather than assert it away.
    assert len(exchanges) == 1
    assert is_stamped_runtime_payload(exchanges[0])
    assert _without_stamp(exchanges[0]) == {
        "exchange_id": exchange_id,
        "user": "The continuity codeword is restart-echo-742.",
        "aura": "I will retain restart-echo-742 across a process restart.",
        "timestamp": "2.0",
        "session_id": "session-live",
        # Present and None. The reconstruction emits both keys on every
        # exchange whether or not the metadata carried them, so a reader never
        # has to ask whether a missing key means "no episode" or "an older
        # row" — and asserting the whole dict is what makes that a contract.
        "action_episode": None,
        "answer_provenance": None,
    }


@pytest.mark.asyncio
async def test_durable_recall_reply_survives_process_memory_clear(monkeypatch):
    persistence = _PersistenceFixture()
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence if name == "persistence" else default
        ),
    )

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    exchange_id = await chat_routes._begin_logged_exchange(
        "The live desktop failure involved the 32B lane losing CognitiveEngine continuity."
    )
    await chat_routes._complete_logged_exchange(
        exchange_id,
        "The live desktop failure involved the 32B lane losing CognitiveEngine continuity.",
        "I tracked that as a live desktop continuity problem, not a backend-only issue.",
        record_experience=False,
    )

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    user_recall = await chat_routes._build_conversation_recall_reply(
        "Can you remind me what I said earlier?"
    )
    aura_recall = await chat_routes._build_conversation_recall_reply(
        "Can you remind me what you answered?"
    )
    topic_recall = await chat_routes._build_conversation_recall_reply(
        "Can you remind me what we discussed?"
    )

    assert user_recall is not None
    assert "32B lane losing CognitiveEngine continuity" in user_recall
    assert aura_recall is not None
    assert "live desktop continuity problem" in aura_recall
    assert topic_recall is not None
    assert "32B lane" in topic_recall
    assert "live desktop continuity problem" in topic_recall


@pytest.mark.asyncio
async def test_recent_context_deduplicates_durable_and_in_memory_exchange(monkeypatch):
    persistence = _PersistenceFixture()
    persistence.record_turn("user", "Keep this one copy.", origin="desktop_ui")
    persistence.record_turn("aura", "One copy retained.", origin="desktop_ui")
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence if name == "persistence" else default
        ),
    )

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.append(
            {
                "id": "same-exchange",
                "user": "Keep this one copy.",
                "aura": "One copy retained.",
                "status": "complete",
                "completed_at": "now",
            }
        )

    exchanges = await chat_routes._recent_completed_conversation_exchanges(
        current_user_message="Continue.",
        limit=6,
    )

    assert len(exchanges) == 1
    assert exchanges[0]["user"] == "Keep this one copy."


def test_durable_reconstruction_joins_interleaved_rows_by_exchange_identity(monkeypatch):
    persistence = _PersistenceFixture()
    persistence.rows = [
        {
            "role": "user",
            "content": "question A",
            "cid": "exchange-a:user",
            "session_id": persistence.session_id,
            "created_at": 1.0,
        },
        {
            "role": "user",
            "content": "question B",
            "cid": "exchange-b:user",
            "session_id": persistence.session_id,
            "created_at": 2.0,
        },
        {
            "role": "aura",
            "content": "answer A",
            "cid": "exchange-a:aura",
            "session_id": persistence.session_id,
            "created_at": 3.0,
        },
        {
            "role": "aura",
            "content": "answer B",
            "cid": "exchange-b:aura",
            "session_id": persistence.session_id,
            "created_at": 4.0,
        },
    ]
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence
            if name == "persistence"
            else default
        ),
    )

    exchanges = chat_routes._load_durable_conversation_exchanges_sync(
        limit=6,
        session_id=persistence.session_id,
    )

    assert [
        (entry["exchange_id"], entry["user"], entry["aura"])
        for entry in exchanges
    ] == [
        ("exchange-a", "question A", "answer A"),
        ("exchange-b", "question B", "answer B"),
    ]


def test_durable_reconstruction_does_not_pair_orphaned_correlated_rows(monkeypatch):
    persistence = _PersistenceFixture()
    persistence.rows = [
        {
            "role": "user",
            "content": "orphan question",
            "cid": "exchange-a:user",
            "session_id": persistence.session_id,
            "created_at": 1.0,
        },
        {
            "role": "aura",
            "content": "different orphan answer",
            "cid": "exchange-b:aura",
            "session_id": persistence.session_id,
            "created_at": 2.0,
        },
    ]
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence
            if name == "persistence"
            else default
        ),
    )

    assert chat_routes._load_durable_conversation_exchanges_sync(
        limit=6,
        session_id=persistence.session_id,
    ) == []


def test_durable_reconstruction_retains_cidless_legacy_pair(monkeypatch):
    persistence = _PersistenceFixture()
    persistence.record_turn("user", "legacy question", origin="desktop_ui")
    persistence.record_turn("aura", "legacy answer", origin="desktop_ui")
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence
            if name == "persistence"
            else default
        ),
    )

    exchanges = chat_routes._load_durable_conversation_exchanges_sync(
        limit=6,
        session_id=persistence.session_id,
    )

    assert len(exchanges) == 1
    assert is_stamped_runtime_payload(exchanges[0])
    assert _without_stamp(exchanges[0]) == {
        "user": "legacy question",
        "aura": "legacy answer",
        "timestamp": "2.0",
        "session_id": persistence.session_id,
    }


class _MultiSessionPersistence:
    """Two conversations: one from before a restart, one after."""

    def __init__(self):
        self.rows: dict[str, list[dict]] = {"before": [], "after": []}
        self.clock = 0.0
        self.history_calls: list[str] = []

    def add(self, session_id, exchange_id, user_text, aura_text):
        for role, content in (("user", user_text), ("aura", aura_text)):
            self.clock += 1.0
            self.rows[session_id].append(
                {
                    "role": role,
                    "content": content,
                    "cid": f"{exchange_id}:{role}",
                    "session_id": session_id,
                    "created_at": self.clock,
                }
            )

    def get_recent_sessions(self, limit=10, *, with_turns_only=False):
        sessions = [
            {"id": "boot-artifact", "last_active": 99.0, "turn_count": 0},
            {"id": "after", "last_active": 50.0, "turn_count": len(self.rows["after"])},
            {"id": "before", "last_active": 10.0, "turn_count": len(self.rows["before"])},
        ]
        if with_turns_only:
            sessions = [s for s in sessions if s["turn_count"] > 0]
        return sessions[:limit]

    def get_session_history(self, session_id=None, limit=100):
        self.history_calls.append(session_id)
        return list(self.rows.get(session_id, []))[-limit:]


def _install(monkeypatch, persistence):
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence if name == "persistence" else default
        ),
    )


def test_recall_reaches_past_the_restart_that_minted_a_new_session(monkeypatch):
    """LIVE DEFECT 2026-08-10: "I can't reach that conversation."

    A restart mints a new session id, so the session a live turn belongs to is
    empty exactly when continuity matters. The loader read only that session;
    its multi-session scan ran solely when no session id was supplied, which
    never happens on a live turn. 34 turns sat on disk, unreachable.
    """
    persistence = _MultiSessionPersistence()
    persistence.add("before", "e1", "which sense would you give up?", "the screen")
    _install(monkeypatch, persistence)

    exchanges = chat_routes._load_durable_conversation_exchanges_sync(
        limit=4,
        session_id="after",
    )

    assert [e["user"] for e in exchanges] == ["which sense would you give up?"]
    assert [e["aura"] for e in exchanges] == ["the screen"]
    assert "boot-artifact" not in persistence.history_calls


def test_a_full_current_session_does_not_reach_back(monkeypatch):
    """Reaching back is for the shortfall only, not a new default."""
    persistence = _MultiSessionPersistence()
    persistence.add("before", "old", "ancient question", "ancient answer")
    for index in range(4):
        persistence.add("after", f"new{index}", f"question {index}", f"answer {index}")
    _install(monkeypatch, persistence)

    exchanges = chat_routes._load_durable_conversation_exchanges_sync(
        limit=2,
        session_id="after",
    )

    assert persistence.history_calls == ["after"]
    assert all("ancient" not in e["user"] for e in exchanges)


def test_earlier_conversations_stay_earlier(monkeypatch):
    """Cross-session ordering is chronological, not scan order."""
    persistence = _MultiSessionPersistence()
    persistence.add("before", "e1", "first question", "first answer")
    persistence.add("after", "e2", "second question", "second answer")
    _install(monkeypatch, persistence)

    exchanges = chat_routes._load_durable_conversation_exchanges_sync(
        limit=4,
        session_id="after",
    )

    assert [e["user"] for e in exchanges] == ["first question", "second question"]


@pytest.mark.asyncio
async def test_in_memory_recall_never_crosses_paired_principals():
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log[:] = [
            {
                "id": "a",
                "user": "A private question",
                "aura": "A private answer",
                "status": "complete",
                "principal_id": "paired-device:a",
                "principal_surface": "paired_device",
            },
            {
                "id": "b",
                "user": "B private question",
                "aura": "B private answer",
                "status": "complete",
                "principal_id": "paired-device:b",
                "principal_surface": "paired_device",
            },
            {
                "id": "legacy",
                "user": "Legacy owner question",
                "aura": "Legacy owner answer",
                "status": "complete",
            },
        ]

    principal_token = chat_routes._CHAT_REQUEST_PRINCIPAL.set("paired-device:a")
    surface_token = chat_routes._CHAT_REQUEST_SURFACE.set("paired_device")
    try:
        exchanges = await chat_routes._recent_completed_conversation_exchanges(
            current_user_message="recall",
            limit=10,
        )
    finally:
        chat_routes._CHAT_REQUEST_SURFACE.reset(surface_token)
        chat_routes._CHAT_REQUEST_PRINCIPAL.reset(principal_token)
        async with chat_routes._get_convo_lock():
            chat_routes._conversation_log.clear()

    assert [(item["user"], item["aura"]) for item in exchanges] == [
        ("A private question", "A private answer")
    ]


def test_durable_loader_never_crosses_paired_principals(monkeypatch, tmp_path):
    persistence = ConversationPersistence(tmp_path / "scoped-chat.db")
    for suffix in ("a", "b"):
        persistence.record_exchange(
            f"{suffix} private question",
            f"{suffix} private answer",
            session_id=f"session-{suffix}",
            cid=f"exchange-{suffix}",
            principal_id=f"paired-device:{suffix}",
            principal_surface="paired_device",
        )
    _install(monkeypatch, persistence)

    principal_token = chat_routes._CHAT_REQUEST_PRINCIPAL.set("paired-device:a")
    surface_token = chat_routes._CHAT_REQUEST_SURFACE.set("paired_device")
    try:
        exchanges = chat_routes._load_durable_conversation_exchanges_sync(
            limit=10,
            session_id="session-a",
        )
    finally:
        chat_routes._CHAT_REQUEST_SURFACE.reset(surface_token)
        chat_routes._CHAT_REQUEST_PRINCIPAL.reset(principal_token)

    assert [(item["user"], item["aura"]) for item in exchanges] == [
        ("a private question", "a private answer")
    ]


@pytest.mark.asyncio
async def test_live_chat_logger_atomically_stamps_authenticated_principal(
    monkeypatch,
    tmp_path,
):
    persistence = ConversationPersistence(tmp_path / "chat-write-scope.db")
    _install(monkeypatch, persistence)
    principal_token = chat_routes._CHAT_REQUEST_PRINCIPAL.set("paired-device:a")
    surface_token = chat_routes._CHAT_REQUEST_SURFACE.set("paired_device")
    try:
        exchange_id = await chat_routes._begin_logged_exchange(
            "store this for principal A",
            session_id="paired-session-a",
        )
        await chat_routes._complete_logged_exchange(
            exchange_id,
            "store this for principal A",
            "principal A reply",
            record_experience=False,
        )
    finally:
        chat_routes._CHAT_REQUEST_SURFACE.reset(surface_token)
        chat_routes._CHAT_REQUEST_PRINCIPAL.reset(principal_token)
        async with chat_routes._get_convo_lock():
            chat_routes._conversation_log.clear()

    assert len(
        persistence.get_session_history(
            "paired-session-a",
            principal_id="paired-device:a",
            principal_surface="paired_device",
        )
    ) == 2
    assert persistence.get_session_history(
        "paired-session-a",
        principal_id="paired-device:b",
        principal_surface="paired_device",
    ) == []
