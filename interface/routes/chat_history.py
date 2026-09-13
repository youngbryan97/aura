"""Restore the UI transcript through the same scoped store used by chat."""

from typing import Any

from fastapi import Request

from interface.auth import paired_device_session_id, request_access_profile
from interface.routes import chat_delivery, chat_memory_state
from interface.routes.chat_common import _CHAT_REQUEST_PRINCIPAL, _CHAT_REQUEST_SURFACE

UI_CONVERSATION_EXCHANGES = 100


async def recent_ui_conversation(
    request: Request | None, *, limit: int = UI_CONVERSATION_EXCHANGES,
) -> list[dict[str, Any]]:
    profile = request_access_profile(request)
    surface = str(profile.get("surface") or "internal").strip().casefold()
    paired = bool(profile.get("conversation_only"))
    session = str(paired_device_session_id(request) or "") if paired else ""
    if paired and not session:
        return []
    principal = chat_delivery._chat_delivery_principal(
        request,
        chat_delivery._authenticated_chat_principal(request),
        chat_delivery._chat_turn_session_key(request, None),
    )
    limit = max(1, min(UI_CONVERSATION_EXCHANGES, int(limit)))

    async def live_snapshot() -> list[dict[str, Any]]:
        async with chat_memory_state._get_convo_lock():
            visible = []
            for entry in chat_memory_state._conversation_log:
                if session and str(entry.get("session_id") or "") != session:
                    continue
                # Old paired rows carry the device session but no principal.
                legacy_device_row = bool(
                    session and not entry.get("principal_id") and not entry.get("user_id")
                )
                if legacy_device_row or chat_memory_state._conversation_record_visible_to_principal(
                    entry, principal_id=principal, principal_surface=surface,
                ):
                    visible.append(dict(entry))
            return visible[-limit:]

    live = await live_snapshot()
    if len(live) >= limit:
        return live
    principal_token = _CHAT_REQUEST_PRINCIPAL.set(principal)
    surface_token = _CHAT_REQUEST_SURFACE.set(surface)
    try:
        durable = await chat_memory_state._load_durable_conversation_exchanges(
            limit=limit, session_id=session, allow_cross_session=not paired,
        )
    finally:
        _CHAT_REQUEST_SURFACE.reset(surface_token)
        _CHAT_REQUEST_PRINCIPAL.reset(principal_token)

    # A turn can finish while disk is read. Reconcile against the newer state.
    live = await live_snapshot()
    ids = {str(entry.get("id")) for entry in live if entry.get("id")}
    restored = []
    for entry in durable:
        exchange_id = str(entry.get("exchange_id") or "")
        if exchange_id and exchange_id in ids:
            continue
        if exchange_id:
            ids.add(exchange_id)
        # Runtime attestation and private reasoning metadata are not UI data.
        restored.append({
            "id": exchange_id,
            "user": entry.get("user", ""),
            "aura": entry.get("aura", ""),
            "timestamp": entry.get("timestamp", ""),
            "session_id": entry.get("session_id", ""),
            "status": "complete",
        })
    return (restored + live)[-limit:]
