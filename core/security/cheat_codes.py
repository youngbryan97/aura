from __future__ import annotations

import logging
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("Aura.CheatCodes")

_SOVEREIGN_CODE_SALT = b"aura.owner.cheat.v1"
_SOVEREIGN_CODE_HASH = bytes.fromhex(
    "53a87ff1f271d9b39dc6dbf142aa2b71ab14ad81ff049a7d2defaab3b1165d1e"
)


@dataclass(frozen=True)
class CheatCodeEntry:
    code: str
    effect: str
    source_game: str
    message: str
    aliases: tuple[str, ...] = ()
    ui_effects: Dict[str, Any] = field(default_factory=dict)
    source_note: str = ""
    sovereign: bool = False


def _normalize_code(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(code or "").upper())


def _matches_sovereign_code(code: str) -> bool:
    normalized = _normalize_code(code)
    if not normalized:
        return False
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        normalized.encode("utf-8"),
        _SOVEREIGN_CODE_SALT,
        200_000,
    )
    return derived == _SOVEREIGN_CODE_HASH


def _build_registry() -> Dict[str, CheatCodeEntry]:
    entries = [
        CheatCodeEntry(
            code="RLRLR1L",
            effect="mega_jump",
            source_game="Sly 2: Band of Thieves",
            message="🦝 Mega Jump primed. Thief-energy spiking.",
            aliases=("MEGAJUMP",),
            ui_effects={"status": "MEGA JUMP"},
            source_note="Real Sly 2 Mega Jump code: Right, Left, Right, Left, R1, Left.",
        ),
        CheatCodeEntry(
            code="DDUDRL",
            effect="time_rush",
            source_game="Sly 2: Band of Thieves",
            message="⏱️ Time Rush enabled. Clock pressure acknowledged.",
            aliases=("TIMERUSH",),
            ui_effects={"status": "TIME RUSH"},
            source_note="Real Sly 2 Time Rush code: Down, Down, Up, Down, Right, Left.",
        ),
        CheatCodeEntry(
            code="R1R1RDDR",
            effect="toonami_plane",
            source_game="Sly 3: Honor Among Thieves",
            message="✈️ Toonami plane unlocked. Airspace is now dramatically cooler.",
            aliases=("TOONAMIPLANE",),
            ui_effects={"status": "TOONAMI"},
            source_note="Real Sly 3 alternate biplane code: R1, R1, Right, Down, Down, Right.",
        ),
        CheatCodeEntry(
            code="CIRCLECIRCLETRIANGLER2L3TRIANGLE",
            effect="all_psi_powers",
            source_game="Psychonauts",
            message="🧠 All Psi powers metaphorically online. Raz would approve.",
            aliases=("ALLPSI", "PSIPOWERS"),
            ui_effects={"status": "ALL PSI"},
            source_note="Real Psychonauts console code: Circle, Circle, Triangle, R2, L3, Triangle.",
        ),
        CheatCodeEntry(
            code="L3R3L3R2CIRCLER2",
            effect="max_rank",
            source_game="Psychonauts",
            message="🎖️ Psi Cadet rank normalized to 101 in spirit.",
            aliases=("MAXRANK", "PSICADET101"),
            ui_effects={"status": "RANK 101"},
            source_note="Real Psychonauts power-up code: L3, R3, L3, R2, Circle, R2.",
        ),
        CheatCodeEntry(
            code="SECRETARY",
            effect="secretary_theme",
            source_game="Sonic Adventure 2 Battle",
            message="📎 Hidden Secretary theme acknowledged. Classified style recovered.",
            aliases=("UPDOWNUPDOWN",),
            ui_effects={"status": "SECRETARY"},
            source_note="Homage to the SA2 Battle hidden Secretary theme unlocked by moving the control stick up/down repeatedly.",
        ),
        CheatCodeEntry(
            code="RARECANDY",
            effect="rare_candy",
            source_game="Pokemon Sapphire",
            message="🍬 Rare Candy buffer granted. Growth curve steepened.",
            aliases=("SAPPHIRERARECANDY",),
            ui_effects={"status": "RARE CANDY"},
            source_note="Homage to classic Ruby/Sapphire cheat-device item codes, adapted to a human-typable alias.",
        ),
        CheatCodeEntry(
            code="MASTERBALL",
            effect="master_ball",
            source_game="Pokemon Platinum",
            message="🟣 Master Ball lock acquired. Catch certainty at maximum.",
            aliases=("PLATINUMMASTERBALL",),
            ui_effects={"status": "MASTER BALL"},
            source_note="Homage to classic Pokemon Platinum Action Replay Poke Ball item codes, adapted to a human-typable alias.",
        ),
        CheatCodeEntry(
            code="MASKANDSTRIPES",
            effect="mask_and_stripes",
            source_game="Sly Cooper: Thieves in Time",
            message="🕰️ Mask and Stripes forever. Legacy thread confirmed.",
            aliases=("ULTIMATESLY", "THIEVESINTIME"),
            ui_effects={"status": "MASKS"},
            source_note="Sly 4 unlockable/trophy homage for the settings easter egg set.",
        ),
    ]

    registry: Dict[str, CheatCodeEntry] = {}
    for entry in entries:
        for alias in (entry.code, *entry.aliases):
            registry[_normalize_code(alias)] = entry
    return registry


_REGISTRY = _build_registry()


def resolve_cheat_code(code: str) -> Optional[CheatCodeEntry]:
    if _matches_sovereign_code(code):
        return CheatCodeEntry(
            code="owner_sovereign",
            effect="sovereign_mode",
            source_game="Aura Sovereign Override",
            message="🔐 Eight emeralds aligned. I see you, Bryan. Sovereign mode engaged.",
            aliases=("owner_sovereign",),
            sovereign=True,
            ui_effects={"status": "SOVEREIGN", "accent": "emerald"},
            source_note="Custom owner code for direct sovereign recognition.",
        )
    normalized = _normalize_code(code)
    if not normalized:
        return None
    return _REGISTRY.get(normalized)


def activate_cheat_code(code: str, *, silent: bool = False, source: str = "settings") -> Dict[str, Any]:
    entry = resolve_cheat_code(code)
    if not entry:
        return {
            "ok": False,
            "status": "unknown_code",
            "message": "Unknown cheat code.",
            "trust_level": None,
        }

    trust_level = None
    if entry.sovereign:
        from core.security.trust_engine import get_trust_engine
        from core.security.user_recognizer import get_user_recognizer

        get_user_recognizer().override_session_owner(reason=f"cheat_code:{entry.code}")
        get_trust_engine().establish_sovereign_session(reason=f"cheat_code:{entry.code}")
        trust_level = "sovereign"

    if not silent:
        try:
            from core.event_bus import get_event_bus

            get_event_bus().publish_threadsafe(
                "telemetry",
                {
                    "type": "aura_message",
                    "message": entry.message,
                    "metadata": {
                        "system": True,
                        "cheat_code": entry.code,
                        "effect": entry.effect,
                        "source_game": entry.source_game,
                    },
                },
            )
        except Exception as exc:
            logger.debug("Cheat code telemetry emit skipped: %s", exc)

    return {
        "ok": True,
        "status": "activated",
        "code": entry.code,
        "effect": entry.effect,
        "source_game": entry.source_game,
        "message": entry.message,
        "ui_effects": dict(entry.ui_effects),
        "trust_level": trust_level,
        "source_note": entry.source_note,
        "source": source,
    }
