"""core/capabilities/__init__.py — Capability Layer Boot Sequence
=================================================================
Wires up all Phase 2-6 modules into the ServiceContainer.

Call `await boot_capabilities()` during Aura's startup to bring all
capability providers online in dependency order.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import importlib
from dataclasses import dataclass

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Capabilities")


@dataclass(frozen=True)
class _Provider:
    """One capability provider and what must be online before it may be.

    ``requires`` is the part that was missing. The boot docstring declared
    PermissionModel "gates all actions" and promised a strict dependency
    order, but every provider was attempted unconditionally in a flat list
    of try/except blocks. If the permission model failed to boot, host
    automation, browser control, the file broker, the clipboard, screen
    perception and the microphone all still came online — a dependency
    fault turned directly into a privileged runtime with no gate in front
    of it, and the status dict reported them as successfully booted.

    A provider whose requirement did not boot is now SKIPPED rather than
    attempted, and reported as blocked with the reason.
    """

    name: str
    module: str
    factory: str
    requires: tuple[str, ...] = ()
    # Reaches outside the process: the host, the filesystem, the network,
    # the screen, the microphone. These are the ones that must never
    # outlive their gate.
    privileged: bool = False


# The gate every privileged provider depends on.
_PERMISSION_GATE = "permission_model"

_PROVIDERS: tuple[_Provider, ...] = (
    # --- Tier 1: Foundation ---
    _Provider("app_registry", "core.capabilities.app_registry", "get_app_registry"),
    _Provider(
        "capability_discovery",
        "core.capabilities.capability_discovery",
        "get_capability_discovery",
    ),
    _Provider(_PERMISSION_GATE, "core.capabilities.permission_model", "get_permission_model"),
    _Provider(
        "host_automation",
        "core.capabilities.host_automation",
        "get_host_automation",
        requires=(_PERMISSION_GATE,),
        privileged=True,
    ),
    _Provider(
        "post_action_verifier",
        "core.capabilities.post_action_verifier",
        "get_post_action_verifier",
    ),
    # --- Tier 2: Adapters ---
    _Provider(
        "browser_controller",
        "core.capabilities.browser_controller",
        "get_browser_controller",
        requires=(_PERMISSION_GATE,),
        privileged=True,
    ),
    _Provider(
        "document_service",
        "core.capabilities.document_service",
        "get_document_service",
        requires=(_PERMISSION_GATE,),
        privileged=True,
    ),
    _Provider(
        "file_broker",
        "core.capabilities.file_broker",
        "get_file_broker",
        requires=(_PERMISSION_GATE,),
        privileged=True,
    ),
    _Provider(
        "web_asset_handler",
        "core.capabilities.web_asset_handler",
        "get_web_asset_handler",
        requires=(_PERMISSION_GATE,),
        privileged=True,
    ),
    _Provider(
        "os_settings",
        "core.capabilities.os_settings",
        "get_os_settings",
        requires=(_PERMISSION_GATE,),
        privileged=True,
    ),
    _Provider(
        "clipboard_manager",
        "core.capabilities.clipboard_manager",
        "get_clipboard_manager",
        requires=(_PERMISSION_GATE,),
        privileged=True,
    ),
    _Provider(
        "source_summarizer",
        "core.capabilities.source_summarizer",
        "get_source_summarizer",
    ),
    # --- Tier 2: Perception ---
    _Provider(
        "screen_perception",
        "core.perception.screen_perception",
        "get_screen_perception",
        requires=(_PERMISSION_GATE,),
        privileged=True,
    ),
    _Provider(
        "perceptual_pump",
        "core.perception.perceptual_pump",
        "get_perceptual_pump",
        # A pump with nothing to pump is not a degraded pump, it is a lie.
        requires=(_PERMISSION_GATE, "screen_perception"),
        privileged=True,
    ),
    _Provider(
        "visual_speech",
        "core.perception.visual_speech",
        "get_visual_speech_engine",
        requires=(_PERMISSION_GATE,),
        privileged=True,
    ),
    # --- Tier 3: Planning ---
    _Provider("task_decomposer", "core.planning.task_decomposer", "get_task_decomposer"),
    _Provider("recovery_engine", "core.planning.recovery_engine", "get_recovery_engine"),
    _Provider("mission_state", "core.planning.mission_state", "get_mission_state"),
    # --- Tier 3: Voice ---
    _Provider(
        "voice_session",
        "core.voice.voice_session",
        "get_voice_session_manager",
        requires=(_PERMISSION_GATE,),
        privileged=True,
    ),
    _Provider(
        "wake_word",
        "core.voice.wake_word",
        "get_wake_word_detector",
        requires=(_PERMISSION_GATE,),
        privileged=True,
    ),
    # --- Tier 3: Philosophical ---
    _Provider(
        "behavioral_proof",
        "core.phenomenal_substrate.philosophical_stance",
        "get_behavioral_proof",
    ),
    _Provider("mind_state_exporter", "core.self.mind_state_export", "get_mind_state_exporter"),
)


async def boot_capabilities() -> dict[str, Any]:
    """Boot all capability providers in dependency order.

    Order matters:
    1. AppRegistry (discovers what's installed)
    2. CapabilityDiscovery (scans machine capabilities)
    3. PermissionModel (gates all actions)
    4. HostAutomation (executes actions)
    5. PostActionVerifier (verifies outcomes)
    6. BrowserController (web automation)
    7. DocumentService (file creation)
    8. FileBroker (sandboxed file ops)
    9. WebAssetHandler (image search/download)
    10. OSSettings (wallpaper, volume, etc.)
    11. ClipboardManager (clipboard ops)
    12. SourceSummarizer (multi-source summarization)
    13. ScreenPerception and PerceptualPump (canonical multimodal perception)
    14. VisualSpeech (consented visual-only speech recognition)

    Planning layer:
    15. TaskDecomposer (NL → TaskGraph)
    16. RecoveryEngine (failure recovery)
    17. MissionState (durable mission progress)

    Voice layer:
    18. VoiceSession (narration + session management)
    19. WakeWord (always-listening detection)

    Philosophical layer:
    20. BehavioralProof (Path A functionalist evidence)
    21. MindStateExporter (mind state export/import)

    Returns a status dict.
    """
    start = time.time()
    booted: list[str] = []
    failed: list[str] = []
    blocked: dict[str, str] = {}

    async def _boot(provider: _Provider) -> None:
        """Boot a single provider, recording success, failure, or refusal."""
        unmet = [name for name in provider.requires if name not in booted]
        if unmet:
            reason = f"required provider(s) not online: {', '.join(unmet)}"
            blocked[provider.name] = reason
            record_degradation(
                f"boot.{provider.name}",
                RuntimeError(reason),
                action="refused to start; a privileged provider may not outlive its gate"
                if provider.privileged
                else "refused to start without its declared dependency",
            )
            logger.warning(
                "%s NOT booted — %s%s",
                provider.name,
                reason,
                " (privileged)" if provider.privileged else "",
            )
            return
        try:
            module = importlib.import_module(provider.module)
            instance = getattr(module, provider.factory)()
            if hasattr(instance, "start"):
                await instance.start()
            booted.append(provider.name)
        except (ImportError, AttributeError, RuntimeError, TypeError, OSError) as e:
            failed.append(provider.name)
            record_degradation(f"boot.{provider.name}", e)
            logger.warning("Failed to boot %s: %s", provider.name, e)

    for provider in _PROVIDERS:
        await _boot(provider)

    duration = time.time() - start
    blocked_privileged = sorted(
        name
        for name in blocked
        if _PROVIDER_BY_NAME[name].privileged
    )
    status = {
        "booted": booted,
        "failed": failed,
        # Blocked is neither booted nor failed: nothing was wrong with the
        # provider, it was refused because its gate was not there. Reporting
        # it as a failure would hide the cause; omitting it would hide the
        # refusal.
        "blocked": dict(blocked),
        "blocked_privileged": blocked_privileged,
        "total": len(_PROVIDERS),
        "success_count": len(booted),
        "failure_count": len(failed),
        "blocked_count": len(blocked),
        "duration_ms": round(duration * 1000, 1),
    }

    logger.info(
        "Capability layer ONLINE: %d/%d providers booted in %.0fms%s%s",
        len(booted), len(_PROVIDERS), duration * 1000,
        f" (FAILED: {', '.join(failed)})" if failed else "",
        f" (BLOCKED: {', '.join(sorted(blocked))})" if blocked else "",
    )
    if blocked_privileged:
        logger.error(
            "Privileged capabilities refused to start without their gate: %s",
            ", ".join(blocked_privileged),
        )
    return status


async def get_capabilities_status() -> dict[str, Any]:
    """Get status of all booted capability providers."""
    from core.container import ServiceContainer

    # Derived from the boot table rather than hand-maintained: a
    # hand-copied roster drifts from the thing it describes, and a provider
    # missing from it is invisible to the status surface.
    providers = [provider.name for provider in _PROVIDERS]

    status = {}
    for name in providers:
        try:
            provider = ServiceContainer.get(name, default=None)
            if provider and hasattr(provider, "get_status"):
                status[name] = provider.get_status()
            elif provider:
                status[name] = {"started": True}
            else:
                status[name] = {"started": False}
        except (ImportError, AttributeError, RuntimeError):
            status[name] = {"started": False, "error": "unavailable"}

    return status


__all__ = ["boot_capabilities", "get_capabilities_status"]


_PROVIDER_BY_NAME: dict[str, _Provider] = {p.name: p for p in _PROVIDERS}
