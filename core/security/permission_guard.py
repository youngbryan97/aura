import asyncio
import logging
import os
import platform
import sys
import time
from enum import Enum, auto
from typing import Any

from core.runtime.base_module import AuraBaseModule
from core.runtime.errors import record_degradation
from core.runtime.resource_observation import get_resource_observer

_PERMISSION_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _load_scripting_bridge_application() -> Any | None:
    try:
        from ScriptingBridge import SBApplication  # type: ignore
    # not a failure: the module is optional here, and its absence is the answer.
    except ImportError:
        return None

    return SBApplication


class PermissionType(Enum):
    MIC = auto()
    CAMERA = auto()
    SCREEN = auto()
    ACCESSIBILITY = auto()
    AUTOMATION = auto()


class PermissionGuard(AuraBaseModule):
    """Handles macOS TCC permission checks without triggering boot-time prompts."""

    def __init__(self):
        super().__init__("PermissionGuard")
        self._cache: dict[PermissionType, dict[str, Any]] = {}
        self._cache_ts: dict[PermissionType, float] = {}
        self._direct_cache: dict[PermissionType, dict[str, Any]] = {}
        self._direct_cache_ts: dict[PermissionType, float] = {}
        # Granted TCC permissions don't get revoked silently, so we cache for
        # 5 minutes once active. Denied/deferred entries still re-probe in 15s.
        self._cache_ttl_s: float = 15.0
        self._cache_ttl_granted_s: float = 300.0
        self._force_refresh_floor_s: float = 20.0

    async def check_permission(self, ptype: PermissionType, force: bool = False) -> dict[str, Any]:
        """Check if a hardware permission is granted.

        Returns:
            {"granted": bool, "status": str, "guidance": str}
        """
        now = time.monotonic()
        cached = self._cache.get(ptype)
        cached_at = float(self._cache_ts.get(ptype, 0.0) or 0.0)
        if cached is not None:
            cache_age = max(0.0, now - cached_at)
            ttl = (
                self._cache_ttl_granted_s
                if cached.get("granted")
                else self._cache_ttl_s
            )
            if force:
                if cache_age < self._force_refresh_floor_s:
                    return cached
            elif cache_age < ttl:
                return cached

        # Env-override path: when the user has granted permission to a parent
        # process Aura can't see (different terminal host), let them assert it
        # with AURA_ASSUME_*_PERMISSION=1 so vision/automation aren't blocked
        # by an inherited TCC identity.
        env_override = self._env_override(ptype)
        if env_override is not None:
            self._cache[ptype] = env_override
            self._cache_ts[ptype] = now
            return env_override

        self.logger.debug("Checking %s permission...", ptype.name)

        native_result = await self._native_bridge_permission_result(ptype)
        if native_result is not None:
            result = native_result
        elif ptype == PermissionType.SCREEN:
            result = await self._check_screen_permission()
        elif ptype == PermissionType.MIC:
            result = await self._check_mic_permission()
        elif ptype == PermissionType.CAMERA:
            result = await self._check_camera_permission()
        elif ptype == PermissionType.ACCESSIBILITY:
            result = await self._check_accessibility_permission()
        elif ptype == PermissionType.AUTOMATION:
            result = await self._check_automation_permission()
        else:
            result = {
                "granted": True,
                "status": "assumed",
                "guidance": "No check implemented for this type yet.",
            }

        self._cache[ptype] = result
        self._cache_ts[ptype] = now
        if (
            ptype
            in {
                PermissionType.SCREEN,
                PermissionType.ACCESSIBILITY,
                PermissionType.AUTOMATION,
            }
            and str(result.get("status") or "")
            not in {"asserted_env", "assumed", "unverified_assertion"}
        ):
            direct_evidence = dict(result)
            direct_evidence["direct_probe"] = True
            direct_evidence["cache_hit"] = False
            direct_evidence["captured_at_unix"] = time.time()
            self._direct_cache[ptype] = direct_evidence
            self._direct_cache_ts[ptype] = time.monotonic()
        return result

    async def check_permission_direct(
        self,
        ptype: PermissionType,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Return recent direct OS evidence without honoring env assertions.

        ``check_permission`` intentionally honors ``AURA_ASSUME_*`` for local
        launch modes where the controlling parent process owns the TCC grant.
        Health and demo-readiness surfaces also need to know whether macOS
        itself can prove the grant for the current process identity. This method
        provides that stricter evidence in a separate, short-lived evidence
        cache so a polling UI cannot create unbounded OS probe work.
        """
        # The signed Aura.app bridge is the authority for desktop control.  The
        # cognitive runtime is usually a Python child; Python's own TCC state
        # can be granted or denied independently and must not override the
        # visible app's grant.  The desktop-access route primes the native
        # bridge cache before calling this method, so this is normally a cheap
        # cache read; local Python probes are fallback evidence only.
        native_result = await self._native_bridge_permission_result(
            ptype, force_one_shot=False
        )
        if native_result is not None:
            direct = dict(native_result)
            direct["direct_probe"] = True
            self._direct_cache[ptype] = direct
            self._direct_cache_ts[ptype] = time.monotonic()
            return direct

        return await self.check_permission_direct_local(ptype, force=force)

    async def check_permission_direct_local(
        self,
        ptype: PermissionType,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Return direct evidence for this process without re-entering Aura.app."""
        now = time.monotonic()
        cached = self._direct_cache.get(ptype)
        cached_at = float(self._direct_cache_ts.get(ptype, 0.0) or 0.0)
        if cached is not None and not force:
            ttl = (
                self._cache_ttl_granted_s
                if cached.get("granted")
                else self._cache_ttl_s
            )
            if now - cached_at < ttl:
                result = dict(cached)
                result["cache_hit"] = True
                result["cache_age_s"] = round(max(0.0, now - cached_at), 3)
                return result

        local_result: dict[str, Any] | None = None
        if ptype == PermissionType.SCREEN:
            loop = asyncio.get_running_loop()
            local_result = await loop.run_in_executor(None, self._screen_preflight_probe)
        elif ptype == PermissionType.ACCESSIBILITY:
            loop = asyncio.get_running_loop()
            local_result = await loop.run_in_executor(
                None, self._accessibility_preflight_probe
            )
        if local_result is not None and local_result.get("granted"):
            direct = dict(local_result)
            direct["direct_probe"] = True
            direct["cache_hit"] = False
            direct["captured_at_unix"] = time.time()
            self._direct_cache[ptype] = dict(direct)
            self._direct_cache_ts[ptype] = time.monotonic()
            return direct
        if local_result is not None:
            result = local_result
        elif ptype == PermissionType.SCREEN:
            result = {
                "granted": False,
                "status": "deferred",
                "guidance": (
                    "Direct Screen Recording preflight is unavailable. "
                    + self.get_guidance(PermissionType.SCREEN)
                ),
            }
        elif ptype == PermissionType.MIC:
            result = await self._check_mic_permission()
        elif ptype == PermissionType.CAMERA:
            result = await self._check_camera_permission()
        elif ptype == PermissionType.ACCESSIBILITY:
            result = {
                "granted": False,
                "status": "deferred",
                "guidance": self.get_guidance(PermissionType.ACCESSIBILITY),
            }
        elif ptype == PermissionType.AUTOMATION:
            result = await self._check_automation_permission()
        else:
            result = {
                "granted": False,
                "status": "unknown",
                "guidance": "No direct probe implemented for this permission type.",
            }
        direct = dict(result or {})
        direct["direct_probe"] = True
        if direct.get("status") == "asserted_env":
            direct["granted"] = False
            direct["status"] = "unverified_assertion"
            direct["guidance"] = self.get_guidance(ptype)
        direct["cache_hit"] = False
        direct["captured_at_unix"] = time.time()
        self._direct_cache[ptype] = dict(direct)
        self._direct_cache_ts[ptype] = time.monotonic()
        return direct

    async def _native_bridge_permission_result(
        self, ptype: PermissionType, *, force_one_shot: bool = False
    ) -> dict[str, Any] | None:
        """Return effective Aura.app permission when the native bridge is trusted.

        Aura's visible desktop app owns the durable macOS TCC grant, while the
        cognitive runtime commonly runs as a Python child.  For desktop control
        health, the relevant question is whether Aura can route the action
        through the trusted app bridge.  This helper keeps that effective grant
        explicit so the UI no longer reports Python's TCC denial as total
        desktop failure when the app bridge can act.
        """
        native_key = {
            PermissionType.SCREEN: "screen_recording",
            PermissionType.ACCESSIBILITY: "accessibility",
            PermissionType.AUTOMATION: "automation",
        }.get(ptype, "")
        if sys.platform != "darwin" or not native_key:
            return None
        try:
            from core.security.native_desktop_bridge import probe_native_desktop_bridge

            # The resident Aura.app bridge is the only production readiness
            # authority. One-shot subprocess probes are diagnostic-only because
            # they create transient launcher processes and can inherit the wrong
            # TCC identity.
            native_probe = await asyncio.to_thread(
                probe_native_desktop_bridge,
                force=force_one_shot,
                prefer_one_shot=force_one_shot,
            )
        except _PERMISSION_RECOVERABLE_ERRORS as exc:
            self.logger.debug("Native Aura.app permission probe unavailable: %s", exc)
            return None
        if not native_probe.get("ok"):
            return None
        if (
            native_probe.get("bridge_transport") == "one_shot_subprocess"
            and not force_one_shot
        ):
            return None
        if not native_probe.get(native_key):
            return None
        return {
            "granted": True,
            "status": "active_native_bridge",
            "guidance": "",
            "native_bridge": True,
            "bridge_executable": str(native_probe.get("bridge_executable", "") or ""),
            "bundle_identifier": str(native_probe.get("bundle_identifier", "") or ""),
        }

    _ENV_OVERRIDE_KEYS: dict[PermissionType, str] = {
        PermissionType.SCREEN: "AURA_ASSUME_SCREEN_PERMISSION",
        PermissionType.ACCESSIBILITY: "AURA_ASSUME_ACCESSIBILITY_PERMISSION",
        PermissionType.AUTOMATION: "AURA_ASSUME_AUTOMATION_PERMISSION",
        PermissionType.MIC: "AURA_ASSUME_MIC_PERMISSION",
        PermissionType.CAMERA: "AURA_ASSUME_CAMERA_PERMISSION",
    }

    def _env_override(self, ptype: PermissionType) -> dict[str, Any] | None:
        """Return a granted result if the user has explicitly asserted the
        permission via env var.

        Assertions are deliberately disabled unless the operator also enables
        ``AURA_PERMISSION_ASSERTIONS_ALLOWED``.  A stale ``AURA_ASSUME_*`` in a
        dotenv file must never make the desktop claim that macOS granted TCC
        access.  Assertions remain available for isolated tests and unusual
        parent-process launchers, but production readiness uses direct probes.
        """
        key = self._ENV_OVERRIDE_KEYS.get(ptype)
        if not key:
            return None
        assertions_allowed = os.getenv(
            "AURA_PERMISSION_ASSERTIONS_ALLOWED", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not assertions_allowed:
            return None
        if os.getenv(key, "0") != "1":
            return None
        return {
            "granted": True,
            "status": "asserted_env",
            "guidance": "",
            "assertion_only": True,
        }

    def _screen_preflight_probe(self) -> dict[str, Any] | None:
        """Use Quartz preflight when available so we don't trigger a capture prompt."""
        try:
            import Quartz  # type: ignore

            preflight = getattr(Quartz, "CGPreflightScreenCaptureAccess", None)
            if callable(preflight):
                granted = bool(preflight())
                return {
                    "granted": granted,
                    "status": "active" if granted else "denied",
                    "guidance": "" if granted else self.get_guidance(PermissionType.SCREEN),
                }
        except _PERMISSION_RECOVERABLE_ERRORS as exc:
            self.logger.debug("Quartz screen preflight unavailable: %s", exc)
        return None

    def _accessibility_preflight_probe(self) -> dict[str, Any] | None:
        """Use AXIsProcessTrusted without prompting so desktop-control checks stay passive."""
        try:
            import ctypes

            framework = "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
            app_services = ctypes.CDLL(framework)
            probe = app_services.AXIsProcessTrusted
            probe.restype = ctypes.c_bool
            granted = bool(probe())
            return {
                "granted": granted,
                "status": "active" if granted else "denied",
                "guidance": "" if granted else self.get_guidance(PermissionType.ACCESSIBILITY),
            }
        except _PERMISSION_RECOVERABLE_ERRORS as exc:
            self.logger.debug("Accessibility preflight unavailable: %s", exc)
        return None

    def request_accessibility_trust(self) -> bool:
        """Actively prompt macOS to grant THIS process Accessibility trust.

        Unlike :meth:`_accessibility_preflight_probe` (a passive, no-prompt
        check), this surfaces the system grant dialog so Aura can acquire
        desktop-control trust for *her own process identity* — independently of
        how she was launched. This matters because the desktop launcher detaches
        the kernel from its parent (reparenting to ``launchd``), which strands
        the TCC trust that would otherwise be inherited from a granted parent
        app. The grant persists for this executable once the user approves.

        Returns the resulting trust state. Safe (no-op True) off macOS.
        """
        if sys.platform != "darwin":
            return True
        try:
            from ApplicationServices import (  # type: ignore
                AXIsProcessTrustedWithOptions,
                kAXTrustedCheckOptionPrompt,
            )

            trusted = bool(
                AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
            )
            # Refresh the cache so callers see the new state without waiting for
            # the denied-entry TTL to expire.
            self._cache[PermissionType.ACCESSIBILITY] = {
                "granted": trusted,
                "status": "active" if trusted else "prompted",
                "guidance": "" if trusted else self.get_guidance(PermissionType.ACCESSIBILITY),
            }
            self._cache_ts[PermissionType.ACCESSIBILITY] = time.monotonic()
            return trusted
        except ImportError as exc:
            probe = self._accessibility_preflight_probe()
            trusted = bool(probe and probe.get("granted") is True)
            self._cache[PermissionType.ACCESSIBILITY] = {
                "granted": trusted,
                "status": "active" if trusted else "prompt_unavailable",
                "guidance": (
                    ""
                    if trusted
                    else (
                        "Aura could not import pyobjc-framework-ApplicationServices "
                        "to surface the Accessibility prompt. Install the declared "
                        "macOS desktop requirements, then approve Aura in System "
                        "Settings -> Privacy & Security -> Accessibility."
                    )
                ),
                "detail": f"{type(exc).__name__}: {exc}",
            }
            self._cache_ts[PermissionType.ACCESSIBILITY] = time.monotonic()
            self.logger.warning(
                "Accessibility prompt unavailable because pyobjc-framework-ApplicationServices "
                "is not installed for %s.",
                sys.executable,
            )
            return trusted
        except _PERMISSION_RECOVERABLE_ERRORS as exc:
            record_degradation("permission_guard", exc)
            self.logger.debug("Accessibility trust prompt unavailable: %s", exc)
            return False

    def request_screen_capture_access(self) -> bool:
        """Actively request Screen Recording access for this process identity.

        macOS grants TCC permissions to the app/executable identity that asks.
        Aura cannot safely edit TCC databases; the production path is to invoke
        the system prompt for the exact process that will read the screen.
        """
        if sys.platform != "darwin":
            return True
        try:
            import Quartz  # type: ignore

            request = getattr(Quartz, "CGRequestScreenCaptureAccess", None)
            if callable(request):
                granted = bool(request())
                self._cache[PermissionType.SCREEN] = {
                    "granted": granted,
                    "status": "active" if granted else "prompted",
                    "guidance": "" if granted else self.get_guidance(PermissionType.SCREEN),
                }
                self._cache_ts[PermissionType.SCREEN] = time.monotonic()
                return granted
        except _PERMISSION_RECOVERABLE_ERRORS as exc:
            record_degradation("permission_guard.screen_capture_prompt", exc)
            self.logger.debug("Screen Recording prompt unavailable: %s", exc)
        return False

    def current_process_identity(self) -> dict[str, Any]:
        """Return the executable/app identity macOS TCC will evaluate.

        This makes permission mismatch diagnosable: if System Settings shows
        Terminal/Codex/Aura enabled but the running detached Python identity is
        different, the UI can tell the user exactly which identity needs trust.
        """
        payload: dict[str, Any] = {
            "platform": platform.platform(),
            "pid": os.getpid(),
            "executable": sys.executable,
            "argv0": sys.argv[0] if sys.argv else "",
            "bundle_identifier": "",
            "bundle_path": "",
            "parent_pid": os.getppid(),
            "parent_name": "",
        }
        if sys.platform == "darwin":
            try:
                from Foundation import NSBundle  # type: ignore

                bundle = NSBundle.mainBundle()
                if bundle is not None:
                    payload["bundle_identifier"] = str(bundle.bundleIdentifier() or "")
                    payload["bundle_path"] = str(bundle.bundlePath() or "")
            except _PERMISSION_RECOVERABLE_ERRORS as exc:
                self.logger.debug("Bundle identity lookup unavailable: %s", exc)
        try:
            parent = get_resource_observer().process(os.getppid())
            if parent is None:
                raise RuntimeError("parent_process_identity_unavailable")
            payload["parent_name"] = parent.name
            payload["parent_executable"] = parent.exe
            payload["observation_source"] = parent.provenance.source.value
            payload["observation_scenario_id"] = parent.provenance.scenario_id
        except _PERMISSION_RECOVERABLE_ERRORS as exc:
            self.logger.debug("Parent process identity lookup unavailable: %s", exc)
        return payload

    def _automation_preflight_probe(self) -> dict[str, Any]:
        """Probe Apple Events access to System Events without shelling out."""
        if sys.platform != "darwin":
            return {
                "granted": True,
                "status": "not_applicable",
                "guidance": "",
            }
        try:
            application_bridge = _load_scripting_bridge_application()
            if application_bridge is None:
                return {
                    "granted": False,
                    "status": "dependency_missing",
                    "guidance": (
                        "Install pyobjc-framework-ScriptingBridge for native "
                        "Automation preflight, then allow Aura to control System Events. "
                        + self.get_guidance(PermissionType.AUTOMATION)
                    ),
                    "detail": "ScriptingBridge is not installed for this Python interpreter.",
                }
            system_events = application_bridge.applicationWithBundleIdentifier_("com.apple.systemevents")
            # Verify permission by querying the process count (safe, single-call)
            _ = len(system_events.processes())
            
            # Retrieve the frontmost application name safely via AppKit
            frontmost_name = ""
            try:
                from AppKit import NSWorkspace
                frontmost = NSWorkspace.sharedWorkspace().frontmostApplication()
                if frontmost:
                    frontmost_name = str(frontmost.localizedName() or "")
            except _PERMISSION_RECOVERABLE_ERRORS as exc:
                self.logger.debug("Unable to read frontmost application during automation probe: %s", exc)

            payload: dict[str, Any] = {
                "granted": True,
                "status": "active",
                "guidance": "",
            }
            if frontmost_name:
                payload["detail"] = frontmost_name[:160]
            return payload
        except _PERMISSION_RECOVERABLE_ERRORS as exc:
            detail = str(exc)
            normalized = detail.lower()
            if "not authorized" in normalized or "-1743" in normalized:
                return {
                    "granted": False,
                    "status": "denied",
                    "guidance": self.get_guidance(PermissionType.AUTOMATION),
                    "detail": detail[:240],
                }
            return {
                "granted": False,
                "status": "deferred",
                "guidance": self.get_guidance(PermissionType.AUTOMATION),
                "detail": detail[:240] or "Native Automation probe unavailable.",
            }

    async def _check_screen_permission(self) -> dict[str, Any]:
        """Probe screen-recording status without forcing a screenshot during boot."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._screen_preflight_probe)
        if result is not None:
            return result

        if os.getenv("AURA_ASSUME_SCREEN_PERMISSION", "0") == "1":
            return {"granted": True, "status": "assumed", "guidance": ""}

        cached = self._cache.get(PermissionType.SCREEN)
        if cached:
            return cached

        return {
            "granted": False,
            "status": "deferred",
            "guidance": (
                "Aura will only request Screen Recording when a screen-aware feature is explicitly used. "
                + self.get_guidance(PermissionType.SCREEN)
            ),
        }

    def _av_media_authorization_probe(
        self, media_attr: str, ptype: PermissionType
    ) -> dict[str, Any]:
        """Passively read the AVFoundation TCC status for audio/video.

        ``authorizationStatusForMediaType_`` reports the current grant WITHOUT
        prompting (only ``requestAccessForMediaType_`` prompts). Status codes:
        0 notDetermined, 1 restricted, 2 denied, 3 authorized. When AVFoundation
        isn't importable we assume granted so a missing pyobjc framework never
        hard-blocks a feature — the feature path surfaces its own real error.
        """
        if sys.platform != "darwin":
            return {"granted": True, "status": "not_applicable", "guidance": ""}
        try:
            import AVFoundation  # type: ignore

            media_type = getattr(AVFoundation, media_attr)
            status = int(
                AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(media_type)
            )
        except _PERMISSION_RECOVERABLE_ERRORS as exc:
            record_degradation("permission_guard", exc)
            self.logger.debug("AVFoundation %s probe unavailable: %s", ptype.name, exc)
            return {"granted": True, "status": "assumed", "guidance": ""}

        if status == 3:  # authorized
            return {"granted": True, "status": "active", "guidance": ""}
        if status == 0:  # notDetermined — macOS prompts on first use
            return {
                "granted": False,
                "status": "undetermined",
                "guidance": (
                    f"macOS will ask for {ptype.name.title()} access the first time "
                    "Aura uses it — click Allow. " + self.get_guidance(ptype)
                ),
            }
        # restricted (1) or denied (2): user must enable it in System Settings.
        return {"granted": False, "status": "denied", "guidance": self.get_guidance(ptype)}

    async def _check_mic_permission(self) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._av_media_authorization_probe, "AVMediaTypeAudio", PermissionType.MIC
        )

    async def _check_camera_permission(self) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._av_media_authorization_probe, "AVMediaTypeVideo", PermissionType.CAMERA
        )

    async def _check_accessibility_permission(self) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._accessibility_preflight_probe)
        if result is not None:
            return result
        return {
            "granted": False,
            "status": "deferred",
            "guidance": self.get_guidance(PermissionType.ACCESSIBILITY),
        }

    async def _check_automation_permission(self) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._automation_preflight_probe)

    def get_guidance(self, ptype: PermissionType) -> str:
        if ptype == PermissionType.SCREEN:
            return (
                "1. Open System Settings\n"
                "2. Go to Privacy & Security\n"
                "3. Select Screen Recording\n"
                "4. Ensure Aura is switched ON. If you use Terminal launch mode, ensure Terminal is switched ON too."
            )
        if ptype == PermissionType.MIC:
            return (
                "1. Open System Settings\n"
                "2. Go to Privacy & Security\n"
                "3. Select Microphone\n"
                "4. Ensure Aura/Terminal is switched ON."
            )
        if ptype == PermissionType.CAMERA:
            return (
                "1. Open System Settings\n"
                "2. Go to Privacy & Security\n"
                "3. Select Camera\n"
                "4. Ensure Aura/Terminal is switched ON for visual processing."
            )
        if ptype == PermissionType.ACCESSIBILITY:
            return (
                "1. Open System Settings\n"
                "2. Go to Privacy & Security\n"
                "3. Select Accessibility\n"
                "4. Ensure Aura is switched ON. If you launched from Terminal or Codex, enable that host app too."
            )
        if ptype == PermissionType.AUTOMATION:
            return (
                "1. Open System Settings\n"
                "2. Go to Privacy & Security\n"
                "3. Select Automation\n"
                "4. Allow Aura/Terminal/Codex to control System Events if you want desktop text and menu-bar access."
            )
        return "Check your macOS Privacy & Security settings."


_SHARED_PERMISSION_GUARD: PermissionGuard | None = None


def reset_permission_guard_for_test() -> None:
    """Drop the shared guard so it is not carried between tests.

    Registered into the container as a side effect of the first call, so the
    test that happens to trigger it is recorded as having changed shared state
    it never asked for.
    """

    global _SHARED_PERMISSION_GUARD
    _SHARED_PERMISSION_GUARD = None


def get_permission_guard() -> PermissionGuard:
    """Return a shared permission guard so passive probes can reuse cache state."""
    global _SHARED_PERMISSION_GUARD

    try:
        from core.container import ServiceContainer
        service_container = ServiceContainer

        existing = service_container.get("permission_guard", default=None)
        if existing is not None:
            return existing
    except _PERMISSION_RECOVERABLE_ERRORS as exc:
        record_degradation("permission_guard", exc)
        logging.getLogger("Aura.PermissionGuard").debug(
            "Shared permission guard lookup failed: %s", exc
        )
        service_container = None

    if _SHARED_PERMISSION_GUARD is None:
        _SHARED_PERMISSION_GUARD = PermissionGuard()

    try:
        if service_container is not None:
            service_container.register_instance("permission_guard", _SHARED_PERMISSION_GUARD, required=False)
    except _PERMISSION_RECOVERABLE_ERRORS as exc:
        record_degradation("permission_guard", exc)
        _SHARED_PERMISSION_GUARD.logger.debug(
            "Shared permission guard registration failed: %s", exc
        )

    return _SHARED_PERMISSION_GUARD
