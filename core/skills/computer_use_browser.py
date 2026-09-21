"""Reading a browser and driving it, lifted out of the one skill.

Seventeen methods about a browser window: where it is pointed, what it is
showing, whether the thing focused is a location bar or an editor canvas.
They are called only by ComputerUseSkill and change when a browser does,
which is not when the rest of computer use changes.


Lifted whole out of `computer_use`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


class _ReadsAndDrivesTheBrowser:
    """Lifted whole out of ComputerUseSkill; see computer_use.py."""

    def _default_browser_name(self) -> str:
        """Which browser this Mac hands http(s) URLs to.

        Read from LaunchServices rather than inferred from the screen: the
        screen answers a different question ("what is in front right now"),
        and immediately after `open` the answer is always still the app that
        called it.
        """
        from .computer_use import (
            _ALLOWED_URL_BROWSERS,
            logger,
        )

        import plistlib

        preference = (
            Path.home()
            / "Library/Preferences/com.apple.LaunchServices"
            / "com.apple.launchservices.secure.plist"
        )
        handlers: list[Any] = []
        try:
            handlers = plistlib.loads(preference.read_bytes()).get("LSHandlers", [])
        except (OSError, ValueError, plistlib.InvalidFileException) as exc:
            # An absent or unreadable preference file is the ordinary state of
            # a Mac whose owner never changed the default — falling through to
            # Safari is the answer, not "" (which reads as "no browser at all"
            # and skips verification entirely, the bug this method exists for).
            logger.debug("Could not read the default browser handler: %s", exc)
        for handler in handlers:
            if not isinstance(handler, dict):
                continue
            if str(handler.get("LSHandlerURLScheme") or "").lower() not in {
                "http",
                "https",
            }:
                continue
            bundle = str(handler.get("LSHandlerRoleAll") or "").lower()
            name = self._BROWSER_BUNDLE_NAMES.get(bundle)
            if name in _ALLOWED_URL_BROWSERS:
                return name
        # No explicit handler registered means Safari, which is the macOS
        # default and not a guess.
        return "Safari" if "Safari" in _ALLOWED_URL_BROWSERS else ""

    def _force_browser_tab_url(self, browser: str, url: str) -> str:
        """Create a foreground tab for a URL in a named browser.

        `open -a Chrome <url>` can be intercepted by restored sessions,
        extension start pages, and auth/login tabs. For live desktop tasks the
        action is not complete until the named browser exposes the requested
        URL as its active tab, so this helper uses the browser's AppleScript
        surface directly as a bounded repair step.
        """
        browser = str(browser or "").strip()
        target = str(url or "").strip()
        if not browser or not target:
            return ""
        quoted_url = self._applescript_string(target)
        if browser in {"Google Chrome", "Arc", "Microsoft Edge"}:
            script = f'''
tell application "{browser}"
    activate
    if (count of windows) = 0 then make new window
    tell front window
        set newTab to make new tab at end of tabs with properties {{URL:{quoted_url}}}
        set active tab index to (count of tabs)
    end tell
end tell
'''
        elif browser == "Safari":
            script = f'''
tell application "Safari"
    activate
    if (count of windows) = 0 then
        make new document with properties {{URL:{quoted_url}}}
    else
        tell front window
            set current tab to (make new tab with properties {{URL:{quoted_url}}})
        end tell
    end if
end tell
'''
        else:
            return ""
        return self._run_applescript(script, timeout=8)

    async def _focus_web_editor_surface(
        self,
        pyautogui: Any,
        *,
        browser: str = "",
        target_url: str = "",
    ) -> tuple[bool, str, str]:
        """Move browser focus from the omnibox into a web editor body.

        Google Docs/Sheets/Slides expose editor bodies through canvas-heavy UI,
        so the invariant must be stronger than "not the URL bar": focus must look
        like an editor surface before a paste/type is allowed.
        """
        from .computer_use import (
            logger,
        )

        if pyautogui is None:
            return False, "", "pyautogui_unavailable_for_web_editor_focus"

        last_snapshot = ""
        try:
            # If navigation left the omnibox active, Escape returns focus to the
            # page without changing the URL or the document contents.
            await asyncio.to_thread(self._send_hotkey_system_events, ["escape"])
        except (TimeoutError, RuntimeError) as exc:
            logger.debug("Web editor focus escape preflight skipped: %s", exc)

        click_points = ((0.50, 0.45), (0.50, 0.55), (0.58, 0.50))
        try:
            screen_w, screen_h = await asyncio.to_thread(pyautogui.size)
        except (RuntimeError, OSError, ValueError, AttributeError) as exc:
            return False, "", f"screen_size_unavailable_for_web_editor_focus: {exc}"

        for x_ratio, y_ratio in click_points:
            try:
                await asyncio.to_thread(
                    pyautogui.click,
                    int(screen_w * x_ratio),
                    int(screen_h * y_ratio),
                )
                await asyncio.sleep(0.35)
                last_snapshot = await asyncio.to_thread(self._focused_element_snapshot)
            except (RuntimeError, OSError, ValueError, AttributeError) as exc:
                return False, last_snapshot, f"web_editor_focus_click_failed: {exc}"
            if self._focused_snapshot_looks_web_editor_surface(last_snapshot):
                return True, last_snapshot, "editable_focus_verified"
            if (
                browser
                and target_url
                and (
                    self._focused_snapshot_looks_web_editor_canvas_candidate(last_snapshot)
                    or not str(last_snapshot or "").strip()
                )
            ):
                active_url, _active_title = await asyncio.to_thread(
                    self._active_browser_location,
                    browser,
                )
                if self._is_resolved_web_editor_url(
                    active_url
                ) and self._url_semantically_matches(target_url, active_url):
                    reason = (
                        "editable_focus_verified_canvas_no_ax_focus"
                        if not str(last_snapshot or "").strip()
                        else "editable_focus_verified_canvas_url"
                    )
                    return True, last_snapshot, reason
            if self._focused_snapshot_looks_browser_location_bar(last_snapshot):
                continue

        if self._focused_snapshot_looks_browser_location_bar(last_snapshot):
            return False, last_snapshot, "browser_location_bar_still_focused"
        if self._focused_snapshot_is_browser_text_entry(last_snapshot):
            return False, last_snapshot, "generic_browser_text_field_focused"
        return False, last_snapshot, "editable_focus_unverified"

    @staticmethod
    def _focused_snapshot_looks_browser_location_bar(snapshot: str) -> bool:
        """Best-effort guard for URL-bar pastes on browser surfaces."""
        raw = str(snapshot or "").strip()
        if not raw:
            return False
        parts = raw.split("\t")
        role = parts[0].strip().lower() if parts else ""
        value = parts[1].strip().lower() if len(parts) > 1 else raw.lower()
        metadata = " ".join(part.strip().lower() for part in parts[1:])
        if role not in {"axtextfield", "axcombobox"}:
            return False
        return bool(
            value.startswith(("http://", "https://"))
            or "search or enter" in metadata
            or "address" in metadata
            or "omnibox" in metadata
            or "url" in metadata
            or "location" in metadata
            or "duckduckgo.com/" in metadata
            or "google.com/search" in metadata
            or "docs.google.com/" in metadata
        )

    @staticmethod
    def _focused_snapshot_is_browser_text_entry(snapshot: str) -> bool:
        raw = str(snapshot or "").strip()
        if not raw:
            return False
        role = raw.split("\t", 1)[0].strip().lower()
        return role in {"axtextfield", "axcombobox"}

    @staticmethod
    def _focused_snapshot_looks_web_editor_canvas_candidate(snapshot: str) -> bool:
        """Positive-but-conservative proof for canvas-backed editors.

        Google Docs can expose the editable canvas as a plain AXWebArea/AXGroup
        without "document" metadata. That shape is only acceptable when the
        caller also proves the active browser URL is still a known editor URL.
        """
        from .computer_use import (
            ComputerUseSkill,
        )

        raw = str(snapshot or "").strip()
        if not raw:
            return False
        if ComputerUseSkill._focused_snapshot_looks_browser_location_bar(raw):
            return False
        if ComputerUseSkill._focused_snapshot_is_browser_text_entry(raw):
            return False
        parts = [part.strip().lower() for part in raw.split("\t")]
        role = parts[0] if parts else ""
        metadata = " ".join(part for part in parts[1:] if part)
        disallowed_hints = (
            "address",
            "email",
            "login",
            "omnibox",
            "password",
            "search or enter",
            "sign in",
            "url",
            "username",
        )
        # Substring, deliberately: `metadata` is the tab-joined accessibility
        # attributes of the focused element, not a sentence. This is the
        # refusal to type into a login, password or address field, and a
        # label whose words run together must still be caught.
        if any(hint in metadata for hint in disallowed_hints):
            return False
        return role in {
            "axgroup",
            "axlayoutarea",
            "axscrollarea",
            "axtextarea",
            "axunknown",
            "axwebarea",
        }

    @staticmethod
    def _focused_snapshot_looks_web_editor_surface(snapshot: str) -> bool:
        """Best-effort positive proof for browser editor focus.

        Generic AXTextField/AXComboBox controls are explicitly rejected because
        they include browser omniboxes, search fields, and sign-in inputs. Google
        Docs often exposes the editing target as a text area, web area, scroll
        area, or group with editor/document descriptors; those are acceptable
        only when the metadata also points at a document/editor body.
        """
        from .computer_use import (
            ComputerUseSkill,
            names_any,
        )

        raw = str(snapshot or "").strip()
        if not raw:
            return False
        parts = [part.strip().lower() for part in raw.split("\t")]
        role = parts[0] if parts else ""
        metadata = " ".join(part for part in parts[1:] if part)
        if ComputerUseSkill._focused_snapshot_looks_browser_location_bar(raw):
            return False
        if role in {"axtextfield", "axcombobox"}:
            return False
        editor_roles = {
            "axtextarea",
            "axwebarea",
            "axgroup",
            "axscrollarea",
            "axlayoutarea",
            "axunknown",
        }
        editor_hints = (
            "document",
            "editor",
            "editing",
            "canvas",
            "body",
            "page",
            "google docs",
            "google sheets",
            "google slides",
        )
        return role in editor_roles and names_any(metadata, editor_hints)

    @staticmethod
    def _normalize_open_url_target(target: str) -> str:
        from .computer_use import (
            urllib,
        )

        text = str(target or "").strip()
        if not text:
            return ""
        if text.startswith(("http://", "https://")):
            return text
        return f"https://duckduckgo.com/?q={urllib.parse.quote_plus(text)}"

    @staticmethod
    def _is_web_editor_url(url: str) -> bool:
        """True for editable Google web-doc surfaces where keystrokes must land in
        the document canvas, not the browser address bar."""
        u = str(url or "").lower()
        return "docs.google.com/" in u and any(
            seg in u for seg in ("/document/", "/spreadsheets/", "/presentation/")
        )

    @staticmethod
    def _is_web_editor_create_url(url: str) -> bool:
        from .computer_use import (
            urllib,
        )

        try:
            parsed = urllib.parse.urlparse(str(url or "").lower())
        except ValueError:
            return False  # not a failure: unparseable is not the create URL
        if parsed.netloc != "docs.google.com":
            return False
        return parsed.path.rstrip("/").endswith("/create")

    @classmethod
    def _is_resolved_web_editor_url(cls, url: str) -> bool:
        return cls._is_web_editor_url(url) and not cls._is_web_editor_create_url(url)

    @staticmethod
    def _url_semantically_matches(expected: str, observed: str) -> bool:
        from .computer_use import (
            urllib,
        )

        expected = str(expected or "").strip()
        observed = str(observed or "").strip()
        if not expected or not observed:
            return False
        if observed.rstrip("/") == expected.rstrip("/") or observed.startswith(expected):
            return True
        try:
            expected_parts = urllib.parse.urlparse(expected)
            observed_parts = urllib.parse.urlparse(observed)
        except ValueError:
            return False  # not a failure: unparseable URLs cannot be shown to match
        if expected_parts.netloc.lower() != observed_parts.netloc.lower():
            return False
        expected_path = expected_parts.path.rstrip("/")
        observed_path = observed_parts.path.rstrip("/")
        if expected_path and observed_path.startswith(expected_path):
            return True
        # Google document creation URLs redirect to a concrete document URL.
        if expected_parts.netloc.lower() == "docs.google.com":
            expected_tokens = {token for token in expected_path.split("/") if token}
            observed_tokens = {token for token in observed_path.split("/") if token}
            return bool(expected_tokens & observed_tokens & {"document", "spreadsheets", "presentation"})
        return False

    @staticmethod
    def _browser_location_script(browser: str) -> str:
        if browser in {"Google Chrome", "Arc", "Microsoft Edge"}:
            return f'''
tell application "{browser}"
    if (count of windows) is 0 then return ""
    set activeUrl to URL of active tab of front window
    set activeTitle to title of active tab of front window
    return activeUrl & linefeed & activeTitle
end tell
'''
        if browser == "Safari":
            return '''
tell application "Safari"
    if (count of windows) is 0 then return ""
    set activeUrl to URL of current tab of front window
    set activeTitle to name of current tab of front window
    return activeUrl & linefeed & activeTitle
end tell
'''
        return ""

    def _active_browser_location(self, browser: str) -> tuple[str, str]:
        from .computer_use import (
            _COMPUTER_USE_RECOVERABLE_ERRORS,
            get_subprocess_gateway,
            logger,
        )

        script = self._browser_location_script(str(browser or "").strip())
        if not script:
            return "", ""
        try:
            result = get_subprocess_gateway().run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5,
                source="computer_use",
                accelerator_capability="none",
            )
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            logger.debug("Active browser URL readback failed: %s", exc)
            return "", ""
        if result.returncode != 0:
            return "", ""
        lines = [line.strip() for line in str(result.stdout or "").splitlines()]
        active_url = lines[0] if lines else ""
        active_title = lines[1] if len(lines) > 1 else ""
        return active_url, active_title

    @staticmethod
    def _browser_source_is_private(url: str) -> bool:
        lowered = str(url or "").lower()
        private_markers = (
            "accounts.google.com",
            "chatgpt.com",
            "gemini.google.com",
            "claude.ai",
            "mail.google.com",
            "bank",
            "password",
            "login",
            "signin",
        )
        # Substring, deliberately: `lowered` is a URL. Its words run into
        # hosts and paths — mybank.example.com, /accounts/login — and this
        # decides whether a page is private enough not to read.
        return any(marker in lowered for marker in private_markers)

    def _browser_execute_javascript(self, browser: str, js: str, *, timeout: int = 8) -> str:
        browser = str(browser or "").strip()
        if browser in {"Google Chrome", "Arc", "Microsoft Edge"}:
            script = f'''
tell application "{browser}"
    activate
    if (count of windows) is 0 then return "{{\\"ok\\":false,\\"error\\":\\"no_browser_window\\"}}"
    tell active tab of front window to execute javascript {self._applescript_string(js)}
end tell
'''
        elif browser == "Safari":
            script = f'''
tell application "Safari"
    activate
    if (count of windows) is 0 then return "{{\\"ok\\":false,\\"error\\":\\"no_browser_window\\"}}"
    do JavaScript {self._applescript_string(js)} in current tab of front window
end tell
'''
        else:
            raise ValueError(f"Browser '{browser}' is not supported for page inspection.")
        return self._run_applescript(script, timeout=timeout)

    def _inspect_browser_page(self, target: str) -> dict[str, Any]:
        from .computer_use import (
            _ALLOWED_URL_BROWSERS,
            _COMPUTER_USE_RECOVERABLE_ERRORS,
            _record_computer_use_degradation,
        )

        try:
            spec = self._target_json(target) if str(target or "").strip().startswith("{") else {}
        except ValueError:
            spec = {}
        browser = str(spec.get("browser") or "").strip()
        if not browser:
            browser = self._frontmost_app_name()
        if browser not in _ALLOWED_URL_BROWSERS:
            return {
                "ok": False,
                "status": "not_browser",
                "error": f"Frontmost app is not an inspectable browser: {browser or 'unavailable'}.",
            }
        mode = str(spec.get("mode") or "text").strip().lower()
        max_chars = max(500, min(int(spec.get("max_chars") or 12000), 60000))
        active_url, active_title = self._active_browser_location(browser)
        wants_source = mode in {"html", "source", "dom", "page_source"}
        allow_private_source = bool(spec.get("allow_private_source") or spec.get("allow_private"))
        if wants_source and self._browser_source_is_private(active_url) and not allow_private_source:
            return {
                "ok": False,
                "status": "private_source_blocked",
                "browser": browser,
                "url": active_url,
                "title": active_title,
                "error": "Page-source inspection is blocked on private or account pages; use text mode unless the user explicitly authorizes source inspection.",
            }
        js = f"""
(() => {{
  const text = (document.body && document.body.innerText || '').slice(0, {max_chars});
  const links = Array.from(document.links || []).slice(0, 80).map(link => ({{
    text: (link.innerText || link.getAttribute('aria-label') || '').trim().slice(0, 180),
    href: link.href || ''
  }})).filter(item => item.href || item.text);
  const payload = {{
    ok: true,
    url: location.href,
    title: document.title || '',
    text,
    links,
    editable_count: document.querySelectorAll('textarea,input[type="text"],input:not([type]),[contenteditable="true"],[role="textbox"]').length
  }};
  if ({str(wants_source).lower()}) {{
    payload.html = (document.documentElement && document.documentElement.outerHTML || '').slice(0, {max_chars});
  }}
  return JSON.stringify(payload);
}})()
"""
        try:
            raw = self._browser_execute_javascript(browser, js, timeout=10)
            data = json.loads(raw or "{}")
            if not data.get("ok"):
                raise RuntimeError(str(data.get("error") or "browser inspection failed"))
            return {
                "ok": True,
                "action": "inspect_browser_page",
                "source": "browser_dom",
                "browser": browser,
                "url": str(data.get("url") or active_url),
                "title": str(data.get("title") or active_title),
                "text": str(data.get("text") or "")[:max_chars],
                "html": str(data.get("html") or "")[:max_chars],
                "links": list(data.get("links") or [])[:80],
                "editable_count": int(data.get("editable_count") or 0),
                "effect_verified": True,
            }
        except _COMPUTER_USE_RECOVERABLE_ERRORS as exc:
            _record_computer_use_degradation(
                exc,
                action="used screen perception fallback after browser DOM inspection failed",
                stage="inspect_browser_page.dom",
                severity="warning",
            )
            try:
                from core.perception.screen_perception import get_screen_perception

                snapshot = get_screen_perception().capture_sync(save_screenshot=True)
                payload = self._screen_snapshot_result(snapshot)
                payload.update(
                    {
                        "action": "inspect_browser_page",
                        "source": "screen_perception_browser_fallback",
                        "browser": browser,
                        "url": active_url,
                        "title": active_title,
                        "effect_verified": bool(payload.get("text")),
                    }
                )
                return payload
            except _COMPUTER_USE_RECOVERABLE_ERRORS as fallback_exc:
                return {
                    "ok": False,
                    "status": "unavailable",
                    "action": "inspect_browser_page",
                    "browser": browser,
                    "url": active_url,
                    "title": active_title,
                    "error": f"Browser inspection failed and OCR fallback failed: {fallback_exc}",
                }

