"""core/capabilities/app_registry.py — Installed App Discovery & Affordance Map
================================================================================
Aura asks "What app/tool can satisfy this task?" rather than "What script
was written for this demo?"

This module:
1. Discovers installed apps on macOS at boot
2. Maps each app to a set of affordances (what it can do)
3. Provides adapter classes for common app categories
4. Lets the TaskDecomposer choose the best tool for each task step

The registry is NOT hardcoded to specific apps. It discovers what's
installed, maps capabilities, and lets the planner choose.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.AppRegistry")


# ---------------------------------------------------------------------------
# App categories and affordances
# ---------------------------------------------------------------------------

class AppCategory(str, Enum):
    TEXT_EDITOR = "text_editor"
    BROWSER = "browser"
    FILE_MANAGER = "file_manager"
    DOCUMENT_RENDERER = "document_renderer"
    TERMINAL = "terminal"
    SYSTEM_SETTINGS = "system_settings"
    IMAGE_VIEWER = "image_viewer"
    MEDIA_PLAYER = "media_player"
    CODE_EDITOR = "code_editor"
    COMMUNICATION = "communication"
    PRODUCTIVITY = "productivity"
    UNKNOWN = "unknown"


class AppAffordance(str, Enum):
    """What an app can DO — not what it IS."""
    CREATE_TEXT = "create_text"
    EDIT_TEXT = "edit_text"
    EXPORT_PDF = "export_pdf"
    OPEN_URL = "open_url"
    BROWSE_WEB = "browse_web"
    EXTRACT_WEB_CONTENT = "extract_web_content"
    MANAGE_TABS = "manage_tabs"
    CREATE_FOLDER = "create_folder"
    MOVE_FILE = "move_file"
    OPEN_FILE = "open_file"
    VIEW_IMAGE = "view_image"
    RENDER_DOCUMENT = "render_document"
    RUN_COMMAND = "run_command"
    CHANGE_SETTINGS = "change_settings"
    SET_WALLPAPER = "set_wallpaper"
    SEND_EMAIL = "send_email"
    SEND_MESSAGE = "send_message"
    CREATE_SPREADSHEET = "create_spreadsheet"
    PRESENT_SLIDES = "present_slides"
    PLAY_MEDIA = "play_media"
    EDIT_CODE = "edit_code"


@dataclass
class InstalledApp:
    """An application installed on this machine."""
    name: str
    bundle_id: str = ""
    path: str = ""
    category: AppCategory = AppCategory.UNKNOWN
    affordances: Set[AppAffordance] = field(default_factory=set)
    launch_method: str = "activate"  # "activate", "open -a", "open -b"
    reliability: float = 0.8         # 0-1, how reliable is automation with this app
    known_issues: List[str] = field(default_factory=list)
    adapter_class: str = ""          # which adapter handles this app
    last_used: float = 0.0
    discovered_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Built-in category → affordance mappings
# ---------------------------------------------------------------------------

# These map app NAMES (lowercase) to categories and affordances.
# The registry uses these as seed data, then augments from discovery.
KNOWN_APP_PROFILES: Dict[str, Dict[str, Any]] = {
    # Text editors
    "notes": {
        "category": AppCategory.TEXT_EDITOR,
        "affordances": {AppAffordance.CREATE_TEXT, AppAffordance.EDIT_TEXT},
        "bundle_id": "com.apple.Notes",
        "reliability": 0.7,
        "known_issues": ["Export to PDF requires UI automation or Share menu"],
    },
    "textedit": {
        "category": AppCategory.TEXT_EDITOR,
        "affordances": {AppAffordance.CREATE_TEXT, AppAffordance.EDIT_TEXT, AppAffordance.EXPORT_PDF},
        "bundle_id": "com.apple.TextEdit",
        "reliability": 0.9,
    },
    "visual studio code": {
        "category": AppCategory.CODE_EDITOR,
        "affordances": {AppAffordance.EDIT_CODE, AppAffordance.CREATE_TEXT, AppAffordance.EDIT_TEXT, AppAffordance.RUN_COMMAND},
        "bundle_id": "com.microsoft.VSCode",
        "reliability": 0.6,
        "known_issues": ["Complex UI, AppleScript limited"],
    },
    "sublime text": {
        "category": AppCategory.CODE_EDITOR,
        "affordances": {AppAffordance.EDIT_CODE, AppAffordance.CREATE_TEXT, AppAffordance.EDIT_TEXT},
        "bundle_id": "com.sublimetext.4",
        "reliability": 0.7,
    },

    # Browsers
    "google chrome": {
        "category": AppCategory.BROWSER,
        "affordances": {AppAffordance.OPEN_URL, AppAffordance.BROWSE_WEB, AppAffordance.MANAGE_TABS, AppAffordance.EXTRACT_WEB_CONTENT},
        "bundle_id": "com.google.Chrome",
        "reliability": 0.85,
        "adapter_class": "BrowserAdapter",
    },
    "safari": {
        "category": AppCategory.BROWSER,
        "affordances": {AppAffordance.OPEN_URL, AppAffordance.BROWSE_WEB, AppAffordance.MANAGE_TABS, AppAffordance.EXTRACT_WEB_CONTENT},
        "bundle_id": "com.apple.Safari",
        "reliability": 0.8,
        "adapter_class": "BrowserAdapter",
    },
    "firefox": {
        "category": AppCategory.BROWSER,
        "affordances": {AppAffordance.OPEN_URL, AppAffordance.BROWSE_WEB, AppAffordance.MANAGE_TABS},
        "bundle_id": "org.mozilla.firefox",
        "reliability": 0.7,
        "adapter_class": "BrowserAdapter",
    },
    "arc": {
        "category": AppCategory.BROWSER,
        "affordances": {AppAffordance.OPEN_URL, AppAffordance.BROWSE_WEB, AppAffordance.MANAGE_TABS},
        "bundle_id": "company.thebrowser.Browser",
        "reliability": 0.6,
        "adapter_class": "BrowserAdapter",
    },

    # File management
    "finder": {
        "category": AppCategory.FILE_MANAGER,
        "affordances": {AppAffordance.CREATE_FOLDER, AppAffordance.MOVE_FILE, AppAffordance.OPEN_FILE},
        "bundle_id": "com.apple.finder",
        "reliability": 0.95,
    },

    # Document rendering
    "preview": {
        "category": AppCategory.DOCUMENT_RENDERER,
        "affordances": {AppAffordance.VIEW_IMAGE, AppAffordance.RENDER_DOCUMENT, AppAffordance.EXPORT_PDF},
        "bundle_id": "com.apple.Preview",
        "reliability": 0.9,
    },
    "pages": {
        "category": AppCategory.PRODUCTIVITY,
        "affordances": {AppAffordance.CREATE_TEXT, AppAffordance.EDIT_TEXT, AppAffordance.EXPORT_PDF, AppAffordance.RENDER_DOCUMENT},
        "bundle_id": "com.apple.iWork.Pages",
        "reliability": 0.7,
    },

    # Terminals
    "terminal": {
        "category": AppCategory.TERMINAL,
        "affordances": {AppAffordance.RUN_COMMAND},
        "bundle_id": "com.apple.Terminal",
        "reliability": 0.95,
    },
    "iterm2": {
        "category": AppCategory.TERMINAL,
        "affordances": {AppAffordance.RUN_COMMAND},
        "bundle_id": "com.googlecode.iterm2",
        "reliability": 0.9,
    },

    # System
    "system settings": {
        "category": AppCategory.SYSTEM_SETTINGS,
        "affordances": {AppAffordance.CHANGE_SETTINGS, AppAffordance.SET_WALLPAPER},
        "bundle_id": "com.apple.systempreferences",
        "reliability": 0.5,
        "known_issues": ["UI is complex and changes between OS versions"],
    },

    # Communication
    "mail": {
        "category": AppCategory.COMMUNICATION,
        "affordances": {AppAffordance.SEND_EMAIL},
        "bundle_id": "com.apple.mail",
        "reliability": 0.7,
    },
    "messages": {
        "category": AppCategory.COMMUNICATION,
        "affordances": {AppAffordance.SEND_MESSAGE},
        "bundle_id": "com.apple.MobileSMS",
        "reliability": 0.8,
        "known_issues": [
            "Inbound history requires Full Disk Access for Aura.app",
            "Outbound automation requires macOS Automation permission for Messages",
        ],
        "adapter_class": "MessagesTransport",
    },
}


# ---------------------------------------------------------------------------
# App Registry
# ---------------------------------------------------------------------------

class AppRegistry:
    """Registry of installed apps and their affordances.

    Usage:
        registry = get_app_registry()
        await registry.discover()

        # Find the best app for creating text
        app = registry.get_best_app_for(AppAffordance.CREATE_TEXT)

        # Find all browsers
        browsers = registry.get_apps_by_category(AppCategory.BROWSER)

        # Check if an app is installed
        installed = registry.is_installed("Google Chrome")
    """

    def __init__(self) -> None:
        self._apps: Dict[str, InstalledApp] = {}
        self._discovered = False
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("app_registry", self, required=False)
        await self.discover()
        self._started = True
        logger.info(
            "AppRegistry ONLINE — discovered %d apps (%s)",
            len(self._apps),
            ", ".join(sorted(self._apps.keys())[:10]),
        )

    async def discover(self) -> None:
        """Scan the system for installed applications."""
        discovered: Dict[str, InstalledApp] = {}

        # 1. Scan /Applications and ~/Applications
        for app_dir in [Path("/Applications"), Path.home() / "Applications"]:
            if not app_dir.exists():
                continue
            try:
                for entry in app_dir.iterdir():
                    if entry.suffix == ".app":
                        name = entry.stem
                        name_lower = name.lower()
                        profile = KNOWN_APP_PROFILES.get(name_lower, {})

                        app = InstalledApp(
                            name=name,
                            path=str(entry),
                            bundle_id=profile.get("bundle_id", ""),
                            category=profile.get("category", AppCategory.UNKNOWN),
                            affordances=set(profile.get("affordances", set())),
                            reliability=profile.get("reliability", 0.5),
                            known_issues=list(profile.get("known_issues", [])),
                            adapter_class=profile.get("adapter_class", ""),
                        )
                        discovered[name_lower] = app

                    # Also check subdirectories one level deep
                    if entry.is_dir() and not entry.suffix:
                        try:
                            for sub in entry.iterdir():
                                if sub.suffix == ".app":
                                    sname = sub.stem
                                    sname_lower = sname.lower()
                                    sprofile = KNOWN_APP_PROFILES.get(sname_lower, {})
                                    discovered[sname_lower] = InstalledApp(
                                        name=sname,
                                        path=str(sub),
                                        bundle_id=sprofile.get("bundle_id", ""),
                                        category=sprofile.get("category", AppCategory.UNKNOWN),
                                        affordances=set(sprofile.get("affordances", set())),
                                        reliability=sprofile.get("reliability", 0.5),
                                    )
                        except PermissionError:
                            continue
            except PermissionError:
                continue

        # 2. Add known system apps that don't live in /Applications
        for name_lower, profile in KNOWN_APP_PROFILES.items():
            if name_lower not in discovered:
                # Check if it's a system app via `mdfind`
                try:
                    proc = await get_subprocess_gateway().spawn_async(
                        ["mdfind", f"kMDItemCFBundleIdentifier == '{profile.get('bundle_id', '')}'"],
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        read_only=True,
                        source="app_registry.bundle_discovery",
                        accelerator_capability="none",
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
                    paths = stdout.decode().strip().split("\n") if stdout else []
                    if paths and paths[0]:
                        discovered[name_lower] = InstalledApp(
                            name=name_lower.title(),
                            path=paths[0],
                            bundle_id=profile.get("bundle_id", ""),
                            category=profile.get("category", AppCategory.UNKNOWN),
                            affordances=set(profile.get("affordances", set())),
                            reliability=profile.get("reliability", 0.5),
                            known_issues=list(profile.get("known_issues", [])),
                            adapter_class=profile.get("adapter_class", ""),
                        )
                except (OSError, asyncio.TimeoutError):
                    continue

        self._apps = discovered
        self._discovered = True

        # Update WorldState with installed apps list
        try:
            ws = ServiceContainer.get("world_state", default=None)
            if ws and hasattr(ws, "installed_apps"):
                ws.installed_apps = sorted(self._apps.keys())
        except (ImportError, AttributeError, RuntimeError):
            pass

    def is_installed(self, app_name: str) -> bool:
        """Check if an app is installed (case-insensitive)."""
        return app_name.lower() in self._apps

    def get_app(self, app_name: str) -> Optional[InstalledApp]:
        """Get app info by name."""
        return self._apps.get(app_name.lower())

    def get_apps_by_category(self, category: AppCategory) -> List[InstalledApp]:
        """Find all installed apps in a category."""
        return [a for a in self._apps.values() if a.category == category]

    def get_apps_with_affordance(self, affordance: AppAffordance) -> List[InstalledApp]:
        """Find all apps that can perform an affordance, sorted by reliability."""
        matching = [a for a in self._apps.values() if affordance in a.affordances]
        matching.sort(key=lambda a: a.reliability, reverse=True)
        return matching

    def get_best_app_for(
        self, affordance: AppAffordance, preferred: str = ""
    ) -> Optional[InstalledApp]:
        """Get the most reliable app for an affordance.

        If preferred is specified and installed, use it if it has the affordance.
        Otherwise, return the highest-reliability app with that affordance.
        """
        if preferred:
            pref_app = self.get_app(preferred)
            if pref_app and affordance in pref_app.affordances:
                return pref_app

        candidates = self.get_apps_with_affordance(affordance)
        return candidates[0] if candidates else None

    def get_preferred_browser(self) -> Optional[InstalledApp]:
        """Get the preferred browser (Chrome > Safari > Firefox > Arc)."""
        preference_order = ["google chrome", "safari", "firefox", "arc"]
        for name in preference_order:
            if name in self._apps:
                return self._apps[name]
        # Fallback: any browser
        browsers = self.get_apps_by_category(AppCategory.BROWSER)
        return browsers[0] if browsers else None

    def get_preferred_text_editor(self) -> Optional[InstalledApp]:
        """Get the preferred text editor (TextEdit > Notes > VS Code)."""
        preference_order = ["textedit", "notes", "visual studio code", "sublime text"]
        for name in preference_order:
            if name in self._apps:
                return self._apps[name]
        editors = self.get_apps_by_category(AppCategory.TEXT_EDITOR)
        return editors[0] if editors else None

    def get_capability_report(self) -> Dict[str, Any]:
        """Produce a summary of what this machine can do.

        Used by TaskDecomposer to plan realistic task graphs.
        """
        available_affordances: Set[str] = set()
        for app in self._apps.values():
            for aff in app.affordances:
                available_affordances.add(aff.value)

        browsers = self.get_apps_by_category(AppCategory.BROWSER)
        editors = self.get_apps_by_category(AppCategory.TEXT_EDITOR)

        return {
            "total_apps": len(self._apps),
            "available_affordances": sorted(available_affordances),
            "has_browser": bool(browsers),
            "preferred_browser": browsers[0].name if browsers else None,
            "has_text_editor": bool(editors),
            "preferred_editor": editors[0].name if editors else None,
            "has_terminal": bool(self.get_apps_by_category(AppCategory.TERMINAL)),
            "has_file_manager": bool(self.get_apps_by_category(AppCategory.FILE_MANAGER)),
            "categories": {
                cat.value: [a.name for a in self.get_apps_by_category(cat)]
                for cat in AppCategory
                if self.get_apps_by_category(cat)
            },
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "discovered": self._discovered,
            "total_apps": len(self._apps),
            "categories": {
                cat.value: len(self.get_apps_by_category(cat))
                for cat in AppCategory
                if self.get_apps_by_category(cat)
            },
        }

    def all_apps(self) -> List[InstalledApp]:
        """Return all discovered apps."""
        return list(self._apps.values())


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[AppRegistry] = None


def get_app_registry() -> AppRegistry:
    global _instance
    if _instance is None:
        _instance = AppRegistry()
    return _instance


__all__ = [
    "AppRegistry",
    "InstalledApp",
    "AppCategory",
    "AppAffordance",
    "KNOWN_APP_PROFILES",
    "get_app_registry",
]
