import os
import plistlib
import shutil
import sys
import tempfile
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.security.macos_bundle_manifest import (  # noqa: E402
    info_plist_overrides,
    write_entitlements_plist,
)

ICON_SOURCE = PROJECT_ROOT / "aura_icon.icns"
LAUNCHER_SOURCE = PROJECT_ROOT / "aura_main.py"

# Target paths
APP_NAME = "Aura.app"
APPLICATIONS_DIR = Path("/Applications")
TARGET_APP_PATH = APPLICATIONS_DIR / APP_NAME
CONTENTS_DIR = TARGET_APP_PATH / "Contents"
MACOS_DIR = CONTENTS_DIR / "MacOS"
RESOURCES_DIR = CONTENTS_DIR / "Resources"


def _write_info_plist(contents_dir: Path, executable_name: str, icon_name: str = "icon.icns") -> None:
    payload = {
        "CFBundleExecutable": executable_name,
        "CFBundleIconFile": icon_name,
        "CFBundleIdentifier": "com.aura.desktop",
        "CFBundleName": "Aura",
        "CFBundleDisplayName": "Aura",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "2026.4.20",
        "CFBundleVersion": "2026.4.20",
        "LSMinimumSystemVersion": "10.13",
        "NSHighResolutionCapable": True,
    }
    payload.update(info_plist_overrides())
    contents_dir.mkdir(parents=True, exist_ok=True)
    with (contents_dir / "Info.plist").open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=True)


def _apply_app_metadata(target_path: Path) -> None:
    contents_dir = target_path / "Contents"
    macos_dir = contents_dir / "MacOS"
    executable_name = "aura-launcher"
    if not (macos_dir / executable_name).exists():
        executable_name = "Aura"
    icon_name = "Aura.icns" if (contents_dir / "Resources" / "Aura.icns").exists() else "icon.icns"
    _write_info_plist(contents_dir, executable_name, icon_name=icon_name)
    write_entitlements_plist(contents_dir / "Resources" / "aura.entitlements")

def install(target_path=TARGET_APP_PATH):
    print(f"Installing Aura to {target_path}...")

    contents_dir = target_path / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"

    # 1. Check if it's a full standalone bundle or just a light wrapper
    is_standalone = (target_path / "Contents" / "Frameworks").exists() or (target_path / "Contents" / "Resources" / "Python3").exists()

    if is_standalone:
        print("  Detected standalone bundle. Syncing source files instead of replacing with wrapper...")
        # Sync source directories
        src_dirs = ["core", "interface", "senses", "memory", "embodiment", "security", "scripts", "brain", "skills"]
        for d in src_dirs:
            src = PROJECT_ROOT / d
            dst = resources_dir / d
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
        _apply_app_metadata(target_path)
        print("  Source sync complete.")
        return

    # 2. Clean existing (for light wrappers only)
    if target_path.exists():
        print(f"  Removing existing light wrapper at {target_path}...")
        shutil.rmtree(target_path)

    # 3. Create structure
    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)

    # 3. Create Shell Script Wrapper
    wrapper_path = macos_dir / "Aura"
    with open(wrapper_path, "w") as f:
        log_path = Path(tempfile.gettempdir()) / "aura_app.log"
        f.write("#!/bin/bash\n")
        f.write(f"export AURA_SOURCE_PATH=\"{PROJECT_ROOT}\"\n")
        f.write("export PATH=\"/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH\"\n")
        f.write(f"python3 \"{LAUNCHER_SOURCE}\" >> \"{log_path}\" 2>&1\n")

    os.chmod(wrapper_path, 0o755)
    print("  Created executable wrapper.")

    # 4. Copy Icon
    if ICON_SOURCE.exists():
        shutil.copy(ICON_SOURCE, resources_dir / "icon.icns")
        print("  Attached icon.")

    # 5. Create Info.plist and hardened-runtime entitlement metadata from the
    # same manifest as the native launcher bundle.  Old light wrappers used a
    # different bundle id and no Apple Events usage string, which made macOS TCC
    # grants look present in Settings while the actually launched app was denied.
    _apply_app_metadata(target_path)
    print("  Generated Info.plist.")

    # 6. Touch the app bundle to refresh Finder
    target_path.touch()

    print(f"\nAura is now installed in {target_path}!")

if __name__ == "__main__":
    # Install to both Applications and Desktop
    install(TARGET_APP_PATH)
    desktop_path = Path.home() / "Desktop" / "Aura.app"
    install(desktop_path)
