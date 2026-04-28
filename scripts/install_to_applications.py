
import os
import sys
import shutil
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ICON_SOURCE = PROJECT_ROOT / "aura_icon.icns"
LAUNCHER_SOURCE = PROJECT_ROOT / "aura_main.py"

# Target paths
APP_NAME = "Aura.app"
APPLICATIONS_DIR = Path("/Applications")
TARGET_APP_PATH = APPLICATIONS_DIR / APP_NAME
CONTENTS_DIR = TARGET_APP_PATH / "Contents"
MACOS_DIR = CONTENTS_DIR / "MacOS"
RESOURCES_DIR = CONTENTS_DIR / "Resources"

def install(target_path=TARGET_APP_PATH):
    print(f"🚀 Installing Aura to {target_path}...")
    
    contents_dir = target_path / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"

    # 1. Check if it's a full standalone bundle or just a light wrapper
    is_standalone = (target_path / "Contents" / "Frameworks").exists() or (target_path / "Contents" / "Resources" / "Python3").exists()
    
    if is_standalone:
        print(f"  Detected standalone bundle. Syncing source files instead of replacing with wrapper...")
        # Sync source directories
        src_dirs = ["core", "interface", "senses", "memory", "embodiment", "security", "scripts", "brain", "skills"]
        for d in src_dirs:
            src = PROJECT_ROOT / d
            dst = resources_dir / d
            if src.exists():
                if dst.exists(): shutil.rmtree(dst)
                shutil.copytree(src, dst)
        print("  Source sync complete.")
        return

    # 2. Clean existing (for light wrappers only)
    if target_path.exists():
        print(f"  Removing existing light wrapper at {target_path}...")
        get_task_tracker().create_task(get_storage_gateway().delete_tree(target_path, cause='install'))
        
    # 3. Create structure
    get_task_tracker().create_task(get_storage_gateway().create_dir(macos_dir, cause='install'))
    get_task_tracker().create_task(get_storage_gateway().create_dir(resources_dir, cause='install'))
    
    # 3. Create Shell Script Wrapper
    wrapper_path = macos_dir / "Aura"
    with open(wrapper_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"export AURA_SOURCE_PATH=\"{PROJECT_ROOT}\"\n")
        f.write(f"export PATH=\"/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH\"\n")
        f.write(f"python3 \"{LAUNCHER_SOURCE}\" >> /tmp/aura_app.log 2>&1\n")
    
    os.chmod(wrapper_path, 0o755)
    print("  Created executable wrapper.")
    
    # 4. Copy Icon
    if ICON_SOURCE.exists():
        shutil.copy(ICON_SOURCE, resources_dir / "icon.icns")
        print("  Attached icon.")
        
    # 5. Create Info.plist
    info_plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>Aura</string>
    <key>CFBundleIconFile</key>
    <string>icon.icns</string>
    <key>CFBundleIdentifier</key>
    <string>com.aura.sovereign</string>
    <key>CFBundleName</key>
    <string>Aura</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>8.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
"""
    with open(contents_dir / "Info.plist", "w") as f:
        f.write(info_plist)
    print("  Generated Info.plist.")

    # 6. Touch the app bundle to refresh Finder
    target_path.touch()
    
    print(f"\n✅ Aura is now installed in {target_path}!")

if __name__ == "__main__":
    # Install to both Applications and Desktop
    install(TARGET_APP_PATH)
    desktop_path = Path.home() / "Desktop" / "Aura.app"
    install(desktop_path)
