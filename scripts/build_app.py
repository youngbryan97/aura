
import os
import sys
import shutil
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

def build_app():
    print("🚀 Building Aura.app with PyInstaller...")
    
    # Clean previous builds
    dist_dir = PROJECT_ROOT / "dist"
    build_dir = PROJECT_ROOT / "build"
    if dist_dir.exists(): shutil.rmtree(dist_dir)
    if build_dir.exists(): shutil.rmtree(build_dir)
    
    # PyInstaller Command
    # We use aura_main.py as entrypoint
    cmd = [
        "pyinstaller",
        "--name=Aura",
        "--noconfirm",
        "--onedir", # Use directory mode for instant launch performance on macOS
        "--windowed", # No console window (macOS .app)
        "--icon=aura_icon.icns", # Use the correct icon file in project root
        "--add-data=interface/static:interface/static", # Include UI assets
        "--add-data=core/identity_base.txt:core", # Include Base Identity
        "--add-data=data:data", # Include all default data/db folders
        "--collect-all=core", # Force collect all internal modules
        "--collect-all=skills",
        "--collect-all=utils",
        "--collect-all=torch",
        "--collect-all=transformers",
        "--collect-all=sentence_transformers",
        "--collect-all=chromadb",
        "--collect-all=cv2",
        "--collect-all=pydantic",
        "--collect-all=fastapi",
        "--collect-all=webview",
        "--collect-all=numpy",
        "--collect-all=pandas",
        "--collect-all=matplotlib",
        "--collect-all=scipy",
        "--collect-all=sklearn",
        "--hidden-import=uvicorn.loops.auto",
        "--hidden-import=uvicorn.protocols.http.auto",
        "--hidden-import=websockets.legacy.server",
        "--hidden-import=websockets.legacy.client",
        "--hidden-import=litellm",
        "--hidden-import=pydantic",
        "--hidden-import=fastapi",
        "--hidden-import=jinja2",
        "--hidden-import=sentence_transformers",
        "--hidden-import=chromadb",
        "--hidden-import=cv2",
        "--hidden-import=pyautogui",
        "--hidden-import=pygetwindow",
        "--hidden-import=pywebview",
        "aura_main.py"
    ]
    
    # Check for icon
    icon_path = PROJECT_ROOT / "aura_icon.icns"
    if not icon_path.exists():
        print("⚠️ Icon not found, skipping icon flag.")
        cmd = [c for c in cmd if not c.startswith("--icon")]

    print(f"Running: {' '.join(cmd)}")
    
    # Explicitly set PYTHONPATH for PyInstaller analysis
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    
    try:
        subprocess.check_call(cmd, cwd=PROJECT_ROOT, env=env)
        print("\n✅ Build Complete!")
        print(f"App Bundle: {PROJECT_ROOT}/dist/Aura.app")
        
        # Move to Desktop for convenience
        desktop = Path.home() / "Desktop"
        target = desktop / "Aura.app"
        if target.exists():
            get_task_tracker().create_task(get_storage_gateway().delete_tree(target, cause='build_app'))
        
        try:
             # PyInstaller 'onedir' + 'windowed' on Mac creates an .app bundle in dist/
             # dist/Aura.app
             source_app = dist_dir / "Aura.app"
             if source_app.exists():
                 # Post-build: Add TCC permissions to Info.plist
                 plist_path = source_app / "Contents" / "Info.plist"
                 if plist_path.exists():
                     print("🔐 Injecting TCC Privacy Permissions into Info.plist...")
                     subprocess.run(["plutil", "-replace", "NSCameraUsageDescription", "-string", "Aura needs camera access for visual processing and spatial awareness.", str(plist_path)])
                     subprocess.run(["plutil", "-replace", "NSMicrophoneUsageDescription", "-string", "Aura needs microphone access for voice interaction and auditory processing.", str(plist_path)])
                     subprocess.run(["plutil", "-replace", "NSSpeechRecognitionUsageDescription", "-string", "Aura needs speech recognition for conversion of audio to text.", str(plist_path)])

                 # 1. Desktop
                 shutil.copytree(source_app, target)
                 print(f"📦 Copied to Desktop: {target}")
                 
                 # 2. Applications (The Sovereign Throne)
                 apps_path = Path("/Applications/Aura.app")
                 if apps_path.exists():
                     get_task_tracker().create_task(get_storage_gateway().delete_tree(apps_path, cause='build_app'))
                 shutil.copytree(source_app, apps_path)
                 print(f"📦 Force-Updated in Applications: {apps_path}")
             else:
                 # It might be just a binary if something went wrong or config diff
                 print(f"⚠️ .app bundle not found in {dist_dir}, check raw binary.")
        except Exception as e:
            print(f"⚠️ Failed to copy to targets: {e}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Build Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_app()
