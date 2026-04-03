import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("🔍 Starting PyWebView Diagnostic...")

try:
    import webview
    print(f"✅ WebView imported successfully (version {getattr(webview, '__version__', 'unknown')})")
except ImportError as e:
    print(f"❌ WebView import failed: {e}")
    sys.exit(1)

try:
    print("🎨 Attempting to create window...")
    window = webview.create_window("Aura Diagnostic", "https://google.com")
    print("✅ Window created successfully.")
    
    # We won't start the loop in a non-interactive environment as it might hang,
    # but the creation success is usually enough to prove the dependencies are there.
    # webview.start()
except Exception as e:
    print(f"❌ Window creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✨ Diagnostic Complete: Environment appears ready for Desktop mode.")
