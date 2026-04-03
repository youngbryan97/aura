################################################################################


import sys
import os
import logging

# Setup logging to console
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("DebugImport")

# Add project root to path (assuming we run from autonomy_engine root)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

print(f"Project Root: {project_root}")
print(f"Sys Path: {sys.path[:3]}")

try:
    print("Attempting to import skills.web_search...")
    import skills.web_search
    print("✅ SUCCESS: Imported skills.web_search")
    
    print("Checking class...")
    from skills.web_search import EnhancedWebSearchSkill
    print(f"✅ Found class: {EnhancedWebSearchSkill}")
    print(f"   Name: {EnhancedWebSearchSkill.name}")

except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ Exception: {e}")
    import traceback
    traceback.print_exc()


##
