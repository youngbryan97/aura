################################################################################


import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def test_ui_obfuscation():
    print(f"🧪 Verifying UI Obfuscation on {BASE_URL}/api/state...")
    try:
        # Fetch state multiple times to check for randomization
        skills = set()
        for i in range(5):
            res = requests.get(f"{BASE_URL}/api/state")
            data = res.json()
            skill_str = data.get("skills", "")
            print(f"   Sample {i+1}: {skill_str}")
            skills.add(skill_str)
            time.sleep(0.1)
            
        if len(skills) > 1:
            print("✅ Success: Skill display is randomized.")
        else:
            print("❌ Failure: Skill display is static (or luck).")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_cognitive_loop_fix():
    print("\n🧪 Verifying Cognitive Loop Fixes...")
    # We can't easily trigger the loop from outside without injecting a specific state,
    # but we can check if the server is responsive and not stuck.
    try:
        res = requests.get(f"{BASE_URL}/health")
        data = res.json()
        print(f"   Health: {data}")
        if data["status"] == "ok":
            print("✅ Success: Server is healthy.")
        else:
            print("❌ Failure: Server is unhealthy.")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_file_upload():
    print("\n🧪 Verifying File Upload...")
    try:
        files = {'file': ('test_upload.txt', 'Hello Aura, this is a test upload.')}
        res = requests.post(f"{BASE_URL}/api/upload", files=files)
        if res.status_code == 200:
            print(f"✅ Success: File uploaded (Status: {res.status_code})")
        else:
            print(f"❌ Failure: Upload failed (Status: {res.status_code})")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    try:
        test_ui_obfuscation()
        test_file_upload()
        test_cognitive_loop_fix()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Server not running on localhost:8000")


##
