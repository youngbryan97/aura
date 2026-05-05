
import pexpect
import os
import time

__test__ = False

nethack_path = "/opt/homebrew/bin/nethack"
env = os.environ.copy()
env["TERM"] = "xterm-256color"

def test_user(name):
    print(f"Testing user: {name}")
    child = pexpect.spawn(f"{nethack_path} -u {name}", env=env, encoding='utf-8')
    time.sleep(1)
    try:
        out = child.read_nonblocking(size=1000, timeout=0.1)
        print(f"Output: {repr(out)}")
        if "Too many hacks" in out:
            print("FAILED: Too many hacks.")
        else:
            print("SUCCESS: Game started.")
    except Exception:
        print("Timeout/EOF.")
    child.terminate()

if __name__ == "__main__":
    test_user("Aura")
    test_user("AuraTest")
