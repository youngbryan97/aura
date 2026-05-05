
import sys
import os
sys.path.append("/Users/bryan/.aura/live-source")

import pexpect
import pyte
import time

nethack_path = "/opt/homebrew/bin/nethack"
env = os.environ.copy()
env["TERM"] = "xterm-256color"
child = pexpect.spawn(f"{nethack_path} -u Aura", env=env, encoding='utf-8')
child.setwinsize(24, 80)

print("Spawned. Reading...")
try:
    for _ in range(5):
        try:
            out = child.read_nonblocking(size=1000, timeout=1.0)
            print(f"Read: {repr(out)}")
        except pexpect.TIMEOUT:
            print("Timeout.")
            # break
except pexpect.EOF:
    print("EOF reached.")

print(f"Is alive: {child.isalive()}")
if child.isalive():
    print("Sending ' ' (space)...")
    child.send(" ")
    time.sleep(1)
    print(f"After space, alive: {child.isalive()}")
