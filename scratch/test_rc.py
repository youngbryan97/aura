
import pexpect
import os
import time

nethack_path = "/opt/homebrew/bin/nethack"
env = os.environ.copy()
env["TERM"] = "xterm-256color"

rc_path = os.path.expanduser("~/.nethackrc_aura")
with open(rc_path, "w") as f:
    f.write("OPTIONS=color,autoquiver,autopickup,hitpointbar,showexp,time,statuslines:2\n")
    f.write("OPTIONS=pettype:none\n")
    f.write("OPTIONS=pickup_types:$\n") # Only pick up gold by default
    
env["NETHACKOPTIONS"] = rc_path

print("Testing user: Aura")
child = pexpect.spawn(f"{nethack_path} -u Aura", env=env, encoding='utf-8')
time.sleep(1)
try:
    out = child.read_nonblocking(size=1000, timeout=0.1)
    print(f"Output: {repr(out)}")
except:
    print("Timeout/EOF.")
print(f"Is alive: {child.isalive()}")
child.terminate()
