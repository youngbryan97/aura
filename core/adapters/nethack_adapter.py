import pexpect
import pyte
import time
import os
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger("Aura.NetHackAdapter")

class NetHackAdapter:
    def __init__(self, nethack_path: str = "/opt/homebrew/bin/nethack"):
        self.nethack_path = nethack_path
        self.screen = pyte.Screen(80, 24)
        self.stream = pyte.Stream(self.screen)
        self.child = None
        self.last_output = ""

    def start(self, name: str = "Aura"):
        env = os.environ.copy()
        env["TERM"] = "vt100"
        
        # Create a custom .nethackrc for automation
        rc_path = os.path.expanduser("~/.nethackrc_aura")
        with open(rc_path, "w") as f:
            f.write("OPTIONS=color,autoquiver,autopickup,hitpointbar,showexp,time,statuslines:2\n")
            f.write("OPTIONS=pettype:none\n")
            f.write("OPTIONS=pickup_types:$\n") # Only pick up gold by default
            
        env["NETHACKOPTIONS"] = rc_path
        
        # Start nethack
        self.child = pexpect.spawn(f"{self.nethack_path} -u {name}", env=env, encoding='utf-8')
        self.child.setwinsize(24, 80)
        time.sleep(1.0)
        self._update_screen()

    def _update_screen(self):
        try:
            # Read all available output
            out = self.child.read_nonblocking(size=10000, timeout=0.1)
            if out:
                self.last_output = out
                self.stream.feed(out)
        except (pexpect.TIMEOUT, pexpect.EOF):
            pass

    def get_screen_text(self) -> str:
        self._update_screen()
        return "\n".join(self.screen.display)

    def send_action(self, action: str):
        """Sends a string to nethack."""
        if not self.child:
            return
        
        # Handle multi-character commands? 
        # Most nethack actions are single keys, but some are extended commands starting with #
        self.child.send(action)
        time.sleep(0.2)
        self._update_screen()
        
        # Check for --More-- or other prompts that block input
        screen = self.get_screen_text()
        if "--More--" in screen or "Hit return to continue" in screen or "Press return" in screen:
            logger.info("Clearing NetHack prompt...")
            self.child.sendline("")
            # Wait briefly for screen update
            time.sleep(0.5)
            # Re-read screen
            self.stream.feed(self.child.read_nonblocking(size=10000, timeout=0.1))
            screen = self.get_screen_text()

    def is_alive(self) -> bool:
        return self.child and self.child.isalive()

    def close(self):
        if self.child:
            self.child.terminate(force=True)

if __name__ == "__main__":
    # Basic smoke test
    adapter = NetHackAdapter()
    adapter.start()
    print("Screen after start:")
    print(adapter.get_screen_text())
    
    print("\nSending 'y' to auto-pick...")
    adapter.send_action("y")
    print(adapter.get_screen_text())
    
    adapter.close()
