"""Replace this process with a command outside the launching shell's session."""

from __future__ import annotations

import os
import sys


def main() -> None:
    command = sys.argv[1:]
    if not command:
        raise SystemExit("exec_detached: a command is required")
    # nohup ignores SIGHUP, but does not survive cleanup of the shell's group.
    # Keep the PID so the launcher and runtime locks identify the same owner.
    if os.getsid(0) != os.getpid():
        os.setsid()
    with open(os.devnull, "rb") as source:
        os.dup2(source.fileno(), 0)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
