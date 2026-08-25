
"""ephemeral test file: must trigger governance lint"""

import subprocess

def use_unsafe() -> None:
    subprocess.run(["echo", "unsafe"], check=False)
