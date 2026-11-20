"""Example paths in an answer are not a leak of her telemetry; a wall of paths on this machine still is (U07).

LIVE 20 November 2026, typed into her window: "Write a Python function that groups a list of file paths by extension,
then run it on a few example paths and show me the output." Her worker destroyed her 845-token answer as a
"telemetry_path_wall" for the example paths the person asked for, and the turn generated again. A leak names files and
folders that exist here; the paths in an example exist nowhere, so only paths on this machine make a wall.
"""
from __future__ import annotations

import pytest

from core.brain.llm.mlx_worker_surface_quality import _telemetry_sanitization_failure_reasons

pytestmark = pytest.mark.unit

ANSWER = """Here's a function that groups paths by extension:

```python
from collections import defaultdict
from pathlib import PurePath

def group_by_extension(paths):
    groups = defaultdict(list)
    for path in paths:
        groups[PurePath(path).suffix or "(none)"].append(path)
    return dict(groups)

paths = ["/home/ana/docs/report.pdf", "/home/ana/docs/notes.txt", "/home/ana/photos/cat.jpg",
         "/home/ana/photos/dog.jpg", "/home/ana/music/song.mp3", "/home/ana/docs/summary.pdf",
         "/home/ana/code/main.py", "/home/ana/code/util.py", "/home/ana/README"]
print(group_by_extension(paths))
```

For /home/ana/docs/report.pdf and /home/ana/docs/summary.pdf the key is .pdf; /home/ana/README has none.
"""


def test_the_example_paths_a_person_asked_for_are_content():
    assert ANSWER.count("/") > 30
    assert "telemetry_path_wall" not in _telemetry_sanitization_failure_reasons(ANSWER)


def test_a_wall_of_paths_on_this_machine_is_still_caught(tmp_path):
    files = []
    for name in ("alpha", "beta", "gamma", "delta"):
        made = tmp_path / "state" / "organs" / name / "ledger.json"
        made.parent.mkdir(parents=True)
        made.write_text("{}")
        files.append(str(made))
    wall = "My state lives at " + ", ".join(files * 3) + "."
    assert wall.count("/") > 30
    assert "telemetry_path_wall" in _telemetry_sanitization_failure_reasons(wall)
