"""A refused recorder must leave the current monitoring owner intact."""

import subprocess
import sys


def test_refused_start_preserves_another_monitoring_owner():
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
from core.connectome.activity import ActivityRecorder, _TOOL_ID
monitoring = sys.monitoring
monitoring.use_tool_id(_TOOL_ID, 'existing.owner')
callback = lambda *args: None
monitoring.register_callback(_TOOL_ID, monitoring.events.PY_START, callback)
monitoring.set_events(_TOOL_ID, monitoring.events.PY_START)
try:
    recorder = ActivityRecorder()
    assert recorder.start() is False
    recorder.stop()
    assert monitoring.get_tool(_TOOL_ID) == 'existing.owner'
    assert monitoring.get_events(_TOOL_ID) == monitoring.events.PY_START
    assert monitoring.register_callback(
        _TOOL_ID, monitoring.events.PY_START, callback
    ) is callback
finally:
    monitoring.set_events(_TOOL_ID, 0)
    monitoring.register_callback(_TOOL_ID, monitoring.events.PY_START, None)
    monitoring.free_tool_id(_TOOL_ID)
"""],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
