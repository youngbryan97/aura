"""A process result survives interruption without claiming an unfinished task passed."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from tools import run_proof_step as proof

pytestmark = pytest.mark.unit


def command(artifact, script, timeout=10):
    return [sys.executable, str(proof.ROOT/'tools/run_proof_step.py'), '--name', 'bounded-fixture',
            '--timeout', str(timeout), '--artifact', str(artifact), '--', sys.executable, '-c', script]


def test_a_child_exit_is_recorded_separately_from_its_task_outcome(tmp_path):
    artifact = tmp_path/'result.json'
    result = subprocess.run(command(artifact, 'raise SystemExit(7)'), timeout=15, capture_output=True)
    record = json.loads(artifact.read_text())
    assert result.returncode == record['returncode'] == 7
    assert record['status'] == 'finished' and record['child_reaped']
    assert not record['passed'] and record['task_outcome'] == 'requires_independent_review'


def test_a_timed_out_child_leaves_a_failed_terminal_record(tmp_path):
    artifact = tmp_path/'timed-out.json'
    result = subprocess.run(command(artifact, 'import time; time.sleep(30)', timeout=.15),
                            timeout=15, capture_output=True)
    record = json.loads(artifact.read_text())
    assert result.returncode == 124 and record['timed_out'] and not record['passed']
    assert record['status'] == 'timed_out' and record['child_reaped']


@pytest.mark.skipif(os.name != 'posix', reason='the proof supervisor owns a POSIX process group')
def test_the_running_checkpoint_exists_before_interrupting_the_supervisor(tmp_path):
    artifact = tmp_path/'interrupted.json'
    proc = subprocess.Popen(command(artifact, 'import time; time.sleep(30)'),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic()+10
        running = None
        while time.monotonic() < deadline:
            if artifact.exists():
                running = json.loads(artifact.read_text())
                if running['status'] == 'running':
                    break
            assert proc.poll() is None
            time.sleep(.02)
        assert running is not None and running['status'] == 'running'
        assert running['supervisor_pid'] == proc.pid and running['child_pid'] > 0
        assert running['passed'] is False and running['returncode'] is None
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=10)
        record = json.loads(artifact.read_text())
        assert proc.returncode == 143 and record['interrupted_by'] == signal.SIGTERM
        assert record['status'] == 'interrupted' and record['child_reaped'] and not record['passed']
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.communicate(timeout=10)


@pytest.mark.parametrize('timeout', ['nan', 'inf', '-1', '0'])
def test_invalid_time_budgets_cannot_start_a_child(monkeypatch, tmp_path, timeout):
    def forbidden(*args, **kwargs):
        raise AssertionError('an invalid budget launched work')
    monkeypatch.setattr(proof.subprocess, 'Popen', forbidden)
    with pytest.raises(SystemExit) as result:
        proof.main(['--name', 'invalid', '--timeout', timeout, '--artifact', str(tmp_path/'none.json'),
                    '--', sys.executable])
    assert result.value.code == 2 and not (tmp_path/'none.json').exists()


def test_failed_running_checkpoint_stops_the_owned_child(monkeypatch, tmp_path):
    saved = []
    spawned = []
    original_spawn = proof.subprocess.Popen
    def spawn(*args, **kwargs):
        proc = original_spawn(*args, **kwargs)
        spawned.append(proc)
        return proc
    def save(path, record):
        saved.append(record.copy())
        if record['status'] == 'running':
            raise OSError('fixture disk refused the running checkpoint')
    monkeypatch.setattr(proof.subprocess, 'Popen', spawn)
    monkeypatch.setattr(proof, '_save', save)
    with pytest.raises(OSError, match='disk refused'):
        proof.main(['--name', 'no-checkpoint', '--timeout', '10', '--artifact', str(tmp_path/'none.json'),
                    '--', sys.executable, '-c', 'import time; time.sleep(30)'])
    assert [r['status'] for r in saved] == ['prepared', 'running']
    assert len(spawned) == 1 and spawned[0].poll() is not None
