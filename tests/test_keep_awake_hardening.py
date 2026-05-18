from core.runtime.keep_awake import MacKeepAwakeController


class _Process:
    def __init__(self, args, *, wait_timeout_once: bool = False):
        self.pid = 4242
        self.args = args
        self.returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self._wait_timeout_once = wait_timeout_once

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        if not self._wait_timeout_once:
            self.returncode = 0

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float):
        if self._wait_timeout_once:
            self._wait_timeout_once = False
            raise TimeoutError("still active")
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _darwin_controller(process_launcher):
    return MacKeepAwakeController(
        process_launcher=process_launcher,
        platform_name="Darwin",
        path_resolver=lambda name: "/usr/bin/caffeinate",
    )


def test_keep_awake_start_uses_injected_launcher():
    launched = []

    def launcher(command):
        launched.append(command)
        return _Process(command)

    controller = _darwin_controller(launcher)
    status = controller.start(reason="continuous runtime", keep_display_awake=True)

    assert status.active is True
    assert status.pid == 4242
    assert status.reason == "continuous runtime"
    assert status.command == ("caffeinate", "-i", "-m", "-s", "-d")
    assert launched == [status.command]


def test_keep_awake_stop_terminates_active_assertion():
    process = _Process(("caffeinate", "-i", "-m", "-s"))
    controller = _darwin_controller(lambda command: process)

    controller.start()
    status = controller.stop()

    assert status.active is False
    assert process.terminate_calls == 1
    assert process.kill_calls == 0


def test_keep_awake_stop_kills_after_timeout():
    process = _Process(("caffeinate", "-i", "-m", "-s"), wait_timeout_once=True)
    controller = _darwin_controller(lambda command: process)

    controller.start()
    status = controller.stop()

    assert status.active is False
    assert process.terminate_calls == 1
    assert process.kill_calls == 1


def test_keep_awake_start_reports_launcher_failure():
    calls = []

    def launcher(command):
        calls.append(command)
        raise OSError("spawn failed")

    controller = _darwin_controller(launcher)
    status = controller.start()

    assert status.active is False
    assert status.supported is True
    assert "spawn failed" in status.reason
    assert len(calls) == 1


def test_keep_awake_unsupported_platform_does_not_launch():
    calls = []
    controller = MacKeepAwakeController(
        process_launcher=lambda command: calls.append(command),
        platform_name="Linux",
        path_resolver=lambda name: "/usr/bin/caffeinate",
    )

    status = controller.start()

    assert status.active is False
    assert status.supported is False
    assert calls == []
