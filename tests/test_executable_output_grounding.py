from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from core.brain.verifiers.code_engine import ground_python_output_claims


@dataclass
class _Result:
    ok: bool = True
    stdout: str = ""
    refused: bool = False
    timed_out: bool = False
    isolation: dict[str, object] = field(
        default_factory=lambda: {
            "sandboxed": True,
            "isolation_level": "kernel:test",
        }
    )


class _Sandbox:
    def __init__(self, result: _Result) -> None:
        self.result = result
        self.calls: list[str] = []

    async def run(self, code: str) -> _Result:
        self.calls.append(code)
        return self.result


_ASYNC_REPLY = '''No. The lock excludes only coroutines that need that lock.

```python
import asyncio

asyncio.run(main())
```

Typical output (order may vary):

```text
A: inside
B: after
```
'''


@pytest.mark.asyncio
async def test_a_wrong_output_claim_is_replaced_by_observed_stdout() -> None:
    sandbox = _Sandbox(_Result(stdout="A: inside\nA: after\nB: inside\nB: after\n"))

    result = await ground_python_output_claims(_ASYNC_REPLY, sandbox=sandbox)

    assert sandbox.calls == ["import asyncio\n\nasyncio.run(main())\n"]
    assert "Typical output" not in result.text
    assert "One observed run:" in result.text
    assert "A: after\nB: inside" in result.text
    assert result.changed is True
    assert result.to_dict()["grounded_count"] == 1
    assert result.receipts[0]["status"] == "grounded_to_observation"


@pytest.mark.asyncio
async def test_a_matching_output_claim_is_left_byte_identical() -> None:
    reply = _ASYNC_REPLY.replace("B: after", "A: after\nB: inside\nB: after")
    sandbox = _Sandbox(_Result(stdout="A: inside\nA: after\nB: inside\nB: after\n"))

    result = await ground_python_output_claims(reply, sandbox=sandbox)

    assert result.text == reply
    assert result.changed is False
    assert result.receipts[0]["status"] == "verified_match"


@pytest.mark.asyncio
async def test_an_unlabelled_example_does_not_trigger_execution() -> None:
    reply = "Here is the example:\n\n```python\nprint('ok')\n```\n"
    sandbox = _Sandbox(_Result(stdout="ok\n"))

    result = await ground_python_output_claims(reply, sandbox=sandbox)

    assert result.text == reply
    assert result.receipts == ()
    assert sandbox.calls == []


@pytest.mark.asyncio
async def test_an_unverified_output_claim_is_removed_without_erasing_the_answer() -> None:
    sandbox = _Sandbox(
        _Result(
            ok=False,
            refused=True,
            isolation={"sandboxed": True, "isolation_level": "kernel:test"},
        )
    )

    result = await ground_python_output_claims(_ASYNC_REPLY, sandbox=sandbox)

    assert result.text.startswith("No. The lock excludes")
    assert "```python" in result.text
    assert "B: after" not in result.text
    assert "no output is claimed here" in result.text
    assert result.receipts[0]["status"] == "execution_refused"


@pytest.mark.asyncio
async def test_only_python_output_pairs_are_execution_claims() -> None:
    reply = """```javascript
console.log('ok')
```

Expected output:

```text
ok
```
"""
    sandbox = _Sandbox(_Result(stdout="ok\n"))

    result = await ground_python_output_claims(reply, sandbox=sandbox)

    assert result.text == reply
    assert result.receipts == ()
    assert sandbox.calls == []


@pytest.mark.asyncio
async def test_shared_markdown_parser_handles_tildes_and_longer_closers() -> None:
    reply = """~~~python
print('ok')
~~~~

Expected output:

~~~~text
wrong
~~~~~
"""
    sandbox = _Sandbox(_Result(stdout="ok\r\n"))

    result = await ground_python_output_claims(reply, sandbox=sandbox)

    assert sandbox.calls == ["print('ok')\n"]
    assert "One observed run:" in result.text
    assert "```text\nok\n```" in result.text


@pytest.mark.asyncio
async def test_output_comparison_ignores_ansi_crlf_and_one_terminal_newline() -> None:
    reply = """```python
print('ok')
```

Observed stdout:

```text
ok
```
"""
    sandbox = _Sandbox(_Result(stdout="\x1b[32mok\x1b[0m\r\n"))

    result = await ground_python_output_claims(reply, sandbox=sandbox)

    assert result.text == reply
    assert result.receipts[0]["status"] == "verified_match"


@pytest.mark.asyncio
async def test_chat_delivery_records_grounding_and_visible_authorship(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.brain.verifiers import code_engine
    from interface.routes import chat

    sandbox = _Sandbox(_Result(stdout="A: inside\nA: after\nB: inside\nB: after\n"))

    async def _ground(text: str):
        return await ground_python_output_claims(text, sandbox=sandbox)

    monkeypatch.setattr(code_engine, "ground_python_output_claims", _ground)
    trace: dict[str, object] = {"turn_id": "test-executable-grounding"}

    delivered = await chat._ground_executable_output_claims_for_delivery(
        trace,
        _ASYNC_REPLY,
    )

    assert "One observed run:" in delivered
    assert trace["executable_output_grounding"]["grounded_count"] == 1
    assert trace["post_generation_repair_applied"] is True
    assert trace["deterministic_repair_applied"] is False
    assert trace["text_mutations"][0]["stage"] == "chat.executable_output_grounding"
    assert trace["text_mutations"][0]["authorship_effect"] == "augmented_by_runtime"
