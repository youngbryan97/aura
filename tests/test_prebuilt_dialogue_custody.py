from core.brain.inference_gate import InferenceGate


def test_history_window_does_not_orphan_an_answer():
    gate = InferenceGate.__new__(InferenceGate)
    messages = [
        {"role": "system", "content": "identity"},
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
        {"role": "user", "content": "why that answer?"},
    ]
    assert gate._compact_prebuilt_messages(messages, history_limit=2) == messages


def test_history_answer_is_not_clipped_when_it_fits_total_budget(monkeypatch):
    monkeypatch.setenv("AURA_CORTEX_CTX", "8192")
    gate = InferenceGate.__new__(InferenceGate)
    answer = "```python\n" + "    value += 1\n" * 260 + "print(value)\n```"
    messages = [
        {"role": "system", "content": "identity"},
        {"role": "user", "content": "show the code"},
        {"role": "assistant", "content": answer},
        {"role": "user", "content": "what does the final line do?"},
    ]
    compact = gate._compact_prebuilt_messages(messages)
    assert compact[2]["content"] == answer


def test_budget_eviction_removes_whole_exchanges():
    gate = InferenceGate.__new__(InferenceGate)
    history = [
        message
        for i in range(4)
        for message in (
            {"role": "user", "content": f"q{i} " + "q" * 850},
            {"role": "assistant", "content": f"a{i} " + "a" * 650},
        )
    ]
    current = {"role": "user", "content": "why?"}
    # A background caller, which is what asks for a count window now. The
    # foreground stopped applying a character budget here on 2026-09-09:
    # `_fit_prompt_to_window` allocates its history with the output reserve
    # known, and a second blind budget underneath it cut history that fitted.
    # The invariant this test holds is the eviction's shape, and eviction is
    # what a windowed caller still gets.
    compact = gate._compact_prebuilt_messages(
        [{"role": "system", "content": "identity"}, *history, current],
        budget_profile="contract",
        history_limit=8,
    )
    contents = [message["content"] for message in compact]
    assert sum(map(len, contents)) <= 2800
    assert compact[-1] == current
    for user, answer in zip(history[::2], history[1::2]):
        assert (user["content"] in contents) == (answer["content"] in contents)
