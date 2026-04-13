import pytest

from core.brain.llm.context_limit import ContextManager, compact_working_memory


def test_context_manager_prune_preserves_recent_suffix():
    manager = ContextManager(max_tokens=20)
    history = "old line\n" + ("x" * 200) + "\nrecent line"

    pruned = manager.prune(history, "system")

    assert pruned.startswith("[...Earlier conversation forgotten...]")
    assert pruned.endswith("recent line")


@pytest.mark.asyncio
async def test_compact_working_memory_preserves_genesis_and_recent_tail():
    history = [{"role": "system", "content": "s0"}, {"role": "user", "content": "u0"}]
    history.extend({"role": "assistant", "content": f"m{i}"} for i in range(10))

    compacted = await compact_working_memory(history, max_raw_turns=6)

    assert compacted[:2] == history[:2]
    assert compacted[-1] == history[-1]
    assert len(compacted) == 6
