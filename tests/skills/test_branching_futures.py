import asyncio
from core.skills.branching_futures import BranchingFuturesSkill, BranchingFutureInput

async def test():
    skill = BranchingFuturesSkill()
    params = BranchingFutureInput(
        goal="Print a test message",
        files_to_copy=["core/skills/branching_futures.py"],
        timeout_minutes=1
    )
    result = await skill.safe_execute(params)
    print(result)

if __name__ == "__main__":
    asyncio.run(test())
