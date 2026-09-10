import asyncio
import threading

from core.consciousness.phenomenological_experiencer import PhenomenologicalExperiencer


def test_stop_settles_producer_before_off_loop_save():
    async def run():
        owner = threading.get_ident()
        settled = []

        async def update():
            try:
                await asyncio.Future()
            finally:
                settled.append(True)

        instance = object.__new__(PhenomenologicalExperiencer)
        instance._running = True
        instance._update_task = asyncio.create_task(update())
        await asyncio.sleep(0)

        def save():
            assert settled == [True]
            assert threading.get_ident() != owner

        instance._save_phenomenal_memory = save
        await instance.stop()
        assert instance._update_task is None
        assert not instance._running

    asyncio.run(run())
