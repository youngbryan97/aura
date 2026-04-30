from core.utils.task_tracker import get_task_tracker
import asyncio
import time
import logging
import sys
import multiprocessing
import os
import json
from pathlib import Path

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup minimal logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Verify.Phase2.2")

# Top-level entry points for multiprocessing pickling
def echo_actor_entry(conn):
    """Refined echo actor that handles the request/response protocol."""
    while True:
        try:
            if conn.poll(1.0):
                raw = conn.recv()
                msg = json.loads(raw) if isinstance(raw, str) else raw
                
                # If it's a request, send back a response with response_to
                if msg.get("is_request") and "request_id" in msg:
                    resp = {
                        "response_to": msg["request_id"],
                        "payload": msg["payload"],
                        "type": "response"
                    }
                    conn.send(json.dumps(resp))
                else:
                    # Just echo normal messages
                    conn.send(raw)
            else:
                continue
        except (EOFError, BrokenPipeError, OSError, json.JSONDecodeError):
             break

def crashing_actor_entry(conn):
    sys.exit(1)

async def test_circuit_breaker():
    logger.info("🧪 Testing Circuit Breaker (Fork Bomb Protection)...")
    from core.supervisor.tree import SupervisionTree, ActorSpec
    
    supervisor = SupervisionTree()
    
    spec = ActorSpec(
        name="bomb_actor",
        entry_point=crashing_actor_entry,
        max_restarts=3,
        restart_delay=0.1,
        window_seconds=10
    )
    
    supervisor.add_actor(spec)
    supervisor.start_actor("bomb_actor")
    
    # Run the poller manually to monitor
    for _ in range(10): 
        supervisor._poll_health()
        time.sleep(0.5)
            
    actor = supervisor._actors["bomb_actor"]
    if actor.is_circuit_broken:
        logger.info("✅ SUCCESS: Circuit breaker engaged after 3 failures.")
    else:
        logger.error(f"❌ FAILURE: Circuit breaker did not engage. Consecutive Failures: {actor.consecutive_failures}")

async def test_pipe_hotswap():
    logger.info("🧪 Testing Pipe Hot-Swap (Ghost Pipe Fix)...")
    from core.supervisor.tree import SupervisionTree, ActorSpec
    from core.bus.actor_bus import ActorBus
    from core.container import ServiceContainer
    
    supervisor = SupervisionTree()
    actor_bus = ActorBus()
    ServiceContainer.register_instance("actor_bus", actor_bus)
    
    hotswap_done = asyncio.Event()

    # Mock restart callback
    async def on_restart_async(name, pipe):
        logger.info(f"🔄 Callback: {name} restarted. Re-binding...")
        await actor_bus.update_actor(name, pipe)
        hotswap_done.set()

    def on_restart(name, pipe):
        get_task_tracker().create_task(on_restart_async(name, pipe))
    
    supervisor.set_restart_callback(on_restart)

    spec = ActorSpec(name="echo", entry_point=echo_actor_entry, restart_delay=0.5)
    supervisor.add_actor(spec)
    pipe = supervisor.start_actor("echo")
    actor_bus.add_actor("echo", pipe, is_child=True)
    
    # 1. Verify it works
    resp = await actor_bus.request("echo", "ping", "hello")
    logger.info(f"   Initial Ping: {resp}")
    
    # 2. Kill it
    logger.info("   💀 Killing echo actor...")
    actor = supervisor._actors["echo"]
    os.kill(actor.process.pid, 9)
    
    # 3. Wait for restart and hot-swap
    # We must ensure we don't block the loop entirely
    for _ in range(20):
        # We wrap _poll_health in a thread if it blocks, but here we just call it
        # and hope the delay isn't too long.
        # Actually, let's just wait for hotswap_done
        supervisor._poll_health()
        if hotswap_done.is_set():
             break
        await asyncio.sleep(0.5)
    
    # 4. Verify re-bound communication
    try:
        if not hotswap_done.is_set():
             logger.warning("   ⚠️ Hotswap event never fired. Checking manual state...")
        
        resp = await actor_bus.request("echo", "ping", "post-restart")
        logger.info(f"   Post-Restart Ping: {resp}")
        logger.info("✅ SUCCESS: Hot-swap re-bound the IPC pipe.")
    except Exception as e:
        logger.error(f"❌ FAILURE: Could not communicate after restart: {e}")

async def test_shm_atomicity():
    logger.info("🧪 Testing SHM Atomicity (Torn Read Prevention)...")
    from core.bus.shared_mem_bus import SharedMemoryTransport
    
    shm_name = "test_atomicity_shm"
    transport_w = SharedMemoryTransport(shm_name, size=1024)
    transport_w.create()
    
    transport_r = SharedMemoryTransport(shm_name)
    transport_r.attach()
    
    from threading import Event
    stop_event = Event()
    
    def writer_loop():
        i = 0
        while not stop_event.is_set():
            data = {"id": i, "payload": "X" * 100}
            try:
                transport_w.write(data)
            except Exception:
                pass
            i += 1
            
    from threading import Thread
    w = Thread(target=writer_loop)
    w.start()
    
    # Read stress test
    errors = 0
    for _ in range(200):
        data = transport_r.read()
        if data and not isinstance(data, dict):
             errors += 1
        time.sleep(0.001)
        
    stop_event.set()
    w.join()
    transport_w.close()
    transport_r.close()
    
    if errors == 0:
        logger.info("✅ SUCCESS: No torn reads detected in 200 stress cycles.")
    else:
        logger.error(f"❌ FAILURE: {errors} torn reads detected.")

async def main():
    await test_circuit_breaker()
    await test_pipe_hotswap()
    await test_shm_atomicity()
    logger.info("🏁 Phase 2.2 Verification Complete.")

if __name__ == "__main__":
    asyncio.run(main())
