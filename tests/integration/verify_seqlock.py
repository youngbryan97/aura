import os
import time
import random
import multiprocessing
from core.bus.shared_mem_bus import SharedMemoryTransport

def writer_process(shm_name, size, stop_event):
    transport = SharedMemoryTransport(shm_name, size)
    transport.create()
    
    # Use simple dicts with repeated characters
    patterns = [{"data": chr(ord('A') + i) * 100} for i in range(10)]
    
    while not stop_event.is_set():
        pattern = random.choice(patterns)
        transport.write(pattern)
        # No sleep, max contention

def reader_process(shm_name, size, stop_event, results_queue):
    transport = SharedMemoryTransport(shm_name, size)
    # Wait for writer to create the segment
    for _ in range(10):
        try:
            transport.attach()
            break
        except:
            time.sleep(0.1)
    else:
        results_queue.put((0, 1)) # Fail to attach
        return

    errors = 0
    reads = 0
    while not stop_event.is_set():
        obj = transport.read()
        if obj is not None:
            reads += 1
            data = obj.get("data", "")
            if data and not all(c == data[0] for c in data):
                errors += 1
    results_queue.put((reads, errors))

def test_seqlock():
    shm_name = "test_seqlock_shm_v2"
    size = 1024 * 16 # 16KB
    stop_event = multiprocessing.Event()
    results_queue = multiprocessing.Queue()
    
    # Cleanup old shm
    try:
        from multiprocessing import shared_memory
        old_shm = shared_memory.SharedMemory(name=shm_name)
        old_shm.close()
        old_shm.unlink()
    except:
        pass

    writer = multiprocessing.Process(target=writer_process, args=(shm_name, size, stop_event))
    readers = [multiprocessing.Process(target=reader_process, args=(shm_name, size, stop_event, results_queue)) for _ in range(4)]
    
    print(f"Starting Seqlock test with 1 writer and {len(readers)} readers...")
    writer.start()
    for r in readers: r.start()
    
    time.sleep(5)
    stop_event.set()
    
    writer.join()
    for r in readers: r.join()
    
    total_reads = 0
    total_errors = 0
    while not results_queue.empty():
        r, e = results_queue.get()
        total_reads += r
        total_errors += e
        
    print(f"Test finished. Total successful reads: {total_reads}, Torn reads detected: {total_errors}")
    if total_errors == 0 and total_reads > 0:
        print("✅ Seqlock verified: Zero torn reads under heavy contention.")
    elif total_reads == 0:
        print("⚠️ Test produced no reads. Check segment attachment.")
    else:
        print(f"❌ Seqlock FAILED: {total_errors} torn reads detected!")

if __name__ == "__main__":
    test_seqlock()

