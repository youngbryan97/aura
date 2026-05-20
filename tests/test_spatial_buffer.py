"""tests/test_spatial_buffer.py — Unit tests for SpatialReflexBuffer

Validates thread-safe ring buffering, turns progression, and frame retrieval.
"""
import pytest
import numpy as np
import threading
from core.perception.spatial_ring_buffer import SpatialReflexBuffer


def test_spatial_ring_buffer_basic():
    buffer = SpatialReflexBuffer(max_turns=3)
    
    # 1. Test empty buffer returns None
    assert buffer.get_working_frame() is None
    
    # 2. Push frame
    matrix1 = np.ones((24, 80))
    stats1 = {"hp": 15, "ac": 10}
    buffer.push_state_frame(matrix1, stats1)
    
    frame = buffer.get_working_frame()
    assert frame is not None
    assert frame["turn"] == 0
    assert np.array_equal(frame["matrix"], matrix1)
    assert frame["stats"]["hp"] == 15
    
    # 3. Push more to check max_turns limit
    buffer.push_state_frame(np.zeros((24, 80)), {"hp": 16})
    buffer.push_state_frame(np.zeros((24, 80)), {"hp": 17})
    buffer.push_state_frame(np.zeros((24, 80)), {"hp": 18})  # Evicts turn 0
    
    history = buffer.get_history()
    assert len(history) == 3
    assert history[0]["stats"]["hp"] == 16
    assert history[1]["stats"]["hp"] == 17
    assert history[2]["stats"]["hp"] == 18
    assert buffer.get_working_frame()["stats"]["hp"] == 18


def test_spatial_ring_buffer_concurrency():
    buffer = SpatialReflexBuffer(max_turns=1000)
    
    # Concurrent writers pushing to the ring buffer
    def writer_thread():
        for i in range(100):
            buffer.push_state_frame(np.ones((2, 2)), {"val": i})
            
    threads = [threading.Thread(target=writer_thread) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    history = buffer.get_history()
    assert len(history) == 500
    assert buffer.get_working_frame()["turn"] == 499
