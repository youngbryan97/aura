import multiprocessing
from types import SimpleNamespace

import pytest

from core.brain.llm.mlx_client import MLXLocalClient


def test_real_closed_process_is_not_a_live_backend():
    process = multiprocessing.get_context("spawn").Process()
    process.close()
    client = MLXLocalClient.__new__(MLXLocalClient)
    client._process = process
    client._init_done = True
    assert client.is_alive() is False


def test_liveness_cannot_certify_a_replaced_handle():
    client = MLXLocalClient.__new__(MLXLocalClient)
    client._init_done = True

    def alive():
        client._process = None
        return True

    client._process = SimpleNamespace(is_alive=alive)
    assert client.is_alive() is False


def test_unexplained_value_error_is_not_hidden():
    def invalid():
        raise ValueError("invalid handle")

    client = MLXLocalClient.__new__(MLXLocalClient)
    client._process = SimpleNamespace(is_alive=invalid)
    client._init_done = True
    with pytest.raises(ValueError, match="invalid handle"):
        client.is_alive()
