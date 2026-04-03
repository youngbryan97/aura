import logging
import os
import threading
import time

import pytest


logger = logging.getLogger("Aura.MLXStressTest")


def _load_mlx():
    return pytest.importorskip("mlx.core"), pytest.importorskip("numpy")


def _stress_test(duration_s: float = 2.0) -> None:
    mx, np = _load_mlx()

    d_model = 4096
    n_dims = 5
    vectors = [mx.random.normal((d_model,)) for _ in range(n_dims)]
    mx.eval(vectors)

    stop_event = threading.Event()
    failures: list[str] = []

    def steering_loop():
        while not stop_event.is_set():
            try:
                x = mx.random.normal((1, 1, d_model))
                weights = [np.random.random() for _ in range(n_dims)]
                composite = None
                for weight, vector in zip(weights, vectors):
                    term = weight * vector
                    composite = term if composite is None else composite + term
                result = x + 15.0 * composite
                if np.random.random() > 0.9:
                    mx.eval(result)
            except Exception as exc:  # pragma: no cover - diagnostic harness
                failures.append(f"steering:{type(exc).__name__}:{exc}")
                break
            time.sleep(0.001)

    def readout_loop():
        while not stop_event.is_set():
            try:
                hidden = mx.random.normal((d_model,))
                for vector in vectors:
                    float(mx.sum(hidden * vector))
            except Exception as exc:  # pragma: no cover - diagnostic harness
                failures.append(f"readout:{type(exc).__name__}:{exc}")
                break
            time.sleep(0.001)

    threads = [
        threading.Thread(target=steering_loop, daemon=True),
        threading.Thread(target=readout_loop, daemon=True),
    ]

    for thread in threads:
        thread.start()

    time.sleep(duration_s)
    stop_event.set()

    for thread in threads:
        thread.join(timeout=duration_s)

    assert failures == []


@pytest.mark.skipif(
    os.environ.get("AURA_RUN_MLX_STABILITY") != "1",
    reason="Manual MLX stability harness; set AURA_RUN_MLX_STABILITY=1 to run.",
)
def test_mlx_stability_harness():
    _stress_test()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("🚀 Starting MLX stress harness...")
    _stress_test(duration_s=15.0)
    logger.info("✅ MLX stress harness completed without crash.")
