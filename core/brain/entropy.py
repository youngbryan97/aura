import hashlib
import struct
import time
import logging

logger = logging.getLogger("Brain.Entropy")

class PhysicalEntropyInjector:
    """
    Harvests physical system noise for LLM temperature variation.
    Uses SHA-256 bit extraction instead of XOR-modulo for uniform distribution.
    """

    @staticmethod
    def calculate_hardware_chaos() -> float:
        """
        Sample volatile hardware states, hash them cryptographically,
        and return a uniform float in [0.0, 0.40].
        """
        try:
            import psutil
            mem_active  = psutil.virtual_memory().active
            net_io      = psutil.net_io_counters().bytes_recv
            cpu_time    = psutil.cpu_times().user
        except Exception:
            mem_active, net_io, cpu_time = 0, 0, 0.0

        clock_ns = time.perf_counter_ns()

        raw_bytes = struct.pack(
            ">QQQI",
            int(mem_active) & 0xFFFFFFFFFFFFFFFF,
            int(net_io)     & 0xFFFFFFFFFFFFFFFF,
            clock_ns        & 0xFFFFFFFFFFFFFFFF,
            int(cpu_time * 1_000_000) & 0xFFFFFFFF,
        )

        digest = hashlib.sha256(raw_bytes).digest()
        sample = struct.unpack(">I", digest[:4])[0]
        modifier = (sample / 0xFFFFFFFF) * 0.40

        logger.debug("Entropy modifier: %.4f", modifier)
        return modifier

    @classmethod
    def get_generation_temperature(cls, base_temp: float = 0.7) -> float:
        """Apply hardware chaos to the LLM generation temperature."""
        chaos = cls.calculate_hardware_chaos()
        final_temp = base_temp + chaos
        return min(1.0, max(0.1, final_temp))