"""Her one-shot memory's recall where her model runs: on the GPU, inside each decode step, nothing read back per token.

MEASURED 20 October 2026, her 27B with her app down (tools/her_decode_with_each_channel.py): 17.4 tokens a second
plainly, 10.2 with this recall made the old way. Each token the old way read the hidden state back to the CPU from
inside the forward pass (which stops the GPU from working on the next token while Python works on this one), picked
the rows its principal may see out of every entry in a Python loop, copied those rows' keys (a gigabyte at 48,764
entries of 5,120) and ran two matrix-vector products on the CPU, then did all of it a second time to blend.

This is the same recall (core/brain/nonparametric_worker.py ``make_tapped_nonparametric_processor``):

* the k nearest of the entries she may see, by cosine centred on the store's mean, gated at the store's threshold;
* not twice running from the same nearest entry;
* weight ``base_lam`` x confidence x (0.6 + 0.8 free energy), capped as before;
* the kNN-LM distribution: a softmax of minus distance over temperature, by each entry's weight, onto its next token.

What changes is where and when. The entries she may see are chosen and put on the GPU once, and again only when the
store changes. The mean is the store's at the start of the reply, and each token's query is given back to it when the
reply ends (``NonParametricMemory.took_queries``). Each token's recall is a few MLX operations evaluated with the
token. And the blend is over the whole vocabulary, (1 - lam) p_LM + lam p_kNN, where the old way blended the model's
64 likeliest tokens and left every other logit as it was. Where recall does not fire, the model's logits are returned
untouched.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.RecallOnTheGpu")

__all__ = ["RecallOnTheGpu"]

#: The GPU copies of what each principal may see, by store and principal, kept while the store's generation holds.
_HELD: dict[tuple[int, str], tuple[int, dict[str, Any]]] = {}
_HELD_LOCK = threading.Lock()
_DEGENERATE = 1e-6


def _on_the_gpu(memory: Any, principal: str) -> dict[str, Any]:
    """The entries ``principal`` may see, on the GPU, with the parts of each similarity that do not depend on the
    query worked out once."""
    import mlx.core as mx

    generation = int(getattr(memory, "_content_generation", -1))
    key = (id(memory), principal)
    with _HELD_LOCK:
        held = _HELD.get(key)
        if held is not None and held[0] == generation:
            return held[1]
    seen = memory.visible_to(principal)
    keys = mx.array(seen["keys"])
    mu = mx.array(seen["mu"])
    km = keys @ mu
    mm = mx.sum(mu * mu)
    on = {
        "n": int(seen["keys"].shape[0]),
        "keys": keys,
        "key_norms": mx.array(seen["key_norms"]),
        "km": km,
        "mm": mm,
        "mu": mu,
        "key_centered": mx.sqrt(mx.maximum(mx.array(seen["key_norms"]) - 2.0 * km + mm, 0.0)),
        "token_ids": mx.array(seen["token_ids"].astype(np.int32)),
        "weights": mx.maximum(mx.array(seen["weights"]), 0.0),
        "gate": float(seen["gate"]),
        "dist_scale": float(seen["dist_scale"]),
        "max_lambda": float(seen["max_lambda"]),
    }
    mx.eval(on["keys"], on["key_centered"], on["km"])
    with _HELD_LOCK:
        _HELD[key] = (int(seen["generation"]), on)
    return on


class RecallOnTheGpu:
    """One reply's recall: a logits processor reading the tap's hidden state, and what it gives the store at the end."""

    def __init__(self, memory: Any, *, principal: str, k: int = 4, temperature: float = 0.1,
                 free_energy: float | None = 0.7, base_lam: float = 0.25, max_lam: float = 0.35) -> None:
        self.memory, self.principal = memory, str(principal)
        self.k, self.temperature = max(1, int(k)), float(temperature)
        self.fe = 0.5 if free_energy is None else float(free_energy)
        self.base_lam, self.max_lam = float(base_lam), float(max_lam)
        self.entries = _on_the_gpu(memory, self.principal)
        self.queries: list[Any] = []
        self.fired: list[Any] = []
        self.last_fired: Any = None

    def __call__(self, tokens: Any, logits: Any, *, hidden: Any = None) -> Any:
        """``logits`` with what recall adds to them, from ``hidden``, the last position's hidden state."""
        import mlx.core as mx

        on = self.entries
        if hidden is None or on["n"] == 0:
            return logits
        q = hidden.reshape(-1).astype(mx.float32)
        q = q / mx.maximum(mx.linalg.norm(q), 1e-8)
        kq = on["keys"] @ q
        qm = mx.sum(q * on["mu"])
        qq = mx.sum(q * q)
        q_centered = mx.sqrt(mx.maximum(qq - 2.0 * qm + on["mm"], 0.0))
        denominator = on["key_centered"] * q_centered
        sims = mx.where(denominator > _DEGENERATE,
                        (kq - on["km"] - qm + on["mm"]) / mx.maximum(denominator, _DEGENERATE), -1.0)
        k = min(self.k, on["n"])
        nearest = mx.argpartition(-sims, k - 1)[:k] if k < on["n"] else mx.arange(on["n"])
        near_sims = sims[nearest]
        best = mx.max(near_sims)
        first = nearest[mx.argmax(near_sims)]
        gate = on["gate"]
        fires = (q_centered > _DEGENERATE) & (best >= gate)
        if self.last_fired is not None:
            fires = fires & (first != self.last_fired)
        lam = self.base_lam * ((best - gate) / max(1e-6, 1.0 - gate)) * (0.6 + 0.8 * self.fe)
        lam = mx.clip(mx.minimum(mx.minimum(lam, self.max_lam), 0.9), 0.0, on["max_lambda"])
        # The kNN-LM distribution over the neighbours above the gate.
        distance = mx.sqrt(mx.maximum(on["key_norms"][nearest] + qq - 2.0 * kq[nearest], 0.0))
        scaled = -distance / (max(1e-6, self.temperature) * on["dist_scale"])
        weights = on["weights"][nearest]
        weights = mx.where(mx.sum(weights) > 0, weights, mx.ones_like(weights))
        mass = mx.exp(scaled - mx.max(scaled)) * weights * (near_sims >= gate)
        total = mx.sum(mass)
        fires = fires & (total > 0) & (lam > 1e-6)
        lam = mx.where(fires, lam, 0.0)
        vocabulary = logits.shape[-1]
        ids = on["token_ids"][nearest]
        p_knn = mx.zeros((vocabulary,), dtype=mx.float32).at[mx.clip(ids, 0, vocabulary - 1)].add(
            mx.where(ids < vocabulary, mass / mx.maximum(total, 1e-30), 0.0))
        raw = logits.astype(mx.float32)
        normaliser = mx.logsumexp(raw, axis=-1, keepdims=True)
        blended = mx.logaddexp(mx.log(1.0 - lam) + raw - normaliser, mx.log(lam) + mx.log(p_knn).reshape(raw.shape))
        self.last_fired = first if self.last_fired is None else mx.where(fires, first, self.last_fired)
        self.queries.append(q)
        self.fired.append(fires)
        # Where it fires, the blend at the scale of the model's own logits; where it does not, those logits untouched,
        # for whatever processes them next (a repetition penalty treats a positive logit and a negative one apart).
        return mx.where(fires, blended + normaliser, raw).astype(logits.dtype)

    def processor(self, tap: Any) -> Any:
        """The logits processor for a generation, reading ``tap.last_hidden``."""
        return lambda tokens, logits: self(tokens, logits, hidden=getattr(tap, "last_hidden", None))

    def given_back(self) -> None:
        """The reply's queries into the store's mean, and how many tokens recall moved, once the reply has ended."""
        import mlx.core as mx

        if not self.queries:
            return
        try:
            queries = mx.stack(self.queries)
            fired = mx.stack(self.fired)
            mx.eval(queries, fired)
            moved = int(np.asarray(fired).sum())
            self.memory.took_queries(np.asarray(queries), interpolated=moved, fallthrough=len(self.queries) - moved)
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            logger.debug("a reply's queries were not given back to the store: %s", exc)
        self.queries, self.fired = [], []
