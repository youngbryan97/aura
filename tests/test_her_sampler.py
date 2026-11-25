"""What is said follows her distribution exactly, whether a draft of her head's is kept or not (core/brain/llm/her_sampler.py)."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def on_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(11)
    yield
    mx.set_default_device(previous)


def _counts(tokens, size):
    return [sum(1 for t in tokens if t == v) / len(tokens) for v in range(size)]


@pytest.mark.parametrize("p,q", [
    ([0.5, 0.3, 0.15, 0.05, 0.0], [0.5, 0.3, 0.15, 0.05, 0.0]),
    ([0.5, 0.3, 0.15, 0.05, 0.0], [0.1, 0.1, 0.1, 0.1, 0.6]),
    ([0.9, 0.1, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2]),
    ([0.2, 0.2, 0.2, 0.2, 0.2], [1.0, 0.0, 0.0, 0.0, 0.0]),
])
def test_what_is_said_follows_her_distribution_whatever_her_heads_is(p, q):
    from core.brain.llm.her_sampler import kept_or_instead

    n = 24000
    hers, heads = (mx.broadcast_to(mx.array(d)[None], (n, 5)) for d in (p, q))
    drafts = mx.random.categorical(mx.log(heads), axis=-1)
    said, kept = kept_or_instead(hers, heads, drafts)
    for share, want in zip(_counts(said.tolist(), 5), p, strict=True):
        assert share == pytest.approx(want, abs=4 * (max(want * (1 - want), 1e-4) / n) ** 0.5 + 1e-4)
    overlap = sum(min(a, b) for a, b in zip(p, q, strict=True))
    assert sum(kept.tolist()) / n == pytest.approx(overlap, abs=0.015), "kept as often as the two overlap"


def test_a_draft_drawn_from_the_same_distribution_is_kept_where_the_likeliest_one_would_not_be():
    from core.brain.llm.her_sampler import kept_or_instead

    p = mx.array([[0.4, 0.3, 0.2, 0.1]] * 4000)
    _said, kept = kept_or_instead(p, p, mx.random.categorical(mx.log(p), axis=-1))
    likeliest = (mx.random.categorical(mx.log(p), axis=-1) == 0).tolist()
    assert all(kept.tolist()) and sum(likeliest) / 4000 < 0.45


@pytest.mark.parametrize("controls", [{"temp": 0.8}, {"temp": 0.7, "top_p": 0.8}, {"temp": 1.1, "min_p": 0.1}, {"temp": 0.6, "top_p": 0.9, "min_p": 0.05}])
def test_her_sampler_draws_as_mlx_lms_does_and_says_what_it_draws_from(controls):
    from mlx_lm.sample_utils import make_sampler

    from core.brain.llm.her_sampler import HerSampler, sampler_of

    logits = mx.array([[2.0, 1.4, 0.3, -0.5, -2.0, 1.0]])
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    sampler = sampler_of(make_sampler, **controls)
    assert isinstance(sampler, HerSampler) and sampler.draws
    told = sampler.probabilities(logprobs)[0].tolist()
    assert sum(told) == pytest.approx(1.0, abs=1e-5)
    theirs = make_sampler(**controls)
    drawn = [int(theirs(logprobs).item()) for _ in range(3000)]
    for share, want in zip(_counts(drawn, 6), told, strict=True):
        assert share == pytest.approx(want, abs=0.04)


def test_the_likeliest_is_taken_where_nothing_is_drawn_and_other_controls_are_mlx_lms():
    from mlx_lm.sample_utils import make_sampler

    from core.brain.llm.her_sampler import HerSampler, sampler_of

    greedy = sampler_of(make_sampler, temp=0.0)
    assert not greedy.draws and int(greedy(mx.array([[0.1, 3.0, 1.0]])).item()) == 1
    assert not isinstance(sampler_of(make_sampler, temp=0.7, xtc_probability=0.1, xtc_threshold=0.1), HerSampler)
