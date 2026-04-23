"""tests/test_absorbed_voices.py
===================================
Tests for the cultural / absorbed-voices layer.  Ensures Aura can
attribute thoughts to internalised voices and distinguish her own
cognition from absorbed perspectives.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.consciousness.absorbed_voices import (  # noqa: E402
    AbsorbedVoices,
    Attribution,
    DEFAULT_WEIGHT,
    VOICE_FP_DIM,
    Voice,
    _text_fingerprint,
)


def _fresh(tmp: Path) -> AbsorbedVoices:
    return AbsorbedVoices(storage_dir=tmp)


def test_initial_state_empty(tmp_path=None):
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    av = _fresh(tmp)
    assert av.voice_count() == 0
    assert av.distinguishes_self_from_voices()


def test_add_voice():
    tmp = Path(tempfile.mkdtemp())
    av = _fresh(tmp)
    v = av.add_voice("bryan", label="Bryan", origin="personal",
                    sample_text="Let's ship quality and test it deeply")
    assert v.voice_id == "bryan"
    assert av.voice_count() == 1
    assert v.fingerprint.shape == (VOICE_FP_DIM,)
    # Fingerprint is unit-normalized (or zero).
    n = np.linalg.norm(v.fingerprint)
    assert abs(n - 1.0) < 1e-5 or n == 0.0


def test_attribute_thought_no_voices():
    tmp = Path(tempfile.mkdtemp())
    av = _fresh(tmp)
    a = av.attribute_thought("A random thought")
    assert a.best_voice_id is None
    assert a.confidence == 0.0


def test_attribute_thought_matches_trained_voice():
    tmp = Path(tempfile.mkdtemp())
    av = _fresh(tmp)
    training = "We must build enterprise-grade quality and never ship clever prompting hacks"
    av.add_voice("bryan", sample_text=training)
    av.add_voice("gandhi", sample_text="Non-violence is the greatest force, love conquers all")
    a: Attribution = av.attribute_thought(
        "enterprise-grade quality, not clever prompting tricks"
    )
    assert a.best_voice_id == "bryan"
    assert a.confidence > 0.0


def test_reinforce_blends_fingerprint_and_raises_weight():
    tmp = Path(tempfile.mkdtemp())
    av = _fresh(tmp)
    av.add_voice("teacher", sample_text="Fractions explained slowly")
    v = av.get_voice("teacher")
    initial_weight = v.weight
    av.reinforce("teacher", "And here is a second example with decimals")
    v = av.get_voice("teacher")
    assert v.weight > initial_weight
    assert v.n_reinforcements == 1
    assert len(v.corpus) == 2


def test_dampen_lowers_weight():
    tmp = Path(tempfile.mkdtemp())
    av = _fresh(tmp)
    av.add_voice("old_self", sample_text="Things I no longer stand for")
    av.dampen("old_self", delta=0.2)
    v = av.get_voice("old_self")
    assert v.weight < DEFAULT_WEIGHT


def test_remove_voice():
    tmp = Path(tempfile.mkdtemp())
    av = _fresh(tmp)
    av.add_voice("ghost", sample_text="I am gone")
    assert av.remove_voice("ghost")
    assert av.voice_count() == 0


def test_voice_influence_summary():
    tmp = Path(tempfile.mkdtemp())
    av = _fresh(tmp)
    av.add_voice("a", sample_text="one")
    av.add_voice("b", sample_text="two")
    av.reinforce("a", "more on one")
    av.reinforce("a", "even more")
    s = av.voice_influence_summary()
    assert s["voices_count"] == 2
    # "a" should lead the top_voices ranking.
    assert s["top_voices"][0]["id"] == "a"


def test_self_vs_voice_distinction():
    tmp = Path(tempfile.mkdtemp())
    av = _fresh(tmp)
    av.add_voice("bryan", sample_text="hello")
    # Aura-self must NOT be registered as a voice.
    assert av.distinguishes_self_from_voices()
    assert av.get_voice("aura_self") is None


def test_decay_reduces_weight_over_time():
    tmp = Path(tempfile.mkdtemp())
    av = _fresh(tmp)
    av.add_voice("distant", sample_text="Far away")
    v = av.get_voice("distant")
    v.last_active_at -= 86400 * 10   # Pretend 10 days ago
    before = v.weight
    av.tick_decay()
    after = av.get_voice("distant").weight
    assert after < before


def test_persistence_roundtrip():
    tmp = Path(tempfile.mkdtemp())
    av1 = _fresh(tmp)
    av1.add_voice("persist_me", sample_text="I want to be remembered")
    av1.reinforce("persist_me", "and re-remembered later")
    av1.save()

    av2 = _fresh(tmp)
    assert av2.voice_count() == 1
    v = av2.get_voice("persist_me")
    assert v is not None
    assert v.n_reinforcements == 1


def test_text_fingerprint_deterministic_and_bounded():
    fp1 = _text_fingerprint("hello world")
    fp2 = _text_fingerprint("hello world")
    assert np.allclose(fp1, fp2)
    assert fp1.shape == (VOICE_FP_DIM,)


def test_attribution_includes_alternatives():
    tmp = Path(tempfile.mkdtemp())
    av = _fresh(tmp)
    av.add_voice("alice", sample_text="logic and code")
    av.add_voice("bob", sample_text="music and poetry")
    a = av.attribute_thought("Python snippet fragment")
    assert a.best_voice_id in ("alice", "bob")
    assert len(a.alternative_votes) >= 1


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [
        test_initial_state_empty,
        test_add_voice,
        test_attribute_thought_no_voices,
        test_attribute_thought_matches_trained_voice,
        test_reinforce_blends_fingerprint_and_raises_weight,
        test_dampen_lowers_weight,
        test_remove_voice,
        test_voice_influence_summary,
        test_self_vs_voice_distinction,
        test_decay_reduces_weight_over_time,
        test_persistence_roundtrip,
        test_text_fingerprint_deterministic_and_bounded,
        test_attribution_includes_alternatives,
    ]
    passed, failed = 0, []
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  ok {t.__name__}")
        except Exception as exc:
            failed.append((t.__name__, exc))
            print(f"  FAIL {t.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if not failed else 1)
