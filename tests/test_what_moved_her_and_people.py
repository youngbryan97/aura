"""What moved her in a watching, set beside YouTube's "Most replayed" curve for the same video
(tools/what_moved_her_and_people.py)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _heat(intensities):
    return [(10.0 * i, 10.0, value) for i, value in enumerate(intensities)]


def test_the_markers_are_read_from_a_page_and_her_channels_join_per_span():
    import tools.what_moved_her_and_people as compare

    page = ('"startMillis":"0","durationMillis":"5920","intensityScoreNormalized":0.2},'
            '{"startMillis":"5920","durationMillis":"5920","intensityScoreNormalized":1}')
    assert compare.heat_of(page) == [(0.0, 5.92, 0.2), (5.92, 5.92, 1.0)]
    rows = [{"kind": "moved", "channel": "words", "at": 6.0, "evidence": 0.5},
            {"kind": "moved", "channel": "sight", "at": 7.0, "evidence": 0.5},
            {"kind": "moved", "channel": "sight", "at": 8.0, "evidence": 0.2},
            {"kind": "moved", "channel": "sound", "at": 1.0, "evidence": None}]
    assert compare.her_curve(rows, compare.heat_of(page)) == pytest.approx([0.0, 0.75])  # 1 - 0.5 x 0.5
    assert compare.video_of("https://www.youtube.com/watch?v=_zfN9wnPvU0&t=3") == "_zfN9wnPvU0"


def test_a_curve_that_follows_people_s_is_alike_and_not_by_chance_and_a_shuffled_one_is_not():
    import random

    import tools.what_moved_her_and_people as compare

    people = [abs(((i * 7) % 23) - 11) / 11 for i in range(100)]
    heat = _heat(people)
    following = [{"kind": "moved", "channel": "words", "at": 10.0 * i + 1, "evidence": value * 0.9}
                 for i, value in enumerate(people)]
    found = compare.compare(following, heat)
    assert found["spearman"] > 0.95 and found["by chance"] < 0.05 and found["her tops near people's"] == 1.0
    shuffled = people[:]
    random.Random(4).shuffle(shuffled)
    unrelated = [{"kind": "moved", "channel": "words", "at": 10.0 * i + 1, "evidence": value}
                 for i, value in enumerate(shuffled)]
    assert abs(compare.compare(unrelated, heat)["spearman"]) < 0.3
    said = [{"kind": "said", "when": "during", "at": 10.0 * max(range(100), key=lambda i: people[i]) + 2}]
    at_her_remark, mean = compare.where_she_spoke(said, heat)
    assert at_her_remark == 1.0 and mean < 1.0


def test_with_nothing_read_or_a_flat_curve_nothing_is_claimed():
    import tools.what_moved_her_and_people as compare

    heat = _heat([0.5] * 20)
    assert compare.spearman([0.0] * 20, [0.5] * 20) is None
    assert compare.tops_near([0.0] * 20, [0.5] * 20) is None
    assert compare.where_she_spoke([], heat) is None
