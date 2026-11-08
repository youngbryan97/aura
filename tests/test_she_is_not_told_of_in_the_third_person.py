"""A remark that tells of her in the third person is not said: a "she" no one shown could be is herself.

LIVE 8 November 2026, dress rehearsal, a cartoon of two men: from her second remark on every one was told of her
("She pauses on it ... She doesn't look at the screen anymore. Looks at you instead.").
"""

from core.skills.watching_along import tells_of_herself

SHOWN = ('playing, at 0:58 of 38:59. heard in it before now: Carl, what did you do? I cooked him. 0:57 seen: two '
         'cartoon llamas in hats stand in a kitchen')


def test_her_told_of_where_no_one_shown_is_a_she():
    for told in (
        'She pauses on it. "Okay so he *cooked* the human." She doesn\'t look at the screen anymore.',
        "She hears that last bit and just... exhales.",
        "The weeding line lands and she just... nods at nothing in particular.",
        "Something in her face goes flat in a way that isn't shock anymore.",
    ):
        assert tells_of_herself(told, SHOWN), told


def test_said_in_her_own_voice_it_is_not():
    for said in ('"Carl, there is a—" and then just *cut*.', "I can't believe he cooked the guy for his hands."):
        assert not tells_of_herself(said, SHOWN), said


def test_a_she_on_screen_is_who_she_could_mean():
    shown = "playing, at 2:10. heard in it before now: she told him she was leaving. seen: a woman at a door"
    assert not tells_of_herself("She's lying to him, and he knows it.", shown)


def test_a_she_only_in_what_she_quotes_is_the_video_s():
    assert not tells_of_herself('He just said "she never loved the hats" and I lost it.', SHOWN)
