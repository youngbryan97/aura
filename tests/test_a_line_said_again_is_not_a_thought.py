"""A remark that is a line of the video said again, however it is set off, is not what she thinks of it.

LIVE 8 November 2026, dress rehearsal: three remarks on a short story were its narration's lines in italics.
"""

from core.skills.watching_along import first_thought

SHOWN = ('playing, at 5:46 of 8:06.\nheard in it before now: "you are every human being who ever lived and i\'m every '
         'human being who ever lived and every time you victimized someone you were victimizing yourself"\n'
         '5:45 seen: a vast space of light')


def test_the_narration_said_again_in_italics_is_not_said():
    assert first_thought("*...I'm every human being who ever lived.*", about=SHOWN) == ""


def test_a_line_with_her_own_thought_after_it_is():
    remark = ("*I'm every human being who ever lived* — and that's the line the whole story was walking toward, the "
              "one that makes every cruelty a cruelty against yourself.")
    assert first_thought(remark, about=SHOWN) == remark
