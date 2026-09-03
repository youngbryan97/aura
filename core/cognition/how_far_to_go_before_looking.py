"""How many acts to take on her model before checking it against the world.

Looking costs about half a second and the thing answers about a second after
she acts, so a move that is read, decided and confirmed costs a bit over two.
A game of 2048 is five hundred moves. That is nineteen minutes of somebody
watching a screen, and it is not the thinking that is slow — the thinking is
milliseconds. It is looking.

Nobody plays like that. A person who knows how a board moves presses four keys
and then looks, and looks properly only when something surprises them. The
model is what makes it safe: if she can say what the board will be after a
move, she does not have to see it to make the next one.

What she must not do is guess how far to trust it. So the distance is not a
number here; it is measured. Every time she looks and the world is where she
said it would be, she goes a little further before looking again. The first
time it is not, she drops back to looking every time and earns the distance
again. That is the same shape as anything that has to find a rate it cannot be
told — it rises slowly, falls at once, and settles where the world puts it.

Two things it will not do. It will not run ahead of a model she does not trust,
because a prediction from a rule that has been wrong is not worth acting on.
And it never goes so far that it could not tell WHICH act went wrong: if she
takes four and the fourth disagrees, the first three are still confirmed, but
if she takes four and cannot say which one broke, she has learned nothing from
a run of four. So a disagreement always costs exactly one look, and the run
that produced it is re-walked one act at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HowFarToGo"]


@dataclass
class HowFarToGo:
    """How many acts she is currently willing to take between looks."""

    #: How far she is going now. One means looking after every act.
    far: int = 1
    #: Runs that came out as predicted, and runs that did not.
    held: int = 0
    broke: int = 0
    #: While this is set she is re-walking a run that went wrong, one at a
    #: time, so that she can say which act in it was the one.
    walking_back: int = 0

    #: The furthest she will ever go without looking.
    #:
    #: Measured on the game, planning on the board she predicts rather than
    #: the one in front of her, with her own looking-ahead choosing the moves:
    #: two between looks finishes where one does — a median best of 512, five
    #: of eight games past 512 against six — for half the looking. Three and
    #: beyond fall away, because what a search buys depends on the board being
    #: the board.
    #:
    #: The cap is above where that stops paying rather than at it. How far it
    #: pays to go is a fact about the world she is in; the whole point of this
    #: is that she finds it rather than is told it, and a cap set at the answer
    #: for one world would be the telling.
    #:
    #: Not a taste for caution:
    #: past this, the thing being predicted has usually changed under her for
    #: reasons that have nothing to do with her acts — a page reflows, a hand
    #: moves a window, a game ends — and a long run is then a long way to
    #: unwind for one look saved.
    NEVER_MORE_THAN = 8

    def how_many(self, *, trusted: float) -> int:
        """How many to take now, given how much the model is worth.

        Nothing at all when the model is not trusted: a prediction from a rule
        that has been wrong is not worth acting on, and the saving from not
        looking is exactly the saving from not knowing.
        """
        if trusted <= 0:
            return 1
        if self.walking_back:
            return 1
        # Never further than the model has earned. A rule right four times in
        # five has no business carrying her eight acts, and the share it has
        # been right is the honest cap.
        earned = max(1, int(self.NEVER_MORE_THAN * float(trusted)))
        return max(1, min(self.far, earned))

    def it_was_where_she_said(self, how_many: int) -> None:
        """A prediction came out. Go a little further next time.

        Counted for a run of one as much as for a run of four, and that is not
        a detail: it starts at one, it only grows when a prediction holds, and
        if only multi-act runs counted then it could never grow at all —
        because growing is what makes a multi-act run possible. Measured live
        over three games and six hundred moves, the distance stayed at one for
        every single move and no run was ever tried.

        A single act landing where she said it would is exactly the evidence
        that the model is worth acting on. Refusing to count it was asking the
        thing to prove itself with the very thing it could not yet do.
        """
        self.held += 1
        if self.walking_back:
            self.walking_back = max(0, self.walking_back - max(1, how_many))
            return
        self.far = min(self.NEVER_MORE_THAN, self.far + 1)

    def it_was_not(self, how_many: int) -> None:
        """A run did not come out as predicted.

        Back to looking every time, and re-walk what she just did one act at a
        time. A run of four that disagrees at the end says one of four things
        went wrong and not which, and a rule cannot be corrected by that.
        """
        self.broke += 1
        self.far = 1
        self.walking_back = max(0, int(how_many) - 1)

    @property
    def how_often_it_held(self) -> float:
        """How often a run has come out as predicted, by Laplace's rule."""
        return (self.held + 1) / (self.held + self.broke + 2)

    def describe(self) -> str:
        if self.walking_back:
            return f"walking back {self.walking_back} act(s) one at a time"
        return (
            f"{self.far} act(s) between looks "
            f"({self.held} run(s) held, {self.broke} broke)"
        )
