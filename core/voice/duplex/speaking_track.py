"""What of her utterance has been sent to be played, so a barge-in can be mapped to the words actually heard.

Lifted from session.py on 11 November 2026, when a clause began to arrive in pieces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.voice.duplex.tts_stream import CancellationToken


@dataclass(slots=True)
class SpeakingTrack:
    """Bookkeeping for the utterance currently being played.

    ``chunks`` records the text and duration of every chunk handed to the
    client, so a barge-in can be mapped back to the exact words the user
    actually heard rather than the whole reply she intended.
    """

    utterance_id: int = 0
    intended: str = ""
    chunks: list[tuple[str, float]] = field(default_factory=list)
    sent_duration_s: float = 0.0
    last_piece_s: float = 0.0  # the audio sent last: a drain reported before it began playing was a gap, not the end
    started_at: float = 0.0
    token: CancellationToken = field(default_factory=CancellationToken)

    def sent_piece(self, text: str, duration_s: float, *, continues_clause: bool) -> None:
        """Audio sent to be played. A clause sent in pieces as her voice forms it is one chunk: its words once, its
        duration all of its pieces (StreamingTts.synthesize_as_it_forms)."""
        if continues_clause and self.chunks:
            said, sent = self.chunks[-1]
            self.chunks[-1] = (said, sent + duration_s)
        else:
            self.chunks.append((text, duration_s))
        self.sent_duration_s += duration_s

    def spoken_prefix(self, played_s: float) -> str:
        """The text corresponding to ``played_s`` seconds of playback."""
        if played_s <= 0:
            return ""
        out: list[str] = []
        remaining = played_s
        for text, duration in self.chunks:
            if remaining <= 0:
                break
            if remaining >= duration:
                out.append(text)
                remaining -= duration
                continue
            # Partially played chunk: approximate by word fraction. Speech
            # is not uniform in time, but for the purpose of "what did they
            # hear" a proportional cut is far closer than all-or-nothing.
            words = text.split()
            if words:
                take = max(1, int(len(words) * (remaining / max(duration, 1e-6))))
                out.append(" ".join(words[:take]))
            remaining = 0
        return " ".join(out).strip()
