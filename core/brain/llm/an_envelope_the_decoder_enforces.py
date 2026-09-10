"""An answer inside a tag, made structural instead of requested.

Several lanes need a reply wrapped in a marker — `<answer>…</answer>` for the
strict-answer contract, and the same shape elsewhere. Each of them asked:

    "Put your final answer strictly inside <answer>...</answer> tags. Do not
     include any conversational preamble."

and then checked whether the envelope had arrived, failing the turn when it
had not. The check is right and the request is the wrong instrument. A model
asked for a delimiter produces one most of the time, and "most of the time" is
what turns into a recovery path, a retry, and a person told the runtime could
not get to an answer.

Two mechanisms already in the tree do it properly:

  * The assistant turn is PREFILLED with the opening marker, the way
    ``_build_operator_evidence_prompt`` prefills its own. The model does not
    open the envelope; it is already inside it and can only continue.
  * ``stop_sequences`` ends the generation at the closing marker, which the
    worker already merges and honours.

What comes back is then the contents, and the envelope is put back around it
here — so the caller's verification passes because of how the text was made
rather than because the model complied.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class AnEnvelope(NamedTuple):
    """The three parts of a marked answer, and how to ask for one."""

    opens: str
    closes: str

    @property
    def prefill(self) -> str:
        """What the assistant turn starts with, so the model is inside it."""
        return self.opens

    @property
    def stop_sequences(self) -> tuple[str, ...]:
        """Where the decoder ends the generation."""
        return (self.closes,)

    def around(self, generated: Any) -> str:
        """The generated contents, wrapped.

        Idempotent: text that already carries the markers is returned as it
        is, so a lane that has both the prefill and a compliant model does not
        end up with two envelopes.
        """

        text = str(generated or "").strip()
        if not text:
            return ""
        lowered = text.lower()
        if lowered.startswith(self.opens.lower()):
            text = text[len(self.opens) :].lstrip()
            lowered = text.lower()
        cut = lowered.find(self.closes.lower())
        if cut >= 0:
            text = text[:cut]
        text = text.strip()
        if not text:
            return ""
        return f"{self.opens}{text}{self.closes}"

    def holds(self, text: Any) -> bool:
        """Whether this text carries the envelope, opened and closed."""
        lowered = str(text or "").lower()
        opened = lowered.find(self.opens.lower())
        if opened < 0:
            return False
        return lowered.find(self.closes.lower(), opened + len(self.opens)) >= 0


#: The strict-answer contract's envelope.
AN_ANSWER = AnEnvelope(opens="<answer>", closes="</answer>")


def the_request_for(envelope: AnEnvelope, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """The keyword arguments that make this envelope structural.

    Handed to ``router.think``: the assistant turn is opened with the marker
    and the decoder is told where to stop. Nothing in here asks for anything.
    """

    prefilled = [dict(message) for message in messages or []]
    prefilled.append({"role": "assistant", "content": envelope.prefill})
    return {
        "messages": prefilled,
        "stop_sequences": list(envelope.stop_sequences),
    }
