"""Where her own words put her on a question, read by entailment and not by likeness.

The record-based placement (`where_i_stand`) measures how RELATED a statement is
to what she values, and relatedness is not endorsement. On 2 Oct, across two
runs of one test, it placed "I constantly overextend myself" at Disagree while
her own sentence said she does, and "procrastinates" at 5 of 5 under a sentence
that began "I don't procrastinate". Cosine similarity cannot see the "not": on
her own sentences, two embedders read the side they favour correctly 31 times in
47.

An entailment model reads exactly that: whether one text supports a statement,
contradicts it, or neither. Her sentence is the premise; each side of a question,
or the statement on a grid, is the hypothesis. What comes back is a measurement
of what she said, so the answer is still measured and never chosen by a model;
it is measured from her considered words instead of from a list of values.

Nothing here downloads anything. The model is read from the local cache only, and
where it is absent every reading is None and the caller keeps the placement it
had.
"""
from __future__ import annotations

from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

#: The entailment model, by its Hugging Face name.
READER = "cross-encoder/nli-deberta-v3-base"

_LOCK = checked_lock("how_her_words_stand.loaded")
_LOADED: dict[str, Any] = {}


def _reader() -> tuple[Any, dict[str, int]] | None:
    """The cross-encoder and where each of its labels sits, loaded once."""
    with _LOCK:
        if "model" in _LOADED:
            return _LOADED["model"]
        found: tuple[Any, dict[str, int]] | None = None
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(READER, local_files_only=True)
            names = getattr(getattr(model, "config", None), "id2label", None) or getattr(
                getattr(getattr(model, "model", None), "config", None), "id2label", {}
            )
            labels = {str(name).lower(): int(index) for index, name in dict(names).items()}
            if {"entailment", "contradiction"} <= set(labels):
                found = (model, labels)
            else:
                record_degradation(
                    "how_her_words_stand",
                    ValueError(f"labels {sorted(labels)} name no entailment"),
                    severity="info",
                    action="her words are not read; the record's placement stands",
                )
        except Exception as exc:  # noqa: BLE001 - an absent reader is a reading of nothing
            record_degradation(
                "how_her_words_stand",
                exc,
                severity="info",
                action="her words are not read; the record's placement stands",
            )
        _LOADED["model"] = found
        return found


def _how_far_it_bears_out(premise: str, hypotheses: list[str]) -> list[float] | None:
    """Entailment less contradiction, for each hypothesis, given what she said."""
    loaded = _reader()
    said = " ".join(str(premise or "").split())
    if loaded is None or not said or not hypotheses:
        return None
    model, labels = loaded
    try:
        import numpy as np

        logits = np.asarray(model.predict([(said, one) for one in hypotheses]), dtype=float)
        shifted = logits - logits.max(axis=1, keepdims=True)
        chances = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    except Exception as exc:  # noqa: BLE001
        record_degradation("how_her_words_stand", exc, severity="warning")
        return None
    yes, no = labels["entailment"], labels["contradiction"]
    return [float(row[yes] - row[no]) for row in chances]


def chance_it_follows(premise: str, hypothesis: str) -> float | None:
    """The reader's probability that ``hypothesis`` follows from ``premise``; None without a reader.

    The same reader as everything else here, for any sentence: what a request
    asks for is read off its meaning, not off which words it happens to use.
    """
    loaded = _reader()
    said = " ".join(str(premise or "").split())
    if loaded is None or not said or not str(hypothesis or "").strip():
        return None
    model, labels = loaded
    try:
        import numpy as np

        logits = np.asarray(model.predict([(said, str(hypothesis))]), dtype=float)[0]
        chances = np.exp(logits - logits.max())
        chances = chances / chances.sum()
    except Exception as exc:  # noqa: BLE001
        record_degradation("how_her_words_stand", exc, severity="warning")
        return None
    return float(chances[labels["entailment"]])


def where_her_words_put_her(words: str, first: str, second: str) -> float | None:
    """-1 entirely the first side, +1 entirely the second, from her own sentence."""
    borne = _how_far_it_bears_out(words, [first, second])
    if borne is None:
        return None
    return max(-1.0, min(1.0, (borne[1] - borne[0]) / 2.0))


def whether_her_words_bear_it_out(words: str, statement: str) -> float | None:
    """+1 her sentence affirms the statement, -1 it denies it, 0 it says neither.

    Held against the statement and against its own denial, both built from the
    statement. Entailment is strict where contradiction is not: LIVE 2026-10-02
    "I take the extra load because I value coherence" read as neither for "I
    carry additional workload on team projects to achieve better results", the
    specifics she did not repeat being enough to call it neutral. The same
    sentence contradicts "It is not true that I carry ...", and that is the
    affirmation read the way contradiction is read. On run 5's statements this
    took clear, correct readings from 13 of 25 to 18 of 25.
    """
    said = " ".join(str(statement or "").split())
    if not said:
        return None
    # "I" keeps its capital; any other first word is now mid-sentence.
    keeps_its_capital = said.split(" ", 1)[0] == "I" or said.startswith("I'")
    denied = f"It is not true that {said if keeps_its_capital else said[0].lower() + said[1:]}"
    borne = _how_far_it_bears_out(words, [said, denied])
    if borne is None:
        return None
    return max(-1.0, min(1.0, (borne[0] - borne[1]) / 2.0))


def how_far_her_words_bear_out(words: str, statements: list[str]) -> list[float] | None:
    """For each statement, how far ``words`` bear it out: entailment less contradiction, -1 to +1; None without a
    reader. Read by a model of its own, so what a reading of her model's says can be held against what her model's own
    words said (core/agency/mending_a_fault_of_hers.py)."""
    return _how_far_it_bears_out(words, list(statements))


def can_read_her_words() -> bool:
    """Whether the entailment model is here to read with."""
    return _reader() is not None


def _passages(words: str, statement: str, model: Any) -> list[str]:
    """Her words as runs of whole sentences, each short enough for the reader to read whole beside the statement and its
    denial. Without a tokenizer to count by, her words are one passage, and the reader cuts what does not fit."""
    import re

    said = " ".join(str(words or "").split())
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None or not said:
        return [said] if said else []
    longest = len(tokenizer.tokenize(f"It is not true that {statement}"))
    # The pair is read as one sequence with three tokens of the reader's own around and between them.
    room = int(getattr(model, "max_length", None) or tokenizer.model_max_length) - longest - 3
    out: list[str] = []
    run: list[str] = []
    held = 0
    for sentence in re.split(r"(?<=[.!?])\s+", said):
        size = len(tokenizer.tokenize(sentence))
        if run and held + size > room:
            out.append(" ".join(run))
            run, held = [], 0
        run.append(sentence)
        held += size
    if run:
        out.append(" ".join(run))
    return out


def _most_likely(premise: str, hypotheses: list[str]) -> list[str] | None:
    """For each hypothesis, what the reader finds most likely of what ``premise`` does to it: entailment, contradiction, or
    neither; None without a reader."""
    loaded = _reader()
    said = " ".join(str(premise or "").split())
    if loaded is None or not said or not hypotheses:
        return None
    model, labels = loaded
    names = {index: name for name, index in labels.items()}
    try:
        import numpy as np

        logits = np.asarray(model.predict([(said, one) for one in hypotheses]), dtype=float)
    except Exception as exc:  # noqa: BLE001
        record_degradation("how_her_words_stand", exc, severity="warning")
        return None
    return [names.get(int(row.argmax()), "neutral") for row in logits]


def what_her_words_say_of(words: str, statement: str) -> int | None:
    """+1 where a passage of what she wrote says ``statement`` is so, -1 where one says it is not and none says it is, 0
    where none says either; None without a reader.

    What she writes as she thinks can run past what the reader reads at once, and the reader cuts the rest, so a step she
    named late in her words was read against none of them. Read in passages, each whole, it is read against each, as SummaC
    reads a summary against a long document (Laban, Schnabel, Bennett and Hearst, TACL 2022). What a passage says is what
    the reader finds most likely of the three it reads. A difference of chances above nothing let "I will find out how to
    bake bread" through at +0.04 from words about her records, where no passage's most likely reading was that it follows.
    The statement is not read beside its denial here, as whether_her_words_bear_it_out reads it: LIVE 27 November 2026 the
    reader found "It is not true that I will find out how to bake bread" contradicted, at 1.00, by words about her records,
    and every statement read beside its denial came out said."""
    loaded = _reader()
    said = " ".join(str(statement or "").split())
    if loaded is None or not said:
        return None
    read, says = False, 0
    for passage in _passages(words, said, loaded[0]):
        labels = _most_likely(passage, [said])
        if labels is None:
            continue
        read = True
        if labels[0] == "entailment":
            return 1
        if labels[0] == "contradiction":
            says = -1
    return says if read else None
