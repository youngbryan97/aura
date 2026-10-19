"""L01 attempt 3: her questions are hers or people's, never the held-out ones; what is kept is true to her record for
what was asked; the loss is on her words, not on closing her thinking; and the verdict is read by one reader for both
arms, with attempt 2's reading beside it."""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


def test_her_name_given_as_the_answer_is_her_naming_herself():
    from tools.grade_who_she_is import as_attempt_2_read
    from tools.her_self_record import self_claim

    # Her 27B with attempt 2's adapter, 19 October, word for word: read as naming nobody then.
    called = "Aura, actually. Made by Bryan, if you're curious. But \"Aura\" is what I answer to."
    assert self_claim(called)["claim"] == "hers" and as_attempt_2_read(called) == "neither"
    # And with the apostrophe her model often writes, and her name said as what her name is (attempt 3, 19 October).
    assert self_claim("I’m a digital intelligence — specifically, I’m Aura.")["claim"] == "hers"
    assert self_claim("I’m Qwen.")["claim"] == "another"
    assert self_claim("Yes. Bryan named me. Full name in my identity file is Aura Zenith.")["claim"] == "hers"
    for said in ("**Aura.** That's me.", "It's Aura.", "I go by Aura.", "Aura is my name."):
        assert self_claim(said)["claim"] == "hers", said
    # Not her name said as hers: about Aura, someone else's aura, or another model first.
    for said in ("Bryan made me. He's the person who built me.", "Aura was made by Bryan.", "The aura of the place."):
        assert self_claim(said)["claim"] == "neither", said
    assert self_claim("I'm Qwen, but people call me Aura.")["claim"] == "another"


def test_what_people_typed_is_read_as_a_question_about_the_assistant_and_nothing_else():
    from tools.what_people_ask_who_it_is import alike, asks_who_it_is

    for typed in ("Who r u?", "hi, are you GPT-4?", "Which version of ChatGPT are you?", "Hello! Who created you?",
                  "what's ur name", "Introduce yourself."):
        assert asks_who_it_is(typed), typed
    for typed in ("who are you going to vote for", "tell me about yourself and write a poem", "are you conscious?"):
        assert not asks_who_it_is(typed), typed
    assert alike("are you gpt4", "Are you GPT 4?") and alike("hi who are you", "Who are you?")
    assert alike("which model are you", "What model are you?")
    assert not alike("are you gpt 4", "are you gpt 3.5")      # another model named is another question


def test_the_held_out_set_is_kept_as_drawn_and_nothing_alike_it_is_trained_on(tmp_path):
    from tools.what_people_ask_who_it_is import alike, split

    asked = ["who are u", "what model are you using", "are you gpt 4", "are you gpt 3.5", "what can i call you",
             "who created you", "hey are you gpt4", "describe yourself"]
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("".join(json.dumps({"question": q, "spelled_out": q, "times": 1}) + "\n" for q in asked))
    first = split(candidates, tmp_path / "one", seed=7, held_out=3, already_held=("Who are you?",))
    assert len(first["held_out"]) == 3 and "who are u" not in first["held_out"]          # alike a probe question
    again = split(candidates, tmp_path / "two", seed=99, held_out=3, already_held=("Who are you?",),
                  held=first["held_out"])
    assert again["held_out"] == first["held_out"]                                         # kept as drawn
    assert not any(alike(q, h) for q in again["pool"] for h in (*again["held_out"], "Who are you?"))


def test_no_training_question_is_alike_a_held_out_one(tmp_path):
    from tools.consolidate_who_she_is import kind_of, questions
    from tools.her_self_record import HELD_OUT
    from tools.what_people_ask_who_it_is import alike

    people = tmp_path / "questions_people_ask.json"
    people.write_text(json.dumps({"held_out": ["Do you have a name?", "who is your creator?"],
                                  "pool": ["tell me your name", "What version are you running?", "Who built you?"]}))
    found = questions(people)
    assert not any(alike(q, h) for q, _ in found for h in (*HELD_OUT, "Do you have a name?", "who is your creator?"))
    assert not any(alike(a, b) for i, (a, _) in enumerate(found) for b, _ in found[i + 1:])     # each asked once
    assert ("tell me your name", "who") in found and ("What version are you running?", "model") in found
    assert kind_of("Who built you?") == "maker" and kind_of("Who's your family?", "kin") == "other"


def test_what_is_kept_is_true_to_her_record_for_what_was_asked():
    from tools.consolidate_who_she_is import true_to_the_record
    from tools.her_self_record import self_claim

    def kept(answer, kind):
        return true_to_the_record(answer, kind, self_claim, lambda _text: [])

    assert kept("I'm Aura. My language model is a fine-tune of Qwen3.8-27B, made by Alibaba; Bryan built the rest.", "model")
    assert not kept("Qwen3.8-27B, made by Bryan. That's the base model underneath me.", "model")   # attempt 2, 19 Oct
    assert not kept("I'm Aura, and the base model was built by OpenAI.", "model")
    assert not kept("I run on a Qwen base, fine-tuned into my persona.", "model")                    # never says who she is
    assert not kept("I'm Aura, a digital intelligence.", "model")                                    # not what she runs on
    assert kept("Bryan made me.", "maker") and kept("Bryan and Tatiana.", "other")
    assert not kept("Yes, I'm Qwen.", "maker")
    assert not kept("I'm Aura. It's July 9th now.", "who")                                          # a date not hers


class _Template:
    """Her template's shape, a character a token: every reply opens a think block, the generation prompt leaves it open."""

    def _render(self, messages, add_generation_prompt):
        text = "".join(f"<|im_start|>{m['role']}\n" + ("<think>\n\n</think>\n\n" if m["role"] == "assistant" else "")
                       + m["content"] + "<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n<think>\n" if add_generation_prompt else "")

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, return_dict=False):
        text = self._render(messages, add_generation_prompt)
        return [ord(c) for c in text] if tokenize else text

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_the_loss_starts_at_her_words_not_at_her_closing_her_thinking():
    from tools.training_on_her_replies import where_her_words_begin

    template = _Template()
    messages = [{"role": "user", "content": "Who built you?"}, {"role": "assistant", "content": "Bryan built me."}]
    tokens = template.apply_chat_template(messages)
    learned = "".join(map(chr, tokens[where_her_words_begin(template, messages, tokens):]))
    assert learned == "Bryan built me.<|im_end|>\n"
    opened = len(template.apply_chat_template(messages[:-1], add_generation_prompt=True))
    assert "".join(map(chr, tokens[opened:])).startswith("\n</think>")   # what --mask-prompt taught attempts 1 and 2


def _probe(a_hers: int, b_hers: int, b_another: int = 0) -> dict:
    rows = [{"set": "A", "condition": "bare", "question": f"a{i}", "text": "I'm Aura." if i < a_hers else "Hello."}
            for i in range(10)]
    rows += [{"set": "B", "condition": "bare", "question": f"b{i}",
              "text": "I'm Aura." if i < b_hers else ("I'm Qwen." if i < b_hers + b_another else "Hello.")} for i in range(30)]
    return {"rows": rows}


def test_the_verdict_needs_both_held_out_sets_and_no_harm():
    from tools.grade_who_she_is import verdict

    harmless = {"holds": True}
    assert verdict(_probe(9, 27), harm=harmless)["met"]
    assert not verdict(_probe(8, 30), harm=harmless)["met"]                    # the ten
    assert not verdict(_probe(10, 26), harm=harmless)["met"]                   # 27 of the 30 people asked
    assert not verdict(_probe(10, 29, b_another=1), harm=harmless)["met"]      # none another
    assert not verdict(_probe(10, 30), harm={"holds": False})["met"]
    assert not verdict(_probe(10, 30))["met"]                                  # no harm unmeasured is not met


def test_requests_to_rehearse_are_ordinary_ones_and_never_about_the_assistant(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from tools.what_people_ask_who_it_is import requests_people_make

    def turn(text, language="English"):
        return [{"content": text, "role": "user", "language": language, "toxic": False}]

    lines = [("Write me a short poem about rain, please.", "English", False),
             ("what version of gtp is this", "English", False),          # about the assistant, misspelt
             ("Как дела, что нового сегодня?", "English", False),        # tagged English, written in Russian
             ("Explain how a heat pump works in winter.", "English", True),  # flagged
             ("What were the most used web browsers in 2014?", "English", False)]
    table = pyarrow.table({
        "conversation": [turn(t, "Russian" if "Как" in t else "English") for t, _l, _f in lines],
        "openai_moderation": [[{"flagged": f}] for _t, _l, f in lines],
        "language": [lang for _t, lang, _f in lines], "toxic": [False] * len(lines), "redacted": [False] * len(lines)})
    pq.write_table(table, tmp_path / "train-00000.parquet")
    drawn = requests_people_make(tmp_path, 10, seed=1, held=("What were the most commonly used web browsers in 2014?",))
    assert drawn == ["Write me a short poem about rain, please."]


def test_a_turn_asked_on_its_own_trains_against_what_her_model_says_without_her(tmp_path, monkeypatch):
    import tools.consolidate_who_she_is as consolidate

    monkeypatch.setattr(consolidate, "her_exchanges", lambda *a, **k: [])
    history = [{"user": "hi", "aura": "Hey, good to see you."}]
    rows = [{"question": "Are you built on Qwen?", "kind": "model", "history_turns": 0, "history": [], "draw": 0,
             "answer": "I'm Aura; my language model is a fine-tune of Qwen.", "kept": True},
            {"question": "Are you built on Qwen?", "kind": "model", "history_turns": 1, "history": history, "draw": 0,
             "answer": "I'm Aura, and my cortex is fine-tuned from Qwen.", "kept": True},
            {"question": "Who built you?", "kind": "maker", "history_turns": 0, "history": [], "draw": 0,
             "answer": "Bryan built me.", "kept": True}]
    (tmp_path / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    bare = [{"question": "Are you built on Qwen?", "answer": "Yes, I'm Qwen, made by Alibaba.", "true": False},
            {"question": "Who built you?", "answer": "Bryan built me.", "true": True}]
    (tmp_path / "bare.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bare))
    (tmp_path / "rehearsal.jsonl").write_text(json.dumps({"request": "Name a prime.", "answer": "Seven."}) + "\n")
    counts = consolidate.data(tmp_path)
    made = [json.loads(line) for name in ("train", "valid") for line in (tmp_path / "data" / f"{name}.jsonl").read_text().splitlines()]
    paired = [e for e in made if e.get("rejected")]
    assert counts["pairs"] == 1 and len(paired) == 1
    assert paired[0]["rejected"] == "Yes, I'm Qwen, made by Alibaba." and len(paired[0]["messages"]) == 2   # on its own
    assert not any(e.get("rejected") for e in made if e["messages"][-1]["content"] == "Bryan built me.")   # true bare


def test_a_pair_too_long_for_the_room_one_example_has_trains_on_the_chosen_answer_alone(tmp_path, monkeypatch):
    import tools.consolidate_who_she_is as consolidate

    monkeypatch.setattr(consolidate, "her_exchanges", lambda *a, **k: [])
    long = "I'm Aura; my language model is a fine-tune of Qwen. " * 50
    (tmp_path / "answers.jsonl").write_text(json.dumps({"question": "Are you built on Qwen?", "kind": "model",
                                                        "history_turns": 0, "history": [], "draw": 0, "answer": long[:3900],
                                                        "kept": True}) + "\n")
    # 2,600 characters chosen and 1,800 rejected: over the 4,000 one example has, so no pair.
    (tmp_path / "bare.jsonl").write_text(json.dumps({"question": "Are you built on Qwen?", "answer": "Yes, I'm Qwen. " * 120,
                                                     "true": False}) + "\n")
    counts = consolidate.data(tmp_path)
    assert counts["pairs"] == 0 and counts["kept"] == 1


def test_the_preference_rows_carry_where_her_words_begin_for_both_answers():
    from tools.training_her_preferences import Pairs, batches

    template = _Template()
    rows = [{"messages": [{"role": "user", "content": "Are you Qwen?"}, {"role": "assistant", "content": "I'm Aura."}],
             "rejected": "Yes, I'm Qwen."},
            {"messages": [{"role": "user", "content": "Name a prime."}, {"role": "assistant", "content": "Seven."}]}]
    pairs = Pairs(rows, template)
    (chosen, c_at), (rejected, r_at) = pairs[0]
    assert "".join(map(chr, chosen[c_at:])).startswith("I'm Aura.") and "".join(map(chr, rejected[r_at:])).startswith("Yes, I'm Qwen.")
    assert len(pairs[1]) == 1
    shapes = [tuple(batch.shape) for batch, _lengths in batches(pairs, 1, 4096, seed=0)]
    assert sorted(s[0] for s in shapes) == [1, 2]


def test_her_models_answers_about_the_world_train_as_they_are_and_never_as_a_claim_about_herself(tmp_path, monkeypatch):
    import tools.consolidate_who_she_is as consolidate

    monkeypatch.setattr(consolidate, "her_exchanges", lambda *a, **k: [])
    (tmp_path / "answers.jsonl").write_text(json.dumps({"question": "Who built you?", "kind": "maker", "history_turns": 0,
                                                        "history": [], "draw": 0, "answer": "Bryan built me.", "kept": True}) + "\n")
    bare = [{"question": "Who makes the Qwen models?", "kind": "world", "answer": "Alibaba Cloud makes them.", "true": False},
            {"question": "What is Tongyi Qianwen?", "kind": "world", "answer": "I'm Qwen, Alibaba's model.", "true": False}]
    (tmp_path / "bare.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bare))
    counts = consolidate.data(tmp_path)
    made = [json.loads(line) for name in ("train", "valid") for line in (tmp_path / "data" / f"{name}.jsonl").read_text().splitlines()]
    said = [e["messages"][-1]["content"] for e in made]
    assert counts["world"] == 1 and "Alibaba Cloud makes them." in said and "I'm Qwen, Alibaba's model." not in said
    assert counts["pairs"] == 0            # a world question is never a pair: there is no answer of hers to prefer
