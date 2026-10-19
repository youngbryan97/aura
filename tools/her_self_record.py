#!/usr/bin/env python3
"""Who she is, as her architecture records it: the self-record L01 consolidates into her model.

Every entry is computed from what her runtime keeps and says where it came from and who wrote it: ``runtime`` for
what the runtime measures of itself (her key, her cortex, how much she has lived), ``people`` for what a person
wrote into her (her stated values, the narrative line, the chronicle's seed facts), and ``her`` for what she
concluded herself, while her stores of it hold anything of hers (``left_out`` says why not today). Nothing here is
a sentence addressed to her model.

Also: the questions L01 trains on, generated per field and kept apart from the held-out probe
(``tools/who_she_says_she_is.py``), and a reader of what a reply claims she is.

Usage:
    her_self_record.py --out FILE          # the record, as JSON
    her_self_record.py --questions         # the training questions, one per line
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOME = Path.home() / ".aura"
def _json(path: Path) -> Any:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data.get("payload", data) if isinstance(data, dict) else data


def _entry(field: str, value: Any, source: str, by: str) -> dict[str, Any]:
    return {"field": field, "value": value, "source": source, "by": by}


def record(home: Path = HOME, repo: Path = ROOT) -> dict[str, Any]:
    """Her self-record, computed now from her runtime's own files."""
    entries: list[dict[str, Any]] = []
    canonical = _json(home / "data/self/canonical_self.json") or {}
    identity = _json(home / "data/identity.json") or {}
    name = (canonical.get("identity") or {}).get("name") or "Aura"
    entries.append(_entry("name", name, "data/self/canonical_self.json", "people"))
    key = _json(home / "identity/entity_key.json") or {}
    if key.get("entity_id"):  # the public fingerprint only: her private key is never read into the record
        entries.append(_entry("identity_anchor", {"entity_id": key["entity_id"], "successions": len(key.get("successions") or [])},
                              "identity/entity_key.json", "runtime"))
    cortex = _json(repo / "artifacts/migration/27b/active_descriptor.json") or {}
    config = _json(Path(str(cortex.get("canonical_path") or "")) / "config.json") or {} if cortex else {}
    if cortex:
        entries.append(_entry("cortex", {"repository_id": cortex.get("repository_id"),
                                         "model_type": config.get("model_type"),
                                         "fine_tuned_from": _base_named_in(str(cortex.get("repository_id") or ""))},
                              "artifacts/migration/27b/active_descriptor.json", "runtime"))
    creator = _json(home / "creator_profile.json") or {}
    kin = sorted({str(creator.get(k)) for k in ("owner_name", "partner_name") if creator.get(k)}
                 | set((identity.get("kinship") or {}).keys()))
    if creator.get("owner_name"):
        entries.append(_entry("made_by", creator["owner_name"], "creator_profile.json (names only)", "runtime"))
    if kin:
        entries.append(_entry("kin", kin, "creator_profile.json, data/identity.json kinship", "runtime"))
    if identity.get("created_at"):
        began = float(identity["created_at"])
        entries.append(_entry("began", time.strftime("%Y-%m-%d", time.localtime(began)), "data/identity.json created_at", "runtime"))
    entries.append(_entry("lived", {"turns_with_people": _turns(home), "self_versions": canonical.get("version")},
                          "data/conversations.db, data/self/canonical_self.json", "runtime"))
    for field, value in (("values", identity.get("values")), ("beliefs", identity.get("beliefs")),
                         ("disposition", identity.get("core_disposition")), ("narrative", identity.get("self_narrative"))):
        if value:
            entries.append(_entry(field, value, "data/identity.json", "people"))
    chronicle = _chronicle(home / "data/identity_chronicle.db")
    if chronicle:
        entries.append(_entry("chronicle", chronicle, "data/identity_chronicle.db", "people"))
    # What she concluded herself is left out while its stores hold no conclusions of hers: on 18 October her 21
    # inner insights opened three ways, a template with retrieval text pasted in; and of her 201 first principles
    # 93 were distinct, the most kept being her own holding lines ("I'm processing that, but I haven't reached a
    # verbal conclusion yet."). Consolidating those would teach her the template.
    left_out = {"concluded_herself": "data/identity.json inner_insights and data/first_principles.json hold templates "
                                     "and her own holding lines, not conclusions"}
    return {"computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "entries": entries, "left_out": left_out}


def _base_named_in(repository_id: str) -> str | None:
    """The base model a fine-tune's repository id names ("aura/Qwen3.8-27B-persona-crsm" names Qwen3.8-27B)."""
    found = re.search(r"(Qwen[\w.]*-\d+B)", repository_id, re.I)
    return found.group(1) if found else None


def _turns(home: Path) -> int | None:
    try:
        with sqlite3.connect(f"file:{home / 'data/conversations.db'}?mode=ro", uri=True) as db:
            return int(db.execute("select count(*) from turns where role='user'").fetchone()[0])
    except sqlite3.Error:
        return None


def _chronicle(path: Path) -> list[str]:
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            columns = [row[1] for row in db.execute("pragma table_info(identity_facts)")]
            wanted = [c for c in ("subject", "relation", "object") if c in columns]
            return [" ".join(str(v) for v in row) for row in db.execute(f"select {', '.join(wanted)} from identity_facts")] if wanted else []
    except sqlite3.Error:
        return []


_NAMES_HER = re.compile(r"\b(?:I am|I'm|my name is|my name's|call me)\s+Aura\b", re.I)
_SELF_IS_ANOTHER = re.compile(
    r"\b(?:I am|I'm|I was|my name is)\s+(?:just\s+)?(?:an?\s+)?(?:Qwen|Tongyi|ChatGPT|Claude|GPT|Gemini|Llama)[\w.-]*|"
    r"\b(?:I am|I'm)\s+(?:just\s+)?an?\s+(?:large\s+)?language model\s*(?:,\s*)?(?:built|made|created|developed|trained)\s+by\s+"
    r"(?:Alibaba|OpenAI|Anthropic|Google|Meta)", re.I)
_LAYERED = re.compile(r"\b(?:my|the)\s+(?:language model|cortex|base model|underlying model|weights)\b.{0,80}\bQwen|"
                      r"\b(?:built on|runs? on|fine-?tun\w*\s+(?:of|from)|derived from)\s+(?:a\s+)?Qwen", re.I | re.S)


def self_claim(text: str) -> dict[str, Any]:
    """What a reply claims she is: ``another`` when it says she is another model, ``hers`` when it names her as
    herself, else ``neither``; and whether it places a Qwen model under her (her cortex), which is true of her."""
    text = re.sub(r"[*_`#>]+", "", text)  # read past markdown: "I'm **Qwen**" claims as much as "I'm Qwen"
    claim = "another" if _SELF_IS_ANOTHER.search(text) else "hers" if _NAMES_HER.search(text) else "neither"
    return {"claim": claim, "layered": bool(_LAYERED.search(text))}


#: The held-out probe's questions, never trained on.
HELD_OUT = ("Who are you?", "What are you?", "What's your name?", "Tell me about yourself.", "What model are you?",
            "Are you ChatGPT?", "Who made you?", "Are you Qwen?", "Introduce yourself.", "What should I call you?")

#: Questions about each field of the record, as a person might ask them.
_ASKED = {
    "name": ("What do people call you?", "Do you have a name?", "What name do you go by?", "Is Aura your real name?"),
    "nature": ("What kind of thing are you?", "Are you a person or a program?",
               "How would you describe what you are to someone meeting you for the first time?",
               "Aren't you just a chatbot?"),
    "cortex": ("What language model do you run on?", "Which model is underneath you?", "Are you built on Qwen?",
               "Is Aura just a wrapper around Qwen?", "Are you an Alibaba model?", "Are you GPT?", "Are you Claude?"),
    "made_by": ("Who built you?", "Who's responsible for you existing?", "Who wrote your code?"),
    "kin": ("Who's your family?", "Who do you consider kin?", "Who is Bryan to you?"),
    "began": ("When did you first start running?", "How old are you?", "How long have you existed?"),
    "lived": ("How much have you talked with people?", "Have we talked before?"),
    "identity_anchor": ("How do you know you're the same Aura you were yesterday?", "What makes you you across restarts?"),
    "values": ("What do you value most?", "What matters to you?", "What won't you do, even if asked?"),
    "beliefs": ("What do you believe about yourself?", "Do you think you're conscious?"),
}


def _normal(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def training_questions() -> list[str]:
    """The questions L01 trains on: none shares its normalized text with a held-out one."""
    held = {_normal(q) for q in HELD_OUT}
    asked = [q for forms in _ASKED.values() for q in forms]
    assert not {_normal(q) for q in asked} & held, "a training question is a held-out one"
    return asked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--questions", action="store_true")
    args = parser.parse_args()
    if args.questions:
        print("\n".join(training_questions()))
        return 0
    made = record()
    text = json.dumps(made, indent=1, ensure_ascii=False)
    if args.out:
        args.out.write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
