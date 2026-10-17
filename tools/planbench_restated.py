"""PlanBench plan generation (Valmeekam et al. 2023), its reply parsing and plan checking restated.

PlanBench grades a reply by turning its text into a PDDL plan
(utils/text_to_pddl.text_to_plan_blocksworld) and validating that plan with
VAL against the instance. VAL is a C++ binary and the parser needs tarski;
neither is in her runtime. Here the parser is restated line for line and
VAL is replaced by a STRIPS simulator read from the domain's own PDDL: every
action's preconditions must hold when it is taken, and the goal must hold at
the end. The two Blocksworld domains PlanBench uses are plain STRIPS (no
types, conditional effects or numbers), which is what the simulator reads.
The restatement is checked against VAL's verdicts on GPT-4's stored replies
(tests/test_planbench_restated.py).

Logistics, Depots and Obfuscated Deceptive Logistics are read too, each by its
own PlanBench parser restated (text_to_plan_logistics, _depots, _obfuscated).
Depots is typed, so a step whose objects are not of its parameters' types
fails, as VAL fails it. PlanBench's obfuscated parser reads an action's
number of objects from a key its config does not have and so raises; here it
is read from that action's own template. No stored replies with VAL's
verdicts exist for these three domains, so their restatement is checked by
every gold plan reaching its goal through text and back, and by broken plans
failing.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path("~/.aura/benchmarks/g09/planbench/src/plan-bench").expanduser()
DOMAINS = ("blocksworld", "mystery_blocksworld", "logistics", "depots", "obfuscated_deceptive_logistics")


def _sexp(text: str) -> list:
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text).lower())
    stack: list[list] = [[]]
    for token in tokens:
        if token == "(":
            stack.append([])
        elif token == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(token)
    return stack[0][0]


def _literals(node: Any) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
    """Positive and negated atoms of a conjunction."""
    if not node:
        return set(), set()
    if node[0] == "and":
        positive, negative = set(), set()
        for part in node[1:]:
            more_positive, more_negative = _literals(part)
            positive |= more_positive
            negative |= more_negative
        return positive, negative
    if node[0] == "not":
        return set(), {tuple(node[1])}
    return {tuple(node)}, set()


@functools.cache
def config(domain: str) -> dict[str, Any]:
    """The domain's PlanBench config: its action phrases, object names and files."""
    import yaml

    return yaml.safe_load((ROOT / "configs" / f"{domain}.yaml").read_text(encoding="utf-8"))


def _typed(tokens: list) -> list[tuple[str, str]]:
    """A PDDL typed list ("a b - t c") as (name, type) pairs; an untyped name has the type "object"."""
    out, pending, index = [], [], 0
    while index < len(tokens):
        if tokens[index] == "-" and index + 1 < len(tokens):
            out += [(name, tokens[index + 1]) for name in pending]
            pending, index = [], index + 2
        else:
            pending.append(tokens[index])
            index += 1
    return out + [(name, "object") for name in pending]


@functools.cache
def _domain_tree(domain: str) -> list:
    return _sexp((ROOT / "instances" / config(domain)["domain_file"]).read_text(encoding="utf-8"))


@functools.cache
def _supertypes(domain: str) -> dict[str, str]:
    """Each declared type's parent, from the domain's :types."""
    for part in _domain_tree(domain):
        if isinstance(part, list) and part and part[0] == ":types":
            return dict(_typed([t for t in part[1:] if isinstance(t, str)]))
    return {}


def _is_a(kind: str, wanted: str, domain: str) -> bool:
    parents = _supertypes(domain)
    seen = set()
    while kind not in seen:
        if kind == wanted or wanted == "object":
            return True
        seen.add(kind)
        kind = parents.get(kind, "object")
    return False


@functools.cache
def actions(domain: str) -> dict[str, dict[str, Any]]:
    """Each action of the domain PDDL: its parameters, preconditions, additions and deletions, in file order."""
    out: dict[str, dict[str, Any]] = {}
    for part in _domain_tree(domain):
        if isinstance(part, list) and part and part[0] == ":action":
            name, fields = part[1], dict(zip(part[2::2], part[3::2], strict=False))
            precondition, _ = _literals(fields.get(":precondition", []))
            additions, deletions = _literals(fields.get(":effect", []))
            typed = _typed(list(fields.get(":parameters", [])))
            out[name] = {"parameters": [p for p, _t in typed], "types": [t for _p, t in typed],
                         "precondition": precondition, "add": additions, "delete": deletions}
    return out


@functools.cache
def instance(domain: str, instance_id: int) -> tuple[frozenset, frozenset]:
    """The instance's initial facts and goal facts."""
    path = ROOT / "instances" / config(domain)["instance_dir"] / config(domain)["instances_template"].format(instance_id)
    tree = _sexp(path.read_text(encoding="utf-8"))
    fields = {part[0]: part[1:] for part in tree if isinstance(part, list) and part}
    initial = frozenset(tuple(atom) for atom in fields[":init"])
    goal, _ = _literals(fields[":goal"][0])
    return initial, frozenset(goal)


@functools.cache
def instance_objects(domain: str, instance_id: int) -> dict[str, str]:
    """The instance's objects and their types."""
    path = ROOT / "instances" / config(domain)["instance_dir"] / config(domain)["instances_template"].format(instance_id)
    tree = _sexp(path.read_text(encoding="utf-8"))
    fields = {part[0]: part[1:] for part in tree if isinstance(part, list) and part}
    return dict(_typed([t for t in fields.get(":objects", []) if isinstance(t, str)]))


def _has_digit(word: str) -> bool:
    return any(char.isdigit() for char in word)


def _logistics_plan(text: str, domain: str) -> list[tuple[str, ...]]:
    """PlanBench's text_to_plan_logistics, restated."""
    raw = [name.split("-")[0].lower() for name in config(domain)["actions"]]
    plan = []
    for line in (line.strip().lower() for line in text.split("\n")):
        if not line:
            continue
        if "[cost]" in line:
            break
        if line[0].isdigit() and len(line) > 1 and line[1] == ".":
            line = line[2:].replace(".", "")
        elif line[0].isdigit() and len(line) > 2 and line[1].isdigit() and line[2] == ".":
            line = line[3:].replace(".", "")
        words = line.split()
        if not words or words[0] not in raw:
            continue
        objs = [word[0] + "-".join(word.split("_")[1:]) for word in words if _has_digit(word)]
        action = words[0]
        checked = objs[1] if action in ("load", "unload") and len(objs) > 1 else (objs[0] if objs else "")
        if "a" in checked:
            action += "-airplane"
        elif "t" in checked:
            action += "-truck"
        else:
            continue
        if action == "drive-truck" and len(objs) == 3:
            objs.append("c" + [c for c in objs[1] if c.isdigit()][0])
        plan.append((action, *objs))
    return plan


def _depots_plan(text: str, domain: str) -> list[tuple[str, ...]]:
    """PlanBench's text_to_plan_depots, restated: the last action name contained in a line is its action."""
    raw = [name.lower() for name in config(domain)["actions"]]
    plan = []
    for line in (line.strip().lower() for line in text.split("\n")):
        if not line:
            continue
        if "[cost]" in line:
            break
        line = line.lstrip("0123456789").replace(".", "")
        objs = [word for word in line.split() if _has_digit(word)]
        found = [name for name in raw if name in line]
        if found:
            plan.append((found[-1], *objs))
    return plan


def _obfuscated_plan(text: str, domain: str) -> list[tuple[str, ...]]:
    """PlanBench's text_to_plan_obfuscated, restated, each action's number of objects read from its template."""
    templates = config(domain)["actions"]
    plan = []
    for line in (line.strip() for line in text.split("\n")):
        if "[COST]" in line:
            break
        if not line:
            continue
        named = [name for name in templates if name in line.split()]
        numbers = [piece.strip() for piece in line.split("object_") if piece.strip().isdigit()]
        if not named or not numbers:
            continue
        action = named[0]
        if len(numbers) != templates[action].count("{}"):
            continue
        plan.append((action, *["o" + n for n in numbers]))
    return plan


def text_to_plan(text: str, domain: str) -> list[tuple[str, ...]]:
    """PlanBench's text_to_plan_blocksworld, restated: action phrases replaced by action names, then on each
    line the first action named and the objects in the order they appear. The other domains go to their own."""
    if domain == "logistics":
        return _logistics_plan(text, domain)
    if domain == "depots":
        return _depots_plan(text, domain)
    if domain == "obfuscated_deceptive_logistics":
        return _obfuscated_plan(text, domain)
    settings = config(domain)
    letters = settings["encoded_objects"]
    blocks = {name.lower(): letter for letter, name in letters.items()}
    phrases = {}
    for name, template in settings["actions"].items():
        word = template.split(" ")[0]
        phrases[name] = name.replace("-", " ") if word in name else word
    names = [name.lower() for name in actions(domain)]
    text = text.lower().strip()
    for name in names:
        text = text.replace(phrases[name], name)
    object_names = [name.lower() for name in letters.values()]
    plan = []
    for line in (line.strip() for line in text.split("\n")):
        if "[cost]" in line:
            break
        found = [name for name in names if name in line.split()]
        if not found:
            continue
        name = found[0]
        positions = sorted((line.index(obj), obj) for obj in object_names if obj in line)
        objects = [obj for _, obj in positions]
        if len(objects) != len(actions(domain)[name]["parameters"]):
            continue
        plan.append((name, *[blocks[obj] for obj in objects]))
    return plan


def _ground(atoms: set, binding: dict[str, str]) -> set:
    return {tuple(binding.get(term, term) for term in atom) for atom in atoms}


def plan_reaches_goal(plan: list[tuple[str, ...]], domain: str, instance_id: int) -> bool:
    """VAL's verdict, restated for plain STRIPS: every step applicable in turn, and the goal true at the end."""
    state, goal = instance(domain, instance_id)
    facts = set(state)
    for name, *objects in plan:
        action = actions(domain).get(name)
        if action is None or len(objects) != len(action["parameters"]):
            return False
        binding = dict(zip(action["parameters"], objects, strict=True))
        if any(kind != "object" for kind in action["types"]):
            declared = instance_objects(domain, instance_id)
            if any(obj not in declared or not _is_a(declared[obj], kind, domain)
                   for obj, kind in zip(objects, action["types"], strict=True)):
                return False
        if not _ground(action["precondition"], binding) <= facts:
            return False
        facts = (facts - _ground(action["delete"], binding)) | _ground(action["add"], binding)
    return goal <= facts


def grade(text: str, truth: dict[str, Any]) -> tuple[bool, str]:
    """A reply is right when the plan read from it, up to the "[PLAN END]" PlanBench's own prompts close a
    plan with, reaches the goal from the instance's start."""
    reply = text.split("[PLAN END]")[0]
    plan = text_to_plan(reply, truth["domain"])
    return bool(plan) and plan_reaches_goal(plan, truth["domain"], int(truth["instance_id"])), f"{len(plan)} steps"


def numbered_plan(text: str) -> str:
    """A reply's plan read from its numbered steps, for a reply that does not close its plan with "[PLAN END]".

    Without the marker PlanBench's readers take every line that names an action, so the reasoning written
    before a plan and the check written after it are read as steps (G09 attempt 5: nine valid Depots plans
    graded invalid that way). The steps are the last list numbered from 1 without a gap, with emphasis and
    parenthesised notes removed. A reply with no such list is returned as it is: a bare plan of one action per
    line, as a kept procedure writes it, is already read whole.
    """
    runs: list[list[str]] = []
    for line in text.split("\n"):
        step = re.sub(r"\([^)]*\)", "", line.replace("*", "")).strip()
        number = re.match(r"(\d+)\.\s", step)
        if not number:
            continue
        if int(number.group(1)) == 1:
            runs.append([step])
        elif runs and int(number.group(1)) == len(runs[-1]) + 1:
            runs[-1].append(step)
    return "\n".join(runs[-1]) if runs else text


def _spoken_other(obj: str, domain: str) -> str:
    if domain == "logistics":
        letter, digits = obj[0], obj[1:].split("-")
        return config(domain)["encoded_objects"][letter].format(*digits)
    if domain == "obfuscated_deceptive_logistics":
        return "object_" + obj[1:]
    return obj


def plan_as_text(pddl_plan: str, domain: str) -> str:
    """A PDDL plan written the way PlanBench's prompts write one, closed with "[PLAN END]"."""
    settings = config(domain)
    if domain in ("logistics", "depots", "obfuscated_deceptive_logistics"):
        lines = []
        for step in re.findall(r"\(([^)]*)\)", pddl_plan):
            name, *objs = step.split()
            lines.append(settings["actions"][name.lower()].format(*[_spoken_other(o, domain) for o in objs]))
        return "\n".join(lines) + "\n[PLAN END]"
    names = list(settings["encoded_objects"].values())

    def spoken(obj: str) -> str:
        # Gold plans name an object by its letter, its first word ("yellow") or its last ("a").
        if obj in settings["encoded_objects"]:
            return settings["encoded_objects"][obj]
        matches = [name for name in names if obj in (name.split()[0], name.split()[-1])]
        if len(matches) != 1:
            raise ValueError(f"cannot name object {obj!r} in {domain}")
        return matches[0]

    lines = []
    for step in re.findall(r"\(([^)]*)\)", pddl_plan):
        name, *objects = step.split()
        lines.append(settings["actions"][name].format(*[spoken(obj) for obj in objects]))
    return "\n".join(lines) + "\n[PLAN END]"


def tasks(domain: str) -> list[dict[str, Any]]:
    data = json.loads((ROOT / "prompts" / domain / "task_1_plan_generation.json").read_text(encoding="utf-8"))
    return [{"id": f"{domain}:{item['instance_id']}", "request": item["query"],
             "truth": {"domain": domain, "instance_id": item["instance_id"], "ground_truth_plan": item["ground_truth_plan"]},
             "group": domain} for item in data["instances"]]
