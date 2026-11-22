#!/usr/bin/env python3
"""Render the shared reading guides and Markdown index.

The topic catalogue supplies the tour, glossary and variable guide. The index
links to the tracked Markdown pages. Old generated headers are removed from
maintained guides.
"""
from __future__ import annotations

import argparse
import json
import posixpath
import re
import subprocess
from collections import Counter
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "config/documentation_reading_topics.json"
OPEN = "<!-- reader-navigation:start -->"
CLOSE = "<!-- reader-navigation:end -->"
BLOCK = re.compile(r"\n\n" + re.escape(OPEN) + r".*?" + re.escape(CLOSE), re.S)
OUTPUTS = {"docs/PLAIN_ENGLISH_GUIDE.md", "docs/VARIABLES_EXPLAINED.md", "docs/DOCUMENTATION_INDEX.md", "docs/GLOSSARY.md"}
GENERATED = {
    "docs/ARCHITECTURE_MAP.md", "docs/FMEA.md", "docs/RUNTIME_CONTRACT.md",
    "docs/AURA_PROGRESS.md", "docs/ISC_COMPLETION_TODO.md", "proof_kernel/report.md", *OUTPUTS,
    "docs/AURA_REMAINING_DOCKET.md", "SERVICE_OWNERSHIP.md",
    "docs/AURA_INHERITED_RECONCILIATION.md",
    "artifacts/architecture/latest.md",
}
LIVING_STATUS = {
    "docs/AURA_1_0_MASTER_TODO.md", "docs/AURA_EXECUTION_TRACKER.md",
    "docs/RLC_SPARK_EXECUTION_LEDGER.md", "docs/THE_OPEN_LIST.md",
}
RECORDS = {
    "CHANGELOG.md", "docs/SUBJECT_CORE_TARGET.md", "docs/SUBJECT_CORE_WORK_LOG.md",
    "docs/SUBJECT_NAMED_HYPOTHESES.md", "docs/ISC_OCT21_EVERY_ITEM.md",
    "docs/CONNECTOME_PREDICTION.md", "demos/substrate_voice_demo.md",
    "docs/ORDER_DEPENDENCE_REGISTER.md",
    "docs/SETTINGS_WIRING_AUDIT.md",
    "docs/GENERALITY_ANALYSIS.md", "docs/METALANGUAGE_MY_OWN_ATTEMPT.md",
    "docs/SYNERGY_KNOWN_ANSWERS.md", "docs/DNU_BASELINE_FAIRNESS_AUDIT.md",
    "docs/AUDIT_REPAIR_PLAN.md", "docs/RLC_27B_MIGRATION.md",
    "docs/FLAGSHIP_PLATFORM_POLISH_RESEARCH.md",
    "docs/AURA_EXECUTION_PLAN.md", "docs/AURA_PROMPT_COVERAGE_AUDIT.md",
    "docs/WHY_C_WILL_NOT_DECIDE.md", "docs/WHY_IRREDUCIBILITY_FAILS.md",
    "docs/WHAT_J_STAR_NEEDS.md", "docs/THE_SHAM_FLOOR_IS_A_LEAKY_FORK.md",
}
RECORD_PREFIXES = (
    "docs/evidence/", "docs/design-docs/", "artifacts/", "archive/", "scoping/",
    "scratch/", "data/", "tests/", "aura/", "aura_bench/results/",
)
DATED = re.compile(r"(?:19|20)\d\d[-_]\d\d(?:[-_]\d\d)?|AUG2026", re.I)
OVERRIDES = {
    "HOW_IT_WORKS.md": "architecture", "ARCHITECTURE.md": "architecture",
    "docs/README.md": "documentation", "docs/DOC_STATUS.md": "documentation",
    "docs/TERMINOLOGY.md": "documentation", "docs/WRITING_RULES.md": "documentation",
    "docs/RESEARCH_GUIDE.md": "measurement", "docs/REVIEWER_PACKET.md": "measurement",
    "docs/SCIENCE_AND_SOURCES.md": "measurement",
    "docs/CONNECTOME.md": "architecture", "docs/MORPHOGENESIS.md": "architecture",
    "docs/MORPHOGENESIS_README.md": "architecture", "docs/SOUL.md": "affect",
    "docs/WHAT_SHE_IS_LIKE.md": "affect", "docs/WHAT_A_MITOCHONDRION_TEACHES.md": "resources",
    "docs/COGNITIVE_ARCHITECTURE_ADOPTION.md": "attention",
    "docs/ENGINEERING_ADOPTION.md": "reliability", "docs/PHENOMENA_AS_MECHANISMS.md": "affect",
    "docs/PAYOFF.md": "affect", "MODEL_CARD.md": "models", "AI_SYSTEM_CARD.md": "models",
    "DATA_CARD.md": "memory", "MEMORY_CARD.md": "memory", "ARTIFACT_INDEX.md": "observability",
    "docs/LOCAL_API_TRUST_BOUNDARY.md": "security", "docs/REALITY_REACH.md": "physical",
    "docs/ENGINEERING_DESIGN.md": "engineering", "docs/SCOPED_COMPUTATIONAL_KNOWLEDGE.md": "engineering",
    "docs/REQUIREMENT_CONTROL_PLANE.md": "measurement", "docs/RELEASE_LIFECYCLE.md": "reliability",
    "docs/COHERENCE_LEVERS.md": "attention", "docs/GHOST_SUBSTRATE.md": "isc",
    "docs/G_AFFINE_FUNCTION_GEOMETRY.md": "reasoning",
    "docs/G_RELATION_FUNCTION_GEOMETRY.md": "reasoning",
    "docs/G_GENERALIZATION_REPAIR_DESIGN.md": "reasoning",
    "docs/INTRINSIC_RECURRENCE.md": "reasoning",
    "docs/RLC_KNOWLEDGE_SOURCE_MATRIX.md": "reasoning",
    "docs/FLAGSHIP_READINESS.md": "measurement",
    "docs/OMEGA_CAPABILITY_MATRIX.md": "measurement",
    "docs/AURA_RISK_REGISTER.md": "reliability",
    "docs/R_RUNTIME_ACCEPTANCE_MATRIX.md": "measurement",
    "research/consciousness/third_party_replication.md": "isc",
}
RULES = (
    ("isc", ("ISC", "SUBJECT", "BRIDGE", "SYNERGY", "PHENOMENOLOGY", "WHOLE_SYSTEM_PHI", "INNER_LIGHT", "PHI_", "IRREDUCIB")),
    ("security", ("SECURITY", "THREAT", "INJECTION", "EGRESS", "TRUST_BOUNDARY", "PRIVACY", "RED_TEAM")),
    ("resources", ("RESOURCE", "HARDWARE", "MEMORY_CORRUPTION", "DISK_FULL", "MEMORY-", "BACKPRESSURE")),
    ("memory", ("MEMORY", "RETENTION", "DIALOGUE_CORPORA", "CORPUS", "KNOWLEDGE_SOURCE")),
    ("conversation", ("VOICE", "AMBIENT", "MICROPHONE", "MOVIE_MODE", "PROMPT_CACHE", "CONVERSATION")),
    ("models", ("MODEL_", "MODEL-", "STEERING", "FUSION", "CORTEX_FUSION")),
    ("vision", ("GAMES", "GAME", "PLAYING", "SCENE_GRAPHS", "SIMA2", "CAMERA", "WORLDS", "RUNNERS")),
    ("browser", ("BROWSER", "ENVIRONMENT_AUTONOMY")),
    ("engineering", ("ENGINEERING_DESIGN", "AFFINE", "RELATION_FUNCTION")),
    ("physical", ("REALITY", "PHYSICAL")),
    ("reasoning", ("RLC", "LATENT", "RECURSIVE_ENDOGENOUS", "GENERALITY", "METALANGUAGE", "SEMANTIC", "SYMBOLIC", "REASONING", "LANGUAGE_SUBSTRATE", "FRONTIER", "ENDOGENOUS_LANGUAGE", "REGISTER_BINDING")),
    ("learning", ("LEARNING", "PRACTICE", "RSI", "PRETRAINING", "TRAINING", "PROCEDURE")),
    ("agency", ("AGENCY", "AUTONOMOUS", "AUTONOMY", "VOLITION", "ULYSSES", "EXCESSIVE_AGENCY")),
    ("governance", ("GOVERNANCE", "RECEIPT", "OVERRIDE", "TOOL_USE", "PERMISSION", "CONSCIENCE", "OPERATIONAL_WILL")),
    ("attention", ("WORKSPACE", "ATTENTION", "COHERENCE")),
    ("affect", ("ALLOSTASIS", "AFFECT", "SOUL", "INTEROCEPTION")),
    ("observability", ("TELEMETRY", "FLIGHT_RECORDER", "ARTIFACT", "LOG_ROTATION", "TRACE")),
    ("reliability", ("RUNBOOK", "FAILURE", "RELIABILITY", "LIFECYCLE", "PRODUCTION", "SHUTDOWN", "CRASH", "LOCK", "BOOT", "STALLED", "ORPHAN", "RESTORE", "VAULT", "LONGEVITY", "SLO", "FMEA", "DEGRADED", "REPAIR")),
    ("getting-started", ("INSTALL", "USER_GUIDE", "OPERATOR", "PLATFORM", "SETTINGS")),
    ("measurement", ("CLAIM", "EVALUAT", "TEST", "PROOF", "VALIDAT", "GAUNTLET", "ABLATION", "QUALITY", "MATURITY", "FAIRNESS", "AUDIT")),
    ("documentation", ("README", "TODO", "LEDGER", "DOCKET", "ROADMAP", "GUIDE", "OWNERSHIP", "RECONCILIATION")),
)


def topic_for(path: str) -> str:
    if path in OVERRIDES:
        return OVERRIDES[path]
    upper = path.upper().replace("-", "_")
    for topic, words in RULES:
        if any(word in upper for word in words):
            return topic
    return "architecture"


def category(path: str, text: str = "") -> str:
    text = BLOCK.sub("", text)
    if path in GENERATED:
        return "Generated reference"
    if path in LIVING_STATUS:
        return "Living status or append-only ledger"
    if (path in RECORDS or path.startswith(RECORD_PREFIXES) or DATED.search(path)
            or any(word in path.upper() for word in ("PREREGISTRATION", "TRACKER", "LEDGER", "_TODO", "HANDOFF", "_RESULTS", "_COUNCIL_"))
            or Path(path).name.upper() in {"TODO.MD", "RESULTS.MD"}
            or "Status: Historical" in text[:600]):
        return "Historical or frozen record"
    if (path.startswith(("security/", "docs/standards/", "specs/", ".github/"))
            or any(word in path.upper() for word in ("_POLICY", "_STANDARD", "BRIDGE_PROOFS"))
            or path in {"AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md", "SECURITY.md"}
            or re.search(r"^# .*\b(?:Policy|Standard)\b", text, re.M)):
        return "Standard, policy or contributor reference"
    return "Guide or reference"


def link(doc: str, target: str, anchor: str = "") -> str:
    relative = posixpath.relpath(target, posixpath.dirname(doc) or ".")
    return quote(relative, safe="/._-") + ("#" + anchor if anchor else "")


def heading(topic: dict) -> str:
    return "ISC" if topic["id"] == "isc" else topic["id"].replace("-", " ").capitalize()


def tour(topics: list[dict]) -> str:
    doc = "docs/PLAIN_ENGLISH_GUIDE.md"
    out = [
        "# Aura in plain English",
        "",
        "Aura is an experimental assistant for Apple Silicon Macs. It saves conversation "
        "records and can look them up when you return with a related question. The app can "
        "also make a decision about a tool request and keep a record of what happened.",
        "",
        "[How a chat turn works](../HOW_IT_WORKS.md) · [Measured results](../CLAIMS_MATRIX.md)",
        "",
        "## Topics",
        "",
    ]
    out.extend(f"- [{topic['title']}](#{topic['id']})" for topic in topics)
    for topic in topics:
        out.extend([
            "",
            f"## {heading(topic)}",
            "",
            topic["definition"],
            "",
            topic["daily"],
            "",
            " · ".join(
                f"[{Path(source).stem.replace('_', ' ').title()}]({link(doc, source)})"
                for source in topic["read_more"]
            ),
        ])
    return "\n".join(out).rstrip() + "\n"


def glossary(topics: list[dict], extra: list[dict], date: str) -> str:
    rows = []
    for topic in topics:
        for variable in topic["variables"]:
            rows.append({**variable, "context": topic["title"],
                         "reference": f"VARIABLES_EXPLAINED.md#{topic['id']}"})
    rows.extend(extra)
    rows.sort(key=lambda row: (row["name"].casefold(), row["context"].casefold()))
    out = ["# Aura glossary: terms in ordinary language", "",
           f"Status: Generated reference · Definitions reviewed {date}", "",
           "Each entry gives the term's meaning in the named subsystem or experiment. Examples are illustrative.", "",
           "[Formulas](FORMULAS.md) · [Project tour](PLAIN_ENGLISH_GUIDE.md) · [ISC results](ISC_STATUS.md)", "",
           "| Term and context | Meaning | Why it matters in practice | Example |",
           "|---|---|---|---|"]
    for row in rows:
        label = row["name"].replace("|", "\\|")
        cells = [f"[{label}]({row['reference']}) ({row['context']})"]
        cells += [row[key].replace("|", "\\|") for key in ("meaning", "practical", "example")]
        out.append("| " + " | ".join(cells) + " |")
    out += ["", "Source definitions are in the [topic catalogue](../config/documentation_reading_topics.json)."]
    return "\n".join(out).rstrip() + "\n"


def variables(topics: list[dict], date: str) -> str:
    out = ["# Aura's variables and measurements explained", "", f"Status: Guide · Definitions updated {date}", "",
           "These tables explain quantities that Aura stores or calculates. The examples illustrate the terms.", "",
           "[ISC worked examples](ISC_VARIABLES.md) · [Formulas](FORMULAS.md)", "",
           "## Find a topic", ""]
    out += [f"- [{t['title']}](#{t['id']})" for t in topics]
    for t in topics:
        out += ["", f"## {heading(t)}", "", t["definition"], "",
                "| Variable or term | Accessible definition | What it means in practice | Example |",
                "|---|---|---|---|"]
        for v in t["variables"]:
            out.append("| " + " | ".join(v[k].replace("|", "\\|") for k in ("name", "meaning", "practical", "example")) + " |")
        out += ["", f"[Why this matters, with a metaphor](PLAIN_ENGLISH_GUIDE.md#{t['id']}) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#{t['related']})"]
    return "\n".join(out).rstrip() + "\n"


def inventory(root: Path) -> list[str]:
    raw = subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.md"],
        cwd=root,
    ).decode()
    return sorted({p for p in raw.split("\0") if p} | OUTPUTS)


def document_map(root: Path, paths: list[str], topics: list[dict]) -> str:
    doc = "docs/DOCUMENTATION_INDEX.md"
    by_id = {t["id"]: t for t in topics}
    groups: dict[str, list[tuple[str, str, str]]] = {}
    for path in paths:
        text = (root / path).read_text(errors="replace") if (root / path).exists() else ""
        title_match = re.search(r"^# (.+)$", text, re.M)
        title = title_match.group(1) if title_match else Path(path).stem.replace("_", " ")
        if path == "docs/DOCUMENTATION_INDEX.md":
            title = "Aura documentation: every tracked Markdown page"
        title = title.replace("|", "\\|").replace("[", "").replace("]", "")
        groups.setdefault(category(path, text), []).append((path, title, topic_for(path)))
    out = ["# Aura documentation: every tracked Markdown page", "",
           f"{len(paths)} Markdown pages. Use the [documentation home](README.md) for common entry points "
           "or the [documentation status page](DOC_STATUS.md) for corrections to older records.", "",
           "Regenerate this index with `python tools/render_reader_navigation.py`.", ""]
    for group in ("Guide or reference", "Living status or append-only ledger", "Standard, policy or contributor reference", "Generated reference", "Historical or frozen record"):
        rows = groups.get(group, [])
        out += [f"## {group}", "", f"{len(rows)} pages.", "",
                "| Page | Topic |", "|---|---|"]
        for path, title, key in rows:
            topic = by_id[key]
            out.append(f"| [{title}]({link(doc, path)}) | [{topic['title']}](PLAIN_ENGLISH_GUIDE.md#{key}) |")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / CONFIG).read_text())
    topics = config["topics"]
    by_id = {t["id"]: t for t in topics}
    if len(by_id) != len(topics):
        raise SystemExit("duplicate topic id")
    paths = inventory(ROOT)
    changes: dict[str, str] = {}
    for path in paths:
        if path in OUTPUTS or not (ROOT / path).is_file():
            continue
        text = (ROOT / path).read_text()
        if category(path, text) == "Guide or reference" and BLOCK.search(text):
            changes[path] = BLOCK.sub("", text)
    # Render these before the index so their actual titles are available.
    planned = {
        "docs/PLAIN_ENGLISH_GUIDE.md": tour(topics),
        "docs/VARIABLES_EXPLAINED.md": variables(topics, config["as_of"]),
        "docs/GLOSSARY.md": glossary(topics, config.get("glossary_extra_terms", []), config["as_of"]),
    }
    for path, body in planned.items():
        if not (ROOT / path).exists() or (ROOT / path).read_text() != body:
            changes[path] = body
    if not args.check:
        for path, body in changes.items():
            (ROOT / path).write_text(body)
    index = document_map(ROOT, paths, topics)
    if not (ROOT / "docs/DOCUMENTATION_INDEX.md").exists() or (ROOT / "docs/DOCUMENTATION_INDEX.md").read_text() != index:
        changes["docs/DOCUMENTATION_INDEX.md"] = index
        if not args.check:
            (ROOT / "docs/DOCUMENTATION_INDEX.md").write_text(index)
    print(json.dumps({"indexed": len(paths), "changed": len(changes), "by_topic": dict(Counter(topic_for(p) for p in paths))}, sort_keys=True))
    if args.check and changes:
        print("navigation has drifted: " + ", ".join(sorted(changes)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
