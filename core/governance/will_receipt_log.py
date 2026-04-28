"""core/will/receipt_log.py

Longitudinal Will-receipt accumulator
=======================================
Persists every UnifiedWill decision to a 30-day rolling JSONL ledger and
exposes an inspector that distills a stable decision policy from the
ledger:

    * top approved domains
    * top refused domains
    * common rationales
    * change-of-mind events (action approved at T1, refused at T2)
    * regret patterns

The inspector is what an external evaluator uses to reconstruct Aura's
"will pattern" without reading any of her replies — only her decisions.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import json
import logging
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.WillReceiptLog")

_DIR = Path.home() / ".aura" / "data" / "will_receipts"
_DIR.mkdir(parents=True, exist_ok=True)
_PATH = _DIR / "receipts.jsonl"


@dataclass
class WillReceiptEntry:
    when: float
    receipt_id: str
    action: str
    domain: str
    approved: bool
    reason: str
    rationale: List[str] = field(default_factory=list)
    regret: Optional[float] = None


def append(entry: WillReceiptEntry) -> None:
    try:
        with open(_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry), default=str) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                pass  # no-op: intentional
    except Exception as exc:
        record_degradation('will_receipt_log', exc)
        logger.warning("will receipt append failed: %s", exc)


def recent(*, days: int = 30) -> List[WillReceiptEntry]:
    cutoff = time.time() - days * 86_400.0
    out: List[WillReceiptEntry] = []
    if not _PATH.exists():
        return out
    try:
        with open(_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if float(rec.get("when", 0.0)) < cutoff:
                    continue
                out.append(WillReceiptEntry(**{k: v for k, v in rec.items() if k in WillReceiptEntry.__dataclass_fields__}))
    except Exception as exc:
        record_degradation('will_receipt_log', exc)
        logger.debug("will receipt read failed: %s", exc)
    return out


def summarize_policy(*, days: int = 30, top_n: int = 10) -> Dict[str, Any]:
    entries = recent(days=days)
    if not entries:
        return {"days": days, "n": 0}

    approved = [e for e in entries if e.approved]
    refused = [e for e in entries if not e.approved]

    approved_domains = Counter(e.domain for e in approved)
    refused_domains = Counter(e.domain for e in refused)
    refused_reasons = Counter(e.reason for e in refused)

    # Change-of-mind: same action+domain approved earlier and refused
    # later (or vice versa) within the window.
    by_action: Dict[str, List[WillReceiptEntry]] = {}
    for e in entries:
        by_action.setdefault(f"{e.action}|{e.domain}", []).append(e)
    change_of_mind = 0
    for k, lst in by_action.items():
        if len({x.approved for x in lst}) > 1:
            change_of_mind += 1

    regrets = [float(e.regret) for e in entries if e.regret is not None]
    avg_regret = (sum(regrets) / len(regrets)) if regrets else None

    return {
        "days": days,
        "n": len(entries),
        "approved_n": len(approved),
        "refused_n": len(refused),
        "top_approved_domains": approved_domains.most_common(top_n),
        "top_refused_domains": refused_domains.most_common(top_n),
        "top_refused_reasons": refused_reasons.most_common(top_n),
        "change_of_mind_actions": change_of_mind,
        "avg_regret": avg_regret,
    }


__all__ = ["WillReceiptEntry", "append", "recent", "summarize_policy"]
