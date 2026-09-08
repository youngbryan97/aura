#!/usr/bin/env python3
"""How much of the neural feed a person could actually read.

The feed's raw sources are internal logger lines. `interface/static/aura.js`
translates them — two rule tables, a pictograph strip, and a fallback for
key=value telemetry — and the question nobody could answer was how much of a
real session that covers.

Measured against `tests/fixtures/neural_feed_sample.json`, which is the
distinct lines from one live desktop session with how often each occurred. The
weighting matters: a rule for a line emitted twice is worth less than a rule
for one emitted two thousand times, and an unweighted count would say the
opposite.

    python tools/neural_feed_legibility.py            # the number
    python tools/neural_feed_legibility.py --untranslated 20

54.6% when first measured on 2026-09-08. The baseline in
`config/neural_feed_legibility_baseline.json` only goes up.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AURA_JS = ROOT / "interface" / "static" / "aura.js"
FIXTURE = ROOT / "tests" / "fixtures" / "neural_feed_sample.json"
BASELINE = ROOT / "config" / "neural_feed_legibility_baseline.json"

#: Loads the card's own translation path out of aura.js and runs the fixture
#: through it. Node rather than a Python reimplementation: a second copy of
#: the rules would drift from the one that renders, and then this would be
#: measuring something nobody sees.
_HARNESS = r"""
const fs = require('fs');
const lines = fs.readFileSync(process.argv[1], 'utf8').split('\n');

function bodyOf(startsWith) {
  const start = lines.findIndex((l) => l.startsWith(startsWith));
  if (start < 0) throw new Error('missing ' + startsWith);
  let depth = 0, seen = false, end = start;
  for (let i = start; i < lines.length; i++) {
    for (const ch of lines[i]) { if (ch === '{') { depth++; seen = true; } else if (ch === '}') depth--; }
    if (seen && depth === 0) { end = i; break; }
  }
  return lines.slice(start, end + 1).join('\n');
}

const start = lines.findIndex((l) => l.startsWith('const PLAIN_LANGUAGE_RULES = ['));
let end = lines.findIndex((l, i) => i > start && l.startsWith('function toPlainEnglish'));
let depth = 0, seen = false;
for (let i = end; i < lines.length; i++) {
  for (const ch of lines[i]) { if (ch === '{') { depth++; seen = true; } else if (ch === '}') depth--; }
  if (seen && depth === 0) { end = i; break; }
}
const block = lines.slice(start, end + 1).join('\n') + '\n' + bodyOf('function stripNeuralPictographs');
const make = new Function('escHtml', block + `
  return {
    // What the card shows.
    shown: function (raw) {
      return plainLanguageThought(toPlainEnglish(stripNeuralPictographs(String(raw || '')).trim()));
    },
    // And the same line with only its pictographs taken off. Removing an
    // emoji is not translating a sentence: measured against the RAW line,
    // "Cortex response received (len=1159)" counted as legible because the
    // tick mark in front of it had gone. That is the measure agreeing with
    // itself. Compare against this instead, and a line counts only when a
    // rule actually rewrote it.
    stripped: function (raw) {
      return stripNeuralPictographs(String(raw || '')).trim();
    },
  };`);
const card = make((s) => String(s));

const fixture = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
let translated = 0, raw = 0, hidden = 0;
const untouched = [];
for (const row of fixture.lines) {
  const shown = card.shown(row.text).trim();
  // A rule line, a banner of box-drawing characters: the card renders
  // nothing for it, and a line nobody sees is neither readable nor
  // unreadable. Counting it as unreadable makes the number about how much
  // decoration the logger emits.
  if (!shown) { hidden += row.count; continue; }
  if (shown !== card.stripped(row.text)) translated += row.count;
  else { raw += row.count; untouched.push(row); }
}
untouched.sort((a, b) => b.count - a.count);
process.stdout.write(JSON.stringify({
  translated_events: translated,
  raw_events: raw,
  hidden_events: hidden,
  events: translated + raw,
  share: translated / Math.max(1, translated + raw),
  untranslated: untouched.slice(0, 40),
}));
"""


def measure() -> dict:
    result = subprocess.run(
        ["node", "-e", _HARNESS, str(AURA_JS), str(FIXTURE)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"could not run the feed's own translator:\n{result.stderr[:2000]}")
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--untranslated", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = measure()
    share = report["share"]
    baseline = 0.0
    if BASELINE.is_file():
        baseline = float(json.loads(BASELINE.read_text()).get("share") or 0.0)

    if args.json:
        print(json.dumps({**report, "baseline": baseline}, indent=2))
        return 0

    print(
        f"{share * 100:.1f}% of a live session's feed events say something a "
        f"person could read ({report['translated_events']:,} of "
        f"{report['events']:,}); baseline {baseline * 100:.1f}%"
    )
    for row in report["untranslated"][: args.untranslated]:
        print(f"  {row['count']:5}  {row['text'][:110]}")
    if share + 1e-9 < baseline:
        print("legibility fell below the baseline", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
