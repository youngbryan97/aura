#!/usr/bin/env python3
"""A page that replays one watching: the video in its own site's player, and what she said beside it as it reaches
the moment she said it.

Reads a watching kept in ``data/watched/`` (core/skills/watching_along.py) and writes one HTML file. The video is not
copied: the page embeds the site's own player and follows its clock. What she said before it played and after it
ended stand above and below.

    replay_of_a_watching.py ~/.aura/data/watched/20261019-221500.jsonl --out replay.html
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


def rows_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def video_id(url: str) -> str:
    found = re.search(r"(?:v=|youtu\.be/|embed/)([\w-]{11})", url or "")
    return found.group(1) if found else ""


def clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def block(label: str, rows: list[dict]) -> str:
    """What she said before or after, as its own section, or nothing when she said nothing then."""
    if not rows:
        return ""
    return f'<section class="aside"><h2>{label}</h2>' + "".join(f"<p>{html.escape(r['words'])}</p>" for r in rows) + "</section>"


def replay(rows: list[dict]) -> str:
    said = [r for r in rows if r.get("kind") == "said"]
    title = next((r.get("title") for r in rows if r.get("title")), "") or "A watching"
    url = next((r.get("url") for r in rows if r.get("url")), "")
    before, during, after = ([r for r in said if r.get("when") == when] for when in ("before", "during", "after"))
    items = "\n".join(
        f'<li data-at="{float(r.get("at") or 0):.1f}"><span class="at">{clock(r.get("at") or 0)}</span>'
        f"<p>{html.escape(r['words'])}</p></li>" for r in during)
    vid = video_id(url)
    player = (f'<iframe id="player" src="https://www.youtube.com/embed/{vid}?enablejsapi=1" title="{html.escape(title)}" '
              'allow="autoplay; encrypted-media" allowfullscreen></iframe>') if vid else "<p>No video address was kept.</p>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Watching with Aura</title>
<style>
  :root {{ --bg: #fbfaf7; --ink: #1f2328; --muted: #6a6f76; --line: #e3e0d8; --now: #fff4d6; --accent: #8a5a00; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #16181b; --ink: #e8e6e1; --muted: #9aa0a6; --line: #2c3036; --now: #3a3220; --accent: #e6b450; }} }}
  :root[data-theme="dark"] {{ --bg: #16181b; --ink: #e8e6e1; --muted: #9aa0a6; --line: #2c3036; --now: #3a3220; --accent: #e6b450; }}
  body {{ margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.55 Georgia, "Iowan Old Style", serif; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 4px; }} .sub {{ color: var(--muted); margin: 0 0 20px; font-family: system-ui, sans-serif; font-size: .9rem; }}
  .grid {{ display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 24px; align-items: start; }}
  @media (max-width: 820px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .video {{ position: sticky; top: 16px; }} iframe {{ width: 100%; aspect-ratio: 16 / 9; border: 0; border-radius: 8px; background: #000; }}
  h2 {{ font: 600 .8rem system-ui, sans-serif; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 0 0 8px; }}
  .aside {{ border-left: 3px solid var(--accent); padding: 4px 0 4px 14px; margin: 0 0 20px; }}
  ol {{ list-style: none; padding: 0; margin: 0; }} li {{ display: grid; grid-template-columns: 52px 1fr; gap: 8px; padding: 10px 8px; border-bottom: 1px solid var(--line); border-radius: 6px; }}
  li.now {{ background: var(--now); }} .at {{ font: 600 .85rem ui-monospace, monospace; color: var(--accent); padding-top: 3px; }} li p {{ margin: 0; }}
</style></head>
<body><main>
<h1>{html.escape(title)}</h1>
<p class="sub">Aura watched this as it played, in real time, and said what she thought. Her words, as she said them.</p>
<div class="grid">
  <div class="video">{player}</div>
  <div>{block("Before it played", before)}<h2>As it played</h2><ol id="said">{items}</ol>{block("After it ended", after)}</div>
</div>
</main>
<script>
  // The player's own clock, asked of it through the messages it answers; the remark it has reached is marked.
  const frame = document.getElementById("player"), rows = [...document.querySelectorAll("#said li")];
  function listen() {{ if (frame) frame.contentWindow.postMessage(JSON.stringify({{event: "listening", id: 1, channel: "widget"}}), "*"); }}
  if (frame) frame.addEventListener("load", () => {{ listen(); setInterval(listen, 2000); }});
  window.addEventListener("message", (e) => {{
    let data; try {{ data = JSON.parse(e.data); }} catch (_) {{ return; }}
    const now = data && data.info && data.info.currentTime;
    if (typeof now !== "number") return;
    let current = null; for (const li of rows) {{ if (Number(li.dataset.at) <= now) current = li; }}
    rows.forEach((li) => li.classList.toggle("now", li === current));
  }});
</script>
</body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("watching", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(replay(rows_of(args.watching)), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
