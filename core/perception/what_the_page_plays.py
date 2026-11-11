"""What a page plays: each sound it starts, heard as an event with when it started and what it was.

A game says much of what happens by sound: a chime as a coin is taken, a crash as a life is lost, a whine before a
shot is fired at her, a beep with each of her own shots. Some of it is said by nothing else; a warning heard before a
blow lands is not on the screen at all. The page's sounds are started through a handful of calls (an audio buffer
started, an oscillator started, a media element played), so they are heard the way her drawing is read
(core/perception/the_drawing_as_objects.py): a script loaded before the page's own wraps those calls, and notes each
sound as it starts. Nothing is played, changed or delayed; the sounds go on as the page meant them, heard or not.

Each sound has a signature, so that the same sound is known again: an oscillator by its wave and pitch, a buffer by
its length, rate and a hash of its samples, a media element by where it came from. What each sound comes with is
learned in play (core/agency/what_a_sound_comes_with.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from core.runtime.page_gateway import page_script

__all__ = ["BOOTSTRAP", "TAKE", "Sound", "heard_with", "sounds_in"]

_LISTENER = r"""
(() => {
  if (window.__auraSoundsHeard) return true;
  const heard = [], LIMIT = 64;
  const note = (kind, sig) => {
    heard.push({at: performance.now(), kind, sig: String(sig).slice(0, 160)});
    if (heard.length > LIMIT) heard.shift();
  };
  const wrap = (proto, name, sign) => {
    if (!proto || typeof proto[name] !== 'function') return;
    const original = proto[name];
    proto[name] = function (...args) {
      try { note(...sign(this)); } catch (_) {}
      return Reflect.apply(original, this, args);
    };
  };
  const g = window;
  wrap(g.AudioBufferSourceNode && g.AudioBufferSourceNode.prototype, 'start', (node) => {
    const b = node.buffer;
    if (!b) return ['buffer', 'buffer'];
    const d = b.getChannelData(0), step = Math.max(1, Math.floor(d.length / 64));
    let h = 0;
    for (let i = 0; i < d.length; i += step) h = (h * 31 + Math.round(d[i] * 1000)) | 0;
    return ['buffer', `buffer ${b.length} ${b.sampleRate} ${h}`];
  });
  wrap(g.OscillatorNode && g.OscillatorNode.prototype, 'start',
       (node) => ['tone', `${node.type} ${Math.round(node.frequency.value)}`]);
  wrap(g.HTMLMediaElement && g.HTMLMediaElement.prototype, 'play',
       (media) => ['media', media.currentSrc || media.src || 'media']);
  window.__auraSoundsHeard = {take() { return {now: performance.now(), heard: heard.splice(0)}; }};
  return true;
})()
"""

#: Loaded before the page's own scripts, so its sounds are heard from the first.
BOOTSTRAP = page_script(_LISTENER)

#: What has been heard since last asked, with the page's clock now: an expression for a page's evaluate.
TAKE = "window.__auraSoundsHeard?.take() ?? null"


@dataclass(frozen=True)
class Sound:
    """One sound the page started: when, on her clock; what kind of call started it; and its signature."""

    at: float
    kind: str
    sig: str


def sounds_in(taken: Any, at: float) -> list[Sound]:
    """The sounds in what the page handed over (``{"now": page ms, "heard": [...]}``), each put on her clock: ``at``
    is when she asked, on hers. Anything else handed over is nothing heard."""
    if not isinstance(taken, dict) or not isinstance(taken.get("heard"), list):
        return []
    try:
        now = float(taken["now"])
    except (KeyError, TypeError, ValueError):
        return []
    found = []
    for item in taken["heard"][:64]:
        if not isinstance(item, dict):
            continue
        try:
            when = float(item["at"])
        except (KeyError, TypeError, ValueError):
            continue
        kind, sig = str(item.get("kind") or ""), str(item.get("sig") or "")[:160]
        if kind in ("buffer", "tone", "media") and sig:
            found.append(Sound(at - max(0.0, now - when) / 1000.0, kind, sig))
    return found


def heard_with(picture: Any, sounds: list[Sound]) -> Any:
    """The picture, carrying the sounds heard since the one before it; as it was where none were."""
    if not sounds:
        return picture
    from core.perception.the_drawing_as_objects import ObservedPicture

    observed = picture if isinstance(picture, ObservedPicture) else picture.view(ObservedPicture)
    observed.sounds = tuple(sounds)
    return observed
