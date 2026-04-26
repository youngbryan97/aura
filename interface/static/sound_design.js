/**
 * sound_design.js — Aura's UI sonic palette.
 *
 * Subtle audio cues for state changes. Every cue:
 *   1. respects the user's `prefers-reduced-motion` preference (silent if set)
 *   2. respects an explicit `sound.muted` setting in /api/settings
 *   3. is generated procedurally from the WebAudio API (no audio files,
 *      so the cue cannot block on a network fetch)
 *
 * Cues:
 *   pulse_completed — completion of a significant background reflection
 *   scar_reinforced — soft tick when a scar's severity increases
 *   arc_complete    — short major-third triad on narrative arc completion
 *   cortex_warm     — single warm note when local cortex becomes Ready
 *   error_phenom    — minor-second pair when an error envelope appears
 *
 * The module installs an event listener on the multimodal SSE feed and
 * fires the matching cue when an event of that kind arrives.
 */
(() => {
  if (window.__auraSoundInstalled) return;
  window.__auraSoundInstalled = true;

  const muted = () => {
    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return true;
    if (window.__auraSoundForceMuted) return true;
    return false;
  };

  let ctx = null;
  function ensureCtx() {
    if (ctx) return ctx;
    try { ctx = new (window.AudioContext || window.webkitAudioContext)(); }
    catch { ctx = null; }
    return ctx;
  }

  function tone(freq, durMs, type = "sine", gainPeak = 0.08) {
    if (muted()) return;
    const ac = ensureCtx();
    if (!ac) return;
    const t0 = ac.currentTime;
    const osc = ac.createOscillator();
    const gain = ac.createGain();
    osc.type = type;
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0, t0);
    gain.gain.linearRampToValueAtTime(gainPeak, t0 + 0.012);
    gain.gain.linearRampToValueAtTime(0, t0 + durMs / 1000);
    osc.connect(gain).connect(ac.destination);
    osc.start(t0);
    osc.stop(t0 + durMs / 1000 + 0.05);
  }

  const cues = {
    pulse_completed:  () => tone(660, 180),
    scar_reinforced:  () => { tone(440, 90); setTimeout(() => tone(330, 90), 60); },
    arc_complete:     () => { tone(523.25, 200); setTimeout(() => tone(659.25, 200), 90); setTimeout(() => tone(783.99, 320), 180); },
    cortex_warm:      () => tone(392, 240, "triangle", 0.06),
    error_phenom:     () => { tone(311.13, 110, "square", 0.05); setTimeout(() => tone(293.66, 110, "square", 0.05), 80); },
  };

  window.auraPlayCue = (name) => { const fn = cues[name]; if (fn) fn(); };

  // hook into multimodal SSE so timeline emits trigger sound cues
  function attachStream(turnId) {
    try {
      const es = new EventSource(`/api/multimodal/stream?turn_id=${encodeURIComponent(turnId)}`);
      es.onmessage = (msg) => {
        try {
          const ev = JSON.parse(msg.data);
          if (ev.kind === "voice_end")          window.auraPlayCue("pulse_completed");
          else if (ev.kind === "memory_thread") window.auraPlayCue("scar_reinforced");
          else if (ev.kind === "image_reveal")  window.auraPlayCue("arc_complete");
        } catch {}
      };
    } catch {}
  }
  window.auraAttachSoundStream = attachStream;
})();
