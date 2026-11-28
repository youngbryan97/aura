// The frame a made program runs in. The program is data and four functions of it: how it begins, what each event does to
// it, and how it is drawn or shown. Everything around that is here, and is the same for every program: the clock, the
// keys and the pointer, the window, what is kept between launches, the sound, the start, pause and end of a game, and
// the look. Because what a program does is a function of its state and an event, it can be tried without a window, and
// is: see the second half of this file.
"use strict";
const Frame = (() => {
  // ── the program ────────────────────────────────────────────────────────────────────────────────────────────────
  let P = null;
  const W = 960, H = 600, STEP = 1 / 60;
  const NONE = { x: W / 2, y: H / 2, down: false };

  // Chance that can be begun again from the same place, so a program can be tried twice the same way.
  let seed = 1;
  function rnd(a, b) {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    const u = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    return a === undefined ? u : b === undefined ? u * a : a + u * (b - a);
  }
  const chance = { rnd, between: (a, b) => a + rnd() * (b - a), int: (a, b) => Math.floor(a + rnd() * (b - a + 1)),
    pick: (list) => list[Math.floor(rnd() * list.length)], shuffle: (list) => {
      const out = list.slice();
      for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [out[i], out[j]] = [out[j], out[i]]; }
      return out; } };

  let live = false, muted = false;
  const played = [];
  function play(name) { if (live) { if (!muted) sound(name); } else played.push(String(name)); }

  // Who is told of each state a test comes to, where somebody asks.
  let watching = null;
  // While a rule is tried with nothing else about: what keeps the lists it does not name empty, whatever comes into them.
  let emptied = null;
  function apply(state, event) {
    // Whatever kind of event it is, it has the time that passed (none, for a key), the keys that are down and the pointer.
    const next = P.update(state, { dt: 0, held: {}, pointer: NONE, ...event }), now = next && typeof next === "object" ? next : state;
    return emptied ? emptied(now) : now;
  }
  const isGame = () => P.kind === "game";
  function ended(state) {
    if (!P.over) return null;
    const said = P.over(state);
    if (!said) return null;
    return typeof said === "object" ? { won: !!said.won, say: String(said.say || "") } : { won: false, say: "" };
  }
  const scoreOf = (state) => (P.score ? Number(P.score(state)) || 0 : 0);

  // ── sound ──────────────────────────────────────────────────────────────────────────────────────────────────────
  let audio = null;
  const VOICES = { blip: [660, 0.06, "square", 0.05], pick: [880, 0.12, "triangle", 0.12, 1320], hit: [140, 0.16, "sawtooth", 0.12, 70],
    win: [523, 0.5, "triangle", 0.14, 1046], lose: [300, 0.6, "sawtooth", 0.1, 80], step: [220, 0.03, "square", 0.03],
    shoot: [980, 0.1, "square", 0.06, 300], jump: [300, 0.18, "square", 0.07, 720], bell: [1046, 0.9, "sine", 0.16], tick: [1400, 0.03, "sine", 0.05] };
  function sound(name) {
    const voice = VOICES[name] || VOICES.blip;
    try {
      audio = audio || new (window.AudioContext || window.webkitAudioContext)();
      const o = audio.createOscillator(), g = audio.createGain(), at = audio.currentTime;
      o.type = voice[2]; o.frequency.setValueAtTime(voice[0], at);
      if (voice[4]) o.frequency.exponentialRampToValueAtTime(voice[4], at + voice[1]);
      g.gain.setValueAtTime(voice[3], at); g.gain.exponentialRampToValueAtTime(0.0001, at + voice[1]);
      o.connect(g).connect(audio.destination); o.start(at); o.stop(at + voice[1] + 0.02);
    } catch (e) { /* a window with no sound is a window with no sound */ }
  }

  // ── the pen a game draws with: a world of 960 by 600, whatever the window is ───────────────────────────────────
  function pen(ctx, t) {
    const opt = (o) => { ctx.globalAlpha = o && o.alpha != null ? Math.max(0, Math.min(1, o.alpha)) : 1;
      ctx.shadowBlur = o && o.glow ? o.glow : 0; ctx.shadowColor = (o && o.glowColor) || (o && o.color) || "#fff"; };
    const fin = (color, o) => { if (o && o.stroke) { ctx.lineWidth = o.width || 2; ctx.strokeStyle = o.stroke; ctx.stroke(); }
      ctx.shadowBlur = 0; ctx.globalAlpha = 1; };
    const g = { W, H, t, ctx, base: ctx.getTransform(),
      // Drawn in the view's own place, wherever the world has been moved to.
      screen(draw) { ctx.save(); ctx.setTransform(g.base); draw(drawn); ctx.restore(); },
      glow(x, y, r, color, strength) { ctx.save(); ctx.globalCompositeOperation = "lighter"; ctx.globalAlpha = strength == null ? 0.6 : Math.max(0, Math.min(1, strength)); const f = ctx.createRadialGradient(x, y, 0, x, y, Math.max(1, r)); f.addColorStop(0, color || "#ffe9a8"); f.addColorStop(1, "rgba(0,0,0,0)"); ctx.fillStyle = f; ctx.beginPath(); ctx.arc(x, y, Math.max(1, r), 0, Math.PI * 2); ctx.fill(); ctx.restore(); },
      clear(color) { ctx.globalAlpha = 1; ctx.fillStyle = color || "#101418"; ctx.fillRect(0, 0, W, H); },
      sky(top, bottom) { const f = ctx.createLinearGradient(0, 0, 0, H); f.addColorStop(0, top); f.addColorStop(1, bottom); ctx.globalAlpha = 1; ctx.fillStyle = f; ctx.fillRect(0, 0, W, H); },
      rect(x, y, w, h, color, o) { if (color && typeof color === "object") { o = color; color = o.fill || o.color || null; } opt({ color, ...o }); ctx.fillStyle = color; ctx.beginPath();
        if (o && o.r && ctx.roundRect) ctx.roundRect(x, y, w, h, o.r); else ctx.rect(x, y, w, h); if (color) ctx.fill(); fin(color, o); },
      circle(x, y, r, color, o) { if (color && typeof color === "object") { o = color; color = o.fill || o.color || null; } opt({ color, ...o }); ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, Math.max(0, r), 0, Math.PI * 2); if (color) ctx.fill(); fin(color, o); },
      ellipse(x, y, rx, ry, color, o) { opt({ color, ...o }); ctx.fillStyle = color; ctx.beginPath(); ctx.ellipse(x, y, Math.max(0, rx), Math.max(0, ry), (o && o.turn) || 0, 0, Math.PI * 2); if (color) ctx.fill(); fin(color, o); },
      line(x1, y1, x2, y2, color, width, o) { opt({ color, ...o }); ctx.strokeStyle = color; ctx.lineWidth = width || 2; ctx.lineCap = "round"; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.shadowBlur = 0; ctx.globalAlpha = 1; },
      poly(points, color, o) { opt({ color, ...o }); ctx.fillStyle = color; ctx.beginPath(); points.forEach((p, i) => { const x = Array.isArray(p) ? p[0] : p.x, y = Array.isArray(p) ? p[1] : p.y; if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.closePath(); if (color) ctx.fill(); fin(color, o); },
      wedge(x, y, r, from, to, color, o) { opt({ color, ...o }); const f = ctx.createRadialGradient(x, y, 0, x, y, Math.max(1, r)); f.addColorStop(0, color); f.addColorStop(1, "rgba(0,0,0,0)"); ctx.fillStyle = o && o.solid ? color : f; ctx.beginPath(); ctx.moveTo(x, y); ctx.arc(x, y, Math.max(0, r), from, to); ctx.closePath(); ctx.fill(); ctx.shadowBlur = 0; ctx.globalAlpha = 1; },
      text(words, x, y, o) { if (typeof words === "number" && typeof y === "string") [words, x, y] = [y, words, x]; o = o || {}; opt({ ...o }); ctx.fillStyle = o.color || "#fff"; ctx.font = `${o.weight || 600} ${o.size || 20}px ${o.font || "ui-rounded, -apple-system, 'SF Pro Rounded', system-ui, sans-serif"}`; ctx.textAlign = o.align || "left"; ctx.textBaseline = o.base || "middle"; ctx.fillText(String(words), x, y); ctx.shadowBlur = 0; ctx.globalAlpha = 1; },
      // What she painted for it, by its name; where there is no such picture, the figure of that name, or a plain shape.
      picture(name, x, y, o) { o = o || {}; const img = painted(name), size = o.size || 40; if (img) { const k = size / Math.max(img.naturalWidth, img.naturalHeight), w = img.naturalWidth * k, h = img.naturalHeight * k; ctx.save(); ctx.translate(x, y); if (o.turn) ctx.rotate(o.turn); if (o.flip) ctx.scale(-1, 1); if (o.alpha != null) ctx.globalAlpha = Math.max(0, Math.min(1, o.alpha)); ctx.drawImage(img, -w / 2, -h / 2, w, h); ctx.restore(); return; } if (typeof figures !== "undefined" && figures.all[name]) return figures.draw(g, name, x, y, { ...o, t: g.t }); g.rect(x - size / 2, y - size / 2, size, size, (P.look && P.look.accent) || "#888", { r: size / 4, alpha: o.alpha }); },
      backdrop(name, o) { o = o || {}; const img = painted(name); g.screen(() => { if (!img) return g.sky("#26304a", "#8593b5"); const k = Math.max(W / img.naturalWidth, H / img.naturalHeight), w = img.naturalWidth * k, h = img.naturalHeight * k, off = (((o.scroll || 0) % (2 * w)) + 2 * w) % (2 * w); ctx.globalAlpha = o.alpha != null ? o.alpha : 1; for (let i = 0; i * w - off < W; i++) { const left = i * w - off; if (left + w < 0) continue; if (i % 2) { ctx.save(); ctx.translate(left + w, (H - h) / 2); ctx.scale(-1, 1); ctx.drawImage(img, 0, 0, w, h); ctx.restore(); } else ctx.drawImage(img, left, (H - h) / 2, w, h); } ctx.globalAlpha = 1; }); },
      at(x, y, turn, draw) { ctx.save(); ctx.translate(x, y); if (turn) ctx.rotate(turn); draw(drawn); ctx.restore(); } };
    // What the pen does not name is the canvas's own, as a canvas is drawn on anywhere: g.fillRect(...), g.fillStyle = ...
    // LIVE 28 November 2026 a book's spine drawn with g.fillRect broke the page it was in, and every test of it held.
    const drawn = new Proxy(g, {
      get: (o, k) => (k in o || typeof k === "symbol" ? o[k] : typeof ctx[k] === "function" ? ctx[k].bind(ctx) : ctx[k]),
      set: (o, k, v) => { if (k in o) o[k] = v; else ctx[k] = v; return true; },
    });
    return drawn;
  }

  // ── what an app is shown with: a tree of plain things, drawn into the page and kept in step with it ────────────
  const node = (tag, props, kids) => ({ tag, props: props || {}, kids: (kids || []).flat(9).filter((k) => k != null && k !== false).map((k) => (typeof k === "object" ? k : { tag: "#", text: String(k) })) });
  const ICONS = {"plus":"M8 3v10M3 8h10","minus":"M3 8h10","check":"m3.5 8.5 3 3 6-7","close":"m4 4 8 8M12 4l-8 8","trash":"M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.6 8.5h5.8l.6-8.5","edit":"m3 13 .5-2.5L10.5 3.5l2 2-7 7zM9 5l2 2","download":"M8 3v7M5 7.5 8 10.5 11 7.5M3.5 13h9","upload":"M8 11V4M5 6.5 8 3.5 11 6.5M3.5 13h9","search":"M7 11.5a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9zM10.3 10.3 13.5 13.5","star":"M8 2.5l1.7 3.5 3.8.5-2.8 2.6.7 3.8L8 11l-3.4 1.9.7-3.8-2.8-2.6 3.8-.5z","heart":"M8 13s-5-3-5-6.5A2.5 2.5 0 0 1 8 5a2.5 2.5 0 0 1 5 1.5C13 10 8 13 8 13z","clock":"M8 13.5a5.5 5.5 0 1 0 0-11 5.5 5.5 0 0 0 0 11zM8 5v3l2 1.5","calendar":"M3 4.5h10v8.5H3zM3 7h10M5.5 3v3M10.5 3v3","play":"M5 3.5v9l7-4.5z","pause":"M5.5 3.5v9M10.5 3.5v9","stop":"M4 4h8v8H4z","refresh":"M12.5 8A4.5 4.5 0 1 1 11 4.6M12.5 2.5v2.6H10","settings":"M8 10a2 2 0 1 0 0-4 2 2 0 0 0 0 4zM8 2v1.5M8 12.5V14M2 8h1.5M12.5 8H14M3.8 3.8l1 1M11.2 11.2l1 1M3.8 12.2l1-1M11.2 4.8l1-1","user":"M8 7.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5zM3.5 13.5c.6-2.6 2.3-4 4.5-4s3.9 1.4 4.5 4","users":"M6 7a2.2 2.2 0 1 0 0-4.4A2.2 2.2 0 0 0 6 7zM2 13c.5-2.3 2-3.5 4-3.5s3.5 1.2 4 3.5M10.5 3a2 2 0 0 1 0 4M11.5 9.6c1.4.4 2.2 1.6 2.5 3.4","cart":"M2.5 3h2l1.5 7h6l1.5-5H5M6.5 13h.1M11.5 13h.1","list":"M5.5 4.5h8M5.5 8h8M5.5 11.5h8M2.5 4.5h.1M2.5 8h.1M2.5 11.5h.1","filter":"M2.5 3.5h11l-4.3 5v4l-2.4 1v-5z","sort":"M5 3v10M2.5 10.5 5 13l2.5-2.5M11 13V3M8.5 5.5 11 3l2.5 2.5","arrow-right":"M3 8h10M9.5 4.5 13 8l-3.5 3.5","arrow-left":"M13 8H3M6.5 4.5 3 8l3.5 3.5","chevron":"m6 4 4 4-4 4","info":"M8 13.5a5.5 5.5 0 1 0 0-11 5.5 5.5 0 0 0 0 11zM8 7.5v3.5M8 5.2v.1","warn":"M8 2.5 14 13H2zM8 6.5v3M8 11.3v.1","copy":"M5.5 5.5h7v7h-7zM3.5 10.5v-7h7","share":"M8 10V2.5M5 5.5 8 2.5l3 3M3.5 8.5v4.5h9V8.5","print":"M4.5 6V2.5h7V6M4.5 11h-2V6h11v5h-2M4.5 9h7v4.5h-7z","home":"M2.5 7.5 8 3l5.5 4.5M4 6.5V13h8V6.5","leaf":"M3 13c0-6 4-9.5 10-10-.5 6-4 10-10 10zM3 13l5-5","drop":"M8 2.5s4.5 4.6 4.5 7.5a4.5 4.5 0 0 1-9 0C3.5 7.1 8 2.5 8 2.5z","sun":"M8 10.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5zM8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1 1M11.6 11.6l1 1M3.4 12.6l1-1M11.6 4.4l1-1","moon":"M13 9.5A5.5 5.5 0 1 1 6.5 3 4.5 4.5 0 0 0 13 9.5z","bell":"M4 11V7.5a4 4 0 0 1 8 0V11l1 1.5H3zM6.5 13.5a1.5 1.5 0 0 0 3 0","tag":"M2.5 2.5h5l6 6-5 5-6-6zM5.5 5.5h.1","book":"M3 3.5c2-.7 3.5-.5 5 .7 1.5-1.2 3-1.4 5-.7v9c-2-.7-3.5-.5-5 .7-1.5-1.2-3-1.4-5-.7zM8 4.2v9","chart":"M2.5 13.5h11M4.5 11V8M8 11V4.5M11.5 11V6.5","money":"M2.5 4.5h11v7h-11zM8 9.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM4.5 6.5v.1M11.5 9.5v.1","pin":"M8 14s4.5-4.2 4.5-7.5a4.5 4.5 0 0 0-9 0C3.5 9.8 8 14 8 14zM8 8a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3z","food":"M4 2.5v4a1.5 1.5 0 0 0 3 0v-4M5.5 2.5v11M11 2.5c-1.4 0-2 1.6-2 4s.6 3 2 3v4","undo":"M5.5 6.5H10a3 3 0 0 1 0 6H7M5.5 6.5 8 4M5.5 6.5 8 9"};
  const glyph = (name) => (ICONS[name] ? node("svg", { viewBox: "0 0 16 16", class: "ico", "aria-hidden": "true" }, [node("path", { d: ICONS[name] })]) : null);
  const ui = {
    column: (...k) => node("div", { class: "column" }, k), row: (...k) => node("div", { class: "row" }, k),
    card: (...k) => node("section", { class: "card" }, k), grid: (...k) => node("div", { class: "grid" }, k),
    title: (t) => node("h1", {}, [t]), heading: (t) => node("h2", {}, [t]), text: (t) => node("p", {}, [t]),
    note: (t) => node("p", { class: "note" }, [t]), big: (t) => node("div", { class: "big" }, [t]), badge: (t, tone) => node("span", { class: "badge " + (tone || "") }, [t]),
    divider: () => node("hr"), spacer: () => node("div", { class: "spacer" }),
    button: (label, name, value, o) => node("button", { class: (o && o.tone) || "", type: "button", disabled: !!(o && o.disabled), title: (o && o.hint) || null, do: [name, value === undefined ? null : value] }, [o && o.icon ? glyph(o.icon) : null, label]),
    icon: (name) => glyph(name),
    // What there is nothing of yet, said where it would be, with what to do about it.
    empty: (title, hint, icon) => node("div", { class: "empty" }, [glyph(icon || "list"), node("strong", {}, [title]), hint ? node("p", { class: "note" }, [hint]) : null]),
    // Over everything while it is shown: what is to be decided before anything else is done.
    dialog: (title, ...k) => node("div", { class: "scrim" }, [node("section", { class: "dialog", role: "dialog", "aria-modal": "true", "aria-label": title }, [node("h2", {}, [title]), ...k])]),
    toast: (words, tone) => (words ? node("div", { class: "toast " + (tone || ""), role: "status", renew: words }, [words]) : null),
    input: (label, name, value, o) => field(label, node("input", { type: (o && o.type) || "text", value: value == null ? "" : String(value), placeholder: (o && o.placeholder) || "", in: name, enter: (o && o.enter) || null, "aria-label": label })),
    number: (label, name, value, o) => field(label, node("input", { type: "number", value: value == null ? "" : String(value), min: o && o.min, max: o && o.max, step: o && o.step, in: name, number: true, enter: (o && o.enter) || null, "aria-label": label })),
    date: (label, name, value) => field(label, node("input", { type: "date", value: value || "", in: name, "aria-label": label })),
    area: (label, name, value, o) => field(label, node("textarea", { value: value == null ? "" : String(value), rows: (o && o.rows) || 5, placeholder: (o && o.placeholder) || "", in: name, "aria-label": label })),
    check: (label, name, on) => node("label", { class: "check" }, [node("input", { type: "checkbox", checked: !!on, in: name, "aria-label": label }), label]),
    toggle: (label, name, value, on) => node("label", { class: "check" }, [node("input", { type: "checkbox", checked: !!on, do: [name, value === undefined ? null : value], "aria-label": label }), label]),
    // What is chosen is what it shows: where nothing that is offered is, it shows that nothing is.
    select: (label, name, value, options) => { const all = (options || []).map((o) => (Array.isArray(o) ? o : [o, o])), at = value == null ? "" : String(value); if (all.length && !all.some(([v]) => String(v) === at)) all.unshift([at, at || "Choose…"]); return field(label, node("select", { value: at, in: name, "aria-label": label }, all.map(([v, t]) => node("option", { value: String(v), selected: String(v) === at }, [t])))); },
    slider: (label, name, value, o) => field(label, node("input", { type: "range", value: String(value), min: (o && o.min) ?? 0, max: (o && o.max) ?? 100, step: (o && o.step) ?? 1, in: name, number: true, "aria-label": label })),
    list: (items, each) => node("ul", { class: "list" }, (items || []).map((item, i) => node("li", {}, [each ? each(item, i) : String(item)]))),
    // Figures stand under each other, and a table with nothing in it says so.
    table: (heads, rows) => { const all = rows || [], figures = (i) => all.length > 0 && all.every((r) => /^[-−+]?\s*[$€£]?\s*[-−]?[\d,. ]*\d\s*%?$/.test(String(r[i] ?? ""))), num = (i) => (figures(i) ? { class: "num" } : {});
      return node("div", { class: "scroll" }, [node("table", {}, [node("thead", {}, [node("tr", {}, (heads || []).map((h, i) => node("th", num(i), [h])))]), node("tbody", {}, all.length ? all.map((r) => node("tr", {}, r.map((c, i) => node("td", num(i), [c])))) : [node("tr", {}, [node("td", { class: "none", colspan: String((heads || []).length || 1) }, ["Nothing yet"])])])])]); },
    tabs: (names, chosen, name) => node("div", { class: "tabs", role: "tablist" }, (names || []).map((n) => node("button", { type: "button", role: "tab", class: n === chosen ? "on" : "", do: [name, n] }, [n]))),
    progress: (value, of) => node("div", { class: "progress", role: "progressbar" }, [node("div", { style: `width:${Math.max(0, Math.min(100, (100 * value) / (of || 1)))}%` })]),
    bars: (rows) => { const top = Math.max(1e-9, ...(rows || []).map((r) => Number(r.value) || 0)); return node("div", { class: "bars" }, (rows || []).map((r) => node("div", { class: "bar" }, [node("span", { class: "bar-label" }, [r.label]), node("div", { class: "bar-track" }, [node("div", { class: "bar-fill", style: `width:${(100 * (Number(r.value) || 0)) / top}%` })]), node("span", { class: "bar-value" }, [String(typeof r.show === "string" || typeof r.show === "number" ? r.show : r.value)])]))); },
    lines: (points) => { const v = (points || []).map(Number), top = Math.max(1e-9, ...v), low = Math.min(0, ...v), n = Math.max(1, v.length - 1); return node("svg", { viewBox: "0 0 600 180", class: "lines" }, [node("polyline", { points: v.map((y, i) => `${(600 * i) / n},${175 - (170 * (y - low)) / (top - low || 1)}`).join(" "), fill: "none" })]); },
    canvas: (name, w, h, draw) => node("canvas", { width: w, height: h, paint: draw, point: name }),
    picture: (name, o) => node("img", { class: "picture", src: (P && P.pictures && P.pictures[name]) || BLANK, alt: (o && o.says) || name, style: `height:${(o && o.height) || 120}px` }),
  };
  const BLANK = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
  // The pictures she painted, as they are drawn: there once they have been read.
  const pics = {};
  function painted(name) { const src = P && P.pictures && P.pictures[name]; if (!src) return null; const img = pics[name] || (pics[name] = Object.assign(new Image(), { src })); return img.complete && img.naturalWidth ? img : null; }
  const field = (label, control) => node("label", { class: "field" }, [node("span", {}, [label]), control]);
  const words = (v) => (v.tag === "#" ? v.text : v.kids.map(words).join(" ")).replace(/\s+/g, " ").trim();
  function every(v, found) { found = found || []; if (v.tag !== "#") { found.push(v); v.kids.forEach((k) => every(k, found)); } return found; }
  // What is worked out from the state is worked out where it is read, and is never kept: kept, it was what the state had been
  // when it was last worked out (LIVE 25 November 2026, a shopping list that went on holding a dish that had been taken off it).
  const whole = (state) => (P && typeof P.derived === "function" && state && typeof state === "object" ? { ...state, ...P.derived(state) } : state);
  function shown(state) { const v = P.view(whole(state), ui); return v && v.tag ? v : node("div", {}, [v]); }

  const SVG = new Set(["svg", "polyline", "path"]);
  function make(v) {
    if (v.tag === "#") return document.createTextNode(v.text);
    const el = SVG.has(v.tag) ? document.createElementNS("http://www.w3.org/2000/svg", v.tag) : document.createElement(v.tag);
    el.__kids = []; fit(el, { tag: v.tag, props: {}, kids: [] }, v); return el;
  }
  function fit(el, was, v) {
    for (const key of Object.keys({ ...was.props, ...v.props })) {
      const value = v.props[key];
      if (key === "do" || key === "in" || key === "enter" || key === "number" || key === "paint" || key === "point" || key === "renew") { el["__" + key] = value; continue; }
      if (key === "value") { if (el.value !== value && document.activeElement !== el) el.value = value ?? ""; else if (el.value !== value && value === "") el.value = ""; continue; }
      if (key === "checked" || key === "disabled" || key === "selected") { el[key] = !!value; continue; }
      if (value == null || value === false) el.removeAttribute(key); else el.setAttribute(key, value === true ? "" : value);
    }
    const old = el.__v ? el.__v.kids : [];
    v.kids.forEach((kid, i) => {
      const at = el.childNodes[i], before = old[i];
      if (!at) el.appendChild(make(kid));
      else if (!before || before.tag !== kid.tag || (kid.tag !== "#" && before.props.renew !== kid.props.renew)) el.replaceChild(make(kid), at);
      else if (kid.tag === "#") { if (at.nodeValue !== kid.text) at.nodeValue = kid.text; }
      else fit(at, before, kid);
    });
    while (el.childNodes.length > v.kids.length) el.removeChild(el.lastChild);
    el.__v = v;
    if (v.props.paint) v.props.paint(pen(el.getContext("2d"), performance.now() / 1000), el.width, el.height);
  }

  // ── the window ─────────────────────────────────────────────────────────────────────────────────────────────────
  const key = (e) => (e.key === " " ? "Space" : e.key.length === 1 ? e.key.toLowerCase() : e.key);
  let state = null, root = null;
  const held = {}, pointer = { x: W / 2, y: H / 2, down: false };
  const store = () => "made:" + P.name;
  function keep() { try { localStorage.setItem(store(), JSON.stringify(state)); } catch (e) { /* kept where it can be */ } }
  function kept() { try { return JSON.parse(localStorage.getItem(store()) || "null"); } catch (e) { return null; } }
  const best = { get() { try { return Number(localStorage.getItem(store() + ":best")) || 0; } catch (e) { return 0; } }, set(n) { try { localStorage.setItem(store() + ":best", String(n)); } catch (e) { /* not kept */ } } };
  const el = (tag, cls, text) => { const x = document.createElement(tag); if (cls) x.className = cls; if (text != null) x.textContent = text; return x; };

  // An accent white words can be read on, darkened until they can (core/making/design.py, that_can_be_read_on), whoever set it:
  // its design, or the program itself with look: { accent }.
  const lightOf = (hex) => { const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
  const readable = (hex) => { if (!/^#[0-9a-f]{6}$/i.test(hex || "")) return null; let c = hex.toLowerCase(); for (let i = 0; i < 40 && 1.05 / (lightOf(c) + 0.05) < 4.8; i++) c = "#" + [1, 3, 5].map((j) => Math.floor(parseInt(c.slice(j, j + 2), 16) * 0.95).toString(16).padStart(2, "0")).join(""); return c; };
  function open(program) {
    P = program; live = true; seed = (Date.now() ^ (Math.random() * 1e9)) | 0;
    document.title = P.name;
    const accent = readable(P.look && P.look.accent);
    if (accent) document.documentElement.style.setProperty("--accent", accent);
    root = document.getElementById("program"); root.textContent = "";
    (isGame() ? openGame : openApp)();
  }
  // Asked of whoever uses it before what cannot be taken back: the first button leaves things as they were, and Escape or a click
  // beside it does too.
  function askFirst(title, words, yes, then) {
    const scrim = el("div", "scrim"), box = el("section", "dialog"), row = el("div", "row"), stay = el("button", "", "Keep it"), go = el("button", "danger", yes);
    box.setAttribute("role", "dialog"); box.setAttribute("aria-modal", "true"); box.setAttribute("aria-label", title); stay.type = go.type = "button";
    const shut = () => { scrim.remove(); document.removeEventListener("keydown", keys, true); };
    const keys = (e) => { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); shut(); } else if (e.key === "Tab") { e.preventDefault(); (document.activeElement === stay ? go : stay).focus(); } };
    stay.onclick = shut; go.onclick = () => { shut(); then(); }; scrim.onclick = (e) => { if (e.target === scrim) shut(); };
    row.append(stay, go); box.append(el("h2", "", title), el("p", "", words), row); scrim.appendChild(box);
    document.body.appendChild(scrim); document.addEventListener("keydown", keys, true); stay.focus();
  }
  function problem(e) { const was = document.querySelector(".problem"); if (was) was.remove(); const p = el("div", "problem", "Something in this program went wrong: " + (e && e.message ? e.message : e)); document.body.appendChild(p); setTimeout(() => p.remove(), 8000); console.error(e); }

  function openApp() {
    document.body.className = "app";
    // Opened where it was left, or where it is to be looked at.
    state = P.initial(); const was = window.__openIn || kept(); if (was && typeof was === "object") state = { ...state, ...was };
    const bar = el("header"); if (P.pictures && P.pictures.icon) { const mark = el("img", "mark"); mark.src = P.pictures.icon; mark.alt = ""; bar.appendChild(mark); } bar.appendChild(el("strong", "", P.name));
    const tools = el("div", "tools"); bar.appendChild(tools);
    const mark = (name) => { const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"), path = document.createElementNS("http://www.w3.org/2000/svg", "path"); svg.setAttribute("viewBox", "0 0 16 16"); svg.setAttribute("class", "ico"); svg.setAttribute("aria-hidden", "true"); path.setAttribute("d", ICONS[name]); svg.appendChild(path); return svg; };
    const tool = (label, hint, act, icon, tone) => { const b = el("button", "tool" + (tone ? " " + tone : "")); b.type = "button"; if (icon) b.appendChild(mark(icon)); b.appendChild(document.createTextNode(label)); b.title = hint; b.onclick = act; tools.appendChild(b); return b; };
    const past = [];
    const undo = tool("Undo", "Undo the last change", () => { if (past.length) { state = past.pop(); show(); keep(); } }, "undo");
    // What clears everything says so, and asks in the app's own look before it does. LIVE 28 November 2026, looking at an app of hers, she
    // said "Start over" was "a plain white button that looks identical to safe actions".
    tool("Start over", "Put everything back as it began", () => askFirst("Start over?", "Everything in it now will be cleared.", "Clear everything", () => { past.push(state); state = P.initial(); show(); keep(); }), "trash", "danger");
    if ((P.controls || []).length || P.about) tool("Help", "What this does and how", () => help(), "info");
    const main = el("main"); root.append(bar, main);
    let was_shown = null, top = null;
    const unsaid = (v) => { if (v.tag === "#") return v; return { ...v, kids: v.kids.filter((k) => !(k.tag === "h1" && words(k) === P.name)).map(unsaid) }; };
    // A dialog that comes up takes the keys: what is pressed is in it, and Escape presses its first button, which leaves things as they were.
    let asked = null;
    const asking = () => { const d = main.querySelector("[role=dialog]"); if (d && d !== asked) { const first = d.querySelector("button, input, select, textarea"); if (first) first.focus(); } asked = d; };
    function show() { undo.disabled = !past.length; try { const v = unsaid(shown(state)); if (!was_shown || was_shown.tag !== v.tag) { main.textContent = ""; top = make(v); main.appendChild(top); } else fit(top, was_shown, v); was_shown = v; asking(); } catch (e) { problem(e); } }
    function send(event, remember) { try { const before = remember ? JSON.parse(JSON.stringify(state)) : null; state = apply(state, event); if (before && JSON.stringify(before) !== JSON.stringify(state)) { past.push(before); if (past.length > 100) past.shift(); } show(); keep(); } catch (e) { problem(e); } }
    const up = (t, what) => { while (t && t !== main && t["__" + what] == null) t = t.parentNode; return t && t !== main ? t : null; };
    main.addEventListener("click", (e) => { const t = up(e.target, "do"); if (t && t.tagName !== "INPUT") send({ type: "do", name: t.__do[0], value: t.__do[1] }, true); });
    main.addEventListener("change", (e) => { const t = e.target; if (t.__do) send({ type: "do", name: t.__do[0], value: t.__do[1] }, true); else if (t.__in && (t.type === "checkbox" || t.tagName === "SELECT")) send({ type: "input", name: t.__in, value: t.type === "checkbox" ? t.checked : t.value }, true); });
    main.addEventListener("input", (e) => { const t = e.target; if (t.__in && t.type !== "checkbox" && t.tagName !== "SELECT") send({ type: "input", name: t.__in, value: t.__number ? Number(t.value) : t.value }); });
    main.addEventListener("keydown", (e) => { const t = e.target; if (e.key === "Enter" && t.__enter && t.tagName !== "TEXTAREA") send({ type: "do", name: t.__enter, value: null }, true); });
    main.addEventListener("pointerdown", (e) => { const t = up(e.target, "point"); if (t) { const r = t.getBoundingClientRect(); send({ type: "do", name: t.__point, value: { x: ((e.clientX - r.left) * t.width) / r.width, y: ((e.clientY - r.top) * t.height) / r.height } }, true); } });
    window.addEventListener("keydown", (e) => { if (asked) { if (e.key === "Escape") { const first = asked.querySelector("button"); if (first) first.click(); e.preventDefault(); } else if (e.key === "Tab") { const all = [...asked.querySelectorAll("button, input, select, textarea")], at = all.indexOf(document.activeElement); if (all.length) { all[(at + (e.shiftKey ? all.length - 1 : 1)) % all.length].focus(); e.preventDefault(); } } return; } if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return; send({ type: "key", key: key(e), down: true }); });
    show();
    if (P.ticks) { let last = performance.now(); setInterval(() => { const now = performance.now(), dt = Math.min(1, (now - last) / 1000); last = now; send({ type: "tick", dt, now: Date.now() }); }, 1000 / (P.ticks === true ? 10 : P.ticks)); }
  }
  function help() {
    const back = el("div", "over help"); const card = el("div", "panel"); back.appendChild(card);
    card.appendChild(el("h2", "", P.name)); if (P.about) card.appendChild(el("p", "", P.about));
    const list = el("dl"); (P.controls || []).forEach((c) => { list.append(el("dt", "", shownKey(c.input)), el("dd", "", c.does)); }); card.appendChild(list);
    const close = el("button", "primary", "Close"); close.onclick = () => back.remove(); card.appendChild(close); document.body.appendChild(back); close.focus();
  }
  const shownKey = (k) => ({ ArrowLeft: "←", ArrowRight: "→", ArrowUp: "↑", ArrowDown: "↓", pointer: "Mouse", Space: "Space" }[k] || (String(k).length === 1 ? String(k).toUpperCase() : String(k)));

  function openGame() {
    document.body.className = "game";
    const hud = el("div", "hud"), canvas = el("canvas"), stage = el("div", "stage"); stage.appendChild(canvas); root.append(hud, stage);
    const ctx = canvas.getContext("2d");
    let mode = "start", over = null, acc = 0, last = 0, stopped = false;
    function size() { const r = stage.getBoundingClientRect(), s = Math.min(r.width / W, r.height / H), d = window.devicePixelRatio || 1; canvas.style.width = W * s + "px"; canvas.style.height = H * s + "px"; canvas.width = Math.round(W * s * d); canvas.height = Math.round(H * s * d); }
    function begin() { state = P.initial(); over = null; mode = "play"; acc = 0; stage.querySelectorAll(".over").forEach((x) => x.remove()); play("blip"); }
    function panel(title, lines, label, act) { stage.querySelectorAll(".over").forEach((x) => x.remove()); const back = el("div", "over"), card = el("div", "panel"); card.appendChild(el("h1", "", title)); lines.forEach((l) => card.appendChild(l)); const b = el("button", "primary", label); b.onclick = act; card.appendChild(b); back.appendChild(card); stage.appendChild(back); b.focus(); }
    function controls() { const list = el("dl"); (P.controls || []).forEach((c) => list.append(el("dt", "", shownKey(c.input)), el("dd", "", c.does))); list.append(el("dt", "", "P"), el("dd", "", "pause"), el("dt", "", "M"), el("dd", "", "sound on or off")); return list; }
    function send(event) { try { state = apply(state, event); } catch (e) { if (!stopped) problem(e); stopped = true; } }
    function where(e) { const r = canvas.getBoundingClientRect(); pointer.x = Math.max(0, Math.min(W, ((e.clientX - r.left) * W) / r.width)); pointer.y = Math.max(0, Math.min(H, ((e.clientY - r.top) * H) / r.height)); }
    function pause() { if (mode === "play") { mode = "pause"; panel("Paused", [controls()], "Go on", () => { mode = "play"; stage.querySelectorAll(".over").forEach((x) => x.remove()); }); } else if (mode === "pause") { mode = "play"; stage.querySelectorAll(".over").forEach((x) => x.remove()); } }
    window.addEventListener("keydown", (e) => { const k = key(e);
      if (k === "p" || k === "Escape") return pause(); if (k === "m") { muted = !muted; return; }
      if (mode === "start" || mode === "end") { if (k === "Space" || k === "Enter") { e.preventDefault(); begin(); } return; }
      if (mode !== "play") return; if (k.startsWith("Arrow") || k === "Space") e.preventDefault();
      if (!held[k]) { held[k] = true; send({ type: "key", key: k, down: true, held, pointer }); } });
    window.addEventListener("keyup", (e) => { const k = key(e); delete held[k]; if (mode === "play") send({ type: "key", key: k, down: false, held, pointer }); });
    window.addEventListener("blur", () => { for (const k of Object.keys(held)) delete held[k]; if (mode === "play") pause(); });
    canvas.addEventListener("pointermove", (e) => { where(e); if (mode === "play") send({ type: "pointer", x: pointer.x, y: pointer.y, down: pointer.down, moved: true, held, pointer }); });
    canvas.addEventListener("pointerdown", (e) => { where(e); pointer.down = true; if (mode === "play") send({ type: "pointer", x: pointer.x, y: pointer.y, down: true, held, pointer }); });
    window.addEventListener("pointerup", () => { pointer.down = false; if (mode === "play") send({ type: "pointer", x: pointer.x, y: pointer.y, down: false, held, pointer }); });
    window.addEventListener("resize", size); size();
    state = P.initial();
    const about = []; if (P.about) about.push(el("p", "", P.about)); about.push(controls()); if (best.get()) about.push(el("p", "note", "Best so far: " + best.get()));
    panel(P.name, about, "Play", begin);
    function frame(now) {
      const dt = Math.min(0.1, (now - last) / 1000 || 0); last = now;
      if (mode === "play" && !stopped) { acc += dt; while (acc >= STEP) { send({ type: "tick", dt: STEP, held, pointer }); acc -= STEP; }
        try { over = ended(state); } catch (e) { problem(e); }
        if (over) { mode = "end"; const n = scoreOf(state), was = best.get(); if (P.score && n > was) best.set(n); play(over.won ? "win" : "lose");
          const lines = []; if (over.say) lines.push(el("p", "", over.say)); if (P.score) { lines.push(el("div", "big", String(n))); lines.push(el("p", "note", n > was ? "A new best!" : "Best: " + was)); }
          panel(over.won ? "You did it" : "Game over", lines, "Play again", begin); } }
      ctx.setTransform(canvas.width / W, 0, 0, canvas.height / H, 0, 0);
      try { P.draw(state, pen(ctx, now / 1000)); } catch (e) { if (!stopped) problem(e); stopped = true; }
      const rows = (P.hud ? P.hud(state) : []).slice(0, 6); if (P.score && !rows.some((r) => /score/i.test(r.label))) rows.unshift({ label: "Score", value: scoreOf(state) });
      const said = rows.map((r) => `${r.label}\u0000${r.value}`).join("\u0001"); if (hud.__said !== said) { hud.__said = said; hud.textContent = ""; rows.forEach((r) => { const c = el("div", "hud-item"); c.append(el("span", "", r.label), el("strong", "", String(r.value))); hud.appendChild(c); }); const b = el("button", "tool", "Pause"); b.onclick = pause; hud.appendChild(b); }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  // ── trying a program without a window ──────────────────────────────────────────────────────────────────────────
  const copy = (x) => JSON.parse(JSON.stringify(x));
  function run(s, seconds, inputs) {
    // inputs: { held: {key: true}, pointer: {x, y, down} }
    const keys = (inputs && inputs.held) || {}, at = (inputs && inputs.pointer) || NONE;
    for (const k of Object.keys(keys)) s = apply(s, { type: "key", key: k, down: true, held: keys, pointer: at });
    if (inputs && inputs.pointer) { s = apply(s, { type: "pointer", x: at.x, y: at.y, down: false, moved: true, held: keys, pointer: { ...at, down: false } }); if (at.down) s = apply(s, { type: "pointer", x: at.x, y: at.y, down: true, held: keys, pointer: at }); }
    for (let i = Math.round(seconds / STEP); i > 0; i--) { s = apply(s, { type: "tick", dt: STEP, held: keys, pointer: at }); if (P.over && i % 6 === 0 && ended(s)) break; }
    for (const k of Object.keys(keys)) s = apply(s, { type: "key", key: k, down: false, held: {}, pointer: at });
    if (inputs && inputs.pointer && at.down) s = apply(s, { type: "pointer", x: at.x, y: at.y, down: false, held: {}, pointer: { ...at, down: false } });
    return s;
  }
  function wrong(x, path, depth) {
    if (typeof x === "number") return Number.isFinite(x) ? null : `${path} became ${x}`;
    if (x && typeof x === "object") { if (depth > 12) return `${path} refers to itself`; const keys = Object.keys(x); if (keys.length > 5000) return `${path} grew to ${keys.length} things`;
      for (const k of keys) { const w = wrong(x[k], `${path}.${k}`, depth + 1); if (w) return w; } return null; }
    return typeof x === "function" ? `${path} is a function: state is plain data` : null;
  }
  const inputsOf = () => { const keys = (P.controls || []).map((c) => c.input).filter((i) => i && i !== "pointer"); const ways = keys.map((k) => ({ held: { [k]: true } })); if ((P.controls || []).some((c) => c.input === "pointer")) for (const x of [0.15, 0.5, 0.85]) for (const y of [0.15, 0.5, 0.85]) for (const down of [false, true]) ways.push({ pointer: { x: W * x, y: H * y, down } }); return ways; };

  // What tells one situation in a program from another: its flags and words, its counts, how many of each thing there are
  // and how many of them are marked, what is on its board, and roughly where what is moved about is.
  function situation(s) {
    const out = [], plain = (x) => x === null || typeof x === "boolean" || typeof x === "string" || Number.isInteger(x);
    const count = (n) => (n < 0 ? "-" + count(-n) : n <= 4 ? String(n) : "~" + Math.ceil(Math.log2(n)));
    const of = (x, path, depth) => {
      if (x === null || typeof x === "boolean") out.push(path + x);
      else if (typeof x === "string") out.push(path + x.slice(0, 16));
      else if (typeof x === "number") { const by = path.endsWith(".x") ? W / 12 : path.endsWith(".y") ? H / 8 : 0; if (by) out.push(path + Math.round(x / by)); else if (Number.isInteger(x)) out.push(path + count(x)); }
      else if (Array.isArray(x)) {
        out.push(path + "#" + count(x.length));
        const flat = x.flat();
        if (flat.length && flat.length <= 400 && flat.every(plain)) out.push(flat.join(","));
        else { const marks = {}; for (const y of x.slice(0, 200)) if (y && typeof y === "object") for (const k of Object.keys(y)) if (typeof y[k] === "boolean" && y[k]) marks[k] = (marks[k] || 0) + 1; for (const k of Object.keys(marks)) out.push(path + "." + k + "#" + count(marks[k])); }
      } else if (typeof x === "object" && depth < 3) for (const k of Object.keys(x).slice(0, 40)) of(x[k], path + "." + k, depth + 1);
    };
    of(s, "", 0); return out.join("|");
  }
  // Playing it to find what can come of it: from each kind of situation come upon, the least gone on from first, a few
  // things are tried, and every new situation is one more to go on from. What takes two steps is found by its first.
  function explore(most, enough) {
    const began = performance.now(), one = inputsOf(), held = one.filter((w) => w.held).slice(0, 6), ways = [null, ...one];
    for (let i = 0; i < held.length; i++) for (let j = i + 1; j < held.length; j++) ways.push({ held: { ...held[i].held, ...held[j].held } });
    const points = one.some((w) => w.pointer); let z = 11; const r = () => (z = (z * 48271) % 2147483647) / 2147483647, any = (list) => list[Math.floor(r() * list.length)];
    const places = (s) => { const at = []; const of = (x, d) => { if (!x || typeof x !== "object" || d > 3 || at.length > 24) return; if (Number.isFinite(x.x) && Number.isFinite(x.y)) at.push(x); for (const k of Object.keys(x)) of(x[k], d + 1); }; of(s, 0); return at; };
    seed = 7; const first = P.initial(), got = { base: scoreOf(first), score: -Infinity, over: null, won: null, ends: 0, tries: 0, newest: 0, does: new Set() };
    const known = new Map(), all = []; const put = (s) => { const k = situation(s); if (!known.has(k)) { const c = { s: copy(s), seed, seen: 0, score: scoreOf(s) }; known.set(k, c); all.push(c); got.newest = got.tries; } };
    put(first);
    while (performance.now() - began < most && all.length < 9000 && got.tries - got.newest < 300 && !(enough && enough(got, all))) {
      let from = any(all); for (let i = 0; i < 5; i++) { const c = any(all); if (c.seen < from.seen || (c.seen === from.seen && c.score > from.score)) from = c; }
      from.seen++; got.tries++; seed = (from.seed + from.seen * 7919) | 0; let s = copy(from.s), way = any(ways);
      for (let i = 0; i < 8; i++) {
        if (i && r() < 0.5) way = any(ways);
        if (points && r() < 0.4) { const at = places(s); if (at.length) { const p = any(at); way = { pointer: { x: Math.max(0, Math.min(W, p.x)), y: Math.max(0, Math.min(H, p.y)), down: r() < 0.6 } }; } }
        const unsure = way && (way.pointer ? ["pointer"] : Object.keys(way.held)).find((k) => !got.does.has(k));
        if (unsure) { // the same with it and without it
          const sd = seed, rest = way.held ? Object.fromEntries(Object.entries(way.held).filter(([k]) => k !== unsure)) : {}, without = JSON.stringify(run(copy(s), 0.3, Object.keys(rest).length ? { held: rest } : null));
          seed = sd; s = run(s, 0.3, way); if (JSON.stringify(s) !== without) got.does.add(unsure);
        } else s = run(s, 0.3, way);
        got.score = Math.max(got.score, scoreOf(s));
        const e = P.over ? ended(s) : null;
        if (e) { got.ends++; got.over = got.over || e; if (e.won) got.won = e; break; }
        put(s);
      }
    }
    got.all = all; return got;
  }
  // What changed in the program between two states, and what a set of states ran between.
  function changed(a, b) { const out = []; const of = (x, y, path, depth) => { if (out.length > 14 || JSON.stringify(x) === JSON.stringify(y)) return; if (x && y && typeof x === "object" && typeof y === "object" && depth < 4 && Array.isArray(x) === Array.isArray(y)) { if (Array.isArray(x) && x.length !== y.length) out.push(`${path} went from ${x.length} long to ${y.length}`); for (const k of new Set([...Object.keys(x), ...Object.keys(y)])) if (!Array.isArray(x) || (k in x && k in y)) of(x[k], y[k], path ? `${path}.${k}` : k, depth + 1); return; } const say = (v) => { const t = typeof v === "number" ? String(Math.round(v * 100) / 100) : JSON.stringify(v); return t === undefined ? "nothing" : t.length > 60 ? t.slice(0, 60) + "…" : t; }; out.push(`${path} ${say(x)} to ${say(y)}`); }; of(a, b, "", 0); return out; }
  function ran(states) { const seen = {}; for (const s of states) { const of = (x, path, depth) => { if (typeof x === "boolean") x = Number(x); if (Array.isArray(x)) { path += " (how many)"; x = x.length; } if (typeof x === "number") { const v = seen[path] || (seen[path] = [x, x, path.split(".").length]); v[0] = Math.min(v[0], x); v[1] = Math.max(v[1], x); } else if (x && typeof x === "object" && depth < 2) for (const k of Object.keys(x).slice(0, 30)) of(x[k], path ? `${path}.${k}` : k, depth + 1); }; of(s, "", 0); } const say = (n) => Math.round(n * 10) / 10; return Object.entries(seen).sort((a, b) => a[1][2] - b[1][2]).map(([k, v]) => (v[0] === v[1] ? `${k} stayed ${say(v[0])}` : `${k} ${say(v[0])} to ${say(v[1])}`)).slice(0, 22).join("; "); }

  let goesOn = null; const later = () => { if (!goesOn) { try { seed = 7; goesOn = Object.keys(run(P.initial(), 2, null)); } catch (e) { goesOn = []; } } return goesOn; };
  function tester(named) {
    const only = (s) => { if (named) for (const k of Object.keys(s)) if (Array.isArray(s[k]) && !named.includes(k)) s[k] = []; return s; };
    emptied = named ? only : null; seed = 7; played.length = 0; let s = only(P.initial());
    const fail = (why) => { throw new Error(why); };
    const nameOf = (v) => String(v.props["aria-label"] || words(v) || v.props.placeholder || "").toLowerCase().trim();
    const whereIn = (name, want) => name.search(new RegExp(`(?<![\\p{L}\\p{N}])${want.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?![\\p{L}\\p{N}])`, "u")), isWords = (name, want) => whereIn(name, want) >= 0;
    const usable = (v, what) => what(v) && !v.props.disabled;
    const called = (tree, want, what) => { const all = every(tree).filter((v) => usable(v, what)); return all.find((v) => nameOf(v) === want) || all.find((v) => isWords(nameOf(v), want)) || (want.length > 3 && all.find((v) => nameOf(v).startsWith(want))) || null; };
    // What stands with some words: the smallest part of what is shown that has them and something to use, as a card has.
    const beside = (tree, want, what) => { let found = null; const of = (v) => { if (v.tag === "#") return; const has = nameOf(v).includes(want) && isWords(nameOf(v), want); if (!has) return; const all = every(v).filter((n) => n !== v && usable(n, what)); if (usable(v, what)) found = v; else if (all.length) found = all[0]; v.kids.forEach(of); }; tree.kids.forEach(of); return found; };
    let prior = null;  // how it stood before the last thing done
    // Found as a person finds it: by its name, by what it stands with, or behind a button that shows it and takes nothing away.
    const find = (label, what, by, soft) => { const want = String(label).toLowerCase().trim(), here = shown(s); const now = called(here, want, what) || (by && beside(here, want, what)); if (now) return now;
      const size = (x) => (x && typeof x === "object" ? Object.values(x).reduce((n, y) => n + (Array.isArray(y) ? y.length : 0) + size(y), 0) : 0), held = size(s); let edge = [s];
      for (let deep = 0; deep < 2; deep++) { const next = []; for (const from of edge) for (const v of every(shown(from)).filter((n) => n.props.do && !n.props.disabled).slice(0, 24)) { let then; try { then = apply(copy(from), { type: "do", name: v.props.do[0], value: v.props.do[1] }); } catch (e) { continue; } if (size(then) !== held || JSON.stringify(then) === JSON.stringify(from)) continue; const got = called(shown(then), want, what); if (got) { act(() => then); return called(shown(s), want, what); } if (next.length < 12) next.push(then); } edge = next; }
      if (soft) return null;
      // What is named more fully than it is called, as a "confirm delete" that says Confirm: the one of those that has just come,
      // where one has, and the one called by most of it.
      if (by) { const before = new Set(prior ? every(shown(JSON.parse(prior))).map(nameOf) : []), names = every(here).filter((v) => usable(v, what) && nameOf(v).length > 2 && isWords(want, nameOf(v))), come = names.filter((v) => !before.has(nameOf(v))), one = (come.length ? come : names).sort((a, b) => nameOf(b).length - nameOf(a).length)[0]; if (one) return one; }
      if (every(here).some((v) => what(v) && v.props.disabled && nameOf(v) === want)) fail(`“${label}” cannot be used now`);
      const all = every(here).filter((v) => usable(v, what)), why = `there is no “${label}” to use; there are: ${all.map((v) => v.props["aria-label"] || words(v)).slice(0, 12).join(", ")}`;
      // What it is worked by is in its design, which the test and the program are both written to: a label the design has not is the test's own. One that comes of what is in it ("Take rice") is told by the words the design's has.
      const designed = (P.controls || []).map((c) => String(c.input).toLowerCase().trim()).filter(Boolean); if (!isGame() && designed.length && !designed.some((d) => d === want || isWords(want, d) || isWords(d, want) || d.split(/\s+/).some((w) => w.length > 2 && isWords(want, w)))) { const e = new Error(`the test uses a control called “${label}”, and the design has none of that name; it has: ${designed.join(", ")}`); e.theTest = true; throw e; }
      return fail(why); };
    // A state that is set is set over one that has begun: what a test does not name is as it is at the beginning.
    // What the test put in each control, while it is there as it was put, and all it put under each label.
    const put = new Map(), putBy = new Map(), done = new Set(), keyOf = (n) => JSON.stringify(n.props.in), valueOf = (n) => String(n.props.value ?? "").toLowerCase();
    const still = () => { if (!put.size) return; const now = new Map(every(shown(s)).filter((n) => n.props.in).map((n) => [keyOf(n), valueOf(n)])); for (const [k, v] of put) if (now.get(k) !== v) put.delete(k); };
    const wasPut = (label, v, value) => { const k = keyOf(v), said = String(value).toLowerCase(), l = String(label).toLowerCase().trim(); put.set(k, said); done.add(k); putBy.set(l, [...(putBy.get(l) || []), said]); };
    let first = null, did = 0; const began = Object.keys(s), act = (f) => { only(s); if (!first) first = copy(s); prior = isGame() ? null : JSON.stringify(s); s = f(s); did++; still(); if (watching) watching(s); return t; };
    // What each control shown sets, seen by setting it: a label that no control has may be what one of them sets. A list kept
    // in a form is set by controls that are there once there is something in it to set, so what adds to it is tried first.
    // LIVE 24 November 2026 the design had a control called "Beneficiaries" and the program a row for each of them, and
    // sixteen of the twenty-one tests that failed said there was no "Beneficiaries" to use.
    const stem = (w) => w.toLowerCase().replace(/(ies|y)$/, "").replace(/(es|s|e)$/, ""), wordsIn = (x) => String(x).replace(/([a-z])([A-Z])/g, "$1 $2").split(/[^\p{L}]+/u).filter((w) => w.length > 1).map(stem);
    const optionsOf = (v) => v.kids.filter((k) => k.tag === "option").map((k) => ({ value: String(k.props.value), said: words(k) }));
    const meant = (v, value) => { if (v.tag === "select") { const want = String(value).toLowerCase().trim(); const all = optionsOf(v), called = all.filter((o) => isWords(o.said.toLowerCase(), want)); return (all.find((o) => o.value.toLowerCase() === want || o.said.toLowerCase() === want) || (want.length > 1 && called.length === 1 && called[0]) || {}).value; } if (v.props.number) return Number.isFinite(Number(value)) ? Number(value) : undefined; return v.props.type === "checkbox" ? !!value : String(value); };
    const markFor = (v) => (v.tag === "select" ? (optionsOf(v).find((o) => o.value !== String(v.props.value)) || {}).value : v.props.number ? 7919 : v.props.type === "checkbox" ? !v.props.checked : v.props.type === "date" ? "2031-03-07" : "zqx");
    const parted = (a, b, path, out) => { if (out.length > 6 || JSON.stringify(a) === JSON.stringify(b)) return out; if (a && b && typeof a === "object" && typeof b === "object") { for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) parted(a[k], b[k], [...path, k], out); } else out.push(path); return out; };
    const setsOf = (from, v) => { const mark = markFor(v); if (mark === undefined) return []; try { return parted(from, apply(copy(from), { type: "input", name: v.props.in, value: mark }), [], []); } catch (e) { return []; } };
    const setting = (from, want, value) => every(shown(from)).filter((n) => n.props.in && !n.props.disabled && meant(n, value) !== undefined && setsOf(from, n).some((path) => path.some((k) => { const has = wordsIn(k); return want.length && want.every((w) => has.includes(w)); })));
    const emptyFirst = (all) => all.find((n) => n.props.number ? !Number(n.props.value) : n.tag === "select" ? !optionsOf(n).some((o) => o.value === String(n.props.value) && o.value !== "") : !String(n.props.value || "")) || null;
    // Puts a value where what a label names is kept; false where nothing shown keeps it.
    const putWhere = (label, value, again) => { const want = wordsIn(label); let all = setting(s, want, value), v = again ? null : emptyFirst(all) || (all.length === 1 ? all[0] : null);
      if (!v) for (const n of every(shown(s)).filter((x) => x.props.do && !x.props.disabled).slice(0, 24)) { let then; try { then = apply(copy(s), { type: "do", name: n.props.do[0], value: n.props.do[1] }); } catch (e) { continue; } const more = setting(then, want, value); if (more.length > all.length) { act(() => then); v = emptyFirst(more) || more[more.length - 1]; break; } }
      v = v || all[all.length - 1]; if (!v) return false; const set = meant(v, value); act((x) => apply(x, { type: "input", name: v.props.in, value: set })); wasPut(label, v, set); return true; };
    // Of several controls of one name, the one somebody would take: one that is empty and not put in yet, then one that is
    // empty, then one not put in, then the last. LIVE 25 November 2026 a test put three people in the rows of a list, a row
    // at a time, and each was put over the one before, in the first row.
    const ofThem = (v) => { const same = every(shown(s)).filter((n) => n.props.in && !n.props.disabled && nameOf(n) === nameOf(v)); if (same.length < 2) return v; const empty = same.filter((n) => emptyFirst([n])), fresh = (n) => !done.has(keyOf(n)); return empty.find(fresh) || empty[0] || same.find(fresh) || same[same.length - 1]; };
    // What is not offered is put in by nobody. Where the program knows of it, or the test put it in before, it should have been
    // offered; where nothing does, the test asks for what nobody could do.
    const refuse = (label, v, value) => { const e = new Error(`“${label}” takes one of ${optionsOf(v).map((o) => o.said || o.value).filter(Boolean).join(", ")}, so nobody could put “${String(value)}” in it`), known = String(value).toLowerCase(); e.theTest = !(JSON.stringify(s).toLowerCase().includes(`"${known}"`) || [...putBy.values()].some((all) => all.includes(known))); throw e; };
    const set = (label, value) => { const named = find(label, (n) => n.props.in, false, true), v = named && ofThem(named);
      if (v && v.tag === "select" && !Array.isArray(value) && optionsOf(v).length && meant(v, value) === undefined && String(value) !== "") refuse(label, v, value);
      if (v && !Array.isArray(value)) { const ready = v.props.number ? Number(value) : v.props.type === "checkbox" ? !!value : v.tag === "select" ? meant(v, value) ?? String(value) : String(value); act((x) => apply(x, { type: "input", name: v.props.in, value: ready })); wasPut(label, v, ready); return t; }
      if (!isGame() && (Array.isArray(value) ? value.length > 0 && value.every((one, i) => putWhere(label, one && typeof one === "object" ? Object.values(one)[0] : one, i > 0)) : putWhere(label, value, false))) return t;
      // What chooses among some things takes one of them: nobody could put another in it.
      const takes = !isGame() && !Array.isArray(value) ? every(shown(s)).filter((n) => n.tag === "select" && meant(n, value) === undefined && setsOf(s, n).some((path) => path.some((k) => { const has = wordsIn(k), want = wordsIn(label); return want.length && want.every((w) => has.includes(w)); })))[0] : null;
      if (takes) refuse(label, takes, value);
      return find(label, (n) => n.props.in); };
    const t = { get s() { return whole(s); }, set s(v) { s = only({ ...P.initial(), ...v }); }, W, H,
      send(e) { return act((s) => apply(s, e)); },
      tick(seconds) { return act((s) => run(s, seconds == null ? STEP : seconds, null)); },
      hold(k, seconds) { return act((s) => run(s, seconds == null ? 0.1 : seconds, { held: { [k]: true } })); },
      press(k) { return t.hold(k, STEP); },
      point(x, y, seconds) { return act((s) => run(s, seconds == null ? STEP : seconds, { pointer: { x, y, down: false } })); },
      click(x, y) { if (typeof x === "string") { const v = find(x, (n) => n.props.do, true); return act((s) => apply(s, { type: "do", name: v.props.do[0], value: v.props.do[1] })); } return act((s) => run(s, STEP, { pointer: { x, y, down: true } })); },
      type(label, text) { return set(label, text); },
      choose(label, value) { return t.type(label, value); },
      enter(label) { const v = find(label, (n) => n.props.enter); return act((s) => apply(s, { type: "do", name: v.props.enter, value: null })); },
      sees(text) { if (isGame()) { const said = [...t.draw().filter((c) => c.what === "text").map((c) => c.args[0]), ...(P.hud ? P.hud(s) : []).flatMap((r) => [r.label, r.value]), ...(P.score ? ["score", scoreOf(s)] : [])].join(" ").toLowerCase(); return said.includes(String(text).toLowerCase()); } const all = words(shown(s)).toLowerCase(); return every(shown(s)).some((v) => String(v.props.value || "").toLowerCase().includes(String(text).toLowerCase())) || all.includes(String(text).toLowerCase()); },
      // That something is not seen: what the test put in a control, and is there as it was put, is not the program showing
      // it, and a thing put in place of another is not the other seen again for having its words in it.
      unseen(text) { if (isGame()) return !t.sees(text); const want = String(text).toLowerCase(), longer = [];
        for (const said of putBy.values()) said.forEach((one, i) => { if (one === want) longer.push(...said.slice(i + 1).filter((later) => later !== want && later.includes(want))); });
        const less = (x) => longer.reduce((left, one) => left.split(one).join(" "), x);
        return !less(words(shown(s)).toLowerCase()).includes(want) && !every(shown(s)).some((v) => less(valueOf(v)).includes(want) && !(v.props.in && put.get(keyOf(v)) === valueOf(v))); },
      // A function of the program's own, given these and nothing else.
      call(name, ...args) { if (typeof P[name] !== "function") { const e = new Error(`the test calls ${name}, and the program has no such function`); e.theTest = true; throw e; } return P[name](...args); },
      // Which of two things shown comes first.
      comesBefore(a, b) { const all = isGame() ? "" : words(shown(s)).toLowerCase(), one = whereIn(all, String(a).toLowerCase()), two = whereIn(all, String(b).toLowerCase()); return one >= 0 && two >= 0 && one < two; },
      get shown() { return isGame() ? t.draw().filter((c) => c.what === "text").map((c) => c.args[0]).join(" ") : words(shown(s)); },
      // What is drawn for the state as it is: each call of the pen, with what it was given.
      draw() { const calls = []; const canvas = document.createElement("canvas"); canvas.width = W; canvas.height = H; const real = pen(canvas.getContext("2d"), 1);
        const g = new Proxy(real, { get: (o, k) => (typeof o[k] === "function" ? (...args) => { calls.push({ what: String(k), args }); if (k === "at") { const [x, y, turn, act] = args; return real.at(x, y, turn, () => act(g)); } return o[k](...args); } : o[k]) });
        P.draw(s, g); return calls; },
      get score() { return scoreOf(s); }, get over() { return ended(s) || false; }, get won() { const e = ended(s); return !!(e && e.won); }, get played() { return played.slice(); },
      expect, initial() { s = P.initial(); return s; }, update(e) { return t.send(e); } };
    const told = { did: () => did, changed: () => (first ? changed(first, s) : []), asked: (body) => readIn(body).filter((k) => !began.includes(k) && !(P.designed && k in P.designed) && !later().includes(k)), held: () => Object.keys(s) };
    return new Proxy(t, { get(o, k) { if (k === "__told") return told; if (k in o || typeof k === "symbol") return o[k];
      // A function of the program's own, asked of the test: given the state as it is, where it was not given one.
      if (typeof P[k] === "function") return (...args) => P[k](...(args.length && args[0] && typeof args[0] === "object" && !Array.isArray(args[0]) ? args : [s, ...args]));
      const e = new Error(`the test asks for t.${String(k)}, and there is no such thing; there are: ${Object.keys(o).join(" ")}`); e.theTest = true; throw e; } });
  }
  // What a test expects: that something is so, or, in the manner of most tests, what something is.
  let waiting = [];
  function expect(value, why) {
    const told = { value, why, judged: false }; waiting.push(told);
    const show = (x) => { try { return JSON.stringify(x); } catch (e) { return String(x); } };
    const same = (a, b) => a === b || (Number.isNaN(a) && Number.isNaN(b)) || (a && b && typeof a === "object" && show(a) === show(b));
    const judge = (ok, what, not) => { told.judged = true; if (ok === !!not) throw new Error(why || `${show(value)} is ${not ? "" : "not "}${what}`); return true; };
    const ways = (not) => ({ toBe: (x) => judge(same(value, x), show(x), not), toEqual: (x) => judge(same(value, x), show(x), not), toStrictEqual: (x) => judge(same(value, x), show(x), not),
      toBeCloseTo: (x, n = 2) => judge(Math.abs(value - x) < 0.5 * 10 ** -n, `near ${x}`, not), toBeGreaterThan: (x) => judge(value > x, `more than ${x}`, not), toBeGreaterThanOrEqual: (x) => judge(value >= x, `at least ${x}`, not),
      toBeLessThan: (x) => judge(value < x, `less than ${x}`, not), toBeLessThanOrEqual: (x) => judge(value <= x, `at most ${x}`, not), toBeTruthy: () => judge(!!value, "so", not), toBeFalsy: () => judge(!value, "not so", not),
      toBeNull: () => judge(value === null, "null", not), toBeUndefined: () => judge(value === undefined, "undefined", not), toBeDefined: () => judge(value !== undefined, "defined", not), toContain: (x) => judge(value != null && value.includes(x), `with ${show(x)} in it`, not),
      toHaveLength: (n) => judge(value != null && value.length === n, `${n} long`, not), toMatch: (r) => judge(new RegExp(r).test(value), `like ${r}`, not), toThrow: () => { let threw = false; try { value(); } catch (e) { threw = true; } judge(threw, "one that throws", not); } });
    return { ...ways(false), not: ways(true) };
  }
  function judged() { const left = waiting.filter((w) => !w.judged); waiting = []; for (const w of left) if (!w.value) throw new Error(w.why || "what was expected is not so"); }
  function tryOne(name, act) { const at = performance.now(); try { const note = act(); return { name, ok: true, why: note || "", ms: Math.round(performance.now() - at) }; } catch (e) { return { name, ok: false, got: (e && e.got) || 0, theTest: !!(e && e.theTest), broke: e instanceof TypeError || e instanceof ReferenceError || e instanceof RangeError || e instanceof SyntaxError, why: String((e && e.message) || e).slice(0, 760) + has(String((e && e.message) || e)) + where(e), ms: Math.round(performance.now() - at) }; } }
  // Where something went wrong, by the names of the program's own functions, the innermost first.
  const OURS = new Set(["apply", "run", "once", "explore", "tryOne", "check", "generic", "rules", "tried", "wander", "look", "act", "shown", "judged", "fail", "must", "expect", "judge", "ways", "every", "words", "pen", "tester", "body", "map", "filter", "forEach", "reduce", "some", "every", "find", "eval", "evaluate", "__try", "get", "send", "tick", "hold", "press", "point", "click", "type", "test"]);
  const where = (e) => { const names = []; for (const line of String((e && e.stack) || "").split("\n").slice(1)) { const m = /at (?:async )?(?:Object\.|Proxy\.|PROGRAM\.)?([A-Za-z_$][\w$]*)(?: \[as [^\]]+\])? \(/.exec(line); if (m && !OURS.has(m[1]) && names[names.length - 1] !== m[1]) names.push(m[1]); if (names.length >= 4) break; } return names.length ? ` (in ${names.join(", called by ")})` : ""; };
  // What a piece can be asked for, where it was asked for what it has not.
  const has = (said) => { const m = /\b([A-Za-z_$][\w$]*)\.([A-Za-z_$][\w$]*) is not a function/.exec(said), piece = m && P && P[m[1]]; return piece && typeof piece === "object" && typeof piece[m[2]] !== "function" ? `; ${m[1]} has ${Object.keys(piece).filter((k) => typeof piece[k] === "function").join(", ")}` : ""; };
  const must = (ok, why) => { if (!ok) throw new Error(why); };

  function generic() {
    const out = []; const check = (name, act) => { if (!only || only.includes(name)) out.push(tryOne(name, act)); };
    const ways = inputsOf();
    const wander = (from, seconds, sd, take) => { seed = sd; let s = from === null ? P.initial() : copy(from); let left = seconds; while (left > 0 && !(P.over && ended(s))) { const way = take ? take(s) : ways.length ? ways[Math.floor(rnd() * ways.length)] : null; s = run(s, 0.25, way); left -= 0.25; } return s; };
    check("it begins", () => { seed = 7; const s = P.initial(); must(s && typeof s === "object", "initial() gives no state"); must(JSON.stringify(s) === JSON.stringify(copy(s)), "its state is not plain data"); const w = wrong(s, "state", 0); must(!w, w); if (P.over) must(!ended(s), "it is over as it begins"); });
    if (out[0] && out[0].ok === false) return out;
    if (P.designed) check("its state is as it was designed", () => { seed = 7; const s = whole(P.initial()), lost = Object.keys(P.designed).filter((k) => !(k in s)); must(!lost.length, `initial() has no ${lost.join(", ")}; the design's state has`); });
    if (P.designed && Object.keys(P.designed).length) check("it goes on from its state as it was designed", () => { seed = 7; let s = { ...P.initial(), ...copy(P.designed) }; if (isGame()) { s = run(s, 1.5, null); P.draw(s, pen(document.createElement("canvas").getContext("2d"), 1)); } else shown(s); const w = wrong(s, "state", 0); must(!w, `begun from the designed state, ${w}: what it holds has only the names the design gives it`); });
    if (isGame()) {
      check("it runs without breaking, whatever is pressed", () => { for (let sd = 1; sd <= 5; sd++) { const s = wander(null, 30, sd); const w = wrong(s, "state", 0); must(!w, w); } });
      check("it is the same twice from the same beginning", () => { const a = JSON.stringify(wander(null, 5, 3)), b = JSON.stringify(wander(null, 5, 3)); must(a === b, "two runs from the same beginning went differently: use rnd() and not Math.random()"); });
      let played_ = null; const explored = () => (played_ = played_ || explore(6000, (g, all) => all.length > 60 && (P.controls || []).every((c) => g.does.has(c.input)) && (!P.score || g.score > g.base) && (!P.over || g.over)));
      check("time passes in it", () => { seed = 7; const a = P.initial(), b = run(copy(a), 2, null); must(JSON.stringify(a) !== JSON.stringify(b) || ways.some((w) => { seed = 7; const x = P.initial(); return JSON.stringify(run(copy(x), 1, w)) !== JSON.stringify(x); }), "nothing happens in two seconds, or when anything is pressed"); });
      for (const c of P.controls || []) check(`${shownKey(c.input)} does something (${c.does})`, () => { const g = explored(); if (!g.does.has(c.input)) must(false, `pressing it changes nothing while it is played, in ${g.all.length} different situations come upon`); });
      if (P.score) check("it can be scored in", () => { const g = explored(); must(g.score > g.base, `played ${g.tries} times over, from ${g.all.length} different situations, its score never rose from ${g.base}. While it was played: ${ran(g.all.map((c) => c.s))}`); return `played, it scored ${Math.round(g.score - g.base)}`; });
      if (P.over) check("it can be lost or won", () => { const g = explored(); let e = g.over; if (!e) { seed = 1; const s = run(P.initial(), 240, null); e = ended(s); } must(e, `it never ended: left alone for four minutes, or played ${g.tries} times over. While it was played: ${ran(g.all.map((c) => c.s))}`); });
      if (P.over) check("it is not over before it can be played", () => { seed = 7; const s = run(P.initial(), 2, null); must(!ended(s), "it is over within two seconds of beginning"); });
      check("it is drawn", () => { const c = document.createElement("canvas"); c.width = W; c.height = H; const ctx = c.getContext("2d", { willReadFrequently: true }); const pic = (s, tm) => { ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, W, H); P.draw(s, pen(ctx, tm)); return ctx.getImageData(0, 0, W, H).data; }; seed = 7; const s0 = P.initial(); const a = pic(s0, 0).slice(0); let other = 0, blank = 0; for (let i = 0; i < a.length; i += 4 * 97) { if (a[i] !== a[0] || a[i + 1] !== a[1] || a[i + 2] !== a[2]) other++; if (a[i + 3] === 0) blank++; } const n = a.length / (4 * 97); must(other > n * 0.005, "nothing is drawn but one colour"); must(blank < n * 0.5, "most of it is left undrawn: begin draw with g.clear(colour)"); const s1 = wander(null, 4, 2); const b = pic(s1, 4); let moved = 0; for (let i = 0; i < a.length; i += 4 * 97) if (a[i] !== b[i] || a[i + 1] !== b[i + 1] || a[i + 2] !== b[i + 2]) moved++; must(moved > 0, "the picture is the same after four seconds of play"); for (let sd = 1; sd <= 3; sd++) pic(wander(null, 20, sd), sd); });
      check("the picture is filled wherever play goes", () => { const c = document.createElement("canvas"); c.width = W; c.height = H; const ctx = c.getContext("2d", { willReadFrequently: true }); const most = (s, tm) => { ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = 1; ctx.clearRect(0, 0, W, H); P.draw(s, pen(ctx, tm)); const d = ctx.getImageData(0, 0, W, H).data, seen = new Map(); let n = 0, top = 0, what = ""; for (let i = 0; i < d.length; i += 4 * 53) { const k = d[i + 3] < 8 ? "nothing at all" : "#" + [d[i], d[i + 1], d[i + 2]].map((v) => (v >> 3 << 3).toString(16).padStart(2, "0")).join(""); const m = (seen.get(k) || 0) + 1; seen.set(k, m); n++; if (m > top) { top = m; what = k; } } return { share: top / n, what }; };
        seed = 7; const first = most(P.initial(), 0), all = explored().all, later = []; for (let i = 1; i <= 8; i++) later.push(most(all[Math.floor((all.length - 1) * (i / 8))].s, 3 + i)); later.sort((a, b) => a.share - b.share); const mid = later[4];
        must(mid.share < 0.62 || mid.share < first.share + 0.22, `at the beginning ${Math.round(first.share * 100)}% of the picture is one colour, and where play goes ${Math.round(mid.share * 100)}% of it is ${mid.what}: what fills the view slides out of it as the world moves. Fill it in the view's own place (the scene pieces do, and g.screen does)`); });
      check("showing it changes nothing in it", () => { const g = pen(document.createElement("canvas").getContext("2d"), 1); for (let sd = 1; sd <= 3; sd++) { const s = wander(null, 6 * sd, sd), before = copy(s); P.draw(s, g); if (P.hud) P.hud(s); if (P.score) P.score(s); if (P.over) P.over(s); must(JSON.stringify(s) === JSON.stringify(before), `drawing it changes its state, so how it goes depends on how often it is drawn: ${changed(before, s).slice(0, 4).join("; ")}`); } });
      check("it does not fill up as it goes on", () => { const size = (x) => (x && typeof x === "object" ? Object.values(x).reduce((n, y) => n + 1 + size(y), 0) : 0); seed = 5; let s = P.initial(); const at = []; for (let i = 0; i < 6; i++) { s = run(s, 20, null); at.push(size(s)); if (P.over && ended(s)) return ""; } must(at[5] < 4000 || at[5] < at[1] * 2.2, `left alone, its state goes from ${at[0]} things in it to ${at[5]} in two minutes: what is out of sight is never let go of`); });
      check("it is quick enough", () => { const c = document.createElement("canvas"); c.width = W; c.height = H; const g = pen(c.getContext("2d"), 0); let s = wander(null, 10, 4); const at = performance.now(); for (let i = 0; i < 120; i++) { s = apply(s, { type: "tick", dt: STEP, held: {}, pointer: NONE }); P.draw(s, g); } const ms = (performance.now() - at) / 120; must(ms < 12, `a frame takes ${ms.toFixed(1)} ms`); return `${ms.toFixed(1)} ms a frame`; });
      if (P.hud) check("what it counts is shown", () => { seed = 7; const rows = P.hud(P.initial()); must(Array.isArray(rows) && rows.every((r) => r && r.label != null && r.value != null), "hud() does not give {label, value} rows"); });
    } else {
      // Shown is drawn too: what is painted on a canvas in it is painted, and what breaks as it is painted breaks it.
      const drawnAt = (v) => { try { document.createElement("div").appendChild(make(v)); } catch (e) { must(false, `drawing it raised: ${e && e.message ? e.message : e}`); } };
      const shownAt = (s) => { const v = shown(s); drawnAt(v); const said = words(v); must(said.length > 0, "nothing is shown"); const odd = /\bundefined\b|\bNaN\b|\[object Object\]|\bInfinity\b|\bInvalid Date\b/.exec(said); must(!odd, `it shows “${odd && odd[0]}” in “${odd && said.slice(Math.max(0, odd.index - 40), odd.index + 30)}”`); const bare = every(v).find((n) => n.kids.length === 1 && n.kids[0].tag === "#" && /^(true|false|null)$/.test(n.kids[0].text)); must(!bare, `it shows a bare “${bare && bare.kids[0].text}”`); return v; };
      check("it is shown", () => { seed = 7; shownAt(P.initial()); if (P.designed) shownAt({ ...P.initial(), ...copy(P.designed) }); });
      check("it begins with nothing in it that nobody put there", () => { seed = 7; const s = P.initial(); const mine = Object.keys(P.designed || {}).filter((k) => Array.isArray(P.designed[k]) && Array.isArray(s[k]) && s[k].some((x) => x && typeof x === "object" && "id" in x && Object.values(x).every((v) => v === null || typeof v !== "object"))); must(!mine.length, `it begins with records already in ${mine.join(", ")}: it is to begin empty, for whoever uses it to fill`);
        // What is worked out from what is put in is nothing, too, where nothing has been: the example of the design's state is an example.
        const tells = (x) => (typeof x === "number" ? x !== 0 : x && typeof x === "object" && Object.values(x).some(tells));
        const copied = Object.keys(P.designed || {}).filter((k) => P.designed[k] && typeof P.designed[k] === "object" && tells(P.designed[k]) && JSON.stringify(P.designed[k]) === JSON.stringify(s[k]));
        must(!copied.length, `it begins with the example's own ${copied.join(", ")} (${String(JSON.stringify(s[copied[0]])).slice(0, 90)}), and nobody has put anything in: what is worked out begins at nothing`); });
      const used = new Map();
      const act = (s) => { const v = shownAt(s), all = every(v).filter((n) => (n.props.do && !n.props.disabled) || n.props.in); if (!all.length) return s; const n = all[Math.floor(rnd() * all.length)]; const label = n.props["aria-label"] || words(n); const before = JSON.stringify(s); let next; if (n.props.do) next = apply(s, { type: "do", name: n.props.do[0], value: n.props.do[1] }); else next = apply(s, { type: "input", name: n.props.in, value: n.props.type === "checkbox" ? rnd() < 0.5 : n.tag === "select" && n.kids.length ? n.kids[Math.floor(rnd() * n.kids.length)].props.value : n.props.number ? Math.floor(rnd() * 50) : n.props.type === "date" ? "2026-11-2" + Math.floor(rnd() * 9) : chance.pick(["milk", "a longer thing to do", "7", "", "Ana"]) }); if (JSON.stringify(next) !== before) used.set(label, true); else if (!used.has(label)) used.set(label, false); return next; };
      check("it runs without breaking, whatever is used", () => { for (let sd = 1; sd <= 6; sd++) { seed = sd; let s = P.initial(); for (let i = 0; i < 160; i++) { s = act(s); if (P.ticks && i % 8 === 0) s = apply(s, { type: "tick", dt: 1, now: 1.79e12 + i * 1000 }); const w = wrong(s, "state", 0); must(!w, w); } shownAt(s); } });
      check("the controls it was designed with are there", () => { const designed = (P.controls || []).map((c) => String(c.input).trim()).filter(Boolean); if (!designed.length) return ""; const said = new Set(), over = new Set(), uses = (n) => n.props.do || n.props.in || n.props.enter;
        // A part of it is there by its title, where what it is worked by stands under that: "Add Expense" over a form is one.
        const titles = (v) => { if (v.tag === "#" || !every(v).some(uses)) return; for (const k of v.kids) { if (k.tag !== "#" && !every(k).some(uses)) over.add(words(k).toLowerCase().trim()); titles(k); } };
        const look = (s) => { const tree = shown(s); titles(tree); for (const n of every(tree)) if (uses(n)) said.add(String(n.props["aria-label"] || words(n) || n.props.placeholder || "").toLowerCase()); };
        for (let sd = 1; sd <= 4; sd++) { seed = sd * 5; let s = sd === 4 && P.designed ? { ...P.initial(), ...copy(P.designed) } : P.initial(); for (let i = 0; i < 70; i++) { look(s); s = act(s); } look(s); }
        const gone = () => { const all = [...said]; return designed.filter((d) => { const w = d.toLowerCase(); return !over.has(w) && !all.some((x) => x === w || x.includes(w) || (x && w.includes(x) && x.length > 2)); }); };
        // What is there only once something has been kept is come to by whoever keeps something: the tests do. LIVE 25 November 2026
        // a mend that put in what was missing was not kept, for "Edit" and "Delete" had not been come upon by chance that time.
        for (const [, body] of (gone().length && window.TESTS) || []) { watching = (then) => { try { look(then); } catch (e) { /* what breaks as it is shown is told by what else is tried */ } }; try { body(tester()); } catch (e) { /* what it finds is told where it is tried */ } finally { watching = null; waiting = []; emptied = null; } if (!gone().length) break; }
        const left = gone(); must(!left.length, `the design has ${left.map((g) => `“${g}”`).join(", ")}, and wherever it was used nothing was called that; there are: ${[...said].filter(Boolean).slice(0, 16).join(", ")}`); });
      check("every control does something", () => { const deadNow = () => [...used].filter(([, did]) => !did).map(([label]) => label);
        // What does something only where something has been put in is pressed where the tests have put it: LIVE 25 November 2026
        // a dialog's Cancel was there only once a dish had been begun, and was called dead, and was.
        const tried = (then) => { for (const n of every(shown(then))) { if (!n.props.do || n.props.disabled) continue; const label = n.props["aria-label"] || words(n); if (used.get(label) !== false) continue; let next; try { next = apply(copy(then), { type: "do", name: n.props.do[0], value: n.props.do[1] }); } catch (e) { continue; } if (JSON.stringify(next) !== JSON.stringify(then)) used.set(label, true); } };
        for (const [, body] of (deadNow().length && window.TESTS) || []) { watching = (then) => { try { tried(then); } catch (e) { /* told where it is tried */ } }; try { body(tester()); } catch (e) { /* told where it is tried */ } finally { watching = null; waiting = []; emptied = null; } if (!deadNow().length) break; }
        // What each sends, and whether update looks at it anywhere: a name nothing looks at is a branch that was never written.
        const sends = new Map(); for (let sd = 1; sd <= 2; sd++) { seed = sd; let x = P.initial(); for (let i = 0; i < 60; i++) { for (const n of every(shown(x))) if (n.props.do) sends.set(n.props["aria-label"] || words(n), String(n.props.do[0])); x = act(x); } }
        const read = Object.values(P).filter((f) => typeof f === "function").map(String).filter((f) => !/\bui\.\w+\(/.test(f)).join("\n"), told = (label) => { const name = sends.get(label); return name ? `${label} (sends “${name}”${read.includes(name) ? "" : ", which update never looks at"})` : label; };
        const dead = deadNow(); must(dead.length === 0, `these change nothing: ${dead.slice(0, 8).map(told).join(", ")}`); must(used.size > 0, "there is nothing to use"); });
      // What is true of any program that keeps things, whatever it is for, and that no test of a rule of it says.
      const records = (s) => { const out = []; const of = (x, path, d) => { if (!x || typeof x !== "object" || d > 4) return; if (Array.isArray(x)) { if (x.length && x.every((y) => y && typeof y === "object" && !Array.isArray(y))) out.push([path, x]); x.slice(0, 60).forEach((y, i) => of(y, `${path}[${i}]`, d + 1)); } else for (const k of Object.keys(x)) of(x[k], path ? `${path}.${k}` : k, d + 1); }; of(s, "", 0); return out; };
      const fill = (s, sd) => { seed = sd; const names = ["Ana", "Ben", "rice", "a longer thing"]; for (const n of every(shown(s)).filter((n) => n.props.in)) { const all = n.tag === "select" ? n.kids : null; s = apply(s, { type: "input", name: n.props.in, value: n.props.type === "checkbox" ? true : all ? (all.length ? all[all.length > 1 ? 1 : 0].props.value : "") : n.props.number || n.props.type === "number" ? 12 : n.props.type === "date" ? "2026-11-20" : names[Math.floor(rnd() * names.length)] }); } return s; };
      const buttons = (s) => every(shown(s)).filter((n) => n.props.do && !n.props.disabled);
      // What is drawn as taking something away takes it, or asks first. LIVE 25 November 2026 the buttons that added a flatmate, a
      // plant and an expense were drawn so, and nothing that was tried said it.
      check("what is marked as taking something away does", () => { const seen = new Map(), size = (x) => (x && typeof x === "object" ? Object.values(x).reduce((n, y) => n + (Array.isArray(y) ? y.length : 0) + size(y), 0) : 0);
        const at = (s) => { for (const n of every(shown(s))) { if (!n.props.do || n.props.disabled || !/(^| )danger( |$)/.test(n.props.class || "")) continue; const label = words(n), had = seen.get(label) || { more: 0, less: 0 }; seen.set(label, had); if (had.less) continue; let then; try { then = apply(copy(s), { type: "do", name: n.props.do[0], value: n.props.do[1] }); } catch (e) { continue; } const by = size(then) - size(s); if (by > 0) had.more++; if (by < 0) had.less++; } };
        for (let sd = 1; sd <= 3; sd++) { seed = sd * 29; let s = sd === 3 && P.designed ? { ...P.initial(), ...copy(P.designed) } : P.initial(); for (let i = 0; i < 24; i++) { at(s); if (i % 3 === 0) { const was = seed; at(fill(copy(s), sd + i)); seed = was; } s = act(s); } }
        const wrong = [...seen].filter(([, had]) => had.more > 0 && !had.less).map(([label]) => `“${label}”`); must(!wrong.length, `${wrong.slice(0, 6).join(", ")}: marked as taking something away, and only ever puts something in`); });
      // A control shows what is put in it. Where what is put in changes the state, and the control goes on showing what it did,
      // it is kept where nothing shows it: the row of a list in a form that is stored by the row's own name.
      check("what is put in a control is what it shows", () => { const lost = new Map(), showsOf = (n) => (n.tag === "select" ? String(n.props.value) : n.props.type === "checkbox" ? !!n.props.checked : String(n.props.value ?? ""));
        for (let sd = 1; sd <= 3; sd++) { seed = sd * 17; let s = sd === 3 && P.designed ? { ...P.initial(), ...copy(P.designed) } : P.initial(); for (let i = 0; i < 40 && lost.size < 6; i++) {
          for (const n of every(shown(s)).filter((x) => x.props.in && !x.props.disabled)) { const label = String(n.props["aria-label"] || n.props.placeholder || n.props.in); if (lost.has(label)) continue;
            const other = n.tag === "select" ? n.kids.filter((k) => k.tag === "option").map((k) => String(k.props.value)).find((v) => v !== String(n.props.value)) : n.props.type === "checkbox" ? !n.props.checked : n.props.number || n.props.type === "number" ? 7 : n.props.type === "date" ? "2031-03-07" : "Ana";
            if (other === undefined || String(other) === String(showsOf(n))) continue; let then; try { then = apply(copy(s), { type: "input", name: n.props.in, value: other }); } catch (e) { continue; }
            const now = every(shown(then)).find((x) => x.props.in === n.props.in); if (!now || showsOf(now) !== showsOf(n)) continue; const kept = changed(s, then); if (kept.length) lost.set(label, `“${label}” shows “${showsOf(now)}” after “${other}” is put in it; what was put in it: ${kept.slice(0, 2).join("; ")}, which it does not show`); }
          s = act(s); } }
        if (lost.size) { const e = new Error([...lost.values()].slice(0, 3).join(". ")); e.got = -lost.size; throw e; } });
      check("showing it changes nothing in it", () => { for (let sd = 1; sd <= 3; sd++) { seed = sd; let s = P.initial(); for (let i = 0; i < 30; i++) s = act(s); const before = copy(s); shown(s); shown(s); must(JSON.stringify(s) === JSON.stringify(before), `view changes the state it is given: ${changed(before, s).slice(0, 4).join("; ")}`); } });
      check("each thing in it can be told from the others", () => { const now = Date.now; let at = 1.79e12; Date.now = () => (at += 1000); try { for (let sd = 1; sd <= 4; sd++) { seed = sd * 13; let s = P.initial(); for (let i = 0; i < 140; i++) { if (i % 7 === 0) s = fill(s, sd + i); s = act(s); for (const [path, list] of records(s)) { const ids = list.map((x) => x.id).filter((x) => x !== undefined), same = ids.find((x, j) => ids.indexOf(x) !== j); must(same === undefined, `two things in ${path} have the id ${JSON.stringify(same)}, so that what is done to one is done to the other: an id is one that nothing there has had`); } } } } finally { Date.now = now; } });
      // What it keeps is put in it from how it opens, by its own controls. LIVE 28 November 2026 a dinner planner opened on "No plan yet. Add a
      // guest and a dish to get started" with nothing to add one by, and a reading tracker had nothing to add a book by; she saw the first.
      check("it can be filled from how it opens", () => { const kept = Object.keys(P.designed || {}).filter((k) => Array.isArray(P.designed[k])); if (!kept.length) return "";
        const sizes = (s) => kept.map((k) => (s && Array.isArray(s[k]) ? s[k].length : 0));
        for (let sd = 1; sd <= 4; sd++) { seed = sd; let s = P.initial(); const at = sizes(s); for (let i = 0; i < 60; i++) { s = fill(s, i + sd); s = act(s); const j = sizes(s).findIndex((n, j) => n > at[j]); if (j >= 0) return `${kept[j]} is filled from how it opens`; } }
        must(false, `nothing shown as it opens, used however it may be, puts anything in ${kept.join(" or ")}: there is nothing to begin with`); });
      check("nothing is put in where nothing was filled in", () => { seed = 7; const s0 = P.initial(); const had = Object.fromEntries(records({ d: P.designed || {} }).map(([path, list]) => [path.slice(2), list[0]])); const lists = (s) => Object.fromEntries(Object.keys(s).filter((k) => Array.isArray(s[k])).map((k) => [k, s[k]]));
        for (const b of buttons(s0).slice(0, 30)) { const s1 = apply(copy(s0), { type: "do", name: b.props.do[0], value: b.props.do[1] }), was = lists(s0), now = lists(s1); for (const k of Object.keys(now)) { const example = had[k]; if (!example || !was[k] || now[k].length !== was[k].length + 1) continue; const first = Object.keys(example).find((f) => f !== "id" && typeof example[f] === "string" && example[f].trim() && !/^\d{4}-\d\d-\d\d/.test(example[f])); const it = now[k].find((x) => !was[k].some((y) => JSON.stringify(x) === JSON.stringify(y))); must(!first || !it || String(it[first] ?? "").trim() !== "", `pressing “${b.props["aria-label"] || words(b)}” with nothing filled in puts a thing with no ${first} in ${k}`); } } });
      check("what is put in and taken out again leaves it as it was", () => { for (let sd = 1; sd <= 2; sd++) { seed = sd; let s0 = P.initial(); for (let i = 0; i < sd * 12; i++) { s0 = fill(s0, i + sd); s0 = act(s0); } const sizes = (s) => Object.fromEntries(Object.keys(s).filter((k) => Array.isArray(s[k])).map((k) => [k, s[k].length]));
        const filled = fill(copy(s0), sd); for (const b of buttons(filled).slice(0, 24)) { const s1 = apply(copy(filled), { type: "do", name: b.props.do[0], value: b.props.do[1] }), a = sizes(s0), z = sizes(s1), grew = Object.keys(z).filter((k) => z[k] === a[k] + 1); if (grew.length !== 1 || Object.keys(z).some((k) => k !== grew[0] && z[k] !== a[k])) continue; const k = grew[0], it = s1[k].find((x) => !s0[k].some((y) => JSON.stringify(x) === JSON.stringify(y))); if (!it || typeof it !== "object" || it.id === undefined) continue;
          for (const c of buttons(s1).filter((n) => JSON.stringify(n.props.do[1] ?? null).includes(JSON.stringify(it.id))).slice(0, 8)) { const s2 = apply(copy(s1), { type: "do", name: c.props.do[0], value: c.props.do[1] }); if (!Array.isArray(s2[k]) || s2[k].length !== a[k] || s2[k].some((x) => x && x.id === it.id) || JSON.stringify(s2[k].map((x) => x.id)) !== JSON.stringify(s0[k].map((x) => x.id))) continue;
            const seenBefore = words(shown(s0)), seenAfter = words(shown(fill(copy(s2), sd))), same = (x, y) => JSON.stringify(x) === JSON.stringify(y); const tell = (x, y, path, d) => { if (typeof x === "number" && typeof y === "number" && x !== y) return `${path} is ${y} and was ${x}`; if (x && y && typeof x === "object" && typeof y === "object" && d < 3 && Array.isArray(x) === Array.isArray(y)) for (const f of new Set([...Object.keys(x), ...Object.keys(y)])) { if (same(x[f], y[f])) continue; const t = tell(x[f] ?? 0, y[f] ?? 0, path ? `${path}.${f}` : f, d + 1); if (t) return t; } return null; };
            const maps = (s) => Object.fromEntries(Object.entries(s).filter(([, v]) => v && typeof v === "object" && !Array.isArray(v) && Object.values(v).some((n) => typeof n === "number"))); const off = tell(maps(s0), maps(s2), "", 0);
            must(!off || seenBefore === words(shown(s2)) || seenBefore === seenAfter, `“${b.props["aria-label"] || words(b)}” and then “${c.props["aria-label"] || words(c)}” for what it put in leaves ${off}`); } } } });
      check("it is quick with a lot in it", () => { seed = 3; let s = P.initial(); if (!P.designed) return ""; const big = { ...s }; for (const k of Object.keys(P.designed)) if (Array.isArray(P.designed[k]) && P.designed[k].length && typeof P.designed[k][0] === "object" && Array.isArray(s[k])) big[k] = Array.from({ length: 300 }, (_, i) => ({ ...copy(P.designed[k][i % P.designed[k].length]), ...(P.designed[k][0].id !== undefined ? { id: typeof P.designed[k][0].id === "number" ? i + 1 : `x${i}` } : {}) })); const at = performance.now(); for (let i = 0; i < 4; i++) shown(big); const ms = (performance.now() - at) / 4; must(ms < 160, `with 300 of each thing in it, it takes ${Math.round(ms)} ms to show`); return `${Math.round(ms)} ms`; });
      check("what is in it is kept", () => { seed = 5; let s = P.initial(); for (let i = 0; i < 60; i++) s = act(s); must(words(shown(copy(s))) === words(shown(s)), "it is not the same after being closed and opened"); });
    }
    return out;
  }
  // The names of the state a test reads or sets, from what is written of it: t.s.name, and name of what t.s was called.
  function readIn(body) { const src = String(body), t = (/^\s*(?:async\s*)?(?:function\s*\w*\s*)?\(?\s*([\w$]+)/.exec(src) || [])[1] || "t", found = new Set(), of = (what) => { for (const m of src.matchAll(new RegExp(`(?<![\\w$.])${what}\\??\\.([A-Za-z_$][\\w$]*)`, "g"))) found.add(m[1]); }; of(t.replace("$", "\\$") + "\\.s"); for (const m of src.matchAll(new RegExp(`(?:const|let|var)\\s+([\\w$]+)\\s*=\\s*${t.replace("$", "\\$")}\\.s\\s*[;,\\n]`, "g"))) of(m[1].replace("$", "\\$")); return [...found]; }
  function rules(tests, nothing) { return (tests || []).map(([name, body]) => { const was = tryOne(name, () => once(body)); if (was.ok || was.broke || was.theTest || nothing || !isGame()) return was;
    // What the test does not name may be what came between: a crow that was near takes the letter before it is had. So it is tried again with nothing else about.
    const alone = tryOne(name, () => { try { once(body, readIn(body).filter((k) => !setEmpty(body, k))); } finally { emptied = null; } return "with nothing else about"; }); return alone.ok ? alone : was; }); }
  // Whether all a test says of a list is that it is empty.
  function setEmpty(body, k) { const src = String(body), all = src.match(new RegExp(`\\.s\\.${k}(?![\\w$])`, "g")) || [], none = src.match(new RegExp(`\\.s\\.${k}\\s*=\\s*\\[\\s*\\]`, "g")) || []; return all.length > 0 && all.length === none.length; }
  // What the program came to in the last test that was run: what it holds, and what it shows.
  let cameTo = null; const last = () => { if (!cameTo) return ""; const s = cameTo.s; let seen = ""; try { seen = isGame() ? (P.hud ? JSON.stringify(P.hud(s)) : "") : words(shown(s)); } catch (e) { seen = "!"; } return JSON.stringify(s) + seen; };
  function once(body, named) { { waiting = []; const t = cameTo = tester(named); try { try { body(t); } catch (e) { judged(); throw e; } judged(); } catch (e) { const t_ = t.__told; e.got = t_.did();
      // What a test made for itself not being what it takes it for is the test's own.
      const own = /^([A-Za-z_$][\w$]*)\.[\w$]+ is not a function/.exec(String(e && e.message)); if (own && own[1] !== "t" && !(own[1] in P) && new RegExp(`\\b(?:const|let|var)\\s+${own[1].replace(/\$/g, "\\$")}\\b|\\b${own[1].replace(/\$/g, "\\$")}\\s*=>`).test(String(body))) e.theTest = true;
      const odd = t_.asked(body), then = t_.changed(); if (odd.length && !(e instanceof ReferenceError)) { e.theTest = true; e.message += `; the test reads s.${odd.join(", s.")}, and the state has no such thing: it has ${t_.held().join(", ")}`; } if (!isGame()) { let said = ""; try { said = t.shown; } catch (x) { said = ""; } e.message += `; it shows “${said.slice(0, 260)}”`; } e.message += then.length ? `; from where the test began it to its end, what changed: ${then.join("; ")}` : "; nothing in the state changed"; throw e; } } }
  function tried(program, tests, nothing) {
    P = { ...program }; live = false; goesOn = emptied = null;
    if (nothing) P.update = (s) => s;
    return { generic: nothing || tests ? [] : generic(), rules: rules(tests, nothing) };
  }
  // Where each of its tests leaves it, as the state it is in then: what it is looked at in, beside how it opens.
  function ends(program, tests) {
    P = { ...program }; live = false; goesOn = emptied = null; const out = [];
    for (const [name, body] of tests || []) { let t = null; try { t = tester(); body(t); } catch (e) { /* where it got to is where it is */ } finally { waiting = []; emptied = null; }
      if (t) { try { out.push([name, JSON.parse(JSON.stringify(t.s))]); } catch (e) { /* a state that cannot be written down is not looked at */ } } }
    return out;
  }
  // Some of the checks of any program, by their names.
  let only = null;
  function checked(program, names) { P = { ...program }; live = false; goesOn = emptied = null; only = names; try { return generic(); } finally { only = null; } }
  return { open, tried, checked, ends, last, rnd, chance, play, expect, ui, pen, W, H };
})();
const { rnd, between, int, pick, shuffle } = { rnd: Frame.rnd, ...Frame.chance };
Object.assign(rnd, { between, int, pick, shuffle });
const play = Frame.play, expect = Frame.expect, ui = Frame.ui, W = Frame.W, H = Frame.H;
