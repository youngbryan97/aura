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
      screen(draw) { ctx.save(); ctx.setTransform(g.base); draw(g); ctx.restore(); },
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
      at(x, y, turn, draw) { ctx.save(); ctx.translate(x, y); if (turn) ctx.rotate(turn); draw(g); ctx.restore(); } };
    return g;
  }

  // ── what an app is shown with: a tree of plain things, drawn into the page and kept in step with it ────────────
  const node = (tag, props, kids) => ({ tag, props: props || {}, kids: (kids || []).flat(9).filter((k) => k != null && k !== false).map((k) => (typeof k === "object" ? k : { tag: "#", text: String(k) })) });
  const ui = {
    column: (...k) => node("div", { class: "column" }, k), row: (...k) => node("div", { class: "row" }, k),
    card: (...k) => node("section", { class: "card" }, k), grid: (...k) => node("div", { class: "grid" }, k),
    title: (t) => node("h1", {}, [t]), heading: (t) => node("h2", {}, [t]), text: (t) => node("p", {}, [t]),
    note: (t) => node("p", { class: "note" }, [t]), big: (t) => node("div", { class: "big" }, [t]), badge: (t, tone) => node("span", { class: "badge " + (tone || "") }, [t]),
    divider: () => node("hr"), spacer: () => node("div", { class: "spacer" }),
    button: (label, name, value, o) => node("button", { class: (o && o.tone) || "", type: "button", disabled: !!(o && o.disabled), title: (o && o.hint) || null, do: [name, value === undefined ? null : value] }, [label]),
    input: (label, name, value, o) => field(label, node("input", { type: (o && o.type) || "text", value: value == null ? "" : String(value), placeholder: (o && o.placeholder) || "", in: name, enter: (o && o.enter) || null, "aria-label": label })),
    number: (label, name, value, o) => field(label, node("input", { type: "number", value: value == null ? "" : String(value), min: o && o.min, max: o && o.max, step: o && o.step, in: name, number: true, enter: (o && o.enter) || null, "aria-label": label })),
    date: (label, name, value) => field(label, node("input", { type: "date", value: value || "", in: name, "aria-label": label })),
    area: (label, name, value, o) => field(label, node("textarea", { value: value == null ? "" : String(value), rows: (o && o.rows) || 5, placeholder: (o && o.placeholder) || "", in: name, "aria-label": label })),
    check: (label, name, on) => node("label", { class: "check" }, [node("input", { type: "checkbox", checked: !!on, in: name, "aria-label": label }), label]),
    toggle: (label, name, value, on) => node("label", { class: "check" }, [node("input", { type: "checkbox", checked: !!on, do: [name, value === undefined ? null : value], "aria-label": label }), label]),
    select: (label, name, value, options) => field(label, node("select", { value: String(value), in: name, "aria-label": label }, (options || []).map((o) => { const [v, t] = Array.isArray(o) ? o : [o, o]; return node("option", { value: String(v), selected: String(v) === String(value) }, [t]); }))),
    slider: (label, name, value, o) => field(label, node("input", { type: "range", value: String(value), min: (o && o.min) ?? 0, max: (o && o.max) ?? 100, step: (o && o.step) ?? 1, in: name, number: true, "aria-label": label })),
    list: (items, each) => node("ul", { class: "list" }, (items || []).map((item, i) => node("li", {}, [each ? each(item, i) : String(item)]))),
    table: (heads, rows) => node("div", { class: "scroll" }, [node("table", {}, [node("thead", {}, [node("tr", {}, (heads || []).map((h) => node("th", {}, [h])))]), node("tbody", {}, (rows || []).map((r) => node("tr", {}, r.map((c) => node("td", {}, [c])))))])]),
    tabs: (names, chosen, name) => node("div", { class: "tabs", role: "tablist" }, (names || []).map((n) => node("button", { type: "button", role: "tab", class: n === chosen ? "on" : "", do: [name, n] }, [n]))),
    progress: (value, of) => node("div", { class: "progress", role: "progressbar" }, [node("div", { style: `width:${Math.max(0, Math.min(100, (100 * value) / (of || 1)))}%` })]),
    bars: (rows) => { const top = Math.max(1e-9, ...(rows || []).map((r) => Number(r.value) || 0)); return node("div", { class: "bars" }, (rows || []).map((r) => node("div", { class: "bar" }, [node("span", { class: "bar-label" }, [r.label]), node("div", { class: "bar-track" }, [node("div", { class: "bar-fill", style: `width:${(100 * (Number(r.value) || 0)) / top}%` })]), node("span", { class: "bar-value" }, [String(typeof r.show === "string" || typeof r.show === "number" ? r.show : r.value)])]))); },
    lines: (points) => { const v = (points || []).map(Number), top = Math.max(1e-9, ...v), low = Math.min(0, ...v), n = Math.max(1, v.length - 1); return node("svg", { viewBox: "0 0 600 180", class: "lines" }, [node("polyline", { points: v.map((y, i) => `${(600 * i) / n},${175 - (170 * (y - low)) / (top - low || 1)}`).join(" "), fill: "none" })]); },
    canvas: (name, w, h, draw) => node("canvas", { width: w, height: h, paint: draw, point: name }),
  };
  const field = (label, control) => node("label", { class: "field" }, [node("span", {}, [label]), control]);
  const words = (v) => (v.tag === "#" ? v.text : v.kids.map(words).join(" ")).replace(/\s+/g, " ").trim();
  function every(v, found) { found = found || []; if (v.tag !== "#") { found.push(v); v.kids.forEach((k) => every(k, found)); } return found; }
  function shown(state) { const v = P.view(state, ui); return v && v.tag ? v : node("div", {}, [v]); }

  const SVG = new Set(["svg", "polyline"]);
  function make(v) {
    if (v.tag === "#") return document.createTextNode(v.text);
    const el = SVG.has(v.tag) ? document.createElementNS("http://www.w3.org/2000/svg", v.tag) : document.createElement(v.tag);
    el.__kids = []; fit(el, { tag: v.tag, props: {}, kids: [] }, v); return el;
  }
  function fit(el, was, v) {
    for (const key of Object.keys({ ...was.props, ...v.props })) {
      const value = v.props[key];
      if (key === "do" || key === "in" || key === "enter" || key === "number" || key === "paint" || key === "point") { el["__" + key] = value; continue; }
      if (key === "value") { if (el.value !== value && document.activeElement !== el) el.value = value ?? ""; else if (el.value !== value && value === "") el.value = ""; continue; }
      if (key === "checked" || key === "disabled" || key === "selected") { el[key] = !!value; continue; }
      if (value == null || value === false) el.removeAttribute(key); else el.setAttribute(key, value === true ? "" : value);
    }
    const old = el.__v ? el.__v.kids : [];
    v.kids.forEach((kid, i) => {
      const at = el.childNodes[i], before = old[i];
      if (!at) el.appendChild(make(kid));
      else if (!before || before.tag !== kid.tag) el.replaceChild(make(kid), at);
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

  function open(program) {
    P = program; live = true; seed = (Date.now() ^ (Math.random() * 1e9)) | 0;
    document.title = P.name;
    if (P.look && P.look.accent) document.documentElement.style.setProperty("--accent", P.look.accent);
    root = document.getElementById("program"); root.textContent = "";
    (isGame() ? openGame : openApp)();
  }
  function problem(e) { const was = document.querySelector(".problem"); if (was) was.remove(); const p = el("div", "problem", "Something in this program went wrong: " + (e && e.message ? e.message : e)); document.body.appendChild(p); setTimeout(() => p.remove(), 8000); console.error(e); }

  function openApp() {
    document.body.className = "app";
    state = P.initial(); const was = kept(); if (was && typeof was === "object") state = { ...state, ...was };
    const bar = el("header"); bar.appendChild(el("strong", "", P.name));
    const tools = el("div", "tools"); bar.appendChild(tools);
    const tool = (label, hint, act) => { const b = el("button", "tool", label); b.title = hint; b.onclick = act; tools.appendChild(b); };
    const past = [];
    tool("Undo", "Undo the last change", () => { if (past.length) { state = past.pop(); show(); keep(); } });
    tool("Start over", "Put everything back as it began", () => { if (confirm("Start over? What is here will be cleared.")) { past.push(state); state = P.initial(); show(); keep(); } });
    if ((P.controls || []).length || P.about) tool("Help", "What this does and how", () => help());
    const main = el("main"); root.append(bar, main);
    let was_shown = null, top = null;
    const unsaid = (v) => { if (v.tag === "#") return v; return { ...v, kids: v.kids.filter((k) => !(k.tag === "h1" && words(k) === P.name)).map(unsaid) }; };
    function show() { try { const v = unsaid(shown(state)); if (!was_shown || was_shown.tag !== v.tag) { main.textContent = ""; top = make(v); main.appendChild(top); } else fit(top, was_shown, v); was_shown = v; } catch (e) { problem(e); } }
    function send(event, remember) { try { const before = remember ? JSON.parse(JSON.stringify(state)) : null; state = apply(state, event); if (before && JSON.stringify(before) !== JSON.stringify(state)) { past.push(before); if (past.length > 100) past.shift(); } show(); keep(); } catch (e) { problem(e); } }
    const up = (t, what) => { while (t && t !== main && t["__" + what] == null) t = t.parentNode; return t && t !== main ? t : null; };
    main.addEventListener("click", (e) => { const t = up(e.target, "do"); if (t && t.tagName !== "INPUT") send({ type: "do", name: t.__do[0], value: t.__do[1] }, true); });
    main.addEventListener("change", (e) => { const t = e.target; if (t.__do) send({ type: "do", name: t.__do[0], value: t.__do[1] }, true); else if (t.__in && (t.type === "checkbox" || t.tagName === "SELECT")) send({ type: "input", name: t.__in, value: t.type === "checkbox" ? t.checked : t.value }, true); });
    main.addEventListener("input", (e) => { const t = e.target; if (t.__in && t.type !== "checkbox" && t.tagName !== "SELECT") send({ type: "input", name: t.__in, value: t.__number ? Number(t.value) : t.value }); });
    main.addEventListener("keydown", (e) => { const t = e.target; if (e.key === "Enter" && t.__enter && t.tagName !== "TEXTAREA") send({ type: "do", name: t.__enter, value: null }, true); });
    main.addEventListener("pointerdown", (e) => { const t = up(e.target, "point"); if (t) { const r = t.getBoundingClientRect(); send({ type: "do", name: t.__point, value: { x: ((e.clientX - r.left) * t.width) / r.width, y: ((e.clientY - r.top) * t.height) / r.height } }, true); } });
    window.addEventListener("keydown", (e) => { if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return; send({ type: "key", key: key(e), down: true }); });
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
    const find = (label, what) => { const want = String(label).toLowerCase().trim(); const all = every(shown(s)).filter(what); const by = (f) => all.find((v) => f((v.props["aria-label"] || words(v) || v.props.placeholder || "").toLowerCase().trim())); return by((n) => n === want) || by((n) => n.startsWith(want)) || by((n) => n.includes(want)) || fail(`there is no “${label}” to use; there are: ${all.map((v) => v.props["aria-label"] || words(v)).slice(0, 12).join(", ")}`); };
    // A state that is set is set over one that has begun: what a test does not name is as it is at the beginning.
    let first = null; const began = Object.keys(s), act = (f) => { only(s); if (!first) first = copy(s); s = f(s); return t; };
    const t = { get s() { return s; }, set s(v) { s = only({ ...P.initial(), ...v }); }, W, H,
      send(e) { return act((s) => apply(s, e)); },
      tick(seconds) { return act((s) => run(s, seconds == null ? STEP : seconds, null)); },
      hold(k, seconds) { return act((s) => run(s, seconds == null ? 0.1 : seconds, { held: { [k]: true } })); },
      press(k) { return t.hold(k, STEP); },
      point(x, y, seconds) { return act((s) => run(s, seconds == null ? STEP : seconds, { pointer: { x, y, down: false } })); },
      click(x, y) { if (typeof x === "string") { const v = find(x, (n) => n.props.do); if (v.props.disabled) fail(`“${x}” cannot be used now`); return act((s) => apply(s, { type: "do", name: v.props.do[0], value: v.props.do[1] })); } return act((s) => run(s, STEP, { pointer: { x, y, down: true } })); },
      type(label, text) { const v = find(label, (n) => n.props.in); const value = v.props.number ? Number(text) : v.props.type === "checkbox" ? !!text : String(text); return act((s) => apply(s, { type: "input", name: v.props.in, value })); },
      choose(label, value) { return t.type(label, value); },
      enter(label) { const v = find(label, (n) => n.props.enter); return act((s) => apply(s, { type: "do", name: v.props.enter, value: null })); },
      sees(text) { if (isGame()) { const said = [...t.draw().filter((c) => c.what === "text").map((c) => c.args[0]), ...(P.hud ? P.hud(s) : []).flatMap((r) => [r.label, r.value]), ...(P.score ? ["score", scoreOf(s)] : [])].join(" ").toLowerCase(); return said.includes(String(text).toLowerCase()); } const all = words(shown(s)).toLowerCase(); return every(shown(s)).some((v) => String(v.props.value || "").toLowerCase().includes(String(text).toLowerCase())) || all.includes(String(text).toLowerCase()); },
      // What is drawn for the state as it is: each call of the pen, with what it was given.
      draw() { const calls = []; const canvas = document.createElement("canvas"); canvas.width = W; canvas.height = H; const real = pen(canvas.getContext("2d"), 1);
        const g = new Proxy(real, { get: (o, k) => (typeof o[k] === "function" ? (...args) => { calls.push({ what: String(k), args }); if (k === "at") { const [x, y, turn, act] = args; return real.at(x, y, turn, () => act(g)); } return o[k](...args); } : o[k]) });
        P.draw(s, g); return calls; },
      get score() { return scoreOf(s); }, get over() { return ended(s) || false; }, get won() { const e = ended(s); return !!(e && e.won); }, get played() { return played.slice(); },
      expect, initial() { s = P.initial(); return s; }, update(e) { return t.send(e); } };
    const told = { changed: () => (first ? changed(first, s) : []), asked: (body) => readIn(body).filter((k) => !began.includes(k) && !(P.designed && k in P.designed) && !later().includes(k)), held: () => Object.keys(s) };
    return new Proxy(t, { get(o, k) { if (k === "__told") return told; if (k in o || typeof k === "symbol") return o[k]; const e = new Error(`the test asks for t.${String(k)}, and there is no such thing; there are: ${Object.keys(o).join(" ")}`); e.theTest = true; throw e; } });
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
  function tryOne(name, act) { const at = performance.now(); try { const note = act(); return { name, ok: true, why: note || "", ms: Math.round(performance.now() - at) }; } catch (e) { return { name, ok: false, theTest: !!(e && e.theTest), broke: e instanceof TypeError || e instanceof ReferenceError || e instanceof RangeError || e instanceof SyntaxError, why: String((e && e.message) || e).slice(0, 760) + has(String((e && e.message) || e)) + where(e), ms: Math.round(performance.now() - at) }; } }
  // Where something went wrong, by the names of the program's own functions, the innermost first.
  const OURS = new Set(["apply", "run", "once", "explore", "tryOne", "check", "generic", "rules", "tried", "wander", "look", "act", "shown", "judged", "fail", "must", "expect", "judge", "ways", "every", "words", "pen", "tester", "body", "map", "filter", "forEach", "reduce", "some", "every", "find", "eval", "evaluate", "__try", "get", "send", "tick", "hold", "press", "point", "click", "type", "test"]);
  const where = (e) => { const names = []; for (const line of String((e && e.stack) || "").split("\n").slice(1)) { const m = /at (?:async )?(?:Object\.|Proxy\.|PROGRAM\.)?([A-Za-z_$][\w$]*)(?: \[as [^\]]+\])? \(/.exec(line); if (m && !OURS.has(m[1]) && names[names.length - 1] !== m[1]) names.push(m[1]); if (names.length >= 4) break; } return names.length ? ` (in ${names.join(", called by ")})` : ""; };
  // What a piece can be asked for, where it was asked for what it has not.
  const has = (said) => { const m = /\b([A-Za-z_$][\w$]*)\.([A-Za-z_$][\w$]*) is not a function/.exec(said), piece = m && P && P[m[1]]; return piece && typeof piece === "object" && typeof piece[m[2]] !== "function" ? `; ${m[1]} has ${Object.keys(piece).filter((k) => typeof piece[k] === "function").join(", ")}` : ""; };
  const must = (ok, why) => { if (!ok) throw new Error(why); };

  function generic() {
    const out = []; const check = (name, act) => out.push(tryOne(name, act));
    const ways = inputsOf();
    const wander = (from, seconds, sd, take) => { seed = sd; let s = from === null ? P.initial() : copy(from); let left = seconds; while (left > 0 && !(P.over && ended(s))) { const way = take ? take(s) : ways.length ? ways[Math.floor(rnd() * ways.length)] : null; s = run(s, 0.25, way); left -= 0.25; } return s; };
    check("it begins", () => { seed = 7; const s = P.initial(); must(s && typeof s === "object", "initial() gives no state"); must(JSON.stringify(s) === JSON.stringify(copy(s)), "its state is not plain data"); const w = wrong(s, "state", 0); must(!w, w); if (P.over) must(!ended(s), "it is over as it begins"); });
    if (out[0].ok === false) return out;
    if (P.designed) check("its state is as it was designed", () => { seed = 7; const s = P.initial(), lost = Object.keys(P.designed).filter((k) => !(k in s)); must(!lost.length, `initial() has no ${lost.join(", ")}; the design's state has`); });
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
      check("it is quick enough", () => { const c = document.createElement("canvas"); c.width = W; c.height = H; const g = pen(c.getContext("2d"), 0); let s = wander(null, 10, 4); const at = performance.now(); for (let i = 0; i < 120; i++) { s = apply(s, { type: "tick", dt: STEP, held: {}, pointer: NONE }); P.draw(s, g); } const ms = (performance.now() - at) / 120; must(ms < 12, `a frame takes ${ms.toFixed(1)} ms`); return `${ms.toFixed(1)} ms a frame`; });
      if (P.hud) check("what it counts is shown", () => { seed = 7; const rows = P.hud(P.initial()); must(Array.isArray(rows) && rows.every((r) => r && r.label != null && r.value != null), "hud() does not give {label, value} rows"); });
    } else {
      const shownAt = (s) => { const v = shown(s); const said = words(v); must(said.length > 0, "nothing is shown"); const odd = /\bundefined\b|\bNaN\b|\[object Object\]|\bInfinity\b|\bInvalid Date\b/.exec(said); must(!odd, `it shows “${odd && odd[0]}” in “${odd && said.slice(Math.max(0, odd.index - 40), odd.index + 30)}”`); const bare = every(v).find((n) => n.kids.length === 1 && n.kids[0].tag === "#" && /^(true|false|null)$/.test(n.kids[0].text)); must(!bare, `it shows a bare “${bare && bare.kids[0].text}”`); return v; };
      check("it is shown", () => { seed = 7; shownAt(P.initial()); if (P.designed) shownAt({ ...P.initial(), ...copy(P.designed) }); });
      check("it begins with nothing in it that nobody put there", () => { seed = 7; const s = P.initial(); const mine = Object.keys(P.designed || {}).filter((k) => Array.isArray(P.designed[k]) && Array.isArray(s[k]) && s[k].some((x) => x && typeof x === "object" && "id" in x && Object.values(x).every((v) => v === null || typeof v !== "object"))); must(!mine.length, `it begins with records already in ${mine.join(", ")}: it is to begin empty, for whoever uses it to fill`); });
      const used = new Map();
      const act = (s) => { const v = shownAt(s), all = every(v).filter((n) => (n.props.do && !n.props.disabled) || n.props.in); if (!all.length) return s; const n = all[Math.floor(rnd() * all.length)]; const label = n.props["aria-label"] || words(n); const before = JSON.stringify(s); let next; if (n.props.do) next = apply(s, { type: "do", name: n.props.do[0], value: n.props.do[1] }); else next = apply(s, { type: "input", name: n.props.in, value: n.props.type === "checkbox" ? rnd() < 0.5 : n.tag === "select" && n.kids.length ? n.kids[Math.floor(rnd() * n.kids.length)].props.value : n.props.number ? Math.floor(rnd() * 50) : n.props.type === "date" ? "2026-11-2" + Math.floor(rnd() * 9) : chance.pick(["milk", "a longer thing to do", "7", "", "Ana"]) }); if (JSON.stringify(next) !== before) used.set(label, true); else if (!used.has(label)) used.set(label, false); return next; };
      check("it runs without breaking, whatever is used", () => { for (let sd = 1; sd <= 6; sd++) { seed = sd; let s = P.initial(); for (let i = 0; i < 160; i++) { s = act(s); if (P.ticks && i % 8 === 0) s = apply(s, { type: "tick", dt: 1, now: 1.79e12 + i * 1000 }); const w = wrong(s, "state", 0); must(!w, w); } shownAt(s); } });
      check("every control does something", () => { const dead = [...used].filter(([, did]) => !did).map(([label]) => label); must(dead.length === 0, `these change nothing: ${dead.slice(0, 8).join(", ")}`); must(used.size > 0, "there is nothing to use"); });
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
  function once(body, named) { { waiting = []; const t = tester(named); try { try { body(t); } catch (e) { judged(); throw e; } judged(); } catch (e) { const t_ = t.__told, odd = t_.asked(body), then = t_.changed(); if (odd.length && !(e instanceof ReferenceError)) { e.theTest = true; e.message += `; the test reads s.${odd.join(", s.")}, and the state has no such thing: it has ${t_.held().join(", ")}`; } e.message += then.length ? `; from where the test began it to its end, what changed: ${then.join("; ")}` : "; nothing in the state changed"; throw e; } } }
  function tried(program, tests, nothing) {
    P = { ...program }; live = false; goesOn = emptied = null;
    if (nothing) P.update = (s) => s;
    return { generic: nothing || tests ? [] : generic(), rules: rules(tests, nothing) };
  }
  return { open, tried, rnd, chance, play, expect, ui, pen, W, H };
})();
const { rnd, between, int, pick, shuffle } = { rnd: Frame.rnd, ...Frame.chance };
Object.assign(rnd, { between, int, pick, shuffle });
const play = Frame.play, expect = Frame.expect, ui = Frame.ui, W = Frame.W, H = Frame.H;
