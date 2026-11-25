/* workbench.js — her work as it goes, as a session: what she says and does in order, the program running beside it, its
 * code, and what holds of it.
 *
 * What happens is streamed (/api/work/{id}/stream) and drawn as it comes: a few letters at a time while she writes, a
 * card for each thing she does, with what changed in it. What is said here reaches the work at its next step. */
(function () {
  const ICON = {
    step: '<circle cx="8" cy="8" r="5.5"/>', done: '<path d="m4 8.500 2.800 2.800L12.500 5"/>', code: '<path d="M6 5 3 8l3 3M10 5l3 3-3 3"/>', tried: '<path d="M6 2.500h4M7 2.500v4L3.500 13h9L9 6.500v-4"/>',
    trouble: '<path d="M8 3 2.500 13h11zM8 7v3M8 11.600v.1"/>', made: '<path d="M3 5.500 8 3l5 2.500v5L8 13l-5-2.500zM3 5.500 8 8l5-2.500M8 8v5"/>', said: '<circle cx="8" cy="5.500" r="2.400"/><path d="M3.500 13c.6-2.600 2.300-4 4.500-4s3.900 1.400 4.500 4"/>',
    design: '<path d="M3 13V3h10v10zM3 6.500h10M6.500 6.500V13"/>', layer: '<path d="m8 3 5.500 3L8 9 2.500 6zM2.500 9.500 8 12.500l5.500-3"/>', asks: '<path d="M6 6.200a2 2 0 1 1 2.700 1.900c-.5.200-.7.500-.7 1.100M8 11.500v.1"/><circle cx="8" cy="8" r="6"/>', painted: '<path d="M2.500 3.500h11v9h-11zM2.500 10.500l3-3 3 3 2-2 3 3M10.500 6h.1"/>',
    ok: '<path d="m3.500 8.500 3 3 6-7"/>', no: '<path d="m4.500 4.500 7 7M11.500 4.500l-7 7"/>', open: '<path d="M7 4H3.500v8.500H12V9M9.500 3H13v3.500M13 3 8 8"/>', chevron: '<path d="m6 4 4 4-4 4"/>', down: '<path d="M8 3.500v9M4.500 9 8 12.500 11.500 9"/>',
    hold: '<path d="M6 4v8M10 4v8"/>', go: '<path d="M5.500 4v8l7-4z"/>', stop: '<rect x="4.500" y="4.500" width="7" height="7" rx="1"/>', send: '<path d="M8 13V3M4 7l4-4 4 4"/>', wide: '<path d="M9.500 3H13v3.500M13 3 9 7M6.500 13H3V9.500M3 13l4-4"/>', again: '<path d="M13 8a5 5 0 1 1-1.600-3.700M11.800 2.500v2.300H9.500"/>', other: '<circle cx="8" cy="8" r="1.500"/>', search: '<circle cx="7" cy="7" r="4"/><path d="m10 10 3.500 3.500"/>',
  };
  const icon = (k) => `<svg class="wb-icon" viewBox="0 0 16 16" aria-hidden="true">${ICON[k] || ICON.other}</svg>`;
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const clock = (at) => new Date(at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const lasted = (s) => (s < 60 ? `${Math.max(0, Math.round(s))}s` : s < 3600 ? `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, '0')}s` : `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`);
  const count = (n) => (n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n || 0));
  const WORDS = { javascript: 'const let var function return if else for while of in new class extends async await true false null undefined this switch case break continue throw try catch finally typeof delete default import export yield', python: 'def class return if elif else for while in not and or is None True False import from as with try except finally raise lambda yield pass break continue global nonlocal assert async await', swift: 'let var func struct class enum return if else guard for while in import init self nil true false switch case break continue throws try catch extension protocol', rust: 'fn let mut struct enum impl trait return if else match for while loop in use mod pub self Self true false where as ref move', sql: 'select from where group by order insert into values create table update delete join left right on as and or not null limit having' };
  WORDS['c'] = WORDS['c++'] = WORDS.javascript + ' int char double float void struct unsigned long short static include define using namespace auto bool template';
  const KEYS = {};
  const keysOf = (l) => (KEYS[l] = KEYS[l] || new Set((WORDS[l] || WORDS.javascript).split(' ')));

  function lit(code, language) {
    const keys = keysOf(language), hash = /^(python|bash|ruby|perl)$/.test(language);
    const rx = new RegExp((hash ? '(#[^\\n]*)' : '(\\/\\/[^\\n]*|\\/\\*[\\s\\S]*?\\*\\/|<!--[\\s\\S]*?-->)') + '|("(?:\\\\.|[^"\\\\\\n])*"|\'(?:\\\\.|[^\'\\\\\\n])*\'|`(?:\\\\.|[^`\\\\])*`)|\\b(\\d+(?:\\.\\d+)?)\\b|([A-Za-z_$][\\w$]*)(?=(\\s*\\()?)', 'g');
    let out = '', last = 0, m;
    while ((m = rx.exec(code))) {
      out += esc(code.slice(last, m.index));
      if (m[1]) out += `<span class="wb-c">${esc(m[1])}</span>`;
      else if (m[2]) out += `<span class="wb-s">${esc(m[2])}</span>`;
      else if (m[3]) out += `<span class="wb-n">${esc(m[3])}</span>`;
      else if (m[4]) out += keys.has(m[4]) ? `<span class="wb-k">${m[4]}</span>` : m[5] !== undefined ? `<span class="wb-f">${esc(m[4])}</span>` : esc(m[4]);
      last = m.index + (m[4] ? m[4].length : m[0].length);
      rx.lastIndex = last;
    }
    return out + esc(code.slice(last));
  }
  // Her words, with the marks people write with: code, weight, and lists.
  function prose(text) {
    const inline = (t) => esc(t).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    const out = []; let list = null;
    for (const line of String(text || '').split('\n')) {
      const m = /^\s*(?:[-*]|\d+\.)\s+(.*)/.exec(line);
      if (m) { if (!list) out.push(list = []); list.push(`<li>${inline(m[1])}</li>`); continue; }
      list = null; if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
    return out.map((x) => (Array.isArray(x) ? `<ul>${x.join('')}</ul>` : x)).join('');
  }
  // What changed between two pieces of text, line by line, with a few lines of what stands about it.
  function diff(a, b) {
    const x = a.split('\n'), y = b.split('\n'); let s = 0, e = 0;
    while (s < x.length && s < y.length && x[s] === y[s]) s++;
    while (e < x.length - s && e < y.length - s && x[x.length - 1 - e] === y[y.length - 1 - e]) e++;
    const p = x.slice(s, x.length - e), q = y.slice(s, y.length - e);
    let mid = [];
    if (p.length * q.length < 160000) {
      const t = Array.from({ length: p.length + 1 }, () => new Uint16Array(q.length + 1));
      for (let i = p.length - 1; i >= 0; i--) for (let j = q.length - 1; j >= 0; j--) t[i][j] = p[i] === q[j] ? t[i + 1][j + 1] + 1 : Math.max(t[i + 1][j], t[i][j + 1]);
      let i = 0, j = 0;
      while (i < p.length && j < q.length) { if (p[i] === q[j]) { mid.push([' ', p[i]]); i++; j++; } else if (t[i + 1][j] >= t[i][j + 1]) mid.push(['-', p[i++]]); else mid.push(['+', q[j++]]); }
      while (i < p.length) mid.push(['-', p[i++]]); while (j < q.length) mid.push(['+', q[j++]]);
    } else mid = [...p.map((l) => ['-', l]), ...q.map((l) => ['+', l])];
    const rows = [...x.slice(Math.max(0, s - 3), s).map((l) => [' ', l]), ...mid, ...y.slice(y.length - e, y.length - e + 3).map((l) => [' ', l])];
    const out = []; let same = [];
    for (const r of rows) { if (r[0] === ' ') { same.push(r); continue; } if (same.length > 7) out.push(...same.slice(0, 3), ['…', `${same.length - 6} lines as they were`], ...same.slice(-3)); else out.push(...same); same = []; out.push(r); }
    out.push(...same.slice(0, 3));
    return { rows: out, added: mid.filter((r) => r[0] === '+').length, gone: mid.filter((r) => r[0] === '-').length };
  }
  const drawnDiff = (d, language) => `<pre class="wb-diff">${d.rows.slice(0, 400).map(([k, l]) => `<div class="${k === '+' ? 'in' : k === '-' ? 'out' : k === '…' ? 'gap' : ''}"><i>${k === '…' ? '' : k}</i><span>${k === '…' ? esc(l) : lit(l, language)}</span></div>`).join('')}</pre>`;
  const codeIn = (text) => { const at = text.indexOf('```'); if (at < 0) return /^\s*[{(]/.test(text) ? { code: text, said: '' } : { code: '', said: text }; const rest = text.slice(at + 3).replace(/^[\w+-]*\n/, ''); const end = rest.indexOf('```'); return { said: text.slice(0, at).trim(), code: end < 0 ? rest : rest.slice(0, end), after: end < 0 ? '' : rest.slice(end + 3).trim() }; };
  /* Her own time is in the list beside her works: what she does with nobody asking, and what came of it. */
  const OWN = 'her-own-time';
  const SORTS = { carry_on: 'Going on with what was asked', find_out: 'Finding things out', look_up: 'Looking into what she set herself', goal: 'Working on what she set herself', tell: 'Telling what she found', make: 'Making something' };
  const signed = (v) => (v == null ? '' : `<span class="${v < 0 ? 'minus' : 'plus'}">${v < 0 ? '−' : '+'}${Math.abs(v).toFixed(2)}</span>`);
  const site = (url) => { try { return new URL(url).hostname.replace(/^www\./, ''); } catch (e) { return ''; } };
  const asked = (q) => `<li class="${{ answered: 'ok', partly: 'ok part' }[q.stands] || (q.stands === 'let go' ? 'no' : 'open')}">${icon({ answered: 'ok', partly: 'ok', 'let go': 'no' }[q.stands] || 'asks')}<span>${esc(q.question)}${q.answer ? `<div class="wb-answer">${esc(q.answer)}${/^https?:/.test(q.where || '') ? ` <a href="${esc(q.where)}" target="_blank" rel="noopener noreferrer">${esc(site(q.where))}</a>` : ''}</div>` : q.stands === 'let go' ? '<div class="wb-why">Nothing she read says.</div>' : ''}</span></li>`;
  function ownTime(h) {
    const b = h.before || {}, now = h.doing && h.doing.what ? `<div class="wb-own-now"><span class="wb-spin"></span><div><strong>${esc(h.doing.what)}</strong>${h.doing.why ? `<div class="wb-why">${esc(h.doing.why)}</div>` : ''}</div></div>`
      : `<div class="wb-own-now">${icon('hold')}<div><strong>At rest</strong>${b.what ? `<div class="wb-why">Last: ${esc(b.what)}${b.came ? `. ${esc(b.came)}` : ''}</div>` : ''}</div></div>`;
    const part = (title, inner) => (inner ? `<h3>${title}</h3>${inner}` : '');
    const finding = (h.finding_out || []).filter((f) => (f.questions || []).length).map((f) => `<details class="wb-card" data-of="${esc(f.of)}"><summary>${icon('search')}<span class="wb-card-name">${esc(f.about)}</span><span class="wb-about">${f.questions.filter((q) => q.stands === 'answered').length} of ${f.questions.length} answered</span><span class="wb-chev">${icon('chevron')}</span><span class="wb-meter"><i style="width:${Math.round(100 * (f.met || 0))}%"></i></span></summary><div class="wb-card-in"><ul class="wb-checks">${f.questions.map(asked).join('')}</ul></div></details>`).join('');
    const did = (h.choices || []).slice(0, 12).map((c) => `<details class="wb-card" data-of="${esc(c.at)}"><summary>${icon(c.reward == null ? 'hold' : c.reward < 0 ? 'no' : 'ok')}<span class="wb-card-name">${esc(c.what)}</span><span class="wb-about">${clock(c.at)} ${signed(c.reward)}</span><span class="wb-chev">${icon('chevron')}</span></summary><div class="wb-card-in">${c.came ? `<p>${esc(c.came)}</p>` : ''}<p class="wb-why whole">${esc(c.why)}</p></div></details>`).join('');
    const kinds = Object.entries((h.learned || {}).kinds || {}).filter(([, k]) => k.done).sort((x, y) => y[1].expected - x[1].expected).map(([k, v]) => `<tr><td>${esc(SORTS[k] || k.replace(/_/g, ' '))}</td><td>${v.done} done</td><td>${signed(v.expected)}</td></tr>`).join('');
    const l = h.learned || {}, better = l.before != null && l.lately != null ? `<p class="wb-why">What came of it lately ${signed(l.lately)}, and before ${signed(l.before)}.</p>` : '';
    return now + part('Finding out', finding) + part('What she did', did) + part('What she expects of each', kinds ? `<table class="wb-learned">${kinds}</table>${better}` : '') || '<div class="wb-empty">Nothing yet.</div>';
  }
  const el = (tag, cls, html) => { const x = document.createElement(tag); if (cls) x.className = cls; if (html != null) x.innerHTML = html; return x; };

  window.mountWorkbench = function (root, options) {
    const wide = !!(options && options.wide);
    let chosen = '', list = [], last = null, after = 0, files = new Map(), checks = [], design = null, source = null, program = null, kept = 0, stopped = false, pinned = true, group = null, timer = null, tab = wide ? 'preview' : 'session', target = '', shownN = 0, liveKey = '', raf = 0, part = '', asking = [], live = null, writing = '';
    const drafts = new Map();
    root.classList.toggle('wb-wide', wide);
    root.innerHTML = `
      ${wide ? '<nav class="wb-rail" aria-label="Her works"><h2>Work</h2><div class="wb-works"></div></nav>' : ''}
      <section class="wb-main">
        <header class="wb-top"><select aria-label="Which work"></select><div class="wb-title"></div><span class="wb-state"><i></i><span></span></span>
          <button data-do="hold" class="wb-quiet" title="Hold it where it is">${icon('hold')}</button><button data-do="open" class="wb-quiet" title="Open what it made">${icon('open')}</button>${wide ? '' : `<button data-do="wide" class="wb-quiet" title="Open in a window of its own">${icon('wide')}</button>`}</header>
        ${wide ? '' : '<div class="wb-tabs" role="tablist"></div>'}
        <div class="wb-session" data-view="session"><div class="wb-log" role="log" aria-live="off"></div></div>
        <button class="wb-latest" hidden>${icon('down')}Latest</button>
        ${wide ? '' : '<div class="wb-view" hidden></div>'}
        <footer class="wb-foot"><div class="wb-status" aria-live="polite"></div><form class="wb-say"><textarea rows="1" placeholder="Say something to her about this" aria-label="Say something to this work"></textarea><button type="button" data-do="stop" class="wb-round wb-stop" title="Stop">${icon('stop')}</button><button type="submit" class="wb-round" title="Say it">${icon('send')}</button></form></footer>
      </section>
      ${wide ? '<aside class="wb-canvas"><div class="wb-tabs" role="tablist"></div><div class="wb-view"></div></aside>' : ''}`;
    const $ = (q) => root.querySelector(q);
    const log = $('.wb-log'), scroller = $('.wb-session'), say = $('.wb-say textarea'), view = $('.wb-view'), latest = $('.wb-latest'), pick = $('select');
    const get = async (url) => { const r = await fetch(url, { credentials: 'same-origin' }); if (!r.ok) throw new Error(String(r.status)); return r.json(); };
    const post = (url, body) => fetch(url, { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : '{}' });
    const visible = () => root.isConnected && root.offsetParent !== null && !document.hidden;
    const down = (smooth) => { scroller.scrollTo({ top: scroller.scrollHeight, behavior: smooth ? 'smooth' : 'auto' }); };

    // ── the session ────────────────────────────────────────────────
    function choose(id) {
      if (id === chosen) return;
      clearTimeout(timer); cancelAnimationFrame(raf); raf = 0; live = null; writing = ''; if (chosen) drafts.set(chosen, say.value);
      chosen = id; last = null; after = 0; files = new Map(); checks = []; design = null; program = null; kept = 0; group = null; target = ''; shownN = 0; liveKey = ''; part = ''; pinned = true; asking = [];
      log.innerHTML = ''; view.innerHTML = ''; say.value = drafts.get(id) || ''; $('.wb-title').textContent = ''; $('.wb-status').textContent = ''; root.classList.toggle('wb-own', id === OWN);
      if (id === OWN) { if (source) { source.close(); source = null; } own(); return; }
      stream(); drawTabs(); drawView();
    }
    async function own() {
      let h; try { h = await get('/api/her_own_time'); } catch (e) { return; }
      if (chosen !== OWN) return;
      const busy = !!(h.doing && h.doing.what), s = $('.wb-state'); s.className = 'wb-state ' + (busy ? 'going' : ''); s.lastElementChild.textContent = busy ? 'Doing something' : 'At rest'; $('.wb-title').textContent = 'Her own time';
      const html = ownTime(h); if (log.__own === html && log.firstChild) return;
      const open = new Set([...log.querySelectorAll('details[open]')].map((d) => d.dataset.of)), at = scroller.scrollTop;
      log.__own = html; log.innerHTML = `<div class="wb-own-time">${html}</div>`; for (const d of log.querySelectorAll('details')) if (open.has(d.dataset.of)) d.open = true; scroller.scrollTop = at;
    }
    scroller.addEventListener('scroll', () => { pinned = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 60; latest.hidden = pinned; });
    latest.addEventListener('click', () => { pinned = true; down(true); });

    function begun(title, at) {
      if (group) { group.el.classList.remove('now'); group.el.classList.add('was'); group.icon.innerHTML = icon('done'); group.time.textContent = lasted(at - group.at); }
      const box = el('div', 'wb-step now'); const head = el('button', 'wb-step-head'); head.type = 'button';
      const mark = el('span', 'wb-mark', '<span class="wb-spin"></span>'), name = el('span', 'wb-step-name'), time = el('span', 'wb-when'); name.textContent = title;
      const inner = el('div', 'wb-step-in'); head.append(mark, name, time); box.append(head, inner); log.appendChild(box);
      head.onclick = () => box.classList.toggle('shut');
      group = { el: box, icon: mark, time, in: inner, at, title };
    }
    const into = () => { if (!group) begun('Beginning', Date.now() / 1000); return group.in; };
    function card(kind, title, about, body, open) {
      const d = el('details', `wb-card ${kind}`); d.open = !!open;
      d.innerHTML = `<summary>${icon(kind)}<span class="wb-card-name">${title}</span><span class="wb-about">${about || ''}</span>${body ? `<span class="wb-chev">${icon('chevron')}</span>` : ''}</summary>`;
      if (body) { const b = el('div', 'wb-card-in'); d.appendChild(b); let drawn = false; const draw = () => { if (!drawn) { drawn = true; b.innerHTML = typeof body === 'function' ? body() : body; } }; if (open) draw(); d.addEventListener('toggle', () => { if (d.open) draw(); }); } else d.classList.add('bare');
      return d;
    }
    const checkRow = (r) => `<li class="${r.ok ? 'ok' : 'no'}">${icon(r.ok ? 'ok' : 'no')}<span>${esc(r.name)}${r.why && !r.ok ? `<div class="wb-why">${esc(r.why)}</div>` : ''}</span></li>`;
    const TELLS = {
      step(h) { begun(h.title, h.at); },
      says(h) { const p = el('div', 'wb-says', prose(h.words)); if (/^Kept\b/.test(h.words)) p.classList.add('good'); into().appendChild(p); },
      said(h) { const p = el('div', 'wb-said'); p.textContent = h.words; log.appendChild(p); group = null; },
      trouble(h) { into().appendChild(el('div', 'wb-trouble', icon('trouble') + `<span>${esc(h.why)}</span>`)); },
      made(h) { const c = el('div', 'wb-made', `<div class="wb-made-pic">${icon('made')}</div><div><strong>${esc(h.title)}</strong><div class="wb-why">${esc(h.where)}</div></div><button data-do="open">${icon('open')}Open it</button>`); log.appendChild(c); const img = new Image(); img.alt = ''; img.onload = () => c.querySelector('.wb-made-pic').replaceChildren(img); img.src = `/api/work/${chosen}/picture/icon`; },
      layer(h) { into().appendChild(card('layer', `Part ${h.n} of ${h.of}`, `${(h.rules || []).length} rules`, `<ul class="wb-plain">${(h.rules || []).map((r) => `<li>${esc(r)}</li>`).join('')}</ul>`)); },
      design(h) { design = h.design || {}; into().appendChild(card('design', `${esc(design.name)}`, `${(design.rules || []).length} rules`, `<p>${esc(design.about)}</p><ul class="wb-plain">${(design.rules || []).map((r) => `<li>${esc(r.says)}${r.of ? `<span class="wb-why"> “${esc(r.of)}”</span>` : ''}</li>`).join('')}</ul>`)); drawTabs(); },
      code(h) {
        const was = files.get(h.name); files.set(h.name, { ...h, was: was ? was.code : '' });
        const d = diff(was ? was.code : '', h.code || '');
        into().appendChild(card('code', `${was ? 'Changed' : 'Wrote'} <b>${esc(h.name)}</b>`, `<span class="plus">+${d.added}</span>${d.gone ? ` <span class="minus">−${d.gone}</span>` : ''}`, () => (was ? drawnDiff(d, h.language) : `<pre class="wb-code">${lit(h.code || '', h.language)}</pre>`)));
      },
      tried(h) {
        const bad = (h.results || []).filter((r) => !r.ok); const before = new Map(checks.map((r) => [r.name, r.ok])); checks = (h.results || []).slice();
        const gained = checks.filter((r) => r.ok && before.get(r.name) === false).length, lost = checks.filter((r) => !r.ok && before.get(r.name) === true).length;
        const c = card('tried', esc(h.title), `${gained ? `<span class="plus">+${gained}</span> ` : ''}${lost ? `<span class="minus">−${lost}</span> ` : ''}<b>${h.good}</b> of ${h.of} hold`, () => `<ul class="wb-checks">${bad.slice(0, 40).map(checkRow).join('')}${bad.length ? '' : '<li class="ok">' + icon('ok') + '<span>Everything it was tried for holds</span></li>'}</ul>`, bad.length > 0 && bad.length < 4);
        c.classList.toggle('all', !bad.length && h.of > 0); c.querySelector('summary').insertAdjacentHTML('beforeend', `<span class="wb-meter"><i style="width:${h.of ? (100 * h.good) / h.of : 0}%"></i></span>`); into().appendChild(c); drawTabs();
      },
      painted(h) { const c = el('figure', 'wb-painted'); c.innerHTML = `<img alt="${esc(h.shows || h.name)}" src="/api/work/${chosen}/picture/${encodeURIComponent(h.name)}"><figcaption>${esc(h.name)}${h.shows ? `<div class="wb-why">${esc(h.shows)}</div>` : ''}</figcaption>`; let strip = into().lastElementChild; if (!strip || !strip.classList.contains('wb-strip')) { strip = el('div', 'wb-strip'); into().appendChild(strip); } strip.appendChild(c); },
      asks(h) { asking.push(h.words); const c = el('div', 'wb-ask'); c.innerHTML = `<div class="wb-ask-q">${icon('asks')}<div>${prose(h.words)}${h.goes_with ? `<div class="wb-why">Until you say, she goes on with ${esc(h.goes_with)}.</div>` : ''}</div></div><form><input aria-label="Your answer" placeholder="Answer her"><button>Answer</button></form>`; c.querySelector('form').onsubmit = async (e) => { e.preventDefault(); const words = c.querySelector('input').value.trim(); if (!words) return; c.classList.add('answered'); c.querySelector('form').remove(); await post(`/api/work/${chosen}/say`, { words }); }; log.appendChild(c); group = null; },
      kept() { kept++; program = null; drawTabs(); if (tab === 'preview' || tab === 'code') drawView(); },
    };
    const MARKED = { held: 'Held', 'goes on': 'Going on', 'told to stop': 'Told to stop', 'taken up again': 'Taken up again', 'left off': 'Left off' };
    function happened(h) {
      if (h.n <= after) return;
      after = h.n;
      if (TELLS[h.what]) TELLS[h.what](h);
      else if (MARKED[h.what]) { log.appendChild(el('div', 'wb-line', `<span>${MARKED[h.what]}${h.why ? `: ${esc(h.why)}` : ''} · ${clock(h.at)}</span>`)); group = null; }
      while (log.children.length > 500) log.removeChild(log.firstChild);
    }

    // ── what she is writing, a few letters at a time ───────────────
    function liveBox() {
      if (live && live.isConnected) return live;
      live = el('div', 'wb-live', '<div class="wb-says"></div><div class="wb-live-code" hidden><header><span class="wb-dots"><i></i><i></i><i></i></span><span class="wb-live-n"></span></header><pre class="wb-code"><code></code></pre></div>');
      live.lines = []; into().appendChild(live); return live;
    }
    function drawLive() {
      raf = 0; if (!target) return;
      if (shownN < target.length) shownN = Math.min(target.length, shownN + Math.max(2, Math.ceil((target.length - shownN) / 9)));
      const box = liveBox(); if (box !== box.parentNode.lastElementChild) box.parentNode.appendChild(box);
      const c = codeIn(target.slice(0, shownN)), words = box.firstElementChild, holder = box.children[1], code = holder.querySelector('code');
      if (words.__t !== c.said) { words.__t = c.said; words.innerHTML = prose(c.said) + (c.code ? '' : '<span class="wb-caret"></span>'); }
      holder.hidden = !c.code;
      if (c.code) {
        // Whole lines are drawn once; the line she is on is drawn as it grows.
        const lines = c.code.split('\n'); const language = (last && last.kind === 'tool' && 'python') || 'javascript';
        if (box.lines.length > lines.length) { code.innerHTML = ''; box.lines = []; }
        while (box.lines.length < lines.length) { const l = el('div', 'wb-l'); code.appendChild(l); box.lines.push(l); const i = box.lines.length - 2; if (i >= 0) box.lines[i].innerHTML = lit(lines[i], language) || '&#8203;'; }
        box.lines[lines.length - 1].innerHTML = lit(lines[lines.length - 1], language) + '<span class="wb-caret"></span>';
        holder.querySelector('.wb-live-n').textContent = `${lines.length} lines`;
        const pre = holder.querySelector('pre'); if (pre.scrollHeight - pre.scrollTop - pre.clientHeight < 80) pre.scrollTop = pre.scrollHeight;
      }
      if (pinned) down(false);
      if (shownN < target.length) raf = requestAnimationFrame(drawLive);
    }
    function wrote(text) {
      if (!text) { cancelAnimationFrame(raf); raf = 0; if (live) { live.remove(); live = null; } target = ''; shownN = 0; return; }
      if (!text.startsWith(target.slice(0, 40)) || text.length < shownN - 200) { shownN = 0; if (live) { live.remove(); live = null; } }
      target = text; if (!raf) raf = requestAnimationFrame(drawLive);
    }

    // ── what stands beside it: the program, its code, what holds ───
    const TABS = () => [...(wide ? [] : [['session', 'Session']]), ['preview', 'Running'], ['code', 'Code'], ['checks', checks.length ? `Checks ${checks.filter((r) => r.ok).length}/${checks.length}` : 'Checks']];
    function drawTabs() { const bar = $('.wb-tabs'); if (!bar) return; bar.innerHTML = TABS().map(([k, t]) => `<button role="tab" aria-selected="${k === tab}" class="${k === tab ? 'on' : ''}" data-tab="${k}">${t}</button>`).join('') + (tab === 'preview' && kept ? `<button class="wb-quiet wb-reload" data-again title="Begin it again">${icon('again')}</button>` : ''); if (tab === 'checks') drawView(); }
    async function drawView() {
      if (!wide) { scroller.hidden = tab !== 'session'; view.hidden = tab === 'session'; if (tab === 'session') return; }
      const id = chosen;
      if (tab === 'preview') {
        const at = view.querySelector('iframe'); const src = `/api/work/${id}/preview?v=${kept}`;
        if (at && at.dataset.src === src) return;
        view.className = 'wb-view wb-preview'; view.innerHTML = kept || (last && last.made) ? `<iframe title="What she has made, running" sandbox="allow-scripts" data-src="${src}" src="${src}"></iframe>` : '<div class="wb-empty">What she makes runs here as soon as there is something to run.</div>';
      } else if (tab === 'code') {
        if (!program || program.of !== id) { let its; try { its = await get(`/api/work/${id}/program`); } catch (e) { its = { parts: {} }; } if (id !== chosen || tab !== 'code') return; program = { ...its, of: id }; }
        const names = Object.keys(program.parts || {});
        if (!names.length) { view.className = 'wb-view'; view.innerHTML = '<div class="wb-empty">Nothing of it is written yet.</div>'; return; }
        if (!names.includes(part)) part = names[0];
        const text = program.parts[part] || '', language = program.language || 'javascript';
        view.className = 'wb-view wb-files'; view.innerHTML = `<nav>${names.map((n) => `<button data-part="${esc(n)}" class="${n === part ? 'on' : ''}">${esc(n.replace(/^.*:/, ''))}</button>`).join('')}</nav><pre class="wb-code wb-numbered">${text.split('\n').map((l) => `<div class="wb-l">${lit(l, language) || '&#8203;'}</div>`).join('')}</pre>`;
      } else if (tab === 'checks') {
        view.className = 'wb-view'; const good = checks.filter((r) => r.ok).length;
        view.innerHTML = checks.length ? `<div class="wb-score"><strong>${good}</strong> of ${checks.length} hold<span class="wb-meter"><i style="width:${(100 * good) / checks.length}%"></i></span></div><ul class="wb-checks">${[...checks].sort((a, b) => a.ok - b.ok).map(checkRow).join('')}</ul>` : `<div class="wb-empty">${design ? 'Its tests are being written.' : 'What has to hold of it shows here, and whether it does.'}</div>`;
      }
    }

    // ── what is going on now ───────────────────────────────────────
    function drawHead() {
      const w = last; if (!w) return;
      const label = { going: 'Working', held: 'Held', done: 'Done', stopped: 'Stopped' }[w.state] || w.state, going = w.state === 'going' || w.state === 'held';
      const s = $('.wb-state'); s.className = 'wb-state ' + w.state; s.lastElementChild.textContent = label; const t = $('.wb-title'); if (t.textContent !== w.title) t.textContent = w.title;
      const hold = $('[data-do=hold]'); hold.innerHTML = icon(w.state === 'held' ? 'go' : 'hold'); hold.title = w.state === 'held' ? 'Let it go on' : 'Hold it where it is'; hold.hidden = !going; $('.wb-stop').hidden = !going;
      if (!going && group && group.el.classList.contains('now')) { group.el.classList.remove('now'); group.el.classList.add('was'); group.icon.innerHTML = icon(w.state === 'done' ? 'done' : 'stop'); group.time.textContent = ''; }
      drawStatus();
    }
    function drawStatus() {
      const w = last, box = $('.wb-status'); if (!w) { box.textContent = ''; return; }
      const n = w.now || {}, now = Date.now() / 1000;
      if (w.state === 'going' || w.state === 'held') {
        const bits = []; if (w.began) bits.push(lasted(now - w.began)); if (n.tokens) bits.push(`${count(n.tokens)} tokens`); if (n.a_second) bits.push(`${n.a_second}/s`);
        const doing = w.state === 'held' ? 'Held' : n.reading ? `Reading what it has so far, ${Math.round((100 * n.reading[0]) / Math.max(1, n.reading[1]))}%` : w.stalled ? w.stalled[0].toUpperCase() + w.stalled.slice(1) : n.what || 'Working';
        const html = `${w.state === 'held' ? '' : '<span class="wb-spin"></span>'}<span class="wb-doing">${esc(doing)}</span><span class="wb-bits">${bits.join(' · ')}</span>`;
        if (box.__h !== html) { box.__h = html; box.innerHTML = html; }
        if (group && group.el.classList.contains('now')) group.time.textContent = lasted(now - group.at);
      } else { const html = `<span class="wb-doing">${w.state === 'done' ? 'Done' : 'Stopped'}${w.ended ? ' at ' + clock(w.ended) : ''}${w.ended && w.began ? ', after ' + lasted(w.ended - w.began) : ''}. Say something and she takes it up again.</span>`; if (box.__h !== html) { box.__h = html; box.innerHTML = html; } }
    }
    function came(w) {
      if (w.id !== chosen) return;
      last = { ...(last || {}), ...w, happened: undefined };
      const n = w.now || {};
      for (const h of w.happened || []) { if (h.n > after && (h.what === 'code' || h.what === 'tried' || h.what === 'step' || h.what === 'says')) wrote(''); happened(h); }
      if (typeof n.writing === 'string') wrote(n.writing); else if (!('writing' in n)) wrote('');
      drawHead(); if ((w.happened || []).length && pinned) requestAnimationFrame(() => down(false));
    }

    // ── streaming, with polling where there is none ────────────────
    function stream() {
      if (source) { source.close(); source = null; }
      if (!chosen || chosen === OWN || !visible() || stopped) return;
      writing = '';
      const id = chosen, s = new EventSource(`/api/work/${id}/stream?after=${after}`); source = s;
      s.addEventListener('seen', (e) => { if (id !== chosen) return; const w = JSON.parse(e.data); const more = w.now && w.now.more; if (w.now) { if (more) writing = writing.slice(0, more[0]) + more[1]; else if (!w.now.is_writing) writing = ''; if (w.now.is_writing) w.now.writing = writing; delete w.now.more; } came(w); });
      s.onerror = () => { s.close(); if (source === s) { source = null; clearTimeout(timer); timer = setTimeout(() => { if (id === chosen) { poll(); stream(); } }, 1500); } };
    }
    async function poll() { if (chosen === OWN) return own(); if (!chosen) return; try { came(await get(`/api/work/${chosen}?after=${after}`)); } catch (e) { /* asked again */ } }
    function railed(hers) {
      const rail = $('.wb-works'); if (!rail) return;
      const html = list.map((w) => `<button data-work="${esc(w.id)}" class="${w.id === chosen ? 'on' : ''}"><span class="wb-thumb">${w.icon ? `<img alt="" src="/api/work/${esc(w.id)}/picture/icon">` : icon('made')}</span><span class="wb-work-name">${esc(w.title)}<small>${{ going: 'Working', held: 'Held', done: 'Done', stopped: 'Stopped' }[w.state] || w.state} · ${clock(w.ended || w.began)}</small></span><i class="${w.state}"></i></button>`).join('')
        + `<button data-work="${OWN}" class="wb-hers ${chosen === OWN ? 'on' : ''}"><span class="wb-thumb">${icon('search')}</span><span class="wb-work-name">${hers.title}<small>With nobody asking</small></span></button>`;
      if (rail.__h !== html) { rail.__h = html; rail.innerHTML = html; }
    }
    async function works() {
      try { list = (await get('/api/work')).works || []; } catch (e) { return; }
      const button = document.querySelector('.tab-btn[data-tab="work"]');
      if (button) { const going = list.some((w) => w.state === 'going' || w.state === 'held'); button.classList.toggle('wb-going', going); button.title = going ? 'She is working on something' : 'Her work'; }
      if (!visible()) { if (source) { source.close(); source = null; } return; }
      const hers = { id: OWN, title: 'Her own time' };
      if (pick) { const html = [...list, hers].map((w) => `<option value="${esc(w.id)}">${esc(w.title)}</option>`).join(''); if (pick.__h !== html) { pick.__h = html; pick.innerHTML = html; } }
      if (chosen === OWN) own();
      if (!list.length && chosen !== OWN) { if (pick) pick.selectedIndex = -1; railed(hers); if (!log.children.length) log.innerHTML = '<div class="wb-empty">Ask her to make something and her work shows here as she does it: what she decides, the code as she writes it, what trying it shows, and the program running. You can speak to it while it goes.</div>'; return; }
      railed(hers);
      const going = list.find((w) => w.state === 'going');
      if (!chosen || (chosen !== OWN && !list.some((w) => w.id === chosen))) choose((going || list[0]).id);
      if (pick) pick.value = chosen;
      if (!source) stream();
    }

    root.addEventListener('click', async (e) => {
      const w = e.target.closest('[data-work]'); if (w) { choose(w.dataset.work); works(); return; }
      const t = e.target.closest('[data-tab]'); if (t && root.contains(t)) { tab = t.dataset.tab; drawTabs(); drawView(); return; }
      if (e.target.closest('[data-again]')) { const f = view.querySelector('iframe'); if (f) f.src = f.dataset.src + '&again=' + Date.now(); return; }
      const y = e.target.closest('.wb-why'); if (y) { y.classList.toggle('whole'); return; }
      const p = e.target.closest('[data-part]'); if (p) { part = p.dataset.part; drawView(); return; }
      const b = e.target.closest('[data-do]'); if (!b) return; const what = b.dataset.do;
      if (!chosen || chosen === OWN) { if (what === 'wide') window.open('/api/workbench', 'aura-workbench', 'width=1360,height=880'); return; }
      if (what === 'wide') return void window.open('/api/workbench', 'aura-workbench', 'width=1360,height=880');
      await post(`/api/work/${chosen}/${what === 'hold' && last && last.state === 'held' ? 'go_on' : what}`); if (!source) poll();
    });
    if (pick) pick.addEventListener('change', () => { choose(pick.value); });
    const grow = () => { say.style.height = 'auto'; say.style.height = Math.min(160, say.scrollHeight) + 'px'; };
    const send = async () => { const words = say.value.trim(); if (!words || !chosen || chosen === OWN) return; say.value = ''; grow(); pinned = true; await post(`/api/work/${chosen}/say`, { words }); if (!source) { poll(); stream(); } };
    $('.wb-say').addEventListener('submit', (e) => { e.preventDefault(); send(); });
    say.addEventListener('input', grow);
    say.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } if (e.key === 'Escape' && last && last.state === 'going') post(`/api/work/${chosen}/stop`); });
    const every = setInterval(() => { if (!stopped) works(); }, 3000), beat = setInterval(() => { if (visible()) drawStatus(); }, 500);
    const seen = () => { if (document.hidden) { if (source) { source.close(); source = null; } } else works(); };
    document.addEventListener('visibilitychange', seen); drawTabs(); works();
    return { stop() { stopped = true; clearInterval(every); clearInterval(beat); clearTimeout(timer); if (source) source.close(); document.removeEventListener('visibilitychange', seen); }, show: works };
  };
})();
