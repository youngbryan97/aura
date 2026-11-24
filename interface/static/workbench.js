/* workbench.js — her work as it goes, in the side panel or on a page of its own.
 *
 * Reads /api/work and /api/work/{id}?after=n, a little at a time, and only while it can be seen. What she is writing is
 * shown as she writes it; what has happened is shown in order; what is said here reaches the work. */
(function () {
  const ICON = {
    step: '<path d="M4 8h8M9 5l3 3-3 3"/>', says: '<path d="M3 4h10v7H7l-3 2.5V11H3z"/>', code: '<path d="M6 5 3 8l3 3M10 5l3 3-3 3"/>', tried: '<path d="M6 2.5h4M7 2.5v4L3.5 13h9L9 6.5v-4"/>',
    trouble: '<path d="M8 3 2.5 13h11zM8 7v3M8 11.6v.1"/>', made: '<path d="M3 5.5 8 3l5 2.5v5L8 13l-5-2.5zM3 5.5 8 8l5-2.5M8 8v5"/>', said: '<circle cx="8" cy="5.5" r="2.4"/><path d="M3.5 13c.6-2.6 2.3-4 4.5-4s3.9 1.4 4.5 4"/>',
    design: '<path d="M3 13V3h10v10zM3 6.5h10M6.5 6.5V13"/>', layer: '<path d="m8 3 5.5 3L8 9 2.5 6zM2.5 9.5 8 12.5l5.5-3"/>', other: '<circle cx="8" cy="8" r="2"/>', ok: '<path d="m3.5 8.5 3 3 6-7"/>', no: '<path d="m4.5 4.5 7 7M11.5 4.5l-7 7"/>',
    hold: '<path d="M6 4v8M10 4v8"/>', go: '<path d="M5.5 4v8l7-4z"/>', stop: '<rect x="4.5" y="4.5" width="7" height="7" rx="1"/>', wide: '<path d="M9.5 3H13v3.5M13 3 9 7M6.5 13H3V9.5M3 13l4-4"/>', open: '<path d="M7 4H3.5v8.5H12V9M9.5 3H13v3.5M13 3 8 8"/>',
  };
  const icon = (k) => `<svg class="wb-icon" viewBox="0 0 16 16" aria-hidden="true">${ICON[k] || ICON.other}</svg>`;
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const clock = (at) => new Date(at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const WORDS = { javascript: 'const let var function return if else for while of in new class extends async await true false null undefined this switch case break continue throw try catch finally typeof delete default import export yield', python: 'def class return if elif else for while in not and or is None True False import from as with try except finally raise lambda yield pass break continue global nonlocal assert async await', swift: 'let var func struct class enum return if else guard for while in import init self nil true false switch case break continue throws try catch extension protocol', rust: 'fn let mut struct enum impl trait return if else match for while loop in use mod pub self Self true false where as ref move', sql: 'select from where group by order insert into values create table update delete join left right on as and or not null limit having' };
  WORDS['c'] = WORDS['c++'] = WORDS.javascript + ' int char double float void struct unsigned long short static include define using namespace auto bool template';

  function lit(code, language) {
    const keys = new Set((WORDS[language] || WORDS.javascript).split(' '));
    const hash = language === 'python' || language === 'bash' || language === 'ruby' || language === 'perl';
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
  const codeIn = (text) => { const at = text.indexOf('```'); if (at < 0) return /^\s*[{(]/.test(text) ? { code: text, said: '' } : { code: '', said: text }; const rest = text.slice(at + 3).replace(/^[\w+-]*\n/, ''); const end = rest.indexOf('```'); return { said: text.slice(0, at).trim(), code: end < 0 ? rest : rest.slice(0, end) }; };

  window.mountWorkbench = function (root, options) {
    const wide = !!(options && options.wide);
    let chosen = '', after = 0, seen = [], files = new Map(), tab = 'writing', timer = null, stopped = false, pinned = true, list = [], last = null;
    root.innerHTML = `<div class="wb-head"><select aria-label="Which work"></select><span class="wb-state"><i></i><span></span></span>
      <button data-do="hold">${icon('hold')}<span>Hold</span></button><button data-do="stop">${icon('stop')}Stop</button><button data-do="open">${icon('open')}Open it</button>${wide ? '' : `<button data-do="wide" title="Open the workbench in a window of its own">${icon('wide')}</button>`}</div>
      <div class="wb-now" aria-live="polite"></div><div class="wb-body"><div class="wb-feed"></div><div class="wb-code-side"><div class="wb-tabs" role="tablist"></div><div class="wb-block"><header></header><pre><code></code></pre></div></div></div>
      <form class="wb-say"><textarea rows="1" placeholder="Say something to this work: change it, ask about it, add to it" aria-label="Say something to this work"></textarea><button type="submit">Say</button></form>`;
    const $ = (q) => root.querySelector(q);
    const pick = $('select'), stateEl = $('.wb-state'), nowEl = $('.wb-now'), feed = $('.wb-feed'), tabs = $('.wb-tabs'), say = $('.wb-say textarea');
    const get = async (url) => { const r = await fetch(url, { credentials: 'same-origin' }); if (!r.ok) throw new Error(String(r.status)); return r.json(); };
    const post = (url, body) => fetch(url, { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : '{}' });
    const visible = () => root.isConnected && root.offsetParent !== null && !document.hidden;

    function choose(id) { if (id === chosen) return; chosen = id; after = 0; seen = []; files = new Map(); tab = 'writing'; last = null; feed.innerHTML = ''; pinned = true; }
    feed.addEventListener('scroll', () => { pinned = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 40; });
    pick.addEventListener('change', () => { choose(pick.value); tick(); });
    root.addEventListener('click', async (e) => {
      const b = e.target.closest('[data-do]'); if (b && chosen) { const what = b.dataset.do; if (what === 'wide') return void window.open('/api/workbench', 'aura-workbench', 'width=1280,height=860'); await post(`/api/work/${chosen}/${what === 'hold' && last && last.state === 'held' ? 'go_on' : what}`); tick(); }
      const t = e.target.closest('[data-tab]'); if (t) { tab = t.dataset.tab; drawCode(); }
    });
    const send = async () => { const words = say.value.trim(); if (!words || !chosen) return; say.value = ''; await post(`/api/work/${chosen}/say`, { words }); tick(); };
    $('.wb-say').addEventListener('submit', (e) => { e.preventDefault(); send(); });
    say.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });

    function row(kind, html) { const d = document.createElement('div'); d.className = 'wb-row'; d.innerHTML = icon(kind) + `<div>${html}</div>`; return d; }
    function drawn(h) {
      if (h.what === 'step') return row('step', `<span class="wb-step">${esc(h.title)}</span><span class="wb-when">${clock(h.at)}</span>`);
      if (h.what === 'says') return row('says', `<div class="wb-says">${esc(h.words)}</div>`);
      if (h.what === 'said') return row('said', `<div class="wb-said">${esc(h.words)}</div>`);
      if (h.what === 'trouble') return row('trouble', `<div class="wb-trouble">${esc(h.why)}</div>`);
      if (h.what === 'made') return row('made', `<strong>${esc(h.title)}</strong><div class="wb-why">${esc(h.where)}</div>`);
      if (h.what === 'layer') return row('layer', `Layer ${h.n} of ${h.of}: ${esc((h.rules || []).join('; '))}`);
      if (h.what === 'design') { const d = h.design || {}; return row('design', `<details class="wb-checks"><summary>The design: ${esc(d.name)}, ${(d.rules || []).length} rules</summary><ul>${(d.rules || []).map((r) => `<li><span></span><span>${esc(r.says)}${r.of ? ` <span class="wb-why">“${esc(r.of)}”</span>` : ''}</span></li>`).join('')}</ul></details>`); }
      if (h.what === 'tried') { const bad = (h.results || []).filter((r) => !r.ok); return row('tried', `<details class="wb-checks" ${bad.length && bad.length < 9 ? 'open' : ''}><summary>${esc(h.title)}: ${h.good} of ${h.of} hold</summary><div class="wb-bar"><div style="width:${h.of ? (100 * h.good) / h.of : 0}%"></div></div><ul>${[...bad, ...(h.results || []).filter((r) => r.ok)].map((r) => `<li class="${r.ok ? 'ok' : 'no'}">${icon(r.ok ? 'ok' : 'no')}<span>${esc(r.name)}${r.why ? `<div class="wb-why">${esc(r.why)}</div>` : ''}</span></li>`).join('')}</ul></details>`); }
      if (h.what === 'code') return row('code', wide ? `<button data-tab="${esc(h.name)}">${esc(h.name)}, ${h.lines} lines</button>` : `<details class="wb-block"><summary style="padding:6px 9px;cursor:pointer">${esc(h.name)}, ${h.lines} lines</summary><pre><code>${lit(h.code, h.language)}</code></pre></details>`);
      return row('other', `<span class="wb-why">${esc(h.what)}</span><span class="wb-when">${clock(h.at)}</span>`);
    }
    function drawCode() {
      const live = last && last.now && last.now.writing;
      if (!wide) { let box = feed.querySelector('.wb-live'); if (!live) { if (box) box.remove(); return; } if (!box) { box = document.createElement('div'); box.className = 'wb-block wb-live'; feed.appendChild(box); } else feed.appendChild(box); const c = codeIn(live); box.innerHTML = `<header><span>writing</span><span>${last.now.tokens || 0} tokens</span><span>${last.now.a_second || ''}${last.now.a_second ? ' a second' : ''}</span></header>${c.said ? `<div class="wb-says" style="padding:8px 10px">${esc(c.said)}</div>` : ''}${c.code ? `<pre><code>${lit(c.code.slice(-6000), 'javascript')}<span class="wb-caret"></span></code></pre>` : ''}`; const pre = box.querySelector('pre'); if (pre) pre.scrollTop = pre.scrollHeight; return; }
      const names = [...(live ? ['writing'] : []), ...[...files.keys()].reverse()];
      if (!names.includes(tab)) tab = names[0] || 'writing';
      tabs.innerHTML = names.map((n) => `<button role="tab" data-tab="${esc(n)}" class="${n === tab ? 'on' : ''}">${esc(n)}</button>`).join('');
      const head = $('.wb-code-side header'), code = $('.wb-code-side code'), pre = $('.wb-code-side pre');
      if (tab === 'writing' && live) { const c = codeIn(live), down = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 60; head.innerHTML = `<span>as she writes it</span><span>${last.now.tokens || 0} tokens</span><span>${last.now.a_second ? last.now.a_second + ' a second' : ''}</span>`; code.innerHTML = (c.said ? `<span class="wb-c">${esc(c.said)}</span>\n\n` : '') + lit(c.code, 'javascript') + '<span class="wb-caret"></span>'; if (down) pre.scrollTop = pre.scrollHeight; }
      else { const f = files.get(tab); head.innerHTML = f ? `<span>${esc(tab)}</span><span>${f.lines} lines</span><span>${clock(f.at)}</span>` : '<span>Nothing is written yet</span>'; code.innerHTML = f ? lit(f.code, f.language) : ''; }
    }
    function drawHead() {
      const w = last; if (!w) return;
      const label = { going: 'working', held: 'held', done: 'done', stopped: 'stopped' }[w.state] || w.state;
      stateEl.className = 'wb-state ' + w.state; stateEl.querySelector('span').textContent = label;
      const hold = $('[data-do=hold]'); hold.innerHTML = w.state === 'held' ? icon('go') + '<span>Go on</span>' : icon('hold') + '<span>Hold</span>';
      const going = w.state === 'going' || w.state === 'held'; hold.disabled = !going; $('[data-do=stop]').disabled = !going;
      const n = w.now || {}; const secs = n.since ? Math.round(Date.now() / 1000 - n.since) : 0;
      nowEl.innerHTML = going ? `<strong>${esc(n.what || 'Beginning')}</strong><span>${secs ? secs + ' s' : ''}</span>${w.stalled ? `<span class="wb-stall">${esc(w.stalled)}</span>` : n.writing ? `<span>writing, ${n.tokens || 0} tokens${n.a_second ? ', ' + n.a_second + ' a second' : ''}</span>` : ''}` : `<span>${w.state === 'done' ? 'Finished' : 'Ended'} ${w.ended ? clock(w.ended) : ''}. Say something below to take it up again.</span>`;
    }
    async function tick() {
      clearTimeout(timer); if (stopped) return;
      let wait = 3000;
      try {
        if (!visible() && !document.hidden) {
          list = (await get('/api/work')).works || []; wait = 6000;
        }
        if (visible()) {
          list = (await get('/api/work')).works || [];
          if (!list.length) { feed.innerHTML = '<div class="wb-empty">When you ask her to make something, her work shows here as she does it: what she decides, the code as she writes it, what trying it shows, and what she is waiting on. You can speak to it while it goes.</div>'; nowEl.textContent = ''; }
          else {
            const had = pick.value; pick.innerHTML = list.map((w) => `<option value="${esc(w.id)}">${esc(w.title)}</option>`).join('');
            const going = list.find((w) => w.state === 'going');
            if (!chosen || !list.some((w) => w.id === chosen)) choose((going || list[0]).id); else if (had && had !== chosen) choose(had);
            pick.value = chosen;
            const w = await get(`/api/work/${chosen}?after=${after}`); last = w;
            for (const h of w.happened || []) { after = Math.max(after, h.n); seen.push(h); if (h.what === 'code') files.set(h.name, h); feed.appendChild(drawn(h)); }
            while (feed.children.length > 600) feed.removeChild(feed.firstChild);
            drawHead(); drawCode(); if (pinned) feed.scrollTop = feed.scrollHeight;
            wait = w.state === 'going' ? 700 : 2500;
          }
        }
        const button = document.querySelector('.tab-btn[data-tab="work"]');
        if (button) { const going = list.some((w) => w.state === 'going' || w.state === 'held'); button.classList.toggle('wb-going', going); button.title = going ? 'She is working on something' : 'Her work'; }
      } catch (e) { wait = 4000; }
      timer = setTimeout(tick, wait);
    }
    tick();
    document.addEventListener('visibilitychange', tick);
    return { stop() { stopped = true; clearTimeout(timer); }, show: tick };
  };
})();
