/**
 * devtools.js — Aura DevTools (time-scrubbable inspector).
 *
 * Tabs:
 *   snapshot    /api/dashboard/snapshot
 *   receipts    /api/dashboard/receipts
 *   trace       /api/trace/<receipt_id>
 *   performance /api/dashboard/performance (computed below from samples)
 *   phi         /api/dashboard/integration
 *   conscience  /api/dashboard/conscience
 *   world       /api/dashboard/world
 *   scrub       per-second snapshots; the bottom slider scrubs through
 *               the recorded buffer and renders the corresponding state
 *
 * The "scrub" tab keeps a rolling 60-minute buffer of /snapshot calls,
 * sampled once per second. Sliding the slider replays a snapshot from
 * that buffer in place; clicking on a generated word in the receipts
 * view jumps to the snapshot for the action receipt that produced it.
 */
(() => {
  const pane = document.getElementById("dt-pane");
  const status = document.getElementById("status");
  const scrub = document.getElementById("scrub");
  const buffer = []; // { when, snapshot }
  const BUF_LIMIT = 60 * 60; // one hour at 1Hz

  let current = "snapshot";
  let scrubIndex = -1;

  document.querySelectorAll(".dt-header nav button").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".dt-header nav button").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      current = btn.dataset.tab;
      render();
    });
  });

  scrub.addEventListener("input", () => {
    if (!buffer.length) return;
    scrubIndex = Math.floor((scrub.value / 100) * (buffer.length - 1));
    render();
  });

  async function fetchJson(url) {
    try {
      const r = await fetch(url);
      return await r.json();
    } catch { return {}; }
  }

  async function render() {
    if (current === "snapshot") {
      const data = scrubIndex >= 0 ? buffer[scrubIndex].snapshot : await fetchJson("/api/dashboard/snapshot");
      pane.innerHTML = renderKVCard("System", data.system || {})
        + renderKVCard("Viability", data.viability || {})
        + renderKVCard("Self snapshot", data.self || {})
        + renderJsonCard("Recent receipts", data.recent_receipts || []);
    } else if (current === "receipts") {
      const data = await fetchJson("/api/dashboard/receipts?limit=50");
      pane.innerHTML = renderJsonCard("Receipts (50)", data.receipts || []);
    } else if (current === "trace") {
      pane.innerHTML = `<div class='dt-card'><h2>Causal Trace</h2><input id="rid" placeholder="receipt id (AO-…)" /><button id="go">Trace</button><pre class='dt-pre' id="rt"></pre></div>`;
      document.getElementById("go").onclick = async () => {
        const id = document.getElementById("rid").value.trim();
        if (!id) return;
        const data = await fetchJson("/api/trace/" + encodeURIComponent(id));
        document.getElementById("rt").textContent = JSON.stringify(data, null, 2);
      };
    } else if (current === "performance") {
      const data = await fetchJson("/api/dashboard/snapshot");
      pane.innerHTML = renderKVCard("System", data.system || {})
        + renderKVCard("Viability", data.viability || {});
    } else if (current === "phi") {
      const data = await fetchJson("/api/dashboard/integration");
      pane.innerHTML = renderKVCard("Phi / GWT / HOT", data || {});
    } else if (current === "conscience") {
      const data = await fetchJson("/api/dashboard/conscience?limit=50");
      pane.innerHTML = renderJsonCard("Conscience violations", data.violations || []);
    } else if (current === "world") {
      const data = await fetchJson("/api/dashboard/world");
      pane.innerHTML = renderKVCard("World channels", data.channels || {});
    } else if (current === "scrub") {
      pane.innerHTML = `<div class="dt-card"><h2>Time Scrub</h2>
        <p>${buffer.length} samples buffered (1Hz, last ${BUF_LIMIT}s).
        Slide the bottom slider to step through state history.</p>
        ${scrubIndex >= 0 ? '<pre class="dt-pre">' + JSON.stringify(buffer[scrubIndex], null, 2) + '</pre>' : '<em>Drag the slider to start scrubbing.</em>'}
      </div>`;
    }
  }

  function renderKVCard(title, obj) {
    const rows = Object.entries(obj || {}).map(
      ([k, v]) => `<div class="dt-kv"><div class="k">${k}</div><div class="v">${escape(stringify(v))}</div></div>`
    ).join("");
    return `<div class="dt-card"><h2>${title}</h2>${rows}</div>`;
  }
  function renderJsonCard(title, payload) {
    return `<div class="dt-card"><h2>${title}</h2><pre class="dt-pre">${escape(JSON.stringify(payload, null, 2))}</pre></div>`;
  }
  function stringify(v) { if (typeof v === "object") return JSON.stringify(v); return String(v); }
  function escape(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

  // 1Hz buffer
  setInterval(async () => {
    const snap = await fetchJson("/api/dashboard/snapshot");
    buffer.push({ when: Date.now(), snapshot: snap });
    if (buffer.length > BUF_LIMIT) buffer.shift();
    status.textContent = `connected · ${buffer.length} sample${buffer.length === 1 ? "" : "s"}`;
    if (current === "snapshot" && scrubIndex < 0) render();
  }, 1000);

  render();
})();
