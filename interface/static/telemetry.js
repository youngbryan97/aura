/* ══════════════════════════════════════════════════════════
   AURA SOVEREIGN — Standalone Telemetry HUD Logic
   ══════════════════════════════════════════════════════════ */

const $ = id => document.getElementById(id);

const state = {
    ws: null,
    connected: false,
    beliefGraphInit: false
};

const DOM = {
    energy: $('bar-energy'),
    curiosity: $('bar-curiosity'),
    frustration: $('bar-frustration'),
    gwt: $('gwt-winner'),
    coherence: $('stat-coherence'),
    vitality: $('stat-vitality'),
    surprise: $('stat-surprise'),
    narrative: $('narrative-box'),
    monologue: $('monologue-feed')
};

function connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Force IPv4 for local testing to avoid ::1 resolution issues with Uvicorn
    const hostname = location.hostname === 'localhost' ? '127.0.0.1' : location.hostname;
    const port = location.port ? ':' + location.port : '';
    state.ws = new WebSocket(`${proto}//${hostname}${port}/ws`);

    // Application-layer heartbeat to prevent silent disconnects
    if (state.pingInterval) clearInterval(state.pingInterval);
    state.pingInterval = setInterval(() => {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({ type: 'ping' }));
        }
    }, 25000);

    state.ws.onopen = () => {
        logToFeed("SYSTEM", "WebSocket connection established.", "info");
        if (!state.beliefGraphInit) initBeliefGraph();
    };

    state.ws.onmessage = e => {
        try {
            const data = JSON.parse(e.data);
            handleWsEvent(data);
        } catch (err) {}
    };

    state.ws.onclose = () => {
        if (state.pingInterval) clearInterval(state.pingInterval);
        logToFeed("SYSTEM", "WebSocket disconnected. Reconnecting in 3s...", "error");
        setTimeout(connect, 3000);
    };

    state.ws.onerror = () => {};
}

function triggerVoiceOrb(state) {
    const micBtn = $('mic-btn');
    const orb = $('voice-orb');
    const stopIcon = $('stop-icon');
    const wrap = $('voice-orb-wrap');

    // Reset
    if (micBtn) micBtn.classList.remove('speaking', 'thinking');
    if (orb) orb.classList.remove('pulsing', 'spinning');
    if (stopIcon) stopIcon.classList.add('hidden');
    if (wrap) wrap.className = 'voice-orb-wrap';

    if (state === 'speaking') {
        if (micBtn) micBtn.classList.add('speaking');
        if (stopIcon) stopIcon.classList.remove('hidden');
        if (wrap) wrap.classList.add('speaking');
    } else if (state === 'thinking') {
        if (micBtn) micBtn.classList.add('thinking');
        if (wrap) wrap.classList.add('thinking');
    } else if (state === 'idle') {
        // back to normal
    }
}

function handleWsEvent(data) {
    const type = data.type;

    if (type === 'telemetry') {
        updateGauges(data);
    } else if (type === 'thought' || type === 'log') {
        const name = data.name || 'THOUGHT';
        const msg = data.message || data.content || "";
        logToFeed(name, msg, data.level?.toLowerCase() || "info");
    } else if (type === 'status' && data.narrative) {
        if (DOM.narrative) DOM.narrative.textContent = data.narrative;
    }
}

function updateGauges(data) {
    if (data.energy != null && DOM.energy) DOM.energy.style.width = data.energy + '%';
    if (data.curiosity != null && DOM.curiosity) DOM.curiosity.style.width = data.curiosity + '%';
    if (data.frustration != null && DOM.frustration) DOM.frustration.style.width = data.frustration + '%';

    if (data.gwt_winner && DOM.gwt) DOM.gwt.textContent = data.gwt_winner;
    if (data.coherence != null && DOM.coherence) DOM.coherence.textContent = (data.coherence * 100).toFixed(0) + '%';
    if (data.vitality != null && DOM.vitality) DOM.vitality.textContent = (data.vitality * 100).toFixed(0) + '%';
    if (data.surprise != null && DOM.surprise) DOM.surprise.textContent = (data.surprise * 100).toFixed(0) + '%';
    if (data.narrative && DOM.narrative) DOM.narrative.textContent = data.narrative;
}

function logToFeed(name, msg, level) {
    if (!DOM.monologue) return;
    const div = document.createElement('div');
    div.className = `log-entry ${level}`;
    const ts = new Date().toLocaleTimeString([], { hour12: false });
    div.innerHTML = `<span class="ts">[${ts}]</span> <span class="tag">@${name}:</span> ${msg}`;
    DOM.monologue.prepend(div);

    if (DOM.monologue.children.length > 50) DOM.monologue.lastChild.remove();
}

// ── Belief Graph (Minimal version for standalone) ─────────
let graphNetwork = null;
function initBeliefGraph() {
    const container = $('belief-graph-container');
    if (!container || typeof vis === 'undefined') return;
    state.beliefGraphInit = true;

    const options = {
        nodes: { shape: 'dot', size: 15, font: { color: '#ccc', face: 'monospace' }, color: { background: '#8a2be2', border: '#00e5ff' } },
        edges: { color: 'rgba(138, 43, 226, 0.4)' },
        physics: { stabilization: true }
    };
    graphNetwork = new vis.Network(container, { nodes: [], edges: [] }, options);
    refreshGraph();
}

async function refreshGraph() {
    try {
        const res = await fetch('/api/knowledge/graph');
        const d = await res.json();
        if (d.nodes && graphNetwork) {
            graphNetwork.setData({
                nodes: new vis.DataSet(d.nodes),
                edges: new vis.DataSet(d.edges || [])
            });
        }
    } catch (e) { }
    setTimeout(refreshGraph, 10000);
}

connect();