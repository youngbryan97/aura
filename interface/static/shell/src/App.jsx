import React, { startTransition, useDeferredValue, useEffect, useRef, useState } from "react";

const TABS = ["neural", "telemetry", "memory", "tools", "settings"];

const DEFAULT_BOOTSTRAP = {
  identity: { name: "Aura Luna", version: "offline", build: "" },
  session: { connected: false, initialized: false, websocket_clients: 0 },
  constitutional: { recent_decisions: [], belief_summary: {} },
  executive: {},
  state: {
    current_objective: "",
    pending_initiatives: 0,
    active_goals: 0,
    policy_mode: "unknown",
    health: {},
    rolling_summary: "",
    coherence_score: 1,
    fragmentation_score: 0,
    contradiction_count: 0,
    phenomenal_state: "",
    thermal_guard: false,
    health_flags: [],
    epistemics: {},
  },
  commitments: { active_count: 0, reliability_score: 1, active: [] },
  tools: [],
  conversation: { recent: [], count: 0 },
  voice: { available: false, microphone_enabled: false, state: "unavailable" },
  telemetry: { cpu_usage: 0, ram_usage: 0, runtime: {}, boot: {} },
  ui: { shell: "react_shell", legacy_fallback_available: true, status_flags: [] },
  timestamp: "",
};

function formatPercent(value) {
  const numeric = Number(value || 0);
  return `${numeric.toFixed(numeric >= 10 ? 0 : 1)}%`;
}

function formatScore(value) {
  return Number(value || 0).toFixed(2);
}

function formatClock(timestamp) {
  if (!timestamp) return "--";
  try {
    const numeric = Number(timestamp);
    const millis = Number.isFinite(numeric) && numeric < 10_000_000_000 ? numeric * 1000 : timestamp;
    return new Date(millis).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "--";
  }
}

function normalizeConversation(conversation) {
  const recent = Array.isArray(conversation?.recent) ? conversation.recent : [];
  const messages = [];
  for (const exchange of recent) {
    if (exchange?.user) {
      messages.push({
        id: `${exchange.timestamp || exchange.created_at || Math.random()}-user`,
        role: "user",
        content: exchange.user,
        createdAt: exchange.timestamp || exchange.created_at || Date.now(),
      });
    }
    if (exchange?.aura) {
      messages.push({
        id: `${exchange.timestamp || exchange.created_at || Math.random()}-assistant`,
        role: "assistant",
        content: exchange.aura,
        createdAt: exchange.timestamp || exchange.created_at || Date.now(),
      });
    }
  }
  return messages;
}

function summarizeTool(tool) {
  const state = tool.available ? "Available" : "Unavailable";
  const qualifier = tool.degraded_reason || tool.last_error || tool.route_class;
  return `${state} | ${qualifier}`;
}

function buildStatusSignals(bootstrap, connectionState) {
  const signals = [];
  const seen = new Set();
  const healthFlags = Array.isArray(bootstrap.state?.health_flags) ? bootstrap.state.health_flags : [];
  const uiFlags = Array.isArray(bootstrap.ui?.status_flags) ? bootstrap.ui.status_flags : [];
  const flags = [...uiFlags, ...healthFlags];

  if (connectionState === "booting") flags.unshift("booting");
  if (connectionState === "reconnecting") flags.unshift("reconnecting");
  if (connectionState === "degraded") flags.unshift("degraded_connection");

  for (const flag of flags) {
    if (!flag || seen.has(flag)) continue;
    seen.add(flag);
    switch (flag) {
      case "booting":
        signals.push({ key: flag, tone: "warn", title: "Booting", body: "Aura is still binding runtime organs and live contracts." });
        break;
      case "reconnecting":
        signals.push({ key: flag, tone: "warn", title: "Reconnecting", body: "Realtime transport dropped. The shell is retrying live deltas." });
        break;
      case "degraded_connection":
        signals.push({ key: flag, tone: "danger", title: "Connection Degraded", body: "Operational truth may lag while transport recovers." });
        break;
      case "thermal_guard":
        signals.push({ key: flag, tone: "warn", title: "Thermal Guard", body: "Inference depth is being downshifted to preserve runtime stability." });
        break;
      case "tool_unavailable":
        signals.push({ key: flag, tone: "info", title: "Tool Degraded", body: "Some governed tools are offline, blocked, or cooling down." });
        break;
      case "executive_hold":
        signals.push({ key: flag, tone: "info", title: "Executive Hold", body: "Autonomous expression is being held to a secondary lane for stability." });
        break;
      case "coherence_low":
        signals.push({ key: flag, tone: "danger", title: "Coherence Low", body: "Aura is biasing toward stabilization and continuity repair." });
        break;
      case "fragmentation_high":
        signals.push({ key: flag, tone: "warn", title: "Fragmentation High", body: "Context load is elevated. Compaction and summary pressure are active." });
        break;
      case "contradictions_present":
        signals.push({ key: flag, tone: "danger", title: "Contradictions Active", body: "Conflicting internal claims are present and require reconciliation." });
        break;
      case "beliefs_contested":
        signals.push({ key: flag, tone: "warn", title: "Beliefs Contested", body: "Epistemic state contains contested beliefs, so confidence is constrained." });
        break;
      default:
        signals.push({ key: flag, tone: "info", title: flag.replace(/_/g, " "), body: "Operational state changed." });
    }
  }

  return signals;
}

export default function App() {
  const [bootstrap, setBootstrap] = useState(DEFAULT_BOOTSTRAP);
  const [connectionState, setConnectionState] = useState("booting");
  const [activeTab, setActiveTab] = useState("neural");
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(true);
  const [messages, setMessages] = useState([]);
  const [streamingMessage, setStreamingMessage] = useState("");
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [telemetry, setTelemetry] = useState({ cpu_usage: 0, ram_usage: 0, cortex: {}, resilience: {}, executive_closure: {} });
  const [neuralFeed, setNeuralFeed] = useState([]);
  const [diagnostics, setDiagnostics] = useState([]);
  const [toolEvents, setToolEvents] = useState([]);
  const [toast, setToast] = useState("");
  const wsRef = useRef(null);
  const reconnectRef = useRef(null);
  const reconnectAttemptRef = useRef(0);
  const seenEventIds = useRef(new Set());
  const deferredMessages = useDeferredValue(messages);

  useEffect(() => {
    let mounted = true;

    const loadBootstrap = async () => {
      try {
        const response = await fetch("/api/ui/bootstrap", { credentials: "same-origin" });
        if (!response.ok) throw new Error(`Bootstrap failed (${response.status})`);
        const payload = await response.json();
        if (!mounted) return;
        startTransition(() => {
          setBootstrap(payload);
          setTelemetry(payload.telemetry || {});
          setMessages(normalizeConversation(payload.conversation));
        });
        setConnectionState(payload.session?.connected ? "connected" : "degraded");
      } catch (error) {
        if (!mounted) return;
        setConnectionState("degraded");
        setToast(error instanceof Error ? error.message : "Bootstrap failed");
      }
    };

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      const hostname = location.hostname === "localhost" ? "127.0.0.1" : location.hostname;
      const port = location.port ? `:${location.port}` : "";
      const ws = new WebSocket(`${proto}//${hostname}${port}/ws`);
      wsRef.current = ws;

      ws.onopen = () => {
        reconnectAttemptRef.current = 0;
        setConnectionState("connected");
        setToast("");
      };

      ws.onclose = () => {
        setConnectionState("reconnecting");
        reconnectAttemptRef.current++;
        const baseDelay = Math.min(30000, 1000 * Math.pow(2, reconnectAttemptRef.current));
        const jitter = Math.random() * 500;
        reconnectRef.current = window.setTimeout(connect, baseDelay + jitter);
      };

      ws.onerror = () => {
        setConnectionState("degraded");
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          const eventId = payload.event_id || payload.id;
          if (eventId && seenEventIds.current.has(eventId)) return;
          if (eventId) {
            seenEventIds.current.add(eventId);
            if (seenEventIds.current.size > 1200) {
              const trimmed = [...seenEventIds.current].slice(-800);
              seenEventIds.current = new Set(trimmed);
            }
          }
          handleEvent(payload);
        } catch (error) {
          console.error("Failed to parse websocket event", error);
        }
      };
    };

    loadBootstrap();
    connect();
    const refreshTimer = window.setInterval(loadBootstrap, 30000);

    return () => {
      mounted = false;
      window.clearInterval(refreshTimer);
      if (reconnectRef.current) window.clearTimeout(reconnectRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  function pushDiagnostics(entry) {
    startTransition(() => {
      setDiagnostics((current) => [entry, ...current].slice(0, 150));
    });
  }

  function appendMessage(message) {
    startTransition(() => {
      setMessages((current) => {
        const last = current[current.length - 1];
        if (last && last.role === message.role && last.content === message.content) {
          return current;
        }
        return [...current, message];
      });
    });
  }

  function handleEvent(payload) {
    const kind = payload.kind || payload.type;
    if (kind === "heartbeat" || kind === "ping") return;

    if (kind === "telemetry") {
      startTransition(() => {
        setTelemetry((current) => ({ ...current, ...payload.payload, ...payload }));
      });
      return;
    }

    if (kind === "thought") {
      const item = {
        id: payload.event_id || crypto.randomUUID(),
        content: payload.content || payload.payload?.content || "...",
        phase: payload.cognitive_phase || payload.payload?.cognitive_phase || "cognition",
        at: payload.event_ts || payload.timestamp || Date.now(),
      };
      startTransition(() => {
        setNeuralFeed((current) => [item, ...current].slice(0, 60));
      });
      return;
    }

    if (kind === "chat_stream_chunk") {
      setStreamingMessage((current) => `${current}${payload.chunk || payload.payload?.chunk || ""}`);
      return;
    }

    if (kind === "aura_message" || kind === "chat_response") {
      const content = payload.message || payload.payload?.message || payload.content || payload.payload?.content;
      if (content) {
        appendMessage({
          id: payload.event_id || crypto.randomUUID(),
          role: "assistant",
          content,
          createdAt: payload.event_ts || payload.timestamp || Date.now(),
        });
        setStreamingMessage("");
      }
      return;
    }

    if (kind === "tool_event") {
      const item = {
        id: payload.event_id || crypto.randomUUID(),
        stage: payload.stage,
        tool: payload.tool,
        success: payload.success,
        reason: payload.decision?.reason || payload.error || "",
        at: payload.event_ts || payload.timestamp || Date.now(),
      };
      startTransition(() => {
        setToolEvents((current) => [item, ...current].slice(0, 50));
      });
      pushDiagnostics({
        id: item.id,
        level: item.success === false ? "error" : item.stage === "rejected" ? "warn" : "info",
        message: `${item.tool} | ${item.stage}${item.reason ? ` | ${item.reason}` : ""}`,
        at: item.at,
      });
      return;
    }

    if (kind === "log") {
      pushDiagnostics({
        id: payload.event_id || crypto.randomUUID(),
        level: payload.level || "info",
        message: payload.message || payload.payload?.message || "",
        at: payload.event_ts || payload.timestamp || Date.now(),
      });
      return;
    }

    if (kind === "skill_status") {
      startTransition(() => {
        setBootstrap((current) => ({
          ...current,
          tools: current.tools.map((tool) =>
            tool.name === payload.skill
              ? { ...tool, state: payload.state, availability: payload.state === "ERROR" ? "unavailable" : tool.availability }
              : tool,
          ),
        }));
      });
    }
  }

  async function sendMessage(event) {
    event.preventDefault();
    const content = input.trim();
    if (!content || sending) return;

    appendMessage({
      id: crypto.randomUUID(),
      role: "user",
      content,
      createdAt: Date.now(),
    });
    setInput("");
    setSending(true);
    setStreamingMessage("");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: content }),
      });
      if (!response.ok) throw new Error(`Chat failed (${response.status})`);
      const payload = await response.json();
      if (payload.response) {
        appendMessage({
          id: crypto.randomUUID(),
          role: "assistant",
          content: payload.response,
          createdAt: Date.now(),
        });
      }
    } catch (error) {
      appendMessage({
        id: crypto.randomUUID(),
        role: "system",
        content: error instanceof Error ? error.message : "Request failed.",
        createdAt: Date.now(),
      });
      setToast(error instanceof Error ? error.message : "Request failed");
    } finally {
      setSending(false);
    }
  }

  async function regenerate() {
    setSending(true);
    try {
      const response = await fetch("/api/chat/regenerate", { method: "POST" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.message || payload.error || "Regenerate failed");
      if (payload.response) {
        appendMessage({
          id: crypto.randomUUID(),
          role: "assistant",
          content: payload.response,
          createdAt: Date.now(),
        });
      }
    } catch (error) {
      setToast(error instanceof Error ? error.message : "Regenerate failed");
    } finally {
      setSending(false);
    }
  }

  async function retryBrain() {
    try {
      const response = await fetch("/api/brain/retry", { method: "POST" });
      const payload = await response.json();
      setToast(payload.status === "retry_sent" ? "Brain retry signaled." : "Brain retry unavailable.");
    } catch (error) {
      setToast("Brain retry failed — check connection");
    }
  }

  async function exportConversation() {
    window.open("/api/export", "_blank", "noopener,noreferrer");
  }

  const closure = bootstrap.executive?.last_reason || telemetry.executive_closure?.dominant_need || "steady";
  const epistemics = bootstrap.state.epistemics || bootstrap.constitutional.belief_summary || {};
  const cognitiveHealth = bootstrap.state.health?.cognitive_health || {};
  const statusSignals = buildStatusSignals(bootstrap, connectionState);
  const headerStats = [
    { label: "State", value: connectionState },
    { label: "Objective", value: bootstrap.state.current_objective || "idle" },
    { label: "Pending", value: String(bootstrap.state.pending_initiatives || 0) },
    { label: "Commitments", value: String(bootstrap.commitments.active_count || 0) },
    { label: "Coherence", value: formatScore(bootstrap.state.coherence_score || 0) },
    { label: "CPU", value: formatPercent(telemetry.cpu_usage || bootstrap.telemetry.cpu_usage) },
    { label: "RAM", value: formatPercent(telemetry.ram_usage || bootstrap.telemetry.ram_usage) },
    { label: "Closure", value: String(closure).slice(0, 22) },
    { label: "Voice", value: bootstrap.voice.state || "offline" },
  ];

  const availableTools = bootstrap.tools.filter((tool) => tool.available);
  const unavailableTools = bootstrap.tools.filter((tool) => !tool.available);

  return (
    <div className={`shell ${connectionState}`}>
      <div className="cosmos" aria-hidden="true">
        <div className="cosmos-gradient cosmos-gradient-a" />
        <div className="cosmos-gradient cosmos-gradient-b" />
        <div className="cosmos-grid" />
      </div>

      <header className="topbar">
        <div className="brand">
          <div className="brand-avatar" />
          <div className="brand-copy">
            <div className="brand-title">{bootstrap.identity.name}</div>
            <div className="brand-subtitle">{bootstrap.identity.version}</div>
          </div>
        </div>

        <div className="stats-ribbon">
          {headerStats.map((item) => (
            <div className="stat-chip" key={item.label}>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </div>
          ))}
        </div>

        <div className="toolbar">
          <button type="button" onClick={retryBrain}>Retry Brain</button>
          <button type="button" onClick={exportConversation}>Export</button>
          <button type="button" onClick={() => window.open("/memory", "_blank", "noopener,noreferrer")}>Black Hole</button>
          <button type="button" onClick={() => setDiagnosticsOpen((open) => !open)}>
            {diagnosticsOpen ? "Hide Diagnostics" : "Show Diagnostics"}
          </button>
        </div>
      </header>

      {toast ? <div className="toast">{toast}</div> : null}

      {statusSignals.length ? (
        <section className="status-rail" aria-label="Operational state">
          {statusSignals.map((signal) => (
            <article className={`status-pill ${signal.tone}`} key={signal.key}>
              <strong>{signal.title}</strong>
              <p>{signal.body}</p>
            </article>
          ))}
        </section>
      ) : null}

      <main className="workspace">
        <section className="chat-panel panel">
          <div className="panel-header">
            <div>
              <div className="eyebrow">Primary Channel</div>
              <h1>Conversation</h1>
            </div>
            <div className="voice-cluster">
              <div className={`voice-orb ${bootstrap.voice.available ? "available" : "disabled"} ${sending ? "active" : ""}`} />
              <div className="voice-meta">
                <span>Voice</span>
                <strong>{bootstrap.voice.state || "offline"}</strong>
              </div>
            </div>
          </div>

          <div className="messages">
            {deferredMessages.map((message) => (
              <article className={`message ${message.role}`} key={message.id}>
                <div className="message-role">{message.role}</div>
                <div className="message-content">{message.content}</div>
                <div className="message-meta">{formatClock(message.createdAt)}</div>
              </article>
            ))}
            {streamingMessage ? (
              <article className="message assistant streaming">
                <div className="message-role">assistant</div>
                <div className="message-content">{streamingMessage}</div>
              </article>
            ) : null}
          </div>

          <form className="composer" onSubmit={sendMessage}>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Talk to Aura..."
              rows={3}
            />
            <div className="composer-actions">
              <button type="button" className="secondary" onClick={regenerate} disabled={sending}>
                Regenerate
              </button>
              <button type="submit" disabled={sending || !input.trim()}>
                {sending ? "Sending..." : "Send"}
              </button>
            </div>
          </form>
        </section>

        <aside className="sidebar panel">
          <div className="tabs">
            {TABS.map((tab) => (
              <button
                type="button"
                key={tab}
                className={tab === activeTab ? "active" : ""}
                onClick={() => setActiveTab(tab)}
              >
                {tab}
              </button>
            ))}
          </div>

          <div className="tab-body">
            {activeTab === "neural" ? (
              <div className="tab-section">
                <div className="section-title">Neural Feed</div>
                <div className="feed-list">
                  {neuralFeed.length === 0 ? <EmptyState label="No active neural feed yet." /> : null}
                  {neuralFeed.map((item) => (
                    <div className="feed-card" key={item.id}>
                      <div className="feed-meta">
                        <span>{item.phase}</span>
                        <span>{formatClock(item.at)}</span>
                      </div>
                      <div className="feed-content">{item.content}</div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {activeTab === "telemetry" ? (
              <div className="tab-section">
                <div className="section-title">Telemetry</div>
                <MetricBar label="CPU" value={telemetry.cpu_usage || bootstrap.telemetry.cpu_usage || 0} />
                <MetricBar label="RAM" value={telemetry.ram_usage || bootstrap.telemetry.ram_usage || 0} />
                <MetricBar label="Curiosity" value={telemetry.cortex?.curiosity || 0} max={100} />
                <MetricBar label="Agency" value={telemetry.cortex?.agency || 0} max={100} />
                <MetricBar label="Closure" value={(telemetry.executive_closure?.need_pressure || 0) * 100} max={100} />
                <MetricBar label="Coherence" value={(bootstrap.state.coherence_score || 0) * 100} max={100} />
                <MetricBar label="Fragmentation" value={(bootstrap.state.fragmentation_score || 0) * 100} max={100} />
                <MetricBar label="Epistemic integrity" value={(epistemics.coherence_score || 0) * 100} max={100} />
                <div className="mini-grid">
                  <MiniStat label="Policy" value={bootstrap.state.policy_mode || "unknown"} />
                  <MiniStat label="Goals" value={String(bootstrap.state.active_goals || 0)} />
                  <MiniStat label="Clients" value={String(bootstrap.session.websocket_clients || 0)} />
                  <MiniStat label="Boot" value={bootstrap.session.initialized ? "ready" : "booting"} />
                </div>
              </div>
            ) : null}

            {activeTab === "memory" ? (
              <div className="tab-section">
                <div className="section-title">Continuity + Memory</div>
                <MiniStat label="Current objective" value={bootstrap.state.current_objective || "idle"} large />
                <MiniStat label="Pending initiatives" value={String(bootstrap.state.pending_initiatives || 0)} large />
                <MiniStat label="Commitment reliability" value={`${Math.round((bootstrap.commitments.reliability_score || 0) * 100)}%`} large />
                <MiniStat label="Contradictions" value={String(bootstrap.state.contradiction_count || 0)} large />
                {bootstrap.state.phenomenal_state ? (
                  <div className="summary-card">
                    <div className="feed-meta">
                      <span>Phenomenal field</span>
                      <span>{formatScore(bootstrap.state.coherence_score || 0)} coherence</span>
                    </div>
                    <div className="feed-content">{bootstrap.state.phenomenal_state}</div>
                  </div>
                ) : null}
                {bootstrap.state.rolling_summary ? (
                  <div className="summary-card">
                    <div className="feed-meta">
                      <span>Subject thread</span>
                      <span>{cognitiveHealth.working_memory_items || bootstrap.conversation.count || 0} hot items</span>
                    </div>
                    <div className="feed-content">{bootstrap.state.rolling_summary}</div>
                  </div>
                ) : null}
                <div className="feed-list compact">
                  {bootstrap.commitments.active.length === 0 ? <EmptyState label="No active commitments." /> : null}
                  {bootstrap.commitments.active.map((commitment) => (
                    <div className="feed-card" key={commitment.id}>
                      <div className="feed-meta">
                        <span>{commitment.status}</span>
                        <span>{commitment.hours_remaining}h</span>
                      </div>
                      <div className="feed-content">{commitment.description}</div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {activeTab === "tools" ? (
              <div className="tab-section">
                <div className="section-title">Tool Surface</div>
                <div className="tool-summary">
                  <MiniStat label="Available" value={String(availableTools.length)} />
                  <MiniStat label="Unavailable" value={String(unavailableTools.length)} />
                </div>
                <div className="feed-list compact">
                  {bootstrap.tools.map((tool) => (
                    <div className={`tool-card ${tool.available ? "available" : "unavailable"}`} key={tool.name}>
                      <div className="tool-card-top">
                        <strong>{tool.name}</strong>
                        <span>{tool.risk_class}</span>
                      </div>
                      <p>{tool.description}</p>
                      <div className="tool-meta-grid">
                        <div>
                          <span>Route</span>
                          <strong>{tool.route_class}</strong>
                        </div>
                        <div>
                          <span>Inputs</span>
                          <strong>{tool.input_summary}</strong>
                        </div>
                        <div>
                          <span>When to use</span>
                          <strong>{tool.example_usage}</strong>
                        </div>
                        <div>
                          <span>Timeout</span>
                          <strong>{tool.timeout_seconds}s</strong>
                        </div>
                      </div>
                      <div className="tool-foot">{summarizeTool(tool)}</div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {activeTab === "settings" ? (
              <div className="tab-section">
                <div className="section-title">Runtime State</div>
                <MiniStat label="Shell" value={bootstrap.ui.shell} large />
                <MiniStat label="Connection" value={connectionState} large />
                <MiniStat label="Voice available" value={bootstrap.voice.available ? "yes" : "no"} large />
                <MiniStat label="Legacy fallback" value={bootstrap.ui.legacy_fallback_available ? "ready" : "absent"} large />
                <div className="mini-grid">
                  <MiniStat label="Beliefs trusted" value={String(epistemics.trusted || 0)} />
                  <MiniStat label="Beliefs contested" value={String(epistemics.contested || 0)} />
                  <MiniStat label="Thermal guard" value={bootstrap.state.thermal_guard ? "active" : "clear"} />
                  <MiniStat label="Fragmentation" value={formatScore(bootstrap.state.fragmentation_score || 0)} />
                </div>
                <div className="feed-list compact">
                  <div className="feed-card">
                    <div className="feed-meta">
                      <span>Constitution</span>
                      <span>{bootstrap.constitutional.recent_decisions?.length || 0} decisions</span>
                    </div>
                    <div className="feed-content">Executive reason: {bootstrap.executive.last_reason || "steady"}</div>
                  </div>
                  <div className="feed-card">
                    <div className="feed-meta">
                      <span>Boot status</span>
                      <span>{bootstrap.telemetry.boot?.status || "unknown"}</span>
                    </div>
                    <div className="feed-content">Initialized: {String(bootstrap.session.initialized)}</div>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </aside>
      </main>

      <section className={`diagnostics panel ${diagnosticsOpen ? "open" : "closed"}`}>
        <div className="panel-header compact">
          <div>
            <div className="eyebrow">Operational Truth</div>
            <h2>Diagnostics</h2>
          </div>
          <div className="diagnostic-summary">
            <span>{diagnostics.length} events</span>
            <span>{toolEvents.length} tool events</span>
          </div>
        </div>
        <div className="diagnostic-grid">
          <div className="diagnostic-column">
            <h3>Subsystem Events</h3>
            <div className="feed-list compact">
              {diagnostics.length === 0 ? <EmptyState label="Waiting for subsystem events." /> : null}
              {diagnostics.map((entry) => (
                <div className={`diag-entry ${entry.level || "info"}`} key={entry.id}>
                  <span>{formatClock(entry.at)}</span>
                  <p>{entry.message}</p>
                </div>
              ))}
            </div>
          </div>
          <div className="diagnostic-column">
            <h3>Tool Lifecycle</h3>
            <div className="feed-list compact">
              {toolEvents.length === 0 ? <EmptyState label="No tool activity yet." /> : null}
              {toolEvents.map((event) => (
                <div className={`diag-entry ${event.success === false ? "error" : event.stage === "rejected" ? "warn" : "info"}`} key={event.id}>
                  <span>{event.stage} · {formatClock(event.at)}</span>
                  <p>{event.tool}{event.reason ? ` | ${event.reason}` : ""}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

function EmptyState({ label }) {
  return <div className="empty-state">{label}</div>;
}

function MiniStat({ label, value, large = false }) {
  return (
    <div className={`mini-stat ${large ? "large" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MetricBar({ label, value, max = 100 }) {
  const normalized = Math.max(0, Math.min(100, (Number(value || 0) / max) * 100));
  return (
    <div className="metric-bar">
      <div className="metric-label">
        <span>{label}</span>
        <strong>{formatPercent(normalized)}</strong>
      </div>
      <div className="metric-track">
        <div className="metric-fill" style={{ width: `${normalized}%` }} />
      </div>
    </div>
  );
}
