/**
 * error_banner.js — universal error UX renderer.
 *
 * The runtime never sends Python tracebacks to the frontend. Every
 * non-200 from the API (or 200 with status="phenomenal") includes an
 * ErrorEnvelope shaped as:
 *
 *   {
 *     envelope_id, phenomenal_state, user_message, technical_summary,
 *     suggested_action, recovery_buttons:[{label, action_id}], severity,
 *     correlation_id, diagnostic_link
 *   }
 *
 * This module renders the envelope at the top of the viewport with the
 * standard [Retry] [Use fallback] [Open diagnostics] buttons. It binds
 * the action_ids to the conversation lane:
 *
 *   - retry        → re-emits the last user message to /api/chat
 *   - fallback     → calls /api/chat with prefer_tier=tertiary
 *   - diagnostics  → opens the Aura DevTools time-scrubber for the
 *                    correlation_id (or /api/dashboard/snapshot if
 *                    DevTools isn't installed)
 *
 * The banner is global; only one is shown at a time. New envelopes
 * replace the previous banner.
 */

(function () {
  if (window.__auraErrorBannerInstalled) return;
  window.__auraErrorBannerInstalled = true;

  let lastUserMessage = "";
  function setLastUserMessage(msg) { lastUserMessage = String(msg || ""); }
  window.auraSetLastUserMessage = setLastUserMessage;

  function ensureCss() {
    if (document.getElementById("aura-error-banner-css")) return;
    const link = document.createElement("link");
    link.id = "aura-error-banner-css";
    link.rel = "stylesheet";
    link.href = "/static/error_banner.css";
    document.head.appendChild(link);
  }

  function renderBanner(envelope) {
    if (!envelope || typeof envelope !== "object") return;
    ensureCss();
    document.querySelectorAll(".aura-error-banner").forEach(el => el.remove());

    const root = document.createElement("div");
    root.className = "aura-error-banner";
    root.setAttribute("role", "alert");
    root.setAttribute("aria-live", "assertive");

    const body = document.createElement("div");
    body.className = "aura-error-banner__body";
    const msg = document.createElement("div");
    msg.className = "aura-error-banner__msg";
    msg.textContent = envelope.user_message || "Something didn't go as expected.";
    const sub = document.createElement("div");
    sub.className = "aura-error-banner__sub";
    sub.textContent = `state: ${envelope.phenomenal_state || "unknown"}` + (envelope.correlation_id ? ` · trace ${envelope.correlation_id}` : "");
    body.appendChild(msg);
    body.appendChild(sub);

    const actions = document.createElement("div");
    actions.className = "aura-error-banner__actions";
    const buttons = Array.isArray(envelope.recovery_buttons) && envelope.recovery_buttons.length
      ? envelope.recovery_buttons
      : [
          { label: "Retry", action_id: "retry" },
          { label: "Use fallback", action_id: "fallback" },
          { label: "Open diagnostics", action_id: "diagnostics" },
        ];
    buttons.forEach((b, idx) => {
      const btn = document.createElement("button");
      btn.className = "aura-error-banner__btn" + (idx === 0 ? " aura-error-banner__btn--primary" : "");
      btn.textContent = b.label;
      btn.addEventListener("click", () => handleAction(b.action_id, envelope));
      actions.appendChild(btn);
    });
    root.appendChild(body);
    root.appendChild(actions);
    document.body.appendChild(root);
  }

  async function handleAction(actionId, envelope) {
    document.querySelectorAll(".aura-error-banner").forEach(el => el.remove());
    if (actionId === "retry" && lastUserMessage) {
      try {
        const resp = await fetch("/api/chat", { method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ message: lastUserMessage }) });
        const data = await resp.json();
        if (data && (data.status === "phenomenal" || data.envelope)) {
          renderBanner(data.envelope || data);
        }
      } catch (e) { console.warn("aura retry failed:", e); }
    } else if (actionId === "fallback" && lastUserMessage) {
      try {
        const resp = await fetch("/api/chat", { method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ message: lastUserMessage, prefer_tier: "tertiary" }) });
        await resp.json();
      } catch (e) { console.warn("aura fallback failed:", e); }
    } else if (actionId === "diagnostics") {
      const link = envelope.diagnostic_link || ("/api/trace/" + (envelope.correlation_id || ""));
      window.open(link, "_blank", "noopener");
    }
  }

  // Hook into fetch so any envelope-bearing response auto-renders.
  const origFetch = window.fetch.bind(window);
  window.fetch = async function (input, init) {
    const resp = await origFetch(input, init);
    try {
      const ct = resp.headers.get("content-type") || "";
      if (ct.includes("application/json")) {
        const cloned = resp.clone();
        cloned.json().then(payload => {
          if (payload && payload.envelope && payload.envelope.envelope_id) {
            renderBanner(payload.envelope);
          } else if (payload && payload.status === "phenomenal" && payload.user_message) {
            renderBanner(payload);
          }
        }).catch(() => {});
      }
    } catch (e) { /* swallow — never break the page */ }
    return resp;
  };

  window.auraRenderErrorBanner = renderBanner;
})();
