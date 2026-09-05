/*
 * The restrained chat window.
 *
 * It talks to the SAME /api/chat as the full desktop. Not a lighter model,
 * not a shorter context, not a companion-mode personality — a second, quieter
 * Aura reachable from the bubble would be a different assistant wearing her
 * icon, and the person clicking the bubble is asking the same one a question.
 * What is restrained here is the SURFACE, not her.
 */
(() => {
  "use strict";

  const log = document.getElementById("log");
  const input = document.getElementById("input");
  const send = document.getElementById("send");
  const thinking = document.getElementById("thinking");
  const thinkingLabel = document.getElementById("thinking-label");
  const expand = document.getElementById("expand");

  // A request wait is bounded so a broken socket cannot pin the surface. The
  // admitted TURN is not deadline-bounded: its durable status is followed
  // until it reaches an authoritative terminal receipt.
  const REQUEST_TIMEOUT_MS = 30000;
  const DELIVERY_POLL_MS = 750;
  const PENDING_KEY = "aura-companion-pending-v1";
  let inFlight = false;
  let lastProgressSequence = 0;

  function idempotencyKey() {
    const id = window.crypto?.randomUUID?.()
      || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 14)}`;
    return `aura-companion-${id}`;
  }

  function chatHeaders(key = "") {
    const headers = {
      "Content-Type": "application/json",
      "X-Aura-Surface": "desktop-ui",
      "X-Aura-Desktop-Request": "same-origin",
      "X-Aura-Require-CognitiveEngine": "true",
    };
    if (key) headers["X-Idempotency-Key"] = key;
    return headers;
  }

  function replyEnvelope(payload) {
    if (payload?.result && typeof payload.result === "object") return payload.result;
    return payload && typeof payload === "object" ? payload : {};
  }

  function replyText(payload) {
    const data = replyEnvelope(payload);
    return String(data.response ?? data.reply ?? data.message ?? "").trim();
  }

  function storePending(item) {
    try { localStorage.setItem(PENDING_KEY, JSON.stringify(item)); } catch (_error) {}
  }

  function clearPending(key) {
    try {
      const saved = JSON.parse(localStorage.getItem(PENDING_KEY) || "null");
      if (!saved || saved.key === key) localStorage.removeItem(PENDING_KEY);
    } catch (_error) {
      try { localStorage.removeItem(PENDING_KEY); } catch (_ignored) {}
    }
  }

  async function jsonFetch(path, options, timeoutMs) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(path, {
        cache: "no-store",
        credentials: "same-origin",
        ...options,
        signal: controller.signal,
      });
      const body = await response.text();
      let payload = {};
      try { payload = body ? JSON.parse(body) : {}; } catch (_error) {
        payload = { status: "invalid_json_response", detail: body.slice(0, 240) };
      }
      return { response, payload };
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async function deliveryStatus(key) {
    return jsonFetch(
      `/api/chat/delivery/${encodeURIComponent(key)}`,
      { headers: chatHeaders() },
      10000,
    );
  }

  function updateProgress(payload, fallback = "Working on your request.") {
    const progress = payload?.progress;
    const sequence = Number(progress?.sequence) || 0;
    if (sequence && sequence < lastProgressSequence) return;
    if (sequence) lastProgressSequence = sequence;
    const message = String(progress?.message || fallback || "").trim();
    if (message) thinkingLabel.textContent = message;
  }

  async function postChat(item) {
    const { response, payload } = await jsonFetch(
      "/api/chat",
      {
        method: "POST",
        headers: chatHeaders(item.key),
        body: JSON.stringify({ message: item.message }),
      },
      REQUEST_TIMEOUT_MS,
    );
    updateProgress(payload);
    if (response.ok && response.status !== 202) return { terminal: true, payload };
    if (response.status === 202) return { terminal: false, payload };
    const detail = replyText(payload) || payload.detail || payload.status || `chat ${response.status}`;
    const error = new Error(detail);
    error.fatal = [400, 401, 403, 409, 422].includes(response.status);
    throw error;
  }

  // The admitted POST may spend a long time in prompt prefill or verified
  // execution before it can return a JSON body. Observe the same durable
  // delivery journal concurrently so this surface shows real phase progress
  // instead of appearing frozen until the POST timeout.
  async function observeDeliveryProgress(item, stop) {
    let delay = DELIVERY_POLL_MS;
    while (!stop()) {
      await new Promise((resolve) => window.setTimeout(resolve, delay));
      if (stop()) return;
      try {
        const { payload } = await deliveryStatus(item.key);
        if (stop()) return;
        updateProgress(payload);
        const state = String(payload?.state || payload?.delivery_state || "").toLowerCase();
        if (
          payload?.terminal === true
          || payload?.delivery_status === "terminal"
          || ["awaiting_approval", "completed", "failed", "ambiguous"].includes(state)
        ) return;
      } catch (_error) {
        // The durable send/replay loop owns transport recovery. This observer
        // is presentation-only and must never cancel or duplicate the turn.
      }
      delay = Math.min(5000, Math.round(delay * 1.5));
    }
  }

  async function awaitDelivery(item) {
    let missingPolls = 0;
    let transportFailures = 0;
    while (true) {
      try {
        const { response, payload } = await deliveryStatus(item.key);
        if (response.ok && (payload.terminal || payload.delivery_status === "terminal")) {
          return payload;
        }
        updateProgress(
          payload,
          response.status === 503
            ? "The delivery journal is recovering; the admitted work is still protected."
            : "The request is admitted and still running.",
        );
        if (response.status === 404) {
          missingPolls += 1;
          // The initial POST may have died before admission. Reusing the exact
          // key is safe: the journal either admits it once or returns its owner.
          if (missingPolls >= 3) {
            const retried = await postChat(item);
            if (retried.terminal) return retried.payload;
            missingPolls = 0;
          }
        } else if (response.status === 401 || response.status === 403) {
          throw Object.assign(new Error(`delivery ${response.status}`), { fatal: true });
        } else if (response.status !== 202 && response.status !== 503) {
          throw Object.assign(new Error(`delivery ${response.status}`), { fatal: true });
        }
        transportFailures = 0;
        const retry = Math.max(250, Number(payload.retry_after_ms) || DELIVERY_POLL_MS);
        await new Promise((resolve) => window.setTimeout(resolve, retry));
      } catch (error) {
        if (error?.fatal) throw error;
        transportFailures += 1;
        updateProgress(null, navigator.onLine === false
          ? "Connection paused. The turn will resume here when the computer is online."
          : "Connection interrupted. Reconnecting to the durable turn.");
        const retry = Math.min(10000, 500 * Math.pow(2, Math.min(5, transportFailures)));
        await new Promise((resolve) => window.setTimeout(resolve, retry));
      }
    }
  }

  async function sendDurably(item) {
    let observing = true;
    observeDeliveryProgress(item, () => !observing).catch(() => {});
    try {
      const posted = await postChat(item);
      if (posted.terminal) return posted.payload;
    } catch (error) {
      if (error?.fatal) throw error;
      // A timed-out or interrupted POST says nothing about the admitted turn.
      // The status loop safely reuses the key if admission never happened.
    } finally {
      observing = false;
    }
    return awaitDelivery(item);
  }

  function bubble(text, kind) {
    const node = document.createElement("div");
    node.className = `msg ${kind}`;
    node.textContent = text;
    log.appendChild(node);
    log.scrollTop = log.scrollHeight;
    return node;
  }

  /*
   * This window keeps running when the host orders it out, so a turn started
   * here and then collapsed is answered into a window nobody is looking at.
   * The bubble it collapsed into is the only thing on screen, and it showed
   * neither state — reported live 2026-08-10: "no typing indicator, no
   * indicator when a message has arrived or is waiting".
   *
   * The host is the authority on whether this window is visible; a WKWebView
   * that has merely been ordered out is not "hidden" by any measure the page
   * can take for itself.
   */
  let windowVisible = true;
  let turnHeartbeat = null;

  window.addEventListener("aura-companion-visibility", (event) => {
    windowVisible = Boolean(event.detail && event.detail.visible);
    // Opening the window IS reading the reply.
    if (windowVisible) reportTurn("read");
  });

  function reportTurn(state) {
    return fetch("/api/ambient/companion-turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state }),
    }).catch(() => {
      /* the bubble simply shows nothing; it must not break the turn */
    });
  }

  function busy(state) {
    inFlight = state;
    send.disabled = state;
    thinking.classList.toggle("on", state);
    if (state) updateProgress(null, "Working on your request.");
    else lastProgressSequence = 0;
    if (state) log.scrollTop = log.scrollHeight;

    window.clearInterval(turnHeartbeat);
    if (state) {
      reportTurn("working");
      // Renewed, so the signal expires on its own if this window goes away
      // mid-turn rather than leaving the bubble working forever.
      turnHeartbeat = window.setInterval(() => reportTurn("working"), 8000);
    } else if (!windowVisible) {
      // The answer landed somewhere the person cannot see it.
      reportTurn("reply_waiting");
    } else {
      reportTurn("idle");
    }
  }

  function autoGrow() {
    input.style.height = "auto";
    input.style.height = `${Math.min(120, input.scrollHeight)}px`;
  }

  async function submit() {
    const text = input.value.trim();
    if (!text || inFlight) return;
    bubble(text, "me");
    input.value = "";
    autoGrow();
    busy(true);

    const item = { message: text, key: idempotencyKey(), queuedAt: Date.now() };
    storePending(item);
    try {
      const data = await sendDurably(item);
      const reply = replyText(data);
      // An empty reply is reported as one. Rendering nothing would leave the
      // window looking like the message was never sent, and the person would
      // send it again.
      bubble(reply || "Aura completed the turn without a deliverable reply.", reply ? "her" : "err");
      clearPending(item.key);
    } catch (error) {
      // The failure is shown in the transcript rather than swallowed: a
      // message that vanishes is indistinguishable from one she ignored.
      bubble(`Could not reach her: ${error.message}`, "err");
    } finally {
      busy(false);
      input.focus();
    }
  }

  send.addEventListener("click", submit);

  input.addEventListener("keydown", (event) => {
    // Enter sends, Shift+Enter is a newline. This window is for one thing
    // said quickly; a send button you must aim at defeats that.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
    if (event.key === "Escape") {
      window.webkit?.messageHandlers?.auraCompanion?.postMessage({ action: "close" });
    }
  });
  input.addEventListener("input", autoGrow);

  // Clicking the mark you clicked to get here takes you back. "close" is the
  // host's word for it and already restores the bubble; there was simply no
  // surface sending it except the Escape key.
  document.getElementById("collapse")?.addEventListener("click", () => {
    if (window.webkit?.messageHandlers?.auraCompanion) {
      window.webkit.messageHandlers.auraCompanion.postMessage({ action: "close" });
    }
  });

  expand.addEventListener("click", () => {
    if (window.webkit?.messageHandlers?.auraCompanion) {
      window.webkit.messageHandlers.auraCompanion.postMessage({ action: "expand" });
      return;
    }
    window.location.href = "/";
  });

  // If she had something queued in the bubble, it is the reason this window
  // was opened. Show it as her first line and clear it, so the thing that
  // prompted the click is not lost behind an empty transcript.
  (async () => {
    try {
      const state = await fetch("/api/ambient/state", { cache: "no-store" }).then((r) =>
        r.json()
      );
      if (state && state.has_utterance && state.utterance) {
        bubble(String(state.utterance), "her");
        await fetch("/api/ambient/clear", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        }).catch(() => {});
      }
    } catch (error) {
      /* an empty transcript is a fine starting state */
    }
    try {
      const pending = JSON.parse(localStorage.getItem(PENDING_KEY) || "null");
      if (pending?.message && pending?.key) {
        busy(true);
        const recovered = await sendDurably(pending);
        const reply = replyText(recovered);
        bubble(reply || "Aura completed the recovered turn without a deliverable reply.", reply ? "her" : "err");
        clearPending(pending.key);
      }
    } catch (error) {
      bubble(`The earlier turn is not settled yet: ${error.message}`, "err");
    } finally {
      busy(false);
      input.focus();
    }
  })();
})();
