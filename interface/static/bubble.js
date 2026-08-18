/*
 * The bubble's behaviour. Deliberately small.
 *
 * It owns three things and nothing else: poll her state, render the two
 * states, and forward three intents (open, clear, moved). Every decision
 * about WHETHER she speaks lives in core/perception/ambient_presence.py and
 * ambient_utterance.py, behind the Will — a surface that could decide to
 * show a message would be a second, quieter authority for unprompted speech,
 * and there is exactly one.
 */
(() => {
  "use strict";

  const pill = document.getElementById("pill");
  const glyph = document.getElementById("glyph");
  const say = document.getElementById("say");
  const close = document.getElementById("close");

  const IDLE_POLL_MS = 4000;
  const ACTIVE_POLL_MS = 1500;
  // A failing backend must not become a spin. Back off, cap, recover.
  const MAX_BACKOFF_MS = 60000;
  /*
   * How long a sentence she offered stays spelled out before it withdraws to
   * the dot. A remark nobody acknowledged must not sit over someone's work
   * all afternoon — but it must not disappear without a trace either, or the
   * only way to learn she had said something is to have been looking.
   *
   * Withdrawing is a PRESENTATION decision and lives here. Whether she has
   * anything to say is the server's, and stays there.
   */
  const WITHDRAW_AFTER_S = 45;

  let backoffMs = 0;
  let lastRendered = "";
  let timer = null;

  function hostBridge() {
    return window.webkit?.messageHandlers?.auraBubble || null;
  }

  async function api(path, options) {
    const response = await fetch(path, {
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!response.ok) throw new Error(`${path} -> ${response.status}`);
    return response.json();
  }

  function render(state) {
    const text = String((state && state.utterance) || "");
    const holding = Boolean(state && state.has_utterance) && text.length > 0;
    // She still has it; she has just stopped holding it open. The server
    // remains the authority on WHETHER there is something — this only decides
    // how long it stays spelled out.
    const age = Number((state && state.utterance_age_s) || 0);
    const withdrawn = holding && age >= WITHDRAW_AFTER_S;
    const speaking = holding && !withdrawn;

    // What the companion window is doing behind her. It keeps running while
    // ordered out, so a message sent there and then collapsed is answered
    // invisibly — and the bubble is the only thing left on screen to say so.
    const working = Boolean(state && state.companion_working);
    const replyWaiting = Boolean(state && state.companion_reply_waiting);

    // Only touch the DOM when something changed. The bubble sits over other
    // windows; a repaint every poll is a flicker in the corner of someone's
    // eye all day.
    const signature = `${speaking ? `1:${text}` : withdrawn ? "2" : "0"}|${
      working ? "w" : ""
    }${replyWaiting ? "r" : ""}`;
    if (signature === lastRendered) return holding;
    lastRendered = signature;

    pill.classList.toggle("speaking", speaking);
    pill.classList.toggle("dormant", !speaking);
    // Unread outlives the text: the dot is what remains of a sentence nobody
    // acknowledged, and it is the whole of "open me when you can".
    // A reply answered into a collapsed window is the same situation by a
    // different route, and gets the same dot.
    pill.classList.toggle("unread", withdrawn || replyWaiting);
    // Working is not unread: nothing is waiting yet. It is the only honest
    // thing to show between pressing send and the answer existing.
    pill.classList.toggle("working", working && !replyWaiting);

    if (speaking) {
      say.textContent = text;
      pill.title = text;
    } else {
      say.textContent = "";
      // The withdrawn message stays reachable on hover rather than being
      // lost — the dot says there is something, and this says what.
      if (withdrawn) pill.title = text;
      else if (replyWaiting) pill.title = "She answered — open to read it";
      else if (working) pill.title = "Working on your message";
      else pill.removeAttribute("title");
    }
    // Keep polling at the active cadence while unread, so acknowledging her
    // anywhere else clears the dot promptly rather than up to four seconds
    // later.
    return holding;
  }

  /*
   * Hand a queued rectangle to the host, which owns the AppKit surface a web
   * page cannot reach. The server popped it for us, so it is drawn once or
   * not at all; if there is no host bridge we are in a browser tab, where
   * there is nothing to draw over and dropping it is correct.
   */
  function forwardHighlight(highlight) {
    if (!highlight) return;
    const bridge = hostBridge();
    if (!bridge) return;
    bridge.postMessage({
      action: "highlight",
      rect: {
        x: highlight.x,
        y: highlight.y,
        width: highlight.width,
        height: highlight.height,
      },
      seconds: highlight.seconds,
    });
  }

  function forwardMove(command) {
    const bridge = hostBridge();
    if (!bridge || !command) return;
    bridge.postMessage({
      action: "move",
      x: command.x,
      y: command.y,
      sequence: command.sequence,
    });
  }

  function reportMeasuredSize() {
    const bridge = hostBridge();
    if (!bridge) return;
    const rect = pill.getBoundingClientRect();
    bridge.postMessage({
      action: "resize",
      width: Math.ceil(rect.width + 16),
      height: Math.ceil(rect.height + 16),
    });
  }

  async function poll() {
    timer = null;
    let speaking = false;
    try {
      // native-bubble marks a WebKit host that can actually draw, and is what
      // lets her refuse to claim she pointed at something when no bubble is
      // listening. It is also what collects the rectangle.
      const surface = hostBridge() ? "native-bubble" : "browser-preview";
      const state = await api(`/api/ambient/state?surface=${surface}`);
      if (state && state.available === false) {
        throw new Error("ambient state unavailable");
      }
      speaking = render(state);
      forwardHighlight(state && state.highlight);
      forwardMove(state && state.bubble_move);
      backoffMs = 0;
    } catch (error) {
      // Stay on screen showing whatever she last said rather than blanking:
      // a message vanishing because the poll failed looks like she withdrew
      // it. Slow down instead.
      backoffMs = Math.min(MAX_BACKOFF_MS, (backoffMs || IDLE_POLL_MS) * 2);
    }
    const next = backoffMs || (speaking ? ACTIVE_POLL_MS : IDLE_POLL_MS);
    timer = window.setTimeout(poll, next);
  }


  if (window.ResizeObserver) {
    const observer = new ResizeObserver(reportMeasuredSize);
    observer.observe(pill);
  } else {
    window.addEventListener("load", reportMeasuredSize, { once: true });
  }

  function openChat() {
    // The bubble does not host a conversation. Clicking asks the host to
    // bring up the restrained chat window; if there is no host bridge we are
    // in a browser tab and the full surface is one navigation away.
    if (window.webkit?.messageHandlers?.auraBubble) {
      window.webkit.messageHandlers.auraBubble.postMessage({ action: "open" });
      return;
    }
    window.location.href = "/";
  }

  glyph.addEventListener("click", (event) => {
    event.preventDefault();
    // Still the click path under the host: the drag recognizer does not delay
    // the primary mouse button, and a pan only begins once the pointer moves.
    // A tap therefore stays a tap.
    openChat();
  });
  say.addEventListener("click", openChat);

  close.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    // Optimistic: clearing must feel instant, and the server is the
    // authority on what comes NEXT, not on what was already dismissed.
    render({ has_utterance: false, utterance: "" });
    try {
      await api("/api/ambient/clear", { method: "POST", body: "{}" });
    } catch (error) {
      /* the next poll reconciles */
    }
  });

  /*
   * Right-click hides her, and hiding is not the same act as clearing.
   *
   * × dismisses a MESSAGE. This dismisses HER: she stops observing, not just
   * being drawn. There was no way to do it at all — the launcher had a hide
   * handler and nothing ever sent it — so the stronger of the two controls
   * was the one with no way to reach it.
   *
   * The context menu carries it rather than a second visible button: this
   * sits over other people's windows all day, and the one thing it must not
   * grow is chrome explaining itself.
   */
  async function hideHer() {
    if (window.webkit?.messageHandlers?.auraBubble) {
      // The host owns the panel AND tells the runtime; one authority for a
      // control that must not end up cosmetic on one path and real on the
      // other.
      window.webkit.messageHandlers.auraBubble.postMessage({ action: "hide" });
      return;
    }
    try {
      await api("/api/ambient/visibility", {
        method: "POST",
        body: JSON.stringify({ mode: "hidden" }),
      });
      render({ has_utterance: false, utterance: "" });
    } catch (error) {
      /* nothing to do; she stays visible, which is the safe direction */
    }
  }

  pill.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    hideHer();
  });

  /*
   * Dragging her somewhere else is NOT handled here.
   *
   * Three attempts lived in this file and none of them could have worked. The
   * stylesheet asked for `-webkit-app-region: drag`, an Electron property that
   * WKWebView does not implement. Replacing it with mousedown plus mousemove
   * deltas posted as {action:"move"} failed for a structural reason: this view
   * is 56x56, WebKit only synthesises mousemove for points inside it, and the
   * gesture therefore died about 28px in — "i still cant drag across the
   * screen". Excluding the glyph from the handle left only the few pixels the
   * ×/reply controls did not claim, which is the "upper right quadrant" that
   * was the last report.
   *
   * A window drag belongs to the window server, not to a page inside the
   * window. AuraLauncher installs TopStripPanGestureRecognizer on this web
   * view with a strip height of zero, meaning the whole surface drags, in
   * global screen coordinates, with no bound to this view's size and no
   * message per pixel. It is the same recognizer the companion window uses.
   *
   * Clicks are unaffected: the recognizer does not delay the primary mouse
   * button and a pan only begins once the pointer moves, so × and the reply
   * control keep taking plain clicks and a tap still opens the chat.
   */

  /*
   * Hand the gesture to AppKit. The page does not drag the window.
   *
   * Three earlier attempts lived here and none could work: `-webkit-app-region:
   * drag` is an Electron property WKWebView ignores, and mousedown + mousemove
   * deltas die about 28px in because this view is 56x56 and WebKit only
   * synthesises mousemove for points inside it.
   *
   * The host runs a modal tracking loop instead, in global screen coordinates,
   * so the whole button drags and the pointer can leave the view. One message,
   * not one per pixel.
   *
   * Every part of the pill is a handle EXCEPT the two controls, which do one
   * thing each and must keep doing it. Clicks survive because the host only
   * treats the gesture as a drag once the pointer actually moves; a tap comes
   * back as `aura-bubble-click`.
   */
  pill.addEventListener("mousedown", (event) => {
    if (event.button !== 0) return;
    if (event.target.closest("#close, #say")) return;
    postToHost({ action: "dragStart" });
  });

  window.addEventListener("aura-bubble-click", openChat);

  function postToHost(payload) {
    const bridge = window.webkit?.messageHandlers?.auraBubble;
    if (!bridge) return false;
    bridge.postMessage(payload);
    return true;
  }

  // Report where the person parked her, so the position survives a restart.
  let moveTimer = null;
  window.addEventListener("aura-bubble-moved", (event) => {
    const detail = event.detail || {};
    window.clearTimeout(moveTimer);
    moveTimer = window.setTimeout(() => {
      api("/api/ambient/position", {
        method: "POST",
        body: JSON.stringify({
          x: Number.isFinite(detail.x) ? detail.x : 0,
          y: Number.isFinite(detail.y) ? detail.y : 0,
          sequence: Number.isInteger(detail.sequence) ? detail.sequence : 0,
        }),
      }).catch(() => {});
    }, 400);
  });

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && timer === null) poll();
  });

  poll();
})();
