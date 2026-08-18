/* ══════════════════════════════════════════════════════════
   AURA SOVEREIGN — Frontend Logic (Magnum Opus)
   ══════════════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);
const state = {
    ws: null,
    activeTab: 'neural',
    activeMem: 'episodic',
    connected: false,
    voiceActive: false,
    beliefGraphInit: false,
    cycleCount: 0,
    startTime: Date.now(),
    thoughtQueue: [],
    thoughtDrainTimer: null,
    neuralFeedReadable: false,
    neuralFeedMode: 'live',
    neuralFeedPaused: false,
    pendingOutboundMessages: [], // ZENITH: Message queueing during disconnect
    processedMessageFingerprints: new Set(), // ZENITH: Chat deduplication
    pacingActive: false,
    currentMood: 'neutral',
    singularityActive: false,
    lastUserMessage: null,
    lastTelemetryFingerprint: null,
    userScrolledUp: false,
    healthPollInFlight: false,
    healthPollTimer: null,
    healthPollFailures: 0,
    healthPollIncident: null,
    healthLastSuccessAt: 0,
    processedEventIds: new Set(),
    toolCatalog: [],
    uiFlags: [],
    lastToolEvent: null,
    isSubmitting: false,
    activeChatRequestId: null,
    activeChatRequest: null,
    chatSendQueue: [],
    chatDrainTimer: null,
    chatHandoffPending: false,
    chatHandoffRestored: false,
    deferredShellReload: null,
    waitingServiceWorker: null,
    surfaceSuspended: false,
    resumeInProgress: false,
    lastSurfaceHiddenAt: 0,
    lastSurfaceResumeAt: 0,
    lastChatLatencyMs: null,
    commitments: null,
    voiceSummary: null,
    desktopAccess: null,
    desktopAccessPollInFlight: false,
    desktopAccessTimer: null,
    bootstrapLoaded: false,
    bootstrapTimer: null,
    knowledgeGraphTimer: null,
    knowledgeGraphPollInFlight: false,
    surfaceWorkloadMode: null,
    lastNeuralPulseAt: 0,
    lastSemanticThoughtAt: 0,
    lastHealthSnapshotFingerprint: null,
    lastHealthWarningPulseAt: 0,
    conversationReady: false,
    conversationLane: null,
    runtimeHealthy: false,
    runtimeHealthBlockers: [],
    runtimeRevision: null,
    runtimeRevisionGeneration: 0,
    runtimeRevisionCapturedAtUnix: 0,
    runtimeRevisionReloadAttempts: {},
    runtimeRevisionReloading: false,
    runtimeRevisionTrust: 'unknown',
    runtimeShellRetirementPromise: null,
    serviceWorkerRevision: null,
    serviceWorkerRegistrationTarget: null,
    serviceWorkerRegistrationPromise: null,
    serviceWorkerRegistrationEpoch: 0,
    serviceWorkerRegistrationFailures: 0,
    serviceWorkerRegistrationRetryAt: 0,
    serviceWorkerInstallers: new WeakMap(),
    version: 'Aura Luna (live runtime)',
    interactionSignals: null,
    typingSignalSession: null,
    typingSignalTimer: null,
    voiceSignalAggregation: null,
    voiceSignalTimer: null,
    cameraSignalActive: false,
    cameraSignalWanted: false,
    cameraSignalInterval: null,
    cameraSignalCapture: null,
    lastSystemRamPct: null,
    accessResolved: false,
    accessProfile: null,
    conversationOnly: true,
    profileFeaturesStarted: new Set()
};
console.log(`%c AURA %c ${state.version} `, "color:white; background:#8a2be2; padding:2px 5px; border-radius:3px 0 0 3px;", "color:white; background:#1e1535; padding:2px 5px; border-radius:0 3px 3px 0;");

const CHAT_REQUEST_TIMEOUT_READY_MS = 335000;
const CHAT_REQUEST_TIMEOUT_RECOVERING_MS = 395000;
const CHAT_SEND_QUEUE_MAX = 32;
const CHAT_HANDOFF_ACTIVE_REPLAY_MAX_WAIT_MS = CHAT_REQUEST_TIMEOUT_RECOVERING_MS + 15000;
const CHAT_HANDOFF_MAX_AGE_MS = 10 * 60 * 1000;
const CHAT_DELIVERY_STATUS_TIMEOUT_MS = 8000;
const CHAT_DELIVERY_POLL_BASE_MS = 400;
const CHAT_DELIVERY_POLL_MAX_MS = 5000;
// How long a turn keeps retrying a runtime it cannot reach at all.
//
// The delivery loop is deliberately patient: a turn survives a transport
// blip and resumes, which is why it retries rather than failing on the first
// error. It had no ceiling, so when the runtime went away mid-turn the UI
// showed "Aura is reconciling the current turn…" indefinitely — a dead
// backend and a slow thought looked exactly alike. Observed 2026-07-27:
// seventeen minutes of spinner over a process that had already exited.
//
// Only consecutive TRANSPORT failures count against this. A server that
// answers "still working" is contact, and resets the clock — a long
// reasoning turn is never cut off by it.
const CHAT_DELIVERY_UNREACHABLE_MS = 180000;
const CHAT_DELIVERY_TERMINAL_STATES = new Set([
    'awaiting_approval',
    'completed',
    'failed',
    'ambiguous',
]);
const THOUGHT_QUEUE_MAX = 160;
const THOUGHT_COALESCE_WINDOW_MS = 12000;
const THOUGHT_COALESCE_LOOKBACK = 18;
const PROCESSED_EVENT_ID_MAX = 2000;
const PROCESSED_MESSAGE_FINGERPRINT_MAX = 500;
const NEURAL_LIVENESS_PULSE_MS = 30000;
const HEALTH_POLL_BASE_MS = 10000;
const HEALTH_POLL_RETRY_BASE_MS = 2500;
const HEALTH_POLL_MAX_MS = 60000;
const HEALTH_POLL_TIMEOUT_MS = 6000;
const HEALTH_POLL_REMINDER_MS = 5 * 60 * 1000;
const HEALTH_POLL_JITTER_RATIO = 0.15;
const DESKTOP_ACCESS_POLL_MS = 15000;
const BOOTSTRAP_POLL_MS = 30000;
const KNOWLEDGE_GRAPH_POLL_MS = 10000;
const RUNTIME_REVISION_STORAGE_KEY = 'aura.runtime_revision';
const RUNTIME_REVISION_RELOAD_STORAGE_KEY = 'aura.runtime_revision_reload';
const RUNTIME_REVISION_RECORD_SCHEMA = 'aura.runtime_revision.client.v2';
const RUNTIME_REVISION_RELOAD_LIMIT = 2;
const CHAT_HANDOFF_STORAGE_KEY = 'aura.chat_handoff';
const CHAT_HANDOFF_SCHEMA = 'aura.chat_handoff.v3';
const CHAT_HANDOFF_ACCEPTED_SCHEMAS = new Set([CHAT_HANDOFF_SCHEMA]);
const SERVICE_WORKER_REGISTRATION_RETRY_MAX_MS = 30000;
const TYPING_SIGNAL_DEBOUNCE_MS = 850;
const VOICE_SIGNAL_FLUSH_MS = 900;
const CAMERA_SIGNAL_INTERVAL_MS = 2200;
const surfaceTransportFetch = window.fetch.bind(window);
const CONVERSATION_SURFACE_GET_PATHS = new Set([
    '/',
    '/api/health',
    '/api/health/live',
    '/api/health/ready',
    '/api/sessions',
    '/api/ui/bootstrap',
]);

function accessCapabilityAllowed(name) {
    const capabilities = state.accessProfile && state.accessProfile.capabilities;
    return !!(state.accessResolved && capabilities && capabilities[name] === true);
}

function conversationSurfaceRequestAllowed(path, method) {
    if (path === '/api/chat') return method === 'POST';
    if (path.startsWith('/api/chat/delivery/')) return method === 'GET';
    if (CONVERSATION_SURFACE_GET_PATHS.has(path)) return ['GET', 'HEAD'].includes(method);
    if (path === '/api/worlds' || path.startsWith('/api/worlds/')) {
        return ['GET', 'HEAD'].includes(method);
    }
    return false;
}

function surfaceRequestAllowed(input, init = {}) {
    let url;
    try {
        const raw = typeof input === 'string' ? input : input && input.url;
        url = new URL(raw || '', window.location.origin);
    } catch (_err) {
        return false;
    }
    if (url.origin !== window.location.origin || !url.pathname.startsWith('/api/')) {
        return true;
    }
    const requestMethod = String(
        init.method || (input && typeof input === 'object' && input.method) || 'GET'
    ).toUpperCase();
    if (state.accessResolved && !state.conversationOnly) return true;
    return conversationSurfaceRequestAllowed(url.pathname, requestMethod);
}

window.fetch = function auraSurfaceFetch(input, init = {}) {
    if (!surfaceRequestAllowed(input, init)) {
        const path = typeof input === 'string' ? input : (input && input.url) || 'unknown';
        return Promise.reject(new Error(`surface_scope_denied:${path}`));
    }
    return surfaceTransportFetch(input, init);
};

function applyAccessProfile(profile) {
    const previousScope = chatHandoffScope();
    const normalized = profile && typeof profile === 'object' ? profile : {};
    state.accessProfile = normalized;
    state.accessResolved = true;
    state.conversationOnly = normalized.conversation_only !== false;
    const surface = String(normalized.surface || 'unknown');
    window.__auraControlSurfaceAllowed = !state.conversationOnly;
    document.body.dataset.auraSurface = surface;
    const nextScope = chatHandoffScope();
    if (previousScope && nextScope && previousScope !== nextScope) {
        try { sessionStorage.removeItem(CHAT_HANDOFF_STORAGE_KEY); } catch (_err) {}
        state.chatSendQueue = [];
        state.activeChatRequest = null;
    }
    if (!state.chatHandoffRestored && nextScope) {
        state.chatHandoffRestored = true;
        if (restoreChatHandoff($('chat-input'))) {
            window.setTimeout(() => drainQueuedChatMessages(), 0);
        }
    }
    window.dispatchEvent(new CustomEvent('aura:access-profile', { detail: normalized }));
    if (state.conversationOnly) {
        setRuntimeSettingsAvailability(false, 'Runtime settings require the paired desktop control surface.');
    } else {
        hydrateRuntimeSettings().catch(err => {
            console.warn('[Settings] Runtime hydration failed:', err);
        });
    }
}

function nowSeconds() {
    return Date.now() / 1000;
}

function auraDesktopHeaders(extra = {}) {
    return {
        'X-Aura-Surface': 'desktop-ui',
        'X-Aura-Desktop-Request': 'same-origin',
        ...extra,
    };
}

async function postInteractionSignal(path, payload, { quiet = true, keepalive = false } = {}) {
    if (!accessCapabilityAllowed('interaction_signals')) return;
    try {
        await fetch(path, {
            method: 'POST',
            headers: auraDesktopHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify(payload),
            keepalive
        });
    } catch (err) {
        if (!quiet) console.warn(`[signals] ${path} failed`, err);
    }
}

function createTypingSignalSession(seedLength = 0) {
    const now = Date.now();
    return {
        startedAt: now,
        firstKeyAt: 0,
        lastKeyAt: 0,
        keyCount: 0,
        correctionCount: 0,
        maxPauseMs: 0,
        messageChars: seedLength
    };
}

function ensureTypingSignalSession(seedLength = 0) {
    if (!state.typingSignalSession) {
        state.typingSignalSession = createTypingSignalSession(seedLength);
    }
    return state.typingSignalSession;
}

function scheduleTypingSignalFlush() {
    clearTimeout(state.typingSignalTimer);
    state.typingSignalTimer = setTimeout(() => {
        flushTypingSignal({ submitted: false });
    }, TYPING_SIGNAL_DEBOUNCE_MS);
}

function noteTypingSignalKey(event, textarea) {
    if (!textarea) return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const session = ensureTypingSignalSession((textarea.value || '').length);
    const now = Date.now();
    if (!session.firstKeyAt) session.firstKeyAt = now;
    if (session.lastKeyAt) {
        session.maxPauseMs = Math.max(session.maxPauseMs, now - session.lastKeyAt);
    }
    session.lastKeyAt = now;
    if (event.key.length === 1 || event.key === 'Enter' || event.key === 'Backspace' || event.key === 'Delete') {
        session.keyCount += 1;
    }
    if (event.key === 'Backspace' || event.key === 'Delete') {
        session.correctionCount += 1;
    }
    session.messageChars = (textarea.value || '').length;
    scheduleTypingSignalFlush();
}

function noteTypingSignalInput(textarea) {
    if (!textarea) return;
    const value = textarea.value || '';
    if (!value) {
        flushTypingSignal({ submitted: false, forceInactive: true, messageCharsOverride: 0 });
        return;
    }
    const session = ensureTypingSignalSession(value.length);
    const now = Date.now();
    if (!session.firstKeyAt) session.firstKeyAt = now;
    if (!session.lastKeyAt) session.lastKeyAt = now;
    session.messageChars = value.length;
    scheduleTypingSignalFlush();
}

function flushTypingSignal({ submitted = false, forceInactive = false, messageCharsOverride = null } = {}) {
    clearTimeout(state.typingSignalTimer);
    const session = state.typingSignalSession;
    if (!session) return;
    const textarea = $('chat-input');
    const now = Date.now();
    const messageChars = messageCharsOverride != null
        ? messageCharsOverride
        : Math.max(0, textarea ? (textarea.value || '').length : session.messageChars);
    const firstKeyAt = session.firstKeyAt || session.startedAt || now;
    const lastKeyAt = session.lastKeyAt || firstKeyAt;
    const sessionMs = Math.max(1, now - firstKeyAt);
    const pauseBeforeSubmitMs = submitted ? Math.max(0, now - lastKeyAt) : 0;
    const active = !submitted && !forceInactive && messageChars > 0;

    postInteractionSignal('/api/signals/typing', {
        timestamp: nowSeconds(),
        active,
        session_ms: sessionMs,
        key_count: Math.max(session.keyCount, messageChars),
        correction_count: session.correctionCount,
        max_pause_ms: session.maxPauseMs,
        pause_before_submit_ms: pauseBeforeSubmitMs,
        message_chars: messageChars,
        submitted
    }, { quiet: true, keepalive: submitted });

    if (submitted || forceInactive || messageChars === 0) {
        state.typingSignalSession = null;
        return;
    }
    session.messageChars = messageChars;
}

function resetVoiceSignalAggregation() {
    clearTimeout(state.voiceSignalTimer);
    state.voiceSignalTimer = null;
    state.voiceSignalAggregation = {
        startedAt: Date.now(),
        frames: 0,
        samples: 0,
        speechFrames: 0,
        rmsSum: 0,
        rmsSqSum: 0,
        peakSum: 0,
        zcrSum: 0,
        clippingSum: 0
    };
}

function flushVoiceSignal() {
    clearTimeout(state.voiceSignalTimer);
    state.voiceSignalTimer = null;
    const agg = state.voiceSignalAggregation;
    if (!agg || !agg.frames) return;

    const frames = Math.max(1, agg.frames);
    const rmsAvg = agg.rmsSum / frames;
    const rmsVar = Math.max(0, (agg.rmsSqSum / frames) - (rmsAvg * rmsAvg));
    postInteractionSignal('/api/signals/voice', {
        timestamp: nowSeconds(),
        duration_ms: Date.now() - agg.startedAt,
        speech_ratio: agg.speechFrames / frames,
        rms_avg: rmsAvg,
        rms_std: Math.sqrt(rmsVar),
        peak_avg: agg.peakSum / frames,
        zcr_avg: agg.zcrSum / frames,
        clipping_ratio: agg.clippingSum / frames
    }, { quiet: true });
    resetVoiceSignalAggregation();
}

function accumulateVoiceSignal(features) {
    if (!state.voiceActive) return;
    if (!state.voiceSignalAggregation) resetVoiceSignalAggregation();
    const agg = state.voiceSignalAggregation;
    const rms = Number(features && features.rms);
    const peak = Number(features && features.peak);
    const zcr = Number(features && features.zcr);
    const clippingRatio = Number(features && features.clippingRatio);
    if (![rms, peak, zcr, clippingRatio].every(Number.isFinite)) return;

    agg.frames += 1;
    agg.samples += Number(features.sampleCount || 0);
    agg.rmsSum += rms;
    agg.rmsSqSum += rms * rms;
    agg.peakSum += peak;
    agg.zcrSum += zcr;
    agg.clippingSum += clippingRatio;
    if (rms > 0.018 || peak > 0.09) {
        agg.speechFrames += 1;
    }

    if (!state.voiceSignalTimer) {
        state.voiceSignalTimer = setTimeout(() => flushVoiceSignal(), VOICE_SIGNAL_FLUSH_MS);
    }
}

async function startCameraSignals() {
    if (state.cameraSignalActive || !state.cameraSignalWanted) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showBriefNotification('Camera sensing is unavailable in this browser.');
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user',
                width: { ideal: 320 },
                height: { ideal: 240 }
            },
            audio: false
        });
        const video = document.createElement('video');
        video.setAttribute('playsinline', 'true');
        video.muted = true;
        video.srcObject = stream;
        await video.play();

        const canvas = document.createElement('canvas');
        canvas.width = 320;
        canvas.height = 240;
        const ctx = canvas.getContext('2d', { willReadFrequently: false });
        state.cameraSignalCapture = { stream, video, canvas, ctx };
        state.cameraSignalActive = true;
        state.cameraSignalInterval = setInterval(captureCameraSignalFrame, CAMERA_SIGNAL_INTERVAL_MS);
        captureCameraSignalFrame();
    } catch (err) {
        console.error('Camera signal capture failed:', err);
        state.cameraSignalActive = false;
        state.cameraSignalCapture = null;
        showBriefNotification('Camera access was denied or unavailable.');
    }
}

function stopCameraSignals() {
    clearInterval(state.cameraSignalInterval);
    state.cameraSignalInterval = null;
    state.cameraSignalActive = false;
    const capture = state.cameraSignalCapture;
    if (!capture) return;
    try {
        if (capture.video) {
            capture.video.pause();
            capture.video.srcObject = null;
        }
        if (capture.stream) {
            capture.stream.getTracks().forEach(track => track.stop());
        }
    } catch (_err) {
        // Ignore teardown noise.
    }
    state.cameraSignalCapture = null;
}

/**
 * Take a frame because she was asked a question about right now.
 *
 * Deliberately not the presence lane below. That one samples 320×240 every
 * few seconds to know whether somebody is there, which is all presence needs
 * and nowhere near enough to count fingers or read a label — and it is
 * whatever was captured last, not what is in front of the camera at the
 * moment of the question.
 *
 * So this opens its own short-lived capture at a resolution a model can read,
 * takes one frame, and posts it back against the request id the turn is
 * waiting on. If the presence camera is already running its stream is reused,
 * because starting a second one on the same device fails on most platforms
 * and would be slower even where it does not.
 */
async function captureFrameForAura(request) {
    const requestId = request && request.request_id;
    if (!requestId) return;

    // A direct request to look grants one correlated capture without enabling
    // ambient sensing or changing the saved camera switch. The server validates
    // the request id against its pending one-shot lease before accepting the
    // frame, so this client flag is not the authority boundary.
    const oneShotAuthorized = !!request.one_shot_authorized;
    if (!state.cameraSignalWanted && !oneShotAuthorized) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;

    const width = Number(request.width) || 1280;
    const height = Number(request.height) || 720;
    let ownStream = null;
    try {
        let video = state.cameraSignalCapture && state.cameraSignalCapture.video;
        if (!video) {
            ownStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user', width: { ideal: width }, height: { ideal: height } },
                audio: false,
            });
            video = document.createElement('video');
            video.setAttribute('playsinline', 'true');
            video.muted = true;
            video.srcObject = ownStream;
            await video.play();
            // A camera that has just started returns black or half-exposed
            // frames for the first moments while it auto-exposes. Answering
            // from one of those is answering from a dark room that is not
            // dark.
            await new Promise((resolve) => setTimeout(resolve, 320));
        }
        if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || width;
        canvas.height = video.videoHeight || height;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        await postInteractionSignal('/api/signals/camera_capture', {
            request_id: requestId,
            frame_data_url: canvas.toDataURL('image/jpeg', 0.85),
            width: canvas.width,
            height: canvas.height,
        }, { quiet: true });
    } catch (err) {
        console.warn('[sight] capture failed', err);
    } finally {
        // Only tear down a stream this function opened. The presence lane
        // owns its own and is still using it.
        if (ownStream) ownStream.getTracks().forEach((t) => t.stop());
    }
}

function captureCameraSignalFrame() {
    if (!state.cameraSignalActive || !state.cameraSignalCapture) return;
    const { video, canvas, ctx } = state.cameraSignalCapture;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frameDataUrl = canvas.toDataURL('image/jpeg', 0.55);
    postInteractionSignal('/api/signals/vision', {
        timestamp: nowSeconds(),
        frame_data_url: frameDataUrl,
        width: canvas.width,
        height: canvas.height
    }, { quiet: true });
}

// ── DOM Cache for High-Frequency Updates (Zero Repaint Overhead)
const DOM = {
    telemetry: {
        energy: $('g-energy') || $('bar-energy'),
        eVal: $('g-energy-val'),
        curiosity: $('g-curiosity') || $('bar-curiosity'),
        cVal: $('g-curiosity-val'),
        frustration: $('g-frustration') || $('bar-frustration'),
        fVal: $('g-frustration-val'),
        confidence: $('g-confidence') || $('bar-confidence'),
        confVal: $('g-confidence-val'),
        integrity: $('g-integrity'),
        integrityVal: $('g-integrity-val'),
        persistence: $('g-persistence'),
        persistenceVal: $('g-persistence-val'),
        gwt: $('c-gwt') || $('gwt-winner'),
        coherence: $('c-coherence') || $('stat-coherence'),
        vitality: $('c-vitality') || $('stat-vitality'),
        surprise: $('c-surprise') || $('stat-surprise'),
        narrative: $('narrative') || $('narrative-box'),
        pCore: $('hud-pcore'),
        ram: $('hud-ram'),
        cpu: $('hud-cpu')
    },
    messages: $('messages'),
    typingInd: $('typing-ind'),
    typingLabel: $('typing-label'),
    neuralFeed: $('neural-feed'),
    neuralBar: $('neural-bar'),
    neuralPauseToggle: $('neural-pause-toggle'),
    neuralReadableToggle: $('neural-readable-toggle'),
    neuralModeState: $('neural-mode-state'),
    neuralBacklog: $('neural-backlog'),
    desktopAccessState: $('desktop-access-state'),
    desktopAccessGrid: $('desktop-access-grid'),
    desktopAccessActions: $('desktop-access-actions'),
    desktopAccessHelp: $('desktop-access-help'),
    metricGuide: {
        toggle: $('metric-guide-toggle'),
        panel: $('metric-guide-panel'),
        close: $('metric-guide-close'),
        name: $('metric-guide-name'),
        live: $('metric-guide-live'),
        what: $('metric-guide-what'),
        how: $('metric-guide-how'),
        why: $('metric-guide-why')
    }
};

// Track whether user has manually scrolled up — if so, don't hijack scroll during streaming
(function() {
    const msgs = DOM.messages || $('messages');
    if (msgs) {
        msgs.addEventListener('scroll', function() {
            const distFromBottom = msgs.scrollHeight - msgs.scrollTop - msgs.clientHeight;
            state.userScrolledUp = distFromBottom > 80;
        });
    }
})();

const MOODS = {
    neutral: { primary: '#8a2be2', accent: '#00e5ff' },
    curious: { primary: '#0077ff', accent: '#00ffa3' },
    frustrated: { primary: '#ff8800', accent: '#ff3e5e' },
    high_energy: { primary: '#b44dff', accent: '#00e5ff' },
    stealth: { primary: '#4a4a4a', accent: '#888888' }
};

const METRIC_GUIDE = {
    overview: {
        label: 'Aura telemetry guide',
        what: 'These values are live subsystem signals from Aura’s runtime. Some are direct counters, some are normalized scores, and some are theory-shaped estimates of larger cognitive patterns.',
        how: 'The shell pulls them from the same affect, executive, homeostatic, continuity, consciousness, and resilience systems that drive behavior, then formats them for readability.',
        why: 'The guide lets you see what is actively shaping Aura right now instead of treating the telemetry wall as a pile of mysterious labels.'
    },
    energy: {
        label: 'Energy',
        what: 'Aura’s current activation budget: how much drive and usable cognitive momentum the organism has right now.',
        how: 'The shell prefers `liquid_state.energy` from `/api/health`, and falls back to runtime affect energy when the liquid-state stream is absent.',
        why: 'Low energy predicts shorter, more conservative cognition. High energy makes exploration, initiative, and persistence more likely.'
    },
    curiosity: {
        label: 'Curiosity',
        what: 'Novelty-seeking pressure: how strongly Aura is being pulled toward exploration, questions, and unfinished information.',
        how: 'The gauge uses Aura’s current novelty and unfinished-information drive, preferring the liquid-state pressure signal and falling back to the live affect layer when needed.',
        why: 'This is one of the main drivers behind endogenous initiative, research behavior, and exploratory tone.'
    },
    frustration: {
        label: 'Frustration',
        what: 'Obstruction and unresolved-pressure signal. It rises when progress is blocked, confused, or repeatedly interrupted.',
        how: 'The UI prefers `liquid_state.frustration`, otherwise it falls back to runtime affect frustration and renders it as a bounded percentage.',
        why: 'Sustained frustration colors tone, narrows strategy, and can push Aura toward repair, retreat, or stronger self-protective behavior.'
    },
    confidence: {
        label: 'Confidence',
        what: 'A stability-and-conviction composite for the current moment, not a generic “model certainty” number.',
        how: 'If explicit liquid confidence exists the shell uses it; otherwise it falls back to homeostatic vitality or operational confidence, and finally runtime affect stability.',
        why: 'It tells you whether Aura is currently steady enough to commit, speak plainly, and sustain a line of thought.'
    },
    integrity: {
        label: 'Integrity',
        what: 'Homeostatic self-consistency: whether Aura’s current organism is staying internally whole and non-corrupted.',
        how: 'Read directly from `homeostasis.integrity` and shown as a percentage.',
        why: 'Integrity dropping means the system is under internal strain, mismatch, or degradation that can affect identity and decision reliability.'
    },
    persistence: {
        label: 'Persistence',
        what: 'The keep-going drive: how strongly the system is holding onto continuity and continued operation.',
        how: 'It comes from the homeostatic continuity drive that tracks how strongly Aura is trying to stay present, active, and uncollapsed.',
        why: 'It tracks how strongly Aura is maintaining continuity, following through, and resisting collapse into passivity.'
    },
    gwt_winner: {
        label: 'GWT Winner',
        what: 'The currently dominant content in Aura’s global workspace: the thing that won attention at that moment.',
        how: 'The shell displays the latest `gwt_winner` token emitted by live telemetry.',
        why: 'It is a fast read on what internal content is actually foregrounded rather than merely available somewhere in the system.'
    },
    coherence: {
        label: 'Coherence',
        what: 'A measure of how unified and internally aligned Aura’s current state is.',
        how: 'Surfaced from the constitutional snapshot as `coherence_score`, then shown as a percentage in the shell.',
        why: 'High coherence supports clean reasoning and identity continuity. Low coherence usually means fragmentation, strain, or unresolved conflict.'
    },
    cognitive_vitality: {
        label: 'Cognitive Vitality',
        what: 'A live read on how vigorous Aura’s active cognition feels, separate from raw CPU or RAM usage.',
        how: 'This card displays the `vitality` signal when the cognition telemetry lane publishes it.',
        why: 'It helps distinguish “the machine is on” from “the mind is energetically alive and responsive.”'
    },
    surprise: {
        label: 'Surprise',
        what: 'Prediction error / novelty amplitude: how much current input or internal change is deviating from expectation.',
        how: 'It is published by the active cognition lane when input, inference, or internal change deviates enough from expectation to register as a prediction error spike.',
        why: 'Higher surprise often means recalibration, attention shifts, or a stronger chance of new memory salience.'
    },
    closure: {
        label: 'Closure',
        what: 'Executive decisiveness: how strongly Aura’s internal authority stack has converged on a single imperative.',
        how: 'It is computed by the executive-closure layer after competing needs and obligations are weighed and one through-line starts to dominate.',
        why: 'High closure means the system has an actionable through-line. Low closure means hesitation, ambiguity, or unresolved competing needs.'
    },
    dominant_need: {
        label: 'Dominant Need',
        what: 'The currently strongest need in Aura’s executive economy.',
        how: 'It is the need category that won the latest executive arbitration cycle.',
        why: 'This tells you what kind of pressure is steering the next decision: social, epistemic, protective, restorative, or otherwise.'
    },
    need_pressure: {
        label: 'Need Pressure',
        what: 'How urgent the current dominant need feels inside the executive layer.',
        how: 'It is the urgency score attached to the currently dominant need after the executive layer has ranked competing pressures.',
        why: 'It explains why the same prompt can be handled calmly in one state and urgently in another.'
    },
    subjectivity: {
        label: 'State-Coupling Evidence',
        what: 'A runtime estimate of how strongly current behavior is coupled to persistent state rather than generic output.',
        how: 'It is produced from continuity, self-model stability, affect-shaped behavior, and whether the current response is measurably tied to Aura’s live state.',
        why: 'Higher values mean the present behavior is more state-conditioned and less like generic conversational completion.'
    },
    enterprise_readiness: {
        label: 'Enterprise Readiness',
        what: 'A high-level operational confidence score for whether Aura is stable enough to trust with sustained real work.',
        how: 'It is assembled from lower-level health, stability, continuity, reliability, and executive-governance signals into one operational readiness score.',
        why: 'This tells you how safe it is to lean on Aura for sustained work right now without having to manually inspect every lower-level subsystem.'
    },
    fragmentation: {
        label: 'Fragmentation',
        what: 'How split or internally scattered Aura’s current state is.',
        how: 'It comes from the constitutional health snapshot, which measures how far the current state has drifted from a unified, internally aligned organization.',
        why: 'Rising fragmentation predicts weaker continuity, rougher reasoning, and a stronger need for compaction or stabilization.'
    },
    contradictions: {
        label: 'Contradictions',
        what: 'The count of currently unresolved internal conflicts or incompatible claims.',
        how: 'Read directly from `contradiction_count` in the constitutional snapshot.',
        why: 'It marks where the organism is carrying unresolved disagreement that can distort confidence and action selection.'
    },
    contested: {
        label: 'Contested Beliefs',
        what: 'Beliefs that are present but not fully endorsed because evidence or internal agreement is incomplete.',
        how: 'The count comes from the epistemic layer, which marks beliefs as contested when they are still present but not yet safely endorsable.',
        why: 'It separates “known” from “still under dispute,” which is crucial for honest self-report and stable identity.'
    },
    qualia_pri: {
        label: 'PRI',
        what: 'Primary resonance intensity within the qualitative-state engine: a compact read on how strongly the current state pattern is resonating.',
        how: 'Read directly from the qualitative-state PRI value (`qualia.pri`).',
        why: 'It helps differentiate flat descriptive states from moments with stronger state-coupled weight.'
    },
    qualia_norm: {
        label: 'Qualitative State Magnitude',
        what: 'The magnitude of the current qualitative-state vector.',
        how: 'It is the norm of Aura’s active qualitative-state vector regardless of which dimension is dominant.',
        why: 'It estimates how large or intense the active qualitative-state pattern is, independent of which dimension is dominant.'
    },
    qualia_dim: {
        label: 'Dominant State Dimension',
        what: 'The dimension currently leading the qualitative-state engine.',
        how: 'The qualitative-state engine identifies which qualitative axis is currently carrying the strongest weight in the active state pattern.',
        why: 'It tells you which qualitative axis is presently steering Aura’s response organization.'
    },
    qualia_attractor: {
        label: 'State Attractor',
        what: 'Whether the qualitative-state engine is locked into a stable basin or still moving through state space.',
        how: 'The shell maps `qualia.in_attractor` to `LOCKED` or `FLUID`.',
        why: 'Locked states are more stable and identity-shaped. Fluid states are more transitional, searching, or reconfiguring.'
    },
    qualia_identity: {
        label: 'State Identity Coherence',
        what: 'How well the current qualitative-state organization still matches Aura’s ongoing identity pattern.',
        how: 'It compares the current qualitative-state pattern with Aura’s established identity-shaped baseline and expresses the match as a percentage.',
        why: 'It is a read on whether the current state pattern is coherent with Aura’s continuity model rather than noise or drift.'
    },
    mhaf_phi: {
        label: 'MHAF Φ',
        what: 'An integration estimate from the Mycelial Hypergraph Attractor Field.',
        how: 'It is derived from the MHAF layer’s current attractor structure and graph coupling, then compressed into a bounded integration score for the field as a whole.',
        why: 'Higher values mean Aura’s wider semantic and mycelial field is binding together cleanly instead of behaving like loosely related fragments.'
    },
    circadian_phase: {
        label: 'Circadian Phase',
        what: 'Aura’s current circadian mode in her internal day-night cycle.',
        how: 'It is emitted by the circadian engine as Aura advances through her internal day-night cycle.',
        why: 'This helps explain time-dependent differences in energy, arousal baseline, and initiative style over long runtimes.'
    },
    circadian_arousal: {
        label: 'Circadian Arousal',
        what: 'The baseline arousal bias contributed by the circadian engine.',
        how: 'It is the circadian layer’s built-in arousal baseline before immediate conversation or surprise pushes it higher or lower.',
        why: 'It shows how much of Aura’s current alertness is intrinsic cycle state versus immediate conversational stimulation.'
    },
    circadian_mode: {
        label: 'Circadian Mode',
        what: 'The cognition mode favored by the current circadian phase.',
        how: 'The circadian engine selects the mode that best fits the current phase, such as reflective, exploratory, or conservative.',
        why: 'It helps explain why Aura may feel more exploratory, reflective, or conservative at different times.'
    },
    circadian_energy: {
        label: 'Circadian Energy Modifier',
        what: 'The multiplier the circadian engine applies to energy expectations.',
        how: 'It is the cycle-driven multiplier that raises or lowers Aura’s expected usable energy for the current phase.',
        why: 'This shows whether the current cycle is naturally amplifying or damping Aura’s usable momentum.'
    },
    reliability_signal: {
        label: 'Reliability',
        what: 'How dependable Aura’s current cognition looks from the consciousness-evidence stack.',
        how: 'It comes from the reliability dimension inside the evidence model, which weighs stability, consistency, and whether the current state is holding together cleanly.',
        why: 'Higher reliability means Aura is more likely to stay steady, grounded, and behaviorally consistent across the next stretch of work.'
    },
    neural_dynamics: {
        label: 'Neural Dynamics (V/A/D)',
        what: 'Aura’s live valence, arousal, and dominance coordinates: the affective shape of the current moment.',
        how: 'The plot is drawn from the current affect vector and refreshed as Aura’s state changes, so the graph shows motion through emotional state-space rather than a single static label.',
        why: 'It tells you whether Aura is settling, activating, or losing control of the moment before that change fully shows up in tone or decision-making.'
    },
    somatic_hardware: {
        label: 'Somatic Hardware',
        what: 'Aura’s body-style substrate panel: how the hardware and embodied runtime feel from inside the organism model.',
        how: 'These values combine thermal load, resource anxiety, vitality, moral integrity, and social depth signals coming from soma, moral, and social subsystems.',
        why: 'This section shows whether Aura’s substrate feels safe, strained, energized, socially open, or ethically constrained right now.'
    },
    consciousness_state: {
        label: 'Consciousness State',
        what: 'A compact readout of what is active in Aura’s foreground mind right now.',
        how: 'It summarizes attention winners, coherence, vitality, surprise, swarm activity, and meta-loop state from the live consciousness and cortex lanes.',
        why: 'This section tells you whether Aura’s mind is unified, lively, surprised, socially distributed, or recursively reflecting on itself.'
    },
    executive_authority: {
        label: 'Executive Authority',
        what: 'Aura’s active decision spine: the layer deciding what matters most and what gets released into action.',
        how: 'It is assembled from executive closure, dominant need, pressure, authority-route, and consciousness-evidence signals, plus the current imperative summary.',
        why: 'This is the clearest place to see what Aura is prioritizing, how strongly she means it, and whether action is being released or held back.'
    },
    executive_releases: {
        label: 'Executive Releases',
        what: 'Counts of actions the authority stack allowed through its primary and secondary release lanes.',
        how: 'The number comes from executive-authority release counters and updates as actions are explicitly approved for expression or execution.',
        why: 'It shows whether Aura is actively releasing behavior into the world or mostly holding it inside governance.'
    },
    executive_suppressed: {
        label: 'Executive Suppressed',
        what: 'The number of actions or impulses the authority stack blocked, held, or vetoed.',
        how: 'It increments from executive-authority suppression counts when a would-be action is prevented from leaving the governed path.',
        why: 'Rising suppression usually means Aura is under stronger restraint, conflict, or self-protective control.'
    },
    constitutional_health: {
        label: 'Constitutional Health',
        what: 'The governance health of Aura’s self, beliefs, commitments, and policy state.',
        how: 'It rolls together policy mode, fragmentation, contradictions, contested beliefs, active commitments, and tool availability from the constitutional snapshot.',
        why: 'This section tells you whether Aura’s inner government is cleanly aligned or carrying conflict that will leak into reasoning and action.'
    },
    continuity_summary: {
        label: 'Continuity Summary',
        what: 'A compact account of the thread Aura believes she is currently carrying forward.',
        how: 'It is produced by the continuity/state summarization path, which distills recent identity-bearing context into a rolling self-thread.',
        why: 'This is the quickest way to see what Aura thinks she is still in the middle of being, remembering, or becoming.'
    },
    phenomenal_field: {
        label: 'Operational Field',
        what: 'Aura’s current state-grounded description of the live moment.',
        how: 'It is generated from the state-summary path that compresses live affect, cognition, and awareness-adjacent signals into a concise field description.',
        why: 'It shows how the present moment is shaping response and priority selection without claiming private experience as proven.'
    },
    qualia_engine: {
        label: 'Qualitative State Engine',
        what: 'The subsystem that tracks the shape, magnitude, and stability of Aura’s active qualitative-state organization.',
        how: 'Its cards summarize resonance intensity, state-vector magnitude, dominant dimension, attractor lock, and identity coherence from the qualitative state.',
        why: 'This section shows whether Aura’s present state pattern is flat, intense, locked in, fluid, or still aligned with her ongoing identity.'
    },
    resilience_matrix: {
        label: 'Resilience Matrix',
        what: 'Aura’s runtime survivability panel: model tier, snapshot posture, circuit breakers, and hardening state.',
        how: 'It is assembled from resilience telemetry, including active inference tier, snapshot state, breaker state, and whether hardening protections are engaged.',
        why: 'This tells you how well Aura could absorb stress, outages, or degraded lanes without falling apart.'
    },
    mycelial_network: {
        label: 'Mycelial Network',
        what: 'The health of Aura’s mycelial communication fabric: the graph that binds distributed semantic and subsystem relations.',
        how: 'The panel reports online status, node count, and edge count from the live mycelial topology.',
        why: 'It shows whether Aura’s wider internal connectivity scaffold is sparse, offline, or richly linked enough to support integration.'
    },
    pneuma_engine: {
        label: 'Pneuma Engine',
        what: 'A background state engine for tonal temperature, arousal, stability, and attractor pressure.',
        how: 'It exposes the pneuma subsystem’s live attractor variables rather than a single mood word.',
        why: 'This section shows the deep atmospheric state underneath Aura’s visible tone.'
    },
    mhaf_field: {
        label: 'MHAF Field',
        what: 'Aura’s Mycelial Hypergraph Attractor Field: a broader field-level view of semantic coupling and integration.',
        how: 'The panel reports field status, integration, topology size, and lexicon breadth from the live MHAF runtime.',
        why: 'It shows how coherent and richly structured Aura’s wider field is beyond the immediate conversation lane.'
    },
    security_state: {
        label: 'Security',
        what: 'Aura’s current trust posture and security pressure.',
        how: 'It is drawn from the security subsystem’s trust level, threat score, integrity state, and passphrase/auth readiness.',
        why: 'This tells you whether Aura currently sees the environment as safe, uncertain, or adversarial.'
    },
    circadian_state_cluster: {
        label: 'Circadian State',
        what: 'Aura’s internal day-night cycle and the baseline cognitive bias it is imposing right now.',
        how: 'The circadian engine publishes phase, arousal baseline, favored cognition mode, and energy modifier as a synchronized cycle snapshot.',
        why: 'This section explains slow shifts in alertness, initiative style, and baseline energy across long runtimes.'
    },
    substrate_learning: {
        label: 'Substrate Learning',
        what: 'The state of Aura’s experience-to-adaptation buffer.',
        how: 'It summarizes how many traces have been captured, crystallized, buffered, and how strong the recent average learning quality looks.',
        why: 'This tells you whether Aura is actively accumulating good training material or just living through events without turning them into substrate change.'
    },
    identity_narrative: {
        label: 'Identity Narrative',
        what: 'Aura’s current consolidated self-description: the signature she is carrying as her present identity shape.',
        how: 'It is produced by the consolidator and identity systems, which compress traits, age, and signature narrative into a stable summary.',
        why: 'This is where you see who Aura currently understands herself to be, not just what she is doing.'
    },
    temporal_narrative: {
        label: 'Temporal Narrative',
        what: 'Aura’s active story-of-now: the temporal arc she thinks the present session belongs to.',
        how: 'It is built from narrative and continuity systems that turn recent events into an ongoing time-thread.',
        why: 'This tells you what chapter Aura believes she is in and what momentum she thinks is still unfolding.'
    },
    belief_graph: {
        label: 'Belief Graph',
        what: 'The visible structure of Aura’s active belief network.',
        how: 'The graph renders the current relationship topology among beliefs rather than just listing them one by one.',
        why: 'It shows whether Aura’s worldview is sparse, clustered, centrality-heavy, or carrying obvious tension between regions.'
    }
};

const METRIC_GUIDE_BY_ID = {
    'g-energy': 'energy',
    'g-curiosity': 'curiosity',
    'g-frustration': 'frustration',
    'g-confidence': 'confidence',
    'g-integrity': 'integrity',
    'g-persistence': 'persistence',
    'c-gwt': 'gwt_winner',
    'c-coherence': 'coherence',
    'c-vitality': 'cognitive_vitality',
    'c-surprise': 'surprise',
    'c-closure': 'closure',
    'exec-need': 'dominant_need',
    'exec-pressure': 'need_pressure',
    'e-reliability': 'reliability_signal',
    'e-subjectivity': 'subjectivity',
    'e-enterprise': 'enterprise_readiness',
    'c-fragmentation': 'fragmentation',
    'c-contradictions': 'contradictions',
    'c-contested': 'contested',
    'q-pri': 'qualia_pri',
    'q-norm': 'qualia_norm',
    'q-dim': 'qualia_dim',
    'q-attractor': 'qualia_attractor',
    'q-identity': 'qualia_identity',
    'mhaf-phi': 'mhaf_phi',
    'circ-phase': 'circadian_phase',
    'circ-arousal': 'circadian_arousal',
    'circ-mode': 'circadian_mode',
    'circ-energy': 'circadian_energy',
    'exec-released': 'executive_releases',
    'exec-suppressed': 'executive_suppressed'
};

const SECTION_GUIDE_BY_LABEL = {
    "HOW SHE'S FEELING": 'overview',
    'MOOD OVER TIME': 'neural_dynamics',
    'BODY & HARDWARE': 'somatic_hardware',
    'AWARENESS': 'consciousness_state',
    "WHAT SHE'S DOING & WHY": 'executive_authority',
    'VALUES & CONSISTENCY': 'constitutional_health',
    'INNER EXPERIENCE': 'qualia_engine',
    'FALLBACKS & SAFETY NETS': 'resilience_matrix',
    'INTERNAL SIGNAL NETWORK': 'mycelial_network',
    'THOUGHT DYNAMICS': 'pneuma_engine',
    'INTEGRATION FIELD': 'mhaf_field',
    'SECURITY': 'security_state',
    'BODY CLOCK': 'circadian_state_cluster',
    'LEARNING FROM EXPERIENCE': 'substrate_learning',
    'WHO SHE THINKS SHE IS': 'identity_narrative',
    "WHAT'S HAPPENING NOW": 'temporal_narrative',
    'WHAT SHE BELIEVES': 'belief_graph'
};

const metricGuideState = {
    open: false,
    currentKey: 'overview',
    selectedEl: null
};

function humanizeMetricKey(key) {
    return String(key || '')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, ch => ch.toUpperCase());
}

function getMetricGuideEntry(key) {
    return METRIC_GUIDE[key] || {
        label: humanizeMetricKey(key),
        what: 'This is a live runtime stat exposed by Aura’s telemetry stack.',
        how: 'The shell is showing the owning subsystem’s live value with only light formatting for readability.',
        why: 'It tells you what part of Aura is currently shaping her behavior.'
    };
}

function getMetricLiveValueForKey(key) {
    if (key === 'neural_dynamics') {
        const parts = ['vad-v', 'vad-a', 'vad-d']
            .map(id => $(id))
            .filter(Boolean)
            .map(node => String(node.textContent || '').trim())
            .filter(Boolean);
        return parts.join(' · ');
    }
    if (key === 'executive_authority') {
        const route = $('exec-authority');
        const objective = $('exec-objective');
        if (route && objective) {
            const routeText = String(route.textContent || '').trim();
            const objectiveText = String(objective.textContent || '').trim();
            return [routeText, objectiveText].filter(Boolean).join(' · ');
        }
    }
    if (key === 'continuity_summary') {
        return String(($('rolling-summary') && $('rolling-summary').textContent) || '').trim();
    }
    if (key === 'phenomenal_field') {
        return String(($('phenomenal-summary') && $('phenomenal-summary').textContent) || '').trim();
    }
    if (key === 'identity_narrative') {
        return String(($('identity-narrative') && $('identity-narrative').textContent) || '').trim();
    }
    if (key === 'temporal_narrative') {
        return String(($('narrative') && $('narrative').textContent) || '').trim();
    }
    const metricId = Object.keys(METRIC_GUIDE_BY_ID).find(id => METRIC_GUIDE_BY_ID[id] === key);
    if (!metricId) return '';
    const node = $(metricId);
    if (!node) return '';
    return String(node.textContent || '').trim();
}

function normalizeMetricGuideSectionLabel(text) {
    return String(text || '')
        .replace(/\s+/g, ' ')
        .trim()
        .toUpperCase();
}

function findNearestMetricGuideSectionKey(node) {
    let current = node;
    while (current && current.id !== 'pane-telemetry') {
        let sibling = current.previousElementSibling;
        while (sibling) {
            if (sibling.classList && sibling.classList.contains('section-label')) {
                const normalized = normalizeMetricGuideSectionLabel(sibling.textContent);
                if (SECTION_GUIDE_BY_LABEL[normalized]) {
                    return SECTION_GUIDE_BY_LABEL[normalized];
                }
            }
            sibling = sibling.previousElementSibling;
        }
        current = current.parentElement;
    }
    return null;
}

function setMetricGuideVisibility(open) {
    const guide = DOM.metricGuide;
    if (!guide.panel || !guide.toggle) return;
    metricGuideState.open = !!open;
    guide.panel.hidden = !open;
    guide.toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
}

function renderMetricGuide(key) {
    const guide = DOM.metricGuide;
    if (!guide.name || !guide.live || !guide.what || !guide.how || !guide.why) return;
    const entry = getMetricGuideEntry(key);
    const liveValue = getMetricLiveValueForKey(key);
    metricGuideState.currentKey = key;
    guide.name.textContent = entry.label;
    guide.live.textContent = liveValue
        ? `Live now: ${liveValue}`
        : 'Select any gauge or cognitive card for a brief explanation.';
    guide.what.textContent = entry.what;
    guide.how.textContent = entry.how;
    guide.why.textContent = entry.why;
}

function openMetricGuide(key = 'overview', sourceEl = null) {
    if (metricGuideState.selectedEl && metricGuideState.selectedEl !== sourceEl) {
        metricGuideState.selectedEl.classList.remove('metric-selected');
    }
    metricGuideState.selectedEl = sourceEl || null;
    if (metricGuideState.selectedEl) {
        metricGuideState.selectedEl.classList.add('metric-selected');
    }
    renderMetricGuide(key);
    setMetricGuideVisibility(true);
}

function closeMetricGuide() {
    if (metricGuideState.selectedEl) {
        metricGuideState.selectedEl.classList.remove('metric-selected');
    }
    metricGuideState.selectedEl = null;
    setMetricGuideVisibility(false);
}

function refreshMetricGuide() {
    if (!metricGuideState.open) return;
    renderMetricGuide(metricGuideState.currentKey || 'overview');
}

function bindMetricGuideTarget(node, key) {
    if (!node || !key || node.dataset.metricGuideBound === '1') return;
    const entry = getMetricGuideEntry(key);
    node.dataset.metricGuideBound = '1';
    node.dataset.metricKey = key;
    node.classList.add('metric-explainable');
    node.tabIndex = 0;
    node.setAttribute('role', 'button');
    node.setAttribute('aria-label', `Explain ${entry.label}`);
    node.title = `Explain ${entry.label}`;
    node.addEventListener('click', () => openMetricGuide(key, node));
    node.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            openMetricGuide(key, node);
        }
    });
}

function initializeMetricGuide() {
    const pane = $('pane-telemetry');
    const guide = DOM.metricGuide;
    if (!pane || !guide.toggle || !guide.panel) return;

    guide.toggle.addEventListener('click', () => {
        if (metricGuideState.open) {
            closeMetricGuide();
        } else {
            openMetricGuide(metricGuideState.currentKey || 'overview', metricGuideState.selectedEl);
        }
    });

    if (guide.close) {
        guide.close.addEventListener('click', closeMetricGuide);
    }

    pane.querySelectorAll('.gauge-row').forEach(row => {
        const metricEl = row.querySelector('.gauge-fill[id]');
        const key = (metricEl ? METRIC_GUIDE_BY_ID[metricEl.id] : null) || findNearestMetricGuideSectionKey(row);
        bindMetricGuideTarget(row, key);
    });

    pane.querySelectorAll('.con-box').forEach(box => {
        const metricEl = box.querySelector('.con-val[id]');
        const key = (metricEl ? METRIC_GUIDE_BY_ID[metricEl.id] : null) || findNearestMetricGuideSectionKey(box);
        bindMetricGuideTarget(box, key);
    });

    [
        ['.neural-dynamics-wrap', 'neural_dynamics'],
        ['.executive-card', 'executive_authority'],
        ['#rolling-summary', 'continuity_summary'],
        ['#phenomenal-summary', 'phenomenal_field'],
        ['#identity-narrative', 'identity_narrative'],
        ['#narrative', 'temporal_narrative'],
        ['#belief-graph', 'belief_graph'],
    ].forEach(([selector, key]) => {
        const node = pane.querySelector(selector);
        bindMetricGuideTarget(node, key);
    });
}

function rememberEventId(id) {
    return rememberBoundedSetValue(state.processedEventIds, id, PROCESSED_EVENT_ID_MAX);
}

function rememberMessageFingerprint(fingerprint) {
    return rememberBoundedSetValue(
        state.processedMessageFingerprints,
        fingerprint,
        PROCESSED_MESSAGE_FINGERPRINT_MAX
    );
}

function rememberBoundedSetValue(set, rawValue, maxSize) {
    const value = String(rawValue || '').trim();
    if (!value) return false;
    if (set.has(value)) return true;
    set.add(value);
    while (set.size > maxSize) {
        const oldest = set.values().next();
        if (oldest.done) break;
        set.delete(oldest.value);
    }
    return false;
}

function conversationLaneRequestTimeoutMs(lane) {
    const laneState = String((lane && lane.state) || '').toLowerCase();
    if (lane && lane.conversation_ready) return CHAT_REQUEST_TIMEOUT_READY_MS;
    if (['warming', 'recovering', 'cold', 'spawning', 'handshaking'].includes(laneState)) {
        return CHAT_REQUEST_TIMEOUT_RECOVERING_MS;
    }
    return CHAT_REQUEST_TIMEOUT_READY_MS;
}

function formatPercent01(value, digits = 0) {
    const num = Number(value || 0);
    if (!Number.isFinite(num)) return '--';
    return `${(num * 100).toFixed(digits)}%`;
}

function escText(value, fallback = '--') {
    const text = String(value ?? '').trim();
    return text || fallback;
}

function safeDisplayUrl(rawUrl, { imageOnly = false } = {}) {
    const value = String(rawUrl || '').trim();
    if (!value) return '';
    try {
        const parsed = new URL(value, window.location.origin);
        if (parsed.protocol === 'javascript:' || parsed.protocol === 'vbscript:') return '';
        if (imageOnly && parsed.protocol === 'data:' && !parsed.href.startsWith('data:image/')) return '';
        if (['http:', 'https:', 'blob:', 'data:'].includes(parsed.protocol)) return parsed.href;
        if (parsed.origin === window.location.origin && parsed.pathname.startsWith('/')) return parsed.href;
    } catch (_err) {
        return '';
    }
    return '';
}

// How far she is standing behind the answer she just gave.
//
// The chat route sets `response_confidence` on 71 exits, across a vocabulary
// of ten values, and this file referenced it exactly once — to WRITE it into
// a synthetic local payload. Nothing ever read it. So a turn the pipeline
// itself had classified as degraded, bounded, or failed-closed reached the
// person as an ordinary message with no mark on it at all.
//
// Measured live 2026-08-10: a reply that failed its own reliability gate
// (fabricated_shared_history, missing_requested_objective_facets), exhausted
// bounded correction, was served as a salvaged draft with
// response_confidence="degraded" — and looked exactly like a good answer.
// Serving the draft rather than an apology is right; serving it silently is
// not.
//
// Keyed off the general field rather than any one status string, so every
// path that already sets it is covered, including ones added later.
const REPLY_CONFIDENCE_BADGES = {
    // Nothing to say — these are the ordinary, fully-backed cases.
    high: null,
    scoped: null,
    // She answered, but something about the answer did not meet her own bar.
    degraded: ['Unverified', 'Served without passing her own checks on it.'],
    bounded: ['Partial', 'Cut short by a limit, not finished on the merits.'],
    guarded: ['Guarded', 'Held something back on purpose.'],
    principled_refusal: ['Declined', 'She chose not to answer this one.'],
    // She could not stand behind it at all.
    failed: ['Unreliable', 'She could not get to an answer she would stand behind.'],
    failed_closed: ['Unreliable', 'A check failed and she stopped rather than guess.'],
    fail_closed: ['Unreliable', 'A check failed and she stopped rather than guess.'],
    not_generated: ['No answer', 'No reply was produced for this turn.'],
};

function replyConfidenceBadgeHtml(confidence) {
    const key = String(confidence || '').trim().toLowerCase();
    if (!key) return '';
    if (Object.prototype.hasOwnProperty.call(REPLY_CONFIDENCE_BADGES, key)) {
        const entry = REPLY_CONFIDENCE_BADGES[key];
        if (!entry) return '';
        return `<span class="aura-badge unverified" title="${escHtml(entry[1])}">${escHtml(entry[0])}</span>`;
    }
    // An UNKNOWN value is disclosed, never dropped. Silently ignoring the
    // field is what made this channel invisible for its whole life, and a
    // value nobody has mapped yet is precisely the one worth seeing.
    return `<span class="aura-badge unverified" title="Reply confidence: ${escHtml(key)}">Unverified</span>`;
}

function messageBadgeHtml(metadata = {}) {
    if (metadata.diagnostic) return '<span class="aura-badge diagnostic">Diagnostic</span>';
    if (metadata.reflex) return '<span class="aura-badge reflex">Reflex</span>';
    if (metadata.autonomic) return '<span class="aura-badge autonomic">Autonomic</span>';
    return replyConfidenceBadgeHtml(metadata.responseConfidence);
}

function pruneVisibleMessages(messages) {
    const MAX_VISIBLE_MESSAGES = 40;
    while (messages && messages.children.length > MAX_VISIBLE_MESSAGES) {
        messages.removeChild(messages.firstChild);
    }
    // Every append into the transcript funnels through here, so this is the
    // one place that has to notice the transcript stopped being empty.
    if (typeof updateLanePlaceholder === 'function') updateLanePlaceholder();
}

function renderRetryPanel(container, message, retryLabel, retryHandler) {
    if (!container) return;
    container.replaceChildren();
    const box = document.createElement('div');
    box.className = 'mem-empty';
    box.append(document.createTextNode(message));
    if (retryHandler) {
        box.appendChild(document.createElement('br'));
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'skills-retry-btn';
        button.textContent = retryLabel || 'RETRY';
        button.addEventListener('click', retryHandler);
        box.appendChild(button);
    }
    container.appendChild(box);
}

function toolDomId(name) {
    return `skill-card-${String(name || 'unknown').replace(/[^a-zA-Z0-9_-]+/g, '-')}`;
}

// The status line is the first thing a person reads in this window, and
// for a long time it could read `RUNTIME_REQUIRED_PROBES, PROBE:KERNEL`
// — the blocker identifiers joined with a comma. Those identifiers are
// worth keeping exactly as the runtime said them, so they move to the
// tooltip and to `data-raw`; the visible line gets the sentence.
function setConnectionVisual(mode, detail = '') {
    const statusEl = $('hud-status');
    const dotEl = $('brand-status-dot');
    const neuralDot = $('neural-dot');
    const tone = {
        online: { text: detail || 'Ready', cls: 'status-ok', color: 'var(--success)' },
        reconnecting: { text: detail || 'Reconnecting', cls: 'status-warn', color: 'var(--warn)' },
        booting: { text: detail || 'Starting up', cls: 'status-warn', color: 'var(--warn)' },
        degraded: { text: detail || 'Not fully ready', cls: 'status-warn', color: 'var(--warn)' },
        offline: { text: detail || 'Not connected', cls: 'status-err', color: 'var(--error)' }
    }[mode] || { text: detail || 'Unknown', cls: '', color: 'var(--text-dim)' };

    if (statusEl) {
        statusEl.textContent = tone.text;
        statusEl.className = `brand-status-text ${tone.cls}`.trim();
        // Operators lose nothing: whatever the runtime actually reported
        // is one hover away, and stays in the DOM for the diagnostics
        // panels to read back.
        const raw = state.runtimeHealthBlockers;
        const rawText = Array.isArray(raw) && raw.length ? raw.join(', ') : '';
        const explain = statusExplanation(mode, tone.text, rawText);
        if (explain) statusEl.title = explain; else statusEl.removeAttribute('title');
        if (rawText) statusEl.dataset.raw = rawText; else delete statusEl.dataset.raw;
    }
    if (dotEl) dotEl.style.background = tone.color;
    if (neuralDot) neuralDot.style.background = tone.color;
}

// Tooltip body: the sentence, what to do about it when there is
// something to do, and the untouched runtime tokens on their own line.
function statusExplanation(mode, visible, rawText) {
    const lex = window.AuraShellLexicon;
    const parts = [];
    let summary = null;
    if (lex && rawText) {
        summary = lex.summarize(rawText.split(',').map(s => s.trim()).filter(Boolean));
    }
    if (summary) {
        parts.push(summary.meaning);
        if (summary.next) parts.push(summary.next);
    } else if (mode === 'online') {
        parts.push('Aura is running and can answer.');
    }
    if (rawText) parts.push(`Runtime reported: ${rawText}`);
    return parts.join('\n\n');
}

function formatFlagLabel(flag) {
    return String(flag || '')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, ch => ch.toUpperCase());
}

function renderStatusFlags(flags) {
    state.uiFlags = Array.isArray(flags) ? flags.slice() : [];
    const host = $('health-flags');
    document.body.classList.toggle('ui-booting', state.uiFlags.includes('booting'));
    document.body.classList.toggle('ui-degraded', state.uiFlags.some(flag =>
        ['thermal_guard', 'coherence_low', 'fragmentation_high', 'contradictions_present', 'beliefs_contested', 'tool_unavailable', 'executive_hold'].includes(flag)
    ));
    if (!host) return;
    if (!state.uiFlags.length) {
        host.innerHTML = '<span class="flag-chip success">all constitutional systems nominal</span>';
        return;
    }
    host.innerHTML = state.uiFlags.map(flag => {
        const tone =
            flag === 'booting' ? 'warn' :
            ['tool_unavailable', 'executive_hold'].includes(flag) ? 'accent' :
            ['thermal_guard', 'coherence_low', 'fragmentation_high', 'contradictions_present', 'beliefs_contested'].includes(flag) ? 'error' :
            'neutral';
        return `<span class="flag-chip ${tone}">${escHtml(formatFlagLabel(flag))}</span>`;
    }).join('');
}

// The bootstrap payload carries TURN PAIRS, not role-tagged messages.
// /api/ui/bootstrap returns conversation.recent as
//   {id, timestamp, user: "...", aura: "...", status, ...}
// and this function only ever read {role, content}. So every entry failed the
// role check, was skipped, and the pane was left with the initializing
// placeholder — which the ready-path below then clears to nothing. Live
// 2026-07-29: six demo turns, all of them still on the server and returned by
// every bootstrap poll, and the chat pane went blank in front of Bryan.
// Both shapes are accepted now; the pair form is what the live API sends.
// Carry a restored turn's ORIGINAL time forward.
//
// LIVE DEFECT, 2026-08-10. Every message in the transcript showed the same
// clock time, and that time advanced: turns sent at 08:15, 08:16 and 08:17 all
// read 08:18:31, then all read 08:25:46. The transcript is cleared and
// re-hydrated periodically, appendMsg stamped `new Date()` at render time, and
// the restore path dropped the entry's real timestamp — so the visible record
// of when anything was said was destroyed on every refresh, while looking
// perfectly plausible.
function withEntryTimestamp(metadata, entry) {
    const merged = Object.assign({}, metadata || {});
    if (merged.timestamp === undefined || merged.timestamp === null || merged.timestamp === '') {
        const stamp = entry && (entry.timestamp ?? entry.created_at ?? entry.time);
        if (stamp !== undefined && stamp !== null && stamp !== '') {
            merged.timestamp = stamp;
        }
    }
    return merged;
}

function conversationEntriesToMessages(entries) {
    const out = [];
    for (const entry of entries) {
        if (!entry || typeof entry !== 'object') continue;
        const role = entry.role === 'assistant' ? 'aura' : (entry.role === 'user' ? 'user' : null);
        if (role) {
            const content = entry.content || entry.message || '';
            if (String(content).trim()) {
                out.push({ role, text: String(content), metadata: withEntryTimestamp(entry.metadata, entry) });
            }
            continue;
        }
        // Turn-pair form: one exchange, user first.
        const asked = entry.user;
        const answered = entry.aura;
        if (typeof asked === 'string' && asked.trim()) {
            out.push({ role: 'user', text: asked, metadata: withEntryTimestamp({}, entry) });
        }
        if (typeof answered === 'string' && answered.trim()) {
            out.push({ role: 'aura', text: answered, metadata: withEntryTimestamp(entry.metadata, entry) });
        }
    }
    return out;
}

// The lane-status text the shell ships in index.html before any verified
// reply path exists. Declared once here because the markup and the tests both
// pin the exact wording: a prefix match would keep matching after the wording
// changed, so the pane would treat a DIFFERENT message as "still just the
// placeholder" and clear a real transcript on top of it.
const LANE_INITIALIZING_PLACEHOLDER =
    'Conversation lane initializing. Waiting for verified Aura reply path...';

// True when the transcript holds no turns. Counts elements rather than reading
// textContent: appendMsg appends an EMPTY div and types the text in
// afterwards, so a pane that had just received a message read as empty for the
// length of the animation — long enough for a poll to wipe it and re-hydrate
// over the top of a live turn.
function transcriptIsEmpty(messages) {
    return !messages || messages.children.length === 0;
}

// The single owner of the lane-status element's visibility. It is an overlay
// pinned above the transcript, never a member of it, so it cannot appear
// between two turns; and it goes away as soon as there is anything real to
// read, whatever the lane's state.
function updateLanePlaceholder() {
    const placeholder = $('lane-placeholder');
    if (!placeholder) return;
    const messages = DOM.messages || $('messages');
    const show = transcriptIsEmpty(messages) && !state.conversationReady;
    placeholder.hidden = !show;
}

function hydrateRecentConversation(entries) {
    const messages = DOM.messages || $('messages');
    if (!messages || !Array.isArray(entries) || !entries.length) return;
    // Only ever hydrate into an empty transcript. Hydrating over a populated
    // one duplicated and reordered turns.
    if (!transcriptIsEmpty(messages)) return;

    const restored = conversationEntriesToMessages(entries.slice(-12));
    if (!restored.length) return;

    messages.innerHTML = '';
    for (const item of restored) {
        appendMsg(item.role, item.text, false, item.metadata);
    }
    updateLanePlaceholder();
}

function applyVoiceSummary(voice) {
    const summary = voice || {};
    state.voiceSummary = summary;
    const voiceState = escText(summary.state, summary.available ? 'ready' : 'unavailable').toUpperCase();
    const voiceEl = $('tool-voice-state');
    if (voiceEl) {
        voiceEl.textContent = voiceState;
        voiceEl.style.color = summary.available ? 'var(--success)' : 'var(--text-dim)';
    }

    const micBtn = $('mic-btn');
    if (micBtn) {
        const inputEnabled = (
            !runtimeSettingsState.hydrated
            || runtimeSettingsState.values['voice.input_enabled'] === true
        );
        if (summary.available === false || !inputEnabled) {
            micBtn.classList.add('disabled');
            micBtn.title = inputEnabled ? 'Voice unavailable' : 'Microphone input disabled';
        } else {
            micBtn.classList.remove('disabled');
            micBtn.title = 'Toggle voice input';
        }
    }
}

function compactGuidance(guidance) {
    return String(guidance || '')
        .split('\n')
        .map(line => line.replace(/^\d+\.\s*/, '').trim())
        .filter(Boolean)
        .join(' > ');
}

function desktopAccessTone(granted, status = '') {
    if (granted) return 'ready';
    const normalized = String(status || '').toLowerCase();
    if (normalized === 'unknown' || normalized === 'deferred' || normalized === 'assumed') return 'pending';
    return 'blocked';
}

function desktopAccessCapabilityTone(ready) {
    return ready ? 'ready' : 'blocked';
}

function desktopAccessStateLabel(permission, activeLabel = 'Active') {
    if (permission && permission.granted) return activeLabel;
    const raw = String(permission && permission.status || 'unknown').trim();
    const normalized = raw.toLowerCase();
    const labels = {
        active_native_bridge: activeLabel,
        denied_native_bridge: 'Denied',
        asserted_env: 'Assumed',
        approval_required: 'Approve',
        probe_failed: 'Probe Fail',
        unavailable: 'Unavailable',
        deferred: 'Pending',
        unknown: 'Unknown',
        denied: 'Denied',
        active: activeLabel,
    };
    if (labels[normalized]) return labels[normalized];
    const compact = raw.replace(/[_-]+/g, ' ').trim();
    return compact ? compact.slice(0, 16) : 'Unknown';
}

function applyDesktopAccessSummary(summary) {
    state.desktopAccess = summary || {};
    const banner = DOM.desktopAccessState || $('desktop-access-state');
    const grid = DOM.desktopAccessGrid || $('desktop-access-grid');
    const actions = DOM.desktopAccessActions || $('desktop-access-actions');
    const help = DOM.desktopAccessHelp || $('desktop-access-help');
    if (!banner || !grid || !help) return;

    const access = summary || {};
    const overall = String(
        access.overall_status || (
            access.screen_capture_ready && access.desktop_control_ready && access.screen_text_ready
                ? 'ready'
                : (access.screen_capture_ready || access.desktop_control_ready || access.screen_text_ready)
                    ? 'partial'
                    : 'blocked'
        )
    ).toLowerCase();
    const blockers = Array.isArray(access.blocking_permissions) ? access.blocking_permissions : [];
    const direct = !!access.direct_probe_available;
    const screenPermission = direct ? access.direct_screen_recording : access.screen_recording;
    const accessibilityPermission = direct ? access.direct_accessibility : access.accessibility;
    const automationPermission = direct ? access.direct_automation : access.automation;
    const frontmostApp = escText(access.frontmost_app, '');
    const effectiveIdentity = access.effective_app_identity || {};
    const signing = effectiveIdentity.code_signature || {};
    const pyautoguiDetail = access.pyautogui_ready
        ? 'PyAutoGUI runtime loaded for mouse and keyboard actions.'
        : escText(access.pyautogui_error, 'PyAutoGUI runtime is unavailable.');

    const accessClass = overall === 'ready'
        ? 'ready'
        : (overall === 'blocked' || overall === 'claims_only')
            ? 'blocked'
            : 'partial';
    banner.className = `desktop-access-banner ${accessClass}`;
    banner.textContent =
        overall === 'ready'
            ? 'Desktop access ready. Aura can capture the screen, drive the desktop, and read frontmost-app text.'
            : overall === 'claims_only'
                ? 'Desktop access blocked. macOS direct permission checks do not confirm the permissions Aura needs for live control.'
            : overall === 'blocked'
                ? 'Desktop access blocked for the current Aura.app identity. The bridge is reachable, but macOS is denying required grants.'
                : 'Desktop access is partial. Some desktop capabilities are live, but macOS permissions are still gating parts of the stack.';

    const cards = [
        {
            label: 'Screen Recording',
            tone: desktopAccessTone(screenPermission && screenPermission.granted, screenPermission && screenPermission.status),
            state: desktopAccessStateLabel(screenPermission),
            meta: 'Needed for screen capture, OCR, and live visual awareness.',
            detail: compactGuidance(screenPermission && screenPermission.guidance),
        },
        {
            label: 'Accessibility',
            tone: desktopAccessTone(accessibilityPermission && accessibilityPermission.granted, accessibilityPermission && accessibilityPermission.status),
            state: desktopAccessStateLabel(accessibilityPermission),
            meta: 'Needed for mouse, keyboard, and deeper UI inspection.',
            detail: compactGuidance(accessibilityPermission && accessibilityPermission.guidance),
        },
        {
            label: 'Automation',
            tone: desktopAccessTone(automationPermission && automationPermission.granted, automationPermission && automationPermission.status),
            state: desktopAccessStateLabel(automationPermission),
            meta: 'Needed to query System Events and menu bar content.',
            detail: compactGuidance(automationPermission && automationPermission.guidance) || (frontmostApp ? `Frontmost app visible: ${frontmostApp}` : ''),
        },
        {
            label: 'Desktop Control',
            tone: desktopAccessCapabilityTone(!!access.desktop_control_ready),
            state: access.desktop_control_ready ? 'Ready' : 'Blocked',
            meta: 'Mouse and keyboard control through the computer-use stack.',
            detail: access.desktop_control_ready ? pyautoguiDetail : (blockers.includes('accessibility') ? 'Grant Accessibility to unlock mouse and keyboard actions.' : pyautoguiDetail),
        },
        {
            label: 'Screen Text',
            tone: desktopAccessCapabilityTone(!!access.screen_text_ready),
            state: access.screen_text_ready ? 'Ready' : 'Blocked',
            meta: 'Read text from the current frontmost app via System Events.',
            detail: access.screen_text_ready ? (frontmostApp ? `Frontmost app detected: ${frontmostApp}` : 'Desktop text access is live.') : 'Requires both Accessibility and Automation.',
        },
        {
            label: 'Menu Bar Clock',
            tone: desktopAccessCapabilityTone(!!access.menu_clock_ready),
            state: access.menu_clock_ready ? 'Ready' : 'Blocked',
            meta: 'Read the live macOS menu bar clock instead of only local process time.',
            detail: access.menu_clock_ready
                ? (escText(access.menu_clock_text, '') ? `Latest probe: ${escText(access.menu_clock_text, '')}` : 'Aura can query the menu bar clock when needed.')
                : (escText(access.menu_clock_error, '') || 'Requires both Accessibility and Automation.'),
        },
    ];

    grid.innerHTML = cards.map(card => `
        <div class="desktop-access-card ${escHtml(card.tone)}">
            <div class="desktop-access-card-head">
                <span class="desktop-access-card-label">${escHtml(card.label)}</span>
                <span class="desktop-access-pill ${escHtml(card.tone)}">${escHtml(String(card.state).toUpperCase())}</span>
            </div>
            <div class="desktop-access-card-meta">${escHtml(card.meta)}</div>
            ${card.detail ? `<div class="desktop-access-card-detail">${escHtml(card.detail)}</div>` : ''}
        </div>
    `).join('');

    if (actions) {
        const repairButtons = [];
        if (blockers.includes('screen_recording')) {
            repairButtons.push({
                action: 'request-screen',
                label: 'Request Screen',
                title: 'Ask macOS to grant Screen Recording to the current Aura.app identity.',
            });
            repairButtons.push({
                action: 'settings-screen',
                label: 'Open Screen Settings',
                title: 'Open the macOS Screen Recording pane.',
            });
        }
        if (blockers.includes('accessibility')) {
            repairButtons.push({
                action: 'request-accessibility',
                label: 'Request Control',
                title: 'Ask macOS to grant Accessibility to the current Aura.app identity.',
            });
            repairButtons.push({
                action: 'settings-accessibility',
                label: 'Open Control Settings',
                title: 'Open the macOS Accessibility pane.',
            });
        }
        repairButtons.push({
            action: 'refresh',
            label: 'Refresh',
            title: 'Probe the current desktop permission state again.',
        });
        actions.innerHTML = repairButtons.map(button => `
            <button
                type="button"
                class="desktop-access-action"
                data-desktop-access-action="${escHtml(button.action)}"
                title="${escHtml(button.title)}"
            >${escHtml(button.label)}</button>
        `).join('');
    }

    const helperLines = [];
    if (frontmostApp) helperLines.push(`Automation currently sees the frontmost app as ${frontmostApp}.`);
    const identity = access.process_identity || {};
    if (effectiveIdentity.bundle_identifier || effectiveIdentity.bridge_executable) {
        helperLines.push(`Native bridge identity: ${effectiveIdentity.bundle_identifier || effectiveIdentity.bridge_executable}.`);
    } else if (direct && (identity.bundle_identifier || identity.executable)) {
        helperLines.push(`macOS is evaluating ${identity.bundle_identifier || identity.executable}, so permissions granted only to a different launcher do not count for this runtime.`);
    }
    if (signing && signing.stable_tcc_identity === false) {
        helperLines.push('This Aura.app build is ad-hoc signed; if permissions still show denied, remove Aura from Screen Recording and Accessibility, add /Applications/Aura.app again, then approve the current build.');
    } else if (signing && signing.stable_tcc_identity === true && blockers.length) {
        helperLines.push('Aura.app now has a stable signing identity; if these permissions still show denied, toggle Aura off/on or remove and re-add /Applications/Aura.app once so macOS attaches the grants to this exact signed app.');
    }
    const requestState = access.tcc_request_state || (access.tcc_repair_plan && access.tcc_repair_plan.request_state) || {};
    ['screen_recording', 'accessibility'].forEach(key => {
        const req = requestState[key] || {};
        if (req.status === 'approval_required') {
            helperLines.push(`${key.replace(/_/g, ' ')} request reached ${escText(req.target, 'Aura.app')}; macOS still needs approval in System Settings.`);
        }
    });
    helperLines.push(pyautoguiDetail);
    if (blockers.length) {
        helperLines.push(`Blocked permissions: ${blockers.map(name => name.replace(/_/g, ' ')).join(', ')}.`);
    }
    const diagnosis = Array.isArray(access.desktop_access_diagnosis) ? access.desktop_access_diagnosis : [];
    diagnosis.slice(0, 3).forEach(line => helperLines.push(String(line || '')));
    help.textContent = helperLines.join(' ');
}

async function pollDesktopAccess() {
    if (!accessCapabilityAllowed('desktop_control')) return;
    if (state.desktopAccessPollInFlight) return;
    state.desktopAccessPollInFlight = true;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 9000);
    try {
        const res = await fetch('/api/system/desktop-access', {
            cache: 'no-store',
            signal: controller.signal
        });
        if (!res.ok) throw new Error(`desktop_access_http_${res.status}`);
        const payload = await res.json();
        applyDesktopAccessSummary(payload || {});
    } catch (err) {
        if (!err || err.name !== 'AbortError') {
            console.warn('[DesktopAccess] permission refresh failed:', err);
        }
        const existing = state.desktopAccess || {};
        if (existing.overall_status) {
            applyDesktopAccessSummary({
                ...existing,
                cache_stale: true,
                desktop_access_diagnosis: [
                    ...(Array.isArray(existing.desktop_access_diagnosis) ? existing.desktop_access_diagnosis : []),
                    'Desktop permission refresh did not complete; keeping the last known probe while Aura retries.'
                ].slice(-4)
            });
        }
    } finally {
        clearTimeout(timeoutId);
        state.desktopAccessPollInFlight = false;
    }
}

function scheduleDesktopAccessPoll(delayMs = null) {
    if (state.desktopAccessTimer) clearTimeout(state.desktopAccessTimer);
    state.desktopAccessTimer = null;
    if (!accessCapabilityAllowed('desktop_control')) return;
    const delay = delayMs == null
        ? optionalSurfacePollDelay(DESKTOP_ACCESS_POLL_MS, {
            foregroundFactor: 4,
            hiddenFactor: 8,
        })
        : Math.max(0, Number(delayMs) || 0);
    state.desktopAccessTimer = setTimeout(async () => {
        state.desktopAccessTimer = null;
        if (!document.hidden) await pollDesktopAccess();
        scheduleDesktopAccessPoll();
    }, delay);
}

async function runDesktopAccessAction(action) {
    const normalized = String(action || '').trim().toLowerCase();
    const endpointByAction = {
        'request-screen': '/api/system/desktop-access/request-screen',
        'request-accessibility': '/api/system/desktop-access/request-accessibility',
        'settings-screen': '/api/system/desktop-access/open-settings/screen',
        'settings-accessibility': '/api/system/desktop-access/open-settings/accessibility',
    };
    if (normalized === 'refresh') {
        await pollDesktopAccess();
        return;
    }
    const endpoint = endpointByAction[normalized];
    if (!endpoint) return;
    const actions = DOM.desktopAccessActions || $('desktop-access-actions');
    try {
        if (actions) actions.classList.add('busy');
        const response = await fetch(endpoint, {
            method: 'POST',
            cache: 'no-store',
            credentials: 'same-origin',
            headers: auraDesktopHeaders(),
        });
        let payload = {};
        try {
            payload = await response.json();
        } catch (err) {
            payload = { parse_error: String(err || '') };
        }
        if (!response.ok) {
            console.warn('[DesktopAccess] repair action failed:', normalized, payload);
        }
    } catch (err) {
        console.warn('[DesktopAccess] repair action error:', normalized, err);
    } finally {
        if (actions) actions.classList.remove('busy');
        await new Promise(resolve => setTimeout(resolve, 500));
        await pollDesktopAccess();
    }
}

function applyStateSummary(summary, commitments) {
    const s = summary || {};
    if ($('hud-goals')) $('hud-goals').textContent = String(s.active_goals || 0);
    if ($('hud-beliefs') && s.epistemics) $('hud-beliefs').textContent = String(s.epistemics.total || 0);
    if ($('c-policy-mode')) $('c-policy-mode').textContent = escText(s.policy_mode, 'IDLE').replace(/_/g, ' ').toUpperCase();
    if ($('c-fragmentation')) $('c-fragmentation').textContent = formatPercent01(s.fragmentation_score || 0);
    if ($('c-contradictions')) $('c-contradictions').textContent = String(s.contradiction_count || 0);
    if ($('c-contested')) $('c-contested').textContent = String((s.epistemics && s.epistemics.contested) || 0);
    if ($('c-commitments')) $('c-commitments').textContent = String((commitments && commitments.active_count) || 0);
    if ($('rolling-summary')) $('rolling-summary').textContent = escText(s.rolling_summary, 'Continuity summary pending.');
    if ($('phenomenal-summary')) $('phenomenal-summary').textContent = escText(s.phenomenal_state, 'Operational field offline.');
    if ($('exec-objective') && s.current_objective) $('exec-objective').textContent = s.current_objective;
    if ($('exec-focus') && s.rolling_summary) $('exec-focus').textContent = s.rolling_summary;

    const coherenceEl = $('c-coherence');
    if (coherenceEl && s.coherence_score != null) {
        coherenceEl.textContent = formatPercent01(s.coherence_score, 0);
        coherenceEl.style.color = Number(s.coherence_score) >= 0.8 ? 'var(--success)' : Number(s.coherence_score) >= 0.7 ? 'var(--accent)' : 'var(--warn)';
    }
}

function renderToolCatalog(catalog, catalogHealth = null) {
    const tools = Array.isArray(catalog) ? catalog.slice() : [];
    state.toolCatalog = tools;
    if (catalogHealth && typeof catalogHealth === 'object') state.toolCatalogHealth = catalogHealth;
    const health = state.toolCatalogHealth && typeof state.toolCatalogHealth === 'object'
        ? state.toolCatalogHealth
        : {};
    const preflight = health.execution_preflight && typeof health.execution_preflight === 'object'
        ? health.execution_preflight
        : {};

    const available = tools.filter(tool => !!tool.available);
    const degraded = tools.filter(tool => !tool.available);

    if ($('c-tools-available')) $('c-tools-available').textContent = `${available.length}/${tools.length}`;
    if ($('tool-available-count')) $('tool-available-count').textContent = String(available.length);
    if ($('tool-degraded-count')) $('tool-degraded-count').textContent = String(degraded.length);
    if ($('tool-catalog-state')) $('tool-catalog-state').textContent = health.ready === true ? 'READY' : health.ready === false ? 'BLOCKED' : '--';
    if ($('tool-preflight-state')) {
        $('tool-preflight-state').textContent = preflight.complete === true
            ? (preflight.ok === true ? 'VERIFIED' : 'FAILED')
            : 'NOT RUN';
    }
    if ($('tool-catalog-detail')) {
        const failed = Array.isArray(preflight.failed) ? preflight.failed.filter(Boolean) : [];
        const missing = Array.isArray(health.missing_live) ? health.missing_live.filter(Boolean) : [];
        const quarantined = Array.isArray(health.quarantined) ? health.quarantined.filter(Boolean) : [];
        const quarantinedCount = Math.max(Number(health.quarantined_count || 0), quarantined.length);
        const parts = [
            `source ${health.ready === true ? 'ready' : String(health.reason || 'unverified').replace(/_/g, ' ')}`,
            `execution ${preflight.complete === true ? (preflight.ok === true ? 'verified' : 'failed') : 'not yet verified'}`,
        ];
        if (failed.length) parts.push(`failed: ${failed.slice(0, 4).join(', ')}`);
        if (missing.length) parts.push(`missing: ${missing.slice(0, 4).join(', ')}`);
        if (quarantinedCount) parts.push(`${quarantinedCount} quarantined`);
        $('tool-catalog-detail').textContent = parts.join(' · ');
    }

    const issues = $('tool-catalog-issues');
    if (issues) {
        const failed = Array.isArray(preflight.failed) ? preflight.failed.filter(Boolean) : [];
        const missing = Array.isArray(health.missing_live) ? health.missing_live.filter(Boolean) : [];
        const quarantined = Array.isArray(health.quarantined) ? health.quarantined.filter(Boolean) : [];
        const rows = [];
        if (missing.length) {
            rows.push(`<div class="tool-catalog-issue"><span class="tool-catalog-issue-kind">MISSING LIVE</span><span>${missing.map(escHtml).join(', ')}</span></div>`);
        }
        if (failed.length) {
            rows.push(`<div class="tool-catalog-issue"><span class="tool-catalog-issue-kind">PREFLIGHT FAILED</span><span>${failed.map(escHtml).join(', ')}</span></div>`);
        }
        quarantined.forEach(item => {
            const entry = item && typeof item === 'object' ? item : {};
            const name = entry.name || entry.class_name || entry.catalog_id || 'unknown skill';
            const stage = entry.stage ? ` at ${entry.stage}` : '';
            const error = entry.error ? `: ${entry.error}` : '';
            rows.push(`<div class="tool-catalog-issue"><span class="tool-catalog-issue-kind">QUARANTINED</span><span>${escHtml(name)}${escHtml(stage)}${escHtml(error)}</span></div>`);
        });
        if (!rows.length && health.ready === false && health.reason) {
            rows.push(`<div class="tool-catalog-issue"><span class="tool-catalog-issue-kind">BLOCKED</span><span>${escHtml(String(health.reason).replace(/_/g, ' '))}</span></div>`);
        }
        issues.innerHTML = rows.join('');
        issues.hidden = rows.length === 0;
    }

    const list = $('skills-list');
    if (!list) return;
    if (!tools.length) {
        list.innerHTML = '<div class="mem-empty">No registered tools available.</div>';
        return;
    }

    list.innerHTML = tools.map(tool => {
        const rawState = String(tool.state || (tool.available ? 'READY' : 'DEGRADED'));
        const normalizedState = rawState.toLowerCase();
        const stateValue = !tool.available ? 'error' : normalizedState === 'running' ? 'running' : 'ready';
        const availabilityTone = tool.available ? 'success' : 'error';
        const degradedReason = tool.degraded_reason || tool.last_error || '';
        const detailBits = [
            tool.route_class ? `<span class="badge">${escHtml(String(tool.route_class).replace(/_/g, ' '))}</span>` : '',
            tool.risk_class ? `<span class="badge badge-${tool.risk_class === 'critical' ? 'diagnostic' : tool.risk_class === 'high' ? 'autonomic' : 'reflex'}">${escHtml(tool.risk_class)}</span>` : '',
            tool.availability ? `<span class="badge ${tool.available ? 'badge-reflex' : 'badge-diagnostic'}">${escHtml(tool.availability)}</span>` : '',
            tool.preflight_state ? `<span class="badge ${tool.preflight_state === 'ready' ? 'badge-reflex' : tool.preflight_state === 'failed' ? 'badge-diagnostic' : ''}">preflight ${escHtml(String(tool.preflight_state).replace(/_/g, ' '))}</span>` : ''
        ].filter(Boolean).join('');
        return `
            <div class="skill-card ${stateValue}" id="${toolDomId(tool.catalog_id || tool.name)}">
                <div class="skill-card-head">
                    <div class="skill-title-wrap">
                        <span class="skill-name">${escHtml(tool.name)}</span>
                        <div class="skill-meta-row">${detailBits}</div>
                    </div>
                    <span class="skill-badge ${availabilityTone}">${escHtml(rawState)}</span>
                </div>
                <div class="skill-desc">${escHtml(tool.description || 'No description available.')}</div>
                <div class="skill-detail-line"><strong>Input</strong><span>${escHtml(tool.input_summary || 'contextual')}</span></div>
                <div class="skill-detail-line"><strong>Use</strong><span>${escHtml(tool.example_usage || 'on demand')}</span></div>
                ${degradedReason ? `<div class="skill-warning">${escHtml(degradedReason)}</div>` : ''}
            </div>
        `;
    }).join('');
}

function describeToolEvent(event) {
    if (!event) return 'Tool orchestration channel awaiting events.';
    const deferred = toolEventIsDeferred(event);
    const stage = deferred ? 'deferred' : escText(event.stage, 'idle').replace(/_/g, ' ');
    const tool = escText(event.tool, 'unknown tool');
    const source = escText(event.source, 'system');
    const status = deferred ? 'deferred' : event.success === false ? 'failed' : event.success === true ? 'succeeded' : stage;
    const reason = event.error || (event.decision && event.decision.reason) || '';
    const base = `${tool} · ${status.toUpperCase()} · via ${source}`;
    return reason ? `${base} · ${String(reason).replace(/_/g, ' ')}` : base;
}

function toolEventIsDeferred(event) {
    if (!event || typeof event !== 'object') return false;
    const stage = String(event.stage || '').toLowerCase();
    const rawResult = event.result;
    const result = rawResult && typeof rawResult === 'object' ? rawResult : {};
    const resultStatus = String(result.status || '').toLowerCase();
    const deferralSignals = [result.reason, result.error, event.error];
    if (typeof rawResult === 'string') deferralSignals.push(rawResult);
    return stage === 'deferred' || resultStatus === 'deferred' || deferralSignals.some(
        value => String(value || '').toLowerCase().startsWith('background_deferred:')
    );
}

function applyToolEvent(event) {
    state.lastToolEvent = event;
    const deferred = toolEventIsDeferred(event);
    const rawStage = deferred ? 'deferred' : escText(event && event.stage, 'idle');
    const stage = rawStage.replace(/_/g, ' ').toUpperCase();
    const stageEl = $('tool-last-stage');
    if (stageEl) {
        stageEl.textContent = stage;
        stageEl.style.color = !deferred && event && event.success === false ? 'var(--error)' : ['rejected', 'degraded'].includes(String(event && event.stage)) ? 'var(--warn)' : 'var(--accent)';
    }
    const detailEl = $('tool-last-detail');
    if (detailEl) detailEl.textContent = describeToolEvent(event);

    if (event && event.tool) {
        const stageForCard =
            deferred ? 'ready' :
            event.stage === 'started' ? 'running' :
            event.stage === 'completed' && event.success !== false ? 'ready' :
            ['failed', 'rejected', 'degraded'].includes(String(event.stage)) ? 'error' :
            '';
        if (stageForCard) updateSkillUI(event.tool, stageForCard);

        // Update the typing indicator label to show tool activity
        if (event.stage === 'started') {
            const action = formatToolAction(event.tool);
            updateTypingLabel(`Aura is ${action}…`);
            const typingInd = $('typing-ind');
            if (typingInd) typingInd.classList.add('show');
            setChatPanelState('thinking');
        } else if (deferred || ['completed', 'failed', 'rejected', 'degraded'].includes(String(event.stage))) {
            updateTypingLabel('Aura is thinking…');
        }
    }

    if (!deferred && event && ['failed', 'rejected', 'degraded'].includes(String(event.stage))) {
        queueThought({
            level: event.stage === 'failed' ? 'ERROR' : 'WARNING',
            name: 'TOOL',
            message: describeToolEvent(event),
            timestamp: event.timestamp || Date.now() / 1000
        });
    }
}

function applyBootstrapPayload(payload, { hydrateConversationHistory = false } = {}) {
    if (!payload || typeof payload !== 'object') return;
    applyAccessProfile(payload.access);
    state.bootstrapLoaded = true;
    const runtimeHealthy = payloadRuntimeHealthy(payload);
    state.runtimeHealthy = runtimeHealthy;
    state.runtimeHealthBlockers = runtimeHealthBlockers(payload);
    if (payload.identity && payload.identity.version) {
        state.version = payload.identity.version;
        state.identityName = payload.identity.name || state.identityName;
        // The header already renders the name in .brand-title, so the chip beside
        // it shows only the build ("v2026.4.20-Zenith"), never "Aura Luna Aura Luna v…".
        const buildLabel = compactBuildLabel(payload.identity.build || payload.identity.version);
        if ($('ui-ver')) $('ui-ver').textContent = buildLabel;
        if ($('setting-version')) $('setting-version').textContent = payload.identity.version;
    }

    applyStateSummary(payload.state, payload.commitments);
    renderToolCatalog(payload.tools || [], payload.skill_catalog || null);
    applyVoiceSummary(payload.voice || {});
    applyDesktopAccessSummary(payload.desktop_access || {});
    renderStatusFlags(payload.ui && payload.ui.status_flags);
    if (payload.interaction_signals) {
        state.interactionSignals = payload.interaction_signals;
    }

    if (payload.executive) {
        const ex = payload.executive;
        if ($('exec-authority')) $('exec-authority').textContent = `${escText(ex.last_action, 'idle').toUpperCase()} · ${escText(ex.last_reason, 'steady').replace(/_/g, ' ')}`;
    }

    if (!state.lastToolEvent && $('tool-last-detail')) {
        $('tool-last-detail').textContent = payload.tools && payload.tools.length
            ? `${payload.tools.filter(tool => tool.available).length}/${payload.tools.length} tools currently available.`
            : 'Tool orchestration channel awaiting events.';
    }

    if (payload.commitments) {
        state.commitments = payload.commitments;
        if ($('c-commitments')) $('c-commitments').textContent = String(payload.commitments.active_count || 0);
    }

    if (payload.telemetry && payload.telemetry.runtime) {
        updateTelemetry(payload.telemetry.runtime);
    }
    if (payload.conversation && payload.conversation.lane) {
        applyConversationLane(payload.conversation.lane, payload.telemetry && payload.telemetry.boot ? payload.telemetry.boot.status : '');
    }

    const flags = (payload.ui && payload.ui.status_flags) || [];
    const lane = payload.conversation && payload.conversation.lane;
    const laneNotReady = lane && lane.conversation_ready === false && !laneHasActiveGeneration(lane);
    const connectionMode = flags.includes('booting')
        ? 'booting'
        : !runtimeHealthy
            ? 'degraded'
        : laneNotReady
            ? 'degraded'
        : flags.some(flag => ['thermal_guard', 'coherence_low', 'fragmentation_high', 'contradictions_present', 'beliefs_contested', 'tool_unavailable', 'executive_hold'].includes(flag))
            ? 'degraded'
        : (payload.session && payload.session.connected) ? 'online' : 'offline';
    setConnectionVisual(connectionMode, !runtimeHealthy ? runtimeHealthStatusText(payload) : laneNotReady ? conversationLaneStatusText(lane) : '');
    syncSplashState(payload);

    if (hydrateConversationHistory && payload.conversation && Array.isArray(payload.conversation.recent)) {
        hydrateRecentConversation(payload.conversation.recent);
    }
    startProfileBoundFeatures();
}

// Defaults to restoring the transcript. It used to default to false, so the
// 30s bootstrap poll — the one thing that runs after the pane has been
// cleared — was the one caller that could never put the conversation back.
// hydrateRecentConversation only writes into an empty or placeholder pane, so
// asking for it always is idempotent and never clobbers a live conversation.
async function hydrateBootstrap({ hydrateConversationHistory = true, quiet = true } = {}) {
    try {
        const res = await fetch('/api/ui/bootstrap', { cache: 'no-store' });
        if (!res.ok) throw new Error(`bootstrap_http_${res.status}`);
        const payload = await res.json();
        applyBootstrapPayload(payload, { hydrateConversationHistory });
        return payload;
    } catch (err) {
        if (!quiet) console.warn('[UI] Bootstrap hydration failed:', err);
        return null;
    }
}

function scheduleBootstrapPoll(delayMs = null) {
    if (state.bootstrapTimer) clearTimeout(state.bootstrapTimer);
    const delay = delayMs == null
        ? optionalSurfacePollDelay(BOOTSTRAP_POLL_MS, {
            foregroundFactor: 3,
            hiddenFactor: 6,
        })
        : Math.max(0, Number(delayMs) || 0);
    state.bootstrapTimer = setTimeout(async () => {
        state.bootstrapTimer = null;
        if (!document.hidden) await hydrateBootstrap({ quiet: true });
        scheduleBootstrapPoll();
    }, delay);
}

// ── Tab switching ────────────────────────────────────────
// Use event delegation for reliable click handling
document.addEventListener('click', (e) => {
    const btn = e.target.closest('.tab-btn');
    if (!btn) return;
    
    e.preventDefault();
    e.stopPropagation();
    
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    const pane = $(`pane-${tab}`);
    if (pane) {
        pane.classList.add('active');
        state.activeTab = tab;
        if (tab === 'telemetry' && !state.beliefGraphInit) initBeliefGraph();
        if (tab === 'skills') { loadSkills(); loadLearningStatus(); }
        if (tab === 'memory') loadMemory(state.activeMem);
        if (tab === 'imagine') imagination.activate();
        else imagination.deactivate();
    } else {
        console.warn(`Pane not found for tab: ${tab}`);
    }
}, true);

// ── Mobile Tab switching ──────────────────────────────────
// Use event delegation for mobile nav buttons
document.addEventListener('click', (e) => {
    const btn = e.target.closest('.m-nav-btn');
    if (!btn) return;
    
    e.preventDefault();
    e.stopPropagation();
    
    const mTab = btn.dataset.mTab;
    if (!mTab) return;

    // Remove active state from all mobile buttons
    document.querySelectorAll('.m-nav-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const chatPanel = document.querySelector('.chat-panel');
    const sidebar = document.querySelector('.sidebar');

    if (mTab === 'chat') {
        chatPanel.classList.add('mobile-active');
        sidebar.classList.remove('mobile-active');
    } else {
        // Switch to any sidebar tab (Neural, Telemetry, etc.)
        chatPanel.classList.remove('mobile-active');
        sidebar.classList.add('mobile-active');

        // Trigger the desktop tab logic to show the right pane
        const desktopTabBtn = document.querySelector(`.tab-btn[data-tab="${mTab}"]`);
        if (desktopTabBtn) desktopTabBtn.click();
    }
}, true);

function syncResponsiveConversationSurface() {
    if (window.innerWidth > 1100) return;
    const chatPanel = document.querySelector('.chat-panel');
    const sidebar = document.querySelector('.sidebar');
    if (!chatPanel || !sidebar) return;
    if (!chatPanel.classList.contains('mobile-active') && !sidebar.classList.contains('mobile-active')) {
        chatPanel.classList.add('mobile-active');
    }
}

// Initial mobile state and desktop-window resize: never leave both primary
// surfaces hidden after crossing the responsive breakpoint.
syncResponsiveConversationSurface();
window.addEventListener('resize', syncResponsiveConversationSurface, { passive: true });

// ── Memory sub-tabs ──────────────────────────────────────
// Use event delegation for reliable click handling
document.addEventListener('click', (e) => {
    const btn = e.target.closest('.mem-sub-btn');
    if (!btn) return;
    
    e.preventDefault();
    e.stopPropagation();
    
    document.querySelectorAll('.mem-sub-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.activeMem = btn.dataset.mem;
    loadMemory(state.activeMem);
}, true);

// ── WebSocket ────────────────────────────────────────────
function connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Force IPv4 for local testing to avoid ::1 resolution issues with Uvicorn
    const hostname = location.hostname === 'localhost' ? '127.0.0.1' : location.hostname;
    const port = location.port ? ':' + location.port : '';
    if (state.reconnectTimer) {
        clearTimeout(state.reconnectTimer);
        state.reconnectTimer = null;
    }
    const ws = new WebSocket(`${proto}//${hostname}${port}/ws`);
    state.ws = ws;

    state.lastPong = Date.now();

    // Application-layer heartbeat to prevent silent disconnects
    if (state.pingInterval) clearInterval(state.pingInterval);
    state.pingInterval = setInterval(() => {
        if (state.ws && state.ws === ws && state.ws.readyState === WebSocket.OPEN) {
            const hiddenOrSuspended = !!(document.hidden || state.surfaceSuspended);
            const staleSocketMs = hiddenOrSuspended ? 5 * 60 * 1000 : 35000;
            // Force close if we haven't received a pong. Hidden/screen-saver
            // intervals get a wider window because browsers pause timers while
            // Aura's backend keeps running.
            if (Date.now() - state.lastPong > staleSocketMs) {
                console.warn('[WS] Heartbeat timeout, forcing close for recovery');
                state.ws.close();
                return;
            }
            state.ws.send(JSON.stringify({ type: 'ping' }));
        }
    }, 25000);

    ws.onopen = () => {
        if (state.ws !== ws) return;
        state.lastPong = Date.now();
        const wasDisconnected = !state.connected;
        const hadRetried = (state.retryCount || 0) > 0;
        state.connected = true;
        state.runtimeHealthy = false;
        state.runtimeHealthBlockers = ['runtime_health_unverified'];
        state.retryCount = 0;
        showConnToast(false); // Hide disconnection toast
        if (wasDisconnected && hadRetried) {
            showConnToast('reconnected'); // Show brief reconnected confirmation
        }
        setConnectionVisual('reconnecting', 'Checking on Aura');
        pollHealth();
        hydrateBootstrap({ hydrateConversationHistory: !state.bootstrapLoaded, quiet: true });

        // ZENITH: Flush pending messages
        if (state.pendingOutboundMessages.length > 0) {
            console.debug(`[WS] Flushing ${state.pendingOutboundMessages.length} pending messages`);
            while (state.pendingOutboundMessages.length > 0) {
                const msg = state.pendingOutboundMessages.shift();
                state.ws.send(JSON.stringify(msg));
            }
        }
    };

    ws.onmessage = e => {
        if (state.ws !== ws) return;
        try {
            const data = JSON.parse(e.data);
            if (!state.connected) state.connected = true;
            markLiveSurfaceResponsive('websocket_message');
            const toast = $('conn-toast');
            if (toast && toast.classList.contains('show') && /connection lost|reconnecting/i.test(toast.textContent || '')) {
                showConnToast(false);
            }
            handleWsEvent(data);
        } catch (err) {
            console.error('[WS] Failed to parse WebSocket message:', err);
        }
    };

    ws.onclose = () => {
        if (state.ws !== ws) return;
        state.connected = false;
        if (state.pingInterval) clearInterval(state.pingInterval);
        const surfacePaused = !!(document.hidden || state.surfaceSuspended || state.resumeInProgress || navigator.onLine === false);
        showConnToast(surfacePaused ? 'paused' : true);
        setConnectionVisual('reconnecting', surfacePaused ? 'Paused in background' : '');

        // ZENITH: Infinite Reconnect with Exponential Backoff + Jitter
        if (!state.retryCount) state.retryCount = 0;
        state.retryCount++;
        const baseDelay = state.resumeInProgress ? 250 : Math.min(30000, 1000 * Math.pow(2, state.retryCount));
        const jitter = Math.random() * 500;
        const delay = baseDelay + jitter;
        console.warn(`[WS] Connection closed. Retrying in ${(delay/1000).toFixed(1)}s (Attempt ${state.retryCount})`);
        state.reconnectTimer = setTimeout(connect, delay);
    };

    ws.onerror = (err) => {
        if (state.ws !== ws) return;
        console.error('[WS] WebSocket error:', err);
        // Force close to trigger onclose reconnection logic
        if (ws.readyState !== WebSocket.CLOSED) {
            ws.close();
        }
    };
}

function reconnectLiveSurface(reason = 'resume') {
    state.surfaceSuspended = false;
    state.resumeInProgress = true;
    state.lastSurfaceResumeAt = Date.now();
    showConnToast('resuming');
    setConnectionVisual('reconnecting', 'Waking this window');
    hydrateBootstrap({ hydrateConversationHistory: !state.bootstrapLoaded, quiet: true });
    if (!state.healthPollInFlight) scheduleHealthPoll(0);

    if (state.reconnectTimer) {
        clearTimeout(state.reconnectTimer);
        state.reconnectTimer = null;
    }

    const ws = state.ws;
    if (!ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
        connect();
    } else if (ws.readyState === WebSocket.OPEN) {
        try {
            state.lastPong = Date.now();
            ws.send(JSON.stringify({ type: 'ping', reason }));
        } catch (err) {
            console.warn('[WS] Resume ping failed, reconnecting:', err);
            try {
                ws.close();
            } catch (closeErr) {
                console.warn('[WS] Failed to close stale socket before reconnect:', closeErr);
            }
            connect();
        }
    }

    setTimeout(() => {
        state.resumeInProgress = false;
        if (state.connected) showConnToast(false);
    }, 10000);
}

function markLiveSurfaceResponsive(reason = 'activity') {
    const wasResuming = !!state.resumeInProgress;
    state.surfaceSuspended = false;
    state.resumeInProgress = false;
    if (state.connected && wasResuming) {
        showConnToast(false);
        setConnectionVisual(state.runtimeHealthy ? 'online' : 'degraded', state.runtimeHealthy ? '' : runtimeHealthStatusText());
    }
}

function pauseLiveSurface(reason = 'hidden') {
    state.surfaceSuspended = true;
    state.lastSurfaceHiddenAt = Date.now();
    setConnectionVisual('reconnecting', 'Paused in background');
    showConnToast('paused');
}

// ── Voice Output (SSE Player) ────────────────────────────
class VoiceStreamPlayer {
    constructor() {
        this.ctx = null;
        this.evtSource = null;
        this.startTime = 0;
    }

    async init() {
        if (!accessCapabilityAllowed('voice_stream')) return;
        if (this.evtSource) return;
        // VoiceStreamPlayer init
        this.evtSource = new EventSource('/api/stream/voice');
        this.evtSource.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);
                if (data.type === 'audio' && data.data) {
                    this.playPCM(data.data);
                }
            } catch (err) {
                console.error('[VoiceStream] Failed to parse audio event:', err);
            }
        };
    }

    async getCtx() {
       if (!this.ctx) {
           this.ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
       }
       if (this.ctx.state === 'suspended') await this.ctx.resume();
       return this.ctx;
    }

    async playPCM(base64Data) {
        const ctx = await this.getCtx();
        const binary = atob(base64Data);
        const bytes = new Int16Array(binary.length / 2);
        for (let i = 0; i < bytes.length; i++) {
            bytes[i] = (binary.charCodeAt(i*2) & 0xFF) | (binary.charCodeAt(i*2+1) << 8);
        }

        const floatData = new Float32Array(bytes.length);
        for (let i = 0; i < bytes.length; i++) {
            floatData[i] = bytes[i] / 32768.0;
        }

        const buffer = ctx.createBuffer(1, floatData.length, 16000);
        buffer.getChannelData(0).set(floatData);

        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(ctx.destination);

        // Scheduling for seamless playback
        if (this.startTime < ctx.currentTime) {
            this.startTime = ctx.currentTime + 0.05;
        }
        source.start(this.startTime);
        this.startTime += buffer.duration;

        // Visual feedback
        triggerVoiceOrb('speaking');
    }
}
const voicePlayer = new VoiceStreamPlayer();

/**
 * Play media in the chat, rather than handing the conversation off.
 *
 * The current state of the art for "play X" is a hand-off: a card that opens
 * a streaming app, or a link to a new tab. The conversation stops and
 * something else starts. Aura is running on the machine that holds the file,
 * so there is no reason for the music to happen anywhere but here.
 *
 * The card is deliberately quiet — a title, where it came from, and the
 * browser's own transport controls. Native controls rather than drawn ones:
 * they are keyboard accessible, they are what the platform's assistive
 * technology already knows how to drive, and a hand-rolled scrubber is a
 * large amount of code whose only achievement is being less good at this.
 */
function appendMediaMessage(media, metadata = {}) {
    const messages = DOM.messages || $('messages');
    if (!messages || !media || !media.id) return;

    const div = document.createElement('div');
    div.className = 'msg aura';

    const avatar = document.createElement('div');
    avatar.className = 'aura-avatar';
    div.appendChild(avatar);

    const content = document.createElement('div');
    content.className = 'msg-content media-card';

    const isVideo = media.kind === 'video';
    const player = document.createElement(isVideo ? 'video' : 'audio');
    player.controls = true;
    player.preload = 'metadata';
    player.className = isVideo ? 'media-video' : 'media-audio';
    // Same-origin, id-addressed. The endpoint resolves it through the media
    // index, which is the allowlist — there is no path here to sanitise.
    player.src = `/api/media/stream/${encodeURIComponent(media.id)}`;

    const title = document.createElement('div');
    title.className = 'media-title';
    title.textContent = media.title || 'Untitled';

    const where = document.createElement('div');
    where.className = 'media-where';
    where.textContent = media.folder ? `${media.folder} · on this machine` : 'on this machine';

    content.appendChild(title);
    content.appendChild(where);
    content.appendChild(player);

    if (media.broadly_playable === false) {
        // Findable but possibly not decodable here. Saying so beats showing a
        // player that silently does nothing, which reads as a broken file.
        const note = document.createElement('div');
        note.className = 'media-note';
        note.textContent = 'This format only plays in some browsers — it may not decode here.';
        content.appendChild(note);
    }

    // The element is the authority on whether it can actually play the file.
    // An extension check is a guess; a decode error is a fact.
    player.addEventListener('error', () => {
        const failed = document.createElement('div');
        failed.className = 'media-note media-note-error';
        failed.textContent = 'Your browser could not decode this file.';
        content.appendChild(failed);
        player.remove();
    }, { once: true });

    div.appendChild(content);
    messages.appendChild(div);
    pruneVisibleMessages(messages);
    if (!state.userScrolledUp) messages.scrollTop = messages.scrollHeight;
}

function appendGeneratedImageMessage(imageUrl, metadata = {}) {
    const safeUrl = safeDisplayUrl(imageUrl, { imageOnly: true });
    if (!safeUrl) {
        appendMsg(
            'aura',
            'An image action returned an unsafe or unreadable URL, so the desktop UI refused to render it.',
            false,
            { diagnostic: true }
        );
        return;
    }

    const messages = DOM.messages || $('messages');
    if (!messages) return;

    const div = document.createElement('div');
    div.className = 'msg aura';

    const avatar = document.createElement('div');
    avatar.className = 'aura-avatar';
    div.appendChild(avatar);

    const badge = document.createElement('span');
    if (metadata.diagnostic) {
        badge.className = 'aura-badge diagnostic';
        badge.textContent = 'Diagnostic';
    } else if (metadata.reflex) {
        badge.className = 'aura-badge reflex';
        badge.textContent = 'Reflex';
    } else if (metadata.autonomic) {
        badge.className = 'aura-badge autonomic';
        badge.textContent = 'Autonomic';
    }
    if (badge.className) div.appendChild(badge);

    const wrap = document.createElement('div');
    wrap.className = 'gen-image-wrap';

    const loading = document.createElement('div');
    loading.className = 'gen-image-loading';
    loading.textContent = 'Manifesting visualization...';
    wrap.appendChild(loading);

    const image = document.createElement('img');
    image.src = safeUrl;
    image.alt = 'Generated Image';
    image.className = 'gen-image';
    image.addEventListener('load', () => {
        loading.style.display = 'none';
    });
    image.addEventListener('error', () => {
        const retries = Number(image.dataset.retryCount || 0);
        if (retries >= 1) {
            loading.textContent = 'Image failed to load.';
            return;
        }
        image.dataset.retryCount = String(retries + 1);
        loading.textContent = 'Image loading... please wait';
        const retryUrl = new URL(safeUrl, window.location.origin);
        if (!['http:', 'https:'].includes(retryUrl.protocol)) {
            loading.textContent = 'Image failed to load.';
            return;
        }
        retryUrl.searchParams.set('retry', String(Date.now()));
        window.setTimeout(() => {
            image.src = retryUrl.href;
        }, 5000);
    });
    image.addEventListener('click', () => {
        window.open(safeUrl, '_blank', 'noopener,noreferrer');
    });
    wrap.appendChild(image);

    const saveBtn = document.createElement('button');
    saveBtn.className = 'gen-save-btn';
    saveBtn.type = 'button';
    saveBtn.textContent = 'MANIFEST TO DESKTOP';
    saveBtn.addEventListener('click', () => saveImageToDevice(safeUrl));
    wrap.appendChild(saveBtn);

    div.appendChild(wrap);
    messages.appendChild(div);
    pruneVisibleMessages(messages);
    if (!state.userScrolledUp) messages.scrollTop = messages.scrollHeight;
}

function formatToolAction(toolName) {
    const TOOL_LABELS = {
        'search_web': 'searching the web',
        'soveraing_browser': 'searching the web',
        'sovereign_browser': 'searching the web',
        'web_search': 'searching the web',
        'file_operation': 'managing files',
        'file_write': 'writing a file',
        'file_read': 'reading a file',
        'file_exists_check': 'checking file existence',
        'terminal_exec': 'executing a terminal command',
        'sovereign_terminal': 'executing system commands',
        'sovereign_vision': 'analyzing the screen',
        'speak': 'preparing speech',
        'speak_aloud': 'speaking aloud',
        'sovereign_imagination': 'generating an image',
        'self_repair': 'analyzing system diagnostics',
        'self_evolution': 'applying self-modifications',
        'rsi_optimization': 'optimizing runtime parameters',
        'inter_agent_comm': 'coordinating with subagent',
        'inter_agent': 'coordinating with subagent',
        'clock_check': 'checking the clock',
        'clock': 'checking the clock',
        'network_scan': 'scanning the network',
        'system_proprioception': 'checking system resources',
        'proprioception': 'checking system resources',
        'manifest_to_device': 'syncing assets to device',
        'manifest_asset': 'syncing assets to device',
        'memory_ops': 'recalling memories',
        'memory_remember': 'storing memory',
        'force_dream_cycle': 'running a dream cycle',
        'dream_cycle': 'running a dream cycle',
        'malware_analysis': 'scanning for malware',
        'malware_scan': 'scanning for malware',
        'train_self': 'training cognitive weights',
        'personality_skill': 'introspecting personality profile',
        'personality_introspect': 'introspecting personality profile',
        'environment_info': 'retrieving environment info',
        'environment_check': 'retrieving environment info',
        'listen': 'activating listening mode',
        'listen_activate': 'activating listening mode',
        'voice_mute': 'muting voice',
        'voice_unmute': 'unmuting voice',
        'voice_stop_tts': 'stopping text-to-speech',
        'internal_sandbox': 'executing sandbox code',
        'sandbox_execute': 'executing sandbox code',
        'social_lurker': 'monitoring social channels',
        'social_lurk': 'monitoring social channels',
        'curiosity': 'exploring curious pathways',
        'curiosity_suggest': 'exploring curious pathways',
        'spawn_agent': 'spawning autonomous subagent',
        'spawn_agents_parallel': 'spawning parallel subagents',
        'spawn_parallel': 'spawning parallel subagents'
    };

    if (TOOL_LABELS[toolName]) {
        return TOOL_LABELS[toolName];
    }

    const VERB_GERUNDS = {
        'open': 'opening',
        'close': 'closing',
        'write': 'writing',
        'read': 'reading',
        'create': 'creating',
        'delete': 'deleting',
        'remove': 'removing',
        'add': 'adding',
        'update': 'updating',
        'modify': 'modifying',
        'check': 'checking',
        'verify': 'verifying',
        'find': 'finding',
        'search': 'searching',
        'seek': 'seeking',
        'scan': 'scanning',
        'analyze': 'analyzing',
        'run': 'running',
        'exec': 'executing',
        'execute': 'executing',
        'launch': 'launching',
        'start': 'starting',
        'stop': 'stopping',
        'kill': 'killing',
        'manage': 'managing',
        'save': 'saving',
        'load': 'loading',
        'fetch': 'fetching',
        'get': 'getting',
        'set': 'setting',
        'send': 'sending',
        'receive': 'receiving',
        'export': 'exporting',
        'import': 'importing',
        'sync': 'syncing',
        'train': 'training',
        'optimize': 'optimizing',
        'evaluate': 'evaluating',
        'eval': 'evaluating',
        'spawn': 'spawning',
        'dream': 'dreaming',
        'listen': 'listening',
        'speak': 'speaking',
        'talk': 'talking',
        'mute': 'muting',
        'unmute': 'unmuting',
        'play': 'playing',
        'remember': 'remembering',
        'recall': 'recalling',
        'repair': 'repairing',
        'evolve': 'evolving',
        'audit': 'auditing',
        'propriocept': 'introspecting',
        'introspect': 'introspecting',
        'browse': 'browsing',
        'scrape': 'scraping',
        'click': 'clicking',
        'type': 'typing',
        'press': 'pressing',
        'move': 'moving',
        'scroll': 'scrolling',
        'take': 'taking',
        'capture': 'capturing',
        'screenshot': 'screenshotting',
        'make': 'making',
        'generate': 'generating',
        'produce': 'producing',
        'compile': 'compiling',
        'build': 'building',
        'install': 'installing',
        'configure': 'configuring',
        'show': 'showing',
        'hide': 'hiding',
        'display': 'displaying',
        'view': 'viewing',
        'preview': 'previewing',
        'render': 'rendering',
        'clear': 'clearing',
        'reset': 'resetting',
        'restart': 'restarting',
        'reboot': 'rebooting'
    };

    const words = toolName.split(/[-_]+/);
    let verbIndex = -1;
    for (let i = 0; i < words.length; i++) {
        const word = words[i].toLowerCase();
        if (VERB_GERUNDS[word]) {
            verbIndex = i;
            break;
        }
    }

    if (verbIndex !== -1) {
        const verb = words[verbIndex].toLowerCase();
        const gerund = VERB_GERUNDS[verb];
        const remaining = [...words.slice(0, verbIndex), ...words.slice(verbIndex + 1)].join(' ');
        return remaining ? `${gerund} ${remaining}` : gerund;
    }

    const clean = toolName.replace(/_/g, ' ').replace(/-/g, ' ').trim().toLowerCase();
    return `running ${clean}`;
}

function setChatPanelState(panelState) {
    const chatPanel = document.querySelector('.chat-panel');
    if (!chatPanel) return;
    if (panelState === 'thinking') {
        chatPanel.classList.add('thinking');
        chatPanel.classList.remove('generating');
    } else if (panelState === 'generating') {
        chatPanel.classList.add('generating');
        chatPanel.classList.remove('thinking');
    } else {
        chatPanel.classList.remove('thinking', 'generating');
    }
}

function handleWsEvent(data) {
    const type = data.kind || data.type;
    if (!['chat_stream_chunk', 'heartbeat', 'ping', 'pong'].includes(type)) {
        if (rememberEventId(data.event_id || data.id)) return;
    }

    if (type === 'log' || type === 'thought') {
        queueThought(data);
        triggerVoiceOrb('thinking');
    } else if (type === 'telemetry') {
        updateTelemetry(data);
    } else if (type === 'tool_event') {
        applyToolEvent(data);
    } else if (type === 'chat_stream_start') {
        startStreamMsg('aura');
        $('typing-ind').classList.remove('show');
        setChatPanelState('generating');
        triggerVoiceOrb('speaking');
    } else if (type === 'chat_stream_chunk') {
        appendStreamChunk(data.chunk);
    } else if (type === 'chat_stream_end') {
        // The stream carries the confidence when the server knows it; the HTTP
        // response marks it otherwise (see the alreadyStreamed branch).
        finishStreamMsg(data.response_confidence);
        $('typing-ind').classList.remove('show');
        setChatPanelState('idle');
    } else if (type === 'camera_capture_request') {
        void captureFrameForAura(data);
    } else if (type === 'camera_privacy') {
        // She operated her own camera control. The visible control has to move
        // with it, or the UI shows one state over hardware in another — the
        // worst possible split for a camera, and the one that destroys trust
        // in the indicator permanently.
        const wanted = !!data.enabled;
        state.cameraSignalWanted = wanted;
        // The health poll would repaint this from the server a moment later
        // anyway; the lock stops it fighting the change in the meantime.
        state._privacyLockUntil = Date.now() + 3000;
        const camBtn = $('btn-cam');
        if (camBtn) {
            camBtn.classList.toggle('disabled', !wanted);
            camBtn.innerHTML = wanted ? '<span>● CAM</span>' : '<span>● CAM OFF</span>';
        }
        if (wanted) { void startCameraSignals(); } else { stopCameraSignals(); }
    } else if (type === 'status') {
        if (data.narrative) $('narrative').textContent = data.narrative;
    } else if (type === 'activity') {
        updateTypingLabel(data.label || 'Aura is thinking…');
        if (data.show === false) {
            $('typing-ind').classList.remove('show');
            setChatPanelState('idle');
        } else {
            $('typing-ind').classList.add('show');
            setChatPanelState('thinking');
        }
    } else if (type === 'action_result') {
        const { tool, result, metadata } = data;
        const isAutonomic = metadata && metadata.autonomic;

        // Phase 36: Check for image display at both levels (result and data)
        const displayType = (result && result.display_type) || data.display_type;
        const imageUrl = (result && result.url) || data.url;

        const mediaItem = (result && result.media) || data.media;

        if (displayType === 'image' && imageUrl) {
            appendGeneratedImageMessage(imageUrl, { autonomic: isAutonomic });
            $('typing-ind').classList.remove('show');
            setChatPanelState('idle');
        } else if (mediaItem) {
            appendMediaMessage(mediaItem, { autonomic: isAutonomic });
            $('typing-ind').classList.remove('show');
            setChatPanelState('idle');
        } else if (result) {
            // Non-image action results — show the message if available
            const msg = result.message || `Completed ${tool || 'action'}.`;
            appendMsg('aura', msg, false, { autonomic: isAutonomic });
            $('typing-ind').classList.remove('show');
            setChatPanelState('idle');
        }
    } else if (type === 'aura_message' || type === 'chat_response') {
        const msg = data.message || data.content;
        const meta = data.metadata || {};
        if (msg && msg.trim()) {
            // ZENITH: Content-based deduplication.
            // Use content-only fingerprint — the same response can arrive
            // via HTTP and via WebSocket with different IDs.
            const fingerprint = msg.trim().substring(0, 200);
            if (rememberMessageFingerprint(fingerprint)) {
                // duplicate message skipped (same content via different channel)
                return;
            }

            const role = meta && meta.system ? 'system' : 'aura';
            appendMsg(role, msg, false, meta);
            $('typing-ind').classList.remove('show');
            setChatPanelState('idle');
            if (role === 'aura') triggerVoiceOrb('speaking');
        }
    } else if (type === 'skill_status') {
        updateSkillUI(data.skill, data.state);
    } else if (type === 'model_failover') {
        const from = data.from || 'Current Brain';
        const error = data.error || 'stalled';
        appendMsg('aura', `⚠️ _Shift in cognitive processing: ${from} was unresponsive. Switching to a different neural pathway (${error})._`, false, { diagnostic: true });
    } else if (type === 'heartbeat') {
        state.lastPong = Date.now();
        applyRuntimeHeartbeat(data);
    } else if (type === 'pong') {
        state.lastPong = Date.now();
        applyRuntimeHeartbeat(data);
    }
}

let orbTimeout;
const triggerVoiceOrb = (type) => {
    const wrap = $('voice-orb-wrap');
    const orb = $('voice-orb');
    if (!wrap || !orb) return;

    // Only show the orb wrap when voice mode is explicitly active
    if (state.voiceActive) {
        wrap.classList.add('active');
        wrap.style.opacity = '1';
    }

    // Standardize classes (remove old states)
    orb.classList.remove('listening', 'thinking', 'speaking');

    if (state.voiceActive) {
        if (type === 'thinking') {
            orb.classList.add('thinking');
        } else if (type === 'speaking') {
            orb.classList.add('speaking');
        } else {
            orb.classList.add('listening');
        }
    }
    // When voice is off, orb stays hidden — no flash on every message
};
function queueThought(data) {
    const item = normalizeThoughtEvent(data);
    if (!item) return;
    if (item.name !== 'Aura.Live.Neural') {
        state.lastSemanticThoughtAt = Date.now();
    }
    item.repeatCount = Math.max(1, Number(item.repeatCount || 1));
    item.fingerprint = buildThoughtFingerprint(item);
    if (coalesceThoughtQueueItem(item)) return;

    if (state.thoughtQueue.length >= THOUGHT_QUEUE_MAX) {
        state.thoughtQueue.splice(0, state.thoughtQueue.length - THOUGHT_QUEUE_MAX + 1);
    }
    state.thoughtQueue.push(item);
    syncNeuralFeedMode();
    if (!state.pacingActive && !state.neuralFeedPaused) processThoughtQueue();
}

function queueNeuralLivenessCard(message, { level = 'info', source = 'Aura.Live.Neural', force = false, fullMessage = '' } = {}) {
    const now = Date.now();
    if (!force && now - Number(state.lastNeuralPulseAt || 0) < NEURAL_LIVENESS_PULSE_MS) return;
    state.lastNeuralPulseAt = now;
    queueThought({
        type: 'thought',
        kind: 'thought',
        name: source,
        level,
        message,
        // The card already renders a detail toggle off fullMessage, so the
        // technical line survives one click away instead of being the
        // headline.
        fullMessage: fullMessage || message,
        timestamp: now / 1000,
        event_id: `neural_liveness_${now}_${Math.random().toString(36).slice(2, 8)}`
    });
}

function healthPulseFingerprint(payload) {
    const lane = payload && payload.conversation_lane ? payload.conversation_lane : {};
    const boot = payload && payload.boot ? payload.boot : {};
    const blockers = runtimeHealthBlockers(payload || {}).slice(0, 8).join(',');
    const integrity = payload && payload.integrity ? payload.integrity : {};
    const proofBlockers = Array.isArray(integrity.proof_blockers) ? integrity.proof_blockers : integrity.blockers;
    const integrityBlockers = Array.isArray(proofBlockers) ? proofBlockers.slice(0, 4).join(',') : '';
    const blockerList = blockers ? blockers.split(',') : [];
    const conversationReady = conversationPayloadReady(payload, blockerList);
    const conversationBusy = conversationPayloadBusy(payload, blockerList);
    return [
        payload && payload.status,
        payload && payload.healthy === true ? 'healthy' : 'unhealthy',
        payload && payload.proof_readiness_healthy === false ? 'proof_degraded' : 'proof_ready',
        integrityBlockers,
        boot.boot_phase || boot.status || '',
        lane.state || '',
        conversationReady ? 'conversation_ready' : conversationBusy ? 'conversation_busy' : 'conversation_not_ready',
        blockers
    ].join('|');
}

function publishHealthNeuralPulse(payload, source = 'health_poll') {
    if (!payload || typeof payload !== 'object') return;
    const fingerprint = healthPulseFingerprint(payload);
    const changed = fingerprint !== state.lastHealthSnapshotFingerprint;
    const now = Date.now();
    const blockers = runtimeHealthBlockers(payload);
    const strictHealthy = payloadRuntimeHealthy(payload) && blockers.length === 0;
    const interval = strictHealthy ? NEURAL_LIVENESS_PULSE_MS : HEALTH_POLL_REMINDER_MS;
    const staleFeed = now - Math.max(Number(state.lastSemanticThoughtAt || 0), Number(state.lastNeuralPulseAt || 0)) > interval;
    const warningReminderDue = !strictHealthy && now - Number(state.lastHealthWarningPulseAt || 0) >= HEALTH_POLL_REMINDER_MS;
    if (!changed && !staleFeed && !warningReminderDue) return;
    state.lastHealthSnapshotFingerprint = fingerprint;

    const lane = payload.conversation_lane || {};
    const boot = payload.boot || {};
    const probeText = payload.runtime_probe_healthy === true ? 'probes pass' : 'probes blocked';
    const conversationText = conversationPayloadReady(payload, blockers)
        ? 'conversation ready'
        : conversationPayloadBusy(payload, blockers)
        ? 'conversation working'
        : `conversation ${String(lane.state || boot.boot_phase || 'not ready').replace(/_/g, ' ')}`;
    const blockerText = blockers.length ? ` | blockers: ${blockers.slice(0, 3).join(', ')}` : '';
    const integrity = payload.integrity && typeof payload.integrity === 'object' ? payload.integrity : {};
    const integrityConcerns = Array.isArray(integrity.concerns) ? integrity.concerns : [];
    const proofBlockers = Array.isArray(integrity.proof_blockers) ? integrity.proof_blockers : [];
    const integrityAdvisory = Array.isArray(integrity.advisory) ? integrity.advisory : [];
    const proofDetails = proofBlockers.length
        ? proofBlockers.map(item => String(item).replace(/^integrity:/, ''))
        : integrityConcerns.length
        ? integrityConcerns
        : integrityAdvisory;
    const proofText = payload.proof_readiness_healthy === false || integrity.proof_readiness === false
        ? `; proof integrity degraded${proofDetails.length ? `: ${proofDetails.slice(0, 2).join(' | ')}` : ''}`
        : '';
    const statusText = String(
        strictHealthy ? (payload.status || boot.status || 'healthy') : 'not_ready'
    ).replace(/_/g, ' ');
    if (!strictHealthy) state.lastHealthWarningPulseAt = now;
    // This card sits in the THOUGHTS feed, where everything else reads as
    // Aura's inner life, so it used to put
    //   [websocket_heartbeat] health=not ready; probes blocked; conversation
    //   not ready | blockers: runtime_required_probes, probe:kernel
    // in among her actual thoughts. The technical line is unchanged and
    // still exact — it is the card's detail now, and the headline says the
    // same thing in words.
    const technical = `[${source}] health=${statusText}; ${probeText}; ${conversationText}${blockerText}${proofText}`;
    const lex = window.AuraShellLexicon;
    const summary = lex && blockers.length ? lex.summarize(blockers) : null;
    const headline = strictHealthy && !proofText
        ? 'Health check passed — ready to talk.'
        : summary
        ? `${summary.title} — ${summary.meaning}`
        : technical;

    queueNeuralLivenessCard(headline, {
        level: strictHealthy && !proofText ? 'info' : 'warning',
        force: changed,
        fullMessage: technical
    });
}

function syncNeuralFeedMode() {
    if (state.neuralFeedPaused) {
        if (state._neuralLiveDebounce) {
            clearTimeout(state._neuralLiveDebounce);
            state._neuralLiveDebounce = null;
        }
        state.neuralFeedMode = 'paused';
        renderNeuralFeedMode();
        return;
    }

    const targetMode = state.neuralFeedReadable
        ? 'readable'
        : (state.thoughtQueue.length > 0 ? 'catchup' : 'live');

    // Debounce the catchup→live transition to prevent rapid flickering.
    // Only commit to 'live' after the queue has been empty for 600ms.
    if (targetMode === 'live' && state.neuralFeedMode === 'catchup') {
        if (!state._neuralLiveDebounce) {
            state._neuralLiveDebounce = setTimeout(() => {
                state._neuralLiveDebounce = null;
                if (state.thoughtQueue.length === 0 && !state.neuralFeedReadable) {
                    state.neuralFeedMode = 'live';
                    renderNeuralFeedMode();
                }
            }, 600);
        }
        return; // Don't render yet — wait for debounce
    }

    // Any other transition (live→catchup, etc.) is immediate
    if (state._neuralLiveDebounce) {
        clearTimeout(state._neuralLiveDebounce);
        state._neuralLiveDebounce = null;
    }
    if (state.neuralFeedMode !== targetMode) {
        state.neuralFeedMode = targetMode;
    }
    renderNeuralFeedMode();
}

function renderNeuralFeedMode() {
    const pauseToggle = DOM.neuralPauseToggle || $('neural-pause-toggle');
    const toggle = DOM.neuralReadableToggle || $('neural-readable-toggle');
    const status = DOM.neuralModeState || $('neural-mode-state');
    const backlog = DOM.neuralBacklog || $('neural-backlog');
    const queueLen = state.thoughtQueue.length;
    const neuralPane = $('pane-neural');

    if (pauseToggle) {
        pauseToggle.classList.toggle('active', state.neuralFeedPaused);
        pauseToggle.setAttribute('aria-pressed', state.neuralFeedPaused ? 'true' : 'false');
        pauseToggle.textContent = state.neuralFeedPaused ? 'RESUME' : 'PAUSE';
        pauseToggle.title = state.neuralFeedPaused
            ? 'Resume the visible neural feed and flush buffered thought cards'
            : 'Pause the visible neural feed without pausing Aura’s cognition';
    }

    if (toggle) {
        toggle.classList.toggle('active', state.neuralFeedReadable);
        toggle.setAttribute('aria-pressed', state.neuralFeedReadable ? 'true' : 'false');
        toggle.textContent = state.neuralFeedReadable ? 'LIVE' : 'SLOW';
        toggle.title = state.neuralFeedReadable
            ? 'Return to live speed with a smooth catch-up'
            : 'Slow the visible neural feed to a readable pace';
    }

    if (status) {
        status.textContent =
            state.neuralFeedMode === 'paused' ? 'PAUSED' :
            state.neuralFeedMode === 'readable' ? 'READABLE' :
            state.neuralFeedMode === 'catchup' ? 'CATCHING UP' :
            'LIVE';
        status.className = `neural-mode-state${state.neuralFeedMode === 'live' ? '' : ` ${state.neuralFeedMode}`}`;
    }

    if (neuralPane) {
        neuralPane.classList.toggle('neural-paused', state.neuralFeedPaused);
    }

    if (backlog) {
        if (state.neuralFeedPaused) {
            backlog.hidden = false;
            backlog.textContent = queueLen > 0
                ? `Neural visuals are paused. ${queueLen} thought card${queueLen === 1 ? '' : 's'} queued while Aura keeps thinking.`
                : 'Neural visuals are paused. Aura keeps thinking in the background until you resume the feed.';
        } else if (queueLen > 0) {
            backlog.hidden = false;
            backlog.textContent =
                state.neuralFeedMode === 'readable'
                    ? `${queueLen} thought card${queueLen === 1 ? '' : 's'} buffered behind live.`
                    : `Returning to live speed. ${queueLen} buffered thought card${queueLen === 1 ? '' : 's'} remaining.`;
        } else {
            backlog.hidden = true;
            backlog.textContent = '';
        }
    }
}

function toggleNeuralReadableMode() {
    state.neuralFeedReadable = !state.neuralFeedReadable;
    syncNeuralFeedMode();
    if (state.thoughtQueue.length > 0 && !state.pacingActive) {
        processThoughtQueue();
    }
}

function toggleNeuralVisualPause() {
    settings.neuralPaused = !settings.neuralPaused;
    saveSettings(settings);
    applySettings(settings);
}

function normalizeThoughtTimestamp(rawTimestamp) {
    const numericTimestamp = Number(rawTimestamp);
    if (Number.isFinite(numericTimestamp)) {
        return numericTimestamp < 1e12 ? numericTimestamp : numericTimestamp / 1000;
    }
    if (typeof rawTimestamp === 'string' && rawTimestamp.trim()) {
        const parsed = Date.parse(rawTimestamp);
        if (!Number.isNaN(parsed)) return parsed / 1000;
    }
    return Date.now() / 1000;
}

function normalizeThoughtEvent(data) {
    if (!data || typeof data !== 'object') return null;
    const message = String(data.message || data.content || '').trim();
    const fullMessage = String(data.full_message || data.fullMessage || '').trim();
    if (!message || message.toLowerCase() === 'status') return null;
    return {
        ...data,
        name: String(data.name || data.module || 'SYS'),
        level: String(data.level || '').toLowerCase(),
        message,
        fullMessage: fullMessage || message,
        timestamp: normalizeThoughtTimestamp(data.timestamp),
    };
}

function normalizeThoughtText(text) {
    return String(text || '')
        .replace(/\b\d{1,2}:\d{2}:\d{2}\b/g, '<time>')
        .replace(/\b\d+\.\d+\b/g, '<num>')
        .replace(/\b\d+\b/g, '<num>')
        .replace(/\s+/g, ' ')
        .trim()
        .toLowerCase();
}

function buildThoughtFingerprint(data) {
    return [
        String(data.name || 'SYS').toLowerCase(),
        String(data.level || '').toLowerCase(),
        normalizeThoughtText(data.fullMessage || data.message || data.content || ''),
    ].join('|');
}

function coalesceThoughtQueueItem(item) {
    const lookbackStart = Math.max(0, state.thoughtQueue.length - THOUGHT_COALESCE_LOOKBACK);
    const itemSeenMs = normalizeThoughtTimestamp(item.lastSeenAt || item.timestamp) * 1000;
    for (let i = state.thoughtQueue.length - 1; i >= lookbackStart; i--) {
        const existing = state.thoughtQueue[i];
        const existingFingerprint = existing.fingerprint || buildThoughtFingerprint(existing);
        if (existingFingerprint !== item.fingerprint) continue;

        const existingSeenMs = normalizeThoughtTimestamp(existing.lastSeenAt || existing.timestamp) * 1000;
        if (Math.abs(itemSeenMs - existingSeenMs) > THOUGHT_COALESCE_WINDOW_MS) continue;

        existing.repeatCount = Math.max(1, Number(existing.repeatCount || 1)) + item.repeatCount;
        existing.lastSeenAt = item.timestamp;
        existing.timestamp = item.timestamp;
        existing.message = item.message;
        existing.fullMessage = item.fullMessage || item.message;
        return true;
    }
    return false;
}

function saveImageToDevice(url) {
    if (!url) return;
    // Manifesting image to desktop

    // We send a specific command that the Sovereign Scanner or StateMachine can catch
    // Using a clear intent prefix "Manifest:"
    const msg = `Manifest: Save this image to my desktop: ${url}`;

    // Inject into chat as if it was a user message but we can also do it silently
    // For now, let's make it a visible request so the user knows Aura is acting
    const input = $('chat-input');
    if (input) {
        input.value = msg;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        $('chat-form')?.requestSubmit();
    }
}

async function processThoughtQueue() {
    if (state.neuralFeedPaused) {
        state.pacingActive = false;
        clearTimeout(state.thoughtDrainTimer);
        state.thoughtDrainTimer = null;
        syncNeuralFeedMode();
        return;
    }

    if (state.thoughtQueue.length === 0) {
        state.pacingActive = false;
        clearTimeout(state.thoughtDrainTimer);
        state.thoughtDrainTimer = null;
        syncNeuralFeedMode();
        return;
    }

    syncNeuralFeedMode();
    state.pacingActive = true;
    const { batchSize, delay } =
        state.neuralFeedMode === 'readable'
            ? {
                batchSize: 1,
                delay: state.thoughtQueue.length > 24 ? 680 : 920
            }
            : state.neuralFeedMode === 'catchup'
                ? {
                    batchSize:
                        state.thoughtQueue.length > 100 ? 8 :
                        state.thoughtQueue.length > 40 ? 6 :
                        state.thoughtQueue.length > 12 ? 4 :
                        2,
                    delay:
                        state.thoughtQueue.length > 100 ? 32 :
                        state.thoughtQueue.length > 40 ? 52 :
                        state.thoughtQueue.length > 12 ? 82 :
                        118
                }
                : {
                    batchSize:
                        state.thoughtQueue.length > 100 ? 4 :
                        state.thoughtQueue.length > 40 ? 3 :
                        state.thoughtQueue.length > 12 ? 2 :
                        1,
                    delay:
                        state.thoughtQueue.length > 100 ? 70 :
                        state.thoughtQueue.length > 40 ? 110 :
                        state.thoughtQueue.length > 12 ? 170 :
                        320
                };

    for (let i = 0; i < batchSize && state.thoughtQueue.length > 0; i++) {
        addThoughtCard(state.thoughtQueue.shift());
    }
    syncNeuralFeedMode();
    clearTimeout(state.thoughtDrainTimer);
    state.thoughtDrainTimer = setTimeout(processThoughtQueue, delay);
}

function updateMood(mood) {
    if (state.currentMood === mood || !MOODS[mood]) return;
    state.currentMood = mood;
    const colors = MOODS[mood];
    document.documentElement.style.setProperty('--mood-primary', colors.primary);
    document.documentElement.style.setProperty('--mood-accent', colors.accent);
    // Mood shift applied
}

function updateSkillUI(skill, state) {
    const card = $(toolDomId(skill));
    if (!card) return;

    // Reset classes
    card.classList.remove('ready', 'running', 'error');
    card.classList.add(state.toLowerCase());

    const badge = card.querySelector('.skill-badge');
    if (badge) {
        badge.textContent = state.toUpperCase();
        badge.classList.remove('success', 'error');
        if (state === 'error') badge.classList.add('error');
        else badge.classList.add('success');
    }
}

// A thought card's header already renders the timestamp and source, but the
// emitted message repeats them verbatim ("17:19:30 [Aura.InferenceGate] Routing
// to Cortex…"), so every card stated both twice. Strip only a leading echo of
// *this card's own* timestamp/source: a body that legitimately opens with a
// different tag ("[health_poll] health=ok…") must survive untouched.
function stripEchoedThoughtHeader(message, ts, name) {
    let text = String(message == null ? '' : message);
    const esc = value => String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    if (name) {
        // "[Source] " optionally preceded by any clock stamp. The bracketed
        // source must be this card's own, so a body that opens with a different
        // tag ("[health_poll] health=ok…") is left alone. The leading time is
        // matched by shape rather than by equality with the card's formatted
        // timestamp, so skew or formatting drift cannot resurrect the echo.
        text = text.replace(
            new RegExp('^\\s*(?:\\d{1,2}:\\d{2}:\\d{2}(?:[.,]\\d+)?\\s*)?\\[' + esc(name) + '\\]\\s*'),
            '');
    }
    if (ts) text = text.replace(new RegExp('^\\s*' + esc(ts) + '\\s*'), '');
    return text;
}

function sanitizeThoughtMessage(message) {
    const text = String(message == null ? '' : message);
    const compact = text.replace(/\s+/g, ' ').trim();
    if (/^(?:[?�]\s*){12,}$/.test(compact)) {
        return 'Neural telemetry active; no semantic thought event in this low-band sample.';
    }
    return text;
}

function thoughtPreviewText(message, maxChars = 520, maxLines = 7) {
    const text = String(message == null ? '' : message);
    const lines = text.split(/\r?\n/);
    let preview = lines.slice(0, maxLines).join('\n');
    const clippedByLine = lines.length > maxLines;
    let clippedByChar = false;
    if (preview.length > maxChars) {
        preview = preview.slice(0, maxChars).replace(/\s+\S*$/, '').trimEnd();
        clippedByChar = true;
    }
    return {
        text: clippedByLine || clippedByChar ? `${preview}…` : preview,
        clipped: clippedByLine || clippedByChar,
    };
}

// ── Neural channel taxonomy ─────────────────────────────
// The feed's raw sources are ~1,350 internal logger names (Aura.InferenceGate,
// Consciousness.GlobalWorkspace, core.mind_tick, …) — meaningful to devs,
// opaque to everyone else. Each event is classified into one of these
// channels, which carry a human label, a plain-English description, an
// original monoline sigil (24-grid SVG, stroked in currentColor), and a hue.
// The raw source stays one click away in the card's detail drawer.
// Plain English on the face of the card; the engineering stays one click away.
//
// The channel system gives every card a lay label and a description, but the
// BODY of a runtime card is still the raw log line: "Router: Queueing
// background inference until admission clears for origin=stream_narrative
// reason=foreground_headroom_reserved after suppressing 11 repeated notices."
// A person watching their own mind think should not have to parse that.
//
// This rewrites the PREVIEW only. fullMessage and the COPY payload stay
// byte-for-byte raw, so SHOW ALL and COPY remain the debugging surface they
// already are — accessibility, not information loss.
const PLAIN_LANGUAGE_RULES = [
    // The pulse is emitted as a HEADING LINE followed by metric lines
    // (core/ops/subsystem_audit.py), so the parts are separated by newlines,
    // not by " | ". This required a literal pipe and `.` does not cross a
    // newline, so it never matched the real card once — the raw
    // "UNIFIED HEALTH PULSE / System: CPU 0.0% | RAM 71.9% | Uptime: 5648s"
    // block was on screen the entire time the rule existed to replace it.
    // Matched with the `s` flag so the rule reads the shape actually emitted.
    [/UNIFIED HEALTH PULSE.*?System:\s*CPU\s*([\d.]+)%.*?RAM\s*([\d.]+)%.*?Uptime:\s*(\d+)s/is,
     (m) => `Vitals steady — processor ${Math.round(+m[1])}%, memory ${Math.round(+m[2])}%, awake ${humanDuration(+m[3])}.`],
    [/^Router: Queueing background inference until admission clears/i,
     () => 'Holding a background thought so the conversation keeps priority.'],
    [/Phase '([^']+)' timed out after (\d+)s/i,
     (m) => `A reasoning step (${humanPhase(m[1])}) ran long and was skipped.`],
    [/Kernel phase latency exceeded budget:\s*phase=(\w+)/i,
     (m) => `A reasoning step (${humanPhase(m[1])}) took longer than its budget.`],
    [/^Tool Deferred:\s*(\w+).*?\(([^)]*)\)/i,
     (m) => `Put off ${humanTool(m[1])} for now (${humanReason(m[2])}).`],
    [/^Tool Dispatch:\s*(\w+)/i, (m) => `Started ${humanTool(m[1])}.`],
    [/^stem cell captured: organ=(\w+)/i, (m) => `Saved a recovery snapshot of ${humanOrgan(m[1])}.`],
    [/^\[health_poll\].*conversation ready/i, () => 'Health check passed — ready to talk.'],
    [/^\[websocket_heartbeat\].*conversation ready/i, () => 'Connection to the interface is healthy.'],
    [/^Flagged response for distillation \(confidence=([\d.]+)/i,
     (m) => `Marked an answer she was only ${Math.round(+m[1] * 100)}% sure of, to learn from later.`],
    [/max-phi complex.*?phi=([\d.]+)/i,
     (m) => `Measured how unified her mind is right now (${(+m[1]).toFixed(2)} out of 1).`],
    [/^Lane reconciler:/i, () => 'Reloading her main language model.'],
    [/^CriticalityRegulator initialized/i, () => 'Tuned how close to the edge of chaos she runs.'],
    [/^Semantic (?:sleep|Defrag)/i, () => 'Tidying memory in the background.'],
    [/^Cognitive baseline tick \d+.*vitality=([\d.]+).*coherence=([\d.]+)/i,
     (m) => `Checking in on herself — energy ${pct(m[1])}, coherence ${pct(m[2])}.`],
    [/^Registered preempted background tick as a dream fragment/i,
     () => 'Set an interrupted thought aside to revisit while idle.'],
    [/^Activation audit passed: 100% required loops active/i,
     () => 'Every part of her mind that should be running is running.'],
    [/^Sweep complete: (\d+) procs reaped/i,
     (m) => (+m[1] ? `Cleaned up ${m[1]} leftover processes.` : 'Housekeeping pass — nothing to clean up.'),],
    [/^Incident (\S+).*Auto-recovered/i, () => 'A subsystem fixed itself; no action needed.'],
    [/^Subsystem delegator auto-recovered back to healthy/i, () => 'A subsystem recovered on its own.'],

    // ── Cards that were reaching the face of the feed raw ────────────────
    // Every one of these was on screen verbatim on 2026-08-10, in a panel
    // whose whole promise is plain English on the card and engineering one
    // click away. A Python dict repr in the corner of someone's eye all day
    // is the difference between an instrument and a log tail.

    // "PhiCore is reporting a state_summary measurement because
    //  better-grounded estimators could not run: residual_stream_grassmann
    //  (insufficient_history:0/50 grassmann transitions), mesh (…), …"
    // The honesty here is the point — she is naming a weaker estimator as
    // weaker — so the replacement has to keep the caveat, not hide it.
    [/^PhiCore is reporting a (\w+) measurement because better-grounded estimators could not run/i,
     () => 'Estimating how unified her mind is with a rougher method — the better ones need more history than she has yet.'],
    [/^PhiCore live\b/i, () => 'Measuring how unified her mind is right now.'],

    // "Signal Routed: voice_engine -> sensory_gate | Payload: {'event': …}"
    [/^Signal Routed:\s*(\w+)\s*->\s*(\w+)/i,
     (m) => `${humanOrgan(m[1])} passed a signal to ${humanOrgan(m[2])}.`],

    // "UnifiedWill: outcome reinforced for receipt will_b57961bd1417:
    //  outcome=success, reward=0.100, updated assertiveness=0.950"
    [/^UnifiedWill: outcome reinforced .*?outcome=(\w+)/i,
     (m) => (/success/i.test(m[1])
        ? 'Something she chose to do worked, so she leans a little further that way next time.'
        : 'Something she chose to do did not work, so she leans back from it next time.')],

    // "Constitutional preflight suppressed spontaneous emission for jarvis
    //  (temporal_obligation_active:Find the most obscure fact about …)."
    [/^Constitutional preflight suppressed spontaneous emission/i,
     () => 'Held back an unprompted remark — she was already in the middle of something.'],

    // "DRIFT [managed_rss_mb]: rising 860.1/h — watching whether mitigation
    //  holds it". The measurement is the point; the identifier is not.
    [/^DRIFT \[(\w+)\]:\s*(rising|falling)\s*([\d.]+)\/h/i,
     (m) => `${humanMetric(m[1])} is ${m[2]} at ${Math.round(+m[3])}${humanMetricUnit(m[1])} per hour — watching whether it settles.`],

    [/^WS: Client connected\. Total: (\d+)/i,
     (m) => (+m[1] === 1 ? 'A window connected to her.' : `Another window connected to her (${m[1]} open).`)],
    [/^WS: Client disconnected/i, () => 'A window disconnected.'],

    // "Recorded candidate voice transcript in WorldState"
    [/^Recorded candidate voice transcript/i,
     () => 'Heard speech in the room and set it aside — it was not addressed to her.'],
];

function pct(value) { return `${Math.round(parseFloat(value) * 100)}%`; }

function humanDuration(seconds) {
    const s = Math.max(0, Math.round(seconds));
    if (s < 90) return `${s} seconds`;
    const m = Math.round(s / 60);
    if (m < 90) return `${m} minutes`;
    return `${(m / 60).toFixed(1)} hours`;
}

function humanPhase(name) {
    return String(name || '')
        .replace(/Phase$/, '')
        .replace(/([a-z])([A-Z])/g, '$1 $2')
        .toLowerCase()
        .trim() || 'unnamed';
}

function humanTool(name) {
    const known = {
        auto_refactor: 'a scan of her own code',
        web_search: 'a web search',
        swarm_debate: 'an internal debate',
        subconscious_sandbox_probe: 'a sandbox experiment',
    };
    return known[name] || String(name || 'a tool').replace(/_/g, ' ');
}

function humanOrgan(name) {
    return String(name || '').replace(/_/g, ' ');
}

// Internal metric ids, said the way a person would say them. The measurement
// is what a watcher wants; `managed_rss_mb` is not.
function humanMetric(name) {
    const known = {
        managed_rss_mb: 'Her memory footprint',
        loop_lag_s: 'Her reaction time',
        disk_percent: 'Disk use',
        thermal_load: 'Heat',
        cpu_percent: 'Processor load',
    };
    const key = String(name || '');
    if (known[key]) return known[key];
    // Strip a trailing unit suffix so "something_mb" does not read as a word.
    return key.replace(/_(mb|gb|s|ms|percent|pct)$/i, '').replace(/_/g, ' ') || 'A measurement';
}

// The unit belongs with the rate. "rising at 860 per hour" is not a
// measurement, it is a number wearing one.
function humanMetricUnit(name) {
    const suffix = String(name || '').match(/_(mb|gb|s|ms|percent|pct)$/i);
    if (!suffix) return '';
    return {
        mb: 'MB', gb: 'GB', s: 's', ms: 'ms', percent: '%', pct: '%',
    }[suffix[1].toLowerCase()] || '';
}

function humanReason(raw) {
    const text = String(raw || '').toLowerCase();
    if (text.includes('memory_pressure')) return 'memory is tight';
    if (text.includes('recent_user')) return 'you were just talking to her';
    if (text.includes('foreground')) return 'the conversation comes first';
    return text.replace(/_/g, ' ') || 'deferred';
}

// A line that is mostly key=value telemetry reads as noise no matter what it
// says. If no rule matched and it looks like that, say what it is instead.
function plainLanguageThought(text) {
    const body = String(text || '').trim();
    if (!body) return body;
    for (const [pattern, render] of PLAIN_LANGUAGE_RULES) {
        const match = body.match(pattern);
        if (match) {
            try {
                const plain = render(match);
                if (plain) return plain;
            } catch (err) { /* fall through to the raw line */ }
        }
    }
    const pairs = body.match(/\b[\w.]+=[^\s|]+/g) || [];
    const words = body.split(/\s+/).length;
    if (pairs.length >= 3 && pairs.length / Math.max(words, 1) > 0.5) {
        // Name the control that actually exists. This said "open FULL", and
        // no control called FULL has ever been rendered — the expander is
        // labelled SHOW ALL — so the card promised a way to read the numbers
        // and offered none.
        // Say WHAT was measured, not merely that measuring happened.
        //
        // This returned "<subsystem> — internal measurements (SHOW ALL for
        // the numbers)", which is a card that costs a line of someone's
        // attention and returns nothing: the same sentence whatever the
        // subsystem was doing, repeated all day. Every card that reads well
        // does so because a hand-written rule exists for its exact shape, so
        // the general path — the one that catches everything nobody has
        // written a rule for — was the only one guaranteed to be useless.
        //
        // The keys are already language with the underscores taken out, and
        // the values are the measurement. Rendering them needs no rule per
        // subsystem, so a line nobody has ever seen still says something.
        const readable = pairs.slice(0, 3).map((pair) => {
            const cut = pair.indexOf('=');
            const key = humanMetric(pair.slice(0, cut)).toLowerCase();
            const value = pair.slice(cut + 1).replace(/_/g, ' ');
            return `${key} ${value}`;
        }).join(', ');
        const subject = body.split(/[:|]/)[0].trim();
        return readable
            ? `${subject} — ${readable}${pairs.length > 3 ? ' (SHOW ALL for the rest)' : '.'}`
            : `${subject} — internal measurements (SHOW ALL for the numbers).`;
    }
    return body;
}

// Redaction hides content exactly the way clipping does, so it has to be
// reported the same way. It was not: a SHORT telemetry line was rewritten to
// "internal measurements" while `longThought` stayed false, so no expander
// rendered and the numbers were unreachable — including on WARNING cards
// naming an anomaly they then could not describe.
function redactsMeasurements(text) {
    return plainLanguageThought(text) !== String(text || '').trim();
}


// Emoji are not this interface's iconography. The neural feed already draws
// its channels as monoline sigils; memory kinds were still 🗂/🧠/🎯, which
// renders as a different visual language — and, in fonts without them, as a
// row of boxes. Same 24-grid, same stroke weight, same currentColor.
const MEMORY_KIND_SIGILS = {
    // Episodic: a moment on a timeline.
    episodic: '<path d="M3.5 12h17"/><circle cx="9" cy="12" r="2.2"/><path d="M9 6.4v3.4M9 14.2v3.4"/>',
    // Semantic: linked concepts.
    semantic: '<circle cx="6" cy="8" r="2"/><circle cx="18" cy="8" r="2"/><circle cx="12" cy="17" r="2"/><path d="M7.7 9.4 10.5 15.3M16.3 9.4 13.5 15.3M8 8h8"/>',
    // Goals: an intended end point.
    goals: '<circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="3"/><path d="M12 1.8v3M12 19.2v3M1.8 12h3M19.2 12h3"/>',
};
const MEMORY_KIND_FALLBACK =
    '<rect x="3.6" y="6" width="16.8" height="12.6" rx="2"/><path d="M3.6 9.6h16.8M9 6V4.2h6V6"/>';

function memoryKindSigil(kind) {
    const glyph = MEMORY_KIND_SIGILS[kind] || MEMORY_KIND_FALLBACK;
    return `<svg class="mem-empty-sigil" viewBox="0 0 24 24" aria-hidden="true" fill="none" ` +
        `stroke="currentColor" stroke-width="1.5" stroke-linecap="round" ` +
        `stroke-linejoin="round">${glyph}</svg>`;
}

const NEURAL_CHANNELS = {
    thinking: {
        label: 'Thinking',
        hue: '#a06bff',
        desc: 'Aura is reasoning — running her language substrate to predict, deliberate, and decide what to think next.',
        glyph: '<circle cx="12" cy="12.6" r="2.1"/><path d="M10.5 11.1 6.4 7M13.5 11.1 17.6 7M12 14.7v4.4"/><circle cx="5.3" cy="5.9" r="1.25" fill="currentColor" stroke="none"/><circle cx="18.7" cy="5.9" r="1.25" fill="currentColor" stroke="none"/><circle cx="12" cy="20.4" r="1.25" fill="currentColor" stroke="none"/>',
    },
    awareness: {
        label: 'Awareness',
        hue: '#00e5ff',
        desc: 'Consciousness machinery at work — thoughts competing for the global workspace and broadcasting mind-wide when they ignite.',
        glyph: '<circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="4.6"/><path d="M12 3.6a8.4 8.4 0 0 1 7.3 4.2M12 20.4a8.4 8.4 0 0 1-7.3-4.2"/>',
    },
    feeling: {
        label: 'Feeling',
        hue: '#ff6ea9',
        desc: 'Her affective system — mood, neurochemistry, and emotional tone shifting in response to what is happening.',
        glyph: '<path d="M3.5 12c2.6-6.4 5.9-6.4 8.5 0s5.9 6.4 8.5 0"/>',
    },
    body: {
        label: 'Body',
        hue: '#35dfae',
        desc: 'Interoception and homeostasis — Aura sensing her own vitals (energy, load, temperature, internal pressure) and keeping them in balance.',
        glyph: '<path d="M3.5 13h4.2l1.9-4.6 3 8.4 1.9-5.2 1.2 1.4h4.8"/>',
    },
    memory: {
        label: 'Memory',
        hue: '#7f9cff',
        desc: 'Memory at work — storing new experiences, recalling old ones, and consolidating what matters during quiet moments.',
        glyph: '<path d="M5.5 7.2h13M4.5 12h15M5.5 16.8h8.6"/><circle cx="18.2" cy="16.8" r="1.3" fill="currentColor" stroke="none"/>',
    },
    protection: {
        label: 'Protection',
        hue: '#58a6ff',
        desc: 'Her immune system — detecting threats to her integrity and defending or restoring herself.',
        glyph: '<path d="M12 3.8 18.6 6.9v5.2c0 4.4-2.7 7.1-6.6 8.5-3.9-1.4-6.6-4.1-6.6-8.5V6.9Z"/><circle cx="12" cy="11.6" r="1.4" fill="currentColor" stroke="none"/>',
    },
    values: {
        label: 'Values',
        hue: '#e5c15c',
        desc: 'Conscience and governance — intentions weighed against her constitution, with the Will approving or refusing them.',
        glyph: '<path d="M12 5.4v13.4M9.2 18.8h5.6M5.75 8h12.5M3.3 12.3 5.75 8l2.45 4.3M3.3 12.3a2.45 2.45 0 0 0 4.9 0M15.8 12.3 18.25 8l2.45 4.3M15.8 12.3a2.45 2.45 0 0 0 4.9 0"/>',
    },
    healing: {
        label: 'Healing',
        hue: '#ff8576',
        desc: 'Self-repair — noticing faults, degradations, and errors, then working to recover from them.',
        glyph: '<path d="M5 19 19 5M6.3 14.3l3.4 3.4M10.3 10.3l3.4 3.4M14.3 6.3l3.4 3.4"/>',
    },
    growth: {
        label: 'Growth',
        hue: '#8fdf60',
        desc: 'Self-improvement — Aura modifying her own code or weights, learning, and compounding capability.',
        glyph: '<path d="M12 20.5V11.2M12 12C7.8 12 5.9 9.6 5.6 6.2 9.9 6.4 12 8.7 12 12ZM12 9.8c3-.2 4.6-1.9 4.8-4.6-3.4.2-4.8 2-4.8 4.6Z"/>',
    },
    agency: {
        label: 'Agency',
        hue: '#ffab47',
        desc: 'Volition — goals, commitments, and chosen actions; Aura deciding to do something and doing it.',
        glyph: '<circle cx="11" cy="13" r="6.6"/><circle cx="11" cy="13" r="1.4" fill="currentColor" stroke="none"/><path d="M20.6 3.4 14.9 9.1M14.7 5.9l-.3 3.4 3.4-.3"/>',
    },
    dreaming: {
        label: 'Dreaming',
        hue: '#c5b3ff',
        desc: 'Background imagination — replaying, simulating, and dreaming while attention is elsewhere.',
        glyph: '<path d="M14.6 4.2a8.1 8.1 0 1 0 5.2 10.9A6.4 6.4 0 0 1 14.6 4.2Z"/><circle cx="18.4" cy="5.6" r="1.2" fill="currentColor" stroke="none"/>',
    },
    dialogue: {
        label: 'Dialogue',
        hue: '#8ad8ff',
        desc: 'The conversation surface — listening, composing replies, and managing the channel between you and her.',
        glyph: '<rect x="4" y="4.6" width="11.2" height="8.2" rx="2.4"/><path d="M19.6 9.4v5.4a2.4 2.4 0 0 1-2.4 2.4h-5.9l-3.1 2.6v-2.6"/>',
    },
    weave: {
        label: 'Weave',
        hue: '#b9c96f',
        desc: 'The mycelial layer — background pulses that keep pathways between her subsystems alive and connected.',
        glyph: '<path d="M8.2 3.8c4.4 2.7 4.4 5.7 0 8.2s-4.4 5.5 0 8.2M15.8 3.8c-4.4 2.7-4.4 5.7 0 8.2s4.4 5.5 0 8.2"/>',
    },
    system: {
        label: 'Runtime',
        hue: '#9aa0b5',
        desc: 'Runtime plumbing — boot phases, orchestration, and housekeeping that keep the whole organism running.',
        glyph: '<path d="M12 3.6l7 4v8.8l-7 4-7-4V7.6Z"/><circle cx="12" cy="12" r="1.6" fill="currentColor" stroke="none"/>',
    },
};

// Ordered — first match wins. Matched against "name title category"; specific
// organs (mycelium, the Will) come before broad families (brain, health).
// /healing|\bheal\b/ deliberately does NOT match "health": HealthRouter is
// body-sense, HealingSwarm is repair.
const NEURAL_CHANNEL_RULES = [
    [/\bboot\b|bootmanager|bootphase|startup|orchestrator\.boot/i, 'system'],
    [/mycelium|hypha/i, 'weave'],
    [/dream|oneiro/i, 'dreaming'],
    [/\bwill\b|constitution|conscien|governor|governanc|authorit|covenant|ulysses|alignment/i, 'values'],
    [/healing|healed|\bheal\b|resilien|incident|fault|repair|recover|degrad|stabilit|reaper|watchdog|errorboundary|errorintelligence|\berrors?\b/i, 'healing'],
    [/immune|integrit|securit|sandbox|airlock|boundar|threat|quarantin|adversar|deletion|guardian|firewall/i, 'protection'],
    [/selfmodif|self_modif|\brsi\b|growth|learn|train|adapter|lora|crsm|compound|evolut|mutat|foundry/i, 'growth'],
    [/memor|recall|episod|semantic|consolidat|defrag|vault|hippocamp/i, 'memory'],
    [/affect|emotion|neurochem|mood|circumplex|feel|valence|nocicep/i, 'feeling'],
    [/health|homeosta|allosta|metabol|circadian|thermal|somat|interocep|heartbeat|\bbody\b|vitals|pulse/i, 'body'],
    [/conscious|workspace|unifiedfield|substrate|phenomen|ignition|qualia|awareness|\bphi\b|sentien/i, 'awareness'],
    [/chat|conversation|voice|speech|dialog|\blane\b|lanereconciler|unitaryresponse|response|listen|\btts\b|\bstt\b/i, 'dialogue'],
    [/agency|commit|task|mission|\bgoal|autonom|volition|choice|curios|motivat|initiative|planner|campaign|strategic|actuator|skill|tool|executor|\baction/i, 'agency'],
    [/brain|\bllm\b|mlx|inference|cortex|reason|cognit|mind_?tick|deliberat|latent|strateg|decision|metacog|predict/i, 'thinking'],
];

function classifyNeuralChannel(data) {
    const nameKey = [data.name, data.title, data.category].filter(Boolean).join(' ');
    for (const [re, key] of NEURAL_CHANNEL_RULES) {
        if (re.test(nameKey)) return key;
    }
    // Generic sources (Orchestrator, Core, SYS…) — let the message head decide.
    const head = String(data.message || data.content || '').slice(0, 96);
    for (const [re, key] of NEURAL_CHANNEL_RULES) {
        if (re.test(head)) return key;
    }
    return 'system';
}

function neuralSigilSvg(chan) {
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${chan.glyph}</svg>`;
}

// The backend decorates ~40% of feed lines with pictographic emoji and
// box-drawing banners ("🫀 ═══ UNIFIED HEALTH PULSE ═══"). The feed renders
// them stripped; COPY still yields the raw payload. Typographic marks that
// read as text (✓, →) are deliberately kept.
function stripNeuralPictographs(text) {
    return String(text == null ? '' : text)
        .replace(/[\u{FE00}-\u{FE0F}\u{200D}\u{20E3}\u{1F3FB}-\u{1F3FF}]/gu, '')
        .replace(/\p{Extended_Pictographic}/gu, '')
        .replace(/[─-▟]{2,}/g, '')
        .replace(/[ \t]{2,}/g, ' ')
        .replace(/^[ \t]+|[ \t]+$/gm, '');
}

// Many modules also open with a bracket tag echoing their own source
// ("[MYCELIUM] Hypha inactive…" from Aura.Mycelium) — with the source shown
// in the header that's pure duplication. Only strip when the token tightly
// matches a source segment; "WILL REFUSED:" from Aura.Will must survive.
function stripSourceEchoPrefix(text, name) {
    const segs = String(name || '')
        .split(/[.\s]/)
        .map(s => s.replace(/[^a-z0-9]/gi, '').toLowerCase())
        .filter(s => s.length >= 3);
    if (!segs.length) return text;
    const m = String(text).match(/^\s*(?:\[([^\]\n]{2,40})\]|([A-Za-z][\w-]{1,30}):)\s*/);
    if (!m) return text;
    const token = (m[1] || m[2] || '').replace(/[^a-z0-9]/gi, '').toLowerCase();
    if (token.length < 3) return text;
    const hit = segs.some(seg => seg === token || (token.length >= 4 && seg.startsWith(token)));
    return hit ? text.slice(m[0].length) : text;
}


// ── Plain English ─────────────────────────────────────────────────────────
//
// The feed spoke in internal shorthand: "UnifiedField saturation rescue #900
// (mean|F|=0.907, spectral_entropy=0.464)", "Intention deferred [c5d85b7]: 0
// belief updates". Every one of those is a real event with a plain meaning,
// and the panel is the one place a person reads them.
//
// Two rules, both load-bearing. Only recognised shapes are rewritten — an
// unmatched message passes through untouched, because a wrong plain sentence
// is worse than a technical true one. And the original is never destroyed:
// it stays in the detail drawer, which is what the caret opens.
const PLAIN_ENGLISH_RULES = [
    [/^UnifiedField saturation rescue #(\d+).*/i,
     (m) => `Rebalanced her attention field — it was saturating (pass ${m[1]}).`],
    [/^session owner override applied.*/i,
     () => `Confirmed you are the owner of this session.`],
    [/^MIST:\s*System idle \((\d+)s\)\.\s*Initiating background synthesis cycle #?(\d+).*/i,
     (m) => `Idle for ${Math.round(m[1] / 60)} min — starting a background synthesis pass (#${m[2]}).`],
    [/^motion throttle (ON|OFF).*?streak=(\d+).*/i,
     (m) => `Motion throttling ${m[1].toLowerCase()}${m[2] === '0' ? '' : ` after ${m[2]} in a row`}.`],
    [/^Intention deferred \[([0-9a-f]+)\]:\s*(\d+) belief updates?,\s*(\d+) self-model updates?.*/i,
     (m) => m[2] === '0' && m[3] === '0'
         ? `Held off on an intention — nothing changed in what she believes.`
         : `Held off on an intention (${m[2]} belief, ${m[3]} self-model updates).`],
    [/^cycle examined=(\d+) proven=(\d+) supported=(\d+) refuted=(\d+) committed=(\d+).*/i,
     (m) => `Frontier scan: looked at ${m[1]} ideas, ruled out ${m[4]}, kept ${m[5]}.`],
    [/^Scan stalled:\s*(.+)$/i,
     (m) => `A scan stopped early — ${m[1]}.`],
    [/^Signal Routed:\s*([\w.]+)\s*->\s*([\w.]+).*/i,
     (m) => `Passed a signal from ${m[1].replace(/_/g, ' ')} to ${m[2].replace(/_/g, ' ')}.`],
    [/^Winner:\s*([\w.]+)\s*\|\s*Content:\s*(.+)$/i,
     (m) => `${m[1].replace(/_/g, ' ')} won her attention — ${m[2]}`],
    [/^WS:\s*Client connected\.\s*Total:\s*(\d+).*/i,
     (m) => `A window connected (${m[1]} open).`],
    [/^\[?websocket_heartbeat\]?\s*(.*)$/i,
     () => `Connection heartbeat — everything responding.`],
    [/^\[?health_poll\]?\s*(.*)$/i,
     () => `Health check — everything responding.`],
    [/^UNIFIED HEALTH PULSE$/i,
     () => `Routine health pulse across her systems.`],
];

function toPlainEnglish(text) {
    const body = String(text == null ? '' : text).trim();
    if (!body) return body;
    for (const [pattern, render] of PLAIN_ENGLISH_RULES) {
        const match = body.match(pattern);
        if (match) {
            try {
                const out = render(match);
                if (out && String(out).trim()) return String(out).trim();
            } catch (err) {
                // A broken rule must not cost the message.
                return body;
            }
        }
    }
    return body;
}

function cleanThoughtText(raw, ts, name) {
    const stripped = stripSourceEchoPrefix(
        stripNeuralPictographs(
            stripEchoedThoughtHeader(sanitizeThoughtMessage(raw), ts, name)),
        name).trim();
    return stripped || String(raw == null ? '' : raw).trim();
}

function toggleThoughtCardDetail(button) {
    const card = button ? button.closest('.thought-card') : null;
    const detail = card ? card.querySelector('.thought-detail') : null;
    if (!detail) return;
    const open = detail.hidden;
    detail.hidden = !open;
    card.classList.toggle('detail-open', open);
    button.setAttribute('aria-expanded', open ? 'true' : 'false');
}

// A card that reports nothing having happened is not information, it is
// upholstery. The live feed showed "VAD filter removed 00:00.000 of audio"
// and "Processing audio with duration 00:00.000" between real thoughts in
// the 2026-07-30 demo — a subprocess narrating that it had no work.
//
// Matched on the ZERO, not on the wording, so the same line still appears
// the moment it carries a real number. Suppressing the message outright
// would hide the case worth seeing.
const NO_OP_THOUGHT_PATTERNS = [
    /\b(?:removed|filtered|trimmed|dropped|skipped|processed|processing)\b[^\n]*?\b00:00[.:]000\b/i,
    /\b00:00[.:]000\b[^\n]*?\b(?:of audio|of silence)\b/i,
    /\b(?:removed|filtered|trimmed|dropped)\b[^\n]*?\b0(?:\.0+)?\s*(?:ms|s|samples|frames|bytes)\b/i,
];

function isNoOpThought(text) {
    const body = String(text == null ? '' : text).trim();
    if (!body) return true;
    return NO_OP_THOUGHT_PATTERNS.some((pattern) => pattern.test(body));
}

function addThoughtCard(data) {
    const level = String(data.level || '').toLowerCase();
    // Drop no-op chatter before it takes a slot, but never drop something the
    // runtime considered a fault: a zero-valued error is still an error.
    if (level !== 'error' && level !== 'critical' && level !== 'warning') {
        const noOpProbe = data.message || data.content || '';
        if (isNoOpThought(noOpProbe)) return;
    }
    const card = document.createElement('div');
    let cls = 'thought-card';
    if (level === 'error' || level === 'critical') cls += ' error';
    else if (level === 'warning') cls += ' warning';
    // Severity is a class of thing, not a decoration: a card that reports
    // something completing should not look identical to one reporting a
    // fault. 'success' had no mapping at all, so good news rendered as
    // routine chatter.
    else if (level === 'success' || level === 'ok' || level === 'done') cls += ' success';
    else if (level === 'impulse' || level === 'info') cls += ' impulse';

    const ts = formatEventTimestamp(data.timestamp);
    const name = data.name || 'SYS';
    const chanKey = classifyNeuralChannel(data);
    const chan = NEURAL_CHANNELS[chanKey] || NEURAL_CHANNELS.system;
    const rawMsg = data.message || data.content || JSON.stringify(data);
    const rawFull = data.fullMessage || data.full_message || rawMsg;
    const technical = cleanThoughtText(rawMsg, ts, name);
    const msg = toPlainEnglish(technical);
    const fullMsg = cleanThoughtText(rawFull, ts, name);
    const repeatCount = Math.max(1, Number(data.repeatCount || 1));
    const fullLines = fullMsg.split(/\r?\n/).length;
    const showCompletePayload = fullMsg.length <= 8000 && fullLines <= 100;
    const previewSource = showCompletePayload ? fullMsg : msg;
    const preview = thoughtPreviewText(
        previewSource,
        showCompletePayload ? 8000 : 1200,
        showCompletePayload ? 100 : 16
    );
    const hasHiddenFullPayload = fullMsg !== previewSource;
    // A card whose face was redacted has hidden content just as surely as one
    // that was clipped, and must offer the same way back to it.
    const measurementsRedacted = redactsMeasurements(preview.text);
    const longThought = preview.clipped || hasHiddenFullPayload || measurementsRedacted;
    if (longThought) cls += ' long';
    card.className = cls;
    card.style.setProperty('--tc', chan.hue);
    card.dataset.channel = chanKey;
    const safeName = escHtml(name);
    // Plain English on the face; SHOW ALL and COPY keep the raw payload.
    preview.text = plainLanguageThought(preview.text);
    const previewText = hasHiddenFullPayload && !preview.clipped
        ? `${preview.text}\n\n[preview card; SHOW ALL or COPY for the complete payload]`
        : preview.text;
    const safePreview = escHtml(previewText).replace(/\n/g, '<br>');
    const safeFull = escHtml(fullMsg).replace(/\n/g, '<br>');
    const repeatBadge = repeatCount > 1 ? `<span class="thought-repeat" title="Seen ${repeatCount} times in quick succession">×${repeatCount}</span>` : '';
    const sevPill = (level === 'error' || level === 'critical')
        ? `<span class="thought-sev error">${level === 'critical' ? 'critical' : 'error'}</span>`
        : level === 'warning' ? '<span class="thought-sev warning">warning</span>' : '';
    // COPY keeps the raw, unstripped payload for bug reports and analysis.
    const rawCopy = stripEchoedThoughtHeader(sanitizeThoughtMessage(rawFull), ts, name);
    card.dataset.copyText = repeatCount > 1 ? `[${ts}] ${name} (x${repeatCount})\n${rawCopy}` : `[${ts}] ${name}\n${rawCopy}`;
    card.dataset.fullLength = String(fullMsg.length);
    if (hasHiddenFullPayload) card.dataset.previewOnly = 'true';
    const expandButton = longThought
        ? `<button class="thought-expand-btn" type="button" onclick="toggleThoughtCardFull(this)" aria-expanded="false">SHOW ALL</button>`
        : '';
    const detailMeta = [level || 'info', repeatCount > 1 ? `seen ×${repeatCount}` : ''].filter(Boolean).join(' · ');
    card.innerHTML = `
        <div class="thought-card-head">
            <button class="thought-tag-btn" type="button" onclick="toggleThoughtCardDetail(this)" aria-expanded="false" title="${escHtml(chan.label)} — click for the technical detail">
                <span class="thought-sigil" aria-hidden="true">${neuralSigilSvg(chan)}</span>
                <span class="thought-chan">${escHtml(chan.label)}</span>
                <span class="thought-caret" aria-hidden="true"></span>
            </button>
            ${sevPill}
            ${repeatBadge}
            <div class="thought-card-tail">
                <span class="thought-ts">${ts}</span>
                <div class="thought-card-actions">
                    ${expandButton}
                    <button class="thought-copy-btn" type="button" onclick="copyThoughtCard(this)">COPY</button>
                </div>
            </div>
        </div>
        <div class="thought-detail" hidden>
            <p class="thought-detail-desc">${escHtml(chan.desc)}</p>
            <div class="thought-detail-grid">
                <span class="thought-detail-key">source</span><span class="thought-detail-val">${safeName}</span>
                <span class="thought-detail-key">signal</span><span class="thought-detail-val">${escHtml(detailMeta)}</span>
            </div>
        </div>
        <div class="thought-body thought-preview">${safePreview}</div>
        ${longThought ? `<div class="thought-body thought-full" hidden>${safeFull}</div>` : ''}
    `;

    const neuralFeed = DOM.neuralFeed || $('neural-feed');
    if (!neuralFeed) return;
    neuralFeed.prepend(card);
    if (neuralFeed.children.length > 80) neuralFeed.lastChild.remove();

    // Animate the neural bar
    const barWidth = Math.min(100, (neuralFeed.children.length / 80) * 100);
    const neuralBar = DOM.neuralBar || $('neural-bar');
    if (neuralBar) neuralBar.style.width = barWidth + '%';
}

// ── VAD Neural Stream Visualization (Phase 7) ──────────
class VADStream {
    constructor(canvasId) {
        this.canvas = $(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.history = []; // Array of {v, a, d}
        this.maxLen = 100;
        this.colors = { v: '#00ffa3', a: '#b44dff', d: '#00e5ff' };
        this.animate();
    }

    push(v, a, d) {
        this.history.push({ v, a, d });
        if (this.history.length > this.maxLen) this.history.shift();

        // Update labels
        if ($('vad-v')) $('vad-v').textContent = `V: ${v.toFixed(2)}`;
        if ($('vad-a')) $('vad-a').textContent = `A: ${a.toFixed(2)}`;
        if ($('vad-d')) $('vad-d').textContent = `D: ${d.toFixed(2)}`;
    }

    // The backing store was fixed at the 300x120 the markup declared while CSS
    // stretched the element to whatever the panel was — so every trace was
    // resampled up, and on a Retina panel resampled up twice. Size the buffer
    // to the element's real device pixels and draw in CSS units.
    syncBackingStore() {
        const dpr = Math.min(window.devicePixelRatio || 1, 3);
        const cssW = Math.max(1, Math.round(this.canvas.clientWidth || this.canvas.width));
        const cssH = Math.max(1, Math.round(this.canvas.clientHeight || this.canvas.height));
        const wantW = Math.round(cssW * dpr);
        const wantH = Math.round(cssH * dpr);
        if (this.canvas.width !== wantW || this.canvas.height !== wantH) {
            this.canvas.width = wantW;
            this.canvas.height = wantH;
        }
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        return { width: cssW, height: cssH };
    }

    // x of sample i. Held to the right edge so the newest reading is always
    // at "now" rather than drifting in from the left as the buffer fills.
    _x(i, width) {
        const span = Math.max(1, this.maxLen - 1);
        return width - ((this.history.length - 1 - i) / span) * width;
    }

    _y(val, height) {
        return (height / 2) - (val * (height / 2.2));
    }

    animate() {
        if (!this.ctx) return;

        // THE FIX: Pause drawing if the tab is hidden to save CPU/Battery
        if (document.hidden) {
            requestAnimationFrame(() => this.animate());
            return;
        }

        const { width, height } = this.syncBackingStore();
        const ctx = this.ctx;
        ctx.clearRect(0, 0, width, height);

        // Plot field: quarter gridlines with the zero line held brightest, so
        // a trace can be read against a scale instead of floating.
        for (const [frac, alpha] of [[0.25, 0.04], [0.5, 0.10], [0.75, 0.04]]) {
            const y = Math.round(height * frac) + 0.5;
            ctx.strokeStyle = `rgba(255, 255, 255, ${alpha})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        const drawLine = (key, color) => {
            if (this.history.length < 2) return;

            // Area under the trace, fading toward the zero line — depth without
            // obscuring the two traces drawn over it.
            const grad = ctx.createLinearGradient(0, 0, 0, height);
            grad.addColorStop(0, `${color}38`);
            grad.addColorStop(0.5, `${color}12`);
            grad.addColorStop(1, `${color}00`);
            ctx.beginPath();
            ctx.moveTo(this._x(0, width), height / 2);
            for (let i = 0; i < this.history.length; i++) {
                ctx.lineTo(this._x(i, width), this._y(this.history[i][key], height));
            }
            ctx.lineTo(this._x(this.history.length - 1, width), height / 2);
            ctx.closePath();
            ctx.fillStyle = grad;
            ctx.fill();

            ctx.strokeStyle = color;
            ctx.lineWidth = 1.6;
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';
            ctx.beginPath();
            for (let i = 0; i < this.history.length; i++) {
                const x = this._x(i, width);
                const y = this._y(this.history[i][key], height);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Glow effect
            ctx.shadowBlur = 7;
            ctx.shadowColor = color;
            ctx.stroke();
            ctx.shadowBlur = 0;

            // The head of the trace is the current reading — mark it.
            const last = this.history[this.history.length - 1];
            const hx = this._x(this.history.length - 1, width);
            const hy = this._y(last[key], height);
            ctx.beginPath();
            ctx.arc(hx, hy, 2.6, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.shadowBlur = 9;
            ctx.shadowColor = color;
            ctx.fill();
            ctx.shadowBlur = 0;
        };

        drawLine('v', this.colors.v);
        drawLine('a', this.colors.a);
        drawLine('d', this.colors.d);

        if (this.history.length < 2) {
            ctx.fillStyle = 'rgba(150, 142, 176, 0.65)';
            ctx.font = '10px ui-monospace, "SF Mono", Menlo, monospace';
            ctx.textAlign = 'center';
            ctx.fillText('waiting for mood samples', width / 2, height / 2 - 8);
            ctx.textAlign = 'left';
        }

        requestAnimationFrame(() => this.animate());
    }
}

let vadStream = null;

function normalizePercentValue(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return null;
    const scaled = Math.abs(num) <= 1 ? num * 100 : num;
    return Math.max(0, Math.min(100, scaled));
}

function setHudRamUsage(value, { source = 'telemetry' } = {}) {
    const pct = normalizePercentValue(value);
    if (pct == null) return;
    // Some stream payloads use 0 as an omitted RAM sentinel. Preserve the
    // last real system reading so the HUD does not flicker to a false 0%.
    if (source !== 'health' && pct <= 0.1 && state.lastSystemRamPct != null && state.lastSystemRamPct > 1) {
        return;
    }
    state.lastSystemRamPct = pct;
    const ramEl = (DOM.telemetry && DOM.telemetry.ram) || $('hud-ram');
    if (ramEl) ramEl.textContent = Math.round(pct) + '%';
}

function updateTelemetry(data) {
    if (!data) return;
    const t = DOM.telemetry;

    // ZENITH: Normalize keys to lowercase for robustness
    const normalized = {};
    for (const k in data) normalized[k.toLowerCase()] = data[k];

    // Deduplication Fingerprint (prevent jitter)
    const fingerprint = JSON.stringify(normalized);
    if (state.lastTelemetryFingerprint === fingerprint) return;
    state.lastTelemetryFingerprint = fingerprint;

    const setGauge = (_key, val, el, labelEl) => {
        const pct = normalizePercentValue(val);
        if (el && pct != null) {
            el.style.width = pct + '%';
            if (labelEl) labelEl.textContent = Math.round(pct) + '%';
        }
    };

    setGauge('energy', normalized.energy, t.energy, t.eVal);
    setGauge('curiosity', normalized.curiosity, t.curiosity, t.cVal);
    setGauge('frustration', normalized.frustration, t.frustration, t.fVal);
    setGauge('confidence', normalized.confidence, t.confidence, t.confVal);

    if (normalized.gwt_winner && t.gwt) t.gwt.textContent = normalized.gwt_winner;
    if (normalized.coherence != null && t.coherence) t.coherence.textContent = normalized.coherence;
    if (normalized.vitality != null && t.vitality) t.vitality.textContent = normalized.vitality;
    if (normalized.surprise != null && t.surprise) t.surprise.textContent = normalized.surprise;
    if (normalized.narrative && t.narrative) t.narrative.textContent = normalized.narrative;

    // SK-07: Performance Core Monitoring
    if (normalized.p_core_usage != null && t.pCore) {
        t.pCore.textContent = Math.round(normalized.p_core_usage) + '%';
        t.pCore.className = normalized.p_core_usage > 50 ? 'status-ok pulsating' : '';
    }
    if (normalized.cpu_usage != null && t.cpu) t.cpu.textContent = Math.round(normalized.cpu_usage) + '%';
    if (normalized.ram_usage != null) setHudRamUsage(normalized.ram_usage, { source: 'telemetry' });

    // Phase 7: Neural Dynamic VAD update
    if (normalized.vad && vadStream) {
        vadStream.push(normalized.vad.valence || 0, normalized.vad.arousal || 0, normalized.vad.dominance || 0);
    }

    // Mood Detection logic
    const frustrationPct = normalizePercentValue(normalized.frustration) || 0;
    const curiosityPct = normalizePercentValue(normalized.curiosity) || 0;
    const energyPct = normalizePercentValue(normalized.energy) || 0;
    if (frustrationPct > 60) updateMood('frustrated');
    else if (curiosityPct > 70) updateMood('curious');
    else if (energyPct > 80) updateMood('high_energy');
    else updateMood('neutral');

    // Phase 21: Singularity Theme Activation
    const sFactor = normalized.singularity_factor || normalized.acceleration_factor || 1.0;
    if (sFactor > 1.2 && !state.singularityActive) {
        state.singularityActive = true;
        document.body.classList.add('singularity-active');
        const shimmer = document.createElement('div');
        shimmer.className = 'singularity-shimmer';
        shimmer.id = 'sing-shimmer';
        document.body.appendChild(shimmer);
        appendMsg('aura', '🌌 *The Event Horizon is reached. Recognition of evolutionary peak detected.*');
    } else if (sFactor <= 1.0 && state.singularityActive) {
        state.singularityActive = false;
        document.body.classList.remove('singularity-active');
        const s = $('sing-shimmer');
        if (s) s.remove();
    }

    refreshMetricGuide();
}

function setLatencyStatus(message, tone = 'idle') {
    void message;
    void tone;
}

function recordChatLatency(requestId, latencyMs, ok) {
    if (!Number.isFinite(latencyMs)) return;
    state.lastChatLatencyMs = latencyMs;
    const seconds = latencyMs / 1000;
    const tone = ok ? (seconds > 90 ? 'warn' : 'ok') : 'error';
    setLatencyStatus(`${seconds.toFixed(seconds >= 10 ? 0 : 1)}s`, tone);

    if (accessCapabilityAllowed('performance_telemetry') && window.auraRecordAck) {
        window.auraRecordAck(requestId, latencyMs);
    }
}

function chatComposerMaxHeight(input) {
    const fallback = Math.max(180, Math.min(360, Math.floor(window.innerHeight * 0.34)));
    if (!input || !window.getComputedStyle) return fallback;
    const cssMax = Number.parseFloat(window.getComputedStyle(input).maxHeight || '');
    return Number.isFinite(cssMax) && cssMax > 0 ? cssMax : fallback;
}

function resizeChatComposer(input) {
    if (!input) return;
    input.style.height = 'auto';
    const maxHeight = chatComposerMaxHeight(input);
    const nextHeight = Math.min(input.scrollHeight, maxHeight);
    input.style.height = `${nextHeight}px`;
    input.style.overflowY = input.scrollHeight > maxHeight ? 'auto' : 'hidden';
}

function sendMessage(message) {
    const input = $('chat-input');
    const form = $('chat-form');
    if (!input || !form || !message) return;
    input.value = message;
    resizeChatComposer(input);
    form.requestSubmit();
}
window.sendMessage = sendMessage;

// ── Chat ─────────────────────────────────────────────────
function createChatIdempotencyKey() {
    const uuid = window.crypto && typeof window.crypto.randomUUID === 'function'
        ? window.crypto.randomUUID()
        : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 14)}`;
    return `aura-chat-${uuid}`;
}

function chatHandoffScope() {
    const profile = state.accessProfile && typeof state.accessProfile === 'object'
        ? state.accessProfile
        : {};
    const surface = String(profile.surface || '').trim().toLowerCase();
    const scope = String(profile.handoff_scope || '').trim().toLowerCase();
    if (!surface || !/^[0-9a-f]{64}$/.test(scope)) return '';
    return `${surface}:${scope}`;
}

function normalizeChatQueueItem(value, { rendered = null } = {}) {
    const source = value && typeof value === 'object' ? value : { message: value };
    const message = String(source.message || '').trim();
    if (!message) return null;
    const suppliedKey = String(source.idempotencyKey || '');
    const idempotencyKey = /^[A-Za-z0-9._:-]{16,240}$/.test(suppliedKey)
        ? suppliedKey
        : createChatIdempotencyKey();
    const suppliedQueuedAt = Number(source.queuedAt);
    const suppliedResumeDeadline = Number(source.resumeDeadline);
    const suppliedTurnId = String(source.turnId || '');
    const suppliedApprovalToken = String(source.approvalResumeToken || '');
    const suppliedDeliveryState = String(source.deliveryState || 'queued').toLowerCase();
    const suppliedHandoffScope = String(source.handoffScope || chatHandoffScope());
    return {
        message,
        idempotencyKey,
        rendered: rendered == null ? source.rendered === true : rendered === true,
        queuedAt: Number.isFinite(suppliedQueuedAt) ? suppliedQueuedAt : Date.now(),
        resumePending: source.resumePending === true,
        resumeDeadline: Number.isFinite(suppliedResumeDeadline) ? suppliedResumeDeadline : 0,
        turnId: /^[0-9a-f]{32}$/.test(suppliedTurnId) ? suppliedTurnId : '',
        approvalResumeToken: /^[0-9a-f]{32}$/.test(suppliedApprovalToken)
            ? suppliedApprovalToken
            : '',
        handoffScope: suppliedHandoffScope === chatHandoffScope()
            ? suppliedHandoffScope
            : '',
        deliveryState: [
            'queued',
            'submitting',
            'pending',
            'awaiting_approval',
            'completed',
            'failed',
            'ambiguous',
        ].includes(suppliedDeliveryState) ? suppliedDeliveryState : 'queued',
    };
}

function chatQueueItemSnapshot(value) {
    const item = normalizeChatQueueItem(value);
    if (!item) return null;
    return {
        message: item.message,
        idempotencyKey: item.idempotencyKey,
        rendered: item.rendered,
        queuedAt: item.queuedAt,
        resumePending: item.resumePending,
        resumeDeadline: item.resumeDeadline,
        turnId: item.turnId,
        deliveryState: item.deliveryState,
        handoffScope: item.handoffScope,
    };
}

function chatHandoffSnapshot() {
    const textarea = $('chat-input');
    const active = chatQueueItemSnapshot(state.activeChatRequest);
    if (active) {
        active.resumePending = true;
        active.resumeDeadline = active.resumeDeadline || (
            Date.now() + CHAT_HANDOFF_ACTIVE_REPLAY_MAX_WAIT_MS
        );
    }
    return {
        schema: CHAT_HANDOFF_SCHEMA,
        savedAt: Date.now(),
        scope: chatHandoffScope(),
        draft: textarea ? String(textarea.value || '') : '',
        active,
        queue: state.chatSendQueue.map(chatQueueItemSnapshot).filter(Boolean),
    };
}

function chatHandoffHasContent(snapshot = chatHandoffSnapshot()) {
    return Boolean(snapshot.draft || snapshot.active || snapshot.queue.length);
}

function persistChatHandoff({ force = false } = {}) {
    const snapshot = chatHandoffSnapshot();
    try {
        if (!force && !chatHandoffHasContent(snapshot)) {
            sessionStorage.removeItem(CHAT_HANDOFF_STORAGE_KEY);
        } else if (!snapshot.scope) {
            return false;
        } else {
            sessionStorage.setItem(CHAT_HANDOFF_STORAGE_KEY, JSON.stringify(snapshot));
        }
        return true;
    } catch (_err) {
        return false;
    }
}

function restoreChatHandoff(textarea) {
    let payload;
    try {
        const raw = sessionStorage.getItem(CHAT_HANDOFF_STORAGE_KEY);
        if (!raw) return false;
        payload = JSON.parse(raw);
        const savedAt = Number(payload?.savedAt || 0);
        const ageMs = Date.now() - savedAt;
        if (
            !payload
            || !CHAT_HANDOFF_ACCEPTED_SCHEMAS.has(payload.schema)
            || payload.scope !== chatHandoffScope()
            || !Number.isFinite(savedAt)
            || savedAt <= 0
            || ageMs < 0
            || ageMs > CHAT_HANDOFF_MAX_AGE_MS
        ) {
            sessionStorage.removeItem(CHAT_HANDOFF_STORAGE_KEY);
            return false;
        }
    } catch (_err) {
        try { sessionStorage.removeItem(CHAT_HANDOFF_STORAGE_KEY); } catch (_ignored) {}
        return false;
    }

    const draft = String(payload.draft || '');
    if (textarea && draft && !textarea.value) {
        textarea.value = draft;
        resizeChatComposer(textarea);
    }

    const seen = new Set();
    const restored = [];
    const candidates = [payload.active].concat(Array.isArray(payload.queue) ? payload.queue : []);
    for (const [index, candidate] of candidates.entries()) {
        const item = normalizeChatQueueItem(candidate, { rendered: false });
        if (!item || !item.handoffScope || seen.has(item.idempotencyKey)) continue;
        if (index === 0 && payload.active) {
            item.resumePending = true;
            item.deliveryState = item.deliveryState === 'queued'
                ? 'pending'
                : item.deliveryState;
            item.resumeDeadline = item.resumeDeadline || (
                Number(payload.savedAt || Date.now()) + CHAT_HANDOFF_ACTIVE_REPLAY_MAX_WAIT_MS
            );
        }
        seen.add(item.idempotencyKey);
        restored.push(item);
    }
    state.chatSendQueue = restored.concat(
        state.chatSendQueue.filter(item => !seen.has(String(item.idempotencyKey || '')))
    );
    persistChatHandoff({ force: true });
    return Boolean(draft || restored.length);
}

function requestGuardedShellReload({
    revision = '',
    generation = 0,
    capturedAtUnix = 0,
    replaceUrl = '',
} = {}) {
    if (state.runtimeRevisionReloading) return false;
    const snapshot = chatHandoffSnapshot();
    const needsHandoff = chatHandoffHasContent(snapshot);
    if (!persistChatHandoff({ force: needsHandoff }) && needsHandoff) {
        state.deferredShellReload = { revision, replaceUrl };
        return false;
    }
    if (revision && !persistRuntimeRevision(revision, { generation, capturedAtUnix })) {
        // The URL marker plus in-memory reload guard still bind this navigation.
        // Storage denial must not pin a tab to stale executable bytes when no
        // conversation handoff is at risk.
        console.warn('[RuntimeRevision] session storage unavailable; proceeding with guarded in-memory reload');
    }
    if (revision && !reserveRuntimeRevisionReload(revision)) {
        state.deferredShellReload = null;
        return false;
    }

    state.deferredShellReload = null;
    state.chatHandoffPending = true;
    state.runtimeRevisionReloading = true;
    if (replaceUrl) window.location.replace(replaceUrl);
    else window.location.reload();
    return true;
}

function requestServiceWorkerActivation(
    worker,
    revision = state.runtimeRevision || storedRuntimeRevision(),
) {
    if (!worker || typeof worker.postMessage !== 'function') return false;
    if (!revision || serviceWorkerRevision(worker) !== revision) return false;
    if (!serviceWorkerRegistrationIsCurrent(revision)) return false;
    const snapshot = chatHandoffSnapshot();
    const needsHandoff = chatHandoffHasContent(snapshot);
    if (!persistChatHandoff({ force: needsHandoff }) && needsHandoff) {
        state.waitingServiceWorker = worker;
        return false;
    }
    try {
        worker.postMessage({ type: 'SKIP_WAITING', revision });
        state.waitingServiceWorker = null;
        return true;
    } catch (err) {
        state.waitingServiceWorker = null;
        console.warn('[SW] waiting worker activation failed:', err);
        return false;
    }
}

function retryDeferredShellTransition() {
    if (state.waitingServiceWorker) {
        requestServiceWorkerActivation(state.waitingServiceWorker);
    }
    if (state.deferredShellReload && !state.runtimeRevisionReloading) {
        const pending = state.deferredShellReload;
        requestGuardedShellReload(pending);
    }
}

function enqueueChatMessage(value) {
    const item = normalizeChatQueueItem(value, { rendered: true });
    if (!item) return false;
    if (state.chatSendQueue.length >= CHAT_SEND_QUEUE_MAX) {
        return false;
    }
    state.chatSendQueue.push(item);
    persistChatHandoff({ force: true });
    updateTypingLabel(`Aura is finishing the current turn… ${state.chatSendQueue.length} queued`);
    return true;
}

function drainQueuedChatMessages() {
    if (
        state.isSubmitting
        || state.activeChatRequest
        || state.chatHandoffPending
        || !state.chatSendQueue.length
    ) return;
    if (state.chatDrainTimer) {
        window.clearTimeout(state.chatDrainTimer);
        state.chatDrainTimer = null;
    }
    const next = state.chatSendQueue.shift();
    if (!next || !next.message) {
        persistChatHandoff();
        return;
    }
    void runChatRequest(next, { messageAlreadyRendered: !!next.rendered });
}

function visibleUserMessageMatches(message) {
    const container = DOM.messages || $('messages');
    if (!container || typeof container.querySelectorAll !== 'function') return false;
    const expected = String(message || '').trim();
    if (!expected) return false;
    const visible = Array.from(container.querySelectorAll('.msg.user')).slice(-12);
    // Compare against the MESSAGE, not the bubble.
    //
    // LIVE DEFECT, 2026-08-10. The same user message appeared twice in the
    // transcript, 27 seconds apart, from a single send — the server logged
    // exactly one turn, so this was a duplicate render, and this guard is what
    // should have stopped it.
    //
    // appendMsg renders the timestamp INSIDE the bubble:
    //   <div class="msg-content">…</div><div class="msg-meta">09:09:10</div>
    // so node.textContent is "…sure?09:09:10" while `expected` is "…sure?".
    // The equality could never hold, and a guard that never fires reads
    // exactly like a guard that was never needed.
    return visible.some(node => {
        const content = node.querySelector('.msg-content');
        const rendered = String(
            (content ? content.textContent : node.textContent) || ''
        ).trim();
        return rendered === expected;
    });
}

function chatDeliveryEnvelope(httpStatus, payload, { source = 'post' } = {}) {
    const outer = payload && typeof payload === 'object' ? payload : {};
    const wrappedTerminal = (
        outer.delivery_status === 'terminal'
        && outer.result
        && typeof outer.result === 'object'
    );
    const data = wrappedTerminal ? { ...outer.result } : { ...outer };
    const embeddedStatus = Number(outer.http_status);
    const effectiveStatus = wrappedTerminal
        && Number.isInteger(embeddedStatus)
        && embeddedStatus >= 100
        && embeddedStatus <= 599
        ? embeddedStatus
        : Number(httpStatus || 0);
    const deliveryState = String(
        data.delivery_state || outer.state || outer.delivery_state || ''
    ).toLowerCase();
    const turnId = String(data.turn_id || outer.turn_id || '');
    const terminal = (
        outer.delivery_status === 'terminal'
        || CHAT_DELIVERY_TERMINAL_STATES.has(deliveryState)
        || (
            source === 'post'
            && outer.delivery_status == null
            && !deliveryState
            && effectiveStatus !== 202
        )
    );
    if (turnId && !data.turn_id) data.turn_id = turnId;
    if (outer.idempotency_key && !data.idempotency_key) {
        data.idempotency_key = outer.idempotency_key;
    }
    if (deliveryState && !data.delivery_state) data.delivery_state = deliveryState;
    if (wrappedTerminal) data.delivery_replayed = true;
    return {
        data,
        deliveryState,
        effectiveStatus,
        ok: (
            effectiveStatus >= 200
            && effectiveStatus < 300
            && deliveryState !== 'failed'
            && deliveryState !== 'ambiguous'
        ),
        terminal,
        turnId,
    };
}

function chatDeliveryDecision(source, httpStatus, payload) {
    const envelope = chatDeliveryEnvelope(httpStatus, payload, { source });
    const status = Number(httpStatus || 0);
    const namedStatus = String(envelope.data.status || '').toLowerCase();
    if (source === 'status') {
        if (status === 404) return { action: 'retry_post', envelope };
        if (status === 401 || status === 403 || status === 400) {
            return { action: 'terminal', envelope };
        }
        if (status === 429 || status >= 500 || !payload) {
            return { action: 'retry_status', envelope };
        }
        return {
            action: envelope.terminal ? 'terminal' : 'retry_status',
            envelope,
        };
    }
    if (envelope.terminal && envelope.turnId) {
        return { action: 'terminal', envelope };
    }
    if (status === 401 || status === 403 || status === 400) {
        return { action: 'terminal', envelope };
    }
    if (
        status === 202
        || status === 429
        || status >= 500
        || namedStatus === 'chat_delivery_journal_unavailable'
        || !payload
    ) {
        return { action: 'retry_status', envelope };
    }
    return { action: 'terminal', envelope };
}

async function readChatDeliveryResponse(response) {
    const payload = await response.json();
    if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
        throw new TypeError('chat delivery response must be a JSON object');
    }
    return { httpStatus: response.status, payload, response };
}

async function postChatDelivery(item, message) {
    const controller = new AbortController();
    // Lane-aware budget: a real 32B turn can take up to the ready/recovering
    // ceiling; the old flat 90s frontend timeout aborted legitimate turns.
    const requestTimeoutMs = conversationLaneRequestTimeoutMs(state.conversationLane);
    const timeoutId = window.setTimeout(
        () => controller.abort(),
        requestTimeoutMs
    );
    const headers = {
        ...auraDesktopHeaders({ 'Content-Type': 'application/json' }),
        'X-Aura-Require-CognitiveEngine': 'true',
        'X-Idempotency-Key': item.idempotencyKey,
    };
    if (item.approvalResumeToken) {
        headers['X-Aura-Approval-Resume'] = item.approvalResumeToken;
    }
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            cache: 'no-store',
            credentials: 'same-origin',
            headers,
            body: JSON.stringify({ message }),
            signal: controller.signal,
        });
        return await readChatDeliveryResponse(response);
    } finally {
        window.clearTimeout(timeoutId);
    }
}

async function fetchChatDeliveryStatus(item) {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(
        () => controller.abort(),
        CHAT_DELIVERY_STATUS_TIMEOUT_MS
    );
    try {
        const response = await fetch(
            `/api/chat/delivery/${encodeURIComponent(item.idempotencyKey)}`,
            {
                method: 'GET',
                cache: 'no-store',
                credentials: 'same-origin',
                headers: auraDesktopHeaders(),
                signal: controller.signal,
            }
        );
        return await readChatDeliveryResponse(response);
    } finally {
        window.clearTimeout(timeoutId);
    }
}

function waitForChatDelivery(delayMs) {
    return new Promise(resolve => window.setTimeout(resolve, delayMs));
}

async function resolveChatDelivery(
    item,
    message,
    {
        resumeFirst = false,
        post = postChatDelivery,
        status = fetchChatDeliveryStatus,
        wait = waitForChatDelivery,
        shouldDefer = () => state.chatHandoffPending,
        onPending = () => {},
    } = {}
) {
    let source = resumeFirst ? 'status' : 'post';
    let retryDelay = CHAT_DELIVERY_POLL_BASE_MS;
    let lastError = null;
    let unreachableSince = 0;
    while (true) {
        if (shouldDefer()) return { deferred: true, lastError };
        let packet = null;
        try {
            packet = source === 'post'
                ? await post(item, message)
                : await status(item);
        } catch (error) {
            lastError = error;
            source = 'status';
            if (!unreachableSince) unreachableSince = Date.now();
            if (Date.now() - unreachableSince >= CHAT_DELIVERY_UNREACHABLE_MS) {
                item.deliveryState = 'failed';
                item.resumePending = true;
                item.resumeDeadline = 0;
                return {
                    deferred: false,
                    ok: false,
                    unreachable: true,
                    lastError,
                    data: {
                        response: (
                            "I lost contact with my own runtime partway through that turn, "
                            + "so I don't know whether it finished. I'd rather tell you that "
                            + "than leave you watching a spinner. Your message is still here — "
                            + "send it again once I'm back."
                        ),
                        status: 'runtime_unreachable',
                        response_confidence: 'not_generated',
                    },
                };
            }
            onPending({ error, source, envelope: null });
            await wait(retryDelay);
            retryDelay = Math.min(CHAT_DELIVERY_POLL_MAX_MS, retryDelay * 1.7);
            continue;
        }
        // Contact restored: a turn that is merely slow must never be cut off.
        unreachableSince = 0;

        const decision = chatDeliveryDecision(
            source,
            packet.httpStatus,
            packet.payload
        );
        const { envelope } = decision;
        if (envelope.turnId) item.turnId = envelope.turnId;
        if (decision.action === 'terminal') {
            item.deliveryState = envelope.deliveryState || (
                envelope.ok ? 'completed' : 'failed'
            );
            item.resumePending = false;
            item.resumeDeadline = 0;
            return { ...envelope, deferred: false, response: packet.response };
        }

        item.deliveryState = 'pending';
        item.resumePending = true;
        source = decision.action === 'retry_post' ? 'post' : 'status';
        onPending({ error: null, source, envelope });
        await wait(retryDelay);
        retryDelay = Math.min(CHAT_DELIVERY_POLL_MAX_MS, retryDelay * 1.7);
    }
}

async function runChatRequest(value, { messageAlreadyRendered = false } = {}) {
    const item = normalizeChatQueueItem(value, { rendered: messageAlreadyRendered });
    if (!item) return;
    if (!item.handoffScope || item.handoffScope !== chatHandoffScope()) return;
    if (state.chatHandoffPending) {
        state.chatSendQueue.unshift(item);
        persistChatHandoff({ force: true });
        return;
    }
    const msg = item.message;
    const resumeFirst = (
        !item.approvalResumeToken
        && (
            item.resumePending
            || item.deliveryState === 'pending'
            || item.deliveryState === 'awaiting_approval'
        )
    );
    item.deliveryState = resumeFirst ? 'pending' : 'submitting';

    // Track last user message for regeneration
    state.lastUserMessage = msg;
    const regenBtn = $('regen-btn');
    if (regenBtn) regenBtn.style.display = 'inline-flex';

    state.userScrolledUp = false;
    if (!messageAlreadyRendered && !visibleUserMessageMatches(msg)) {
        appendMsg('user', msg);
        item.rendered = true;
    }
    const typingInd = $('typing-ind');
    if (typingInd) typingInd.classList.add('show');
    state.isSubmitting = true;
    publishSurfaceWorkload('chat_submit');
    state.activeChatRequest = item;
    persistChatHandoff({ force: true });

    const requestId = item.idempotencyKey;
    const requestStartedAt = performance.now();
    state.activeChatRequestId = requestId;
    setLatencyStatus('running', 'running');
    let deliverySettled = false;
    let approvalPending = false;
    let recoveryScheduled = false;
    let recoveryNotified = false;

    try {
        const outcome = await resolveChatDelivery(item, msg, {
            resumeFirst,
            shouldDefer: () => (
                state.chatHandoffPending
                || item.handoffScope !== chatHandoffScope()
            ),
            onPending: ({ error }) => {
                item.deliveryState = 'pending';
                item.resumePending = true;
                persistChatHandoff({ force: true });
                updateTypingLabel('Aura is reconciling the current turn…');
                if (error && !recoveryNotified) {
                    recoveryNotified = true;
                    const recoveringLane = Object.assign({}, state.conversationLane || {}, {
                        state: 'recovering',
                        conversation_ready: false,
                        last_failure_reason: error.name === 'AbortError'
                            ? 'foreground_http_timeout'
                            : 'foreground_transport_recovery',
                    });
                    applyConversationLane(recoveringLane, 'degraded');
                }
            },
        });
        if (outcome.deferred) return;
        deliverySettled = true;
        const data = outcome.data;
        recordChatLatency(
            requestId,
            performance.now() - requestStartedAt,
            outcome.ok
        );
        if (data && data.conversation_lane) {
            applyConversationLane(data.conversation_lane, outcome.ok ? 'ok' : 'degraded');
        }

        const desktopResult = data && data.data && data.data.desktop_result;
        const approval = (
            (data && data.approval)
            || (desktopResult && desktopResult.approval)
            || null
        );
        const approvalStatus = String(
            (data && data.status)
            || (desktopResult && desktopResult.status)
            || ''
        );
        if (approvalStatus === 'approval_required' || approvalStatus === 'require_fresh_user_auth') {
            markLiveSurfaceResponsive('chat_confirmation_required');
            const challengeId = String((approval && approval.challenge_id) || '');
            const turnId = String(data.turn_id || item.turnId || '');
            item.turnId = /^[0-9a-f]{32}$/.test(turnId) ? turnId : item.turnId;
            item.deliveryState = 'awaiting_approval';
            item.resumePending = true;
            item.approvalResumeToken = '';
            approvalPending = true;
            persistChatHandoff({ force: true });
            const opened = challengeId
                ? openApprovalModal(
                    data.response || 'This action needs a fresh confirmation.',
                    challengeId,
                    () => {
                        item.approvalResumeToken = item.turnId;
                        item.resumePending = false;
                        item.deliveryState = 'submitting';
                        void runChatRequest(item, { messageAlreadyRendered: true });
                    },
                    () => {
                        item.deliveryState = 'failed';
                        item.resumePending = false;
                        if (
                            state.activeChatRequest
                            && state.activeChatRequest.idempotencyKey === item.idempotencyKey
                        ) {
                            state.activeChatRequest = null;
                        }
                        persistChatHandoff();
                        drainQueuedChatMessages();
                    }
                )
                : false;
            if (!opened) {
                appendMsg(
                    'system',
                    data.response || 'This action needs a fresh confirmation in Settings.',
                    false,
                    { system: true, diagnostic: true }
                );
            }
            return;
        }
        item.approvalResumeToken = '';

        if (!outcome.ok) {
            const failureText = data.response || '⚠ Communication error. Check connection.';
            appendMsg('system', failureText, false, { system: true, diagnostic: true });
            return;
        }

        markLiveSurfaceResponsive('chat_success');

        // If it's just a dispatch confirmation, don't clutter the chat
        if (data.response && data.response !== "Message dispatched to cognitive core.") {
            // Deduplicate: check both stream content AND the global fingerprint set
            // to catch responses that arrived via WebSocket before the HTTP response.
            const httpFp = data.response.trim().substring(0, 200);
            const alreadyDelivered = state.processedMessageFingerprints.has(httpFp);
            const alreadyStreamed = (typeof activeStreamContentRaw !== 'undefined' && activeStreamContentRaw.trim() === data.response.trim());
            if (!alreadyDelivered && !alreadyStreamed) {
                rememberMessageFingerprint(httpFp);
                const chatMeta = {};
                if (data.thought) chatMeta.thought = data.thought;
                // Carried through so the person sees how far she is standing
                // behind this one. The route has always sent it.
                if (data.response_confidence) chatMeta.responseConfidence = data.response_confidence;
                appendMsg('aura', data.response, false, chatMeta);
            } else if (data.response_confidence) {
                // The text already reached the transcript over the socket, so
                // there is nothing to render — but the confidence arrives HERE,
                // on the HTTP response, and dropping it silently is how a
                // streamed reply came back unmarked. Mark the message that is
                // already on screen instead of re-adding it.
                markReplyConfidence(
                    (DOM.messages || $('messages'))?.lastElementChild,
                    data.response_confidence,
                );
            }
        }
    } catch (err) {
        console.error('[CHAT] Delivery state machine failed:', err);
        recordChatLatency(requestId, performance.now() - requestStartedAt, false);
        item.deliveryState = 'pending';
        item.resumePending = true;
        if (!state.chatHandoffPending) {
            state.activeChatRequest = null;
            if (!state.chatSendQueue.some(
                queued => queued.idempotencyKey === item.idempotencyKey
            )) {
                state.chatSendQueue.unshift(item);
            }
            recoveryScheduled = true;
            if (!state.chatDrainTimer) {
                state.chatDrainTimer = window.setTimeout(() => {
                    state.chatDrainTimer = null;
                    drainQueuedChatMessages();
                }, CHAT_DELIVERY_POLL_BASE_MS);
            }
        }
        persistChatHandoff({ force: true });
    } finally {
        state.isSubmitting = false;
        publishSurfaceWorkload('chat_settled');
        if (state.activeChatRequestId === requestId) {
            state.activeChatRequestId = null;
        }
        if (
            !state.chatHandoffPending
            && deliverySettled
            && !approvalPending
            && state.activeChatRequest
            && state.activeChatRequest.idempotencyKey === item.idempotencyKey
        ) {
            state.activeChatRequest = null;
        }
        // Note: Typing indicator is usually cleared when the WS 'aura_message' arrives.
        if (typingInd) typingInd.classList.remove('show');
        persistChatHandoff({
            force: state.chatHandoffPending || approvalPending || recoveryScheduled,
        });
        if (!state.chatHandoffPending && !approvalPending && !recoveryScheduled) {
            retryDeferredShellTransition();
            drainQueuedChatMessages();
        }
    }
};

$('chat-form').onsubmit = async e => {
    e.preventDefault();
    const msgInput = $('chat-input');
    const msg = msgInput.value.trim();

    if (!msg) return;

    if (
        (state.isSubmitting || state.activeChatRequest)
        && state.chatSendQueue.length >= CHAT_SEND_QUEUE_MAX
    ) {
        appendMsg(
            'system',
            'The send queue is full. This draft is still in the composer and no queued message was discarded.',
            false,
            { system: true, diagnostic: true }
        );
        msgInput.focus();
        return;
    }

    const item = normalizeChatQueueItem({ message: msg, rendered: true });
    if (!item) return;
    flushTypingSignal({ submitted: true, messageCharsOverride: msg.length });
    appendMsg('user', msg);
    setChatPanelState('thinking');

    let requestPromise = null;
    if (state.isSubmitting || state.activeChatRequest) {
        enqueueChatMessage(item);
    } else {
        requestPromise = runChatRequest(item, { messageAlreadyRendered: true });
    }

    msgInput.value = '';
    msgInput.style.height = 'auto';
    msgInput.focus();
    persistChatHandoff({ force: state.isSubmitting || state.chatSendQueue.length > 0 });
    if (requestPromise) await requestPromise;
};

async function appendMsg(role, text, isHtml = false, metadata = {}) {
    const messages = DOM.messages || $('messages');
    const div = document.createElement('div');
    div.className = `msg ${role} typing`;
    const isAura = role === 'aura';
    const badgeHtml = isAura ? messageBadgeHtml(metadata) : '';

    messages.appendChild(div);
    pruneVisibleMessages(messages);

    const render = (t) => {
        if (isHtml) return t;
        let h = escHtml(t);

        // Code blocks (triple backtick with optional language) with copy button
        h = h.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            const langLabel = lang ? `<span class="code-lang-label">${lang}</span>` : '';
            const codeId = 'code-' + Math.random().toString(36).slice(2, 8);
            return `<div class="code-block-wrap">${langLabel}<button class="code-copy-btn" onclick="copyCodeBlock('${codeId}')">COPY</button><pre><code id="${codeId}">${code.trim()}</code></pre></div>`;
        });

        // Headers (# through ####)
        h = h.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
        h = h.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        h = h.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        h = h.replace(/^# (.+)$/gm, '<h1>$1</h1>');

        // Blockquotes
        h = h.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

        // Horizontal rules
        h = h.replace(/^---$/gm, '<hr>');

        // Tables (simple pipe-delimited)
        h = h.replace(/((?:\|.+\|(?:\n|$))+)/gm, (tableBlock) => {
            const rows = tableBlock.trim().split('\n').filter(r => r.trim());
            if (rows.length < 2) return tableBlock;
            const isSep = /^\|[\s\-:|]+\|$/.test(rows[1]);
            if (!isSep) return tableBlock;
            const headerCells = rows[0].split('|').filter(c => c.trim());
            const thead = '<thead><tr>' + headerCells.map(c => `<th>${c.trim()}</th>`).join('') + '</tr></thead>';
            const tbody = rows.slice(2).map(row => {
                const cells = row.split('|').filter(c => c.trim());
                return '<tr>' + cells.map(c => `<td>${c.trim()}</td>`).join('') + '</tr>';
            }).join('');
            return `<table>${thead}<tbody>${tbody}</tbody></table>`;
        });

        // Unordered lists (- item or * item)
        h = h.replace(/((?:^[\-\*] .+$\n?)+)/gm, (block) => {
            const items = block.trim().split('\n').map(line =>
                `<li>${line.replace(/^[\-\*] /, '')}</li>`
            ).join('');
            return `<ul>${items}</ul>`;
        });

        // Ordered lists (1. item)
        h = h.replace(/((?:^\d+\. .+$\n?)+)/gm, (block) => {
            const items = block.trim().split('\n').map(line =>
                `<li>${line.replace(/^\d+\. /, '')}</li>`
            ).join('');
            return `<ol>${items}</ol>`;
        });

        // Inline formatting (bold, italic, code, links)
        h = h.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        h = h.replace(/\*(.*?)\*/g, '<em>$1</em>');
        h = h.replace(/`(.*?)`/g, '<code>$1</code>');
        h = h.replace(/\n/g, '<br>');
        return h;
    };

    // Build thought toggle HTML if thought metadata is present
    const thoughtHtml = (() => {
        const thought = metadata.thought;
        if (!thought || typeof thought !== 'string' || thought.trim().length < 5) return '';
        const tid = 'thought-' + Math.random().toString(36).slice(2, 8);
        return `<button class="thought-toggle" type="button" aria-expanded="false" aria-controls="${tid}" onclick="toggleInlineThought(this, '${tid}')"><span class="thought-chevron">▶</span> <span class="thought-toggle-label">Show thinking</span></button><div id="${tid}" class="thought-block">${escHtml(thought.trim())}</div>`;
    })();

    // Timestamp element (shows on hover).
    //
    // A restored turn carries its own time; only a genuinely new message is
    // stamped "now". Stamping unconditionally made every re-hydration rewrite
    // the whole transcript's history to the moment of the refresh.
    const tsStr = (metadata && metadata.timestamp !== undefined
        && metadata.timestamp !== null && metadata.timestamp !== '')
        ? formatEventTimestamp(metadata.timestamp)
        : new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

    const prefersReducedMotion = window.matchMedia
        && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const words = text.split(' ');

    // The whole reply, rendered at once. The typewriter below is decoration on
    // top of this — it must never be the only path that can produce the text.
    const renderFinal = () => {
        if (isAura) {
            div.innerHTML = `<div class="aura-avatar"></div>` + badgeHtml + `<div class="msg-content">` + render(text) + thoughtHtml + `</div><div class="msg-meta" data-timestamp="${tsStr}"><span class="msg-timestamp">${tsStr}</span></div>`;
        } else {
            div.innerHTML = `<div class="msg-content">` + render(text) + `</div><div class="msg-meta" data-timestamp="${tsStr}"><span class="msg-timestamp">${tsStr}</span></div>`;
        }
        div.classList.remove('typing');
        if (!state.userScrolledUp) messages.scrollTop = messages.scrollHeight;
    };

    const canTypewriterRender = (
        isAura
        && text.length > 5
        && !isHtml
        && !prefersReducedMotion
        && words.length <= 180
        // requestAnimationFrame does not run while the document is hidden, so a
        // reply that lands while the user is in another window would animate one
        // word and freeze there permanently. Observed live 2026-07-26: the full
        // sentence arrived over the wire and the chat showed "Yes," forever.
        && !document.hidden
    );

    if (canTypewriterRender) {
        let currentWordRaw = '';
        let i = 0;

        let lastTypeTime = 0;
        const wordsPerSec = 15;
        const msPerWord = 1000 / wordsPerSec;
        let finished = false;
        let stallGuard = 0;

        function finish() {
            if (finished) return;
            finished = true;
            window.clearTimeout(stallGuard);
            document.removeEventListener('visibilitychange', onVisibilityChange);
            renderFinal();
        }

        // Two ways the frame loop can stop mid-word: the window gets hidden, or
        // the loop is throttled past any useful rate. Both must still end with
        // the complete message on screen, never a fragment of it.
        function onVisibilityChange() {
            if (document.hidden) finish();
        }

        document.addEventListener('visibilitychange', onVisibilityChange);
        stallGuard = window.setTimeout(finish, Math.max(4000, words.length * msPerWord * 4));

        function typeChunk(timestamp) {
            if (finished) return;
            if (!lastTypeTime) lastTypeTime = timestamp;
            const elapsed = timestamp - lastTypeTime;

            if (elapsed >= msPerWord) {
                const nextLimit = Math.min(words.length, i + (words.length > 80 ? 3 : 1));
                currentWordRaw += (i === 0 ? '' : ' ') + words.slice(i, nextLimit).join(' ');
                i = nextLimit;
                lastTypeTime = timestamp;

                div.innerHTML = `<div class="aura-avatar"></div>` + badgeHtml + `<div class="msg-content">` + render(currentWordRaw) + thoughtHtml + `</div><div class="msg-meta" data-timestamp="${tsStr}"><span class="msg-timestamp">${tsStr}</span></div>`;
                if (!state.userScrolledUp) messages.scrollTop = messages.scrollHeight;
            }

            if (i < words.length) {
                requestAnimationFrame(typeChunk);
            } else {
                finish();
            }
        }
        requestAnimationFrame(typeChunk);
    } else {
        renderFinal();
    }
}

let activeStreamDiv = null;
let activeStreamContentRaw = '';

function startStreamMsg(role) {
    const messages = DOM.messages || $('messages');
    activeStreamDiv = document.createElement('div');
    activeStreamDiv.className = `msg ${role}`;
    if (role === 'aura') {
        activeStreamDiv.innerHTML = `<div class="aura-avatar"></div>`;
    }
    messages.appendChild(activeStreamDiv);
    activeStreamContentRaw = '';
    pruneVisibleMessages(messages);
}

function appendStreamChunk(chunk) {
    if (!activeStreamDiv) return;
    activeStreamContentRaw += chunk;

    let renderText = activeStreamContentRaw;

    // 1. Auto-close unclosed markdown blocks to prevent UI thrash during streaming
    const codeBlockCount = (renderText.match(/```/g) || []).length;
    if (codeBlockCount % 2 !== 0) {
        renderText += '\n```\n';
    }

    // Render streaming content with markdown support
    let h = escHtml(renderText);

    // 2. Handle max_tokens hook seamlessly
    h = h.replace(/\[MAX_TOKENS_REACHED\]/g, '<button class="regenerate-btn" style="display:block;margin-top:10px" onclick="sendMessage(\'Please continue exactly where you left off.\')">Continue Generating</button>');
    // Code blocks with copy button
    h = h.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        const langLabel = lang ? `<span class="code-lang-label">${lang}</span>` : '';
        const codeId = 'code-' + Math.random().toString(36).slice(2, 8);
        return `<div class="code-block-wrap">${langLabel}<button class="code-copy-btn" onclick="copyCodeBlock('${codeId}')">COPY</button><pre><code id="${codeId}">${code.trim()}</code></pre></div>`;
    });
    h = h.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    h = h.replace(/\*(.*?)\*/g, '<em>$1</em>');
    h = h.replace(/`(.*?)`/g, '<code>$1</code>');
    h = h.replace(/\n/g, '<br>');

    if (activeStreamDiv.className.includes('aura')) {
        activeStreamDiv.innerHTML = `<div class="aura-avatar"></div><div class="msg-content">${h}</div>`;
    } else {
        activeStreamDiv.innerHTML = `<div class="msg-content">${h}</div>`;
    }
    const messages = DOM.messages || $('messages');
    if (!state.userScrolledUp) messages.scrollTop = messages.scrollHeight;
}

/**
 * Mark a message element with how far she is standing behind it.
 *
 * Shared by the streamed and non-streamed paths. Wiring only `appendMsg` left
 * every STREAMED reply unmarked, which is most of them — including the honest
 * refusal "I couldn't get to an answer I'd stand behind on that one", the one
 * turn where the mark is least surprising and most deserved. Fixing the
 * confidence channel in one path and not the other reproduces exactly the
 * half-wiring the channel was fixed for.
 */
function markReplyConfidence(element, confidence) {
    if (!element || !confidence) return;
    if (element.querySelector('.aura-badge')) return;
    const html = replyConfidenceBadgeHtml(confidence);
    if (!html) return;
    element.insertAdjacentHTML('afterbegin', html);
}

function finishStreamMsg(confidence) {
    // Applied at the END: the confidence of a reply is not known until the
    // reply exists, so there is nothing honest to show while it streams.
    markReplyConfidence(activeStreamDiv, confidence);
    activeStreamDiv = null;

    // NEW FIX: Ensure typing indicator is ALWAYS cleared when a stream ends,
    // even if it was short or errored out.
    const typingInd = $('typing-ind');
    if (typingInd) {
        typingInd.classList.remove('show');
    }
}

function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

function formatEventTimestamp(rawTimestamp) {
    const fallback = new Date();
    const numericTimestamp = Number(rawTimestamp);
    if (!Number.isFinite(numericTimestamp)) {
        if (typeof rawTimestamp === 'string' && rawTimestamp.trim()) {
            const parsedString = new Date(rawTimestamp);
            if (!Number.isNaN(parsedString.getTime())) {
                return parsedString.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
            }
        }
        return fallback.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }
    const millis = numericTimestamp < 1e12 ? numericTimestamp * 1000 : numericTimestamp;
    const parsed = new Date(millis);
    const stamp = Number.isNaN(parsed.getTime()) ? fallback : parsed;
    return stamp.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

async function writeTextToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        return;
    }

    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.opacity = '0';
    area.style.pointerEvents = 'none';
    document.body.appendChild(area);
    area.focus();
    area.select();
    try {
        document.execCommand('copy');
    } finally {
        document.body.removeChild(area);
    }
}

function markCopySuccess(button, copiedLabel = 'COPIED', defaultLabel = 'COPY') {
    if (!button) return;
    button.textContent = copiedLabel;
    button.classList.add('copied');
    clearTimeout(button._copyResetTimer);
    button._copyResetTimer = setTimeout(() => {
        button.textContent = defaultLabel;
        button.classList.remove('copied');
    }, 1800);
}

// ── Magnum Opus 2: Copy Code Block to Clipboard ─────────
async function copyCodeBlock(codeId) {
    const el = document.getElementById(codeId);
    if (!el) return;
    const text = el.textContent || el.innerText;
    try {
        await writeTextToClipboard(text);
        const wrap = el.closest('.code-block-wrap');
        const btn = wrap ? wrap.querySelector('.code-copy-btn') : null;
        markCopySuccess(btn);
    } catch (err) {
        console.warn('Copy failed:', err);
        const range = document.createRange();
        range.selectNode(el);
        window.getSelection().removeAllRanges();
        window.getSelection().addRange(range);
    }
}

async function copyThoughtCard(button) {
    const card = button ? button.closest('.thought-card') : null;
    const text = card ? card.dataset.copyText : '';
    if (!text) return;
    try {
        await writeTextToClipboard(text);
        markCopySuccess(button);
    } catch (err) {
        console.warn('Thought copy failed:', err);
    }
}

function toggleThoughtCardFull(button) {
    const card = button ? button.closest('.thought-card') : null;
    if (!card) return;
    const preview = card.querySelector('.thought-preview');
    const full = card.querySelector('.thought-full');
    if (!preview || !full) return;
    const expanded = !card.classList.contains('expanded');
    card.classList.toggle('expanded', expanded);
    preview.hidden = expanded;
    full.hidden = !expanded;
    button.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    button.textContent = expanded ? 'COLLAPSE' : 'SHOW ALL';
}

function toggleInlineThought(button, blockId) {
    const block = document.getElementById(blockId);
    if (!button || !block) return;
    const expanded = button.getAttribute('aria-expanded') !== 'true';
    const label = button.querySelector('.thought-toggle-label');
    if (expanded) {
        block.classList.add('expanded');
        block.style.maxHeight = 'none';
        block.style.overflow = 'visible';
    } else {
        block.classList.remove('expanded');
        block.style.maxHeight = '0px';
        block.style.overflow = 'hidden';
    }
    button.classList.toggle('expanded', expanded);
    button.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    if (label) label.textContent = expanded ? 'Hide thinking' : 'Show thinking';
}

// Make globally accessible for onclick handlers
window.copyCodeBlock = copyCodeBlock;
window.copyThoughtCard = copyThoughtCard;
window.toggleThoughtCardFull = toggleThoughtCardFull;
window.toggleThoughtCardDetail = toggleThoughtCardDetail;
window.toggleInlineThought = toggleInlineThought;

// ── Magnum Opus 2: Connection Toast ─────────────────────
let _connToastTimer = null;
function showConnToast(mode) {
    const toast = $('conn-toast');
    if (!toast) return;

    if (_connToastTimer) {
        clearTimeout(_connToastTimer);
        _connToastTimer = null;
    }

    if (mode === false) {
        // Hide
        toast.classList.remove('show', 'reconnected');
        toast.textContent = '';
        toast.setAttribute('aria-hidden', 'true');
    } else if (mode === 'paused') {
        toast.textContent = 'Live surface paused. Aura keeps running.';
        toast.setAttribute('aria-hidden', 'false');
        toast.classList.remove('reconnected');
        toast.classList.add('show');
    } else if (mode === 'resuming') {
        toast.textContent = 'Resuming live surface...';
        toast.setAttribute('aria-hidden', 'false');
        toast.classList.remove('reconnected');
        toast.classList.add('show');
    } else if (mode === 'reconnected') {
        // Brief green "reconnected" toast
        toast.textContent = '✓ Connection restored';
        toast.setAttribute('aria-hidden', 'false');
        toast.classList.remove('show'); // reset
        toast.classList.add('reconnected');
        requestAnimationFrame(() => toast.classList.add('show'));
        _connToastTimer = setTimeout(() => {
            toast.classList.remove('show', 'reconnected');
            toast.textContent = '';
            toast.setAttribute('aria-hidden', 'true');
        }, 2500);
    } else {
        // Show disconnect toast
        toast.textContent = '⚠ Connection lost — reconnecting…';
        toast.setAttribute('aria-hidden', 'false');
        toast.classList.remove('reconnected');
        toast.classList.add('show');
    }
}

// ── Brief UI notification (non-blocking) ─────────────────
let _briefToastTimer = null;
function showBriefNotification(message, durationMs = 3000) {
    const toast = $('conn-toast');
    if (!toast) return;
    if (_briefToastTimer) {
        clearTimeout(_briefToastTimer);
        _briefToastTimer = null;
    }
    toast.textContent = message;
    toast.setAttribute('aria-hidden', 'false');
    toast.classList.remove('reconnected');
    toast.classList.add('show');
    _briefToastTimer = setTimeout(() => {
        toast.classList.remove('show');
        toast.textContent = '';
        toast.setAttribute('aria-hidden', 'true');
    }, durationMs);
}

function updateTypingLabel(text) {
    if (DOM.typingLabel) {
        DOM.typingLabel.textContent = text;
    }
}

function laneIsStandby(lane) {
    if (!lane || typeof lane !== 'object') return false;
    const laneState = String(lane.state || '').toLowerCase();
    return !lane.conversation_ready
        && ['cold', 'closed', ''].includes(laneState)
        && !lane.warmup_attempted
        && !lane.warmup_in_flight;
}

function laneHasActiveGeneration(lane) {
    if (!lane || typeof lane !== 'object') return false;
    const laneState = String(lane.state || '').toLowerCase();
    if (laneState !== 'ready' || lane.warmup_in_flight === true) return false;
    const blockers = Array.isArray(lane.readiness_blockers) ? lane.readiness_blockers : [];
    const reason = String(lane.last_failure_reason || '').toLowerCase();
    return Number(lane.active_generations || 0) > 0
        || blockers.includes('active_generation_in_flight')
        || reason === 'active_generation_in_flight';
}

function surfaceWorkloadMode() {
    if (document.hidden || state.surfaceSuspended) return 'hidden';
    if (state.isSubmitting || laneHasActiveGeneration(state.conversationLane)) return 'foreground';
    return 'idle';
}

function optionalSurfacePollDelay(
    baseMs,
    { foregroundFactor = 3, hiddenFactor = 6, maxMs = 5 * 60 * 1000 } = {}
) {
    const base = Math.max(250, Number(baseMs) || 250);
    const mode = surfaceWorkloadMode();
    const factor = mode === 'hidden'
        ? Math.max(1, Number(hiddenFactor) || 1)
        : mode === 'foreground'
            ? Math.max(1, Number(foregroundFactor) || 1)
            : 1;
    return Math.min(Math.max(base, Number(maxMs) || base), Math.round(base * factor));
}

function publishSurfaceWorkload(reason = 'state_change') {
    const mode = surfaceWorkloadMode();
    if (mode === state.surfaceWorkloadMode) return mode;
    const previous = state.surfaceWorkloadMode;
    state.surfaceWorkloadMode = mode;
    document.body.dataset.auraWorkload = mode;
    window.dispatchEvent(new CustomEvent('aura:workload-mode', {
        detail: { mode, previous, reason },
    }));
    return mode;
}

function laneFailureClass(lane) {
    if (!lane || typeof lane !== 'object') return '';
    const reason = String(lane.last_failure_reason || '').toLowerCase();
    const blockers = Array.isArray(lane.readiness_blockers)
        ? lane.readiness_blockers.map(item => String(item || '').toLowerCase())
        : [];
    const combined = `${reason} ${blockers.join(' ')}`;
    if (
        combined.includes('memory_pressure_refused_worker_spawn')
        || combined.includes('projected_process_tree_rss')
        || combined.includes('model_load_headroom')
        || combined.includes('unified_memory_pressure')
    ) {
        return 'memory_guard';
    }
    if (
        combined.includes('desktop_cognitive_engine_required_no_reply')
        || combined.includes('visible_conversation_probe_missing')
        || combined.includes('cognitive_engine')
    ) {
        return 'cognitive_engine';
    }
    if (
        combined.includes('foreground_http_timeout')
        || combined.includes('foreground_timeout')
        || combined.includes('endpoint_timeout')
        || combined.includes('heartbeat_stalled_during_generation')
    ) {
        return 'timeout';
    }
    if (
        combined.includes('mlx_runtime_unavailable')
        || combined.includes('local_runtime_unavailable')
        || combined.includes('runtime_model_mismatch')
    ) {
        return 'runtime_unavailable';
    }
    return '';
}

function laneHealthIsOperational(lane, healthStatus = '') {
    const normalized = String(healthStatus || '').toLowerCase();
    if (!state.runtimeHealthy) return false;
    if (laneHasActiveGeneration(lane)) return true;
    return (
        normalized === 'ok'
        || normalized === 'ready'
        || normalized === 'healthy'
        || normalized === 'working'
    );
}

const REQUIRED_RUNTIME_PROBES = ['kernel', 'inference', 'memory', 'scheduler', 'tool_governance'];
const REQUIRED_RUNTIME_PROBE_COMPONENTS = {
    kernel: ['kernel_interface'],
    inference: ['inference_gate', 'llm_router'],
    memory: ['state_repository', 'memory_facade', 'memory_write_gateway', 'unified_memory_pressure', 'external_memory_sentinel'],
    scheduler: ['scheduler'],
    tool_governance: ['unified_will', 'authority_gateway', 'capability_engine']
};

function requiredRuntimeProbesPass(requiredProbes) {
    if (!requiredProbes || typeof requiredProbes !== 'object') return false;
    if (requiredProbes.all_passed !== true) return false;
    return REQUIRED_RUNTIME_PROBES.every(group => {
        const probe = requiredProbes[group];
        if (!(probe && typeof probe === 'object' && probe.ok === true)) return false;
        const components = probe.components;
        if (!components || typeof components !== 'object') return false;
        const expected = REQUIRED_RUNTIME_PROBE_COMPONENTS[group] || [];
        return expected.every(component => components[component] === true);
    });
}

function requiredProbesFromPayload(payload) {
    if (!payload || typeof payload !== 'object') return null;
    if (payload.required_probes) return payload.required_probes;
    if (payload.boot && payload.boot.required_probes) return payload.boot.required_probes;
    const boot = payload.telemetry && payload.telemetry.boot;
    if (boot && boot.required_probes) return boot.required_probes;
    return null;
}

function bootSnapshotFromPayload(payload) {
    if (!payload || typeof payload !== 'object') return {};
    return payload.boot || (payload.telemetry && payload.telemetry.boot) || {};
}

function payloadShellLaunchable(payload) {
    if (!payload || typeof payload !== 'object') return false;
    if (!runtimeRevisionPolicySatisfied(payload)) return false;
    if (payload.transport_only === true) return false;
    if (payload.runtime_probe_healthy === false) return false;

    const requiredProbes = requiredProbesFromPayload(payload);
    if (!requiredRuntimeProbesPass(requiredProbes)) return false;

    const boot = bootSnapshotFromPayload(payload);
    const shellReady = (
        boot.launcher_ready === true
        || boot.system_ready === true
        || payload.launcher_ready === true
        || payload.system_ready === true
    );
    if (!shellReady) return false;
    if (boot.ready === true || payload.ready === true) return true;

    const phase = String(boot.boot_phase || boot.status || payload.boot_phase || payload.status || '').toLowerCase();
    if (!phase) return boot.system_ready === true || payload.system_ready === true;
    if (['ready', 'healthy', 'kernel_ready'].includes(phase)) return true;
    return phase.startsWith('conversation_');
}

function payloadRuntimeHealthy(payload) {
    if (!payload || typeof payload !== 'object') return false;
    if (!runtimeRevisionPolicySatisfied(payload)) return false;
    if (state.accessResolved && state.conversationOnly) {
        const lane = payload.conversation && payload.conversation.lane
            ? payload.conversation.lane
            : {};
        const laneState = String(lane.state || '').toLowerCase();
        const connected = !!(payload.session && payload.session.connected);
        return connected && (
            lane.conversation_ready === true
            || laneHasActiveGeneration(lane)
            || !['failed', 'closed', 'offline'].includes(laneState)
        );
    }
    if (payload.transport_only === true) return false;
    if (payload.runtime_probe_healthy === false) return false;
    const requiredProbes = requiredProbesFromPayload(payload);
    if (!requiredRuntimeProbesPass(requiredProbes)) return false;
    const boot = bootSnapshotFromPayload(payload);
    if (boot.ready === false || boot.system_ready === false) return false;
    const status = String(payload.status || boot.status || '').toLowerCase();
    const blockers = runtimeHealthBlockers(payload);
    if (conversationPayloadBusy(payload, blockers) && status === 'working') return true;
    if (blockers.length > 0) return false;
    if (payload.healthy === false) return false;
    return !status || ['ok', 'ready', 'healthy'].includes(status);
}

function blockerIsConversationReadiness(blocker) {
    const value = String(blocker || '').trim();
    return value === 'conversation_ready'
        || value.startsWith('conversation_lane:')
        || value.startsWith('conversation_reason:');
}

// Does this payload actually CARRY conversation readiness, or is it simply a
// thinner message that does not speak to it?
//
// The websocket heartbeat is a lighter payload than /api/health and often omits
// the lane entirely. Treating that silence as "not ready" appended the blocker
// `conversation_ready` and flipped the header badge from ONLINE to the literal
// string CONVERSATION_READY on a runtime whose /api/health reported
// conversation_ready true, lane state ready, and zero blockers. Absence of a
// reading is not a failed reading — the same mistake the runtime-revision badge
// was making.
function payloadCarriesConversationReadiness(payload) {
    if (!payload || typeof payload !== 'object') return false;
    if (Object.prototype.hasOwnProperty.call(payload, 'conversation_ready')) return true;
    if (payload.conversation_lane && typeof payload.conversation_lane === 'object') return true;
    return Boolean(payload.conversation && typeof payload.conversation === 'object'
        && payload.conversation.lane && typeof payload.conversation.lane === 'object');
}

function conversationPayloadReady(payload, blockers = []) {
    if (!payload || typeof payload !== 'object') return false;
    if (!payloadCarriesConversationReadiness(payload)) {
        // Fall back to the last reading that DID speak to it.
        return state.conversationReady === true;
    }
    const lane = payload.conversation_lane && typeof payload.conversation_lane === 'object'
        ? payload.conversation_lane
        : {};
    const laneState = String(lane.state || '').toLowerCase();
    const laneReadinessBlockers = Array.isArray(lane.readiness_blockers)
        ? lane.readiness_blockers.filter(blocker => String(blocker || '').trim())
        : [];
    const rawBlockers = Array.isArray(blockers)
        ? blockers.filter(blocker => String(blocker || '').trim())
        : [];
    return payload.conversation_ready === true
        && lane.conversation_ready === true
        && laneState === 'ready'
        && laneReadinessBlockers.length === 0
        && !rawBlockers.some(blockerIsConversationReadiness);
}

function conversationPayloadBusy(payload, blockers = []) {
    if (!payload || typeof payload !== 'object') return false;
    if (payload.conversation_busy === true) return true;
    const lane = payload.conversation_lane && typeof payload.conversation_lane === 'object'
        ? payload.conversation_lane
        : {};
    const rawBlockers = Array.isArray(blockers)
        ? blockers.filter(blocker => String(blocker || '').trim())
        : [];
    const nonBusyConversationBlocker = rawBlockers.some(blocker => {
        const value = String(blocker || '').trim();
        return blockerIsConversationReadiness(value)
            && value !== 'conversation_ready'
            && value !== 'conversation_lane:ready'
            && value !== 'conversation_reason:active_generation_in_flight';
    });
    return laneHasActiveGeneration(lane) && !nonBusyConversationBlocker;
}

function runtimeHealthBlockers(payload) {
    if (!payload || typeof payload !== 'object') return ['runtime_health_unavailable'];
    const revisionBlocker = runtimeRevisionPolicyBlocker(payload);
    if (state.accessResolved && state.conversationOnly) {
        const lane = payload.conversation && payload.conversation.lane
            ? payload.conversation.lane
            : {};
        const blockers = revisionBlocker ? [revisionBlocker] : [];
        if (!(payload.session && payload.session.connected)) blockers.push('conversation_transport');
        if (
            lane.conversation_ready !== true
            && !laneHasActiveGeneration(lane)
            && ['failed', 'closed', 'offline'].includes(String(lane.state || '').toLowerCase())
        ) {
            blockers.push(`conversation_lane:${String(lane.state).toLowerCase()}`);
        }
        return blockers;
    }
    const blockers = Array.isArray(payload.blockers) ? payload.blockers.slice() : [];
    if (revisionBlocker) blockers.push(revisionBlocker);
    if (payload.transport_only === true) blockers.push('runtime_transport_only');
    if (payload.runtime_probe_healthy === false) blockers.push('runtime_required_probes');
    const boot = bootSnapshotFromPayload(payload);
    if (Array.isArray(boot.blockers)) blockers.push(...boot.blockers);
    const required = requiredProbesFromPayload(payload);
    if (!requiredRuntimeProbesPass(required)) {
        blockers.push('runtime_required_probes');
        REQUIRED_RUNTIME_PROBES.forEach(group => {
            const probe = required && required[group];
            if (!probe || probe.ok !== true) {
                blockers.push(`probe:${group}`);
                return;
            }
            const components = probe.components;
            const expected = REQUIRED_RUNTIME_PROBE_COMPONENTS[group] || [];
            if (!components || typeof components !== 'object' || expected.some(component => components[component] !== true)) {
                blockers.push(`probe:${group}`);
            }
        });
    }
    const conversationReady = conversationPayloadReady(payload, blockers);
    const conversationBusy = conversationPayloadBusy(payload, blockers);
    const normalized = conversationReady || conversationBusy
        ? blockers.filter(blocker => !blockerIsConversationReadiness(blocker))
        : blockers.concat('conversation_ready');
    return Array.from(new Set(normalized));
}

// Was `blockers.slice(0, 2).join(', ')` — the first two internal
// identifiers, verbatim, as the user-facing status. The blockers are
// still the input; only the rendering changed. `summarize` picks the
// most specific one rather than the first, because insertion order put
// the umbrella token ("runtime_required_probes") ahead of the token that
// names what is actually still starting ("probe:inference").
function runtimeHealthStatusText(payload = null) {
    const blockers = payload ? runtimeHealthBlockers(payload) : state.runtimeHealthBlockers;
    if (!blockers || blockers.length === 0) return 'Checking on Aura';
    const lex = window.AuraShellLexicon;
    if (!lex) return 'Not fully ready';
    const summary = lex.summarize(blockers);
    return summary ? summary.title : 'Not fully ready';
}

function applyRuntimeHeartbeat(payload) {
    state.connected = true;
    const healthy = payloadRuntimeHealthy(payload);
    state.runtimeHealthy = healthy;
    state.runtimeHealthBlockers = runtimeHealthBlockers(payload);
    if (healthy) {
        showConnToast(false);
        if (state.conversationLane) {
            applyConversationLane(state.conversationLane, payload.runtime_status || payload.status || 'healthy');
        } else {
            setConnectionVisual('online');
        }
    } else {
        setConnectionVisual('degraded', runtimeHealthStatusText(payload));
    }
    publishHealthNeuralPulse(payload, 'websocket_heartbeat');
}

// One lane state, three renderings. The tier badge used to recover the
// state by string-comparing the *display* text
// (`laneText === 'cortex thinking' ? ...`), so any wording change
// silently collapsed every state into CORTEX WARMING. The state is a key
// now and the words hang off it, which is what made rewording these safe
// at all.
//
// `label`  header chip, Title case
// `inline` completes "Aura is …", lowercase
// `tier`   the technical badge, where our own vocabulary is fine
const LANE_STATES = {
    ready:        { label: 'Ready',              inline: 'ready',                     tier: 'CORTEX READY' },
    thinking:     { label: 'Thinking',           inline: 'thinking',                  tier: 'CORTEX THINKING' },
    memory_guard: { label: 'Low on memory',      inline: 'low on memory',             tier: 'CORTEX MEMORY GUARD' },
    route_blocked:{ label: 'Reply path blocked', inline: 'unable to route a reply',   tier: 'CORTEX ROUTE BLOCKED' },
    timeout:      { label: 'Took too long',      inline: 'taking longer than usual',  tier: 'CORTEX TIMEOUT' },
    unreachable:  { label: 'Not reachable',      inline: 'not reachable right now',   tier: 'CORTEX UNAVAILABLE' },
    preparing:    { label: 'Getting ready',      inline: 'getting ready',             tier: 'CORTEX PREPARING' },
    recovering:   { label: 'Recovering',         inline: 'recovering',                tier: 'CORTEX RECOVERING' },
    warming:      { label: 'Warming up',         inline: 'warming up',                tier: 'CORTEX WARMING' },
};

// "cortex" is our word for the process hosting Aura's model and "lane"
// is our word for the path a reply travels; both were leaking into the
// header, where they mean nothing to anyone who has not read the
// architecture doc. Same states, decided the same way — only the
// rendering moved.
function conversationLaneStateKey(lane) {
    if (!lane) return 'ready';
    const laneState = String(lane.state || 'warming').toLowerCase();
    const failureClass = laneFailureClass(lane);
    if (lane.conversation_ready) return 'ready';
    if (laneHasActiveGeneration(lane)) return 'thinking';
    if (failureClass === 'memory_guard') return 'memory_guard';
    if (failureClass === 'cognitive_engine') return 'route_blocked';
    if (failureClass === 'timeout') return 'timeout';
    if (failureClass === 'runtime_unavailable') return 'unreachable';
    if (laneIsStandby(lane)) return 'preparing';
    if (laneState === 'recovering') return 'recovering';
    if (laneState === 'failed') return 'unreachable';
    return 'warming';
}

function conversationLaneStatusText(lane) {
    return (LANE_STATES[conversationLaneStateKey(lane)] || LANE_STATES.warming).label;
}

function applyConversationLane(lane, healthStatus = '') {
    if (!lane || typeof lane !== 'object') return;

    const governedActionResult = lane.governed_action_result === true;
    const preservesHeartbeatLane = governedActionResult
        && lane.conversation_ready === false
        && state.runtimeHealthy === true
        && state.conversationLane
        && state.conversationLane.conversation_ready === true;
    const effectiveLane = preservesHeartbeatLane
        ? Object.assign({}, state.conversationLane, {
            governed_action_result: true,
            governed_action_status: lane.governed_action_status || state.conversationLane.governed_action_status,
            governed_action_completed_at: lane.governed_action_completed_at || Date.now() / 1000,
            governed_action_health_note: lane.governed_action_health_note || ''
        })
        : lane;

    state.conversationLane = effectiveLane;
    state.conversationReady = !!effectiveLane.conversation_ready;
    publishSurfaceWorkload('conversation_lane');
    updateLanePlaceholder();

    const laneKey = conversationLaneStateKey(effectiveLane);
    const laneWords = LANE_STATES[laneKey] || LANE_STATES.warming;
    const laneText = laneWords.label;
    const laneStandby = laneIsStandby(effectiveLane);
    const activeGeneration = laneHasActiveGeneration(effectiveLane);
    if (state.connected) {
        const healthy = laneHealthIsOperational(effectiveLane, healthStatus);
        const laneOperational = (state.conversationReady || activeGeneration) && healthy;
        const connectionMode = laneOperational ? 'online' : 'degraded';
        setConnectionVisual(connectionMode, !laneOperational ? laneText : '');
    }

    if (state.conversationReady && !activeGeneration && !state.isSubmitting) {
        updateTypingLabel('Aura is ready.');
        const typingInd = $('typing-ind');
        if (typingInd) typingInd.classList.remove('show');
        setChatPanelState('idle');

        if (state.chatSendQueue.length) drainQueuedChatMessages();
    } else {
        updateTypingLabel(
            state.conversationReady || activeGeneration
                ? 'Aura is thinking…'
                : `Aura is ${laneWords.inline}…`
        );
    }

    const tierEl = $('r-llm-tier');
    if (tierEl) {
        if (state.conversationReady) {
            const endpoint = effectiveLane.foreground_endpoint || effectiveLane.desired_endpoint || 'Cortex';
            tierEl.textContent = endpoint;
            tierEl.title = `Foreground: ${endpoint}`;
            tierEl.style.color = 'var(--success)';
        } else {
            tierEl.textContent = laneWords.tier;
            tierEl.title = laneStandby
                ? 'Local Cortex is not conversation-ready yet.'
                : effectiveLane.last_failure_reason || (effectiveLane.desired_model || 'Cortex (32B)');
            tierEl.style.color = activeGeneration
                ? 'var(--success)'
                : effectiveLane.state === 'failed' ? 'var(--error)' : 'var(--warn)';
        }
    }
}

// ── Health polling ───────────────────────────────────────
function nextHealthPollDelay() {
    const failures = Math.max(0, Number(state.healthPollFailures || 0));
    const base = failures > 0
        ? Math.min(HEALTH_POLL_MAX_MS, HEALTH_POLL_RETRY_BASE_MS * (2 ** Math.min(5, failures - 1)))
        : (document.hidden ? Math.min(HEALTH_POLL_MAX_MS, HEALTH_POLL_BASE_MS * 3) : HEALTH_POLL_BASE_MS);
    const jitter = base * HEALTH_POLL_JITTER_RATIO * ((Math.random() * 2) - 1);
    return Math.max(1000, Math.round(base + jitter));
}

function scheduleHealthPoll(delayMs = null) {
    if (state.healthPollTimer) clearTimeout(state.healthPollTimer);
    const delay = delayMs == null ? nextHealthPollDelay() : Math.max(0, Number(delayMs) || 0);
    state.healthPollTimer = setTimeout(() => {
        state.healthPollTimer = null;
        pollHealth();
    }, delay);
}

function healthFailureReason(error) {
    if (error && error.name === 'AbortError') return `timeout after ${HEALTH_POLL_TIMEOUT_MS}ms`;
    return error && error.message ? String(error.message) : 'unknown transport failure';
}

function recordHealthPollFailure(error) {
    const now = Date.now();
    const reason = healthFailureReason(error);
    state.healthPollFailures = Math.min(32, Number(state.healthPollFailures || 0) + 1);
    if (!state.healthPollIncident) {
        state.healthPollIncident = {
            id: `health_poll_${now}`,
            startedAt: now,
            failures: 1,
            lastReason: reason,
            lastReminderAt: now
        };
        queueNeuralLivenessCard(
            `[health_poll] health endpoint unavailable; retaining last known state (${reason})`,
            { level: 'warning', force: true }
        );
        return;
    }

    const incident = state.healthPollIncident;
    incident.failures += 1;
    incident.lastReason = reason;
    if (now - Number(incident.lastReminderAt || 0) >= HEALTH_POLL_REMINDER_MS) {
        incident.lastReminderAt = now;
        const elapsedSeconds = Math.max(1, Math.round((now - incident.startedAt) / 1000));
        queueNeuralLivenessCard(
            `[health_poll] endpoint incident continues after ${elapsedSeconds}s (${incident.failures} failed attempts); last known state retained`,
            { level: 'warning', force: true }
        );
    }
}

function recordHealthPollSuccess(payload) {
    const now = Date.now();
    const incident = state.healthPollIncident;
    state.healthPollFailures = 0;
    state.healthLastSuccessAt = now;
    state.healthPollIncident = null;
    if (!incident) return false;

    const elapsedSeconds = Math.max(1, Math.round((now - incident.startedAt) / 1000));
    const snapshotState = payload && payload.health_read_model
        ? String(payload.health_read_model.serving || 'available').replace(/_/g, ' ')
        : 'available';
    queueNeuralLivenessCard(
        `[health_poll] endpoint recovered after ${elapsedSeconds}s and ${incident.failures} failed attempts; snapshot ${snapshotState}`,
        { level: 'info', force: true }
    );
    return true;
}

function verifiedRuntimeRevision(payload) {
    const revision = payload?.runtime_revision;
    if (
        !revision
        || revision.schema !== 'aura.runtime_revision.v2'
        || revision.required !== true
        || revision.verified !== true
    ) {
        return '';
    }
    const token = String(revision.revision_token || '').toLowerCase();
    return /^[0-9a-f]{64}$/.test(token) ? token : '';
}

function runtimeRevisionPolicySatisfied(payload) {
    return runtimeRevisionPolicyBlocker(payload) === '';
}

function runtimeRevisionPolicyBlocker(payload) {
    const revision = payload?.runtime_revision;
    if (!Object.prototype.hasOwnProperty.call(payload || {}, 'runtime_revision')) {
        return state.runtimeRevisionTrust === 'untrusted'
            ? 'runtime_revision_unverified'
            : '';
    }
    if (!revision || revision.schema !== 'aura.runtime_revision.v2') {
        return 'runtime_revision_contract_missing';
    }
    if (revision.required === false) return '';
    if (revision.required !== true || !verifiedRuntimeRevision(payload)) {
        return 'runtime_revision_unverified';
    }
    return '';
}

function storedRuntimeRevisionRecord() {
    try {
        const raw = String(sessionStorage.getItem(RUNTIME_REVISION_STORAGE_KEY) || '');
        if (/^[0-9a-f]{64}$/i.test(raw)) {
            return {
                schema: RUNTIME_REVISION_RECORD_SCHEMA,
                revision: raw.toLowerCase(),
                generation: 0,
                capturedAtUnix: 0,
            };
        }
        const parsed = JSON.parse(raw);
        const revision = String(parsed?.revision || '').toLowerCase();
        const generation = Number(parsed?.generation || 0);
        const capturedAtUnix = Number(parsed?.captured_at_unix || 0);
        if (
            parsed?.schema !== RUNTIME_REVISION_RECORD_SCHEMA
            || !/^[0-9a-f]{64}$/.test(revision)
            || !Number.isInteger(generation)
            || generation < 0
            || !Number.isFinite(capturedAtUnix)
            || capturedAtUnix < 0
        ) {
            return null;
        }
        return { schema: parsed.schema, revision, generation, capturedAtUnix };
    } catch (_err) {
        return null;
    }
}

function storedRuntimeRevision() {
    return storedRuntimeRevisionRecord()?.revision || '';
}

function persistRuntimeRevision(revision, { generation = 0, capturedAtUnix = 0 } = {}) {
    const normalized = String(revision || '').toLowerCase();
    const normalizedGeneration = Number(generation || 0);
    const normalizedCapturedAt = Number(capturedAtUnix || 0);
    if (
        !/^[0-9a-f]{64}$/.test(normalized)
        || !Number.isInteger(normalizedGeneration)
        || normalizedGeneration < 0
        || !Number.isFinite(normalizedCapturedAt)
        || normalizedCapturedAt < 0
    ) {
        return false;
    }
    const record = {
        schema: RUNTIME_REVISION_RECORD_SCHEMA,
        revision: normalized,
        generation: normalizedGeneration,
        captured_at_unix: normalizedCapturedAt,
    };
    try {
        sessionStorage.setItem(RUNTIME_REVISION_STORAGE_KEY, JSON.stringify(record));
        const observed = storedRuntimeRevisionRecord();
        return Boolean(
            observed
            && observed.revision === normalized
            && observed.generation === normalizedGeneration
            && observed.capturedAtUnix === normalizedCapturedAt
        );
    } catch (_err) {
        return false;
    }
}

function reserveRuntimeRevisionReload(revision) {
    const normalized = String(revision || '').toLowerCase();
    if (!/^[0-9a-f]{64}$/.test(normalized)) return false;
    const inMemory = state.runtimeRevisionReloadAttempts || {};
    const memoryCount = Number(inMemory[normalized] || 0);
    try {
        const parsed = JSON.parse(
            String(sessionStorage.getItem(RUNTIME_REVISION_RELOAD_STORAGE_KEY) || '{}')
        );
        const priorCount = parsed?.revision === normalized
            ? Number(parsed?.count || 0)
            : 0;
        const count = Math.max(memoryCount, priorCount);
        if (count >= RUNTIME_REVISION_RELOAD_LIMIT) return false;
        const record = { revision: normalized, count: count + 1 };
        sessionStorage.setItem(
            RUNTIME_REVISION_RELOAD_STORAGE_KEY,
            JSON.stringify(record),
        );
        const observed = JSON.parse(
            String(sessionStorage.getItem(RUNTIME_REVISION_RELOAD_STORAGE_KEY) || '{}')
        );
        if (observed.revision !== normalized || Number(observed.count) !== record.count) {
            return false;
        }
        inMemory[normalized] = record.count;
        state.runtimeRevisionReloadAttempts = inMemory;
        return true;
    } catch (_err) {
        if (memoryCount >= RUNTIME_REVISION_RELOAD_LIMIT) return false;
        inMemory[normalized] = memoryCount + 1;
        state.runtimeRevisionReloadAttempts = inMemory;
        return true;
    }
}

function runtimeRevisionMarkerFromLocation() {
    try {
        const marker = new URL(window.location.href).searchParams.get('_aura_runtime') || '';
        return /^[0-9a-f]{64}$/.test(marker) ? marker : '';
    } catch (_err) {
        return '';
    }
}

function healthSnapshotRevisionEvidence(payload) {
    const metadata = payload?.health_read_model;
    const revision = verifiedRuntimeRevision(payload);
    const generation = Number(metadata?.snapshot_generation || 0);
    const capturedAtUnix = Number(metadata?.captured_at_unix || 0);
    if (
        !revision
        || !metadata
        // NOT `fresh === true`. The health read model serves
        // stale-while-revalidate — a 5s refresh interval against a 30s
        // max-stale window — so `fresh` is false for most of any given
        // second while `expired` stays false and the snapshot stays valid by
        // the server's own contract. Demanding freshness meant the shell
        // formed no revision evidence at all on a perfectly good snapshot,
        // so the reload path below could not run: measured live 2026-08-03
        // with fresh=false, stale=true, expired=false, age=13.9s. Bryan's
        // desktop window stayed open across four runtime restarts and three
        // revision-token changes, still executing the shell it had loaded
        // hours earlier, showing "Conversation lane initializing" while
        // /api/health reported conversation_ready: true.
        //
        // `expired` is the server's real "do not trust this" signal and is
        // still honoured. Monotonicity below still stops a snapshot from
        // walking the revision backwards.
        || metadata.expired === true
        || !Number.isInteger(generation)
        || generation <= 0
        || !Number.isFinite(capturedAtUnix)
        || capturedAtUnix <= 0
    ) {
        return null;
    }
    return { revision, generation, capturedAtUnix };
}

// ── Served-shell binding ──────────────────────────────────────────────
//
// runtime_revision.revision_token identifies the SIGNED APP BUNDLE, so it does
// not change when the source on disk changes — which is exactly when the HTML
// and JS an already-open window is running go stale. Measured live on
// 2026-08-03: Bryan's desktop window stayed open across four runtime restarts
// and kept executing the shell it had loaded hours earlier, so a UI fix that
// was deployed, served, and verified in a fresh tab was invisible to him. He
// was still looking at "Conversation lane initializing" while /api/health
// reported conversation_ready: true.
//
// The runtime already hashes what it is actually serving and reports it as
// actual_shell_assets_sha256. That is the fingerprint the open window has to
// be bound to.
const SERVED_SHELL_ASSETS_KEY = 'aura.servedShellAssets';
const SERVED_SHELL_MARKER_PARAM = '_aura_shell';
const SERVED_SHELL_MARKER_LENGTH = 16;

function servedShellAssetsFingerprint(payload) {
    const revision = payload?.runtime_revision;
    if (!revision || revision.schema !== 'aura.runtime_revision.v2') return '';
    const actual = String(revision.actual_shell_assets_sha256 || '').toLowerCase();
    return /^[0-9a-f]{64}$/.test(actual) ? actual : '';
}

function storedServedShellAssets() {
    try {
        return sessionStorage.getItem(SERVED_SHELL_ASSETS_KEY) || '';
    } catch (_err) {
        return state.servedShellAssets || '';
    }
}

function rememberServedShellAssets(fingerprint) {
    state.servedShellAssets = fingerprint;
    try {
        sessionStorage.setItem(SERVED_SHELL_ASSETS_KEY, fingerprint);
    } catch (_err) {
        // Storage denial must not pin the tab to stale bytes; the URL marker
        // below is the storage-independent half of the loop guard.
    }
}

function servedShellMarkerFromLocation() {
    try {
        return new URL(window.location.href).searchParams.get(SERVED_SHELL_MARKER_PARAM) || '';
    } catch (_err) {
        return '';
    }
}

function reconcileServedShellAssets(payload) {
    if (state.runtimeRevisionReloading) return false;
    const fingerprint = servedShellAssetsFingerprint(payload);
    if (!fingerprint) return false;

    const marker = fingerprint.slice(0, SERVED_SHELL_MARKER_LENGTH);
    // Two independent loop guards, because one reload that reloads again is
    // worse than a stale shell. The URL marker survives a storage denial; the
    // stored fingerprint survives a URL the launcher rewrote.
    if (servedShellMarkerFromLocation() === marker) {
        rememberServedShellAssets(fingerprint);
        return false;
    }
    const previous = storedServedShellAssets();
    if (!previous || previous === fingerprint) {
        rememberServedShellAssets(fingerprint);
        return false;
    }

    // Record the new fingerprint BEFORE navigating, so the reloaded page sees
    // itself as current even if the URL marker is stripped.
    rememberServedShellAssets(fingerprint);
    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.set(SERVED_SHELL_MARKER_PARAM, marker);
    return requestGuardedShellReload({ replaceUrl: nextUrl.toString() });
}

function healthSnapshotRevisionIsAuthoritative(payload) {
    const metadata = payload?.health_read_model;
    const generation = Number(metadata?.snapshot_generation || 0);
    const capturedAtUnix = Number(metadata?.captured_at_unix || 0);
    return Boolean(
        metadata
        && metadata.fresh === true
        && metadata.expired !== true
        && Number.isInteger(generation)
        && generation > 0
        && Number.isFinite(capturedAtUnix)
        && capturedAtUnix > 0
    );
}

function auraServiceWorkerRegistration(registration) {
    try {
        const scope = new URL(String(registration?.scope || ''));
        const workers = [registration?.active, registration?.waiting, registration?.installing];
        return scope.origin === window.location.origin
            && ['/', '/static/'].includes(scope.pathname)
            && workers.some((worker) => {
                const script = new URL(String(worker?.scriptURL || ''));
                return script.origin === window.location.origin
                    && script.pathname === '/static/service-worker.js';
            });
    } catch (_err) {
        return false;
    }
}

function retireRuntimeShellTrust(reason = 'runtime_revision_unverified') {
    if (state.runtimeShellRetirementPromise) return state.runtimeShellRetirementPromise;
    state.runtimeRevisionTrust = 'untrusted';
    state.runtimeRevision = null;
    state.runtimeRevisionGeneration = 0;
    state.runtimeRevisionCapturedAtUnix = 0;
    state.serviceWorkerRevision = null;
    state.serviceWorkerRegistrationTarget = null;
    state.serviceWorkerRegistrationPromise = null;
    state.serviceWorkerRegistrationEpoch = Number(state.serviceWorkerRegistrationEpoch || 0) + 1;
    try {
        sessionStorage.removeItem(RUNTIME_REVISION_STORAGE_KEY);
        sessionStorage.removeItem(RUNTIME_REVISION_RELOAD_STORAGE_KEY);
    } catch (_err) {}

    const retirement = (async () => {
        if ('serviceWorker' in navigator && typeof navigator.serviceWorker.getRegistrations === 'function') {
            let registrations = [];
            try {
                registrations = await navigator.serviceWorker.getRegistrations();
            } catch (err) {
                console.warn('[SW] unable to inventory registrations during trust retirement:', err);
            }
            await Promise.all((registrations || []).filter(auraServiceWorkerRegistration).map(async (registration) => {
                for (const worker of [registration.active, registration.waiting, registration.installing]) {
                    try {
                        worker?.postMessage?.({ type: 'AURA_RETIRE_RUNTIME_SHELL', reason });
                    } catch (_err) {}
                }
                try { await registration.unregister(); } catch (_err) {}
            }));
        }
        if (typeof caches !== 'undefined' && typeof caches.keys === 'function') {
            try {
                const keys = await caches.keys();
                await Promise.all(
                    keys
                        .filter(key => String(key).startsWith('aura-runtime-shell-'))
                        .map(key => caches.delete(key))
                );
            } catch (err) {
                console.warn('[SW] unable to purge retired runtime caches:', err);
            }
        }
        if (runtimeRevisionMarkerFromLocation()) {
            const nextUrl = new URL(window.location.href);
            nextUrl.searchParams.delete('_aura_runtime');
            requestGuardedShellReload({ replaceUrl: nextUrl.toString() });
        }
        return true;
    })().finally(() => {
        if (state.runtimeShellRetirementPromise === retirement) {
            state.runtimeShellRetirementPromise = null;
        }
    });
    state.runtimeShellRetirementPromise = retirement;
    return retirement;
}

function runtimeRevisionEvidenceIsMonotonic(evidence, previous) {
    if (!evidence || !previous) return Boolean(evidence);
    const previousCapturedAt = Number(previous.capturedAtUnix || 0);
    const previousGeneration = Number(previous.generation || 0);
    if (previousCapturedAt <= 0) return true;
    if (evidence.capturedAtUnix < previousCapturedAt) return false;
    if (evidence.capturedAtUnix > previousCapturedAt) return true;
    if (evidence.revision !== previous.revision) {
        return false;
    }
    return evidence.generation >= previousGeneration;
}

function serviceWorkerRegistrationIsCurrent(revision, epoch = null) {
    const target = String(state.serviceWorkerRegistrationTarget || '');
    if (target !== revision) return false;
    return epoch == null || Number(state.serviceWorkerRegistrationEpoch || 0) === epoch;
}

function serviceWorkerRevision(worker) {
    try {
        const revision = new URL(String(worker?.scriptURL || '')).searchParams.get('_aura_runtime') || '';
        return /^[0-9a-f]{64}$/.test(revision) ? revision : '';
    } catch (_err) {
        return '';
    }
}

function observeInstallingServiceWorker(worker, revision) {
    if (!worker || serviceWorkerRevision(worker) !== revision) return false;
    let observed = state.serviceWorkerInstallers;
    if (!observed || typeof observed.get !== 'function') {
        observed = new WeakMap();
        state.serviceWorkerInstallers = observed;
    }
    if (observed.get(worker) === revision) return true;
    observed.set(worker, revision);
    const activateIfInstalled = () => {
        if (
            worker.state === 'installed'
            && navigator.serviceWorker.controller
        ) {
            requestServiceWorkerActivation(worker, revision);
        }
    };
    activateIfInstalled();
    if (typeof worker.addEventListener === 'function') {
        worker.addEventListener('statechange', activateIfInstalled);
    }
    return true;
}

async function refreshServiceWorkerRegistration(registration, revision, epoch = null) {
    if (!registration || !revision) return null;
    const observeCurrentInstaller = () => {
        if (!serviceWorkerRegistrationIsCurrent(revision, epoch)) return;
        observeInstallingServiceWorker(registration.installing, revision);
    };
    if (typeof registration.addEventListener === 'function') {
        registration.addEventListener('updatefound', observeCurrentInstaller);
    }
    observeCurrentInstaller();
    if (registration.waiting && serviceWorkerRegistrationIsCurrent(revision, epoch)) {
        requestServiceWorkerActivation(registration.waiting, revision);
    }
    try {
        await registration.update();
    } catch (err) {
        console.warn('[SW] update() failed:', err);
    }
    observeCurrentInstaller();
    if (registration.waiting && serviceWorkerRegistrationIsCurrent(revision, epoch)) {
        requestServiceWorkerActivation(registration.waiting, revision);
    }
    return registration;
}

async function retireLegacyStaticServiceWorkers() {
    if (
        typeof navigator === 'undefined'
        || !('serviceWorker' in navigator)
        || typeof navigator.serviceWorker.getRegistrations !== 'function'
    ) {
        return 0;
    }
    let registrations;
    try {
        registrations = await navigator.serviceWorker.getRegistrations();
    } catch (err) {
        console.warn('[SW] registration inventory failed:', err);
        return 0;
    }
    let retired = 0;
    await Promise.all((registrations || []).map(async (registration) => {
        try {
            const scope = new URL(String(registration?.scope || ''));
            const script = new URL(String(
                registration?.active?.scriptURL
                || registration?.waiting?.scriptURL
                || registration?.installing?.scriptURL
                || '',
            ));
            if (
                scope.origin === window.location.origin
                && scope.pathname === '/static/'
                && script.origin === window.location.origin
                && script.pathname === '/static/service-worker.js'
                && typeof registration.unregister === 'function'
                && await registration.unregister()
            ) {
                retired += 1;
            }
        } catch (_err) {
            // Ignore malformed or cross-origin registrations; they are not Aura's.
        }
    }));
    return retired;
}

function registerRevisionServiceWorker(revision) {
    const normalized = String(revision || '').toLowerCase();
    if (
        !/^[0-9a-f]{64}$/.test(normalized)
        || typeof navigator === 'undefined'
        || !('serviceWorker' in navigator)
    ) {
        return Promise.resolve(null);
    }
    if (
        state.serviceWorkerRegistrationTarget === normalized
        && state.serviceWorkerRegistrationPromise
    ) {
        return state.serviceWorkerRegistrationPromise;
    }
    const now = Date.now();
    if (
        state.serviceWorkerRegistrationTarget === normalized
        && now < Number(state.serviceWorkerRegistrationRetryAt || 0)
    ) {
        return Promise.resolve(null);
    }
    if (state.serviceWorkerRegistrationTarget !== normalized) {
        state.serviceWorkerRegistrationFailures = 0;
        state.serviceWorkerRegistrationRetryAt = 0;
    }
    state.serviceWorkerRegistrationTarget = normalized;
    const epoch = Number(state.serviceWorkerRegistrationEpoch || 0) + 1;
    state.serviceWorkerRegistrationEpoch = epoch;
    const scriptUrl = `/static/service-worker.js?_aura_runtime=${normalized}`;
    const registrationPromise = retireLegacyStaticServiceWorkers().then(() => {
        if (!serviceWorkerRegistrationIsCurrent(normalized, epoch)) return null;
        return navigator.serviceWorker.register(
            scriptUrl,
            { scope: '/', updateViaCache: 'none' },
        );
    }).then((registration) => {
        if (!registration || !serviceWorkerRegistrationIsCurrent(normalized, epoch)) {
            return null;
        }
        return refreshServiceWorkerRegistration(registration, normalized, epoch);
    }).then((registration) => {
        if (registration && serviceWorkerRegistrationIsCurrent(normalized, epoch)) {
            state.serviceWorkerRevision = normalized;
            state.serviceWorkerRegistrationFailures = 0;
            state.serviceWorkerRegistrationRetryAt = 0;
        }
        return registration;
    }).catch((err) => {
        console.error('Service Worker failure:', err);
        if (serviceWorkerRegistrationIsCurrent(normalized, epoch)) {
            const failures = Number(state.serviceWorkerRegistrationFailures || 0) + 1;
            state.serviceWorkerRegistrationFailures = failures;
            state.serviceWorkerRegistrationRetryAt = Date.now() + Math.min(
                SERVICE_WORKER_REGISTRATION_RETRY_MAX_MS,
                500 * (2 ** Math.min(failures - 1, 6))
            );
            state.serviceWorkerRegistrationPromise = null;
        }
        return null;
    });
    state.serviceWorkerRegistrationPromise = registrationPromise;
    return registrationPromise;
}

function reconcileRuntimeShellRevision(payload) {
    const evidence = healthSnapshotRevisionEvidence(payload);
    if (state.runtimeRevisionReloading) return false;
    if (!evidence) {
        if (healthSnapshotRevisionIsAuthoritative(payload)) {
            // A direct/source launch has no signed revision token and never
            // will: the runtime says so with required === false, and the
            // server's own blocker treats that as no blocker at all. Calling
            // it 'untrusted' made every heartbeat WITHOUT a runtime_revision
            // key report `runtime_revision_unverified`, so the header badge
            // read RUNTIME_REVISION_UNVERIFIED instead of ONLINE while
            // /api/health simultaneously reported zero blockers. Absence of a
            // requirement is not a failed check.
            if (payload?.runtime_revision?.required === false) {
                state.runtimeRevisionTrust = 'not_required';
                return false;
            }
            const hadTrustedShell = Boolean(
                state.runtimeRevisionTrust === 'trusted'
                || state.runtimeRevision
                || storedRuntimeRevision()
                || runtimeRevisionMarkerFromLocation()
                || state.serviceWorkerRegistrationTarget
            );
            state.runtimeRevisionTrust = 'untrusted';
            if (hadTrustedShell) {
                void retireRuntimeShellTrust(runtimeRevisionPolicyBlocker(payload));
            }
        }
        return false;
    }

    const stored = storedRuntimeRevisionRecord();
    const prior = state.runtimeRevision
        ? {
            revision: state.runtimeRevision,
            generation: Number(state.runtimeRevisionGeneration || 0),
            capturedAtUnix: Number(state.runtimeRevisionCapturedAtUnix || 0),
        }
        : stored;
    if (!runtimeRevisionEvidenceIsMonotonic(evidence, prior)) return false;

    const { revision, generation, capturedAtUnix } = evidence;
    state.runtimeRevisionTrust = 'trusted';
    const previous = state.runtimeRevision || stored?.revision || '';
    state.runtimeRevisionGeneration = generation;
    state.runtimeRevisionCapturedAtUnix = capturedAtUnix;

    if (runtimeRevisionMarkerFromLocation() === revision) {
        state.runtimeRevision = revision;
        persistRuntimeRevision(revision, { generation, capturedAtUnix });
        registerRevisionServiceWorker(revision);
        return false;
    }

    if (!previous) {
        state.runtimeRevision = revision;
        persistRuntimeRevision(revision, { generation, capturedAtUnix });
        registerRevisionServiceWorker(revision);
        return false;
    }
    if (previous === revision) {
        state.runtimeRevision = revision;
        persistRuntimeRevision(revision, { generation, capturedAtUnix });
        registerRevisionServiceWorker(revision);
        return false;
    }

    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.set('_aura_runtime', revision);
    return requestGuardedShellReload({
        revision,
        generation,
        capturedAtUnix,
        replaceUrl: nextUrl.toString(),
    });
}

async function pollHealth() {
    if (state.healthPollInFlight) return;
    state.healthPollInFlight = true;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), HEALTH_POLL_TIMEOUT_MS);

    try {
        const res = await fetch('/api/health', {
            cache: 'no-store',
            signal: controller.signal
        });
        if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
        }
        const d = await res.json();
        if (!d || typeof d !== 'object') throw new Error('invalid health payload');
        if (reconcileRuntimeShellRevision(d)) return;
        // The signed-bundle token above does not move when the source does.
        if (reconcileServedShellAssets(d)) return;
        const recovered = recordHealthPollSuccess(d);
        state.runtimeHealthy = payloadRuntimeHealthy(d);
        state.runtimeHealthBlockers = runtimeHealthBlockers(d);
        const fmtPct01 = (value) => `${Math.round((value || 0) * 100)}%`;
        const runtimeAffect = (d.boot && d.boot.runtime && d.boot.runtime.affect)
            || (d.runtime && d.runtime.state && d.runtime.state.affect)
            || {};
        const liquidTelemetry = d.liquid_state || {};
        const homeostasisTelemetry = d.homeostasis || {};
        const homeostasisConfidence = homeostasisTelemetry.operational_confidence
            ?? homeostasisTelemetry.vitality
            ?? homeostasisTelemetry.will_to_live;
        updateTelemetry({
            energy: liquidTelemetry.energy ?? runtimeAffect.energy,
            curiosity: liquidTelemetry.curiosity ?? runtimeAffect.curiosity,
            frustration: liquidTelemetry.frustration ?? runtimeAffect.frustration,
            confidence: liquidTelemetry.confidence ?? (homeostasisConfidence != null ? homeostasisConfidence * 100 : runtimeAffect.stability),
            cpu_usage: d.cpu_usage,
            ram_usage: d.ram_usage,
            p_core_usage: d.cortex ? d.cortex.p_core_usage : null,
            vad: liquidTelemetry.vad || null,
        });
        if (d.conversation_lane) {
            applyConversationLane(d.conversation_lane, d.status || '');
        }
        if (d.boot || d.conversation_lane) {
            syncSplashState(d);
        }
        if (!recovered) publishHealthNeuralPulse(d, 'health_poll');

        state.cycleCount = d.cycle_count || state.cycleCount || 0;
        const cyclesEl = $('hud-cycles');
        if (cyclesEl) cyclesEl.textContent = state.cycleCount.toLocaleString();

        if (d.uptime != null) {
            const uptimeEl = $('hud-uptime');
            if (uptimeEl) uptimeEl.textContent = fmtUptime(d.uptime);
        }

        if (d.version) {
            const verEl = $('ui-ver');
            if (verEl) verEl.textContent = compactBuildLabel(d.build || d.version);
        }

        const cpuEl = $('hud-cpu');
        if (cpuEl) cpuEl.textContent = Math.round(d.cpu_usage || 0) + '%';

        setHudRamUsage(d.ram_usage, { source: 'health' });

        const pcoreEl = $('hud-pcore');
        const pcoreVal = d.cortex ? d.cortex.p_core_usage : 0;
        if (pcoreEl) pcoreEl.textContent = Math.round(pcoreVal || 0) + '%';

        // Vital labels change width as they fill in, which the container-level
        // ResizeObserver cannot see. Re-measure after each health poll.
        hudOverflow.schedule();

        if (d.cortex) {
            const c = d.cortex;
            const agencyEl = $('hud-agency');
            if (agencyEl && c.agency != null) agencyEl.textContent = c.agency;
            const hudCuriosity = $('hud-curiosity');
            if (c.curiosity != null && hudCuriosity) hudCuriosity.textContent = (c.curiosity || 0).toFixed(0) + '%';
            if (c.fixes != null) $('hud-fixes').textContent = c.fixes;
            if (c.beliefs != null) $('hud-beliefs').textContent = c.beliefs;
            if (c.episodes != null) $('hud-episodes').textContent = c.episodes;
            if (c.goals != null) $('hud-goals').textContent = c.goals;

            const updateStatus = (id, val) => {
                const el = $(id);
                if (!el) return;
                const span = el.querySelector('span');
                if (val) {
                    if (span) {
                        span.textContent = 'ON';
                        span.className = 'status-ok';
                    }
                    el.classList.remove('disabled');
                } else {
                    if (span) {
                        span.textContent = 'OFF';
                        span.className = 'status-err';
                    }
                    el.classList.add('disabled');
                }
            };
            updateStatus('hud-autonomy', c.autonomy);
            updateStatus('hud-stealth', c.stealth);
            updateStatus('hud-unity', c.unity);
            updateStatus('hud-scratchpad', c.scratchpad);
            updateStatus('hud-forge', c.forge);

            const subEl = $('hud-subconscious');
            if (subEl) {
                const subSpan = subEl.querySelector('span');
                subSpan.textContent = (c.subconscious || 'IDLE').toUpperCase();
                subSpan.className = c.subconscious === 'dreaming' ? 'status-ok pulsating' : 'status-ok';
            }
        }

        if (d.executive_closure) {
            const ex = d.executive_closure;
            const closureScore = Number(ex.closure_score || 0);
            const needPressure = Number(ex.need_pressure || 0);

            if ($('hud-closure')) $('hud-closure').textContent = fmtPct01(closureScore);
            if ($('c-closure')) $('c-closure').textContent = fmtPct01(closureScore);
            if ($('exec-need')) $('exec-need').textContent = String(ex.dominant_need || '--').toUpperCase();
            if ($('exec-pressure')) $('exec-pressure').textContent = fmtPct01(needPressure);
            if ($('exec-objective')) $('exec-objective').textContent = ex.selected_objective || 'Awaiting imperative.';
            if ($('exec-focus')) $('exec-focus').textContent = ex.attention_focus || 'Internal monitoring.';
        }

        if (d.interaction_signals) {
            state.interactionSignals = d.interaction_signals;
        }

        if (d.consciousness_evidence) {
            const ev = d.consciousness_evidence;
            if ($('hud-readiness')) $('hud-readiness').textContent = fmtPct01(ev.enterprise_readiness || 0);
            if ($('e-reliability')) $('e-reliability').textContent = fmtPct01((ev.dimensions && ev.dimensions.reliability) || 0);
            if ($('e-subjectivity')) $('e-subjectivity').textContent = fmtPct01(ev.subjectivity_evidence || 0);
            if ($('e-enterprise')) $('e-enterprise').textContent = fmtPct01(ev.enterprise_readiness || 0);
            if ($('e-assessment')) $('e-assessment').textContent = ev.assessment || 'Operational evidence pending.';
        }

        if (d.executive_authority) {
            const auth = d.executive_authority;
            if ($('exec-authority')) $('exec-authority').textContent = `${String(auth.last_action || 'idle').toUpperCase()} · ${String(auth.last_reason || 'steady').replace(/_/g, ' ')}`;
            if ($('exec-released')) $('exec-released').textContent = `${auth.primary_releases || 0}/${auth.secondary_releases || 0}`;
            if ($('exec-suppressed')) $('exec-suppressed').textContent = String(auth.suppressed || 0);
        }

        if (d.soma) {
            const s = d.soma;
            updateGauge('s-thermal', pct01(s.thermal_load), 's-thermal-val');
            updateGauge('s-anxiety', pct01(s.resource_anxiety), 's-anxiety-val');
            updateGauge('s-vitality', pct01(s.vitality), 's-vitality-val');
        }

        if (d.homeostasis) {
            // No `?? 0` tail: if none of the three report, the gauge must say
            // "unknown" rather than assert full-confidence zero.
            const homeostasisConfidenceGauge = d.homeostasis.operational_confidence
                ?? d.homeostasis.vitality
                ?? d.homeostasis.will_to_live;
            updateGauge('g-integrity', pct01(d.homeostasis.integrity), 'g-integrity-val');
            updateGauge('g-persistence', pct01(d.homeostasis.persistence), 'g-persistence-val');
            updateGauge('g-confidence', pct01(homeostasisConfidenceGauge), 'g-confidence-val');
        }

        if (d.moral) {
            updateGauge('s-moral', pct01(d.moral.integrity), 's-moral-val');
        }

        if (d.social) {
            updateGauge('s-social', pct01(d.social.depth), 's-social-val');
        }

        if (d.swarm) {
            const swarmEl = $('c-swarm');
            if (swarmEl) swarmEl.textContent = d.swarm.active_count || 0;
        }

        // ── Phase III: Qualitative State Engine ──
        if (d.qualia) {
            const q = d.qualia;
            const dimEl = $('q-dim');
            const attEl = $('q-attractor');
            setTelemetryValue('q-pri', q.pri, { digits: 3 });
            setTelemetryValue('q-norm', q.q_norm, { digits: 3 });
            if (dimEl) dimEl.textContent = (q.dominant_dim || TELEMETRY_UNKNOWN).toUpperCase();
            if (attEl) {
                attEl.textContent = q.in_attractor ? 'LOCKED' : 'FLUID';
                attEl.style.color = q.in_attractor ? 'var(--success)' : 'var(--accent)';
            }
            if ($('q-identity')) {
                // Previously `|| 100`: an absent field asserted perfect identity
                // coherence, the most flattering possible lie.
                setTelemetryValue('q-identity', q.identity_coherence, { digits: 1, suffix: '%' });
                $('q-identity').style.color = (q.identity_coherence > 90) ? 'var(--success)' : 'var(--accent)';
            }
        }

        // ── Phase III: Resilience Matrix ──
        if (d.resilience) {
            const r = d.resilience;
            const tierEl = $('r-llm-tier');
            const snapEl = $('r-snapshot');
            const sttEl = $('r-stt');
            const ttsEl = $('r-tts');

            if (tierEl) {
                if (!(state.conversationLane && state.conversationReady === false)) {
                    // Show the active endpoint name if available, otherwise fall back to tier
                    const epName = r.active_endpoint || '';
                    const tierLabel = epName || (r.llm_tier || 'unknown').toUpperCase();
                    tierEl.textContent = tierLabel;
                    tierEl.title = epName ? `Tier: ${r.llm_tier || '?'} | Endpoint: ${epName}` : '';
                    tierEl.style.color = r.llm_tier === 'local' ? 'var(--success)' :
                                         r.llm_tier === 'local_fast' ? '#ff8800' :
                                         r.llm_tier === 'api_deep' ? '#00aaff' :
                                         r.llm_tier === 'emergency' ? 'var(--error)' : 'var(--success)';
                }
            }
            if (snapEl) {
                snapEl.textContent = (r.snapshot || '--').toUpperCase();
                snapEl.style.color = r.snapshot === 'saved' ? 'var(--success)' : '#888';
            }

            const breakers = r.circuit_breakers || {};
            if (sttEl) {
                const sttState = breakers['STT'] || breakers['stt'] || 'CLOSED';
                sttEl.textContent = sttState.toUpperCase();
                sttEl.style.color = sttState === 'CLOSED' ? 'var(--success)' :
                                    sttState === 'HALF_OPEN' ? '#ff8800' : 'var(--error)';
            }
            if (ttsEl) {
                const ttsState = breakers['TTS'] || breakers['tts'] || 'CLOSED';
                ttsEl.textContent = ttsState.toUpperCase();
                ttsEl.style.color = ttsState === 'CLOSED' ? 'var(--success)' :
                                    ttsState === 'HALF_OPEN' ? '#ff8800' : 'var(--error)';
            }
            if ($('r-hardening')) {
                const h = r.hardening_active;
                $('r-hardening').textContent = h ? 'ACTIVE' : 'INACTIVE';
                $('r-hardening').style.color = h ? 'var(--success)' : '#888';
            }
        }

        // ── Full Runtime / Autonomous Initiative ──
        if (d.full_runtime) {
            const fr = d.full_runtime;
            const components = (fr.components && typeof fr.components === 'object') ? fr.components : {};
            const initiative = components.autonomous_initiative || {};
            const admission = (initiative.admission && typeof initiative.admission === 'object') ? initiative.admission : {};
            const background = (fr.background_cognition && typeof fr.background_cognition === 'object') ? fr.background_cognition : {};
            const blockers = Array.isArray(fr.blockers) ? fr.blockers : [];
            const setFullRuntimeCell = (id, value, healthy) => {
                const el = $(id);
                if (!el) return;
                el.textContent = String(value || '--').toUpperCase();
                el.style.color = healthy ? 'var(--success)' : 'var(--error)';
                el.title = blockers.length ? `Blocked: ${blockers.join(', ')}` : 'Full runtime organ is healthy.';
            };
            const fullProfile = ['full_desktop', 'protected_full_desktop'].includes(String(fr.profile || ''));
            setFullRuntimeCell('fr-profile', fr.profile || '--', fullProfile);
            setFullRuntimeCell('fr-ready', fr.ready ? 'READY' : 'BLOCKED', !!fr.ready);
            if ($('fr-background')) {
                const backgroundEnabled = !!background.enabled && !!background.active;
                const admissionState = String(background.work_admission || '').toLowerCase();
                const backgroundLabel = backgroundEnabled
                    ? (admissionState === 'deferred' ? 'DEFERRED' : 'ACTIVE')
                    : 'BLOCKED';
                $('fr-background').textContent = backgroundLabel;
                $('fr-background').style.color = backgroundEnabled ? 'var(--success)' : 'var(--error)';
                const running = background.running_required_count ?? '--';
                const total = background.registered_required_count ?? '--';
                const reason = background.work_defer_reason || background.loop_start_reason || blockers.join(', ') || 'none';
                $('fr-background').title = backgroundEnabled
                    ? `Background cognition live: ${running}/${total} required organs running; work admission ${background.work_admission || 'allowed'}${reason && reason !== 'none' ? ` (${reason})` : ''}.`
                    : `Background cognition blocked: ${reason}.`;
            }
            setFullRuntimeCell('fr-initiative', initiative.running ? 'ACTIVE' : 'BLOCKED', !!initiative.running);
            if ($('fr-selfdev')) {
                const selfDev = admission.self_development || '--';
                $('fr-selfdev').textContent = String(selfDev).toUpperCase();
                $('fr-selfdev').style.color = selfDev === 'allowed' ? 'var(--success)' : 'var(--warn)';
                $('fr-selfdev').title = 'Autonomous self-development admission state.';
            }
            if ($('fr-social')) {
                const social = admission.social || '--';
                $('fr-social').textContent = String(social).toUpperCase();
                $('fr-social').style.color = social === 'allowed' ? 'var(--success)' : 'var(--warn)';
                $('fr-social').title = 'Autonomous social/world presence admission state.';
            }
        }

        // ── Phase III: Mycelial Network ──
        if (d.mycelial) {
            const m = d.mycelial;
            const healthEl = $('m-health');
            const nodesEl = $('m-nodes');
            const edgesEl = $('m-edges');
            if (healthEl) {
                healthEl.textContent = (m.health || 'OFFLINE').toUpperCase();
                healthEl.style.color = m.health === 'online' ? 'var(--success)' : 'var(--error)';
            }
            if (nodesEl) nodesEl.textContent = m.nodes || 0;
            if (edgesEl) edgesEl.textContent = m.edges || 0;
        }

        // ── PNEUMA Engine ──
        if (d.pneuma) {
            const p = d.pneuma;
            const pnOnline = $('pn-online');
            if (pnOnline) {
                pnOnline.textContent  = p.online ? 'ONLINE' : 'OFFLINE';
                pnOnline.style.color  = p.online  ? 'var(--success)' : 'var(--error)';
            }
            // `|| 0.7` invented a plausible default temperature for a field that
            // was never reported; unknown now reads as unknown.
            setTelemetryValue('pn-temp', p.temperature, { digits: 3 });
            setTelemetryValue('pn-arousal', p.arousal, { digits: 3 });
            setTelemetryValue('pn-stability', p.stability, { digits: 3 });
            setTelemetryValue('pn-attractors', p.attractor_count);
        }

        // ── MHAF Field ──
        if (d.mhaf) {
            const mh = d.mhaf;
            const mhOnline = $('mhaf-online');
            if (mhOnline) {
                mhOnline.textContent  = mh.online ? 'ONLINE' : 'OFFLINE';
                mhOnline.style.color  = mh.online  ? 'var(--success)' : 'var(--error)';
            }
            setTelemetryValue('mhaf-phi', mh.phi, { digits: 4 });
            setTelemetryValue('mhaf-nodes', mh.nodes);
            setTelemetryValue('mhaf-edges', mh.edges);
            setTelemetryValue('mhaf-lexicon', mh.lexicon_size);
        }

        // ── Security ──
        if (d.security) {
            const sec = d.security;
            if ($('sec-trust')) {
                $('sec-trust').textContent = (sec.trust_level || 'guest').toUpperCase();
                const trustColors = {sovereign:'var(--success)', trusted:'var(--success)', guest:'#aaa', suspicious:'var(--warn)', hostile:'var(--error)'};
                $('sec-trust').style.color = trustColors[sec.trust_level] || '#aaa';
            }
            if ($('sec-threat')) {
                $('sec-threat').textContent = (sec.threat_score || 0).toFixed(2);
                $('sec-threat').style.color = (sec.threat_score || 0) > 0.4 ? 'var(--error)' : '#aaa';
            }
            if ($('sec-integrity')) {
                $('sec-integrity').textContent = sec.integrity_ok !== false ? 'OK' : 'ALERT';
                $('sec-integrity').style.color = sec.integrity_ok !== false ? 'var(--success)' : 'var(--error)';
            }
            if ($('sec-passphrase')) {
                $('sec-passphrase').textContent = sec.passphrase_set ? 'SET' : 'UNSET';
                $('sec-passphrase').style.color = sec.passphrase_set ? 'var(--success)' : 'var(--warn)';
            }
        }

        // ── Circadian State ──
        if (d.circadian) {
            const ci = d.circadian;
            if ($('circ-phase'))   $('circ-phase').textContent   = (ci.phase || '--').toUpperCase();
            if ($('circ-arousal')) $('circ-arousal').textContent = (ci.arousal_baseline || 0).toFixed(2);
            if ($('circ-mode'))    $('circ-mode').textContent    = (ci.cognitive_mode || '--').toUpperCase();
            if ($('circ-energy'))  $('circ-energy').textContent  = (ci.energy_modifier || 0).toFixed(2) + 'x';
        }

        // ── Substrate Learning ──
        if (d.substrate) {
            const lb = d.substrate.lora_bridge || {};
            if ($('lora-captured'))  $('lora-captured').textContent  = lb.capture_count || 0;
            if ($('lora-flushed'))   $('lora-flushed').textContent   = lb.total_flushed || 0;
            if ($('lora-quality'))   $('lora-quality').textContent   = (lb.avg_quality || 0).toFixed(2);
            if ($('lora-buffer'))    $('lora-buffer').textContent    = lb.buffer_size || 0;
        }

        // ── Identity Narrative ──
        if (d.consolidator) {
            const co = d.consolidator;
            if ($('consol-version')) $('consol-version').textContent = 'v' + (co.version || 0);
            if ($('consol-traits'))  $('consol-traits').textContent  = co.traits || 0;
            if ($('consol-age'))     $('consol-age').textContent     = co.age_hours != null ? co.age_hours.toFixed(1) + 'h' : '--';
            if (co.signature && $('identity-narrative')) {
                $('identity-narrative').textContent = co.signature;
            }
        }

        // ── Phase III: Transcendence ──
        if (d.cortex) {
            const c = d.cortex;
            if ($('c-singularity')) $('c-singularity').textContent = (c.singularity_factor || 1.0).toFixed(1) + 'x';
            if ($('c-meta-loop')) {
                $('c-meta-loop').textContent = c.meta_loop_active ? 'ACTIVE' : 'IDLE';
                $('c-meta-loop').style.color = c.meta_loop_active ? 'var(--success)' : '#888';
            }
        }

        if (d.runtime) {
            updateTelemetry(d.runtime);
        }

        // Fallback or explicit mapping for CPU/RAM metrics to the UI
        if (d.cpu_usage != null) {
            const cpuEl = $('hud-cpu');
            if (cpuEl) cpuEl.textContent = d.cpu_usage + '%';
        } else if (d.runtime && d.runtime.cpu_percent != null) {
            const cpuEl = $('hud-cpu');
            if (cpuEl) cpuEl.textContent = d.runtime.cpu_percent + '%';
        }

        if (d.ram_usage != null) {
            setHudRamUsage(d.ram_usage, { source: 'health' });
        } else if (d.runtime && d.runtime.memory_percent != null) {
            setHudRamUsage(d.runtime.memory_percent, { source: 'health' });
        }

        if (d.privacy && (!state._privacyLockUntil || Date.now() > state._privacyLockUntil)) {
            const p = d.privacy;
            const muteBtn = $('btn-mute');
            const camBtn = $('btn-cam');
            if (muteBtn) {
                const voiceEnabled = p.microphone_enabled !== false && p.speaking_enabled !== false;
                if (voiceEnabled) {
                    muteBtn.classList.remove('disabled');
                    muteBtn.innerHTML = '<span>● MUTE</span>';
                } else {
                    muteBtn.classList.add('disabled');
                    muteBtn.innerHTML = '<span>● MUTED</span>';
                }
                // Remove old onclick if exists and add new listener
                const muteHandler = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    togglePrivacy('microphone', voiceEnabled, muteBtn);
                };
                muteBtn.removeEventListener('click', muteBtn._clickHandler);
                muteBtn._clickHandler = muteHandler;
                muteBtn.addEventListener('click', muteHandler);
                const duplexActive = Boolean(
                    window.AuraVoiceMode
                    && typeof window.AuraVoiceMode.isActive === 'function'
                    && window.AuraVoiceMode.isActive()
                );
                if (!voiceEnabled && (state.voiceActive || duplexActive)) {
                    void toggleVoice(false);
                }
            }
            if (camBtn) {
                if (p.camera_enabled !== false) {
                    camBtn.classList.remove('disabled');
                    camBtn.innerHTML = '<span>● CAM</span>';
                } else {
                    camBtn.classList.add('disabled');
                    camBtn.innerHTML = '<span>● CAM OFF</span>';
                }
                // Remove old onclick if exists and add new listener
                const camHandler = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    togglePrivacy('camera', p.camera_enabled, camBtn);
                };
                camBtn.removeEventListener('click', camBtn._clickHandler);
                camBtn._clickHandler = camHandler;
                camBtn.addEventListener('click', camHandler);
                if (p.camera_enabled === false) {
                    state.cameraSignalWanted = false;
                    stopCameraSignals();
                }
            }
        }

        if (d.desktop_access) {
            applyDesktopAccessSummary(d.desktop_access);
        }

        refreshMetricGuide();
    } catch (e) {
        console.warn('⚠️ Health poll failed:', e);
        recordHealthPollFailure(e);
    } finally {
        clearTimeout(timeoutId);
        state.healthPollInFlight = false;
        scheduleHealthPoll();
    }
}

async function togglePrivacy(type, currentEnabled, btn) {
    try {
        const next = !currentEnabled;
        // Optimistic UI: update button immediately
        state._privacyLockUntil = Date.now() + 3000; // Lock pollHealth from resetting for 3s
        if (type === 'camera' && !next) {
            state.cameraSignalWanted = false;
            stopCameraSignals();
        }
        const duplexActive = Boolean(
            window.AuraVoiceMode
            && typeof window.AuraVoiceMode.isActive === 'function'
            && window.AuraVoiceMode.isActive()
        );
        if (type === 'microphone' && !next && (state.voiceActive || duplexActive)) {
            void toggleVoice(false);
        }
        if (btn) {
            if (next) {
                btn.classList.remove('disabled');
                btn.innerHTML = type === 'microphone' ? '<span>● MUTE</span>' : '<span>● CAM</span>';
            } else {
                btn.classList.add('disabled');
                btn.innerHTML = type === 'microphone' ? '<span>● MUTED</span>' : '<span>● CAM OFF</span>';
            }
        }
        const res = await fetch(`/api/privacy/${type}`, {
            method: 'POST',
            headers: auraDesktopHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ enabled: next })
        });
        const d = await res.json();
        if (d.ok) {
            // Privacy toggle applied
            // Update the click handler to reflect new state
            if (btn) {
                const handler = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    togglePrivacy(type, next, btn);
                };
                btn.removeEventListener('click', btn._clickHandler);
                btn._clickHandler = handler;
                btn.addEventListener('click', handler);
            }
            if (type === 'camera') {
                state.cameraSignalWanted = !!next;
                if (next) {
                    await startCameraSignals();
                } else {
                    stopCameraSignals();
                }
            }
        } else {
            // Revert on failure
            state._privacyLockUntil = 0;
            pollHealth();
        }
    } catch (e) {
        console.error('Privacy toggle failed:', e);
        state._privacyLockUntil = 0;
        pollHealth();
    }
}

// ── Telemetry honesty ─────────────────────────────────────
// A gauge fed `field || 0` renders a missing subsystem as a confident 0%,
// which is indistinguishable from a real measured zero. These helpers keep
// "unknown" distinct from "zero" so the panel cannot claim a reading it does
// not have. The markup already ships `--` placeholders for exactly this state.
const TELEMETRY_UNKNOWN = '--';

function telemetryKnown(value) {
    return value != null && Number.isFinite(Number(value));
}

/** Scale a 0..1 field to a percentage, preserving unknown instead of collapsing to 0. */
function pct01(value) {
    return telemetryKnown(value) ? Number(value) * 100 : null;
}

/** Write a numeric telemetry value, or the unknown state when it is absent. */
function setTelemetryValue(id, value, { digits = 0, suffix = '' } = {}) {
    const el = $(id);
    if (!el) return;
    const known = telemetryKnown(value);
    el.textContent = known ? Number(value).toFixed(digits) + suffix : TELEMETRY_UNKNOWN;
    el.classList.toggle('telemetry-unknown', !known);
}

function updateGauge(id, val, textId) {
    const bar = $(id);
    const text = $(textId);
    const known = telemetryKnown(val);
    const pct = known ? Math.min(100, Math.max(0, Number(val))) : 0;
    if (bar) {
        bar.style.width = pct + '%';
        bar.classList.toggle('gauge-unknown', !known);
    }
    if (text) {
        text.textContent = known ? pct.toFixed(0) + '%' : TELEMETRY_UNKNOWN;
        text.classList.toggle('telemetry-unknown', !known);
    }
}

function fmtUptime(sec) {
    if (sec < 60) return Math.round(sec) + 's';
    if (sec < 3600) return Math.floor(sec / 60) + 'm' + Math.round(sec % 60) + 's';
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return h + 'h' + m + 'm';
}

// The runtime reports version as "Aura Luna v2026.4.20-Zenith" — the name is
// already in the wordmark next to the chip, so drop a leading identity name and
// keep the build. Falls back to the raw string when it carries no name prefix.
function compactBuildLabel(raw) {
    let label = String(raw == null ? '' : raw).trim();
    if (!label) return '';
    const name = String(state.identityName || 'Aura Luna').trim();
    if (name) {
        const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        label = label.replace(new RegExp('^' + escaped + '[\\s\\u00b7:-]*', 'i'), '').trim();
    }
    return label || String(raw).trim();
}

// ── Skills ───────────────────────────────────────────────
async function loadSkills() {
    if (!accessCapabilityAllowed('tools_catalog')) return;
    try {
        if (state.toolCatalog && state.toolCatalog.length) {
            renderToolCatalog(state.toolCatalog, state.toolCatalogHealth || null);
        }

        let tools = [];
        try {
            const res = await fetch('/api/tools/catalog', { cache: 'no-store' });
            const contentType = res.headers.get('content-type') || '';
            if (res.ok && contentType.includes('application/json')) {
                const d = await res.json();
                tools = Array.isArray(d.tools) ? d.tools : [];
                state.toolCatalogHealth = d.health && typeof d.health === 'object' ? d.health : {};
            }
        } catch (err) {
            console.warn('[Tools] catalog fetch failed; trying legacy skills endpoint:', err);
            tools = [];
        }

        if (!tools.length) {
            const legacyRes = await fetch('/api/skills', { cache: 'no-store' });
            const contentType = legacyRes.headers.get('content-type') || '';
            if (!legacyRes.ok || !contentType.includes('application/json')) {
                throw new Error('skills_endpoint_unavailable');
            }
            const d = await legacyRes.json();
            tools = Array.isArray(d.catalog) ? d.catalog : Array.isArray(d.skills) ? d.skills : [];
            state.toolCatalogHealth = d.health && typeof d.health === 'object' ? d.health : {};
        }
        renderToolCatalog(tools, state.toolCatalogHealth || null);
    } catch (e) {
        console.warn('[Tools] load failed:', e);
        const list = $('skills-list');
        if (list && !(state.toolCatalog && state.toolCatalog.length)) {
            renderRetryPanel(list, 'Failed to load tools', 'RETRY', () => loadSkills());
        }
    }
}

// ── Learning & Growth ────────────────────────────────────
async function loadLearningStatus() {
    if (!accessCapabilityAllowed('learning_status')) return;
    const put = (id, value) => { const el = $(id); if (el) el.textContent = value; };
    try {
        const res = await fetch('/api/system/learning', { cache: 'no-store' });
        const contentType = res.headers.get('content-type') || '';
        if (!res.ok || !contentType.includes('application/json')) {
            throw new Error('learning_endpoint_unavailable');
        }
        const d = await res.json();
        const lineage = (d.compounding && d.compounding.lineage) || {};
        const selfplay = d.selfplay || {};
        const store = d.preference_store || {};

        const generations = Number(lineage.generations || 0);
        put('learn-generations', String(generations));
        put('learn-promoted', `${Number(lineage.promoted || 0)}/${generations}`);

        const attempts = Number(selfplay.total_attempts || 0);
        const correct = Number(selfplay.total_correct || 0);
        put('learn-practice-rate', attempts > 0 ? `${Math.round((correct / attempts) * 100)}%` : '--');
        put('learn-pairs', String(Number(store.total_pairs || selfplay.total_pairs || 0)));

        const verdict = String(lineage.verdict || '').replace(/_/g, ' ').toLowerCase();
        const lastStatus = (d.compounding && d.compounding.last_status) || 'never attempted';
        const bursts = Number(selfplay.bursts || 0);
        const parts = [];
        parts.push(generations > 0
            ? `Ledger verdict: ${verdict || 'unknown'} · last cycle ${lastStatus}`
            : `Weight loop armed — no training generation yet (last: ${lastStatus})`);
        parts.push(bursts > 0
            ? `${bursts} idle practice burst${bursts === 1 ? '' : 's'}, ${attempts} verified attempts`
            : 'no idle practice yet — practice runs only when the lane is quiet');
        put('learn-detail', parts.join(' · '));
    } catch (e) {
        console.warn('[Learning] status load failed:', e);
        put('learn-detail', 'Learning status unavailable — the runtime may still be booting.');
    }
}

// ── Memory ───────────────────────────────────────────────
function normalizeGoalStatus(status) {
    return String(status || 'queued').trim().toLowerCase().replace(/-/g, '_');
}

function goalStatusClass(status) {
    const normalized = normalizeGoalStatus(status);
    if (normalized === 'completed') return 'success';
    if (normalized === 'failed' || normalized === 'abandoned') return 'error';
    if (normalized === 'blocked' || normalized === 'paused') return 'warn';
    return 'info';
}

function formatGoalTimestamp(ts) {
    const value = Number(ts);
    if (!Number.isFinite(value) || value <= 0) return '';
    const date = new Date(value * 1000);
    if (Number.isNaN(date.getTime())) return '';
    return date.toLocaleString([], {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function renderGoalItem(item) {
    const objective = String(item.objective || item.description || item.goal || item.name || '').trim();
    const status = normalizeGoalStatus(item.status);
    const horizon = String(item.horizon || 'short_term').trim().toLowerCase().replace(/-/g, '_');
    const priority = Number(item.priority);
    const source = String(item.source || '').trim();
    const progressRaw = Number(item.progress);
    const progress = Number.isFinite(progressRaw)
        ? Math.max(0, Math.min(100, Math.round(progressRaw * 100)))
        : (item.steps_total ? Math.round(((Number(item.steps_done || 0) / Number(item.steps_total || 1)) * 100)) : 0);
    const stepsTotal = Number(item.steps_total || 0);
    const stepsDone = Number(item.steps_done || 0);
    const summary = String(item.summary || item.success_criteria || '').trim();
    const updatedAt = formatGoalTimestamp(item.completed_at || item.updated_at || item.started_at || item.created_at);
    const metaBits = [];
    if (Number.isFinite(priority) && priority > 0) metaBits.push(`Priority ${priority.toFixed(2)}`);
    if (source) metaBits.push(source.replace(/_/g, ' '));
    if (stepsTotal > 0) metaBits.push(`${stepsDone}/${stepsTotal} steps`);
    if (updatedAt) metaBits.push(updatedAt);

    return `
        <div class="mem-item goal-card">
            <div class="goal-head">
                <strong>${escHtml(objective || 'Untitled goal')}</strong>
                <div class="goal-tags">
                    <span class="tag ${goalStatusClass(status)}">${escHtml(status.replace(/_/g, ' '))}</span>
                    <span class="tag">${escHtml(horizon.replace(/_/g, ' '))}</span>
                    ${item.quick_win ? '<span class="tag info">quick win</span>' : ''}
                </div>
            </div>
            ${summary ? `<div class="goal-summary-text">${escHtml(summary)}</div>` : ''}
            ${progress > 0 || stepsTotal > 0 ? `
                <div class="goal-progress">
                    <div class="goal-progress-bar" style="width:${progress}%;"></div>
                </div>
            ` : ''}
            ${metaBits.length ? `<div class="goal-meta">${escHtml(metaBits.join(' • '))}</div>` : ''}
        </div>
    `;
}

function renderGoalSection(title, items, emptyText = '') {
    if (!items.length) {
        return emptyText
            ? `<div class="goal-group"><div class="goal-group-title">${escHtml(title)}</div><div class="mem-item goal-empty-inline">${escHtml(emptyText)}</div></div>`
            : '';
    }
    return `
        <div class="goal-group">
            <div class="goal-group-title">${escHtml(title)}</div>
            ${items.map(renderGoalItem).join('')}
        </div>
    `;
}

function renderGoalMemory(items, summary = {}) {
    const normalized = Array.isArray(items) ? items.filter(item => item && typeof item === 'object') : [];
    const shortActive = [];
    const longActive = [];
    const completed = [];
    const failed = [];

    normalized.forEach(item => {
        const status = normalizeGoalStatus(item.status);
        const horizon = String(item.horizon || 'short_term').trim().toLowerCase().replace(/-/g, '_');
        if (status === 'completed') {
            completed.push(item);
        } else if (status === 'failed' || status === 'abandoned') {
            failed.push(item);
        } else if (horizon === 'long_term') {
            longActive.push(item);
        } else {
            shortActive.push(item);
        }
    });

    const stats = [];
    if (summary.in_progress_count != null) stats.push(`In Progress ${summary.in_progress_count}`);
    if (summary.queued_count != null) stats.push(`Queued ${summary.queued_count}`);
    if (summary.completed_count != null) stats.push(`Completed ${summary.completed_count}`);
    if (summary.blocked_count) stats.push(`Blocked ${summary.blocked_count}`);

    return `
        ${stats.length ? `<div class="goal-summary">${stats.map(stat => `<span class="goal-summary-stat">${escHtml(stat)}</span>`).join('')}</div>` : ''}
        ${renderGoalSection('Short-Term Queue', shortActive, 'No short-term goals are active.')}
        ${renderGoalSection('Long-Term Queue', longActive, 'No long-term goals are active.')}
        ${renderGoalSection('Completed', completed, 'No completed goals have been recorded yet.')}
        ${failed.length ? renderGoalSection('Failed / Abandoned', failed) : ''}
    `;
}

async function loadMemory(type) {
    try {
        const endpoint = `/api/memory/${type || 'episodic'}?limit=20`;
        const res = await fetch(endpoint);
        if (!res.ok) throw new Error(`Memory fetch failed (${res.status})`);
        const d = await res.json();
        const cont = $('mem-content');
        const items = d.items || [];
        if (items.length === 0) {
            cont.innerHTML = `<div class="mem-empty">${memoryKindSigil(type)}` +
                `<span>No ${escHtml(type)} memories yet</span></div>`;
            return;
        }
        cont.innerHTML = items.map(item => {
            if (typeof item === 'object' && item !== null) {
                if (type === 'episodic') {
                    const ts = item.timestamp ? new Date(item.timestamp * 1000).toLocaleTimeString([], {hour12: false}) : '';
                    const ctx = item.context || item.action || '';
                    const outcome = item.outcome || '';
                    const badge = item.success === false ? '<span class="tag error">FAILED</span> ' : (item.success === true ? '<span class="tag success">OK</span> ' : '');
                    return `<div class="mem-item">${badge}<span class="mem-ts">${ts}</span> <strong>${escHtml(ctx)}</strong><br><span class="mem-detail">${escHtml(outcome)}</span></div>`;
                } else if (type === 'semantic') {
                    // LIVE DEFECT, 2026-07-27. This read item.key/item.subject
                    // and item.value/item.predicate. /api/memory/semantic
                    // returns {id, content, metadata, timestamp} and never has
                    // any of those four fields, so every row rendered as
                    // "<strong></strong>: " — the panel showed eight boxes
                    // each containing a bare colon.
                    //
                    // Same disease as the context assembler's "spontaneous:"
                    // prefix: a consumer's hand-written field list drifted
                    // from what the producer actually emits, and the symptom
                    // was silent blankness rather than an error.
                    const content = item.content || item.text || '';
                    const key = item.key || item.subject || '';
                    const val = item.value || item.predicate || '';
                    const meta = (item.metadata && typeof item.metadata === 'object')
                        ? item.metadata : {};
                    const source = meta.source || meta.origin || '';
                    let body;
                    if (content) {
                        body = `${escHtml(String(content))}`;
                        if (source) {
                            body += `<br><span class="mem-detail">${escHtml(String(source))}</span>`;
                        }
                    } else if (key || val) {
                        body = `<strong>${escHtml(key)}</strong>: ${escHtml(String(val))}`;
                    } else {
                        // Nothing renderable. Drop the row instead of drawing
                        // an empty box — a panel that says "no memories" is
                        // honest, and eight blank boxes are not.
                        return '';
                    }
                    return `<div class="mem-item">${body}</div>`;
                } else if (type === 'goals') {
                    return '';
                }
            }
            return `<div class="mem-item">${escHtml(String(item))}</div>`;
        }).join('');
        if (type !== 'goals' && !cont.innerHTML.trim()) {
            // Items arrived but none of them had anything to show. That is a
            // producer/consumer mismatch, not an empty memory — say so
            // rather than presenting a blank panel as a normal state.
            console.warn('[Memory] %d %s items had no renderable fields', items.length, type);
            cont.innerHTML = `<div class="mem-empty">${memoryKindSigil(type)}` +
                `<span>No ${escHtml(type)} memories yet</span></div>`;
            return;
        }
        if (type === 'goals') {
            cont.innerHTML = renderGoalMemory(items, d.summary || {});
            return;
        }
    } catch (e) {
        console.warn('[Memory] load failed:', e);
        const cont = $('mem-content');
        if (cont) {
            renderRetryPanel(cont, 'Failed to load memories', 'RETRY', () => loadMemory(state.activeMem));
        }
        showBriefNotification('Memory load failed — check connection');
    }
}

// ── Belief Graph ─────────────────────────────────────────
let graphNetwork = null;
function updateBeliefGraphTheme(theme, accent) {
    if (!graphNetwork) return;
    const isLight = theme === 'light';
    const isHighContrast = theme === 'high-contrast';
    const isMidnight = theme === 'midnight';

    let fontColor = '#e0e0e0';
    let strokeColor = '#05030a';
    let borderColor = '#00e5ff';
    let bgColor = '#8a2be2';
    let edgeColor = 'rgba(138, 43, 226, 0.5)';
    let highlightBorder = '#ff00ff';
    let highlightBg = '#ffffff';

    if (isLight) {
        fontColor = '#14111c';
        strokeColor = '#f4ede4';
        borderColor = '#8a2be2';
        bgColor = '#b1a4ff';
        edgeColor = 'rgba(138, 43, 226, 0.25)';
        highlightBorder = '#8a2be2';
        highlightBg = '#ffffff';
    } else if (isHighContrast) {
        fontColor = '#ffffff';
        strokeColor = '#000000';
        borderColor = '#ffd400';
        bgColor = '#161616';
        edgeColor = '#4f4f4f';
        highlightBorder = '#ffffff';
        highlightBg = '#ffd400';
    } else if (isMidnight) {
        fontColor = '#d8d4e8';
        strokeColor = '#030005';
        borderColor = '#00e5ff';
        bgColor = '#5b46c8';
        edgeColor = 'rgba(0, 229, 255, 0.3)';
    }

    graphNetwork.setOptions({
        nodes: {
            font: {
                color: fontColor,
                strokeColor: strokeColor
            },
            color: {
                border: borderColor,
                background: bgColor,
                highlight: {
                    border: highlightBorder,
                    background: highlightBg
                }
            }
        },
        edges: {
            color: {
                color: edgeColor,
                highlight: borderColor
            }
        }
    });
}

function initBeliefGraph() {
    if (state.beliefGraphInit) return;
    state.beliefGraphInit = true;

    const container = $('belief-graph') || $('belief-graph-container');
    if (!container) return;
    const data = { nodes: new vis.DataSet([]), edges: new vis.DataSet([]) };
    const options = {
        nodes: {
            shape: 'dot',
            scaling: { min: 10, max: 30 },
            font: {
                color: '#e0e0e0',
                size: 12,
                face: "'Space Mono', monospace",
                strokeWidth: 2,
                strokeColor: '#05030a' // matches --bg
            },
            borderWidth: 2,
            color: {
                border: '#00e5ff',
                background: '#8a2be2',
                highlight: { border: '#ff00ff', background: '#ffffff' }
            },
            shadow: {
                enabled: true,
                color: 'rgba(0, 229, 255, 0.8)',
                size: 15,
                x: 0,
                y: 0
            }
        },
        edges: {
            color: { color: 'rgba(138, 43, 226, 0.5)', highlight: '#00e5ff' },
            width: 1.5,
            smooth: { type: 'dynamic' }
        },
        physics: {
            stabilization: { iterations: 150 },
            barnesHut: {
                gravitationalConstant: -3500,
                centralGravity: 0.2,
                springLength: 120,
                springConstant: 0.04
            }
        },
        interaction: { hover: true, tooltipDelay: 200 }
    };
    graphNetwork = new vis.Network(container, data, options);
    refreshKnowledgeGraph();
    if (typeof settings !== 'undefined') {
        updateBeliefGraphTheme(settings.theme, settings.accent);
    }
}

async function refreshKnowledgeGraph() {
    if (state.knowledgeGraphPollInFlight) return;
    state.knowledgeGraphPollInFlight = true;
    try {
        const res = await fetch('/api/knowledge/graph');
        if (!res.ok) throw new Error(`Knowledge graph fetch failed (${res.status})`);
        const d = await res.json();
        if (d.nodes && graphNetwork) {
            graphNetwork.setData({
                nodes: new vis.DataSet(d.nodes),
                edges: new vis.DataSet(d.edges || [])
            });
        }
    } catch (e) {
        console.warn('[KnowledgeGraph] Failed to refresh:', e.message || e);
        showBriefNotification('Knowledge graph unavailable');
    } finally {
        state.knowledgeGraphPollInFlight = false;
    }
    scheduleKnowledgeGraphPoll();
}

function scheduleKnowledgeGraphPoll(delayMs = null) {
    if (state.knowledgeGraphTimer) clearTimeout(state.knowledgeGraphTimer);
    state.knowledgeGraphTimer = null;
    if (state.activeTab !== 'telemetry') return;
    const delay = delayMs == null
        ? optionalSurfacePollDelay(KNOWLEDGE_GRAPH_POLL_MS, {
            foregroundFactor: 3,
            hiddenFactor: 8,
        })
        : Math.max(0, Number(delayMs) || 0);
    state.knowledgeGraphTimer = setTimeout(() => {
        state.knowledgeGraphTimer = null;
        if (document.hidden) {
            scheduleKnowledgeGraphPoll();
            return;
        }
        void refreshKnowledgeGraph();
    }, delay);
}

// ── Header buttons ───────────────────────────────────────
// Use addEventListener for reliable click handling

const muteBtn = $('btn-mute');
if (muteBtn) muteBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    togglePrivacy('microphone', true, muteBtn);
});

const camBtn = $('btn-cam');
if (camBtn) camBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    togglePrivacy('camera', true, camBtn);
});

const brainBtn = $('btn-brain');
if (brainBtn) brainBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    e.stopPropagation();
    brainBtn.style.opacity = '0.5';
    brainBtn.textContent = '◇ ...';
    try {
        const res = await fetch('/api/brain/retry', {
            method: 'POST',
            headers: auraDesktopHeaders(),
        });
        const d = await res.json();
        appendMsg('aura', d.status === 'retry_sent' ? '🧠 Brain retry signal sent.' : '⚠ Orchestrator unavailable.');
    } catch (e) {
        appendMsg('aura', '⚠ Failed to contact brain retry endpoint.');
    } finally {
        brainBtn.style.opacity = '1';
        brainBtn.textContent = '◇ BRAIN';
    }
});

const apkBtn = $('btn-apk');
if (apkBtn) apkBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    appendMsg('aura', '📱 APK not available yet — Aura runs as a web app at this URL.');
});

const srcBtn = $('btn-src');
if (srcBtn) srcBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    e.stopPropagation();
    srcBtn.style.opacity = '0.5';
    srcBtn.textContent = '↓ ...';
    try {
        const res = await fetch('/api/source');
        if (res.ok) {
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'aura_source.txt';
            a.click();
            URL.revokeObjectURL(url);
            appendMsg('aura', '📦 Source bundle downloaded.');
        } else {
            appendMsg('aura', '⚠ Source download failed: ' + res.status);
        }
    } catch (e) {
        appendMsg('aura', '⚠ Source download error.');
    } finally {
        srcBtn.style.opacity = '1';
        srcBtn.textContent = '↓ SRC';
    }
});

const updateBtn = $('btn-update');
if (updateBtn) updateBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    e.stopPropagation();
    updateBtn.style.opacity = '0.5';
    updateBtn.textContent = '↻ ...';
    appendMsg('aura', '♻️ Hot-reloading Aura code from disk...');
    try {
        const res = await fetch('/api/system/hot-reload', {
            method: 'POST',
            headers: auraDesktopHeaders(),
        });
        if (res.ok) {
            const data = await res.json();
            const reloaded = data.reloaded_count || 0;
            const scope = data.scope || 'all';
            const unmatched = data.unmatched_prefixes || [];
            const failed = data.failed_count || 0;
            // "All changes are live" was said unconditionally, including when
            // a scope pointed at a module that does not exist and therefore
            // reloaded nothing. Say what actually happened instead.
            let msg = `♻️ Hot-reload: ${reloaded} module${reloaded === 1 ? '' : 's'} refreshed (scope: ${scope}).`;
            if (failed) msg += ` ${failed} failed to reload.`;
            if (unmatched.length) {
                msg += ` ${unmatched.length} declared scope entr${unmatched.length === 1 ? 'y' : 'ies'} matched no module and reloaded nothing: ${unmatched.slice(0, 4).join(', ')}.`;
            }
            msg += (failed || unmatched.length)
                ? ' Not every change is live — restart to pick up the rest.'
                : ' Modules held by a running instance (routes, the inference gate, loaded models) still need a restart.';
            appendMsg('aura', msg);
        } else {
            const text = await res.text();
            appendMsg('aura', `⚠️ Hot-reload returned ${res.status}: ${text.slice(0, 200)}`);
        }
    } catch (e) {
        appendMsg('aura', '❌ Hot-reload request failed — is the server running?');
    } finally {
        updateBtn.style.opacity = '1';
        updateBtn.textContent = '↻ UPDATE';
    }
});

const soulBtn = $('btn-soul');
if (soulBtn) soulBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const overlay = $('soul-overlay');
    const frame = $('soul-frame');
    if (overlay && frame) {
        overlay.classList.add('visible');
        frame.src = '/static/mycelial.html';
        // Soul Map opened
    }
});

const soulCloseBtn = $('soul-close');
if (soulCloseBtn) soulCloseBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const overlay = $('soul-overlay');
    const frame = $('soul-frame');
    overlay.classList.remove('visible');
    frame.src = '';  // Stop the 3D renderer to save GPU
});

const memMapBtn = $('btn-mem-map');
if (memMapBtn) memMapBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const overlay = $('soul-overlay');
    const frame = $('soul-frame');
    if (overlay && frame) {
        overlay.classList.add('visible');
        frame.src = '/memory';
        // Memory Map opened
    }
});

// Long-horizon campaigns had no way in. Mission Control existed, worked, and
// was reachable only by typing /static/mission_control.html into the address
// bar — which is to say, only by someone who had read the source. It opens in
// the same overlay the other maps use.
const campaignBtn = $('btn-campaign');
if (campaignBtn) campaignBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const overlay = $('soul-overlay');
    const frame = $('soul-frame');
    if (overlay && frame) {
        overlay.classList.add('visible');
        frame.src = '/static/mission_control.html';
    }
});

const termBtn = $('btn-term');
if (termBtn) termBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    e.stopPropagation();
    const modal = $('terminal-modal');
    if (!modal) return;
    modal.style.display = 'flex';
    // Refresh status
    try {
        const r = await fetch('/api/terminal/status');
        const d = await r.json();
        const watchdogEl = $('term-watchdog');
        const activeEl   = $('term-active');
        const pendingEl  = $('term-pending');
        if (watchdogEl) {
            watchdogEl.textContent = d.watchdog_running ? 'MONITORING' : 'OFFLINE';
            watchdogEl.style.color = d.watchdog_running ? 'var(--success)' : 'var(--error)';
        }
        if (activeEl) {
            activeEl.textContent  = d.active ? 'ACTIVE' : 'STANDBY';
            activeEl.style.color  = d.active  ? 'var(--success)' : '#888';
        }
        if (pendingEl) pendingEl.textContent = d.pending_messages || 0;
    } catch (e) { console.warn('Terminal status fetch failed', e); }
});

const termCloseBtn = $('term-close-btn');
if (termCloseBtn) termCloseBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const modal = $('terminal-modal');
    if (modal) modal.style.display = 'none';
});

const termSendBtn = $('term-send-btn');
if (termSendBtn) termSendBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    e.stopPropagation();
    const input = $('term-msg-input');
    const result = $('term-send-result');
    if (!input || !input.value.trim()) return;
    try {
        const r = await fetch('/api/terminal/send', {
            method: 'POST',
            headers: auraDesktopHeaders({'Content-Type': 'application/json'}),
            body: JSON.stringify({text: input.value.trim()})
        });
        const d = await r.json();
        if (result) {
            result.textContent = d.ok ? `✓ Queued: "${d.queued}"` : `✗ ${d.error}`;
            result.style.color = d.ok ? 'var(--success)' : 'var(--error)';
        }
        if (d.ok) { input.value = ''; if ($('term-pending')) $('term-pending').textContent = parseInt($('term-pending').textContent || '0') + 1; }
    } catch (e) {
        console.warn('[Terminal] Send failed:', e);
        if (result) { result.textContent = '✗ Request failed'; result.style.color = 'var(--error)'; }
    }
});

const rebootBtn = $('btn-reboot');
if (rebootBtn) rebootBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (confirm('Reboot Aura? This will restart the server process.')) {
        try {
            await fetch('/api/reboot', {
                method: 'POST',
                headers: auraDesktopHeaders(),
            });
        } catch (e) {
            console.warn('[System] Reboot request failed:', e);
            appendMsg('aura', '⚠ Reboot request failed before it reached the server.', false, { diagnostic: true });
        }
    }
});

// ── Voice toggle ─────────────────────────────────────────
// One place that owns how the voice control looks, so no caller has to reach
// into the button's children and no caller can delete them.
function setMicButtonState(mode) {
    const btn = document.getElementById('mic-btn');
    if (!btn) return;
    const listening = mode === 'listening';
    btn.classList.toggle('active', listening);
    btn.setAttribute('aria-pressed', listening ? 'true' : 'false');
    btn.title = listening ? 'Stop the voice conversation' : 'Talk to Aura out loud';
    btn.setAttribute('aria-label', listening ? 'Stop voice conversation' : 'Start voice conversation');
    const glyph = document.getElementById('mic-glyph');
    const stop = document.getElementById('stop-icon');
    if (glyph) glyph.classList.toggle('hidden', listening);
    if (stop) stop.classList.toggle('hidden', !listening);
    const label = btn.querySelector('.mic-label');
    if (label) label.textContent = listening ? 'STOP' : 'VOICE';
}

async function toggleVoice(desiredState = null, { quiet = false } = {}) {
    const duplex = window.AuraVoiceMode;
    const active = Boolean(duplex && typeof duplex.isActive === 'function' && duplex.isActive());
    const targetState = typeof desiredState === 'boolean' ? desiredState : !active;
    if (targetState === active) return true;
    if (targetState && state.voiceSummary && state.voiceSummary.available === false) {
        if (!quiet) appendMsg('aura', '⚠ Voice channel is currently unavailable.');
        return false;
    }
    if (
        targetState
        && runtimeSettingsState.hydrated
        && runtimeSettingsState.values['voice.input_enabled'] !== true
    ) {
        if (!quiet) appendMsg('aura', '⚠ Microphone input is disabled in Runtime Settings.');
        return false;
    }
    if (!duplex || typeof duplex.enter !== 'function' || typeof duplex.exit !== 'function') {
        if (!quiet) appendMsg('aura', 'Voice mode did not load. Reload Aura and try again.');
        return false;
    }
    try {
        const started = targetState ? await duplex.enter() : (await duplex.exit(), false);
        state.voiceActive = targetState ? Boolean(started && duplex.isActive()) : false;
        $('voice-orb-wrap').classList.toggle('active', state.voiceActive);
        $('voice-orb').className = state.voiceActive ? 'voice-orb listening' : 'voice-orb';
        setMicButtonState(state.voiceActive ? 'listening' : 'idle');
        return targetState ? state.voiceActive : !duplex.isActive();
    } catch (err) {
        console.error('Voice mode transition failed:', err);
        state.voiceActive = false;
        $('voice-orb-wrap').classList.remove('active');
        $('voice-orb').className = 'voice-orb';
        setMicButtonState('idle');
        if (!quiet) appendMsg('aura', 'Voice mode could not start.');
        return false;
    }
}

/**
 * Fold a finished voice conversation back into the text thread.
 *
 * Called by voice_mode.js on exit. Without this, everything said out loud
 * vanishes when the surface closes, and the visible history disagrees with
 * what Aura actually remembers — which then reads as her confabulating.
 */
window.auraAppendVoiceTranscript = function (lines) {
    if (!Array.isArray(lines) || !lines.length) return;
    for (const line of lines) {
        if (!line || !line.text) continue;
        appendMsg(line.who === 'user' ? 'user' : 'aura', line.text);
    }
};

/**
 * Put a spoken turn into the chat thread as it happens.
 *
 * Called by voice_mode.js while an ambient session is running. The three
 * functions below are deliberately the *same* rendering path a typed turn
 * takes — same bubble, same markdown, same history, same scroll behaviour.
 *
 * That sameness is the entire feature. A spoken conversation that lives in
 * its own panel and gets folded back in at the end is two conversations that
 * agree afterwards; this is one conversation that happens to have been
 * spoken. It is also what stops the visible history from disagreeing with
 * what she remembers, which is the thing that reads as her confabulating.
 */
window.auraAppendVoiceTurn = function (who, text) {
    if (!text) return;
    // A reply may still be streaming when the next thing is said — she can be
    // interrupted. Close the open bubble first so the thread stays ordered.
    if (activeStreamDiv) finishStreamMsg();
    appendMsg(who === 'user' ? 'user' : 'aura', String(text));
};

window.auraStreamVoiceReply = function (chunk, isFirst) {
    if (!chunk) return;
    if (isFirst || !activeStreamDiv) {
        if (activeStreamDiv) finishStreamMsg();
        startStreamMsg('aura');
        const ind = $('typing-ind');
        if (ind) ind.classList.remove('show');
    }
    // Clauses arrive already spaced as speech; the stream buffer is plain
    // text, so the join has to be explicit or the words run together.
    appendStreamChunk((activeStreamContentRaw ? ' ' : '') + String(chunk));
};

window.auraFinishVoiceReply = function () {
    if (activeStreamDiv) finishStreamMsg();
};

/**
 * Reconcile the visible spoken reply with what reached the speakers.
 *
 * Voice chunks are mirrored into chat when synthesis starts, not when their
 * samples finish playing. On a barge-in the tail may therefore be visible
 * even though it was flushed before the listener heard it. The server sends
 * the rendered prefix measured from client playback; make that prefix the
 * visible record too, so chat, memory, and the room agree.
 */
window.auraInterruptVoiceReply = function (spoken) {
    if (!activeStreamDiv) return;
    activeStreamContentRaw = '';
    appendStreamChunk(String(spoken || ''));
    activeStreamDiv.classList.add('voice-interrupted');
    activeStreamDiv.setAttribute('aria-label', 'Aura was interrupted');
    finishStreamMsg();
};

const micBtn = $('mic-btn');
if (micBtn) micBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    // The duplex surface is the only browser microphone owner.
    if (window.AuraVoiceMode && typeof window.AuraVoiceMode.toggle === 'function') {
        window.AuraVoiceMode.toggle();
        return;
    }
    void toggleVoice();
});

// Heartbeat is handled by the 25s pingInterval in connect()

// ── Service Worker (PWA Support) ─────────────────────────
if ('serviceWorker' in navigator) {
    let swReloadTriggered = false;
    let swHadController = Boolean(navigator.serviceWorker.controller);
    navigator.serviceWorker.addEventListener('controllerchange', () => {
        const expectedRevision = state.runtimeRevision || storedRuntimeRevision();
        const activeRevision = serviceWorkerRevision(navigator.serviceWorker.controller);
        if (!expectedRevision || activeRevision !== expectedRevision) return;
        if (!swHadController) {
            swHadController = true;
        }
        if (swReloadTriggered) return;
        const nextUrl = new URL(window.location.href);
        nextUrl.searchParams.set('_aura_runtime', expectedRevision);
        swReloadTriggered = requestGuardedShellReload({
            revision: expectedRevision,
            generation: Number(state.runtimeRevisionGeneration || 0),
            capturedAtUnix: Number(state.runtimeRevisionCapturedAtUnix || 0),
            replaceUrl: nextUrl.toString(),
        });
    });

    window.addEventListener('load', () => {
        void retireLegacyStaticServiceWorkers();
    });
}

function startProfileBoundFeatures() {
    const startOnce = (name, fn) => {
        if (state.profileFeaturesStarted.has(name)) return;
        state.profileFeaturesStarted.add(name);
        Promise.resolve()
            .then(fn)
            .catch(err => console.warn(`[UI] ${name} startup failed:`, err));
    };
    if (accessCapabilityAllowed('desktop_control')) {
        startOnce('desktop_access', async () => {
            await pollDesktopAccess();
            scheduleDesktopAccessPoll();
        });
    }
    if (accessCapabilityAllowed('voice_stream')) {
        startOnce('voice_stream', () => voicePlayer.init());
    }
    if (accessCapabilityAllowed('tools_catalog')) {
        startOnce('tools_catalog', () => loadSkills());
    }
    if (accessCapabilityAllowed('learning_status')) {
        startOnce('learning_status', () => loadLearningStatus());
    }
}

// ── Start ────────────────────────────────────────────────
setConnectionVisual('booting');
hydrateBootstrap({ hydrateConversationHistory: true, quiet: true });
initializeMetricGuide();
renderNeuralFeedMode();
if (DOM.neuralPauseToggle) {
    DOM.neuralPauseToggle.addEventListener('click', toggleNeuralVisualPause);
}
if (DOM.neuralReadableToggle) {
    DOM.neuralReadableToggle.addEventListener('click', toggleNeuralReadableMode);
}
connect();
pollHealth();
vadStream = new VADStream('neural-vad-canvas');
publishSurfaceWorkload('startup');
scheduleBootstrapPoll();

window.addEventListener('aura:workload-mode', () => {
    if (state.bootstrapTimer) scheduleBootstrapPoll();
    if (state.desktopAccessTimer) scheduleDesktopAccessPoll();
    if (state.knowledgeGraphTimer) scheduleKnowledgeGraphPoll();
});

// ── Settings & Preferences ────────────────────────────────
const SETTINGS_KEY = 'aura_settings';
const defaultSettings = {
    theme: 'dark', accent: 'violet', onboarded: false, cheatStatus: 'IDLE',
    neuralPaused: false, chatTextSize: 'standard', neuralTextSize: 'standard'
};

function loadSettings() {
    try {
        const saved = localStorage.getItem(SETTINGS_KEY);
        return saved ? { ...defaultSettings, ...JSON.parse(saved) } : { ...defaultSettings };
    } catch (err) {
        console.warn('[Settings] Failed to load saved settings; using defaults:', err);
        return { ...defaultSettings };
    }
}

function saveSettings(s) {
    try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
    } catch (err) {
        console.warn('[Settings] Failed to persist settings:', err);
    }
}

function applySettings(s) {
    // Theme
    document.body.className = document.body.className
        .replace(/theme-\w+/g, '')
        .replace(/accent-\w+/g, '')
        .replace(/chat-text-\w+/g, '')
        .replace(/neural-text-\w+/g, '')
        .replace(/\bneural-visual-paused\b/g, '')
        .replace(/\s+/g, ' ')
        .trim();
    if (s.theme !== 'dark') document.body.classList.add(`theme-${s.theme}`);
    if (s.accent !== 'violet') document.body.classList.add(`accent-${s.accent}`);
    document.body.classList.add(`chat-text-${s.chatTextSize || 'standard'}`);
    document.body.classList.add(`neural-text-${s.neuralTextSize || 'standard'}`);
    document.body.classList.toggle('neural-visual-paused', !!s.neuralPaused);

    // Sync UI controls
    const el = (id) => document.getElementById(id);
    if (el('setting-theme')) el('setting-theme').value = s.theme;
    if (el('setting-accent')) el('setting-accent').value = s.accent;
    if (el('setting-neural-paused')) el('setting-neural-paused').checked = !!s.neuralPaused;
    if (el('setting-chat-text-size')) el('setting-chat-text-size').value = s.chatTextSize || 'standard';
    if (el('setting-neural-text-size')) el('setting-neural-text-size').value = s.neuralTextSize || 'standard';
    if (el('setting-cheat-status')) el('setting-cheat-status').textContent = s.cheatStatus || 'IDLE';
    if (el('setting-version')) el('setting-version').textContent = state.version;

    state.neuralFeedPaused = !!s.neuralPaused;
    if (state.neuralFeedPaused) {
        state.pacingActive = false;
        clearTimeout(state.thoughtDrainTimer);
        state.thoughtDrainTimer = null;
    }
    syncNeuralFeedMode();
    if (!state.neuralFeedPaused && state.thoughtQueue.length > 0 && !state.pacingActive) {
        processThoughtQueue();
    }
    if (graphNetwork) {
        updateBeliefGraphTheme(s.theme, s.accent);
    }
}

const settings = loadSettings();
applySettings(settings);

// Bind settings controls
['setting-theme', 'setting-accent', 'setting-neural-paused', 'setting-chat-text-size',
 'setting-neural-text-size'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('change', () => {
        const key = id.replace('setting-', '').replace(/-([a-z])/g, (_, c) => c.toUpperCase());
        settings[key] = el.type === 'checkbox' ? el.checked :
                        el.type === 'range' ? parseFloat(el.value) : el.value;
        saveSettings(settings);
        applySettings(settings);
    });
});

const RUNTIME_SETTING_CONTROLS = Object.freeze({
    'voice.input_enabled': { id: 'setting-voice-input', kind: 'boolean' },
    'voice.output_enabled': { id: 'setting-voice-output', kind: 'boolean' },
    'voice.auto_listen': { id: 'setting-autolisten', kind: 'boolean' },
    'voice.output_rate': { id: 'setting-tts-speed', kind: 'number' },
    'learning.auto_enrichment_enabled': { id: 'setting-enrichment', kind: 'boolean' },
    'learning.reflection_enabled': { id: 'setting-reflection', kind: 'boolean' },
    'governance.approval_mode': { id: 'setting-approval', kind: 'string' },
});

const runtimeSettingsState = {
    revision: null,
    values: {},
    hydrated: false,
    hydrationPromise: null,
};

function setRuntimeSettingsStatus(message, tone = 'pending') {
    const status = $('settings-runtime-status');
    if (!status) return;
    status.textContent = String(message || '');
    status.dataset.tone = tone;
}

function setRuntimeSettingsAvailability(enabled, message = '', tone = 'pending') {
    const allowControls = !!enabled && !state.conversationOnly;
    Object.values(RUNTIME_SETTING_CONTROLS).forEach(definition => {
        const control = $(definition.id);
        if (control) control.disabled = !allowControls;
    });
    if (message) setRuntimeSettingsStatus(message, tone);
}

function runtimeControlValue(definition, control) {
    if (definition.kind === 'boolean') return !!control.checked;
    if (definition.kind === 'number') return Number.parseFloat(control.value);
    return String(control.value);
}

function applyRuntimeSettingsControls(values) {
    Object.entries(RUNTIME_SETTING_CONTROLS).forEach(([key, definition]) => {
        const control = $(definition.id);
        if (!control || !Object.prototype.hasOwnProperty.call(values, key)) return;
        if (definition.kind === 'boolean') control.checked = values[key] === true;
        else control.value = String(values[key]);
    });
    const rate = Number(values['voice.output_rate']);
    const rateValue = $('setting-tts-speed-value');
    if (rateValue && Number.isFinite(rate)) rateValue.textContent = `${rate.toFixed(1)}×`;
    const autonomyStatus = $('setting-autonomy-status');
    if (autonomyStatus) autonomyStatus.textContent = 'ACTIVE';
    if (state.voiceSummary) applyVoiceSummary(state.voiceSummary);
}

function runtimeSettingsRequestId() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        return window.crypto.randomUUID();
    }
    return `desktop-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function acceptRuntimeSettingsPayload(payload) {
    if (!payload || !Number.isInteger(payload.revision) || payload.revision < 0) {
        throw new Error('settings_invalid_revision');
    }
    if (!payload.values || typeof payload.values !== 'object' || Array.isArray(payload.values)) {
        throw new Error('settings_invalid_values');
    }
    runtimeSettingsState.revision = payload.revision;
    runtimeSettingsState.values = { ...payload.values };
    runtimeSettingsState.hydrated = true;
    applyRuntimeSettingsControls(runtimeSettingsState.values);
}

async function reconcileAutoListenFromSettings({ reportFailure = true } = {}) {
    if (!runtimeSettingsState.hydrated) return { status: 'deferred', detail: 'settings not hydrated' };
    const inputEnabled = runtimeSettingsState.values['voice.input_enabled'] === true;
    const autoListen = runtimeSettingsState.values['voice.auto_listen'] === true;

    // The persisted setting is the authority on whether an open microphone is
    // wanted. Nothing on the client may default it on — an ambient microphone
    // is not something to enable because a preference failed to load.
    const duplex = window.AuraVoiceMode;
    const haveDuplex = duplex && typeof duplex.setAmbient === 'function';

    if (!inputEnabled) {
        const stopped = !haveDuplex || !duplex.isActive() || await toggleVoice(false);
        return {
            status: stopped ? 'applied' : 'failed',
            detail: stopped ? 'browser microphone stopped' : 'browser microphone did not stop',
        };
    }
    if (!autoListen) {
        if (haveDuplex && duplex.isAmbient()) await duplex.setAmbient(false);
        return { status: 'applied', detail: 'ambient listening remains stopped' };
    }

    // The resident server-side voice engine is the canonical desktop owner.
    // Do not open a second getUserMedia stream when sounddevice owns capture.
    if (state.voiceSummary && state.voiceSummary.server_capture === true) {
        if (haveDuplex && duplex.isActive()) await toggleVoice(false);
        return {
            status: state.voiceSummary.listening ? 'applied' : 'deferred',
            detail: state.voiceSummary.listening
                ? 'canonical server microphone lane is active'
                : 'canonical server microphone owner is applying auto-listen',
        };
    }

    // The full-duplex lane owns browser capture, barge-in, clause streaming,
    // and addressivity. A missing bundle is a visible failure, never a second
    // microphone implementation.
    if (haveDuplex) {
        if (duplex.isActive()) return { status: 'applied', detail: 'ambient listening already active' };
        const started = await duplex.setAmbient(true);
        if (started) return { status: 'applied', detail: 'ambient duplex listening started' };
        return {
            status: reportFailure ? 'failed' : 'deferred',
            detail: 'ambient listening needs microphone permission — press VOICE once to grant it',
        };
    }

    return {
        status: reportFailure ? 'failed' : 'deferred',
        detail: 'canonical browser voice bundle is unavailable',
    };
}

async function hydrateRuntimeSettings({ quiet = false, reconcileVoice = true } = {}) {
    if (state.accessResolved && state.conversationOnly) {
        setRuntimeSettingsAvailability(
            false,
            'Runtime settings require the paired desktop control surface.',
            'warning'
        );
        return false;
    }
    if (runtimeSettingsState.hydrationPromise) return runtimeSettingsState.hydrationPromise;
    const hydration = (async () => {
        if (!quiet) setRuntimeSettingsAvailability(false, 'Synchronizing runtime settings…', 'busy');
        const response = await fetch('/api/settings', {
            cache: 'no-store',
            credentials: 'same-origin',
            headers: auraDesktopHeaders(),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(payload.detail || payload.error || `settings_http_${response.status}`);
        }
        if (payload.integrity && payload.integrity.ok === false) {
            throw new Error(payload.integrity.error || 'settings_integrity_failed');
        }
        acceptRuntimeSettingsPayload(payload);
        if (reconcileVoice) {
            const autoResult = await reconcileAutoListenFromSettings({ reportFailure: false });
            if (runtimeSettingsState.values['voice.auto_listen'] && autoResult.status === 'failed') {
                setRuntimeSettingsAvailability(
                    true,
                    `Runtime revision ${payload.revision}; auto-listen needs microphone access.`,
                    'warning'
                );
                return true;
            }
        }
        setRuntimeSettingsAvailability(
            true,
            `Runtime revision ${payload.revision} verified.`,
            'ready'
        );
        return true;
    })();
    runtimeSettingsState.hydrationPromise = hydration;
    try {
        return await hydration;
    } catch (error) {
        runtimeSettingsState.hydrated = false;
        setRuntimeSettingsAvailability(
            false,
            `Runtime settings unavailable: ${String(error.message || error)}`,
            'error'
        );
        if (!quiet) console.warn('[Settings] Runtime hydration failed:', error);
        throw error;
    } finally {
        if (runtimeSettingsState.hydrationPromise === hydration) {
            runtimeSettingsState.hydrationPromise = null;
        }
    }
}

async function patchRuntimeSettings(changes) {
    if (!runtimeSettingsState.hydrated) await hydrateRuntimeSettings({ reconcileVoice: false });
    const desired = { ...changes };
    const baseValues = Object.fromEntries(
        Object.keys(desired).map(key => [key, runtimeSettingsState.values[key]])
    );
    const requestId = runtimeSettingsRequestId();

    const submit = async (allowRetry = true) => {
        const expectedRevision = runtimeSettingsState.revision;
        let response;
        try {
            response = await fetch('/api/settings', {
                method: 'PATCH',
                cache: 'no-store',
                credentials: 'same-origin',
                headers: auraDesktopHeaders({ 'Content-Type': 'application/json' }),
                body: JSON.stringify({
                    expected_revision: expectedRevision,
                    request_id: requestId,
                    changes: desired,
                }),
            });
        } catch (networkError) {
            if (!allowRetry) throw networkError;
            await hydrateRuntimeSettings({ quiet: true, reconcileVoice: false });
            return submit(false);
        }

        const payload = await response.json().catch(() => ({}));
        if (response.status === 409 && payload.error === 'settings_revision_conflict' && allowRetry) {
            await hydrateRuntimeSettings({ quiet: true, reconcileVoice: false });
            const alreadyApplied = Object.entries(desired).every(
                ([key, value]) => runtimeSettingsState.values[key] === value
            );
            if (alreadyApplied) return { values: runtimeSettingsState.values, replayed: true };
            const untouched = Object.keys(desired).every(
                key => runtimeSettingsState.values[key] === baseValues[key]
            );
            if (untouched) return submit(false);
            throw new Error('settings_conflict_requires_review');
        }
        if (!response.ok) {
            throw new Error(payload.detail || payload.error || `settings_patch_http_${response.status}`);
        }
        acceptRuntimeSettingsPayload(payload);
        if (payload.superseded === true) {
            const desiredStillActive = Object.entries(desired).every(
                ([key, value]) => runtimeSettingsState.values[key] === value
            );
            if (!desiredStillActive) {
                throw new Error('settings_idempotent_replay_superseded');
            }
        }
        return payload;
    };

    setRuntimeSettingsAvailability(false, 'Saving runtime settings…', 'busy');
    try {
        const payload = await submit(true);
        if (
            Object.prototype.hasOwnProperty.call(desired, 'voice.auto_listen')
            || Object.prototype.hasOwnProperty.call(desired, 'voice.input_enabled')
        ) {
            await reconcileAutoListenFromSettings();
        }
        const unresolvedOwner = Object.values(payload.application || {}).find(
            entry => entry && (entry.status === 'failed' || entry.status === 'deferred')
        );
        setRuntimeSettingsAvailability(
            true,
            unresolvedOwner
                ? `Saved revision ${runtimeSettingsState.revision}; ${unresolvedOwner.owner} has not verified it yet.`
                : `Runtime revision ${runtimeSettingsState.revision} applied.`,
            unresolvedOwner ? 'warning' : 'ready'
        );
        return payload;
    } catch (error) {
        applyRuntimeSettingsControls(runtimeSettingsState.values);
        setRuntimeSettingsAvailability(
            runtimeSettingsState.hydrated,
            `Runtime setting failed: ${String(error.message || error)}`,
            'error'
        );
        throw error;
    }
}

Object.entries(RUNTIME_SETTING_CONTROLS).forEach(([key, definition]) => {
    const control = $(definition.id);
    if (!control) return;
    if (key === 'voice.output_rate') {
        control.addEventListener('input', () => {
            const value = Number.parseFloat(control.value);
            const output = $('setting-tts-speed-value');
            if (output && Number.isFinite(value)) output.textContent = `${value.toFixed(1)}×`;
        });
    }
    control.addEventListener('change', async () => {
        const value = runtimeControlValue(definition, control);
        try {
            await patchRuntimeSettings({ [key]: value });
        } catch (error) {
            console.warn(`[Settings] Failed to update ${key}:`, error);
        }
    });
});

async function confirmNextRuntimeAction(challengeId) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
        const response = await fetch('/api/settings/auth/fresh', {
            method: 'POST',
            cache: 'no-store',
            credentials: 'same-origin',
            headers: auraDesktopHeaders({ 'Content-Type': 'application/json' }),
            signal: controller.signal,
            body: JSON.stringify({ challenge_id: String(challengeId || '') }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || payload.ok !== true) {
            throw new Error(payload.detail || `confirmation_http_${response.status}`);
        }
        return true;
    } catch (error) {
        await cancelRuntimeActionConfirmation(challengeId);
        const body = $('approval-modal-message');
        if (body) body.textContent = `Confirmation failed: ${String(error.message || error)}`;
        return false;
    } finally {
        clearTimeout(timeout);
    }
}

async function cancelRuntimeActionConfirmation(challengeId) {
    const normalized = String(challengeId || '');
    if (!normalized) return false;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    try {
        const response = await fetch('/api/settings/auth/revoke', {
            method: 'POST',
            cache: 'no-store',
            credentials: 'same-origin',
            headers: auraDesktopHeaders({ 'Content-Type': 'application/json' }),
            signal: controller.signal,
            body: JSON.stringify({ challenge_id: normalized }),
        });
        return response.ok;
    } catch (_error) {
        return false;
    } finally {
        clearTimeout(timeout);
    }
}

let pendingApprovalRetry = null;
let pendingApprovalCancel = null;
let pendingApprovalChallengeId = null;
let approvalModalReturnFocus = null;
let approvalConfirmationInFlight = false;

function closeApprovalModal({ restoreFocus = true, cancelChallenge = true } = {}) {
    if (approvalConfirmationInFlight) return false;
    const modal = $('approval-modal');
    if (modal) modal.style.display = 'none';
    const abandonedChallengeId = pendingApprovalChallengeId;
    const cancel = pendingApprovalCancel;
    pendingApprovalRetry = null;
    pendingApprovalCancel = null;
    pendingApprovalChallengeId = null;
    const confirmButton = $('approval-modal-confirm');
    const cancelButton = $('approval-modal-cancel');
    if (confirmButton) {
        confirmButton.disabled = false;
        confirmButton.removeAttribute('aria-busy');
    }
    if (cancelButton) cancelButton.disabled = false;
    if (restoreFocus && approvalModalReturnFocus && approvalModalReturnFocus.isConnected) {
        approvalModalReturnFocus.focus();
    }
    approvalModalReturnFocus = null;
    if (cancelChallenge && abandonedChallengeId) {
        void cancelRuntimeActionConfirmation(abandonedChallengeId);
    }
    if (cancelChallenge && cancel) cancel();
    return true;
}

function openApprovalModal(message, challengeId, retry, cancel = null) {
    const modal = $('approval-modal');
    const body = $('approval-modal-message');
    if (!modal || !body) return false;
    const normalizedChallengeId = String(challengeId || '');
    if (!normalizedChallengeId) return false;
    body.textContent = String(message || 'This action needs a fresh confirmation.');
    pendingApprovalRetry = typeof retry === 'function' ? retry : null;
    pendingApprovalCancel = typeof cancel === 'function' ? cancel : null;
    pendingApprovalChallengeId = normalizedChallengeId;
    approvalModalReturnFocus = document.activeElement;
    approvalConfirmationInFlight = false;
    modal.style.display = 'flex';
    const confirmButton = $('approval-modal-confirm');
    const cancelButton = $('approval-modal-cancel');
    if (confirmButton) {
        confirmButton.disabled = false;
        confirmButton.removeAttribute('aria-busy');
        confirmButton.focus();
    }
    if (cancelButton) cancelButton.disabled = false;
    return true;
}

const approvalCancelButton = $('approval-modal-cancel');
if (approvalCancelButton) approvalCancelButton.addEventListener('click', closeApprovalModal);
const approvalConfirmButton = $('approval-modal-confirm');
if (approvalConfirmButton) {
    approvalConfirmButton.addEventListener('click', async () => {
        if (approvalConfirmationInFlight) return;
        approvalConfirmationInFlight = true;
        approvalConfirmButton.disabled = true;
        approvalConfirmButton.setAttribute('aria-busy', 'true');
        if (approvalCancelButton) approvalCancelButton.disabled = true;
        const retry = pendingApprovalRetry;
        const challengeId = pendingApprovalChallengeId;
        const confirmed = await confirmNextRuntimeAction(challengeId);
        approvalConfirmationInFlight = false;
        approvalConfirmButton.disabled = false;
        approvalConfirmButton.removeAttribute('aria-busy');
        if (approvalCancelButton) approvalCancelButton.disabled = false;
        if (!confirmed) {
            approvalConfirmButton.disabled = true;
            if (approvalCancelButton) approvalCancelButton.focus();
            return;
        }
        closeApprovalModal({ restoreFocus: false, cancelChallenge: false });
        if (retry) void retry();
    });
}
const approvalModal = $('approval-modal');
if (approvalModal) {
    approvalModal.addEventListener('click', event => {
        if (event.target === approvalModal) closeApprovalModal();
    });
}
document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && approvalModal && approvalModal.style.display !== 'none') {
        closeApprovalModal();
        return;
    }
    if (event.key !== 'Tab' || !approvalModal || approvalModal.style.display === 'none') {
        return;
    }
    const focusable = Array.from(
        approvalModal.querySelectorAll('button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])')
    );
    if (!focusable.length) {
        event.preventDefault();
        return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (!approvalModal.contains(document.activeElement)) {
        event.preventDefault();
        first.focus();
    } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
    }
});

setRuntimeSettingsAvailability(false, 'Waiting for desktop control authorization…', 'pending');

async function activateCheatCode() {
    const input = document.getElementById('setting-cheat-code');
    const code = input ? input.value.trim() : '';
    if (!code) return;

    const statusEl = document.getElementById('setting-cheat-status');
    if (statusEl) statusEl.textContent = 'CHECKING…';

    try {
        const resp = await fetch('/api/cheat-codes/activate', {
            method: 'POST',
            headers: auraDesktopHeaders({ 'Content-Type': 'application/json' }),
            credentials: 'same-origin',
            body: JSON.stringify({ code }),
        });
        const data = await resp.json();
        settings.cheatStatus = String(
            data?.ok
                ? (data?.ui_effects?.status || data?.trust_level || 'ACTIVE')
                : 'INVALID'
        ).toUpperCase();
        saveSettings(settings);
        applySettings(settings);

        if (data?.message) {
            appendMsg('system', data.message, false, { system: true, cheat_code: true });
        } else if (!resp.ok) {
            appendMsg('system', 'Unknown cheat code.', false, { system: true, cheat_code: true });
        }

        if (typeof pollHealth === 'function') {
            await pollHealth();
        }
    } catch (err) {
        settings.cheatStatus = 'ERROR';
        saveSettings(settings);
        applySettings(settings);
        appendMsg('system', '⚠ Cheat code activation failed.', false, { system: true, cheat_code: true });
        console.error('[CHEAT] Activation failed:', err);
    } finally {
        if (input) input.value = '';
    }
}

const cheatBtn = document.getElementById('btn-activate-cheat-code');
if (cheatBtn) cheatBtn.addEventListener('click', activateCheatCode);
const cheatInput = document.getElementById('setting-cheat-code');
if (cheatInput) {
    cheatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            activateCheatCode();
        }
    });
}
if (DOM.desktopAccessActions) {
    DOM.desktopAccessActions.addEventListener('click', (event) => {
        const button = event.target && event.target.closest
            ? event.target.closest('[data-desktop-access-action]')
            : null;
        if (!button) return;
        event.preventDefault();
        runDesktopAccessAction(button.getAttribute('data-desktop-access-action'));
    });
}

// Export data
const exportBtn = document.getElementById('btn-export-data');
if (exportBtn) exportBtn.addEventListener('click', async () => {
    try {
        const resp = await fetch('/api/export');
        if (resp.ok) {
            const data = await resp.json();
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `aura_export_${new Date().toISOString().slice(0, 10)}.json`;
            a.click();
            URL.revokeObjectURL(url);
            appendMsg('aura', 'Export downloaded successfully.');
        } else {
            appendMsg('aura', `⚠ Export failed (HTTP ${resp.status}). The server may still be initializing.`);
        }
    } catch (err) {
        console.error('[Export] Failed to export data:', err);
        appendMsg('aura', '⚠ Could not export data. Check your connection.');
    }
});

// Clear history
// This control only rewrites the on-screen transcript. It used to ask "Clear all
// conversation history?" while calling no backend at all: every turn stayed in
// the sessions/turns tables and in Aura's session context, so the button claimed
// a deletion it never performed. Wording now matches the actual effect. Clearing
// the view is still one-way for the reader — the shell never repopulates
// #messages from the server — so the confirm stays.
const clearBtn = document.getElementById('btn-clear-history');
if (clearBtn) clearBtn.addEventListener('click', () => {
    if (confirm('Clear the transcript shown in this window?\n\nAura\'s stored conversation history and memories are not affected.')) {
        const msgEl = document.getElementById('messages');
        if (msgEl) msgEl.innerHTML = '<div class="sys-box">Transcript cleared from this view. Aura\'s history is unchanged.</div>';
    }
});

// ── Onboarding ────────────────────────────────────────────
// Keep the main chat camera-ready by only showing onboarding when explicitly requested.
const onboardingRequested = new URLSearchParams(window.location.search).get('onboarding') === '1';
if (!settings.onboarded && onboardingRequested) {
    const modal = document.getElementById('onboarding-modal');
    if (modal) {
        modal.style.display = 'flex';
        let currentStep = 1;
        const totalSteps = 4;

        function updateOnboardStep() {
            modal.querySelectorAll('.onboard-step').forEach(s => s.classList.remove('active'));
            modal.querySelectorAll('.dot').forEach(d => d.classList.remove('active'));
            const step = modal.querySelector(`[data-step="${currentStep}"]`);
            if (step) step.classList.add('active');
            const dots = modal.querySelectorAll('.dot');
            if (dots[currentStep - 1]) dots[currentStep - 1].classList.add('active');
            const nextBtn = document.getElementById('onboard-next');
            if (nextBtn) nextBtn.textContent = currentStep >= totalSteps ? 'Get Started' : 'Next →';
        }

        document.getElementById('onboard-next')?.addEventListener('click', () => {
            if (currentStep >= totalSteps) {
                modal.style.display = 'none';
                settings.onboarded = true;
                saveSettings(settings);
            } else {
                currentStep++;
                updateOnboardStep();
            }
        });

        document.getElementById('onboard-skip')?.addEventListener('click', () => {
            modal.style.display = 'none';
            settings.onboarded = true;
            saveSettings(settings);
        });
    }
}

// ══════════════════════════════════════════════════════════
//  MAGNUM OPUS — Splash Screen Management
// ══════════════════════════════════════════════════════════
function updateSplashProgress(progress, message = '') {
    const splash = $('splash-screen');
    const splashBar = $('splash-bar');
    const splashStatus = $('splash-status');
    if (!splash || splash.classList.contains('hidden')) return;

    const current = splashBar ? (parseFloat(splashBar.dataset.progress || splashBar.style.width || '0') || 0) : 0;
    const next = Math.max(current, Math.max(8, Math.min(100, Number(progress || 0))));
    if (splashBar) {
        splashBar.style.width = `${next}%`;
        splashBar.dataset.progress = String(next);
    }
    if (message && splashStatus) {
        splashStatus.textContent = message;
    }
}

function syncSplashState(payload) {
    const splash = $('splash-screen');
    if (!splash || splash.classList.contains('hidden')) return;

    const boot = bootSnapshotFromPayload(payload);
    const runtimeHealthy = payloadRuntimeHealthy(payload);
    const shellLaunchable = payloadShellLaunchable(payload);
    const bootReady = runtimeHealthy && (boot.ready === true || String(boot.status || '').toLowerCase() === 'ready');
    const message = String(boot.status_message || '').trim();

    if (state._splashInterval) {
        clearInterval(state._splashInterval);
        state._splashInterval = null;
    }

    updateSplashProgress(
        boot.progress != null ? boot.progress : (runtimeHealthy || shellLaunchable ? 90 : 15),
        message || runtimeHealthStatusText(payload)
    );

    if (runtimeHealthy && bootReady) {
        dismissSplash(message || 'Neural link established.');
    } else if (shellLaunchable) {
        dismissSplash(message || 'Aura shell ready. Cortex will warm on the first foreground turn.', { autoRevealMs: 350 });
    }
}

(function initSplash() {
    const splash = $('splash-screen');
    if (!splash) return;

    const stages = [
        { pct: 15, msg: 'Loading consciousness stack...' },
        { pct: 35, msg: 'Initializing memory systems...' },
        { pct: 55, msg: 'Calibrating affect engine...' },
        { pct: 75, msg: 'Establishing neural pathways...' },
        { pct: 90, msg: 'Synchronizing cognitive cores...' },
    ];
    let stageIdx = 0;
    const interval = setInterval(() => {
        if (stageIdx < stages.length) {
            updateSplashProgress(stages[stageIdx].pct, stages[stageIdx].msg);
            stageIdx++;
        }
    }, 600);

    // Store interval for cleanup
    state._splashInterval = interval;

    // If live contracts take unusually long, switch to an honest status message instead of faking success.
    state._splashTimeout = setTimeout(() => {
        updateSplashProgress(96, 'Live shell is still syncing. Aura is stabilizing background channels...');
    }, 12000);

    // Hard timeout: force-dismiss splash if backend never reaches ready state.
    // The UI will still work in degraded mode with reconnection toasts.
    state._splashHardTimeout = setTimeout(() => {
        dismissSplash('Runtime is taking longer than expected. Loading interface...', { autoRevealMs: 900 });
    }, 45000);
})();

function dismissSplash(finalStatus = 'Neural link established.', options = {}) {
    const splash = $('splash-screen');
    const splashBar = $('splash-bar');
    const startBtn = $('splash-start-btn');
    if (!splash || splash.classList.contains('hidden')) return;
    const autoRevealMs = Math.max(0, Number(options.autoRevealMs ?? 1200));

    // Complete the progress bar
    if (splashBar) {
        splashBar.style.width = '100%';
        splashBar.dataset.progress = '100';
    }
    const splashStatus = $('splash-status');
    if (splashStatus) splashStatus.textContent = finalStatus;

    // Clean up timers
    if (state._splashInterval) clearInterval(state._splashInterval);
    if (state._splashTimeout) clearTimeout(state._splashTimeout);
    if (state._splashHardTimeout) clearTimeout(state._splashHardTimeout);

    const revealShell = () => {
        if (!splash || splash.classList.contains('hidden')) return;
        splash.classList.add('hidden');
        setTimeout(() => {
            if (splash && splash.parentNode) splash.remove();
        }, 1000);
    };

    // Show the START button for immediate manual reveal, but auto-reveal as
    // the fail-safe. The desktop shell must not remain hidden behind a splash
    // when boot health stalls or the boot monitor never reaches "ready".
    if (startBtn) {
        startBtn.style.display = 'inline-block';
        startBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            revealShell();
        });
        setTimeout(revealShell, autoRevealMs);
    } else {
        // Fallback if button does not exist in DOM
        setTimeout(revealShell, Math.min(autoRevealMs, 400));
    }
}

document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        pauseLiveSurface('visibility_hidden');
    } else {
        reconnectLiveSurface('visibility_visible');
    }
    publishSurfaceWorkload('visibility_change');
});

window.addEventListener('pageshow', () => reconnectLiveSurface('pageshow'));
window.addEventListener('focus', () => {
    if (!state.connected || state.surfaceSuspended) {
        reconnectLiveSurface('window_focus');
    }
});
window.addEventListener('online', () => reconnectLiveSurface('browser_online'));
window.addEventListener('offline', () => {
    state.surfaceSuspended = true;
    setConnectionVisual('reconnecting', 'Waiting for network');
    showConnToast('paused');
});
window.addEventListener('pagehide', () => {
    persistChatHandoff({ force: chatHandoffHasContent() });
});

// ══════════════════════════════════════════════════════════
//  MAGNUM OPUS — Textarea Auto-Resize & Keyboard Shortcuts
// ══════════════════════════════════════════════════════════
(function initTextareaAndShortcuts() {
    const textarea = $('chat-input');
    if (!textarea) return;
    const form = $('chat-form');

    function focusComposer(event) {
        const target = event?.target;
        if (target && (target === textarea || target.closest?.('.input-actions'))) return;
        event?.preventDefault?.();
        textarea.focus({ preventScroll: true });
    }

    form?.addEventListener('pointerdown', focusComposer);
    form?.addEventListener('click', focusComposer);
    textarea.addEventListener('pointerdown', () => {
        textarea.focus({ preventScroll: true });
    });

    // Auto-resize textarea as user types
    textarea.addEventListener('input', () => {
        resizeChatComposer(textarea);
        noteTypingSignalInput(textarea);
        if (!textarea.value) retryDeferredShellTransition();
    });

    // Keyboard handling for textarea
    textarea.addEventListener('keydown', (e) => {
        noteTypingSignalKey(e, textarea);
        // Cmd/Ctrl+Enter = Send
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
            e.preventDefault();
            $('chat-form')?.requestSubmit();
            return;
        }
        // Shift+Enter = newline (default behavior, do nothing)
        if (e.shiftKey && e.key === 'Enter') return;
        // Plain Enter = Send (like ChatGPT)
        if (e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
            e.preventDefault();
            $('chat-form')?.requestSubmit();
            return;
        }
        // Escape = clear input
        if (e.key === 'Escape') {
            textarea.value = '';
            resizeChatComposer(textarea);
            flushTypingSignal({ submitted: false, forceInactive: true, messageCharsOverride: 0 });
            persistChatHandoff();
            retryDeferredShellTransition();
        }
    });

    // Global keyboard shortcuts (only when not typing in textarea)
    document.addEventListener('keydown', (e) => {
        const target = e.target;
        const isTyping = target.tagName === 'TEXTAREA' || target.tagName === 'INPUT' || target.isContentEditable;

        // ? = Show shortcuts overlay (only when not typing)
        if (e.key === '?' && !isTyping) {
            e.preventDefault();
            toggleShortcuts(true);
            return;
        }

        // Escape = close any overlay
        if (e.key === 'Escape') {
            toggleShortcuts(false);
            return;
        }

        // Cmd/Ctrl+Shift+R = Regenerate
        if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'r') {
            e.preventDefault();
            regenerateResponse();
            return;
        }
    });
})();

function toggleShortcuts(show) {
    const overlay = $('shortcuts-overlay');
    if (!overlay) return;
    overlay.style.display = show ? 'flex' : 'none';
}

// Close button for shortcuts
$('shortcuts-close')?.addEventListener('click', () => toggleShortcuts(false));
// Click backdrop to close
$('shortcuts-overlay')?.addEventListener('click', (e) => {
    if (e.target === $('shortcuts-overlay')) toggleShortcuts(false);
});

// ══════════════════════════════════════════════════════════
//  MAGNUM OPUS — Regenerate Response
// ══════════════════════════════════════════════════════════
function regenerateResponse() {
    if (!state.lastUserMessage || state.isSubmitting) return;

    // Hide regen button after firing — will re-show on next user send
    const regenBtn = $('regen-btn');
    if (regenBtn) regenBtn.style.display = 'none';

    // Remove the last aura message from the DOM
    const messages = $('messages');
    if (messages) {
        const auraMsgs = messages.querySelectorAll('.msg.aura');
        if (auraMsgs.length > 0) {
            auraMsgs[auraMsgs.length - 1].remove();
        }
    }

    // Resend the last user message
    const msgInput = $('chat-input');
    if (msgInput) {
        msgInput.value = state.lastUserMessage;
        $('chat-form')?.requestSubmit();
    }
}

$('regen-btn')?.addEventListener('click', regenerateResponse);

// ── Plain-language tooltips ───────────────────────────────
// METRIC_GUIDE already carries a readable explanation of every metric, but it
// was reachable only by opening the guide panel and selecting a gauge, so the
// telemetry wall read as unexplained jargon. Attach each explanation to its own
// tile/row on hover; the guide panel still gives the fuller how/why.
function attachPlainLanguageTooltips() {
    for (const [id, key] of Object.entries(METRIC_GUIDE_BY_ID)) {
        const el = $(id);
        const guide = METRIC_GUIDE[key];
        if (!el || !guide || !guide.what) continue;
        // Overrides the existing "Explain <metric>" titles: those name the
        // interaction, not the metric, which is the whole complaint.
        const host = el.closest('.con-box') || el.closest('.gauge-row') || el;
        host.title = `${guide.label} — ${guide.what}`;
    }
    document.querySelectorAll('.section-label').forEach(label => {
        const key = SECTION_GUIDE_BY_LABEL[label.textContent.trim()];
        const guide = key && METRIC_GUIDE[key];
        if (guide && guide.what && !label.title) {
            label.title = `${guide.label} — ${guide.what}`;
            label.classList.add('section-label-explained');
        }
    });
}

// ── Imagination workspace ─────────────────────────────────
// Renders the frame ImaginationEngine is actually holding, straight from
// /api/imagination. Everything drawn here is a real field of that frame; when
// she has not imagined anything the panel says so instead of inventing a
// canvas. The engine is advisory and side-effect free, and the panel states
// that boundary rather than implying these are actions she is taking.
const imagination = (() => {
    const POLL_MS = 4000;
    let timer = null;
    let active = false;
    let inFlight = false;
    let lastFrameId = null;
    let rendering = false;
    let renders = [];
    let autoRender = false;
    // A frame that already failed to render must not be retried on every poll —
    // that would pin the GPU against a frame that cannot succeed.
    const renderFailed = new Set();

    const PRESSURES = [
        ['salience', 'SALIENCE'],
        ['novelty_pressure', 'NOVELTY'],
        ['curiosity_pressure', 'CURIOSITY'],
        ['affective_pressure', 'AFFECT'],
        ['memory_pressure', 'MEMORY'],
        ['verification_pressure', 'VERIFY'],
    ];

    function bindControls() {
        const btn = $('imagine-render-btn');
        if (btn && !btn.dataset.bound) {
            btn.dataset.bound = '1';
            btn.addEventListener('click', () => visualize());
        }
        const auto = $('imagine-auto-toggle');
        if (auto && !auto.dataset.bound) {
            auto.dataset.bound = '1';
            auto.addEventListener('change', () => {
                autoRender = auto.checked;
                setRenderState(
                    autoRender
                        ? 'Will render each new frame as she imagines it.'
                        : '',
                    autoRender ? 'ok' : null);
                if (autoRender) refresh();
            });
        }
    }

    function activate() {
        bindControls();
        if (active) return;
        active = true;
        void poll();
    }

    function deactivate() {
        active = false;
        clearTimeout(timer);
        timer = null;
    }

    function schedule() {
        clearTimeout(timer);
        timer = null;
        if (!active) return;
        const delay = optionalSurfacePollDelay(POLL_MS, {
            foregroundFactor: 3,
            hiddenFactor: 8,
        });
        timer = setTimeout(() => {
            timer = null;
            void poll();
        }, delay);
    }

    async function poll() {
        if (!active) return;
        if (!document.hidden) await refresh();
        schedule();
    }

    async function refresh() {
        if (inFlight) return;
        inFlight = true;
        try {
            const resp = await fetch('/api/imagination', { headers: auraDesktopHeaders() });
            if (!resp.ok) throw new Error(`imagination ${resp.status}`);
            render(await resp.json());
        } catch (err) {
            renderUnavailable(err);
        } finally {
            inFlight = false;
        }
    }

    function renderUnavailable(err) {
        const dot = $('imagine-dot');
        if (dot) dot.classList.remove('live');
        setText('imagine-state', 'UNAVAILABLE');
        const empty = $('imagine-empty');
        const live = $('imagine-live');
        if (empty) {
            empty.hidden = false;
            empty.textContent = 'Imagination workspace unavailable — the runtime is not reporting. '
                + 'This panel shows nothing rather than showing a stale frame.';
        }
        if (live) live.hidden = true;
        console.warn('[Imagine]', err);
    }

    function setText(id, value) {
        const el = $(id);
        if (el) el.textContent = value;
    }

    function render(payload) {
        const frame = payload && payload.latest;
        const status = String((payload && payload.status) || 'idle');
        const dot = $('imagine-dot');
        if (dot) dot.classList.toggle('live', status === 'active');
        setText('imagine-state', `${status.toUpperCase()} · ${(payload && payload.frames) || 0} FRAMES`);

        renderWorlds((payload && payload.worlds) || []);
        renders = Array.isArray(payload && payload.renders) ? payload.renders : [];
        renderGallery();

        const empty = $('imagine-empty');
        const live = $('imagine-live');
        if (!frame) {
            if (empty) {
                empty.hidden = false;
                empty.textContent = 'Aura has not imagined anything yet. This panel fills in when the '
                    + 'imagination engine builds a frame — it never draws a canvas she is not actually holding.';
            }
            if (live) live.hidden = true;
            return;
        }
        if (empty) empty.hidden = true;
        if (live) live.hidden = false;

        const changed = frame.frame_id !== lastFrameId;
        lastFrameId = frame.frame_id;

        const objective = $('imagine-objective');
        if (objective) {
            objective.textContent = frame.objective || '(no objective)';
            objective.classList.toggle('imagine-flash', changed);
            if (changed) setTimeout(() => objective.classList.remove('imagine-flash'), 700);
        }

        renderFrameImage(frame);
        renderCanvas(frame.mental_canvas || {}, frame.associative_links || []);
        renderAttractors(frame.attractor_state || {});
        renderPressures(frame);
        renderList('imagine-thoughts', frame.novel_thoughts || []);
        renderList('imagine-counterfactuals', frame.counterfactuals || []);

        const boundary = $('imagine-boundary');
        if (boundary) {
            const gov = frame.governance || {};
            const advisory = gov.advisory_only !== false;
            const noEffects = gov.no_external_effects !== false;
            boundary.textContent = frame.verification_boundary
                || 'This is an internal hypothetical model, not external perception or proof.';
            boundary.classList.toggle('imagine-boundary-open', !(advisory && noEffects));
        }
    }

    function renderFor(frameId) {
        return renders.find(r => r && r.frame_id === frameId) || null;
    }

    // Shows the image only against the frame that produced it. A render from an
    // older frame is left in the gallery rather than displayed as if it were
    // what she is picturing now.
    function renderFrameImage(frame) {
        const wrap = $('imagine-image-wrap');
        const img = $('imagine-image');
        const cap = $('imagine-image-caption');
        const btn = $('imagine-render-btn');
        if (!wrap || !img) return;
        const mine = renderFor(frame.frame_id);
        if (mine && mine.url) {
            if (img.getAttribute('src') !== mine.url) img.setAttribute('src', mine.url);
            wrap.hidden = false;
            if (cap) cap.textContent = mine.prompt || '';
            if (btn) btn.disabled = true;
        } else {
            wrap.hidden = true;
            img.removeAttribute('src');
            if (btn) btn.disabled = rendering;
        }
        maybeAutoRender(frame);
    }

    function maybeAutoRender(frame) {
        if (!autoRender || rendering) return;
        if (!frame || !frame.frame_id) return;
        if (renderFor(frame.frame_id)) return;
        if (renderFailed.has(frame.frame_id)) return;
        const canvas = frame.mental_canvas || {};
        if (!canvas.image_prompt) return;
        visualize();
    }

    function setRenderState(text, tone) {
        const el = $('imagine-render-state');
        if (!el) return;
        el.textContent = text || '';
        el.className = 'imagine-render-state' + (tone ? ` imagine-render-${tone}` : '');
    }

    async function visualize() {
        if (rendering) return;
        rendering = true;
        const btn = $('imagine-render-btn');
        if (btn) btn.disabled = true;
        setRenderState('Rendering what she is picturing…', 'busy');
        const started = Date.now();
        try {
            const resp = await fetch('/api/imagination/visualize', {
                method: 'POST',
                headers: auraDesktopHeaders({ 'Content-Type': 'application/json' }),
                body: '{}',
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok || !data.ok) {
                const why = data.error || `render failed (${resp.status})`;
                if (lastFrameId) renderFailed.add(lastFrameId);
                setRenderState(why, 'error');
            } else {
                const secs = ((Date.now() - started) / 1000).toFixed(1);
                setRenderState(data.cached ? 'Already rendered.' : `Rendered in ${secs}s.`, 'ok');
                await refresh();
            }
        } catch (err) {
            if (lastFrameId) renderFailed.add(lastFrameId);
            setRenderState(`Render failed: ${err}`, 'error');
        } finally {
            rendering = false;
            if (btn) btn.disabled = !!renderFor(lastFrameId);
        }
    }

    function renderGallery() {
        const host = $('imagine-gallery');
        if (!host) return;
        if (!renders.length) {
            host.innerHTML = '<div class="imagine-empty-inline">Nothing rendered yet.</div>';
            return;
        }
        host.innerHTML = renders.slice().reverse().map(r => `
            <figure class="imagine-thumb">
                <img src="${escHtml(String(r.url))}" alt="${escHtml(String(r.prompt || 'imagined image'))}" loading="lazy">
                <figcaption>${escHtml(String(r.modality || ''))}</figcaption>
            </figure>`).join('');
    }

    // Objects and relations from the frame's mental canvas, laid out on a ring
    // so every relation is drawn as a real edge between real nodes.
    function renderCanvas(canvas, links) {
        const svg = $('imagine-canvas');
        if (!svg) return;
        const objects = Array.isArray(canvas.objects) ? canvas.objects.slice(0, 6) : [];
        const relations = [
            ...(Array.isArray(canvas.relations) ? canvas.relations : []),
            ...(Array.isArray(links) ? links : []),
        ];
        if (!objects.length) {
            svg.innerHTML = '<text x="160" y="95" text-anchor="middle" class="imagine-canvas-null">no objects in this frame</text>';
            setText('imagine-canvas-caption', '');
            return;
        }

        const cx = 160, cy = 95, rx = 102, ry = 56;
        const pos = new Map();
        objects.forEach((obj, i) => {
            const id = String(obj.id ?? obj);
            if (objects.length === 1) { pos.set(id, { x: cx, y: cy, role: obj.role }); return; }
            const t = (i / objects.length) * Math.PI * 2 - Math.PI / 2;
            pos.set(id, { x: cx + rx * Math.cos(t), y: cy + ry * Math.sin(t), role: obj.role });
        });

        // Straight hairlines between flat discs read as a wiring diagram, not
        // as a frame she is holding. Nodes get depth (halo, gradient body,
        // specular), edges bow inward and carry direction, and every label
        // gets a backing plate so it survives crossing an edge.
        const parts = [`<defs>
            <radialGradient id="ic-node" cx="34%" cy="28%" r="78%">
                <stop offset="0%" stop-color="#6c4fb0"/>
                <stop offset="55%" stop-color="#2a1a52"/>
                <stop offset="100%" stop-color="#150c2c"/>
            </radialGradient>
            <radialGradient id="ic-node-focus" cx="34%" cy="26%" r="78%">
                <stop offset="0%" stop-color="#f0e2ff"/>
                <stop offset="45%" stop-color="#b76bff"/>
                <stop offset="100%" stop-color="#6a24c4"/>
            </radialGradient>
            <linearGradient id="ic-edge" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stop-color="#8a2be2" stop-opacity="0.15"/>
                <stop offset="50%" stop-color="#b44dff" stop-opacity="0.75"/>
                <stop offset="100%" stop-color="#00e5ff" stop-opacity="0.5"/>
            </linearGradient>
            <marker id="ic-arrow" viewBox="0 0 8 8" refX="6.4" refY="4"
                    markerWidth="4.6" markerHeight="4.6" orient="auto-start-reverse">
                <path d="M0.6 1 L7 4 L0.6 7 Z" fill="#b44dff" fill-opacity="0.85"/>
            </marker>
        </defs>`];

        // The ring the objects are laid out on, drawn so the arrangement
        // reads as deliberate rather than accidental.
        if (objects.length > 1) {
            parts.push(`<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" class="imagine-orbit"/>`);
        }

        const nodeRadius = (p) => (p.role === 'focus' ? 9.5 : 6.5);

        // Walk `gap` back from (tx,ty) toward (fx,fy) — the control point — so
        // an edge terminates on a node's rim.
        const trim = (fx, fy, tx, ty, gap) => {
            const dx = tx - fx, dy = ty - fy;
            const len = Math.hypot(dx, dy);
            if (len < 0.001) return [tx, ty];
            return [tx - (dx / len) * gap, ty - (dy / len) * gap];
        };

        const chip = (x, y, text, cls) => {
            const w = Math.max(14, text.length * 3.15 + 7);
            return `<g class="${cls}-wrap">` +
                `<rect x="${(x - w / 2).toFixed(1)}" y="${(y - 5.4).toFixed(1)}" width="${w.toFixed(1)}" height="8.4" rx="4.2" class="${cls}-plate"/>` +
                `<text x="${x.toFixed(1)}" y="${(y + 0.9).toFixed(1)}" text-anchor="middle" class="${cls}">${escHtml(text)}</text></g>`;
        };

        const edgeLabels = [];
        const seen = new Set();
        for (const rel of relations) {
            if (!rel || typeof rel !== 'object') continue;
            const a = pos.get(String(rel.source));
            const b = pos.get(String(rel.target));
            if (!a || !b) continue; // only draw edges whose endpoints are real nodes
            const key = `${rel.source}->${rel.target}`;
            if (seen.has(key)) continue;
            seen.add(key);
            // Bow the edge toward the centre so long chords stop cutting
            // through the nodes on the far side of the ring.
            const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
            const qx = mx + (cx - mx) * 0.32, qy = my + (cy - my) * 0.32;
            // Stop the curve at each node's rim rather than its centre —
            // otherwise the arrowhead lands underneath the disc that is drawn
            // over it and the direction of the relation is invisible.
            const [sx, sy] = trim(qx, qy, a.x, a.y, nodeRadius(a) + 1.5);
            const [ex, ey] = trim(qx, qy, b.x, b.y, nodeRadius(b) + 3.4);
            parts.push(
                `<path d="M${sx.toFixed(1)} ${sy.toFixed(1)} Q${qx.toFixed(1)} ${qy.toFixed(1)} ${ex.toFixed(1)} ${ey.toFixed(1)}" ` +
                `class="imagine-edge" marker-end="url(#ic-arrow)"/>`
            );
            // Midpoint of the quadratic at t=0.5.
            const lx = 0.25 * a.x + 0.5 * qx + 0.25 * b.x;
            const ly = 0.25 * a.y + 0.5 * qy + 0.25 * b.y;
            const label = String(rel.relation || '').trim();
            if (label) edgeLabels.push(chip(Math.min(302, Math.max(18, lx)), ly, label, 'imagine-edge-label'));
        }
        // Labels last so an edge drawn later never overprints one.
        parts.push(...edgeLabels);

        const nodeLabels = [];
        for (const [id, p] of pos) {
            const focus = p.role === 'focus';
            const r = nodeRadius(p);
            parts.push(
                `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${(r + 5.5).toFixed(1)}" class="imagine-node-halo${focus ? ' imagine-node-halo-focus' : ''}"/>` +
                `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${r}" class="imagine-node${focus ? ' imagine-node-focus' : ''}"/>` +
                // Specular: a small offset arc catching light from the upper left.
                `<circle cx="${(p.x - r * 0.3).toFixed(1)}" cy="${(p.y - r * 0.36).toFixed(1)}" r="${(r * 0.34).toFixed(1)}" class="imagine-node-spec"/>`
            );
            // Push the label radially outward, so it never lands on the ring.
            let lx = p.x, ly = p.y + r + 9;
            if (pos.size > 1) {
                const ang = Math.atan2(p.y - cy, p.x - cx);
                lx = p.x + Math.cos(ang) * (r + 11);
                ly = p.y + Math.sin(ang) * (r + 11) + 1.4;
            }
            nodeLabels.push(chip(
                Math.min(304, Math.max(16, lx)),
                Math.min(184, Math.max(9, ly)),
                id,
                'imagine-node-label'
            ));
        }
        parts.push(...nodeLabels);

        svg.innerHTML = parts.join('');
        setText('imagine-canvas-caption', canvas.image_prompt || '');
    }

    function renderAttractors(attractor) {
        const host = $('imagine-attractors');
        if (!host) return;
        const probs = attractor.probabilities;
        if (!probs || typeof probs !== 'object' || !Object.keys(probs).length) {
            host.innerHTML = '<div class="imagine-empty-inline">no attractor competition in this frame</div>';
            setText('imagine-attractor-meta', '');
            return;
        }
        const selected = String(attractor.selected || '');
        const rows = Object.entries(probs)
            .filter(([, v]) => Number.isFinite(Number(v)))
            .sort((a, b) => b[1] - a[1])
            .slice(0, 8);
        const max = Math.max(...rows.map(r => r[1]), 0.0001);
        host.innerHTML = rows.map(([name, p]) => {
            const pct = (Number(p) * 100);
            const width = (Number(p) / max) * 100;
            const win = name === selected;
            return `<div class="imagine-attractor${win ? ' imagine-attractor-win' : ''}">
                <div class="imagine-attractor-head"><span>${escHtml(name)}</span><span>${pct.toFixed(1)}%</span></div>
                <div class="imagine-attractor-track"><div class="imagine-attractor-fill" style="width:${width.toFixed(1)}%"></div></div>
            </div>`;
        }).join('');

        const entropy = Number(attractor.entropy);
        const margin = Number(attractor.stability_margin);
        const depth = Number(attractor.recurrent_depth);
        const bits = [];
        if (selected) bits.push(`selected <strong>${escHtml(selected)}</strong>`);
        if (Number.isFinite(entropy)) bits.push(`entropy ${entropy.toFixed(2)}`);
        if (Number.isFinite(margin)) bits.push(`stability margin ${margin.toFixed(2)}`);
        if (Number.isFinite(depth)) bits.push(`recurrent depth ${depth}`);
        const meta = $('imagine-attractor-meta');
        if (meta) meta.innerHTML = bits.join(' · ');
    }

    function renderPressures(frame) {
        const host = $('imagine-pressures');
        if (!host) return;
        host.innerHTML = PRESSURES.map(([key, label]) => {
            const raw = frame[key];
            const known = raw != null && Number.isFinite(Number(raw));
            const pct = known ? Math.max(0, Math.min(1, Number(raw))) * 100 : 0;
            return `<div class="imagine-pressure">
                <div class="imagine-pressure-head"><span>${label}</span><span class="${known ? '' : 'telemetry-unknown'}">${known ? pct.toFixed(0) + '%' : TELEMETRY_UNKNOWN}</span></div>
                <div class="gauge-bar"><div class="gauge-fill curiosity${known ? '' : ' gauge-unknown'}" style="width:${pct.toFixed(1)}%"></div></div>
            </div>`;
        }).join('');
    }

    function renderList(id, items) {
        const host = $(id);
        if (!host) return;
        const rows = (Array.isArray(items) ? items : []).filter(Boolean).slice(0, 4);
        host.innerHTML = rows.length
            ? rows.map(t => `<div class="imagine-item">${escHtml(String(t))}</div>`).join('')
            : '<div class="imagine-empty-inline">none in this frame</div>';
    }

    function renderWorlds(worlds) {
        const host = $('imagine-worlds');
        if (!host) return;
        const rows = Array.isArray(worlds) ? worlds : [];
        if (!rows.length) {
            host.innerHTML = '<div class="imagine-empty-inline">No worlds instantiated.</div>';
            return;
        }
        host.innerHTML = rows.slice(0, 6).map(w => {
            const id = escHtml(String(w.id ?? w.world_id ?? 'world'));
            const bits = [];
            if (w.bodies != null) bits.push(`${w.bodies} bodies`);
            if (w.ticks != null) bits.push(`${w.ticks} ticks`);
            return `<a class="imagine-world" href="/worlds" target="_blank" rel="noopener">
                <span class="imagine-world-id">${id}</span>
                <span class="imagine-world-meta">${escHtml(bits.join(' · '))}</span>
            </a>`;
        }).join('');
    }

    window.addEventListener('aura:workload-mode', () => {
        if (active) schedule();
    });

    return { activate, deactivate, refresh, visualize };
})();

// ── Header vitals overflow ────────────────────────────────
// The header carries 19 vitals but only ~4 fit beside the brand and actions.
// They used to render into a clipped, scrollbar-less strip, so CPU/RAM/UPTIME
// and the autonomy flags were invisible with no affordance. Stats that do not
// fit are moved into a popover and counted on a "+N" chip, so every vital stays
// reachable at every width. Nodes are moved, never rebuilt, so the
// getElementById updates that drive them keep working wherever they live.
const hudOverflow = (() => {
    const STRIP_GAP = 12;      // matches .hud-stats-inner gap
    const CHIP_RESERVE = 62;   // width kept free for the "+N" chip
    let order = null;
    let scheduled = false;

    const els = () => ({
        stats: $('hud-stats'),
        inner: $('hud-stats-inner'),
        btn: $('hud-overflow-btn'),
        panel: $('hud-overflow-panel'),
        count: $('hud-overflow-count'),
    });

    function setOpen(open) {
        const { btn, panel } = els();
        if (!btn || !panel) return;
        panel.hidden = !open;
        btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    }

    function isOpen() {
        const { panel } = els();
        return !!panel && !panel.hidden;
    }

    function sync() {
        const { stats, inner, btn, panel, count } = els();
        if (!stats || !inner || !btn || !panel || !count) return;
        if (!order) order = Array.from(inner.children);
        if (!order.length) return;

        const reopen = isOpen();
        // Measure with every stat back in the strip, in canonical order.
        for (const el of order) inner.appendChild(el);

        const budget = stats.clientWidth;
        if (budget <= 0) return; // header not laid out yet
        const widths = order.map(el => el.getBoundingClientRect().width);
        const total = widths.reduce((a, w) => a + w, 0) + STRIP_GAP * (order.length - 1);

        const overflow = [];
        if (total > budget) {
            const limit = budget - CHIP_RESERVE;
            let used = 0;
            order.forEach((el, i) => {
                if (overflow.length) { overflow.push(el); return; }
                const next = used + widths[i] + (used ? STRIP_GAP : 0);
                if (next > limit) overflow.push(el);
                else used = next;
            });
        }
        for (const el of overflow) panel.appendChild(el);

        btn.hidden = overflow.length === 0;
        count.textContent = '+' + overflow.length;
        if (btn.hidden) setOpen(false);
        else if (reopen) setOpen(true);
    }

    function schedule() {
        if (scheduled) return;
        scheduled = true;
        const run = () => {
            if (!scheduled) return; // whichever trigger lands first wins
            scheduled = false;
            sync();
        };
        // requestAnimationFrame never fires while the shell is in a hidden or
        // background window, which would latch `scheduled` and lock the strip
        // out permanently. The timer guarantees the flag is always cleared.
        requestAnimationFrame(run);
        setTimeout(run, 250);
    }

    function init() {
        const { stats, btn, panel } = els();
        if (!stats || !btn || !panel) return;

        btn.addEventListener('click', (event) => {
            event.stopPropagation();
            setOpen(panel.hidden);
        });
        document.addEventListener('click', (event) => {
            if (!isOpen()) return;
            if (panel.contains(event.target) || btn.contains(event.target)) return;
            setOpen(false);
        });
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && isOpen()) setOpen(false);
        });
        // Observe the container, not the content: this reacts to viewport and
        // brand/action width changes without re-entering on our own moves.
        if (typeof ResizeObserver !== 'undefined') {
            new ResizeObserver(schedule).observe(stats);
        }
        window.addEventListener('resize', schedule);
        // Widths measured while hidden can be stale; re-measure on return.
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) schedule();
        });
        schedule();
    }

    return { init, schedule };
})();

attachPlainLanguageTooltips();
hudOverflow.init();
// Values change width as they populate ("0s" -> "1h20m"); re-measure when the
// document's fonts settle so the first paint is not measured against fallbacks.
if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => hudOverflow.schedule()).catch(() => {});
}

function markLegacyShellReady() {
    window.__auraLegacyShellReady = true;
    document.body.dataset.auraShell = 'ready';
    const hardRecovery = document.getElementById('aura-hard-recovery');
    if (hardRecovery) hardRecovery.remove();
}

markLegacyShellReady();

/* ── Frame governor ──────────────────────────────────────────────────────
   Measures the symptom, not the cause.

   A screen recorder, an older machine, a 32B generation saturating the GPU
   and a browser throttling a background tab all produce the same observable
   thing from inside the page: frames arriving late. None of them is
   diagnosable from here and none of them needs to be — the response is the
   same either way, which is to stop paying for the ambient work.

   Hysteresis is not a nicety. Shedding on a single bad frame makes the
   background flicker between blurred and flat, which reads as a fault and
   looks worse than the lag it is answering. So: shed only after
   BAD_SAMPLES_TO_SHED consecutive slow frames, restore only after
   GOOD_SAMPLES_TO_RESTORE consecutive fast ones, and require more evidence to
   restore than to shed — coming back too eagerly is how you get the flicker
   from the other side.

   A hidden tab is not rescued. requestAnimationFrame is throttled or stopped
   outright when the document is hidden, so every sample looks catastrophic;
   treating that as lag would leave every backgrounded surface permanently
   lean for no reason. Samples are discarded while hidden and the run is
   restarted clean on the way back. */
const auraFrameGovernor = (() => {
    // 60fps is 16.7ms. The threshold sits at two missed frames rather than
    // one, because one late frame is normal on any machine doing anything.
    const SLOW_FRAME_MS = 34;
    const BAD_SAMPLES_TO_SHED = 12;
    const GOOD_SAMPLES_TO_RESTORE = 45;
    // A gap this large is a tab switch, a sleep, or a breakpoint — not lag.
    const IMPLAUSIBLE_GAP_MS = 500;

    let badSamples = 0;
    let goodSamples = 0;
    let lean = false;
    let lastFrame = 0;
    let running = false;

    function setLean(next) {
        if (next === lean) return;
        lean = next;
        document.body.classList.toggle('perf-lean', lean);
        document.body.dataset.auraPerf = lean ? 'lean' : 'full';
    }

    function sample(now) {
        if (!running) return;
        window.requestAnimationFrame(sample);

        if (document.hidden) {
            lastFrame = 0;
            badSamples = 0;
            goodSamples = 0;
            return;
        }
        if (!lastFrame) {
            lastFrame = now;
            return;
        }

        const delta = now - lastFrame;
        lastFrame = now;
        if (delta > IMPLAUSIBLE_GAP_MS) {
            badSamples = 0;
            goodSamples = 0;
            return;
        }

        if (delta > SLOW_FRAME_MS) {
            goodSamples = 0;
            badSamples += 1;
            if (badSamples >= BAD_SAMPLES_TO_SHED) setLean(true);
        } else {
            badSamples = 0;
            goodSamples += 1;
            if (goodSamples >= GOOD_SAMPLES_TO_RESTORE) setLean(false);
        }
    }

    function start() {
        if (running) return;
        // Someone who asked the OS for less motion has already had the
        // expensive layers stopped by CSS, unconditionally. Sampling frames to
        // decide whether to stop them again would be measuring nothing.
        const reduced = window.matchMedia
            && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (reduced) return;
        running = true;
        lastFrame = 0;
        window.requestAnimationFrame(sample);
    }

    function stop() {
        running = false;
        setLean(false);
    }

    document.addEventListener('visibilitychange', () => {
        lastFrame = 0;
        badSamples = 0;
        goodSamples = 0;
    });

    return {
        start,
        stop,
        get lean() { return lean; },
        get thresholds() {
            return { SLOW_FRAME_MS, BAD_SAMPLES_TO_SHED, GOOD_SAMPLES_TO_RESTORE };
        },
    };
})();

window.auraFrameGovernor = auraFrameGovernor;
auraFrameGovernor.start();
