/**
 * voice_mode.js — Aura's full-duplex voice surface.
 *
 * Design notes worth keeping:
 *
 * Two AudioContexts, not one. Capture runs at 16 kHz because that is what
 * Silero and Whisper want; playback runs at 24 kHz because that is Kokoro's
 * native rate and resampling her voice down to 16 k audibly dulls it. A
 * single context cannot have two rates.
 *
 * Echo cancellation is requested explicitly and is not optional. Without it,
 * on speakers, her own voice re-enters the microphone, trips barge-in, and
 * she interrupts herself — which looks exactly like a model failure. The
 * server has a transcript-level echo guard as the last line, but the browser's
 * WebRTC canceller is what does the real work.
 *
 * The client reports playback position continuously. The server needs it to
 * know what the user actually heard before an interruption, which is what
 * keeps her memory honest about her own half of the conversation.
 */
(function () {
    'use strict';

    const CAPTURE_RATE = 16000;
    const PLAYBACK_RATE = 24000;

    const OPCODE = { SPEECH: 1, BACKCHANNEL: 2, FILLER: 3 };
    const HEADER_BYTES = 8;

    const state = {
        active: false,
        // Ambient: the microphone is open and she is in the room, but there
        // is no modal, no orb taking over the screen, and every turn lands in
        // the chat thread like any other. The full-screen surface below still
        // exists and is still what the VOICE button opens — it is now the
        // *focused* mode rather than the only mode.
        ambient: false,
        ws: null,
        captureCtx: null,
        playbackCtx: null,
        micStream: null,
        micNode: null,
        speechNode: null,
        asideNode: null,
        micLevel: 0,
        auraLevel: 0,
        sessionState: 'idle',
        muted: false,
        playedMs: 0,
        bufferedMs: 0,
        transcript: [],
        partialStable: '',
        partialTentative: '',
        currentReply: '',
        spokenSoFar: '',
        voices: [],
        currentVoice: '',
        metrics: null,
        ready: false,
        serverReady: false,
        audioStartPromise: null,
        captureGeneration: 0,
        captureRecoveryPromise: null,
        captureMuteTimer: null,
        captureStopIntentional: false,
        terminalFailure: false,
        serverState: 'idle',
        speechUtteranceId: 0,
        expectedSpeechSeq: 0,
        interruptedUtterances: new Set(),
        rejectAsides: false,
        rejectPendingSpeech: false,
        returnFocus: null,
        lastVisualScale: null,
        lastVisualGlow: null,
        lastPlaybackReportMs: -1,
        currentReplyMirrored: false,
        reconnectAttempts: 0,
        reconnectTimer: null,
        closingIntentionally: false,
        rafHandle: null,
    };

    // ── ambient presence ─────────────────────────────────────────────────
    //
    // Everything that makes an open microphone legible without making it
    // loud. Three jobs: put spoken turns into the chat thread where every
    // other turn lives, show what she heard but let pass, and never once
    // prompt for anything the user did not ask for.

    const AMBIENT_PREF_KEY = 'aura.voice.ambient';

    const ambient = (() => {
        let indicator = null;
        let overheardEl = null;
        let overheardTimer = null;
        let streaming = false;

        function ensureIndicator() {
            if (indicator) return indicator;
            indicator = document.createElement('div');
            indicator.className = 'ambient-listening';
            indicator.setAttribute('role', 'status');
            indicator.setAttribute('aria-live', 'polite');
            indicator.innerHTML =
                '<span class="ambient-dot"></span>' +
                '<span class="ambient-label">listening</span>' +
                '<div class="ambient-overheard" id="ambient-overheard"></div>';
            // Clicking the indicator opens the focused surface — the fastest
            // route from "she is around" to "I want the full thing".
            indicator.addEventListener('click', () => { surface(true); });
            document.body.appendChild(indicator);
            overheardEl = indicator.querySelector('#ambient-overheard');
            return indicator;
        }

        return {
            enabledByUser() {
                // Mirrors the persisted `voice.auto_listen` runtime setting,
                // which is the authority — aura.js pushes it here as soon as
                // settings hydrate. The local copy exists only so a reload
                // can resume listening in the same frame instead of waiting
                // a round trip, and it defaults to *off*: an ambient
                // microphone must never be enabled because a preference
                // failed to load.
                try {
                    return window.localStorage.getItem(AMBIENT_PREF_KEY) === 'on';
                } catch (_e) {
                    return false;
                }
            },

            setPreference(on) {
                try {
                    window.localStorage.setItem(AMBIENT_PREF_KEY, on ? 'on' : 'off');
                } catch (_e) { /* preference simply will not persist */ }
            },

            setEnabled(on) {
                if (!on) {
                    if (indicator) indicator.classList.remove('ambient-on');
                    return;
                }
                ensureIndicator().classList.add('ambient-on');
            },

            setState(s) {
                if (!indicator) return;
                indicator.dataset.state = s || '';
                const label = indicator.querySelector('.ambient-label');
                if (!label) return;
                label.textContent =
                    s === 'user_speaking' ? 'hearing you'
                        : s === 'thinking' ? 'thinking'
                            : s === 'speaking' ? 'speaking'
                                : 'listening';
            },

            // A turn she took. It belongs in the chat thread exactly like a
            // typed one — same bubble, same history, same everything. That
            // identity is the point of the whole feature: there is no
            // separate place where the spoken conversation lives.
            userSaid(text) {
                if (typeof window.auraAppendVoiceTurn === 'function') {
                    window.auraAppendVoiceTurn('user', text);
                    return true;
                }
                streaming = false;
                return false;
            },

            auraChunk(text) {
                if (!text) return false;
                if (typeof window.auraStreamVoiceReply !== 'function') return false;
                window.auraStreamVoiceReply(text, !streaming);
                streaming = true;
                return true;
            },

            replyDone() {
                if (streaming && typeof window.auraFinishVoiceReply === 'function') {
                    window.auraFinishVoiceReply();
                }
                const wasStreaming = streaming;
                streaming = false;
                return wasStreaming;
            },

            interrupted(spoken) {
                if (typeof window.auraInterruptVoiceReply === 'function') {
                    window.auraInterruptVoiceReply(spoken || '');
                } else {
                    this.replyDone();
                }
                streaming = false;
            },

            showOverheard(text, why) {
                ensureIndicator();
                if (!overheardEl) return;
                const reason = Array.isArray(why) && why.length ? why[0] : '';
                overheardEl.textContent = reason ? `heard "${text}" — ${reason}` : `heard "${text}"`;
                indicator.classList.add('ambient-overheard-show');
                clearTimeout(overheardTimer);
                overheardTimer = setTimeout(
                    () => indicator.classList.remove('ambient-overheard-show'),
                    4200,
                );
            },
        };
    })();

    // ── DOM ──────────────────────────────────────────────────────────────

    let root = null;
    const el = {};

    function buildSurface() {
        if (root) return root;
        root = document.createElement('div');
        root.className = 'vm-root';
        root.setAttribute('role', 'dialog');
        root.setAttribute('aria-label', 'Voice conversation');
        root.setAttribute('aria-modal', 'true');
        root.hidden = true;
        root.innerHTML = `
          <div class="vm-scrim"></div>
          <div class="vm-stage">
            <div class="vm-heading"><span class="vm-identity">Aura</span><span class="vm-status" id="vm-status" role="status">Connecting</span></div>

            <div class="vm-orb-wrap" id="vm-orb-wrap">
              <div class="vm-orb-glow" id="vm-orb-glow"></div>
              <div class="vm-orb" id="vm-orb">
                <div class="vm-orb-inner"></div>
                <img class="vm-neuron" src="/static/neuron_glyph.svg" alt="">
                <div class="vm-orb-ring vm-ring-1"></div>
                <div class="vm-orb-ring vm-ring-2"></div>
              </div>
              <div class="vm-listening-dots" id="vm-dots">
                <span></span><span></span><span></span>
              </div>
            </div>

            <div class="vm-caption" id="vm-caption">
              <div class="vm-caption-user" id="vm-caption-user"></div>
              <div class="vm-caption-aura" id="vm-caption-aura"></div>
            </div>

            <div class="vm-transcript" id="vm-transcript" aria-live="polite"></div>

            <div class="vm-hint" id="vm-hint">You can interrupt me at any time.</div>
            <div class="vm-controls">
              <button type="button" class="vm-btn" id="vm-mute" title="Mute microphone" aria-label="Mute microphone">
                <span class="vm-ico" id="vm-mute-ico"></span>
                <span class="vm-btn-label">Mute</span>
              </button>
              <button type="button" class="vm-btn vm-btn-stop" id="vm-interrupt" title="Stop Aura speaking" aria-label="Stop Aura speaking">
                <span class="vm-ico vm-ico-stop"></span>
                <span class="vm-btn-label">Stop</span>
              </button>
              <button type="button" class="vm-btn" id="vm-transcript-toggle" title="Show transcript" aria-label="Show transcript">
                <span class="vm-ico vm-ico-list"></span>
                <span class="vm-btn-label">Transcript</span>
              </button>
              <button type="button" class="vm-btn" id="vm-voice-btn" title="Change voice" aria-label="Change voice">
                <span class="vm-ico vm-ico-wave"></span>
                <span class="vm-btn-label">Voice</span>
              </button>
              <button type="button" class="vm-btn vm-btn-end" id="vm-end" title="Leave voice mode" aria-label="Leave voice mode">
                <span class="vm-ico vm-ico-end"></span>
                <span class="vm-btn-label">End</span>
              </button>
            </div>

            <div class="vm-voice-picker" id="vm-voice-picker" role="listbox" aria-label="Voice"></div>

            <div class="vm-meta" id="vm-meta"></div>
          </div>`;
        document.body.appendChild(root);

        el.status = root.querySelector('#vm-status');
        el.hint = root.querySelector('#vm-hint');
        el.orbWrap = root.querySelector('#vm-orb-wrap');
        el.orb = root.querySelector('#vm-orb');
        el.glow = root.querySelector('#vm-orb-glow');
        el.dots = root.querySelector('#vm-dots');
        el.captionUser = root.querySelector('#vm-caption-user');
        el.captionAura = root.querySelector('#vm-caption-aura');
        el.transcript = root.querySelector('#vm-transcript');
        el.mute = root.querySelector('#vm-mute');
        el.muteIco = root.querySelector('#vm-mute-ico');
        el.interrupt = root.querySelector('#vm-interrupt');
        el.transcriptToggle = root.querySelector('#vm-transcript-toggle');
        el.end = root.querySelector('#vm-end');
        el.meta = root.querySelector('#vm-meta');
        el.voiceBtn = root.querySelector('#vm-voice-btn');
        el.voicePicker = root.querySelector('#vm-voice-picker');

        el.voiceBtn.addEventListener('click', () => {
            root.classList.toggle('vm-show-voices');
            el.voiceBtn.classList.toggle('vm-active');
            renderVoicePicker();
        });

        el.mute.addEventListener('click', toggleMute);
        el.interrupt.addEventListener('click', interruptPlayback);
        el.transcriptToggle.addEventListener('click', () => {
            root.classList.toggle('vm-show-transcript');
            el.transcriptToggle.classList.toggle('vm-active');
        });
        el.end.addEventListener('click', () => leaveFocusedMode());

        // Tapping the orb interrupts — the fastest possible target, and the
        // gesture people reach for instinctively when they want it to stop.
        el.orbWrap.addEventListener('click', () => {
            if (state.sessionState === 'speaking') {
                interruptPlayback();
            }
        });

        document.addEventListener('keydown', onKeyDown);
        return root;
    }

    function onKeyDown(e) {
        if (!state.active) return;
        if (state.ambient) {
            // Ambient listening does not own the keyboard. Space is a page
            // scroll and Escape closes whatever the user was actually looking
            // at; stealing either because a microphone happens to be open is
            // the kind of thing that makes a background feature feel like a
            // foreground one.
            return;
        }
        if (e.key === 'Tab') {
            const controls = [...root.querySelectorAll('button:not(:disabled), [tabindex="0"]')]
                .filter(node => node.getClientRects().length);
            const at = controls.indexOf(document.activeElement);
            if (!controls.length) return;
            if (e.shiftKey && at <= 0) { e.preventDefault(); controls[controls.length - 1].focus(); }
            else if (!e.shiftKey && (at < 0 || at === controls.length - 1)) { e.preventDefault(); controls[0].focus(); }
        } else if (e.key === 'Escape') {
            e.preventDefault();
            leaveFocusedMode();
        } else if (e.code === 'Space' && e.target === document.body) {
            // Space is the universal "stop talking" key here.
            e.preventDefault();
            if (state.sessionState === 'speaking') {
                interruptPlayback();
            }
        } else if (e.key.toLowerCase() === 'm' && e.target === document.body) {
            e.preventDefault();
            toggleMute();
        }
    }

    // ── audio capture ────────────────────────────────────────────────────

    function sendDeviceState(deviceState, reason, generation = state.captureGeneration) {
        sendCommand('device_state', {
            state: deviceState,
            reason: String(reason || '').slice(0, 160),
            generation,
        });
    }

    async function startCapture({ reason = 'capture_started' } = {}) {
        const currentTrack = state.micStream && state.micStream.getAudioTracks()[0];
        if (
            currentTrack
            && currentTrack.readyState === 'live'
            && state.captureCtx
            && state.micNode
        ) return true;

        let stream = null;
        let ctx = null;
        let node = null;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    // Not optional. See the note at the top of this file.
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    channelCount: 1,
                    sampleRate: CAPTURE_RATE,
                },
            });
            const track = stream.getAudioTracks()[0];
            if (!track) throw new Error('microphone stream has no audio track');

            ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: CAPTURE_RATE });
            await ctx.audioWorklet.addModule('/static/voice-capture-processor.js');
            const source = ctx.createMediaStreamSource(stream);
            node = new AudioWorkletNode(ctx, 'voice-capture-processor', {
                processorOptions: { targetSampleRate: CAPTURE_RATE },
            });
            source.connect(node);

            state.captureGeneration += 1;
            const generation = state.captureGeneration;
            state.micStream = stream;
            state.captureCtx = ctx;
            state.micNode = node;

            track.addEventListener('ended', () => {
                if (!state.captureStopIntentional && generation === state.captureGeneration) {
                    void recoverCapture('track_ended');
                }
            });
            track.addEventListener('mute', () => {
                clearTimeout(state.captureMuteTimer);
                state.captureMuteTimer = setTimeout(() => {
                    if (
                        track.muted
                        && !state.captureStopIntentional
                        && generation === state.captureGeneration
                    ) void recoverCapture('track_muted');
                }, 1200);
            });
            track.addEventListener('unmute', () => {
                clearTimeout(state.captureMuteTimer);
                if (generation === state.captureGeneration) {
                    sendDeviceState('active', 'track_unmuted', generation);
                }
            });

            node.port.onmessage = (e) => {
                const d = e.data;
                if (d.type === 'pcm') {
                    if (state.ready && state.ws && state.ws.readyState === WebSocket.OPEN && !state.muted) {
                        state.ws.send(d.pcm);
                    }
                } else if (d.type === 'level') {
                    state.micLevel = d.level;
                }
            };

            // Do NOT connect capture to destination: that creates feedback.
            sendDeviceState('active', reason, generation);
            return true;
        } catch (err) {
            if (node) { try { node.disconnect(); } catch (_e) { /* partial setup */ } }
            if (stream) stream.getTracks().forEach((track) => track.stop());
            if (ctx && ctx.state !== 'closed') {
                try { await ctx.close(); } catch (_e) { /* partial setup */ }
            }
            throw err;
        }
    }

    async function stopCapture({ notify = true, reason = 'capture_stopped' } = {}) {
        state.captureStopIntentional = true;
        clearTimeout(state.captureMuteTimer);
        if (state.micNode) {
            try { state.micNode.disconnect(); } catch (_e) { /* already gone */ }
            state.micNode = null;
        }
        if (state.micStream) {
            state.micStream.getTracks().forEach((track) => track.stop());
            state.micStream = null;
        }
        const ctx = state.captureCtx;
        state.captureCtx = null;
        if (ctx && ctx.state !== 'closed') {
            try { await ctx.close(); } catch (_e) { /* already closed */ }
        }
        state.micLevel = 0;
        state.captureStopIntentional = false;
        if (notify) sendDeviceState('stopped', reason);
    }

    async function recoverCapture(reason) {
        if (state.captureRecoveryPromise) return state.captureRecoveryPromise;
        state.captureRecoveryPromise = (async () => {
            state.ready = false;
            setStatus('recovering');
            sendDeviceState('recovering', reason);
            await stopCapture({ notify: false, reason });
            const delays = [0, 250, 750, 1500, 3000];
            let lastError = null;
            for (const delay of delays) {
                if (
                    !state.active
                    || state.muted
                    || !state.serverReady
                    || !state.ws
                    || state.ws.readyState !== WebSocket.OPEN
                ) return false;
                if (delay) await new Promise((resolve) => setTimeout(resolve, delay));
                try {
                    await startCapture({ reason: `recovered:${reason}` });
                    state.ready = true;
                    setStatus(state.serverState === 'idle' ? 'listening' : state.serverState);
                    showMeta('microphone recovered');
                    return true;
                } catch (err) {
                    lastError = err;
                }
            }
            sendDeviceState(
                'unavailable',
                `${reason}:${lastError && lastError.name ? lastError.name : 'reopen_failed'}`,
            );
            setStatus('error');
            showMeta('Microphone unavailable. Reconnecting a device will retry.');
            return false;
        })();
        try {
            return await state.captureRecoveryPromise;
        } finally {
            state.captureRecoveryPromise = null;
        }
    }

    async function handleMediaDeviceChange() {
        if (!state.active || state.muted || !state.serverReady) return;
        const track = state.micStream && state.micStream.getAudioTracks()[0];
        if (!track || track.readyState !== 'live') {
            await recoverCapture('device_change_no_live_track');
            return;
        }
        try {
            const currentId = String((track.getSettings && track.getSettings().deviceId) || '');
            const devices = await navigator.mediaDevices.enumerateDevices();
            const currentPresent = !currentId || devices.some(
                (device) => device.kind === 'audioinput' && device.deviceId === currentId
            );
            if (!currentPresent) await recoverCapture('active_input_removed');
        } catch (err) {
            console.warn('[voice] device-change inspection failed', err);
            if (track.readyState !== 'live') await recoverCapture('device_change_inspection_failed');
        }
    }

    async function startPlayback() {
        if (state.playbackCtx && state.speechNode && state.asideNode) return true;
        const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: PLAYBACK_RATE });
        state.playbackCtx = ctx;
        await ctx.audioWorklet.addModule('/static/voice-playback-processor.js');

        // Two independent lanes so a backchannel can overlap speech without
        // either one flushing the other.
        state.speechNode = new AudioWorkletNode(ctx, 'voice-playback-processor', {
            processorOptions: { bufferSeconds: 60, sourceSampleRate: PLAYBACK_RATE },
            outputChannelCount: [1],
        });
        state.asideNode = new AudioWorkletNode(ctx, 'voice-playback-processor', {
            processorOptions: { bufferSeconds: 10, sourceSampleRate: PLAYBACK_RATE },
            outputChannelCount: [1],
        });

        state.speechNode.port.onmessage = (e) => {
            const d = e.data;
            if (d.type === 'progress') {
                state.playedMs = d.playedMs;
                state.bufferedMs = d.bufferedMs;
                state.auraLevel = d.level;
                // The server needs this to know what was actually heard.
                if (
                    state.speechUtteranceId > 0
                    && (state.lastPlaybackReportMs < 0
                        || Math.abs(d.playedMs - state.lastPlaybackReportMs) >= 35)
                ) {
                    state.lastPlaybackReportMs = d.playedMs;
                    sendCommand('playback', {
                        played_ms: d.playedMs,
                        utterance_id: state.speechUtteranceId,
                    });
                }
            } else if (d.type === 'drained' && state.speechUtteranceId > 0) {
                state.playedMs = d.playedMs;
                state.bufferedMs = 0;
                state.lastPlaybackReportMs = d.playedMs;
                sendCommand('playback', {
                    played_ms: d.playedMs,
                    utterance_id: state.speechUtteranceId,
                    drained: true,
                });
            } else if (d.type === 'overflow') {
                console.warn('[voice] playback buffer overflow, dropped', d.dropped, 'samples');
                sendCommand('playback', {
                    played_ms: state.playedMs,
                    utterance_id: state.speechUtteranceId,
                    overflow_samples: d.dropped,
                });
            }
        };
        state.asideNode.port.onmessage = (e) => {
            if (e.data.type === 'progress') {
                state.auraLevel = Math.max(state.auraLevel, e.data.level);
            }
        };

        state.speechNode.connect(ctx.destination);
        state.asideNode.connect(ctx.destination);
        return true;
    }

    // ── websocket ────────────────────────────────────────────────────────

    function wsUrl() {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const mode = state.ambient ? 'ambient' : 'focused';
        return `${proto}//${window.location.host}/ws/voice?mode=${mode}`;
    }

    function connect() {
        state.ready = false;
        state.serverReady = false;
        state.serverState = 'idle';
        state.interruptedUtterances.clear();
        state.rejectPendingSpeech = false;
        state.rejectAsides = false;
        resetSpeechPlayback();
        const ws = new WebSocket(wsUrl());
        ws.binaryType = 'arraybuffer';
        state.ws = ws;

        ws.onopen = () => {
            state.reconnectAttempts = 0;
            setStatus('preparing');
            const token = (window.AURA_API_TOKEN || '');
            if (token) ws.send(JSON.stringify({ type: 'auth', token }));
        };

        ws.onmessage = (e) => {
            if (typeof e.data === 'string') {
                let msg;
                try { msg = JSON.parse(e.data); } catch (_err) { return; }
                handleEvent(msg);
            } else {
                handleAudio(new Uint8Array(e.data));
            }
        };

        ws.onclose = () => { void handleSocketClose(ws); };

        ws.onerror = () => { /* onclose handles recovery */ };
    }

    async function handleSocketClose(closedSocket) {
        if (state.ws !== closedSocket) return;
        state.ready = false;
        state.serverReady = false;
        await teardownAudio();
        if (state.closingIntentionally || state.terminalFailure || !state.active) return;
        scheduleReconnect();
    }

    async function activateAudioAfterReady(msg) {
        if (state.audioStartPromise) return state.audioStartPromise;
        state.audioStartPromise = (async () => {
            try {
                await startPlayback();
                if (!state.muted) await startCapture();
                await ensureAudioRunning();
                if (
                    !state.active
                    || !state.serverReady
                    || !state.ws
                    || state.ws.readyState !== WebSocket.OPEN
                ) {
                    await teardownAudio();
                    return false;
                }
                state.ready = true;
                setStatus(state.serverState === 'idle' ? 'listening' : state.serverState);
                ambient.setEnabled(state.ambient);
                if (!msg.asr_available) showMeta('Speech recognition is unavailable.');
                sendCommand('list_voices');
                sendCommand('set_floor', { open: !state.ambient });
                return true;
            } catch (err) {
                console.error('[voice] authenticated audio init failed', err);
                const denied = err && err.name === 'NotAllowedError';
                state.ready = false;
                state.terminalFailure = true;
                await teardownAudio();
                if (state.ambient) {
                    state.active = false;
                    state.ambient = false;
                    ambient.setEnabled(false);
                } else {
                    showMeta(denied ? 'Microphone permission denied.' : 'Could not start audio.');
                    setStatus('error');
                }
                if (state.ws && state.ws.readyState === WebSocket.OPEN) {
                    state.ws.close(4004, 'browser microphone unavailable');
                }
                return false;
            } finally {
                state.audioStartPromise = null;
            }
        })();
        return state.audioStartPromise;
    }

    function scheduleReconnect() {
        // Exponential backoff, capped. A voice session that silently stops
        // reconnecting looks identical to one that is merely listening,
        // so the status line always says which is happening.
        state.reconnectAttempts += 1;
        if (state.reconnectAttempts > 6) {
            setStatus('disconnected');
            showMeta('Connection lost. Press End and try again.');
            return;
        }
        const delay = Math.min(8000, 400 * Math.pow(2, state.reconnectAttempts - 1));
        setStatus('reconnecting');
        clearTimeout(state.reconnectTimer);
        state.reconnectTimer = setTimeout(() => { if (state.active) connect(); }, delay);
    }

    function sendCommand(command, extra) {
        if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
        state.ws.send(JSON.stringify(Object.assign({ command }, extra || {})));
    }

    // ── events ───────────────────────────────────────────────────────────

    function handleEvent(msg) {
        switch (msg.type) {
            case 'voice.ready':
                // Device capture begins only after socket authentication,
                // server-side microphone lease admission, and ASR/TTS warmup.
                // Before this event getUserMedia is never called.
                state.serverReady = true;
                void activateAudioAfterReady(msg);
                break;

            case 'voice.state':
                state.serverState = msg.state || 'idle';
                if (typeof msg.muted === 'boolean') {
                    state.muted = msg.muted;
                    if (el.mute) el.mute.classList.toggle('vm-active', state.muted);
                }
                setStatus(state.ready ? state.serverState : 'preparing');
                ambient.setState(msg.state);
                break;

            case 'voice.partial':
                state.partialStable = msg.stable || '';
                state.partialTentative = msg.tentative || '';
                renderUserCaption();
                break;

            case 'voice.final':
                if (msg.text && msg.addressed !== false) state.rejectPendingSpeech = false;
                state.partialStable = '';
                state.partialTentative = '';
                if (msg.text && msg.addressed === false) {
                    // She heard it and decided it was not for her. Showing
                    // nothing would make the gate invisible: the user could
                    // not tell "did not hear me" from "chose not to answer",
                    // and an ambient microphone whose decisions you cannot
                    // see is one you cannot trust. Showing it as a chat
                    // message would be worse — the thread would fill with
                    // everything said in the room. So it surfaces as a
                    // transient line that says what she heard and why she
                    // let it pass, and then goes away.
                    ambient.showOverheard(msg.text, msg.address_why || []);
                    if (el.captionUser) el.captionUser.textContent = '';
                    break;
                }
                if (msg.text) {
                    const mirrored = ambient.userSaid(msg.text);
                    addTranscript('user', msg.text, mirrored);
                    if (el.captionUser) el.captionUser.textContent = msg.text;
                } else if (el.captionUser) {
                    el.captionUser.textContent = '';
                }
                break;

            case 'voice.reply':
                state.currentReply = msg.text || '';
                state.spokenSoFar = '';
                addTranscript('aura', state.currentReply, state.currentReplyMirrored);
                state.currentReplyMirrored = false;
                break;

            case 'voice.chunk':
                if (state.rejectPendingSpeech || state.interruptedUtterances.has(Number(msg.utterance_id))) break;
                // Caption follows the audio clause by clause, so the words
                // on screen are the words currently in the air.
                if (msg.utterance_id && msg.utterance_id !== state.speechUtteranceId) {
                    beginSpeechUtterance(msg.utterance_id, msg.seq || 0);
                }
                state.spokenSoFar = (state.spokenSoFar ? state.spokenSoFar + ' ' : '') + msg.text;
                if (el.captionAura) el.captionAura.textContent = state.spokenSoFar;
                state.currentReplyMirrored = ambient.auraChunk(msg.text)
                    || state.currentReplyMirrored;
                if (state.currentReplyMirrored) markLatestAuraTranscriptMirrored();
                break;

            case 'voice.backchannel':
                flashAside(msg.text);
                break;

            case 'voice.filler':
                flashAside(msg.text);
                break;

            case 'voice.interrupted':
                if (state.speechNode) state.speechNode.port.postMessage({ type: 'flush' });
                markInterrupted(msg.spoken || '');
                ambient.interrupted(msg.spoken || '');
                state.currentReplyMirrored = true;
                break;

            case 'voice.metrics_done':
                ambient.replyDone();
                break;

            case 'voice.flush':
                if (state.speechNode) state.speechNode.port.postMessage({ type: 'flush' });
                break;

            case 'voice.duck':
                // She lowers her voice the instant you start talking, and
                // comes back up if it turns out you were just saying "mhm".
                if (state.speechNode) {
                    state.speechNode.port.postMessage({
                        type: 'duck', gain: msg.gain, ramp_ms: msg.ramp_ms,
                    });
                }
                if (root) root.classList.toggle('vm-ducked', (msg.gain || 1) < 0.9);
                break;

            case 'voice.style':
                showMeta(`voice: ${msg.change}`);
                break;

            case 'voice.voices':
                state.voices = msg.voices || [];
                state.currentVoice = msg.current || '';
                renderVoicePicker();
                break;

            case 'voice.metrics':
                state.metrics = msg;
                // Metrics are emitted once the utterance has actually been
                // heard, which is the honest moment to close the chat bubble.
                ambient.replyDone();
                break;

            case 'voice.device':
                if (msg.state === 'recovering') showMeta('microphone reconnecting');
                if (msg.state === 'unavailable') showMeta('microphone unavailable');
                if (msg.state === 'active' && String(msg.reason || '').startsWith('recovered:')) {
                    showMeta('microphone recovered');
                }
                break;

            case 'voice.error':
                showMeta(msg.message || 'Voice error');
                setStatus('error');
                if (new Set([
                    'paired_device_voice_scope_denied',
                    'paired_device_session_revoked',
                    'owner_session_revoked',
                    'voice_disabled_in_settings',
                    'voice_session_capacity_reached',
                    'microphone_owner_disabled',
                    'microphone_device_busy',
                    'microphone_lease_revoked',
                ]).has(msg.status)) {
                    state.terminalFailure = true;
                    state.ready = false;
                    state.serverReady = false;
                    void stopCapture();
                    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
                        state.ws.close(4004, 'terminal voice admission failure');
                    }
                }
                break;

            default:
                break;
        }
    }

    function handleAudio(bytes) {
        if (bytes.length < HEADER_BYTES) return;
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const opcode = view.getUint8(0);
        const flags = view.getUint8(1);
        const seq = view.getUint16(2, true);
        const utteranceId = view.getUint32(4, true);
        const payload = bytes.subarray(HEADER_BYTES);

        if (opcode === OPCODE.SPEECH) {
            if (state.rejectPendingSpeech || state.interruptedUtterances.has(utteranceId)) return;
            if (utteranceId !== state.speechUtteranceId) {
                beginSpeechUtterance(utteranceId, seq);
            } else if (seq !== state.expectedSpeechSeq) {
                console.warn('[voice] non-contiguous speech sequence', {
                    utteranceId, expected: state.expectedSpeechSeq, received: seq,
                });
                // A missing audio frame makes transcript/playback accounting
                // uncertain. Flush rather than joining two non-contiguous
                // waveform regions with an audible click.
                if (state.speechNode) state.speechNode.port.postMessage({ type: 'flush' });
                state.expectedSpeechSeq = seq;
            }
            if (payload.length > 0) state.expectedSpeechSeq = (seq + 1) & 0xffff;
            if ((flags & 1) && payload.length === 0) return;
        }
        if (payload.length === 0 || (opcode !== OPCODE.SPEECH && state.rejectAsides)) return;

        // Copy: the worklet takes ownership of the buffer it is handed.
        const pcm = new Int16Array(payload.length / 2);
        for (let i = 0; i < pcm.length; i++) {
            pcm[i] = view.getInt16(HEADER_BYTES + i * 2, true);
        }

        const node = opcode === OPCODE.SPEECH ? state.speechNode : state.asideNode;
        if (!node) return;
        node.port.postMessage({ type: 'push', pcm: pcm.buffer }, [pcm.buffer]);
    }

    function interruptPlayback() {
        if (!state.active) return;
        if (state.speechUtteranceId) {
            state.interruptedUtterances.add(state.speechUtteranceId);
            if (state.interruptedUtterances.size > 32) state.interruptedUtterances.delete(state.interruptedUtterances.values().next().value);
        }
        state.rejectAsides = true;
        state.rejectPendingSpeech = true;
        // Silence the audio thread before waiting for the server's receipt.
        for (const node of [state.speechNode, state.asideNode]) node?.port.postMessage({ type: 'flush' });
        state.auraLevel = 0;
        setStatus('listening');
        sendCommand('barge_in', { played_ms: state.playedMs, utterance_id: state.speechUtteranceId });
    }

    function beginSpeechUtterance(utteranceId, seq) {
        state.rejectAsides = false;
        state.speechUtteranceId = Number(utteranceId || 0);
        state.expectedSpeechSeq = Number(seq || 0);
        state.playedMs = 0;
        state.bufferedMs = 0;
        state.lastPlaybackReportMs = -1;
        if (state.speechNode) state.speechNode.port.postMessage({ type: 'reset' });
    }

    function resetSpeechPlayback() {
        state.speechUtteranceId = 0;
        state.expectedSpeechSeq = 0;
        state.playedMs = 0;
        state.bufferedMs = 0;
        state.lastPlaybackReportMs = -1;
        if (state.speechNode) state.speechNode.port.postMessage({ type: 'reset' });
    }

    // ── rendering ────────────────────────────────────────────────────────

    function setStatus(s) {
        state.sessionState = s;
        if (!el.status) return;
        const labels = {
            idle: 'ready', listening: 'listening', user_speaking: 'listening',
            thinking: 'thinking', speaking: 'speaking', closed: 'ended',
            connecting: 'connecting', reconnecting: 'reconnecting',
            preparing: 'preparing voice', disconnected: 'disconnected', error: 'error',
        };
        el.status.textContent = state.muted ? 'Microphone off' : (labels[s] || s);
        if (el.hint) el.hint.textContent = state.muted ? 'Your microphone is off.' : s === 'speaking' ? 'Speak to interrupt, or tap Stop.' : s === 'thinking' ? 'I’m with you.' : s === 'connecting' || s === 'preparing' ? 'Getting ready to listen.' : 'You can speak naturally.';
        if (el.interrupt) el.interrupt.disabled = s !== 'speaking';
        if (el.mute) {
            el.mute.setAttribute('aria-pressed', String(state.muted));
            el.mute.setAttribute('aria-label', state.muted ? 'Unmute microphone' : 'Mute microphone');
            el.mute.querySelector('.vm-btn-label').textContent = state.muted ? 'Unmute' : 'Mute';
        }
        if (root) {
            root.dataset.state = s;
        }
    }

    function renderUserCaption() {
        if (!el.captionUser) return;
        const stable = state.partialStable;
        const tail = state.partialTentative;
        // The unstable tail is rendered faintly: Whisper may still revise it,
        // and showing it at full weight makes the caption look like it is
        // correcting itself constantly.
        el.captionUser.innerHTML =
            escapeHtml(stable) + (tail ? ` <span class="vm-tentative">${escapeHtml(tail)}</span>` : '');
    }

    function addTranscript(who, text, mirroredToChat = false) {
        if (!text) return;
        state.transcript.push({ who, text, at: Date.now(), mirroredToChat });
        const line = document.createElement('div');
        line.className = `vm-line vm-line-${who}`;
        line.innerHTML = `<span class="vm-who">${who === 'user' ? 'You' : 'Aura'}</span><span class="vm-text">${escapeHtml(text)}</span>`;
        el.transcript.appendChild(line);
        el.transcript.scrollTop = el.transcript.scrollHeight;
    }

    function markLatestAuraTranscriptMirrored() {
        for (let i = state.transcript.length - 1; i >= 0; i--) {
            if (state.transcript[i].who !== 'aura') continue;
            state.transcript[i].mirroredToChat = true;
            return;
        }
    }

    function markInterrupted(spoken) {
        for (let i = state.transcript.length - 1; i >= 0; i--) {
            if (state.transcript[i].who !== 'aura') continue;
            state.transcript[i].text = spoken;
            state.transcript[i].mirroredToChat = true;
            break;
        }
        const lines = el.transcript.querySelectorAll('.vm-line-aura');
        const last = lines[lines.length - 1];
        if (!last) return;
        last.classList.add('vm-interrupted');
        const textEl = last.querySelector('.vm-text');
        if (textEl && spoken) {
            // Show exactly what was heard, with the rest struck through —
            // this mirrors what her memory now holds.
            const full = textEl.textContent || '';
            const rest = full.startsWith(spoken) ? full.slice(spoken.length) : '';
            textEl.innerHTML = escapeHtml(spoken) +
                (rest ? `<span class="vm-unheard">${escapeHtml(rest)}</span>` : '');
        }
        el.captionAura.textContent = spoken;
    }

    // Kokoro's ids are opaque ("af_heart"). Render them as a language/gender
    // grouping plus a readable name, so choosing a voice is a two-second
    // decision instead of a lookup table.
    const VOICE_PREFIX = {
        af: 'US · female', am: 'US · male',
        bf: 'UK · female', bm: 'UK · male',
        ef: 'ES · female', em: 'ES · male',
        ff: 'FR · female', hf: 'HI · female', hm: 'HI · male',
        if: 'IT · female', im: 'IT · male',
        jf: 'JP · female', jm: 'JP · male',
        pf: 'PT · female', pm: 'PT · male',
        zf: 'ZH · female', zm: 'ZH · male',
    };

    function prettyVoice(id) {
        const [prefix, ...rest] = String(id).split('_');
        const name = rest.join('_') || id;
        return {
            group: VOICE_PREFIX[prefix] || 'other',
            name: name.charAt(0).toUpperCase() + name.slice(1),
        };
    }

    function renderVoicePicker() {
        if (!el.voicePicker) return;
        if (!state.voices.length) {
            el.voicePicker.innerHTML = '<div class="vm-voice-empty">No alternate voices available.</div>';
            return;
        }
        const groups = new Map();
        for (const id of state.voices) {
            const { group, name } = prettyVoice(id);
            if (!groups.has(group)) groups.set(group, []);
            groups.get(group).push({ id, name });
        }
        el.voicePicker.innerHTML = '';
        for (const [group, items] of groups) {
            const wrap = document.createElement('div');
            wrap.className = 'vm-voice-group';
            wrap.innerHTML = `<div class="vm-voice-group-label">${escapeHtml(group)}</div>`;
            const row = document.createElement('div');
            row.className = 'vm-voice-row';
            for (const item of items) {
                const chip = document.createElement('button');
                chip.type = 'button';
                chip.className = 'vm-voice-chip' + (item.id === state.currentVoice ? ' vm-active' : '');
                chip.textContent = item.name;
                chip.setAttribute('role', 'option');
                chip.setAttribute('aria-selected', String(item.id === state.currentVoice));
                chip.addEventListener('click', () => {
                    state.currentVoice = item.id;
                    sendCommand('set_voice', { voice: item.id });
                    // Speak a line in the new voice immediately: picking a
                    // voice you cannot hear is guesswork.
                    sendCommand('text', { text: 'Okay — this is how I sound now.' });
                    renderVoicePicker();
                });
                row.appendChild(chip);
            }
            wrap.appendChild(row);
            el.voicePicker.appendChild(wrap);
        }
    }

    function flashAside(text) {
        const node = document.createElement('div');
        node.className = 'vm-aside';
        node.textContent = text;
        el.orbWrap.appendChild(node);
        setTimeout(() => node.remove(), 1800);
    }

    function showMeta(text) {
        if (!el.meta) return;
        el.meta.textContent = text;
        el.meta.classList.add('vm-meta-show');
        clearTimeout(el.metaTimer);
        el.metaTimer = setTimeout(() => el.meta.classList.remove('vm-meta-show'), 3200);
    }

    function escapeHtml(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // ── animation ────────────────────────────────────────────────────────

    function tick() {
        if (!state.active) return;
        // The orb answers to whoever is talking: the user's mic level while
        // listening, her own output level while speaking. That single rule
        // makes turn-taking legible without any text.
        const speaking = state.sessionState === 'speaking';
        const level = speaking ? state.auraLevel : state.micLevel;
        const eased = Math.min(1, Math.pow(level * (speaking ? 2.6 : 4.2), 0.62));

        const scale = 1 + eased * (speaking ? 0.26 : 0.17);
        const glow = 0.25 + eased * 0.75;

        const animate = !document.hidden && !state.ambient && root?.classList.contains('vm-open');
        const motion = !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
        if (animate && motion && (state.lastVisualScale === null || Math.abs(scale - state.lastVisualScale) > .006)) {
            if (el.orb) el.orb.style.transform = `scale(${scale.toFixed(3)})`;
            state.lastVisualScale = scale;
        }
        if (animate && (state.lastVisualGlow === null || Math.abs(glow - state.lastVisualGlow) > .02)) {
            if (el.glow) el.glow.style.opacity = glow.toFixed(2);
            state.lastVisualGlow = glow;
        }

        state.rafHandle = requestAnimationFrame(tick);
    }

    // ── entry / exit ─────────────────────────────────────────────────────

    // The surface is always built, even for an ambient session that never
    // shows it. Guarding twenty `el.*` references on a flag is how a UI grows
    // two subtly different code paths; building the DOM and simply not
    // revealing it costs a few nodes and keeps one path.
    async function openSession({ ambient: wantAmbient }) {
        if (state.active) {
            // Already listening. A focused request just reveals the surface.
            if (!wantAmbient && state.ambient) surface(true);
            return true;
        }
        buildSurface();
        state.active = true;
        state.ambient = Boolean(wantAmbient);
        state.closingIntentionally = false;
        state.terminalFailure = false;
        state.serverReady = false;
        setStatus('connecting');
        surface(!state.ambient);
        // Connect first. The server authenticates the principal, admits the
        // canonical microphone lease, and warms both models before emitting
        // voice.ready. Only that event is allowed to call getUserMedia.
        connect();
        state.rafHandle = requestAnimationFrame(tick);
        return true;
    }

    function surface(show) {
        if (!root) return;
        if (show) {
            // Animate in on the next frame so the transition never appears to
            // hang on the click that started it.
            state.returnFocus = document.activeElement;
            root.hidden = false;
            requestAnimationFrame(() => { root.classList.add('vm-open'); el.mute.focus(); });
            document.body.classList.add('vm-active');
            state.ambient = false;
        } else {
            root.classList.remove('vm-open');
            root.hidden = true;
            document.body.classList.remove('vm-active');
            state.returnFocus?.focus?.();
        }
        // Opening the focused surface is the user saying "everything I say
        // now is for you". The server stands its addressivity gate down for
        // the duration; without this, focused voice mode would still be
        // second-guessing whether it was being spoken to, which is the one
        // place that judgement is not wanted.
        sendCommand('set_floor', { open: Boolean(show) });
    }

    async function enterVoiceMode() {
        return openSession({ ambient: false });
    }

    async function enterAmbient() {
        return openSession({ ambient: true });
    }

    async function teardownAudio() {
        await stopCapture();
        if (state.speechNode) { try { state.speechNode.disconnect(); } catch (_e) { /* already gone */ } state.speechNode = null; }
        if (state.asideNode) { try { state.asideNode.disconnect(); } catch (_e) { /* already gone */ } state.asideNode = null; }
        for (const key of ['playbackCtx']) {
            const ctx = state[key];
            if (ctx && ctx.state !== 'closed') { try { await ctx.close(); } catch (_e) { /* already closed */ } }
            state[key] = null;
        }
        state.ready = false;
        state.serverReady = false;
        resetSpeechPlayback();
    }

    async function ensureAudioRunning() {
        for (const ctx of [state.captureCtx, state.playbackCtx]) {
            if (!ctx || ctx.state === 'running' || ctx.state === 'closed') continue;
            try { await ctx.resume(); } catch (_e) { /* next user gesture retries */ }
        }
        const suspended = [state.captureCtx, state.playbackCtx]
            .some((ctx) => ctx && ctx.state === 'suspended');
        if (suspended && !state.ambient) showMeta('Tap once to enable audio.');
        return !suspended;
    }

    // Leaving the focused surface is not the same as ending the conversation.
    // If ambient listening is on, closing the modal drops back to it rather
    // than shutting the microphone — otherwise "End" would silently turn off
    // a setting the user never touched.
    async function leaveFocusedMode() {
        if (!state.active) return;
        if (ambient.enabledByUser()) {
            surface(false);
            state.ambient = true;
            exportTranscript();
            return;
        }
        await exitVoiceMode();
    }

    async function exitVoiceMode() {
        if (!state.active) return;
        state.active = false;
        state.ambient = false;
        state.closingIntentionally = true;
        state.serverReady = false;
        clearTimeout(state.reconnectTimer);
        if (state.rafHandle) cancelAnimationFrame(state.rafHandle);

        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            sendCommand('stop');
            state.ws.close(1000, 'user ended voice mode');
        }
        state.ws = null;
        await teardownAudio();

        // Hand the conversation back to the text thread so nothing said out
        // loud is lost when the surface closes.
        exportTranscript();

        root.classList.remove('vm-open');
        root.hidden = true;
        document.body.classList.remove('vm-active');
        state.returnFocus?.focus?.();
        setTimeout(() => { if (!state.active && root) root.classList.remove('vm-show-transcript'); }, 400);
    }

    function exportTranscript() {
        if (!state.transcript.length) return;
        const sink = window.auraAppendVoiceTranscript;
        if (typeof sink === 'function') {
            const missing = state.transcript.filter((line) => !line.mirroredToChat);
            try { if (missing.length) sink(missing); } catch (err) { console.warn('[voice] transcript export failed', err); }
        }
        state.transcript = [];
        if (el.transcript) el.transcript.innerHTML = '';
    }

    async function toggleMute() {
        const muting = !state.muted;
        if (muting) {
            state.muted = true;
            sendCommand('mute');
            // Muted means the physical browser track is gone, not merely that
            // PCM is discarded after capture. Reacquisition happens only from
            // an already-authenticated, still-authoritative voice session.
            await stopCapture();
        } else {
            if (!state.serverReady || !state.ws || state.ws.readyState !== WebSocket.OPEN) {
                showMeta('Voice is not ready yet.');
                return;
            }
            try {
                await startCapture();
                await ensureAudioRunning();
                state.muted = false;
                sendCommand('unmute');
            } catch (err) {
                console.error('[voice] microphone reacquisition failed', err);
                state.muted = true;
                sendCommand('mute');
                showMeta('Microphone could not be reopened.');
            }
        }
        el.mute.classList.toggle('vm-active', state.muted);
        el.muteIco.classList.toggle('vm-ico-muted', state.muted);
        setStatus(state.sessionState);
    }

    // ── ambient startup ──────────────────────────────────────────────────

    /**
     * Start listening at launch, if — and only if — that is already settled.
     *
     * Two consents are required and neither is asked for here. The browser
     * must already hold a microphone grant from a previous, deliberate act,
     * and the user must not have turned ambient listening off. A page that
     * pops a microphone prompt on load has decided something on the user's
     * behalf; the VOICE button is where that decision belongs, because
     * pressing it *is* the request.
     *
     * So the first run of a fresh profile is silent, and every run after the
     * user has once said yes is ambient. That is the behaviour people
     * actually want from something living on their machine: it is there when
     * they open it, and it never once surprises them into being recorded.
     */
    async function maybeStartAmbient() {
        if (!ambient.enabledByUser()) return false;
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return false;

        let granted = false;
        try {
            if (navigator.permissions && navigator.permissions.query) {
                const status = await navigator.permissions.query({ name: 'microphone' });
                granted = status.state === 'granted';
                // If the user grants it later from the browser's own UI,
                // honour that without making them reload.
                status.onchange = () => {
                    if (status.state === 'granted' && !state.active) enterAmbient();
                };
            }
        } catch (_e) {
            // Firefox has historically rejected the 'microphone' descriptor.
            // No permission API means no way to know without prompting, and
            // prompting is the thing being avoided, so stay quiet.
            granted = false;
        }
        if (!granted) return false;
        return enterAmbient();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => { void maybeStartAmbient(); });
    } else {
        void maybeStartAmbient();
    }

    // Browsers may create a suspended AudioContext even when microphone
    // permission already exists. Any later deliberate gesture is sufficient
    // to resume it; install one bounded shared retry instead of letting a
    // session remain visibly active but inaudible.
    document.addEventListener('pointerdown', () => {
        if (state.active) void ensureAudioRunning();
    }, { passive: true });
    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
        navigator.mediaDevices.addEventListener('devicechange', () => {
            void handleMediaDeviceChange();
        });
    }

    window.AuraVoiceMode = {
        enter: enterVoiceMode,
        exit: exitVoiceMode,
        // The VOICE button toggles the *focused* surface. When ambient
        // listening is on, closing it returns to ambient rather than going
        // deaf — the button controls the window, not the microphone.
        toggle: async () => {
            if (state.active && state.sessionState === 'error') {
                await exitVoiceMode();
                return enterVoiceMode();
            }
            if (state.active && !state.ambient) return leaveFocusedMode();
            return enterVoiceMode();
        },
        isActive: () => state.active,
        isAmbient: () => state.active && state.ambient,
        ambientEnabled: () => ambient.enabledByUser(),
        setAmbient: async (on) => {
            ambient.setPreference(on);
            if (on) return enterAmbient();
            await exitVoiceMode();
            ambient.setEnabled(false);
            return false;
        },
        setVoice: (v) => sendCommand('set_voice', { voice: v }),
        voices: () => state.voices.slice(),
        status: () => ({
            state: state.sessionState,
            ambient: state.ambient,
            muted: state.muted,
            metrics: state.metrics,
        }),
    };
})();
