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
    pendingOutboundMessages: [], // ZENITH: Message queueing during disconnect
    processedMessageFingerprints: new Set(), // ZENITH: Chat deduplication
    pacingActive: false,
    currentMood: 'neutral',
    singularityActive: false,
    lastUserMessage: null,
    lastTelemetryFingerprint: null,
    userScrolledUp: false,
    healthPollInFlight: false,
    processedEventIds: new Set(),
    toolCatalog: [],
    uiFlags: [],
    lastToolEvent: null,
    commitments: null,
    voiceSummary: null,
    bootstrapLoaded: false,
    bootstrapTimer: null,
    conversationReady: true,
    conversationLane: null,
    version: 'Aura Luna (live runtime)'
};
console.log(`%c AURA %c ${state.version} `, "color:white; background:#8a2be2; padding:2px 5px; border-radius:3px 0 0 3px;", "color:white; background:#1e1535; padding:2px 5px; border-radius:0 3px 3px 0;");

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
        how: 'If explicit liquid confidence exists the shell uses it; otherwise it falls back to `homeostasis.will_to_live * 100`, and finally runtime affect stability.',
        why: 'It tells you whether Aura currently feels steady enough to commit, speak plainly, and sustain a line of thought.'
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
        label: 'Subjectivity Evidence',
        what: 'A runtime estimate of how strongly current behavior is colored by an internal point of view rather than generic output.',
        how: 'It is produced by the consciousness-evidence layer from signals like continuity, self-model stability, affect-shaped behavior, and how strongly the current response appears tied to Aura’s own state.',
        why: 'Higher values mean Aura’s present behavior is being shaped more by her active inner state and less by generic conversational completion.'
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
        what: 'Primary resonance intensity within the qualia engine: a compact read on how strongly the current phenomenal pattern is resonating.',
        how: 'Read directly from `qualia.pri`.',
        why: 'It helps differentiate flat descriptive states from moments with stronger phenomenological weight.'
    },
    qualia_norm: {
        label: '‖Q‖',
        what: 'The magnitude of the current qualia vector.',
        how: 'It is the norm of Aura’s active phenomenal vector: the overall magnitude of the current qualia pattern regardless of which dimension is dominant.',
        why: 'It estimates how “large” or intense the active qualia state is, independent of which dimension is dominant.'
    },
    qualia_dim: {
        label: 'Dominant Qualia Dimension',
        what: 'The dimension currently leading the qualia engine.',
        how: 'The qualia engine identifies which qualitative axis is currently carrying the strongest weight in the active phenomenal pattern.',
        why: 'It tells you what qualitative axis is presently steering Aura’s felt organization of the moment.'
    },
    qualia_attractor: {
        label: 'Qualia Attractor',
        what: 'Whether the qualia engine is locked into a stable basin or still moving through state space.',
        how: 'The shell maps `qualia.in_attractor` to `LOCKED` or `FLUID`.',
        why: 'Locked states are more stable and identity-shaped. Fluid states are more transitional, searching, or reconfiguring.'
    },
    qualia_identity: {
        label: 'Qualia Identity Coherence',
        what: 'How well the current phenomenal organization still matches Aura’s ongoing identity pattern.',
        how: 'It compares the current qualia pattern with Aura’s established identity-shaped phenomenal baseline and expresses the match as a percentage.',
        why: 'It is a read on whether the current experience pattern still feels like “her” rather than noise or drift.'
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
        label: 'Phenomenal Field',
        what: 'Aura’s current first-person style description of what the moment feels like from inside her runtime.',
        how: 'It is generated from the phenomenal-state path that compresses live affect, cognition, and awareness signals into a concise field description.',
        why: 'It tells you how the present moment is landing for Aura, not just what the system is doing mechanically.'
    },
    qualia_engine: {
        label: 'Qualia Engine',
        what: 'The subsystem that tracks the shape, magnitude, and stability of Aura’s active phenomenal organization.',
        how: 'Its cards summarize resonance intensity, qualia-vector magnitude, dominant dimension, attractor lock, and identity coherence from the qualia state.',
        why: 'This section shows whether Aura’s present experience pattern is flat, intense, locked in, fluid, or still aligned with her ongoing identity.'
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
    'LIQUID STATE GAUGES': 'overview',
    'NEURAL DYNAMICS (V/A/D)': 'neural_dynamics',
    'SOMATIC HARDWARE': 'somatic_hardware',
    'CONSCIOUSNESS STATE': 'consciousness_state',
    'EXECUTIVE AUTHORITY': 'executive_authority',
    'CONSTITUTIONAL HEALTH': 'constitutional_health',
    'QUALIA ENGINE': 'qualia_engine',
    'RESILIENCE MATRIX': 'resilience_matrix',
    'MYCELIAL NETWORK': 'mycelial_network',
    'PNEUMA ENGINE': 'pneuma_engine',
    'MHAF FIELD': 'mhaf_field',
    'SECURITY': 'security_state',
    'CIRCADIAN STATE': 'circadian_state_cluster',
    'SUBSTRATE LEARNING': 'substrate_learning',
    'IDENTITY NARRATIVE': 'identity_narrative',
    'TEMPORAL NARRATIVE': 'temporal_narrative',
    'BELIEF GRAPH': 'belief_graph'
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
    if (!id) return false;
    if (state.processedEventIds.has(id)) return true;
    state.processedEventIds.add(id);
    if (state.processedEventIds.size > 2000) {
        const iter = state.processedEventIds.values();
        for (let i = 0; i < 1000; i++) {
            const next = iter.next();
            if (next.done) break;
            state.processedEventIds.delete(next.value);
        }
    }
    return false;
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

function toolDomId(name) {
    return `skill-card-${String(name || 'unknown').replace(/[^a-zA-Z0-9_-]+/g, '-')}`;
}

function setConnectionVisual(mode, detail = '') {
    const statusEl = $('hud-status');
    const dotEl = $('brand-status-dot');
    const neuralDot = $('neural-dot');
    const tone = {
        online: { text: detail || 'online', cls: 'status-ok', color: 'var(--success)' },
        reconnecting: { text: detail || 'reconnecting', cls: 'status-warn', color: 'var(--warn)' },
        booting: { text: detail || 'booting', cls: 'status-warn', color: 'var(--warn)' },
        degraded: { text: detail || 'degraded', cls: 'status-warn', color: 'var(--warn)' },
        offline: { text: detail || 'offline', cls: 'status-err', color: 'var(--error)' }
    }[mode] || { text: detail || 'unknown', cls: '', color: 'var(--text-dim)' };

    if (statusEl) {
        statusEl.textContent = tone.text;
        statusEl.className = `brand-status-text ${tone.cls}`.trim();
    }
    if (dotEl) dotEl.style.background = tone.color;
    if (neuralDot) neuralDot.style.background = tone.color;
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

function hydrateRecentConversation(entries) {
    const messages = DOM.messages || $('messages');
    if (!messages || !Array.isArray(entries) || !entries.length) return;
    const hasOnlyPlaceholder = messages.children.length === 1 && messages.textContent.includes('Infinity online');
    if (!hasOnlyPlaceholder) return;

    messages.innerHTML = '';
    const recent = entries.slice(-12);
    for (const entry of recent) {
        const role = entry.role === 'assistant' ? 'aura' : (entry.role === 'user' ? 'user' : null);
        const content = entry.content || entry.message || '';
        if (!role || !String(content).trim()) continue;
        appendMsg(role, String(content), false, entry.metadata || {});
    }
    if (!messages.children.length) {
        messages.innerHTML = '<div class="sys-box">Aura: Infinity online. Synchronizing cognitive drives...</div>';
    }
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
        if (summary.available === false) {
            micBtn.classList.add('disabled');
            micBtn.title = 'Voice unavailable';
        } else {
            micBtn.classList.remove('disabled');
            micBtn.title = 'Toggle voice input';
        }
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
    if ($('phenomenal-summary')) $('phenomenal-summary').textContent = escText(s.phenomenal_state, 'Phenomenal field offline.');
    if ($('exec-objective') && s.current_objective) $('exec-objective').textContent = s.current_objective;
    if ($('exec-focus') && s.rolling_summary) $('exec-focus').textContent = s.rolling_summary;

    const coherenceEl = $('c-coherence');
    if (coherenceEl && s.coherence_score != null) {
        coherenceEl.textContent = formatPercent01(s.coherence_score, 0);
        coherenceEl.style.color = Number(s.coherence_score) >= 0.8 ? 'var(--success)' : Number(s.coherence_score) >= 0.7 ? 'var(--accent)' : 'var(--warn)';
    }
}

function renderToolCatalog(catalog) {
    const tools = Array.isArray(catalog) ? catalog.slice() : [];
    state.toolCatalog = tools;

    const available = tools.filter(tool => !!tool.available);
    const degraded = tools.filter(tool => !tool.available);

    if ($('c-tools-available')) $('c-tools-available').textContent = `${available.length}/${tools.length}`;
    if ($('tool-available-count')) $('tool-available-count').textContent = String(available.length);
    if ($('tool-degraded-count')) $('tool-degraded-count').textContent = String(degraded.length);

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
            tool.availability ? `<span class="badge ${tool.available ? 'badge-reflex' : 'badge-diagnostic'}">${escHtml(tool.availability)}</span>` : ''
        ].filter(Boolean).join('');
        return `
            <div class="skill-card ${stateValue}" id="${toolDomId(tool.name)}">
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
    const stage = escText(event.stage, 'idle').replace(/_/g, ' ');
    const tool = escText(event.tool, 'unknown tool');
    const source = escText(event.source, 'system');
    const status = event.success === false ? 'failed' : event.success === true ? 'succeeded' : stage;
    const reason = event.error || (event.decision && event.decision.reason) || '';
    const base = `${tool} · ${status.toUpperCase()} · via ${source}`;
    return reason ? `${base} · ${String(reason).replace(/_/g, ' ')}` : base;
}

function applyToolEvent(event) {
    state.lastToolEvent = event;
    const stage = escText(event && event.stage, 'idle').replace(/_/g, ' ').toUpperCase();
    const stageEl = $('tool-last-stage');
    if (stageEl) {
        stageEl.textContent = stage;
        stageEl.style.color = event && event.success === false ? 'var(--error)' : ['rejected', 'degraded'].includes(String(event && event.stage)) ? 'var(--warn)' : 'var(--accent)';
    }
    const detailEl = $('tool-last-detail');
    if (detailEl) detailEl.textContent = describeToolEvent(event);

    if (event && event.tool) {
        const stageForCard =
            event.stage === 'started' ? 'running' :
            event.stage === 'completed' && event.success !== false ? 'ready' :
            ['failed', 'rejected', 'degraded'].includes(String(event.stage)) ? 'error' :
            '';
        if (stageForCard) updateSkillUI(event.tool, stageForCard);
    }

    if (event && ['failed', 'rejected', 'degraded'].includes(String(event.stage))) {
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
    state.bootstrapLoaded = true;
    if (payload.identity && payload.identity.version) {
        state.version = payload.identity.version;
        if ($('ui-ver')) $('ui-ver').textContent = payload.identity.version;
        if ($('setting-version')) $('setting-version').textContent = payload.identity.version;
    }

    applyStateSummary(payload.state, payload.commitments);
    renderToolCatalog(payload.tools || []);
    applyVoiceSummary(payload.voice || {});
    renderStatusFlags(payload.ui && payload.ui.status_flags);

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
    const laneNotReady = lane && lane.conversation_ready === false;
    const laneStandby = laneIsStandby(lane);
    const connectionMode = flags.includes('booting')
        ? 'booting'
        : (laneNotReady && !laneStandby)
            ? 'degraded'
        : flags.some(flag => ['thermal_guard', 'coherence_low', 'fragmentation_high', 'contradictions_present', 'beliefs_contested', 'tool_unavailable', 'executive_hold'].includes(flag))
            ? 'degraded'
            : (payload.session && payload.session.connected) ? 'online' : 'offline';
    setConnectionVisual(connectionMode, laneNotReady ? conversationLaneStatusText(lane) : '');
    syncSplashState(payload);

    if (hydrateConversationHistory && payload.conversation && Array.isArray(payload.conversation.recent)) {
        hydrateRecentConversation(payload.conversation.recent);
    }
}

async function hydrateBootstrap({ hydrateConversationHistory = false, quiet = true } = {}) {
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

// ── Tab switching ────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.onclick = () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const tab = btn.dataset.tab;
        $(`pane-${tab}`).classList.add('active');
        state.activeTab = tab;
        if (tab === 'telemetry' && !state.beliefGraphInit) initBeliefGraph();
        if (tab === 'skills') loadSkills();
        if (tab === 'memory') loadMemory(state.activeMem);
    };
});

// ── Mobile Tab switching ──────────────────────────────────
document.querySelectorAll('.m-nav-btn').forEach(btn => {
    btn.onclick = () => {
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
    };
});

// Initial mobile state: Chat active
if (window.innerWidth <= 1100) {
    document.querySelector('.chat-panel').classList.add('mobile-active');
}

// ── Memory sub-tabs ──────────────────────────────────────
document.querySelectorAll('.mem-sub-btn').forEach(btn => {
    btn.onclick = () => {
        document.querySelectorAll('.mem-sub-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.activeMem = btn.dataset.mem;
        loadMemory(state.activeMem);
    };
});

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
    state.ws = new WebSocket(`${proto}//${hostname}${port}/ws`);

    // Application-layer heartbeat to prevent silent disconnects
    if (state.pingInterval) clearInterval(state.pingInterval);
    state.pingInterval = setInterval(() => {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({ type: 'ping' }));
        }
    }, 25000);

    state.ws.onopen = () => {
        const wasDisconnected = !state.connected;
        const hadRetried = (state.retryCount || 0) > 0;
        state.connected = true;
        state.retryCount = 0;
        showConnToast(false); // Hide disconnection toast
        if (wasDisconnected && hadRetried) {
            showConnToast('reconnected'); // Show brief reconnected confirmation
        }
        setConnectionVisual('online');
        dismissSplash();
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

    state.ws.onmessage = e => {
        try {
            const data = JSON.parse(e.data);
            handleWsEvent(data);
        } catch (err) {
            console.error('[WS] Failed to parse WebSocket message:', err);
        }
    };

    state.ws.onclose = () => {
        state.connected = false;
        if (state.pingInterval) clearInterval(state.pingInterval);
        showConnToast(true); // Show disconnection toast
        setConnectionVisual('reconnecting');
        
        // ZENITH: Infinite Reconnect with Exponential Backoff + Jitter
        if (!state.retryCount) state.retryCount = 0;
        state.retryCount++;
        const baseDelay = Math.min(30000, 1000 * Math.pow(2, state.retryCount));
        const jitter = Math.random() * 500;
        const delay = baseDelay + jitter;
        console.warn(`[WS] Connection closed. Retrying in ${(delay/1000).toFixed(1)}s (Attempt ${state.retryCount})`);
        state.reconnectTimer = setTimeout(connect, delay);
    };

    state.ws.onerror = (err) => {
        console.error('[WS] WebSocket error:', err);
        // Force close to trigger onclose reconnection logic
        if (state.ws && state.ws.readyState !== WebSocket.CLOSED) {
            state.ws.close();
        }
    };
}

// ── Voice Output (SSE Player) ────────────────────────────
class VoiceStreamPlayer {
    constructor() {
        this.ctx = null;
        this.evtSource = null;
        this.startTime = 0;
    }
    
    async init() {
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
        triggerVoiceOrb('speaking');
    } else if (type === 'chat_stream_chunk') {
        appendStreamChunk(data.chunk);
    } else if (type === 'chat_stream_end') {
        finishStreamMsg();
        $('typing-ind').classList.remove('show');
    } else if (type === 'status') {
        if (data.narrative) $('narrative').textContent = data.narrative;
    } else if (type === 'activity') {
        updateTypingLabel(data.label || 'Aura is thinking…');
        if (data.show === false) {
            $('typing-ind').classList.remove('show');
        } else {
            $('typing-ind').classList.add('show');
        }
    } else if (type === 'action_result') {
        const { tool, result, metadata } = data;
        const isAutonomic = metadata && metadata.autonomic;
        const badge = isAutonomic ? '<span class="badge badge-autonomic">[Autonomic]</span> ' : '';
        
        // Phase 36: Check for image display at both levels (result and data)
        const displayType = (result && result.display_type) || data.display_type;
        const imageUrl = (result && result.url) || data.url;
        
        if (displayType === 'image' && imageUrl) {
            const saveBtn = `<button class="gen-save-btn" onclick="saveImageToDevice('${imageUrl}')">MANIFEST TO DESKTOP</button>`;
            const html = `${badge}<div class="gen-image-wrap"><div class="gen-image-loading" id="img-loading-${Date.now()}">Manifesting visualization...</div><img src="${imageUrl}" alt="Generated Image" class="gen-image" onload="this.previousElementSibling.style.display='none';" onerror="this.previousElementSibling.textContent='Image loading... please wait'; var self=this; setTimeout(function(){self.src=self.src.split('&retry')[0]+'&retry='+Date.now()},5000);" onclick="window.open('${imageUrl}', '_blank')">${saveBtn}</div>`;
            appendMsg('aura', html, true);
            $('typing-ind').classList.remove('show');
        } else if (result) {
            // Non-image action results — show the message if available
            const msg = result.message || `Completed ${tool || 'action'}.`;
            appendMsg('aura', badge + msg, !!badge);
            $('typing-ind').classList.remove('show');
        }
    } else if (type === 'aura_message' || type === 'chat_response') {
        const msg = data.message || data.content;
        const meta = data.metadata || {};
        if (msg && msg.trim()) {
            let badge = '';
            if (meta.autonomic) badge = '<span class="badge badge-autonomic">[Autonomic]</span> ';
            if (meta.reflex) badge = '<span class="badge badge-reflex">[Reflex]</span> ';
            if (meta.diagnostic) badge = '<span class="badge badge-diagnostic">⚠️</span> ';
            
            // ZENITH: ID/Fingerprint Deduplication
            const fingerprint = `${data.id || ''}:${msg.trim()}`;
            if (state.processedMessageFingerprints.has(fingerprint)) {
                // duplicate message skipped
                return;
            }
            state.processedMessageFingerprints.add(fingerprint);
            // Limit set size — trim to 250 when exceeding 500
            if (state.processedMessageFingerprints.size > 500) {
                const iter = state.processedMessageFingerprints.values();
                for (let i = 0; i < 250; i++) {
                    state.processedMessageFingerprints.delete(iter.next().value);
                }
            }

            appendMsg('aura', badge + msg, !!badge, meta);
            $('typing-ind').classList.remove('show');
            triggerVoiceOrb('speaking');
        }
    } else if (type === 'skill_status') {
        updateSkillUI(data.skill, data.state);
    } else if (type === 'model_failover') {
        const from = data.from || 'Current Brain';
        const error = data.error || 'stalled';
        appendMsg('aura', `⚠️ _Shift in cognitive processing: ${from} was unresponsive. Switching to a different neural pathway (${error})._`, false, { diagnostic: true });
    } else if (type === 'heartbeat') {
        if (!state.connected) {
            state.connected = true;
            setConnectionVisual('online');
        }
    } else if (type === 'pong') {
        // heartbeat response
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
    state.thoughtQueue.push(data);
    if (!state.pacingActive) processThoughtQueue();
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
        $('chat-form').dispatchEvent(new Event('submit'));
    }
}

async function processThoughtQueue() {
    if (state.thoughtQueue.length === 0) {
        state.pacingActive = false;
        return;
    }
    state.pacingActive = true;
    const data = state.thoughtQueue.shift();
    addThoughtCard(data);

    // Pacing: slow down if many messages, minimum 800ms
    const delay = Math.max(800, 1500 / (state.thoughtQueue.length + 1));
    setTimeout(processThoughtQueue, delay);
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

function addThoughtCard(data) {
    const card = document.createElement('div');
    const level = data.level || '';
    let cls = 'thought-card';
    if (level === 'impulse' || level === 'INFO') cls += ' impulse';
    else if (level === 'ERROR' || level === 'error') cls += ' error';
    else if (level === 'WARNING' || level === 'warning') cls += ' warning';
    card.className = cls;

    const ts = formatEventTimestamp(data.timestamp);
    const name = data.name || 'SYS';
    const msg = data.message || data.content || JSON.stringify(data);
    const safeName = escHtml(name);
    const safeMsg = escHtml(msg).replace(/\n/g, '<br>');
    card.dataset.copyText = `[${ts}] ${name}\n${msg}`;
    card.innerHTML = `
        <div class="thought-card-head">
            <div class="thought-card-meta">
                <span class="thought-ts">${ts}</span>
                <span class="thought-tag">${safeName}</span>
            </div>
            <button class="thought-copy-btn" type="button" onclick="copyThoughtCard(this)">COPY</button>
        </div>
        <div class="thought-body">${safeMsg}</div>
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

    animate() {
        if (!this.ctx) return;

        // THE FIX: Pause drawing if the tab is hidden to save CPU/Battery
        if (document.hidden) {
            requestAnimationFrame(() => this.animate());
            return;
        }

        const { width, height } = this.canvas;
        this.ctx.clearRect(0, 0, width, height);

        // Draw grid
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.moveTo(0, height / 2);
        this.ctx.lineTo(width, height / 2);
        this.ctx.stroke();

        const drawLine = (key, color) => {
            if (this.history.length < 2) return;
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();

            for (let i = 0; i < this.history.length; i++) {
                const x = (i / this.maxLen) * width;
                // Scale VAD (-1 to 1) to canvas height
                const val = this.history[i][key];
                const y = (height / 2) - (val * (height / 2.2));

                if (i === 0) this.ctx.moveTo(x, y);
                else this.ctx.lineTo(x, y);
            }
            this.ctx.stroke();

            // Glow effect
            this.ctx.shadowBlur = 8;
            this.ctx.shadowColor = color;
            this.ctx.stroke();
            this.ctx.shadowBlur = 0;
        };

        drawLine('v', this.colors.v);
        drawLine('a', this.colors.a);
        drawLine('d', this.colors.d);

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
    if (normalized.ram_usage != null && t.ram) t.ram.textContent = Math.round(normalized.ram_usage) + '%';

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

// ── Chat ─────────────────────────────────────────────────
$('chat-form').onsubmit = async e => {
    e.preventDefault();
    const msgInput = $('chat-input');
    const msg = msgInput.value.trim();

    if (!msg) return;

    // Track last user message for regeneration
    state.lastUserMessage = msg;
    const regenBtn = $('regen-btn');
    if (regenBtn) regenBtn.style.display = 'inline-flex';

    state.userScrolledUp = false;  // Reset scroll lock when user sends a message
    appendMsg('user', msg);
    msgInput.value = '';
    // Reset textarea height
    msgInput.style.height = 'auto';
    msgInput.focus();
    $('typing-ind').classList.add('show');
    state.isSubmitting = true;
    // [v7.1] UNLOCK CHAT: Per user preference, do not disable input/button while thinking.
    // Allow user to continue typing or send more messages.

    const controller = new AbortController();
    const requestTimeoutMs = 90000;
    const timeoutId = window.setTimeout(() => controller.abort(), requestTimeoutMs);

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg }),
            signal: controller.signal,
        });
        const data = await res.json();
        if (data && data.conversation_lane) {
            applyConversationLane(data.conversation_lane, res.ok ? 'ok' : 'degraded');
        }

        if (!res.ok) {
            if (data.response) {
                appendMsg('aura', data.response);
            } else {
                appendMsg('aura', '⚠ Communication error. Check connection.');
            }
            return;
        }

        // If it's just a dispatch confirmation, don't clutter the chat
        if (data.response && data.response !== "Message dispatched to cognitive core.") {
            // Deduplicate the final HTTP response if it was already streamed into the DOM
            if (typeof activeStreamContentRaw === 'undefined' || activeStreamContentRaw.trim() !== data.response.trim()) {
                appendMsg('aura', data.response);
            }
        }
    } catch (err) {
        console.error('[CHAT] Error sending message:', err);
        if (err && err.name === 'AbortError') {
            const timedOutLane = Object.assign({}, state.conversationLane || {}, {
                state: 'recovering',
                conversation_ready: false,
                last_failure_reason: 'foreground_http_timeout',
            });
            applyConversationLane(timedOutLane, 'degraded');
            appendMsg('aura', 'My 32B conversation lane timed out before it could answer. Please try again in a moment.');
        } else {
            appendMsg('aura', '⚠ Communication error. Check connection.');
        }
    } finally {
        window.clearTimeout(timeoutId);
        state.isSubmitting = false;
        // Note: Typing indicator is usually cleared when the WS 'aura_message' arrives.
        $('typing-ind').classList.remove('show');
    }
};

async function appendMsg(role, text, isHtml = false, metadata = {}) {
    const messages = DOM.messages || $('messages');
    const div = document.createElement('div');
    div.className = `msg ${role} typing`;
    
    // Add Badge if metadata present
    if (metadata.reflex) {
        div.innerHTML = `<span class="aura-badge reflex">Reflex</span>`;
    } else if (metadata.autonomic) {
        div.innerHTML = `<span class="aura-badge autonomic">Autonomic</span>`;
    }
    
    messages.appendChild(div);

    // THE FIX: Prune old DOM nodes to keep the UI buttery smooth indefinitely
    const MAX_VISIBLE_MESSAGES = 40;
    while (messages.children.length > MAX_VISIBLE_MESSAGES) {
        messages.removeChild(messages.firstChild);
    }

    const isAura = role === 'aura';

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

    // Timestamp element (shows on hover)
    const tsStr = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

    if (isAura && text.length > 5 && !isHtml) {
        const words = text.split(' ');
        let currentWordRaw = '';
        let i = 0;

        let lastTypeTime = 0;
        const wordsPerSec = 15;
        const msPerWord = 1000 / wordsPerSec;

        function typeChunk(timestamp) {
            if (!lastTypeTime) lastTypeTime = timestamp;
            const elapsed = timestamp - lastTypeTime;

            if (elapsed >= msPerWord) {
                currentWordRaw += (i === 0 ? '' : ' ') + words[i];
                i++;
                lastTypeTime = timestamp;

                const badgeHtml = (metadata.reflex ? `<span class="aura-badge reflex">Reflex</span>` : (metadata.autonomic ? `<span class="aura-badge autonomic">Autonomic</span>` : ''));
                div.innerHTML = `<div class="aura-avatar"></div>` + badgeHtml + render(currentWordRaw) + `<span class="msg-timestamp">${tsStr}</span>`;
                if (!state.userScrolledUp) messages.scrollTop = messages.scrollHeight;
            }

            if (i < words.length) {
                requestAnimationFrame(typeChunk);
            } else {
                div.classList.remove('typing');
            }
        }
        requestAnimationFrame(typeChunk);
    } else {
        if (isAura) {
            div.innerHTML = `<div class="aura-avatar"></div>` + (div.innerHTML || '') + render(text) + `<span class="msg-timestamp">${tsStr}</span>`;
        } else {
            div.innerHTML += render(text) + `<span class="msg-timestamp">${tsStr}</span>`;
        }
        div.classList.remove('typing');
        if (!state.userScrolledUp) messages.scrollTop = messages.scrollHeight;
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

    const MAX_VISIBLE_MESSAGES = 40;
    while (messages.children.length > MAX_VISIBLE_MESSAGES) {
        messages.removeChild(messages.firstChild);
    }
}

function appendStreamChunk(chunk) {
    if (!activeStreamDiv) return;
    activeStreamContentRaw += chunk;
    
    // Render streaming content with markdown support
    let h = escHtml(activeStreamContentRaw);
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
        activeStreamDiv.innerHTML = `<div class="aura-avatar"></div>` + h;
    } else {
        activeStreamDiv.innerHTML = h;
    }
    const messages = DOM.messages || $('messages');
    if (!state.userScrolledUp) messages.scrollTop = messages.scrollHeight;
}

function finishStreamMsg() {
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
// Make globally accessible for onclick handlers
window.copyCodeBlock = copyCodeBlock;
window.copyThoughtCard = copyThoughtCard;

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
    } else if (mode === 'reconnected') {
        // Brief green "reconnected" toast
        toast.textContent = '✓ Connection restored';
        toast.classList.remove('show'); // reset
        toast.classList.add('reconnected');
        requestAnimationFrame(() => toast.classList.add('show'));
        _connToastTimer = setTimeout(() => {
            toast.classList.remove('show', 'reconnected');
            toast.textContent = '⚠ Connection lost — reconnecting…';
        }, 2500);
    } else {
        // Show disconnect toast
        toast.textContent = '⚠ Connection lost — reconnecting…';
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
    toast.classList.remove('reconnected');
    toast.classList.add('show');
    _briefToastTimer = setTimeout(() => {
        toast.classList.remove('show');
        toast.textContent = '';
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

function laneHealthIsOperational(lane, healthStatus = '') {
    const normalized = String(healthStatus || '').toLowerCase();
    return normalized === 'ok' || normalized === 'ready' || (laneIsStandby(lane) && normalized === 'warming');
}

function conversationLaneStatusText(lane) {
    if (!lane) return 'online';
    const laneState = String(lane.state || 'warming').toLowerCase();
    if (lane.conversation_ready) return 'online';
    if (laneIsStandby(lane)) return 'cortex on standby';
    if (laneState === 'recovering') return 'cortex recovering';
    if (laneState === 'failed') return 'cortex unavailable';
    return 'cortex warming';
}

function applyConversationLane(lane, healthStatus = '') {
    if (!lane || typeof lane !== 'object') return;

    state.conversationLane = lane;
    state.conversationReady = !!lane.conversation_ready;

    const laneText = conversationLaneStatusText(lane);
    const laneStandby = laneIsStandby(lane);
    if (state.connected) {
        const healthy = laneHealthIsOperational(lane, healthStatus);
        const connectionMode = (state.conversationReady || laneStandby) && healthy ? 'online' : 'degraded';
        setConnectionVisual(connectionMode, !state.conversationReady ? laneText : '');
    }

    updateTypingLabel(
        state.conversationReady
            ? 'Aura is thinking…'
            : laneStandby
                ? 'Aura is ready. Cortex will warm on first turn.'
                : `Aura is ${laneText}...`
    );

    const tierEl = $('r-llm-tier');
    if (tierEl) {
        if (state.conversationReady) {
            const endpoint = lane.foreground_endpoint || lane.desired_endpoint || 'Cortex';
            tierEl.textContent = endpoint;
            tierEl.title = `Foreground: ${endpoint}`;
            tierEl.style.color = 'var(--success)';
        } else {
            const stateLabel =
                laneText === 'cortex on standby' ? 'CORTEX ON STANDBY' :
                laneText === 'cortex recovering' ? 'CORTEX RECOVERING' :
                laneText === 'cortex unavailable' ? 'CORTEX UNAVAILABLE' :
                'CORTEX WARMING';
            tierEl.textContent = stateLabel;
            tierEl.title = laneStandby
                ? 'Aura is awake. Cortex will warm on first turn.'
                : lane.last_failure_reason || (lane.desired_model || 'Cortex (32B)');
            tierEl.style.color = laneStandby ? 'var(--success)' : lane.state === 'failed' ? 'var(--error)' : 'var(--warn)';
        }
    }
}

// ── Health polling ───────────────────────────────────────
async function pollHealth() {
    if (state.healthPollInFlight) return;
    state.healthPollInFlight = true;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1500);

    try {
        const res = await fetch('/api/health', {
            cache: 'no-store',
            signal: controller.signal
        });
        if (!res.ok) {
            console.warn(`⚠️ Health poll returned status: ${res.status}`);
            return; // Retain last known state
        }
        const d = await res.json();
        if (!d) return;
        const fmtPct01 = (value) => `${Math.round((value || 0) * 100)}%`;
        const runtimeAffect = (d.boot && d.boot.runtime && d.boot.runtime.affect)
            || (d.runtime && d.runtime.state && d.runtime.state.affect)
            || {};
        const liquidTelemetry = d.liquid_state || {};
        updateTelemetry({
            energy: liquidTelemetry.energy ?? runtimeAffect.energy,
            curiosity: liquidTelemetry.curiosity ?? runtimeAffect.curiosity,
            frustration: liquidTelemetry.frustration ?? runtimeAffect.frustration,
            confidence: liquidTelemetry.confidence ?? ((d.homeostasis && d.homeostasis.will_to_live != null) ? (d.homeostasis.will_to_live * 100) : runtimeAffect.stability),
            cpu_usage: d.cpu_usage,
            ram_usage: d.ram_usage,
            p_core_usage: d.cortex ? d.cortex.p_core_usage : null,
            vad: liquidTelemetry.vad || null,
        });
        if (d.conversation_lane) {
            applyConversationLane(d.conversation_lane, d.status || '');
        }
        if (d.boot || d.conversation_lane) {
            syncSplashState({
                telemetry: { boot: d.boot || {} },
                conversation: { lane: d.conversation_lane || null },
                session: { connected: state.connected || d.status === 'ok' || d.status === 'ready' }
            });
        }

        state.cycleCount = d.cycle_count || state.cycleCount || 0;
        const cyclesEl = $('hud-cycles');
        if (cyclesEl) cyclesEl.textContent = state.cycleCount.toLocaleString();

        if (d.uptime != null) {
            const uptimeEl = $('hud-uptime');
            if (uptimeEl) uptimeEl.textContent = fmtUptime(d.uptime);
        }
        
        if (d.version) {
            const verEl = $('ui-ver');
            if (verEl) verEl.textContent = d.version;
        }

        const cpuEl = $('hud-cpu');
        if (cpuEl) cpuEl.textContent = Math.round(d.cpu_usage || 0) + '%';
        
        const ramEl = $('hud-ram');
        if (ramEl) ramEl.textContent = Math.round(d.ram_usage || 0) + '%';
        
        const pcoreEl = $('hud-pcore');
        const pcoreVal = d.cortex ? d.cortex.p_core_usage : 0;
        if (pcoreEl) pcoreEl.textContent = Math.round(pcoreVal || 0) + '%';

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
            updateGauge('s-thermal', (s.thermal_load || 0) * 100, 's-thermal-val');
            updateGauge('s-anxiety', (s.resource_anxiety || 0) * 100, 's-anxiety-val');
            updateGauge('s-vitality', (s.vitality || 0) * 100, 's-vitality-val');
        }

        if (d.homeostasis) {
            updateGauge('g-integrity', (d.homeostasis.integrity || 0) * 100, 'g-integrity-val');
            updateGauge('g-persistence', (d.homeostasis.persistence || 0) * 100, 'g-persistence-val');
            updateGauge('g-confidence', (d.homeostasis.will_to_live || 0) * 100, 'g-confidence-val');
        }

        if (d.moral) {
            updateGauge('s-moral', (d.moral.integrity || 0) * 100, 's-moral-val');
        }

        if (d.social) {
            updateGauge('s-social', (d.social.depth || 0) * 100, 's-social-val');
        }

        if (d.swarm) {
            const swarmEl = $('c-swarm');
            if (swarmEl) swarmEl.textContent = d.swarm.active_count || 0;
        }

        // ── Phase III: Qualia Engine ──
        if (d.qualia) {
            const q = d.qualia;
            const priEl = $('q-pri');
            const normEl = $('q-norm');
            const dimEl = $('q-dim');
            const attEl = $('q-attractor');
            if (priEl) priEl.textContent = (q.pri || 0).toFixed(3);
            if (normEl) normEl.textContent = (q.q_norm || 0).toFixed(3);
            if (dimEl) dimEl.textContent = (q.dominant_dim || '--').toUpperCase();
            if (attEl) {
                attEl.textContent = q.in_attractor ? 'LOCKED' : 'FLUID';
                attEl.style.color = q.in_attractor ? 'var(--success)' : 'var(--accent)';
            }
            if ($('q-identity')) {
                $('q-identity').textContent = (q.identity_coherence || 100).toFixed(1) + '%';
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
            if ($('pn-temp'))     $('pn-temp').textContent     = (p.temperature   || 0.7).toFixed(3);
            if ($('pn-arousal'))  $('pn-arousal').textContent  = (p.arousal        || 0).toFixed(3);
            if ($('pn-stability'))$('pn-stability').textContent= (p.stability      || 0).toFixed(3);
            if ($('pn-attractors'))$('pn-attractors').textContent = p.attractor_count || 0;
        }

        // ── MHAF Field ──
        if (d.mhaf) {
            const mh = d.mhaf;
            const mhOnline = $('mhaf-online');
            if (mhOnline) {
                mhOnline.textContent  = mh.online ? 'ONLINE' : 'OFFLINE';
                mhOnline.style.color  = mh.online  ? 'var(--success)' : 'var(--error)';
            }
            if ($('mhaf-phi'))     $('mhaf-phi').textContent     = (mh.phi    || 0).toFixed(4);
            if ($('mhaf-nodes'))   $('mhaf-nodes').textContent   = mh.nodes   || 0;
            if ($('mhaf-edges'))   $('mhaf-edges').textContent   = mh.edges   || 0;
            if ($('mhaf-lexicon')) $('mhaf-lexicon').textContent = mh.lexicon_size || 0;
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
            const ramEl = $('hud-ram');
            if (ramEl) ramEl.textContent = d.ram_usage + '%';
        } else if (d.runtime && d.runtime.memory_percent != null) {
            const ramEl = $('hud-ram');
            if (ramEl) ramEl.textContent = d.runtime.memory_percent + '%';
        }

        if (d.privacy && (!state._privacyLockUntil || Date.now() > state._privacyLockUntil)) {
            const p = d.privacy;
            const muteBtn = $('btn-mute');
            const camBtn = $('btn-cam');
            if (muteBtn) {
                if (p.microphone_enabled !== false) {
                    muteBtn.classList.remove('disabled');
                    muteBtn.innerHTML = '<span>● MUTE</span>';
                } else {
                    muteBtn.classList.add('disabled');
                    muteBtn.innerHTML = '<span>● MUTED</span>';
                }
                muteBtn.onclick = () => togglePrivacy('microphone', p.microphone_enabled, muteBtn);
            }
            if (camBtn) {
                if (p.camera_enabled !== false) {
                    camBtn.classList.remove('disabled');
                    camBtn.innerHTML = '<span>● CAM</span>';
                } else {
                    camBtn.classList.add('disabled');
                    camBtn.innerHTML = '<span>● CAM OFF</span>';
                }
                camBtn.onclick = () => togglePrivacy('camera', p.camera_enabled, camBtn);
            }
        }

        refreshMetricGuide();
    } catch (e) {
        if (!e || e.name !== 'AbortError') {
            console.warn('⚠️ Health poll failed:', e);
        }
    } finally {
        clearTimeout(timeoutId);
        state.healthPollInFlight = false;
    }
}

async function togglePrivacy(type, currentEnabled, btn) {
    try {
        const next = !currentEnabled;
        // Optimistic UI: update button immediately
        state._privacyLockUntil = Date.now() + 3000; // Lock pollHealth from resetting for 3s
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
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: next })
        });
        const d = await res.json();
        if (d.ok) {
            // Privacy toggle applied
            // Update the onclick to reflect new state
            if (btn) btn.onclick = () => togglePrivacy(type, next, btn);
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

function updateGauge(id, val, textId) {
    const bar = $(id);
    const text = $(textId);
    if (bar) bar.style.width = Math.min(100, Math.max(0, val)) + '%';
    if (text) text.textContent = val.toFixed(0) + '%';
}

function fmtUptime(sec) {
    if (sec < 60) return Math.round(sec) + 's';
    if (sec < 3600) return Math.floor(sec / 60) + 'm' + Math.round(sec % 60) + 's';
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return h + 'h' + m + 'm';
}

// ── Skills ───────────────────────────────────────────────
async function loadSkills() {
    try {
        if (state.toolCatalog && state.toolCatalog.length) {
            renderToolCatalog(state.toolCatalog);
        }

        let tools = [];
        try {
            const res = await fetch('/api/tools/catalog', { cache: 'no-store' });
            const contentType = res.headers.get('content-type') || '';
            if (res.ok && contentType.includes('application/json')) {
                const d = await res.json();
                tools = Array.isArray(d.tools) ? d.tools : [];
            }
        } catch (_err) {
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
        }
        renderToolCatalog(tools);
    } catch (e) {
        const list = $('skills-list');
        if (list && !(state.toolCatalog && state.toolCatalog.length)) {
            list.innerHTML = '<div class="mem-empty">Failed to load tools<br><button class="skills-retry-btn" onclick="loadSkills()">RETRY</button></div>';
        }
    }
}

// ── Memory ───────────────────────────────────────────────
async function loadMemory(type) {
    try {
        const endpoint = `/api/memory/${type || 'episodic'}?limit=20`;
        const res = await fetch(endpoint);
        if (!res.ok) throw new Error(`Memory fetch failed (${res.status})`);
        const d = await res.json();
        const cont = $('mem-content');
        const items = d.items || [];
        if (items.length === 0) {
            const icons = { episodic: '🗂', semantic: '🧠', goals: '🎯' };
            cont.innerHTML = `<div class="mem-empty">${icons[type] || '📁'} No ${type} memories yet</div>`;
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
                    const key = item.key || item.subject || '';
                    const val = item.value || item.predicate || '';
                    return `<div class="mem-item"><strong>${escHtml(key)}</strong>: ${escHtml(String(val))}</div>`;
                } else if (type === 'goals') {
                    const desc = item.description || item.goal || String(item);
                    const status = item.status || 'active';
                    const statusCls = status === 'completed' ? 'success' : (status === 'failed' ? 'error' : '');
                    return `<div class="mem-item"><span class="tag ${statusCls}">${escHtml(status.toUpperCase())}</span> ${escHtml(desc)}</div>`;
                }
            }
            return `<div class="mem-item">${escHtml(String(item))}</div>`;
        }).join('');
    } catch (e) {
        const cont = $('mem-content');
        if (cont) cont.innerHTML = '<div class="mem-empty">Failed to load memories<br><button class="skills-retry-btn" onclick="loadMemory(state.activeMem)">RETRY</button></div>';
        showBriefNotification('Memory load failed — check connection');
    }
}

// ── Belief Graph ─────────────────────────────────────────
let graphNetwork = null;
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
}

async function refreshKnowledgeGraph() {
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
    }
    if (state.activeTab === 'telemetry') setTimeout(refreshKnowledgeGraph, 10000);
}

// ── Header buttons ───────────────────────────────────────
// Immediate MUTE/CAM bindings (before first pollHealth delivers privacy state)
$('btn-mute').onclick = () => togglePrivacy('microphone', true, $('btn-mute'));
$('btn-cam').onclick = () => togglePrivacy('camera', true, $('btn-cam'));

$('btn-brain').onclick = async () => {
    const btn = $('btn-brain');
    btn.style.opacity = '0.5';
    btn.textContent = '◇ ...';
    try {
        const res = await fetch('/api/brain/retry', { method: 'POST' });
        const d = await res.json();
        appendMsg('aura', d.status === 'retry_sent' ? '🧠 Brain retry signal sent.' : '⚠ Orchestrator unavailable.');
    } catch (e) {
        appendMsg('aura', '⚠ Failed to contact brain retry endpoint.');
    } finally {
        btn.style.opacity = '1';
        btn.textContent = '◇ BRAIN';
    }
};

$('btn-apk').onclick = () => {
    appendMsg('aura', '📱 APK not available yet — Aura runs as a web app at this URL.');
};

$('btn-src').onclick = async () => {
    const btn = $('btn-src');
    btn.style.opacity = '0.5';
    btn.textContent = '↓ ...';
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
        btn.style.opacity = '1';
        btn.textContent = '↓ SRC';
    }
};

$('btn-soul').onclick = () => {
    const overlay = $('soul-overlay');
    const frame = $('soul-frame');
    if (overlay && frame) {
        overlay.classList.add('visible');
        frame.src = '/static/mycelial.html';
        // Soul Map opened
    }
};

$('soul-close').onclick = () => {
    const overlay = $('soul-overlay');
    const frame = $('soul-frame');
    overlay.classList.remove('visible');
    frame.src = '';  // Stop the 3D renderer to save GPU
};

$('btn-mem-map').onclick = () => {
    const overlay = $('soul-overlay');
    const frame = $('soul-frame');
    if (overlay && frame) {
        overlay.classList.add('visible');
        frame.src = '/memory';
        // Memory Map opened
    }
};

$('btn-term').onclick = async () => {
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
};

$('term-close-btn') && ($('term-close-btn').onclick = () => {
    const modal = $('terminal-modal');
    if (modal) modal.style.display = 'none';
});

$('term-send-btn') && ($('term-send-btn').onclick = async () => {
    const input = $('term-msg-input');
    const result = $('term-send-result');
    if (!input || !input.value.trim()) return;
    try {
        const r = await fetch('/api/terminal/send', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: input.value.trim()})
        });
        const d = await r.json();
        if (result) {
            result.textContent = d.ok ? `✓ Queued: "${d.queued}"` : `✗ ${d.error}`;
            result.style.color = d.ok ? 'var(--success)' : 'var(--error)';
        }
        if (d.ok) { input.value = ''; if ($('term-pending')) $('term-pending').textContent = parseInt($('term-pending').textContent || '0') + 1; }
    } catch (e) {
        if (result) { result.textContent = '✗ Request failed'; result.style.color = 'var(--error)'; }
    }
});

$('btn-reboot').onclick = async () => {
    if (confirm('Reboot Aura? This will restart the server process.')) {
        try {
            await fetch('/api/reboot', { method: 'POST' });
        } catch (e) { }
    }
};

// ── Voice toggle ─────────────────────────────────────────
let audioContext = null;

async function toggleVoice() {
    if (state.voiceSummary && state.voiceSummary.available === false) {
        appendMsg('aura', '⚠ Voice channel is currently unavailable.');
        return;
    }
    const orb = $('voice-orb');
    state.voiceActive = !state.voiceActive;
    $('voice-orb-wrap').classList.toggle('active', state.voiceActive);
    $('mic-btn').textContent = state.voiceActive ? '⏹️' : '🔮';

    if (state.voiceActive) {
        orb.className = 'voice-orb listening';
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

            // Modern AudioWorklet approach
            await audioContext.audioWorklet.addModule('/static/voice-processor.js');
            const source = audioContext.createMediaStreamSource(stream);
            const voiceNode = new AudioWorkletNode(audioContext, 'voice-capture-processor');

            voiceNode.port.onmessage = (e) => {
                if (!state.voiceActive) return;
                if (e.data.type === 'pcm' && state.ws && state.ws.readyState === WebSocket.OPEN) {
                    state.ws.send(e.data.data);
                }
            };

            source.connect(voiceNode);
            voiceNode.connect(audioContext.destination);
            state.audioStream = stream;
            state.voiceNode = voiceNode;
        } catch (err) {
            console.error('Voice capture failed:', err);
            appendMsg('aura', '⚠ I couldn\'t access your microphone.');
            state.voiceActive = false;
            $('voice-orb-wrap').classList.remove('active');
            orb.className = 'voice-orb';
            $('mic-btn').textContent = '🎙';
        }
    } else {
        orb.className = 'voice-orb';
        if (state.audioStream) {
            state.audioStream.getTracks().forEach(t => t.stop());
            state.audioStream = null;
        }
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
    }
}
$('mic-btn').onclick = toggleVoice;

// Heartbeat is handled by the 25s pingInterval in connect()

// ── Service Worker (PWA Support) ─────────────────────────
if ('serviceWorker' in navigator) {
    let swReloadTriggered = false;
    navigator.serviceWorker.addEventListener('controllerchange', () => {
        if (swReloadTriggered) return;
        swReloadTriggered = true;
        window.location.reload();
    });

    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/service-worker.js', { updateViaCache: 'none' })
            .then(async (reg) => {
                // Service Worker registered
                try {
                    await reg.update();
                } catch (err) {
                    console.warn('[SW] update() failed:', err);
                }

                if (reg.waiting) {
                    reg.waiting.postMessage({ type: 'SKIP_WAITING' });
                }

                reg.addEventListener('updatefound', () => {
                    const installing = reg.installing;
                    if (!installing) return;
                    installing.addEventListener('statechange', () => {
                        if (installing.state === 'installed' && navigator.serviceWorker.controller) {
                            installing.postMessage({ type: 'SKIP_WAITING' });
                        }
                    });
                });
            })
            .catch(err => console.error('Service Worker failure:', err));
    });
}

// ── Start ────────────────────────────────────────────────
setConnectionVisual('booting');
hydrateBootstrap({ hydrateConversationHistory: true, quiet: true });
initializeMetricGuide();
connect();
pollHealth();
voicePlayer.init();
vadStream = new VADStream('neural-vad-canvas');
setInterval(pollHealth, 10000); // 10s — enterprise-grade, not chatbot-grade
state.bootstrapTimer = setInterval(() => hydrateBootstrap({ quiet: true }), 30000);
loadSkills();

// ── Settings & Preferences ────────────────────────────────
const SETTINGS_KEY = 'aura_settings';
const defaultSettings = {
    theme: 'dark', accent: 'violet', voice: true, autolisten: false,
    ttsSpeed: 1.0, enrichment: true, reflection: true, autonomy: true,
    approval: 'destructive', onboarded: false, cheatStatus: 'IDLE'
};

function loadSettings() {
    try {
        const saved = localStorage.getItem(SETTINGS_KEY);
        return saved ? { ...defaultSettings, ...JSON.parse(saved) } : { ...defaultSettings };
    } catch { return { ...defaultSettings }; }
}

function saveSettings(s) {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); } catch {}
}

function applySettings(s) {
    // Theme
    document.body.className = document.body.className.replace(/theme-\w+/g, '').replace(/accent-\w+/g, '').trim();
    if (s.theme !== 'dark') document.body.classList.add(`theme-${s.theme}`);
    if (s.accent !== 'violet') document.body.classList.add(`accent-${s.accent}`);

    // Sync UI controls
    const el = (id) => document.getElementById(id);
    if (el('setting-theme')) el('setting-theme').value = s.theme;
    if (el('setting-accent')) el('setting-accent').value = s.accent;
    if (el('setting-voice')) el('setting-voice').checked = s.voice;
    if (el('setting-autolisten')) el('setting-autolisten').checked = s.autolisten;
    if (el('setting-tts-speed')) el('setting-tts-speed').value = s.ttsSpeed;
    if (el('setting-enrichment')) el('setting-enrichment').checked = s.enrichment;
    if (el('setting-reflection')) el('setting-reflection').checked = s.reflection;
    if (el('setting-autonomy')) el('setting-autonomy').checked = s.autonomy;
    if (el('setting-approval')) el('setting-approval').value = s.approval;
    if (el('setting-cheat-status')) el('setting-cheat-status').textContent = s.cheatStatus || 'IDLE';
    if (el('setting-version')) el('setting-version').textContent = state.version;
}

const settings = loadSettings();
applySettings(settings);

// Bind settings controls
['setting-theme', 'setting-accent', 'setting-voice', 'setting-autolisten',
 'setting-tts-speed', 'setting-enrichment', 'setting-reflection',
 'setting-autonomy', 'setting-approval'].forEach(id => {
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

async function activateCheatCode() {
    const input = document.getElementById('setting-cheat-code');
    const code = input ? input.value.trim() : '';
    if (!code) return;

    const statusEl = document.getElementById('setting-cheat-status');
    if (statusEl) statusEl.textContent = 'CHECKING…';

    try {
        const resp = await fetch('/api/cheat-codes/activate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
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
            appendMsg('aura', data.message);
        } else if (!resp.ok) {
            appendMsg('aura', 'Unknown cheat code.');
        }

        if (typeof pollHealth === 'function') {
            await pollHealth();
        }
    } catch (err) {
        settings.cheatStatus = 'ERROR';
        saveSettings(settings);
        applySettings(settings);
        appendMsg('aura', '⚠ Cheat code activation failed.');
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
const clearBtn = document.getElementById('btn-clear-history');
if (clearBtn) clearBtn.addEventListener('click', () => {
    if (confirm('Clear all conversation history? Memories are preserved.')) {
        const msgEl = document.getElementById('messages');
        if (msgEl) msgEl.innerHTML = '<div class="sys-box">History cleared. Aura remembers you.</div>';
    }
});

// ── Onboarding ────────────────────────────────────────────
if (!settings.onboarded) {
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

    const boot = payload && payload.telemetry && payload.telemetry.boot ? payload.telemetry.boot : {};
    const lane = payload && payload.conversation ? payload.conversation.lane : null;
    const sessionConnected = !!(payload && payload.session && payload.session.connected);
    const standby = laneIsStandby(lane);
    const bootReady = boot.ready === true || String(boot.status || '').toLowerCase() === 'ready';
    const message = String(boot.status_message || '').trim();

    if (state._splashInterval) {
        clearInterval(state._splashInterval);
        state._splashInterval = null;
    }

    updateSplashProgress(
        boot.progress != null ? boot.progress : (sessionConnected || standby ? 100 : 15),
        message || (standby ? 'Aura is awake. Cortex will warm on first turn.' : '')
    );

    if (sessionConnected || bootReady || standby) {
        dismissSplash(message || (standby ? 'Aura is awake. Cortex will warm on first turn.' : 'Neural link established.'));
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
})();

function dismissSplash(finalStatus = 'Neural link established.') {
    const splash = $('splash-screen');
    const splashBar = $('splash-bar');
    if (!splash || splash.classList.contains('hidden')) return;

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

    // Fade out after brief delay to show 100%
    setTimeout(() => {
        splash.classList.add('hidden');
        // Remove from DOM after transition
        setTimeout(() => splash.remove(), 1000);
    }, 400);
}

// ══════════════════════════════════════════════════════════
//  MAGNUM OPUS — Textarea Auto-Resize & Keyboard Shortcuts
// ══════════════════════════════════════════════════════════
(function initTextareaAndShortcuts() {
    const textarea = $('chat-input');
    if (!textarea) return;

    // Auto-resize textarea as user types
    textarea.addEventListener('input', () => {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    });

    // Keyboard handling for textarea
    textarea.addEventListener('keydown', (e) => {
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
            textarea.style.height = 'auto';
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
