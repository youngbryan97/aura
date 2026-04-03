// --- Phase 3: The Security Bridge ---

const fetchJson = async (url, options = {}, timeoutMs = 4000) => {
    const controller = new AbortController();
    const timerId = globalThis.setTimeout(() => controller.abort(), timeoutMs);

    try {
        const res = await fetch(url, {
            ...options,
            signal: controller.signal,
        });
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        return await res.json();
    } catch (e) {
        if (e && e.name === 'AbortError') {
            throw new Error(`Request timed out after ${timeoutMs}ms`);
        }
        throw e;
    } finally {
        globalThis.clearTimeout(timerId);
    }
};

export const callQuantumLLM = async (prompt) => {
    try {
        return await fetchJson('/api/think', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt })
        }, 15000);
    } catch (e) {
        console.error("Neural link severed:", e);
        return { error: "Link lost", details: e.message };
    }
};

export const fetchMemoryStats = async () => {
    try {
        return await fetchJson('/memory/api/memory');
    } catch (e) {
        console.error("Dashboard link severed:", e);
        throw e;
    }
};

// Standardized Persistence Helpers
export const getLocalState = (key, fallback) => {
    try {
        const item = localStorage.getItem(`aura_${key}`);
        return item ? JSON.parse(item) : fallback;
    } catch (e) {
        console.error("Singularity access denied:", e);
        return fallback;
    }
};

export const setLocalState = (key, value) => {
    try {
        localStorage.setItem(`aura_${key}`, JSON.stringify(value));
    } catch (e) {
        console.error("Singularity collapse during write:", e);
    }
};
