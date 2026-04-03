/**
 * Memory Engine for Aura's Black Hole Memory.
 */
export const lz77Compress = (text) => {
    // Pure JS simplified LZ77 stub for UI representation
    if (!text) return "";
    return btoa(text).substring(0, text.length * 0.8); // Visual shorthand
};

export const cosineSim = (v1, v2) => {
    let dot = 0, n1 = 0, n2 = 0;
    for (let i = 0; i < v1.length; i++) {
        dot += v1[i] * v2[i];
        n1 += v1[i] * v1[i];
        n2 += v2[i] * v2[i];
    }
    return dot / (Math.sqrt(n1) * Math.sqrt(n2));
};

export const seededRand = (seed) => {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
};
