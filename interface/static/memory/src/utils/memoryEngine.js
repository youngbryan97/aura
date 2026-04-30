/**
 * Memory Engine for Aura's Black Hole Memory.
 */
export const lz77Compress = (text) => {
    if (!text) return "";
    const input = String(text);
    const windowSize = 128;
    const lookaheadSize = 32;
    const tokens = [];
    let i = 0;

    while (i < input.length) {
        const windowStart = Math.max(0, i - windowSize);
        const window = input.slice(windowStart, i);
        let bestOffset = 0;
        let bestLength = 0;

        for (let offset = 1; offset <= window.length; offset += 1) {
            let length = 0;
            while (
                length < lookaheadSize &&
                i + length < input.length &&
                input[i + length] === input[i - offset + length]
            ) {
                length += 1;
            }
            if (length > bestLength) {
                bestLength = length;
                bestOffset = offset;
            }
        }

        if (bestLength >= 3) {
            const nextChar = input[i + bestLength] || "";
            tokens.push(`${bestOffset}:${bestLength}:${nextChar}`);
            i += bestLength + (nextChar ? 1 : 0);
        } else {
            tokens.push(`0:0:${input[i]}`);
            i += 1;
        }
    }

    return tokens.join("|");
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
