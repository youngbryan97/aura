/**
 * Physics Engine for Aura's Black Hole Memory.
 */
export const bekensteinBound = (radius_cm, energy) => {
    // S <= 2 * pi * E * R / (hbar * c)
    // Simplified for UI visualization
    const limit = (radius_cm * energy) / 1024;
    return limit.toFixed(2);
};

export const hawkingDecay = (mass, age_seconds) => {
    // Evaporation simulation
    const decay = mass / Math.pow(age_seconds + 1, 0.5);
    return decay.toFixed(3);
};

export const gravitationalSort = (items) => {
    return [...items].sort((a, b) => b.gravity - a.gravity);
};
