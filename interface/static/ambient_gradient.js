/* A small colour field, sampled twice a second instead of repainting blurred
 * full-window layers at display refresh rate. */
(() => {
    'use strict';
    const panel = document.querySelector('.chat-panel');
    if (!panel) return;
    const canvas = document.createElement('canvas');
    canvas.className = 'chat-colour-field';
    canvas.width = 160;
    canvas.height = 120;
    canvas.setAttribute('aria-hidden', 'true');
    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) return;
    panel.prepend(canvas);
    panel.classList.add('has-colour-field');
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    let timer = null;
    let phase = 0;
    let previous = performance.now();
    let classes = document.body.className;

    function paint(now) {
        const started = performance.now();
        if (!reduced.matches) phase += Math.min(2, (now - previous) / 1000) / 26;
        previous = now;
        const light = document.body.classList.contains('theme-light');
        const working = panel.classList.contains('thinking') || panel.classList.contains('generating');
        const alpha = light ? .14 : working ? .24 : .17;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const colours = [[153, 86, 240], [36, 168, 199], [201, 88, 163]];
        for (let i = 0; i < colours.length; i++) {
            const angle = phase + i * 2.094;
            const x = 80 + Math.cos(angle) * 60;
            const y = 60 + Math.sin(angle * .83 + i) * 55;
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, 100);
            const rgb = colours[i].join(',');
            gradient.addColorStop(0, `rgba(${rgb},${alpha})`);
            gradient.addColorStop(1, `rgba(${rgb},0)`);
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, 160, 120);
        }
        canvas.dataset.paintMs = (performance.now() - started).toFixed(2);
        canvas.dataset.phase = phase.toFixed(4);
    }

    function schedule() {
        clearTimeout(timer);
        timer = null;
        if (document.hidden || reduced.matches || document.body.classList.contains('vm-active')) return;
        const lean = document.body.classList.contains('perf-lean')
            || document.body.classList.contains('aura-throttle-motion');
        timer = setTimeout(() => { paint(performance.now()); schedule(); }, lean ? 2000 : 500);
    }
    document.addEventListener('visibilitychange', () => {
        previous = performance.now();
        schedule();
    });
    reduced.addEventListener('change', () => { paint(performance.now()); schedule(); });
    new MutationObserver(() => {
        if (classes === document.body.className) return;
        const themeChanged = classes.includes('theme-light') !== document.body.classList.contains('theme-light');
        classes = document.body.className;
        if (themeChanged && !document.hidden) paint(performance.now());
        schedule();
    }).observe(document.body, { attributes: true, attributeFilter: ['class'] });
    paint(previous);
    schedule();
})();
