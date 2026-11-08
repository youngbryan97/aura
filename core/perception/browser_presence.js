/* Installed only in Aura's owned browser context. No page event is intercepted. */
(() => {
    'use strict';
    if (window.top !== window || window.__auraBrowserPresence) return;
    let host, root, timer, lastBox, suspended = 0, active = false;
    const SVG = 'http://www.w3.org/2000/svg';
    const STYLE = `
        :host { pointer-events:none!important; }
        * { box-sizing:border-box; pointer-events:none!important; }
        [hidden] { display:none!important; }
        .frame { position:fixed;inset:3px;border:1.5px solid #ad99ffbb;border-radius:13px;box-shadow:inset 0 0 0 1px #17121f77,inset 0 0 18px #ad99ff0d; }
        .label { position:fixed;top:15px;right:18px;display:flex;align-items:center;gap:9px;padding:8px 13px 8px 9px;border:1px solid #ae9aff40;border-radius:12px;background:#17131ff5;box-shadow:0 3px 12px #0003,inset 0 1px 0 #ffffff08;color:#ede7f6;font:12px/1.4 system-ui,sans-serif; }
        .identity { display:flex;align-items:center;gap:7px;font-weight:600; }
        .identity svg { width:21px;height:21px; }
        .label .activity { border-left:1px solid #ffffff1c;padding-left:9px;color:#d7cfe5; }
        .target { position:fixed;border:1.5px solid #b5a0ff;border-radius:8px;background:#b29aff0c;box-shadow:0 0 0 3px #b29aff0b; }
        .cursor { position:fixed;width:34px;height:34px;filter:drop-shadow(0 2px 2px #0008); }
        .cursor svg { width:34px;height:34px;display:block; }
        .point { position:fixed;width:6px;height:6px;border:1px solid #fff;border-radius:50%;background:#ad99ff;transform:translate(-50%,-50%); }
        .point.clicked { animation:click .65s ease-out; }
        @keyframes click { to { box-shadow:0 0 0 18px #b49aff00;outline:1px solid #b49aff00;transform:translate(-50%,-50%) scale(2.5); } }
        @media(prefers-reduced-motion:reduce) { .point.clicked { animation:none; } }`;
    // Built node by node. A page that requires Trusted Types refuses every
    // assignment of markup, and YouTube is one: there, on 7 and 8 November
    // 2026, every mount of the mark failed and it was never drawn.
    function part(name, parent) {
        const node = document.createElement('div');
        node.className = name;
        parent.append(node);
        return node;
    }
    // The two marks share one shadow tree, so each takes its own ids.
    function neuron([tag, attributes, children], ids) {
        const node = document.createElementNS(SVG, tag);
        for (const [name, value] of Object.entries(attributes)) {
            node.setAttribute(name, name === 'id' ? ids + value : value.replace(/url\(#([^)]+)\)/g, `url(#${ids}$1)`));
        }
        for (const child of children) node.append(typeof child === 'string' ? child : neuron(child, ids));
        return node;
    }
    function dress(shadow) {
        try {
            const sheet = new CSSStyleSheet();
            sheet.replaceSync(STYLE);
            shadow.adoptedStyleSheets = [sheet];
        } catch (_) {
            const style = document.createElement('style');
            style.textContent = STYLE;
            shadow.append(style);
        }
    }
    function mount() {
        if (host?.isConnected) return;
        host = document.createElement('div');
        host.dataset.auraPresence = 'true';
        host.setAttribute('aria-hidden', 'true');
        host.style.cssText = 'all:initial!important;position:fixed!important;inset:0!important;pointer-events:none!important;z-index:2147483647!important;';
        // A closed shadow tree keeps the mark out of page text and control enumeration.
        root = host.attachShadow({ mode: 'closed' });
        dress(root);
        part('frame', root);
        const label = part('label', root);
        const identity = document.createElement('span');
        identity.className = 'identity';
        identity.append(neuron(__AURA_NEURON__, 'label-'), document.createTextNode('Aura'));
        const activity = document.createElement('span');
        activity.className = 'activity';
        label.append(identity, activity);
        for (const name of ['target', 'cursor', 'point']) part(name, root).hidden = true;
        root.querySelector('.cursor').append(neuron(__AURA_NEURON__, ''));
        (document.documentElement || document).append(host);
    }
    function visibility() { if (host) host.style.setProperty('display', active && !suspended ? 'block' : 'none', 'important'); }
    function show(data) {
        mount();
        clearTimeout(timer);
        active = data.action !== 'closed';
        const labels = { navigating:'Opening this page', reading:'Reading this page', clicking:'Clicking here', typing:'Typing here', scrolling:'Scrolling this page' };
        const completed = { navigating:'Page opened', reading:'Page read', clicking:'Clicked here', typing:'Finished typing', scrolling:'Scrolled' };
        root.querySelector('.activity').textContent = `${(data.stage === 'completed' ? completed[data.action] : labels[data.action]) || 'Working here'}${data.stage === 'failed' ? ' · stopped' : ''}`;
        const box = data.box || (data.action === 'typing' ? lastBox : null);
        lastBox = box;
        const target = root.querySelector('.target');
        const cursor = root.querySelector('.cursor');
        const point = root.querySelector('.point');
        const hasPoint = box && [box.x, box.y, box.width, box.height].every(Number.isFinite);
        target.hidden = cursor.hidden = point.hidden = !hasPoint;
        if (hasPoint) {
            const x = Number.isFinite(box.clickX) ? box.clickX : box.x + box.width / 2;
            const y = Number.isFinite(box.clickY) ? box.clickY : box.y + box.height / 2;
            target.style.cssText = `left:${box.x}px;top:${box.y}px;width:${box.width}px;height:${box.height}px`;
            cursor.style.cssText = `left:${x + 5}px;top:${y + 5}px`;
            point.style.cssText = `left:${x}px;top:${y}px`;
            point.classList.toggle('clicked', data.stage === 'completed');
        }
        visibility();
        timer = setTimeout(() => { active = false; visibility(); }, data.stage === 'started' ? 5000 : 1100);
    }
    Object.defineProperty(window, '__auraBrowserPresence', { value: Object.freeze({
        show,
        suspend() { suspended++; visibility(); return true; },
        resume() { suspended = Math.max(0, suspended - 1); visibility(); },
    }) });
})();
