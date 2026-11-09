/* Installed only in Aura's owned browser context. No page event is intercepted. */
(() => {
    'use strict';
    if (window.top !== window || window.__auraBrowserPresence) return;
    // `held` from her first action on this page until her browser closes: the
    // person can tell at any moment that the window is in her hands, and where
    // her pointer is. Before 8 November 2026 the mark lasted five seconds after
    // an action began, so through a whole video it was not on screen at all.
    let host, root, timer, lastPoint = null, held = false;
    const suspensions = [];
    const SVG = 'http://www.w3.org/2000/svg';
    //: How far into the window the hue reaches, in CSS pixels. A picture whose
    //: clip stays further in than this does not have the hue in it.
    const HUE_REACH = 22;
    const STYLE = `
        :host { pointer-events:none!important; }
        * { box-sizing:border-box; pointer-events:none!important; }
        [hidden] { display:none!important; }
        .hue { position:fixed;inset:0;box-shadow:inset 0 0 6px 2px #9a78ffe6,inset 0 0 18px 3px #8f6bff73;animation:breathe 4.8s ease-in-out infinite; }
        @keyframes breathe { 50% { opacity:.72; } }
        .label { position:fixed;top:15px;right:18px;display:flex;align-items:center;gap:9px;padding:8px 13px 8px 9px;border:1px solid #ae9aff40;border-radius:12px;background:#17131ff5;box-shadow:0 3px 12px #0003,inset 0 1px 0 #ffffff08;color:#ede7f6;font:12px/1.4 system-ui,sans-serif; }
        .identity { display:flex;align-items:center;gap:7px;font-weight:600; }
        .identity svg { width:21px;height:21px; }
        .label .activity { border-left:1px solid #ffffff1c;padding-left:9px;color:#d7cfe5; }
        .label .activity:empty { display:none; }
        .cursor { position:fixed;left:0;top:0;width:30px;height:30px;margin:-15px 0 0 -15px;filter:drop-shadow(0 0 5px #b49affa0) drop-shadow(0 1px 2px #0009);transition:transform .32s cubic-bezier(.2,.8,.2,1); }
        .cursor svg { width:30px;height:30px;display:block; }
        .ripple { position:fixed;left:0;top:0;width:30px;height:30px;margin:-15px 0 0 -15px;border-radius:50%;border:1.5px solid #c9b8ff;opacity:0; }
        .ripple.go { animation:ripple .6s ease-out; }
        @keyframes ripple { from { opacity:.9;transform:var(--at) scale(.5); } to { opacity:0;transform:var(--at) scale(2.2); } }
        @media(prefers-reduced-motion:reduce) { .hue, .ripple.go { animation:none; } .cursor { transition:none; } }`;
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
        part('hue', root);
        const label = part('label', root);
        const identity = document.createElement('span');
        identity.className = 'identity';
        identity.append(neuron(__AURA_NEURON__, 'label-'), document.createTextNode('Aura'));
        const activity = document.createElement('span');
        activity.className = 'activity';
        label.append(identity, activity);
        part('ripple', root);
        const cursor = part('cursor', root);
        cursor.append(neuron(__AURA_NEURON__, ''));
        cursor.hidden = true;
        (document.documentElement || document).append(host);
    }
    function overlaps(a, b) {
        return a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
    }
    // A picture of the whole window (no clip) leaves out all of the mark; a
    // picture of part of it leaves out only what of the mark falls inside, so
    // the hue does not blink each time she looks at the video she is watching.
    function inPicture(name) {
        if (!suspensions.length) return false;
        return suspensions.some(clip => {
            if (!clip) return true;
            if (name === 'hue') {
                return clip.x < HUE_REACH || clip.y < HUE_REACH
                    || clip.x + clip.width > innerWidth - HUE_REACH || clip.y + clip.height > innerHeight - HUE_REACH;
            }
            const node = root.querySelector(`.${name}`);
            if (!node || node.hidden) return false;
            const box = node.getBoundingClientRect();
            return overlaps({ x: box.x, y: box.y, width: box.width, height: box.height }, clip);
        });
    }
    function visibility() {
        if (!host) return;
        host.style.setProperty('display', held ? 'block' : 'none', 'important');
        if (!held) return;
        for (const name of ['hue', 'label', 'cursor', 'ripple']) {
            const node = root.querySelector(`.${name}`);
            if (node) node.style.setProperty('visibility', inPicture(name) ? 'hidden' : 'visible');
        }
    }
    function moveCursor(point) {
        const cursor = root.querySelector('.cursor');
        if (!point) { cursor.hidden = !lastPoint; return; }
        // It glides from where it was; where it was nowhere, it appears in place.
        const appearing = !lastPoint || cursor.hidden;
        if (appearing) cursor.style.transition = 'none';
        lastPoint = point;
        cursor.hidden = false;
        cursor.style.transform = `translate(${point.x}px, ${point.y}px)`;
        if (appearing) { void cursor.offsetWidth; cursor.style.transition = ''; }
    }
    function ripple(point) {
        const node = root.querySelector('.ripple');
        node.style.setProperty('--at', `translate(${point.x}px, ${point.y}px)`);
        node.style.transform = `translate(${point.x}px, ${point.y}px)`;
        node.classList.remove('go');
        void node.offsetWidth;
        node.classList.add('go');
    }
    function pointOf(data) {
        const box = data.box;
        if (box && [box.x, box.y, box.width, box.height].every(Number.isFinite)) {
            return {
                x: Number.isFinite(box.clickX) ? box.clickX : box.x + box.width / 2,
                y: Number.isFinite(box.clickY) ? box.clickY : box.y + box.height / 2,
            };
        }
        const at = data.point;
        return at && Number.isFinite(at.x) && Number.isFinite(at.y) ? { x: at.x, y: at.y } : null;
    }
    function show(data) {
        mount();
        clearTimeout(timer);
        held = data.action !== 'closed';
        const labels = { navigating:'Opening this page', reading:'Reading this page', clicking:'Clicking here', typing:'Typing here', scrolling:'Scrolling this page' };
        const completed = { navigating:'Page opened', reading:'Page read', clicking:'Clicked here', typing:'Finished typing', scrolling:'Scrolled' };
        const activity = root.querySelector('.activity');
        activity.textContent = `${(data.stage === 'completed' ? completed[data.action] : labels[data.action]) || 'Working here'}${data.stage === 'failed' ? ' · stopped' : ''}`;
        const point = pointOf(data);
        moveCursor(point);
        if (point && data.action === 'clicking' && data.stage === 'completed') ripple(point);
        visibility();
        // What she is doing is said while she does it; her name, her hue and
        // her pointer stay for as long as the window is hers.
        timer = setTimeout(() => { activity.textContent = ''; }, data.stage === 'started' ? 5000 : 1100);
        return lastPoint;
    }
    Object.defineProperty(window, '__auraBrowserPresence', { value: Object.freeze({
        show,
        suspend(clip) {
            const valid = clip && [clip.x, clip.y, clip.width, clip.height].every(Number.isFinite);
            suspensions.push(valid ? { x: clip.x, y: clip.y, width: clip.width, height: clip.height } : null);
            visibility();
            return true;
        },
        resume() { suspensions.pop(); visibility(); },
    }) });
})();
