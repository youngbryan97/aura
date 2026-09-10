// What a person using a keyboard, a screen reader, or ordinary eyes can get
// from this interface, measured rather than asserted.
//
// Run it against the live page. It reports what it checked as well as what it
// found, because a check that silently measures nothing reads exactly like a
// check that found nothing wrong — which is what happened twice while this was
// being written. The first version parsed only `rgb()` and the browser reports
// `color(srgb ...)`, so every ratio came back 1.02 and 341 of 406 nodes looked
// broken. The second refused to judge anything sitting under a gradient, and
// since the page ground IS a gradient it measured zero of 291.
//
//     const report = auraAccessibilityAudit();
//
// Everything here is read-only.

(function (global) {
    'use strict';

    //: WCAG 2.1 AA for text, and for text large enough to carry itself.
    const ORDINARY_TEXT = 4.5;
    const LARGE_TEXT = 3.0;
    const LARGE_PX = 24;
    const LARGE_BOLD_PX = 18.66;

    //: The smallest a pointer target may be, from WCAG 2.5.5 at AAA and the
    //: platform guidance both Apple and Google publish.
    const A_TARGET_WORTH_AIMING_AT = 24;

    function channelsOf(colour) {
        if (!colour) return null;
        let found = String(colour).match(
            /^color\(srgb\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s*\/\s*([\d.]+))?/i
        );
        if (found) {
            return [
                +found[1] * 255, +found[2] * 255, +found[3] * 255,
                found[4] === undefined ? 1 : +found[4],
            ];
        }
        found = String(colour).match(/rgba?\(([^)]+)\)/i);
        if (found) {
            const parts = found[1].split(/[,\s/]+/).filter(Boolean).map(Number);
            return [parts[0], parts[1], parts[2], parts[3] === undefined ? 1 : parts[3]];
        }
        found = String(colour).trim().match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
        if (found) {
            const hex = found[1].length === 3
                ? found[1].split('').map((c) => c + c).join('')
                : found[1];
            return [0, 2, 4].map((at) => parseInt(hex.slice(at, at + 2), 16)).concat([1]);
        }
        return null;
    }

    function relativeLuminance(channels) {
        const linear = channels.slice(0, 3).map((value) => {
            const c = value / 255;
            return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
        });
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
    }

    function composite(front, back) {
        const alpha = front[3] === undefined ? 1 : front[3];
        return [0, 1, 2]
            .map((i) => front[i] * alpha + back[i] * (1 - alpha))
            .concat([1]);
    }

    function contrast(one, other) {
        const a = relativeLuminance(one);
        const b = relativeLuminance(other);
        return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
    }

    function colourStopsIn(image) {
        const found = [];
        const pattern = /(?:color\(srgb[^)]*\)|rgba?\([^)]*\)|#[0-9a-f]{3,8})/gi;
        let hit;
        while ((hit = pattern.exec(image))) {
            const parsed = channelsOf(hit[0]);
            if (parsed) found.push(parsed);
        }
        return found;
    }

    // Every colour this text could be sitting on.
    //
    // A gradient is not one colour, so the audit collects each of its stops
    // and judges the text against the one that flatters it MOST. A failure is
    // then a failure everywhere on the element rather than an artefact of
    // where along the ramp the reading was taken.
    function backdropsUnder(element) {
        let node = element;
        const layers = [];
        while (node && node !== document.documentElement) {
            const style = getComputedStyle(node);
            if (style.backgroundImage && style.backgroundImage !== 'none') {
                const stops = colourStopsIn(style.backgroundImage);
                if (stops.length) layers.push(stops);
            }
            const painted = channelsOf(style.backgroundColor);
            if (painted && painted[3] > 0) {
                layers.push([painted]);
                if (painted[3] >= 0.999) break;
            }
            node = node.parentElement;
        }
        const ground = channelsOf(getComputedStyle(document.body).backgroundColor)
            || [5, 3, 10, 1];
        if (!layers.length) return [ground];
        const candidates = [];
        for (const layer of layers) {
            for (const colour of layer) {
                candidates.push(colour[3] >= 0.999 ? colour : composite(colour, ground));
            }
        }
        return candidates.length ? candidates : [ground];
    }

    function onScreen(element) {
        let node = element;
        while (node && node !== document.documentElement) {
            const style = getComputedStyle(node);
            if (style.display === 'none' || style.visibility === 'hidden') return false;
            if (parseFloat(style.opacity) === 0) return false;
            if (node.hasAttribute
                && (node.hasAttribute('hidden') || node.getAttribute('aria-hidden') === 'true')) {
                return false;
            }
            node = node.parentElement;
        }
        const box = element.getBoundingClientRect();
        return box.width > 0 && box.height > 0;
    }

    function accessibleName(element) {
        const label = (element.getAttribute('aria-label') || '').trim();
        if (label) return label;
        const by = element.getAttribute('aria-labelledby');
        if (by) {
            const joined = by.split(/\s+/)
                .map((id) => ((document.getElementById(id) || {}).textContent || ''))
                .join(' ').trim();
            if (joined) return joined;
        }
        const text = (element.textContent || '').replace(/\s+/g, ' ').trim();
        if (text) return text;
        const title = (element.getAttribute('title') || '').trim();
        if (title) return title;
        const image = element.querySelector('img[alt]');
        if (image && image.alt.trim()) return image.alt.trim();
        if (element.labels && element.labels.length) {
            const viaLabel = [...element.labels]
                .map((l) => (l.textContent || '').trim()).join(' ').trim();
            if (viaLabel) return viaLabel;
        }
        const placeholder = (element.getAttribute('placeholder') || '').trim();
        return placeholder;
    }

    function leafTextNodes() {
        return [...document.querySelectorAll(
            'button,a,label,p,span,div,li,h1,h2,h3,h4,h5,h6,td,th,summary,option'
        )]
            .filter((el) => (el.textContent || '').trim().length > 0)
            .filter((el) => ![...el.children].some(
                (child) => (child.textContent || '').trim().length > 0
            ))
            .filter(onScreen);
    }

    function contrastFindings() {
        const findings = [];
        const already = new Set();
        let measured = 0;
        for (const element of leafTextNodes()) {
            const style = getComputedStyle(element);
            const ink = channelsOf(style.color);
            if (!ink) continue;
            // Ink with no alpha is not hard to read; it is not painted.
            //
            // Below 480px the mute control becomes a 42-pixel pill with a
            // coloured dot drawn by `::after`, and its label is hidden with
            // `color: transparent` — a deliberate icon-only button, still
            // named for a screen reader by the text that is there. Judged as
            // contrast it reads 1:1 and looks like the worst defect on the
            // page. What matters for such a control is whether it has a name,
            // and `controlFindings` asks that of every control already.
            if (ink[3] === 0) continue;
            measured += 1;
            let best = 0;
            for (const backdrop of backdropsUnder(element)) {
                const ratio = contrast(composite(ink, backdrop), backdrop);
                if (ratio > best) best = ratio;
            }
            const size = parseFloat(style.fontSize) || 14;
            const bold = (parseInt(style.fontWeight, 10) || 400) >= 700;
            const needed = (size >= LARGE_PX || (size >= LARGE_BOLD_PX && bold))
                ? LARGE_TEXT : ORDINARY_TEXT;
            if (best >= needed) continue;
            const shape = `${style.color}|${size}|${element.className}`;
            if (already.has(shape)) continue;
            already.add(shape);
            findings.push({
                kind: 'contrast',
                where: (element.className || '').toString().split(' ')[0]
                    || element.tagName.toLowerCase(),
                text: (element.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 40),
                ratio: Number(best.toFixed(2)),
                needed,
                px: Number(size.toFixed(1)),
                colour: style.color,
            });
        }
        findings.sort((a, b) => a.ratio - b.ratio);
        return { measured, findings };
    }

    function controlFindings() {
        const controls = [...document.querySelectorAll(
            'button,[role=button],a[href],input,select,textarea,[tabindex]'
        )].filter(onScreen);
        const findings = [];
        for (const control of controls) {
            if (!accessibleName(control)) {
                findings.push({
                    kind: 'unnamed control',
                    where: control.id || (control.className || '').toString().split(' ')[0]
                        || control.tagName.toLowerCase(),
                    text: '',
                });
            }
            const box = control.getBoundingClientRect();
            const smallest = Math.min(box.width, box.height);
            if (smallest > 0 && smallest < A_TARGET_WORTH_AIMING_AT) {
                findings.push({
                    kind: 'small target',
                    where: control.id || (control.className || '').toString().split(' ')[0]
                        || control.tagName.toLowerCase(),
                    text: accessibleName(control).slice(0, 30),
                    size: `${Math.round(box.width)}x${Math.round(box.height)}`,
                    needed: A_TARGET_WORTH_AIMING_AT,
                });
            }
        }
        return { measured: controls.length, findings };
    }

    function structureFindings() {
        const findings = [];
        const headings = [...document.querySelectorAll('h1,h2,h3,h4,h5,h6')]
            .filter(onScreen).map((h) => Number(h.tagName[1]));
        if (!document.querySelector('h1')) {
            findings.push({
                kind: 'no first-level heading',
                where: 'document',
                text: 'a page with no h1 has no title in the accessibility tree',
            });
        }
        for (let at = 1; at < headings.length; at++) {
            if (headings[at] - headings[at - 1] > 1) {
                findings.push({
                    kind: 'heading level skipped',
                    where: `h${headings[at - 1]} to h${headings[at]}`,
                    text: '',
                });
            }
        }
        if (!(document.documentElement.lang || '').trim()) {
            findings.push({ kind: 'no document language', where: 'html', text: '' });
        }
        const images = [...document.querySelectorAll('img')]
            .filter(onScreen).filter((img) => !img.hasAttribute('alt'));
        for (const image of images) {
            findings.push({
                kind: 'image without alt',
                where: image.getAttribute('src') || 'img',
                text: '',
            });
        }
        return { measured: headings.length + images.length + 2, findings };
    }

    function auraAccessibilityAudit() {
        const ink = contrastFindings();
        const controls = controlFindings();
        const structure = structureFindings();
        const findings = [...ink.findings, ...controls.findings, ...structure.findings];
        return {
            at: new Date().toISOString(),
            viewport: `${window.innerWidth}x${window.innerHeight}`,
            checked: {
                text: ink.measured,
                controls: controls.measured,
                structure: structure.measured,
            },
            failures: findings.length,
            findings,
        };
    }

    global.auraAccessibilityAudit = auraAccessibilityAudit;
    global.auraContrastBetween = contrast;
    global.auraChannelsOf = channelsOf;
})(typeof window !== 'undefined' ? window : globalThis);
