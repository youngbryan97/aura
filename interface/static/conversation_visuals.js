/* Native views of reply data. The source stays available beside each view. */
(() => {
    'use strict';
    const svgNS = 'http://www.w3.org/2000/svg';
    let sequence = 0;
    const element = (tag, className, text) => {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    };
    const svg = (tag, attributes, text) => {
        const node = document.createElementNS(svgNS, tag);
        for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
        if (text !== undefined) node.textContent = text;
        return node;
    };

    function numberOf(text) {
        // Missing, approximate, ranged, and ratio values keep their table.
        const match = String(text).trim().match(/^([−-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(%|ms|s|px|GB|MB|KB|tok\/s)?$/);
        if (!match) return null;
        const value = Number(match[1].replaceAll(',', '').replace('−', '-'));
        return Number.isFinite(value) ? { value, unit: match[2] || '', label: text } : null;
    }

    function sourceView(source, label) {
        const details = element('details', 'conversation-visual-source');
        details.append(element('summary', '', label));
        source.replaceWith(details);
        details.append(source);
        return details;
    }

    function chartTable(table) {
        if (table.closest('.conversation-visual-source')) return;
        const rows = [...table.rows].map(row => [...row.cells].map(cell => cell.textContent.trim()));
        const headers = rows.shift();
        if (!headers || headers.length < 2 || headers.length > 4 || !rows.length || rows.length > 12) return;
        if (rows.some(row => row.length !== headers.length || !row[0] || row[0].length > 100)) return;
        // Identifiers such as years and account numbers are not measurements.
        if (rows.every(row => numberOf(row[0]))) return;
        const columns = headers.slice(1).map((title, index) => ({
            title, values: rows.map(row => numberOf(row[index + 1])),
        }));
        if (columns.some(col => !col.title || col.values.some(value => value === null)
            || new Set(col.values.map(value => value.unit)).size !== 1)) return;

        const figure = element('figure', 'conversation-visual');
        figure.append(element('figcaption', '', rows.length === 1 ? 'At a glance' : 'By the numbers'));
        const metrics = element('div', 'conversation-metrics');
        for (const col of columns) {
            if (rows.length === 1) {
                const metric = element('div', 'conversation-metric');
                metric.append(element('strong', '', col.values[0].label), element('span', '', col.title));
                metrics.append(metric);
                continue;
            }
            const group = element('div', 'conversation-chart');
            group.append(element('h4', '', col.title));
            const low = Math.min(0, ...col.values.map(value => value.value));
            const high = Math.max(0, ...col.values.map(value => value.value));
            const span = high - low || 1;
            const origin = -low / span * 100;
            for (let i = 0; i < rows.length; i++) {
                const value = col.values[i];
                const row = element('div', 'conversation-chart-row');
                row.append(element('span', 'conversation-chart-label', rows[i][0]));
                const track = element('div', 'conversation-chart-track');
                track.setAttribute('aria-hidden', 'true');
                const bar = element('span', 'conversation-chart-bar');
                bar.style.left = `${Math.min(origin, (value.value - low) / span * 100)}%`;
                bar.style.width = `${Math.abs(value.value) / span * 100}%`;
                bar.dataset.negative = String(value.value < 0);
                const baseline = element('span', 'conversation-chart-zero');
                baseline.style.left = `${origin}%`;
                track.append(bar, baseline);
                row.append(track, element('strong', 'conversation-chart-value', value.label));
                group.append(row);
            }
            figure.append(group);
        }
        if (rows.length === 1) {
            figure.append(element('p', 'conversation-metric-label', rows[0][0]), metrics);
        }
        const details = sourceView(table, 'View data table');
        details.before(figure);
        figure.append(details);
    }

    function flowOf(text) {
        const lines = text.trim().split(/\n|;\s*(?=[A-Za-z])/).map(line => line.trim()).filter(line => line && !line.startsWith('%%'));
        const header = lines.shift()?.match(/^(?:flowchart|graph)\s+(LR|TD|TB)$/);
        if (!header || !lines.length || lines.length > 40) return null;
        const nodes = new Map();
        const edges = [];
        const nodeOf = text => {
            const match = text.trim().match(/^([A-Za-z][\w-]*)(?:\[([^\]\n]+)\]|\{([^}\n]+)\}|\(([^)\n]+)\))?$/);
            if (!match) return null;
            const id = match[1];
            const label = (match[2] || match[3] || match[4] || '').replace(/^"(.*)"$/, '$1');
            if (label.length > 60) return null;
            if (!nodes.has(id)) nodes.set(id, { id, label: label || id });
            else if (label) nodes.get(id).label = label;
            return id;
        };
        for (const line of lines) {
            const parts = line.split(/\s*(-->(?:\|[^|\n]+\|)?)\s*/);
            let from = nodeOf(parts[0]);
            if (!from) return null;
            for (let i = 1; i < parts.length; i += 2) {
                const to = nodeOf(parts[i + 1] || '');
                if (!to) return null;
                edges.push({ from, to, label: parts[i].match(/\|([^|]+)\|/)?.[1] || '' });
                from = to;
            }
        }
        if (nodes.size < 2 || nodes.size > 12 || !edges.length || edges.length > 24) return null;
        const levels = new Map();
        let pending = [...nodes.keys()];
        for (let pass = 0; pending.length && pass < nodes.size; pass++) {
            const ready = pending.filter(id => edges.filter(edge => edge.to === id).every(edge => levels.has(edge.from)));
            if (!ready.length) return null; // Cycles retain their source.
            for (const id of ready) {
                levels.set(id, Math.max(0, ...edges.filter(edge => edge.to === id).map(edge => levels.get(edge.from) + 1)));
            }
            pending = pending.filter(id => !levels.has(id));
        }
        return { nodes, edges, levels, horizontal: header[1] === 'LR' };
    }

    function diagram(code, expanded = false) {
        const source = code.closest('.code-block-wrap');
        if (!source || source.closest('.conversation-visual-source')
            || source.querySelector('.code-lang-label')?.textContent !== 'mermaid') return;
        const flow = flowOf(code.textContent);
        if (!flow) return;
        const { nodes, edges, levels } = flow;
        const horizontal = flow.horizontal && (source.closest('.msg-content')?.clientWidth || window.innerWidth) >= 700;
        const counts = new Map();
        const places = new Map();
        for (const node of nodes.values()) {
            const level = levels.get(node.id);
            const slot = counts.get(level) || 0;
            counts.set(level, slot + 1);
            places.set(node.id, horizontal ? [24 + level * 232, 24 + slot * 120] : [24 + slot * 244, 24 + level * 132]);
        }
        const width = Math.max(...[...places.values()].map(([x]) => x)) + 224;
        const height = Math.max(...[...places.values()].map(([, y]) => y)) + 96;
        const figure = element('figure', 'conversation-visual conversation-flow');
        figure.append(element('figcaption', '', 'How it connects'));
        const picture = svg('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img' });
        const titleId = `conversation-flow-${++sequence}`;
        picture.setAttribute('aria-labelledby', titleId);
        picture.append(svg('title', { id: titleId }, edges.map(edge => `${nodes.get(edge.from).label} → ${nodes.get(edge.to).label}${edge.label ? ` (${edge.label})` : ''}`).join('. ')));
        const defs = svg('defs', {});
        const marker = svg('marker', { id: `${titleId}-arrow`, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' });
        marker.append(svg('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: 'currentColor' }));
        defs.append(marker);
        picture.append(defs);
        for (const edge of edges) {
            const [x, y] = places.get(edge.from);
            const [tx, ty] = places.get(edge.to);
            const start = horizontal ? [x + 200, y + 36] : [x + 100, y + 72];
            const end = horizontal ? [tx - 4, ty + 36] : [tx + 100, ty - 4];
            picture.append(svg('line', { x1: start[0], y1: start[1], x2: end[0], y2: end[1], 'marker-end': `url(#${titleId}-arrow)` }));
            if (edge.label) picture.append(svg('text', { x: (start[0] + end[0]) / 2 + 8, y: (start[1] + end[1]) / 2 - 8, class: 'flow-edge-label' }, edge.label));
        }
        for (const node of nodes.values()) {
            const [x, y] = places.get(node.id);
            picture.append(svg('rect', { x, y, width: 200, height: 72, rx: 16 }));
            const label = svg('text', { x: x + 100, y: y + 30, 'text-anchor': 'middle' });
            const words = node.label.match(/.{1,22}(?:\s|$)|.{1,22}/g) || [node.label];
            words.slice(0, 3).forEach((word, index) => label.append(svg('tspan', { x: x + 100, dy: index ? 17 : 0 }, word.trim())));
            picture.append(label);
        }
        const canvas = element('div', 'conversation-flow-canvas');
        picture.style.minWidth = `${width}px`;
        canvas.append(picture);
        figure.append(canvas);
        const details = sourceView(source, 'View diagram source');
        details.open = expanded;
        details.before(figure);
        figure.append(details);
        if (typeof ResizeObserver !== 'undefined' && flow.horizontal) {
            const observer = new ResizeObserver(() => {
                if (!figure.isConnected) { observer.disconnect(); return; }
                const nowHorizontal = (figure.closest('.msg-content')?.clientWidth || window.innerWidth) >= 700;
                if (nowHorizontal === horizontal) return;
                observer.disconnect();
                const open = details.open;
                figure.replaceWith(source);
                diagram(code, open);
            });
            observer.observe(figure);
        }
    }

    function enhance(root) {
        if (!root) return;
        for (const table of root.querySelectorAll('table')) chartTable(table);
        for (const code of root.querySelectorAll('.code-block-wrap code')) diagram(code);
    }

    // Older shells can still call this entry point while their assets update.
    function activity() {
        const stale = document.getElementById?.('aura-browser-activity');
        if (stale) {
            clearTimeout(stale.expiry);
            stale.remove();
        }
    }
    activity();

    window.AuraConversationVisuals = Object.freeze({ enhance, activity, numberOf, flowOf });
})();
