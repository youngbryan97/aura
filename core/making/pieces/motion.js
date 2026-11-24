//# vec: points and directions in the plane
//  vec.add(a,b) vec.sub(a,b) vec.mul(a,k) vec.len(a) vec.unit(a) vec.dist(a,b) vec.dot(a,b) vec.angle(a) vec.from(angle,length) vec.lerp(a,b,t) vec.turn(a,angle) vec.toward(from,to,step) -> {x,y} or a number
const vec = { add: (a, b) => ({ x: a.x + b.x, y: a.y + b.y }), sub: (a, b) => ({ x: a.x - b.x, y: a.y - b.y }), mul: (a, k) => ({ x: a.x * k, y: a.y * k }),
  len: (a) => Math.hypot(a.x, a.y), unit: (a) => { const n = Math.hypot(a.x, a.y) || 1; return { x: a.x / n, y: a.y / n }; }, dist: (a, b) => Math.hypot(a.x - b.x, a.y - b.y),
  dot: (a, b) => a.x * b.x + a.y * b.y, angle: (a) => Math.atan2(a.y, a.x), from: (t, n = 1) => ({ x: Math.cos(t) * n, y: Math.sin(t) * n }), lerp: (a, b, t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t }),
  turn: (a, t) => ({ x: a.x * Math.cos(t) - a.y * Math.sin(t), y: a.x * Math.sin(t) + a.y * Math.cos(t) }),
  toward: (a, b, step) => { const d = Math.hypot(b.x - a.x, b.y - a.y); return d <= step ? { x: b.x, y: b.y } : { x: a.x + ((b.x - a.x) / d) * step, y: a.y + ((b.y - a.y) / d) * step }; } };
//? vec.len({x:3,y:4}) === 5
//? Math.abs(vec.turn({x:1,y:0}, Math.PI/2).y - 1) < 1e-9
//? vec.toward({x:0,y:0},{x:10,y:0},4).x === 4 && vec.toward({x:0,y:0},{x:1,y:0},4).x === 1

//# angles: turning toward a heading the short way round
//  angles.wrap(a) -> -PI..PI; angles.between(a,b) -> signed shortest difference; angles.toward(a,b,step) -> a turned toward b; angles.inside(angle,centre,width) -> bool
const angles = { wrap: (a) => { a = (a + Math.PI) % (2 * Math.PI); return (a < 0 ? a + 2 * Math.PI : a) - Math.PI; }, between: (a, b) => angles.wrap(b - a),
  toward: (a, b, step) => { const d = angles.wrap(b - a); return Math.abs(d) <= step ? b : a + Math.sign(d) * step; }, inside: (a, centre, width) => Math.abs(angles.wrap(a - centre)) <= width / 2 };
//? Math.abs(angles.between(3, -3) - (2*Math.PI - 6)) < 1e-9
//? angles.inside(0.1, 0, 0.4) && !angles.inside(1, 0, 0.4)

//# hits: whether shapes touch, and where a moving one first does
//  hits.circles(a,b) with {x,y,r}; hits.rects(a,b) with {x,y,w,h}; hits.circleRect(c,r); hits.inRect(p,r); hits.inPoly(p,points); hits.segments(a,b,c,d) -> {x,y}|null; hits.segCircle(a,b,c) -> bool; hits.swept(r,dx,dy,block) -> {t,nx,ny}|null (t 0..1 of the move)
const hits = { circles: (a, b) => Math.hypot(a.x - b.x, a.y - b.y) <= a.r + b.r, rects: (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h,
  circleRect: (c, r) => Math.hypot(c.x - Math.max(r.x, Math.min(c.x, r.x + r.w)), c.y - Math.max(r.y, Math.min(c.y, r.y + r.h))) <= c.r,
  inRect: (p, r) => p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h,
  inPoly: (p, pts) => { let inside = false; for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) { const a = pts[i], b = pts[j]; if (a.y > p.y !== b.y > p.y && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) inside = !inside; } return inside; },
  segments: (a, b, c, d) => { const rx = b.x - a.x, ry = b.y - a.y, sx = d.x - c.x, sy = d.y - c.y, den = rx * sy - ry * sx; if (!den) return null; const t = ((c.x - a.x) * sy - (c.y - a.y) * sx) / den, u = ((c.x - a.x) * ry - (c.y - a.y) * rx) / den; return t >= 0 && t <= 1 && u >= 0 && u <= 1 ? { x: a.x + t * rx, y: a.y + t * ry } : null; },
  segCircle: (a, b, c) => { const dx = b.x - a.x, dy = b.y - a.y, n = dx * dx + dy * dy || 1, t = Math.max(0, Math.min(1, ((c.x - a.x) * dx + (c.y - a.y) * dy) / n)); return Math.hypot(a.x + t * dx - c.x, a.y + t * dy - c.y) <= c.r; },
  swept: (r, dx, dy, b) => { const inv = (n) => (n === 0 ? Infinity : 1 / n); const ex = dx > 0 ? (b.x - (r.x + r.w)) : (b.x + b.w - r.x), lx = dx > 0 ? b.x + b.w - r.x : b.x - (r.x + r.w), ey = dy > 0 ? b.y - (r.y + r.h) : b.y + b.h - r.y, ly = dy > 0 ? b.y + b.h - r.y : b.y - (r.y + r.h);
    if (dx === 0 && !(r.x < b.x + b.w && r.x + r.w > b.x)) return null; if (dy === 0 && !(r.y < b.y + b.h && r.y + r.h > b.y)) return null;
    const tx = dx === 0 ? -Infinity : ex * inv(dx), ty = dy === 0 ? -Infinity : ey * inv(dy), ox = dx === 0 ? Infinity : lx * inv(dx), oy = dy === 0 ? Infinity : ly * inv(dy), t = Math.max(tx, ty), out = Math.min(ox, oy); if (t > out || t < 0 || t > 1) return null; return tx > ty ? { t, nx: -Math.sign(dx), ny: 0 } : { t, nx: 0, ny: -Math.sign(dy) }; } };
//? hits.circles({x:0,y:0,r:5},{x:9,y:0,r:5}) && !hits.circles({x:0,y:0,r:4},{x:9,y:0,r:4})
//? hits.inPoly({x:1,y:1},[{x:0,y:0},{x:4,y:0},{x:0,y:4}]) && !hits.inPoly({x:3,y:3},[{x:0,y:0},{x:4,y:0},{x:0,y:4}])
//? hits.segments({x:0,y:0},{x:2,y:2},{x:0,y:2},{x:2,y:0}).x === 1
//? hits.swept({x:0,y:0,w:10,h:10},100,0,{x:50,y:0,w:10,h:10}).t === 0.4
//? hits.swept({x:0,y:0,w:10,h:10},10,0,{x:50,y:0,w:10,h:10}) === null

//# bodies: things that move with speed, gravity, friction and bounce
//  bodies.move(b,dt,{gravity,drag,max}) changes b.x,b.y,b.vx,b.vy; bodies.bounds(b,{x,y,w,h},bounce) keeps it in, gives "left"|"right"|"top"|"bottom"|null; bodies.bounce(a,b) two circles {x,y,r,vx,vy,m} knock apart; bodies.wrap(b,w,h,margin); bodies.push(b,fromX,fromY,strength)
const bodies = { move: (b, dt, o = {}) => { b.vy = (b.vy || 0) + (o.gravity || 0) * dt; const k = o.drag ? Math.exp(-o.drag * dt) : 1; b.vx = (b.vx || 0) * k; b.vy *= k; if (o.max) { const n = Math.hypot(b.vx, b.vy); if (n > o.max) { b.vx *= o.max / n; b.vy *= o.max / n; } } b.x += b.vx * dt; b.y += b.vy * dt; return b; },
  bounds: (b, area, bounce = 0) => { const r = b.r || 0, w = b.w || 0, h = b.h || 0, left = area.x + r, right = area.x + area.w - (w || r), top = area.y + r, bottom = area.y + area.h - (h || r); let hit = null; if (b.x < left) { b.x = left; b.vx = Math.abs(b.vx || 0) * bounce; hit = "left"; } else if (b.x > right) { b.x = right; b.vx = -Math.abs(b.vx || 0) * bounce; hit = "right"; } if (b.y < top) { b.y = top; b.vy = Math.abs(b.vy || 0) * bounce; hit = "top"; } else if (b.y > bottom) { b.y = bottom; b.vy = -Math.abs(b.vy || 0) * bounce; hit = "bottom"; } return hit; },
  bounce: (a, b, e = 1) => { const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1e-9; if (d > a.r + b.r) return false; const nx = dx / d, ny = dy / d, ma = a.m || 1, mb = b.m || 1, gap = (a.r + b.r - d) / (ma + mb); a.x -= nx * gap * mb; a.y -= ny * gap * mb; b.x += nx * gap * ma; b.y += ny * gap * ma; const v = ((b.vx || 0) - (a.vx || 0)) * nx + ((b.vy || 0) - (a.vy || 0)) * ny; if (v < 0) { const j = (-(1 + e) * v) / (1 / ma + 1 / mb); a.vx = (a.vx || 0) - (j / ma) * nx; a.vy = (a.vy || 0) - (j / ma) * ny; b.vx = (b.vx || 0) + (j / mb) * nx; b.vy = (b.vy || 0) + (j / mb) * ny; } return true; },
  wrap: (b, w, h, m = 0) => { if (b.x < -m) b.x = w + m; else if (b.x > w + m) b.x = -m; if (b.y < -m) b.y = h + m; else if (b.y > h + m) b.y = -m; return b; },
  push: (b, x, y, k) => { const dx = b.x - x, dy = b.y - y, d = Math.hypot(dx, dy) || 1; b.vx = (b.vx || 0) + (dx / d) * k; b.vy = (b.vy || 0) + (dy / d) * k; return b; } };
//? bodies.move({x:0,y:0,vx:10,vy:0},0.5).x === 5
//? bodies.bounds({x:-3,y:5,vx:-9,r:2},{x:0,y:0,w:100,h:100},1) === "left"
//? (() => { const a={x:0,y:0,r:5,vx:10,vy:0}, b={x:9,y:0,r:5,vx:0,vy:0}; bodies.bounce(a,b); return Math.abs(a.vx) < 1e-9 && Math.abs(b.vx-10) < 1e-9; })()

//# tiles: a world of squares written as rows of characters, and moving through it
//  tiles.read(rows) -> {w,h,at}; tiles.at(map,x,y) -> char (outside is "#"); tiles.set(map,x,y,c); tiles.find(map,c) -> [{x,y}]; tiles.move(box,dx,dy,map,size,solid) moves {x,y,w,h} and stops at solid squares -> {left,right,top,bottom} it ran into; solid is a string of characters, default "#"
const tiles = { read: (rows) => ({ w: Math.max(...rows.map((r) => r.length)), h: rows.length, at: rows.map((r) => r.split("")) }), at: (m, x, y) => (y >= 0 && y < m.h && x >= 0 && x < m.w ? m.at[y][x] || " " : "#"), set: (m, x, y, c) => { if (y >= 0 && y < m.h && x >= 0 && x < m.w) m.at[y][x] = c; return m; },
  find: (m, c) => { const out = []; m.at.forEach((r, y) => r.forEach((k, x) => { if (k === c) out.push({ x, y }); })); return out; },
  move: (b, dx, dy, m, size, solid = "#") => { const hit = { left: false, right: false, top: false, bottom: false }; const blocked = () => { for (let y = Math.floor(b.y / size); y <= Math.floor((b.y + b.h - 1e-6) / size); y++) for (let x = Math.floor(b.x / size); x <= Math.floor((b.x + b.w - 1e-6) / size); x++) if (solid.includes(tiles.at(m, x, y))) return { x, y }; return null; };
    const go = (d, axis) => { let left = d; while (left) { const s = Math.abs(left) > size / 2 ? (Math.sign(left) * size) / 2 : left; b[axis] += s; const c = blocked(); if (c) { if (axis === "x") { if (s > 0) { b.x = c.x * size - b.w; hit.right = true; } else { b.x = (c.x + 1) * size; hit.left = true; } } else if (s > 0) { b.y = c.y * size - b.h; hit.bottom = true; } else { b.y = (c.y + 1) * size; hit.top = true; } return; } left -= s; } };
    go(dx, "x"); go(dy, "y"); return hit; } };
//? tiles.move({x:10,y:10,w:10,h:10},100,0,tiles.read(["#####","#   #","#####"]),32).right === true
//? (() => { const b={x:40,y:40,w:20,h:20}; tiles.move(b,0,500,tiles.read(["#####","#   #","#   #","#####"]),32); return b.y === 76; })()

//# particles: sparks, smoke and splashes
//  particles.burst(list,x,y,{n,speed,life,color,size,spread,angle,gravity}); particles.step(list,dt) -> list still alive; particles.draw(list,g)
const particles = { burst: (list, x, y, o = {}) => { for (let i = 0; i < (o.n || 12); i++) { const a = (o.angle || 0) + (o.spread == null ? Math.PI * 2 : o.spread) * (rnd() - 0.5), v = (o.speed || 120) * (0.4 + rnd() * 0.6), life = (o.life || 0.6) * (0.6 + rnd() * 0.4); list.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life, full: life, color: o.color || "#ffd166", size: o.size || 4, g: o.gravity || 0 }); } return list; },
  step: (list, dt) => list.filter((p) => { p.life -= dt; p.vy += p.g * dt; p.x += p.vx * dt; p.y += p.vy * dt; return p.life > 0; }), draw: (list, g) => { for (const p of list) g.circle(p.x, p.y, p.size * Math.max(0.15, p.life / p.full), p.color, { alpha: p.life / p.full }); } };
//? particles.step(particles.burst([],0,0,{n:5,life:1}),2).length === 0

//# camera: a view that follows, shakes and zooms
//  camera.make(x,y) ; camera.follow(c,target,dt,{speed,left,top,right,bottom}); camera.shake(c,amount); camera.step(c,dt); camera.view(c,g,draw) draws the world through it; camera.point(c,x,y) -> where in the world a place on screen is
const camera = { make: (x = 480, y = 300) => ({ x, y, zoom: 1, shaking: 0 }), follow: (c, t, dt, o = {}) => { const k = 1 - Math.exp(-(o.speed || 6) * dt); c.x += (t.x - c.x) * k; c.y += (t.y - c.y) * k; if (o.right != null) { c.x = Math.max((o.left || 0) + W / 2, Math.min(o.right - W / 2, c.x)); c.y = Math.max((o.top || 0) + H / 2, Math.min(o.bottom - H / 2, c.y)); } return c; },
  shake: (c, n) => { c.shaking = Math.max(c.shaking, n); return c; }, step: (c, dt) => { c.shaking = Math.max(0, c.shaking - c.shaking * 6 * dt - 0.5 * dt); return c; },
  view: (c, g, draw) => { const k = c.shaking; g.ctx.save(); g.ctx.translate(W / 2 + Math.sin(g.t * 90) * k, H / 2 + Math.cos(g.t * 77) * k); g.ctx.scale(c.zoom, c.zoom); g.ctx.translate(-c.x, -c.y); draw(g); g.ctx.restore(); }, point: (c, x, y) => ({ x: (x - W / 2) / c.zoom + c.x, y: (y - H / 2) / c.zoom + c.y }) };
//? Math.abs(camera.point({x:100,y:50,zoom:2},W/2+20,H/2).x - 110) < 1e-9
//? camera.follow(camera.make(0,0),{x:100,y:0},10).x > 99

//# sprites: pictures made of squares, written as rows of characters
//  sprites.draw(g,rows,x,y,size,palette,{flip}) draws centred, palette maps a character to a colour, " " and "." are empty; sprites.frame(frames,t,perSecond) -> which to show
const sprites = { draw: (g, rows, x, y, size, palette, o = {}) => { const w = Math.max(...rows.map((r) => r.length)), p = size / Math.max(w, rows.length); rows.forEach((r, j) => { for (let i = 0; i < r.length; i++) { const c = palette[r[i]]; if (c) { g.ctx.globalAlpha = 1; g.ctx.fillStyle = c; g.ctx.fillRect(x + ((o.flip ? w - 1 - i : i) - w / 2) * p, y + (j - rows.length / 2) * p, p + 0.6, p + 0.6); } } }); }, frame: (frames, t, n = 8) => frames[Math.floor(t * n) % frames.length] };
//? sprites.frame(["a","b"],0.2,8) === "b"

//# ease: movement that starts and stops gently
//  ease.in(t) ease.out(t) ease.both(t) ease.bounce(t) ease.elastic(t) with t 0..1; ease.over(from,to,elapsed,length,how) -> value; ease.pulse(t,perSecond) -> 0..1
const ease = { in: (t) => t * t * t, out: (t) => 1 - Math.pow(1 - t, 3), both: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2), bounce: (t) => { const n = 7.5625, d = 2.75; if (t < 1 / d) return n * t * t; if (t < 2 / d) return n * (t -= 1.5 / d) * t + 0.75; if (t < 2.5 / d) return n * (t -= 2.25 / d) * t + 0.9375; return n * (t -= 2.625 / d) * t + 0.984375; }, elastic: (t) => (t === 0 || t === 1 ? t : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * ((2 * Math.PI) / 3)) + 1),
  over: (a, b, t, n, how = "both") => a + (b - a) * ease[how](Math.max(0, Math.min(1, t / n))), pulse: (t, n = 1) => 0.5 + 0.5 * Math.sin(t * n * Math.PI * 2) };
//? ease.over(0,10,1,1) === 10 && ease.over(0,10,0,1) === 0

//# timers: countdowns, cooldowns and things that happen every so often
//  timers.every(s,key,seconds,dt) -> how many times it came due this step; timers.ready(s,key,seconds,dt?) cooldown: true when it may be used, and starts it; timers.step(s,dt) lets cooldowns run; timers.left(s,key); s is the state, they are kept in s.timers
const timers = { of: (s) => (s.timers = s.timers || {}), every: (s, k, n, dt) => { const t = timers.of(s); t[k] = (t[k] || 0) + dt; let due = 0; while (t[k] >= n - 1e-9) { t[k] -= n; due++; } return due; }, ready: (s, k, n) => { const t = timers.of(s), key = "cd:" + k; if ((t[key] || 0) > 1e-9) return false; t[key] = n; return true; }, step: (s, dt) => { const t = timers.of(s); for (const k of Object.keys(t)) if (k.startsWith("cd:")) t[k] = Math.max(0, t[k] - dt); }, left: (s, k) => timers.of(s)["cd:" + k] || 0 };
//? (() => { const s={}; let n=0; for (let i=0;i<60;i++) n += timers.every(s,"a",0.25,1/60); return n === 4; })()
//? (() => { const s={}; const a=timers.ready(s,"shoot",0.5), b=timers.ready(s,"shoot",0.5); timers.step(s,0.6); return a && !b && timers.ready(s,"shoot",0.5); })()

//# waves: what comes when, harder as it goes
//  waves.gap(seconds,{from,to,over}) -> seconds between arrivals at this time; waves.level(score,per) -> 1,2,3..; waves.due(s,key,dt,gap) -> how many arrive now; waves.edge(margin) -> {x,y,side} a place just outside the world
const waves = { gap: (t, o) => o.to + (o.from - o.to) * Math.exp(-t / (o.over || 60)), level: (score, per) => 1 + Math.floor(score / per), due: (s, k, dt, gap) => { s.waves = s.waves || {}; s.waves[k] = (s.waves[k] == null ? gap : s.waves[k]) - dt; let n = 0; while (s.waves[k] <= 0) { s.waves[k] += gap; n++; } return n; }, edge: (m = 20) => { const side = int(0, 3); return side === 0 ? { x: between(0, W), y: -m, side: "top" } : side === 1 ? { x: W + m, y: between(0, H), side: "right" } : side === 2 ? { x: between(0, W), y: H + m, side: "bottom" } : { x: -m, y: between(0, H), side: "left" }; } };
//? waves.gap(0,{from:2,to:0.5,over:30}) === 2 && waves.gap(1000,{from:2,to:0.5,over:30}) < 0.51
//? (() => { const s={}; let n=0; for (let i=0;i<600;i++) n+=waves.due(s,"a",1/60,1); return n >= 9 && n <= 10; })()

//# flock: birds, fish and crowds that keep together
//  flock.step(list,dt,{near,speed,apart,together,align,area}) moves [{x,y,vx,vy}] as a flock
const flock = { step: (list, dt, o = {}) => { const near = o.near || 70, speed = o.speed || 90; for (const b of list) { let n = 0, cx = 0, cy = 0, ax = 0, ay = 0, sx = 0, sy = 0; for (const c of list) { if (c === b) continue; const dx = c.x - b.x, dy = c.y - b.y, d = Math.hypot(dx, dy); if (d < near) { n++; cx += c.x; cy += c.y; ax += c.vx; ay += c.vy; if (d < near / 3 && d > 0) { sx -= dx / d; sy -= dy / d; } } } if (n) { b.vx += ((cx / n - b.x) * (o.together || 0.6) + (ax / n - b.vx) * (o.align || 1) + sx * (o.apart || 120)) * dt; b.vy += ((cy / n - b.y) * (o.together || 0.6) + (ay / n - b.vy) * (o.align || 1) + sy * (o.apart || 120)) * dt; } const v = Math.hypot(b.vx, b.vy) || 1; b.vx *= speed / v; b.vy *= speed / v; b.x += b.vx * dt; b.y += b.vy * dt; if (o.area) { const A = o.area; if (b.x < A.x) b.x += A.w; if (b.x > A.x + A.w) b.x -= A.w; if (b.y < A.y) b.y += A.h; if (b.y > A.y + A.h) b.y -= A.h; } } return list; } };
//? flock.step([{x:0,y:0,vx:1,vy:0},{x:10,y:0,vx:0,vy:1}],0.1,{speed:50}).every(b => Math.abs(Math.hypot(b.vx,b.vy)-50) < 1e-6)

//# ropes: cloth, ropes and chains held together
//  ropes.make(x1,y1,x2,y2,n) -> {points,links}; ropes.step(r,dt,{gravity,drag,rounds}) ; points[i].pin = true holds one still
const ropes = { make: (x1, y1, x2, y2, n) => { const points = [], links = []; for (let i = 0; i <= n; i++) { const x = x1 + ((x2 - x1) * i) / n, y = y1 + ((y2 - y1) * i) / n; points.push({ x, y, px: x, py: y, pin: i === 0 }); if (i) links.push([i - 1, i, Math.hypot(x2 - x1, y2 - y1) / n]); } return { points, links }; },
  step: (r, dt, o = {}) => { for (const p of r.points) { if (p.pin) continue; const vx = (p.x - p.px) * (o.drag || 0.99), vy = (p.y - p.py) * (o.drag || 0.99); p.px = p.x; p.py = p.y; p.x += vx; p.y += vy + (o.gravity == null ? 900 : o.gravity) * dt * dt; } for (let k = 0; k < (o.rounds || 12); k++) for (const [i, j, n] of r.links) { const a = r.points[i], b = r.points[j], dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1e-9, f = (d - n) / d, wa = a.pin ? 0 : b.pin ? 1 : 0.5, wb = b.pin ? 0 : a.pin ? 1 : 0.5; a.x += dx * f * wa; a.y += dy * f * wa; b.x -= dx * f * wb; b.y -= dy * f * wb; } return r; } };
//? (() => { const r=ropes.make(0,0,100,0,10); for (let i=0;i<240;i++) ropes.step(r,1/60); const e=r.points[10]; return Math.hypot(e.x,e.y) < 112 && e.y > 60; })()
