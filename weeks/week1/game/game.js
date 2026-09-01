/* ===========================================================================
   WEB-CRAWLER: navigate Wikipedia's Marvel graph in the fog.

   Data: the course's frozen week-1 snapshot (303 nodes, 1784 directed edges),
   preprocessed into data/marvel_week1.js by scripts/build_graph.py.

   The idea: you stand on an article and can only click onward along links
   going OUT of it, exactly like a real reader. The map does not exist until
   you have stood somewhere and seen what it links to. The mission is a round
   trip: find the target, then get home. "Par" is the BFS shortest route both
   ways, so you are playing against the structure of the graph, not the clock.
   =========================================================================== */

(function () {
  'use strict';

  const D = window.MARVEL_DATA;
  if (!D) {
    document.body.innerHTML = '<p style="padding:40px;font:16px sans-serif;color:#fff">' +
      'data/marvel_week1.js is missing. Run <code>python scripts/build_graph.py</code>.</p>';
    return;
  }

  const N = D.nodes.length;
  const OUT = D.out, IN = D.in;
  const byId = new Map(D.nodes.map((n, i) => [n.id, i]));

  const WORLD = 3000;                      // world extent in pixels at zoom 1
  const HERO_SCALE = 1.95;                 // makes the hero ~80 world units tall
  const TAU = Math.PI * 2;

  // Node geometry: radius scales with in-degree, so the hubs actually look like hubs.
  const nodes = D.nodes.map((n, i) => ({
    ...n, idx: i,
    wx: n.x * WORLD, wy: n.y * WORLD,
    r: 8.5 + 2.15 * Math.sqrt(IN[i].length),
    inDeg: IN[i].length, outDeg: OUT[i].length,
  }));

  // ------------------------------------------------------------------ helpers
  const $ = (id) => document.getElementById(id);
  const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
  const lerp = (a, b, t) => a + (b - a) * t;
  const easeInOut = (t) => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  const shortName = (s) => s.replace(/\s*\((character|comics|Marvel Comics|Marvel Comics character|characters)\)\s*$/i, '');

  /** BFS in one direction. adj = OUT for "where can I go", IN for "who reaches here". */
  function bfs(src, adj) {
    const dist = new Int16Array(N).fill(-1);
    dist[src] = 0;
    const q = [src];
    for (let h = 0; h < q.length; h++) {
      const u = q[h], d = dist[u] + 1;
      for (const v of adj[u]) if (dist[v] < 0) { dist[v] = d; q.push(v); }
    }
    return dist;
  }

  /** Shortest route from a to b along OUT edges, or null if there is none. */
  function shortestPath(a, b) {
    if (a === b) return [a];
    const prev = new Int32Array(N).fill(-1);
    const seen = new Uint8Array(N); seen[a] = 1;
    const q = [a];
    for (let h = 0; h < q.length; h++) {
      const u = q[h];
      for (const v of OUT[u]) {
        if (seen[v]) continue;
        seen[v] = 1; prev[v] = u;
        if (v === b) { const p = [b]; let c = b; while (c !== a) { c = prev[c]; p.push(c); } return p.reverse(); }
        q.push(v);
      }
    }
    return null;
  }

  // -------------------------------------------------------------------- audio
  // Everything is synthesised with WebAudio so the site ships no sound files.
  const Sfx = {
    ctx: null, on: true,
    boot() {
      if (!this.ctx) { const A = window.AudioContext || window.webkitAudioContext; if (A) this.ctx = new A(); }
      if (this.ctx && this.ctx.state === 'suspended') this.ctx.resume();
    },
    tone(freq, dur, type, gain, slideTo) {
      if (!this.on || !this.ctx) return;
      const t = this.ctx.currentTime;
      const o = this.ctx.createOscillator(), g = this.ctx.createGain();
      o.type = type || 'sine'; o.frequency.setValueAtTime(freq, t);
      if (slideTo) o.frequency.exponentialRampToValueAtTime(slideTo, t + dur);
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(gain || 0.05, t + 0.012);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      o.connect(g).connect(this.ctx.destination); o.start(t); o.stop(t + dur + 0.02);
    },
    noise(dur, gain) {
      if (!this.on || !this.ctx) return;
      const t = this.ctx.currentTime, len = Math.floor(this.ctx.sampleRate * dur);
      const buf = this.ctx.createBuffer(1, len, this.ctx.sampleRate), d = buf.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / len);
      const s = this.ctx.createBufferSource(); s.buffer = buf;
      const f = this.ctx.createBiquadFilter(); f.type = 'highpass'; f.frequency.value = 1400;
      const g = this.ctx.createGain(); g.gain.value = gain || 0.05;
      s.connect(f).connect(g).connect(this.ctx.destination); s.start(t);
    },
    thwip() { this.noise(0.14, 0.05); this.tone(1400, 0.16, 'sawtooth', 0.028, 420); },
    // Arrival pitch rises with the node's in-degree, so hubs literally sound bigger.
    step(deg) { this.tone(220 + Math.min(deg, 40) * 9, 0.13, 'triangle', 0.035); },
    ding() { this.tone(880, 0.18, 'sine', 0.05); setTimeout(() => this.tone(1320, 0.3, 'sine', 0.045), 90); },
    win() { [523, 659, 784, 1046].forEach((f, i) => setTimeout(() => this.tone(f, 0.32, 'triangle', 0.055), i * 130)); },
    buzz() { this.tone(150, 0.22, 'square', 0.035, 90); },
  };

  // -------------------------------------------------------------------- state
  const S = {
    hero: null, home: -1, target: -1, goal: -1, phase: 0,
    cur: -1, path: [], steps: 0, par: 0, legPar: [0, 0],
    seen: new Uint8Array(N), visited: new Uint8Array(N), edges: new Set(),
    moving: null, senses: 3, hint: -1, hintUntil: 0,
    t0: 0, running: false, finished: false, explore: false,
    difficulty: 'normal',
  };

  const cam = { x: WORLD / 2, y: WORLD / 2, z: 1, tx: WORLD / 2, ty: WORLD / 2, tz: 1, free: false };
  let particles = [];
  let hoverNode = -1;

  const DIFFS = {
    easy: { label: 'The neighbourhood', note: '2 to 3 steps out', range: [2, 3], senses: 3 },
    normal: { label: 'Across town', note: '3 to 4 steps out', range: [3, 4], senses: 2 },
    hard: { label: 'The multiverse', note: '5+ steps out', range: [5, 9], senses: 1 },
  };

  // ---------------------------------------------------------------- fog of war
  function reveal(i) {
    S.seen[i] = 1;
    for (const v of OUT[i]) { S.seen[v] = 1; S.edges.add(i * N + v); }
    // Edges between places you already know also surface, so the map knits itself together.
    for (const u of IN[i]) if (S.visited[u]) S.edges.add(u * N + i);
  }

  // -------------------------------------------------------------- round setup
  function pickTarget(home, range) {
    const fromHome = bfs(home, OUT);          // how far out each node is
    const toHome = bfs(home, IN);             // reverse BFS: who can still reach home
    let pool = [];
    for (let i = 0; i < N; i++) {
      if (i === home) continue;
      const a = fromHome[i], b = toHome[i];
      if (a < 0 || b < 0) continue;           // must be reachable AND have a way back
      if (a >= range[0] && a <= range[1]) pool.push(i);
    }
    if (!pool.length) {                        // fall back to anything on a round trip
      for (let i = 0; i < N; i++) if (i !== home && fromHome[i] > 1 && toHome[i] > 0) pool.push(i);
    }
    const idx = pool[Math.floor(Math.random() * pool.length)];
    return { idx, out: fromHome[idx], back: toHome[idx] };
  }

  function newRound(hero, difficulty) {
    S.hero = hero;
    S.difficulty = difficulty;
    S.home = byId.get(hero.node);
    const cfg = DIFFS[difficulty];
    const t = pickTarget(S.home, cfg.range);
    S.target = t.idx;
    S.legPar = [t.out, t.back];
    S.par = t.out + t.back;
    S.goal = S.target; S.phase = 0;
    S.cur = S.home; S.path = [S.home];
    S.steps = 0; S.senses = cfg.senses; S.hint = -1;
    S.seen = new Uint8Array(N); S.visited = new Uint8Array(N); S.edges = new Set();
    S.moving = null; S.finished = false; S.explore = false;
    S.visited[S.home] = 1; reveal(S.home);
    S.t0 = performance.now(); S.running = true;
    particles = [];

    const h = nodes[S.home];
    cam.x = cam.tx = h.wx; cam.y = cam.ty = h.wy; cam.z = cam.tz = 1.05; cam.free = false;
    syncUI();
    toast(`Mission: find <b>${shortName(nodes[S.target].name)}</b>, then get back home.`, 4200);
  }

  // --------------------------------------------------------------------- moves
  function travelTo(v) {
    if (!S.running || S.moving || S.finished) return;
    if (!OUT[S.cur].includes(v)) {
      Sfx.buzz();
      toast('No link that way. You can only follow links <b>out</b> of the page you are on.');
      return;
    }
    const a = nodes[S.cur], b = nodes[v];
    const dist = Math.hypot(b.wx - a.wx, b.wy - a.wy);
    S.moving = {
      from: S.cur, to: v, t0: performance.now(),
      dur: clamp(dist / 420 * 900, 520, 1500),
      facing: (b.wx >= a.wx) ? 1 : -1,
      anchor: { x: lerp(a.wx, b.wx, 0.5) + (Math.random() - 0.5) * 60, y: Math.min(a.wy, b.wy) - dist * 0.55 - 120 },
    };
    S.hint = -1;
    cam.free = false;
    Sfx.boot();
    if (S.hero.travel === 'swing') Sfx.thwip();
    else if (S.hero.travel === 'run') Sfx.noise(0.22, 0.035);
    else Sfx.tone(520, 0.3, 'sine', 0.03, 900);
    renderLinks();
  }

  function arrive(v) {
    S.cur = v; S.visited[v] = 1; S.steps++;
    S.path.push(v);
    reveal(v);
    burst(nodes[v].wx, nodes[v].wy, S.hero.tint, 16);
    Sfx.step(nodes[v].inDeg);

    if (S.phase === 0 && v === S.target) {
      S.phase = 1; S.goal = S.home;
      Sfx.ding();
      burst(nodes[v].wx, nodes[v].wy, '#ffc93f', 40);
      toast(`Found <b>${shortName(nodes[v].name)}</b>. Now get home to <b>${shortName(nodes[S.home].name)}</b>.`, 4200);
    } else if (S.phase === 1 && v === S.home) {
      finish();
      return;
    } else if (OUT[v].length === 0) {
      Sfx.buzz();
      toast('<b>Dead end.</b> This article links to nobody else in the category. Use Web-zip.', 4500);
    }
    syncUI();
  }

  function webZip() {
    if (!S.running || S.moving || S.finished || S.path.length < 2) return;
    S.path.pop();
    const back = S.path[S.path.length - 1];
    S.steps += 2;
    S.cur = back;
    // Deliberately no phase rollback: once the target is found it stays found.
    Sfx.thwip();
    burst(nodes[back].wx, nodes[back].wy, '#5fe3ff', 20);
    if (S.phase === 1 && back === S.home) { finish(); return; }
    cam.free = false;
    syncUI();
    toast('Web-zip: one step back, two steps on the meter.');
  }

  function spiderSense() {
    if (!S.running || S.moving || S.finished) return;
    if (S.senses <= 0) { Sfx.buzz(); toast('Out of spider-sense.'); return; }
    const p = shortestPath(S.cur, S.goal);
    if (!p || p.length < 2) {
      Sfx.buzz();
      toast('No route to the objective from here. You have to zip back.', 4000);
      return;
    }
    S.senses--; S.steps += 3;
    S.hint = p[1]; S.hintUntil = performance.now() + 7000;
    Sfx.tone(1200, 0.4, 'sine', 0.04, 1800);
    toast(`Spider-sense tingling: go via <b>${shortName(nodes[S.hint].name)}</b>, ${p.length - 1} steps left. (+3 steps)`, 5000);
    syncUI();
  }

  function finish() {
    S.finished = true; S.running = false;
    const secs = (performance.now() - S.t0) / 1000;
    Sfx.win();
    burst(nodes[S.home].wx, nodes[S.home].wy, '#ffc93f', 70);

    const diff = S.steps - S.par;
    let verdict, line;
    if (diff <= 0) {
      verdict = 'PERFECT ROUTE!';
      line = 'You walked the shortest route that exists in this graph. That is not luck, that is reading the network.';
    } else if (diff <= 2) {
      verdict = 'SPECTACULAR!';
      line = 'Almost optimal. You aimed at the hubs and let them do the work.';
    } else if (diff <= 5) {
      verdict = 'AMAZING!';
      line = 'Solid navigation. A couple of detours, but you found your way back.';
    } else {
      verdict = 'YOU MADE IT HOME.';
      line = 'The long way round. The graph is directed, and what took you out rarely brings you back.';
    }

    $('wVerdict').textContent = verdict;
    $('wLine').textContent = line;
    $('wSteps').textContent = S.steps;
    $('wPar').textContent = S.par;
    $('wDiff').textContent = (diff > 0 ? '+' : '') + diff;
    $('wDiffBox').className = 'big ' + (diff <= 0 ? 'good' : diff > 5 ? 'bad' : '');
    $('wTime').textContent = fmtTime(secs);
    $('wSeen').textContent = S.seen.reduce((a, b) => a + b, 0);

    $('wPath').innerHTML = '<span class="k">Your route (' + (S.path.length - 1) + ' hops)</span>' +
      S.path.map(i => '<em>' + shortName(nodes[i].name) + '</em>').join('<span class="arrow">&rarr;</span>');

    const p1 = shortestPath(S.home, S.target) || [];
    const p2 = shortestPath(S.target, S.home) || [];
    const opt = p1.concat(p2.slice(1));
    $('wOpt').innerHTML = '<span class="k">Shortest possible route (par ' + S.par + ')</span>' +
      opt.map(i => '<em>' + shortName(nodes[i].name) + '</em>').join('<span class="arrow">&rarr;</span>');

    $('vWin').classList.remove('gone');
  }

  // ---------------------------------------------------------------- particles
  function burst(x, y, color, n) {
    for (let i = 0; i < n; i++) {
      const a = Math.random() * TAU, s = 40 + Math.random() * 190;
      particles.push({ x, y, vx: Math.cos(a) * s, vy: Math.sin(a) * s, life: 1, color, size: 1 + Math.random() * 2.6 });
    }
  }

  function stepParticles(dt) {
    for (const p of particles) {
      p.x += p.vx * dt; p.y += p.vy * dt;
      p.vx *= 0.94; p.vy *= 0.94; p.vy += 60 * dt;
      p.life -= dt * 1.35;
    }
    particles = particles.filter(p => p.life > 0);
    if (particles.length > 600) particles.splice(0, particles.length - 600);
  }

  // ------------------------------------------------------------------- canvas
  const cv = $('cv'), ctx = cv.getContext('2d');
  let VW = 0, VH = 0, DPR = 1;

  function resize() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    const r = cv.getBoundingClientRect();
    VW = r.width; VH = r.height;
    cv.width = Math.round(VW * DPR); cv.height = Math.round(VH * DPR);
  }
  window.addEventListener('resize', resize);

  /* The side panel covers the right edge of the canvas, so the middle of the
     window is not the middle of the map. We move the view centre to the middle
     of the VISIBLE field, otherwise the hero ends up under the panel every time
     the camera recentres. */
  function viewCx() {
    const p = $('panel');
    const w = p.classList.contains('hidden') ? 0 : Math.min(p.getBoundingClientRect().width, VW * 0.5);
    return (VW - w) / 2;
  }
  const viewCy = () => VH / 2;

  const toScreen = (wx, wy) => ({ x: (wx - cam.x) * cam.z + viewCx(), y: (wy - cam.y) * cam.z + viewCy() });

  // Fixed star dust in world space, so panning actually feels like movement.
  const dust = (() => {
    let s = 12345;
    const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    return Array.from({ length: 420 }, () => ({
      x: rnd() * WORLD * 1.6 - WORLD * 0.3, y: rnd() * WORLD * 1.6 - WORLD * 0.3,
      r: 0.5 + rnd() * 1.5, a: 0.05 + rnd() * 0.22,
    }));
  })();

  let lastFacing = 1;

  /** Where the hero is standing or flying right now, in world coordinates (feet). */
  function heroPos(now) {
    if (!S.moving) {
      const n = nodes[S.cur];
      return { x: n.wx, y: n.wy + 3, prog: 0, moving: false, facing: lastFacing };
    }
    const m = S.moving;
    const raw = clamp((now - m.t0) / m.dur, 0, 1);
    const a = nodes[m.from], b = nodes[m.to];
    const e = easeInOut(raw);
    let x = lerp(a.wx, b.wx, e), y = lerp(a.wy, b.wy, e);
    const dx = b.wx - a.wx, dy = b.wy - a.wy, len = Math.hypot(dx, dy) || 1;
    const px = -dy / len, py = dx / len;
    const arc = Math.sin(raw * Math.PI);
    // Each hero moves along a different curve: a big swing, a gliding wobble, a flat sprint.
    if (S.hero.travel === 'swing') { x += px * arc * len * 0.30; y += py * arc * len * 0.30 - arc * 46; }
    else if (S.hero.travel === 'float') { const w = Math.sin(raw * Math.PI * 2.2) * 22; x += px * w; y += py * w - arc * 16; }
    else { y -= arc * 10; }
    lastFacing = m.facing;
    return { x, y: y + 3, prog: raw, moving: true, facing: m.facing };
  }

  function draw(now, dt) {
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, VW, VH);

    if (S.moving && now - S.moving.t0 >= S.moving.dur) { const to = S.moving.to; S.moving = null; arrive(to); }

    const hp = heroPos(now);

    if (!cam.free) { cam.tx = hp.x; cam.ty = hp.y; }
    const k = 1 - Math.pow(0.0016, dt);
    cam.x = lerp(cam.x, cam.tx, k); cam.y = lerp(cam.y, cam.ty, k); cam.z = lerp(cam.z, cam.tz, k);

    for (const d of dust) {
      const p = toScreen(d.x, d.y);
      if (p.x < -20 || p.x > VW + 20 || p.y < -20 || p.y > VH + 20) continue;
      ctx.fillStyle = `rgba(150,190,255,${d.a})`;
      ctx.beginPath(); ctx.arc(p.x, p.y, d.r, 0, TAU); ctx.fill();
    }

    drawEdges(now);
    drawTrail();
    drawNodes(now);
    drawParticles();
    drawHeroOnMap(now, hp);
    drawCompass(hp);

    stepParticles(dt);
  }

  function edgeCurve(a, b) {
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1;
    return { cx: mx - dy / L * L * 0.10, cy: my + dx / L * L * 0.10 };
  }

  function drawEdges(now) {
    const frontier = new Set(OUT[S.cur]);
    ctx.lineCap = 'round';

    // Everything you have revealed so far, dimmed.
    ctx.strokeStyle = 'rgba(120,150,220,.18)';
    ctx.lineWidth = Math.max(0.6, 1.1 * cam.z);
    ctx.beginPath();
    for (const key of S.edges) {
      const u = Math.floor(key / N), v = key % N;
      if (u === S.cur && frontier.has(v)) continue;
      const a = toScreen(nodes[u].wx, nodes[u].wy), b = toScreen(nodes[v].wx, nodes[v].wy);
      if (Math.max(a.x, b.x) < -50 || Math.min(a.x, b.x) > VW + 50) continue;
      if (Math.max(a.y, b.y) < -50 || Math.min(a.y, b.y) > VH + 50) continue;
      const c = edgeCurve(a, b);
      ctx.moveTo(a.x, a.y); ctx.quadraticCurveTo(c.cx, c.cy, b.x, b.y);
    }
    ctx.stroke();

    // The edges you can actually take right now, lit up with arrowheads.
    const pulse = 0.62 + 0.28 * Math.sin(now / 320);
    const a0 = toScreen(nodes[S.cur].wx, nodes[S.cur].wy);
    for (const v of frontier) {
      const b = toScreen(nodes[v].wx, nodes[v].wy);
      const c = edgeCurve(a0, b);
      const isHint = v === S.hint && now < S.hintUntil;
      const isGoal = v === S.goal;
      ctx.strokeStyle = isHint ? `rgba(255,201,63,${pulse})`
        : isGoal ? `rgba(255,75,80,${pulse})`
          : v === hoverNode ? 'rgba(160,240,255,.95)'
            : `rgba(95,227,255,${0.34 + 0.12 * Math.sin(now / 400 + v)})`;
      ctx.lineWidth = (isHint || isGoal || v === hoverNode ? 2.6 : 1.5) * Math.max(0.6, cam.z);
      ctx.beginPath(); ctx.moveTo(a0.x, a0.y); ctx.quadraticCurveTo(c.cx, c.cy, b.x, b.y); ctx.stroke();

      const t = 0.82, mt = 1 - t;
      const px = mt * mt * a0.x + 2 * mt * t * c.cx + t * t * b.x;
      const py = mt * mt * a0.y + 2 * mt * t * c.cy + t * t * b.y;
      const dxb = 2 * mt * (c.cx - a0.x) + 2 * t * (b.x - c.cx);
      const dyb = 2 * mt * (c.cy - a0.y) + 2 * t * (b.y - c.cy);
      const ang = Math.atan2(dyb, dxb), s = 6 * Math.max(0.6, cam.z);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px - Math.cos(ang - 0.45) * s, py - Math.sin(ang - 0.45) * s);
      ctx.moveTo(px, py);
      ctx.lineTo(px - Math.cos(ang + 0.45) * s, py - Math.sin(ang + 0.45) * s);
      ctx.stroke();
    }
  }

  function drawTrail() {
    if (S.path.length < 2) return;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,201,63,.42)';
    ctx.lineWidth = 2.4 * Math.max(0.5, cam.z);
    ctx.setLineDash([7 * cam.z, 6 * cam.z]);
    ctx.beginPath();
    for (let i = 0; i < S.path.length - 1; i++) {
      const a = toScreen(nodes[S.path[i]].wx, nodes[S.path[i]].wy);
      const b = toScreen(nodes[S.path[i + 1]].wx, nodes[S.path[i + 1]].wy);
      const c = edgeCurve(a, b);
      ctx.moveTo(a.x, a.y); ctx.quadraticCurveTo(c.cx, c.cy, b.x, b.y);
    }
    ctx.stroke();
    ctx.restore();
  }

  function drawNodes(now) {
    const frontier = new Set(OUT[S.cur]);
    const labels = [];

    for (let i = 0; i < N; i++) {
      if (!S.seen[i]) continue;
      const n = nodes[i];
      const p = toScreen(n.wx, n.wy);
      const R = Math.max(2.5, n.r * cam.z);
      if (p.x < -R - 60 || p.x > VW + R + 60 || p.y < -R - 60 || p.y > VH + R + 60) continue;

      const isCur = i === S.cur, isGoal = i === S.goal, isHome = i === S.home;
      const isFront = frontier.has(i), isHover = i === hoverNode;
      const isHint = i === S.hint && now < S.hintUntil;

      if (isGoal || isFront || isCur || isHint) {
        const col = isHint ? '255,201,63' : isGoal ? '255,75,80' : isCur ? '255,201,63' : '95,227,255';
        const g = ctx.createRadialGradient(p.x, p.y, R * 0.4, p.x, p.y, R * 3.2);
        g.addColorStop(0, `rgba(${col},${isGoal ? .34 : .22})`);
        g.addColorStop(1, `rgba(${col},0)`);
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(p.x, p.y, R * 3.2, 0, TAU); ctx.fill();
      }

      let fill = 'rgba(40,52,86,.9)', stroke = 'rgba(120,150,215,.45)', lw = 1;
      if (S.visited[i]) { fill = 'rgba(96,78,32,.92)'; stroke = '#ffc93f'; lw = 1.6; }
      if (isFront) { fill = 'rgba(24,64,88,.95)'; stroke = '#5fe3ff'; lw = 1.8; }
      if (isHint) { stroke = '#ffc93f'; lw = 2.6; }
      if (isGoal) { fill = 'rgba(96,22,26,.95)'; stroke = '#ff4b50'; lw = 2.6; }
      if (isHover && isFront) { lw = 3; }

      ctx.fillStyle = fill;
      ctx.beginPath(); ctx.arc(p.x, p.y, R * (isHover && isFront ? 1.18 : 1), 0, TAU); ctx.fill();
      ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke();

      if (isHome) {                                    // home gets a slowly turning star
        ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(now / 2600);
        ctx.strokeStyle = '#ffc93f'; ctx.lineWidth = 1.6;
        ctx.beginPath();
        for (let k = 0; k < 10; k++) {
          const rr = R * (k % 2 ? 1.28 : 1.85), a = k * Math.PI / 5 - Math.PI / 2;
          k ? ctx.lineTo(Math.cos(a) * rr, Math.sin(a) * rr) : ctx.moveTo(Math.cos(a) * rr, Math.sin(a) * rr);
        }
        ctx.closePath(); ctx.stroke(); ctx.restore();
      }

      if (isGoal && !isHome) {                         // pulsing objective ring
        const rr = R * (1.6 + 0.35 * Math.sin(now / 260));
        ctx.strokeStyle = `rgba(255,75,80,${0.7 - 0.3 * Math.sin(now / 260)})`;
        ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(p.x, p.y, rr, 0, TAU); ctx.stroke();
      }

      // Priority decides who keeps their label when two collide.
      const prio = isCur || isGoal ? 4 : isHint ? 3.5 : isFront ? 3 : S.visited[i] ? 2 : n.r > 15 ? 1 : 0;
      if (prio > 0 || S.explore) labels.push({ i, p, R, isFront, isGoal, isCur, prio });
    }

    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    labels.sort((a, b) => b.prio - a.prio || b.R - a.R);
    const taken = [];
    for (const L of labels) {
      const n = nodes[L.i];
      const big = L.prio >= 3;
      const fs = clamp((big ? 13 : 11) * Math.max(0.75, Math.min(cam.z, 1.35)), 9, 17);
      ctx.font = `${big ? 700 : 500} ${fs}px Inter, system-ui, sans-serif`;
      const txt = shortName(n.name);
      const w = ctx.measureText(txt).width, y = L.p.y - L.R - 5;
      const box = { x0: L.p.x - w / 2 - 3, x1: L.p.x + w / 2 + 3, y0: y - fs - 1, y1: y + 2 };
      let hit = false;
      for (const t of taken) {
        if (box.x0 < t.x1 && box.x1 > t.x0 && box.y0 < t.y1 && box.y1 > t.y0) { hit = true; break; }
      }
      if (hit && L.prio < 4) continue;                 // current node and objective always win
      taken.push(box);

      ctx.lineWidth = 3.2; ctx.strokeStyle = 'rgba(4,7,16,.9)';
      ctx.strokeText(txt, L.p.x, y);
      ctx.fillStyle = L.isGoal ? '#ff8b8f' : L.isCur ? '#ffc93f' : L.isFront ? '#cdeeff' : 'rgba(190,205,235,.6)';
      ctx.fillText(txt, L.p.x, y);
    }
  }

  function drawParticles() {
    for (const p of particles) {
      const s = toScreen(p.x, p.y);
      ctx.globalAlpha = clamp(p.life, 0, 1);
      ctx.fillStyle = p.color;
      ctx.beginPath(); ctx.arc(s.x, s.y, p.size * cam.z, 0, TAU); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function drawHeroOnMap(now, hp) {
    const p = toScreen(hp.x, hp.y);
    const sc = HERO_SCALE * cam.z;
    const t = now / 1000;

    if (hp.moving && S.hero.travel === 'swing' && S.moving) {
      const an = toScreen(S.moving.anchor.x, S.moving.anchor.y);
      const hand = window.webHand(S.hero, t, hp.prog);
      const hx = p.x + hand.x * sc * hp.facing, hy = p.y + hand.y * sc;
      ctx.strokeStyle = 'rgba(235,248,255,.85)'; ctx.lineWidth = Math.max(1, 1.4 * cam.z);
      ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(an.x, an.y); ctx.stroke();
      ctx.fillStyle = 'rgba(235,248,255,.9)';
      ctx.beginPath(); ctx.arc(an.x, an.y, 2.6 * cam.z, 0, TAU); ctx.fill();
    }

    if (hp.moving && S.hero.travel === 'run' && S.moving) {
      const a = nodes[S.moving.from], b = nodes[S.moving.to];
      const ang = Math.atan2(b.wy - a.wy, b.wx - a.wx);
      ctx.strokeStyle = 'rgba(242,197,33,.30)'; ctx.lineWidth = 1.6;
      for (let i = 0; i < 5; i++) {
        const off = (i - 2) * 7 * cam.z, L = (22 + i * 9) * cam.z;
        ctx.beginPath();
        ctx.moveTo(p.x - Math.cos(ang) * 9 * cam.z - Math.sin(ang) * off,
          p.y - 22 * cam.z - Math.sin(ang) * 9 * cam.z + Math.cos(ang) * off);
        ctx.lineTo(p.x - Math.cos(ang) * L - Math.sin(ang) * off,
          p.y - 22 * cam.z - Math.sin(ang) * L + Math.cos(ang) * off);
        ctx.stroke();
      }
      if (Math.random() < 0.4) {
        particles.push({ x: hp.x, y: hp.y, vx: (Math.random() - .5) * 40, vy: -Math.random() * 30, life: .6, color: 'rgba(242,197,33,.7)', size: 1.6 });
      }
    }

    if (hp.moving && S.hero.travel === 'float' && Math.random() < 0.6) {
      particles.push({ x: hp.x, y: hp.y - 20, vx: (Math.random() - .5) * 50, vy: (Math.random() - .5) * 50, life: .8, color: '#ffb03a', size: 1.4 });
    }

    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.scale(sc, sc);
    window.drawHero(ctx, S.hero, { time: t, facing: hp.facing, moving: hp.moving, progress: hp.prog });
    ctx.restore();
  }

  /** Edge-of-screen compass. You know WHO you are looking for, never the route. */
  function drawCompass(hp) {
    if (S.finished || S.explore) return;
    const g = nodes[S.goal];
    const p = toScreen(g.wx, g.wy);
    const m = 54;
    const right = viewCx() * 2;
    if (p.x > m && p.x < right - m && p.y > m + 30 && p.y < VH - m) return;

    const cxs = viewCx(), cys = viewCy();
    const ang = Math.atan2(p.y - cys, p.x - cxs);
    const rx = (cxs - m) / Math.abs(Math.cos(ang) || 1e-6);
    const ry = (VH / 2 - m) / Math.abs(Math.sin(ang) || 1e-6);
    const R = Math.min(rx, ry);
    const x = cxs + Math.cos(ang) * R, y = cys + Math.sin(ang) * R;
    const col = S.phase === 1 ? '#ffc93f' : '#ff4b50';

    ctx.save();
    ctx.translate(x, y); ctx.rotate(ang);
    ctx.fillStyle = col; ctx.shadowColor = col; ctx.shadowBlur = 12;
    ctx.beginPath(); ctx.moveTo(13, 0); ctx.lineTo(-9, 8); ctx.lineTo(-5, 0); ctx.lineTo(-9, -8);
    ctx.closePath(); ctx.fill();
    ctx.restore();

    ctx.font = '700 10px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    const lx = clamp(x - Math.cos(ang) * 26, 40, right - 40), ly = clamp(y - Math.sin(ang) * 26, 22, VH - 22);
    ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(4,7,16,.9)';
    const label = shortName(g.name).toUpperCase();
    ctx.strokeText(label, lx, ly); ctx.fillStyle = col; ctx.fillText(label, lx, ly);
  }

  // -------------------------------------------------------------------- input
  let drag = null;

  function nodeAt(sx, sy) {
    let best = -1, bestD = 1e9;
    for (let i = 0; i < N; i++) {
      if (!S.seen[i]) continue;
      const p = toScreen(nodes[i].wx, nodes[i].wy);
      const R = Math.max(9, nodes[i].r * cam.z) + 6;
      const d = Math.hypot(p.x - sx, p.y - sy);
      if (d < R && d < bestD) { best = i; bestD = d; }
    }
    return best;
  }

  cv.addEventListener('pointerdown', (e) => {
    cv.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, y: e.clientY, ox: cam.tx, oy: cam.ty, moved: 0 };
    cv.classList.add('grabbing');
  });

  cv.addEventListener('pointermove', (e) => {
    const r = cv.getBoundingClientRect();
    const sx = e.clientX - r.left, sy = e.clientY - r.top;
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.moved += Math.abs(dx) + Math.abs(dy);
      if (drag.moved > 5) {
        cam.free = true;
        cam.tx = drag.ox - dx / cam.z; cam.ty = drag.oy - dy / cam.z;
        cam.x = cam.tx; cam.y = cam.ty;
      }
    } else {
      hoverNode = nodeAt(sx, sy);
      cv.classList.toggle('pointing', hoverNode >= 0 && !S.explore && OUT[S.cur].includes(hoverNode));
      renderLinks(true);
    }
  });

  cv.addEventListener('pointerup', (e) => {
    cv.classList.remove('grabbing');
    const wasDrag = drag && drag.moved > 5;
    drag = null;
    if (wasDrag || S.explore) return;
    const r = cv.getBoundingClientRect();
    const i = nodeAt(e.clientX - r.left, e.clientY - r.top);
    if (i >= 0) travelTo(i);
  });

  cv.addEventListener('pointerleave', () => { hoverNode = -1; cv.classList.remove('grabbing'); });

  cv.addEventListener('wheel', (e) => {
    e.preventDefault();
    cam.tz = clamp(cam.tz * Math.exp(-e.deltaY * 0.0016), 0.22, 2.6);
  }, { passive: false });

  window.addEventListener('keydown', (e) => {
    if (!$('vStart').classList.contains('gone')) { if (e.key === 'Enter') $('bStart').click(); return; }
    if (e.key === 'c' || e.key === 'C') { cam.free = false; cam.tz = 1.05; }
    if (e.key === 'm' || e.key === 'M') $('bSound').click();
    if (e.key === 'z' || e.key === 'Z') webZip();
    if (e.key === 'h' || e.key === 'H') spiderSense();
    if (e.key === '+' || e.key === '=') cam.tz = clamp(cam.tz * 1.2, 0.22, 2.6);
    if (e.key === '-') cam.tz = clamp(cam.tz / 1.2, 0.22, 2.6);
    const n = parseInt(e.key, 10);
    if (n >= 1 && n <= 9 && !S.explore) { const v = OUT[S.cur][n - 1]; if (v !== undefined) travelTo(v); }
  });

  // ----------------------------------------------------------------------- UI
  let toastTimer = null;
  function toast(html, ms) {
    const t = $('toast');
    t.innerHTML = html; t.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), ms || 2800);
  }

  function fmtTime(s) { return Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0'); }

  function syncUI() {
    const n = nodes[S.cur];
    $('sSteps').textContent = S.steps;
    $('sPar').textContent = S.par;
    $('sSeen').textContent = S.seen.reduce((a, b) => a + b, 0);
    $('hName').textContent = shortName(n.name);
    $('hDesc').textContent = n.desc || 'No description in the dataset.';
    $('hIn').textContent = n.inDeg + ' in';
    $('hOut').textContent = n.outDeg + ' out';
    $('hStep').textContent = 'step ' + S.steps;
    $('hWiki').href = n.url;
    $('senseN').textContent = S.senses;
    $('bSense').disabled = S.senses <= 0;
    $('bBack').disabled = S.path.length < 2;

    const obj = $('objective');
    obj.classList.toggle('home', S.phase === 1);
    $('objText').textContent = S.phase === 0
      ? 'FIND ' + shortName(nodes[S.target].name).toUpperCase()
      : 'GET HOME TO ' + shortName(nodes[S.home].name).toUpperCase();
    renderLinks();
  }

  function renderLinks(hoverOnly) {
    const box = $('links');
    if (hoverOnly) {
      [...box.children].forEach(el => el.classList.toggle('hl', +el.dataset.i === hoverNode));
      return;
    }
    const outs = OUT[S.cur].slice().sort((a, b) => nodes[b].inDeg - nodes[a].inDeg);
    $('linkCount').textContent = outs.length;
    box.innerHTML = '';
    if (!outs.length) {
      box.innerHTML = '<div class="deadend"><b>Dead end.</b> This article links to none of the other 302 characters. ' +
        'Use <b>Web-zip</b> to step back.</div>';
      return;
    }
    outs.forEach((v) => {
      const n = nodes[v];
      const b = document.createElement('button');
      b.className = 'link' + (S.visited[v] ? ' visited' : '') + (v === S.goal ? ' isgoal' : '') +
        (v === S.hint && performance.now() < S.hintUntil ? ' hint' : '');
      b.dataset.i = v;
      b.innerHTML = `<span class="dot"></span><span class="nm">${shortName(n.name)}</span>` +
        `<span class="dg">${n.inDeg}&darr; ${n.outDeg}&uarr;</span>`;
      b.title = n.desc || '';
      b.onclick = () => travelTo(v);
      b.onmouseenter = () => { hoverNode = v; };
      b.onmouseleave = () => { hoverNode = -1; };
      box.appendChild(b);
    });
  }

  // ------------------------------------------------------------- start screen
  let chosen = HEROES[0], chosenDiff = 'normal';
  let previewCanvases = [];

  function buildStart() {
    const wrap = $('heroPick');
    wrap.innerHTML = '';
    previewCanvases = [];
    HEROES.forEach((h) => {
      const i = byId.get(h.node);
      const card = document.createElement('button');
      card.className = 'heroCard' + (h === chosen ? ' sel' : '');
      if (h === chosen) card.style.borderColor = h.tint;
      card.innerHTML =
        '<canvas width="200" height="240"></canvas>' +
        `<div class="hn comic" style="color:${h.tint}">${h.label}</div>` +
        `<div class="hr">${h.role}</div>` +
        `<div class="hd">${h.blurb}</div>` +
        `<div class="deg"><span class="chip in">${IN[i].length} in</span><span class="chip out">${OUT[i].length} out</span></div>`;
      card.onclick = () => { chosen = h; Sfx.boot(); Sfx.tone(660, 0.12, 'triangle', 0.04); buildStart(); };
      wrap.appendChild(card);
      previewCanvases.push({ cv: card.querySelector('canvas'), hero: h });
    });

    const dwrap = $('diffPick');
    dwrap.innerHTML = '';
    Object.entries(DIFFS).forEach(([key, d]) => {
      const b = document.createElement('button');
      b.className = 'diff' + (key === chosenDiff ? ' sel' : '');
      b.innerHTML = `${d.label}<small>${d.note}</small>`;
      b.onclick = () => { chosenDiff = key; buildStart(); };
      dwrap.appendChild(b);
    });
  }

  function drawPreviews(now) {
    const t = now / 1000;
    for (const p of previewCanvases) {
      if (!p.cv.isConnected) continue;
      const c = p.cv.getContext('2d');
      c.setTransform(1, 0, 0, 1, 0, 0);
      c.clearRect(0, 0, p.cv.width, p.cv.height);
      c.save();
      c.translate(p.cv.width / 2, p.cv.height - 26);
      c.scale(4.4, 4.4);
      window.drawHero(c, p.hero, { time: t, facing: 1, moving: true, progress: (t * 0.4) % 1 });
      c.restore();
    }
  }

  function showStart() {
    buildStart();
    $('vStart').classList.remove('gone');
    $('vWin').classList.add('gone');
    S.running = false;
  }

  function revealAll() {
    for (let i = 0; i < N; i++) { S.seen[i] = 1; for (const v of OUT[i]) S.edges.add(i * N + v); }
    S.explore = true;
    cam.free = true; cam.tx = WORLD / 2; cam.ty = WORLD / 2; cam.tz = 0.28;
    $('vWin').classList.add('gone');
    toast('The whole network: 303 nodes, 1784 edges. The ring on the outside is the 17 isolates plus the small islands ' +
      'outside the giant component.', 7000);
    syncUI();
  }

  // ------------------------------------------------------------------ buttons
  $('bStart').onclick = () => { Sfx.boot(); $('vStart').classList.add('gone'); resize(); newRound(chosen, chosenDiff); };
  $('bAgain').onclick = () => { $('vWin').classList.add('gone'); newRound(chosen, chosenDiff); };
  $('bMenu').onclick = showStart;
  $('bMenu2').onclick = showStart;
  $('bExplore').onclick = revealAll;
  $('bSense').onclick = spiderSense;
  $('bBack').onclick = webZip;
  $('bCenter').onclick = () => { cam.free = false; cam.tz = 1.05; S.explore = false; };
  $('bSound').onclick = () => { Sfx.on = !Sfx.on; $('bSound').classList.toggle('on', Sfx.on); Sfx.boot(); };
  $('panelToggle').onclick = () => {
    const p = $('panel'), t = $('panelToggle');
    p.classList.toggle('hidden');
    const hid = p.classList.contains('hidden');
    t.classList.toggle('closed', hid);
    t.innerHTML = hid ? '&#9654;' : '&#9664;';
    document.documentElement.style.setProperty('--panelw', hid ? '0px' : '');
  };

  // --------------------------------------------------------------- main loop
  let last = performance.now();
  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.05); last = now;
    if (!$('vStart').classList.contains('gone')) drawPreviews(now);
    else {
      draw(now, dt);
      if (S.running && !S.finished) $('sTime').textContent = fmtTime((now - S.t0) / 1000);
    }
    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------------------ startup
  resize();
  S.hero = HEROES[0];
  S.home = S.cur = S.goal = S.target = byId.get(HEROES[0].node);
  showStart();
  requestAnimationFrame(frame);
})();
