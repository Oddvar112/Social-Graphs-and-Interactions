/* ===========================================================================
   KINGPIN: shatter the Marvel graph.

   Data: the course's frozen week-1 snapshot (303 nodes, 1784 directed edges),
   played UNDIRECTED, the way the week-3 course page treats betweenness and
   fragmentation. All graph algorithms live in core.js, which the test suite
   runs under node and diffs against networkx.

   The idea: you get a fixed number of hits. Each one removes a character and
   every link through them; whoever loses their last route to the big cluster
   is cut off and stops counting. When the hits run out, three rival hitmen
   run the same contract on the same board (degree-greedy, betweenness-greedy,
   random x30), so the score screen is one long "compared to what". The map
   shows degree. It does not show betweenness. That gap is the game.
   =========================================================================== */

(function () {
  'use strict';

  const D = window.MARVEL_DATA;
  if (!D) {
    document.body.innerHTML = '<p style="padding:40px;font:16px sans-serif;color:#fff">' +
      'data/marvel_week1.js is missing. Run <code>python scripts/build_graph.py</code>.</p>';
    return;
  }

  const Core = window.KingpinCore;
  const { N, adj } = Core.buildArena(D);
  const WORLD = 3000;
  const TAU = Math.PI * 2;

  // ------------------------------------------------------------ intact graph
  const c0 = Core.components(N, adj, null);
  const inG0 = new Uint8Array(N);                 // the 277-strong core
  for (let i = 0; i < N; i++) if (c0.comp[i] === c0.giantId) inG0[i] = 1;
  const G0SIZE = c0.giantSize;

  const deg0 = Core.aliveDegrees(N, adj, null);
  const btw0 = Core.brandes(N, adj, null);
  const BTW_NORM = (G0SIZE - 1) * (G0SIZE - 2) / 2;

  /** Rank within the intact core, 1 = biggest, ties to the lowest index. */
  function ranksOf(score) {
    const members = [];
    for (let i = 0; i < N; i++) if (inG0[i]) members.push(i);
    members.sort((a, b) => score[b] - score[a] || a - b);
    const rank = new Int32Array(N).fill(-1);
    members.forEach((m, r) => { rank[m] = r + 1; });
    return rank;
  }
  const degRank = ranksOf(deg0);
  const btwRank = ranksOf(btw0);
  const brokerTop10 = (() => {
    const m = [];
    for (let i = 0; i < N; i++) if (inG0[i]) m.push(i);
    m.sort((a, b) => btw0[b] - btw0[a] || a - b);
    return new Set(m.slice(0, 10));
  })();

  const nodes = D.nodes.map((n, i) => ({
    ...n, idx: i,
    wx: n.x * WORLD, wy: n.y * WORLD,
    deg: deg0[i],
    r: 6 + 2.2 * Math.sqrt(deg0[i]),
  }));

  // ------------------------------------------------------------------ helpers
  const $ = (id) => document.getElementById(id);
  const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
  const lerp = (a, b, t) => a + (b - a) * t;
  const shortName = (s) => s.replace(/\s*\([^)]*\)\s*$/, '');

  // -------------------------------------------------------------------- audio
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
      const f = this.ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = 900;
      const g = this.ctx.createGain(); g.gain.value = gain || 0.06;
      s.connect(f).connect(g).connect(this.ctx.destination); s.start(t);
    },
    boom() { this.noise(0.3, 0.09); this.tone(95, 0.5, 'sine', 0.09, 42); this.tone(46, 0.7, 'sine', 0.07, 30); },
    crack() { this.tone(1900, 0.06, 'square', 0.02); setTimeout(() => this.tone(620, 0.14, 'triangle', 0.035, 300), 40); },
    laser() { this.tone(2200, 0.16, 'sawtooth', 0.016, 3600); },
    heart() { this.tone(58, 0.12, 'sine', 0.10, 40); setTimeout(() => this.tone(52, 0.16, 'sine', 0.08, 36), 150); },
    buzz() { this.tone(150, 0.22, 'square', 0.035, 90); },
    fanfare(good) {
      const seq = good ? [523, 659, 784, 1046] : [330, 262, 208, 165];
      seq.forEach((f, i) => setTimeout(() => this.tone(f, 0.32, 'triangle', 0.055), i * 130));
    },
  };

  // -------------------------------------------------------------------- state
  const CONTRACTS = {
    3: { label: 'A warning', note: '3 hits' },
    5: { label: 'A message', note: '5 hits' },
    8: { label: 'The purge', note: '8 hits' },
  };

  const S = {
    budget: 8, alive: null, comp: null, giantId: -1, coreSize: G0SIZE,
    inCore: null, hits: [], curve: [],
    running: false, finished: false, reveal: false,
  };

  const cam = { x: WORLD / 2, y: WORLD / 2, z: 0.3, tx: WORLD / 2, ty: WORLD / 2, tz: 0.3 };
  let particles = [];
  let hoverNode = -1;
  let shake = { mag: 0, until: 0 };
  const killedAt = new Float64Array(N);           // performance.now() of each death, for the ✗ animation
  let splashes = [];                              // comic onomatopoeia, world-anchored
  let shot = null;                                // the in-flight sniper sequence
  let flashUntil = 0;                             // red vignette after a kill
  let severFx = new Map();                        // node -> drift-away animation
  let nextBeat = 0;                               // heartbeat timer on the last bullet
  let NOW = performance.now();                    // frame clock for posOf()

  /** Where a node sits right now: severed nodes drift away from the shot. */
  function posOf(i) {
    const n = nodes[i];
    const fx = severFx.get(i);
    if (!fx) return { x: n.wx, y: n.wy };
    const t = Math.min(1, (NOW - fx.t0) / 1200);
    const e = 1 - Math.pow(1 - t, 3);
    const wob = Math.sin(NOW / 900 + fx.ph) * 4 * e;
    return { x: n.wx + fx.dx * e + wob, y: n.wy + fx.dy * e + wob * 0.6 };
  }

  function addSplash(wx, wy, txt, color, size, delay) {
    splashes.push({
      x: wx, y: wy, txt, color, size,
      t0: performance.now() + (delay || 0), dur: 1400,
      rot: (Math.random() - 0.5) * 0.3,
    });
  }

  // The bounding box of the core, for fitting the camera.
  const bbox = (() => {
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (const n of nodes) {
      if (!inG0[n.idx]) continue;
      x0 = Math.min(x0, n.wx); y0 = Math.min(y0, n.wy);
      x1 = Math.max(x1, n.wx); y1 = Math.max(y1, n.wy);
    }
    return { x0, y0, x1, y1 };
  })();

  // Undirected edge list (u < v), split by whether both ends sit in the core.
  const edges = (() => {
    const out = [];
    for (let u = 0; u < N; u++) for (const v of adj[u]) if (u < v) out.push([u, v]);
    return out;
  })();

  // ---------------------------------------------------------------- recompute
  function recompute() {
    S.comp = Core.components(N, adj, S.alive);
    S.giantId = S.comp.giantId;
    S.coreSize = S.comp.giantSize;
    const inCore = new Uint8Array(N);
    for (let i = 0; i < N; i++) if (S.alive[i] && S.comp.comp[i] === S.giantId) inCore[i] = 1;
    S.inCore = inCore;
  }

  // -------------------------------------------------------------- round setup
  function newRound(budget) {
    S.budget = budget;
    S.alive = new Uint8Array(N).fill(1);
    S.hits = []; S.curve = [G0SIZE];
    S.running = true; S.finished = false; S.reveal = false;
    killedAt.fill(0);
    recompute();
    particles = []; splashes = []; shot = null; severFx = new Map(); flashUntil = 0;
    fitCamera();
    // Fly in: start the camera pulled back and let the lerp carry it home.
    cam.x = cam.tx; cam.y = cam.ty; cam.z = cam.tz * 0.45;
    syncUI();
    renderWanted();
    renderHits();
    setAim(-1);
    toast(`Contract of <b>${budget}</b>: click any purple character to fire. Your score is <b>CORE</b>, top right &mdash; make it as small as you can.`, 6000);
  }

  function fitCamera() {
    const vw = Math.max(200, VW - panelWidth()), vh = Math.max(200, VH);
    const spanX = bbox.x1 - bbox.x0 + 300, spanY = bbox.y1 - bbox.y0 + 300;
    cam.tz = clamp(Math.min(vw / spanX, vh / spanY), 0.12, 2.6);
    cam.tx = (bbox.x0 + bbox.x1) / 2;
    cam.ty = (bbox.y0 + bbox.y1) / 2;
  }

  // --------------------------------------------------------------------- hits
  function hit(i) {
    if (!S.running || S.finished || shot) return;
    if (!S.alive[i]) return;
    if (!inG0[i]) {
      Sfx.buzz();
      toast(`<b>${shortName(nodes[i].name)}</b> was never part of the core. No contract on them.`);
      return;
    }
    if (!S.inCore[i]) {
      Sfx.buzz();
      toast(`<b>${shortName(nodes[i].name)}</b> is already cut off. Don't waste a bullet.`);
      return;
    }
    // Lock on, then fire: the kill itself lands when the laser reaches them.
    shot = { target: i, t0: performance.now(), dur: 300, fired: false };
    Sfx.boot(); Sfx.laser();
  }

  function applyHit(i) {
    const before = S.coreSize;
    const wasCore = S.inCore;
    S.alive[i] = 0;
    killedAt[i] = performance.now();
    recompute();

    const drop = before - S.coreSize;              // includes the victim
    S.hits.push({ i, drop });
    S.curve.push(S.coreSize);

    Sfx.boom();
    burst(nodes[i].wx, nodes[i].wy, '#ff4b50', 34);
    shake = { mag: 11, until: performance.now() + 420 };
    flashUntil = performance.now() + 320;
    cam.z = Math.min(cam.z * 1.05, cam.tz * 1.25);   // zoom punch; the lerp settles it back
    const WORDS = ['BLAM!', 'BAM!', 'POW!', 'KRAK!', 'WHAM!'];
    addSplash(nodes[i].wx, nodes[i].wy - 26, WORDS[Math.floor(Math.random() * WORDS.length)], '#ffd94a', 95);

    // Everyone newly severed drifts away from the shot, so a broker hit LOOKS different.
    let severed = 0, cx = 0, cy = 0;
    for (let v = 0; v < N; v++) {
      if (S.alive[v] && wasCore[v] && !S.inCore[v]) {
        severed++;
        cx += nodes[v].wx; cy += nodes[v].wy;
        burst(nodes[v].wx, nodes[v].wy, 'rgba(150,160,180,.8)', 10);
        let dx = nodes[v].wx - nodes[i].wx, dy = nodes[v].wy - nodes[i].wy;
        const L = Math.hypot(dx, dy) || 1;
        severFx.set(v, {
          t0: performance.now() + 150,
          dx: dx / L * (34 + Math.random() * 22), dy: dy / L * (34 + Math.random() * 22),
          ph: Math.random() * TAU,
        });
      }
    }
    if (severed > 0) {
      setTimeout(() => Sfx.crack(), 120);
      addSplash(cx / severed, cy / severed + 40, `×${severed} CUT OFF`, '#aebbdd', 52, 380);
    }

    const tail = severed > 0
      ? `took <b>${severed}</b> down with ${severed === 1 ? 'him' : 'them'}. Core: <b>${S.coreSize}</b>.`
      : `nobody else lost their route. Core: <b>${S.coreSize}</b>.`;
    toast(`<b>${shortName(nodes[i].name)}</b> eliminated &mdash; ${tail}`, 3600);

    syncUI();
    renderWanted();
    renderHits();
    setAim(-1);

    if (S.hits.length >= S.budget) {
      S.running = false;
      setTimeout(finish, 1100);
    }
  }

  // ------------------------------------------------------------------- finish
  function finish() {
    S.finished = true;

    // The rivals run the same contract on the same board.
    const hub = Core.attack(N, adj, S.budget, 'degree');
    const broker = Core.attack(N, adj, S.budget, 'betweenness');
    const rnd = Core.randomAttackAvg(N, adj, S.budget, 30, 7);
    const rec = Core.RECORDS[S.budget];

    const you = S.coreSize;
    const hubF = hub.curve[S.budget], brokerF = broker.curve[S.budget];
    const rndF = rnd.curve[S.budget];
    const bestBot = Math.min(hubF, brokerF), worstBot = Math.max(hubF, brokerF);

    let verdict, line;
    if (rec && you <= rec.giant) {
      verdict = 'UNDERWORLD LEGEND';
      line = 'You matched the best hit-list our search ever found. Both bots are greedy: they shoot the biggest ' +
        'number and never plan two hits ahead. You just did, and that is the only way to beat them here.';
    } else if (you < bestBot) {
      verdict = 'COLD-BLOODED';
      line = 'You beat both machines. Greedy centrality, degree or betweenness, sheds leaves one hub at a time; ' +
        'you found hits that work together. The record below shows how deep that idea goes.';
    } else if (you <= worstBot) {
      verdict = 'PROFESSIONAL HIT';
      line = 'You matched the machines. On this network the hub bot and the broker bot nearly tie, because for the ' +
        'top names betweenness IS degree (the week-3 lesson: Spider-Man is exactly as between as his 106 links predict).';
    } else if (you < rndF) {
      verdict = 'HIRED MUSCLE';
      line = 'Better than chance, worse than the machines. The big names shed the most, but notice how little even ' +
        'they shed: this core barely shatters, and that robustness is a real finding about the network.';
    } else {
      verdict = 'AMATEUR HOUR';
      line = 'A random hitman does this well on average. Almost every character you can shoot has a path around them: ' +
        'that is what a giant component with a heavy-tailed degree distribution buys.';
    }

    $('wVerdict').textContent = verdict;
    $('wLine').textContent = line;
    $('wYou').textContent = you;
    $('wHub').textContent = hubF;
    $('wBroker').textContent = brokerF;
    $('wRandom').textContent = rndF.toFixed(1);
    $('wRecord').textContent = rec ? rec.giant : '?';
    $('wYou').parentElement.classList.toggle('good', you <= bestBot);

    const rows = S.hits.map((h, k) => {
      const n = nodes[h.i];
      const gone = h.drop - 1;
      return `${k + 1}. <em>${shortName(n.name)}</em> &mdash; hub #${degRank[h.i]} &middot; broker #${btwRank[h.i]}` +
        ` &middot; <span class="m">${gone > 0 ? `cut off ${gone}` : 'cut off nobody'}</span>`;
    });
    let recLine = '';
    if (rec) {
      const names = rec.ids.map(i => shortName(nodes[i].name)).join(', ');
      recLine = `<span class="k" style="margin-top:9px">The record crew &middot; core ${rec.giant}</span>` +
        `<em>${names}</em> &mdash; found by simulated annealing, not by any single centrality. ` +
        `Black Widow is hub #${degRank[26]} yet on every record list: she is the only door to her corner of the map.`;
    }
    $('wReport').innerHTML = `<span class="k">Your hit-list</span>` + rows.join('<br>') + '<br>' + recLine;

    Sfx.fanfare(you <= worstBot);
    $('vWin').classList.remove('gone');
    // Count the numbers down from 277 while the chart draws itself in. The
    // canvas has no size until the veil is laid out, hence the first rAF.
    const vals = [
      ['wYou', you, 0], ['wHub', hubF, 0], ['wBroker', brokerF, 0],
      ['wRandom', rndF, 1], ['wRecord', rec ? rec.giant : you, 0],
    ];
    requestAnimationFrame(() => {
      const t0 = performance.now();
      (function anim(now) {
        const p = Math.min(1, (now - t0) / 1100);
        const e = 1 - Math.pow(1 - p, 3);
        for (const [id, v, dec] of vals) $(id).textContent = (G0SIZE + (v - G0SIZE) * e).toFixed(dec);
        drawChart(hub.curve, broker.curve, rnd.curve, rec ? rec.giant : null, e);
        if (p < 1) requestAnimationFrame(anim);
      })(t0);
    });
    syncUI();
  }

  // -------------------------------------------------------------- score chart
  function drawChart(hubC, brokerC, rndC, recG, prog = 1) {
    const cvC = $('chart');
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = cvC.getBoundingClientRect();
    cvC.width = Math.round(r.width * dpr); cvC.height = Math.round(r.height * dpr);
    const g = cvC.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);

    const W = r.width, H = r.height, padL = 34, padR = 10, padT = 8, padB = 20;
    const B = S.budget;
    let lo = Math.min(...S.curve, ...hubC, ...brokerC, ...rndC);
    if (recG != null) lo = Math.min(lo, recG);
    lo = Math.floor(lo - 2); const hi = G0SIZE;
    const X = (i) => padL + (W - padL - padR) * (i / B);
    const Y = (v) => padT + (H - padT - padB) * (1 - (v - lo) / (hi - lo));

    g.strokeStyle = 'rgba(160,145,196,.18)'; g.fillStyle = 'rgba(160,145,196,.75)';
    g.font = '600 9px Inter, sans-serif'; g.textAlign = 'right'; g.textBaseline = 'middle';
    const step = Math.max(2, Math.ceil((hi - lo) / 5 / 2) * 2);
    for (let v = hi; v >= lo; v -= step) {
      g.beginPath(); g.moveTo(padL, Y(v)); g.lineTo(W - padR, Y(v)); g.stroke();
      g.fillText(String(v), padL - 5, Y(v));
    }
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (let i = 0; i <= B; i++) g.fillText(String(i), X(i), H - padB + 5);
    g.fillText('hits', W - padR - 12, H - padB + 5);

    // Draws the first `prog` of a curve, interpolating the last partial segment.
    function line(curve, color, dash, width) {
      const upto = (curve.length - 1) * prog;
      g.strokeStyle = color; g.lineWidth = width || 2; g.setLineDash(dash || []);
      g.beginPath();
      g.moveTo(X(0), Y(curve[0]));
      for (let i = 1; i <= Math.floor(upto); i++) g.lineTo(X(i), Y(curve[i]));
      const f = upto - Math.floor(upto);
      if (f > 0) {
        const i = Math.floor(upto);
        g.lineTo(X(upto), Y(curve[i] + (curve[i + 1] - curve[i]) * f));
      }
      g.stroke(); g.setLineDash([]);
    }
    // Your curve first and fat, the bots on top of it, so that when the runs
    // coincide (they often do) the bot lines visibly ride inside yours.
    if (recG != null) line(new Array(B + 1).fill(recG), 'rgba(74,222,128,.7)', [4, 4], 1.5);
    line(rndC, '#8a93a8', [2, 3], 1.5);
    line(S.curve, '#ffc93f', [], 4);
    line(hubC, '#5fe3ff', [], 1.6);
    line(brokerC, '#ff4b50', [7, 5], 1.6);

    g.fillStyle = '#ffc93f';
    S.curve.forEach((v, i) => {
      if (i > (S.curve.length - 1) * prog) return;
      g.beginPath(); g.arc(X(i), Y(v), 3, 0, TAU); g.fill();
    });
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

  function panelWidth() {
    const p = $('panel');
    return p.classList.contains('hidden') ? 0 : Math.min(p.getBoundingClientRect().width, VW * 0.5);
  }
  const viewCx = () => (VW - panelWidth()) / 2;
  const viewCy = () => VH / 2;
  const toScreen = (wx, wy) => ({ x: (wx - cam.x) * cam.z + viewCx(), y: (wy - cam.y) * cam.z + viewCy() });

  // Fixed dust in world space, same trick as WEB-CRAWLER.
  const dust = (() => {
    let s = 54321;
    const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    return Array.from({ length: 420 }, () => ({
      x: rnd() * WORLD * 1.6 - WORLD * 0.3, y: rnd() * WORLD * 1.6 - WORLD * 0.3,
      r: 0.5 + rnd() * 1.5, a: 0.04 + rnd() * 0.18,
    }));
  })();

  /** In reveal mode nodes are re-sized by intact betweenness, not degree. */
  function nodeRadius(i) {
    if (!S.reveal) return nodes[i].r;
    return 5 + 68 * Math.sqrt(btw0[i] / BTW_NORM);
  }

  function draw(now, dt) {
    NOW = now;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, VW, VH);

    if (shot && !shot.fired && now - shot.t0 >= shot.dur) {
      shot.fired = true;
      const t = shot.target;
      shot = null;
      applyHit(t);
    }
    if (S.running && !S.finished && S.budget - S.hits.length === 1 && now > nextBeat) {
      Sfx.heart();
      nextBeat = now + 1300;
    }

    const k = 1 - Math.pow(0.0016, dt);
    cam.x = lerp(cam.x, cam.tx, k); cam.y = lerp(cam.y, cam.ty, k); cam.z = lerp(cam.z, cam.tz, k);

    let sx = 0, sy = 0;
    if (now < shake.until) {
      const decay = (shake.until - now) / 420;
      sx = (Math.random() - 0.5) * shake.mag * decay;
      sy = (Math.random() - 0.5) * shake.mag * decay;
    }
    ctx.save();
    ctx.translate(sx, sy);

    for (const d of dust) {
      const p = toScreen(d.x, d.y);
      if (p.x < -20 || p.x > VW + 20 || p.y < -20 || p.y > VH + 20) continue;
      ctx.fillStyle = `rgba(180,160,255,${d.a})`;
      ctx.beginPath(); ctx.arc(p.x, p.y, d.r, 0, TAU); ctx.fill();
    }

    drawEdges();
    drawNodes(now);
    drawParticles();
    drawShot(now);
    drawSplashes(now);
    ctx.restore();
    drawVignette(now);

    stepParticles(dt);
  }

  /** The sniper sequence: a laser from offscreen and a lock-on ring closing in. */
  function drawShot(now) {
    if (!shot) return;
    const t = clamp((now - shot.t0) / shot.dur, 0, 1);
    const n = posOf(shot.target);
    const p = toScreen(n.x, n.y);
    // The muzzle sits offscreen, on the far side of the view centre.
    const dx = p.x - viewCx(), dy = p.y - viewCy();
    const L = Math.hypot(dx, dy) || 1;
    const reach = Math.max(VW, VH);
    const mx = p.x + dx / L * reach, my = p.y + dy / L * reach;
    ctx.lineCap = 'round';
    ctx.strokeStyle = `rgba(255,60,60,${0.10 + 0.25 * t})`;
    ctx.lineWidth = 6;
    ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(p.x, p.y); ctx.stroke();
    ctx.strokeStyle = `rgba(255,120,120,${0.35 + 0.55 * t})`;
    ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(p.x, p.y); ctx.stroke();

    const R = Math.max(9, nodeRadius(shot.target) * cam.z);
    const ring = R + (1 - t) * 46;
    ctx.save();
    ctx.translate(p.x, p.y); ctx.rotate(now / 240);
    ctx.strokeStyle = `rgba(255,201,63,${0.5 + 0.5 * t})`;
    ctx.lineWidth = 2;
    ctx.setLineDash([ring * 0.5, ring * 0.55]);
    ctx.beginPath(); ctx.arc(0, 0, ring, 0, TAU); ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
    if (t > 0.85) {                                 // muzzle flash on impact
      ctx.fillStyle = `rgba(255,240,220,${(t - 0.85) / 0.15 * 0.85})`;
      ctx.beginPath(); ctx.arc(p.x, p.y, R * 1.7, 0, TAU); ctx.fill();
    }
  }

  function drawSplashes(now) {
    splashes = splashes.filter(s => now < s.t0 + s.dur);
    for (const s of splashes) {
      if (now < s.t0) continue;
      const t = (now - s.t0) / s.dur;
      const pop = t < 0.22 ? backOut(t / 0.22) : 1;
      const alpha = t > 0.68 ? 1 - (t - 0.68) / 0.32 : 1;
      const p = toScreen(s.x, s.y);
      const fs = Math.max(14, s.size * cam.z * pop);
      ctx.save();
      ctx.translate(p.x, p.y - t * 26 * cam.z);
      ctx.rotate(s.rot);
      ctx.globalAlpha = alpha;
      ctx.font = `${fs}px Bangers, Impact, sans-serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.lineWidth = Math.max(3, fs * 0.14); ctx.strokeStyle = 'rgba(10,6,14,.95)';
      ctx.strokeText(s.txt, 0, 0);
      ctx.fillStyle = s.color;
      ctx.fillText(s.txt, 0, 0);
      ctx.restore();
      ctx.globalAlpha = 1;
    }
  }
  const backOut = (t) => { const c = 1.70158; t -= 1; return t * t * ((c + 1) * t + c) + 1; };

  /** Red edges of the screen: a pulse on every kill, a slow burn on the last bullet. */
  function drawVignette(now) {
    let a = 0;
    if (now < flashUntil) a = (flashUntil - now) / 320 * 0.30;
    if (S.running && !S.finished && S.budget - S.hits.length === 1) {
      a = Math.max(a, 0.09 + 0.05 * Math.sin(now / 280));
    }
    if (a <= 0.004) return;
    const g = ctx.createRadialGradient(VW / 2, VH / 2, Math.min(VW, VH) * 0.35, VW / 2, VH / 2, Math.max(VW, VH) * 0.75);
    g.addColorStop(0, 'rgba(230,36,41,0)');
    g.addColorStop(1, `rgba(230,36,41,${a})`);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, VW, VH);
  }

  function drawEdges() {
    ctx.lineCap = 'round';
    // Two batches: living core edges, then everything severed or outside.
    ctx.lineWidth = Math.max(0.8, 1.0 * cam.z);
    for (const pass of [0, 1]) {
      ctx.strokeStyle = pass === 0 ? 'rgba(166,120,255,.32)' : 'rgba(140,150,170,.12)';
      ctx.beginPath();
      for (const [u, v] of edges) {
        if (!S.alive[u] || !S.alive[v]) continue;
        const core = !!(S.inCore[u] && S.inCore[v]);
        if ((pass === 0) !== core) continue;
        const pu = posOf(u), pv = posOf(v);
        const a = toScreen(pu.x, pu.y), b = toScreen(pv.x, pv.y);
        if (Math.max(a.x, b.x) < -50 || Math.min(a.x, b.x) > VW + 50) continue;
        if (Math.max(a.y, b.y) < -50 || Math.min(a.y, b.y) > VH + 50) continue;
        ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
      }
      ctx.stroke();
    }
    // Edges of the character under the crosshair, lit.
    if (hoverNode >= 0 && S.alive[hoverNode]) {
      ctx.strokeStyle = 'rgba(255,201,63,.55)';
      ctx.lineWidth = Math.max(0.8, 1.6 * cam.z);
      ctx.beginPath();
      const ph = posOf(hoverNode);
      const a = toScreen(ph.x, ph.y);
      for (const v of adj[hoverNode]) {
        if (!S.alive[v]) continue;
        const pv = posOf(v);
        const b = toScreen(pv.x, pv.y);
        ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
      }
      ctx.stroke();
    }
  }

  function drawNodes(now) {
    const labels = [];

    for (let i = 0; i < N; i++) {
      const n = nodes[i];
      const w = posOf(i);
      const p = toScreen(w.x, w.y);
      const R = Math.max(2, nodeRadius(i) * cam.z);
      if (p.x < -R - 60 || p.x > VW + R + 60 || p.y < -R - 60 || p.y > VH + R + 60) continue;

      if (!S.alive[i]) {                          // the fallen: a fading red ✗
        const age = (now - killedAt[i]) / 1000;
        const ring = Math.min(age * 3, 1);
        if (ring < 1) {
          ctx.strokeStyle = `rgba(255,75,80,${(1 - ring) * 0.8})`;
          ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(p.x, p.y, R * (1 + ring * 3), 0, TAU); ctx.stroke();
        }
        // Crime-scene tape: a faint dashed ring that stays on the map.
        ctx.strokeStyle = 'rgba(255,201,63,.28)';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.arc(p.x, p.y, R * 1.55, 0, TAU); ctx.stroke();
        ctx.setLineDash([]);
        const s = Math.max(3, R * 0.7);
        ctx.strokeStyle = 'rgba(255,75,80,.85)';
        ctx.lineWidth = Math.max(1.4, 2.2 * cam.z);
        ctx.beginPath();
        ctx.moveTo(p.x - s, p.y - s); ctx.lineTo(p.x + s, p.y + s);
        ctx.moveTo(p.x + s, p.y - s); ctx.lineTo(p.x - s, p.y + s);
        ctx.stroke();
        labels.push({ i, p, R, prio: 3, style: 'dead' });
        continue;
      }

      const core = !!S.inCore[i];
      const isHover = i === hoverNode;
      const isBroker = S.reveal && brokerTop10.has(i);

      if (isHover && core && S.running) {
        const g = ctx.createRadialGradient(p.x, p.y, R * 0.4, p.x, p.y, R * 3.2);
        g.addColorStop(0, 'rgba(255,75,80,.30)');
        g.addColorStop(1, 'rgba(255,75,80,0)');
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(p.x, p.y, R * 3.2, 0, TAU); ctx.fill();
      }
      if (isBroker) {
        const g = ctx.createRadialGradient(p.x, p.y, R * 0.4, p.x, p.y, R * 2.6);
        g.addColorStop(0, 'rgba(255,201,63,.30)');
        g.addColorStop(1, 'rgba(255,201,63,0)');
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(p.x, p.y, R * 2.6, 0, TAU); ctx.fill();
      }
      if (!S.reveal && core && n.deg >= 30) {      // the hubs smoulder a little
        const breathe = 0.10 + 0.04 * Math.sin(now / 800 + i);
        const g = ctx.createRadialGradient(p.x, p.y, R * 0.5, p.x, p.y, R * 2.4);
        g.addColorStop(0, `rgba(166,120,255,${breathe})`);
        g.addColorStop(1, 'rgba(166,120,255,0)');
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(p.x, p.y, R * 2.4, 0, TAU); ctx.fill();
      }

      let fill, stroke, lw = 1.1, alpha = 1;
      if (core) { fill = 'rgba(52,40,86,.94)'; stroke = '#8f7bd8'; }
      else if (inG0[i]) { fill = 'rgba(46,50,62,.75)'; stroke = 'rgba(130,140,160,.5)'; }
      else { fill = 'rgba(40,44,54,.5)'; stroke = 'rgba(110,120,140,.3)'; alpha = 0.6; }
      if (isBroker) { stroke = '#ffc93f'; lw = 2.2; }
      if (isHover) { stroke = core && S.running ? '#ff4b50' : '#ffc93f'; lw = 2.4; }

      ctx.globalAlpha = alpha;
      ctx.fillStyle = fill;
      ctx.beginPath(); ctx.arc(p.x, p.y, R * (isHover ? 1.15 : 1), 0, TAU); ctx.fill();
      ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke();
      ctx.globalAlpha = 1;

      if (isHover && core && S.running) {          // crosshair on a live target
        ctx.strokeStyle = 'rgba(255,75,80,.9)'; ctx.lineWidth = 1.4;
        const o = R * 1.5, l = R * 0.7;
        ctx.beginPath();
        ctx.moveTo(p.x - o - l, p.y); ctx.lineTo(p.x - o, p.y);
        ctx.moveTo(p.x + o, p.y); ctx.lineTo(p.x + o + l, p.y);
        ctx.moveTo(p.x, p.y - o - l); ctx.lineTo(p.x, p.y - o);
        ctx.moveTo(p.x, p.y + o); ctx.lineTo(p.x, p.y + o + l);
        ctx.stroke();
      }

      const prio = isHover ? 4 : isBroker ? 3.5 : (core && degRank[i] > 0 && degRank[i] <= 14) ? 2 : 0;
      if (prio > 0) labels.push({ i, p, R, prio, style: core ? 'core' : 'sev' });
    }

    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    labels.sort((a, b) => b.prio - a.prio || b.R - a.R);
    const taken = [];
    for (const L of labels) {
      const big = L.prio >= 3;
      const fs = clamp((big ? 13 : 11) * Math.max(0.75, Math.min(cam.z, 1.35)), 9, 17);
      ctx.font = `${big ? 700 : 500} ${fs}px Inter, system-ui, sans-serif`;
      const txt = shortName(nodes[L.i].name);
      const w = ctx.measureText(txt).width, y = L.p.y - L.R - 5;
      const box = { x0: L.p.x - w / 2 - 3, x1: L.p.x + w / 2 + 3, y0: y - fs - 1, y1: y + 2 };
      let hitBox = false;
      for (const t of taken) {
        if (box.x0 < t.x1 && box.x1 > t.x0 && box.y0 < t.y1 && box.y1 > t.y0) { hitBox = true; break; }
      }
      if (hitBox && L.prio < 4) continue;
      taken.push(box);
      ctx.lineWidth = 3.2; ctx.strokeStyle = 'rgba(8,5,15,.9)';
      ctx.strokeText(txt, L.p.x, y);
      ctx.fillStyle = L.style === 'dead' ? '#ff8b8f' : L.i === hoverNode ? '#ffc93f'
        : L.style === 'sev' ? 'rgba(170,180,200,.6)' : 'rgba(205,190,240,.75)';
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

  // -------------------------------------------------------------------- input
  let drag = null;

  function nodeAt(sx, sy) {
    let best = -1, bestD = 1e9;
    for (let i = 0; i < N; i++) {
      const w = posOf(i);
      const p = toScreen(w.x, w.y);
      const R = Math.max(8, nodeRadius(i) * cam.z) + 5;
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
        cam.tx = drag.ox - dx / cam.z; cam.ty = drag.oy - dy / cam.z;
        cam.x = cam.tx; cam.y = cam.ty;
      }
    } else {
      const h = nodeAt(sx, sy);
      if (h !== hoverNode) { hoverNode = h; setAim(h); highlightWanted(h); }
      cv.classList.toggle('aim', h >= 0 && S.running && S.alive[h] && S.inCore && S.inCore[h]);
    }
  });

  cv.addEventListener('pointerup', (e) => {
    cv.classList.remove('grabbing');
    const wasDrag = drag && drag.moved > 5;
    drag = null;
    if (wasDrag) return;
    const r = cv.getBoundingClientRect();
    const i = nodeAt(e.clientX - r.left, e.clientY - r.top);
    if (i >= 0) hit(i);
  });

  cv.addEventListener('pointerleave', () => { hoverNode = -1; setAim(-1); highlightWanted(-1); cv.classList.remove('grabbing'); });

  cv.addEventListener('wheel', (e) => {
    e.preventDefault();
    cam.tz = clamp(cam.tz * Math.exp(-e.deltaY * 0.0016), 0.12, 2.6);
  }, { passive: false });

  window.addEventListener('keydown', (e) => {
    if (!$('vStart').classList.contains('gone')) { if (e.key === 'Enter') $('bStart').click(); return; }
    if (e.key === 'c' || e.key === 'C') fitCamera();
    if (e.key === 'm' || e.key === 'M') $('bSound').click();
  });

  // ----------------------------------------------------------------------- UI
  let toastTimer = null;
  function toast(html, ms) {
    const t = $('toast');
    t.innerHTML = html; t.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), ms || 2800);
  }

  function syncUI() {
    $('sCore').textContent = S.coreSize;
    const removed = S.hits.length;
    $('sCut').textContent = G0SIZE - S.coreSize - removed;
    $('sHits').textContent = S.budget - removed;
    $('hitCount').textContent = removed;
    const obj = $('objective');
    if (S.hits.length >= S.budget) {
      obj.classList.add('done');
      $('objText').textContent = 'CONTRACT COMPLETE';
    } else {
      obj.classList.remove('done');
      $('objText').textContent = `SHRINK THE CORE · ${S.budget - removed} HIT${S.budget - removed === 1 ? '' : 'S'} LEFT`;
    }
  }

  /** The intel card: whoever the crosshair is on right now. */
  function setAim(i) {
    if (i < 0) {
      $('hKick').textContent = 'You are aiming at';
      $('hName').textContent = 'Nobody yet';
      $('hDesc').textContent = 'Hover a character on the map, or pick one from the list below. One click is one hit. There is no undo in this business.';
      $('hChips').innerHTML = '';
      return;
    }
    const n = nodes[i];
    $('hKick').textContent = S.alive[i] ? 'You are aiming at' : 'Already eliminated';
    $('hName').textContent = shortName(n.name);
    $('hDesc').textContent = n.desc || 'No description in the dataset.';
    const alive = S.alive[i], core = alive && S.inCore[i];
    // Current degree, so the card and the most-wanted list never disagree.
    let degNow = 0;
    for (const v of adj[i]) if (S.alive[v]) degNow++;
    let status;
    if (!alive) status = '<span class="chip dead">eliminated</span>';
    else if (core) status = '<span class="chip core">in the core</span>';
    else if (inG0[i]) status = '<span class="chip out">cut off</span>';
    else status = '<span class="chip out">never in the core</span>';
    $('hChips').innerHTML =
      `<span class="chip deg">${alive ? degNow : deg0[i]} links</span>` + status +
      (S.reveal || S.finished ? `<span class="chip">broker #${btwRank[i] > 0 ? btwRank[i] : '&ndash;'}</span>` : '');
  }

  function renderWanted() {
    const box = $('wanted');
    const degNow = Core.aliveDegrees(N, adj, S.alive);
    const list = [];
    for (let i = 0; i < N; i++) if (S.inCore[i]) list.push(i);
    list.sort((a, b) => degNow[b] - degNow[a] || a - b);
    box.innerHTML = '';
    list.slice(0, 15).forEach((i, k) => {
      const b = document.createElement('button');
      b.className = 'row';
      b.dataset.i = i;
      b.innerHTML = `<span class="rk">${k + 1}</span><span class="dot"></span>` +
        `<span class="nm">${shortName(nodes[i].name)}</span><span class="dg">${degNow[i]} links</span>`;
      b.onclick = () => hit(i);
      b.onmouseenter = () => { hoverNode = i; setAim(i); };
      b.onmouseleave = () => { hoverNode = -1; setAim(-1); };
      box.appendChild(b);
    });
  }

  function highlightWanted(i) {
    [...$('wanted').children].forEach(el => el.classList.toggle('hl', +el.dataset.i === i));
  }

  function renderHits() {
    const box = $('hits');
    box.innerHTML = '';
    if (!S.hits.length) {
      box.innerHTML = '<div class="row" style="color:var(--dim);font-size:11px">Nobody yet. ' +
        'Every hit lands here with how much of the core it took down.</div>';
      return;
    }
    [...S.hits].reverse().forEach((h) => {
      const d = document.createElement('div');
      d.className = 'row dead';
      d.innerHTML = `<span class="rk">&#10007;</span><span class="dot"></span>` +
        `<span class="nm">${shortName(nodes[h.i].name)}</span><span class="drop">core &minus;${h.drop}</span>`;
      box.appendChild(d);
    });
  }

  // ------------------------------------------------------------- start screen
  let chosenBudget = 8;

  function buildStart() {
    const dwrap = $('diffPick');
    dwrap.innerHTML = '';
    Object.entries(CONTRACTS).forEach(([key, d]) => {
      const b = document.createElement('button');
      b.className = 'diff' + (+key === chosenBudget ? ' sel' : '');
      b.innerHTML = `${d.label}<small>${d.note}</small>`;
      b.onclick = () => { chosenBudget = +key; buildStart(); };
      dwrap.appendChild(b);
    });
  }

  function showStart() {
    buildStart();
    $('vStart').classList.remove('gone');
    $('vWin').classList.add('gone');
    S.running = false;
  }

  function revealBrokers() {
    S.reveal = true;
    $('vWin').classList.add('gone');
    fitCamera();
    toast('Nodes re-sized by <b>betweenness</b> on the intact network. The gold ten are the brokers. ' +
      'Compare with the degree list on the right: mostly the same names, and the differences ' +
      '(Hercules, Black Widow, U.S. Agent) are the interesting part.', 9000);
    syncUI();
  }

  // ------------------------------------------------------------------ buttons
  $('bStart').onclick = () => { Sfx.boot(); $('vStart').classList.add('gone'); resize(); newRound(chosenBudget); };
  $('bAgain').onclick = () => { $('vWin').classList.add('gone'); newRound(S.budget); };
  $('bMenu').onclick = showStart;
  $('bMenu2').onclick = showStart;
  $('bReveal').onclick = revealBrokers;
  $('bCenter').onclick = fitCamera;
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
    draw(now, dt);
    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------------------ startup
  resize();
  S.alive = new Uint8Array(N).fill(1);
  recompute();
  fitCamera();
  cam.x = cam.tx; cam.y = cam.ty; cam.z = cam.tz;
  syncUI();
  renderWanted();
  showStart();
  requestAnimationFrame(frame);
})();
