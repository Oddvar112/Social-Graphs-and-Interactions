/* ORIGIN STORY - replay the Marvel network by debut year, or grow a model of the same size.
 *
 * Four modes share one machine. A mode is a list of arrivals (one node per
 * step) and a list of edges, each stamped with the step at which its later
 * endpoint arrives. The state at time t is simply "every node with rank <= t
 * and every edge stamped <= t", so scrubbing backwards costs nothing.
 *
 *   marvel    the frozen course snapshot, 303 heroes in order of first
 *             appearance, directed edges, degree = in-degree from characters
 *             already present
 *   combined  the week-2 crawl: the same heroes plus 363 villains from
 *             Category:Marvel Comics supervillains, 602 characters, villains
 *             drawn red, heroes cyan, the 64 filed as both in gold
 *   ba        preferential attachment with the endpoint-list trick: a newcomer
 *             picks a random entry of the list of all edge endpoints, i.e. a
 *             node with probability k_i / sum k_j. Four or five links per
 *             newcomer, exactly enough fives to land on the real m = 1434
 *   uniform   the same growth, but the newcomer picks an existing node at random
 *
 * Layout is a sunflower spiral by arrival rank: centre = first, rim = last.
 * That is the whole point of the picture. If the first to arrive become the
 * hubs, the centre glows.
 */
(function () {
  "use strict";
  const D = window.MARVEL_DATA, YEARS = window.MARVEL_YEARS || {}, IMAGES = window.MARVEL_IMAGES || {};
  const CMB = window.MARVEL_COMBINED || null;
  const GA = Math.PI * (3 - Math.sqrt(5));               // golden angle
  const COL = { bg: "#070a14", line: "#26304f", text: "#e9edf9", dim: "#8d99bd", web: "#5fe3ff",
                gold: "#ffc93f", red: "#ff4b50", edge: "rgba(95,110,170,0.28)", cross: "rgba(176,108,255,0.30)",
                hot: "rgba(255,201,63,0.85)" };
  const SIDE = { hero: "#5fe3ff", villain: "#ff4b50", both: "#ffc93f" };
  const PLASMA = [[13, 8, 135], [126, 3, 168], [204, 71, 120], [248, 149, 64], [240, 249, 33]];

  // ------------------------------------------------------------ helpers
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function plasma(x) {
    x = Math.max(0, Math.min(1, x)) * (PLASMA.length - 1);
    const i = Math.min(Math.floor(x), PLASMA.length - 2), f = x - i;
    const a = PLASMA[i], b = PLASMA[i + 1];
    return `rgb(${Math.round(a[0] + f * (b[0] - a[0]))},${Math.round(a[1] + f * (b[1] - a[1]))},${Math.round(a[2] + f * (b[2] - a[2]))})`;
  }
  function pretty(name) {
    return name.replace(/ \((character|characters|comics|Marvel Comics|Marvel Comics character)\)$/, "");
  }
  function ordinal(n) {
    const s = (n % 10 === 1 && n % 100 !== 11) ? "st" : (n % 10 === 2 && n % 100 !== 12) ? "nd"
            : (n % 10 === 3 && n % 100 !== 13) ? "rd" : "th";
    return n + s;
  }

  // ------------------------------------------------------------ the modes
  function replay(nodes, edgesByIndex, key, withSides) {
    // nodes: [{id, name, year, url, side}], edges in node-index space, directed.
    const n = nodes.length;
    const idx = nodes.map((_, i) => i);
    idx.sort((a, b) => (nodes[a].year || 9999) - (nodes[b].year || 9999) || nodes[a].name.localeCompare(nodes[b].name));
    const rank = new Array(n);
    idx.forEach((node, r) => { rank[node] = r; });
    const edges = edgesByIndex.map(e => [rank[e[0]], rank[e[1]]]);
    return {
      key, n, directed: true,
      ids: idx.map(i => nodes[i].id),
      urls: idx.map(i => nodes[i].url),
      sides: withSides ? idx.map(i => nodes[i].side || "hero") : null,
      labels: idx.map(i => pretty(nodes[i].name)),
      when: idx.map(i => nodes[i].year ? String(nodes[i].year) : "year unknown"),
      years: idx.map(i => nodes[i].year || null),
      edges,
      stamp: edges.map(e => Math.max(e[0], e[1])),
    };
  }

  function buildMarvel() {
    const nodes = D.nodes.map(nd => ({ id: nd.id, name: nd.name, url: nd.url, year: YEARS[nd.id] || null, side: "hero" }));
    const edges = [];
    D.out.forEach((targets, u) => targets.forEach(v => edges.push([u, v])));
    return replay(nodes, edges, "marvel", false);
  }

  function buildCombined() {
    return replay(CMB.nodes, CMB.edges, "combined", true);
  }

  function buildModel(preferential, seed) {
    const rnd = mulberry32(seed);
    const n = D.nodes.length, M_TARGET = 1434, SEED = 6;
    const edges = [];
    for (let i = 1; i < SEED; i++) edges.push([0, i]);          // a six-node star
    const ends = [];
    edges.forEach(e => ends.push(e[0], e[1]));
    const newcomers = n - SEED;
    const fives = M_TARGET - edges.length - 4 * newcomers;      // how many newcomers bring five links
    const bringsFive = new Array(n).fill(false);
    const pool = [];
    for (let i = SEED; i < n; i++) pool.push(i);
    for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
    for (let i = 0; i < fives; i++) bringsFive[pool[i]] = true;
    for (let v = SEED; v < n; v++) {
      const want = bringsFive[v] ? 5 : 4;
      const chosen = new Set();
      while (chosen.size < want) {
        chosen.add(preferential ? ends[Math.floor(rnd() * ends.length)] : Math.floor(rnd() * v));
      }
      chosen.forEach(u => { edges.push([v, u]); ends.push(v, u); });
    }
    const labels = [], when = [];
    for (let i = 0; i < n; i++) { labels.push("node #" + (i + 1)); when.push("arrived #" + (i + 1)); }
    return {
      key: preferential ? "ba" : "uniform", n, directed: false,
      labels, when, years: null, sides: null, edges,
      stamp: edges.map(e => Math.max(e[0], e[1])),
    };
  }

  // ------------------------------------------------------------ state
  const ARRIVE_MS = 480, EDGE_MS = 260;   // durations for the new-arrival burst / edge draw-in
  let mode = buildMarvel();
  let seed = 7;
  let t = 0;                     // rank of the newest node present
  let playing = false;
  let acc = 0, last = 0;
  let inspect = -1;              // rank of a clicked node, pinned in the card until released
  let lastDrawnT = -1, arrivalStart = 0;

  const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
  const cc = document.getElementById("ccdf"), cctx = cc.getContext("2d");
  const scrub = document.getElementById("scrub"), speed = document.getElementById("speed");
  const playBtn = document.getElementById("play");

  function degrees(upto) {
    const deg = new Int32Array(mode.n);
    const E = mode.edges, S = mode.stamp;
    for (let i = 0; i < E.length; i++) {
      if (S[i] > upto) continue;
      if (mode.directed) deg[E[i][1]]++; else { deg[E[i][0]]++; deg[E[i][1]]++; }
    }
    return deg;
  }
  function outDegreeAt(node, upto) {
    if (!mode.directed) return 0;
    const E = mode.edges, S = mode.stamp;
    let c = 0;
    for (let i = 0; i < E.length; i++) if (S[i] <= upto && E[i][0] === node) c++;
    return c;
  }
  function linksPresent(upto) {
    let c = 0;
    for (let i = 0; i < mode.stamp.length; i++) if (mode.stamp[i] <= upto) c++;
    return c;
  }

  // ------------------------------------------------------------ the arrival card
  // Pictures are fetched from Wikipedia as the playhead approaches a character,
  // thirty arrivals ahead, so the card is rarely waiting on the network.
  const imgCache = new Map();
  function preload(fromRank) {
    if (!mode.ids) return;
    for (let r = fromRank; r < Math.min(mode.n, fromRank + 30); r++) {
      const meta = IMAGES[mode.ids[r]];
      if (!meta || imgCache.has(meta.src)) continue;
      const im = new Image();
      im.decoding = "async";
      im.src = meta.src;
      imgCache.set(meta.src, im);
    }
  }
  let cardShown = null;
  function updateCard(deg) {
    const card = document.getElementById("card");
    const img = document.getElementById("card-img"), ph = document.getElementById("card-ph");
    const show = inspect >= 0 ? inspect : t;          // a pinned node takes over the card
    const pinned = inspect >= 0 && inspect !== t;
    const own = deg[show];
    card.classList.toggle("model", !mode.directed);
    card.classList.toggle("pinned", pinned);
    const side = mode.sides ? mode.sides[show] : null;
    const sideWord = side === "villain" ? " · villain" : side === "both" ? " · hero and villain" : "";
    document.getElementById("card-burst").textContent = mode.directed
      ? (pinned ? `Pinned · ${mode.when[show]}${sideWord}` : `Just arrived · ${mode.when[show]}${sideWord}`)
      : mode.when[show];
    card.style.setProperty("--burst", pinned ? "#a678ff" : side === "villain" ? COL.red : side === "both" ? "#ff9f43" : COL.gold);
    document.getElementById("card-name").textContent = mode.labels[show];
    const outOwn = outDegreeAt(show, t);
    document.getElementById("card-info").innerHTML = mode.directed
      ? `<b>${own}</b> in-link${own === 1 ? "" : "s"} from the ${t} characters here so far, <b>${outOwn}</b> out-link${outOwn === 1 ? "" : "s"} to others.`
      : `Arrived with <b>${own}</b> links, chosen ${mode.key === "ba" ? "in proportion to degree" : "uniformly at random"}.`;
    if (mode.directed) {
      const meta = IMAGES[mode.ids[show]];
      card.href = mode.urls[show];
      if (meta) {
        img.onerror = () => { img.hidden = true; ph.hidden = false; };
        img.src = meta.src;
        img.alt = mode.labels[show];
        img.hidden = false;
        ph.hidden = true;
        document.getElementById("card-credit").textContent = "Image: Wikipedia, " + meta.file.replace(/_/g, " ");
      } else {
        img.onerror = null;
        img.removeAttribute("src");
        img.hidden = true;
        ph.hidden = false;
        document.getElementById("card-credit").textContent = "No lead image on the Wikipedia article";
      }
      ph.innerHTML = `${mode.labels[show]}<small>no picture on Wikipedia</small>`;
      preload(t + 1);
    } else {
      card.removeAttribute("href");
    }
    const stamp = mode.key + ":" + show;
    if (cardShown !== stamp) {
      cardShown = stamp;
      card.classList.remove("pop");
      void card.offsetWidth;                      // restart the animation
      card.classList.add("pop");
    }
  }

  // ------------------------------------------------------------ layout
  function layout() {
    const w = cv.width, h = cv.height;
    const R = Math.min(w, h) * 0.46;
    const cx = w / 2, cy = h / 2;
    const pos = new Array(mode.n);
    for (let r = 0; r < mode.n; r++) {
      const rad = R * Math.sqrt((r + 0.5) / mode.n);
      const a = r * GA;
      pos[r] = [cx + rad * Math.cos(a), cy + rad * Math.sin(a)];
    }
    return { pos, R, cx, cy };
  }

  function rings() {
    // Where to draw the faint rings: decade starts for the replays, every 50 for the models.
    const out = [];
    if (mode.years) {
      let lastDecade = null;
      for (let r = 0; r < mode.n; r++) {
        const y = mode.years[r];
        if (!y) break;
        const dec = Math.floor(y / 10) * 10;
        if (dec !== lastDecade && dec >= 1940) { out.push({ r, label: String(dec) }); }
        lastDecade = dec;
      }
    } else {
      for (let r = 50; r < mode.n; r += 50) out.push({ r, label: "#" + r });
    }
    return out;
  }

  // ------------------------------------------------------------ drawing
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = cv.getBoundingClientRect();
    cv.width = Math.round(rect.width * dpr);
    cv.height = Math.round(rect.height * dpr);
    const r2 = cc.getBoundingClientRect();
    cc.width = Math.round(r2.width * dpr);
    cc.height = Math.round(r2.height * dpr);
    draw();
  }

  function nodeRadius(k, dpr) {
    const scale = mode.n > 400 ? 1.25 : 1.55;              // 602 nodes need a little less room each
    return (2.0 + scale * Math.sqrt(k)) * dpr;
  }

  function draw() {
    const w = cv.width, h = cv.height, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const now = performance.now();
    if (t !== lastDrawnT) { arrivalStart = now; lastDrawnT = t; }
    if (inspect > t) inspect = -1;                          // a pinned node can outrun a reset/scrub-back
    const burstProg = Math.min(1, (now - arrivalStart) / ARRIVE_MS);

    ctx.clearRect(0, 0, w, h);
    const { pos, R, cx, cy } = layout();
    const deg = degrees(t);
    const KSCALE = mode.key === "combined" ? 201 : 106;     // the colour scale tops out at Spider-Man
    const E = mode.edges, S = mode.stamp, sides = mode.sides;

    // rings
    ctx.lineWidth = 1 * dpr;
    ctx.font = `${10 * dpr}px Inter, sans-serif`;
    ctx.textAlign = "left";
    rings().forEach(({ r, label }) => {
      const rad = R * Math.sqrt((r + 0.5) / mode.n);
      ctx.strokeStyle = "rgba(141,153,189,0.16)";
      ctx.beginPath(); ctx.arc(cx, cy, rad, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = r <= t ? "rgba(141,153,189,0.75)" : "rgba(141,153,189,0.32)";
      ctx.fillText(label, cx + rad * 0.7071 + 4 * dpr, cy - rad * 0.7071 - 3 * dpr);
    });

    // edges present; in the combined replay a link across the hero-villain line is purple
    const crosses = i => !!sides && sides[E[i][0]] !== sides[E[i][1]] && sides[E[i][0]] !== "both" && sides[E[i][1]] !== "both";
    ctx.lineWidth = 0.7 * dpr;
    for (const [colour, wantCross] of [[COL.edge, false], [COL.cross, true]]) {
      if (wantCross && !sides) continue;
      ctx.strokeStyle = colour;
      ctx.beginPath();
      for (let i = 0; i < E.length; i++) {
        if (S[i] >= t || crosses(i) !== wantCross) continue;
        ctx.moveTo(pos[E[i][0]][0], pos[E[i][0]][1]);
        ctx.lineTo(pos[E[i][1]][0], pos[E[i][1]][1]);
      }
      ctx.stroke();
    }
    // edges landing this step draw in over EDGE_MS, growing outward from the newcomer itself.
    // A newcomer with a lot of links (an unknown-year character forced to the very last rank,
    // say) gets its edges staggered a few ms apart instead of the whole fan snapping in as one
    // flash - it reads as a quick cascade rather than "connected to everything at once".
    const arriving = [];
    for (let i = 0; i < E.length; i++) if (S[i] === t) arriving.push(i);
    ctx.lineWidth = 1.3 * dpr;
    ctx.strokeStyle = COL.hot;
    ctx.beginPath();
    arriving.forEach((i, k) => {
      const delay = Math.min(k, 12) * 14;
      const p = Math.max(0, Math.min(1, ((now - arrivalStart) - delay) / EDGE_MS));
      if (p <= 0) return;
      const from = E[i][0] === t ? E[i][0] : E[i][1], to = from === E[i][0] ? E[i][1] : E[i][0];
      const a = pos[from], b = pos[to];
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p);
    });
    ctx.stroke();

    // the current leader gets a slow pulsing glow, so "who's winning" reads without the board
    let leader = -1, leaderK = 0;
    for (let r = 0; r <= t; r++) if (deg[r] > leaderK) { leaderK = deg[r]; leader = r; }
    if (leader >= 0) {
      const pulse = 0.5 + 0.5 * Math.sin(now / 450);
      const glow = sides ? (SIDE[sides[leader]] || COL.gold) : COL.gold;
      ctx.save();
      ctx.shadowColor = glow;
      ctx.shadowBlur = (14 + 10 * pulse) * dpr;
      ctx.beginPath();
      ctx.arc(pos[leader][0], pos[leader][1], nodeRadius(leaderK, dpr) + 3 * dpr, 0, Math.PI * 2);
      ctx.strokeStyle = glow;
      ctx.globalAlpha = 0.55 + 0.25 * pulse;
      ctx.lineWidth = 2 * dpr;
      ctx.stroke();
      ctx.restore();
    }

    // click any node that has arrived to pin it; its incident edges get picked out
    if (inspect >= 0) {
      ctx.lineWidth = 1.1 * dpr;
      ctx.strokeStyle = "rgba(233,237,249,0.55)";
      ctx.beginPath();
      for (let i = 0; i < E.length; i++) {
        if (S[i] > t || (E[i][0] !== inspect && E[i][1] !== inspect)) continue;
        ctx.moveTo(pos[E[i][0]][0], pos[E[i][0]][1]);
        ctx.lineTo(pos[E[i][1]][0], pos[E[i][1]][1]);
      }
      ctx.stroke();
    }

    // nodes, small first so hubs sit on top
    const order = [];
    for (let r = 0; r <= t; r++) order.push(r);
    order.sort((a, b) => deg[a] - deg[b]);
    for (const r of order) {
      const k = deg[r];
      const pop = (r === t && burstProg < 1) ? 0.5 + 0.5 * (1 - Math.pow(1 - burstProg, 3)) : 1;
      ctx.beginPath();
      ctx.arc(pos[r][0], pos[r][1], nodeRadius(k, dpr) * pop, 0, Math.PI * 2);
      if (sides) {
        ctx.fillStyle = SIDE[sides[r]] || COL.web;
        ctx.globalAlpha = 0.45 + 0.55 * Math.min(1, Math.log1p(k) / Math.log1p(KSCALE) * 1.6);
      } else {
        ctx.fillStyle = plasma(Math.log1p(k) / Math.log1p(KSCALE));
      }
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.lineWidth = (r === t ? 2.2 : r === inspect ? 2 : 0.6) * dpr;
      ctx.strokeStyle = r === t ? COL.gold : r === inspect ? "#e9edf9" : COL.bg;
      ctx.stroke();
    }

    // labels for the top hubs
    const top = order.slice(-8).reverse().filter(r => deg[r] > 0);
    ctx.font = `700 ${11 * dpr}px Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.lineWidth = 3.5 * dpr;
    ctx.lineJoin = "round";
    top.forEach(r => {
      const text = mode.directed ? mode.labels[r] : `#${r + 1}`;
      const y = pos[r][1] - nodeRadius(deg[r], dpr) - 4 * dpr;
      ctx.strokeStyle = COL.bg;
      ctx.strokeText(text, pos[r][0], y);
      ctx.fillStyle = sides ? (SIDE[sides[r]] || COL.text) : COL.text;
      ctx.fillText(text, pos[r][0], y);
    });

    // a burst of particles where the newest character just landed
    if (burstProg < 1) {
      const burstColor = sides ? (SIDE[sides[t]] || COL.hot) : COL.hot;
      drawBurst(pos[t], nodeRadius(deg[t], dpr), burstProg, dpr, burstColor);
    }

    maybeSplash();
    drawCCDF(deg);
    updatePanel(deg);
  }

  function drawBurst(p, r0, prog, dpr, color) {
    const ease = 1 - Math.pow(1 - prog, 3);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.globalAlpha = (1 - ease) * 0.85;
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    ctx.arc(p[0], p[1], r0 + ease * 24 * dpr, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = color;
    const N = 8;
    for (let i = 0; i < N; i++) {
      const ang = (i / N) * Math.PI * 2 + 0.35;
      const dist = ease * (18 + (i % 3) * 6) * dpr;
      ctx.globalAlpha = (1 - ease) * 0.8;
      ctx.beginPath();
      ctx.arc(p[0] + Math.cos(ang) * dist, p[1] + Math.sin(ang) * dist, Math.max(0.6, (1 - ease) * 2.2) * dpr, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  // ------------------------------------------------------------ decade splash cards
  // Fires once per decade (or every 50 arrivals in the models) the first time it's reached,
  // even across mode switches or a reseed - not on every scrub back and forth over the same ground.
  let announced = new Set();
  let splashTimer = null;
  function showSplash(label) {
    const el = document.getElementById("splash");
    el.innerHTML = `<span>${label}</span>`;
    const span = el.firstElementChild;
    requestAnimationFrame(() => requestAnimationFrame(() => span.classList.add("show")));
    clearTimeout(splashTimer);
    splashTimer = setTimeout(() => span.classList.remove("show"), 1000);
  }
  function maybeSplash() {
    let label = null;
    if (mode.years) {
      const y = mode.years[t];
      if (y) { const dec = Math.floor(y / 10) * 10; if (dec >= 1940) label = dec + "s"; }
    } else if (t > 0 && t % 50 === 0) {
      label = "#" + t;
    }
    if (label && !announced.has(label)) { announced.add(label); showSplash(label); }
  }

  function drawCCDF(deg) {
    const w = cc.width, h = cc.height, dpr = Math.min(window.devicePixelRatio || 1, 2);
    cctx.clearRect(0, 0, w, h);
    const ks = [];
    for (let r = 0; r <= t; r++) if (deg[r] > 0) ks.push(deg[r]);
    if (ks.length < 2) return;
    ks.sort((a, b) => a - b);
    const n = ks.length;
    const pts = [];
    for (let i = 0; i < n; i++) if (i === 0 || ks[i] !== ks[i - 1]) pts.push([ks[i], (n - i) / n]);
    const padL = 30 * dpr, padB = 20 * dpr, padT = 10 * dpr, padR = 10 * dpr;
    const xmax = Math.log10(250), ymin = Math.log10(1 / 602);
    const X = k => padL + (Math.log10(k) / xmax) * (w - padL - padR);
    const Y = p => padT + (Math.log10(p) / ymin) * (h - padT - padB);
    cctx.strokeStyle = "rgba(38,48,79,0.9)";
    cctx.lineWidth = 1 * dpr;
    cctx.font = `${9 * dpr}px Inter, sans-serif`;
    cctx.fillStyle = COL.dim;
    cctx.textAlign = "center";
    [1, 10, 100].forEach(k => {
      cctx.beginPath(); cctx.moveTo(X(k), padT); cctx.lineTo(X(k), h - padB); cctx.stroke();
      cctx.fillText(String(k), X(k), h - 6 * dpr);
    });
    cctx.textAlign = "right";
    [1, 0.1, 0.01].forEach(p => {
      cctx.beginPath(); cctx.moveTo(padL, Y(p)); cctx.lineTo(w - padR, Y(p)); cctx.stroke();
      cctx.fillText(String(p), padL - 4 * dpr, Y(p) + 3 * dpr);
    });
    // reference slope: a straight line with the Marvel in-degree exponent, alpha - 1 = 1.22
    cctx.strokeStyle = "rgba(255,75,80,0.55)";
    cctx.setLineDash([4 * dpr, 4 * dpr]);
    cctx.beginPath();
    cctx.moveTo(X(5), Y(0.37));
    cctx.lineTo(X(200), Y(0.37 * Math.pow(40, -1.22)));
    cctx.stroke();
    cctx.setLineDash([]);
    cctx.fillStyle = COL.web;
    pts.forEach(([k, p]) => { cctx.beginPath(); cctx.arc(X(k), Y(p), 2.2 * dpr, 0, Math.PI * 2); cctx.fill(); });
    cctx.fillStyle = COL.dim;
    cctx.textAlign = "right";
    cctx.fillText("dashed: Marvel's fitted in-degree slope", w - padR - 4 * dpr, padT + 9 * dpr);
  }

  function updatePanel(deg) {
    const big = document.getElementById("ro-big"), sub = document.getElementById("ro-sub");
    let clock = "?";
    if (mode.years) { for (let r = t; r >= 0; r--) if (mode.years[r]) { clock = String(mode.years[r]); break; } }
    big.textContent = mode.years ? clock : "#" + (t + 1);
    let extra = "";
    if (mode.sides) {
      const c = { hero: 0, villain: 0, both: 0 };
      for (let r = 0; r <= t; r++) c[mode.sides[r]]++;
      extra = ` · ${c.hero} heroes, ${c.villain} villains, ${c.both} both`;
    }
    sub.textContent = `${t + 1} of ${mode.n} characters · ${linksPresent(t)} links${extra}`;
    updateCard(deg);
    scrub.max = mode.n - 1;
    scrub.value = t;
    document.getElementById("scrub-val").textContent = `${t + 1} / ${mode.n}`;

    const order = [];
    for (let r = 0; r <= t; r++) order.push(r);
    order.sort((a, b) => deg[b] - deg[a] || a - b);
    const top = order.slice(0, 10);
    const kmax = deg[top[0]] || 1;
    const board = document.getElementById("board");
    board.innerHTML = top.map((r, i) => {
      const dot = mode.sides ? `<i class="dot" style="background:${SIDE[mode.sides[r]]}"></i>` : "";
      return `<li class="${i === 0 && deg[r] > 0 ? "lead" : ""}${r === t ? " newest" : ""}">
         <span class="bar" style="width:${(100 * deg[r] / kmax).toFixed(1)}%"></span>
         <span class="rk">${i + 1}</span>
         <span class="nm">${dot}${mode.labels[r]}<small>${mode.when[r]}</small></span>
         <span class="k">${deg[r]}</span></li>`;
    }).join("");
    document.getElementById("board-title").textContent =
      mode.directed ? "Most linked, right now (in-degree)" : "Most linked, right now (degree)";

    // the first-mover readout: where in the arrival order do the current top ten sit?
    const fact = document.getElementById("fact");
    if (t < 20) {
      fact.innerHTML = mode.key === "combined"
        ? "Press play. Villains arrive in red, heroes in cyan, the 64 characters Wikipedia files as both in gold. Purple links cross the line. Watch which side collects the links."
        : mode.directed
          ? "Press play. The Golden Age arrives first: the Human Torch, Namor's era, the 1940s. Watch how few links they hold once the 1960s land."
          : "Press play. The six seed nodes are the centre. Watch whether anybody who arrives later ever catches them.";
    } else {
      const pct = top.filter(r => deg[r] > 0).map(r => 100 * r / t);
      const avg = pct.reduce((a, b) => a + b, 0) / pct.length;
      const leader = order[0];
      let villainNote = "";
      if (mode.sides) {
        const topV = order.find(r => mode.sides[r] === "villain" && deg[r] > 0);
        if (topV !== undefined) {
          const rankV = order.indexOf(topV);
          villainNote = ` The best-linked villain is <b>${mode.labels[topV]}</b> (${mode.when[topV]}), ${ordinal(rankV + 1)} overall with ${deg[topV]} in-links.`;
        }
      }
      fact.innerHTML = `The current top ten arrived, on average, at the <b>${ordinal(Math.round(avg))} percentile</b> of everyone here so far. ` +
        (mode.directed
          ? `Leader: <b>${mode.labels[leader]}</b> (${mode.when[leader]}) with ${deg[leader]} in-links.${villainNote}`
          : `Leader: <b>${mode.labels[leader]}</b> with ${deg[leader]} links. Under preferential attachment the top ten are almost always the first ten.`);
    }
    playBtn.innerHTML = playing ? "&#10074;&#10074; Pause" : "&#9654; Play";
    playBtn.disabled = false;
    document.getElementById("reseed").disabled = mode.directed;
  }

  // ------------------------------------------------------------ control
  function setT(v) { t = Math.max(0, Math.min(mode.n - 1, v)); draw(); }
  function tick(now) {
    if (playing) {
      const dt = (now - last) / 1000;
      last = now;
      acc += dt * Number(speed.value);
      while (acc >= 1) { acc -= 1; if (t < mode.n - 1) t++; else { playing = false; break; } }
      draw();
    } else if (now - arrivalStart < Math.max(ARRIVE_MS, EDGE_MS)) {
      draw();                    // keep animating the burst/edge draw-in a beat after a manual step
    }
    requestAnimationFrame(tick);
  }
  function setMode(key) {
    if (key === "marvel") mode = buildMarvel();
    else if (key === "combined" && CMB) mode = buildCombined();
    else mode = buildModel(key === "ba", seed);
    document.querySelectorAll(".mode").forEach(b => b.classList.toggle("on", b.dataset.mode === mode.key));
    playing = false;
    inspect = -1;
    announced = new Set();
    setT(0);
  }
  function hitTest(clientX, clientY) {
    const rect = cv.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const mx = (clientX - rect.left) * dpr, my = (clientY - rect.top) * dpr;
    const { pos } = layout();
    const deg = degrees(t);
    let best = -1, bestD = Infinity;
    for (let r = 0; r <= t; r++) {
      const dx = pos[r][0] - mx, dy = pos[r][1] - my;
      const d2 = dx * dx + dy * dy;
      const hr = nodeRadius(deg[r], dpr) + 5 * dpr;
      if (d2 <= hr * hr && d2 < bestD) { bestD = d2; best = r; }
    }
    return best;
  }

  playBtn.addEventListener("click", () => {
    if (t >= mode.n - 1) t = 0;
    playing = !playing;
    last = performance.now();
    acc = 0;
    draw();
  });
  document.getElementById("step").addEventListener("click", () => { playing = false; setT(t + 1); });
  document.getElementById("reset").addEventListener("click", () => { playing = false; setT(0); });
  document.getElementById("reseed").addEventListener("click", () => {
    seed = (seed * 48271 + 11) % 2147483647;
    setMode(mode.key);
  });
  scrub.addEventListener("input", () => { playing = false; setT(Number(scrub.value)); });
  speed.addEventListener("input", () => { document.getElementById("speed-val").textContent = `${speed.value} characters / s`; });
  document.querySelectorAll(".mode").forEach(b => b.addEventListener("click", () => setMode(b.dataset.mode)));
  cv.addEventListener("mousemove", e => { cv.style.cursor = hitTest(e.clientX, e.clientY) >= 0 ? "pointer" : "default"; });
  cv.addEventListener("mouseleave", () => { cv.style.cursor = "default"; });
  cv.addEventListener("click", e => {
    const hit = hitTest(e.clientX, e.clientY);
    // clicking empty canvas, or the newest node (already the star of the show), releases any pin
    inspect = (hit === -1 || hit === t || hit === inspect) ? -1 : hit;
    draw();
  });
  window.addEventListener("resize", resize);
  document.addEventListener("keydown", e => {
    if (e.code === "Space") { e.preventDefault(); playBtn.click(); }
    if (e.code === "ArrowRight") { playing = false; setT(t + 1); }
    if (e.code === "ArrowLeft") { playing = false; setT(t - 1); }
  });

  // ?mode=combined&t=150 opens the page at a moment, so the post can link to one.
  const q = new URLSearchParams(location.search);
  if (["combined", "ba", "uniform"].includes(q.get("mode"))) setMode(q.get("mode"));
  if (q.get("t")) t = Math.max(0, Math.min(mode.n - 1, Number(q.get("t")) || 0));
  resize();
  requestAnimationFrame(tick);
})();
