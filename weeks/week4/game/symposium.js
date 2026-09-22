/* ===========================================================================
   SYMPOSIUM: seat the history of philosophy.

   Data: the course's frozen week-4 philosophers snapshot, undirected giant
   component (1374 philosophers, 9139 links), laid out once in Python. All
   graph algorithms live in core.js, which the test suite runs under node and
   diffs against networkx.

   ONE TABLE (the main game): you get one philosopher at the head of an
   empty table and 3, 5 or 9 seats. The score is that table's share of
   modularity times m, "links above chance": links inside minus what a
   degree-preserving shuffle would put there. Rivals: a greedy host, the
   best table a deterministic swap search finds, a random pick from the
   host's friends; the reveal shows the best table.

   THE WHOLE ROOM: a few place cards and up to nine tables. Each card seats
   one philosopher by hand; everyone else joins the table where most of the
   people they link to are sitting, ring by ring, until the room settles.
   Score: modularity of the whole seating. Rivals: Louvain, greedy, by
   century, at random; the reveal shows where you and Louvain disagree, and
   how often Louvain disagrees with itself.
   =========================================================================== */
(function () {
  'use strict';

  const D = window.SYMPOSIUM;
  if (!D) {
    document.body.innerHTML = '<p style="padding:40px;font:16px serif;color:#fff">' +
      'data/week4_symposium.js is missing. Run <code>python scripts/build_week4_data.py</code>.</p>';
    return;
  }
  const Core = window.SymposiumCore;
  const IMAGES = window.PHILOSOPHER_IMAGES || {};
  const G = Core.buildGraph(D);
  const N = G.N;
  const WORLD = 3200;
  const TAU = Math.PI * 2;

  const $ = (id) => document.getElementById(id);
  const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
  const lerp = (a, b, t) => a + (b - a) * t;
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');

  const TABLE_COLORS = ['#d9a648', '#7fb069', '#5b8fd6', '#e07a3f', '#d4536a', '#a985e0', '#45c2b1', '#e79ab9', '#b7b64f', '#c9c1b3'];
  const ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'];
  const MAX_TABLES = 9;
  const UNSEATED = '#4d3f46';
  const LADDER_MAX = 0.58;

  const nodes = D.nodes.map((n, i) => ({
    ...n, idx: i,
    wx: n.x * WORLD, wy: n.y * WORLD,
    r: 2.3 + 1.15 * Math.sqrt(n.deg),
  }));
  const edges = D.edges;
  const LOU = D.louvain.labels;
  const LOU_K = D.louvain.k;
  const famousSet = new Set(D.famous.slice(0, 30));

  // Louvain's communities: centroids for the reveal, and a colour each.
  const louCentroid = (() => {
    const sx = new Float64Array(LOU_K), sy = new Float64Array(LOU_K), n = new Int32Array(LOU_K);
    for (let i = 0; i < N; i++) { sx[LOU[i]] += nodes[i].wx; sy[LOU[i]] += nodes[i].wy; n[LOU[i]]++; }
    return Array.from({ length: LOU_K }, (_, c) => ({ x: sx[c] / n[c], y: sy[c] / n[c], size: n[c] }));
  })();

  // ------------------------------------------------------------------ state
  const LEVELS = {
    6: { label: 'A quiet dinner', note: '6 place cards' },
    12: { label: 'The banquet', note: '12 place cards' },
    20: { label: 'The whole Academy', note: '20 place cards' },
  };

  const SEATS = {
    3: { label: 'A small circle', note: '3 seats' },
    5: { label: 'A table for six', note: '5 seats' },
    9: { label: 'The long table', note: '9 seats' },
  };
  const MODES = {
    one: { label: 'ONE TABLE', note: 'You get a guest. Fill 3, 5 or 9 seats so the table beats chance by the most.' },
    room: { label: 'THE WHOLE ROOM', note: 'A few place cards, everybody else follows their friends. Modularity of the whole seating.' },
  };

  const S = {
    mode: 'room', host: -1, k: 5, members: [], score: null, rivals: null, revealMembers: null, hintsLeft: 1,   // mode is set when a round starts
    cards: 12, tables: [], active: -1, activeChosen: false, seeds: new Map(),
    labels: new Int32Array(N).fill(-1), disp: new Int32Array(N).fill(-1),
    q: 0, stats: null, weighted: false,
    running: false, finished: false, reveal: false, history: [],
  };
  const anim = new Map();                         // node -> { t0, to }
  const cam = { x: WORLD / 2, y: WORLD / 2, z: 0.3, tx: WORLD / 2, ty: WORLD / 2, tz: 0.3 };
  let hoverNode = -1;
  let NOW = performance.now();
  let roundStartedAt = 0;
  const SX = new Float32Array(N), SY = new Float32Array(N);   // screen coords, refreshed every frame

  const bbox = (() => {
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (const n of nodes) { x0 = Math.min(x0, n.wx); y0 = Math.min(y0, n.wy); x1 = Math.max(x1, n.wx); y1 = Math.max(y1, n.wy); }
    return { x0, y0, x1, y1 };
  })();

  const cardsLeft = () => S.cards - S.seeds.size;

  // Portraits: Wikipedia's page image for each philosopher, loaded on first
  // use and drawn inside the node once it is on screen at a readable size.
  const imgCache = new Map();
  function portraitOf(i) {
    const meta = IMAGES[nodes[i].id];
    if (!meta) return null;
    let e = imgCache.get(i);
    if (!e) {
      e = { img: new Image(), ok: false };
      e.img.onload = () => { e.ok = true; };
      e.img.src = meta.src;
      imgCache.set(i, e);
    }
    return e.ok ? e.img : null;
  }
  const portraitTag = (i, cls) => {
    const meta = IMAGES[nodes[i].id];
    return meta ? `<img class="${cls}" loading="lazy" src="${meta.src}" alt="">`
      : `<span class="pini">${esc(nodes[i].name[0])}</span>`;
  };
  const colorOf = (i) => S.reveal ? TABLE_COLORS[LOU[i] % TABLE_COLORS.length] : (S.disp[i] < 0 ? UNSEATED : TABLE_COLORS[S.disp[i]]);
  const rgba = (hex, a) => `rgba(${parseInt(hex.slice(1, 3), 16)},${parseInt(hex.slice(3, 5), 16)},${parseInt(hex.slice(5, 7), 16)},${a})`;

  // ------------------------------------------------------------ the seating
  /** Recompute everyone's table from the placed guests; queue the ink animation for whoever changed. */
  function recompute(animate) {
    const prev = S.labels;
    const seeds = [...S.seeds].map(([i, t]) => [i, t]);
    let labels, rounds = [], sweeps = [];
    if (seeds.length === 0) {
      labels = new Int32Array(N).fill(-1);
    } else {
      const P = Core.propagate(G, seeds, { weighted: S.weighted });
      labels = P.labels; rounds = P.rounds; sweeps = P.sweeps;
    }
    S.labels = labels;
    S.q = Core.modularity(G, labels, false);
    S.stats = Core.tableStats(G, labels, Math.max(1, S.tables.length));
    attribute();

    const arrival = new Int32Array(N);
    rounds.forEach((ch, r) => { for (let k = 0; k < ch.length; k += 2) arrival[ch[k]] = r + 1; });
    sweeps.forEach((ch, s) => { for (let k = 0; k < ch.length; k += 2) arrival[ch[k]] = rounds.length + 1 + s; });

    const now = performance.now();
    let changed = 0, followed = 0;
    for (let i = 0; i < N; i++) {
      if (labels[i] === prev[i]) continue;
      changed++;
      if (labels[i] === S.active) followed++;
      if (!animate || labels[i] < 0) { S.disp[i] = labels[i]; anim.delete(i); }
      else anim.set(i, { t0: now + 40 + arrival[i] * 70, to: labels[i] });
    }
    return { changed, followed };
  }

  /* Who brought whom: walk out from every placed guest along links that stay
     inside their table, so each seated guest is credited to the nearest
     place card at their table. Display only; the score does not use it. */
  function attribute() {
    const attr = new Int32Array(N).fill(-1);
    const dist = new Int32Array(N).fill(-1);
    const queue = [];
    for (const [i] of S.seeds) { attr[i] = i; dist[i] = 0; queue.push(i); }
    for (let h = 0; h < queue.length; h++) {
      const u = queue[h], lu = S.labels[u];
      for (const v of G.adj[u]) {
        if (dist[v] >= 0 || S.labels[v] !== lu) continue;
        dist[v] = dist[u] + 1; attr[v] = attr[u]; queue.push(v);
      }
    }
    const brought = new Map();
    for (let i = 0; i < N; i++) if (attr[i] >= 0 && attr[i] !== i) brought.set(attr[i], (brought.get(attr[i]) || 0) + 1);
    S.attr = attr; S.brought = brought;
  }
  const broughtBy = (i) => (S.brought && S.brought.get(i)) || 0;

  function stepAnim(now) {
    for (const [i, a] of anim) {
      if (now >= a.t0) { S.disp[i] = a.to; if (now >= a.t0 + 520) anim.delete(i); }
    }
  }
  function popOf(i, now) {
    const a = anim.get(i);
    if (!a || now < a.t0) return 1;
    return 1 + 0.9 * Math.sin(Math.PI * Math.min(1, (now - a.t0) / 520));
  }

  function snapshot() {
    if (S.mode === 'one') return { members: S.members.slice() };
    return { seeds: new Map(S.seeds), tables: S.tables.length, active: S.active };
  }
  function pushHistory() {
    S.history.push(snapshot());
    if (S.history.length > 80) S.history.shift();
    $('bUndo').disabled = false;
  }
  function undo() {
    const h = S.history.pop();
    if (!h) return;
    if (S.mode === 'one') {
      S.members = h.members;
      updateOne(true);
    } else {
      S.seeds = h.seeds;
      S.tables = Array.from({ length: h.tables }, () => ({}));
      S.active = h.active;
      recompute(true);
    }
    syncAll();
    $('bUndo').disabled = S.history.length === 0;
    toast('Taken back.');
  }

  // ------------------------------------------------------------ ONE TABLE
  /* Build the best community around one guest: the host sits first, you
     fill k seats, and the score is the table's links above chance. */
  function newOneRound(host, k) {
    S.mode = 'one'; S.host = host; S.k = k; S.members = [host]; S.history = [];
    S.tables = [{}]; S.active = 0; S.activeChosen = false;
    S.labels = new Int32Array(N).fill(-1); S.disp = new Int32Array(N).fill(-1);
    S.q = 0; S.stats = null; S.attr = null; S.brought = null; S.revealMembers = null; S.hintsLeft = 1;
    S.running = true; S.finished = false; S.reveal = false;
    roundStartedAt = performance.now();
    anim.clear();
    const greedy = Core.greedyTable(G, host, k);
    const record = Core.improveTable(G, host, greedy);
    S.rivals = {
      greedy, record,
      greedyScore: Core.tableScore(G, greedy), recordScore: Core.tableScore(G, record),
      random: Core.randomTableAvg(G, host, k, 20, 3),
    };
    $('stage').classList.add('one');
    $('bUndo').disabled = true;
    updateOne(false);
    fitCamera();
    cam.x = cam.tx; cam.y = cam.ty; cam.z = cam.tz * 0.6;
    renderLadder();
    syncAll();
    closeResults();
    setAim(-1);
    toast(`<b>${esc(nodes[host].name)}</b> is at the head of the table, ${k} seats to fill. Seat people who link to the table, and mind how many links they have in all: hubs are expensive.`, 9000);
  }

  /** Recolour the map and rescore after the table changed. */
  function updateOne(animate) {
    const members = S.reveal && S.revealMembers ? S.revealMembers : S.members;
    const prev = S.labels;
    const labels = new Int32Array(N).fill(-1);
    for (const i of members) labels[i] = 0;
    S.labels = labels;
    S.score = Core.tableScore(G, members);
    S.q = S.score.q;
    S.seeds = new Map(members.map(i => [i, 0]));           // the map draws them with a ring and a face
    const now = performance.now();
    for (let i = 0; i < N; i++) {
      if (labels[i] === prev[i]) continue;
      if (!animate || labels[i] < 0) { S.disp[i] = labels[i]; anim.delete(i); }
      else anim.set(i, { t0: now + 40, to: 0 });
    }
  }

  const inSetOf = (members) => { const s = new Uint8Array(N); for (const i of members) s[i] = 1; return s; };
  /** What guest number idx adds to links above chance, given everyone else at the table. */
  function marginalOf(members, idx) {
    return Core.tableScore(G, members).above - Core.tableScore(G, members.filter((_, j) => j !== idx)).above;
  }
  const signed = (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1);

  function seatOne(i, from) {
    if (!S.running) return;
    if (S.reveal) { toast('This is the best table we found. Press <b>R</b> to go back to yours.'); return; }
    if (i === S.host) { toast(`<b>${esc(nodes[i].name)}</b> is the host, and the host stays.`); return; }
    if (S.members.includes(i)) { unseatOne(i); return; }
    if (S.members.length - 1 >= S.k) { toast('Every seat is taken. Click a chair to send someone away, or <b>serve dinner</b>.'); return; }
    const before = S.score.above;
    const links = Core.linksInto(G, inSetOf(S.members), i);
    pushHistory();
    S.members.push(i);
    updateOne(true);
    syncAll();
    fly(i, from);
    const d = S.score.above - before;
    const n = nodes[i];
    if (d >= 0) {
      toast(`<b>${esc(n.name)}</b> sits down: ${links} link${links === 1 ? '' : 's'} to the table, ${n.deg} in all, <b>${signed(d)}</b> above chance.`, 4500);
    } else {
      toast(`<b>${esc(n.name)}</b> sits down and <b>costs ${Math.abs(d).toFixed(1)}</b>: only ${links} link${links === 1 ? '' : 's'} to the table, but ${n.deg} links in all, so chance expected more.`, 5500);
    }
  }

  function unseatOne(i) {
    if (!S.running || S.reveal || i === S.host || !S.members.includes(i)) return;
    pushHistory();
    S.members = S.members.filter(x => x !== i);
    updateOne(true);
    syncAll();
    toast(`<b>${esc(nodes[i].name)}</b> leaves the table.`);
  }

  /** One hint per table: the guest who would raise the score most right now. */
  function hintOne() {
    if (!S.running || S.reveal || S.mode !== 'one') return;
    if (S.hintsLeft <= 0) { toast('No hints left for this table.'); return; }
    if (S.members.length - 1 >= S.k) { toast('The table is full. Serve dinner, or send someone away first.'); return; }
    const inSet = inSetOf(S.members);
    let best = -1, bestG = -Infinity;
    for (const v of Core.candidates(G, S.members)) {
      const g = Core.addGain(G, inSet, S.score.dsum, v);
      if (g > bestG) { bestG = g; best = v; }
    }
    if (best < 0) return;
    S.hintsLeft--;
    syncHud();
    const l = Core.linksInto(G, inSet, best);
    toast(`<span style="color:var(--laurel);font-weight:600">HINT</span> <b>${esc(nodes[best].name)}</b> would add <b>${signed(bestG)}</b>: ` +
      `${l} link${l === 1 ? '' : 's'} to the table and ${nodes[best].deg} links in all, so chance expects little from them.`, 9000);
    highlightGuest(best);
    const row = guestRows.find(b => +b.dataset.i === best);
    if (row) row.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }

  function addTable() {
    if (S.tables.length >= MAX_TABLES) { toast('Nine tables is all this room holds.'); return -1; }
    S.tables.push({});
    S.active = S.tables.length - 1;
    S.stats = Core.tableStats(G, S.labels, S.tables.length);
    renderTables();
    syncHud();
    return S.active;
  }
  function removeTable(t) {
    for (const tt of S.seeds.values()) if (tt === t) return;
    pushHistory();
    S.tables.splice(t, 1);
    const ns = new Map();
    for (const [i, tt] of S.seeds) ns.set(i, tt > t ? tt - 1 : tt);
    S.seeds = ns;
    if (S.active >= S.tables.length) S.active = S.tables.length - 1;
    recompute(false);
    syncAll();
  }

  /** Seat guest i at the active table. `from` is where their portrait should fly in from (a DOMRect-like). */
  function seat(i, from) {
    if (S.mode === 'one') return seatOne(i, from);
    if (!S.running) return;
    if (S.reveal) { toast('This is Louvain&rsquo;s seating. Press <b>R</b> to go back to yours.'); return; }
    const n = nodes[i];
    if (S.seeds.has(i)) {
      const t = S.seeds.get(i);
      pushHistory();
      if (S.active < 0 || t === S.active) {
        S.seeds.delete(i);
        recompute(true); syncAll();
        toast(`<b>${esc(n.name)}</b> hands back the place card and leaves the chair.`);
      } else {
        S.seeds.set(i, S.active);
        recompute(true); syncAll(); fly(i, from);
        toast(`<b>${esc(n.name)}</b> moves to Table ${ROMAN[S.active]} and brings <b>${broughtBy(i)}</b> guests along.`);
      }
      return;
    }
    if (cardsLeft() <= 0) { toast('No place cards left. Click a chair to take a card back, or <b>serve dinner</b>.'); return; }
    if (S.tables.length === 0 || S.active < 0) { if (addTable() < 0) return; }
    // A new guest gets a new table unless you picked one: one table for everyone scores exactly zero,
    // so the default has to be the interesting move, not the trap.
    else if (!S.activeChosen && tableHasSeed(S.active) && S.tables.length < MAX_TABLES) addTable();
    pushHistory();
    S.seeds.set(i, S.active);
    S.activeChosen = false;
    const r = recompute(true);
    syncAll();
    fly(i, from);
    const others = r.changed - r.followed;
    const b = broughtBy(i);
    let hint = '';
    if (S.seeds.size >= 2 && S.q === 0) hint = ' <span style="color:var(--dim)">Everyone is at one table, so Q is 0: choose or open another table for the next guest.</span>';
    toast(`<b>${esc(n.name)}</b> sits down at Table ${ROMAN[S.seeds.get(i)]} and brings <b>${b}</b> guest${b === 1 ? '' : 's'} along` +
      (others > 0 ? `; ${others} elsewhere change tables.` : '.') + hint, hint ? 7000 : 4200);
  }
  const tableHasSeed = (t) => { for (const tt of S.seeds.values()) if (tt === t) return true; return false; };

  /** Take a place card back from a chair, whichever table is active. */
  function unseat(i) {
    if (S.mode === 'one') return unseatOne(i);
    if (!S.running || S.reveal || !S.seeds.has(i)) return;
    pushHistory();
    S.seeds.delete(i);
    recompute(true); syncAll();
    toast(`<b>${esc(nodes[i].name)}</b> hands back the place card and leaves the chair.`);
  }

  /** The portrait travels from wherever you picked the guest to their new chair. */
  function fly(i, from) {
    const chair = document.querySelector(`.chair[data-seat="${i}"]`);
    if (!chair) return;
    const to = chair.getBoundingClientRect();
    if (!from) from = { left: to.left - 60, top: to.top + 120, width: 40, height: 40 };
    const meta = IMAGES[nodes[i].id];
    const el = document.createElement(meta ? 'img' : 'div');
    el.className = 'fly';
    if (meta) el.src = meta.src; else el.textContent = nodes[i].name[0];
    el.style.left = (from.left + from.width / 2 - 20) + 'px';
    el.style.top = (from.top + from.height / 2 - 20) + 'px';
    document.body.appendChild(el);
    const dx = (to.left + to.width / 2) - (from.left + from.width / 2);
    const dy = (to.top + to.height / 2) - (from.top + from.height / 2);
    chair.style.visibility = 'hidden';
    const a = el.animate(
      [{ transform: 'translate(0,0) scale(1.5)', opacity: 0.95 }, { transform: `translate(${dx}px,${dy}px) scale(1)`, opacity: 1 }],
      { duration: 560, easing: 'cubic-bezier(.2,.8,.2,1)' });
    a.onfinish = () => { el.remove(); chair.style.visibility = ''; chair.classList.add('pop'); };
  }

  // ---------------------------------------------------------------- rounds
  function newRound(cards) {
    S.mode = 'room'; S.host = -1; S.members = []; S.score = null; S.rivals = null; S.revealMembers = null;
    $('stage').classList.remove('one');
    S.cards = cards;
    S.tables = []; S.active = -1; S.activeChosen = false; S.seeds = new Map(); S.history = [];
    S.labels = new Int32Array(N).fill(-1); S.disp = new Int32Array(N).fill(-1);
    S.q = 0; S.stats = Core.tableStats(G, S.labels, 1);
    S.attr = null; S.brought = null;
    S.running = true; S.finished = false; S.reveal = false;
    roundStartedAt = performance.now();
    anim.clear();
    $('bUndo').disabled = true;
    fitCamera();
    cam.x = cam.tx; cam.y = cam.ty; cam.z = cam.tz * 0.6;
    addTable();
    renderLadder();
    syncAll();
    closeResults();
    renderGuests();
    setAim(-1);
    toast(`<b>${cards}</b> place cards. Type a name to seat someone. Each new guest opens a new table, unless you click a table first to seat them there.`, 7000);
  }

  function fitCamera() {
    const vw = Math.max(200, VW - panelWidth()), vh = Math.max(200, VH);
    const spanX = bbox.x1 - bbox.x0 + 240, spanY = bbox.y1 - bbox.y0 + 240;
    cam.tz = clamp(Math.min(vw / spanX, vh / spanY), 0.1, 3);
    cam.tx = (bbox.x0 + bbox.x1) / 2;
    cam.ty = (bbox.y0 + bbox.y1) / 2;
  }

  // ---------------------------------------------------------------- finish
  function scoreBoxes(items) {
    $('scoreRow').innerHTML = items.map(x => `<div class="big ${x.c || ''}" title="${esc(x.t || '')}"><b>${x.v}</b><span>${x.l}</span></div>`).join('');
  }
  function nmiBoxes(items) {
    $('nmiRow').innerHTML = items.map(x => `<div class="nmi" title="${esc(x.t || '')}"><b>${x.v}</b><span>${x.l}</span></div>`).join('');
  }

  function finishOne() {
    if (S.members.length < 2) { toast('Seat at least one guest before you serve.'); return; }
    S.finished = true; S.running = false; S.reveal = false;
    const sc = Core.tableScore(G, S.members);
    const R = S.rivals;
    const you = sc.above, rec = R.recordScore.above, gre = R.greedyScore.above, rnd = R.random;
    const host = nodes[S.host].name;
    const guests = S.members.slice(1);
    const same = guests.filter(i => LOU[i] === LOU[S.host]).length;
    const louName = D.louvain.names[LOU[S.host]], louSize = louCentroid[LOU[S.host]].size;

    let verdict, line;
    if (you >= rec - 1e-6) {
      verdict = 'A PERFECT TABLE';
      line = `You matched the best table our search found around ${esc(host)}. Every seat earns its keep: nobody at the table costs more in links than they bring.`;
    } else if (you >= gre) {
      verdict = 'BETTER THAN THE GREEDY HOST';
      line = `The greedy host adds the safest guest each time and never looks two seats ahead. You did. The best table we found is still above you, at ${rec.toFixed(1)}.`;
    } else if (you > rnd) {
      verdict = 'A REAL TABLE';
      line = `Above a random pick from ${esc(host)}&rsquo;s friends, but the greedy host beats you. Look for chairs marked with a minus: those guests cost more in links than they bring.`;
    } else if (you > 0) {
      verdict = 'ABOVE CHANCE, BARELY';
      line = `A random pick from ${esc(host)}&rsquo;s friends does about as well. The links your guests have elsewhere are eating the links they have here.`;
    } else {
      verdict = 'BELOW CHANCE';
      line = `Your table holds fewer links than a shuffled network would give it. Hubs did this: every link a guest has anywhere raises what chance expects at your table.`;
    }
    $('wVerdict').textContent = verdict;
    $('wLine').innerHTML = line;
    $('scoreOf').textContent = 'Links above chance at each host’s table';
    scoreBoxes([
      { v: you.toFixed(1), l: 'You', c: 'you' + (you >= gre ? ' good' : ''), t: 'Your table' },
      { v: rec.toFixed(1), l: 'Best found', c: 'lou', t: 'The best table our swap search found around your host' },
      { v: gre.toFixed(1), l: 'Greedy host', t: 'Adds whoever raises the score most, one seat at a time' },
      { v: rnd.toFixed(1), l: 'Random', t: `${S.k} people linked to the host, drawn at random, average of 20` },
    ]);
    $('wNote').innerHTML = `<b>Links above chance</b> is one table&rsquo;s share of modularity, times the ${G.m} links in the network: ` +
      `the links inside the table minus what a shuffled network with the same links would put there. The <b>greedy host</b> adds the best next ` +
      `guest each time and never looks ahead; the <b>best found</b> swaps guests in and out until nothing improves.`;
    nmiBoxes([
      { v: `${sc.inside}`, l: `links inside · ${sc.expected.toFixed(1)} expected by chance` },
      { v: `${same} of ${guests.length}`, l: `sit at Louvain’s table for ${host} (${louName}, ${louSize})`, t: 'How many of your guests Louvain puts in the same community as your host' },
      { v: sc.q.toFixed(4), l: 'this table’s share of Q' },
    ]);
    const inSet = inSetOf(S.members);
    const rows = guests.map((i, j) => `${IMAGES[nodes[i].id] ? portraitTag(i, 'pimg') : ''}<em>${esc(nodes[i].name)}</em> &mdash; ` +
      `${Core.linksInto(G, inSet, i)} to the table, ${nodes[i].deg} in all, <span class="${marginalOf(S.members, j + 1) >= 0 ? 'm' : 'w'}">${signed(marginalOf(S.members, j + 1))}</span>`);
    const recSet = inSetOf(R.record);
    const recRows = R.record.slice(1).map(i => `${IMAGES[nodes[i].id] ? portraitTag(i, 'pimg') : ''}<em>${esc(nodes[i].name)}</em> &mdash; ` +
      `${Core.linksInto(G, recSet, i)} to the table, ${nodes[i].deg} in all`);
    $('wReport').innerHTML =
      `<span class="k">Your table · ${sc.inside} links inside</span>` + rows.join('<br>') + '<br>' +
      `<span class="k" style="margin-top:8px">The best table we found · ${R.recordScore.inside} links inside, ${R.recordScore.dsum} links in all</span>` + recRows.join('<br>');
    $('bReveal').textContent = 'SHOW THE BEST TABLE';
    $('vWin').classList.remove('gone');
    syncAll();
  }

  function finish() {
    if (S.mode === 'one') return finishOne();
    if (S.seeds.size === 0) { toast('Seat at least one guest before you serve.'); return; }
    S.finished = true; S.running = false; S.reveal = false;

    const you = S.q;
    const lou = D.louvain.q, gre = D.greedy.q, cen = D.century.q, shu = D.shuffle.mean;
    const T = Math.max(1, S.tables.length);
    const rnd = Core.randomSeatingAvg(G, [...S.seeds], T, 20, 5);
    const nmi = Core.nmi(S.labels, LOU);
    const dis = Core.disagreements(S.labels, LOU);
    const [lo, hi] = D.louvain.qRange;

    let verdict, line;
    if (you >= lo) {
      verdict = 'THE HOST OF HOSTS';
      line = `Your seating scores inside Louvain&rsquo;s own range. The algorithm runs twenty times on this network and lands between ${lo.toFixed(3)} and ${hi.toFixed(3)}; you got there with ${S.seeds.size} place cards.`;
    } else if (you >= gre) {
      verdict = 'MASTER OF CEREMONIES';
      line = `You beat the greedy merger, which starts with everyone alone and only ever joins tables together. Louvain also lets guests move between tables afterwards, and that is worth the gap still above you.`;
    } else if (you >= cen) {
      verdict = 'BETTER THAN THE HISTORIAN';
      line = `Seating everyone by the century they were born in scores ${cen.toFixed(3)}. You beat it, so your tables follow traditions more than birthdays. Louvain follows them better still.`;
    } else if (you >= shu) {
      verdict = 'AN HONEST SEATING';
      line = `You found real structure: a shuffled copy of this network, every philosopher keeping their number of links, scores about ${shu.toFixed(2)}. But the historian, who knows nothing except birth centuries, still beats you.`;
    } else {
      verdict = 'A CHAOTIC EVENING';
      line = `A shuffled network scores about ${shu.toFixed(2)}, and your room is below it. Usually one table has swallowed the whole room, or a card went to a guest whose friends all sit somewhere else.`;
    }

    $('wVerdict').textContent = verdict;
    $('wLine').innerHTML = line;
    $('scoreOf').textContent = 'Modularity Q of each host’s seating';
    scoreBoxes([
      { v: you.toFixed(3), l: 'You', c: 'you' + (you >= gre ? ' good' : ''), t: 'Your seating' },
      { v: lou.toFixed(3), l: 'Louvain', c: 'lou', t: 'Louvain, best of 20 runs on this network' },
      { v: gre.toFixed(3), l: 'Greedy', t: 'Greedy modularity: start with everyone alone, keep merging the two tables that raise Q most' },
      { v: cen.toFixed(3), l: 'By century', t: 'Seat everyone by the century they were born in' },
      { v: shu.toFixed(3), l: 'Shuffled', t: 'Louvain on a degree-preserving shuffle of the same network, average of 20' },
      { v: rnd.toFixed(3), l: 'Random', t: 'Your placed guests, everyone else seated at random, average of 20' },
    ]);
    $('wNote').innerHTML = `Same room, same links, five hosts. <b>Louvain</b> is what the course page runs, best of twenty seeds. ` +
      `<b>Greedy</b> merges tables and never splits. <b>By century</b> is the historian&rsquo;s seating. <b>Shuffled</b> is Louvain on a ` +
      `degree-preserving shuffle of the same network, the honest zero. <b>Random</b> keeps your placed guests and seats the rest by lot.`;
    nmiBoxes([
      { v: nmi.toFixed(2), l: 'you vs Louvain (NMI)', t: 'Normalized mutual information between your seating and Louvain’s: 1 means the same tables, 0 means no relation.' },
      { v: D.selfNmi.mean.toFixed(2), l: `Louvain vs Louvain (${D.selfNmi.min.toFixed(2)}–${D.selfNmi.max.toFixed(2)} over 20 runs)`, t: 'How much Louvain agrees with itself across random seeds' },
      { v: `${dis.nodes.length}`, l: 'seated differently', t: 'Guests you and Louvain seat at different tables, after matching each of your tables to the community it overlaps most.' },
    ]);
    $('bReveal').textContent = 'SHOW LOUVAIN’S SEATING';

    // the report: your tables, then the disagreements that matter
    const sizes = Array.from(S.stats.size).map((s, t) => ({ t, s })).filter(x => x.s > 0).sort((a, b) => b.s - a.s);
    const tableLine = sizes.map(x => {
      const head = topOfTable(x.t, 1)[0];
      return `<em style="color:${TABLE_COLORS[x.t]}">Table ${ROMAN[x.t]}</em> ${x.s}${head != null ? ` (${esc(nodes[head].name)})` : ''}`;
    }).join(' &middot; ');
    const louLine = D.louvain.names.map((nm, c) => `<em style="color:${TABLE_COLORS[c % 10]}">${esc(nm)}</em> ${louCentroid[c].size}`).join(' &middot; ');

    const contested = dis.nodes.filter(i => D.flips[i] >= 5).length;
    const shown = dis.nodes.slice().sort((a, b) => nodes[b].deg - nodes[a].deg).slice(0, 14);
    const rows = shown.map(i => {
      const yours = S.labels[i] >= 0 ? `Table ${ROMAN[S.labels[i]]}` : 'nowhere';
      const f = D.flips[i];
      return `${IMAGES[nodes[i].id] ? portraitTag(i, 'pimg') : ''}<em>${esc(nodes[i].name)}</em> &mdash; you: ${yours}, Louvain: ${esc(D.louvain.names[LOU[i]])}` +
        (f > 0 ? ` &middot; <span class="m">Louvain moves them in ${f} of its 19 other runs</span>` : ` &middot; <span class="w">Louvain never moves them</span>`);
    });
    let summary;
    if (dis.nodes.length === 0) summary = 'You and Louvain agree on every guest.';
    else summary = `Of the <em>${dis.nodes.length}</em> guests you and Louvain seat differently, <em>${contested}</em> are guests Louvain itself moves in at least ` +
      `5 of its own runs. Disagreeing with Louvain about those is not being wrong; it is finding the same soft borders the algorithm finds.`;
    $('wReport').innerHTML =
      `<span class="k">Your tables</span>${tableLine}<br>` +
      `<span class="k" style="margin-top:8px">Louvain&rsquo;s tables</span>${louLine}<br>` +
      `<span class="k" style="margin-top:8px">Seated differently &middot; the most linked ${shown.length}</span>${summary}<br>` + rows.join('<br>');

    $('vWin').classList.remove('gone');
    syncAll();
  }

  function topOfTable(t, k) {
    const m = [];
    for (let i = 0; i < N; i++) if (S.labels[i] === t) m.push(i);
    m.sort((a, b) => nodes[b].deg - nodes[a].deg || a - b);
    return m.slice(0, k);
  }

  function reveal(on) {
    S.reveal = on;
    $('vWin').classList.add('gone');
    if (S.mode === 'one') {
      S.revealMembers = on ? S.rivals.record : null;
      S.running = !on;
      updateOne(true);
      syncAll();
      toast(on ? `<b>The best table we found</b> around ${esc(nodes[S.host].name)}: ${S.rivals.recordScore.above.toFixed(1)} links above chance. Press <b>R</b> to return to yours.`
               : 'Back to your table.', on ? 8000 : 2500);
      return;
    }
    if (on) {
      S.running = false;
      toast(`<b>Louvain&rsquo;s seating</b>, ${LOU_K} tables named after who sits at them. Hover anyone to see where you put them. Press <b>R</b> to return to your own room.`, 9000);
    } else {
      S.running = true;
      toast('Back to your seating.');
    }
    syncAll();
  }

  // ---------------------------------------------------------------- canvas
  const cv = $('cv'), ctx = cv.getContext('2d');
  let VW = 0, VH = 0, DPR = 1;

  function resize() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    const r = cv.getBoundingClientRect();
    VW = r.width; VH = r.height;
    cv.width = Math.round(VW * DPR); cv.height = Math.round(VH * DPR);
  }
  window.addEventListener('resize', resize);

  const panelWidth = () => 0;                      // the map has its own box now; nothing overlaps it
  const viewCx = () => VW / 2;
  const viewCy = () => VH / 2;

  // Candle-light dust in world space.
  const dust = (() => {
    const rnd = Core.mulberry32(99);
    return Array.from({ length: 360 }, () => ({
      x: rnd() * WORLD * 1.6 - WORLD * 0.3, y: rnd() * WORLD * 1.6 - WORLD * 0.3,
      r: 0.5 + rnd() * 1.4, a: 0.03 + rnd() * 0.14,
    }));
  })();

  function draw(now, dt) {
    NOW = now;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, VW, VH);

    const k = 1 - Math.pow(0.0016, dt);
    cam.x = lerp(cam.x, cam.tx, k); cam.y = lerp(cam.y, cam.ty, k); cam.z = lerp(cam.z, cam.tz, k);
    stepAnim(now);

    const cx = viewCx(), cy = viewCy(), z = cam.z;
    for (let i = 0; i < N; i++) {
      SX[i] = (nodes[i].wx - cam.x) * z + cx;
      SY[i] = (nodes[i].wy - cam.y) * z + cy;
    }

    for (const d of dust) {
      const x = (d.x - cam.x) * z + cx, y = (d.y - cam.y) * z + cy;
      if (x < -20 || x > VW + 20 || y < -20 || y > VH + 20) continue;
      ctx.fillStyle = `rgba(240,200,120,${d.a})`;
      ctx.beginPath(); ctx.arc(x, y, d.r, 0, TAU); ctx.fill();
    }

    drawEdges();
    drawNodes(now);
    if (S.reveal) drawLouvainNames();
    if (S.running && anim.size > 0) drawInkHint(now);
  }

  const labelForEdge = (i) => S.reveal ? LOU[i] % TABLE_COLORS.length : S.disp[i];

  function drawEdges() {
    const lw = Math.max(0.45, 0.65 * cam.z);
    ctx.lineCap = 'round';
    // links between tables, or with an unseated end: faint
    ctx.lineWidth = lw;
    ctx.strokeStyle = 'rgba(216,207,192,0.085)';
    ctx.beginPath();
    for (let e = 0; e < edges.length; e++) {
      const u = edges[e][0], v = edges[e][1];
      const lu = labelForEdge(u), lv = labelForEdge(v);
      if (lu >= 0 && lu === lv) continue;
      if (!segVisible(u, v)) continue;
      ctx.moveTo(SX[u], SY[u]); ctx.lineTo(SX[v], SY[v]);
    }
    ctx.stroke();
    // links inside a table, in the table's colour
    ctx.lineWidth = lw * 1.15;
    for (let t = 0; t < TABLE_COLORS.length; t++) {
      ctx.strokeStyle = rgba(TABLE_COLORS[t], 0.30);
      ctx.beginPath();
      let any = false;
      for (let e = 0; e < edges.length; e++) {
        const u = edges[e][0], v = edges[e][1];
        if (labelForEdge(u) !== t || labelForEdge(v) !== t) continue;
        if (!segVisible(u, v)) continue;
        ctx.moveTo(SX[u], SY[u]); ctx.lineTo(SX[v], SY[v]); any = true;
      }
      if (any) ctx.stroke();
    }
    // the hovered guest's links, lit
    if (hoverNode >= 0) {
      ctx.strokeStyle = 'rgba(240,199,102,.75)';
      ctx.lineWidth = Math.max(0.9, 1.5 * cam.z);
      ctx.beginPath();
      for (const v of G.adj[hoverNode]) { ctx.moveTo(SX[hoverNode], SY[hoverNode]); ctx.lineTo(SX[v], SY[v]); }
      ctx.stroke();
    }
  }
  function segVisible(u, v) {
    if (Math.max(SX[u], SX[v]) < -40 || Math.min(SX[u], SX[v]) > VW + 40) return false;
    if (Math.max(SY[u], SY[v]) < -40 || Math.min(SY[u], SY[v]) > VH + 40) return false;
    return true;
  }

  function drawNodes(now) {
    const labels = [];
    const hoverSet = hoverNode >= 0 ? new Set(G.adj[hoverNode]) : null;
    for (let i = 0; i < N; i++) {
      const n = nodes[i];
      const x = SX[i], y = SY[i];
      const pop = popOf(i, now);
      const isSeed = !S.reveal && S.seeds.has(i);
      const isHover = i === hoverNode;
      // the guests you placed, and the one under the cursor, are always big enough to show a face
      let R = Math.max(1.6, n.r * cam.z) * pop;
      if (isSeed) R = Math.max(R, 11); else if (isHover) R = Math.max(R, 9);
      if (x < -R - 40 || x > VW + R + 40 || y < -R - 40 || y > VH + R + 40) continue;

      const isNb = hoverSet && hoverSet.has(i);
      const col = colorOf(i);

      if (isSeed || (isHover && S.running)) {
        const g = ctx.createRadialGradient(x, y, R * 0.5, x, y, R * 3.2);
        g.addColorStop(0, isHover ? 'rgba(240,199,102,.35)' : 'rgba(217,166,72,.22)');
        g.addColorStop(1, 'rgba(217,166,72,0)');
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(x, y, R * 3.2, 0, TAU); ctx.fill();
      }
      if (pop > 1.05) {                              // the ink arriving
        ctx.strokeStyle = rgba(col, 0.55 * (pop - 1));
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(x, y, R * 1.9, 0, TAU); ctx.stroke();
      }

      ctx.fillStyle = col;
      ctx.globalAlpha = (S.disp[i] < 0 && !S.reveal) ? 0.75 : 1;
      ctx.beginPath(); ctx.arc(x, y, R, 0, TAU); ctx.fill();
      ctx.globalAlpha = 1;
      // a face, for the guests you placed and the famous, once the node is big enough to show one
      if (R >= 7 && (isSeed || isHover || famousSet.has(i))) {
        const img = portraitOf(i);
        if (img) {
          const ir = R - 1.2;
          const s = Math.max(2 * ir / img.width, 2 * ir / img.height);
          const w = img.width * s, h = img.height * s;
          ctx.save();
          ctx.beginPath(); ctx.arc(x, y, ir, 0, TAU); ctx.clip();
          ctx.drawImage(img, x - w / 2, y - h * 0.42, w, h);
          ctx.restore();
          ctx.strokeStyle = col; ctx.lineWidth = Math.max(1.5, 2.6 * Math.min(cam.z, 1.2));
          ctx.beginPath(); ctx.arc(x, y, R, 0, TAU); ctx.stroke();
        }
      }
      if (isSeed) {
        ctx.strokeStyle = '#f0c766'; ctx.lineWidth = Math.max(1.4, 2.4 * Math.min(cam.z, 1.2));
        ctx.beginPath(); ctx.arc(x, y, R + 2.2, 0, TAU); ctx.stroke();
      } else if (isHover || isNb) {
        ctx.strokeStyle = isHover ? '#fff2d0' : 'rgba(240,199,102,.8)'; ctx.lineWidth = isHover ? 2 : 1;
        ctx.beginPath(); ctx.arc(x, y, R + 1, 0, TAU); ctx.stroke();
      } else if (R > 4) {
        ctx.strokeStyle = 'rgba(12,6,9,.55)'; ctx.lineWidth = 0.8;
        ctx.beginPath(); ctx.arc(x, y, R, 0, TAU); ctx.stroke();
      }

      const prio = isHover ? 4 : isSeed ? 3 : (VW >= 520 && famousSet.has(i)) ? (n.deg >= 90 ? 2 : 1) : 0;
      if (prio > 0) labels.push({ i, x, y, R, prio });
    }

    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    labels.sort((a, b) => b.prio - a.prio || b.R - a.R);
    const taken = [];
    const zf = Math.max(0.8, Math.min(cam.z, 1.4));
    for (const L of labels) {
      const big = L.prio >= 3;
      const fs = clamp((big ? 15 : L.prio === 2 ? 13 : 11.5) * zf, 10, 20);
      ctx.font = `${big ? '600' : '500'} ${fs}px "Cormorant Garamond", Georgia, serif`;
      const txt = nodes[L.i].name;
      const w = ctx.measureText(txt).width, y = L.y - L.R - 4;
      const box = { x0: L.x - w / 2 - 3, x1: L.x + w / 2 + 3, y0: y - fs - 1, y1: y + 2 };
      let hit = false;
      for (const t of taken) if (box.x0 < t.x1 && box.x1 > t.x0 && box.y0 < t.y1 && box.y1 > t.y0) { hit = true; break; }
      if (hit && L.prio < 4) continue;
      taken.push(box);
      ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(12,6,9,.92)';
      ctx.strokeText(txt, L.x, y);
      ctx.fillStyle = L.i === hoverNode ? '#fff2d0' : big ? '#f0c766' : 'rgba(241,230,210,.85)';
      ctx.fillText(txt, L.x, y);
      // under a placed guest: how many they brought to the table
      if (S.mode === 'room' && !S.reveal && S.seeds.has(L.i)) {
        ctx.save();
        ctx.textBaseline = 'top';
        ctx.font = `600 ${Math.max(9, fs * 0.72)}px Cinzel, Georgia, serif`;
        const tag = `+${broughtBy(L.i)}`;
        ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(12,6,9,.92)';
        ctx.strokeText(tag, L.x, L.y + L.R + 3);
        ctx.fillStyle = TABLE_COLORS[S.seeds.get(L.i)];
        ctx.fillText(tag, L.x, L.y + L.R + 3);
        ctx.restore();
      }
    }
  }

  function drawLouvainNames() {
    if (VW < 520) return;                          // the small map cannot fit nine names; the room shows them
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    const fs = clamp(22 * Math.max(0.7, Math.min(cam.z, 1.3)), 14, 30);
    ctx.font = `600 ${fs}px Cinzel, Georgia, serif`;
    for (let c = 0; c < LOU_K; c++) {
      if (louCentroid[c].size < 20) continue;
      const x = (louCentroid[c].x - cam.x) * cam.z + viewCx(), y = (louCentroid[c].y - cam.y) * cam.z + viewCy();
      const txt = D.louvain.names[c].toUpperCase();
      ctx.lineWidth = 6; ctx.strokeStyle = 'rgba(12,6,9,.9)';
      ctx.strokeText(txt, x, y);
      ctx.fillStyle = TABLE_COLORS[c % 10];
      ctx.fillText(txt, x, y);
    }
  }

  function drawInkHint() {
    if (S.mode !== 'room') return;
    // a quiet line at the bottom while the room is still settling
    ctx.font = '500 13px "Cormorant Garamond", Georgia, serif';
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillStyle = 'rgba(191,169,143,.7)';
    ctx.fillText(`${anim.size} guests still finding their table…`, 14, VH - 10);
  }

  // ----------------------------------------------------------------- input
  let drag = null;

  function nodeAt(sx, sy) {
    let best = -1, bestD = 1e9;
    for (let i = 0; i < N; i++) {
      const R = Math.max(7, nodes[i].r * cam.z) + 3;
      const dx = SX[i] - sx, dy = SY[i] - sy;
      if (Math.abs(dx) > R || Math.abs(dy) > R) continue;
      const d = Math.hypot(dx, dy);
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
      if (h !== hoverNode) { hoverNode = h; setAim(h); highlightGuest(h); }
      cv.classList.toggle('seat', h >= 0 && S.running);
    }
  });
  cv.addEventListener('pointerup', (e) => {
    cv.classList.remove('grabbing');
    const wasDrag = drag && drag.moved > 5;
    drag = null;
    if (wasDrag) return;
    if (performance.now() - roundStartedAt < 400) return;   // the click that opened the doors is not a seating
    const r = cv.getBoundingClientRect();
    const i = nodeAt(e.clientX - r.left, e.clientY - r.top);
    if (i >= 0) seat(i, { left: e.clientX - 20, top: e.clientY - 20, width: 40, height: 40 });
  });
  cv.addEventListener('pointerleave', () => { hoverNode = -1; setAim(-1); highlightGuest(-1); cv.classList.remove('grabbing'); });
  cv.addEventListener('wheel', (e) => {
    e.preventDefault();
    cam.tz = clamp(cam.tz * Math.exp(-e.deltaY * 0.0016), 0.1, 3);
  }, { passive: false });

  window.addEventListener('keydown', (e) => {
    const inSearch = document.activeElement === $('search') || e.target === $('search');
    if (!$('vStart').classList.contains('gone')) { if (e.key === 'Enter') $('bStart').click(); return; }
    if (inSearch) return;
    if (e.key === '/') { e.preventDefault(); $('search').focus(); return; }
    if (e.key === 'c' || e.key === 'C') fitCamera();
    if (e.key === 'm' || e.key === 'M') setMapMode(!$('stage').classList.contains('mapmode'));
    if (e.key === 't' || e.key === 'T') { if (S.mode === 'room' && S.running && !S.reveal && addTable() >= 0) { S.activeChosen = true; syncSearchHint(); } }
    if (e.key === 'h' || e.key === 'H') hintOne();
    if (e.key === 'z' || e.key === 'Z') { if (S.running && !S.reveal) undo(); }
    if (e.key === 'r' || e.key === 'R') { if (S.finished) reveal(!S.reveal); }
    if (e.key === 'Enter' || e.key === 'Return') { if (S.running && !S.reveal && $('vWin').classList.contains('gone')) finish(); }
    if (e.key === 'Escape') { if (S.reveal) reveal(false); }
  });

  // -------------------------------------------------------------------- UI
  let toastTimer = null;
  function toast(html, ms) {
    const t = $('toast');
    t.innerHTML = html; t.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), ms || 3200);
  }

  function syncHud() {
    if (S.mode === 'one') {
      const sc = S.score || Core.tableScore(G, S.members);
      const left = S.k - (S.members.length - 1);
      $('sQ').textContent = sc.above.toFixed(1); $('sQl').textContent = 'above chance';
      $('sCards').textContent = left; $('sCardsl').textContent = 'seats';
      $('sSeated').textContent = sc.inside; $('sSeatedl').textContent = 'inside';
      const obj = $('objective');
      obj.classList.toggle('done', S.reveal || (S.finished && !S.running));
      if (S.reveal) $('objText').textContent = `THE BEST TABLE WE FOUND · ${S.rivals.recordScore.above.toFixed(1)} · R TO RETURN`;
      else if (S.finished && !S.running) $('objText').textContent = `DINNER SERVED · ${sc.above.toFixed(1)} ABOVE CHANCE`;
      else if (left > 0) $('objText').textContent = `${nodes[S.host].name.toUpperCase()}’S TABLE · ${left} SEAT${left === 1 ? '' : 'S'} LEFT`;
      else $('objText').textContent = 'TABLE FULL · SERVE DINNER';
      $('bServe').disabled = !S.running || S.members.length < 2;
      $('bHint').disabled = !S.running || S.reveal || S.hintsLeft <= 0 || left === 0;
      $('hintN').textContent = S.hintsLeft;
      return;
    }
    $('sQl').textContent = 'Q'; $('sCardsl').textContent = 'Cards'; $('sSeatedl').textContent = 'Seated';
    $('sQ').textContent = S.q.toFixed(3);
    $('sCards').textContent = cardsLeft();
    let seated = 0;
    for (let i = 0; i < N; i++) if (S.labels[i] >= 0) seated++;
    $('sSeated').textContent = seated;
    const obj = $('objective');
    if (S.reveal) {
      obj.classList.add('done');
      $('objText').textContent = `LOUVAIN’S SEATING · Q ${D.louvain.q.toFixed(3)} · R TO RETURN`;
    } else if (S.finished && !S.running) {
      obj.classList.add('done');
      $('objText').textContent = `DINNER SERVED · Q ${S.q.toFixed(3)}`;
    } else {
      obj.classList.remove('done');
      const c = cardsLeft();
      $('objText').textContent = c > 0 ? `SEAT THE ROOM · ${c} PLACE CARD${c === 1 ? '' : 'S'} LEFT` : 'NO CARDS LEFT · SERVE DINNER';
    }
    $('bServe').disabled = !S.running || S.seeds.size === 0;
    $('bTable').disabled = !S.running || S.reveal || S.tables.length >= MAX_TABLES;
  }

  /** One round table: the top with the head count, and the chairs around it. */
  function roundTable(size, share, chairs, color, topHtml, Dfixed) {
    const D = Dfixed || Math.round(118 + 92 * Math.sqrt(size / N));   // a table that seats the room gets big
    const W = D + 2 * 46;
    const r = D / 2 + 25;
    const n = chairs.length;
    const html = chairs.map((c, k) => {
      const a = -Math.PI / 2 + k * 2 * Math.PI / Math.max(1, n);
      const x = (W / 2 + r * Math.cos(a)).toFixed(1), y = (W / 2 + r * Math.sin(a)).toFixed(1);
      return c.replace('%POS%', `left:${x}px;top:${y}px`);
    }).join('');
    const q = share == null ? '' : `<span class="qs">Q ${share >= 0 ? '+' : ''}${share.toFixed(3)}</span>`;
    const top = topHtml || `<span class="cnt${size >= 1000 ? '' : ' big'}">${size}</span><span class="lbl">${size === 1 ? 'guest' : 'guests'}</span>${size ? q : ''}`;
    return `<div class="round" style="width:${W}px;height:${W}px;--tc:${color}">` +
      `<div class="top" style="width:${D}px;height:${D}px">${top}</div>${html}</div>`;
  }

  /** ONE TABLE: the host at the head, your guests around, each chair with what they add. */
  function renderOneTable(box) {
    const members = S.reveal && S.revealMembers ? S.revealMembers : S.members;
    const sc = Core.tableScore(G, members);
    const inSet = inSetOf(members);
    const color = TABLE_COLORS[0];
    const chairs = members.map((i, idx) => {
      const meta = IMAGES[nodes[i].id];
      const links = Core.linksInto(G, inSet, i);
      const marg = idx === 0 ? null : marginalOf(members, idx);
      const badge = idx === 0 ? 'HOST' : signed(marg);
      const cls = 'chair' + (idx === 0 ? ' host' : '') + (marg != null && marg < 0 ? ' neg' : '');
      const title = idx === 0
        ? `${nodes[i].name}, your host: ${links} links to the table, ${nodes[i].deg} links in all`
        : `${nodes[i].name}: ${links} link${links === 1 ? '' : 's'} to the table, ${nodes[i].deg} in all, ${badge} to the score.${S.reveal ? '' : ' Click to send them away.'}`;
      return `<button class="${cls}" data-who="${i}" ${idx === 0 || S.reveal ? '' : `data-seat="${i}"`} style="%POS%" title="${esc(title)}">` +
        (meta ? `<img src="${meta.src}" alt="">` : `<span class="ini">${esc(nodes[i].name[0])}</span>`) + `<span class="n">${badge}</span></button>`;
    });
    if (!S.reveal) for (let s = members.length - 1; s < S.k; s++) {
      chairs.push(`<button class="chair empty" data-empty="0" style="%POS%" title="An empty seat: type a name, or pick someone linked to the table">+</button>`);
    }
    const top = `<span class="cnt big">${sc.above.toFixed(1)}</span><span class="lbl">links above chance</span>` +
      `<span class="qs">${sc.inside} inside &middot; ${sc.expected.toFixed(1)} by chance</span>`;
    const card = document.createElement('div');
    card.className = 'tcard solo on';
    card.style.setProperty('--tc', color);
    const head = S.reveal ? 'THE BEST TABLE WE FOUND' : `${nodes[S.host].name.toUpperCase()}’S TABLE`;
    const guests = members.slice(1);
    const who = guests.length
      ? guests.map((i, j) => `<b>${esc(nodes[i].name)}</b> ${signed(marginalOf(members, j + 1))}`).join(' &middot; ')
      : 'Nobody yet. Type a name, or pick someone from the list of people linked to the table.';
    card.innerHTML = `<div class="thead"><span class="sw" style="background:${color}"></span><span>${esc(head)}</span></div>` +
      roundTable(members.length, null, chairs, color, top, 196 + 6 * S.k) +
      `<div class="tsub">${who}</div>`;
    card.onclick = (e) => {
      const ch = e.target.closest('.chair');
      if (!ch) return;
      if (ch.dataset.seat !== undefined) unseatOne(+ch.dataset.seat);
      else if (ch.dataset.empty !== undefined) $('search').focus();
    };
    card.onmouseover = (e) => { const ch = e.target.closest('.chair'); if (ch && ch.dataset.who !== undefined) { hoverNode = +ch.dataset.who; setAim(hoverNode); } };
    card.onmouseout = (e) => { const ch = e.target.closest('.chair'); if (ch && ch.dataset.who !== undefined) { hoverNode = -1; setAim(-1); } };
    box.appendChild(card);
  }

  function renderTables() {
    const box = $('tables');
    box.innerHTML = '';
    if (S.mode === 'one') { renderOneTable(box); return; }
    if (S.reveal) { renderLouvainTables(box); return; }
    if (S.tables.length === 0) {
      box.innerHTML = '<div class="roomnote">No table open yet. Press <b>+ Table</b>, or type a name above and a table opens for them.</div>';
      return;
    }
    const st = S.stats;
    S.tables.forEach((_, t) => {
      const color = TABLE_COLORS[t];
      const seeds = [...S.seeds].filter(([, tt]) => tt === t).map(([i]) => i).sort((a, b) => broughtBy(b) - broughtBy(a) || a - b);
      const size = st ? st.size[t] : 0;
      const share = st ? st.share[t] : 0;
      const chairs = seeds.map(i => {
        const meta = IMAGES[nodes[i].id];
        const b = broughtBy(i);
        return `<button class="chair" data-seat="${i}" style="%POS%" title="${esc(nodes[i].name)} brought ${b} guests. Click to take the card back.">` +
          (meta ? `<img src="${meta.src}" alt="">` : `<span class="ini">${esc(nodes[i].name[0])}</span>`) +
          `<span class="n">+${b}</span></button>`;
      });
      if (cardsLeft() > 0 && S.running) chairs.push(`<button class="chair empty" data-empty="${t}" style="%POS%" title="Seat someone here: choose this table and type a name">+</button>`);
      const card = document.createElement('div');
      card.className = 'tcard' + (t === S.active ? ' on' : '') + (size === 0 ? ' empty' : '');
      card.style.setProperty('--tc', color);
      const seedNames = new Set(seeds.map(i => nodes[i].name));
      const also = topOfTable(t, 6).map(i => nodes[i].name).filter(nm => !seedNames.has(nm)).slice(0, 3);
      const whoLine = seeds.length
        ? seeds.map(i => `<b>${esc(nodes[i].name)}</b> +${broughtBy(i)}`).join(' &middot; ')
        : 'Nobody yet. Choose this table and type a name.';
      card.innerHTML =
        `<div class="thead"><span class="sw" style="background:${color}"></span><span>Table ${ROMAN[t]}</span>` +
        (size === 0 && S.tables.length > 1 ? `<button class="x" data-x="${t}" title="Close this empty table">&#10005;</button>` : '') + `</div>` +
        roundTable(size, share, chairs, color) +
        `<div class="tsub">${whoLine}</div>` +
        (also.length ? `<div class="tsub" style="margin-top:2px;color:var(--dim2)">with ${also.map(esc).join(', ')}&hellip;</div>` : '');
      card.onclick = (e) => {
        const x = e.target.closest('.x');
        if (x) { removeTable(+x.dataset.x); return; }
        const ch = e.target.closest('.chair');
        if (ch && ch.dataset.seat !== undefined) { unseat(+ch.dataset.seat); return; }
        S.active = t; S.activeChosen = true; renderTables(); syncSearchHint();
        if (ch && ch.dataset.empty !== undefined) $('search').focus();
      };
      card.onmouseover = (e) => { const ch = e.target.closest('.chair'); if (ch && ch.dataset.seat !== undefined) { hoverNode = +ch.dataset.seat; setAim(hoverNode); } };
      card.onmouseout = (e) => { const ch = e.target.closest('.chair'); if (ch && ch.dataset.seat !== undefined) { hoverNode = -1; setAim(-1); } };
      box.appendChild(card);
    });
  }

  /** In the reveal, the room shows Louvain's tables instead, with the most linked guests in the chairs. */
  function renderLouvainTables(box) {
    const order = Array.from({ length: LOU_K }, (_, c) => c).sort((a, b) => louCentroid[b].size - louCentroid[a].size);
    for (const c of order) {
      const color = TABLE_COLORS[c % TABLE_COLORS.length];
      const size = louCentroid[c].size;
      const chairs = D.louvain.top[c].slice(0, Math.min(5, size)).map(i => {
        const meta = IMAGES[nodes[i].id];
        return `<button class="chair" data-lou="${i}" style="%POS%" title="${esc(nodes[i].name)}, ${nodes[i].deg} links">` +
          (meta ? `<img src="${meta.src}" alt="">` : `<span class="ini">${esc(nodes[i].name[0])}</span>`) + `</button>`;
      });
      const card = document.createElement('div');
      card.className = 'tcard';
      card.style.setProperty('--tc', color);
      card.innerHTML =
        `<div class="thead"><span class="sw" style="background:${color}"></span><span>${esc(D.louvain.names[c])}</span></div>` +
        roundTable(size, null, chairs, color) +
        `<div class="tsub">${D.louvain.top[c].slice(0, 5).map(i => esc(nodes[i].name)).join(', ')}</div>`;
      card.onmouseover = (e) => { const ch = e.target.closest('.chair'); if (ch) { hoverNode = +ch.dataset.lou; setAim(hoverNode); } };
      card.onmouseout = (e) => { const ch = e.target.closest('.chair'); if (ch) { hoverNode = -1; setAim(-1); } };
      box.appendChild(card);
    }
  }

  function syncSearchHint() {
    const s = $('search');
    if (S.mode === 'one' && S.host >= 0) {
      s.placeholder = `Who joins ${nodes[S.host].name}’s table? Type a name…  (Enter seats the first match)`;
      return;
    }
    s.placeholder = S.active >= 0 && S.tables.length
      ? `Who sits at Table ${ROMAN[S.active]}? Type a name…  (Enter seats the first match)`
      : 'Type a name…  (press /, Enter seats the first match)';
  }

  const ladderMax = () => S.mode === 'one' && S.rivals ? Math.max(1, S.rivals.recordScore.above * 1.15) : LADDER_MAX;
  const ladderValue = () => S.mode === 'one' ? Math.max(0, S.score ? S.score.above : 0) : S.q;
  function renderLadder() {
    const bar = $('ladderBar');
    const max = ladderMax();
    const marks = S.mode === 'one' && S.rivals
      ? [['random', S.rivals.random, 'below'], ['greedy', S.rivals.greedyScore.above, ''], ['best found', S.rivals.recordScore.above, 'lou below']]
      : [['shuffled', D.shuffle.mean, ''], ['century', D.century.q, 'below'], ['greedy', D.greedy.q, ''], ['Louvain', D.louvain.q, 'lou below']];
    const fmt = (q) => S.mode === 'one' ? q.toFixed(1) : q.toFixed(2);
    bar.innerHTML = marks.map(([n, q, c]) => `<div class="tick ${c}" style="left:${clamp(q / max * 100, 0, 100).toFixed(1)}%"><span>${n} ${fmt(q)}</span></div>`).join('') +
      `<div class="you" id="ladderYou" style="left:${clamp(ladderValue() / max * 100, 0, 100).toFixed(1)}%"></div>`;
  }
  function syncLadder() {
    const y = $('ladderYou');
    if (y) y.style.left = clamp(ladderValue() / ladderMax() * 100, 0, 100).toFixed(1) + '%';
    const q = S.q;
    let cap;
    if (S.mode === 'one') {
      const v = S.score ? S.score.above : 0, R = S.rivals;
      if (S.members.length < 2) cap = 'Links above chance at your table, against a random pick, the greedy host and the best table we found.';
      else if (R && v >= R.recordScore.above - 1e-6) cap = 'You matched the best table we know of. Serve dinner.';
      else if (R && v >= R.greedyScore.above) cap = 'Ahead of the greedy host. The best table found is above you.';
      else if (R && v > R.random) cap = 'Ahead of a random pick. The greedy host is next.';
      else cap = 'Below a random pick: someone at the table costs more than they bring.';
      $('ladderCap').textContent = cap;
      return;
    }
    if (S.seeds.size === 0) cap = 'Your Q against the rival hosts. Louvain is the one to beat.';
    else if (q >= D.louvain.qRange[0]) cap = 'You are in Louvain’s own range. Serve dinner.';
    else if (q >= D.greedy.q) cap = 'Ahead of the greedy merger. Louvain is still above you.';
    else if (q >= D.century.q) cap = 'Ahead of seating by century. Greedy is next.';
    else if (q >= D.shuffle.mean) cap = 'Above the shuffled network, so this is real structure. The historian is next.';
    else cap = 'Below a shuffled network. One table is probably swallowing the room.';
    $('ladderCap').textContent = cap;
  }

  function setAim(i) {
    const por = $('hPortrait');
    if (i < 0) {
      $('hKick').textContent = 'Under your gaze';
      $('hName').textContent = 'Nobody yet';
      $('hDesc').textContent = 'Hover a philosopher on the map, or find one in the list below. Click to seat them at the active table.';
      $('hChips').innerHTML = '';
      $('hCredit').textContent = '';
      por.classList.remove('has'); por.title = ''; $('hInitial').textContent = '?';
      return;
    }
    const n = nodes[i];
    const seed = S.seeds.has(i);
    const meta = IMAGES[n.id];
    if (meta) {
      const im = $('hImg');
      if (im.getAttribute('src') !== meta.src) im.src = meta.src;
      por.classList.add('has');
      por.title = 'Wikipedia, ' + meta.file.replace(/_/g, ' ');
      $('hCredit').textContent = 'Portrait: Wikipedia, ' + meta.file.replace(/_/g, ' ');
    } else {
      por.classList.remove('has'); por.title = '';
      $('hInitial').textContent = n.name[0];
      $('hCredit').textContent = 'No portrait on the Wikipedia article';
    }
    $('hKick').textContent = seed ? 'Placed by you' : S.reveal ? 'At Louvain’s table' : 'Under your gaze';
    $('hName').textContent = n.name;
    $('hDesc').textContent = n.desc || 'No description in the snapshot.';
    let chips = `<span class="chip">${n.deg} links</span><span class="chip era">${D.eraShort[n.era]}</span>`;
    if (n.sub) chips += `<span class="chip era">${esc(n.sub.split(',')[0])}</span>`;
    if (S.mode === 'one') {
      const members = S.reveal && S.revealMembers ? S.revealMembers : S.members;
      if (i === S.host) chips += `<span class="chip card">the host</span>`;
      else if (members.includes(i)) chips += `<span class="chip tbl" style="background:${TABLE_COLORS[0]}">at the table</span>`;
      else { const l = Core.linksInto(G, inSetOf(members), i); chips += `<span class="chip">${l} link${l === 1 ? '' : 's'} to the table</span>`; }
    } else {
      if (S.labels[i] >= 0) chips += `<span class="chip tbl" style="background:${TABLE_COLORS[S.labels[i]]}">Table ${ROMAN[S.labels[i]]}</span>`;
      else if (S.seeds.size > 0) chips += `<span class="chip">not seated</span>`;
      if (seed) chips += `<span class="chip card">place card &middot; brought ${broughtBy(i)}</span>`;
      else if (S.attr && S.attr[i] >= 0 && !S.reveal) chips += `<span class="chip">followed ${esc(nodes[S.attr[i]].name)}</span>`;
    }
    if (S.finished || S.reveal) {
      chips += `<span class="chip tbl" style="background:${TABLE_COLORS[LOU[i] % 10]}">Louvain: ${esc(D.louvain.names[LOU[i]])}</span>`;
      if (D.flips[i] > 0) chips += `<span class="chip">moves in ${D.flips[i]}/19 runs</span>`;
    }
    $('hChips').innerHTML = chips;
  }

  // The search box: matches drop down under it; Enter seats the first one.
  let resultRows = [];
  function renderResults(query) {
    const box = $('results');
    const q = (query || '').trim().toLowerCase();
    box.innerHTML = '';
    resultRows = [];
    if (!q) { box.classList.remove('open'); return; }
    let list = [];
    for (let i = 0; i < N && list.length < 80; i++) if (nodes[i].name.toLowerCase().includes(q)) list.push(i);
    list.sort((a, b) => nodes[b].deg - nodes[a].deg);
    list = list.slice(0, 8);
    if (!list.length) {
      box.innerHTML = '<div class="row" style="color:var(--dim);font-style:italic">Nobody by that name among the 1374.</div>';
      box.classList.add('open');
      return;
    }
    list.forEach((i, k) => {
      const b = document.createElement('button');
      b.className = 'row' + (k === 0 ? ' first' : '') + (S.seeds.has(i) ? ' seated' : '');
      b.dataset.i = i;
      let where;
      if (S.mode === 'one') {
        if (i === S.host) where = 'the host';
        else if (S.members.includes(i)) where = 'at the table';
        else { const l = Core.linksInto(G, inSetOf(S.members), i); where = `${l} link${l === 1 ? '' : 's'} to the table`; }
      } else {
        where = S.seeds.has(i) ? `placed at Table ${ROMAN[S.seeds.get(i)]}` : S.labels[i] >= 0 ? `sits at Table ${ROMAN[S.labels[i]]}` : 'not seated yet';
      }
      b.innerHTML = `${portraitTag(i, 'pimg')}<span class="nm">${esc(nodes[i].name)}<small>${D.eraShort[nodes[i].era]} &middot; ${nodes[i].deg} links &middot; ${where}</small></span>` +
        (k === 0 ? '<span class="key">ENTER</span>' : '');
      b.onclick = () => { const from = (b.querySelector('.pimg, .pini') || b).getBoundingClientRect(); closeResults(); seat(i, from); };
      b.onmouseenter = () => { hoverNode = i; setAim(i); };
      b.onmouseleave = () => { hoverNode = -1; setAim(-1); };
      box.appendChild(b);
      resultRows.push(b);
    });
    box.classList.add('open');
  }
  function closeResults() {
    $('search').value = '';
    $('results').classList.remove('open');
    $('results').innerHTML = '';
    resultRows = [];
  }

  let guestRows = [];
  function renderGuests() {
    const box = $('guests');
    box.innerHTML = '';
    guestRows = [];
    if (S.mode === 'one' && S.host >= 0) {
      // everyone linked to the table, the ones with the most links to it first, the modest before the hubs
      const members = S.reveal && S.revealMembers ? S.revealMembers : S.members;
      const inSet = inSetOf(members);
      const cand = Core.candidates(G, members).map(i => ({ i, l: Core.linksInto(G, inSet, i) }));
      cand.sort((a, b) => b.l - a.l || nodes[a.i].deg - nodes[b.i].deg || a.i - b.i);
      $('guestsHead').textContent = 'Linked to the table';
      $('guestsNote').textContent = `${cand.length} people · to the table · in all`;
      for (const { i, l } of cand.slice(0, 90)) {
        const b = document.createElement('button');
        b.className = 'row';
        b.dataset.i = i;
        b.innerHTML = `${portraitTag(i, 'pimg')}<span class="nm">${esc(nodes[i].name)}<small>${D.eraShort[nodes[i].era]}</small></span>` +
          `<span class="two"><b>${l}</b><span>${nodes[i].deg}</span></span>`;
        b.onclick = () => seat(i, (b.querySelector('.pimg, .pini') || b).getBoundingClientRect());
        b.onmouseenter = () => { hoverNode = i; setAim(i); };
        b.onmouseleave = () => { hoverNode = -1; setAim(-1); };
        box.appendChild(b);
        guestRows.push(b);
      }
      syncGuestRows();
      return;
    }
    $('guestsHead').textContent = 'Who’s who';
    $('guestsNote').textContent = 'by links, top 80 · click to seat';
    const list = D.famous;
    for (const i of list) {
      const b = document.createElement('button');
      b.className = 'row';
      b.dataset.i = i;
      b.innerHTML = `<span class="dot"></span>${portraitTag(i, 'pimg')}<span class="nm">${esc(nodes[i].name)}</span>` +
        `<span class="era">${D.eraShort[nodes[i].era]}</span><span class="dg">${nodes[i].deg}</span>`;
      b.onclick = () => seat(i, (b.querySelector('.pimg, .pini') || b).getBoundingClientRect());
      b.onmouseenter = () => { hoverNode = i; setAim(i); };
      b.onmouseleave = () => { hoverNode = -1; setAim(-1); };
      box.appendChild(b);
      guestRows.push(b);
    }
    syncGuestRows();
  }
  function syncGuestRows() {
    for (const b of guestRows) {
      const i = +b.dataset.i;
      const dot = b.firstElementChild;
      dot.style.background = colorOf(i);
      b.classList.toggle('seated', S.seeds.has(i));
    }
  }
  function highlightGuest(i) {
    for (const b of guestRows) b.classList.toggle('hl', +b.dataset.i === i);
  }

  function syncAll() {
    syncHud();
    renderTables();
    syncSearchHint();
    syncLadder();
    if (S.mode === 'one') renderGuests(); else syncGuestRows();
    if (hoverNode >= 0) setAim(hoverNode);
  }

  // ------------------------------------------------------------ start screen
  let chosen = 12, chosenK = 5, chosenMode = 'one', chosenHost = -1;
  function pickHost() {
    const pool = D.famous.slice(0, 60);
    let h = pool[Math.floor(Math.random() * pool.length)];
    if (h === chosenHost) h = pool[(pool.indexOf(h) + 1) % pool.length];
    chosenHost = h;
  }
  function buildStart() {
    const m = $('modePick');
    m.innerHTML = '';
    Object.entries(MODES).forEach(([key, M]) => {
      const b = document.createElement('button');
      b.className = 'mode' + (key === chosenMode ? ' sel' : '');
      b.innerHTML = `<b>${M.label}</b><span>${M.note}</span>`;
      b.onclick = () => { chosenMode = key; buildStart(); };
      m.appendChild(b);
    });
    const one = chosenMode === 'one';
    $('rulesOne').style.display = one ? '' : 'none';
    $('rulesRoom').style.display = one ? 'none' : '';
    const hp = $('hostPick');
    if (one) {
      if (chosenHost < 0) pickHost();
      const meta = IMAGES[nodes[chosenHost].id];
      hp.innerHTML = `<span>Your guest:</span>${meta ? `<img src="${meta.src}" alt="">` : `<span class="hini">${esc(nodes[chosenHost].name[0])}</span>`}` +
        `<b>${esc(nodes[chosenHost].name)}</b><span>${D.eraShort[nodes[chosenHost].era]}, ${nodes[chosenHost].deg} links</span>` +
        `<button class="ghost" id="bReroll">Another guest</button>`;
      $('bReroll').onclick = () => { pickHost(); buildStart(); };
    } else hp.innerHTML = '';
    const d = $('diffPick');
    d.innerHTML = '';
    const opts = one ? SEATS : LEVELS;
    Object.entries(opts).forEach(([key, L]) => {
      const b = document.createElement('button');
      b.className = 'diff' + (+key === (one ? chosenK : chosen) ? ' sel' : '');
      b.innerHTML = `${L.label}<small>${L.note}</small>`;
      b.onclick = () => { if (one) chosenK = +key; else chosen = +key; buildStart(); };
      d.appendChild(b);
    });
  }
  function showStart() {
    buildStart();
    $('vStart').classList.remove('gone');
    $('vWin').classList.add('gone');
    S.running = false;
  }

  // ---------------------------------------------------------------- buttons
  $('bStart').onclick = () => {
    $('vStart').classList.add('gone'); resize();
    if (chosenMode === 'one') newOneRound(chosenHost, chosenK); else newRound(chosen);
  };
  $('bHint').onclick = hintOne;
  $('bMenu').onclick = showStart;
  $('bMenu2').onclick = showStart;
  $('bServe').onclick = finish;
  $('bTable').onclick = () => { if (S.mode === 'room' && S.running && !S.reveal && addTable() >= 0) { S.activeChosen = true; syncSearchHint(); $('search').focus(); } };
  $('bUndo').onclick = undo;
  $('bReveal').onclick = () => reveal(true);
  $('bBack').onclick = () => { $('vWin').classList.add('gone'); S.finished = false; S.running = true; S.reveal = false; syncAll(); };
  $('bCenter').onclick = fitCamera;
  $('cWeighted').onchange = (e) => {
    S.weighted = e.target.checked;
    if (S.seeds.size) { const r = recompute(true); syncAll(); toast(`${S.weighted ? 'Repeated links now count.' : 'Every link counts once again.'} <b>${r.changed}</b> guests change tables.`); }
  };
  $('search').oninput = (e) => renderResults(e.target.value);
  $('search').onfocus = (e) => renderResults(e.target.value);
  $('search').addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { closeResults(); $('search').blur(); return; }
    if (e.key !== 'Enter' && e.key !== 'Return') return;
    e.preventDefault();
    const first = resultRows[0];
    if (!first || !$('search').value.trim()) return;
    const from = (first.querySelector('.pimg, .pini') || first).getBoundingClientRect();
    closeResults();
    seat(+first.dataset.i, from);
    $('search').blur();
  });
  document.addEventListener('pointerdown', (e) => {
    if (!e.target.closest('.search')) { $('results').classList.remove('open'); }
  });
  function setMapMode(on) {
    $('stage').classList.toggle('mapmode', on);
    $('bMap').textContent = on ? 'Small map' : 'Big map';
    $('mapCap').textContent = on ? 'The room, as a network' : 'The room, as a network';
    requestAnimationFrame(() => { resize(); fitCamera(); cam.x = cam.tx; cam.y = cam.ty; cam.z = cam.tz; });
  }
  $('bMap').onclick = () => setMapMode(!$('stage').classList.contains('mapmode'));

  // -------------------------------------------------------------- main loop
  let last = performance.now();
  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.05); last = now;
    draw(now, dt);
    requestAnimationFrame(frame);
  }

  // ---------------------------------------------------------------- startup
  resize();
  S.stats = Core.tableStats(G, S.labels, 1);
  fitCamera();
  cam.x = cam.tx; cam.y = cam.ty; cam.z = cam.tz;
  renderLadder();
  renderGuests();
  syncAll();
  showStart();
  if (window.ResizeObserver) {
    // the map box changes size with the window and with the big-map toggle; keep the room fitted
    new ResizeObserver(() => { resize(); fitCamera(); cam.x = cam.tx; cam.y = cam.ty; cam.z = cam.tz; }).observe($('mapbox'));
  }
  requestAnimationFrame(frame);
})();
