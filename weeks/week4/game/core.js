/* ===========================================================================
   SYMPOSIUM's engine: the philosophers network, modularity, the seating
   rule (label propagation from the guests you placed), NMI, and the random
   host. No DOM in here, so scripts/test_symposium_rules.py can run this
   exact file under node and diff every number against networkx.

   The network is the undirected giant component of the course's week-4
   philosophers snapshot (1374 philosophers, 9139 links; a link either way
   means the two articles mention each other, weight = how many times).
   Modularity is unweighted unless asked otherwise, which is how the course
   page runs Louvain on this network.
   =========================================================================== */
(function (global) {
  'use strict';

  /** Adjacency with weights from the data file's edge list. */
  function buildGraph(D) {
    const N = D.nodes.length;
    const nb = Array.from({ length: N }, () => []);
    const wt = Array.from({ length: N }, () => []);
    let W = 0;
    for (const [u, v, w] of D.edges) {
      nb[u].push(v); wt[u].push(w);
      nb[v].push(u); wt[v].push(w);
      W += w;
    }
    const adj = nb.map(a => Int32Array.from(a));
    const w = wt.map(a => Int32Array.from(a));
    const deg = new Int32Array(N);
    const str = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      deg[i] = adj[i].length;
      let s = 0; for (const x of w[i]) s += x;
      str[i] = s;
    }
    return { N, adj, w, deg, str, m: D.edges.length, W, edges: D.edges };
  }

  /* Modularity of a labelling, by the per-community formula
       Q = sum_c [ L_c / m  -  (D_c / 2m)^2 ]
     with L_c the links inside c and D_c the degree sum of c. A label of -1
     means "not seated": that node is its own community of one (no internal
     links, so it only pays the degree term). Weighted: L_c is the weight
     inside, D_c the strength sum, m the total weight. */
  function modularity(G, labels, weighted) {
    const m = weighted ? G.W : G.m;
    if (m === 0) return 0;
    const inside = new Map(), dsum = new Map();
    let q = 0;
    for (let u = 0; u < G.N; u++) {
      const lu = labels[u];
      const ku = weighted ? G.str[u] : G.deg[u];
      if (lu < 0) { q -= (ku / (2 * m)) * (ku / (2 * m)); continue; }
      dsum.set(lu, (dsum.get(lu) || 0) + ku);
      const a = G.adj[u], w = G.w[u];
      for (let j = 0; j < a.length; j++) {
        const v = a[j];
        if (v > u && labels[v] === lu) inside.set(lu, (inside.get(lu) || 0) + (weighted ? w[j] : 1));
      }
    }
    for (const [c, d] of dsum) q += (inside.get(c) || 0) / m - (d / (2 * m)) * (d / (2 * m));
    return q;
  }

  /* Which table node i would pick, looking at its neighbours' current
     labels: the most seated neighbours (weighted: the most repeated links),
     ties broken by the stronger links, then by the lower table number.
     Returns -1 when no neighbour is seated yet. */
  function bestTable(G, labels, i, tables, weighted, votes, tie) {
    votes.fill(0); tie.fill(0);
    const a = G.adj[i], w = G.w[i];
    let any = false;
    for (let j = 0; j < a.length; j++) {
      const lv = labels[a[j]];
      if (lv < 0) continue;
      votes[lv] += weighted ? w[j] : 1;
      tie[lv] += w[j];
      any = true;
    }
    if (!any) return -1;
    let best = -1;
    for (let t = 0; t < tables; t++) {
      if (votes[t] === 0) continue;
      if (best < 0 || votes[t] > votes[best] || (votes[t] === votes[best] && tie[t] > tie[best])) best = t;
    }
    return best;
  }

  /* The seating rule, in two phases.

     Phase 1, the ink: the guests you placed sit still. Every round, everyone
     who is not seated yet but has a seated neighbour looks at those
     neighbours, all at the same time, and joins the table most of them are
     at. So every table grows by one hop per round, and nobody's table gets
     a head start from where they happen to fall in the list.

     Phase 2, the room settles: now seated guests may move. In index order,
     each guest moves to the table where most of their neighbours sit right
     now, staying put when their own table ties for the top. Each move adds
     at least one link inside a table, so this always stops. A guest never
     ends up alone at a table that way; only the ones you placed are fixed.

     Returns the labels, the rounds of phase 1 and the sweeps of phase 2,
     each as a flat list [node, table, node, table, ...] the map can animate. */
  function propagate(G, seeds, opts) {
    const weighted = !!(opts && opts.weighted);
    const maxSweeps = (opts && opts.maxSweeps) || 200;
    const N = G.N;
    const labels = new Int32Array(N).fill(-1);
    const fixed = new Uint8Array(N);
    let tables = 0;
    for (const [i, t] of seeds) { labels[i] = t; fixed[i] = 1; if (t + 1 > tables) tables = t + 1; }
    const votes = new Float64Array(tables), tie = new Float64Array(tables);
    const rounds = [], sweeps = [];

    // phase 1: synchronous rings
    for (;;) {
      const changed = [];
      for (let i = 0; i < N; i++) {
        if (labels[i] >= 0) continue;
        const t = bestTable(G, labels, i, tables, weighted, votes, tie);
        if (t >= 0) changed.push(i, t);
      }
      if (changed.length === 0) break;
      for (let k = 0; k < changed.length; k += 2) labels[changed[k]] = changed[k + 1];
      rounds.push(changed);
    }

    // phase 2: sequential settling
    let converged = false, moves = 0;
    while (sweeps.length < maxSweeps) {
      const changed = [];
      for (let i = 0; i < N; i++) {
        if (fixed[i] || labels[i] < 0) continue;
        const cur = labels[i];
        const t = bestTable(G, labels, i, tables, weighted, votes, tie);
        if (t < 0 || t === cur) continue;
        if (votes[cur] === votes[t]) continue;            // staying put wins ties
        labels[i] = t; changed.push(i, t);
      }
      if (changed.length === 0) { converged = true; break; }
      sweeps.push(changed);
      moves += changed.length / 2;
    }
    return { labels, rounds, sweeps, moves, converged, tables };
  }

  /** Normalized mutual information, arithmetic-mean normalisation (sklearn's default). */
  function nmi(a, b) {
    const n = a.length;
    const ca = new Map(), cb = new Map(), cab = new Map();
    for (let i = 0; i < n; i++) {
      const x = a[i] < 0 ? -1 - i : a[i], y = b[i] < 0 ? -1 - i : b[i];  // unseated = a singleton
      ca.set(x, (ca.get(x) || 0) + 1);
      cb.set(y, (cb.get(y) || 0) + 1);
      const k = x + ':' + y;
      cab.set(k, (cab.get(k) || 0) + 1);
    }
    let ha = 0, hb = 0, mi = 0;
    for (const c of ca.values()) ha -= c / n * Math.log(c / n);
    for (const c of cb.values()) hb -= c / n * Math.log(c / n);
    for (const [k, c] of cab) {
      const [x, y] = k.split(':').map(Number);
      mi += c / n * Math.log((c / n) / ((ca.get(x) / n) * (cb.get(y) / n)));
    }
    const denom = (ha + hb) / 2;
    return denom > 0 ? mi / denom : 1;
  }

  /** For every table in `labels`, the community in `ref` it overlaps most (ties to the lowest id). */
  function majorityMap(labels, ref) {
    const ov = new Map();
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] < 0) continue;
      let row = ov.get(labels[i]);
      if (!row) { row = new Map(); ov.set(labels[i], row); }
      row.set(ref[i], (row.get(ref[i]) || 0) + 1);
    }
    const mp = new Map();
    for (const [t, row] of ov) {
      let best = -1, bc = -1;
      for (const [c, k] of row) if (k > bc || (k === bc && c < best)) { bc = k; best = c; }
      mp.set(t, best);
    }
    return mp;
  }

  /** Nodes your seating and the reference disagree on, after the majority mapping. */
  function disagreements(labels, ref) {
    const mp = majorityMap(labels, ref);
    const out = [];
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] < 0 || mp.get(labels[i]) !== ref[i]) out.push(i);
    }
    return { nodes: out, map: mp };
  }

  /** Deterministic small RNG so the random host is reproducible. */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* The random host: keeps your placed guests, seats everyone else at a
     table drawn uniformly at random, and reports the average modularity
     over `runs` seatings. */
  function randomSeatingAvg(G, seeds, tables, runs, seed) {
    const base = new Int32Array(G.N).fill(-1);
    for (const [i, t] of seeds) base[i] = t;
    let total = 0;
    for (let r = 0; r < runs; r++) {
      const rng = mulberry32((seed || 11) + r * 7919);
      const lab = base.slice();
      for (let i = 0; i < G.N; i++) if (lab[i] < 0) lab[i] = Math.floor(rng() * tables);
      total += modularity(G, lab, false);
    }
    return total / runs;
  }

  /** Per-table bookkeeping for the side panel: size, links inside, degree sum, Q share. */
  function tableStats(G, labels, tables) {
    const size = new Int32Array(tables), inside = new Int32Array(tables), dsum = new Int32Array(tables);
    for (let u = 0; u < G.N; u++) {
      const lu = labels[u];
      if (lu < 0) continue;
      size[lu]++; dsum[lu] += G.deg[u];
      for (const v of G.adj[u]) if (v > u && labels[v] === lu) inside[lu]++;
    }
    const share = new Float64Array(tables);
    for (let t = 0; t < tables; t++) share[t] = inside[t] / G.m - (dsum[t] / (2 * G.m)) * (dsum[t] / (2 * G.m));
    return { size, inside, dsum, share };
  }

  /* ---- ONE TABLE: build the best community around one guest -------------

     A single table's share of modularity is
         Q_c = L_c / m  -  (D_c / 2m)^2
     with L_c the links inside and D_c the degree sum. Multiplied by m it is
     "links inside minus the links a degree-preserving shuffle would put
     there", which is the number the game shows: links above chance. */

  /** Links inside a set of members, its degree sum, what chance predicts, and the difference. */
  function tableScore(G, members) {
    const inSet = new Uint8Array(G.N);
    for (const i of members) inSet[i] = 1;
    let inside = 0, dsum = 0;
    for (const u of members) {
      dsum += G.deg[u];
      for (const v of G.adj[u]) if (v > u && inSet[v]) inside++;
    }
    const expected = dsum * dsum / (4 * G.m);
    return { inside, dsum, expected, above: inside - expected, q: inside / G.m - (dsum / (2 * G.m)) * (dsum / (2 * G.m)) };
  }

  /** How many links v has into the set. */
  function linksInto(G, inSet, v) {
    let l = 0;
    for (const u of G.adj[v]) if (inSet[u]) l++;
    return l;
  }

  /** Change in "links above chance" from adding v to a table with degree sum dsum. */
  function addGain(G, inSet, dsum, v) {
    const d = G.deg[v];
    return linksInto(G, inSet, v) - (2 * dsum * d + d * d) / (4 * G.m);
  }

  /** Everyone linked to the table but not at it, in index order. */
  function candidates(G, members) {
    const inSet = new Uint8Array(G.N), seen = new Uint8Array(G.N);
    for (const i of members) inSet[i] = 1;
    const out = [];
    for (const u of members) for (const v of G.adj[u]) if (!inSet[v] && !seen[v]) { seen[v] = 1; out.push(v); }
    out.sort((a, b) => a - b);
    return out;
  }

  /* The greedy host: k times, add whoever raises links-above-chance most
     among the people linked to the table (ties to the lowest index). */
  function greedyTable(G, host, k) {
    const members = [host];
    const inSet = new Uint8Array(G.N);
    inSet[host] = 1;
    let dsum = G.deg[host];
    for (let s = 0; s < k; s++) {
      let best = -1, bestG = -Infinity;
      for (const v of candidates(G, members)) {
        const g = addGain(G, inSet, dsum, v);
        if (g > bestG + 1e-12 || (Math.abs(g - bestG) <= 1e-12 && v < best)) { bestG = g; best = v; }
      }
      if (best < 0) break;
      members.push(best); inSet[best] = 1; dsum += G.deg[best];
    }
    return members;
  }

  /* The record: start from a table and keep swapping one guest (never the
     host, who sits first) for one candidate as long as some swap raises
     links-above-chance; take the best swap each round, ties to the earliest
     seat and then the lowest candidate index. Deterministic, so the test
     can replay it. */
  function improveTable(G, host, start, maxRounds) {
    const members = start.slice();
    if (members[0] !== host) throw new Error('the host sits first');
    for (let round = 0; round < (maxRounds || 60); round++) {
      const inSet = new Uint8Array(G.N);
      for (const i of members) inSet[i] = 1;
      const base = tableScore(G, members);
      const cand = candidates(G, members);
      const into = new Map();                      // links from each candidate into the current table
      for (const v of cand) into.set(v, linksInto(G, inSet, v));
      let bestVal = base.above + 1e-9, bestA = -1, bestV = -1;
      for (let a = 1; a < members.length; a++) {
        const out = members[a];
        const nbOut = new Set(G.adj[out]);
        let insideOut = 0;                          // links from the leaving guest to the others
        for (const u of members) if (u !== out && nbOut.has(u)) insideOut++;
        const dsumW = base.dsum - G.deg[out];
        for (const v of cand) {
          const l = into.get(v) - (nbOut.has(v) ? 1 : 0);
          const dsum2 = dsumW + G.deg[v];
          const val = base.inside - insideOut + l - dsum2 * dsum2 / (4 * G.m);
          if (val > bestVal + 1e-12) { bestVal = val; bestA = a; bestV = v; }
        }
      }
      if (bestA < 0) break;
      members[bestA] = bestV;
    }
    return members;
  }

  /* Someone picking k guests at random from the people linked to the host
     (or from anyone, if the host knows fewer than k), averaged over runs. */
  function randomTableAvg(G, host, k, runs, seed) {
    let pool = Array.from(G.adj[host]);
    if (pool.length < k) { pool = []; for (let i = 0; i < G.N; i++) if (i !== host) pool.push(i); }
    let total = 0;
    for (let r = 0; r < runs; r++) {
      const rng = mulberry32((seed || 3) + r * 7919);
      const bag = pool.slice();
      const members = [host];
      for (let s = 0; s < k; s++) { const j = Math.floor(rng() * bag.length); members.push(bag[j]); bag.splice(j, 1); }
      total += tableScore(G, members).above;
    }
    return total / runs;
  }

  const SymposiumCore = { buildGraph, modularity, propagate, nmi, majorityMap, disagreements, mulberry32, randomSeatingAvg, tableStats,
    tableScore, linksInto, addGain, candidates, greedyTable, improveTable, randomTableAvg };
  if (typeof module !== 'undefined' && module.exports) module.exports = SymposiumCore;
  global.SymposiumCore = SymposiumCore;
})(typeof window !== 'undefined' ? window : globalThis);
