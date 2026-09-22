/* ===========================================================================
   KINGPIN's graph engine: components, Brandes betweenness, and the three
   rival hitmen (degree-greedy, betweenness-greedy, random).

   Kept free of any DOM so scripts/test_kingpin_rules.py can run this exact
   file under node and check every number against networkx. The game plays on
   the UNDIRECTED version of the week-1 snapshot (a link either way means the
   pages know each other), which is the graph the week-3 course page uses for
   betweenness and fragmentation.
   =========================================================================== */
(function (global) {
  'use strict';

  /** Undirected adjacency from the directed snapshot: union of out and in. */
  function buildArena(D) {
    const N = D.nodes.length;
    const sets = Array.from({ length: N }, () => new Set());
    for (let u = 0; u < N; u++) {
      for (const v of D.out[u]) if (v !== u) { sets[u].add(v); sets[v].add(u); }
    }
    const adj = sets.map(s => Int32Array.from(Array.from(s).sort((a, b) => a - b)));
    return { N, adj };
  }

  /** Connected components over the alive nodes (alive = null means everyone). */
  function components(N, adj, alive) {
    const comp = new Int32Array(N).fill(-1);
    const sizes = [];
    const queue = new Int32Array(N);
    for (let s = 0; s < N; s++) {
      if ((alive && !alive[s]) || comp[s] >= 0) continue;
      const id = sizes.length;
      let size = 0, qh = 0, qt = 0;
      comp[s] = id; queue[qt++] = s;
      while (qh < qt) {
        const u = queue[qh++]; size++;
        for (const v of adj[u]) if ((!alive || alive[v]) && comp[v] < 0) { comp[v] = id; queue[qt++] = v; }
      }
      sizes.push(size);
    }
    let giantId = -1, giantSize = 0;
    for (let i = 0; i < sizes.length; i++) if (sizes[i] > giantSize) { giantSize = sizes[i]; giantId = i; }
    return { comp, sizes, giantId, giantSize, count: sizes.length };
  }

  /** Degree counting only alive neighbours. */
  function aliveDegrees(N, adj, alive) {
    const deg = new Int32Array(N);
    for (let u = 0; u < N; u++) {
      if (alive && !alive[u]) continue;
      let d = 0;
      for (const v of adj[u]) if (!alive || alive[v]) d++;
      deg[u] = d;
    }
    return deg;
  }

  /* Brandes (2001): one BFS per source, then walk the rings back inward
     accumulating each node's share of the shortest paths through it. Returns
     raw pair counts (each unordered pair once), the same scale as networkx's
     betweenness_centrality(normalized=False) on an undirected graph. */
  function brandes(N, adj, alive) {
    const bc = new Float64Array(N);
    const dist = new Int32Array(N).fill(-1);
    const sigma = new Float64Array(N);
    const delta = new Float64Array(N);
    const stack = new Int32Array(N);
    const queue = new Int32Array(N);
    const preds = Array.from({ length: N }, () => []);

    for (let s = 0; s < N; s++) {
      if (alive && !alive[s]) continue;
      let qh = 0, qt = 0, sp = 0;
      dist[s] = 0; sigma[s] = 1; queue[qt++] = s;
      while (qh < qt) {
        const u = queue[qh++];
        stack[sp++] = u;
        const du = dist[u] + 1, su = sigma[u];
        for (const v of adj[u]) {
          if (alive && !alive[v]) continue;
          if (dist[v] < 0) { dist[v] = du; queue[qt++] = v; }
          if (dist[v] === du) { sigma[v] += su; preds[v].push(u); }
        }
      }
      for (let i = sp - 1; i >= 1; i--) {
        const w = stack[i];
        const coeff = (1 + delta[w]) / sigma[w];
        for (const p of preds[w]) delta[p] += sigma[p] * coeff;
        bc[w] += delta[w];
      }
      for (let i = 0; i < sp; i++) {
        const w = stack[i];
        dist[w] = -1; sigma[w] = 0; delta[w] = 0; preds[w].length = 0;
      }
    }
    for (let i = 0; i < N; i++) bc[i] /= 2;   // every pair was counted from both ends
    return bc;
  }

  /** Deterministic small RNG so the random hitman is reproducible. */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* One rival hitman running the contract: repeatedly shoot a member of the
     CURRENT giant component, chosen by the strategy, and record how the giant
     shrinks. Ties break on the lowest node index, so the run is reproducible
     in any language. Greedy choices do not depend on the budget, so a shorter
     contract is always a prefix of a longer one. */
  function attack(N, adj, budget, strategy, rngSeed) {
    const alive = new Uint8Array(N).fill(1);
    const rng = mulberry32(rngSeed || 1);
    let c = components(N, adj, alive);
    const curve = [c.giantSize];
    const order = [];
    for (let h = 0; h < budget; h++) {
      let target = -1;
      if (strategy === 'random') {
        const g = [];
        for (let i = 0; i < N; i++) if (alive[i] && c.comp[i] === c.giantId) g.push(i);
        target = g[Math.floor(rng() * g.length)];
      } else {
        const score = strategy === 'degree' ? aliveDegrees(N, adj, alive) : brandes(N, adj, alive);
        let best = -Infinity;
        for (let i = 0; i < N; i++) {
          if (!alive[i] || c.comp[i] !== c.giantId) continue;
          if (score[i] > best) { best = score[i]; target = i; }
        }
      }
      alive[target] = 0;
      order.push(target);
      c = components(N, adj, alive);
      curve.push(c.giantSize);
    }
    return { order, curve };
  }

  /** The random hitman, averaged over many runs (his curve, not his luck). */
  function randomAttackAvg(N, adj, budget, runs, seed) {
    const avg = new Float64Array(budget + 1);
    let sample = null;
    for (let r = 0; r < runs; r++) {
      const res = attack(N, adj, budget, 'random', (seed || 7) + r * 101);
      if (!sample) sample = res;
      for (let i = 0; i < res.curve.length; i++) avg[i] += res.curve[i];
    }
    return { curve: Array.from(avg, v => v / runs), order: sample.order };
  }

  /* Best hit-lists we ever found, per contract size: simulated annealing with
     restarts over hubs + articulation points + high single-node damage
     (12 restarts x 600-stall local search). Node indices into the week-1
     snapshot; scripts/test_kingpin_rules.py re-verifies the claimed core
     sizes, so if the data file is ever regenerated differently this fails
     loudly instead of lying to the player. Black Widow (26) is in all three:
     degree 25, but she is the only door to her corner of the universe. */
  const RECORDS = {
    3: { giant: 264, ids: [26, 73, 246] },                       // Black Widow, Doctor Strange, Spider-Man
    5: { giant: 259, ids: [26, 62, 73, 246, 298] },              // + Deadpool, Wolverine
    8: { giant: 253, ids: [26, 62, 63, 73, 100, 246, 289, 298] }, // + Deathlok, Genis-Vell, War Machine
  };

  const KingpinCore = { buildArena, components, aliveDegrees, brandes, mulberry32, attack, randomAttackAvg, RECORDS };
  if (typeof module !== 'undefined' && module.exports) module.exports = KingpinCore;
  global.KingpinCore = KingpinCore;
})(typeof window !== 'undefined' ? window : globalThis);
