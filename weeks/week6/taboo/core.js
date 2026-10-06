/* ===========================================================================
   TABOO-IDF's engine: the machine that guesses, the taboo rule, the score.
   No DOM in here, so scripts/test_taboo.py can run this exact file and diff it
   against numpy.

   The machine is the lookalikes measure with the query being your words: the
   cosine between a binary vector of the words you typed and a page's
   unit-length TF-IDF vector. That is the sum, over your words, of the page's
   weight for the word (the postings in data/week6_taboo.js, in hundred-
   thousandths), divided by the square root of how many words you typed. The
   ranking needs only the sum.

   Rules. A word must be letters only. If fewer than two pages use it, the
   machine has never seen it: no cost. In NAMES BANNED mode a word the name
   rule calls a name is a TABOO strike and costs one word. A word you already
   typed is ignored. You win when the target is ahead of every other page.
   Every word used (typed, struck, or bought as a hint) costs a point.
   =========================================================================== */
(function (global) {
  'use strict';

  const DEFAULTS = { mode: 'banned', maxUsed: 12, hintCost: 3, top: 10 };

  function build(D) {
    const index = new Map();
    D.words.forEach((w, i) => index.set(w, i));
    const S = D.meta.scale, N = D.chars.length;
    return { D, N, S, index, ps: D.ps, pi: D.pi, pv: D.pv };
  }

  /** What typing `raw` would do. status: ok | empty | invalid | unknown | taboo | dup. */
  function lookup(M, raw, typed, mode) {
    const w = String(raw == null ? '' : raw).trim().toLowerCase();
    if (!w) return { status: 'empty', w };
    if (!/^[a-z]+$/.test(w)) return { status: 'invalid', w };
    const id = M.index.get(w);
    if (id === undefined) return { status: 'unknown', w };
    if (typed.indexOf(id) >= 0) return { status: 'dup', w, id };
    if (mode === 'banned' && M.D.kind[id] === 1) return { status: 'taboo', w, id };
    return { status: 'ok', w, id };
  }

  /** Sum over the typed words of every page's weight: the numerator of the cosine. */
  function scores(M, ids) {
    const out = new Float64Array(M.N);
    for (const id of ids) {
      for (let p = M.ps[id]; p < M.ps[id + 1]; p++) out[M.pi[p]] += M.pv[p] / M.S;
    }
    return out;
  }

  /** Pages best first (ties to the lower index), the target's rank, and whether it is strictly ahead of everyone. */
  function ranking(M, ids, target) {
    const sc = scores(M, ids);
    const order = Array.from(sc.keys()).sort((a, b) => sc[b] - sc[a] || a - b);
    const rank = order.indexOf(target) + 1;
    let runner = -1;
    for (let k = 0; k < order.length; k++) if (order[k] !== target) { runner = order[k]; break; }
    const k = ids.length;
    const norm = k ? 1 / Math.sqrt(k) : 0;
    const won = k > 0 && sc[target] > sc[runner] + 1e-12;
    return { sc, order, rank, won, norm, margin: sc[target] - sc[runner] };
  }

  /** The targets: pages with enough text, and (famous) the best-known `n` of them by in-degree. */
  function targetPool(D, famous, n) {
    const pool = [];
    D.chars.forEach((c, i) => { if (c.oracle) pool.push(i); });
    if (!famous) return pool;
    return pool.sort((a, b) => D.chars[b].kin - D.chars[a].kin || a - b).slice(0, n || 80);
  }

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function pickTarget(D, pool, seed, exclude) {
    const rnd = mulberry32(seed);
    const ok = pool.filter(i => !exclude || exclude.indexOf(i) < 0);
    return (ok.length ? ok : pool)[Math.floor(rnd() * (ok.length ? ok.length : pool.length))];
  }

  /** Points for a win after `used` words used: 11 - used, at least 1. A loss or a give-up is 0. */
  function points(won, used) { return won ? Math.max(1, 11 - used) : 0; }

  /** The word the oracle would have typed for a target, with the margin it wins by. */
  function oracleWord(D, target, mode) {
    const c = D.chars[target];
    const [id, margin] = mode === 'banned' ? c.oracle : c.oracle_any;
    return { id, word: D.words[id], margin };
  }
  /** The target's fingerprint: its twelve strongest plain words. */
  function fingerprint(D, target) { return D.chars[target].print.map(id => D.words[id]); }

  const API = { DEFAULTS, build, lookup, scores, ranking, targetPool, mulberry32, pickTarget, points, oracleWord, fingerprint };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else global.TabooCore = API;
})(typeof window !== 'undefined' ? window : globalThis);
