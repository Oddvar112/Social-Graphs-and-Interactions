/* ===========================================================================
   SECRET IDENTITY's engine: cases, clues, the bag-of-words machine and the
   score. No DOM in here, so scripts/test_secret_identity.py can run this exact
   file under node and diff every number against a Python re-implementation.

   The data (data/week5_secret.js, from scripts/build_week5_data.py) holds, per
   character, the clue words under three weightings, a truncated bag-of-words
   vector (stopwords removed, L2-normalised, top 200 terms) and the five
   lookalike pages. The machine that plays against you scores each suspect by
   the cosine between the clues revealed so far (a binary vector) and that
   suspect's page vector, which is exactly the week's "lookalikes" measure with
   the query being your clue list. Ties go to the suspect listed first.

   The machine does not peek at the answer. It reads at least two clues, then
   commits as soon as its leading suspect scores at least twice the runner-up,
   or on the last clue if that never happens; a wrong commitment scores zero,
   like yours would. (Two clues and a factor of two: with one clue there is
   nothing to corroborate, and a lead of less than double is a coin toss between
   lookalikes. Both numbers are in DEFAULTS and the test re-implements them.)
   =========================================================================== */
(function (global) {
  'use strict';

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function shuffle(arr, rnd) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
  function sample(arr, n, rnd) { return shuffle(arr.slice(), rnd).slice(0, n); }
  /** A seed from a calendar day, so everybody gets the same case of the day. */
  function dailySeed(dateStr) {
    let h = 2166136261;
    for (const ch of dateStr) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619) >>> 0; }
    return h >>> 1;
  }

  const DEFAULTS = { rounds: 12, mix: 'mixed', weighting: 'distinct', decoys: 'lookalikes', roster: 'famous', famous: 80,
                     seed: 1, maxClues: 10, margin: 2, minClues: 2 };

  /** Per-character Map(vocab id -> weight), built once. */
  function buildIndex(D) {
    return { vec: D.chars.map(c => new Map(c.vec)) };
  }

  /** Characters with enough text to have a bag worth guessing from. */
  function fullPool(D, weighting, maxClues) {
    const out = [];
    for (let i = 0; i < D.chars.length; i++) {
      const c = D.chars[i];
      if (c.tokens >= 500 && c.clues[weighting || 'distinct'].length >= (maxClues || 10)) out.push(i);
    }
    return out;
  }
  /** The n best-known of them, by how many characters link to their page. */
  function famousPool(D, n, weighting, maxClues) {
    return fullPool(D, weighting, maxClues).sort((a, b) => D.chars[b].kin - D.chars[a].kin || a - b).slice(0, n || 80);
  }

  /** The clue list of one character under a weighting: [{w, count, df, vid}], best first. */
  function clueList(D, ci, weighting) {
    return D.chars[ci].clues[weighting].map(([vid, count]) => ({ vid, w: D.vocab[vid], count, df: D.df[vid] }));
  }

  /** Cosine-style score of every option against the revealed clue vids. */
  function machineScores(D, idx, options, clueVids) {
    return options.map(o => {
      const v = idx.vec[o];
      let s = 0;
      for (const vid of clueVids) s += v.get(vid) || 0;
      return s;
    });
  }
  function argmax(sc) {
    let best = 0;
    for (let i = 1; i < sc.length; i++) if (sc[i] > sc[best]) best = i;
    return best;
  }
  /** What the machine would answer right now: its leader, or -1 while every score is zero. */
  function machinePick(D, idx, options, clueVids) {
    const sc = machineScores(D, idx, options, clueVids);
    const best = argmax(sc);
    return sc[best] > 0 ? options[best] : -1;
  }
  /** The fewest clues after which the leader is the right suspect and stays it
      (an oracle's view, for the reveal); null if never. */
  function machinePar(D, idx, answer, options, clueVids) {
    let par = null;
    for (let k = 1; k <= clueVids.length; k++) {
      const pick = machinePick(D, idx, options, clueVids.slice(0, k));
      if (pick === answer) { if (par === null) par = k; }
      else par = null;
    }
    return par;
  }
  /**
   * How the machine actually plays: after each clue it looks at the scores and,
   * once it has read at least `minClues`, commits to the leader as soon as the
   * leader is at least `margin` times the runner-up (and above zero). If that
   * never happens it is forced to guess on the last clue. Returns
   * { k, pick, forced, trace } with trace = the scores after every clue.
   */
  function machineCommit(D, idx, options, clueVids, margin, minClues) {
    const m = margin || DEFAULTS.margin;
    const min = minClues || DEFAULTS.minClues;
    const trace = [];
    for (let k = 1; k <= clueVids.length; k++) {
      const sc = machineScores(D, idx, options, clueVids.slice(0, k));
      trace.push(sc);
      const best = argmax(sc);
      const second = Math.max(...sc.filter((_, i) => i !== best));
      if (k >= min && sc[best] > 0 && sc[best] >= m * second) return { k, pick: options[best], forced: false, trace };
    }
    const last = trace[trace.length - 1];
    const best = argmax(last);
    return { k: clueVids.length, pick: last[best] > 0 ? options[best] : -1, forced: true, trace };
  }

  /** Three decoys for a suspect: random pages from the pool, or the lookalikes
      (cosine neighbours) that are in the pool, topped up from the pool. */
  function decoys(D, answer, mode, rnd, pool) {
    const inPool = pool ? new Set(pool) : null;
    const others = (pool || fullPool(D)).filter(i => i !== answer);
    if (mode === 'lookalikes') {
      const look = D.chars[answer].look.filter(i => !inPool || inPool.has(i)).slice(0, 3);
      const rest = others.filter(i => !look.includes(i));
      return look.concat(sample(rest, 3 - look.length, rnd));
    }
    return sample(others, 3, rnd);
  }

  const MIX = {
    mixed: r => { const u = Math.round(r * 0.5), i = Math.round(r * 0.25); return { unmask: u, impostor: i, redacted: r - u - i }; },
    unmask: r => ({ unmask: r, impostor: 0, redacted: 0 }),
    impostor: r => ({ unmask: 0, impostor: r, redacted: 0 }),
    redacted: r => ({ unmask: 0, impostor: 0, redacted: r }),
  };
  /**
   * A case file: `rounds` rounds of the requested mix, deterministic for a seed.
   * opts = { rounds, mix, weighting: raw|nostop|distinct, decoys: random|lookalikes,
   *          roster: famous|full, famous, seed, maxClues, margin }
   */
  function makeCase(D, idx, opts) {
    const o = Object.assign({}, DEFAULTS, opts);
    const rnd = mulberry32(o.seed);
    const n = MIX[o.mix](o.rounds);
    const rounds = [];
    const pool = o.roster === 'famous' ? famousPool(D, o.famous, o.weighting, o.maxClues) : fullPool(D, o.weighting, o.maxClues);
    for (const answer of sample(pool, n.unmask, rnd)) {
      const options = shuffle([answer].concat(decoys(D, answer, o.decoys, rnd, pool)), rnd);
      const clues = clueList(D, answer, o.weighting).slice(0, o.maxClues);
      const clueVids = clues.map(c => c.vid);
      const commit = machineCommit(D, idx, options, clueVids, o.margin, o.minClues);
      rounds.push({ type: 'unmask', answer, options, clues, par: machinePar(D, idx, answer, options, clueVids), commit });
    }
    const reals = [], fakes = [];
    D.impostor.forEach((it, i) => (it.real ? reals : fakes).push(i));
    const nReal = Math.ceil(n.impostor / 2);
    const picks = shuffle(sample(reals, nReal, rnd).concat(sample(fakes, n.impostor - nReal, rnd)), rnd);
    for (const i of picks) rounds.push({ type: 'impostor', item: i });
    for (const i of sample(D.redacted.map((_, k) => k), n.redacted, rnd)) rounds.push({ type: 'redacted', item: i });
    // shuffle, but open with an unmask round when there is one: it is the game's signature
    shuffle(rounds, rnd);
    const first = rounds.findIndex(r => r.type === 'unmask');
    if (first > 0) rounds.unshift(rounds.splice(first, 1)[0]);
    return { opts: o, rounds };
  }

  /** Points: unmask 11 - clues used (max 10), the other two 4 each; wrong = 0. */
  function points(type, correct, cluesUsed) {
    if (!correct) return 0;
    if (type === 'unmask') return Math.max(1, 11 - cluesUsed);
    return 4;
  }
  function maxPoints(rounds) {
    return rounds.reduce((s, r) => s + (r.type === 'unmask' ? 10 : 4), 0);
  }
  /** The machine's points on one unmask round: its commitment, scored like yours. */
  function machinePoints(r) {
    return r.commit && r.commit.pick === r.answer ? Math.max(1, 11 - r.commit.k) : 0;
  }
  /** What the machine scores on the whole case: unmask rounds only. */
  function machineTotal(rounds) {
    return rounds.reduce((s, r) => s + (r.type === 'unmask' ? machinePoints(r) : 0), 0);
  }

  const API = { mulberry32, shuffle, sample, dailySeed, buildIndex, fullPool, famousPool, clueList, machineScores,
                machinePick, machinePar, machineCommit, decoys, makeCase, points, maxPoints, machinePoints, machineTotal,
                MIX, DEFAULTS };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else global.SecretCore = API;
})(typeof window !== 'undefined' ? window : globalThis);
