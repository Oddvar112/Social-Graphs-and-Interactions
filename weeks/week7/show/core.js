/* ===========================================================================
   IN OTHER WORDS: the rules. No DOM in here, so scripts/test_in_other_words.py
   can run this exact file under node and check it against the data.

   An episode is ten clues from data/week7_show.js: four in the page's own
   words, five in other words, one about a character who is not in the corpus,
   no character twice, in a shuffled order fixed by the episode's seed.
   "Tonight's episode" is seeded by the date, so everyone gets the same show.

   Each clue: you point at one of six portraits, or at NOBODY HERE. The two
   engines point at their number one of all 303 pages (never at nobody: a
   search engine always returns something). One point per right answer.
   =========================================================================== */
(function (global) {
  'use strict';

  const DEFAULTS = { rounds: 10, mix: { page: 4, other: 5, absent: 1 }, timer: 40 };
  const EPOCH = Date.UTC(2026, 9, 20);          // episode 1 was 20 October 2026

  function build(D) {
    const byStyle = { page: [], other: [], absent: [] };
    D.clues.forEach((c, i) => { if (c.play !== false) byStyle[c.style].push(i); });
    return { D, byStyle };
  }

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function shuffle(a, rnd) {
    for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  }

  /** Episode number and seed for a calendar day (local time). */
  function episodeOfDate(date) {
    const d = date || new Date();
    const day = Math.floor((Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()) - EPOCH) / 86400000);
    return { number: day + 1, seed: 1000003 * (day + 1) + 7 };
  }

  /** The clues of an episode: indices into D.clues, no character twice, styles mixed as DEFAULTS.mix says. */
  function episode(M, seed, mix) {
    mix = mix || DEFAULTS.mix;
    const rnd = mulberry32(seed);
    const used = new Set();
    const chosen = [];
    for (const style of ['absent', 'page', 'other']) {
      const pool = shuffle(M.byStyle[style].slice(), rnd);
      let n = mix[style] || 0;
      for (const i of pool) {
        if (n <= 0) break;
        const key = M.D.clues[i].style === 'absent' ? 'absent:' + M.D.clues[i].name : 'page:' + M.D.clues[i].target;
        if (used.has(key)) continue;
        used.add(key);
        chosen.push(i);
        n -= 1;
      }
    }
    return shuffle(chosen, rnd);
  }

  /** The verdicts for one clue. pick: a page index, -1 for NOBODY HERE, null for no answer. */
  function judge(clue, pick) {
    const absent = clue.style === 'absent';
    const kwPick = clue.kw.top[0][0], dnPick = clue.dn.top[0][0];
    return {
      you: pick === null || pick === undefined ? false : (absent ? pick === -1 : pick === clue.target),
      kw: !absent && kwPick === clue.target,
      dn: !absent && dnPick === clue.target,
      kwPick, dnPick, absent,
    };
  }

  /** Totals over a list of verdicts, and who won (ties shared). */
  function tally(verdicts) {
    const t = { you: 0, kw: 0, dn: 0, rounds: verdicts.length };
    verdicts.forEach(v => { t.you += v.you; t.kw += v.kw; t.dn += v.dn; });
    const best = Math.max(t.you, t.kw, t.dn);
    t.winners = ['you', 'kw', 'dn'].filter(k => t[k] === best);
    return t;
  }

  /** A result to paste: one line per contestant, a square per round. */
  function shareText(number, verdicts) {
    const t = tally(verdicts);
    const row = (label, key) => label.padEnd(8) + verdicts.map(v => v[key] ? '■' : '□').join('') + '  ' + t[key] + '/' + t.rounds;
    return ['IN OTHER WORDS · episode ' + number, row('YOU', 'you'), row('KEYWORD', 'kw'), row('MEANING', 'dn')].join('\n');
  }

  const API = { DEFAULTS, build, mulberry32, episodeOfDate, episode, judge, tally, shareText };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else global.ShowCore = API;
})(typeof window !== 'undefined' ? window : globalThis);
