/* ===========================================================================
   THE NAME DIAL's engine: re-ranking lookalikes while two dials are turned.
   No DOM in here, so scripts/test_name_dial.py can run this exact file and
   diff it against Python.

   The data (data/week6_dial.js, from scripts/build_week6_dial.py) holds three
   303 x 303 matrices, in millionths, that add up to the dot product of two
   unit-length TF-IDF vectors: Pn from names, Ph from habit words, Po from every
   other word. Turn the names dial to wn and the habit dial to wh (0 to 1) and
   scale that part of each vector by it, then renormalise:

       cos(i, j) = ( wn^2 Pn[i][j] + wh^2 Ph[i][j] + Po[i][j] ) / ( norm(i) norm(j) )
       norm(i)^2 = wn^2 Pn[i][i] + wh^2 Ph[i][i] + Po[i][i]

   wn = wh = 1 is TF-IDF as the course defines it; wn = 0, wh = 1 removes the
   names; wn = wh = 0 removes names and habit words. The three terms of the
   numerator, divided by the same norms, are the three parts of the cosine and
   add up to it.
   =========================================================================== */
(function (global) {
  'use strict';

  /** Prepare typed arrays and the adjacency once. */
  function build(D) {
    const N = D.chars.length, S = D.meta.scale;
    const toF = a => { const f = new Float64Array(a.length); for (let k = 0; k < a.length; k++) f[k] = a[k] / S; return f; };
    const adj = Array.from({ length: N }, () => new Uint8Array(N));
    for (const [a, b] of D.edges) { adj[a][b] = 1; adj[b][a] = 1; }
    return { D, N, n: toF(D.n), h: toF(D.h), o: toF(D.o), adj };
  }

  function norms(M, wn, wh) {
    const N = M.N, out = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const k = i * N + i;
      const v = wn * wn * M.n[k] + wh * wh * M.h[k] + M.o[k];
      out[i] = v > 0 ? Math.sqrt(v) : 1;
    }
    return out;
  }

  /** Cosine of page i with every page at the given dials; -1 for i itself. */
  function cosRow(M, i, wn, wh, nr) {
    const N = M.N, out = new Float64Array(N), nm = nr || norms(M, wn, wh);
    const a = wn * wn, b = wh * wh, base = i * N;
    for (let j = 0; j < N; j++) {
      out[j] = j === i ? -1 : (a * M.n[base + j] + b * M.h[base + j] + M.o[base + j]) / (nm[i] * nm[j]);
    }
    return out;
  }

  /** Indices of the k largest entries, best first, ties to the lower index. */
  function topK(row, k) {
    const idx = Array.from(row.keys());
    idx.sort((x, y) => row[y] - row[x] || x - y);
    return idx.slice(0, k);
  }

  /** The three parts of cos(i, j) at the given dials; they add up to the cosine. */
  function parts(M, i, j, wn, wh, nr) {
    const N = M.N, nm = nr || norms(M, wn, wh), d = nm[i] * nm[j], k = i * N + j;
    return { n: wn * wn * M.n[k] / d, h: wh * wh * M.h[k] / d, o: M.o[k] / d };
  }

  /** Mean over all pages of how many of the k nearest neighbours are linked to the page. */
  function agreement(M, wn, wh, k) {
    const kk = k || 10, nr = norms(M, wn, wh);
    let total = 0;
    for (let i = 0; i < M.N; i++) {
      const row = cosRow(M, i, wn, wh, nr);
      for (const j of topK(row, kk)) total += M.adj[i][j];
    }
    return total / M.N;
  }

  /** The agreement curve as the names dial goes from 0 to 1 in `steps` steps, habit dial fixed. */
  function curve(M, wh, steps, k) {
    const out = [];
    for (let s = 0; s <= steps; s++) out.push({ wn: s / steps, a: agreement(M, s / steps, wh, k) });
    return out;
  }

  /** The k nearest neighbours of page i, each with its cosine, its three parts, whether it is linked, and the rank
      it had at the starting dials (both at 1), so the interface can show who moved. */
  function neighbours(M, i, wn, wh, k, baseRank) {
    const nr = norms(M, wn, wh), row = cosRow(M, i, wn, wh, nr);
    return topK(row, k).map((j, r) => {
      const p = parts(M, i, j, wn, wh, nr);
      return { j, rank: r + 1, cos: row[j], n: p.n, h: p.h, o: p.o, linked: !!M.adj[i][j], base: baseRank ? baseRank[j] : null };
    });
  }

  /** rank of every page in page i's list at the starting dials (1 = nearest). */
  function baseRanks(M, i) {
    const order = topK(cosRow(M, i, 1, 1), M.N - 1), rank = new Int32Array(M.N);
    order.forEach((j, r) => { rank[j] = r + 1; });
    return rank;
  }

  const API = { build, norms, cosRow, topK, parts, agreement, curve, neighbours, baseRanks };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else global.DialCore = API;
})(typeof window !== 'undefined' ? window : globalThis);
