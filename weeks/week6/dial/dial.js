/* ===========================================================================
   THE NAME DIAL: the interface. All arithmetic lives in core.js.
   State: one page, two dials. Everything else is derived, and the state is
   kept in the address (#p=<page index>&n=<names>&h=<habit>) so a link keeps it.
   =========================================================================== */
(function () {
  'use strict';
  const D = window.DIAL;
  if (!D) {
    document.body.innerHTML = '<p style="padding:40px;font:16px sans-serif;color:#fff">data/week6_dial.js is missing. Run <code>python scripts/build_week6_dial.py</code>.</p>';
    return;
  }
  const Core = window.DialCore;
  const M = Core.build(D);
  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const SHOW = 12;
  const QUICK = ['Storm (Marvel Comics)', 'Wolverine (character)', 'Venom (character)', 'Doctor Strange', 'Black Widow (Natasha Romanova)', 'Spider-Man'];
  const byId = new Map(D.chars.map((c, i) => [c.id.replace(/_/g, ' '), i]));
  const byName = new Map();
  D.chars.forEach((c, i) => { if (!byName.has(c.name.toLowerCase())) byName.set(c.name.toLowerCase(), i); });

  const state = { page: 0, wn: 1, wh: 1 };
  let base = null;                 // ranks at the starting dials, for the page on show
  let curveTimer = null;

  const idx = id => (byId.has(id) ? byId.get(id) : -1);
  state.page = Math.max(0, idx('Storm (Marvel Comics)'));

  // ------------------------------------------------------------ address
  function readHash() {
    try {
      const q = new URLSearchParams(location.hash.slice(1));
      const p = Number(q.get('p')), n = Number(q.get('n')), h = Number(q.get('h'));
      if (q.has('p') && p >= 0 && p < M.N && Number.isInteger(p)) state.page = p;
      if (q.has('n') && n >= 0 && n <= 100) state.wn = Math.round(n) / 100;
      if (q.has('h') && h >= 0 && h <= 100) state.wh = Math.round(h) / 100;
    } catch (e) { /* no hash support: keep defaults */ }
  }
  function writeHash() {
    try { history.replaceState(null, '', `#p=${state.page}&n=${Math.round(state.wn * 100)}&h=${Math.round(state.wh * 100)}`); } catch (e) { /* previews */ }
  }

  // ------------------------------------------------------------ drawing
  function bar(r, scale) {
    const w = x => (100 * Math.max(0, x) / scale).toFixed(2);
    return `<span class="bar"><i class="n" style="width:${w(r.n)}%"></i><i class="h" style="width:${w(r.h)}%"></i><i class="o" style="width:${w(r.o)}%"></i></span>`;
  }
  function renderList() {
    const rows = Core.neighbours(M, state.page, state.wn, state.wh, SHOW, base);
    const scale = Math.max(rows[0].cos, 1e-9);
    $('list').innerHTML = rows.map(r => {
      const c = D.chars[r.j];
      const was = r.base === r.rank ? 'same place' : 'was #' + r.base;
      return `<li class="nbr"><span class="rk">${r.rank}</span>` +
        `<button class="nm" data-j="${r.j}" title="Show ${esc(c.name)}'s lookalikes">${esc(c.name)}<small>${was}</small></button>` +
        `${bar(r, scale)}<span class="cs">${r.cos.toFixed(3)}</span>` +
        `<span class="lk ${r.linked ? 'yes' : 'no'}">${r.linked ? 'linked' : 'not linked'}</span></li>`;
    }).join('');
    const top10 = Core.neighbours(M, state.page, state.wn, state.wh, 10, null);
    const k = top10.filter(r => r.linked).length;
    $('r-page').textContent = `${k} / 10`;
    const c = D.chars[state.page];
    $('title').textContent = c.name.toUpperCase();
    $('sub').innerHTML = `${c.tokens.toLocaleString()} words on the page, ${c.kin} pages link here. <a href="${esc(c.url)}" target="_blank" rel="noopener">Read it on Wikipedia &rarr;</a>`;
    $('r-page-l').textContent = `of ${c.name}'s ten nearest neighbours are linked to it`;
  }
  function renderControls() {
    $('wn').value = Math.round(state.wn * 100);
    $('wh').value = Math.round(state.wh * 100);
    $('on').textContent = Math.round(state.wn * 100) + '%';
    $('oh').textContent = Math.round(state.wh * 100) + '%';
    document.querySelectorAll('#chips .chip').forEach(b => b.classList.toggle('on', Number(b.dataset.j) === state.page));
  }

  const CH = { w: 360, h: 190, l: 36, r: 12, t: 12, b: 30 };
  function renderAgreement() {
    const pts = Core.curve(M, state.wh, 10);
    const all = Core.agreement(M, state.wn, state.wh);
    $('r-all').textContent = all.toFixed(2);
    const maxY = Math.max(4.5, ...pts.map(p => p.a)) * 1.04;
    const X = v => CH.l + v * (CH.w - CH.l - CH.r), Y = v => CH.h - CH.b - (v / maxY) * (CH.h - CH.b - CH.t);
    let s = '';
    for (const g of [0, 1, 2, 3, 4]) s += `<line x1="${CH.l}" x2="${CH.w - CH.r}" y1="${Y(g)}" y2="${Y(g)}" stroke="#26304f" stroke-width="1"/><text x="${CH.l - 6}" y="${Y(g) + 4}" text-anchor="end" fill="#8d99bd" font-size="10">${g}</text>`;
    for (const g of [0, 25, 50, 75, 100]) s += `<text x="${X(g / 100)}" y="${CH.h - 12}" text-anchor="middle" fill="#8d99bd" font-size="10">${g}</text>`;
    s += `<text x="${(CH.l + CH.w - CH.r) / 2}" y="${CH.h - 1}" text-anchor="middle" fill="#8d99bd" font-size="10">names count (%)</text>`;
    s += `<line x1="${CH.l}" x2="${CH.w - CH.r}" y1="${Y(chance)}" y2="${Y(chance)}" stroke="#66739a" stroke-dasharray="4 4"/><text x="${CH.w - CH.r}" y="${Y(chance) - 4}" text-anchor="end" fill="#66739a" font-size="9.5">ten random pages</text>`;
    s += `<polyline fill="none" stroke="#5fe3ff" stroke-width="2.5" stroke-linejoin="round" points="${pts.map(p => `${X(p.wn).toFixed(1)},${Y(p.a).toFixed(1)}`).join(' ')}"/>`;
    s += `<line x1="${X(state.wn)}" x2="${X(state.wn)}" y1="${CH.t}" y2="${CH.h - CH.b}" stroke="#ffc93f" stroke-width="1" stroke-dasharray="3 3"/>`;
    s += `<circle cx="${X(state.wn)}" cy="${Y(all)}" r="5.5" fill="#ffc93f" stroke="#0b0d16" stroke-width="2"/>`;
    $('chart').innerHTML = s;
  }
  let chance = 0;
  (function () { let deg = 0; for (let i = 0; i < M.N; i++) for (let j = 0; j < M.N; j++) deg += M.adj[i][j]; chance = 10 * (deg / M.N) / (M.N - 1); $('r-chance').textContent = chance.toFixed(2); })();

  function render(full) {
    renderControls();
    renderList();
    writeHash();
    if (full) { renderAgreement(); return; }
    clearTimeout(curveTimer);
    curveTimer = setTimeout(renderAgreement, 40);
  }
  function setPage(j) {
    state.page = j;
    base = Core.baseRanks(M, j);
    $('pick').value = '';
    render(true);
  }

  // ------------------------------------------------------------ wiring
  $('people').innerHTML = D.chars.map(c => `<option value="${esc(c.name)}"></option>`).join('');
  $('chips').innerHTML = QUICK.map(id => { const j = idx(id.replace(/_/g, ' ')); return j < 0 ? '' : `<button class="chip" data-j="${j}">${esc(D.chars[j].name.replace(/ \(.*\)$/, ''))}</button>`; }).join('');
  $('chips').addEventListener('click', e => { const b = e.target.closest('button'); if (b) setPage(Number(b.dataset.j)); });
  $('list').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { setPage(Number(b.dataset.j)); window.scrollTo && window.scrollTo(0, 0); } });
  $('pick').addEventListener('change', () => {
    const v = $('pick').value.trim().toLowerCase();
    if (byName.has(v)) setPage(byName.get(v));
    else {
      const hit = D.chars.findIndex(c => c.name.toLowerCase().startsWith(v));
      if (v && hit >= 0) setPage(hit);
    }
  });
  $('wn').addEventListener('input', () => { state.wn = Number($('wn').value) / 100; render(false); });
  $('wh').addEventListener('input', () => { state.wh = Number($('wh').value) / 100; render(false); });
  $('presets').addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b) return;
    state.wn = Number(b.dataset.n) / 100;
    state.wh = Number(b.dataset.h) / 100;
    render(true);
  });
  window.addEventListener('hashchange', () => { const p = state.page; readHash(); if (p !== state.page) base = Core.baseRanks(M, state.page); render(true); });

  // When this page sits in an iframe (the week 6 post), tell the parent how tall the content really is, so the frame fits it.
  function reportHeight() {
    try {
      if (!window.parent || window.parent === window || !document.body || !document.body.getBoundingClientRect) return;
      window.parent.postMessage({ type: 'name-dial-height', h: Math.ceil(document.body.getBoundingClientRect().height) + 2 }, '*');
    } catch (e) { /* cross-origin frames or previews: keep the fallback height */ }
  }
  window.addEventListener('load', reportHeight);
  if (typeof ResizeObserver !== 'undefined' && document.body) new ResizeObserver(reportHeight).observe(document.body);

  readHash();
  base = Core.baseRanks(M, state.page);
  render(true);
  reportHeight();
})();
