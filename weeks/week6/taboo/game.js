/* ===========================================================================
   TABOO-IDF: the interface. All rules live in core.js.

   You describe a secret page one word at a time. After every word a machine
   ranks all 303 pages by cosine against your words. With names banned, a name
   is a strike. Get the target to number one before you have used 12 words.
   =========================================================================== */
(function () {
  'use strict';
  const D = window.TABOO;
  if (!D) {
    document.body.innerHTML = '<p style="padding:40px;font:16px sans-serif;color:#fff">data/week6_taboo.js is missing. Run <code>python scripts/build_week6_taboo.py</code>.</p>';
    return;
  }
  const Core = window.TabooCore;
  const M = Core.build(D);
  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const MAX = Core.DEFAULTS.maxUsed, HINT = Core.DEFAULTS.hintCost, TOP = Core.DEFAULTS.top;
  const settings = { mode: 'banned', roster: 'famous' };
  const total = { score: 0, wins: 0 };
  let g = null;     // { target, typed: [ids], log: [{w, kind, rank}], strikes, hints, over, won }
  let seen = [];
  let counter = 0;

  function used() { return g.typed.length + g.strikes + g.hints * HINT; }
  function best() { try { return Number(localStorage.getItem('taboo-idf:best') || 0); } catch (e) { return 0; } }
  function saveBest() { try { if (total.score > best()) localStorage.setItem('taboo-idf:best', String(total.score)); } catch (e) { /* private mode */ } }

  // ------------------------------------------------------------ a new target
  function newTarget() {
    const pool = Core.targetPool(D, settings.roster === 'famous', 80);
    counter += 1;
    const seed = (Date.now() ^ (Math.random() * 1e9) ^ (counter * 2654435761)) >>> 1;
    const target = Core.pickTarget(D, pool, seed, seen.slice(-Math.min(12, pool.length - 1)));
    seen.push(target);
    g = { target, typed: [], log: [], strikes: 0, hints: 0, over: false, won: false };
    const c = D.chars[target];
    $('tname').textContent = c.name;
    $('tmeta').textContent = `${c.tokens.toLocaleString()} words on the page, ${c.kin} pages link here. Describe them; do not say who.`;
    $('pic').innerHTML = c.img ? `<img src="${esc(c.img)}" alt="${esc(c.name)}" onerror="this.parentNode.innerHTML='<div class=ph comic>?</div>'">` : '<div class="ph comic">?</div>';
    $('reveal').hidden = true;
    $('reveal').innerHTML = '';
    $('next').hidden = true;
    $('word').disabled = false;
    $('send').disabled = false;
    $('giveup').disabled = false;
    $('word').value = '';
    say('', '');
    render();
    $('word').focus();
  }

  // ------------------------------------------------------------ feedback
  function say(text, cls) { const m = $('msg'); m.textContent = text; m.className = 'msg ' + (cls || ''); }
  function render() {
    // typed words
    $('typed').innerHTML = g.log.map(x => {
      const cls = x.kind === 'taboo' ? 'taboo' : (x.kind === 'hint' ? 'hint' : (x.delta > 0 ? 'up' : (x.delta < 0 ? 'down' : '')));
      const tag = x.kind === 'taboo' ? 'strike: a name' : (x.kind === 'hint' ? `-${HINT}` : `#${x.rank}${x.delta ? (x.delta > 0 ? ' ▲' + x.delta : ' ▼' + (-x.delta)) : ''}`);
      return `<li class="${cls} type">${esc(x.w)}<small>${tag}</small></li>`;
    }).join('') || '<li style="border-style:dashed;color:var(--dim2);font-size:13px">your words will appear here</li>';
    $('s-score').textContent = total.score;
    $('s-wins').textContent = total.wins;
    $('s-used').textContent = `${used()} / ${MAX}`;
    const left = Math.max(0, 2 - g.hints);
    $('hint').disabled = g.over || left === 0;
    $('hintcost').textContent = left ? `(−${HINT} words, ${left} left)` : '(none left)';
    // the board
    const r = Core.ranking(M, g.typed, g.target);
    if (!g.typed.length) {
      $('board').innerHTML = '<li class="empty">Nothing to rank yet. Say a word.</li>';
      $('rank').hidden = true;
      $('sub').textContent = 'Type a word. The pages re-rank after every one.';
      return r;
    }
    const k = g.typed.length;
    const top = r.order.slice(0, TOP);
    const maxc = Math.max(r.sc[top[0]] * r.norm, 1e-9);
    const row = (p, rank) => `<li class="row${p === g.target ? ' me' : ''}"><span class="rk">${rank}</span><span class="nm">${esc(D.chars[p].name)}</span>` +
      `<span class="bar"><i style="width:${(100 * r.sc[p] * r.norm / maxc).toFixed(1)}%"></i></span><span class="cs">${(r.sc[p] * r.norm).toFixed(3)}</span></li>`;
    let html = top.map((p, i) => row(p, i + 1)).join('');
    if (r.rank > TOP) html += `<li class="row gap">&middot; &middot; &middot;</li>` + row(g.target, r.rank);
    $('board').innerHTML = html;
    $('rank').hidden = false;
    $('rank').className = 'rank' + (r.rank === 1 ? ' first' : '');
    $('rank').innerHTML = r.rank === 1 ? `<b>#1</b> ${esc(D.chars[g.target].name)} is first of 303.`
      : `${esc(D.chars[g.target].name)} is <b>#${r.rank}</b> of 303 with ${k} word${k > 1 ? 's' : ''}.`;
    $('sub').textContent = `Cosine between your ${k} word${k > 1 ? 's' : ''} and each page's TF-IDF vector.`;
    return r;
  }
  function currentRank() { return Core.ranking(M, g.typed, g.target).rank; }
  /** Hints spell out the target's own strongest words, not the oracle's odd one: first letter and length, then three letters. */
  function hintWord(level) { return Core.fingerprint(D, g.target)[(level === undefined ? g.hints : level) === 0 ? 0 : 1]; }

  // ------------------------------------------------------------ one word
  function submit(raw) {
    if (!g || g.over) return;
    const before = g.typed.length ? currentRank() : D.chars.length;
    const res = Core.lookup(M, raw, g.typed, settings.mode);
    if (res.status === 'empty') return;
    if (res.status === 'invalid') return say('One word, letters only.', 'warn');
    if (res.status === 'unknown') return say(`"${res.w}": fewer than two pages use that word, so the machine has never seen it. No cost.`, 'warn');
    if (res.status === 'dup') return say(`You already said "${res.w}".`, 'warn');
    if (res.status === 'taboo') {
      g.strikes += 1;
      g.log.push({ w: res.w, kind: 'taboo' });
      say(`TABOO. "${res.w}" is a name, by the rule: capitalised in the middle of sentences, or in a page title. Strike: one word used.`, 'bad');
      $('word').value = '';
      return afterMove();
    }
    g.typed.push(res.id);
    const after = currentRank();
    g.log.push({ w: res.w, kind: 'word', rank: after, delta: g.typed.length === 1 ? 0 : before - after });
    $('word').value = '';
    const n = g.typed.length;
    say(after === 1 ? `"${res.w}" does it.` : (g.typed.length === 1 ? `"${res.w}": ${D.chars[g.target].name} is #${after}.` :
      (before > after ? `"${res.w}" helps: #${before} to #${after}.` : (before < after ? `"${res.w}" hurts: #${before} to #${after}.` : `"${res.w}" changes nothing at the top.`))), after === 1 ? 'ok' : (before > after ? 'ok' : (before < after ? 'bad' : '')));
    afterMove();
  }
  function afterMove() {
    const r = render();
    if (g.typed.length && r.won) return finish(true);
    if (used() >= MAX) return finish(false);
  }

  function hint() {
    if (g.over || g.hints >= 2) return;
    const w = hintWord(g.hints);
    g.hints += 1;
    const shown = g.hints === 1 ? w[0] + ' ' + '_ '.repeat(w.length - 1).trim() + ` (${w.length} letters)` : w.slice(0, 3) + ' ' + '_ '.repeat(Math.max(0, w.length - 3)).trim() + ` (${w.length} letters)`;
    g.log.push({ w: shown, kind: 'hint' });
    say(`Hint: one of ${D.chars[g.target].name}'s strongest words begins like this. It cost ${HINT} words.`, 'warn');
    afterMove();
  }

  // ------------------------------------------------------------ the end of a target
  function finish(won, gaveUp) {
    g.over = true;
    g.won = won;
    const n = used();
    const pts = Core.points(won, n);
    total.score += pts;
    if (won) total.wins += 1;
    saveBest();
    $('s-score').textContent = total.score;
    $('s-wins').textContent = total.wins;
    $('word').disabled = true;
    $('send').disabled = true;
    $('giveup').disabled = true;
    $('hint').disabled = true;
    $('next').hidden = false;
    $('next').focus();
    const c = D.chars[g.target];
    const fp = Core.fingerprint(D, g.target);
    const typedWords = g.typed.map(id => D.words[id]);
    const oB = Core.oracleWord(D, g.target, 'banned'), oA = Core.oracleWord(D, g.target, 'allowed');
    const head = won ? `SOLVED in ${n} word${n > 1 ? 's' : ''}: +${pts}` : (gaveUp ? 'GAVE UP' : `OUT OF WORDS: ${c.name}`);
    $('reveal').hidden = false;
    $('reveal').innerHTML = `<h3 class="comic ${won ? 'ok' : 'bad'}">${esc(head)}</h3>
      <p>${won ? `${esc(c.name)} is first of 303 after ${g.typed.length} word${g.typed.length > 1 ? 's' : ''}${g.strikes ? `, plus ${g.strikes} strike${g.strikes > 1 ? 's' : ''}` : ''}${g.hints ? `, plus ${g.hints} hint${g.hints > 1 ? 's' : ''}` : ''}.` :
        `${esc(c.name)} finished at #${currentRank()} of 303.`} <a href="${esc(c.url)}" target="_blank" rel="noopener">Read the page &rarr;</a></p>
      <p><b>The oracle</b>, which has read all 303 pages, wins in <b>one word</b>: <b>${esc(oB.word)}</b> with names banned${oA.id !== oB.id ? `, <b>${esc(oA.word)}</b> with names allowed` : ''}. Its margin over the runner-up is ${oB.margin.toFixed(3)}. The machine is not hard; guessing the page's vocabulary is.</p>
      <p><b>${esc(c.name)}'s fingerprint</b>: the twelve non-name, non-habit words with the largest TF-IDF weight on the page. Gold ones you said.</p>
      <div class="pills">${fp.map(w => `<span class="type${typedWords.indexOf(w) >= 0 ? ' hit' : ''}">${esc(w)}</span>`).join('')}</div>`;
    say('', '');
    $('reveal').scrollIntoView && $('reveal').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  // ------------------------------------------------------------ wiring
  $('form').addEventListener('submit', e => { e.preventDefault(); submit($('word').value); });
  $('hint').addEventListener('click', hint);
  $('giveup').addEventListener('click', () => { if (g && !g.over) finish(false, true); });
  $('next').addEventListener('click', newTarget);
  document.querySelectorAll('.seg').forEach(seg => {
    seg.addEventListener('click', e => {
      const b = e.target.closest('button');
      if (!b) return;
      seg.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
      settings[seg.id.slice(2)] = b.dataset.v;
      if (g && !g.over && !g.typed.length && !g.strikes) newTarget();
      else if (g && !g.over) say('Applies from the next character.', 'warn');
    });
  });
  newTarget();
})();
