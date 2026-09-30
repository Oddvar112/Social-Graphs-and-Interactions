/* ===========================================================================
   SECRET IDENTITY: the interface. All rules live in core.js.

   Three case types, dealt from data/week5_secret.js:
     UNMASK    a character's page as a bag of words, one clue word at a time,
               four suspects; fewer clues, more points. A bag-of-words machine
               reads the same clues and commits, by confidence, on its own
               schedule: you see WHEN it commits, never to whom, until you do.
     IMPOSTOR  one sentence: from a real page, or from a trigram model trained
               on one Louvain community's pages?
     REDACTED  a concordance line with one word blanked; two of the three
               decoys share contexts with the answer (NLTK's similar()).
   Case of the day: a fixed case for everyone, seeded by the date, with a
   result you can copy and compare.
   =========================================================================== */
(function () {
  'use strict';
  const D = window.SECRET;
  if (!D) {
    document.body.innerHTML = '<p style="padding:40px;font:16px sans-serif;color:#fff">data/week5_secret.js is missing. ' +
      'Run <code>python scripts/build_week5_data.py</code>.</p>';
    return;
  }
  const Core = window.SecretCore;
  const IDX = Core.buildIndex(D);
  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const WEIGHT_NAME = { distinct: 'distinctive', nostop: 'no stopwords', raw: 'raw counts' };
  const ROSTER_NAME = { famous: 'the 80 best-known', full: 'the full roster' };
  const KEYS = ['1', '2', '3', '4'];
  const TODAY = new Date().toISOString().slice(0, 10);
  const DAILY = { rounds: 12, mix: 'mixed', weighting: 'distinct', decoys: 'lookalikes', roster: 'famous' };

  const settings = { rounds: 12, mix: 'mixed', weighting: 'distinct', decoys: 'lookalikes', roster: 'famous' };
  let game = null;        // { case, cur, score, machine, streak, results: [], daily }

  // ------------------------------------------------------------ persistence
  function storeKey(daily) {
    return daily ? `secret-identity:daily:${daily}`
      : `secret-identity:${settings.rounds}:${settings.mix}:${settings.weighting}:${settings.decoys}:${settings.roster}`;
  }
  function loadBest(daily) { try { return JSON.parse(localStorage.getItem(storeKey(daily)) || 'null'); } catch (e) { return null; } }
  function saveBest(b, daily) { try { localStorage.setItem(storeKey(daily), JSON.stringify(b)); } catch (e) { /* private mode */ } }
  function showBest(el, daily) {
    const b = loadBest(daily);
    el.textContent = b ? `Your best ${daily ? 'today' : 'with these settings'}: ${b.score} of ${b.max} (machine ${b.machine}).`
      : (daily ? '' : 'No case with these settings yet.');
  }

  // ------------------------------------------------------------ options
  document.querySelectorAll('.seg').forEach(seg => {
    seg.addEventListener('click', e => {
      const b = e.target.closest('button');
      if (!b) return;
      seg.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
      const key = seg.id.slice(2);
      settings[key] = key === 'rounds' ? Number(b.dataset.v) : b.dataset.v;
      showBest($('best'));
    });
  });

  // ------------------------------------------------------------ the case
  function startCase(daily) {
    const opts = daily ? Object.assign({ seed: Core.dailySeed(TODAY) }, DAILY)
      : Object.assign({ seed: (Date.now() ^ (Math.random() * 1e9)) >>> 1 }, settings);
    const cs = Core.makeCase(D, IDX, opts);
    game = { case: cs, cur: 0, score: 0, machine: 0, streak: 0, bonus: 0, results: [], daily: daily ? TODAY : null };
    $('start').hidden = true;
    $('end').hidden = true;
    renderSettings();
    renderStrip();
    renderRound();
    updateHud();
  }
  function updateHud() {
    $('s-score').textContent = game ? game.score : 0;
    $('s-machine').textContent = game ? game.machine : 0;
    $('s-streak').textContent = game ? game.streak : 0;
    $('s-round').textContent = game ? `${Math.min(game.cur + 1, game.case.rounds.length)} / ${game.case.rounds.length}` : '–';
  }
  function renderSettings() {
    const o = game.case.opts;
    $('settings').innerHTML = (game.daily ? `<div class="kv"><span>Case of the day</span><span>${esc(game.daily)}</span></div>` : '') +
      `<div class="kv"><span>Case types</span><span>${esc(o.mix)}</span></div>` +
      `<div class="kv"><span>Suspects from</span><span>${esc(ROSTER_NAME[o.roster])}</span></div>` +
      `<div class="kv"><span>Clue order</span><span>${esc(WEIGHT_NAME[o.weighting])}</span></div>` +
      `<div class="kv"><span>Decoys</span><span>${esc(o.decoys)}</span></div>` +
      `<div class="kv"><span>Max points</span><span>${Core.maxPoints(game.case.rounds)}</span></div>`;
  }
  function renderStrip() {
    const rs = game.case.rounds;
    $('strip').innerHTML = rs.map((r, i) => {
      const res = game.results[i];
      const cls = ['tile', r.type[0], i === game.cur ? 'now' : '', res ? (res.correct ? 'ok' : 'bad') : ''].join(' ');
      return `<div class="${cls}" title="${r.type}">${r.type[0].toUpperCase()}${res ? `<span class="pts">${res.points}</span>` : ''}</div>`;
    }).join('');
  }

  // ------------------------------------------------------------ rounds
  let state = null;       // per-round working state
  function renderRound() {
    const r = game.case.rounds[game.cur];
    const card = $('card');
    card.classList.remove('pop');
    void card.offsetWidth;
    card.classList.add('pop');
    state = { shown: 0, done: false, machineSeen: false };
    if (r.type === 'unmask') renderUnmask(r, card);
    else if (r.type === 'impostor') renderImpostor(r, card);
    else renderRedacted(r, card);
    renderStrip();
    $('desk').scrollTop = 0;
  }

  function clueRow(c, i) {
    return `<li class="clue"><span class="no">#${i + 1}</span><span class="w type">${esc(c.w)}</span>` +
      `<span class="meta"><b>&times;${c.count}</b> on this page<br>in ${c.df} of 303 pages</span></li>`;
  }
  function renderUnmask(r, card) {
    const n = game.cur + 1;
    card.innerHTML = `<span class="tab">Unmask &middot; case ${n}</span><span class="caseno type">file ${String(r.answer + 1).padStart(3, '0')}</span>
      <h2 class="comic">Who is this page about?</h2>
      <p class="hint">One page, reduced to its words, <b>${esc(WEIGHT_NAME[game.case.opts.weighting])}</b> first. Their own names are blacked out.
      Every clue you take costs a point: name them after one clue for 10, after all ten for 1. The machine reads along and commits on its own.</p>
      <div class="race"><ul class="clues" id="clues"></ul>
        <div class="machine" id="mach"><span class="lbl">The machine</span><span class="st reading" id="mach-st">waiting for a clue</span>
        <span class="sub" id="mach-sub">it reads at least two clues and commits once one suspect scores twice the runner-up; you won't see to whom until you commit</span></div></div>
      <div class="actions"><button class="btn gold" id="b-clue">Another clue <span id="left"></span></button></div>
      <span class="redline"></span>
      <div class="suspects" id="suspects">${r.options.map((ci, i) => `<button class="suspect" data-i="${i}"><span class="key">${i + 1}</span><span class="nm">${esc(D.chars[ci].name)}<small>${esc(D.chars[ci].tokens.toLocaleString())} words on the page</small></span></button>`).join('')}</div>
      <div id="reveal"></div>`;
    $('b-clue').addEventListener('click', addClue);
    card.querySelectorAll('.suspect').forEach(b => b.addEventListener('click', () => guessUnmask(Number(b.dataset.i))));
    const img = D.chars[r.answer].img;
    if (img) { const im = new Image(); im.decoding = 'async'; im.src = img; }
    addClue();
  }
  function addClue() {
    const r = game.case.rounds[game.cur];
    if (state.done || state.shown >= r.clues.length) return;
    const c = r.clues[state.shown];
    $('clues').insertAdjacentHTML('beforeend', clueRow(c, state.shown));
    state.shown += 1;
    const left = r.clues.length - state.shown;
    $('left').textContent = left ? `(${left} left)` : '';
    $('b-clue').disabled = left === 0;
    updateMachineWidget(r);
  }
  function updateMachineWidget(r) {
    const st = $('mach-st'), sub = $('mach-sub'), box = $('mach');
    if (!st) return;
    if (r.commit.k <= state.shown) {
      st.textContent = `committed after ${r.commit.k} clue${r.commit.k > 1 ? 's' : ''}`;
      st.className = 'st locked';
      sub.textContent = r.commit.forced ? 'it never got a clear lead and had to guess on the last clue' : 'it is done reading; its suspect is sealed';
      if (!state.machineSeen) { state.machineSeen = true; box.classList.remove('slam'); void box.offsetWidth; box.classList.add('slam'); }
    } else {
      st.textContent = `still reading (${state.shown} of ${r.clues.length})`;
      st.className = 'st reading';
      sub.textContent = state.shown < 2 ? 'it will not commit on a single clue' : 'no suspect scores twice the runner-up yet';
    }
  }
  function guessUnmask(i) {
    if (state.done) return;
    state.done = true;
    const r = game.case.rounds[game.cur];
    const pick = r.options[i];
    const correct = pick === r.answer;
    const mpts = Core.machinePoints(r);
    const pts = Core.points('unmask', correct, state.shown);
    finishRound(correct, pts, mpts, { clues: state.shown, commit: r.commit.k, mright: r.commit.pick === r.answer });
    const card = $('card');
    card.querySelectorAll('.suspect').forEach((b, k) => {
      b.disabled = true;
      if (r.options[k] === r.answer) b.classList.add('right');
      else if (k === i) b.classList.add('wrong');
      if (r.options[k] === r.commit.pick) b.classList.add('machine');
    });
    $('b-clue').disabled = true;
    // the widget unseals
    $('mach-st').textContent = `committed after ${r.commit.k}: ${r.commit.pick >= 0 ? esc(D.chars[r.commit.pick].name) : 'nobody'}`;
    $('mach-st').className = 'st ' + (r.commit.pick === r.answer ? 'won' : 'lost');
    $('mach-sub').textContent = r.commit.pick === r.answer ? `right, ${mpts} points` : 'wrong, 0 points';
    const ch = D.chars[r.answer];
    const mname = r.commit.pick >= 0 ? esc(D.chars[r.commit.pick].name) : 'nobody';
    const machineLine = r.commit.pick === r.answer
      ? `The machine committed to ${esc(ch.name)} after <b>${r.commit.k}</b> clue${r.commit.k > 1 ? 's' : ''}${r.commit.forced ? ', forced on the last one,' : ''} and scores ${mpts}.`
      : `The machine committed to <b>${mname}</b> after ${r.commit.k} clue${r.commit.k > 1 ? 's' : ''}${r.commit.forced ? ' (a forced guess on the last one)' : ''}: wrong, 0 points.` +
        (r.par ? ` Reading on, it would have been right from clue ${r.par}.` : ' Even all ten clues never pointed at the right page.');
    // the race, suspect by suspect, at the moment the machine committed
    const sc = r.commit.trace[r.commit.k - 1];
    const smax = Math.max(...sc, 1e-9);
    const race = r.options.map((ci, k) => {
      const lead = r.commit.trace.findIndex(t => t.indexOf(Math.max(...t)) === k && Math.max(...t) > 0);
      return `<div class="rrow${ci === r.answer ? ' ans' : ''}${ci === r.commit.pick ? ' mpick' : ''}"><span class="rn">${esc(D.chars[ci].name)}</span>` +
        `<span class="rb"><i style="width:${(100 * sc[k] / smax).toFixed(1)}%"></i></span><span class="rv">${sc[k].toFixed(3)}${lead >= 0 && lead < r.commit.k ? ` &middot; led at clue ${lead + 1}` : ''}</span></div>`;
    }).join('');
    const lists = ['distinct', 'nostop', 'raw'].map(w => `<div><b>${esc(WEIGHT_NAME[w])}</b>${Core.clueList(D, r.answer, w).slice(0, 7).map(c => `<i>${esc(c.w)}</i>`).join(', ')}</div>`).join('');
    $('reveal').innerHTML = `<div class="reveal${ch.img ? '' : ' noimg'}">${ch.img ? `<div class="pic"><img src="${esc(ch.img)}" alt="${esc(ch.name)}" onerror="this.parentNode.innerHTML='<div class=ph>${esc(ch.name)}</div>'"></div>` : ''}
      <div><h3 class="comic ${correct ? 'ok' : 'bad'}">${correct ? 'UNMASKED' : 'WRONG SUSPECT'}: ${esc(ch.name)}</h3>
      <p class="pts">${correct ? `You needed ${state.shown} clue${state.shown > 1 ? 's' : ''}: ${pts} points.` : `It was <a href="${esc(ch.url)}" target="_blank" rel="noopener">${esc(ch.name)}</a>. 0 points.`}${bonusText()}</p>
      <p>${machineLine}</p>
      <div class="racebox"><b>Cosine against the clues, after ${r.commit.k} clue${r.commit.k > 1 ? 's' : ''}</b>${race}</div>
      <p>${esc(ch.name)}: ${ch.kin} characters link here, ${ch.tokens.toLocaleString()} words, ${ch.types.toLocaleString()} distinct${ch.com >= 0 ? `, community <b>${esc(D.communities.names[ch.com])}</b>` : ''}. <a href="${esc(ch.url)}" target="_blank" rel="noopener">Read the page &rarr;</a></p>
      <div class="lists">${lists}</div>
      ${ch.img ? '<p class="credit">Lead image of the Wikipedia article, loaded from Wikipedia; the art belongs to Marvel and is shown for identification.</p>' : ''}
      <div class="actions"><button class="btn primary" id="b-next">${nextLabel()}</button></div></div></div>`;
    card.insertAdjacentHTML('beforeend', `<div class="stamp ${correct ? 'ok' : ''}">${correct ? 'UNMASKED' : 'WRONG'}</div>`);
    $('b-next').addEventListener('click', nextRound);
    $('b-next').focus();
  }

  function renderImpostor(r, card) {
    const it = D.impostor[r.item];
    card.innerHTML = `<span class="tab imp">Impostor &middot; case ${game.cur + 1}</span><span class="caseno type">exhibit ${String(r.item + 1).padStart(3, '0')}</span>
      <h2 class="comic">Real sentence, or Markov chain?</h2>
      <p class="hint">One of the 303 pages wrote this, or a <b>trigram model</b> trained on one community's pages did: it only ever knows the last two words. Both were put through the same detokenizer, so the formatting cannot tell you.</p>
      <div class="quote type">&ldquo;${esc(it.t)}&rdquo;</div>
      <div class="big2"><button class="btn" id="b-real">REAL <small style="display:block;font-size:11px;font-weight:600">a Wikipedia page said this (R)</small></button>
      <button class="btn gold" id="b-fake">MARKOV <small style="display:block;font-size:11px;font-weight:600">a trigram model made it up (M)</small></button></div>
      <div id="reveal"></div>`;
    $('b-real').addEventListener('click', () => guessImpostor(1));
    $('b-fake').addEventListener('click', () => guessImpostor(0));
  }
  function guessImpostor(saidReal) {
    if (state.done) return;
    state.done = true;
    const r = game.case.rounds[game.cur];
    const it = D.impostor[r.item];
    const correct = saidReal === it.real;
    const pts = Core.points('impostor', correct, 0);
    finishRound(correct, pts, 0, {});
    $('b-real').disabled = $('b-fake').disabled = true;
    const src = it.real ? `<b>Real.</b> It is on <a href="${esc(D.chars[it.src].url)}" target="_blank" rel="noopener">${esc(D.chars[it.src].name)}</a>'s page.`
      : `<b>Generated.</b> The <b>${esc(D.communities.names[it.com])}</b> community's trigram model wrote it, trained on the ${D.communities.sizes[it.com]} pages of that community. No nine-word run of it appears anywhere in the corpus.`;
    $('reveal').innerHTML = `<div class="reveal noimg"><div><h3 class="comic ${correct ? 'ok' : 'bad'}">${correct ? 'CORRECT' : 'FOOLED'}</h3>
      <p class="pts">${pts} points.${bonusText()}</p><p>${src}</p>
      <p>${it.real ? 'Real sentences carry long-range structure a trigram model cannot: a subject that is still the subject twelve words later.' : 'Every three consecutive words of a generated sentence occurred in the corpus. The seams are where the sentence forgets what it was about.'}</p>
      <div class="actions"><button class="btn primary" id="b-next">${nextLabel()}</button></div></div></div>`;
    $('card').insertAdjacentHTML('beforeend', `<div class="stamp ${correct ? 'ok' : ''}">${correct ? (it.real ? 'REAL' : 'MARKOV') : 'FOOLED'}</div>`);
    $('b-next').addEventListener('click', nextRound);
    $('b-next').focus();
  }

  function renderRedacted(r, card) {
    const it = D.redacted[r.item];
    card.innerHTML = `<span class="tab red">Redacted &middot; case ${game.cur + 1}</span><span class="caseno type">line ${String(r.item + 1).padStart(3, '0')}</span>
      <h2 class="comic">Which word was blacked out?</h2>
      <p class="hint">A concordance line from one of the pages with the key word removed. Two of the wrong answers are words that occur in the <b>same contexts</b> as the right one somewhere in the corpus: NLTK's <i>similar()</i>. Similar context is not similar meaning.</p>
      <div class="quote type" id="quote">&ldquo;${esc(it.t).replace('▮▮▮▮', '<span class="blank">▮▮▮▮▮▮</span>')}&rdquo;</div>
      <div class="choices">${it.o.map((w, i) => `<button class="suspect" data-i="${i}"><span class="key">${i + 1}</span><span class="nm type">${esc(w)}</span></button>`).join('')}</div>
      <div id="reveal"></div>`;
    card.querySelectorAll('.suspect').forEach(b => b.addEventListener('click', () => guessRedacted(Number(b.dataset.i))));
  }
  function guessRedacted(i) {
    if (state.done) return;
    state.done = true;
    const r = game.case.rounds[game.cur];
    const it = D.redacted[r.item];
    const correct = it.o[i].toLowerCase() === it.a.toLowerCase();
    const pts = Core.points('redacted', correct, 0);
    finishRound(correct, pts, 0, {});
    $('card').querySelectorAll('.suspect').forEach((b, k) => {
      b.disabled = true;
      if (it.o[k].toLowerCase() === it.a.toLowerCase()) b.classList.add('right');
      else if (k === i) b.classList.add('wrong');
    });
    const q = $('quote');
    q.classList.add('rev');
    q.querySelector('.blank').textContent = it.a;
    const sims = it.o.filter((w, k) => it.sim[k]);
    $('reveal').innerHTML = `<div class="reveal noimg"><div><h3 class="comic ${correct ? 'ok' : 'bad'}">${correct ? 'RESTORED' : 'STILL REDACTED'}: ${esc(it.a)}</h3>
      <p class="pts">${pts} points.${bonusText()}</p>
      <p>From <a href="${esc(D.chars[it.src].url)}" target="_blank" rel="noopener">${esc(D.chars[it.src].name)}</a>'s page. <b>${esc(it.a.toLowerCase())}</b> occurs ${it.n} times across ${it.df} of the 303 pages.</p>
      <p>${sims.length ? `<b>${sims.map(esc).join('</b> and <b>')}</b> came from <i>similar()</i>: somewhere in the corpus they sit between the same left and right words as ${esc(it.a.toLowerCase())}. That is all the method knows; it has no dictionary.` : 'No word in the corpus shared enough contexts with this one to serve as a decoy, so all three are frequency-matched strangers.'}</p>
      <div class="actions"><button class="btn primary" id="b-next">${nextLabel()}</button></div></div></div>`;
    $('card').insertAdjacentHTML('beforeend', `<div class="stamp ${correct ? 'ok' : ''}">${correct ? 'RESTORED' : 'WRONG'}</div>`);
    $('b-next').addEventListener('click', nextRound);
    $('b-next').focus();
  }

  // ------------------------------------------------------------ scoring
  function finishRound(correct, pts, mpts, extra) {
    let bonus = 0;
    if (correct) {
      game.streak += 1;
      if (game.streak % 3 === 0) bonus = 2;
    } else game.streak = 0;
    game.bonus = bonus;
    game.score += pts + bonus;
    game.machine += mpts;
    game.results[game.cur] = Object.assign({ correct, points: pts + bonus, machine: mpts }, extra);
    updateHud();
    renderStrip();
  }
  function bonusText() { return game.bonus ? ` <b>+${game.bonus} streak bonus.</b>` : ''; }
  function nextLabel() { return game.cur + 1 >= game.case.rounds.length ? 'Close the case' : 'Next'; }
  function nextRound() {
    game.cur += 1;
    if (game.cur >= game.case.rounds.length) return endCase();
    renderRound();
    updateHud();
  }

  function shareText() {
    const rs = game.case.rounds, res = game.results;
    const squares = rs.map((r, i) => (res[i].correct ? '🟩' : '🟥')).join('');
    const head = game.daily ? `SECRET IDENTITY · case of the day ${game.daily}` : `SECRET IDENTITY · ${game.case.opts.rounds} rounds, ${game.case.opts.mix}, ${WEIGHT_NAME[game.case.opts.weighting]}, ${game.case.opts.decoys}`;
    return `${head}\nme ${game.score} / ${Core.maxPoints(rs)} · the bag-of-words machine ${game.machine}\n${squares}\n${location.href.split('?')[0]}`;
  }
  function endCase() {
    const rs = game.case.rounds, res = game.results;
    const max = Core.maxPoints(rs);
    const byType = {};
    rs.forEach((r, i) => {
      const t = byType[r.type] = byType[r.type] || { n: 0, ok: 0, clues: 0, mk: 0, mok: 0 };
      t.n += 1;
      t.ok += res[i].correct ? 1 : 0;
      if (r.type === 'unmask') { t.clues += res[i].clues; t.mk += r.commit.k; t.mok += r.commit.pick === r.answer ? 1 : 0; }
    });
    const u = byType.unmask;
    $('e-grid').innerHTML = `<div class="fact"><b>${game.score}</b><span>your score of ${max}</span></div>
      <div class="fact m"><b>${game.machine}</b><span>the machine's</span></div>` +
      Object.entries(byType).map(([t, v]) => `<div class="fact"><b>${v.ok}/${v.n}</b><span>${t} right</span></div>`).join('') +
      (u ? `<div class="fact"><b>${(u.clues / u.n).toFixed(1)}</b><span>your clues per unmask</span></div>` : '') +
      (u ? `<div class="fact m"><b>${(u.mk / u.n).toFixed(1)}</b><span>machine's clues, ${u.mok}/${u.n} right</span></div>` : '');
    let verdict;
    if (!u) verdict = `No unmask rounds, so no machine to beat. ${game.score} of ${max} points.`;
    else if (game.score > game.machine) verdict = `<b>You beat the bag of words</b> by ${game.score - game.machine} points. The machine only ever adds up weights and commits on a lead; you know that "Queens" and "Bugle" are one person and it does not.`;
    else if (game.score === game.machine) verdict = `<b>A draw with the machine.</b> Cosine similarity between a clue list and a page is a surprisingly good detective when the clues are distinctive words.`;
    else verdict = `<b>The machine wins</b> by ${game.machine - game.score}. It commits as soon as one suspect scores twice the runner-up, which with distinctive words is often the second or third clue. Try the same case with raw counts to watch it drown in "the" and "of" like everyone else.`;
    if (game.case.opts.weighting === 'raw') verdict += ' Raw counts hand every page the same first words, which is Zipf\'s law played as a game.';
    if (game.daily) verdict += ` This was the case of the day, ${game.daily}: everyone who opens it today gets the same rounds, so the numbers compare.`;
    $('e-verdict').innerHTML = verdict;
    $('e-rounds').innerHTML = rs.map((r, i) => {
      const x = res[i];
      let what;
      if (r.type === 'unmask') what = `<b>${esc(D.chars[r.answer].name)}</b> &middot; you ${x.clues} clue${x.clues > 1 ? 's' : ''}, machine ${r.commit.k} (${r.commit.pick === r.answer ? 'right' : 'wrong'})`;
      else if (r.type === 'impostor') { const it = D.impostor[r.item]; what = it.real ? `real, from <b>${esc(D.chars[it.src].name)}</b>` : `generated, <b>${esc(D.communities.names[it.com])}</b> model`; }
      else { const it = D.redacted[r.item]; what = `blank was <b>${esc(it.a)}</b>, from ${esc(D.chars[it.src].name)}`; }
      return `<li><span>${r.type[0].toUpperCase()}</span><span>${what}</span><span class="r ${x.correct ? 'ok' : 'bad'}">${x.correct ? '+' + x.points : '0'}</span></li>`;
    }).join('');
    $('e-title').textContent = game.score >= game.machine ? 'CASE CLOSED' : 'OUTSMARTED BY A VECTOR';
    $('e-kicker').textContent = game.daily ? `Case of the day · ${game.daily}` : 'Case closed';
    const best = loadBest(game.daily);
    if (!best || game.score > best.score) saveBest({ score: game.score, max, machine: game.machine }, game.daily);
    showBest($('e-best'), game.daily);
    $('b-share').textContent = 'Copy result';
    $('end').hidden = false;
  }

  // ------------------------------------------------------------ wiring
  $('b-start').addEventListener('click', () => startCase(false));
  $('b-daily').addEventListener('click', () => startCase(true));
  $('b-again').addEventListener('click', () => startCase(!!game.daily));
  $('b-settings').addEventListener('click', () => { $('end').hidden = true; $('start').hidden = false; showBest($('best')); });
  $('b-new').addEventListener('click', () => { $('end').hidden = true; $('start').hidden = false; showBest($('best')); });
  $('b-share').addEventListener('click', () => {
    const text = shareText();
    const done = () => { $('b-share').textContent = 'Copied'; };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, () => fallbackCopy(text, done));
    else fallbackCopy(text, done);
  });
  function fallbackCopy(text, done) {
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); done(); } catch (e) { window.prompt('Copy your result', text); }
    document.body.removeChild(ta);
  }
  document.addEventListener('keydown', e => {
    if (!game || !$('start').hidden || !$('end').hidden) return;
    const r = game.case.rounds[game.cur];
    if (!r) return;
    if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault();
      if (state.done) nextRound();
      else if (r.type === 'unmask') addClue();
      return;
    }
    if (state.done) return;
    const k = KEYS.indexOf(e.key);
    if (k >= 0) { if (r.type === 'unmask') guessUnmask(k); else if (r.type === 'redacted') guessRedacted(k); }
    if (r.type === 'impostor') { if (e.key === 'r' || e.key === 'R') guessImpostor(1); if (e.key === 'm' || e.key === 'M') guessImpostor(0); }
  });
  $('daily-date').textContent = TODAY;
  showBest($('best'));
  showBest($('daily-best'), TODAY);
  $('card').innerHTML = '<span class="tab">Case file</span><h2 class="comic">The desk is empty.</h2><p class="hint">Open a case from the sheet on top.</p>';
})();
