/* ===========================================================================
   IN OTHER WORDS: the studio. The rules are in core.js; this file is the
   lights, the podiums, the cards, the sound and the instant replay.

   A round: the banner slams in, the host's clue types itself out while both
   machines search and lock in, six portraits flip in, you answer (or the
   clock runs out), the machines' picks drop onto the cards, the answer lights
   up, the scores bump, and the instant replay shows what each machine went on:
   KEYWORD's words, coloured by what they added to its score, and MEANING's
   passages, with the clue's words marked in them.
   =========================================================================== */
(function () {
  'use strict';
  const D = window.SHOW;
  if (!D || !window.ShowCore) {
    document.body.innerHTML = '<p style="padding:40px;font:16px sans-serif;color:#fff">data/week7_show.js is missing. Run <code>python scripts/build_week7_show.py</code>.</p>';
    return;
  }
  const C = window.ShowCore;
  const M = C.build(D);
  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const reduced = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);
  const wait = ms => new Promise(r => setTimeout(r, reduced ? Math.min(ms, 80) : ms));
  const NAME = { you: 'YOU', kw: 'KEYWORD', dn: 'MEANING' };

  // ------------------------------------------------------------------ storage (per browser, optional)
  const store = {
    get(k, d) { try { const v = localStorage.getItem('in-other-words:' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('in-other-words:' + k, JSON.stringify(v)); } catch (e) { /* private mode */ } },
  };
  const settings = { sound: store.get('sound', true), clock: store.get('clock', true), hard: store.get('hard', false) };

  // ------------------------------------------------------------------ the contestants
  const AV = {
    you: '<svg viewBox="0 0 64 64" class="bot" aria-hidden="true"><circle cx="32" cy="32" r="28" fill="#18203c" stroke="#ffc93f" stroke-width="2.5"/>' +
      '<path d="M9 30C13 21 24 20 32 27C40 20 51 21 55 30C53 39 46 42 40 41C36 40 34 37 32 35C30 37 28 40 24 41C18 42 11 39 9 30Z" fill="#ffc93f"/>' +
      '<g class="eyes"><ellipse cx="23" cy="31" rx="4.6" ry="2.9" fill="#05070f"/><ellipse cx="41" cy="31" rx="4.6" ry="2.9" fill="#05070f"/>' +
      '<circle cx="24.5" cy="30.4" r="1.1" fill="#fff"/><circle cx="42.5" cy="30.4" r="1.1" fill="#fff"/></g>' +
      '<path d="M23 47q9 7 18 0" stroke="#ffc93f" stroke-width="2.5" fill="none" stroke-linecap="round"/></svg>',
    kw: '<svg viewBox="0 0 64 64" class="bot" aria-hidden="true"><line x1="32" y1="5" x2="32" y2="13" stroke="#5fe3ff" stroke-width="2"/>' +
      '<circle class="ant" cx="32" cy="5" r="3" fill="#5fe3ff"/><rect x="12" y="13" width="40" height="28" rx="6" fill="#18203c" stroke="#5fe3ff" stroke-width="2"/>' +
      '<g class="eyes"><rect x="19.5" y="21" width="9" height="7" rx="2" fill="#5fe3ff"/><rect x="35.5" y="21" width="9" height="7" rx="2" fill="#5fe3ff"/></g>' +
      '<rect class="lid" x="18" y="20" width="28" height="10" fill="#18203c"/>' +
      '<path d="M24 34h16" stroke="#3b4a78" stroke-width="3" stroke-linecap="round"/><path d="M27 34v0M32 34v0M37 34v0" stroke="#5fe3ff" stroke-width="3" stroke-linecap="round"/>' +
      '<rect x="16" y="43" width="32" height="17" rx="4" fill="#111830" stroke="#5fe3ff" stroke-width="2"/>' +
      '<text x="32" y="54.5" text-anchor="middle" font-size="7.2" font-weight="800" fill="#5fe3ff" font-family="Inter,Arial,sans-serif">TF-IDF</text>' +
      '<g class="glass"><circle cx="50" cy="40" r="7" fill="rgba(95,227,255,.2)" stroke="#e9edf9" stroke-width="2"/><line x1="55" y1="45" x2="60" y2="51" stroke="#e9edf9" stroke-width="3" stroke-linecap="round"/></g></svg>',
    dn: '<svg viewBox="0 0 64 64" class="bot" aria-hidden="true"><g class="waves"><path d="M7 18q-5 8 0 16M57 18q5 8 0 16" stroke="#a678ff" stroke-width="2" fill="none" stroke-linecap="round"/>' +
      '<path d="M3 14q-7 12 0 24M61 14q7 12 0 24" stroke="#a678ff" stroke-width="1.5" fill="none" stroke-linecap="round" opacity=".6"/></g>' +
      '<path d="M14 31A18 18 0 0 1 50 31Z" fill="rgba(166,120,255,.25)" stroke="#a678ff" stroke-width="2"/>' +
      '<path d="M20 27q3-6 7-2q3-6 7 0q3-6 7-1q3-2 4 3M23 20q4-4 8 0q4-4 8 0" stroke="#d6c2ff" stroke-width="1.6" fill="none" stroke-linecap="round"/>' +
      '<rect x="14" y="31" width="36" height="16" rx="8" fill="#18203c" stroke="#a678ff" stroke-width="2"/>' +
      '<g class="eyes"><circle cx="25" cy="39" r="3.4" fill="#a678ff"/><circle cx="39" cy="39" r="3.4" fill="#a678ff"/></g>' +
      '<rect class="lid" x="19" y="34" width="26" height="10" fill="#18203c" style="transform-origin:32px 39px"/>' +
      '<rect x="19" y="49" width="26" height="12" rx="4" fill="#111830" stroke="#a678ff" stroke-width="2"/>' +
      '<text x="32" y="57.8" text-anchor="middle" font-size="6.6" font-weight="800" fill="#a678ff" font-family="Inter,Arial,sans-serif">mxbai</text></svg>',
  };
  const PH = '<svg viewBox="0 0 64 64" aria-hidden="true"><circle cx="32" cy="22" r="12" fill="#8d99bd"/><path d="M8 60c2-14 12-22 24-22s22 8 24 22z" fill="#8d99bd"/>' +
    '<path d="M21 21q5-5 11 0q6-5 11 0q-2 5-6 5q-3 0-5-3q-2 3-5 3q-4 0-6-5z" fill="#05070f"/></svg>';

  // ------------------------------------------------------------------ sound: synthesised, no files
  const Sound = (function () {
    let ctx = null, noiseBuf = null;
    function ac() {
      if (!settings.sound) return null;
      try {
        if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
        if (ctx.state === 'suspended') ctx.resume();
        return ctx;
      } catch (e) { return null; }
    }
    function tone(f, t0, dur, type, vol, f2) {
      const a = ac(); if (!a) return;
      const o = a.createOscillator(), g = a.createGain(), s = a.currentTime + t0;
      o.type = type || 'sine';
      o.frequency.setValueAtTime(f, s);
      if (f2) o.frequency.exponentialRampToValueAtTime(f2, s + dur);
      g.gain.setValueAtTime(0.0001, s);
      g.gain.exponentialRampToValueAtTime(vol || 0.15, s + 0.012);
      g.gain.exponentialRampToValueAtTime(0.0001, s + dur);
      o.connect(g); g.connect(a.destination);
      o.start(s); o.stop(s + dur + 0.05);
    }
    function noise(t0, dur, vol, freq) {
      const a = ac(); if (!a) return;
      if (!noiseBuf) {
        noiseBuf = a.createBuffer(1, a.sampleRate * 0.5, a.sampleRate);
        const d = noiseBuf.getChannelData(0);
        for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
      }
      const src = a.createBufferSource(), f = a.createBiquadFilter(), g = a.createGain(), s = a.currentTime + t0;
      src.buffer = noiseBuf; f.type = 'bandpass'; f.frequency.value = freq || 1400; f.Q.value = 0.9;
      g.gain.setValueAtTime(vol || 0.06, s); g.gain.exponentialRampToValueAtTime(0.0001, s + dur);
      src.connect(f); f.connect(g); g.connect(a.destination);
      src.start(s); src.stop(s + dur + 0.02);
    }
    return {
      ding() { tone(880, 0, 0.2, 'triangle', 0.2); tone(1320, 0.11, 0.45, 'triangle', 0.2); tone(1760, 0.22, 0.5, 'sine', 0.08); },
      buzz() { tone(150, 0, 0.5, 'sawtooth', 0.11, 95); tone(156, 0, 0.5, 'square', 0.05, 98); },
      lock() { tone(520, 0, 0.07, 'square', 0.05); tone(780, 0.06, 0.09, 'square', 0.05); },
      tick() { tone(1850, 0, 0.035, 'square', 0.035); },
      pick() { tone(640, 0, 0.08, 'triangle', 0.12); tone(960, 0.05, 0.1, 'triangle', 0.08); },
      flip() { noise(0, 0.09, 0.035, 2600); },
      slam() { noise(0, 0.35, 0.09, 300); tone(110, 0, 0.3, 'sine', 0.18, 60); },
      drum() { for (let i = 0; i < 12; i++) noise(i * 0.05, 0.05, 0.03 + i * 0.004, 240); },
      fanfare() {
        [523, 659, 784, 1047].forEach((f, i) => tone(f, i * 0.13, 0.32, 'triangle', 0.15));
        tone(1047, 0.55, 1.0, 'triangle', 0.13); tone(784, 0.55, 1.0, 'sine', 0.07); tone(659, 0.55, 1.0, 'sine', 0.06);
      },
      sad() { tone(392, 0, 0.32, 'triangle', 0.13); tone(370, 0.3, 0.32, 'triangle', 0.13); tone(349, 0.6, 0.7, 'triangle', 0.13); },
    };
  })();

  // ------------------------------------------------------------------ confetti
  const Confetti = (function () {
    const cv = $('confetti'), cx = cv.getContext('2d');
    const colors = ['#ffc93f', '#5fe3ff', '#a678ff', '#ff4b50', '#4ade80', '#ffffff'];
    let parts = [], raf = 0;
    function size() { cv.width = innerWidth * devicePixelRatio; cv.height = innerHeight * devicePixelRatio; }
    addEventListener('resize', size);
    size();
    function burst(x, y, n, spread) {
      if (reduced) return;
      for (let i = 0; i < n; i++) {
        const a = -Math.PI / 2 + (Math.random() - 0.5) * (spread || 2.4), v = 6 + Math.random() * 9;
        parts.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, r: 3 + Math.random() * 4, c: colors[i % colors.length],
          rot: Math.random() * 6, vr: (Math.random() - 0.5) * 0.4, life: 0, sq: Math.random() < 0.6 });
      }
      if (!raf) raf = requestAnimationFrame(step);
    }
    function step() {
      const d = devicePixelRatio;
      cx.clearRect(0, 0, cv.width, cv.height);
      parts = parts.filter(p => p.life < 170 && p.y < innerHeight + 60);
      for (const p of parts) {
        p.life += 1; p.vy += 0.27; p.vx *= 0.99; p.x += p.vx; p.y += p.vy; p.rot += p.vr;
        cx.save(); cx.translate(p.x * d, p.y * d); cx.rotate(p.rot);
        cx.globalAlpha = Math.max(0, 1 - p.life / 170); cx.fillStyle = p.c;
        if (p.sq) cx.fillRect(-p.r * d, -p.r * 0.45 * d, p.r * 2 * d, p.r * 0.9 * d);
        else { cx.beginPath(); cx.arc(0, 0, p.r * 0.8 * d, 0, 7); cx.fill(); }
        cx.restore();
      }
      raf = parts.length ? requestAnimationFrame(step) : 0;
      if (!raf) cx.clearRect(0, 0, cv.width, cv.height);
    }
    return {
      fromEl(el, n) { if (!el) return; const r = el.getBoundingClientRect(); burst(r.left + r.width / 2, r.top + r.height / 3, n || 80); },
      rain(n) { for (let i = 0; i < 5; i++) setTimeout(() => burst(innerWidth * (0.12 + 0.19 * i), innerHeight * 0.3, n || 40, 3.2), i * 150); },
    };
  })();

  // ------------------------------------------------------------------ helpers
  function announce(text) { $('live').textContent = text; }
  function cardOf(p) { return $('lineup').querySelector(`.card[data-p="${p}"]`); }
  const hasChip = (card, k) => card.querySelector(`.chip.${k}`);
  function addChip(card, k, label, asked) {
    const box = card.querySelector('.chips');
    let chip = hasChip(card, k);
    if (!chip) { chip = document.createElement('span'); box.appendChild(chip); }
    chip.className = `chip ${k}${asked ? ' ask' : ''}`;
    chip.innerHTML = AV[k] + esc(label);
  }
  function setPressed(id, on) { $(id).setAttribute('aria-pressed', String(!!on)); }
  /** Split a text around the occurrences of a name, so a word-level pass can mark the rest. */
  function markPassage(text, content, name) {
    let shared = 0;
    const seen = new Set();
    const words = s => s.split(/([A-Za-z]+)/).map(part => {
      if (!/^[A-Za-z]+$/.test(part)) return esc(part);
      const w = part.toLowerCase();
      if (content.has(w)) { if (!seen.has(w)) { seen.add(w); shared += 1; } return `<mark>${esc(part)}</mark>`; }
      return esc(part);
    }).join('');
    if (!name) return { html: words(text), shared };
    const pieces = text.split(name);
    return { html: pieces.map(words).join(`<mark class="who">${esc(name)}</mark>`), shared };
  }
  function shortName(c) { return c.style === 'absent' ? 'Nobody (' + c.name + ')' : D.chars[c.target].name; }
  const RULES = D.meta.rules || {};
  const fmt = x => Number(x).toFixed(2);
  /** One sentence on why a machine said nobody, or why it did not when it should have. */
  function nobodyWhy(c, k, v) {
    const r = RULES[k];
    if (!r) return '';
    const x = c[k][r.feature], pick = D.chars[c[k].top[0][0]].name;
    const measure = r.feature === 'gap' ? 'standard deviations ahead of the runner-up' : 'standard deviations above the average page';
    if (c[k].nobody) return ` It said <b>NOBODY HERE</b>: its best page, ${esc(pick)}, was only ${fmt(x)} ${measure}, and it answers only from ${fmt(r.threshold)}.`;
    if (v.absent) return ` It answered anyway: its best page was ${fmt(x)} ${measure}, over its line of ${fmt(r.threshold)} for saying nobody.`;
    return '';
  }

  // ------------------------------------------------------------------ state
  let S = null;     // the episode
  let T = null;     // the clock

  function renderDots() {
    if (!S) { $('dots').innerHTML = ''; return; }
    const n = S.clues.length;
    $('dots').innerHTML = Array.from({ length: n }, (_, i) => {
      let cls = '';
      if (S && i < S.verdicts.length) cls = S.verdicts[i].you ? 'ok' : 'no';
      else if (S && i === S.i) cls = 'now';
      return `<li class="${cls}" title="Round ${i + 1}"></li>`;
    }).join('');
  }

  function renderPodiums() {
    const P = [['you', 'Picks from six portraits'], ['kw', 'TF-IDF &middot; searches all 303 pages'], ['dn', 'mxbai &middot; searches all 303 pages']];
    $('podiums').innerHTML = P.map(([k, sub]) => `<div class="podium ${k}" id="p-${k}"><div class="av">${AV[k]}</div><h3 class="comic">${NAME[k]}</h3>` +
      `<div class="score" id="s-${k}">${S.score[k]}</div><div class="sub">${sub}</div><div class="status"><span class="lamp"></span><span id="st-${k}">Ready</span>` +
      (k === 'you' ? '' : `<button class="ask" type="button" data-k="${k}"${S.asked[k] ? ' disabled' : ''} title="Once an episode: see its top three">ASK ${NAME[k]}</button>`) + '</div></div>').join('');
    $('podiums').querySelectorAll('.ask').forEach(b => b.addEventListener('click', () => ask(b.dataset.k)));
  }
  function podiumState(k, cls, label) {
    const el = $('p-' + k); if (!el) return;
    el.classList.remove('thinking', 'locked', 'right', 'wrong');
    if (cls) el.classList.add(cls);
    $('st-' + k).textContent = label;
  }
  function bumpScore(k) {
    const el = $('s-' + k); if (!el) return;
    el.textContent = S.score[k];
    el.classList.remove('bump'); void el.offsetWidth; el.classList.add('bump');
  }

  // ------------------------------------------------------------------ the intro
  function renderIntro() {
    const cast = [
      ['you', 'The contestant', 'Knows comics. Reads the clue, looks at six portraits and points. Gets one look at each machine\'s top three per episode.'],
      ['kw', 'TF-IDF &middot; week 6\'s machine', 'Counts the clue\'s words on every page; a rare word counts for a lot, a common one for nothing. A word that no page uses does not exist for it. Says nobody here when its best page is not clearly ahead of the next.'],
      ['dn', 'mxbai-embed-large &middot; a sentence encoder', 'Turns the clue and 4,479 passages into 1,024 numbers each and picks the page with the closest passage. Reads meaning, not spelling. Says nobody here when no page stands out from the crowd.'],
    ];
    $('cast').innerHTML = cast.map(([k, role, text]) => `<div class="who ${k}"><div class="av">${AV[k]}</div><h3 class="comic">${NAME[k]}</h3><div class="role">${role}</div><p>${text}</p></div>`).join('');
    const ep = C.episodeOfDate();
    $('ep-no').textContent = '#' + ep.number;
    const rec = store.get('record', null);
    $('record').textContent = rec && rec.played ? `On this browser: ${rec.played} episode${rec.played > 1 ? 's' : ''} played, both machines beaten in ${rec.beat}, best score ${rec.best} of 10.` : '';
    setPressed('o-clock', settings.clock);
    setPressed('o-hard', settings.hard);
    document.body.classList.toggle('hard', settings.hard);
  }
  function toIntro() {
    stopTimer();
    S = null;
    $('game').hidden = true; $('final').hidden = true; $('intro').hidden = false;
    $('eplabel').textContent = '';
    renderDots(); renderIntro();
    window.scrollTo({ top: 0 });
  }

  // ------------------------------------------------------------------ an episode
  function startEpisode(seed, number) {
    S = { seed, number, clues: C.episode(M, seed), i: -1, verdicts: [], score: { you: 0, kw: 0, dn: 0 }, asked: { kw: false, dn: false }, round: null };
    $('eplabel').textContent = 'Episode ' + number;
    $('intro').hidden = true; $('final').hidden = true; $('game').hidden = false;
    renderPodiums(); renderDots();
    window.scrollTo({ top: 0 });
    Sound.pick();
    nextRound();
  }

  async function nextRound() {
    if (!S) return;
    if (S.round && !S.round.over) return;            // one round at a time (a click and Enter together must not skip one)
    stopTimer();
    S.i += 1;
    if (S.i >= S.clues.length) return finish();
    const idx = S.clues[S.i], c = D.clues[idx];
    const R = S.round = { idx, c, pick: undefined, over: false, live: false, locked: {} };
    $('replay').hidden = true; $('replay').innerHTML = '';
    $('clue').innerHTML = ''; $('lineup').innerHTML = '';
    $('b-nobody').disabled = true; $('b-nobody').className = 'nobody';
    $('b-nobody').querySelector('.chips').innerHTML = '';
    document.querySelectorAll('.bubble').forEach(b => b.remove());
    setTimer(1, '');
    $('timer').classList.remove('hot'); $('r-sec').classList.remove('hot');
    $('timer').classList.toggle('off', !settings.clock);
    $('r-label').textContent = `Round ${S.i + 1} of ${S.clues.length}`;
    renderDots();
    podiumState('you', '', 'Listen to the clue');
    podiumState('kw', 'thinking', 'Searching 303 pages');
    podiumState('dn', 'thinking', 'Searching 4,479 passages');
    window.scrollTo({ top: 0, behavior: reduced ? 'auto' : 'smooth' });
    await banner(S.i === S.clues.length - 1 ? 'FINAL CLUE!' : `ROUND ${S.i + 1}`);
    if (S.round !== R) return;
    // the machines are instant; the pause before they lock in is theatre
    setTimeout(() => lockIn(R, 'kw'), 700 + Math.random() * 500);
    setTimeout(() => lockIn(R, 'dn'), 1300 + Math.random() * 600);
    await typeClue(c.clue);
    if (S.round !== R) return;
    await deal(R);
    if (S.round !== R) return;
    R.live = true;
    $('b-nobody').disabled = false;
    podiumState('you', 'thinking', 'Your move');
    announce(`Round ${S.i + 1}. ${c.clue}. Choose one of six, or nobody here.`);
    startTimer(R);
  }

  async function banner(text) {
    const b = $('banner'), t = $('banner-txt');
    t.textContent = text;
    b.hidden = false;
    t.style.animation = 'none'; void t.offsetWidth; t.style.animation = '';
    Sound.slam();
    await wait(1050);
    b.hidden = true;
  }

  function typeClue(text) {
    const words = text.split(/\s+/), step = reduced ? 0 : 40;
    $('clue').innerHTML = '<span class="q">“</span>' + words.map((w, i) => `<span class="w" style="animation-delay:${i * step}ms">${esc(w)}</span>`).join(' ') +
      `<span class="w q" style="animation-delay:${words.length * step}ms">”</span>`;
    return wait(words.length * step + 300);
  }

  async function deal(R) {
    const box = $('lineup');
    box.innerHTML = R.c.lineup.map((p, k) => {
      const ch = D.chars[p];
      const pic = ch.img ? `<img src="${esc(ch.img)}" alt="" decoding="async">` : `<span class="ph">${PH}</span>`;
      return `<button class="card" type="button" data-p="${p}" aria-label="${k + 1}: ${esc(ch.name)}" disabled>` +
        `<span class="frame">${pic}<span class="num">${k + 1}</span><span class="stamp"></span></span><span class="nm">${esc(ch.name)}</span><span class="chips"></span></button>`;
    }).join('');
    const cards = Array.from(box.children);
    cards.forEach(el => {
      const img = el.querySelector('img');
      if (img) img.addEventListener('error', () => { const s = document.createElement('span'); s.className = 'ph'; s.innerHTML = PH; img.replaceWith(s); });
      el.addEventListener('click', () => answer(R, Number(el.dataset.p)));
      // once it has flipped in, hand the card back to plain CSS so hover can lift it (a finished animation would pin its transform)
      const done = e => { if (e.target !== el) return; el.classList.remove('in'); el.classList.add('shown'); el.removeEventListener('animationend', done); };
      el.addEventListener('animationend', done);
    });
    cards.forEach((el, k) => setTimeout(() => { el.classList.add('in'); Sound.flip(); }, reduced ? 0 : k * 105));
    await wait(cards.length * 105 + 420);
    cards.forEach(el => { el.disabled = false; });
  }

  function lockIn(R, k) {
    if (!S || S.round !== R || R.locked[k]) return;
    R.locked[k] = true;
    podiumState(k, 'locked', 'Locked in');
    Sound.lock();
  }

  // ------------------------------------------------------------------ the clock
  function setTimer(frac, label) { $('timer-bar').style.transform = `scaleX(${Math.max(0, frac)})`; $('r-sec').textContent = label; }
  function startTimer(R) {
    stopTimer();
    if (!settings.clock) { setTimer(1, ''); return; }
    const t0 = performance.now(), dur = C.DEFAULTS.timer * 1000;
    let last = C.DEFAULTS.timer + 1;
    T = { R, raf: 0 };
    const tick = now => {
      if (!T || T.R !== R || R.over) return;
      const left = Math.max(0, dur - (now - t0)), sec = Math.ceil(left / 1000);
      setTimer(left / dur, sec + 's');
      if (sec !== last) { last = sec; if (sec <= 5 && sec > 0) Sound.tick(); }
      $('timer').classList.toggle('hot', sec <= 5); $('r-sec').classList.toggle('hot', sec <= 5);
      if (left <= 0) { answer(R, null); return; }
      T.raf = requestAnimationFrame(tick);
    };
    T.raf = requestAnimationFrame(tick);
  }
  function stopTimer() { if (T) cancelAnimationFrame(T.raf); T = null; }

  // ------------------------------------------------------------------ a lifeline
  function ask(k) {
    const R = S && S.round;
    if (!R || R.over || !R.live || S.asked[k]) return;
    S.asked[k] = true;
    const el = $('p-' + k);
    el.querySelector('.ask').disabled = true;
    const top = R.c[k].top.slice(0, 3);
    const nobody = !!R.c[k].nobody;
    const b = document.createElement('div');
    b.className = 'bubble';
    b.innerHTML = `<button class="x" type="button" aria-label="Close">&times;</button><b>${NAME[k]}:</b> my top three of 303 pages` +
      `<ol>${top.map(([p, s]) => `<li><b>${esc(D.chars[p].name)}</b> <span style="color:#55607d">${s.toFixed(3)}</span>${cardOf(p) ? '' : ' <span style="color:#55607d">(not in your six)</span>'}</li>`).join('')}</ol>` +
      `<div style="margin-top:6px">${nobody ? 'My answer: <b>NOBODY HERE</b>. None of them stands out.' : `My answer: <b>${esc(D.chars[top[0][0]].name)}</b>.`}</div>`;
    el.appendChild(b);
    b.querySelector('.x').addEventListener('click', () => b.remove());
    top.forEach(([p], i) => { const card = cardOf(p); if (card) addChip(card, k, `#${i + 1}`, true); });
    if (nobody) addChip($('b-nobody'), k, 'ITS ANSWER', true);
    Sound.lock();
    announce(`${NAME[k]} says: ${top.map(([p]) => D.chars[p].name).join(', ')}. ${nobody ? 'Its answer: nobody here.' : ''}`);
  }

  // ------------------------------------------------------------------ an answer, and the reveal
  async function answer(R, pick) {
    if (!S || S.round !== R || R.over) return;
    if (!R.live && pick !== null) return;
    R.over = true;
    R.pick = pick;
    stopTimer();
    lockIn(R, 'kw'); lockIn(R, 'dn');
    const v = C.judge(R.c, pick);
    S.verdicts.push(v);
    Array.from($('lineup').children).forEach(el => { el.disabled = true; });
    $('b-nobody').disabled = true;
    if (pick === -1) $('b-nobody').classList.add('picked');
    else if (pick !== null) { const el = cardOf(pick); if (el) el.classList.add('picked'); }
    Sound.pick();
    podiumState('you', 'locked', pick === null ? 'Out of time' : 'Locked in');
    document.querySelectorAll('.bubble').forEach(b => b.remove());
    await reveal(R, v);
  }

  async function reveal(R, v) {
    const c = R.c;
    Sound.drum();
    document.body.classList.add('flash');
    await wait(650);
    // each machine's answer: its number one, or the NOBODY HERE buzzer
    const kc = v.kwNobody ? $('b-nobody') : cardOf(v.kwPick), dc = v.dnNobody ? $('b-nobody') : cardOf(v.dnPick);
    if (kc) addChip(kc, 'kw', 'KEYWORD');
    Sound.lock();
    await wait(430);
    if (dc) addChip(dc, 'dn', 'MEANING');
    Sound.lock();
    await wait(560);
    document.body.classList.remove('flash');
    const cards = Array.from($('lineup').children);
    cards.forEach(el => { el.classList.add('reveal'); if (Number(el.dataset.p) !== c.target) el.classList.add('dim'); });
    if (v.absent) $('b-nobody').classList.add('answer');
    else {
      const el = cardOf(c.target);
      el.classList.remove('dim'); el.classList.add('answer');
      el.querySelector('.frame').insertAdjacentHTML('beforeend', '<span class="ribbon">THE ANSWER</span>');
    }
    const wrong = new Set();
    if (!v.you && R.pick !== null && R.pick !== -1) wrong.add(R.pick);
    if (!v.kw && !v.kwNobody) wrong.add(v.kwPick);
    if (!v.dn && !v.dnNobody) wrong.add(v.dnPick);
    wrong.forEach(p => { const el = cardOf(p); if (el) { el.classList.remove('dim'); el.querySelector('.stamp').innerHTML = '<b>&times;</b>'; } });
    if (v.you) { Sound.ding(); Confetti.fromEl(v.absent ? $('b-nobody') : cardOf(c.target), 90); } else Sound.buzz();
    const said = { kw: v.kwNobody, dn: v.dnNobody };
    ['you', 'kw', 'dn'].forEach((k, i) => setTimeout(() => {
      if (v[k]) { S.score[k] += 1; bumpScore(k); }
      let label = v[k] ? 'Right' : 'Wrong';
      if (k === 'you' && R.pick === null) label = 'Out of time';
      if (k !== 'you' && said[k]) label = v[k] ? 'Right: nobody here' : 'Said nobody here';
      podiumState(k, v[k] ? 'right' : 'wrong', label);
    }, 200 + i * 170));
    renderDots();
    const answerOf = (nobody, p) => nobody ? 'nobody here' : D.chars[p].name;
    announce(`${v.you ? 'Right' : 'Wrong'}. The answer: ${v.absent ? 'nobody here, ' + c.name + ' has no page' : D.chars[c.target].name}. ` +
      `KEYWORD said ${answerOf(v.kwNobody, v.kwPick)}, MEANING said ${answerOf(v.dnNobody, v.dnPick)}.`);
    await wait(950);
    replay(R, v);
  }

  // ------------------------------------------------------------------ the instant replay
  function replay(R, v) {
    const c = R.c, box = $('replay');
    const kind = c.style === 'page' ? ['page', "In the page's own words"] : (c.style === 'other' ? ['other', 'In other words'] : ['absent', 'Not in the corpus']);
    let head;
    if (v.absent) {
      const said = v.kwNobody && v.dnNobody ? 'Both machines said nobody here.'
        : [v.kwNobody ? 'KEYWORD said nobody here' : `KEYWORD answered ${esc(D.chars[v.kwPick].name)}`,
          v.dnNobody ? 'MEANING said nobody here' : `MEANING answered ${esc(D.chars[v.dnPick].name)}`].join('; ') + '.';
      head = `<div class="pic none">?</div><div><span class="kind absent">${kind[1]}</span><h3 class="comic none">NOBODY HERE: ${esc(c.name.toUpperCase())}</h3>` +
        `<p>${esc(c.name)} has no page among the 303, so the right answer was <b>nobody here</b>. ${said} ` +
        `A search engine always has a number one; it only says nobody if you give it a rule for when its number one does not stand out.</p></div>`;
    } else {
      const t = D.chars[c.target];
      head = `<div class="pic">${t.img ? `<img src="${esc(t.img)}" alt="${esc(t.name)}">` : PH}</div><div><span class="kind ${kind[0]}">${kind[1]}</span><h3 class="comic">${esc(t.name.toUpperCase())}</h3>` +
        `<p>Where the right page ranked among 303: <b class="kw">KEYWORD #${c.ranks.kw}</b> &middot; <b class="dn">MEANING #${c.ranks.dn}</b> ` +
        `<small>(MEANING with 60-word passages #${c.ranks.dn60}; the course's smaller MiniLM #${c.ranks.mini})</small> &middot; ` +
        `<a href="${esc(t.url)}" target="_blank" rel="noopener">Read the page &rarr;</a></p></div>`;
    }
    const last = S.i === S.clues.length - 1;
    box.innerHTML = `<div class="rp-top"><span class="rec"></span>Instant replay</div><div class="rp-head">${head}</div>` +
      `<div class="rp-cols">${keywordColumn(c, v)}${meaningColumn(c, v)}</div>` +
      `<div class="rp-next"><span class="hint">or press Enter</span><button class="big" id="b-next" type="button">${last ? 'The final scores' : 'Next clue'} &#9654;</button></div>`;
    box.hidden = false;
    box.querySelectorAll('.pass .more').forEach(b => b.addEventListener('click', () => {
      const p = b.closest('.pass'); p.classList.toggle('open');
      b.textContent = p.classList.contains('open') ? 'Show less' : 'Show the whole passage';
    }));
    $('b-next').addEventListener('click', nextRound);
    $('b-next').focus({ preventScroll: true });
    setTimeout(() => box.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' }), 200);
    const next = S.clues[S.i + 1];
    if (next !== undefined) D.clues[next].lineup.forEach(p => { if (D.chars[p].img) { const im = new Image(); im.src = D.chars[p].img; } });
  }

  function keywordColumn(c, v) {
    const words = new Map(c.kw.words.map(([w, a, b, df]) => [w, { a, b, df }]));
    const unknown = new Set(c.kw.unknown);
    const pickScore = c.kw.top[0][1];
    const showRight = !v.absent && !v.kw;
    const toks = c.clue.split(/([A-Za-z]+)/).map(part => {
      if (!/^[A-Za-z]+$/.test(part)) return esc(part);
      const w = part.toLowerCase(), x = words.get(w);
      if (unknown.has(w)) return `<span class="unk" title="${esc(w)}: no page uses this word, so KEYWORD cannot see it">${esc(part)}</span>`;
      if (!x) return `<span class="hab">${esc(part)}</span>`;
      const share = pickScore > 0 ? x.a / pickScore : 0;
      const cls = [];
      let style = '';
      if (share > 0.02) {
        const alpha = 0.28 + 0.72 * Math.min(1, share * 1.5);
        cls.push('hit');
        style = ` style="background:rgba(95,227,255,${alpha.toFixed(2)});color:${alpha > 0.55 ? '#03222b' : '#e9fbff'}"`;
      }
      if (showRight && x.b > 0.0008) cls.push('rgt');
      if (!cls.length && x.df > 100) cls.push('hab');
      const tip = `${w}: on ${x.df} of 303 pages; adds ${x.a.toFixed(3)} to its pick` + (v.absent ? '' : `, ${x.b.toFixed(3)} to the right page`);
      return `<span class="${cls.join(' ')}"${style} title="${esc(tip)}">${esc(part)}</span>`;
    }).join('');
    let best = null;
    c.kw.words.forEach(([w, a]) => { if (!best || a > best.a) best = { w, a }; });
    const name = D.chars[v.kwPick].name;
    const share = best && pickScore > 0 ? best.a / pickScore : 0;
    const inName = best && (name.toLowerCase().match(/[a-z]+/g) || []).indexOf(best.w) >= 0;
    let text = (v.kw ? '<span class="ok">Right.</span>' : '<span class="bad">Wrong.</span>');
    text += c.kw.nobody ? nobodyWhy(c, 'kw', v) : ` Its number one: <b>${esc(name)}</b>, cosine ${pickScore.toFixed(3)}.` + nobodyWhy(c, 'kw', v);
    if (best && share >= 0.5) text += ` One word carried ${Math.round(share * 100)} percent of ${c.kw.nobody ? 'that page\'s score' : 'it'}: <b>${esc(best.w)}</b>` +
      `${inName && v.kwPick !== c.target ? `, which is part of the name ${esc(name)}` : ''}.`;
    if (unknown.size) text += ` It never saw ${[...unknown].map(w => `<b>${esc(w)}</b>`).join(', ')}: no page uses ${unknown.size > 1 ? 'those words' : 'that word'}.`;
    if (showRight) text += ` The right page came #${c.ranks.kw}.`;
    return `<div class="rp-col kw"><h4>${AV.kw}KEYWORD counted these words</h4><p class="toks">${toks}</p>` +
      `<div class="legend"><span><i class="l1"></i>counted toward its pick</span>${showRight ? '<span><i class="l2"></i>would have counted for the right page</span>' : ''}` +
      `${unknown.size ? '<span><i class="l3"></i>a word no page uses</span>' : ''}</div><p class="verdict">${text}</p></div>`;
  }

  function meaningColumn(c, v) {
    const P = c.dn.passages;
    const content = new Set(c.kw.words.filter(x => x[3] <= 100).map(x => x[0]));
    const who = v.absent ? c.name : null;
    const block = (p, cls, label) => {
      const x = P[String(p)];
      if (!x) return '';
      const m = markPassage(x.text, content, who);
      return `<div class="pass ${cls}"><header><b>${esc(D.chars[p].name)}</b><span>${label}</span><span class="sc">${x.score.toFixed(3)}</span></header>` +
        `<blockquote>${m.html}</blockquote><button class="more" type="button">Show the whole passage</button></div>`;
    };
    const nob = !!c.dn.nobody;
    const pickIsRight = !v.absent && v.dnPick === c.target;
    let body = pickIsRight ? block(v.dnPick, 'right', nob ? 'its best page, the right one, but it said nobody' : 'its number one, and the right page')
      : block(v.dnPick, 'pick', nob ? 'its best page (it said nobody)' : 'its number one') + (v.absent ? '' : block(c.target, 'right', `the right page's closest passage, #${c.ranks.dn}`));
    const x = P[String(v.dnPick)];
    const shared = markPassage(x.text, content, null).shared;
    let text = (v.dn ? '<span class="ok">Right.</span>' : '<span class="bad">Wrong.</span>');
    text += nob ? nobodyWhy(c, 'dn', v)
      : ` Its number one is the page with the passage closest to the clue in meaning: <b>${esc(D.chars[v.dnPick].name)}</b>, similarity ${x.score.toFixed(3)}.` + nobodyWhy(c, 'dn', v);
    text += ` Of the clue's ${content.size} distinctive words, ${shared} ${shared === 1 ? 'turns' : 'turn'} up in that passage.`;
    if (v.absent && x.text.indexOf(c.name) >= 0) text += ` The passage mentions <b>${esc(c.name)}</b>, on somebody else's page: the right text, the wrong person.`;
    return `<div class="rp-col dn"><h4>${AV.dn}MEANING matched this passage</h4>${body}<p class="verdict">${text}</p></div>`;
  }

  // ------------------------------------------------------------------ the end of the show
  function finish() {
    stopTimer();
    const t = C.tally(S.verdicts);
    const youAlone = t.winners.length === 1 && t.winners[0] === 'you';
    const youShare = t.winners.indexOf('you') >= 0;
    const rec = store.get('record', { played: 0, beat: 0, best: 0 });
    rec.played += 1; if (t.you > t.kw && t.you > t.dn) rec.beat += 1; rec.best = Math.max(rec.best, t.you);
    store.set('record', rec);
    const title = youAlone ? 'You beat both machines!' : (youShare ? 'A tie at the top.' : (t.winners.length > 1 ? 'The machines take it.' : `${NAME[t.winners[0]]} takes it.`));
    const by = { page: { n: 0, you: 0, kw: 0, dn: 0 }, other: { n: 0, you: 0, kw: 0, dn: 0 }, absent: { n: 0, you: 0, kw: 0, dn: 0 } };
    S.verdicts.forEach((v, i) => { const b = by[D.clues[S.clues[i]].style]; b.n += 1; b.you += v.you; b.kw += v.kw; b.dn += v.dn; });
    const max = Math.max(1, t.you, t.kw, t.dn);
    const split = (label, b) => `<div><h4>${label}</h4><p><b class="you">You ${b.you}/${b.n}</b> &middot; <b class="kw">KEYWORD ${b.kw}/${b.n}</b> &middot; <b class="dn">MEANING ${b.dn}/${b.n}</b></p></div>`;
    const mark = ok => ok ? '<span class="y">&#10003;</span>' : '<span class="n">&times;</span>';
    const rows = S.verdicts.map((v, i) => { const c = D.clues[S.clues[i]]; return `<tr><td>${i + 1}</td><td><b>${esc(shortName(c))}</b></td><td class="cl">${esc(c.clue)}</td><td class="m">${mark(v.you)}</td><td class="m">${mark(v.kw)}</td><td class="m">${mark(v.dn)}</td></tr>`; }).join('');
    const all = D.clues.filter(c => c.style === 'other');
    const kwAll = all.filter(c => C.judge(c, null).kw).length, dnAll = all.filter(c => C.judge(c, null).dn).length;
    const miniAll = all.filter(c => c.ranks.mini === 1).length;
    const share = C.shareText(S.number, S.verdicts);
    $('game').hidden = true;
    $('final').hidden = false;
    $('final').innerHTML = `<h2 class="final-title comic">THAT'S THE SHOW!</h2><p class="winner">${title}</p>` +
      `<div class="bars">${['you', 'kw', 'dn'].map(k => `<div class="bar ${k}${t.winners.indexOf(k) >= 0 ? ' win' : ''}"><div class="av">${AV[k]}</div>` +
        `<div class="col" data-h="${Math.round(100 * (0.14 + 0.86 * t[k] / max))}"><b>${t[k]}</b></div><div class="lbl">${NAME[k]}</div></div>`).join('')}</div>` +
      `<div class="split">${split("In the page's own words", by.page)}${split('In other words', by.other)}${split('Nobody here', by.absent)}</div>` +
      `<p class="foot" style="margin-top:16px">Across all ${all.length} clues we wrote in other words, KEYWORD gets ${kwAll} right and MEANING ${dnAll} ` +
      `(the course's smaller MiniLM would have found ${miniAll}). ` +
      `<a href="../index.html">Why, in the post &rarr;</a></p>` +
      `<div class="recap"><table><thead><tr><th>#</th><th>Answer</th><th class="cl">Clue</th><th>You</th><th>KW</th><th>MN</th></tr></thead><tbody>${rows}</tbody></table></div>` +
      `<pre class="share" id="share">${esc(share)}</pre>` +
      `<div class="go"><button class="big" id="b-copy" type="button">Copy result</button><button class="alt" id="b-again" type="button">Another episode</button><a class="alt" href="../index.html">How the machines work &rarr;</a></div>`;
    window.scrollTo({ top: 0 });
    setTimeout(() => $('final').querySelectorAll('.bar .col').forEach(el => { el.style.height = el.dataset.h + '%'; }), 120);
    if (youShare) { Sound.fanfare(); setTimeout(() => Confetti.rain(60), 500); } else Sound.sad();
    $('b-copy').addEventListener('click', () => copy(share));
    $('b-again').addEventListener('click', () => startEpisode(randomSeed(), 'R' + String(Math.floor(Math.random() * 9000) + 1000)));
    $('eplabel').textContent = 'Episode ' + S.number + ' · final';
    announce(`That's the show. You ${t.you}, KEYWORD ${t.kw}, MEANING ${t.dn}. ${title}`);
  }
  function copy(text) {
    const done = () => { $('b-copy').textContent = 'Copied'; };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, () => fallback());
    else fallback();
    function fallback() {
      const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy'); done(); } catch (e) { window.prompt('Copy your result', text); }
      ta.remove();
    }
  }
  function randomSeed() { return (Date.now() ^ Math.floor(Math.random() * 1e9)) >>> 1; }

  // ------------------------------------------------------------------ wiring
  $('b-tonight').addEventListener('click', () => { const ep = C.episodeOfDate(); startEpisode(ep.seed, String(ep.number)); });
  $('b-random').addEventListener('click', () => startEpisode(randomSeed(), 'R' + String(Math.floor(Math.random() * 9000) + 1000)));
  $('b-nobody').addEventListener('click', () => { const R = S && S.round; if (R) answer(R, -1); });
  $('home').addEventListener('click', e => { e.preventDefault(); toIntro(); });
  $('o-clock').addEventListener('click', () => { settings.clock = !settings.clock; store.set('clock', settings.clock); setPressed('o-clock', settings.clock); });
  $('o-hard').addEventListener('click', () => { settings.hard = !settings.hard; store.set('hard', settings.hard); setPressed('o-hard', settings.hard); document.body.classList.toggle('hard', settings.hard); });
  function soundUI() {
    setPressed('b-sound', settings.sound);
    $('snd-on').toggleAttribute('hidden', !settings.sound);
    $('snd-off').toggleAttribute('hidden', settings.sound);
    $('snd-lbl').textContent = settings.sound ? 'Sound on' : 'Sound off';
  }
  $('b-sound').addEventListener('click', () => { settings.sound = !settings.sound; store.set('sound', settings.sound); soundUI(); if (settings.sound) Sound.pick(); });
  document.addEventListener('keydown', e => {
    if (e.altKey || e.ctrlKey || e.metaKey || !S || $('game').hidden) return;
    const R = S.round;
    if (!R) return;
    if (!R.over && R.live) {
      if (/^[1-6]$/.test(e.key)) { const p = R.c.lineup[Number(e.key) - 1]; if (p !== undefined) { e.preventDefault(); answer(R, p); } }
      else if (e.key === 'n' || e.key === 'N') { e.preventDefault(); answer(R, -1); }
    } else if (R.over && !$('replay').hidden && e.key === 'Enter') {
      if (e.target && e.target.closest && e.target.closest('button, a')) return;   // the focused button handles its own Enter
      e.preventDefault(); nextRound();
    }
  });
  soundUI();
  renderDots();
  renderIntro();
})();
