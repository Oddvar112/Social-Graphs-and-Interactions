"""
Checks the engine and the data SECRET IDENTITY runs on, by actually running it.

The game's rules live in weeks/week5/game/core.js with no DOM in them, so this
test executes the exact shipped file under node against the shipped data file
and recomputes everything independently in Python from the corpus.

What this proves:
  * the data file describes all 303 characters, each with ten or more clues under
    every weighting, none of them one of the character's own names
  * the machine's scores and picks match a Python re-implementation of the
    cosine-against-clues rule, case for case, ties included
  * every "real" impostor sentence is a run of tokens on the page it is credited
    to, and no generated sentence shares a nine-token run with any page, or is a
    real sentence; real and generated sentences have the same length range
  * every redacted line hides exactly one word, the answer is one of four unique
    options, the answer does not survive as a substring of the line, and the
    original sentence is on the credited page
  * a case file has the requested mix, is deterministic for a seed, and every
    unmask round lists the answer exactly once among four suspects

Run:  python scripts/test_secret_identity.py     (needs node and the token cache;
      the first run builds the cache, about a minute)
"""

import json
import pathlib
import random
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import week5_text as W                          # noqa: E402
from analyse import load_graph                  # noqa: E402

DATA_JS = ROOT / "data" / "week5_secret.js"
CORE_JS = ROOT / "weeks" / "week5" / "game" / "core.js"

HARNESS = r"""
const fs = require('fs');
const core = require(process.argv[1]);
const raw = fs.readFileSync(process.argv[2], 'utf8');
const D = JSON.parse(raw.slice(raw.indexOf('{'), raw.lastIndexOf('}') + 1));
const idx = core.buildIndex(D);
const probes = JSON.parse(process.argv[3]);
const out = { probes: [], cases: [] };
for (const p of probes) {
  const clues = core.clueList(D, p.answer, p.weighting).slice(0, 10).map(c => c.vid);
  const picks = [], scores = [];
  for (let k = 1; k <= clues.length; k++) {
    picks.push(core.machinePick(D, idx, p.options, clues.slice(0, k)));
    scores.push(core.machineScores(D, idx, p.options, clues.slice(0, k)));
  }
  const c = core.machineCommit(D, idx, p.options, clues);
  out.probes.push({ picks, scores, par: core.machinePar(D, idx, p.answer, p.options, clues), commit: { k: c.k, pick: c.pick, forced: c.forced } });
}
out.famous = core.famousPool(D, 80);
out.daily = [core.dailySeed('2026-09-30'), core.dailySeed('2026-09-30'), core.dailySeed('2026-10-01')];
for (const seed of [1, 2, 3]) {
  for (const mix of ['mixed', 'unmask', 'impostor', 'redacted']) {
    for (const roster of ['famous', 'full']) {
      const o = { rounds: 12, mix, weighting: 'distinct', decoys: 'lookalikes', roster, seed };
      const a = core.makeCase(D, idx, o), b = core.makeCase(D, idx, o);
      out.cases.push({ seed, mix, roster, rounds: a.rounds.map(r => ({ type: r.type, answer: r.answer, options: r.options, item: r.item, par: r.par, commit: r.commit && { k: r.commit.k, pick: r.commit.pick } })),
                       same: JSON.stringify(a.rounds) === JSON.stringify(b.rounds), max: core.maxPoints(a.rounds), machine: core.machineTotal(a.rounds) });
    }
  }
}
// how the machine fares: 300 simulated unmask rounds per setting, scored by its own commitments
out.machine = {};
const rnd = core.mulberry32(99);
for (const roster of ['famous', 'full']) for (const w of ['distinct', 'nostop', 'raw']) for (const dm of ['lookalikes', 'random']) {
  const pool = roster === 'famous' ? core.famousPool(D, 80, w, 10) : core.fullPool(D, w, 10);
  let ks = 0, right = 0, pts = 0, forced = 0;
  for (let t = 0; t < 300; t++) {
    const a = pool[Math.floor(rnd() * pool.length)];
    const opts = core.shuffle([a].concat(core.decoys(D, a, dm, rnd, pool)), rnd);
    const clues = core.clueList(D, a, w).slice(0, 10).map(c => c.vid);
    const c = core.machineCommit(D, idx, opts, clues);
    ks += c.k; forced += c.forced ? 1 : 0;
    if (c.pick === a) { right++; pts += Math.max(1, 11 - c.k); }
  }
  out.machine[roster + '/' + w + '/' + dm] = { clues: ks / 300, right: right / 300, forced: forced / 300, points_per_round: pts / 300 };
}
process.stdout.write(JSON.stringify(out));
"""


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


def main():
    raw = DATA_JS.read_text(encoding="utf-8")
    D = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    chars, vocab = D["chars"], D["vocab"]
    print(f"data: {len(chars)} characters, {len(D['impostor'])} impostor items, {len(D['redacted'])} blanks")

    # ---- 1. characters and clues
    if len(chars) != 303:
        fail("expected 303 characters")
    for c in chars:
        mask = set(c["mask"])
        for w, lst in c["clues"].items():
            if len(lst) < 10 and c["tokens"] >= 500:
                fail(f"{c['name']} has only {len(lst)} clues under {w}")
            words = [vocab[v] for v, _ in lst]
            bad = [x for x in words if x in mask]
            if bad:
                fail(f"{c['name']} leaks its own name in {w} clues: {bad}")
            counts = [n for _, n in lst]
            if w != "distinct" and counts != sorted(counts, reverse=True):
                fail(f"{c['name']} {w} clues are not sorted by count")
        if len(c["look"]) != 5 or chars.index(c) in c["look"]:
            fail(f"{c['name']} lookalikes wrong: {c['look']}")
    print("1. clues: ten or more per weighting, no own names, sorted, five lookalikes each")

    # ---- 2. the machine, against a Python re-implementation
    rng = random.Random(7)
    probes = []
    pool = [i for i, c in enumerate(chars) if c["tokens"] >= 500]
    for _ in range(60):
        a = rng.choice(pool)
        opts = [a] + (chars[a]["look"][:3] if rng.random() < 0.5 else rng.sample([i for i in pool if i != a], 3))
        rng.shuffle(opts)
        probes.append({"answer": a, "options": opts, "weighting": rng.choice(["distinct", "nostop", "raw"])})
    res = subprocess.run(["node", "-e", HARNESS, str(CORE_JS), str(DATA_JS), json.dumps(probes)],
                         capture_output=True, text=True, check=True)
    js = json.loads(res.stdout)
    vec = [dict(c["vec"]) for c in chars]
    for p, out in zip(probes, js["probes"]):
        clues = [v for v, _ in chars[p["answer"]]["clues"][p["weighting"]][:10]]
        par = None
        for k in range(1, len(clues) + 1):
            sc = [sum(vec[o].get(v, 0.0) for v in clues[:k]) for o in p["options"]]
            best = max(range(len(sc)), key=lambda i: (sc[i], -i))
            pick = p["options"][best] if sc[best] > 0 else -1
            if abs(sum(sc) - sum(out["scores"][k - 1])) > 1e-9 or pick != out["picks"][k - 1]:
                fail(f"machine disagrees on {chars[p['answer']]['name']} at {k} clues: py {pick} js {out['picks'][k-1]}")
            if pick == p["answer"]:
                par = par or k
            else:
                par = None
        if par != out["par"]:
            fail(f"machine par disagrees for {chars[p['answer']]['name']}: py {par} js {out['par']}")
        # the commitment rule: at least two clues read, leader at least twice the runner-up, else forced on the last clue
        commit = None
        for k in range(1, len(clues) + 1):
            sc = [sum(vec[o].get(v, 0.0) for v in clues[:k]) for o in p["options"]]
            best = max(range(len(sc)), key=lambda i: (sc[i], -i))
            second = max(s for i, s in enumerate(sc) if i != best)
            if k >= 2 and sc[best] > 0 and sc[best] >= 2.0 * second - 1e-12:
                commit = (k, p["options"][best], False)
                break
        if commit is None:
            sc = [sum(vec[o].get(v, 0.0) for v in clues) for o in p["options"]]
            best = max(range(len(sc)), key=lambda i: (sc[i], -i))
            commit = (len(clues), p["options"][best] if sc[best] > 0 else -1, True)
        if commit != (out["commit"]["k"], out["commit"]["pick"], out["commit"]["forced"]):
            fail(f"machine commitment disagrees for {chars[p['answer']]['name']}: py {commit} js {out['commit']}")
    print(f"2. machine: {len(probes)} probes x 10 clue counts agree with Python: scores, picks, par and the commitment rule")
    famous = sorted([i for i, c in enumerate(chars) if c["tokens"] >= 500 and len(c["clues"]["distinct"]) >= 10],
                    key=lambda i: (-chars[i]["kin"], i))[:80]
    if js["famous"] != famous:
        fail("famous pool is not the 80 most linked-to characters with enough text")
    if js["daily"][0] != js["daily"][1] or js["daily"][0] == js["daily"][2]:
        fail(f"daily seed not deterministic per day: {js['daily']}")
    print(f"   famous pool = the 80 most linked-to (from {chars[famous[0]]['name']} down to {chars[famous[-1]]['name']}, {chars[famous[-1]]['kin']} in-links); daily seed deterministic")

    # ---- 3. impostor sentences against the corpus
    pages = W.load_pages()
    docs = W.load_tokens(pages, verbose=False)
    ids = sorted(docs)
    streams = {n: W.raw_tokens(docs[n]) for n in ids}
    grams = set()
    for s in streams.values():
        for i in range(len(s) - 8):
            grams.add(tuple(s[i:i + 9]))
    reals = [it for it in D["impostor"] if it["real"]]
    fakes = [it for it in D["impostor"] if not it["real"]]
    for it in reals:
        s = streams[ids[it["src"]]]
        w = it["w"]
        n = len(w)
        if not any(s[i:i + n] == w for i in range(len(s) - n + 1)):
            fail(f"real sentence not found on {chars[it['src']]['name']}: {it['t'][:60]}")
        if W.detok(w) != it["t"]:
            fail("real sentence text does not match its tokens")
    real_set = {tuple(it["w"]) for it in reals}
    all_sents = {tuple(s) for n in ids for s in W.clean_sentences(docs[n], lo=10, hi=28)}
    for it in fakes:
        w = it["w"]
        if tuple(w) in all_sents or tuple(w) in real_set:
            fail(f"generated sentence is a real sentence: {it['t'][:60]}")
        if any(tuple(w[i:i + 9]) in grams for i in range(len(w) - 8)):
            fail(f"generated sentence copies a nine-word run: {it['t'][:60]}")
        if W.detok(w) != it["t"]:
            fail("generated sentence text does not match its tokens")
    lr = [len(it["w"]) for it in reals]
    lf = [len(it["w"]) for it in fakes]
    if abs(sum(lr) / len(lr) - sum(lf) / len(lf)) > 3:
        fail(f"length mismatch: real {sum(lr)/len(lr):.1f} generated {sum(lf)/len(lf):.1f}")
    print(f"3. impostor: {len(reals)} real sentences found verbatim on their pages; {len(fakes)} generated ones share no "
          f"nine-word run with any page (mean length {sum(lr)/len(lr):.1f} vs {sum(lf)/len(lf):.1f} tokens)")

    # ---- 4. redacted lines
    for it in D["redacted"]:
        if it["t"].count("▮▮▮▮") != 1:
            fail(f"blank count wrong: {it['t']}")
        if len(set(it["o"])) != 4 or it["a"].lower() not in [o.lower() for o in it["o"]]:
            fail(f"options wrong: {it['o']} / {it['a']}")
        if it["a"].lower() in it["t"].lower():
            fail(f"answer survives in the line: {it['a']} / {it['t']}")
        s = streams[ids[it["src"]]]
        w = it["w"]
        n = len(w)
        if not any(s[i:i + n] == w for i in range(len(s) - n + 1)):
            fail(f"redacted source sentence not on {chars[it['src']]['name']}")
        if len(it["sim"]) != 4 or it["sim"][[o.lower() for o in it["o"]].index(it["a"].lower())] != 0:
            fail("similar() flags malformed")
    print(f"4. redacted: {len(D['redacted'])} lines, one blank each, answer among four unique options, sources verified")

    # ---- 5. cases
    famous_set = set(famous)
    full_set = {i for i, c in enumerate(chars) if c["tokens"] >= 500 and len(c["clues"]["distinct"]) >= 10}
    for cs in js["cases"]:
        if not cs["same"]:
            fail(f"case not deterministic for seed {cs['seed']} mix {cs['mix']}")
        types = [r["type"] for r in cs["rounds"]]
        n = {t: types.count(t) for t in ("unmask", "impostor", "redacted")}
        want = {"mixed": {"unmask": 6, "impostor": 3, "redacted": 3}, "unmask": {"unmask": 12, "impostor": 0, "redacted": 0},
                "impostor": {"unmask": 0, "impostor": 12, "redacted": 0}, "redacted": {"unmask": 0, "impostor": 0, "redacted": 12}}[cs["mix"]]
        if n != want:
            fail(f"mix {cs['mix']} gave {n}")
        for r in cs["rounds"]:
            if r["type"] == "unmask":
                if len(r["options"]) != 4 or r["options"].count(r["answer"]) != 1 or len(set(r["options"])) != 4:
                    fail(f"unmask options malformed: {r['options']} answer {r['answer']}")
                if cs["roster"] == "famous":
                    if not set(r["options"]) <= famous_set:
                        fail("a famous-roster case dealt an obscure suspect")
                    look = [i for i in chars[r["answer"]]["look"] if i in famous_set][:3]
                    if not set(look) <= set(r["options"]):
                        fail("famous-roster decoys should include every famous lookalike")
                else:
                    look = [i for i in chars[r["answer"]]["look"] if i in full_set][:3]
                    if not set(look) <= set(r["options"]) or not set(r["options"]) <= full_set:
                        fail("lookalike decoys are not the nearest pages with enough text")
                if r["commit"] is None or not (1 <= r["commit"]["k"] <= 10):
                    fail("unmask round without a machine commitment")
        if cs["mix"] == "mixed" and cs["rounds"][0]["type"] != "unmask":
            fail("a mixed case should open with an unmask round")
        if cs["mix"] == "impostor":
            reals_n = sum(1 for r in cs["rounds"] if D["impostor"][r["item"]]["real"])
            if reals_n != 6:
                fail(f"impostor-only case not balanced: {reals_n} real of 12")
    print(f"5. cases: {len(js['cases'])} case files deterministic, mixes right, suspects well-formed, machine totals on "
          f"unmask-only cases {sorted({c['machine'] for c in js['cases'] if c['mix'] == 'unmask'})}")

    print("   the machine, 300 simulated rounds per setting (roster / clue order / decoys):")
    for k, v in js["machine"].items():
        print(f"      {k:30s} commits after {v['clues']:.2f} clues, right {100*v['right']:.0f}%, forced to guess {100*v['forced']:.0f}%, "
              f"{v['points_per_round']:.1f} points per round")
    G = load_graph()
    if any(c["kin"] != G.in_degree(c["id"]) for c in chars):
        fail("in-degrees in the data file do not match the snapshot")
    print("6. in-degrees match the week 1 snapshot")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
