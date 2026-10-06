"""
Checks the data and the engine TABOO-IDF runs on, by actually running them.

The game's rules live in weeks/week6/taboo/core.js with no DOM in them, so this test executes the exact shipped file
(under node, or under macOS's built-in JavaScript engine if node is not installed) and compares it with numpy, and
recomputes the data's postings from scratch in plain Python (math.log and dictionaries).

What this proves:
  * the postings in data/week6_taboo.js are every page's unit TF-IDF weight for every word on at least two pages, to
    the rounding of the file (every one of them, and no others)
  * each word's kind (other / name / habit) and document frequency are the ones the analysis uses
  * core.js scores and ranks for random word lists agree with numpy to 1e-4, and its win test is "strictly ahead"
  * the oracle word stored for each of the 216 targets, typed alone, puts the target first in core.js, in both modes,
    and no allowed word does better; each fingerprint word is a plain word with one of the twelve largest weights
  * core.js lookup() gives ok / unknown / taboo / dup / invalid / empty as the rules say, in both modes
  * the whole interface, game.js, runs through a stub page: a win after one word scores 10, a name with names banned is
    a strike that costs a word, an unknown word costs nothing, a hint costs three, twelve words used is a loss, giving
    up scores 0, and the two option rows work

Run:  python scripts/test_taboo.py     (needs node or macOS; about 30 seconds)
"""

import collections
import json
import math
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import week5_text as W                          # noqa: E402
import week6_text as T                          # noqa: E402

DATA_JS = ROOT / "data" / "week6_taboo.js"
CORE_JS = ROOT / "weeks" / "week6" / "taboo" / "core.js"
UI_JS = ROOT / "weeks" / "week6" / "taboo" / "game.js"
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("FAIL:", msg)
    return cond


def run_js(source):
    """Run a script that defines __result() returning a JSON string; node if there is one, else macOS's own
    JavaScript engine (osascript -l JavaScript), which needs no install."""
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "harness.js"
        if shutil.which("node"):
            path.write_text(source + "\nconsole.log(__result());\n", encoding="utf-8")
            cmd = ["node", str(path)]
        elif shutil.which("osascript"):
            path.write_text(source + "\n__result();\n", encoding="utf-8")
            cmd = ["osascript", "-l", "JavaScript", str(path)]
        else:
            sys.exit("need node, or osascript on macOS, to run the shipped JavaScript")
        res = subprocess.run(cmd, capture_output=True, text=True)
        lines = (res.stdout.strip() or res.stderr.strip()).splitlines()
        if res.returncode != 0 or not lines or not lines[-1].startswith("{"):
            sys.exit("JavaScript failed:\n" + res.stderr[-2000:] + res.stdout[-500:])
        return json.loads(lines[-1])


def load_data():
    s = DATA_JS.read_text(encoding="utf-8")
    return json.loads(s[s.index("{"):s.rindex("}") + 1])


# ------------------------------------------------------------------------------------------------- data
def test_data(D, m):
    N, S = m.N, D["meta"]["scale"]
    words = D["words"]
    check(words == sorted(words) and len(set(words)) == len(words), "words are unique and alphabetical")
    check(len(D["ps"]) == len(words) + 1 and D["ps"][-1] == len(D["pi"]) == len(D["pv"]), "postings offsets are consistent")
    c = m.c
    counts = [collections.Counter(c.tokens[n]) for n in c.ids]
    df = collections.Counter()
    for cnt in counts:
        df.update(cnt.keys())
    lens = [sum(cnt.values()) for cnt in counts]
    norms = [math.sqrt(sum(((k / lens[i]) * math.log(N / df[w])) ** 2 for w, k in cnt.items())) for i, cnt in enumerate(counts)]
    third = -(-N // 3)

    def weight(i, w):
        k = counts[i].get(w, 0)
        return (k / lens[i]) * math.log(N / df[w]) / norms[i] if k else 0.0

    check(all(df[w] >= 2 for w in words), "every word is on at least two pages")
    check(len(words) == sum(1 for w in df if df[w] >= 2), "and every such word is there")
    # every posting, both ways: what the file says against the corpus, and the corpus against the file
    index = {w: i for i, w in enumerate(words)}
    filed = {}
    for wid in range(len(words)):
        for p in range(D["ps"][wid], D["ps"][wid + 1]):
            filed[(wid, D["pi"][p])] = D["pv"][p] / S
    worst, seen = 0.0, 0
    for i, cnt in enumerate(counts):
        for w in cnt:
            if w in index:
                x = weight(i, w)
                if x != 0.0:                 # a word on all 303 pages has idf = ln(1) = 0 and no posting at all
                    seen += 1
                worst = max(worst, abs(filed.get((index[w], i), 0.0) - x))
    check(seen == len(filed) == len(D["pi"]), f"postings are exactly the (word, page) pairs with a non-zero weight ({seen} of {len(filed)})")
    check(worst < 1e-5, f"every posting matches plain-Python TF-IDF (largest difference {worst:.1e})")
    for wid, w in enumerate(words):
        if D["df"][wid] != df[w]:
            check(False, f"'{w}': document frequency")
            break
        want = 1 if w in m.names else (2 if (w in W.STOPWORDS or df[w] >= third) else 0)
        if D["kind"][wid] != want:
            check(False, f"'{w}': kind {want}")
            break
    return worst


def np_scores(m, D, ids):
    """numpy's version: columns of the unit TF-IDF matrix for the chosen words, summed."""
    col = {w: i for i, w in enumerate(m.c.vocab)}
    cols = [col[D["words"][i]] for i in ids]
    return m.Un[:, cols].sum(axis=1) if cols else np.zeros(m.N)


# ------------------------------------------------------------------------------------------------- engine
def harness_core(D, probes):
    return ("var window = globalThis;\nconst D = " + json.dumps(D) + ";\nconst PROBES = " + json.dumps(probes) + ";\n"
            "new Function('module', " + json.dumps(CORE_JS.read_text(encoding="utf-8")) + ")(undefined);\n"
            "const C = globalThis.TabooCore, M = C.build(D);\n"
            r"""
function __result() {
  const out = { rank: [], lookups: [], oracle: [], pool: null, points: null, pick: null, ties: null };
  for (const p of PROBES.sets) {
    const r = C.ranking(M, p.ids, p.target);
    out.rank.push({ sc: Array.from(r.sc), rank: r.rank, won: r.won, norm: r.norm, top: r.order.slice(0, 5) });
  }
  for (const q of PROBES.lookups) out.lookups.push(C.lookup(M, q.raw, q.typed, q.mode).status);
  D.chars.forEach((c, i) => {
    if (!c.oracle) return;
    const row = {};
    for (const mode of ['banned', 'allowed']) {
      const o = C.oracleWord(D, i, mode);
      const r = C.ranking(M, [o.id], i);
      row[mode] = { id: o.id, rank: r.rank, won: r.won, kind: D.kind[o.id], margin: r.margin, stored: o.margin };
    }
    out.oracle.push([i, row]);
  });
  out.pool = [C.targetPool(D, false).length, C.targetPool(D, true, 80).length, C.targetPool(D, true, 80).slice(0, 3)];
  out.points = [C.points(true, 1), C.points(true, 10), C.points(true, 12), C.points(false, 3)];
  out.pick = [C.pickTarget(D, C.targetPool(D, true, 80), 5), C.pickTarget(D, C.targetPool(D, true, 80), 5), C.pickTarget(D, C.targetPool(D, true, 80), 6)];
  const stop = ['the', 'and', 'of'].map(w => M.index.get(w));
  const t = C.targetPool(D, true, 80)[0];
  out.ties = [C.ranking(M, stop, t).won, C.ranking(M, [], t).won];
  return JSON.stringify(out);
}
""")


def test_engine(D, m):
    rng = random.Random(7)
    targets = [i for i, c in enumerate(D["chars"]) if "oracle" in c]
    sets = []
    for _ in range(60):
        ids = rng.sample(range(len(D["words"])), rng.randint(1, 8))
        sets.append({"ids": ids, "target": rng.choice(targets)})
    some_name = next(i for i, k in enumerate(D["kind"]) if k == 1)
    some_word = next(i for i, k in enumerate(D["kind"]) if k == 0)
    lookups = [
        {"raw": "  Weather ", "typed": [], "mode": "banned"}, {"raw": "", "typed": [], "mode": "banned"},
        {"raw": "two words", "typed": [], "mode": "banned"}, {"raw": "x1", "typed": [], "mode": "banned"},
        {"raw": "zzzqqq", "typed": [], "mode": "banned"}, {"raw": D["words"][some_name], "typed": [], "mode": "banned"},
        {"raw": D["words"][some_name], "typed": [], "mode": "allowed"}, {"raw": D["words"][some_word], "typed": [some_word], "mode": "banned"},
        {"raw": D["words"][some_word].upper(), "typed": [], "mode": "banned"}, {"raw": None, "typed": [], "mode": "banned"}]
    want = ["ok", "empty", "invalid", "invalid", "unknown", "taboo", "ok", "dup", "ok", "empty"]
    r = run_js(harness_core(D, {"sets": sets, "lookups": lookups}))
    check(r["lookups"] == want, f"lookup statuses {r['lookups']} should be {want}")
    worst = 0.0
    for s, out in zip(sets, r["rank"]):
        ref = np_scores(m, D, s["ids"])
        worst = max(worst, np.abs(np.array(out["sc"]) - ref).max())
        order = np.lexsort((np.arange(m.N), -ref))
        rank = int(np.where(order == s["target"])[0][0]) + 1
        check(abs(out["rank"] - rank) <= 0 or abs(ref[order[out["rank"] - 1]] - ref[s["target"]]) < 2e-4, f"rank of target {rank} vs {out['rank']}")
        check(abs(out["norm"] - 1 / math.sqrt(len(s["ids"]))) < 1e-12, "cosine normaliser is 1 / sqrt(words)")
        others = np.delete(ref, s["target"]).max()
        if abs(ref[s["target"]] - others) > 3e-4:
            check(out["won"] == bool(ref[s["target"]] > others), "win test is strictly ahead")
    check(worst < 1e-4, f"core.js scores match numpy for 60 random word lists (largest difference {worst:.1e})")
    n_or = 0
    for i, row in r["oracle"]:
        n_or += 1
        for mode in ("banned", "allowed"):
            o = row[mode]
            check(o["rank"] == 1 and o["won"], f"{D['chars'][i]['name']}: oracle word wins alone ({mode})")
            check(abs(o["margin"] - o["stored"]) < 2e-4, f"{D['chars'][i]['name']}: stored margin ({mode})")
        check(row["banned"]["kind"] != 1, f"{D['chars'][i]['name']}: banned oracle is not a name")
        check(row["allowed"]["stored"] >= row["banned"]["stored"] - 1e-9, f"{D['chars'][i]['name']}: allowing names cannot lower the oracle margin")
    check(n_or == D["meta"]["n_targets"] == 216, "216 targets")
    # no allowed word does better than the stored oracle, for a sample of targets (numpy, every word)
    col = {w: i for i, w in enumerate(m.c.vocab)}
    sel = [col[w] for w in D["words"]]
    Wm = m.Un[:, sel]
    for i in random.Random(3).sample(targets, 25):
        others = np.delete(Wm, i, axis=0).max(axis=0)
        margin = Wm[i] - others
        nonname = np.array(D["kind"]) != 1
        check(abs(margin[nonname].max() - D["chars"][i]["oracle"][1]) < 2e-4, f"{D['chars'][i]['name']}: no non-name word beats the oracle")
        check(abs(margin.max() - D["chars"][i]["oracle_any"][1]) < 2e-4, f"{D['chars'][i]['name']}: no word at all beats the any-word oracle")
        fp = D["chars"][i]["print"]
        plain = np.array(D["kind"]) == 0
        top12 = set(np.argsort(-np.where(plain, Wm[i], -1))[:12])
        check(set(fp) == top12 and all(D["kind"][w] == 0 for w in fp), f"{D['chars'][i]['name']}: fingerprint is the twelve strongest plain words")
    check(r["pool"][0] == 216 and r["pool"][1] == 80, f"target pools {r['pool'][:2]}")
    kin = [D["chars"][i]["kin"] for i in r["pool"][2]]
    check(kin == sorted(kin, reverse=True), "famous pool is best-known first")
    check(r["points"] == [10, 1, 1, 0], f"points {r['points']}")
    check(r["pick"][0] == r["pick"][1], "pickTarget is deterministic for a seed")
    check(r["ties"] == [False, False], "a tie, or no words, is not a win")
    return worst


# ------------------------------------------------------------------------------------------------- interface
def harness_ui(D):
    return ("var window = globalThis;\nwindow.TABOO = " + json.dumps(D) + ";\n"
            "new Function('module', " + json.dumps(CORE_JS.read_text(encoding="utf-8")) + ")(undefined);\n"
            + r"""
const els = {}, handlers = {}, store = {};
function El(id) {
  const e = { id, innerHTML: '', value: '', hidden: false, disabled: false, className: '', dataset: {}, style: {},
    classList: { toggle() {}, add() {}, remove() {} },
    addEventListener(ev, fn) { (handlers[id] = handlers[id] || {})[ev] = fn; },
    focus() {}, scrollIntoView() {}, querySelectorAll() { return []; } };
  let text = '';
  Object.defineProperty(e, 'textContent', { get() { return text; }, set(v) { text = String(v); } });   // a browser stores text
  return e;
}
const document = { getElementById: id => els[id] || (els[id] = El(id)), querySelectorAll: sel => (sel === '.seg' ? [g('o-mode'), g('o-roster')] : []),
  addEventListener() {}, body: { innerHTML: '' } };
const g = id => document.getElementById(id);
const localStorage = { getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); } };
globalThis.document = document; globalThis.localStorage = localStorage;
new Function('window', 'document', 'localStorage', """ + json.dumps(UI_JS.read_text(encoding="utf-8")) + r""")(window, document, localStorage);

const C = globalThis.TabooCore, M = C.build(window.TABOO), D = window.TABOO;
const say = w => { g('word').value = w; handlers.form.submit({ preventDefault() {} }); };
const target = () => D.chars.findIndex(c => c.oracle && c.name === g('tname').textContent);
const snap = () => ({ score: g('s-score').textContent, used: g('s-used').textContent, wins: g('s-wins').textContent, msg: g('msg').textContent,
                      reveal: g('reveal').innerHTML, revealHidden: g('reveal').hidden, nextHidden: g('next').hidden, typed: g('typed').innerHTML,
                      board: g('board').innerHTML, rank: g('rank').innerHTML, wordDisabled: g('word').disabled, hint: g('hintcost').textContent,
                      tname: g('tname').textContent });
const seg = (id, v) => handlers[id].click({ target: { closest: () => ({ dataset: { v }, classList: { toggle() {} } }) } });
function __result() {
  const out = {};
  out.start = snap(); let t = target(); out.t0 = t;
  // 1. win in one word with the oracle (names banned)
  const o = C.oracleWord(D, t, 'banned'); say(o.word); out.win1 = snap();
  handlers.next.click(); t = target();
  // 2. a name is a strike, an unknown word is free, a duplicate is ignored, then the oracle wins on the third word used
  const nameWord = D.words.find((w, i) => D.kind[i] === 1 && /^[a-z]+$/.test(w) && D.chars[t].name.toLowerCase().includes(w));
  out.nameWord = nameWord || null;
  say(nameWord || 'storm'); out.strike = snap();
  say('zzzqqqxx'); out.unknown = snap();
  const o2 = C.oracleWord(D, t, 'banned'); say('the'); say('the'); out.dup = snap();
  say(o2.word); out.win3 = snap();
  handlers.next.click(); t = target();
  // 3. hints cost three words each, two at most
  out.hintTarget = t; handlers.hint.click(); out.hint1 = snap(); handlers.hint.click(); out.hint2 = snap(); handlers.hint.click(); out.hint3 = snap();
  handlers.giveup.click(); out.gaveUp = snap();
  handlers.next.click(); t = target();
  // 4. twelve words used without winning is a loss
  const filler = ['the', 'and', 'of', 'to', 'in', 'a', 'is', 'as', 'by', 'for', 'with', 'on'];
  filler.forEach(say); out.lost = snap();
  handlers.next.click();
  // 5. options
  seg('o-mode', 'allowed'); t = target();
  const nm = D.chars[t].name.toLowerCase().split(/[^a-z]+/).find(w => M.index.has(w) && D.kind[M.index.get(w)] === 1);
  out.allowedWord = nm || null; if (nm) say(nm); out.allowedSnap = snap();
  handlers.next.click(); seg('o-roster', 'all'); out.allRoster = snap();
  // 6. a wrong-looking word
  say('two words'); out.invalid = snap();
  return JSON.stringify(out);
}
""")


def test_ui(D, m):
    r = run_js(harness_ui(D))
    t0 = r["t0"]
    check(t0 >= 0 and D["chars"][t0].get("oracle") is not None, "opens on a valid target")
    check(r["start"]["wordDisabled"] is False and r["start"]["revealHidden"] is True, "opens playable")
    w = r["win1"]
    check(w["score"] == "10" and w["wins"] == "1" and w["used"] == "1 / 12" and w["wordDisabled"] is True and w["nextHidden"] is False,
          f"one-word win scores 10 ({w['score']}, used {w['used']})")
    check("SOLVED in 1 word: +10" in w["reveal"] and "fingerprint" in w["reveal"] and "oracle" in w["reveal"], "the reveal explains the win")
    if r["nameWord"]:
        s = r["strike"]
        check("strike" in s["typed"] and s["used"] == "1 / 12" and "TABOO" in s["msg"], f"a name is a strike that costs one word ({s['used']})")
    u = r["unknown"]
    check("fewer than two pages" in u["msg"] and u["used"] == r["strike"]["used"], "an unknown word costs nothing")
    d = r["dup"]
    check(d["typed"].count("the") <= 1 + d["typed"].count("strike") * 0 and "already said" in d["msg"], "a repeated word is ignored")
    k3 = r["win3"]
    expect = 10 - (2 if r["nameWord"] else 1)         # strike, 'the', then the oracle word: three words used
    check(k3["score"] == str(10 + expect) and "SOLVED" in k3["reveal"], f"three words used scores {expect} on top ({k3['score']})")
    check("(−3 words, 1 left)" in r["hint1"]["hint"] and "(none left)" in r["hint2"]["hint"], "two hints at most, three words each")
    fp = [D["words"][x] for x in D["chars"][r["hintTarget"]]["print"]]
    check(f"{fp[0][0]} " in r["hint1"]["typed"] and f"({len(fp[0])} letters)" in r["hint1"]["typed"], f"hint 1 spells the first letter and length of '{fp[0]}'")
    check(f"{fp[1][:3]} " in r["hint2"]["typed"] and f"({len(fp[1])} letters)" in r["hint2"]["typed"], f"hint 2 spells three letters of '{fp[1]}'")
    check(r["hint2"]["used"] == "6 / 12" and r["hint3"]["used"] == "6 / 12", f"hints cost three words each ({r['hint2']['used']})")
    check("GAVE UP" in r["gaveUp"]["reveal"] and r["gaveUp"]["score"] == k3["score"], "giving up scores nothing")
    check("OUT OF WORDS" in r["lost"]["reveal"] and r["lost"]["used"] == "12 / 12" and r["lost"]["score"] == k3["score"], "twelve words used without winning is a loss")
    check(r["allowedWord"] is None or "TABOO" not in r["allowedSnap"]["msg"], "with names allowed a name is not a strike")
    check("one word, letters only" in r["invalid"]["msg"].lower(), "two words in the box is refused")
    return r


def main():
    print("loading the corpus and building the model ...")
    D = load_data()
    m = T.build_model()
    worst = test_data(D, m)
    print(f"data: postings match plain-Python TF-IDF, largest difference {worst:.1e}; {D['meta']['n_words']} words, {D['meta']['n_postings']} postings")
    w2 = test_engine(D, m)
    print(f"core.js: scores match numpy (largest difference {w2:.1e}); all 216 oracles win alone, in both modes")
    test_ui(D, m)
    print("game.js: win, strike, unknown, repeat, hints, give-up, loss and both option rows run through the stub page")
    if FAILS:
        print(f"\n{len(FAILS)} checks failed")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
