"""
Checks the data and the engine THE NAME DIAL runs on, by actually running them.

The dial's arithmetic lives in weeks/week6/dial/core.js with no DOM in it, so this test executes the exact shipped
file (under node, or under macOS's built-in JavaScript engine if node is not installed) and compares it with the
Python pipeline, and recomputes the data's three matrices from scratch in plain Python (math.log and dictionaries).

What this proves:
  * the three matrices in data/week6_dial.js are, for every diagonal entry and a random sample of 600 pairs, the dot
    products of unit-length TF-IDF vectors split into names, habit words and other words, to the rounding of the file;
    each diagonal sums to 1; the matrices are symmetric; the edge list is the week 1 network, undirected
  * core.js at the dials (1, 1), (0, 1) and (0, 0) gives the same cosine for every pair as TF-IDF with names kept,
    with names removed, and with names and habit words removed in numpy (largest difference reported)
  * the agreement of the ten nearest neighbours at those dials is the number the post quotes (4.02, 1.99, 2.28)
  * the three parts of every neighbour's cosine add up to it at any dial setting, and a dial at zero zeroes its part
  * the nearest neighbours of six pages, at three dial settings, are the same pages with the same cosines as in numpy
  * the whole interface, dial.js, runs through a stub page: presets, sliders, clicking a neighbour, and the numbers it
    prints are the engine's numbers

Run:  python scripts/test_name_dial.py     (needs node or macOS; about 10 seconds)
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

DATA_JS = ROOT / "data" / "week6_dial.js"
CORE_JS = ROOT / "weeks" / "week6" / "dial" / "core.js"
UI_JS = ROOT / "weeks" / "week6" / "dial" / "dial.js"
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
    check(len(D["chars"]) == N == 303, "303 pages")
    check([c["id"] for c in D["chars"]] == m.ids, "pages are in the corpus order")
    mats = {k: np.array(D[k], dtype=float).reshape(N, N) / S for k in "nho"}
    for k, M in mats.items():
        check(np.abs(M - M.T).max() < 1.5 / S, f"matrix {k} is symmetric")
    diag = sum(M.diagonal() for M in mats.values())
    check(np.abs(diag - 1).max() < 3 / S, f"diagonals sum to 1 (off by at most {np.abs(diag - 1).max():.1e})")

    # from scratch: dictionaries and math.log only
    c = m.c
    counts = [collections.Counter(c.tokens[n]) for n in c.ids]
    df = collections.Counter()
    for cnt in counts:
        df.update(cnt.keys())
    third = -(-N // 3)
    lens = [sum(cnt.values()) for cnt in counts]
    vec = []
    for i, cnt in enumerate(counts):
        v = {w: (k / lens[i]) * math.log(N / df[w]) for w, k in cnt.items()}
        nrm = math.sqrt(sum(x * x for x in v.values()))
        vec.append({w: x / nrm for w, x in v.items()})

    def kind(w):
        return "n" if w in m.names else ("h" if (w in W.STOPWORDS or df[w] >= third) else "o")

    rng = random.Random(6)
    pairs = [(i, i) for i in range(N)] + [tuple(sorted(rng.sample(range(N), 2))) for _ in range(600)]
    worst = 0.0
    for i, j in pairs:
        parts = {"n": 0.0, "h": 0.0, "o": 0.0}
        for w in vec[i].keys() & vec[j].keys():
            parts[kind(w)] += vec[i][w] * vec[j][w]
        for k in "nho":
            worst = max(worst, abs(parts[k] - mats[k][i, j]))
    check(worst < 1.5 / S, f"matrices match plain-Python TF-IDF (largest difference {worst:.1e})")

    edges = {(min(m.ix[a], m.ix[b]), max(m.ix[a], m.ix[b])) for a, b in m.U.edges() if a != b}
    check({tuple(e) for e in D["edges"]} == edges and len(edges) == D["meta"]["n_edges"], f"edge list is the undirected network ({len(edges)} links)")
    return mats, worst


# ------------------------------------------------------------------------------------------------- engine
def numpy_cos(m, wn, wh):
    """The same dials in numpy: scale the name and habit columns of the unit TF-IDF vectors, renormalise, cosine."""
    scale = np.where(m.isname, wn, np.where(m.ishabit, wh, 1.0))
    return T.cosine_matrix(m.Un * scale)


def harness_core(D, probes):
    return ("var window = globalThis;\nconst D = " + json.dumps(D) + ";\nconst PROBES = " + json.dumps(probes) + ";\n"
            "new Function('module', " + json.dumps(CORE_JS.read_text(encoding="utf-8")) + ")(undefined);\n"
            "const C = globalThis.DialCore, M = C.build(D);\n"
            r"""
function __result() {
  const out = { rows: [], agree: [], lists: [], sums: [], zero: [], curve: null, ranks: null };
  for (const [wn, wh] of [[1, 1], [0, 1], [0, 0]]) {
    out.rows.push(Array.from(C.cosRow(M, 7, wn, wh)));
    out.agree.push(C.agreement(M, wn, wh));
  }
  for (const p of PROBES) {
    const nb = C.neighbours(M, p.page, p.wn, p.wh, 10, null);
    out.lists.push({ page: p.page, wn: p.wn, wh: p.wh, j: nb.map(r => r.j), cos: nb.map(r => r.cos), linked: nb.map(r => r.linked) });
    out.sums.push(nb.map(r => Math.abs(r.n + r.h + r.o - r.cos)));
    out.zero.push({ wn: p.wn, wh: p.wh, n: nb.map(r => r.n), h: nb.map(r => r.h) });
  }
  out.curve = C.curve(M, 1, 10);
  const br = C.baseRanks(M, 7); out.ranks = [br[C.topK(C.cosRow(M, 7, 1, 1), 1)[0]], Math.max(...br), Math.min(...Array.from(br).filter((x, i) => i !== 7))];
  return JSON.stringify(out);
}
""")


def test_engine(D, m):
    ix = m.ix
    pages = [ix[n] for n in ("Storm_(Marvel_Comics)", "Wolverine_(character)", "Venom_(character)", "Doctor_Strange",
                             "Black_Widow_(Natasha_Romanova)", "Eddie_Brock")]
    settings = [(1.0, 1.0), (0.0, 1.0), (0.3, 0.6)]
    probes = [{"page": p, "wn": a, "wh": b} for p in pages for a, b in settings]
    r = run_js(harness_core(D, probes))
    worst = 0.0
    for (a, b), row in zip([(1, 1), (0, 1), (0, 0)], r["rows"]):
        S = numpy_cos(m, a, b)[7]
        row = np.array(row)
        mask = np.arange(m.N) != 7
        worst = max(worst, np.abs(row[mask] - S[mask]).max())
        check(row[7] == -1, f"dials ({a},{b}): a page is not its own neighbour")
    check(worst < 3e-5, f"core.js cosine matches numpy at three dial settings (largest difference {worst:.1e})")
    # the dials are the representations the analysis calls TF-IDF, names out, names and habit words out
    check(np.abs(numpy_cos(m, 1, 1) - m.S_kept).max() < 1e-12, "dials (1, 1) are TF-IDF as the course defines it")
    check(np.abs(numpy_cos(m, 0, 1) - T.cosine_matrix(m.X * ~m.isname)).max() < 1e-12, "dials (0, 1) are TF-IDF with names removed")
    check(np.abs(numpy_cos(m, 0, 0) - m.S_clean).max() < 1e-12, "dials (0, 0) are names and habit words removed")
    stats = json.load(open(ROOT / "data" / "stats_week6.json"))["agreement"]
    for got, key in zip(r["agree"], ("TF-IDF", "TF-IDF, names out", "TF-IDF, names and habit words out")):
        check(abs(got - stats[key]) < 0.012, f"agreement {key}: dial engine {got:.3f}, analysis {stats[key]:.3f}")
    for L, sums, z in zip(r["lists"], r["sums"], r["zero"]):
        label = f"{D['chars'][L['page']]['name']} at names {L['wn']}, habit {L['wh']}"
        S = numpy_cos(m, L["wn"], L["wh"])[L["page"]]
        want = np.sort(S)[::-1][:10]
        check(np.abs(np.array(L["cos"]) - want).max() < 3e-5, f"{label}: the ten nearest cosines match numpy")
        check(np.abs(S[L["j"]] - np.array(L["cos"])).max() < 3e-5, f"{label}: each listed page has the cosine listed")
        check(max(sums) < 1e-9, f"{label}: the three parts add up to the cosine")
        check(L["linked"] == [bool(m.U.has_edge(m.ids[L["page"]], m.ids[j])) for j in L["j"]], f"{label}: linked flags")
        if L["wn"] == 0:
            check(max(abs(x) for x in z["n"]) < 1e-12, f"{label}: names dial at zero zeroes the names part")
    check(len(r["curve"]) == 11 and abs(r["curve"][-1]["a"] - r["agree"][0]) < 1e-9 and abs(r["curve"][0]["a"] - r["agree"][1]) < 1e-9,
          "the curve starts at names out and ends at TF-IDF")
    check(r["ranks"][0] == 1 and r["ranks"][1] == m.N - 1 and r["ranks"][2] == 1, "base ranks run from 1 to 302")
    return r, worst


# ------------------------------------------------------------------------------------------------- interface
def harness_ui(D):
    return ("var window = globalThis;\nwindow.DIAL = " + json.dumps(D) + ";\n"
            "new Function('module', " + json.dumps(CORE_JS.read_text(encoding="utf-8")) + ")(undefined);\n"
            + r"""
const els = {}, handlers = {}, calls = { hash: [], scroll: 0 };
function El(id) {
  return { id, innerHTML: '', textContent: '', className: '', value: '', hidden: false, dataset: {}, style: {}, classList: { toggle() {}, add() {}, remove() {} },
    addEventListener(ev, fn) { (handlers[id] = handlers[id] || {})[ev] = fn; } };
}
const document = { getElementById: id => els[id] || (els[id] = El(id)), querySelectorAll: () => [], body: { innerHTML: '', getBoundingClientRect: () => ({ height: 987.2 }) } };
const location = { hash: '' };
const history = { replaceState(a, b, url) { calls.hash.push(url); } };
const store = {};
const localStorage = { getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); } };
globalThis.document = document; globalThis.location = location; globalThis.history = history; globalThis.localStorage = localStorage;
const msgs = [];
const win = { addEventListener(ev, fn) { handlers.window = handlers.window || {}; handlers.window[ev] = fn; }, scrollTo() { calls.scroll += 1; },
  parent: { postMessage(m, o) { msgs.push([m, o]); } } };
const timers = [];
const fakeSetTimeout = fn => { timers.push(fn); return timers.length; };
const flush = () => { while (timers.length) timers.shift()(); };
new Function('window', 'document', 'location', 'history', 'setTimeout', 'clearTimeout', 'URLSearchParams', """ + json.dumps(UI_JS.read_text(encoding="utf-8")) + r""")(
  Object.assign(win, { DIAL: D => 0 }, { DIAL: window.DIAL, DialCore: globalThis.DialCore }), document, location, history, fakeSetTimeout, () => {}, globalThis.URLSearchParams || function () { throw new Error('none'); });

const g = id => document.getElementById(id);
const snap = () => ({ title: g('title').textContent, list: g('list').innerHTML, rall: g('r-all').textContent, rpage: g('r-page').textContent,
                      on: g('on').textContent, oh: g('oh').textContent, chart: g('chart').innerHTML, sub: g('sub').innerHTML, chance: g('r-chance').textContent,
                      chips: g('chips').innerHTML, people: g('people').innerHTML });
function __result() {
  flush();
  const out = { start: snap() };
  handlers.presets.click({ target: { closest: () => ({ dataset: { n: '0', h: '0' } }) } });
  flush(); out.cleanPreset = snap();
  handlers.presets.click({ target: { closest: () => ({ dataset: { n: '100', h: '100' } }) } });
  flush(); out.asIs = snap();
  g('wn').value = '40'; handlers.wn.input(); flush(); out.slider = snap();
  handlers.list.click({ target: { closest: () => ({ dataset: { j: '12' } }) } });
  flush(); out.moved = snap();
  handlers.chips.click({ target: { closest: () => ({ dataset: { j: '5' } }) } });
  flush(); out.chip = snap();
  g('pick').value = 'venom'; handlers.pick.change(); flush(); out.typed = snap();
  out.hashes = calls.hash; out.scrolls = calls.scroll;
  handlers.window.load(); out.msgs = msgs;
  return JSON.stringify(out);
}
""")


def test_ui(D, m, eng):
    r = run_js(harness_ui(D))
    s = r["start"]
    check(s["title"] == "STORM", f"opens on Storm ({s['title']!r})")
    check(s["list"].count('class="nbr"') == 12, "twelve neighbours listed")
    check(s["rpage"].endswith("/ 10"), "the page readout is out of ten")
    check(abs(float(s["rall"]) - eng["agree"][0]) < 0.006, f"opening agreement {s['rall']} is the engine's {eng['agree'][0]:.3f}")
    check(len(s["people"].split("<option")) - 1 == 303, "303 pages in the picker")
    check(s["chance"] == "0.31", f"chance level {s['chance']}")
    check(abs(float(r["cleanPreset"]["rall"]) - eng["agree"][2]) < 0.006, "preset 'names and habit words out' prints the engine's agreement")
    check(r["cleanPreset"]["on"] == "0%" and r["cleanPreset"]["oh"] == "0%", "the preset moves both dials")
    check(r["asIs"]["list"] == s["list"] and r["asIs"]["rall"] == s["rall"], "the as-is preset returns to the opening list exactly")
    check(r["slider"]["on"] == "40%" and r["slider"]["list"] != s["list"], "moving the names slider re-ranks the list")
    check(r["moved"]["title"] != "STORM" and r["moved"]["title"] == D["chars"][12]["name"].upper(), "clicking a neighbour moves to that page")
    check(r["chip"]["title"] == D["chars"][5]["name"].upper(), "clicking a chip moves to that page")
    check(r["typed"]["title"] == "VENOM", f"typing a name moves to that page ({r['typed']['title']!r})")
    check(all(h.startswith("#p=") and "&n=" in h and "&h=" in h for h in r["hashes"]) and len(r["hashes"]) > 5, "the state is kept in the address")
    check("polyline" in s["chart"] and "circle" in s["chart"], "the agreement chart is drawn")
    check(r["scrolls"] >= 1, "clicking a neighbour scrolls to the top")
    check(r["msgs"] and r["msgs"][-1][0] == {"type": "name-dial-height", "h": 990}, f"inside a frame the page reports its content height to the post ({r['msgs'][-1:] if r['msgs'] else None})")
    return r


def main():
    print("loading the corpus and building the model ...")
    D = load_data()
    m = T.build_model()
    mats, worst = test_data(D, m)
    print(f"data: matrices match plain-Python TF-IDF, largest difference {worst:.1e}")
    eng, w2 = test_engine(D, m)
    print(f"core.js: agreement at (1,1), (0,1), (0,0) = {', '.join(f'{x:.2f}' for x in eng['agree'])}; largest cosine difference from numpy {w2:.1e}")
    test_ui(D, m, eng)
    print("dial.js: presets, slider, neighbour click, chip, typed name and address all run through the stub page")
    if FAILS:
        print(f"\n{len(FAILS)} checks failed")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
