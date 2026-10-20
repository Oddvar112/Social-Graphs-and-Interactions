"""
Checks the data and the rules IN OTHER WORDS runs on, by recomputing the data and running the shipped engine.

What this proves:
  * every KEYWORD top ten and rank in data/week7_show.js is the cosine between the clue's TF-IDF vector and the
    page's, recomputed here from the corpus with the week 6 definitions (to the rounding of the file)
  * every MEANING top ten and rank is the best-passage dot product, recomputed from the cached passage vectors;
    every stored passage really is the best passage of that page for that clue, and sits on that page
  * every lineup has six distinct portraits, contains the right page (unless the character is absent) and the pick
    of each engine, and never contains an absent character's page (there is none)
  * the overlap counts (clue words on the page, rare ones) match a recount
  * core.js under node: an episode has ten clues, four / five / one by style, no character twice, and is the same
    for the same seed and different for another; judge() gives you a point for the right page, or for NOBODY HERE
    on an absent clue, and never for the engines on an absent clue; tally() and shareText() add up

Run:  python scripts/test_in_other_words.py     (needs node; about 20 seconds after the embeddings are cached)
"""

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import week6_text as T                          # noqa: E402
import week7_text as X                          # noqa: E402

DATA_JS = ROOT / "data" / "week7_show.js"
CORE_JS = ROOT / "weeks" / "week7" / "show" / "core.js"
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("FAIL:", msg)
    return cond


def load_data():
    s = DATA_JS.read_text(encoding="utf-8")
    return json.loads(s[s.index("{"):s.rindex("}") + 1])


def run_js(source):
    if not shutil.which("node"):
        sys.exit("need node to run the shipped JavaScript")
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "harness.js"
        path.write_text(source + "\nconsole.log(__result());\n", encoding="utf-8")
        res = subprocess.run(["node", str(path)], capture_output=True, text=True, encoding="utf-8")
        if res.returncode != 0:
            sys.exit("node failed:\n" + res.stderr)
        return json.loads(res.stdout.strip().splitlines()[-1])


def main():
    D = load_data()
    clues = X.load_clues()
    check(len(clues) == len(D["clues"]) == D["meta"]["n_clues"], "clue count")
    m = T.build_model()
    d = X.Dense(m.c.pages, m.ids, D["meta"]["passage_words"])
    check(len(d.texts) == D["meta"]["n_passages"], "passage count")
    Q = X.encoder().encode([c["clue"] for c in clues])
    chars = D["chars"]
    check([c["id"] for c in chars] == m.ids, "chars are the 303 pages in order")

    for k, (c, s) in enumerate(zip(clues, D["clues"])):
        check(c["id"] == s["id"] and c["clue"] == s["clue"] and c["style"] == s["style"], f"{c['id']}: clue text and style")
        absent = c["style"] == "absent"
        t = -1 if absent else m.ix[c["target"]]
        check(s["target"] == t, f"{c['id']}: target index")
        kw, parts, unknown = X.keyword_scores(m, c["clue"])
        mean, best, arg = d.scores(Q[k])
        for key, sc in (("kw", kw), ("dn", best)):
            top = X.topk(sc, len(s[key]["top"]))
            check([p for p, _ in s[key]["top"]] == top, f"{c['id']}: {key} top ten order")
            check(all(abs(sc[p] - v) < 6e-5 for p, v in s[key]["top"]), f"{c['id']}: {key} top ten scores")
            if not absent:
                check(s["ranks"][key] == X.rank_of(sc, t), f"{c['id']}: {key} rank")
        check(sorted(s["kw"]["unknown"]) == unknown, f"{c['id']}: unknown words")
        for w, a, b, df in s["kw"]["words"]:
            check(w in parts and abs(parts[w][s["kw"]["top"][0][0]] - a) < 6e-5, f"{c['id']}: contribution of '{w}' to the pick")
            check(absent or abs(parts[w][t] - b) < 6e-5, f"{c['id']}: contribution of '{w}' to the right page")
            check(df == int(m.df[m.c.vi[w]]), f"{c['id']}: df of '{w}'")
        for p, pas in s["dn"]["passages"].items():
            p = int(p)
            check(pas["text"] == d.texts[arg[p]] and d.owner[arg[p]] == p, f"{c['id']}: best passage of page {p}")
            check(abs(pas["score"] - best[p]) < 6e-5, f"{c['id']}: passage score of page {p}")
        lu = s["lineup"]
        check(len(lu) == D["meta"]["lineup"] == len(set(lu)), f"{c['id']}: six distinct portraits")
        check(absent or t in lu, f"{c['id']}: the right page is in the lineup")
        check(s["kw"]["top"][0][0] in lu and s["dn"]["top"][0][0] in lu, f"{c['id']}: both picks are in the lineup")
        picks = {s["kw"]["top"][0][0], s["dn"]["top"][0][0], t}
        check(all(chars[p]["img"] for p in lu if p not in picks), f"{c['id']}: every runner-up decoy has a portrait")
        plausible = set(X.topk(kw, 40)) | set(X.topk(best, 40)) | ({t} if not absent else set())
        check(sum(p in plausible for p in lu) == len(lu), f"{c['id']}: every decoy is in one engine's top forty")
        check(s["play"] == (c["id"] != "ironman-absent"), f"{c['id']}: played in the game unless set aside")
        if not absent:
            hits = X.distinctive_words(m, c["clue"], t)
            check(s["overlap"]["page"] == len(hits), f"{c['id']}: overlap count")
            check(s["overlap"]["rare"] == sum(m.df[m.c.vi[w]] <= 30 for w in hits), f"{c['id']}: rare overlap count")
    print(f"data: {len(D['clues'])} clues recomputed ({len(FAILS)} failures so far)")

    # ---------------------------------------------------------------- core.js under node
    core = CORE_JS.read_text(encoding="utf-8")
    data = DATA_JS.read_text(encoding="utf-8").replace("window.SHOW", "globalThis.SHOW")
    harness = data + "\n" + core + r"""
function __result() {
  const C = globalThis.ShowCore || module.exports;
  const D = globalThis.SHOW;
  const M = C.build(D);
  const out = {};
  const ep = C.episode(M, 12345), ep2 = C.episode(M, 12345), ep3 = C.episode(M, 54321);
  out.ep = ep; out.same = JSON.stringify(ep) === JSON.stringify(ep2); out.different = JSON.stringify(ep) !== JSON.stringify(ep3);
  out.styles = {}; out.keys = []; out.unplayed = 0;
  ep.forEach(i => { const c = D.clues[i]; out.styles[c.style] = (out.styles[c.style] || 0) + 1; out.keys.push(c.style === 'absent' ? 'a:' + c.name : 'p:' + c.target); });
  for (let s = 1; s <= 400; s++) C.episode(M, s).forEach(i => { if (D.clues[i].play === false) out.unplayed += 1; });
  out.verdicts = D.clues.map(c => ({
    right: C.judge(c, c.style === 'absent' ? -1 : c.target), wrongNobody: C.judge(c, -1), none: C.judge(c, null),
    decoy: C.judge(c, c.lineup.find(p => p !== c.target)),
  }));
  const vs = ep.map(i => C.judge(D.clues[i], D.clues[i].target === -1 ? -1 : D.clues[i].target));
  out.tally = C.tally(vs); out.share = C.shareText(3, vs);
  out.date = C.episodeOfDate(new Date(2026, 9, 20)); out.date2 = C.episodeOfDate(new Date(2026, 9, 27));
  return JSON.stringify(out);
}
"""
    r = run_js(harness)
    check(len(r["ep"]) == 10 and len(set(r["ep"])) == 10, "an episode is ten distinct clues")
    check(r["styles"] == {"page": 4, "other": 5, "absent": 1}, f"episode mix: {r['styles']}")
    check(len(set(r["keys"])) == 10, "no character twice in an episode")
    check(r["same"] and r["different"], "episodes are fixed by their seed")
    check(r["unplayed"] == 0, "a clue set aside never turns up in 400 episodes")
    for c, v in zip(D["clues"], r["verdicts"]):
        absent = c["style"] == "absent"
        check(v["right"]["you"], f"{c['id']}: the right answer scores")
        check(v["none"]["you"] is False, f"{c['id']}: no answer scores nothing")
        check(v["wrongNobody"]["you"] == absent, f"{c['id']}: NOBODY HERE scores only on an absent clue")
        check(not v["decoy"]["you"] or absent is False and c["lineup"][0] == c["target"], f"{c['id']}: a decoy scores nothing")
        check(v["right"]["kw"] == (not absent and c["kw"]["top"][0][0] == c["target"]), f"{c['id']}: keyword verdict")
        check(v["right"]["dn"] == (not absent and c["dn"]["top"][0][0] == c["target"]), f"{c['id']}: meaning verdict")
        check(not absent or (v["right"]["kw"] is False and v["right"]["dn"] is False), f"{c['id']}: engines never score on an absent clue")
    check(r["tally"]["you"] == 10 and r["tally"]["rounds"] == 10, "tally adds up")
    check(r["share"].startswith("IN OTHER WORDS · episode 3") and r["share"].count("\n") == 3, "share text has four lines")
    check(r["date"]["number"] == 1 and r["date2"]["number"] == 8, "episode numbers count from 20 October 2026")
    print("core.js: episode, judge, tally, shareText checked")

    print("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILURES")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
