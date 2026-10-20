"""
Data for IN OTHER WORDS (weeks/week7/show): a game show in which a host reads a description of a Marvel character
and you race two search engines to point at the right portrait.

For every clue in data/week7_clues.tsv (written by hand before any of this ran) the two engines rank all 303 pages:

  KEYWORD   sparse retrieval: the cosine between the clue's TF-IDF vector and each page's, with the week 6
            definitions (tf = count / words, idf = ln(303 / df), unit length). A word the corpus has never seen is
            invisible to it.
  MEANING   dense retrieval: all-MiniLM-L6-v2 encodes the clue and every passage of every page (at most 180 words,
            split at sentence boundaries; 4,479 passages); a page's score is its best passage. The page-mean score
            and 60-word passages are computed too, for the post, and their ranks are stored.

Stored per clue: the top ten of each engine with scores, the rank of the right page under each variant, the
evidence (for KEYWORD, each clue word's contribution to its pick's score and to the right page's, and the words it
never read; for MEANING, the best-matching passage of its pick, of the right page and of KEYWORD's pick, in full,
so the text stays visible), and a lineup of six portraits: the right answer, each engine's pick, and then the next
pages down both engines' lists, so that every decoy is a page one of the machines found plausible, in a fixed
shuffled order. A clue about a character who is not in the corpus has no right page; the lineup is the two picks
plus their runners-up and the right answer is "nobody here".

One clue is kept out of the game (but not out of the post): see NOT_PLAYED.

Writes data/week7_show.js (window.SHOW). Deterministic given the frozen data and the cached embeddings.

Run:  python scripts/build_week7_show.py            (the first run encodes 19,000 passages: a minute on a GPU)
      python scripts/build_week7_show.py --check    only the vocabulary check of the clues (no model needed)
"""

import datetime
import json
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import week6_text as T                              # noqa: E402
import week7_text as X                              # noqa: E402
from analyse import pretty                          # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "week7_show.js"
TOP = 10
LINEUP = 6
RARE_DF = 30

# Clues that stay in the analysis but are not dealt in the game, and why. We noticed this one after the first ranking:
# the clue describes Iron Man, who has no page, but the corpus has the Ultimate Marvel version of the same Tony Stark,
# and the clue fits him too. In a lineup that shows his portrait, "nobody here" would be a trick, not a test.
NOT_PLAYED = {"ironman-absent": "the corpus has the Ultimate Marvel Iron Man, and the clue describes him too"}


def load_images():
    path = DATA / "week2_images.js"
    if not path.exists():
        return {}
    s = path.read_text(encoding="utf-8")
    return json.loads(s[s.index("{"):s.rindex("}") + 1])


def vocabulary_check(m, clues):
    """Which of each clue's words are on its page, and how rare they are. The rule for an 'other' clue was at most
    one rare page word (df <= 30); 'page' clues should have at least three page words."""
    print(f"{'clue':20s} {'style':6s} {'on page':>8s}  rare page words (df)")
    for c in clues:
        if c["style"] == "absent":
            continue
        t = m.ix[c["target"]]
        hits = X.distinctive_words(m, c["clue"], t)
        rare = [f"{w}({int(m.df[m.c.vi[w]])})" for w in hits if m.df[m.c.vi[w]] <= RARE_DF]
        flag = "  <-- leaks" if c["style"] == "other" and len(rare) > 1 else ("  <-- weak" if c["style"] == "page" and len(hits) < 3 else "")
        print(f"{c['id']:20s} {c['style']:6s} {len(hits):8d}  {', '.join(rare)}{flag}")


def main(check_only=False):
    m = T.build_model()
    clues = X.load_clues()
    if check_only:
        vocabulary_check(m, clues)
        return
    ids, N = m.ids, m.N
    images = load_images()
    enc = X.encoder()
    d180 = X.Dense(m.c.pages, ids, 180)
    d60 = X.Dense(m.c.pages, ids, 60)
    Q = enc.encode([c["clue"] for c in clues])
    with_img = [i for i, n in enumerate(ids) if images.get(n, {}).get("src")]
    with_set = set(with_img)

    chars = [{"id": n, "name": pretty(n), "url": "https://en.wikipedia.org/wiki/" + n, "img": images.get(n, {}).get("src"),
              "tokens": int(m.length[i]), "kin": int(m.G.in_degree(n))} for i, n in enumerate(ids)]

    out = []
    tally = {}
    for k, c in enumerate(clues):
        absent = c["style"] == "absent"
        t = -1 if absent else m.ix[c["target"]]
        kw, parts, unknown = X.keyword_scores(m, c["clue"])
        mean, best, arg = d180.scores(Q[k])
        _, best60, _ = d60.scores(Q[k])
        kw_top, dn_top = X.topk(kw, TOP), X.topk(best, TOP)
        ranks = None if absent else {"kw": X.rank_of(kw, t), "dn": X.rank_of(best, t), "dn60": X.rank_of(best60, t), "mean": X.rank_of(mean, t)}
        # evidence: keyword, word by word, for its pick and for the right page
        pick_kw, pick_dn = kw_top[0], dn_top[0]
        words = []
        for w in dict.fromkeys(X.query_tokens(c["clue"])):
            if w in parts:
                words.append([w, round(float(parts[w][pick_kw]), 4), None if absent else round(float(parts[w][t]), 4),
                              int(m.df[m.c.vi[w]])])
        passages = {}
        for p in dict.fromkeys([pick_dn, pick_kw] + ([] if absent else [t])):
            passages[str(p)] = {"text": d180.texts[arg[p]], "score": round(float(best[p]), 4)}
        # the lineup: the right page, both picks, then the runners-up of both engines (with a portrait), alternating
        rnd = random.Random("in-other-words:" + c["id"])
        lineup = [] if absent else [t]
        for p in (pick_kw, pick_dn):
            if p not in lineup:
                lineup.append(p)
        kw_all, dn_all = X.topk(kw, 40), X.topk(best, 40)
        for p in (q for pair in zip(kw_all, dn_all) for q in pair):
            if len(lineup) == LINEUP:
                break
            if p not in lineup and p in with_set:
                lineup.append(p)
        pool = [i for i in with_img if i not in lineup]
        lineup += rnd.sample(pool, LINEUP - len(lineup))
        rnd.shuffle(lineup)
        hits = [] if absent else X.distinctive_words(m, c["clue"], t)
        out.append({"id": c["id"], "style": c["style"], "target": t, "name": c["target"] if absent else pretty(c["target"]),
                    "clue": c["clue"], "play": c["id"] not in NOT_PLAYED, "why_not": NOT_PLAYED.get(c["id"]),
                    "kw": {"top": [[p, round(float(kw[p]), 4)] for p in kw_top], "words": words, "unknown": unknown},
                    "dn": {"top": [[p, round(float(best[p]), 4)] for p in dn_top], "passages": passages},
                    "ranks": ranks, "lineup": lineup,
                    "overlap": {"page": len(hits), "rare": sum(1 for w in hits if m.df[m.c.vi[w]] <= RARE_DF),
                                "words": len([w for w in dict.fromkeys(X.query_tokens(c["clue"])) if w in m.c.vi and not m.ishabit[m.c.vi[w]]])}})
        st = tally.setdefault(c["style"], {"n": 0, "kw1": 0, "dn1": 0, "kw3": 0, "dn3": 0})
        st["n"] += 1
        if ranks:
            st["kw1"] += ranks["kw"] == 1
            st["dn1"] += ranks["dn"] == 1
            st["kw3"] += ranks["kw"] <= 3
            st["dn3"] += ranks["dn"] <= 3

    data = {"meta": {"built": datetime.date.today().isoformat(), "model": X.MODEL, "passage_words": X.PASSAGE_WORDS,
                     "n_passages": len(d180.texts), "n_passages_60": len(d60.texts), "n_chars": N, "n_clues": len(out),
                     "lineup": LINEUP, "top": TOP, "styles": {s: v["n"] for s, v in tally.items()}},
            "chars": chars, "clues": out}
    OUT.write_text("// Autogenerated by scripts/build_week7_show.py - do not edit by hand.\n"
                   "window.SHOW = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} kB): {len(out)} clues, {len(d180.texts)} passages")
    for s, v in tally.items():
        if s == "absent":
            continue
        print(f"  {s:6s} n={v['n']:3d}  top-1  keyword {v['kw1']:3d}  meaning {v['dn1']:3d}   |  top-3  keyword {v['kw3']:3d}  meaning {v['dn3']:3d}")
    for c in out:
        if c["style"] == "absent":
            print(f"  absent {c['name']:16s} keyword -> {chars[c['kw']['top'][0][0]]['name']} ({c['kw']['top'][0][1]:.3f})   "
                  f"meaning -> {chars[c['dn']['top'][0][0]]['name']} ({c['dn']['top'][0][1]:.3f})")


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
