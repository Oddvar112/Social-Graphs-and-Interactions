"""
Data for IN OTHER WORDS (weeks/week7/show): a game show in which a host reads a description of a Marvel character
and you race two search engines to point at the right portrait, or to say that nobody here fits.

For every clue in data/week7_clues.tsv (written by hand before any of this ran) the two machines rank all 303 pages:

  KEYWORD   sparse retrieval: the cosine between the clue's TF-IDF vector and each page's, with the week 6
            definitions (tf = count / words, idf = ln(303 / df), unit length). A word the corpus has never seen is
            invisible to it.
  MEANING   dense retrieval with mxbai-embed-large-v1 (week7_text.GAME_MODEL), the strongest of the six encoders
            scripts/compare_week7_encoders.py tried: it encodes the clue and every passage of every page (at most
            180 words, split at sentence boundaries; 4,479 passages), and a page's score is its best passage. The
            post's own analysis is about the course's smaller MiniLM; its rank is stored too, for the replay.

Each machine also says NOBODY HERE when its number one does not stand out: KEYWORD when its number one is less than
a threshold ahead of the runner-up (gap = (best - second) / standard deviation of the 303 scores), MEANING when its
number one stands less than a threshold above the average page (z = (best - mean) / standard deviation). Those are
the measures that worked best for each in the comparison; the thresholds are fitted here, on the clues the game deals,
to the game's own scoring (4 clues in the page's words, 5 in other words, 1 about a missing character). Fitted on the
same clues they are judged on, they flatter the machines a little; the post quotes the leave-one-out numbers.

Stored per clue: each machine's top ten with scores, its standout measures and whether it says nobody; the rank of
the right page (MEANING at 180 and 60 words, MiniLM at 180, KEYWORD); the evidence (for KEYWORD, each clue word's
contribution to its pick's score and to the right page's, and the words it never read; for MEANING, the best passage
of its pick, of the right page and of KEYWORD's pick, in full, so the text stays visible); and a lineup of six
portraits: the right answer, each machine's number one, and then the next pages down both lists, so that every decoy
is a page one of the machines found plausible, in a fixed shuffled order. A clue about a character who is not in the
corpus has no right page; its lineup is the two number ones plus their runners-up.

One clue is kept out of the game (but not out of the post): see NOT_PLAYED.

Writes data/week7_show.js (window.SHOW). Deterministic given the frozen data and the cached embeddings.

Run:  python scripts/build_week7_show.py            (the first run encodes the passages: a few minutes on a GPU)
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

# Which standout measure each machine's NOBODY HERE rule uses (scripts/compare_week7_encoders.py: the best by
# leave-one-out episode score for each engine).
RULES = {"kw": "gap", "dn": "z"}


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
    d180 = X.Dense(m.c.pages, ids, 180, X.GAME_MODEL)
    d60 = X.Dense(m.c.pages, ids, 60, X.GAME_MODEL)
    Q = X.encoder(X.GAME_MODEL).encode([c["clue"] for c in clues], query=True)
    X.release(X.GAME_MODEL)
    mini = X.Dense(m.c.pages, ids, 180, X.MODEL)
    Qm = X.encoder(X.MODEL).encode([c["clue"] for c in clues], query=True)
    with_img = [i for i, n in enumerate(ids) if images.get(n, {}).get("src")]
    with_set = set(with_img)

    chars = [{"id": n, "name": pretty(n), "url": "https://en.wikipedia.org/wiki/" + n, "img": images.get(n, {}).get("src"),
              "tokens": int(m.length[i]), "kin": int(m.G.in_degree(n))} for i, n in enumerate(ids)]

    out = []
    for k, c in enumerate(clues):
        absent = c["style"] == "absent"
        t = -1 if absent else m.ix[c["target"]]
        kw, parts, unknown = X.keyword_scores(m, c["clue"])
        _, best, arg = d180.scores(Q[k])
        _, best60, _ = d60.scores(Q[k])
        _, bestm, _ = mini.scores(Qm[k])
        kw_top, dn_top = X.topk(kw, TOP), X.topk(best, TOP)
        ranks = None if absent else {"kw": X.rank_of(kw, t), "dn": X.rank_of(best, t), "dn60": X.rank_of(best60, t), "mini": X.rank_of(bestm, t)}
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
        sk, sd = X.standout(kw), X.standout(best)
        out.append({"id": c["id"], "style": c["style"], "target": t, "name": c["target"] if absent else pretty(c["target"]),
                    "clue": c["clue"], "play": c["id"] not in NOT_PLAYED, "why_not": NOT_PLAYED.get(c["id"]),
                    "kw": {"top": [[p, round(float(kw[p]), 4)] for p in kw_top], "words": words, "unknown": unknown,
                           "z": round(sk["z"], 4), "gap": round(sk["gap"], 4)},
                    "dn": {"top": [[p, round(float(best[p]), 4)] for p in dn_top], "passages": passages,
                           "z": round(sd["z"], 4), "gap": round(sd["gap"], 4)},
                    "ranks": ranks, "lineup": lineup,
                    "overlap": {"page": len(hits), "rare": sum(1 for w in hits if m.df[m.c.vi[w]] <= RARE_DF),
                                "words": len([w for w in dict.fromkeys(X.query_tokens(c["clue"])) if w in m.c.vi and not m.ishabit[m.c.vi[w]]])}})

    # NOBODY HERE: fit each machine's threshold on the clues the game deals, then flag every clue
    played = [c for c in out if c["play"]]
    styles = np.array([c["style"] for c in played])
    rules = {}
    for key, feat in RULES.items():
        right = np.array([c["style"] != "absent" and c[key]["top"][0][0] == c["target"] for c in played])
        values = np.array([c[key][feat] for c in played])
        thr, score = X.fit_nobody(values, styles, right)
        # store the threshold rounded, and decide with the stored value, so the page and this file agree exactly
        thr = round(thr, 4)
        for c in out:
            c[key]["nobody"] = bool(c[key][feat] < thr)
        nob = np.array([c[key]["nobody"] for c in played])
        rules[key] = {"feature": feat, "threshold": thr, "episode": round(X.episode_score(styles, right, nob), 3),
                      "episode_always_answers": round(X.episode_score(styles, right, np.zeros(len(played), bool)), 3),
                      "missing_caught": int(nob[styles == "absent"].sum()), "missing": int((styles == "absent").sum()),
                      "right_given_up": int((nob & right).sum())}

    tally = {}
    for c in played:
        st = tally.setdefault(c["style"], {"n": 0, "kw": 0, "dn": 0})
        st["n"] += 1
        for key in ("kw", "dn"):
            ok = c[key]["nobody"] if c["style"] == "absent" else (not c[key]["nobody"] and c[key]["top"][0][0] == c["target"])
            st[key] += ok

    data = {"meta": {"built": datetime.date.today().isoformat(), "model": X.GAME_MODEL, "model_short": X.SPECS[X.GAME_MODEL]["short"],
                     "course_model": X.MODEL, "passage_words": 180, "n_passages": len(d180.texts), "n_passages_60": len(d60.texts),
                     "n_chars": N, "n_clues": len(out), "lineup": LINEUP, "top": TOP, "rules": rules,
                     "styles": {s: sum(1 for c in out if c["style"] == s) for s in ("page", "other", "absent")}},
            "chars": chars, "clues": out}
    OUT.write_text("// Autogenerated by scripts/build_week7_show.py - do not edit by hand.\n"
                   "window.SHOW = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} kB): {len(out)} clues, {len(d180.texts)} passages, MEANING = {X.GAME_MODEL}")
    for key, r in rules.items():
        print(f"  {key}: nobody when {r['feature']} < {r['threshold']}: {r['episode']:.2f} points an episode "
              f"(always answering: {r['episode_always_answers']:.2f}); says nobody to {r['missing_caught']} of {r['missing']} missing, "
              f"gives up {r['right_given_up']} right answers")
    for s, v in tally.items():
        print(f"  {s:6s} n={v['n']:3d}  right in the game: keyword {v['kw']:3d}  meaning {v['dn']:3d}")


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
