"""
Week 7: change the encoder (exercise 7.10), and teach the machines to say NOBODY HERE.

The post's MEANING is the course's encoder, all-MiniLM-L6-v2. Here the same 128 clues, the same passages (at most 180
or 60 words, split at sentence boundaries) and the same scoring (a page is its best passage) are run with five larger
sentence encoders, each used the way its model card says (pooling, query and passage prefixes, input length; see
week7_text.SPECS). KEYWORD, the week 6 TF-IDF, is the reference.

For every engine it reports what the post reports (right first, in the top three, median rank, for clues in the page's
own words and in other words) and the game's measure: expected points in a ten-clue episode of 4 page-words clues,
5 other-words clues and 1 about a missing character, which a machine that always answers can never score.

Then the NOBODY HERE rule: say nobody when the number one does not stand out from the 303 pages enough, measured as
  top   the best score itself
  z     (best - mean) / standard deviation of all 303 scores
  gap   (best - runner-up) / standard deviation
with the threshold that maximises the episode score. A threshold fitted on the clues it is then judged on flatters
it, so the score that counts is leave-one-out: every clue judged by a threshold fitted without it. The missing
characters are the five the game deals (Iron Man is set aside: the corpus has his Ultimate Marvel version).

Outputs data/week7_encoders.json. Downloads the five encoders on the first run (about 2 GB in the Hugging Face cache;
mxbai-embed-large is 670 MB of it) and encodes 19,464 passages with each; after that, seconds.

Run:  python scripts/compare_week7_encoders.py
"""

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import week6_text as T                              # noqa: E402
import week7_text as X                              # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "week7_encoders.json"
FEATURES = ("top", "z", "gap")
SET_ASIDE = {"ironman-absent"}


def summary(rows):
    s = {}
    for st in ("page", "other"):
        r = np.array([x["rank"] for x in rows if x["style"] == st])
        s[st] = {"n": int(len(r)), "top1": int((r == 1).sum()), "top3": int((r <= 3).sum()), "top10": int((r <= 10).sum()),
                 "median": float(np.median(r))}
    played = [x for x in rows if x["id"] not in SET_ASIDE]
    styles = np.array([x["style"] for x in played])
    right = np.array([x["style"] != "absent" and x["rank"] == 1 for x in played])
    s["episode"] = round(X.episode_score(styles, right, np.zeros(len(played), bool)), 3)
    rules = {}
    for f in FEATURES:
        v = np.array([x[f] for x in played])
        t, e_in = X.fit_nobody(v, styles, right)
        loo = X.loo_nobody(v, styles, right)
        rules[f] = {"threshold": round(t, 4), "episode_in": round(e_in, 3), "episode_loo": round(X.episode_score(styles, right, loo), 3),
                    "caught_loo": int(sum(loo[styles == "absent"])), "missing": int(sum(styles == "absent")),
                    "right_lost_loo": int(sum(loo & right)),
                    "caught_in": int(sum((v < t)[styles == "absent"])), "right_lost_in": int(sum((v < t) & right))}
    best = max(FEATURES, key=lambda f: (rules[f]["episode_loo"], -FEATURES.index(f)))
    s["nobody"] = {"rules": rules, "best": best}
    return s


def engine_rows(clues, m, score_fn):
    rows = []
    for k, c in enumerate(clues):
        sc = score_fn(k, c)
        t = None if c["style"] == "absent" else m.ix[c["target"]]
        rows.append({"id": c["id"], "style": c["style"], "rank": None if t is None else X.rank_of(sc, t),
                     "pick": m.ids[int(np.argmax(sc))], **X.standout(sc)})
    return rows


def main():
    m = T.build_model()
    clues = X.load_clues()
    out = {"meta": {"clues": len(clues), "set_aside": sorted(SET_ASIDE), "mix": X.EPISODE_MIX, "features": FEATURES}, "engines": []}

    kw_rows = engine_rows(clues, m, lambda k, c: X.keyword_scores(m, c["clue"])[0])
    out["engines"].append({"name": "KEYWORD (TF-IDF)", "model": None, "words": None, "summary": summary(kw_rows), "rows": kw_rows})

    for model, spec in X.SPECS.items():
        t0 = time.time()
        enc = X.encoder(model)
        Q = enc.encode([c["clue"] for c in clues], query=True)
        n_params = sum(p.numel() for p in enc.model.parameters())
        for words in (180, 60):
            d = X.Dense(m.c.pages, m.ids, words, model)
            rows = engine_rows(clues, m, lambda k, c, d=d: d.scores(Q[k])[1])
            out["engines"].append({"name": spec["short"], "model": model, "words": words, "params_m": round(n_params / 1e6),
                                   "passages": len(d.texts), "summary": summary(rows), "rows": rows})
        X.release(model)
        print(f"  {spec['short']:18s} {round(time.time() - t0):4d} s", flush=True)

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}\n")
    print(f"{'engine':20s} {'words':>5s} {'params':>6s} | {'page 1st':>8s} {'top3':>4s} | {'other 1st':>9s} {'top3':>4s} {'med':>4s} | "
          f"{'episode':>7s} | best nobody rule (leave one out)")
    for e in out["engines"]:
        s = e["summary"]
        r = s["nobody"]["rules"][s["nobody"]["best"]]
        print(f"{e['name']:20s} {str(e['words'] or ''):>5s} {str(e.get('params_m') or ''):>6s} | {s['page']['top1']:8d} {s['page']['top3']:4d} | "
              f"{s['other']['top1']:9d} {s['other']['top3']:4d} {s['other']['median']:4.0f} | {s['episode']:7.2f} | "
              f"{s['nobody']['best']:3s} < {r['threshold']:.3f}: {r['episode_loo']:.2f} an episode, says nobody to {r['caught_loo']} of {r['missing']} "
              f"missing, gives up {r['right_lost_loo']} right answers")


if __name__ == "__main__":
    main()
