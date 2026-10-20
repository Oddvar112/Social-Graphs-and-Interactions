"""
Week 7 analysis: say it in other words.

The question: when does searching by meaning beat searching by words, and what does each engine do when the answer
is not there at all? We wrote 128 clues before computing a single ranking (data/week7_clues.tsv): for 61 characters,
one clue in the page's own words and one with the same facts in other words, plus six characters who have no page.
Then two engines rank all 303 pages for every clue, and we record where the right page lands.

  KEYWORD          the week 6 TF-IDF (tf = count / words, idf = ln(303 / df), unit length), cosine with the clue's
                   TF-IDF vector. Sparse: a word it never read does not exist for it.
  MEANING          all-MiniLM-L6-v2, best passage: every page cut into passages of at most 180 words at sentence
                   boundaries (4,479 passages), a page scored by its best passage. Dense.
  and three changes of one choice (exercise 7.10):
  MEANING, 60      the same with passages of at most 60 words (14,985 passages)
  MEANING, mean    a page is the mean of its 180-word passage vectors, renormalised: one vector per page
  MEANING, cut     plain encode() of the whole page, which silently keeps the first 256 word pieces only

Outputs
  assets/figures/week7_paraphrase.png    the price of paraphrase for each engine, and how sure each one is
  data/stats_week7.json                  every number in the post
Reads
  data/week7_clues.tsv                   the clues, written before any ranking
  data/week7_meaning_misses.tsv          MEANING's 45 wrong number ones on other-words clues, read by hand and labelled
                                         (checked here against the ranking: same clues, same picks)

Run:  python scripts/analyse_week7.py        (a minute the first time, while the passages are encoded; then seconds)
"""

import collections
import json
import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import week5_text as W                              # noqa: E402
import week6_text as T                              # noqa: E402
import week7_text as X                              # noqa: E402
from analyse import (BG, FG, DIM, GRID, CYAN, GOLD, RED, PURPLE, FIGS, pretty, style)   # noqa: E402,F401

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "stats_week7.json"
READ = ROOT / "data" / "week7_meaning_misses.tsv"
VARIANTS = ["kw", "dn", "dn60", "mean", "cut"]
LABEL = {"kw": "KEYWORD (TF-IDF)", "dn": "MEANING (best 180-word passage)", "dn60": "MEANING, 60-word passages",
         "mean": "MEANING, page mean", "cut": "MEANING, first 256 word pieces"}


def summary(ranks):
    r = np.array(ranks)
    return {"n": int(len(r)), "top1": int((r == 1).sum()), "top3": int((r <= 3).sum()), "top10": int((r <= 10).sum()),
            "median": float(np.median(r)), "mrr": round(float(np.mean(1 / r)), 3)}


def auc(pos, neg):
    """Probability that a random right answer scored higher than a random wrong one (ties count half)."""
    if not pos or not neg:
        return None
    p, n = np.array(pos)[:, None], np.array(neg)[None, :]
    return round(float((p > n).mean() + 0.5 * (p == n).mean()), 3)


def content_words(m, text):
    return [w for w in dict.fromkeys(X.query_tokens(text)) if w in m.c.vi and not m.ishabit[m.c.vi[w]]]


def main():
    m = T.build_model()
    ids, N = m.ids, m.N
    clues = X.load_clues()
    enc = X.encoder()
    d180 = X.Dense(m.c.pages, ids, 180)
    d60 = X.Dense(m.c.pages, ids, 60)
    cut = X.encode_cached([m.c.pages[n] for n in ids], "whole")          # truncated to 256 word pieces by the encoder
    Q = enc.encode([c["clue"] for c in clues], query=True)
    pieces = [enc.pieces(m.c.pages[n]) for n in ids]

    rows = []
    for k, c in enumerate(clues):
        absent = c["style"] == "absent"
        t = None if absent else m.ix[c["target"]]
        kw, parts, unknown = X.keyword_scores(m, c["clue"])
        mean, best, arg = d180.scores(Q[k])
        _, best60, _ = d60.scores(Q[k])
        sc = {"kw": kw, "dn": best, "dn60": best60, "mean": mean, "cut": cut @ Q[k]}
        row = {"id": c["id"], "style": c["style"], "target": c["target"], "clue": c["clue"], "unknown": unknown,
               "pick": {v: ids[int(np.argmax(s))] for v, s in sc.items()},
               "top": {v: round(float(np.max(s)), 4) for v, s in sc.items()}}
        if not absent:
            row["rank"] = {v: X.rank_of(s, t) for v, s in sc.items()}
            hits = X.distinctive_words(m, c["clue"], t)
            row["page_words"] = len(hits)
            row["rare_page_words"] = int(sum(m.df[m.c.vi[w]] <= 30 for w in hits))
            row["content_words"] = len(content_words(m, c["clue"]))
            # which word carried KEYWORD's pick, and is it part of that page's own name?
            p = int(np.argmax(kw))
            if parts:
                w, v = max(((w, float(a[p])) for w, a in parts.items()), key=lambda x: x[1])
                row["kw_carrier"] = {"word": w, "share": round(v / float(kw[p]), 3) if kw[p] > 0 else 0.0,
                                     "in_name": w in T.own_words(ids[p]), "df": int(m.df[m.c.vi[w]])}
            # how many of the clue's content words are in the passage MEANING matched on the right page
            cw = content_words(m, c["clue"])
            ptoks = set(X.query_tokens(d180.texts[arg[t]]))
            row["dn_passage_overlap"] = [sum(w in ptoks for w in cw), len(cw)]
        rows.append(row)

    by = collections.defaultdict(list)
    for r in rows:
        by[r["style"]].append(r)
    s = {"meta": {"model": X.MODEL, "n_pages": N, "pages_over_256_pieces": int(sum(p > 256 for p in pieces)),
                  "median_pieces": int(np.median(pieces)), "n_passages_180": len(d180.texts), "n_passages_60": len(d60.texts),
                  "n_clues": len(rows), "styles": {k: len(v) for k, v in by.items()}},
         "accuracy": {st: {v: summary([r["rank"][v] for r in by[st]]) for v in VARIANTS} for st in ("page", "other")}}

    # ------------------------------------------------------------ the price of paraphrase, paired by character
    pairs = {}
    for r in rows:
        if r["style"] in ("page", "other"):
            pairs.setdefault(r["target"], {})[r["style"]] = r
    paired = [(v["page"], v["other"]) for v in pairs.values() if "page" in v and "other" in v]
    s["paired"] = {"n": len(paired)}
    for v in ("kw", "dn"):
        a = np.array([p["rank"][v] for p, _ in paired])
        b = np.array([o["rank"][v] for _, o in paired])
        s["paired"][v] = {"median_page": float(np.median(a)), "median_other": float(np.median(b)),
                          "worse": int((b > a).sum()), "same": int((b == a).sum()), "better": int((b < a).sum()),
                          "top3_both": int(((a <= 3) & (b <= 3)).sum())}
    o = by["other"]
    s["head_to_head_other"] = {"meaning_better": sum(r["rank"]["dn"] < r["rank"]["kw"] for r in o),
                               "keyword_better": sum(r["rank"]["kw"] < r["rank"]["dn"] for r in o),
                               "tie": sum(r["rank"]["kw"] == r["rank"]["dn"] for r in o)}
    p = by["page"]
    s["head_to_head_page"] = {"meaning_better": sum(r["rank"]["dn"] < r["rank"]["kw"] for r in p),
                              "keyword_better": sum(r["rank"]["kw"] < r["rank"]["dn"] for r in p),
                              "tie": sum(r["rank"]["kw"] == r["rank"]["dn"] for r in p)}

    # ------------------------------------------------------------ why KEYWORD misses: the word that carried its pick
    misses = [r for r in o if r["rank"]["kw"] > 1 and "kw_carrier" in r]
    s["keyword_misses_other"] = {
        "n": len(misses),
        "carrier_in_pick_name": sum(r["kw_carrier"]["in_name"] for r in misses),
        "median_carrier_share": round(float(np.median([r["kw_carrier"]["share"] for r in misses])), 3),
        "carrier_over_half": sum(r["kw_carrier"]["share"] >= 0.5 for r in misses),
        "picks": collections.Counter(pretty(r["pick"]["kw"]) for r in misses).most_common(6),
        "carriers": collections.Counter(r["kw_carrier"]["word"] for r in misses).most_common(10),
        "examples": [{"id": r["id"], "pick": pretty(r["pick"]["kw"]), **r["kw_carrier"]} for r in misses],
    }
    s["unknown_words"] = {st: {"clues_with": sum(bool(r["unknown"]) for r in by[st]), "words": sum(len(r["unknown"]) for r in by[st])}
                          for st in ("page", "other")}
    # KEYWORD's success on other-words clues by how many rare page words leaked into the clue
    s["keyword_by_leak"] = {str(k): summary([r["rank"]["kw"] for r in o if min(r["rare_page_words"], 2) == k]) for k in (0, 1, 2)
                            if any(min(r["rare_page_words"], 2) == k for r in o)}
    # MEANING's right answers on other-words clues: the matched passage shares few of the clue's words
    hit = [r for r in o if r["rank"]["dn"] == 1]
    s["meaning_wins_overlap"] = {"n": len(hit), "shared": int(sum(r["dn_passage_overlap"][0] for r in hit)),
                                 "content": int(sum(r["dn_passage_overlap"][1] for r in hit)),
                                 "median_share": round(float(np.median([r["dn_passage_overlap"][0] / max(1, r["dn_passage_overlap"][1]) for r in hit])), 3)}

    # ------------------------------------------------------------ MEANING's misses, read by hand (data/week7_meaning_misses.tsv)
    read = {}
    with open(READ, encoding="utf-8", newline="") as f:
        for line in f:
            if line.startswith("#") or line.startswith("id\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                read[parts[0]] = {"pick": parts[1], "label": parts[2], "note": parts[3] if len(parts) > 3 else ""}
    missed = {r["id"]: r for r in o if r["rank"]["dn"] > 1}
    if set(read) != set(missed):
        sys.exit(f"{READ.name} must list exactly MEANING's misses: missing {sorted(set(missed) - set(read))}, extra {sorted(set(read) - set(missed))}")
    for k, r in missed.items():
        if read[k]["pick"] != pretty(r["pick"]["dn"]):
            sys.exit(f"{READ.name}: {k} says MEANING picked {read[k]['pick']}, the ranking says {pretty(r['pick']['dn'])}")
    s["meaning_misses_read"] = {"n": len(read), "labels": dict(collections.Counter(v["label"] for v in read.values()).most_common())}

    # ------------------------------------------------------------ nobody here: confidence is not validity
    conf = {}
    for v in ("kw", "dn"):
        right = [r["top"][v] for r in rows if r["style"] != "absent" and r["rank"][v] == 1]
        wrong = [r["top"][v] for r in rows if r["style"] != "absent" and r["rank"][v] > 1]
        gone = [r["top"][v] for r in by["absent"]]
        conf[v] = {"right_median": round(float(np.median(right)), 3), "wrong_median": round(float(np.median(wrong)), 3),
                   "absent": [round(x, 3) for x in gone], "auc_right_vs_wrong": auc(right, wrong),
                   "auc_right_vs_absent": auc(right, gone), "n_right": len(right), "n_wrong": len(wrong),
                   "right_below_median_absent": int(sum(x < np.median(gone) for x in right))}
    s["confidence"] = conf
    s["absent"] = [{"name": r["target"], "keyword": pretty(r["pick"]["kw"]), "keyword_score": r["top"]["kw"],
                    "meaning": pretty(r["pick"]["dn"]), "meaning_score": r["top"]["dn"]} for r in by["absent"]]
    s["clues"] = [{k: r.get(k) for k in ("id", "style", "target", "rank", "pick", "top", "unknown", "page_words",
                                         "rare_page_words", "kw_carrier", "dn_passage_overlap")} for r in rows]

    OUT.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    figure(paired, rows, by)
    print_summary(s)


# ---------------------------------------------------------------------------- the figure
def figure(paired, rows, by):
    fig = plt.figure(figsize=(13.6, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.32)
    ticks = [1, 3, 10, 30, 100, 300]

    for k, (v, col, name) in enumerate((("kw", CYAN, "KEYWORD (TF-IDF)"), ("dn", PURPLE, "MEANING (MiniLM, best passage)"))):
        ax = fig.add_subplot(gs[0, k])
        a = np.array([p["rank"][v] for p, _ in paired])
        b = np.array([o["rank"][v] for _, o in paired])
        jit = np.random.default_rng(7 + k).uniform(-0.06, 0.06, len(a))
        for i in range(len(a)):
            keep = b[i] <= 3
            ax.plot([0 + jit[i], 1 + jit[i]], [a[i], b[i]], color=col if keep else DIM, alpha=0.85 if keep else 0.33,
                    lw=1.5 if keep else 0.9, zorder=3 if keep else 2)
        ax.scatter(np.zeros(len(a)) + jit, a, s=13, color=GOLD, zorder=4, edgecolor=BG, lw=0.4)
        ax.scatter(np.ones(len(b)) + jit, b, s=13, color=col, zorder=4, edgecolor=BG, lw=0.4)
        ax.set_yscale("log")
        ax.set_ylim(400, 0.8)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"#{t}" for t in ticks])
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([0, 1])
        top3a, top3b = int((a <= 3).sum()), int((b <= 3).sum())
        ax.set_xticklabels([f"the page's own words\ntop 3: {top3a} of {len(a)}", f"other words\ntop 3: {top3b} of {len(b)}"],
                           fontsize=9.5)
        for lab, c in zip(ax.get_xticklabels(), (GOLD, col)):
            lab.set_color(c)
        ax.axhspan(0.8, 3.5, color=col, alpha=0.06, zorder=0)
        style(ax, name, None, "rank of the right page among 303" if k == 0 else None)
        ax.grid(axis="x", visible=False)
        ax.text(0.99, 0.015, f"median rank #{np.median(a):g} → #{np.median(b):g}", transform=ax.transAxes, ha="right",
                va="bottom", color=FG, fontsize=9.5)

    ax = fig.add_subplot(gs[0, 2])
    groups = [("right", lambda r, v: r["style"] != "absent" and r["rank"][v] == 1, GOLD),
              ("wrong", lambda r, v: r["style"] != "absent" and r["rank"][v] > 1, DIM),
              ("nobody\nis there", lambda r, v: r["style"] == "absent", RED)]
    rng = np.random.default_rng(3)
    where = {}
    for j, (v, col, name) in enumerate((("kw", CYAN, "KEYWORD"), ("dn", PURPLE, "MEANING"))):
        for g, (lab, f, c) in enumerate(groups):
            rs = [r for r in rows if f(r, v)]
            ys = [r["top"][v] for r in rs]
            xs = j * 3.6 + g + rng.uniform(-0.22, 0.22, len(ys))
            ax.scatter(xs, ys, s=16 if lab.startswith("nobody") else 10, color=c, alpha=0.95 if lab.startswith("nobody") else 0.6,
                       edgecolor=BG, lw=0.4, zorder=3)
            if v == "dn" and lab.startswith("nobody"):
                where.update({r["target"]: (x, y) for r, x, y in zip(rs, xs, ys)})
            if ys:
                ax.plot([j * 3.6 + g - 0.3, j * 3.6 + g + 0.3], [np.median(ys)] * 2, color=FG, lw=1.6, zorder=4)
        ax.text(j * 3.6 + 1, 0.9, name, ha="center", color=col, fontsize=10.5, fontweight="bold")
    for r in by["absent"]:
        if r["target"] in ("Thor", "Magneto"):
            up = r["target"] == "Thor"
            ax.annotate(f"{r['target']} → {pretty(r['pick']['dn'])}", where[r["target"]],
                        xytext=(5.75, 0.68 if up else 0.32), ha="center", va="center", color=FG, fontsize=8.5,
                        arrowprops=dict(arrowstyle="-", color=DIM, lw=0.6, shrinkA=4, shrinkB=3))
    ax.set_xticks([0, 1, 2, 3.6, 4.6, 5.6])
    ax.set_xticklabels([g[0] for g in groups] * 2, fontsize=8.8)
    ax.set_ylim(0, 0.95)
    ax.set_xlim(-0.6, 6.5)
    style(ax, "How sure was the number one?", None, "score of the engine's number one")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIGS / "week7_paraphrase.png", bbox_inches="tight")
    plt.close(fig)


def print_summary(s):
    print(f"pages over 256 word pieces: {s['meta']['pages_over_256_pieces']} of {s['meta']['n_pages']} "
          f"(median {s['meta']['median_pieces']}); passages {s['meta']['n_passages_180']} at 180, {s['meta']['n_passages_60']} at 60")
    for st in ("page", "other"):
        print(f"\n{st} words (n={s['accuracy'][st]['kw']['n']})")
        for v in VARIANTS:
            a = s["accuracy"][st][v]
            print(f"  {LABEL[v]:34s} top1 {a['top1']:3d}  top3 {a['top3']:3d}  top10 {a['top10']:3d}  median {a['median']:6.1f}  MRR {a['mrr']:.3f}")
    print("\npaired:", json.dumps(s["paired"]))
    print("head to head, other words:", s["head_to_head_other"], "| page words:", s["head_to_head_page"])
    km = s["keyword_misses_other"]
    print(f"keyword misses on other words: {km['n']}; carrier word is in the pick's own name in {km['carrier_in_pick_name']}; "
          f"carries >= half the pick's score in {km['carrier_over_half']} (median share {km['median_carrier_share']})")
    print("  picks:", km["picks"])
    print("  carriers:", km["carriers"])
    print("unknown words:", s["unknown_words"])
    print("keyword on other words by leaked rare page words:", {k: (v["n"], v["top1"], v["top3"]) for k, v in s["keyword_by_leak"].items()})
    print("meaning's right answers on other words, clue words in the matched passage:", s["meaning_wins_overlap"])
    print("meaning's misses on other words, read by hand:", s["meaning_misses_read"])
    for v in ("kw", "dn"):
        c = s["confidence"][v]
        print(f"confidence {v}: right median {c['right_median']} (n={c['n_right']}), wrong median {c['wrong_median']} (n={c['n_wrong']}), "
              f"absent {c['absent']}, AUC right/wrong {c['auc_right_vs_wrong']}, right/absent {c['auc_right_vs_absent']}, "
              f"right answers below the absent median: {c['right_below_median_absent']}")
    for a in s["absent"]:
        print(f"  {a['name']:16s} keyword -> {a['keyword']} ({a['keyword_score']:.3f})  meaning -> {a['meaning']} ({a['meaning_score']:.3f})")


if __name__ == "__main__":
    main()
