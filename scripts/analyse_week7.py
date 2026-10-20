"""
Week 7 analysis: say it in other words.

The question: when does searching by meaning beat searching by words, and what does each engine do when the answer
is not there at all? We wrote 128 clues before computing a single ranking (data/week7_clues.tsv): for 61 characters,
one clue in the page's own words and one with the same facts in other words, plus six characters who have no page.
Then three engines rank all 303 pages for every clue, and we record where the right page lands.

  KEYWORD          the week 6 TF-IDF (tf = count / words, idf = ln(303 / df), unit length), cosine with the clue's
                   TF-IDF vector. Sparse: a word it never read does not exist for it.
  MiniLM           the course's encoder, all-MiniLM-L6-v2 (23 M parameters), best passage: every page cut into passages
                   of at most 180 words at sentence boundaries (4,479 passages), a page scored by its best passage.
  mxbai            mxbai-embed-large-v1 (335 M parameters), the strongest of the six encoders that
                   scripts/compare_week7_encoders.py tried, the same passages and scoring; the game's MEANING.
  and three changes of one choice for each encoder (exercise 7.10):
  60               passages of at most 60 words (14,985 passages)
  mean             a page is the mean of its 180-word passage vectors, renormalised: one vector per page
  cut              plain encode() of the whole page, which silently keeps the first 256 (MiniLM) or 512 (mxbai) pieces

And the NOBODY HERE rule: say nobody when the number one does not stand out from the 303 pages, z = (best - mean) / sd
for the encoders, gap = (best - runner-up) / sd for KEYWORD, with the threshold that scores best in the game's episode
(4 page-words clues, 5 other-words clues, 1 missing character), judged leave-one-out.

Outputs
  assets/figures/week7_paraphrase.png    the price of paraphrase for each engine
  assets/figures/week7_nobody.png        how sure each engine's number one was, and how far it stood out
  data/stats_week7.json                  every number in the post
Reads
  data/week7_clues.tsv                   the clues, written before any ranking
  data/week7_meaning_misses.tsv          MiniLM's 45 wrong number ones on other-words clues, read by hand and labelled
  data/week7_meaning_misses_mxbai.tsv    mxbai's 36, the same way (both checked here against the ranking)

Run:  python scripts/analyse_week7.py        (minutes the first time, while the passages are encoded; then seconds)
"""

import collections
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import week6_text as T                              # noqa: E402
import week7_text as X                              # noqa: E402
from analyse import (BG, FG, DIM, CYAN, GOLD, RED, PURPLE, FIGS, pretty, style)   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "stats_week7.json"
READ = {"dn": ROOT / "data" / "week7_meaning_misses.tsv", "mx": ROOT / "data" / "week7_meaning_misses_mxbai.tsv"}
PINK = "#ff7ac8"
ENGINES = ("kw", "dn", "mx")
VARIANTS = ["kw", "dn", "dn60", "mean", "cut", "mx", "mx60", "mxmean", "mxcut"]
LABEL = {"kw": "KEYWORD (TF-IDF)", "dn": "MiniLM, best 180-word passage", "dn60": "MiniLM, 60-word passages",
         "mean": "MiniLM, page mean", "cut": "MiniLM, first 256 word pieces", "mx": "mxbai, best 180-word passage",
         "mx60": "mxbai, 60-word passages", "mxmean": "mxbai, page mean", "mxcut": "mxbai, first 512 word pieces"}
NAME = {"kw": "KEYWORD (TF-IDF)", "dn": "MiniLM (the course's)", "mx": "mxbai (the game's MEANING)"}
COLOR = {"kw": CYAN, "dn": PINK, "mx": PURPLE}
RULE = {"kw": "gap", "dn": "z", "mx": "z"}
SET_ASIDE = {"ironman-absent"}


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


def read_labels(path):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for line in f:
            if line.startswith("#") or line.startswith("id\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                out[parts[0]] = {"pick": parts[1], "label": parts[2], "note": parts[3] if len(parts) > 3 else ""}
    return out


def main():
    m = T.build_model()
    ids, N = m.ids, m.N
    clues = X.load_clues()
    texts = [c["clue"] for c in clues]
    whole = [m.c.pages[n] for n in ids]
    dense, Q, cut = {}, {}, {}
    for key, model in (("dn", X.MODEL), ("mx", X.GAME_MODEL)):
        dense[key] = (X.Dense(m.c.pages, ids, 180, model), X.Dense(m.c.pages, ids, 60, model))
        cut[key] = X.encode_cached(whole, "whole", model)          # truncated by the encoder to its input length
        enc = X.encoder(model)
        Q[key] = enc.encode(texts, query=True)
        if key == "dn":
            pieces = [enc.pieces(p) for p in whole]
        X.release(model)

    rows = []
    for k, c in enumerate(clues):
        absent = c["style"] == "absent"
        t = None if absent else m.ix[c["target"]]
        kw, parts, unknown = X.keyword_scores(m, c["clue"])
        sc, args = {"kw": kw}, {}
        for key in ("dn", "mx"):
            d180, d60 = dense[key]
            mean, best, arg = d180.scores(Q[key][k])
            _, best60, _ = d60.scores(Q[key][k])
            pre = "" if key == "dn" else "mx"
            sc[key] = best
            sc[pre + "60" if key == "mx" else "dn60"] = best60
            sc[pre + "mean" if key == "mx" else "mean"] = mean
            sc[pre + "cut" if key == "mx" else "cut"] = cut[key] @ Q[key][k]
            args[key] = arg
        row = {"id": c["id"], "style": c["style"], "target": c["target"], "clue": c["clue"], "unknown": unknown,
               "pick": {v: ids[int(np.argmax(s))] for v, s in sc.items()},
               "top": {v: round(float(np.max(s)), 4) for v, s in sc.items()},
               "standout": {e: {f: round(x, 4) for f, x in X.standout(sc[e]).items()} for e in ENGINES}}
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
            # how many of the clue's content words are in the passage each encoder matched on the right page
            cw = content_words(m, c["clue"])
            row["passage_overlap"] = {}
            for key in ("dn", "mx"):
                ptoks = set(X.query_tokens(dense[key][0].texts[args[key][t]]))
                row["passage_overlap"][key] = [sum(w in ptoks for w in cw), len(cw)]
        rows.append(row)

    by = collections.defaultdict(list)
    for r in rows:
        by[r["style"]].append(r)
    o, pg = by["other"], by["page"]
    s = {"meta": {"models": {"dn": X.MODEL, "mx": X.GAME_MODEL}, "n_pages": N,
                  "pages_over_256_pieces": int(sum(p > 256 for p in pieces)), "pages_over_512_pieces": int(sum(p > 512 for p in pieces)),
                  "median_pieces": int(np.median(pieces)), "n_passages_180": len(dense["dn"][0].texts),
                  "n_passages_60": len(dense["dn"][1].texts), "n_clues": len(rows), "styles": {k: len(v) for k, v in by.items()}},
         "accuracy": {st: {v: summary([r["rank"][v] for r in by[st]]) for v in VARIANTS} for st in ("page", "other")}}

    # ------------------------------------------------------------ the price of paraphrase, paired by character
    pairs = {}
    for r in rows:
        if r["style"] in ("page", "other"):
            pairs.setdefault(r["target"], {})[r["style"]] = r
    paired = [(v["page"], v["other"]) for v in pairs.values() if "page" in v and "other" in v]
    s["paired"] = {"n": len(paired)}
    for v in ENGINES:
        a = np.array([p["rank"][v] for p, _ in paired])
        b = np.array([q["rank"][v] for _, q in paired])
        s["paired"][v] = {"median_page": float(np.median(a)), "median_other": float(np.median(b)),
                          "worse": int((b > a).sum()), "same": int((b == a).sum()), "better": int((b < a).sum()),
                          "top3_both": int(((a <= 3) & (b <= 3)).sum())}

    def h2h(group, x, y):
        return {"x_better": sum(r["rank"][x] < r["rank"][y] for r in group), "y_better": sum(r["rank"][y] < r["rank"][x] for r in group),
                "tie": sum(r["rank"][x] == r["rank"][y] for r in group)}
    s["head_to_head"] = {f"{x}_vs_{y}": {"other": h2h(o, x, y), "page": h2h(pg, x, y)} for x, y in (("dn", "kw"), ("mx", "kw"), ("mx", "dn"))}

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
    pm = [r for r in pg if r["rank"]["kw"] > 1 and "kw_carrier" in r]
    s["keyword_misses_page"] = {"n": len(pm), "carrier_in_pick_name": sum(r["kw_carrier"]["in_name"] for r in pm)}
    s["unknown_words"] = {st: {"clues_with": sum(bool(r["unknown"]) for r in by[st]), "words": sum(len(r["unknown"]) for r in by[st])}
                          for st in ("page", "other")}
    s["keyword_by_leak"] = {str(k): summary([r["rank"]["kw"] for r in o if min(r["rare_page_words"], 2) == k]) for k in (0, 1, 2)
                            if any(min(r["rare_page_words"], 2) == k for r in o)}

    # ------------------------------------------------------------ the encoders' right answers and misses on other words
    s["meaning_wins_overlap"], s["meaning_misses_read"] = {}, {}
    for key in ("dn", "mx"):
        hit = [r for r in o if r["rank"][key] == 1]
        ov = [r["passage_overlap"][key] for r in hit]
        s["meaning_wins_overlap"][key] = {"n": len(hit), "shared": int(sum(a for a, _ in ov)), "content": int(sum(b for _, b in ov)),
                                          "median_share": round(float(np.median([a / max(1, b) for a, b in ov])), 3)}
        read = read_labels(READ[key])
        missed = {r["id"]: r for r in o if r["rank"][key] > 1}
        if set(read) != set(missed):
            sys.exit(f"{READ[key].name} must list exactly the misses: missing {sorted(set(missed) - set(read))}, extra {sorted(set(read) - set(missed))}")
        for k, r in missed.items():
            if read[k]["pick"] != pretty(r["pick"][key]):
                sys.exit(f"{READ[key].name}: {k} says the pick was {read[k]['pick']}, the ranking says {pretty(r['pick'][key])}")
        s["meaning_misses_read"][key] = {"n": len(read), "labels": dict(collections.Counter(v["label"] for v in read.values()).most_common())}
    both = [r for r in o if r["rank"]["dn"] > 1 and r["rank"]["mx"] > 1]
    s["meaning_misses_both"] = {"n": len(both), "same_pick": sum(r["pick"]["dn"] == r["pick"]["mx"] for r in both)}

    # ------------------------------------------------------------ nobody here: the score, and how far the number one stands out
    conf = {}
    for v in ENGINES:
        right = [r["top"][v] for r in rows if r["style"] != "absent" and r["rank"][v] == 1]
        wrong = [r["top"][v] for r in rows if r["style"] != "absent" and r["rank"][v] > 1]
        gone = [r["top"][v] for r in by["absent"]]
        mx_gone = max(gone)
        conf[v] = {"right_median": round(float(np.median(right)), 3), "wrong_median": round(float(np.median(wrong)), 3),
                   "absent": [round(x, 3) for x in gone], "auc_right_vs_wrong": auc(right, wrong), "n_right": len(right), "n_wrong": len(wrong),
                   "right_at_or_below_absent_max": int(sum(x <= mx_gone for x in right)),
                   "other_right": sum(1 for r in o if r["rank"][v] == 1),
                   "other_right_at_or_below_absent_max": int(sum(1 for r in o if r["rank"][v] == 1 and r["top"][v] <= mx_gone))}
    s["confidence"] = conf
    played = [r for r in rows if r["id"] not in SET_ASIDE]
    styles = np.array([r["style"] for r in played])
    s["nobody_rule"] = {}
    for v in ENGINES:
        f = RULE[v]
        right = np.array([r["style"] != "absent" and r["rank"][v] == 1 for r in played])
        vals = np.array([r["standout"][v][f] for r in played])
        thr, e_in = X.fit_nobody(vals, styles, right)
        loo = X.loo_nobody(vals, styles, right)
        s["nobody_rule"][v] = {"feature": f, "threshold": round(thr, 4),
                               "episode_always": round(float(X.episode_score(styles, right, np.zeros(len(played), bool))), 3),
                               "episode_in": round(e_in, 3), "episode_loo": round(float(X.episode_score(styles, right, loo)), 3),
                               "caught_in": int(((vals < thr) & (styles == "absent")).sum()), "lost_in": int(((vals < thr) & right).sum()),
                               "caught_loo": int((loo & (styles == "absent")).sum()), "lost_loo": int((loo & right).sum()),
                               "missing": int((styles == "absent").sum())}
    s["absent"] = [{"name": r["target"], **{e: {"pick": pretty(r["pick"][e]), "score": r["top"][e], f: r["standout"][e][RULE[e]]}
                                              for e, f in RULE.items()}} for r in by["absent"]]
    s["clues"] = [{k: r.get(k) for k in ("id", "style", "target", "rank", "pick", "top", "standout", "unknown", "page_words",
                                         "rare_page_words", "kw_carrier", "passage_overlap")} for r in rows]

    OUT.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    figure_paraphrase(paired)
    figure_nobody(rows, s)
    print_summary(s)


# ---------------------------------------------------------------------------- the figures
def figure_paraphrase(paired):
    fig = plt.figure(figsize=(14.4, 6.0))
    gs = fig.add_gridspec(1, 3, wspace=0.28)
    ticks = [1, 3, 10, 30, 100, 300]
    for k, v in enumerate(ENGINES):
        col = COLOR[v]
        ax = fig.add_subplot(gs[0, k])
        a = np.array([p["rank"][v] for p, _ in paired])
        b = np.array([q["rank"][v] for _, q in paired])
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
        ax.set_xticklabels([f"the page's own words\ntop 3: {top3a} of {len(a)}", f"other words\ntop 3: {top3b} of {len(b)}"], fontsize=9.5)
        for lab, c in zip(ax.get_xticklabels(), (GOLD, col)):
            lab.set_color(c)
        ax.axhspan(0.8, 3.5, color=col, alpha=0.06, zorder=0)
        style(ax, NAME[v], None, "rank of the right page among 303" if k == 0 else None)
        ax.grid(axis="x", visible=False)
        ax.text(0.99, 0.015, f"median rank #{np.median(a):g} → #{np.median(b):g}", transform=ax.transAxes, ha="right",
                va="bottom", color=FG, fontsize=9.5)
    fig.savefig(FIGS / "week7_paraphrase.png", bbox_inches="tight")
    plt.close(fig)


def figure_nobody(rows, s):
    fig = plt.figure(figsize=(14.4, 5.6))
    gs = fig.add_gridspec(1, 2, wspace=0.18)
    groups = [("right", lambda r, v: r["style"] != "absent" and r["rank"][v] == 1, GOLD),
              ("wrong", lambda r, v: r["style"] != "absent" and r["rank"][v] > 1, DIM),
              ("nobody\nis there", lambda r, v: r["style"] == "absent" and r["id"] not in SET_ASIDE, RED)]
    xs_of = {}
    for panel, (title, ylabel) in enumerate((("How sure was the number one?", "score of the engine's number one"),
                                             ("How far did it stand out from the other 302?", "standout: gap (KEYWORD) or z (the encoders)"))):
        ax = fig.add_subplot(gs[0, panel])
        rng = np.random.default_rng(3 + panel)
        clip = 8.6 if panel == 1 else None                    # a few KEYWORD gaps run off the chart: drawn at the top as triangles
        for j, v in enumerate(ENGINES):
            for g, (lab, f, c) in enumerate(groups):
                rs = [r for r in rows if f(r, v)]
                ys = [r["top"][v] if panel == 0 else r["standout"][v][RULE[v]] for r in rs]
                xs = j * 3.8 + g + rng.uniform(-0.22, 0.22, len(ys))
                big = lab.startswith("nobody")
                ya = np.array(ys, dtype=float)
                over = ya > clip if clip else np.zeros(len(ya), bool)
                ax.scatter(xs[~over], ya[~over], s=18 if big else 9, color=c, alpha=0.95 if big else 0.55, edgecolor=BG, lw=0.4, zorder=3)
                if over.any():
                    ax.scatter(xs[over], np.full(over.sum(), clip), s=16, marker="^", color=c, alpha=0.8, edgecolor=BG, lw=0.4, zorder=3)
                if big and panel == 1 and v == "mx":
                    xs_of.update({r["target"]: (x, y) for r, x, y in zip(rs, xs, ys)})
                if ys:
                    ax.plot([j * 3.8 + g - 0.3, j * 3.8 + g + 0.3], [np.median(ys)] * 2, color=FG, lw=1.6, zorder=4)
            if panel == 1:
                thr = s["nobody_rule"][v]["threshold"]
                ax.plot([j * 3.8 - 0.5, j * 3.8 + 2.5], [thr, thr], color=COLOR[v], lw=1.2, ls="--", zorder=5)
                ax.text(j * 3.8 + 2.55, thr, "nobody\nbelow", color=COLOR[v], fontsize=7.5, va="center")
        top = 0.95 if panel == 0 else 10.0
        for j, v in enumerate(ENGINES):
            ax.text(j * 3.8 + 1, top * 0.945, ("KEYWORD", "MiniLM", "mxbai")[j], ha="center", color=COLOR[v], fontsize=10.5, fontweight="bold")
        if clip:
            ax.text(0.995, 0.995, "▲ above 8.6, drawn at 8.6", transform=ax.transAxes, color=DIM, fontsize=7.5, va="top", ha="right")
        if panel == 1:
            for name, dy in (("Thor", 0.9), ("Magneto", -1.1)):
                if name in xs_of:
                    x, y = xs_of[name]
                    pick = next(a["mx"]["pick"] for a in s["absent"] if a["name"] == name)
                    ax.annotate(f"{name} → {pick}", (x, y), xytext=(x - 1.1, y + dy), ha="right", va="center", color=FG, fontsize=8.3,
                                arrowprops=dict(arrowstyle="-", color=DIM, lw=0.6, shrinkA=3, shrinkB=3))
        ax.set_xticks([j * 3.8 + g for j in range(3) for g in range(3)])
        ax.set_xticklabels([g[0] for g in groups] * 3, fontsize=8.2)
        ax.set_ylim(0, top)
        ax.set_xlim(-0.6, 3 * 3.8 + 0.3)
        style(ax, title, None, ylabel)
        ax.grid(axis="x", visible=False)
    fig.savefig(FIGS / "week7_nobody.png", bbox_inches="tight")
    plt.close(fig)


def print_summary(s):
    print(f"pages over 256 word pieces: {s['meta']['pages_over_256_pieces']}, over 512: {s['meta']['pages_over_512_pieces']} of {s['meta']['n_pages']} "
          f"(median {s['meta']['median_pieces']}); passages {s['meta']['n_passages_180']} at 180, {s['meta']['n_passages_60']} at 60")
    for st in ("page", "other"):
        print(f"\n{st} words (n={s['accuracy'][st]['kw']['n']})")
        for v in VARIANTS:
            a = s["accuracy"][st][v]
            print(f"  {LABEL[v]:32s} top1 {a['top1']:3d}  top3 {a['top3']:3d}  top10 {a['top10']:3d}  median {a['median']:6.1f}  MRR {a['mrr']:.3f}")
    print("\npaired:", json.dumps(s["paired"]))
    print("head to head:", json.dumps(s["head_to_head"]))
    km = s["keyword_misses_other"]
    print(f"keyword misses on other words: {km['n']}; carrier in the pick's own name in {km['carrier_in_pick_name']}; "
          f">= half in {km['carrier_over_half']} (median share {km['median_carrier_share']}); page words: {s['keyword_misses_page']}")
    print("  picks:", km["picks"])
    print("unknown words:", s["unknown_words"])
    print("keyword on other words by leaked rare page words:", {k: (v["n"], v["top1"], v["top3"]) for k, v in s["keyword_by_leak"].items()})
    print("right answers on other words, clue words in the matched passage:", s["meaning_wins_overlap"])
    print("misses on other words, read by hand:", s["meaning_misses_read"], "| missed by both:", s["meaning_misses_both"])
    for v in ENGINES:
        c = s["confidence"][v]
        print(f"confidence {v}: right median {c['right_median']} (n={c['n_right']}), wrong median {c['wrong_median']}, absent {c['absent']}, "
              f"right at or below the absent max: {c['right_at_or_below_absent_max']} (other words: {c['other_right_at_or_below_absent_max']} of {c['other_right']})")
    for v in ENGINES:
        print(f"nobody rule {v}:", s["nobody_rule"][v])
    for a in s["absent"]:
        print(f"  {a['name']:16s} " + "  ".join(f"{e} -> {a[e]['pick']} ({a[e]['score']:.3f}, {RULE[e]} {a[e][RULE[e]]:.2f})" for e in ENGINES))


if __name__ == "__main__":
    main()
