"""
Week 6 analysis: names or meaning?

The course's last opener for exercise 6.11 asks: TF-IDF put Storm next to the Human Torch because of a
surname; find a way to tell name-driven matches from story-driven ones across the whole corpus, and show
what the lookalikes look like once the names are handled. This script is that way, and everything the
week 6 post quotes.

  I.   Which words are names (capitalised mid-sentence, or in a page title), and what that costs.
  II.  Network agreement of ten nearest neighbours under six representations, and why names carry
       half of it: most links are written as the other page's name.
  III. The split: cos(a, b) = names + habit words + other words, exactly, for every lookalike pair, and the
       verdict (NAME / HABIT / STORY) that follows. Tested against 45 pairs read by hand.
  IV.  What the names were hiding: lookalikes with names and habit words removed, and the pairs that are
       close in text and far in the network.

Outputs
  assets/figures/week6_names.png         agreement by representation, and every pair with and without names
  assets/figures/week6_split.png         fifteen pairs, each cosine split into its three parts
  data/stats_week6.json                  every number in the post
  data/week6_pair_audit.tsv              45 pairs read by hand (kept; read back and compared with the rule)

Run:  python scripts/analyse_week6.py        (about a minute)
"""

import collections
import csv
import json
import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import networkx as nx                               # noqa: E402
import numpy as np                                  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import week5_text as W                              # noqa: E402
import week6_text as T                              # noqa: E402
from analyse import (BG, CYAN, DIM, FG, GOLD, GRID, PURPLE, pretty, style)   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGS = ROOT / "assets" / "figures"
DATA = ROOT / "data"
K = 10                       # neighbours per page, as in the lookalikes explorable
KIND_COLOR = {"name": GOLD, "habit": PURPLE, "story": CYAN}


def pid(prefix, ids):
    hits = [n for n in ids if n == prefix or n.startswith(prefix)]
    assert hits, prefix
    exact = [n for n in hits if n == prefix]
    return exact[0] if exact else hits[0]


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    m = T.build_model()
    c, G, ids, ix, N = m.c, m.G, m.ids, m.ix, m.N
    names, isname, isstop, ishabit = m.names, m.isname, m.isstop, m.ishabit
    C, X, idf, df, Un, length = m.C, m.X, m.idf, m.df, m.Un, m.length
    dist, U, Q, comms, comm_of, base_words = m.dist, m.U, m.Q, m.comms, m.comm_of, m.base_words
    stats = {}

    def linked(a, b):
        return U.has_edge(ids[a], ids[b])

    def d(a, b):
        return dist.get(ids[a], {}).get(ids[b])

    def same_comm(a, b):
        ca, cb = comm_of.get(ids[a]), comm_of.get(ids[b])
        return None if ca is None or cb is None else ca == cb

    # ---------------------------------------------------------------- I. which words are names
    reasons = collections.Counter(names.values())
    course = T.course_rule_names(c)
    only_course = sorted(course - set(names), key=lambda w: -c.seen[w])
    total_tokens = int(C.sum())
    stats["names"] = {
        "pages": N, "tokens": total_tokens, "types": len(c.vocab),
        "name_types": len(names), "by_reason": dict(reasons),
        "name_token_share": float(C[:, isname].sum() / total_tokens),
        "course_rule_types": len(course), "course_rule_only": len(only_course),
        "course_rule_only_examples": only_course[:12],
        "title_only": sorted(w for w, r in names.items() if r == "title"),
        "title_words_left_alone": sorted(w for w in base_words if w in c.vi and w not in names and w not in W.STOPWORDS),
    }

    # ---------------------------------------------------------------- II. agreement with the network
    Cs = C.copy()
    Cs[:, isstop] = 0
    reps = {
        "raw counts": T.cosine_matrix(C),
        "counts, stopwords out": T.cosine_matrix(Cs),
        "TF-IDF": T.cosine_matrix(X),
        "TF-IDF, names out": T.cosine_matrix(X * ~isname),
        "TF-IDF, names and stopwords out": m.S_nostop,
        "TF-IDF, names and habit words out": m.S_clean,
    }

    def overlap(S):
        top = T.topk(S, K)
        return float(np.mean([sum(linked(i, j) for j in top[i]) for i in range(N)]))

    rng = np.random.default_rng(6)
    rand = []
    for _ in range(20):
        R = rng.random((N, N))
        np.fill_diagonal(R, -1)
        rand.append(overlap(R))
    agree = {k: overlap(S) for k, S in reps.items()}
    agree["ten random pages"] = float(np.mean(rand))
    stats["agreement"] = agree

    cn = W.character_names(G)
    hit = tot = 0
    for a, b in G.edges():
        if not cn.get(b):
            continue
        tot += 1
        hit += any(x in c.pages[a] for x in cn[b])
    stats["anchor"] = {"links": G.number_of_edges(), "links_with_usable_name": tot, "named_on_source_page": hit,
                       "share": hit / tot}

    # ---------------------------------------------------------------- III. the split
    S_kept, S_clean = m.S_kept, m.S_clean
    top_kept, top_clean = T.topk(S_kept, K), T.topk(S_clean, K)

    pair_set = T.pair_set

    def analyse_pairs(pairs):
        out = []
        for a, b in pairs:
            dec = T.decompose(Un, a, b, isname, ishabit)
            contrib = dec.pop("contrib")
            nm = np.where(isname, contrib, -1)
            t = int(np.argmax(nm))
            dec["top_name"] = c.vocab[t] if nm[t] > 0 else None
            dec["name_kind"] = T.name_kind(dec["top_name"], a, b, ids, base_words) if dec["top_name"] else None
            dec.update(a=a, b=b, linked=linked(a, b), dist=d(a, b), same=same_comm(a, b))
            out.append(dec)
        return out

    def summarise(rows):
        n = len(rows)
        out = {"pairs": n}
        for k in ("name", "habit", "story"):
            sel = [r for r in rows if r["klass"] == k]
            ds = [r["dist"] for r in sel if r["dist"] is not None]
            sc = [r["same"] for r in sel if r["same"] is not None]
            out[k] = {"n": len(sel), "share": len(sel) / n,
                      "linked": float(np.mean([r["linked"] for r in sel])) if sel else None,
                      "same_community": float(np.mean(sc)) if sc else None,
                      "mean_distance": float(np.mean(ds)) if ds else None,
                      "median_cos": float(np.median([r["cos"] for r in sel])) if sel else None}
        nm = [r for r in rows if r["klass"] == "name" and r["name_kind"]]
        out["name_kinds"] = dict(collections.Counter(r["name_kind"] for r in nm))
        return out

    rows_kept = analyse_pairs(pair_set(top_kept))
    rows_clean = analyse_pairs(pair_set(top_clean))
    rows_both = {(r["a"], r["b"]): r for r in rows_kept + rows_clean}
    stats["split_kept_lists"] = summarise(rows_kept)
    stats["split_clean_lists"] = summarise(rows_clean)
    shares = np.array([r["share"] for r in rows_kept])
    stats["split_kept_lists"]["share_quantiles"] = dict(zip(
        ("p10", "p25", "p50", "p75", "p90"), map(float, np.quantile(shares, [.1, .25, .5, .75, .9]))))
    order = np.argsort(-S_kept[np.triu_indices(N, 1)])[:500]
    iu = np.triu_indices(N, 1)
    top500 = [T.decompose(Un, iu[0][o], iu[1][o], isname, ishabit) for o in order]
    stats["top500"] = {"name": sum(r["klass"] == "name" for r in top500),
                       "median_share": float(np.median([r["share"] for r in top500]))}

    # names out only (stopwords kept): what does the list look like then?
    S_nonames = reps["TF-IDF, names out"]
    top_nn = T.topk(S_nonames, K)
    Xn = X * ~isname
    Un_nn = T.unit(Xn)
    pron = {"she", "her", "he", "his", "hers", "him", "herself", "himself"}
    topterm = collections.Counter()
    for a, b in pair_set(top_nn):
        t = int(np.argmax(Un_nn[a] * Un_nn[b]))
        topterm[c.vocab[t]] += 1
    n_nn = sum(topterm.values())
    stats["names_out_only"] = {"pairs": n_nn, "top_term_is_pronoun": sum(v for w, v in topterm.items() if w in pron),
                               "most_common_top_terms": topterm.most_common(8)}

    # every pair of pages, not just the lookalike lists: the three parts for all 45,753 at once
    Mn, Mh = Un * isname, Un * (ishabit & ~isname)
    Mo = Un * ~(isname | ishabit)
    Pn, Ph, Po = Mn @ Mn.T, Mh @ Mh.T, Mo @ Mo.T
    iu_all = np.triu_indices(N, 1)
    pn, ph, po = Pn[iu_all], Ph[iu_all], Po[iu_all]
    tot_all = pn + ph + po
    kl = np.where(pn >= 0.5 * tot_all, "name", np.where(ph >= po, "habit", "story"))
    buckets = {}
    for lo in (0.0, 0.02, 0.05, 0.1, 0.2):
        sel = tot_all >= lo
        buckets[f">={lo}"] = {"pairs": int(sel.sum()), **{k: float((kl[sel] == k).mean()) for k in ("name", "habit", "story")}}
    stats["all_pairs"] = {"pairs": int(len(tot_all)), "by_min_cosine": buckets,
                          "median_cosine": float(np.median(tot_all)), "p99_cosine": float(np.quantile(tot_all, .99))}

    # ---------------------------------------------------------------- audit against hand labels
    audit = []
    with open(DATA / "week6_pair_audit.tsv", encoding="utf-8") as f:
        for r in csv.DictReader((ln for ln in f if not ln.startswith("#")), delimiter="\t"):
            a, b = ix[r["page_a"]], ix[r["page_b"]]
            dec = T.decompose(Un, a, b, isname, ishabit)
            audit.append({"a": r["page_a"], "b": r["page_b"], "label": r["label"], "rule": dec["klass"],
                          "share": dec["share"], "note": r["note"]})
    want = {"N": "name", "S": "story", "H": "habit"}
    clear = [r for r in audit if r["label"] in want]
    right = [r for r in clear if want[r["label"]] == r["rule"]]
    wrong = [r for r in clear if want[r["label"]] != r["rule"]]
    stats["audit"] = {"pairs": len(audit), "mixed": len(audit) - len(clear), "clear": len(clear), "agree": len(right),
                      "by_label": {k: sum(r["label"] == k for r in audit) for k in "NSHM"},
                      "disagree": [{"a": W.base_title(r["a"]), "b": W.base_title(r["b"]), "label": r["label"],
                                    "rule": r["rule"], "share": round(r["share"], 2), "note": r["note"]} for r in wrong],
                      "near_boundary": sum(abs(r["share"] - .5) < .03 for r in wrong)}

    # ---------------------------------------------------------------- spotlights and neighbour lists
    def spot(x, y):
        a, b = ix[pid(x, ids)], ix[pid(y, ids)]
        r = T.decompose(Un, a, b, isname, ishabit)
        contrib = r.pop("contrib")
        terms = [(c.vocab[t], bool(isname[t]), float(contrib[t] / r["cos"])) for t in np.argsort(-contrib)[:5]]
        clean = float(S_clean[a, b])
        return {"a": pretty(ids[a]), "b": pretty(ids[b]), "cos": r["cos"], "name": r["name"], "habit": r["habit"],
                "other": r["other"], "share": r["share"], "klass": r["klass"], "cos_clean": clean,
                "linked": linked(a, b), "dist": d(a, b), "terms": terms}

    pairs_fig = [
        ("Emma_Frost", "Jack_Frost_(Marvel"), ("Anne_Weying", "Turbo_(comics)"), ("Storm_(Marvel", "Human_Torch"),
        ("Shear", "Toxyn"), ("Jessica_Jones", "Spider-Woman_(Jessica_Drew"), ("NFL_SuperPro", "Spider-Woman_(Jessica_Drew"),
        ("Eddie_Brock", "Venom_(character"), ("Mania", "Venom_(character"), ("Blade_(", "Morbius"),
        ("Iron_Fist", "Lei_Kung"), ("She-Hulk", "Toxyn"), ("Betsy_Braddock", "Emma_Frost"),
        ("Garrison_Kane", "Raza_Longknife"), ("Doom_2099", "Ogre"), ("Anole", "Shatterstar"),
    ]
    spots = [spot(x, y) for x, y in pairs_fig]
    stats["spotlights"] = spots

    def neighbours(name, S, k=5):
        a = ix[pid(name, ids)]
        return [{"page": pretty(ids[j]), "cos": float(S[a, j]), "linked": linked(a, j)}
                for j in np.argsort(-S[a])[:k]]

    lists = {}
    for who in ("Storm_(Marvel", "Wolverine_(character)", "Venom_(character", "Doctor_Strange", "Black_Widow_(Natasha"):
        lists[pretty(pid(who, ids))] = {"names kept": neighbours(who, S_kept),
                                        "names out": neighbours(who, reps["TF-IDF, names out"]),
                                        "names and habit words out": neighbours(who, S_clean)}
    stats["neighbour_lists"] = lists

    # ---------------------------------------------------------------- IV. story twins far apart
    long_enough = length >= 500
    twins = []
    for a, b in zip(*np.triu_indices(N, 1)):
        if not (long_enough[a] and long_enough[b]) or linked(a, b):
            continue
        dd = d(a, b)
        if dd is not None and dd < 3:
            continue
        twins.append((float(S_clean[a, b]), a, b, dd))
    twins.sort(reverse=True)
    Uc = T.unit(X * ~isname * ~ishabit)
    twin_rows = []
    for cos_c, a, b, dd in twins[:30]:
        contrib = Uc[a] * Uc[b]
        t = [c.vocab[i] for i in np.argsort(-contrib)[:6]]
        twin_rows.append({"a": pretty(ids[a]), "b": pretty(ids[b]), "cos_clean": cos_c, "cos_kept": float(S_kept[a, b]),
                          "dist": dd, "same_community": same_comm(a, b), "terms": t})
    stats["story_twins"] = twin_rows
    stats["twins_universe"] = {"pairs_long_enough": int(long_enough.sum() * (long_enough.sum() - 1) // 2),
                               "unlinked_and_far": len(twins)}

    n_all = N * (N - 1) // 2
    sizes = [len(cm) for cm in comms]
    g = sum(sizes)
    stats["louvain"] = {"modularity": Q, "communities": len(comms), "sizes": sizes, "networkx": nx.__version__}
    stats["baseline"] = {"link_density": U.number_of_edges() / n_all, "undirected_links": U.number_of_edges(),
                         "pairs": n_all, "same_community_chance": sum(x * x for x in sizes) / (g * g)}

    # ---------------------------------------------------------------- figure 1
    fig = plt.figure(figsize=(13.2, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.2], wspace=0.34)
    ax = fig.add_subplot(gs[0])
    labels = ["ten random pages", "raw counts", "counts, stopwords out", "TF-IDF, names and habit words out",
              "TF-IDF, names and stopwords out", "TF-IDF, names out", "TF-IDF"]
    vals = [agree[k] for k in labels]
    colors = [GRID, DIM, DIM, CYAN, CYAN, CYAN, GOLD]
    bars = ax.barh(range(len(labels)), vals, color=colors, height=0.66)
    for i, v in enumerate(vals):
        ax.text(v + 0.06, i, f"{v:.2f}", va="center", color=FG, fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0, 4.7)
    style(ax, "Agreement with the network", "of the ten nearest neighbours, how many are linked to the page")
    ax.grid(axis="y", alpha=0)

    ax = fig.add_subplot(gs[1])
    pts = rows_kept
    FLOOR = 0.004                    # a cosine of exactly zero cannot sit on a log axis
    xs = np.maximum([r["cos"] for r in pts], FLOOR)
    ys = np.maximum([float(S_clean[r["a"], r["b"]]) for r in pts], FLOOR)
    klass = np.array([r["klass"] for r in pts])
    for k in ("name", "habit", "story"):
        sel = klass == k
        ax.scatter(xs[sel], ys[sel], s=10, c=KIND_COLOR[k], alpha=0.5 if k == "name" else 0.8, lw=0,
                   label=f"{k.upper()}  ({int(sel.sum())})")
    ax.plot([FLOOR, 1], [FLOOR, 1], color=DIM, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(FLOOR, 1)
    ax.set_ylim(FLOOR, 1)
    marks = {"Emma Frost | Jack Frost": (6, -30, "left"), "Storm | Human Torch": (10, -34, "left"),
             "Eddie Brock | Venom": (-46, -22, "right"), "Shear | Toxyn": (20, -48, "left"),
             "Garrison Kane | Raza Longknife": (24, 30, "left"), "Anole | Shatterstar": (24, 40, "left")}
    for sp in spots:
        key = f"{sp['a']} | {sp['b']}"
        if key in marks:
            dx, dy, ha = marks[key]
            ax.annotate(key, (sp["cos"], sp["cos_clean"]), xytext=(dx, dy), textcoords="offset points", color=FG,
                        fontsize=8.5, ha=ha, arrowprops=dict(arrowstyle="-", color=DIM, lw=0.6))
    ax.legend(loc="upper left", fontsize=9, markerscale=2.2)
    style(ax, f"Every lookalike pair, with and without its names ({len(pts)} pairs)",
          "cosine, TF-IDF with names kept (log scale)", "cosine with names and habit words removed (log scale)")
    fig.savefig(FIGS / "week6_names.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- figure 2
    fig, ax = plt.subplots(figsize=(12.4, 6.6))
    rows = sorted(spots, key=lambda s: -s["share"])
    for i, s in enumerate(rows):
        tot_ = s["name"] + s["habit"] + s["other"]
        left = 0
        for part, col in (("name", GOLD), ("habit", PURPLE), ("other", CYAN)):
            w = s[part] / tot_
            ax.barh(i, w, left=left, color=col, height=0.7)
            if w > 0.1:
                ax.text(left + w / 2, i, f"{w * 100:.0f}", ha="center", va="center", color=BG, fontsize=8.5, fontweight="bold")
            left += w
        ax.text(1.015, i, f"cos {s['cos']:.2f}  →  {s['cos_clean']:.2f}", va="center", color=DIM, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{s['a']}  |  {s['b']}" for s in rows], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.axvline(0.5, color=FG, lw=0.7, ls=":")
    style(ax, "What each cosine is made of", "percent of the cosine from  names (gold),  habit words (purple),  other words (cyan)")
    ax.grid(axis="y", alpha=0)
    ax.text(1.015, -0.9, "cosine kept \u2192 names and\nhabit words removed", color=DIM, fontsize=8.5, va="bottom")
    fig.savefig(FIGS / "week6_split.png", bbox_inches="tight")
    plt.close(fig)

    (DATA / "stats_week6.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print_summary(stats)


def print_summary(s):
    print("names:", s["names"]["name_types"], "types,", f"{s['names']['name_token_share']:.1%} of tokens,", s["names"]["by_reason"])
    print("agreement:", {k: round(v, 2) for k, v in s["agreement"].items()})
    print("anchor:", s["anchor"])
    for k in ("split_kept_lists", "split_clean_lists"):
        x = s[k]
        print(k, x["pairs"], {c_: (x[c_]["n"], round(x[c_]["linked"] or 0, 2)) for c_ in ("name", "habit", "story")}, x["name_kinds"])
    print("top500", s["top500"], "names-out-only", s["names_out_only"])
    print("all pairs", s["all_pairs"]["by_min_cosine"])
    print("audit", {k: v for k, v in s["audit"].items() if k != "disagree"})
    print("twins universe", s["twins_universe"])
    print("wrote data/stats_week6.json, week6_names.png, week6_split.png")


if __name__ == "__main__":
    main()
