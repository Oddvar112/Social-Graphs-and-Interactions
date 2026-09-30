"""
Week 5 analysis: the Marvel network gets its text.

Everything the week 5 post quotes, in the order the post asks it:
  I.   Count first: tokens, types, hapax under three pipelines; Zipf on the short
       descriptions, the full pages and Moby Dick; how much the top words cover.
  II.  Does fame buy words? Page length against in-degree, the outliers, and Heaps'
       law with the pages added most-linked first and least-linked first.
  III. Every link is a sentence: the sentence on A's page that mentions B, parsed,
       and labelled foe / ally / family (scripts/week5_text.RelationLabeller), then
       tested against Louvain communities with a label shuffle.
  IV.  Wikipedia copies itself: pages that share long runs of identical text.
  V.   A search engine in 20 lines, and who it cannot find because they are not there.
  VI.  Lookalikes: cosine neighbours against network neighbours; the weirdest pages.
  VII. One trigram generator per community: a fake sentence from each.

Outputs
  assets/figures/week5_*.png
  data/stats_week5.json          every number in the post
  data/week5_label_audit.tsv     60 labelled sentences we read by hand (written once;
                                 the verdict column is filled in by a human and kept)

Run:  python scripts/analyse_week5.py         (three to four minutes; the first run
                                              tokenizes the pages and caches them)
"""

import collections
import csv
import itertools
import json
import pathlib
import random
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.collections import LineCollection   # noqa: E402
import networkx as nx                               # noqa: E402
import numpy as np                                  # noqa: E402
from scipy import stats as sps                      # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import week5_text as W                              # noqa: E402
from analyse import (BG, CYAN, DIM, FG, GOLD, GREEN, GRID, HALO, PURPLE, RED,   # noqa: E402
                     load_graph, place_labels, pretty, style)

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGS = ROOT / "assets" / "figures"
DATA = ROOT / "data"
ORANGE = "#ff9f43"
COMM_COLOURS = [CYAN, GOLD, GREEN, PURPLE, RED, ORANGE, "#ff7ab6"]

SEARCH_QUERIES = [
    ("norse god of thunder", "Thor"), ("web slinging teenager from queens", "Spider-Man"),
    ("king of wakanda", "Black_Panther_(character)"), ("sorcerer supreme", "Doctor_Strange"),
    ("mutant with adamantium claws and a healing factor", "Wolverine_(character)"),
    ("green monster gamma radiation", "Hulk"), ("blind lawyer from hell's kitchen", "Daredevil"),
    ("russian spy avenger", "Black_Widow_(Natasha_Romanova)"), ("master archer avenger", "Hawkeye_(comics)"),
    ("super soldier with a shield", "Captain_America"), ("billionaire in a suit of armor", "Iron_Man"),
    ("talking raccoon", "Rocket_Raccoon"), ("healthcare robot big hero 6", "Baymax"),
    ("merc with a mouth", "Deadpool"), ("symbiote", "Venom_(character)"),
    ("queen of the inhumans", "Medusa_(comics)"), ("vampire hunter", "Blade_(character)"),
    ("canadian superhero team", "Guardian_(Marvel_Comics)"), ("sister of thor", "Angela_(character)"),
    ("the first mutant", "Apocalypse_(character)"),
]
ABSENT = ["Captain America", "Iron Man", "Thor", "Magneto", "Apocalypse", "Thanos", "Daredevil", "Namor", "Vision", "Mary Jane",
          "Steve Rogers", "Ultron", "Mephisto", "Doctor Doom", "Nick Fury", "Kingpin", "Norman Osborn", "Beast",
          "Galactus", "Tony Stark", "Loki", "Punisher", "Dracula", "Ant-Man", "Kang", "Green Goblin", "Sabretooth",
          "Red Skull", "Wasp", "Silver Surfer", "Bruce Banner", "Rogue", "Professor X", "Doctor Octopus",
          "Reed Richards", "Colossus", "Charles Xavier", "Mister Fantastic", "Falcon", "Nightcrawler", "Gambit",
          "Juggernaut", "Mister Sinister", "Invisible Woman"]
ROSTER_HUBS = ["Spider-Man", "Hulk", "Wolverine", "Deadpool", "Doctor Strange", "Black Widow"]
COMMUNITY_NAMES = {  # by their best-known members, filled in after Louvain runs (see name_communities)
}


def name_communities(comms, G):
    """A human name per community, from who sits there."""
    names = []
    for c in comms:
        top = [pretty(n) for n in sorted(c, key=lambda n: -G.degree(n))[:3]]
        names.append(" / ".join(top[:2]))
    return names


# ----------------------------------------------------------------------------- I. counts
def zipf_slope(counter, rmax=None):
    f = np.array(sorted(counter.values(), reverse=True), float)
    r = np.arange(1, len(f) + 1)
    if rmax:
        f, r = f[:rmax], r[:rmax]
    s, _ = np.polyfit(np.log(r), np.log(f), 1)
    return -float(s)


def counts_section(pages, docs, G, stats):
    raw = [w for d in docs.values() for w in W.raw_tokens(d)]
    lower = [w for d in docs.values() for w in W.words(d)]
    nostop = [w for d in docs.values() for w in W.words(d, stop=True)]
    out = {}
    for key, toks in (("raw", raw), ("lower", lower), ("nostop", nostop)):
        c = collections.Counter(toks)
        hap = sum(v == 1 for v in c.values())
        out[key] = {"tokens": len(toks), "types": len(c), "hapax": hap, "hapax_share": round(hap / len(c), 3),
                    "top": c.most_common(12)}
    cL = collections.Counter(lower)
    f = np.array(sorted(cL.values(), reverse=True))
    out["coverage"] = {"top10": round(float(f[:10].sum() / f.sum()), 3), "top100": round(float(f[:100].sum() / f.sum()), 3),
                       "top1000": round(float(f[:1000].sum() / f.sum()), 3)}
    out["power_count"] = cL["power"]
    out["chars"] = sum(len(p) for p in pages.values())
    desc = [w for n in G for w in re.findall(r"[a-z]+", G.nodes[n].get("description", "").lower())]
    cD = collections.Counter(desc)
    out["descriptions"] = {"tokens": len(desc), "types": len(cD), "top": cD.most_common(8),
                           "zipf_s_all": round(zipf_slope(cD), 2), "zipf_s_top100": round(zipf_slope(cD, 100), 2)}
    out["zipf"] = {"full_top100": round(zipf_slope(cL, 100), 2), "full_top1000": round(zipf_slope(cL, 1000), 2),
                   "full_all": round(zipf_slope(cL), 2)}
    moby = None
    try:
        import nltk
        try:
            nltk.data.find("corpora/gutenberg")
        except LookupError:
            nltk.download("gutenberg", quiet=True)
        from nltk.corpus import gutenberg
        md = [w.lower() for w in gutenberg.words("melville-moby_dick.txt") if w.isalpha()]
        cM = collections.Counter(md)
        moby = {"tokens": len(md), "types": len(cM), "hapax_share": round(sum(v == 1 for v in cM.values()) / len(cM), 3),
                "zipf_s_top1000": round(zipf_slope(cM, 1000), 2), "zipf_s_all": round(zipf_slope(cM), 2)}
    except Exception as exc:                       # noqa: BLE001
        print("   (Moby Dick unavailable:", exc, ")")
        cM = None
    out["moby_dick"] = moby
    stats["counts"] = out
    return cL, cD, cM


def fig_zipf(cL, cD, cM):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    ax = axes[0]
    for c, colour, label in ((cL, CYAN, f"full pages, {sum(cL.values()):,} tokens"),
                             (cD, GOLD, f"short descriptions, {sum(cD.values()):,} tokens"),
                             (cM, GREEN, "Moby Dick" if cM else None)):
        if c is None:
            continue
        f = np.array(sorted(c.values(), reverse=True), float)
        r = np.arange(1, len(f) + 1)
        ax.plot(r, f / f.sum(), ".", ms=3.2, color=colour, alpha=0.85, label=label)
    r = np.arange(1, 30000)
    ax.plot(r, 0.06 / r, "--", color=RED, lw=1.4, label="ideal Zipf, s = 1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(5e-7, 0.2)
    style(ax, "Rank against relative frequency, log-log", "frequency rank", "share of all tokens")
    ax.legend(labelcolor=FG, fontsize=8.6, loc="lower left")
    ax = axes[1]
    f = np.array(sorted(cL.values(), reverse=True), float)
    ax.bar(np.arange(1, 41), f[:40], color=CYAN, alpha=0.85)
    for i, (w, n) in enumerate(cL.most_common(8)):
        ax.text(i + 1, n, w, ha="center", va="bottom", fontsize=8.5, color=FG, rotation=0)
    style(ax, "The same counts on linear axes, ranks 1 to 40", "frequency rank", "count in the full pages")
    fig.suptitle("Is Marvel Zipf-like? The short descriptions, the full pages and a novel",
                 color=FG, fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "week5_zipf.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- II. fame, Heaps
def fame_section(docs, G, stats):
    kin = dict(G.in_degree())
    L = {n: len(W.words(docs[n])) for n in docs}
    ids = list(docs)
    x = np.array([kin[n] for n in ids], float)
    y = np.array([L[n] for n in ids], float)
    rho = sps.spearmanr(x, y)
    mask = x > 0
    b, a = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    res = {ids[i]: float(np.log(y[i]) - (a + b * np.log(max(x[i], 1)))) for i in range(len(ids))}
    srt = sorted(res.items(), key=lambda kv: kv[1])
    order_hub = sorted(docs, key=lambda n: -kin[n])
    order_min = order_hub[::-1]

    def heaps(order):
        seen, xs, ys, n = set(), [], [], 0
        for nid in order:
            for w in W.words(docs[nid]):
                n += 1
                seen.add(w)
            xs.append(n)
            ys.append(len(seen))
        return np.array(xs), np.array(ys)

    def new_rate(order):
        seen, rates = set(), []
        for nid in order:
            ws = W.words(docs[nid])
            new = sum(1 for w in set(ws) if w not in seen)
            seen |= set(ws)
            rates.append(1000 * new / max(len(ws), 1))
        return rates

    xh, yh = heaps(order_hub)
    xm, ym = heaps(order_min)
    rh, rm = new_rate(order_hub), new_rate(order_min)
    stats["fame"] = {
        "spearman": round(float(rho.correlation), 3), "spearman_p": float(rho.pvalue),
        "loglog_slope": round(float(b), 3),
        "median_tokens": float(np.median(y)), "min_tokens": int(y.min()), "max_tokens": int(y.max()),
        "spread": round(float(y.max() / y.min()), 1),
        "short_for_fame": [(pretty(n), kin[n], L[n]) for n, _ in srt[:6]],
        "long_for_fame": [(pretty(n), kin[n], L[n]) for n, _ in srt[-6:][::-1]],
        "longest": [(pretty(n), L[n], kin[n]) for n in sorted(L, key=lambda n: -L[n])[:5]],
        "shortest": [(pretty(n), L[n], kin[n]) for n in sorted(L, key=lambda n: L[n])[:5]],
        "heaps": {"beta_hub_first": round(float(np.polyfit(np.log(xh), np.log(yh), 1)[0]), 3),
                  "beta_minor_first": round(float(np.polyfit(np.log(xm), np.log(ym), 1)[0]), 3),
                  "types_at_100k_hub_first": int(yh[np.searchsorted(xh, 100000)]),
                  "types_at_100k_minor_first": int(ym[np.searchsorted(xm, 100000)]),
                  "new_per_1000_last50_hub_first": round(float(np.mean(rh[-50:])), 1),
                  "new_per_1000_last50_minor_first": round(float(np.mean(rm[-50:])), 1),
                  "total_types": int(yh[-1])},
    }
    return L, (xh, yh, xm, ym), (a, b), res


def fig_fame(docs, G, L, heaps, fit, res):
    kin = dict(G.in_degree())
    ids = list(docs)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.3))
    ax = axes[0]
    x = np.array([kin[n] for n in ids], float)
    y = np.array([L[n] for n in ids], float)
    ax.scatter(x, y, s=18 + 0.6 * x, c=np.log1p(x), cmap="plasma", alpha=0.8, edgecolor=BG, lw=0.4, zorder=3)
    a, b = fit
    grid = np.linspace(0, np.log(110), 50)
    ax.plot(np.exp(grid), np.exp(a + b * grid), "--", color=RED, lw=1.5, label=f"words ∝ in-degree^{b:.2f}")
    srt = sorted(res.items(), key=lambda kv: kv[1])
    labels = [(kin[n], L[n], pretty(n), GOLD, 8.4) for n, _ in srt[-5:]] + \
             [(kin[n], L[n], pretty(n), CYAN, 8.4) for n, _ in srt[:3] if kin[n] > 0] + \
             [(kin[n], L[n], pretty(n), FG, 8.6) for n in ["Spider-Man", "Scarlet_Witch"]]
    ax.set_xscale("symlog", linthresh=1.5)
    ax.set_yscale("log")
    ax.set_xlim(-0.6, 140)
    ax.set_ylim(150, 20000)
    style(ax, "Page length against in-degree", "in-degree (how many characters link here)", "words on the page")
    ax.legend(labelcolor=FG, fontsize=8.6, loc="lower right")
    place_labels(ax, labels, fig)
    ax = axes[1]
    xh, yh, xm, ym = heaps
    ax.plot(xh, yh, color=CYAN, lw=2, label="most-linked characters first")
    ax.plot(xm, ym, color=GOLD, lw=2, label="least-linked characters first")
    style(ax, "Heaps' law: vocabulary against corpus size, two orders", "tokens read so far", "distinct types seen")
    ax.legend(labelcolor=FG, fontsize=8.6, loc="lower right")
    ax.text(0.03, 0.95, "the vocabulary never flattens:\nevery page keeps bringing new words",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.6, color=DIM, linespacing=1.6)
    fig.suptitle("Does network fame buy you words, and do minor characters bring new ones?",
                 color=FG, fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "week5_fame_heaps.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- III. relations
def audit(labels, stats):
    """The 60 sentences a human read. Written once with an empty verdict column;
    afterwards the file is the source of truth and this only checks it is still
    about the labels the code produces today."""
    path = DATA / "week5_label_audit.tsv"
    if not path.exists():
        rng = random.Random(2026)
        rows = []
        for lab in ("foe", "ally", "family"):
            es = sorted(e for e, l in labels.items() if l[0] == lab)
            for u, v in rng.sample(es, 20):
                ev = [(s, r) for s, r in labels[(u, v)][1] for k, _, _ in r if k == lab]
                s, rel = ev[0] if ev else labels[(u, v)][1][0]
                rel = [r for r in rel if r[0] == lab] or rel
                rows.append({"label": lab, "source": u, "target": v, "cue": rel[0][1], "pattern": rel[0][2],
                             "sentence": " ".join(s.split())[:400], "verdict": ""})
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print("   wrote", path.relative_to(ROOT), "- fill in the verdict column (yes/no) by reading")
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
    out, stale = {}, 0
    for lab in ("foe", "ally", "family"):
        rs = [r for r in rows if r["label"] == lab]
        for r in rs:
            if labels.get((r["source"], r["target"]), ("?",))[0] != lab:
                stale += 1
        judged = [r for r in rs if r["verdict"].strip().lower() in ("yes", "no")]
        yes = sum(1 for r in judged if r["verdict"].strip().lower() == "yes")
        out[lab] = {"read": len(judged), "right": yes, "precision": round(yes / len(judged), 2) if judged else None}
    out["stale_rows"] = stale
    if stale:
        print(f"   WARNING: {stale} audited rows no longer carry the audited label; re-read them")
    stats["relations"]["audit"] = out


def relations_section(pages, docs, G, stats):
    names = W.character_names(G)
    RL = W.RelationLabeller(G, pages, docs, names)
    labels = RL.label_all()
    UG = G.to_undirected()
    giant = UG.subgraph(max(nx.connected_components(UG), key=len)).copy()
    q, comms, seed = W.louvain_best(giant)
    com = {n: i for i, c in enumerate(comms) for n in c}
    cnames = name_communities(comms, G)
    keys = [(u, v) for (u, v) in labels if u in com and v in com]
    labs = np.array([labels[k][0] for k in keys])
    cross = np.array([com[u] != com[v] for u, v in keys])
    rng = np.random.default_rng(0)
    test = {}
    for k in ("foe", "ally", "family"):
        obs = float(cross[labs == k].mean())
        null = np.array([cross[rng.permutation(labs) == k].mean() for _ in range(3000)])
        test[k] = {"n": int((labs == k).sum()), "cross": round(obs, 3), "null_mean": round(float(null.mean()), 3),
                   "null_sd": round(float(null.std()), 3), "z": round((obs - null.mean()) / null.std(), 1),
                   "p": round((np.sum(null <= obs) + 1) / (len(null) + 1), 4)}
    agree = collections.Counter()
    frenemies, mutual = [], collections.defaultdict(list)
    for (u, v), l in labels.items():
        if (v, u) in labels and u < v and l[0] in ("foe", "ally", "family") and labels[(v, u)][0] in ("foe", "ally", "family"):
            pair = tuple(sorted((l[0], labels[(v, u)][0])))
            agree[pair] += 1
            if l[0] == labels[(v, u)][0]:
                mutual[l[0]].append((pretty(u), pretty(v)))
            elif pair == ("ally", "foe"):
                frenemies.append((pretty(u), l[0], pretty(v), labels[(v, u)][0]))
    incoming = {}
    for lab in ("foe", "ally", "family"):
        c = collections.Counter(v for (u, v), l in labels.items() if l[0] == lab)
        incoming[lab] = [(pretty(n), k) for n, k in c.most_common(8)]
    counts = collections.Counter(l[0] for l in labels.values())
    stats["relations"] = {
        "counts": dict(counts), "edges": G.number_of_edges(),
        "with_sentence": int(sum(v for k, v in counts.items() if k not in ("none", "ambiguous"))),
        "louvain": {"q": round(q, 3), "k": len(comms), "seed": seed, "sizes": [len(c) for c in comms],
                    "names": cnames, "members_top": [[pretty(n) for n in sorted(c, key=lambda n: -G.degree(n))[:6]] for c in comms]},
        "cross_all": round(float(cross.mean()), 3),
        "test": test, "mutual_agreement": {f"{a}/{b}": n for (a, b), n in agree.most_common()},
        "mutual": {k: v[:12] for k, v in mutual.items()}, "frenemies": frenemies, "incoming": incoming,
    }
    audit(labels, stats)
    return labels, comms, com, giant


def fig_relations(G, giant, labels, comms, com, stats):
    fig = plt.figure(figsize=(13.6, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(BG)
    pos = nx.spring_layout(giant, k=0.55, iterations=500, seed=42)
    colour = {"foe": RED, "ally": GREEN, "family": GOLD}
    segs = collections.defaultdict(list)
    for (u, v), l in labels.items():
        if u in pos and v in pos:
            segs[l[0] if l[0] in colour else "other"].append((pos[u], pos[v]))
    ax.add_collection(LineCollection(segs["other"], colors="#3b466e", linewidths=0.25, alpha=0.22, zorder=1))
    for lab, lw, alpha in (("ally", 1.0, 0.55), ("foe", 1.1, 0.7), ("family", 1.5, 0.85)):
        ax.add_collection(LineCollection(segs[lab], colors=colour[lab], linewidths=lw, alpha=alpha, zorder=2))
    deg = dict(giant.degree())
    order = sorted(giant.nodes(), key=lambda n: deg[n])
    ax.scatter([pos[n][0] for n in order], [pos[n][1] for n in order], s=[10 + 3.2 * deg[n] for n in order],
               c=[COMM_COLOURS[com[n] % len(COMM_COLOURS)] for n in order], linewidths=0.5, edgecolors=BG,
               zorder=3, alpha=0.92)
    top = sorted(giant.nodes(), key=lambda n: -deg[n])[:14]
    place_labels(ax, [(pos[n][0], pos[n][1], pretty(n), FG, 8.4) for n in top], fig)
    px = np.array([p[0] for p in pos.values()])
    py = np.array([p[1] for p in pos.values()])
    ax.set_xlim(np.percentile(px, 0.6) - 0.05, np.percentile(px, 99.4) + 0.05)
    ax.set_ylim(np.percentile(py, 0.6) - 0.05, np.percentile(py, 99.4) + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    t = stats["relations"]["test"]
    ax.set_title(f"Node colour = Louvain community (7 of them). Red = foe ({t['foe']['n']}), green = ally ({t['ally']['n']}),\n"
                 f"gold = family ({t['family']['n']}); faint grey = the links whose sentence says nothing", color=FG,
                 loc="left", fontsize=10, pad=10)
    ax2 = fig.add_subplot(gs[0, 1])
    labs = ["foe", "ally", "family"]
    ys = np.arange(3)
    for i, lab in enumerate(labs):
        r = t[lab]
        ax2.barh(i, r["cross"], color=colour[lab], alpha=0.85, height=0.55)
        ax2.errorbar(r["null_mean"], i, xerr=2 * r["null_sd"], fmt="o", color=FG, ms=5, capsize=6, lw=1.4, zorder=5)
        ax2.text(r["cross"] + 0.012, i - 0.02, f"{100 * r['cross']:.0f}%  (z = {r['z']:+.1f})", va="center",
                 fontsize=9.5, color=FG, fontweight="bold")
    ax2.axvline(stats["relations"]["cross_all"], color=DIM, ls="--", lw=1.2)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([f"{lab} links\n(n = {t[lab]['n']})" for lab in labs], fontsize=10)
    ax2.set_xlim(0, 0.62)
    ax2.set_ylim(2.6, -0.7)
    ax2.text(stats["relations"]["cross_all"] + 0.008, -0.5, f"all links: {100 * stats['relations']['cross_all']:.0f}%",
             color=DIM, fontsize=8.6, va="center")
    style(ax2, "Share of each kind of link that\ncrosses a community line", "share crossing communities", None)
    ax2.text(0.02, 0.03, "white dot = shuffled labels, mean ± 2 sd", transform=ax2.transAxes, fontsize=8.6, color=DIM)
    fig.suptitle("Every link is a sentence. Foes cross communities like any link; family stays home.",
                 color=FG, fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.99)
    fig.savefig(FIGS / "week5_relations.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- IV. copying
def copying_section(pages, docs, G, com, stats, n=8):
    seqs = {nid: [(t[W.LOWER], t[W.IDX]) for t in docs[nid]["tokens"] if t[W.IS_ALPHA]] for nid in docs}
    where = collections.defaultdict(set)
    for nid, s in seqs.items():
        ws = [w for w, _ in s]
        for i in range(len(ws) - n + 1):
            where[tuple(ws[i:i + n])].add(nid)
    npages = collections.Counter(len(v) for v in where.values())
    boiler = sorted((g for g, v in where.items() if len(v) >= 30), key=lambda g: -len(where[g]))
    pair = collections.Counter()
    for g, v in where.items():
        if 2 <= len(v) <= 4:
            for a, b in itertools.combinations(sorted(v), 2):
                pair[(a, b)] += 1

    def longest_run(a, b):
        wa = [w for w, _ in seqs[a]]
        wb = [w for w, _ in seqs[b]]
        sb = collections.defaultdict(list)
        for i in range(len(wb) - n + 1):
            sb[tuple(wb[i:i + n])].append(i)
        best, i = (0, 0), 0
        while i < len(wa) - n + 1:
            g = tuple(wa[i:i + n])
            if g in sb:
                j, k = sb[g][0], n
                while i + k < len(wa) and j + k < len(wb) and wa[i + k] == wb[j + k]:
                    k += 1
                if k > best[0]:
                    best = (k, i)
                i += k - n + 1
            else:
                i += 1
        return best

    top = []
    for (a, b), c in pair.most_common(20):
        k, i = longest_run(a, b)
        off = seqs[a][i][1]
        top.append({"a": pretty(a), "b": pretty(b), "shared": c, "run": k,
                    "snippet": " ".join(pages[a][off:off + 170].split()),
                    "same_community": bool(com.get(a) == com.get(b)) if a in com and b in com else None})
    H = nx.Graph()
    for (a, b), c in pair.items():
        if c >= 5:
            H.add_edge(a, b, weight=c)
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    strong = [(u, v) for u, v, d in H.edges(data=True) if d["weight"] >= 15]
    stats["copying"] = {
        "pairs_15plus": len(strong), "pages_15plus": len({x for e in strong for x in e}),
        "ngram": n, "distinct_ngrams": len(where), "in_one_page": npages[1], "in_two_pages": npages[2],
        "boilerplate_30plus": len(boiler), "boilerplate_examples": [" ".join(g) for g in boiler[:6]],
        "most_shared": {" ".join(boiler[0]): len(where[boiler[0]])} if boiler else {},
        "pairs_any": len(pair), "pairs_5plus": sum(1 for v in pair.values() if v >= 5),
        "pairs_20plus": sum(1 for v in pair.values() if v >= 20),
        "top_pairs": top, "network_nodes": H.number_of_nodes(), "network_edges": H.number_of_edges(),
        "clusters": [[pretty(x) for x in sorted(c, key=lambda x: -H.degree(x))] for c in comps[:8]],
    }
    return H


def fig_copying(H, G, com, min_shared=15):
    """The strong copies as a gallery: every connected component laid out on its
    own, then packed on shelves, biggest first, so the pairs read as pairs."""
    import math
    H = H.edge_subgraph([(u, v) for u, v, d in H.edges(data=True) if d["weight"] >= min_shared]).copy()
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    cells = []
    for c in comps:
        sub = H.subgraph(c)
        side = 1.0 + 0.62 * math.sqrt(len(c))
        if len(c) == 2:
            a, b = sorted(c)
            p = {a: (-0.5, 0.0), b: (0.5, 0.0)}
        else:
            p = nx.spring_layout(sub, k=0.9, iterations=600, seed=3, weight=None)
        xs = np.array([v[0] for v in p.values()])
        ys = np.array([v[1] for v in p.values()])
        span = max(np.ptp(xs), np.ptp(ys), 1e-6)
        cells.append((side, {n: ((v[0] - xs.mean()) / span * side * 0.7, (v[1] - ys.mean()) / span * side * 0.55)
                             for n, v in p.items()}))
    width = math.sqrt(sum(s * s for s, _ in cells)) * 1.35
    pos, x, y, rowh = {}, 0.0, 0.0, 0.0
    for side, p in cells:
        if x + side > width and x > 0:
            x, y, rowh = 0.0, y - rowh - 0.5, 0.0
        for n, (px, py) in p.items():
            pos[n] = (x + side / 2 + px, y - side / 2 + py)
        x += side + 0.35
        rowh = max(rowh, side)
    fig, ax = plt.subplots(figsize=(13, 9.8))
    ax.set_facecolor(BG)
    wmax = max(d["weight"] for _, _, d in H.edges(data=True))
    for u, v, d in H.edges(data=True):
        tu, tv = W.base_title(u).lower(), W.base_title(v).lower()
        same = tu.split()[0] == tv.split()[0] or any(w in tv for w in tu.split() if len(w) > 4) or \
            any(w in tu for w in tv.split() if len(w) > 4)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=GOLD if same else CYAN,
                lw=1.2 + 6 * math.sqrt(d["weight"] / wmax), alpha=0.8, zorder=1, solid_capstyle="round")
    deg = dict(H.degree(weight="weight"))
    ax.scatter([pos[n][0] for n in H], [pos[n][1] for n in H], s=[70 + 0.7 * deg[n] for n in H],
               c=[COMM_COLOURS[com[n] % len(COMM_COLOURS)] if n in com else "#555" for n in H],
               edgecolors=BG, linewidths=0.8, zorder=3)
    items = [(pos[n][0], pos[n][1], pretty(n), FG, 8.4) for n in sorted(H, key=lambda n: -deg[n])]
    place_labels(ax, items, fig)
    ax.margins(0.04)
    ax.axis("off")
    ax.set_title(f"Pairs of pages sharing at least {min_shared} eight-word runs that occur in fewer than five pages "
                 f"({H.number_of_nodes()} pages, {H.number_of_edges()} pairs), one component per box.\n"
                 "Edge width = how many runs. Gold = the two pages share a name; node colour = Louvain community.",
                 color=FG, loc="left", fontsize=10.5, pad=12)
    fig.suptitle("Wikipedia copies itself: the copying network of the Marvel pages, as a gallery of pairs",
                 color=FG, fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.99)
    fig.savefig(FIGS / "week5_copying.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- V. search, absentees
def search_section(pages, docs, G, stats):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import normalize
    ids = sorted(docs)
    idx = {n: i for i, n in enumerate(ids)}
    corpus = [" ".join(W.words(docs[n], stop=True)) for n in ids]
    vec = CountVectorizer(token_pattern=r"\S+")
    X = vec.fit_transform(corpus)
    Xn = normalize(X)
    results = []
    for qtext, want in SEARCH_QUERIES:
        qwords = [w for w in re.findall(r"[a-z']+", qtext.lower()) if w not in W.STOPWORDS]
        qv = normalize(vec.transform([" ".join(qwords)]))
        sc = (Xn @ qv.T).toarray().ravel()
        order = np.argsort(-sc)
        rank = int(np.where(order == idx[want])[0][0]) + 1 if want in idx else None
        results.append({"query": qtext, "wanted": pretty(want), "in_corpus": want in idx, "rank": rank,
                        "top3": [(pretty(ids[j]), round(float(sc[j]), 3)) for j in order[:3]]})
    text = "\n".join(pages.values())
    # a variant page (Ultimate universe, film series) does not make the character present
    roster_titles = {W.base_title(n).lower() for n in G if not re.search(r"Ultimate|film", n)}
    absent = []
    for a in ABSENT:
        rx = re.compile(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])")
        absent.append({"name": a, "mentions": len(rx.findall(text)),
                       "pages": sum(1 for p in pages.values() if rx.search(p)),
                       "in_roster": a.lower() in roster_titles})
    hubs = []
    for a in ROSTER_HUBS:
        rx = re.compile(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])")
        hubs.append({"name": a, "mentions": len(rx.findall(text)), "pages": sum(1 for p in pages.values() if rx.search(p)),
                     "in_roster": a.lower() in roster_titles})
    absent.sort(key=lambda r: -r["mentions"])
    stats["search"] = {"matrix_shape": list(X.shape), "zero_share": round(1 - X.nnz / (X.shape[0] * X.shape[1]), 3),
                       "results": results, "hits_top3": sum(1 for r in results if r["rank"] and r["rank"] <= 3),
                       "not_in_corpus": [r["wanted"] for r in results if not r["in_corpus"]],
                       "absent": absent, "hubs": hubs}
    return X, Xn, ids


def fig_absent(stats):
    rows = [r for r in stats["search"]["absent"] if not r["in_roster"]][:16]
    hubs = [r for r in stats["search"]["hubs"] if r["in_roster"]][:4]
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    names = [r["name"] for r in hubs] + [r["name"] for r in rows]
    vals = [r["mentions"] for r in hubs] + [r["mentions"] for r in rows]
    cols = [CYAN] * len(hubs) + [RED] * len(rows)
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols, alpha=0.88, height=0.7)
    for i, (v, r) in enumerate(zip(vals, hubs + rows)):
        ax.text(v + 12, i, f"{v}  on {r['pages']} pages", va="center", fontsize=8.6, color=FG)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.28)
    style(ax, None, "mentions across the 303 pages", None)
    ax.set_title("Cyan: characters with a page in the corpus. Red: the most-mentioned characters who have none.",
                 color=FG, loc="left", fontsize=11, pad=12)
    fig.suptitle("The hero who isn't there: no Captain America, no Thor, no Iron Man page in the playground",
                 color=FG, fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "week5_absent.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- VI. lookalikes, weird
def lookalikes_section(docs, G, X, Xn, ids, stats):
    from sklearn.preprocessing import normalize
    UG = G.to_undirected()
    idx = {n: i for i, n in enumerate(ids)}
    S = np.asarray((Xn @ Xn.T).todense())
    np.fill_diagonal(S, -1)
    rng = np.random.default_rng(1)
    ov, rnd = [], []
    for n in ids:
        nb = set(UG[n])
        top = [ids[j] for j in np.argsort(-S[idx[n]])[:10]]
        ov.append(len(nb & set(top)))
        rnd.append(len(nb & set(rng.choice([m for m in ids if m != n], 10, replace=False))))
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if not UG.has_edge(ids[i], ids[j]):
                pairs.append((float(S[i, j]), pretty(ids[i]), pretty(ids[j])))
    pairs.sort(reverse=True)
    w = idx["Wolverine_(character)"]
    centroid = normalize(np.asarray(Xn.mean(axis=0)).reshape(1, -1))
    cos_c = np.asarray(Xn @ centroid.T).ravel() if not hasattr(Xn @ centroid.T, 'toarray') else (Xn @ centroid.T).toarray().ravel()
    rowsum = np.asarray(X.sum(axis=1)).ravel()
    big = rowsum >= 800
    weird = [j for j in np.argsort(cos_c) if big[j]][:6]
    vocab = np.array(list(sorted(
        __import__("sklearn.feature_extraction.text", fromlist=["CountVectorizer"]).CountVectorizer(token_pattern=r"\S+")
        .fit([" ".join(W.words(docs[n], stop=True)) for n in ids]).vocabulary_.items(), key=lambda kv: kv[1])))[:, 0]
    tot = np.asarray(X.sum(axis=0)).ravel()
    distinct = {}
    for j in weird:
        row = np.asarray(X[j].todense()).ravel()
        rel = (row / row.sum()) / (tot / tot.sum())
        rel[row < 3] = 0
        distinct[pretty(ids[j])] = list(vocab[np.argsort(-rel)[:8]])
    stats["lookalikes"] = {
        "network_neighbours_in_top10": round(float(np.mean(ov)), 2), "random_baseline": round(float(np.mean(rnd)), 2),
        "wolverine": [(pretty(ids[j]), round(float(S[w, j]), 3), UG.has_edge(ids[w], ids[j])) for j in np.argsort(-S[w])[:5]],
        "closest_unlinked": [(round(s, 3), a, b) for s, a, b in pairs[:6]],
        "least_like_corpus": [(pretty(ids[j]), round(float(cos_c[j]), 3), int(rowsum[j])) for j in weird],
        "least_like_corpus_any_size": [(pretty(ids[j]), round(float(cos_c[j]), 3), int(rowsum[j])) for j in np.argsort(cos_c)[:5]],
        "most_like_corpus": [(pretty(ids[j]), round(float(cos_c[j]), 3), int(rowsum[j])) for j in np.argsort(-cos_c)[:4]],
        "distinctive_words": distinct,
    }


# ----------------------------------------------------------------------------- VII. generators
def generator_section(docs, comms, cnames, stats):
    cs = {n: W.clean_sentences(docs[n]) for n in docs}
    everything = [s for v in cs.values() for s in v]
    out = []
    for i, c in enumerate(comms):
        m = W.TrigramModel([s for n in c for s in cs.get(n, [])])
        m.add_guard(everything)
        rng = random.Random(100 + i)
        gen = m.generate(rng, 2, lo=9, hi=24)
        out.append({"community": cnames[i], "size": len(c), "sentences": len([s for n in c for s in cs.get(n, [])]),
                    "generated": [W.detok(s) for s in gen]})
    stats["generators"] = {"clean_sentences": len(everything), "per_community": out}


# ----------------------------------------------------------------------------- main
def main():
    pages = W.load_pages()
    docs = W.load_tokens(pages)
    G = load_graph()
    stats = {"network": {"n": G.number_of_nodes(), "m": G.number_of_edges()}}

    print("I.   counts")
    cL, cD, cM = counts_section(pages, docs, G, stats)
    fig_zipf(cL, cD, cM)
    print("II.  fame and Heaps")
    L, heaps, fit, res = fame_section(docs, G, stats)
    fig_fame(docs, G, L, heaps, fit, res)
    print("III. relations")
    labels, comms, com, giant = relations_section(pages, docs, G, stats)
    fig_relations(G, giant, labels, comms, com, stats)
    print("IV.  copying")
    H = copying_section(pages, docs, G, com, stats)
    fig_copying(H, G, com)
    print("V.   search")
    X, Xn, ids = search_section(pages, docs, G, stats)
    fig_absent(stats)
    print("VI.  lookalikes")
    lookalikes_section(docs, G, X, Xn, ids, stats)
    print("VII. generators")
    generator_section(docs, comms, stats["relations"]["louvain"]["names"], stats)

    (DATA / "stats_week5.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    c = stats["counts"]
    print(f"\ntokens {c['lower']['tokens']:,} types {c['lower']['types']:,} hapax {c['lower']['hapax_share']:.0%}; "
          f"Zipf s(top 1000) full {c['zipf']['full_top1000']} descriptions {c['descriptions']['zipf_s_all']}")
    t = stats["relations"]["test"]
    print("relations:", stats["relations"]["counts"])
    for k in ("foe", "ally", "family"):
        print(f"   {k:7s} n={t[k]['n']:3d} cross {t[k]['cross']:.3f} null {t[k]['null_mean']:.3f} ± {t[k]['null_sd']:.3f} z={t[k]['z']:+.1f} p={t[k]['p']}")
    print("audit:", stats["relations"]["audit"])
    print("search hits in top 3:", stats["search"]["hits_top3"], "of", len(SEARCH_QUERIES), "; not in corpus:", stats["search"]["not_in_corpus"])
    print("wrote data/stats_week5.json and", sorted(p.name for p in FIGS.glob("week5_*.png")))


if __name__ == "__main__":
    main()
