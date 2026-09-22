"""
Week 2 analysis: models and null models on the Marvel network, and the same
toolkit on its natural sequel, the villains.

Five questions, in the order the post asks them:
  1. Random, Barabasi-Albert, or neither?  CCDFs against both models, and a
     proper Clauset-Shalizi-Newman fit (scripts/heavytail.py) with a bootstrap
     test and alternatives, on the frozen snapshot.
  2. The friendship paradox: how strong, who drives it, and who is the one
     character nobody out-popularities.
  3. Shuffle-test twelve numbers under two nulls (degree-preserving swaps and
     G(n,m)) and sort them into survivors and casualties.
  4. Grow a preferential-attachment Marvel with the same n and m, put it next
     to the real one, and test the model's central prediction, that the first
     to arrive become the hubs, against real debut years from Wikipedia.
  5. Add Category:Marvel Comics supervillains (data/week2_*.tsv, from
     scripts/crawl_week2.py) and run everything again, plus the one test the
     villains make possible: do villains link to heroes more than chance?

Outputs
  assets/figures/week2_*.png    the figures in weeks/week2/index.html
  data/stats_week2.json         every number the post quotes

Run:  python scripts/analyse_week2.py            (about ten minutes)
      python scripts/analyse_week2.py --quick    (a minute, 30 shuffles)
"""

import argparse
import collections
import csv
import json
import math
import pathlib
import sys
import time

import numpy as np
import networkx as nx
from scipy import stats as sps

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import heavytail as ht                                       # noqa: E402
from analyse import (BG, CYAN, DIM, FG, GOLD, GREEN, GRID, HALO, PURPLE, RED,   # noqa: E402
                     load_graph, place_labels, pretty, style)
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.collections import LineCollection             # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGS = ROOT / "assets" / "figures"
DATA = ROOT / "data"
ORANGE = "#ff9f43"


def read_tsv(path, fieldnames=None):
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t", fieldnames=fieldnames))


def load_combined():
    """The week-2 crawl: heroes + villains, with a side label and a debut year."""
    H = nx.DiGraph()
    for r in read_tsv(DATA / "week2_nodes.tsv"):
        H.add_node(r["node_id"], name=r["name"], side=r["side"],
                   year=int(r["debut_year"]) if r["debut_year"] else None)
    for r in read_tsv(DATA / "week2_edges.tsv"):
        H.add_edge(r["source"], r["target"])
    return H


# ---------------------------------------------------------------- null models
def directed_swap(G, nswap, rng):
    """Degree-preserving shuffle for a directed network.

    Pick a->b and c->d, rewire to a->d and c->b. Every in-degree and every
    out-degree survives; who links to whom does not. Self-loops and duplicate
    links are refused, which is what keeps it a simple graph. Ten swaps per
    link is the usual rule for forgetting the original wiring.
    """
    edges = list(G.edges())
    E = set(edges)
    m = len(edges)
    done = tries = 0
    while done < nswap and tries < 50 * nswap:
        tries += 1
        i, j = rng.integers(m, size=2)
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if a == d or c == b or a == c or b == d or (a, d) in E or (c, b) in E:
            continue
        E.remove((a, b))
        E.remove((c, d))
        E.add((a, d))
        E.add((c, b))
        edges[i], edges[j] = (a, d), (c, b)
        done += 1
    out = nx.DiGraph()
    out.add_nodes_from(G.nodes(data=True))
    out.add_edges_from(E)
    return out


def undirected_swap(G, rng):
    g = G.copy()
    m = g.number_of_edges()
    nx.double_edge_swap(g, nswap=10 * m, max_tries=200 * m, seed=int(rng.integers(2**31)))
    return g


def gnm_like(G, rng, directed=False):
    g = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges(), directed=directed,
                            seed=int(rng.integers(2**31)))
    g = nx.relabel_nodes(g, dict(enumerate(G.nodes())))
    for n, d in G.nodes(data=True):
        g.nodes[n].update(d)
    return g


def null_summary(real, samples):
    """z-score and one-sided empirical p-value, with the +1 that stops p = 0."""
    v = np.asarray(samples, dtype=float)
    mean, sd = float(v.mean()), float(v.std())
    if sd < 1e-9:
        sd = 0.0
    if sd > 0:
        z = (real - mean) / sd
    else:
        z = 0.0 if abs(real - mean) < 1e-9 else math.copysign(math.inf, real - mean)
    extreme = int(np.sum(v >= real)) if real >= mean else int(np.sum(v <= real))
    p = (extreme + 1) / (len(v) + 1)
    return {"real": float(real), "mean": round(mean, 4), "sd": round(sd, 4),
            "z": (round(z, 2) if math.isfinite(z) else ("+inf" if z > 0 else "-inf")),
            "p": round(p, 4), "n": int(len(v)), "samples": [float(x) for x in v]}


# ------------------------------------------------------ measured quantities
def undirected_quantities(g):
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    gc = g.subgraph(comps[0])
    k = np.array([d for _, d in g.degree()], dtype=float)
    return {
        "clustering": nx.average_clustering(g),
        "transitivity": nx.transitivity(g),
        "aspl": nx.average_shortest_path_length(gc),
        "diameter": nx.diameter(gc),
        "assortativity": nx.degree_assortativity_coefficient(g),
        "max_core": max(nx.core_number(g).values()),
        "hub_share": k.max() / g.number_of_edges(),
        "friend_mean": friendship(g)["friend_mean"],
    }


def whole_quantities(g):
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    return {
        "islands": sum(1 for c in comps[1:] if len(c) > 1),
        "isolates": nx.number_of_isolates(g),
    }


def directed_quantities(g):
    sccs = sorted(nx.strongly_connected_components(g), key=len, reverse=True)
    return {
        "reciprocity": nx.reciprocity(g),
        "n_scc": len(sccs),
        "giant_scc": len(sccs[0]),
    }


def side_of(g):
    lab = nx.get_node_attributes(g, "side")
    return {n: {"hero": "H", "villain": "V", "both": "B"}[lab[n]] for n in g}


def combined_quantities(g):
    s = side_of(g)
    pairs = collections.Counter(s[u] + s[v] for u, v in g.edges())
    m = g.number_of_edges()
    mutual_cross = sum(1 for u, v in g.edges()
                       if u < v and g.has_edge(v, u) and {s[u], s[v]} == {"H", "V"})
    vh = pairs["VH"] or 1
    return {
        "HH": pairs["HH"], "HV": pairs["HV"], "VH": pairs["VH"], "VV": pairs["VV"],
        "same_side_share": sum(c for t, c in pairs.items() if t[0] == t[1]) / m,
        "nemesis_pairs": mutual_cross,
        "reciprocity_VH": sum(1 for u, v in g.edges() if s[u] == "V" and s[v] == "H" and g.has_edge(v, u)) / vh,
        "reciprocity": nx.reciprocity(g),
        "clustering": nx.average_clustering(g.to_undirected()),
    }


def friendship(ug):
    """The friendship paradox, computed exactly rather than sampled.

    Pick a character uniformly among those with at least one link, then one of
    their neighbours uniformly. P(friend = v) = sum over u in N(v) of 1/(n k_u).
    """
    k = dict(ug.degree())
    nodes = [n for n in ug if k[n] > 0]
    n = len(nodes)
    p_friend = collections.Counter()
    for u in nodes:
        for v in ug[u]:
            p_friend[v] += 1.0 / (n * k[u])
    ks = np.array([k[u] for u in nodes], dtype=float)
    ge = sum(1.0 / (n * k[u]) for u in nodes for v in ug[u] if k[v] >= k[u])
    return {
        "person_mean": float(ks.mean()),
        "friend_mean": float(sum(p * k[v] for v, p in p_friend.items())),
        "edge_end_mean": float((ks ** 2).mean() / ks.mean()),
        "p_friend_at_least": float(ge),
        "share_nodes_below_neighbours": float(np.mean([np.mean([k[v] for v in ug[u]]) > k[u] for u in nodes])),
        "p_friend": p_friend,
        "kings": sorted((u for u in nodes if all(k[v] <= k[u] for v in ug[u])), key=lambda u: -k[u]),
    }


# ------------------------------------------------------------ growing Marvel
def grow_ba(n, m_edges, rng, preferential=True):
    """Preferential attachment with the endpoint-list trick, tuned to hit m exactly.

    Seed: a six-node star. Each newcomer makes four or five links (never to the
    same node twice); exactly as many make five as needed for the total to land
    on m_edges. Picking uniformly from the list of edge endpoints IS picking a
    node with probability k_i / sum k_j, which is the whole model.
    """
    seed = 6
    edges = [(0, i) for i in range(1, seed)]
    ends = [x for e in edges for x in e]
    newcomers = list(range(seed, n))
    n_five = m_edges - len(edges) - 4 * len(newcomers)
    assert 0 <= n_five <= len(newcomers), "m is out of reach with 4-5 links per newcomer"
    five = set(rng.choice(newcomers, size=n_five, replace=False).tolist())
    for new in newcomers:
        want = 5 if new in five else 4
        chosen = set()
        while len(chosen) < want:
            chosen.add(ends[rng.integers(len(ends))] if preferential else int(rng.integers(new)))
        for t in chosen:
            edges.append((new, t))
            ends += [new, t]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    return g


# ------------------------------------------------------------------ figures
def ccdf_band(runs, kmax):
    """5-95 percentile band of P(K >= k) across model runs, on every integer k."""
    ks = np.arange(1, kmax + 1)
    mat = np.array([[np.mean(np.asarray(r) >= k) for k in ks] for r in runs])
    return ks, np.percentile(mat, 5, axis=0), np.percentile(mat, 50, axis=0), np.percentile(mat, 95, axis=0)


def place_labels2(ax, items, fig):
    """place_labels from week 1 with eight candidate slots instead of four, so a
    label in a crowded centre (Norman Osborn next to Spider-Man) still finds room."""
    import matplotlib.transforms
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    candidates = [(0, 7, "center", "bottom"), (0, -7, "center", "top"), (9, 0, "left", "center"),
                  (-9, 0, "right", "center"), (8, 8, "left", "bottom"), (-8, 8, "right", "bottom"),
                  (8, -8, "left", "top"), (-8, -8, "right", "top"), (0, 16, "center", "bottom"),
                  (0, -16, "center", "top")]
    taken = []
    for x, y, text, colour, fs in items:
        for dx, dy, ha, va in candidates:
            t = ax.text(x, y, text, fontsize=fs, color=colour, ha=ha, va=va, fontweight="bold", zorder=6,
                        path_effects=HALO, transform=ax.transData)
            t.set_transform(ax.transData + matplotlib.transforms.ScaledTranslation(dx / 72, dy / 72,
                                                                                    fig.dpi_scale_trans))
            bb = t.get_window_extent(renderer=renderer)
            box = (bb.x0 - 2, bb.y0 - 2, bb.x1 + 2, bb.y1 + 2)
            if any(box[0] < o[2] and box[2] > o[0] and box[1] < o[3] and box[3] > o[1] for o in taken):
                t.remove()
                continue
            taken.append(box)
            break


def fig_ccdf(UG, G, ba_degrees, er_degrees, fit_in, alts_in, stats):
    ku = np.array([d for _, d in UG.degree()])
    kin = np.array([d for _, d in G.in_degree()])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.3))

    # Left: the real undirected degree against both models, same n and m.
    ax = axes[0]
    kmax = 130
    for runs, colour, label in ((er_degrees, GREEN, "random G(n,m)"), (ba_degrees, PURPLE, "preferential attachment")):
        ks, lo, med, hi = ccdf_band(runs, kmax)
        keep = med > 0
        ax.fill_between(ks[keep], np.maximum(lo[keep], 1 / 303 / 3), hi[keep], color=colour, alpha=0.18, lw=0)
        ax.plot(ks[keep], med[keep], color=colour, lw=1.6, label=f"{label}, median of {len(runs)} runs")
    xs, ps = ht.ccdf(ku[ku > 0])
    ax.plot(xs, ps, "o-", color=CYAN, ms=4.2, lw=1.1, mec=BG, mew=0.4, label="Marvel, undirected degree", zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1 / 303 / 1.6, 1.3)
    ax.set_xlim(0.9, 160)
    style(ax, "Random, preferential attachment, or neither?", "degree k", "P(K ≥ k)   (CCDF, no binning)")
    ax.legend(labelcolor=FG, fontsize=8.4, loc="lower left")
    ax.annotate(f"Spider-Man, k = {ku.max()}", (ku.max(), 1 / 303), textcoords="offset points", xytext=(-8, 14),
                ha="right", fontsize=8.5, color=CYAN, arrowprops=dict(arrowstyle="-", color=CYAN, lw=0.8))
    ax.text(0.97, 0.95, f"biggest hub\nreal: {ku.max()}\nBA: {np.mean([max(r) for r in ba_degrees]):.0f} ± "
            f"{np.std([max(r) for r in ba_degrees]):.0f}\nrandom: {np.mean([max(r) for r in er_degrees]):.0f}",
            transform=ax.transAxes, fontsize=8.4, color=DIM, ha="right", va="top", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.5", fc="#111830", ec=GRID, lw=1))

    # Right: the in-degree tail with the three fitted shapes.
    ax = axes[1]
    xs, ps = ht.ccdf(kin[kin > 0])
    ax.plot(xs, ps, "o", color=CYAN, ms=4.4, mec=BG, mew=0.4, label="Marvel, in-degree", zorder=5)
    kmin, alpha = fit_in["kmin"], fit_in["alpha"]
    tail_share = np.mean(kin >= kmin)
    grid = np.arange(kmin, 200)
    from scipy import special
    pl = special.zeta(alpha, grid) / special.zeta(alpha, kmin) * tail_share
    ax.plot(grid, pl, "--", color=RED, lw=1.7, label=f"power law, α = {alpha:.2f} (k ≥ {kmin})")
    tp = alts_in["truncated_powerlaw"]
    w = grid ** (-tp["alpha"]) * np.exp(-tp["lambda"] * grid)
    ax.plot(grid, np.cumsum(w[::-1])[::-1] / w.sum() * tail_share, "-", color=GOLD, lw=1.7,
            label=f"power law with cut-off, α = {tp['alpha']:.2f}, λ = {tp['lambda']:.3f}")
    ln = alts_in["lognormal"]
    w = np.exp(-((np.log(grid) - ln["mu"]) ** 2) / (2 * ln["sigma"] ** 2)) / grid
    ax.plot(grid, np.cumsum(w[::-1])[::-1] / w.sum() * tail_share, ":", color=GREEN, lw=2.0,
            label=f"log-normal, μ = {ln['mu']:.2f}, σ = {ln['sigma']:.2f}")
    ax.axvline(kmin, color=DIM, lw=0.8, ls=":")
    ax.text(kmin * 1.06, 0.9, f"k_min = {kmin}", color=DIM, fontsize=8.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1 / 303 / 1.6, 1.3)
    ax.set_xlim(0.9, 160)
    style(ax, "Is the in-degree tail a power law? Three fits, one tail", "in-degree k", "P(K ≥ k)")
    ax.legend(labelcolor=FG, fontsize=8.2, loc="lower left")
    ax.text(0.97, 0.95, f"bootstrap p(power law) = {stats['fit']['in']['p_boot']:.3f}\n"
            f"cut-off beats pure power law: p = {tp['p']:.3f}\n"
            f"log-normal vs power law: p = {ln['p']:.2f} (a draw)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.2, color=DIM, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.5", fc="#111830", ec=GRID, lw=1))

    fig.suptitle("The Marvel degree distribution against the two models, and against a proper fit",
                 color=FG, fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "week2_ccdf.png", bbox_inches="tight")
    plt.close(fig)


def fig_friendship(UG, G, fp, stats):
    k = dict(UG.degree())
    nodes = [n for n in UG if k[n] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # Left: the two distributions the paradox compares, exact rather than sampled,
    # drawn as CCDFs so the shift is one curve sitting to the right of the other.
    ax = axes[0]
    ks = np.arange(1, 110)
    p_person = np.array([np.mean([k[u] == x for u in nodes]) for x in ks])
    p_friend = np.array([sum(p for v, p in fp["p_friend"].items() if k[v] == x) for x in ks])
    c_person = np.cumsum(p_person[::-1])[::-1]
    c_friend = np.cumsum(p_friend[::-1])[::-1]
    ax.plot(ks, c_person, "-", color=CYAN, lw=2.2, label=f"a random character (mean degree {fp['person_mean']:.1f})")
    ax.plot(ks, c_friend, "-", color=RED, lw=2.2, label=f"a random friend of one (mean degree {fp['friend_mean']:.1f})")
    ax.fill_between(ks, c_person, c_friend, color=RED, alpha=0.12, lw=0)
    ax.axvline(fp["person_mean"], color=CYAN, ls="--", lw=1.1)
    ax.axvline(fp["friend_mean"], color=RED, ls="--", lw=1.1)
    ax.set_xscale("log")
    ax.set_xlim(1, 120)
    ax.set_ylim(0, 1.02)
    style(ax, "Your friends have more friends than you", "degree k", "fraction with degree ≥ k")
    ax.legend(labelcolor=FG, fontsize=8.6, loc="upper right")
    ax.text(0.03, 0.04, f"P(friend at least as connected as you) = {fp['p_friend_at_least']:.2f}\n"
            f"Spider-Man is the friend {100 * fp['p_friend']['Spider-Man']:.1f}% of the time\n"
            f"half of all friends have degree ≥ {int(ks[np.argmax(c_friend <= 0.5)])}, "
            f"half of all characters ≥ {int(ks[np.argmax(c_person <= 0.5)])}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.3, color=DIM, linespacing=1.6)

    # Right: the directed version, every character at once.
    ax = axes[1]
    kin = dict(G.in_degree())
    src = [u for u in G if G.out_degree(u) > 0]
    x = np.array([kin[u] for u in src], dtype=float)
    y = np.array([np.mean([kin[v] for v in G.successors(u)]) for u in src])
    above = y > x
    rng = np.random.default_rng(3)
    jx = x + rng.uniform(-0.15, 0.15, len(x))
    ax.scatter(jx[above], y[above], s=22, color=CYAN, alpha=0.7, edgecolor=BG, lw=0.4,
               label=f"the pages you link to are more famous ({100 * above.mean():.0f}%)", zorder=3)
    ax.scatter(jx[~above], y[~above], s=30, color=GOLD, alpha=0.95, edgecolor=BG, lw=0.4,
               label=f"they are not ({100 * (~above).mean():.0f}%)", zorder=4)
    lim = 130
    ax.plot([0.3, lim], [0.3, lim], color=DIM, ls="--", lw=1, alpha=0.6)
    ax.text(40, 30, "equal fame", color=DIM, fontsize=8.5, rotation=32)
    labels = [(u, kin[u], np.mean([kin[v] for v in G.successors(u)])) for u in src
              if all(kin[v] <= kin[u] for v in G.successors(u))]
    ax.set_xscale("symlog", linthresh=1.5)
    ax.set_yscale("symlog", linthresh=1.5)
    ax.set_xlim(-0.6, lim)
    ax.set_ylim(-0.6, lim)
    style(ax, "The pages you link to are more famous than you", "your in-degree",
          "mean in-degree of the pages you link to")
    ax.legend(labelcolor=FG, fontsize=8.4, loc="upper left")
    labels.sort(key=lambda it: -it[1])
    place_labels2(ax, [(xx, yy, pretty(u), GOLD, 8.4) for u, xx, yy in labels], fig)

    fig.suptitle("The friendship paradox among superheroes, undirected (left) and as Wikipedia actually links (right)",
                 color=FG, fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "week2_friendship.png", bbox_inches="tight")
    plt.close(fig)


SURVIVOR_LABELS = {
    "clustering": ("Average clustering C", "{:.3f}"),
    "transitivity": ("Transitivity", "{:.3f}"),
    "aspl": ("Average shortest path", "{:.2f}"),
    "diameter": ("Diameter", "{:.0f}"),
    "assortativity": ("Degree assortativity r", "{:.3f}"),
    "max_core": ("Largest k-core", "{:.0f}"),
    "hub_share": ("Biggest hub's share of links", "{:.3f}"),
    "friend_mean": ("Mean degree of a random friend", "{:.1f}"),
    "islands": ("Islands outside the giant component", "{:.0f}"),
    "isolates": ("Isolated characters", "{:.0f}"),
    "reciprocity": ("Reciprocity (directed)", "{:.3f}"),
    "n_scc": ("Strongly connected components", "{:.0f}"),
    "giant_scc": ("Size of the largest strongly connected part", "{:.0f}"),
}


def fig_survivors(results, order, stats):
    n = len(order)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12.8, 3.0 * rows))
    verdicts = {}
    for ax, key in zip(axes.flat, order):
        label, fmt = SURVIVOR_LABELS[key]
        real = results[key]["swap"]["real"]
        sw = np.array(results[key]["swap"]["samples"])
        er = np.array(results[key]["gnm"]["samples"])
        lo = min(real, sw.min(), er.min())
        hi = max(real, sw.max(), er.max())
        span = (hi - lo) or abs(real) or 1.0
        bins = np.linspace(lo - 0.06 * span, hi + 0.06 * span, 34)
        ax.hist(er, bins=bins, color=PURPLE, alpha=0.55, label="random G(n,m)")
        ax.hist(sw, bins=bins, color=CYAN, alpha=0.65, label="degree-preserving shuffle")
        ax.axvline(real, color=RED, lw=2.2, label="real network")
        zs, ze = results[key]["swap"]["z"], results[key]["gnm"]["z"]
        ps = results[key]["swap"]["p"]
        ax.set_ylim(0, ax.get_ylim()[1] * 1.55)
        rel = (real - results[key]["swap"]["mean"]) / abs(results[key]["swap"]["mean"]) if results[key]["swap"]["mean"] else 0
        survives = isinstance(zs, str) or abs(zs) >= 3
        if key in ("isolates", "hub_share", "max_core") and results[key]["swap"]["sd"] == 0 and sw[0] == real:
            verdict, colour = "fixed by the degree sequence", DIM
            survives = None
        elif survives:
            verdict, colour = "SURVIVES the shuffle", GOLD
        else:
            verdict, colour = "dies: explained by degrees", DIM
        verdicts[key] = {"verdict": verdict, "survives": survives}
        ax.set_title(f"{label}: real {fmt.format(real)}", color=FG, loc="left", fontsize=10.2, pad=8)
        off = f", real is {rel:+.0%} off its mean" if 0.005 < abs(rel) < 5 else ""
        ax.text(0.02, 0.95, f"z = {zs} vs shuffle (p = {ps}){off}\nz = {ze} vs random",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color=DIM, linespacing=1.5)
        ax.text(0.98, 0.78, verdict, transform=ax.transAxes, ha="right", va="top", fontsize=8.6,
                color=colour, fontweight="bold")
        ax.set_yticks([])
        ax.grid(alpha=0.2, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].legend(labelcolor=FG, fontsize=7.8, loc="center right", bbox_to_anchor=(1.0, 0.45))
    fig.suptitle("Shuffle test: what survives when every character keeps its degree, and what dies",
                 color=FG, fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(FIGS / "week2_survivors.png", bbox_inches="tight")
    plt.close(fig)
    stats["verdicts"] = verdicts


def draw_graph(ax, g, pos, size_by, colour_by, labels, label_colour=FG, edge_colour="#4a5a92", title=None):
    segs = [(pos[u], pos[v]) for u, v in g.edges()]
    ax.add_collection(LineCollection(segs, colors=edge_colour, linewidths=0.3, alpha=0.32, zorder=1))
    order = sorted(g.nodes(), key=lambda n: size_by[n])
    xs = [pos[n][0] for n in order]
    ys = [pos[n][1] for n in order]
    ss = np.array([size_by[n] for n in order], dtype=float)
    cs = np.array([colour_by[n] for n in order], dtype=float)
    sc = ax.scatter(xs, ys, s=14 + 5.6 * ss, c=np.log1p(cs), cmap="plasma", vmin=0, vmax=np.log1p(106),
                    linewidths=0.4, edgecolors=BG, zorder=3, alpha=0.95)
    px = np.array([p[0] for p in pos.values()])
    py = np.array([p[1] for p in pos.values()])
    ax.set_xlim(np.percentile(px, 0.5) - 0.05, np.percentile(px, 99.5) + 0.05)
    ax.set_ylim(np.percentile(py, 0.5) - 0.05, np.percentile(py, 99.5) + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, color=FG, loc="left", fontsize=11.5, pad=10)
    return sc


def fig_grow(UG, ba_graph, stats):
    giant = UG.subgraph(max(nx.connected_components(UG), key=len))
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.6))
    kreal = dict(giant.degree())
    pos = nx.spring_layout(giant, k=0.55, iterations=500, seed=42)
    draw_graph(axes[0], giant, pos, kreal, kreal, None,
               title=f"Real Marvel: {len(giant)} characters in the giant component, {giant.number_of_edges()} links")
    top = sorted(giant.nodes(), key=lambda n: -kreal[n])[:9]
    place_labels2(axes[0], [(pos[n][0], pos[n][1], pretty(n), FG, 8.6) for n in top], fig)

    kba = dict(ba_graph.degree())
    pos2 = nx.spring_layout(ba_graph, k=0.55, iterations=500, seed=42)
    draw_graph(axes[1], ba_graph, pos2, kba, kba, None,
               title=f"Grown Marvel: preferential attachment to n = 303, m = {ba_graph.number_of_edges()}")
    top2 = sorted(ba_graph.nodes(), key=lambda n: -kba[n])[:6]
    place_labels2(axes[1], [(pos2[n][0], pos2[n][1], f"arrived #{n + 1}", GOLD, 8.6) for n in top2], fig)
    fig.suptitle("Same size, same number of links: the real network and one preferential-attachment run. "
                 "Node size and colour are degree.", color=FG, fontsize=12.5, fontweight="bold", x=0.012,
                 ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "week2_grow.png", bbox_inches="tight")
    plt.close(fig)


def fig_firstmover(G, years, ba_graph, stats):
    kin = dict(G.in_degree())
    nodes = [n for n in G if n in years]
    y = np.array([years[n] for n in nodes], dtype=float)
    k = np.array([kin[n] for n in nodes], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))

    ax = axes[0]
    rng = np.random.default_rng(5)
    ax.scatter(y + rng.uniform(-0.3, 0.3, len(y)), k, s=18 + 1.4 * k, c=k, cmap="plasma", vmin=0, vmax=60,
               alpha=0.85, edgecolor=BG, lw=0.4, zorder=3)
    edges_d = [1939] + list(range(1950, 2031, 10))
    means = [k[(y >= a) & (y < b)].mean() if np.any((y >= a) & (y < b)) else np.nan
             for a, b in zip(edges_d[:-1], edges_d[1:])]
    ax.step(edges_d, means + [means[-1]], where="post", color=GOLD, lw=1.8, alpha=0.9,
            label="mean in-degree per decade", zorder=4)
    rho = sps.spearmanr(y, k).correlation
    ax.set_yscale("symlog", linthresh=5)
    ax.set_ylim(-0.5, 160)
    ax.set_xlim(1931, 2024)
    style(ax, f"Wikipedia: debut year against in-degree, Spearman ρ = {rho:.2f}", "year of first appearance",
          "in-degree today")
    ax.legend(labelcolor=FG, fontsize=8.5, loc="upper right")
    named = sorted(nodes, key=lambda n: -kin[n])[:7] + ["Human_Torch_(android)"]
    place_labels2(ax, [(years[n], kin[n], f"{pretty(n)} ({years[n]})", FG, 8.3) for n in named], fig)

    ax = axes[1]
    kba = np.array([ba_graph.degree(i) for i in range(ba_graph.number_of_nodes())], dtype=float)
    t = np.arange(1, len(kba) + 1)
    ax.scatter(t, kba, s=18 + 1.4 * kba, c=kba, cmap="plasma", vmin=0, vmax=60, alpha=0.85, edgecolor=BG,
               lw=0.4, zorder=3)
    m_mean = ba_graph.number_of_edges() / ba_graph.number_of_nodes()
    ax.plot(t, m_mean * np.sqrt(len(kba) / t), color=GOLD, lw=1.8, alpha=0.9, label="model: k ∝ (n / arrival)½")
    rho_ba = sps.spearmanr(t, kba).correlation
    ax.set_yscale("symlog", linthresh=5)
    ax.set_ylim(-0.5, 160)
    style(ax, f"Preferential attachment: arrival order against degree, ρ = {rho_ba:.2f}", "arrival order (1 = first)",
          "degree at n = 303")
    ax.legend(labelcolor=FG, fontsize=8.5, loc="upper right")
    ax.text(0.97, 0.78, f"top-10 hubs arrived, on average:\nmodel: first {stats['firstmover']['ba_top10_percentile']:.0f}% "
            f"of nodes\nWikipedia: {stats['firstmover']['marvel_top10_percentile']:.0f}th percentile",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.4, color=DIM, linespacing=1.6)

    fig.suptitle("The model's central prediction, the first to arrive become the hubs, against real debut years",
                 color=FG, fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "week2_firstmover.png", bbox_inches="tight")
    plt.close(fig)


def fig_villains_network(H, stats):
    s = side_of(H)
    UH = H.to_undirected()
    giant = UH.subgraph(max(nx.connected_components(UH), key=len))
    pos = nx.spring_layout(giant, k=0.42, iterations=600, seed=7)
    kin = dict(H.in_degree())
    colour = {"H": CYAN, "V": RED, "B": GOLD}

    fig, ax = plt.subplots(figsize=(13, 10.4))
    segs = {"same": [], "cross": []}
    for u, v in H.edges():
        if u in pos and v in pos:
            segs["cross" if {s[u], s[v]} == {"H", "V"} else "same"].append((pos[u], pos[v]))
    ax.add_collection(LineCollection(segs["same"], colors="#3f4d80", linewidths=0.28, alpha=0.28, zorder=1))
    ax.add_collection(LineCollection(segs["cross"], colors="#b06cff", linewidths=0.30, alpha=0.22, zorder=2))
    order = sorted(giant.nodes(), key=lambda n: kin[n])
    ax.scatter([pos[n][0] for n in order], [pos[n][1] for n in order],
               s=[12 + 4.2 * kin[n] for n in order], c=[colour[s[n]] for n in order],
               linewidths=0.45, edgecolors=BG, zorder=3, alpha=0.92)
    items = []
    for side_key, nmax in (("H", 7), ("V", 8), ("B", 4)):
        top = sorted((n for n in giant if s[n] == side_key), key=lambda n: -kin[n])[:nmax]
        items += [(pos[n][0], pos[n][1], pretty(n), colour[side_key], 8.2 + 3.0 * kin[n] / 201) for n in top]
    items.sort(key=lambda it: -it[4])
    px = np.array([p[0] for p in pos.values()])
    py = np.array([p[1] for p in pos.values()])
    ax.set_xlim(np.percentile(px, 0.4) - 0.04, np.percentile(px, 99.6) + 0.04)
    ax.set_ylim(np.percentile(py, 0.4) - 0.04, np.percentile(py, 99.6) + 0.04)
    ax.set_aspect("equal")
    ax.axis("off")
    place_labels2(ax, items, fig)
    sides = stats["villains"]["sides"]
    ax.set_title(f"{sides['hero']} heroes (cyan), {sides['villain']} villains (red) and {sides['both']} characters "
                 f"Wikipedia files as both (gold). Purple links cross the line.", color=FG, loc="left",
                 fontsize=12.5, pad=14)
    fig.tight_layout()
    fig.savefig(FIGS / "week2_villains_network.png", bbox_inches="tight")
    plt.close(fig)


def fig_villains_mixing(H, res_c, fit_v, fit_h, stats):
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.7))

    # Left: the four kinds of link, real against the two nulls.
    ax = axes[0]
    keys = ["HH", "HV", "VH", "VV"]
    names = ["hero → hero", "hero → villain", "villain → hero", "villain → villain"]
    x = np.arange(4)
    real = [res_c[k]["swap"]["real"] for k in keys]
    sw = [res_c[k]["swap"]["mean"] for k in keys]
    sw_sd = [res_c[k]["swap"]["sd"] for k in keys]
    er = [res_c[k]["gnm"]["mean"] for k in keys]
    er_sd = [res_c[k]["gnm"]["sd"] for k in keys]
    ax.bar(x - 0.28, real, width=0.26, color=RED, label="real")
    ax.bar(x, sw, width=0.26, color=CYAN, yerr=sw_sd, ecolor=FG, capsize=3, label="degree-preserving shuffle")
    ax.bar(x + 0.28, er, width=0.26, color=PURPLE, yerr=er_sd, ecolor=FG, capsize=3, label="random G(n,m)")
    for i, k in enumerate(keys):
        top = max(real[i], sw[i] + sw_sd[i], er[i] + er_sd[i])
        ax.text(x[i], top + 25, f"z = {res_c[k]['swap']['z']:+.1f}\nvs shuffle", ha="center", va="bottom",
                fontsize=7.8, color=GOLD, linespacing=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.2)
    ax.set_ylim(0, max(max(real), max(er)) * 1.42)
    style(ax, "Who links to whom, against chance", None, "number of links")
    ax.legend(labelcolor=FG, fontsize=7.8, loc="upper left", ncol=3, columnspacing=0.8, handlelength=1.2)

    # Middle: nemesis pairs, the survivor.
    ax = axes[1]
    r = res_c["nemesis_pairs"]
    sw = np.array(r["swap"]["samples"])
    er = np.array(r["gnm"]["samples"])
    bins = np.arange(0, max(r["swap"]["real"], sw.max()) + 12, 4)
    ax.hist(er, bins=bins, color=PURPLE, alpha=0.55, label="random G(n,m)")
    ax.hist(sw, bins=bins, color=CYAN, alpha=0.65, label="degree-preserving shuffle")
    ax.axvline(r["swap"]["real"], color=RED, lw=2.2, label=f"real: {r['swap']['real']:.0f} pairs")
    style(ax, "Hero and villain that link to each other", "mutual hero ↔ villain pairs", "shuffles")
    ax.text(0.5, 0.6, f"z = {r['swap']['z']} vs shuffle\nz = {r['gnm']['z']} vs random", transform=ax.transAxes,
            ha="center", fontsize=8.4, color=DIM, linespacing=1.6)
    ax.legend(labelcolor=FG, fontsize=7.8, loc="upper right")

    # Right: the same toolkit, the villains' in-degree tail.
    ax = axes[2]
    s = side_of(H)
    kin = dict(H.in_degree())
    for side_key, colour, label, fit in (("H", CYAN, "heroes", fit_h), ("V", RED, "villains", fit_v)):
        ks = np.array([kin[n] for n in H if s[n] == side_key and kin[n] > 0])
        xs, ps = ht.ccdf(ks)
        ax.plot(xs, ps, "o-", color=colour, ms=3.8, lw=1.0, mec=BG, mew=0.4,
                label=f"{label}: α = {fit['alpha']:.2f} (k ≥ {fit['kmin']}), bootstrap p = {fit['p_boot']:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, "In-degree CCDF, heroes and villains", "in-degree (combined network)", "P(K ≥ k)")
    ax.legend(labelcolor=FG, fontsize=7.8, loc="lower left")

    fig.suptitle("The villains: raw counts say they obsess over heroes; the shuffle says both sides keep to themselves",
                 color=FG, fontsize=12.5, fontweight="bold", x=0.012, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "week2_villains_mixing.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- main
def run_nulls(G, measure, make_swap, make_gnm, n_shuffles, rng, label):
    t0 = time.time()
    real = measure(G)
    swaps = [measure(make_swap(G, rng)) for _ in range(n_shuffles)]
    gnms = [measure(make_gnm(G, rng)) for _ in range(n_shuffles)]
    out = {key: {"swap": null_summary(real[key], [s[key] for s in swaps]),
                 "gnm": null_summary(real[key], [s[key] for s in gnms])} for key in real}
    print(f"   {label}: {n_shuffles} shuffles x 2 nulls in {time.time() - t0:.0f}s")
    for key in real:
        print(f"      {key:18s} real {real[key]:9.3f}   swap {out[key]['swap']['mean']:9.3f} ± {out[key]['swap']['sd']:.3f} "
              f"z={out[key]['swap']['z']!s:>6}   gnm {out[key]['gnm']['mean']:9.3f} z={out[key]['gnm']['z']!s:>6}")
    return out


def strip_samples(results):
    """Keep every shuffled value: the post promises the histograms can be rebuilt from the JSON."""
    return {k: {null: {kk: ([round(x, 5) for x in vv] if kk == "samples" else vv) for kk, vv in d.items()}
                for null, d in v.items()} for k, v in results.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=300)
    ap.add_argument("--ba-runs", type=int, default=200)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--quick", action="store_true", help="30 shuffles, 40 BA runs, 100 bootstraps")
    args = ap.parse_args()
    if args.quick:
        args.shuffles, args.ba_runs, args.boot = 30, 40, 100
    rng = np.random.default_rng(2026)
    stats = {"settings": vars(args)}

    G = load_graph()
    UG = G.to_undirected()
    giant = UG.subgraph(max(nx.connected_components(UG), key=len)).copy()
    H = load_combined()
    years = {n: d["year"] for n, d in H.nodes(data=True) if d["year"]}
    kin = np.array([d for _, d in G.in_degree()])
    kout = np.array([d for _, d in G.out_degree()])
    ku = np.array([d for _, d in UG.degree()])
    stats["network"] = {"n": G.number_of_nodes(), "m_directed": G.number_of_edges(),
                        "m_undirected": UG.number_of_edges(), "giant": giant.number_of_nodes(),
                        "giant_edges": giant.number_of_edges(), "mean_k_undirected": round(float(ku.mean()), 2),
                        "max_k_undirected": int(ku.max()),
                        "top_undirected": [(pretty(n), int(d)) for n, d in sorted(UG.degree(), key=lambda x: -x[1])[:6]]}

    # 1. Fits ------------------------------------------------------------------
    print("1. heavy-tail fits")
    stats["fit"] = {}
    fits = {}
    for name, k in (("in", kin), ("out", kout), ("undirected", ku)):
        f = ht.fit_tail(k)
        p = ht.bootstrap_pvalue(k, f, n_boot=args.boot, seed=11)
        alts = ht.compare(k, f)
        fits[name] = (f, alts)
        stats["fit"][name] = {"kmin": f["kmin"], "alpha": round(f["alpha"], 2), "alpha_se": round(f["alpha_se"], 2),
                              "n_tail": f["n_tail"], "ks": round(f["ks"], 3), "p_boot": round(p, 3), "alternatives": alts}
        print(f"   {name:10s} k_min {f['kmin']:3d}  alpha {f['alpha']:.2f} ± {f['alpha_se']:.2f}  tail {f['n_tail']:3d}  "
              f"p_boot {p:.3f}  cutoff p {alts['truncated_powerlaw']['p']}  lognormal p {alts['lognormal']['p']}")

    # The same fit on models of the same size: what the test can and cannot do at n = 303.
    print("   power of the test at n = 303")
    ba_degrees, er_degrees, ba_stats = [], [], []
    ba_alpha, ba_p, er_alpha, er_p = [], [], [], []
    for i in range(args.ba_runs):
        g = grow_ba(303, UG.number_of_edges(), rng)
        kb = np.array([d for _, d in g.degree()])
        ba_degrees.append(kb)
        fr = friendship(g)
        ba_stats.append({"max_k": int(kb.max()), "clustering": nx.average_clustering(g),
                         "transitivity": nx.transitivity(g), "aspl": nx.average_shortest_path_length(g),
                         "diameter": nx.diameter(g), "assortativity": nx.degree_assortativity_coefficient(g),
                         "spearman_arrival": sps.spearmanr(np.arange(303), kb).correlation,
                         "top10_percentile": float(np.mean(np.argsort(-kb)[:10]) / 303 * 100),
                         "friend_mean": fr["friend_mean"], "p_friend_at_least": fr["p_friend_at_least"]})
        ge = gnm_like(UG, rng)
        ke = np.array([d for _, d in ge.degree()])
        er_degrees.append(ke)
        if i < 25:
            fb = ht.fit_tail(kb)
            ba_alpha.append(fb["alpha"])
            ba_p.append(ht.bootstrap_pvalue(kb, fb, n_boot=max(50, args.boot // 5), seed=i))
            fe = ht.fit_tail(ke)
            er_alpha.append(fe["alpha"])
            er_p.append(ht.bootstrap_pvalue(ke, fe, n_boot=max(50, args.boot // 5), seed=i))
    stats["fit"]["ba_303"] = {"alpha_mean": round(float(np.mean(ba_alpha)), 2), "alpha_sd": round(float(np.std(ba_alpha)), 2),
                              "p_boot_median": round(float(np.median(ba_p)), 3),
                              "share_rejected_05": round(float(np.mean(np.array(ba_p) < 0.05)), 2)}
    stats["fit"]["er_303"] = {"alpha_mean": round(float(np.mean(er_alpha)), 2), "alpha_sd": round(float(np.std(er_alpha)), 2),
                              "p_boot_median": round(float(np.median(er_p)), 3),
                              "share_rejected_05": round(float(np.mean(np.array(er_p) < 0.05)), 2)}
    print(f"   BA(303): fitted alpha {stats['fit']['ba_303']['alpha_mean']} ± {stats['fit']['ba_303']['alpha_sd']}, "
          f"rejected at 5%: {stats['fit']['ba_303']['share_rejected_05']};  "
          f"ER(303): alpha {stats['fit']['er_303']['alpha_mean']}, rejected at 5%: {stats['fit']['er_303']['share_rejected_05']}")

    fig_ccdf(UG, G, ba_degrees, er_degrees, fits["in"][0], fits["in"][1], stats)

    # 2. Friendship paradox ------------------------------------------------------
    print("2. friendship paradox")
    fp = friendship(UG)
    top_friends = fp["p_friend"].most_common(12)
    src = [u for u in G if G.out_degree(u) > 0]
    kin_d = dict(G.in_degree())
    above = [np.mean([kin_d[v] for v in G.successors(u)]) > kin_d[u] for u in src]
    ge_d = sum(1.0 / (len(src) * G.out_degree(u)) for u in src for v in G.successors(u) if kin_d[v] >= kin_d[u])
    stats["friendship"] = {
        "person_mean": round(fp["person_mean"], 2), "friend_mean": round(fp["friend_mean"], 2),
        "edge_end_mean": round(fp["edge_end_mean"], 2), "p_friend_at_least": round(fp["p_friend_at_least"], 3),
        "share_nodes_below_neighbours": round(fp["share_nodes_below_neighbours"], 3),
        "top_friends": [(pretty(v), round(100 * p, 2), int(UG.degree(v))) for v, p in top_friends],
        "top10_friend_share": round(100 * sum(p for _, p in top_friends[:10]), 1),
        "kings": [(pretty(u), int(UG.degree(u))) for u in fp["kings"]],
        "directed_share_linking_upward": round(float(np.mean(above)), 3),
        "directed_p_linked_at_least": round(float(ge_d), 3),
        "directed_mean_in_of_linkers": round(float(np.mean([kin_d[u] for u in src])), 2),
        "directed_mean_in_of_linked": round(float(np.mean([np.mean([kin_d[v] for v in G.successors(u)]) for u in src])), 2),
        "link_only_downward": [(pretty(u), kin_d[u], G.out_degree(u)) for u in src
                               if all(kin_d[v] <= kin_d[u] for v in G.successors(u))],
        "more_famous_than_all_linkers": [(pretty(u), kin_d[u]) for u in G if G.in_degree(u) > 0
                                         and all(kin_d[v] < kin_d[u] for v in G.predecessors(u))],
        "ba_friend_mean": round(float(np.mean([b["friend_mean"] for b in ba_stats])), 2),
        "ba_p_friend_at_least": round(float(np.mean([b["p_friend_at_least"] for b in ba_stats])), 3),
    }
    print(f"   person {fp['person_mean']:.1f}, friend {fp['friend_mean']:.1f}, P(friend >= you) {fp['p_friend_at_least']:.3f}, "
          f"kings {stats['friendship']['kings']}")
    fig_friendship(UG, G, fp, stats)

    # 3. Shuffle tests -----------------------------------------------------------
    print("3. shuffle tests")
    res_u = run_nulls(giant, undirected_quantities, undirected_swap, lambda g, r: gnm_like(g, r), args.shuffles, rng,
                      "undirected giant component")
    res_w = run_nulls(UG, whole_quantities, undirected_swap, lambda g, r: gnm_like(g, r), args.shuffles, rng,
                      "whole undirected network")
    res_d = run_nulls(G, directed_quantities, lambda g, r: directed_swap(g, 10 * g.number_of_edges(), r),
                      lambda g, r: gnm_like(g, r, directed=True), args.shuffles, rng, "directed network")
    results = {**res_u, **res_w, **res_d}
    order = ["clustering", "transitivity", "aspl", "diameter", "assortativity", "max_core",
             "hub_share", "islands", "isolates", "reciprocity", "n_scc", "giant_scc"]
    fig_survivors(results, order, stats)
    stats["shuffle"] = strip_samples(results)
    stats["friendship"]["null_friend_mean_swap"] = results["friend_mean"]["swap"]["mean"]
    stats["friendship"]["null_friend_mean_gnm"] = results["friend_mean"]["gnm"]["mean"]

    # 4. Grow your own Marvel ----------------------------------------------------
    print("4. grow your own Marvel")
    ba_graph = grow_ba(303, UG.number_of_edges(), np.random.default_rng(7))
    real_q = undirected_quantities(giant)
    summary = {}
    for key in ("max_k", "clustering", "transitivity", "aspl", "diameter", "assortativity", "spearman_arrival",
                "top10_percentile"):
        vals = np.array([b[key] for b in ba_stats], dtype=float)
        summary[key] = {"mean": round(float(vals.mean()), 3), "sd": round(float(vals.std()), 3)}
    er_q = {"max_k": float(np.mean([k.max() for k in er_degrees]))}
    nodes_y = [n for n in G if n in years]
    yv = np.array([years[n] for n in nodes_y], dtype=float)
    kv = np.array([G.in_degree(n) for n in nodes_y], dtype=float)
    order_y = sorted(nodes_y, key=lambda n: years[n])
    rank = {n: i for i, n in enumerate(order_y)}
    top10 = sorted(nodes_y, key=lambda n: -G.in_degree(n))[:10]
    decade = {}
    for d in range(1930, 2030, 10):
        sel = (yv >= d) & (yv < d + 10)
        if sel.any():
            decade[f"{d}s"] = {"n": int(sel.sum()), "mean_in": round(float(kv[sel].mean()), 1),
                               "median_in": float(np.median(kv[sel])), "max_in": int(kv[sel].max()),
                               "max_name": pretty(nodes_y[int(np.flatnonzero(sel)[np.argmax(kv[sel])])])}
    stats["firstmover"] = {
        "heroes_with_year": len(nodes_y),
        "spearman_year_indegree": round(float(sps.spearmanr(yv, kv).correlation), 3),
        "spearman_p": float(sps.spearmanr(yv, kv).pvalue),
        "ba_spearman_arrival_degree": summary["spearman_arrival"]["mean"],
        "ba_top10_percentile": summary["top10_percentile"]["mean"],
        "marvel_top10_percentile": round(float(np.mean([100 * rank[n] / len(order_y) for n in top10])), 1),
        "top10": [(pretty(n), G.in_degree(n), years[n], round(100 * rank[n] / len(order_y))) for n in top10],
        "golden_age": [(pretty(n), G.in_degree(n), years[n]) for n in order_y[:12]],
        "decades": decade,
    }
    stats["grow"] = {"real": {k: round(float(v), 3) for k, v in real_q.items()} | {"max_k": int(ku.max()),
                     "isolates": int(nx.number_of_isolates(UG)), "islands": whole_quantities(UG)["islands"]},
                     "ba": summary | {"isolates": 0, "islands": 0, "m": ba_graph.number_of_edges()},
                     "er": er_q}
    print(f"   BA max k {summary['max_k']['mean']} ± {summary['max_k']['sd']}, C {summary['clustering']['mean']}, "
          f"first-mover rho {summary['spearman_arrival']['mean']} vs Marvel {stats['firstmover']['spearman_year_indegree']}")
    fig_grow(UG, ba_graph, stats)
    fig_firstmover(G, years, ba_graph, stats)

    # 5. The villains -------------------------------------------------------------
    print("5. the villains")
    s = side_of(H)
    kinH = dict(H.in_degree())
    meta = json.loads((DATA / "week2_crawl.json").read_text(encoding="utf-8"))
    pairs = collections.Counter(s[u] + s[v] for u, v in H.edges())
    out_share = {}
    for a in "HV":
        outs = [v for u, v in H.edges() if s[u] == a]
        out_share[a] = {b: round(sum(s[v] == b for v in outs) / len(outs), 3) for b in "HVB"}
    linked_by_villains = collections.Counter(v for u, v in H.edges() if s[u] == "V")
    linked_by_heroes = collections.Counter(v for u, v in H.edges() if s[u] == "H")
    nemesis = [(u, v) for u, v in H.edges() if u < v and H.has_edge(v, u) and {s[u], s[v]} == {"H", "V"}]
    nemesis.sort(key=lambda e: -(kinH[e[0]] + kinH[e[1]]))
    UH = H.to_undirected()
    fpH = friendship(UH)
    kin_v = np.array([kinH[n] for n in H if s[n] == "V"])
    kin_h = np.array([kinH[n] for n in H if s[n] == "H"])
    fit_v = ht.fit_tail(kin_v)
    fit_h = ht.fit_tail(kin_h)
    fit_v["p_boot"] = ht.bootstrap_pvalue(kin_v, fit_v, n_boot=args.boot, seed=21)
    fit_h["p_boot"] = ht.bootstrap_pvalue(kin_h, fit_h, n_boot=args.boot, seed=22)
    alts_v = ht.compare(kin_v, fit_v)
    stats["villains"] = {
        "crawled": meta["crawled"][:10], "category": meta["category"], "category_members": meta["category_members"],
        "category_redirects": meta["category_redirects"], "category_lists": meta["category_lists"],
        "villains_kept": meta["villains_kept"], "dropped": meta["dropped"],
        "n": H.number_of_nodes(), "m": H.number_of_edges(), "sides": meta["sides"], "both": meta["both"],
        "isolates": nx.number_of_isolates(H),
        "hero_hero_jaccard": meta["hero_hero_jaccard"], "hero_hero_live": meta["hero_hero_live"],
        "hero_hero_shared": meta["hero_hero_shared"],
        "pairs": dict(pairs), "out_share": out_share,
        "in_share": {a: round(sum(kinH[n] for n in H if s[n] == a) / H.number_of_edges(), 3) for a in "HVB"},
        "node_share": {a: round(sum(1 for n in H if s[n] == a) / H.number_of_nodes(), 3) for a in "HVB"},
        "top_in": [(pretty(n), kinH[n], s[n]) for n in sorted(H, key=lambda n: -kinH[n])[:12]],
        "top_villains": [(pretty(n), kinH[n]) for n in sorted((n for n in H if s[n] == "V"), key=lambda n: -kinH[n])[:12]],
        "most_linked_by_villains": [(pretty(n), c, s[n]) for n, c in linked_by_villains.most_common(10)],
        "most_linked_by_heroes": [(pretty(n), c, s[n]) for n, c in linked_by_heroes.most_common(10)],
        "villains_most_linked_by_heroes": [(pretty(n), c) for n, c in linked_by_heroes.most_common() if s[n] == "V"][:8],
        "nemesis_count": len(nemesis),
        "nemesis_top": [(pretty(u) if s[u] == "H" else pretty(v), pretty(v) if s[u] == "H" else pretty(u))
                        for u, v in nemesis[:14]],
        "spidey_in": kinH["Spider-Man"], "spidey_in_from_villains": linked_by_villains["Spider-Man"],
        "friendship": {"person_mean": round(fpH["person_mean"], 2), "friend_mean": round(fpH["friend_mean"], 2),
                       "p_friend_at_least": round(fpH["p_friend_at_least"], 3),
                       "kings": [(pretty(u), int(UH.degree(u)), s[u]) for u in fpH["kings"]],
                       "top_friends": [(pretty(v), round(100 * p, 2), s[v]) for v, p in fpH["p_friend"].most_common(8)]},
        "fit_villains_in": {"kmin": fit_v["kmin"], "alpha": round(fit_v["alpha"], 2), "alpha_se": round(fit_v["alpha_se"], 2),
                            "n_tail": fit_v["n_tail"], "p_boot": round(fit_v["p_boot"], 3), "alternatives": alts_v},
        "fit_heroes_in": {"kmin": fit_h["kmin"], "alpha": round(fit_h["alpha"], 2), "alpha_se": round(fit_h["alpha_se"], 2),
                          "n_tail": fit_h["n_tail"], "p_boot": round(fit_h["p_boot"], 3)},
        "max_out_villain": max((H.out_degree(n), pretty(n)) for n in H if s[n] == "V"),
    }
    res_c = run_nulls(H, combined_quantities, lambda g, r: directed_swap(g, 10 * g.number_of_edges(), r),
                      lambda g, r: gnm_like(g, r, directed=True), args.shuffles, rng, "heroes + villains")
    stats["villains"]["shuffle"] = strip_samples(res_c)
    fig_villains_network(H, stats)
    fig_villains_mixing(H, res_c, fit_v, fit_h, stats)

    (DATA / "stats_week2.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=float),
                                           encoding="utf-8")
    print("\nwrote data/stats_week2.json and", sorted(p.name for p in FIGS.glob("week2_*.png")))


if __name__ == "__main__":
    main()
