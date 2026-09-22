"""
Week 3 analysis: who matters, and why.

Everything the week-3 post quotes, on the frozen snapshot:
  1. Paths and distances: BFS numbers on the undirected giant component, and
     what keeping the arrows does to them (strongly connected core, who
     Spider-Man can actually reach, the most asymmetric pair we can find).
  2. Four answers to "who matters": degree, closeness/harmonic, betweenness,
     eigenvector and PageRank, with their top tens and how much they agree.
  3. Compared to what: betweenness under the degree-preserving shuffle, which
     is the test that separates real brokers (Hercules, Black Widow) from
     characters whose centrality is just their degree (Spider-Man, Black Cat).
  4. Cliques: triangles, the clique number, and who the biggest clique is.
  5. The KINGPIN cross-check: single-character damage, articulation points,
     the two greedy bots' shrink curves, and the shipped record hit-lists
     re-verified (weeks/week3/game/core.js is the file the game runs).

Outputs
  assets/figures/week3_centralities.png   betweenness against degree, the
                                          Krackhardt cases ringed and named
  assets/figures/week3_brokers.png        the shuffle test: betweenness
                                          z-scores against degree
  data/stats_week3.json                   every number the post quotes

Run:  python scripts/analyse_week3.py            (a few minutes; 200 shuffles)
      python scripts/analyse_week3.py --quick    (30 shuffles, for a look)
"""

import argparse
import collections
import csv
import itertools
import json
import pathlib
import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats as sstats

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGS = ROOT / "assets" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

BG = "#0a0f1f"
FG = "#e9edf9"
DIM = "#8d99bd"
GRID = "#26304f"
CYAN = "#5fe3ff"
GOLD = "#ffc93f"
RED = "#ff4b50"
GREEN = "#4ade80"
PURPLE = "#a678ff"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": GRID,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": GRID,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlepad": 12,
    "legend.frameon": False, "figure.dpi": 130,
})


def read_tsv(path, fieldnames=None):
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t", fieldnames=fieldnames))


def load_graph():
    G = nx.DiGraph()
    for row in read_tsv(ROOT / "week1_nodes.tsv"):
        G.add_node(row["node_id"], name=row["name"])
    for row in read_tsv(ROOT / "week1_edges.tsv", ["source", "target"]):
        G.add_edge(row["source"], row["target"])
    return G


def pretty(node_id):
    return re.sub(r"_", " ", re.sub(r"_\([^)]*\)$", "", node_id))


def style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, color=FG, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5)
    ax.grid(alpha=0.28, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def top10(d, names):
    return [[pretty(v), round(float(s), 4)] for v, s in
            sorted(d.items(), key=lambda kv: -kv[1])[:10]]


# ---------------------------------------------------------------------------
# 1. Paths and distances
# ---------------------------------------------------------------------------
def distances(G, Gg, stats):
    n = Gg.number_of_nodes()
    ecc = nx.eccentricity(Gg)
    dist_hist = collections.Counter()
    for _, dists in nx.all_pairs_shortest_path_length(Gg):
        dist_hist.update(dists.values())
    del dist_hist[0]
    pairs = sum(dist_hist.values())          # ordered pairs, symmetric anyway
    stats["giant"] = n
    stats["mean_distance"] = round(sum(d * c for d, c in dist_hist.items()) / pairs, 3)
    stats["diameter"] = max(ecc.values())
    stats["radius"] = min(ecc.values())
    stats["center"] = [pretty(v) for v, e in ecc.items() if e == min(ecc.values())]
    stats["frac_pairs_2_or_3"] = round((dist_hist[2] + dist_hist[3]) / pairs, 3)

    # Directed: the arrows change everything.
    scc = max(nx.strongly_connected_components(G), key=len)
    stats["largest_scc"] = len(scc)
    stats["reach_from_spiderman"] = len(nx.descendants(G, "Spider-Man"))
    reachable_pairs = sum(len(nx.descendants(G, v)) for v in G)
    stats["frac_ordered_pairs_reachable"] = round(
        reachable_pairs / (G.number_of_nodes() * (G.number_of_nodes() - 1)), 3)

    # The most asymmetric pair: one step there, as far back as it gets.
    best = None
    dist_from = {v: nx.single_source_shortest_path_length(G, v) for v in G}
    for a, b in G.edges():
        back = dist_from[b].get(a)
        if back is not None and (best is None or back > best[2]):
            best = (a, b, back)
    stats["asymmetric_pair"] = {
        "from": pretty(best[0]), "to": pretty(best[1]), "there": 1, "back": best[2]}


# ---------------------------------------------------------------------------
# 2. The centralities and how much they agree
# ---------------------------------------------------------------------------
def centralities(G, Gg, stats):
    deg = dict(Gg.degree())
    clo = nx.closeness_centrality(Gg)
    har = nx.harmonic_centrality(Gg)
    har = {v: s / (len(Gg) - 1) for v, s in har.items()}
    btw = nx.betweenness_centrality(Gg)
    eig = nx.eigenvector_centrality_numpy(Gg)
    pr = nx.pagerank(G, alpha=0.85)

    names = {"degree": deg, "closeness": clo, "harmonic": har,
             "betweenness": btw, "eigenvector": eig, "pagerank": pr}
    stats["top10"] = {k: top10(d, names) for k, d in names.items()}

    order = sorted(Gg)
    kv = [deg[v] for v in order]
    stats["spearman_vs_degree"] = {
        k: round(float(sstats.spearmanr(kv, [d[v] for v in order]).statistic), 3)
        for k, d in names.items() if k not in ("degree", "pagerank")}
    top_deg = {v for v, _ in sorted(deg.items(), key=lambda x: -x[1])[:10]}
    top_btw = {v for v, _ in sorted(btw.items(), key=lambda x: -x[1])[:10]}
    stats["top10_shared_degree_betweenness"] = len(top_deg & top_btw)

    stats["spiderman"] = {
        "closeness": round(clo["Spider-Man"], 3),
        "mean_distance": round(1 / clo["Spider-Man"], 3),
        "betweenness": round(btw["Spider-Man"], 4),
        "out_degree": G.out_degree("Spider-Man"),
    }
    stats["median_closeness"] = round(float(np.median(list(clo.values()))), 3)

    # PageRank's oddities: who ranks high on few in-links because of who links.
    pr_rank = {v: r + 1 for r, (v, _) in
               enumerate(sorted(pr.items(), key=lambda x: -x[1]))}
    inter = []
    for v in ("Black_Cat_(Marvel_Comics)", "Mayday_Parker"):
        if v in pr_rank:
            inter.append({"who": pretty(v), "pagerank_rank": pr_rank[v],
                          "in_degree": G.in_degree(v)})
    stats["pagerank_upsets"] = inter
    bc = set(Gg.neighbors("Black_Cat_(Marvel_Comics)"))
    stats["black_cat_shelter"] = {
        "degree": len(bc),
        "neighbors_shared_with_spiderman": len(bc & set(Gg.neighbors("Spider-Man"))),
        "spiderman_is_neighbor": "Spider-Man" in bc}
    stats["hulk"] = {"in": G.in_degree("Hulk"), "out": G.out_degree("Hulk")}
    stats["betsy"] = {"in": G.in_degree("Betsy_Braddock"),
                      "out": G.out_degree("Betsy_Braddock")}
    return deg, btw


# ---------------------------------------------------------------------------
# 3. Compared to what: the degree-preserving shuffle
# ---------------------------------------------------------------------------
def shuffle_test(Gg, deg, btw, n_shuffles, stats, rng_seed=20260914):
    m = Gg.number_of_edges()
    rng = random.Random(rng_seed)
    samples = collections.defaultdict(list)
    r_samples = []
    for i in range(n_shuffles):
        H = Gg.copy()
        nx.double_edge_swap(H, nswap=10 * m, max_tries=100 * m,
                            seed=rng.randrange(2**31))
        r_samples.append(nx.degree_assortativity_coefficient(H))
        for v, b in nx.betweenness_centrality(H).items():
            samples[v].append(b)
        print(f"  shuffle {i + 1}/{n_shuffles}", end="\r")
    print()

    z = {}
    ratio = {}
    for v, vals in samples.items():
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        z[v] = (btw[v] - mu) / sd if sd > 0 else 0.0
        ratio[v] = btw[v] / mu if mu > 0 else float("inf")

    def report(node_id):
        vals = samples[node_id]
        return {"who": pretty(node_id), "degree": deg[node_id],
                "real": round(btw[node_id], 4),
                "shuffle_mean": round(float(np.mean(vals)), 4),
                "shuffle_sd": round(float(np.std(vals)), 4),
                "z": round(z[node_id], 1)}

    stats["shuffles"] = n_shuffles
    stats["betweenness_vs_shuffle"] = [report(v) for v in (
        "Spider-Man", "Hercules_(Marvel_Comics)", "Black_Widow_(Natasha_Romanova)",
        "U.S._Agent", "Black_Cat_(Marvel_Comics)")]
    stats["surprising_brokers"] = sorted(
        (round(z[v], 1), pretty(v), deg[v]) for v in z if z[v] > 3)[::-1]
    stats["assortativity"] = {
        "real": round(nx.degree_assortativity_coefficient(Gg), 3),
        "shuffle_mean": round(float(np.mean(r_samples)), 3),
        "shuffle_sd": round(float(np.std(r_samples)), 3)}
    return z


# ---------------------------------------------------------------------------
# 4. Cliques
# ---------------------------------------------------------------------------
def cliques(Gg, stats):
    stats["triangles"] = sum(nx.triangles(Gg).values()) // 3
    maximal = list(nx.find_cliques(Gg))
    stats["maximal_cliques"] = len(maximal)
    biggest = max(maximal, key=len)
    stats["clique_number"] = len(biggest)
    # The largest clique is not unique: report all of them and their overlap.
    top = [c for c in maximal if len(c) == len(biggest)]
    common = set.intersection(*map(set, top))
    stats["largest_cliques"] = {
        "count": len(top),
        "common_core": sorted(pretty(v) for v in common),
        "guests": sorted({pretty(v) for c in top for v in c} - {pretty(v) for v in common}),
    }
    fives = set()
    for c in maximal:
        if len(c) >= 5:
            for sub in itertools.combinations(sorted(c), 5):
                fives.add(sub)
    stats["five_cliques"] = len(fives)


# ---------------------------------------------------------------------------
# 5. The KINGPIN cross-check
# ---------------------------------------------------------------------------
def kingpin(Gg, stats):
    giant = set(Gg)

    def final_giant(removed):
        H = Gg.copy()
        H.remove_nodes_from(removed)
        return max(len(c) for c in nx.connected_components(H))

    damage = sorted(((len(giant) - 1 - final_giant([v]), v) for v in giant),
                    reverse=True)
    stats["single_hit_damage"] = [
        {"who": pretty(v), "degree": Gg.degree(v), "cut_off": d}
        for d, v in damage[:5]]
    stats["articulation_points"] = len(list(nx.articulation_points(Gg)))

    def greedy(strategy, budget=8):
        H = Gg.copy()
        cur = max(nx.connected_components(H), key=len)
        curve = [len(cur)]
        for _ in range(budget):
            score = dict(H.degree()) if strategy == "degree" \
                else nx.betweenness_centrality(H, normalized=False)
            target = max(sorted(cur), key=lambda v: score[v])
            H.remove_node(target)
            cur = max(nx.connected_components(H), key=len)
            curve.append(len(cur))
        return curve

    stats["greedy_curves"] = {s: greedy(s) for s in ("degree", "betweenness")}

    # Re-verify the record hit-lists the game ships, straight from its code.
    core = (ROOT / "weeks/week3/game/core.js").read_text(encoding="utf-8")
    idx_to_node = {}
    data = (ROOT / "data/marvel_week1.js").read_text(encoding="utf-8")
    D = json.loads(data[data.index("{"):data.rstrip().rstrip(";").rindex("}") + 1])
    for i, node in enumerate(D["nodes"]):
        idx_to_node[i] = node["id"]
    records = {}
    for budget, giant_claim, ids in re.findall(
            r"(\d+):\s*\{\s*giant:\s*(\d+),\s*ids:\s*\[([\d,\s]+)\]", core):
        members = [idx_to_node[int(i)] for i in ids.split(",")]
        achieved = final_giant(members)
        assert achieved == int(giant_claim), \
            f"record {budget}: core.js claims {giant_claim}, networkx says {achieved}"
        records[budget] = {"core": achieved,
                           "crew": sorted(pretty(v) for v in members)}
    stats["records"] = records


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
LABELLED = {
    "Spider-Man": "Spider-Man", "Hulk": "Hulk",
    "Hercules_(Marvel_Comics)": "Hercules",
    "Black_Widow_(Natasha_Romanova)": "Black Widow",
    "U.S._Agent": "U.S. Agent",
    "Black_Cat_(Marvel_Comics)": "Black Cat",
    "Rockman_(character)": "Rockman",
}


def fig_centralities(Gg, deg, btw):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ks = np.array([deg[v] for v in Gg])
    bs = np.array([btw[v] for v in Gg])
    floor = 6e-5                                     # rail for betweenness zero
    ax.scatter(ks[bs > 0], bs[bs > 0], s=22, c=CYAN, alpha=0.55, lw=0)
    ax.scatter(ks[bs == 0], np.full((bs == 0).sum(), floor), s=14, c=DIM,
               alpha=0.5, lw=0)
    offsets = {"Spider-Man": (-8, 6, "right"), "Hercules_(Marvel_Comics)": (2, 10, "left"),
               "Black_Widow_(Natasha_Romanova)": (9, -3, "left"),
               "U.S._Agent": (-9, -6, "right"), "Black_Cat_(Marvel_Comics)": (9, -3, "left")}
    for v, label in LABELLED.items():
        colour = GOLD if v in ("Hercules_(Marvel_Comics)",
                               "Black_Widow_(Natasha_Romanova)", "U.S._Agent",
                               "Rockman_(character)") else \
            RED if v == "Black_Cat_(Marvel_Comics)" else FG
        y = max(btw[v], floor)
        dx, dy, ha = offsets.get(v, (7, 5, "left"))
        ax.scatter([deg[v]], [y], s=46, facecolor="none", edgecolor=colour, lw=1.6)
        ax.annotate(label, (deg[v], y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=9, color=colour, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(floor * 1.6, color=GRID, lw=0.8, ls="--")
    style(ax, "Betweenness against degree, one point per character",
          "degree (log)", "betweenness, pair-normalized (log; zeros on the rail)")
    fig.tight_layout()
    fig.savefig(FIGS / "week3_centralities.png", bbox_inches="tight")
    plt.close(fig)


def fig_brokers(Gg, deg, z):
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ks = np.array([deg[v] for v in Gg])
    zs = np.array([z[v] for v in Gg])
    ax.axhspan(-2, 2, color=GRID, alpha=0.35, lw=0)
    ax.axhline(0, color=DIM, lw=0.8)
    ax.scatter(ks, zs, s=22, c=CYAN, alpha=0.55, lw=0)
    offsets = {"Spider-Man": (-8, 8, "right"), "Hercules_(Marvel_Comics)": (2, -16, "left"),
               "Black_Widow_(Natasha_Romanova)": (9, 4, "left"),
               "U.S._Agent": (-9, 4, "right"), "Black_Cat_(Marvel_Comics)": (9, -4, "left"),
               "Rockman_(character)": (9, -3, "left")}
    for v, label in LABELLED.items():
        colour = GOLD if z[v] > 2 else RED if z[v] < -2 else FG
        dx, dy, ha = offsets.get(v, (7, 5, "left"))
        ax.scatter([deg[v]], [z[v]], s=46, facecolor="none", edgecolor=colour, lw=1.6)
        ax.annotate(label, (deg[v], z[v]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=9, color=colour, fontweight="bold")
    ax.set_xscale("log")
    style(ax, "The same betweenness, compared to what the degree sequence forces",
          "degree (log)",
          "z-score against the degree-preserving shuffle")
    ax.text(0.03, 0.55, "grey band: |z| < 2,\nbetweenness explained by degree",
            transform=ax.transAxes, ha="left", fontsize=8.5, color=DIM)
    fig.tight_layout()
    fig.savefig(FIGS / "week3_brokers.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="30 shuffles instead of 200")
    args = ap.parse_args()
    n_shuffles = 30 if args.quick else 200

    G = load_graph()
    U = G.to_undirected()
    Gg = U.subgraph(max(nx.connected_components(U), key=len)).copy()
    stats = {"settings": {"shuffles": n_shuffles, "quick": args.quick}}

    print("distances ...")
    distances(G, Gg, stats)
    print("centralities ...")
    deg, btw = centralities(G, Gg, stats)
    print(f"shuffle test ({n_shuffles} shuffles) ...")
    z = shuffle_test(Gg, deg, btw, n_shuffles, stats)
    print("cliques ...")
    cliques(Gg, stats)
    print("kingpin cross-check ...")
    kingpin(Gg, stats)
    print("figures ...")
    fig_centralities(Gg, deg, btw)
    fig_brokers(Gg, deg, z)

    out = ROOT / "data" / "stats_week3.json"
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
