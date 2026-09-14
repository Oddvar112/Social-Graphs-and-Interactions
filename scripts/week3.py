"""
Week 3: who matters, and compared to what.

Runs every 🔬 tool exercise from week 3 on the frozen snapshot and writes the
figures and numbers the week 3 post quotes.

  3.3  distances: <d>, diameter, radius, centre, the distance distribution
       against G(n,m) and against degree-preserving shuffles, the directed
       version, and a hand-written BFS validated against networkx
  3.5  degree / closeness / harmonic / betweenness, top tens, the
       betweenness-vs-degree scatter, and what happens when you keep arrows
  3.7  PageRank against in-degree, plus a hand-written power iteration
       validated to six decimals for three values of alpha
  3.8  betweenness and closeness z-scores against 200 degree-preserving
       shuffles: who is surprising once you control for degree
  3.9  degree assortativity, k_nn(k), its shuffle z-score, and the directed
       out->in correlations
  3.10 cliques
  3.12 the go-nuts piece: dismantling the network in different orders

Outputs
  assets/figures/week3_*.png
  data/week3.json               every number the post quotes
  data/week3_dismantle.js       removal orders for the DISMANTLE explorable

Run:  python scripts/week3.py          (about two minutes; the shuffles dominate)
"""

import collections
import json
import math
import pathlib
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyse import (BG, CYAN, DIM, FG, GOLD, GREEN, GRID, PURPLE, RED,  # noqa: E402
                     load_graph, place_labels, pretty, style)

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGS = ROOT / "assets" / "figures"
SHUFFLES = 200
SEED = 2026
rng = np.random.default_rng(SEED)
random.seed(SEED)

OUT = {}          # everything the post quotes ends up in here
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def giant_undirected(G):
    U = G.to_undirected()
    return U.subgraph(max(nx.connected_components(U), key=len)).copy()


def dist_matrix(H, nodes):
    """All-pairs shortest paths via scipy, which is ~50x faster than looping
    networkx BFS when you need it hundreds of times."""
    A = nx.to_scipy_sparse_array(H, nodelist=nodes, dtype=np.int8, format="csr")
    return shortest_path(A, directed=H.is_directed(), unweighted=True)


def mean_distance(H):
    """<d> over the giant component of H (undirected)."""
    gc = H.subgraph(max(nx.connected_components(H), key=len))
    D = dist_matrix(gc, list(gc))
    iu = np.triu_indices_from(D, k=1)
    return float(D[iu].mean()), gc.number_of_nodes(), D


def degree_preserving_shuffle(H, seed):
    """Double-edge swap, ten swaps per edge. Same degree sequence, wiring
    randomised: the null every 'compared to what' question in week 3 uses."""
    S = nx.Graph(H)
    nx.double_edge_swap(S, nswap=10 * S.number_of_edges(),
                        max_tries=100 * S.number_of_edges(), seed=seed)
    return S


def top(d, k=10, reverse=True):
    return sorted(d.items(), key=lambda kv: -kv[1] if reverse else kv[1])[:k]


def zscore(real, samples):
    s = np.asarray(samples, dtype=float)
    sd = s.std()
    return float((real - s.mean()) / sd) if sd > 0 else float("nan")


# ---------------------------------------------------------------------------
# 3.3  distances
# ---------------------------------------------------------------------------
def section_distances(G):
    log("3.3 distances")
    gc = giant_undirected(G)
    nodes = list(gc)
    n_und, m_und = G.to_undirected().number_of_nodes(), G.to_undirected().number_of_edges()
    D = dist_matrix(gc, nodes)
    iu = np.triu_indices_from(D, k=1)
    dists = D[iu].astype(int)

    ecc = D.max(axis=1)
    diameter, radius = int(ecc.max()), int(ecc.min())
    centre = [pretty(nodes[i]) for i in np.where(ecc == radius)[0]]
    d_real = float(dists.mean())

    hist_real = np.bincount(dists, minlength=diameter + 1)[1:] / len(dists)

    # G(n,m): same n and m as the UNDIRECTED network, i.e. what week 2's table used.
    er_d, er_gc, er_hist = [], [], []
    for _ in range(SHUFFLES):
        H = nx.gnm_random_graph(n_und, m_und, seed=int(rng.integers(2**31)))
        d, size, DD = mean_distance(H)
        er_d.append(d); er_gc.append(size)
        v = DD[np.triu_indices_from(DD, k=1)].astype(int)
        er_hist.append(np.bincount(v, minlength=12)[1:12] / len(v))

    # Degree-preserving shuffles of the giant component.
    cm_d, cm_gc, cm_hist = [], [], []
    for _ in range(SHUFFLES):
        S = degree_preserving_shuffle(gc, int(rng.integers(2**31)))
        d, size, DD = mean_distance(S)
        cm_d.append(d); cm_gc.append(size)
        v = DD[np.triu_indices_from(DD, k=1)].astype(int)
        cm_hist.append(np.bincount(v, minlength=12)[1:12] / len(v))

    er_d, cm_d = np.array(er_d), np.array(cm_d)
    k_mean = 2 * m_und / n_und
    OUT["distances"] = {
        "n_undirected": n_und, "m_undirected": m_und, "mean_degree_undirected": round(k_mean, 2),
        "giant": gc.number_of_nodes(),
        "mean_distance": round(d_real, 3), "diameter": diameter, "radius": radius, "centre": centre,
        "share_at_2_or_3": round(float(((dists == 2) | (dists == 3)).mean()), 3),
        "ln_n_over_ln_k": round(math.log(n_und) / math.log(k_mean), 3),
        "gnm": {"mean": round(float(er_d.mean()), 3), "sd": round(float(er_d.std()), 3),
                "giant_mean": round(float(np.mean(er_gc)), 1),
                "z_real": round(zscore(d_real, er_d), 2)},
        "shuffled": {"mean": round(float(cm_d.mean()), 3), "sd": round(float(cm_d.std()), 3),
                     "giant_mean": round(float(np.mean(cm_gc)), 1),
                     "z_real": round(zscore(d_real, cm_d), 2)},
        "clustering_real": round(nx.average_clustering(gc), 3),
        "clustering_gnm": round(float(np.mean([nx.average_clustering(nx.gnm_random_graph(n_und, m_und, seed=s))
                                                for s in range(20)])), 3),
    }

    # ---- figure: distance distribution, real vs the two nulls
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    xs = np.arange(1, 12)
    er_h = np.array(er_hist); cm_h = np.array(cm_hist)
    ax.bar(xs[:len(hist_real)] - 0.27, hist_real, width=0.27, color=GOLD, label=f"Marvel  ⟨d⟩ = {d_real:.2f}")
    ax.bar(xs, er_h.mean(0), width=0.27, color=DIM, alpha=.9, label=f"G(n,m)  ⟨d⟩ = {er_d.mean():.2f} ± {er_d.std():.2f}")
    ax.errorbar(xs, er_h.mean(0), yerr=er_h.std(0), fmt="none", ecolor=FG, elinewidth=.8, capsize=2)
    ax.bar(xs + 0.27, cm_h.mean(0), width=0.27, color=CYAN, alpha=.9, label=f"degree-preserving shuffle  ⟨d⟩ = {cm_d.mean():.2f} ± {cm_d.std():.2f}")
    ax.errorbar(xs + 0.27, cm_h.mean(0), yerr=cm_h.std(0), fmt="none", ecolor=FG, elinewidth=.8, capsize=2)
    ax.set_xlim(0.3, 8.7)
    ax.set_xticks(range(1, 9))
    style(ax, "Distance between pairs of characters, giant component", "shortest path length d", "share of pairs")
    ax.legend(labelcolor=FG, fontsize=8.5, loc="upper right")

    ax = axes[1]
    ax.hist(er_d, bins=24, color=DIM, alpha=.85, label="G(n,m)")
    ax.hist(cm_d, bins=24, color=CYAN, alpha=.85, label="degree-preserving shuffle")
    ax.axvline(d_real, color=GOLD, lw=2.2, label=f"Marvel {d_real:.3f}")
    style(ax, f"⟨d⟩ over {SHUFFLES} draws of each null", "⟨d⟩", "draws")
    ax.legend(labelcolor=FG, fontsize=8.5)
    ax.text(0.03, 0.96, f"z vs G(n,m) = {OUT['distances']['gnm']['z_real']:+.1f}\n"
                        f"z vs shuffle = {OUT['distances']['shuffled']['z_real']:+.1f}",
            transform=ax.transAxes, va="top", fontsize=9, color=FG,
            bbox=dict(boxstyle="round,pad=.45", fc="#111830", ec=GRID))
    fig.suptitle("2.67 versus 2.8 is not the interesting comparison",
                 color=FG, fontsize=13, fontweight="bold", x=0.012, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "week3_distances.png", bbox_inches="tight")
    plt.close(fig)

    # ---- directed distances
    scc = max(nx.strongly_connected_components(G), key=len)
    Dd = dist_matrix(G, list(G))
    finite = np.isfinite(Dd) & ~np.eye(len(G), dtype=bool)
    OUT["directed"] = {
        "largest_scc": len(scc),
        "reachable_pairs_share": round(float(finite.mean()), 3),
        "mean_directed_distance": round(float(Dd[finite].mean()), 3),
        "reachable_pairs": int(finite.sum()),
        "all_ordered_pairs": int(len(G) * (len(G) - 1)),
    }

    # Pairs with d_AB = 1 and d_BA >= 5: a link that takes five hops to answer.
    gl = list(G)
    idx = {v: i for i, v in enumerate(gl)}
    asym = []
    for a, b in G.edges():
        back = Dd[idx[b], idx[a]]
        if back >= 5:
            asym.append((pretty(a), pretty(b), int(back) if np.isfinite(back) else None))
    asym.sort(key=lambda t: -(t[2] or 999))
    OUT["directed"]["one_way_links"] = asym[:12]
    OUT["directed"]["one_way_count"] = len(asym)
    OUT["directed"]["one_way_unreachable"] = sum(1 for t in asym if t[2] is None)


# ---------------------------------------------------------------------------
# 3.3.6  BFS by hand, validated
# ---------------------------------------------------------------------------
def my_bfs(adj, src):
    """Ring by ring. dist[v] = number of hops from src, -1 if unreachable."""
    dist = {src: 0}
    frontier = [src]
    ring = 0
    while frontier:
        ring += 1
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = ring
                    nxt.append(v)
        frontier = nxt
    return dist


def section_bfs_validation(G):
    log("3.3.6 hand-written BFS vs networkx")
    gc = giant_undirected(G)
    adj = {u: list(gc[u]) for u in gc}
    sample = random.Random(SEED).sample(list(gc), 20)
    mismatches = 0
    for s in sample:
        mine = my_bfs(adj, s)
        ref = nx.single_source_shortest_path_length(gc, s)
        if mine != ref:
            mismatches += 1
    OUT["bfs_validation"] = {"sources_checked": len(sample), "mismatches": mismatches}


# ---------------------------------------------------------------------------
# 3.5  centralities
# ---------------------------------------------------------------------------
def section_centralities(G):
    log("3.5 centralities")
    gc = giant_undirected(G)
    deg = dict(gc.degree())
    clo = nx.closeness_centrality(gc)
    har = nx.harmonic_centrality(gc)
    bet = nx.betweenness_centrality(gc, normalized=True)
    meas = {"degree": deg, "closeness": clo, "harmonic": har, "betweenness": bet}

    tops = {k: [(pretty(n), round(float(v), 4)) for n, v in top(v_, 10)] for k, v_ in meas.items()}
    top_sets = [set(n for n, _ in top(v_, 10)) for v_ in meas.values()]
    in_all_four = sorted(pretty(n) for n in set.intersection(*top_sets))

    def rank_of(d, node):
        order = [n for n, _ in sorted(d.items(), key=lambda kv: -kv[1])]
        return order.index(node) + 1 if node in order else None

    named = {}
    for nid in ("Hercules_(Marvel_Comics)", "Black_Widow_(Natasha_Romanova)", "Rockman_(comics)", "Spider-Man"):
        if nid in gc:
            named[pretty(nid)] = {
                "degree": deg[nid], "degree_rank": rank_of(deg, nid),
                "betweenness": round(bet[nid], 4), "betweenness_rank": rank_of(bet, nid),
                "closeness": round(clo[nid], 3), "closeness_rank": rank_of(clo, nid),
                "neighbours": sorted(pretty(v) for v in gc[nid])[:12],
            }

    # ---- figure: betweenness vs degree, log axes, outliers labelled
    nodes = list(gc)
    x = np.array([deg[v] for v in nodes], dtype=float)
    y = np.array([bet[v] for v in nodes], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6.6))
    ax.scatter(x, np.maximum(y, 1e-5), s=22 + 2.2 * x, c=np.log1p(x), cmap="plasma",
               alpha=.85, edgecolor=BG, lw=.4, zorder=3)
    # fit b ~ k^a in log space to define "outlier"
    mask = y > 0
    a, b0 = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    xs = np.linspace(1, x.max(), 50)
    ax.plot(xs, np.exp(b0) * xs ** a, ls="--", color=RED, lw=1.3, label=f"fit  b ∝ k^{a:.2f}")
    resid = np.log(np.maximum(y, 1e-5)) - (b0 + a * np.log(x))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(8e-6, 0.4)
    # Label the top of the cloud and the biggest residuals above the fit,
    # most important first, dropping anything that would overlap.
    picks = list(dict.fromkeys(list(np.argsort(-y)[:5]) + list(np.argsort(-resid)[:10])))
    place_labels(ax, [(x[i], max(y[i], 1e-5), pretty(nodes[i]), FG, 8.8) for i in picks], fig)
    style(ax, "Betweenness against degree, undirected giant component", "degree k", "betweenness (normalised)")
    ax.legend(labelcolor=FG, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "week3_betweenness_vs_degree.png", bbox_inches="tight")
    plt.close(fig)

    # ---- keep the arrows: rank shifts
    scc = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
    d_har = nx.harmonic_centrality(scc)
    d_bet = nx.betweenness_centrality(scc, normalized=True)
    shifts = []
    for nid in scc:
        if nid in gc:
            shifts.append((pretty(nid),
                           rank_of(bet, nid), rank_of(d_bet, nid),
                           rank_of(har, nid), rank_of(d_har, nid)))
    shifts.sort(key=lambda t: -abs((t[1] or 0) - (t[2] or 0)))
    OUT["centralities"] = {
        "top10": tops, "in_all_four_top10": in_all_four, "named": named,
        "betweenness_degree_fit_exponent": round(float(a), 2),
        "biggest_betweenness_rank_shifts_when_directed": [
            {"name": t[0], "undirected_rank": t[1], "directed_rank": t[2]} for t in shifts[:8]],
        "directed_top10_betweenness": [(pretty(n), round(v, 4)) for n, v in top(d_bet, 10)],
        "directed_top10_harmonic": [(pretty(n), round(v, 2)) for n, v in top(d_har, 10)],
    }
    return gc, deg, clo, bet


# ---------------------------------------------------------------------------
# 3.7  PageRank
# ---------------------------------------------------------------------------
def my_pagerank(G, alpha=0.85, tol=1e-12, max_iter=1000):
    """Power iteration. Dangling nodes (no out-links) hand their whole score
    to everyone uniformly, exactly as networkx does by default, so the two
    can be compared to the last decimal."""
    nodes = list(G)
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    out_deg = np.array([G.out_degree(v) for v in nodes], dtype=float)
    dangling = out_deg == 0
    # column-stochastic transition matrix M[j, i] = 1/out(i) if i -> j
    M = np.zeros((n, n))
    for u, v in G.edges():
        M[idx[v], idx[u]] += 1.0 / out_deg[idx[u]]
    pr = np.full(n, 1.0 / n)
    for it in range(max_iter):
        leak = alpha * pr[dangling].sum()            # dangling mass, redistributed uniformly
        new = alpha * M @ pr + (leak + (1 - alpha)) / n
        new /= new.sum()
        if np.abs(new - pr).sum() < tol:
            pr = new
            break
        pr = new
    return dict(zip(nodes, pr)), it + 1


def section_pagerank(G):
    log("3.7 PageRank")
    pr = nx.pagerank(G, alpha=0.85)
    indeg = dict(G.in_degree())
    top_pr = [pretty(n) for n, _ in top(pr, 10)]
    top_in = [pretty(n) for n, _ in top(indeg, 10)]
    only_pr = [n for n in top_pr if n not in top_in]
    only_in = [n for n in top_in if n not in top_pr]

    # who sends the PageRank-only characters their score?
    sources = {}
    inv = {pretty(n): n for n in G}
    for name in only_pr:
        nid = inv[name]
        preds = sorted(G.predecessors(nid), key=lambda u: -pr[u] / max(G.out_degree(u), 1))[:4]
        sources[name] = [f"{pretty(u)} (PR {pr[u]:.4f}, out {G.out_degree(u)})" for u in preds]

    validation = {}
    for alpha in (0.5, 0.85, 0.95):
        mine, iters = my_pagerank(G, alpha=alpha)
        ref = nx.pagerank(G, alpha=alpha, tol=1e-12, max_iter=1000)
        max_abs = max(abs(mine[v] - ref[v]) for v in G)
        validation[str(alpha)] = {"max_abs_diff": float(f"{max_abs:.2e}"), "agree_to_6dp": bool(max_abs < 5e-7),
                                  "iterations": iters}

    OUT["pagerank"] = {
        "top10_pagerank": [(pretty(n), round(v, 4)) for n, v in top(pr, 10)],
        "top10_indegree": [(pretty(n), v) for n, v in top(indeg, 10)],
        "only_in_pagerank_top10": only_pr, "only_in_indegree_top10": only_in,
        "sources_of_pagerank_only": sources,
        "dangling_nodes": int(sum(1 for v in G if G.out_degree(v) == 0)),
        "validation": validation,
    }


# ---------------------------------------------------------------------------
# 3.8  brokers, compared to what
# ---------------------------------------------------------------------------
def section_shuffle_test(gc, deg, clo, bet):
    log(f"3.8 betweenness and closeness against {SHUFFLES} degree-preserving shuffles")
    nodes = list(gc)
    B = np.zeros((SHUFFLES, len(nodes)))
    C = np.zeros((SHUFFLES, len(nodes)))
    for s in range(SHUFFLES):
        S = degree_preserving_shuffle(gc, int(rng.integers(2**31)))
        b = nx.betweenness_centrality(S, normalized=True)
        c = nx.closeness_centrality(S)
        B[s] = [b[v] for v in nodes]
        C[s] = [c[v] for v in nodes]
        if s % 50 == 49:
            log(f"   shuffle {s + 1}/{SHUFFLES}")

    real_b = np.array([bet[v] for v in nodes]); real_c = np.array([clo[v] for v in nodes])
    k = np.array([deg[v] for v in nodes], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        zb = (real_b - B.mean(0)) / B.std(0)
        zc = (real_c - C.mean(0)) / C.std(0)
    zb = np.nan_to_num(zb, nan=0.0, posinf=0.0, neginf=0.0)
    zc = np.nan_to_num(zc, nan=0.0, posinf=0.0, neginf=0.0)

    def rows(z, real, mean, sd, sel):
        return [{"name": pretty(nodes[i]), "degree": int(k[i]), "real": round(float(real[i]), 4),
                 "null_mean": round(float(mean[i]), 4), "null_sd": round(float(sd[i]), 4),
                 "z": round(float(z[i]), 1)} for i in sel]

    hi_b = np.argsort(-zb)[:12]; lo_b = np.argsort(zb)[:8]
    hi_c = np.argsort(-zc)[:8]; lo_c = np.argsort(zc)[:8]
    OUT["shuffle_test"] = {
        "shuffles": SHUFFLES,
        "betweenness_most_surprising_high": rows(zb, real_b, B.mean(0), B.std(0), hi_b),
        "betweenness_most_surprising_low": rows(zb, real_b, B.mean(0), B.std(0), lo_b),
        "closeness_most_surprising_high": rows(zc, real_c, C.mean(0), C.std(0), hi_c),
        "closeness_most_surprising_low": rows(zc, real_c, C.mean(0), C.std(0), lo_c),
        "betweenness_z_sd": round(float(zb.std()), 2), "closeness_z_sd": round(float(zc.std()), 2),
        "share_abs_z_over_2_betweenness": round(float((np.abs(zb) > 2).mean()), 3),
        "share_abs_z_over_2_closeness": round(float((np.abs(zc) > 2).mean()), 3),
    }
    for nid in ("Hercules_(Marvel_Comics)", "Black_Widow_(Natasha_Romanova)", "Rockman_(comics)", "Spider-Man"):
        if nid in nodes:
            i = nodes.index(nid)
            OUT["shuffle_test"][pretty(nid)] = {
                "degree": int(k[i]), "betweenness": round(float(real_b[i]), 4),
                "null_mean": round(float(B.mean(0)[i]), 4), "null_sd": round(float(B.std(0)[i]), 4),
                "z": round(float(zb[i]), 1),
                "betweenness_rank_real": int((real_b > real_b[i]).sum() + 1),
                "z_rank": int((zb > zb[i]).sum() + 1),
            }

    # ---- figure: z against degree, both measures
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
    for ax, z, title in ((axes[0], zb, "betweenness"), (axes[1], zc, "closeness")):
        ax.axhspan(-2, 2, color=DIM, alpha=.12, lw=0)
        ax.axhline(0, color=DIM, lw=.8)
        ax.scatter(k, z, s=18 + 1.6 * k, c=z, cmap="coolwarm", vmin=-6, vmax=6, edgecolor=BG, lw=.4, zorder=3)
        sel = list(np.argsort(-z)[:6]) + list(np.argsort(z)[:4])
        for i in sel:
            ax.annotate(pretty(nodes[i]), (k[i], z[i]), textcoords="offset points", xytext=(6, 4),
                        fontsize=8.3, color=FG, fontweight="bold")
        ax.set_xscale("log")
        style(ax, f"{title}: z-score against degree-preserving shuffles", "degree k", "z")
    fig.suptitle("Surprising, once you control for degree", color=FG, fontsize=13, fontweight="bold",
                 x=0.012, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "week3_centrality_z.png", bbox_inches="tight")
    plt.close(fig)
    return zb


# ---------------------------------------------------------------------------
# 3.9  mixing
# ---------------------------------------------------------------------------
def section_mixing(G, gc):
    log("3.9 assortativity")
    r = nx.degree_assortativity_coefficient(gc)
    knn = nx.average_degree_connectivity(gc)
    r_null = [nx.degree_assortativity_coefficient(degree_preserving_shuffle(gc, int(rng.integers(2**31))))
              for _ in range(SHUFFLES)]
    # directed correlations: which end of an arrow predicts which
    src_out = np.array([G.out_degree(u) for u, v in G.edges()], dtype=float)
    src_in = np.array([G.in_degree(u) for u, v in G.edges()], dtype=float)
    tgt_out = np.array([G.out_degree(v) for u, v in G.edges()], dtype=float)
    tgt_in = np.array([G.in_degree(v) for u, v in G.edges()], dtype=float)
    OUT["mixing"] = {
        "r_real": round(float(r), 3),
        "r_shuffled_mean": round(float(np.mean(r_null)), 3), "r_shuffled_sd": round(float(np.std(r_null)), 3),
        "z": round(zscore(r, r_null), 1),
        "directed_out_in": round(float(np.corrcoef(src_out, tgt_in)[0, 1]), 3),
        "directed_in_in": round(float(np.corrcoef(src_in, tgt_in)[0, 1]), 3),
        "directed_out_out": round(float(np.corrcoef(src_out, tgt_out)[0, 1]), 3),
        "directed_in_out": round(float(np.corrcoef(src_in, tgt_out)[0, 1]), 3),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = np.array(sorted(knn)); ys = np.array([knn[k] for k in ks])
    ax.scatter(ks, ys, s=36, color=GOLD, edgecolor=BG, zorder=3, label="Marvel")
    S = degree_preserving_shuffle(gc, 7); knn_s = nx.average_degree_connectivity(S)
    ks2 = np.array(sorted(knn_s)); ax.plot(ks2, [knn_s[k] for k in ks2], color=CYAN, lw=1.4, alpha=.9,
                                          label="one degree-preserving shuffle")
    ax.set_xscale("log"); ax.set_yscale("log")
    style(ax, f"k_nn(k): r = {r:.3f} real, {np.mean(r_null):.3f} ± {np.std(r_null):.3f} shuffled  (z = {zscore(r, r_null):+.1f})",
          "degree k", "mean degree of neighbours")
    ax.legend(labelcolor=FG, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "week3_knn.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3.10  cliques
# ---------------------------------------------------------------------------
def section_cliques(gc):
    log("3.10 cliques")
    cl = list(nx.find_cliques(gc))
    sizes = collections.Counter(len(c) for c in cl)
    omega = max(sizes)
    biggest = [sorted(pretty(v) for v in c) for c in cl if len(c) == omega]
    tri = sum(nx.triangles(gc).values()) // 3
    # count all (not just maximal) k-cliques for k = 5
    five = sum(math.comb(len(c), 5) for c in cl)   # overcounts shared sub-cliques; report maximal separately
    OUT["cliques"] = {
        "maximal_cliques": len(cl), "clique_number": omega, "largest": biggest,
        "maximal_by_size": {str(k): v for k, v in sorted(sizes.items())},
        "triangles": tri,
        "min_degree_for_8_clique": 7,
        "nodes_with_degree_ge_7": int(sum(1 for _, d in gc.degree() if d >= 7)),
        "nodes_total": gc.number_of_nodes(),
    }


# ---------------------------------------------------------------------------
# 3.12  go nuts: dismantle it
# ---------------------------------------------------------------------------
def dismantle(gc, order):
    """Remove nodes in `order`; return giant-component size after each removal."""
    H = gc.copy()
    sizes = [H.number_of_nodes()]
    for v in order:
        H.remove_node(v)
        sizes.append(max((len(c) for c in nx.connected_components(H)), default=0))
    return sizes


def adaptive_order(gc, measure, steps):
    """Recompute the measure after every removal. Expensive, but it is the
    honest version of 'remove the most central node'."""
    H = gc.copy(); order = []
    for _ in range(steps):
        if H.number_of_edges() == 0:
            break
        sc = measure(H)
        v = max(sc, key=sc.get)
        order.append(v); H.remove_node(v)
    return order


def section_dismantle(G, gc, deg, bet, zb):
    log("3.12 dismantling")
    nodes = list(gc); n = len(nodes)
    pr = nx.pagerank(G, alpha=0.85)
    clo = nx.closeness_centrality(gc)
    static = {
        "degree": sorted(nodes, key=lambda v: -deg[v]),
        "betweenness": sorted(nodes, key=lambda v: -bet[v]),
        "pagerank": sorted(nodes, key=lambda v: -pr[v]),
        "closeness": sorted(nodes, key=lambda v: -clo[v]),
    }
    steps = 80
    adaptive = {
        "betweenness (recomputed)": adaptive_order(gc, lambda H: nx.betweenness_centrality(H), steps),
        "degree (recomputed)": adaptive_order(gc, lambda H: dict(H.degree()), steps),
    }
    curves = {k: dismantle(gc, v[:steps]) for k, v in {**static, **adaptive}.items()}
    rnd = np.array([dismantle(gc, random.Random(s).sample(nodes, steps)) for s in range(30)])
    curves["random"] = rnd.mean(0).tolist()

    # how many removals to halve the giant component
    def to_half(c):
        for i, s in enumerate(c):
            if s <= n / 2:
                return i
        return None
    halves = {k: to_half(c) for k, c in curves.items()}
    # area under curve: robustness R (Schneider et al.)
    R = {k: round(float(np.sum(np.array(c) / n) / n), 4) for k, c in curves.items()}

    OUT["dismantle"] = {
        "steps": steps, "giant_start": n,
        "removals_to_halve": halves, "robustness_R": R,
        "orders_first_12": {k: [pretty(v) for v in v_[:12]] for k, v_ in {**static, **adaptive}.items()},
        "order_overlap_first_20_betweenness_vs_adaptive": len(
            set(static["betweenness"][:20]) & set(adaptive["betweenness (recomputed)"][:20])),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    cols = {"degree": GOLD, "betweenness": RED, "pagerank": PURPLE, "closeness": GREEN,
            "betweenness (recomputed)": "#ff9a3c", "degree (recomputed)": "#ffe28a", "random": DIM}
    for k, c in curves.items():
        xs = np.arange(len(c)) / n
        when = f"halved after {halves[k]}" if halves[k] is not None else f"not halved in {steps}"
        ax.plot(xs, np.array(c) / n, color=cols[k], lw=2.2 if "recomputed" in k else 1.6,
                ls="--" if "recomputed" in k else "-", label=f"{k}  ({when})")
    ax.axhline(0.5, color=FG, lw=.7, alpha=.5)
    style(ax, "Remove characters one by one: how fast does the giant component collapse?",
          "fraction of characters removed", "giant component, fraction of original")
    ax.legend(labelcolor=FG, fontsize=8.6, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "week3_dismantle.png", bbox_inches="tight")
    plt.close(fig)

    # ---- data for the explorable: orders as indices into the week-1 node list
    js_src = (ROOT / "data" / "marvel_week1.js").read_text(encoding="utf-8")
    week1 = json.loads(js_src[js_src.index("{"):js_src.rstrip().rstrip(";").rindex("}") + 1])
    by_id = {nd["id"]: i for i, nd in enumerate(week1["nodes"])}
    payload = {
        "giant": [by_id[v] for v in nodes],
        "orders": {k: [by_id[v] for v in v_[:steps]] for k, v_ in {**static, **adaptive}.items()},
        "random_curve": curves["random"],
        "steps": steps,
    }
    (ROOT / "data" / "week3_dismantle.js").write_text(
        "// Generated by scripts/week3.py. Removal orders for the DISMANTLE explorable.\n"
        "window.DISMANTLE = " + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def main():
    G = load_graph()
    section_distances(G)
    section_bfs_validation(G)
    gc, deg, clo, bet = section_centralities(G)
    section_pagerank(G)
    zb = section_shuffle_test(gc, deg, clo, bet)
    section_mixing(G, gc)
    section_cliques(gc)
    section_dismantle(G, gc, deg, bet, zb)

    (ROOT / "data" / "week3.json").write_text(json.dumps(OUT, indent=2, ensure_ascii=False), encoding="utf-8")
    log("wrote data/week3.json")
    d = OUT["distances"]
    print(f"\n<d> real {d['mean_distance']}  G(n,m) {d['gnm']['mean']}±{d['gnm']['sd']} (z {d['gnm']['z_real']})"
          f"  shuffle {d['shuffled']['mean']}±{d['shuffled']['sd']} (z {d['shuffled']['z_real']})")
    print("diameter", d["diameter"], "radius", d["radius"], "centre", d["centre"])
    print("in all four top tens:", OUT["centralities"]["in_all_four_top10"])
    print("most surprising betweenness:", [(r["name"], r["z"]) for r in OUT["shuffle_test"]["betweenness_most_surprising_high"][:6]])
    print("assortativity:", OUT["mixing"])
    print("clique number", OUT["cliques"]["clique_number"], OUT["cliques"]["largest"])
    print("halve giant after:", OUT["dismantle"]["removals_to_halve"])


if __name__ == "__main__":
    main()
