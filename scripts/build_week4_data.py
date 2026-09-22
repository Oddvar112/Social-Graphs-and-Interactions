"""
Data for SYMPOSIUM (weeks/week4/game) and the numbers in the week 4 post.

Reads the course's frozen philosophers snapshot (week4_philosophers_nodes.tsv,
week4_philosophers_edges.tsv), builds the undirected weighted network the
course page uses (sum both directions), keeps its giant component, and writes
data/week4_symposium.js with:

  * the giant component: nodes with era, subfields, a short blurb, a fixed
    force-directed layout, degree and strength; edges with summed weights
  * Louvain, best of 20 seeds (unweighted, the way the course page runs it),
    with a hand-written name per community, and how often each philosopher
    changes community across the 20 runs
  * greedy modularity, the seating by century, and two null models
    (degree-preserving shuffle and G(n,m)) for the "compared to what" screen

Everything the game shows as a rival is recomputed by scripts/test_symposium_rules.py
against this file, so a stale file fails loudly.

Run:  python scripts/build_week4_data.py            (about a minute)
      python scripts/build_week4_data.py --png DIR  (also renders the layout to DIR)
"""
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
NODES = ROOT / "week4_philosophers_nodes.tsv"
EDGES = ROOT / "week4_philosophers_edges.tsv"
OUT = ROOT / "data" / "week4_symposium.js"
SEEDS = 20
DESC_MAX = 150

ERAS = [
    "centuries BC",
    "1st through 10th centuries",
    "11th through 14th centuries",
    "15th and 16th centuries",
    "17th century",
    "18th century",
    "19th century",
]
ERA_SHORT = ["BC", "1st-10th c.", "11th-14th c.", "15th-16th c.", "17th c.", "18th c.", "19th c."]

# Names for Louvain's communities, keyed by the member with the highest degree
# in that community (so the key survives a re-run as long as the community
# does). Anything not listed falls back to "the <top member> table".
NAMES = {
    "Aristotle": "the Scholastics",
    "Plato": "the Greeks",
    "Thomas Aquinas": "the Scholastics",
    "Augustine of Hippo": "the Church Fathers",
    "Immanuel Kant": "the Germans",
    "Georg Wilhelm Friedrich Hegel": "the Germans",
    "John Locke": "the British",
    "David Hume": "the British",
    "Avicenna": "the Islamic Golden Age",
    "Averroes": "the Islamic Golden Age",
    "Confucius": "the Chinese",
    "Zhu Xi": "the Chinese",
    "Adi Shankara": "the Indians",
    "Nagarjuna": "the Indians",
    "Karl Marx": "the Marxists",
    "Friedrich Nietzsche": "the Germans",
    "Bertrand Russell": "the moderns",
    "Gottlob Frege": "the moderns",
    "The Buddha": "the Indians",
    "Władysław Tatarkiewicz": "the Poles",
    "Jules Barthélemy-Saint-Hilaire": "a table for two",
    "René Descartes": "the Enlightenment",
    "Baruch Spinoza": "the Enlightenment",
    "Gottfried Wilhelm Leibniz": "the Enlightenment",
    "Maimonides": "the Jewish tradition",
    "John Stuart Mill": "the utilitarians",
    "Jeremy Bentham": "the utilitarians",
    "Charles Sanders Peirce": "the pragmatists",
    "William James": "the pragmatists",
    "Martin Luther": "the Reformation",
    "Erasmus": "the humanists",
    "Auguste Comte": "the positivists",
    "Adam Smith": "the economists",
}


def read():
    nodes = pd.read_csv(NODES, sep="\t", comment="#", quoting=3, encoding="utf-8", keep_default_na=False)
    edges = pd.read_csv(EDGES, sep="\t", comment="#", encoding="utf-8")
    G = nx.Graph()
    G.add_nodes_from(nodes.node_id)
    for s, t, w in edges.itertuples(index=False):
        if s == t:
            continue
        if G.has_edge(s, t):
            G[s][t]["weight"] += int(w)
        else:
            G.add_edge(s, t, weight=int(w))
    meta = {r.node_id: r for r in nodes.itertuples(index=False)}
    return G, meta


def shorten(text, limit=DESC_MAX):
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:.") + "..."


def labels_of(partition, order):
    lab = {}
    for c, members in enumerate(partition):
        for n in members:
            lab[n] = c
    return [lab[n] for n in order]


def nmi(a, b):
    """Normalized mutual information, arithmetic-mean normalisation (sklearn's default)."""
    a = list(a)
    b = list(b)
    n = len(a)
    ca = Counter(a)
    cb = Counter(b)
    cab = Counter(zip(a, b))
    ha = -sum(c / n * math.log(c / n) for c in ca.values())
    hb = -sum(c / n * math.log(c / n) for c in cb.values())
    mi = sum(c / n * math.log((c / n) / ((ca[x] / n) * (cb[y] / n))) for (x, y), c in cab.items())
    denom = (ha + hb) / 2
    return mi / denom if denom > 0 else 1.0


def majority_map(labels, ref):
    """Map each community in `labels` to the community of `ref` it overlaps most."""
    ov = defaultdict(Counter)
    for l, r in zip(labels, ref):
        ov[l][r] += 1
    return {l: c.most_common(1)[0][0] for l, c in ov.items()}


def layout(H, seed=42):
    """ForceAtlas2 in LinLog mode with hub dissuasion, so the traditions pull
    apart into visible regions instead of one hairball; strong gravity keeps
    the Chinese and Indian arms close enough to fit on one screen. About
    three minutes on a laptop."""
    pos = nx.forceatlas2_layout(H, max_iter=700, seed=seed, scaling_ratio=2.0,
                                gravity=6.0, dissuade_hubs=True, linlog=True, jitter_tolerance=1.0)
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    span = max(xs.max() - xs.min(), ys.max() - ys.min()) or 1.0
    return {n: (0.5 + 0.9 * (p[0] - cx) / span, 0.5 + 0.9 * (p[1] - cy) / span) for n, p in pos.items()}


def main():
    G, meta = read()
    giant_nodes = max(nx.connected_components(G), key=len)
    H = G.subgraph(giant_nodes).copy()
    order = sorted(H.nodes())
    index = {n: i for i, n in enumerate(order)}
    N = len(order)
    m = H.number_of_edges()
    print(f"network: {G.number_of_nodes()} philosophers, {G.number_of_edges()} links; "
          f"giant component {N} / {m}")

    deg = dict(H.degree())
    strength = dict(H.degree(weight="weight"))

    # ---- Louvain, 20 seeds, unweighted
    runs = [nx.community.louvain_communities(H, weight=None, seed=s) for s in range(SEEDS)]
    qs = [nx.community.modularity(H, p, weight=None) for p in runs]
    best = int(np.argmax(qs))
    L = [labels_of(p, order) for p in runs]
    pair = [nmi(L[i], L[j]) for i in range(SEEDS) for j in range(i + 1, SEEDS)]
    print(f"louvain: Q {min(qs):.4f}..{max(qs):.4f} (best seed {best}: {qs[best]:.4f}, "
          f"{len(runs[best])} communities); self-NMI {min(pair):.3f}..{max(pair):.3f}, mean {np.mean(pair):.3f}")

    # Communities sorted by size so table 0 is the biggest; relabel accordingly.
    comms = sorted(runs[best], key=lambda c: (-len(c), min(index[n] for n in c)))
    lou = [0] * N
    for c, members in enumerate(comms):
        for n in members:
            lou[index[n]] = c
    ref = lou
    flips = np.zeros(N, dtype=int)
    for r in range(SEEDS):
        if r == best:
            continue
        mp = majority_map(L[r], ref)
        for i in range(N):
            if mp[L[r][i]] != ref[i]:
                flips[i] += 1

    names, tops = [], []
    for c, members in enumerate(comms):
        top = sorted(members, key=lambda n: (-deg[n], n))[:8]
        tops.append([index[n] for n in top])
        head = meta[top[0]].name
        names.append(NAMES.get(head, f"the {head} table"))
        eras = Counter(meta[n].era for n in members).most_common(2)
        print(f"  {len(members):4d}  {names[c]:28s} {', '.join(meta[n].name for n in top[:5])}"
              f"   [{'; '.join(f'{e} {k}' for e, k in eras)}]")

    # ---- greedy modularity
    greedy = nx.community.greedy_modularity_communities(H, weight=None)
    gq = nx.community.modularity(H, greedy, weight=None)
    gl = labels_of(sorted(greedy, key=lambda c: (-len(c), min(index[n] for n in c))), order)
    print(f"greedy: Q {gq:.4f}, {len(greedy)} communities, NMI vs louvain {nmi(gl, lou):.3f}")

    # ---- the seating by century
    era_idx = [ERAS.index(meta[n].era) for n in order]
    era_part = [{n for n in order if ERAS.index(meta[n].era) == e} for e in range(len(ERAS))]
    era_q = nx.community.modularity(H, [c for c in era_part if c], weight=None)
    era_nmi = nmi(era_idx, lou)
    print(f"by century: Q {era_q:.4f}, NMI vs louvain {era_nmi:.3f}")

    # ---- null models
    sq, rq = [], []
    for s in range(SEEDS):
        S = H.copy()
        nx.double_edge_swap(S, nswap=10 * m, max_tries=200 * m, seed=s)
        sq.append(nx.community.modularity(S, nx.community.louvain_communities(S, weight=None, seed=s), weight=None))
        R = nx.gnm_random_graph(N, m, seed=s)
        rq.append(nx.community.modularity(R, nx.community.louvain_communities(R, seed=s)))
    print(f"degree-preserving shuffle: Q {np.mean(sq):.4f} +/- {np.std(sq):.4f};  "
          f"G(n,m): {np.mean(rq):.4f} +/- {np.std(rq):.4f}")

    # ---- weighted Q of the same partition, for the weights toggle
    lou_w = nx.community.modularity(H, comms, weight="weight")
    print(f"weighted Q of louvain's partition: {lou_w:.4f}")

    # ---- layout
    pos = layout(H)

    nodes_out = []
    for n in order:
        r = meta[n]
        nodes_out.append({
            "id": n,
            "name": r.name,
            "era": ERAS.index(r.era),
            "sub": r.subfields or "",
            "desc": shorten(r.description),
            "x": round(float(pos[n][0]), 4),
            "y": round(float(pos[n][1]), 4),
            "deg": int(deg[n]),
            "str": int(strength[n]),
        })
    edges_out = sorted([index[u], index[v], d["weight"]] if index[u] < index[v] else [index[v], index[u], d["weight"]]
                       for u, v, d in H.edges(data=True))
    famous = [index[n] for n in sorted(order, key=lambda n: (-deg[n], n))[:80]]

    payload = {
        "meta": {
            "source": "02805 Social Graphs and Interactions - week 4 philosophers snapshot (2026-09-15)",
            "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "giant_nodes": N, "giant_edges": m, "seeds": SEEDS,
        },
        "eras": ERAS, "eraShort": ERA_SHORT,
        "nodes": nodes_out,
        "edges": edges_out,
        "famous": famous,
        "louvain": {"q": round(qs[best], 6), "qw": round(lou_w, 6), "k": len(comms), "seed": best,
                    "labels": lou, "names": names, "top": tops,
                    "qRange": [round(min(qs), 4), round(max(qs), 4)]},
        "flips": flips.tolist(),
        "selfNmi": {"mean": round(float(np.mean(pair)), 4), "min": round(min(pair), 4), "max": round(max(pair), 4)},
        "greedy": {"q": round(gq, 6), "k": len(greedy), "labels": gl, "nmi": round(nmi(gl, lou), 4)},
        "century": {"q": round(era_q, 6), "nmi": round(era_nmi, 4)},
        "shuffle": {"mean": round(float(np.mean(sq)), 4), "sd": round(float(np.std(sq)), 4), "n": SEEDS},
        "gnm": {"mean": round(float(np.mean(rq)), 4), "sd": round(float(np.std(rq)), 4), "n": SEEDS},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        "// Generated by scripts/build_week4_data.py from the course's philosophers snapshot. Do not edit by hand.\n"
        "window.SYMPOSIUM = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} kB)")

    if "--png" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        i = sys.argv.index("--png")
        outdir = pathlib.Path(sys.argv[i + 1]) if len(sys.argv) > i + 1 else ROOT
        from matplotlib import patheffects
        # the game's table colours, so the figure and the reveal screen agree
        cols = ["#d9a648", "#7fb069", "#5b8fd6", "#e07a3f", "#d4536a", "#a985e0", "#45c2b1", "#e79ab9", "#b7b64f", "#c9c1b3"]
        fig, ax = plt.subplots(figsize=(12, 11))
        fig.patch.set_facecolor("#160d12")
        ax.set_facecolor("#160d12")
        for u, v in H.edges():
            same = lou[index[u]] == lou[index[v]]
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=cols[lou[index[u]] % 10] if same else "#d8cfc0", lw=0.18 if same else 0.12,
                    alpha=0.28 if same else 0.10, zorder=1)
        for c, members in enumerate(comms):
            ax.scatter([pos[n][0] for n in members], [pos[n][1] for n in members],
                       s=[5 + deg[n] * 0.55 for n in members], color=cols[c % 10], zorder=2, lw=0.3, edgecolor="#0c0609")
        for c, members in enumerate(comms):
            if len(members) < 20:
                continue
            cx = np.mean([pos[n][0] for n in members])
            cy = np.mean([pos[n][1] for n in members])
            ax.text(cx, cy, names[c].upper(), color=cols[c % 10], fontsize=13, ha="center", va="center", zorder=3,
                    fontweight="bold", path_effects=[patheffects.withStroke(linewidth=4, foreground="#0c0609")])
        for n in sorted(H.nodes(), key=lambda n: -deg[n])[:14]:
            ax.text(pos[n][0], pos[n][1] - 0.012, meta[n].name, color="#f1e6d2", fontsize=7.5, ha="center", va="top", zorder=4,
                    path_effects=[patheffects.withStroke(linewidth=2.5, foreground="#0c0609")])
        ax.axis("off")
        fig.savefig(outdir / "week4_tables.png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
        print("figure written to", outdir / "week4_tables.png")


if __name__ == "__main__":
    main()
