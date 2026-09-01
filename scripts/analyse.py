"""
Week 1 analysis of the Marvel Wikipedia network.

Reads the frozen course snapshot (week1_edges.tsv + week1_nodes.tsv), builds the
directed graph WITHOUT dropping the 17 isolates, and writes every figure and
number the week-1 post quotes.

Outputs
  assets/figures/*.png      figures embedded in weeks/week1.html
  data/stats.json           every number the post quotes, so the prose and the
                            data can never drift apart

Run:  python scripts/analyse.py
"""

import collections
import csv
import json
import math
import pathlib

import matplotlib
import matplotlib.transforms
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGS = ROOT / "assets" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Palette lifted from the site CSS so the figures sit inside the page, not on it.
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
    """All 303 nodes first, then the edges. Skip step one and you lose 17 characters."""
    G = nx.DiGraph()
    for row in read_tsv(ROOT / "week1_nodes.tsv"):
        G.add_node(row["node_id"], name=row["name"], url=row["url"],
                   description=row.get("description", ""))
    for row in read_tsv(ROOT / "week1_edges.tsv", ["source", "target"]):
        G.add_edge(row["source"], row["target"])
    return G


def pretty(node_id):
    return node_id.replace("_", " ").replace(" (character)", "").replace(" (Marvel Comics)", "") \
                  .replace(" (comics)", "").replace(" (characters)", "")


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


# ---------------------------------------------------------------------------
# Figure 1: degree distributions, linear and log-log, in and out
# ---------------------------------------------------------------------------
def fig_degree_distributions(G, stats):
    kin = np.array([d for _, d in G.in_degree()])
    kout = np.array([d for _, d in G.out_degree()])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))

    for ax, k, color, label in ((axes[0, 0], kin, CYAN, "in-degree"),
                                (axes[0, 1], kout, GOLD, "out-degree")):
        ax.hist(k, bins=np.arange(0, k.max() + 2) - 0.5, color=color, alpha=0.85, edgecolor=BG, lw=0.4)
        ax.axvline(k.mean(), color=RED, ls="--", lw=1.4, label=f"mean = {k.mean():.2f}")
        ax.axvline(np.median(k), color=GREEN, ls=":", lw=1.6, label=f"median = {np.median(k):.0f}")
        style(ax, f"{label}, linear axes", f"k ({label})", "number of characters")
        ax.legend(labelcolor=FG, fontsize=8.5)
        ax.set_xlim(-1, min(k.max() + 1, 115))

    # Log-log needs binned densities; drop k = 0 because log(0) has no home on this plot.
    for ax, k, color, label in ((axes[1, 0], kin, CYAN, "in-degree"),
                                (axes[1, 1], kout, GOLD, "out-degree")):
        counts = collections.Counter(k[k > 0])
        xs = np.array(sorted(counts))
        ys = np.array([counts[x] for x in xs], dtype=float) / len(k)
        ax.scatter(xs, ys, s=34, color=color, alpha=0.85, edgecolor=BG, lw=0.5, zorder=3)

        # Least-squares slope on the log-log cloud. Descriptive only: with 303
        # nodes this is a shape hint, not evidence of a power law.
        lx, ly = np.log10(xs), np.log10(ys)
        slope, intercept = np.polyfit(lx, ly, 1)
        grid = np.linspace(lx.min(), lx.max(), 40)
        ax.plot(10 ** grid, 10 ** (slope * grid + intercept), color=RED, lw=1.5, ls="--",
                label=f"slope = {slope:.2f}", zorder=2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        style(ax, f"{label}, log-log", f"k ({label})", "P(k)")
        ax.legend(labelcolor=FG, fontsize=8.5)
        stats[f"loglog_slope_{'in' if label.startswith('in') else 'out'}"] = round(float(slope), 2)

    fig.suptitle("Degree distributions of the Marvel Wikipedia network (303 nodes, 1784 directed edges)",
                 color=FG, fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(FIGS / "degree_distributions.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: in-degree against out-degree, the heart of the week's question
# ---------------------------------------------------------------------------
def fig_in_vs_out(G, stats):
    ids = list(G.nodes())
    kin = np.array([G.in_degree(n) for n in ids], dtype=float)
    kout = np.array([G.out_degree(n) for n in ids], dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 7.2))

    # Jitter keeps the pile of low-degree nodes from becoming one opaque blob.
    rng = np.random.default_rng(7)
    jx = kin + rng.uniform(-0.18, 0.18, len(kin))
    jy = kout + rng.uniform(-0.18, 0.18, len(kout))

    sizes = 16 + 3.2 * (kin + kout)
    ax.scatter(jx, jy, s=sizes, c=kin - kout, cmap="coolwarm_r",
               alpha=0.78, edgecolor=BG, lw=0.5, zorder=3)

    lim = max(kin.max(), kout.max()) + 6
    ax.plot([0.2, lim], [0.2, lim], color=DIM, ls="--", lw=1, alpha=0.55, zorder=1)
    ax.text(23, 27, "in = out", color=DIM, fontsize=9, rotation=34, alpha=0.9)

    # Two families of extremes, labelled on opposite sides so they never collide:
    # the pages everyone links TO, and the pages that do the linking.
    top_in = sorted(ids, key=lambda n: -G.in_degree(n))[:5]
    top_out = sorted(ids, key=lambda n: -G.out_degree(n))[:5]
    for rank, n in enumerate(top_in):
        ax.annotate(pretty(n), (G.in_degree(n), G.out_degree(n)),
                    textcoords="offset points", xytext=(-11, -13 - 2 * rank),
                    ha="right", fontsize=9.5, color=CYAN, fontweight="bold", zorder=5)
    for rank, n in enumerate(top_out):
        if n in top_in:
            continue
        ax.annotate(pretty(n), (G.in_degree(n), G.out_degree(n)),
                    textcoords="offset points", xytext=(9, 9 + 2 * rank),
                    ha="left", fontsize=9.5, color=GOLD, fontweight="bold", zorder=5)

    ax.set_xscale("symlog", linthresh=1.5)
    ax.set_yscale("symlog", linthresh=1.5)
    ax.set_xlim(-0.7, 190)
    ax.set_ylim(-0.7, 60)
    style(ax, None, "in-degree: how many characters link TO this page",
          "out-degree: how many characters this page links to")
    ax.set_title("Being linked to and doing the linking are two different jobs",
                 color=FG, loc="left", fontsize=13, pad=16)

    note = "\n".join([
        f"Pearson r = {stats['pearson_in_out']:.2f}",
        "cyan = top 5 in-degree",
        "gold = top 5 out-degree",
    ])
    ax.text(0.985, 0.045, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=DIM, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.55", fc="#111830", ec=GRID, lw=1))

    fig.tight_layout()
    fig.savefig(FIGS / "in_vs_out.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: draw the network
# ---------------------------------------------------------------------------
HALO = [pe.withStroke(linewidth=3.2, foreground=BG)]


def place_labels(ax, items, fig):
    """Greedy label placement, most important first.

    items = [(x, y, text, colour, fontsize)] already sorted by importance. Each
    label gets four tries (above, below, right, left) before it is dropped, so a
    genuine hub like Hulk does not disappear just because it sits next to
    Spider-Man. Overlap is tested in display space, the only space where labels
    of different font sizes are comparable.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    candidates = [(0, 6, "center", "bottom"), (0, -6, "center", "top"),
                  (9, 0, "left", "center"), (-9, 0, "right", "center")]
    taken = []
    for x, y, text, colour, fs in items:
        for dx, dy, ha, va in candidates:
            t = ax.text(x, y, text, fontsize=fs, color=colour, ha=ha, va=va,
                        fontweight="bold", zorder=6, path_effects=HALO,
                        transform=ax.transData)
            t.set_transform(ax.transData + matplotlib.transforms.ScaledTranslation(
                dx / 72, dy / 72, fig.dpi_scale_trans))
            bb = t.get_window_extent(renderer=renderer)
            box = (bb.x0 - 2, bb.y0 - 2, bb.x1 + 2, bb.y1 + 2)
            if any(box[0] < o[2] and box[2] > o[0] and box[1] < o[3] and box[3] > o[1] for o in taken):
                t.remove()
                continue
            taken.append(box)
            break


def fig_network(G, stats):
    UG = G.to_undirected()
    comps = sorted(nx.connected_components(UG), key=len, reverse=True)
    giant = UG.subgraph(comps[0])

    # Higher k spreads the hairball out; without it everything piles into one blob.
    pos = nx.spring_layout(giant, k=0.55, iterations=500, seed=42)

    fig, ax = plt.subplots(figsize=(12, 9.6))
    ax.set_facecolor(BG)

    kin = {n: G.in_degree(n) for n in giant}
    kmax = max(kin.values())

    # Edges that end at a big hub get drawn brighter, which makes the star
    # patterns around Spider-Man, Hulk and Wolverine visible instead of implied.
    seg_dim, seg_hot = [], []
    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        (seg_hot if kin[v] >= 33 else seg_dim).append((pos[u], pos[v]))

    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(seg_dim, colors="#4a5a92", linewidths=0.32, alpha=0.30, zorder=1))
    ax.add_collection(LineCollection(seg_hot, colors=CYAN, linewidths=0.42, alpha=0.22, zorder=2))

    order = sorted(giant.nodes(), key=lambda n: kin[n])
    xs = [pos[n][0] for n in order]
    ys = [pos[n][1] for n in order]
    ks = np.array([kin[n] for n in order], dtype=float)
    sc = ax.scatter(xs, ys, s=16 + 6.2 * ks, c=np.log1p(ks), cmap="plasma",
                    linewidths=0.5, edgecolors=BG, zorder=3, alpha=0.96)

    top = sorted(giant.nodes(), key=lambda n: -kin[n])[:22]
    items = [(pos[n][0], pos[n][1], pretty(n), FG, 8.2 + 2.6 * kin[n] / kmax) for n in top]

    # Trim the empty margins the few far-flung nodes create.
    px = np.array([p[0] for p in pos.values()])
    py = np.array([p[1] for p in pos.values()])
    ax.set_xlim(np.percentile(px, 0.6) - 0.05, np.percentile(px, 99.4) + 0.05)
    ax.set_ylim(np.percentile(py, 0.6) - 0.05, np.percentile(py, 99.4) + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")

    place_labels(ax, items, fig)

    cb = fig.colorbar(sc, ax=ax, fraction=0.024, pad=0.01)
    cb.set_label("in-degree (log scale)", color=DIM, fontsize=9)
    cb.ax.tick_params(colors=DIM, labelsize=8)
    cb.outline.set_edgecolor(GRID)
    ticks = [1, 3, 10, 30, 106]
    cb.set_ticks(np.log1p(ticks))
    cb.set_ticklabels([str(t) for t in ticks])

    ax.set_title(f"The giant component: {len(giant)} of 303 characters. "
                 f"Highlighted edges end at a top-5 hub.",
                 color=FG, loc="left", fontsize=13, pad=14)
    fig.tight_layout()
    fig.savefig(FIGS / "network.png", bbox_inches="tight")
    plt.close(fig)

    return comps


# ---------------------------------------------------------------------------
# Figure 4: the islands, which is where the interesting anomalies live
# ---------------------------------------------------------------------------
def fig_islands(G, comps, stats):
    islands = [c for c in comps[1:] if len(c) > 1]
    isolates = sorted((n for n in G if G.degree(n) == 0), key=pretty)

    fig = plt.figure(figsize=(12.4, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.06)

    # Left panel: the one island that survives on its own.
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(BG)
    isl = G.subgraph(islands[0])
    p = nx.spring_layout(isl.to_undirected(), seed=11, k=1.1, iterations=400)

    nx.draw_networkx_edges(isl, p, ax=ax, edge_color=CYAN, width=1.25, alpha=0.55,
                           arrows=True, arrowsize=8, node_size=260,
                           connectionstyle="arc3,rad=0.10")
    nx.draw_networkx_nodes(isl, p, ax=ax, node_size=260, node_color=CYAN,
                           alpha=0.92, edgecolors=BG, linewidths=1.4)
    for n in isl:
        ax.text(p[n][0], p[n][1] - 0.13, pretty(n), fontsize=9, color=FG,
                ha="center", va="top", fontweight="bold", path_effects=HALO)

    ax.set_title(f"The only island: {len(isl)} characters, {isl.number_of_edges()} edges,\n"
                 f"and not one link to the other 294",
                 color=CYAN, loc="left", fontsize=12.5, pad=12)
    ax.axis("off")
    ax.margins(0.22)

    # Right panel: the characters with no link in either direction.
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG)
    cols = 2
    rows = math.ceil(len(isolates) / cols)
    for i, n in enumerate(isolates):
        col, row = i % cols, i // cols
        x, y = col * 1.0, -row * 1.0
        ax2.scatter([x], [y], s=130, color="#3b466e", alpha=0.9,
                    edgecolors=GRID, lw=1.2, zorder=3)
        ax2.text(x + 0.11, y, pretty(n), fontsize=9.2, color=DIM, va="center", ha="left")
    ax2.set_xlim(-0.25, cols - 1 + 0.95)
    ax2.set_ylim(-rows + 0.4, 0.75)
    ax2.set_title(f"{len(isolates)} isolates: no link in, no link out.\n"
                  f"Build from the edge list alone and they vanish",
                  color=DIM, loc="left", fontsize=12.5, pad=12)
    ax2.axis("off")

    fig.suptitle("Everything outside the giant component",
                 color=FG, fontsize=13.5, fontweight="bold", x=0.012, ha="left", y=1.02)
    fig.savefig(FIGS / "islands.png", bbox_inches="tight")
    plt.close(fig)

    stats["islands"] = [sorted(pretty(x) for x in c) for c in islands]
    stats["isolates_list"] = [pretty(n) for n in isolates]


# ---------------------------------------------------------------------------
# Figure 5: how far is home? Round-trip distances, the maths behind the game
# ---------------------------------------------------------------------------
def fig_roundtrip(G, stats):
    hubs = ["Spider-Man", "Wolverine_(character)", "Doctor_Strange"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), sharey=True)
    colours = {"Spider-Man": RED, "Wolverine_(character)": GOLD, "Doctor_Strange": PURPLE}

    summary = {}
    for ax, hub in zip(axes, hubs):
        out_d = nx.single_source_shortest_path_length(G, hub)
        back_d = nx.single_source_shortest_path_length(G.reverse(copy=False), hub)
        both = [(out_d[n] + back_d[n]) for n in G if n in out_d and n in back_d and n != hub]
        vals, counts = np.unique(both, return_counts=True)
        ax.bar(vals, counts, color=colours[hub], alpha=0.85, edgecolor=BG, lw=0.5)
        ax.axvline(np.mean(both), color=FG, ls="--", lw=1.2)
        style(ax, pretty(hub), "round-trip length (out + back)", "characters")
        ax.text(0.97, 0.93, f"reachable both ways: {len(both)}\nmean par: {np.mean(both):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.6, color=DIM)
        summary[pretty(hub)] = {
            "round_trip_targets": len(both),
            "mean_par": round(float(np.mean(both)), 2),
            "min_par": int(min(both)),
            "max_par": int(max(both)),
        }

    fig.suptitle("How hard is it to get home? Round-trip distance from each playable hero",
                 color=FG, fontsize=13, fontweight="bold", x=0.012, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "roundtrip.png", bbox_inches="tight")
    plt.close(fig)
    stats["heroes"] = summary


# ---------------------------------------------------------------------------
def main():
    G = load_graph()
    UG = G.to_undirected()
    kin = np.array([d for _, d in G.in_degree()], dtype=float)
    kout = np.array([d for _, d in G.out_degree()], dtype=float)

    comps_w = sorted(nx.connected_components(UG), key=len, reverse=True)
    comps_s = sorted(nx.strongly_connected_components(G), key=len, reverse=True)

    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_isolates": len(list(nx.isolates(G))),
        "n_sinks": int((kout == 0).sum()),
        "n_sources": int((kin == 0).sum()),
        "density": round(nx.density(G), 5),
        "mean_degree": round(float(kin.mean()), 2),
        "max_in": int(kin.max()),
        "max_out": int(kout.max()),
        "median_in": int(np.median(kin)),
        "median_out": int(np.median(kout)),
        "pearson_in_out": round(float(np.corrcoef(kin, kout)[0, 1]), 3),
        "reciprocity": round(nx.reciprocity(G), 3),
        "n_weak_components": len(comps_w),
        "giant_weak": len(comps_w[0]),
        "n_strong_components": len(comps_s),
        "giant_strong": len(comps_s[0]),
        "top_in": [(pretty(n), int(d)) for n, d in sorted(G.in_degree(), key=lambda x: -x[1])[:10]],
        "top_out": [(pretty(n), int(d)) for n, d in sorted(G.out_degree(), key=lambda x: -x[1])[:10]],
    }

    # Spider-Man's share of the network, the number the course description teases.
    spidey_in = G.in_degree("Spider-Man")
    stats["spidey_in"] = int(spidey_in)
    stats["spidey_out"] = int(G.out_degree("Spider-Man"))
    stats["spidey_share"] = round(100 * spidey_in / (G.number_of_nodes() - 1), 1)
    stats["top_in_share"] = round(
        100 * sum(d for _, d in sorted(G.in_degree(), key=lambda x: -x[1])[:10]) / G.number_of_edges(), 1)

    # Average shortest path inside the largest strongly connected component.
    sub = G.subgraph(comps_s[0])
    stats["giant_strong_aspl"] = round(nx.average_shortest_path_length(sub), 2)
    stats["giant_strong_diameter"] = int(nx.diameter(sub))

    fig_degree_distributions(G, stats)
    fig_in_vs_out(G, stats)
    comps = fig_network(G, stats)
    fig_islands(G, comps, stats)
    fig_roundtrip(G, stats)

    (ROOT / "data" / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("islands", "isolates_list", "top_in", "top_out")}, indent=2))
    print("\ntop in :", stats["top_in"][:5])
    print("top out:", stats["top_out"][:5])
    print("islands:", stats["islands"])
    print("\nfigures written to", FIGS)


if __name__ == "__main__":
    main()
