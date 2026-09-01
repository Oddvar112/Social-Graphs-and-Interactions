"""
Bygger en spillklar versjon av uke-1-snapshotet.

Leser week1_edges.tsv + week1_nodes.tsv (frosset snapshot fra kursets datasidee),
regner ut et fast force-directed layout og skriver data/marvel_week1.js som
setter window.MARVEL_DATA. Vi skriver .js (ikke .json) sa spillet ogsaa kan
kjores rett fra file:// uten en lokal server.

Kjor:  python scripts/build_graph.py
"""
import csv
import json
import math
import pathlib

import networkx as nx

ROOT = pathlib.Path(__file__).resolve().parent.parent
EDGES = ROOT / "week1_edges.tsv"
NODES = ROOT / "week1_nodes.tsv"
OUT = ROOT / "data" / "marvel_week1.js"

DESC_MAX = 220


def read_tsv(path, fieldnames=None):
    """TSV-leser som hopper over '#'-kommentarer (headeren i edges-fila er kommentert ut)."""
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t", fieldnames=fieldnames))


def shorten(text, limit=DESC_MAX):
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:.") + "..."


def build_graph():
    # Alle 303 nodene forst, sa kantene -- ellers mister vi de 17 isolerte stille.
    node_rows = read_tsv(NODES)
    edge_rows = read_tsv(EDGES, ["source", "target"])

    G = nx.DiGraph()
    for row in node_rows:
        G.add_node(row["node_id"], **row)
    for row in edge_rows:
        G.add_edge(row["source"], row["target"])
    return G, node_rows


def layout(G):
    """Spring layout paa den storste komponenten, isolerte noder i en ytre ring."""
    UG = G.to_undirected()
    components = sorted(nx.connected_components(UG), key=len, reverse=True)
    giant = UG.subgraph(components[0])

    pos = nx.spring_layout(giant, k=1.6 / math.sqrt(len(giant)), iterations=600, seed=42)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    # Normaliser den store komponenten inn i [0.12, 0.88]^2.
    pos = {n: (0.5 + 0.76 * (p[0] - cx) / span, 0.5 + 0.76 * (p[1] - cy) / span) for n, p in pos.items()}

    # Alt som ikke er i den store komponenten legges i en ring utenfor -- "the void".
    outside = [n for n in G if n not in pos]
    for i, n in enumerate(sorted(outside)):
        angle = 2 * math.pi * i / max(len(outside), 1)
        radius = 0.62 + 0.05 * (i % 3)
        pos[n] = (0.5 + radius * math.cos(angle), 0.5 + radius * math.sin(angle))
    return pos


def main():
    G, node_rows = build_graph()
    pos = layout(G)

    ids = sorted(G.nodes())
    index = {n: i for i, n in enumerate(ids)}
    meta = {row["node_id"]: row for row in node_rows}

    nodes = []
    for n in ids:
        row = meta.get(n, {})
        x, y = pos[n]
        nodes.append({
            "id": n,
            "name": row.get("name") or n.replace("_", " "),
            "url": row.get("url") or f"https://en.wikipedia.org/wiki/{n}",
            "desc": shorten(row.get("description")),
            "x": round(x, 5),
            "y": round(y, 5),
        })

    out_adj = [sorted(index[m] for m in G.successors(n)) for n in ids]
    in_adj = [sorted(index[m] for m in G.predecessors(n)) for n in ids]

    payload = {
        "meta": {
            "source": "02805 Social Graphs and Interactions - week 1 frozen snapshot (2026-08-26)",
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "isolates": len(list(nx.isolates(G))),
            "sinks": sum(1 for _, d in G.out_degree() if d == 0),
            "sources": sum(1 for _, d in G.in_degree() if d == 0),
        },
        "nodes": nodes,
        "out": out_adj,
        "in": in_adj,
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        "// Autogenerert av scripts/build_graph.py - ikke rediger for hand.\n"
        "window.MARVEL_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"Skrev {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.0f} kB)")
    print(f"  {payload['meta']['nodes']} noder, {payload['meta']['edges']} kanter, "
          f"{payload['meta']['isolates']} isolerte, {payload['meta']['sinks']} blindveier")

if __name__ == "__main__":
    main()
