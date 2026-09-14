"""
Checks the graph engine KINGPIN runs on, by actually running it.

The game's algorithms live in weeks/week3/game/core.js with no DOM in them,
so unlike week 1 we do not have to re-implement anything: this test executes
the exact shipped file under node, then recomputes everything independently
with networkx and diffs the two.

What this actually proves:
  * the game's arena is the correct undirected giant component (277 of 303)
  * its Brandes betweenness matches networkx to floating-point accuracy,
    and reproduces the week-3 course page's top-ten table
  * the two rival hitmen (degree-greedy, betweenness-greedy) shoot exactly
    the characters networkx says they should, so the "compared to what"
    scoreboard the player is judged against is real
  * the betweenness hitman beats or matches the degree hitman on this graph,
    so the game's lesson is true, not scripted

Run:  python scripts/test_kingpin_rules.py     (needs node and networkx)
"""

import json
import pathlib
import subprocess
import sys

import networkx as nx

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUDGET = 8  # the longest contract; shorter ones are prefixes of it

raw = (ROOT / "data" / "marvel_week1.js").read_text(encoding="utf-8")
DATA = json.loads(raw[raw.index("{"):raw.rstrip().rstrip(";").rindex("}") + 1])
NODES = DATA["nodes"]
N = len(NODES)

HARNESS = """
const fs = require('fs');
const core = require(process.argv[1]);
const raw = fs.readFileSync(process.argv[2], 'utf8');
const D = JSON.parse(raw.slice(raw.indexOf('{'), raw.lastIndexOf('}') + 1));
const { N, adj } = core.buildArena(D);
const budget = parseInt(process.argv[3], 10);
const c = core.components(N, adj, null);
process.stdout.write(JSON.stringify({
  giant: c.giantSize,
  components: c.count,
  betweenness: Array.from(core.brandes(N, adj, null)),
  degree: core.attack(N, adj, budget, 'degree'),
  broker: core.attack(N, adj, budget, 'betweenness'),
  random: core.randomAttackAvg(N, adj, budget, 30, 7),
  records: core.RECORDS,
}));
"""


def short(name):
    import re
    return re.sub(r"\s*\((character|comics|Marvel Comics|Marvel Comics character|characters)\)\s*$",
                  "", name, flags=re.I)


def build_graph():
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for u, outs in enumerate(DATA["out"]):
        for v in outs:
            if u != v:
                G.add_edge(u, v)
    return G


def giant_of(G):
    return max(nx.connected_components(G), key=len)


def nx_attack(G0, budget, strategy):
    """The rival hitmen, replicated: shoot the argmax over the current giant,
    ties to the lowest index, exactly as core.js attack() does."""
    G = G0.copy()
    giant = giant_of(G)
    curve, order = [len(giant)], []
    for _ in range(budget):
        if strategy == "degree":
            score = dict(G.degree())
        else:
            score = nx.betweenness_centrality(G, normalized=False)
        best, target = -1.0, -1
        for i in sorted(giant):
            if score[i] > best:
                best, target = score[i], i
        G.remove_node(target)
        order.append(target)
        giant = giant_of(G)
        curve.append(len(giant))
    return order, curve


def main():
    failures = []

    js = subprocess.run(
        ["node", "-e", HARNESS, "--",
         str(ROOT / "weeks/week3/game/core.js"),
         str(ROOT / "data" / "marvel_week1.js"),
         str(BUDGET)],
        capture_output=True, text=True)
    if js.returncode != 0:
        print(js.stderr)
        sys.exit("node harness failed")
    R = json.loads(js.stdout)

    G = build_graph()
    giant = giant_of(G)
    print(f"arena: {N} nodes, {G.number_of_edges()} undirected edges, "
          f"giant component {len(giant)}")

    if R["giant"] != len(giant):
        failures.append(f"giant: js={R['giant']} nx={len(giant)}")
    if R["components"] != nx.number_connected_components(G):
        failures.append(f"components: js={R['components']} nx={nx.number_connected_components(G)}")

    # --- betweenness against networkx, and against the course page's table ---
    bc_nx = nx.betweenness_centrality(G, normalized=False)
    worst = max(abs(R["betweenness"][i] - bc_nx[i]) for i in range(N))
    print(f"betweenness: max |js - networkx| = {worst:.2e} over all {N} nodes")
    if worst > 1e-6:
        failures.append(f"betweenness disagrees with networkx by {worst}")

    n_g = len(giant)
    norm = (n_g - 1) * (n_g - 2) / 2
    top = sorted(giant, key=lambda i: -bc_nx[i])[:10]
    print("top ten by betweenness (normalized):")
    for i in top:
        print(f"  {short(NODES[i]['name']):24s} {bc_nx[i] / norm:.4f}")
    expected_top3 = ["Spider-Man", "Hulk", "Wolverine"]
    got_top3 = [short(NODES[i]["name"]) for i in top[:3]]
    if got_top3 != expected_top3:
        failures.append(f"top-3 betweenness {got_top3} != course page {expected_top3}")
    spidey = bc_nx[top[0]] / norm
    if abs(spidey - 0.235) > 0.005:
        failures.append(f"Spider-Man normalized betweenness {spidey:.4f}, course page says 0.235")

    # --- the rival hitmen, move for move ---
    for strategy, key in (("degree", "degree"), ("betweenness", "broker")):
        order, curve = nx_attack(G, BUDGET, strategy)
        if R[key]["order"] != order:
            failures.append(f"{strategy} hitman order: js={R[key]['order']} nx={order}")
        if R[key]["curve"] != curve:
            failures.append(f"{strategy} hitman curve: js={R[key]['curve']} nx={curve}")
        names = " > ".join(short(NODES[i]["name"]) for i in order)
        print(f"{strategy:12s} hitman: {names}")
        print(f"{'':12s}  giant: {' '.join(map(str, curve))}")

    # --- the shipped record hit-lists must achieve exactly what they claim ---
    for b_str, rec in R["records"].items():
        b = int(b_str)
        if len(rec["ids"]) != b:
            failures.append(f"record {b}: {len(rec['ids'])} ids for a {b}-hit contract")
        H = G.copy()
        H.remove_nodes_from(rec["ids"])
        achieved = len(giant_of(H))
        names = ", ".join(short(NODES[i]["name"]) for i in rec["ids"])
        print(f"record {b} hits: core {achieved} ({names})")
        if achieved != rec["giant"]:
            failures.append(f"record {b}: claims core {rec['giant']}, actually {achieved}")
        # The verdict ladder assumes record <= both bots at that budget.
        for key in ("degree", "broker"):
            if rec["giant"] > R[key]["curve"][b]:
                failures.append(f"record {b} ({rec['giant']}) is worse than the {key} bot "
                                f"({R[key]['curve'][b]}); the verdict tiers are out of order")

    # --- the lesson must be true: the broker bot is at least as destructive ---
    if R["broker"]["curve"][-1] > R["degree"]["curve"][-1]:
        failures.append("degree-greedy beat betweenness-greedy; the game's verdict text lies")

    # --- the random hitman must be clearly worse than both bots ---
    rnd = R["random"]["curve"]
    if not all(b <= a for a, b in zip(rnd, rnd[1:])):
        failures.append(f"random average curve is not monotone: {rnd}")
    if rnd[-1] <= R["degree"]["curve"][-1]:
        failures.append("random hitman matched the degree bot; the scoreboard tiers collapse")
    print(f"random hitman (avg of 30): giant {rnd[0]:.0f} -> {rnd[-1]:.1f} "
          f"(bots reach {R['degree']['curve'][-1]} and {R['broker']['curve'][-1]})")

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("\nthe shipped engine agrees with networkx on every number the game shows")


if __name__ == "__main__":
    main()
