"""
Checks the engine SYMPOSIUM runs on, by actually running it.

The game's algorithms live in weeks/week4/game/core.js with no DOM in them,
so this test executes the exact shipped file under node, then recomputes
everything independently from the course's TSV files with networkx and
diffs the two.

What this actually proves:
  * the shipped data file is the correct giant component of the philosophers
    network (1374 of 1444, 9139 links), with the right degrees and weights
  * the game's modularity matches networkx to floating-point accuracy, on
    Louvain's partition, greedy's, the seating by century, and the seatings
    the game produces
  * the shipped Louvain partition really is what networkx gives for that
    seed, and its Q really is the best of the 20 seeds the post talks about
  * the seating rule (label propagation from the placed guests) does exactly
    what a plain Python re-implementation does, guest for guest
  * the game's NMI matches scikit-learn's, when scikit-learn is installed
  * the ladder on the score screen is in the order the verdict text assumes:
    Louvain > greedy > by century > degree-preserving shuffle > random host
  * ONE TABLE: a table's "links above chance" is m times its modularity share
    (checked against networkx on the partition {table, everyone else alone}),
    and the greedy host and the record table are exactly what a plain Python
    re-implementation produces, guest for guest, with record >= greedy

Run:  python scripts/test_symposium_rules.py     (needs node, networkx, pandas)
"""

import json
import pathlib
import subprocess
import sys

import networkx as nx

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_week4_data as B  # noqa: E402

DATA_JS = ROOT / "data" / "week4_symposium.js"
CORE_JS = ROOT / "weeks" / "week4" / "game" / "core.js"

HARNESS = r"""
const fs = require('fs');
const core = require(process.argv[1]);
const raw = fs.readFileSync(process.argv[2], 'utf8');
const D = JSON.parse(raw.slice(raw.indexOf('{'), raw.lastIndexOf('}') + 1));
const G = core.buildGraph(D);
const byName = new Map(D.nodes.map((n, i) => [n.name, i]));
const seedSets = JSON.parse(process.argv[3]).map(set => set.map(([name, t]) => [byName.get(name), t]));
const era = D.nodes.map(n => n.era);
const out = {
  N: G.N, m: G.m, W: G.W,
  deg: Array.from(G.deg), str: Array.from(G.str),
  qLouvain: core.modularity(G, D.louvain.labels, false),
  qwLouvain: core.modularity(G, D.louvain.labels, true),
  qGreedy: core.modularity(G, D.greedy.labels, false),
  qCentury: core.modularity(G, era, false),
  nmiCentury: core.nmi(era, D.louvain.labels),
  nmiGreedy: core.nmi(D.greedy.labels, D.louvain.labels),
  seatings: [],
  tables: [],
};
for (const name of JSON.parse(process.argv[4])) {
  const host = byName.get(name);
  for (const k of [3, 5, 9]) {
    const greedy = core.greedyTable(G, host, k);
    const record = core.improveTable(G, host, greedy);
    out.tables.push({ name, host, k, greedy, record,
      greedyScore: core.tableScore(G, greedy), recordScore: core.tableScore(G, record),
      random: core.randomTableAvg(G, host, k, 20, 3) });
  }
}
for (const [k, seeds] of seedSets.entries()) {
  for (const weighted of [false, true]) {
    const P = core.propagate(G, seeds, { weighted });
    out.seatings.push({
      set: k, weighted, seeds,
      labels: Array.from(P.labels), rounds: P.rounds.length, sweeps: P.sweeps.length, converged: P.converged,
      q: core.modularity(G, P.labels, false),
      nmi: core.nmi(P.labels, D.louvain.labels),
      random: core.randomSeatingAvg(G, seeds, P.tables, 20, 5),
    });
  }
}
process.stdout.write(JSON.stringify(out));
"""

HOSTS = ["Immanuel Kant", "Aristotle", "Confucius", "Boethius", "Alexandre Koyré", "Jules Lequier"]

SEED_SETS = [
    [["Plato", 0], ["Immanuel Kant", 1], ["Thomas Aquinas", 2]],
    [["Aristotle", 0], ["Plato", 1], ["Immanuel Kant", 2], ["René Descartes", 3],
     ["Bertrand Russell", 4], ["The Buddha", 5], ["Confucius", 6], ["Thomas Aquinas", 7]],
    [["Friedrich Nietzsche", 0]],
    [["Karl Marx", 0], ["Adam Smith", 1], ["Laozi", 2], ["Avicenna", 0]],
]


def py_propagate(adj, w, seeds, weighted, max_sweeps=200):
    """The seating rule, re-implemented plainly. Phase 1: unseated guests
    with a seated neighbour join, all at once, the table most of those
    neighbours are at (ties: the stronger links, then the lower table).
    Phase 2: in index order, seated guests move to their neighbours'
    majority table, staying put on a tie, until nobody moves."""
    N = len(adj)
    labels = [-1] * N
    fixed = set()
    for i, t in seeds:
        labels[i] = t
        fixed.add(i)
    T = max(t for _, t in seeds) + 1

    def best_table(i):
        votes = [0.0] * T
        tie = [0.0] * T
        seen = False
        for v, ww in zip(adj[i], w[i]):
            lv = labels[v]
            if lv < 0:
                continue
            votes[lv] += ww if weighted else 1
            tie[lv] += ww
            seen = True
        if not seen:
            return -1, votes
        best = -1
        for t in range(T):
            if votes[t] == 0:
                continue
            if best < 0 or votes[t] > votes[best] or (votes[t] == votes[best] and tie[t] > tie[best]):
                best = t
        return best, votes

    rounds = 0
    while True:
        changed = []
        for i in range(N):
            if labels[i] >= 0:
                continue
            t, _ = best_table(i)
            if t >= 0:
                changed.append((i, t))
        if not changed:
            break
        for i, t in changed:
            labels[i] = t
        rounds += 1

    sweeps = 0
    while sweeps < max_sweeps:
        moved = 0
        for i in range(N):
            if i in fixed or labels[i] < 0:
                continue
            cur = labels[i]
            t, votes = best_table(i)
            if t < 0 or t == cur or votes[cur] == votes[t]:
                continue
            labels[i] = t
            moved += 1
        if moved == 0:
            break
        sweeps += 1
    return labels, rounds, sweeps


def py_table(adj, deg, m, members):
    """Links inside, degree sum, and links above chance for a set of guests."""
    s = set(members)
    inside = sum(1 for u in members for v in adj[u] if v > u and v in s)
    dsum = sum(deg[u] for u in members)
    return inside, dsum, inside - dsum * dsum / (4 * m)


def py_candidates(adj, members):
    s = set(members)
    return sorted({v for u in members for v in adj[u] if v not in s})


def py_greedy(adj, deg, m, host, k):
    """The greedy host: add whoever raises links-above-chance most, ties to the lowest index."""
    members = [host]
    for _ in range(k):
        s = set(members)
        dsum = sum(deg[u] for u in members)
        best, best_g = -1, None
        for v in py_candidates(adj, members):
            links = sum(1 for u in adj[v] if u in s)
            g = links - (2 * dsum * deg[v] + deg[v] * deg[v]) / (4 * m)
            if best_g is None or g > best_g + 1e-12:
                best, best_g = v, g
        if best < 0:
            break
        members.append(best)
    return members


def py_improve(adj, deg, m, host, start, max_rounds=60):
    """The record: best single swap each round (never the host), ties to the earliest seat, then lowest index."""
    members = list(start)
    for _ in range(max_rounds):
        _, _, base = py_table(adj, deg, m, members)
        best_val, best = base + 1e-9, None
        cand = py_candidates(adj, members)
        for a in range(1, len(members)):
            without = members[:a] + members[a + 1:]
            for v in cand:
                _, _, val = py_table(adj, deg, m, without + [v])
                if val > best_val + 1e-12:
                    best_val, best = val, (a, v)
        if best is None:
            break
        members[best[0]] = best[1]
    return members


def main():
    failures = []
    raw = DATA_JS.read_text(encoding="utf-8")
    D = json.loads(raw[raw.index("{"):raw.rstrip().rstrip(";").rindex("}") + 1])
    ids = [n["id"] for n in D["nodes"]]
    N = len(ids)

    js = subprocess.run(
        ["node", "-e", HARNESS, "--", str(CORE_JS), str(DATA_JS), json.dumps(SEED_SETS), json.dumps(HOSTS)],
        capture_output=True, text=True, encoding="utf-8")
    if js.returncode != 0:
        print(js.stderr)
        sys.exit("node harness failed")
    R = json.loads(js.stdout)

    # --- the shipped network against the TSVs
    G, meta = B.read()
    H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    print(f"network: {G.number_of_nodes()} philosophers, {G.number_of_edges()} links; "
          f"giant component {H.number_of_nodes()} / {H.number_of_edges()}")
    if sorted(H.nodes()) != ids:
        failures.append("shipped node list is not the giant component in index order")
    index = {n: i for i, n in enumerate(ids)}
    shipped_edges = {(u, v): w for u, v, w in D["edges"]}
    nx_edges = {(min(index[u], index[v]), max(index[u], index[v])): d["weight"] for u, v, d in H.edges(data=True)}
    if shipped_edges != nx_edges:
        failures.append(f"edges differ: shipped {len(shipped_edges)}, networkx {len(nx_edges)}")
    if R["N"] != H.number_of_nodes() or R["m"] != H.number_of_edges():
        failures.append(f"js graph {R['N']}/{R['m']} vs networkx {H.number_of_nodes()}/{H.number_of_edges()}")
    if R["W"] != H.size(weight="weight"):
        failures.append(f"total weight js={R['W']} nx={H.size(weight='weight')}")
    deg = dict(H.degree())
    strength = dict(H.degree(weight="weight"))
    if any(R["deg"][index[n]] != deg[n] for n in ids) or any(D["nodes"][index[n]]["deg"] != deg[n] for n in ids):
        failures.append("degrees disagree")
    if any(R["str"][index[n]] != strength[n] for n in ids):
        failures.append("strengths disagree")

    # --- modularity, on every partition the game shows
    def part(labels):
        groups = {}
        for n, l in zip(ids, labels):
            groups.setdefault(l, set()).add(n)
        return list(groups.values())

    lou = D["louvain"]["labels"]
    checks = [
        ("louvain", R["qLouvain"], nx.community.modularity(H, part(lou), weight=None), D["louvain"]["q"]),
        ("louvain weighted", R["qwLouvain"], nx.community.modularity(H, part(lou), weight="weight"), D["louvain"]["qw"]),
        ("greedy", R["qGreedy"], nx.community.modularity(H, part(D["greedy"]["labels"]), weight=None), D["greedy"]["q"]),
        ("by century", R["qCentury"], nx.community.modularity(H, part([n["era"] for n in D["nodes"]]), weight=None), D["century"]["q"]),
    ]
    for name, q_js, q_nx, q_shipped in checks:
        print(f"Q {name:18s} js {q_js:.6f}  networkx {q_nx:.6f}  shipped {q_shipped:.6f}")
        if abs(q_js - q_nx) > 1e-9:
            failures.append(f"Q {name}: js {q_js} vs networkx {q_nx}")
        if abs(q_shipped - q_nx) > 1e-5:
            failures.append(f"Q {name}: shipped {q_shipped} vs networkx {q_nx}")

    # --- the shipped Louvain partition is networkx's, for the seed it names, and the best of 20
    seed = D["louvain"]["seed"]
    again = nx.community.louvain_communities(H, weight=None, seed=seed)
    if B.nmi(B.labels_of(again, ids), lou) < 1 - 1e-12:
        failures.append(f"shipped Louvain partition is not networkx's for seed {seed}")
    qs = [nx.community.modularity(H, nx.community.louvain_communities(H, weight=None, seed=s), weight=None)
          for s in range(D["meta"]["seeds"])]
    print(f"louvain over {len(qs)} seeds: Q {min(qs):.4f}..{max(qs):.4f}; shipped best {D['louvain']['q']:.4f}")
    if abs(max(qs) - D["louvain"]["q"]) > 1e-6:
        failures.append(f"shipped Louvain Q {D['louvain']['q']} is not the best of the seeds ({max(qs)})")
    if [round(min(qs), 4), round(max(qs), 4)] != D["louvain"]["qRange"]:
        failures.append("shipped qRange is stale")

    # --- NMI against scikit-learn if we have it, else against the build script's own
    try:
        from sklearn.metrics import normalized_mutual_info_score as sk_nmi
        ref = "scikit-learn"
    except ImportError:
        sk_nmi, ref = B.nmi, "build script"
    era = [n["era"] for n in D["nodes"]]
    for name, js_v, a, b in (("century vs louvain", R["nmiCentury"], era, lou),
                             ("greedy vs louvain", R["nmiGreedy"], D["greedy"]["labels"], lou)):
        v = sk_nmi(a, b)
        print(f"NMI {name:20s} js {js_v:.6f}  {ref} {v:.6f}")
        if abs(js_v - v) > 1e-9:
            failures.append(f"NMI {name}: js {js_v} vs {ref} {v}")

    # --- the seating rule, guest for guest
    adj = [[] for _ in ids]
    w = [[] for _ in ids]
    for u, v, ww in D["edges"]:
        adj[u].append(v); w[u].append(ww)
        adj[v].append(u); w[v].append(ww)
    for s in R["seatings"]:
        labels, rounds, sweeps = py_propagate(adj, w, s["seeds"], s["weighted"])
        tag = f"set {s['set']} ({len(s['seeds'])} guests, {'weighted' if s['weighted'] else 'plain'})"
        q_nx = nx.community.modularity(H, part(labels), weight=None)
        sizes = sorted((labels.count(t) for t in set(labels)), reverse=True)
        print(f"seating {tag}: {rounds} rounds + {sweeps} sweeps, Q {s['q']:.4f}, NMI vs louvain {s['nmi']:.3f}, "
              f"tables {sizes}, random host {s['random']:+.4f}")
        if labels != s["labels"]:
            diff = sum(1 for a, b in zip(labels, s["labels"]) if a != b)
            failures.append(f"seating {tag}: js and python disagree on {diff} guests")
        if rounds != s["rounds"] or sweeps != s["sweeps"] or not s["converged"]:
            failures.append(f"seating {tag}: rounds js={s['rounds']} py={rounds}, sweeps js={s['sweeps']} "
                            f"py={sweeps}, converged={s['converged']}")
        if abs(q_nx - s["q"]) > 1e-9:
            failures.append(f"seating {tag}: Q js {s['q']} vs networkx {q_nx}")
        if -1 in labels:
            failures.append(f"seating {tag}: someone was left unseated on a connected network")
        if abs(s["random"]) > 0.02:
            failures.append(f"seating {tag}: the random host scores {s['random']}, should be about 0")

    # --- ONE TABLE: scores, the greedy host and the record, guest for guest
    deg_list = [len(a) for a in adj]
    m = H.number_of_edges()
    for t in R["tables"]:
        host, k = t["host"], t["k"]
        tag = f"{t['name']} k={k}"
        g_py = py_greedy(adj, deg_list, m, host, k)
        r_py = py_improve(adj, deg_list, m, host, g_py)
        if g_py != t["greedy"]:
            failures.append(f"greedy table {tag}: js {t['greedy']} vs python {g_py}")
        if r_py != t["record"]:
            failures.append(f"record table {tag}: js {t['record']} vs python {r_py}")
        for label, members, score in (("greedy", t["greedy"], t["greedyScore"]), ("record", t["record"], t["recordScore"])):
            inside, dsum, above = py_table(adj, deg_list, m, members)
            # the share of modularity of one table = its term in networkx's modularity of {table, singletons}
            others = [{ids[i]} for i in range(len(ids)) if i not in set(members)]
            q_nx = nx.community.modularity(H, [{ids[i] for i in members}] + others, weight=None)
            q_single = sum(-(deg_list[i] / (2 * m)) ** 2 for i in range(len(ids)) if i not in set(members))
            if inside != score["inside"] or dsum != score["dsum"] or abs(above - score["above"]) > 1e-9:
                failures.append(f"{label} table {tag}: score js {score} vs python {(inside, dsum, above)}")
            if abs((q_nx - q_single) - score["q"]) > 1e-9:
                failures.append(f"{label} table {tag}: Q share js {score['q']} vs networkx {q_nx - q_single}")
            if abs(score["above"] - m * score["q"]) > 1e-6:
                failures.append(f"{label} table {tag}: above {score['above']} is not m * q {m * score['q']}")
        if t["recordScore"]["above"] < t["greedyScore"]["above"] - 1e-9:
            failures.append(f"{tag}: the record is worse than greedy")
        if t["random"] > t["greedyScore"]["above"]:
            failures.append(f"{tag}: random {t['random']} beats greedy {t['greedyScore']['above']}")
        names = ", ".join(D["nodes"][i]["name"] for i in t["record"][1:])
        print(f"one table {tag:24s} greedy {t['greedyScore']['above']:6.2f}  record {t['recordScore']['above']:6.2f}  "
              f"random {t['random']:5.2f}  | record: {names}")

    # --- the ladder the verdicts assume
    ladder = [("louvain", D["louvain"]["q"]), ("greedy", D["greedy"]["q"]), ("by century", D["century"]["q"]),
              ("shuffle", D["shuffle"]["mean"]), ("random host", 0.0)]
    for (a, qa), (b, qb) in zip(ladder, ladder[1:]):
        if not qa > qb:
            failures.append(f"ladder out of order: {a} {qa} <= {b} {qb}")
    print("ladder: " + " > ".join(f"{n} {q:.3f}" for n, q in ladder))

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("\nthe shipped engine agrees with networkx on every number the game shows")


if __name__ == "__main__":
    main()
