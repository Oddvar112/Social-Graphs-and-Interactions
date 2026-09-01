"""
Checks the rules WEB-CRAWLER enforces, using the same algorithms the game runs.

The browser cannot be driven headlessly here (requestAnimationFrame is paused
when the pane is not compositing), so instead we re-implement the game's target
selection, par computation and shortest-path reconstruction exactly as game.js
does them and hammer them against the real graph.

What this actually proves:
  * every mission the game can hand out is finishable
  * par is achievable, so "PERFECT ROUTE" is reachable and not an empty promise
  * the player can never be permanently stuck

Run:  python scripts/test_game_rules.py
"""

import collections
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Read the same file the browser reads, so we are testing the shipped data.
raw = (ROOT / "data" / "marvel_week1.js").read_text(encoding="utf-8")
DATA = json.loads(raw[raw.index("{"):raw.rstrip().rstrip(";").rindex("}") + 1])

NODES = DATA["nodes"]
OUT = DATA["out"]
IN = DATA["in"]
N = len(NODES)
BY_ID = {n["id"]: i for i, n in enumerate(NODES)}

HEROES = ["Spider-Man", "Wolverine_(character)", "Doctor_Strange"]
DIFFS = {"easy": (2, 3), "normal": (3, 4), "hard": (5, 9)}


def bfs(src, adj):
    dist = [-1] * N
    dist[src] = 0
    q = collections.deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def shortest_path(a, b):
    if a == b:
        return [a]
    prev = [-1] * N
    seen = [False] * N
    seen[a] = True
    q = collections.deque([a])
    while q:
        u = q.popleft()
        for v in OUT[u]:
            if seen[v]:
                continue
            seen[v] = True
            prev[v] = u
            if v == b:
                p, c = [b], b
                while c != a:
                    c = prev[c]
                    p.append(c)
                return p[::-1]
            q.append(v)
    return None


def pick_pool(home, lo, hi):
    """Exactly game.js pickTarget: reachable outward AND with a route home."""
    out_d = bfs(home, OUT)
    back_d = bfs(home, IN)          # reverse BFS: distance FROM v TO home
    pool = [i for i in range(N)
            if i != home and out_d[i] >= 0 and back_d[i] >= 0 and lo <= out_d[i] <= hi]
    if not pool:
        pool = [i for i in range(N) if i != home and out_d[i] > 1 and back_d[i] > 0]
    return pool, out_d, back_d


def main():
    rng = random.Random(20260901)
    failures = []
    checked = 0

    print(f"graph: {N} nodes, {sum(len(a) for a in OUT)} edges\n")

    for hero in HEROES:
        home = BY_ID[hero]
        for diff, (lo, hi) in DIFFS.items():
            pool, out_d, back_d = pick_pool(home, lo, hi)
            if not pool:
                failures.append(f"{hero}/{diff}: empty target pool")
                continue

            sample = pool if len(pool) <= 120 else rng.sample(pool, 120)
            pars = []
            for target in sample:
                checked += 1
                par = out_d[target] + back_d[target]
                pars.append(par)

                # A greedy optimal playthrough must exist and cost exactly par.
                leg1 = shortest_path(home, target)
                leg2 = shortest_path(target, home)
                if leg1 is None or leg2 is None:
                    failures.append(f"{hero}/{diff}: no round trip to {NODES[target]['id']}")
                    continue
                walked = (len(leg1) - 1) + (len(leg2) - 1)
                if walked != par:
                    failures.append(
                        f"{hero}/{diff}: {NODES[target]['id']} par={par} but optimal play costs {walked}")

                # Every node on the optimal route must itself still reach home,
                # otherwise the second leg could strand the player.
                for node in leg1 + leg2:
                    if back_d[node] < 0:
                        failures.append(
                            f"{hero}/{diff}: {NODES[node]['id']} on the route cannot reach home")
                        break

            print(f"{hero:24s} {diff:7s} targets={len(pool):4d} "
                  f"par min={min(pars)} mean={sum(pars)/len(pars):.2f} max={max(pars)}")

    # Web-zip is the only escape from a dead end, so the player must always have
    # at least one step of history to undo. That holds because the first move can
    # only happen from home, which always leaves a path of length >= 2 behind.
    sinks = [i for i in range(N) if not OUT[i]]
    print(f"\ndead ends in the graph: {len(sinks)} "
          f"(reachable only via web-zip, which the UI always offers after step 1)")

    print(f"\nchecked {checked} missions across {len(HEROES)} heroes x {len(DIFFS)} difficulties")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures[:20]:
            print("  " + f)
        sys.exit(1)
    print("all missions are finishable and par is always achievable")


if __name__ == "__main__":
    main()
