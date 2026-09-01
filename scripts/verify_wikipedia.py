"""
Stretch goal: re-derive a corner of the snapshot straight from Wikipedia and see
whether it still agrees.

The course network was harvested from the raw wiki source, where an internal link
is written [[Page name]] or [[Page name|display text]]. We ask the API for that
same raw source for a handful of characters, extract the links ourselves, keep
only the ones pointing at another character in the 303-node roster, and diff the
result against the frozen edge list.

Two things make an honest diff harder than it looks:

  1. Redirects. [[Peter Parker]] and [[Spider-Man]] are the same article. The
     roster is redirect-resolved, so we have to resolve too, in one batched
     query, or every redirect shows up as a fake disagreement.
  2. Time. The snapshot is frozen at 2026-08-26 and Wikipedia is not. Any diff we
     find is a real edit, not a bug, so we report it as drift rather than error.

Wikipedia answers 403 without a User-Agent that identifies the client, hence the
UA below.

Run:  python scripts/verify_wikipedia.py [--n 12] [--all]
"""

import argparse
import csv
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
API = "https://en.wikipedia.org/w/api.php"
UA = ("SocialGraphs2026-Week1/1.0 (DTU 02805 student project; "
      "https://github.com/Oddvar112/Social-Graphs-and-Interactions)")

# [[Target]] or [[Target|label]], skipping [[File:...]], [[Category:...]] and #anchors.
LINK_RE = re.compile(r"\[\[([^\[\]|#]+?)(?:\|[^\[\]]*?)?\]\]")


def api(params):
    params = dict(params, format="json", formatversion="2")
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def read_tsv(path, fieldnames=None):
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t", fieldnames=fieldnames))


def fetch_wikitext(titles):
    """Raw wiki source for up to 50 titles per request."""
    out = {}
    for i in range(0, len(titles), 20):
        chunk = titles[i:i + 20]
        data = api({"action": "query", "prop": "revisions", "rvprop": "content",
                    "rvslots": "main", "titles": "|".join(chunk)})
        for page in data.get("query", {}).get("pages", []):
            if "revisions" not in page:
                continue
            out[page["title"]] = page["revisions"][0]["slots"]["main"]["content"]
        time.sleep(0.2)
    return out


def resolve_redirects(titles):
    """Map every raw link target onto the article it actually lands on."""
    mapping = {}
    titles = list(titles)
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        data = api({"action": "query", "redirects": "1", "titles": "|".join(chunk)})
        q = data.get("query", {})
        norm = {r["from"]: r["to"] for r in q.get("normalized", [])}
        redir = {r["from"]: r["to"] for r in q.get("redirects", [])}
        for t in chunk:
            cur = norm.get(t, t)
            cur = redir.get(cur, cur)
            mapping[t] = cur
        time.sleep(0.2)
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="how many characters to re-derive")
    ap.add_argument("--all", action="store_true", help="re-derive all 303 (slow)")
    args = ap.parse_args()

    roster = read_tsv(ROOT / "week1_nodes.tsv")
    ids = [r["node_id"] for r in roster]
    id_set = set(ids)
    # node_id uses underscores, the API speaks spaces.
    to_title = {i: i.replace("_", " ") for i in ids}
    to_id = {v: k for k, v in to_title.items()}

    snapshot = {}
    for row in read_tsv(ROOT / "week1_edges.tsv", ["source", "target"]):
        snapshot.setdefault(row["source"], set()).add(row["target"])

    # A deterministic spread: the biggest hubs, a few mid-sized pages, a few isolates.
    if args.all:
        sample = ids
    else:
        out_counts = {i: len(snapshot.get(i, ())) for i in ids}
        ranked = sorted(ids, key=lambda i: -out_counts[i])
        isolates = [i for i in ids if out_counts[i] == 0]
        n = args.n
        sample = ranked[:n // 2] + ranked[len(ranked) // 2: len(ranked) // 2 + max(1, n // 3)]
        sample += isolates[:max(1, n - len(sample))]
        sample = list(dict.fromkeys(sample))[:n]

    print(f"Re-deriving {len(sample)} of {len(ids)} characters from live Wikipedia\n")

    pages = fetch_wikitext([to_title[i] for i in sample])

    # One batched redirect resolution for every raw target we saw.
    raw_targets = set()
    per_page_raw = {}
    for title, text in pages.items():
        found = {m.group(1).strip() for m in LINK_RE.finditer(text)}
        found = {f for f in found if ":" not in f}          # drop File:, Category:, etc.
        per_page_raw[title] = found
        raw_targets |= found
    resolved = resolve_redirects(sorted(raw_targets))

    rows, tot_snap, tot_live, tot_both = [], 0, 0, 0
    for node_id in sample:
        title = to_title[node_id]
        if title not in pages:
            print(f"  (no wikitext returned for {title}, skipped)")
            continue
        live = set()
        for raw in per_page_raw[title]:
            landed = resolved.get(raw, raw)
            nid = to_id.get(landed)
            if nid and nid in id_set and nid != node_id:
                live.add(nid)
        snap = snapshot.get(node_id, set())
        both = snap & live
        rows.append((node_id, len(snap), len(live), len(both), sorted(snap - live), sorted(live - snap)))
        tot_snap += len(snap)
        tot_live += len(live)
        tot_both += len(both)

    w = max(len(r[0]) for r in rows) + 2
    print(f"{'character':<{w}}{'snap':>6}{'live':>6}{'both':>6}   drift")
    print("-" * (w + 30))
    for node_id, s, l, b, only_snap, only_live in rows:
        drift = ""
        if only_snap:
            drift += "only in snapshot: " + ", ".join(only_snap[:3])
        if only_live:
            drift += ("  |  " if drift else "") + "only live now: " + ", ".join(only_live[:3])
        print(f"{node_id:<{w}}{s:>6}{l:>6}{b:>6}   {drift}")

    jac = tot_both / (tot_snap + tot_live - tot_both) if (tot_snap + tot_live - tot_both) else 1.0
    print("-" * (w + 30))
    print(f"{'TOTAL':<{w}}{tot_snap:>6}{tot_live:>6}{tot_both:>6}")
    print(f"\nagreement (Jaccard over all sampled edges): {jac:.3f}")
    print(f"edges in the snapshot we could not reproduce: {tot_snap - tot_both}")
    print(f"edges live Wikipedia has that the snapshot does not: {tot_live - tot_both}")

    out = ROOT / "data" / "verification.json"
    out.write_text(json.dumps({
        "sampled": len(rows),
        "snapshot_edges": tot_snap,
        "live_edges": tot_live,
        "shared": tot_both,
        "jaccard": round(jac, 3),
        "rows": [{"node": r[0], "snapshot": r[1], "live": r[2], "shared": r[3],
                  "only_snapshot": r[4], "only_live": r[5]} for r in rows],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
