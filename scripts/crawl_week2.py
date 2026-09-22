"""
Week 2 crawl: the heroes' natural follow-up, the villains.

Builds a second network next to the frozen course snapshot: every article in
Wikipedia's Category:Marvel Comics supervillains, filtered the way the course
filtered the heroes (no redirects, no "List of" pages, no teams), plus the 303
heroes of the week-1 roster, and every link between any two of them.

The result is one directed network with a side label on every node:
  hero      in the week-1 roster only
  villain   in the supervillain category only
  both      in the roster AND the category (the turncoats)

Also harvested, because week 2 is about growth models: each character's debut
year, read from the "debut" field of the comics-character infobox. That is the
number that lets us test the Barabasi-Albert prediction that the first to
arrive become the hubs.

Outputs
  data/week2_nodes.tsv     node_id  name  side  debut_year  url
  data/week2_edges.tsv     source  target   (directed, within the union)
  data/week2_crawl.json    counts, the filter decisions, and the date

Wikipedia answers 403 without a User-Agent that identifies the client. Every
request below carries one, and we sleep between requests.

Run:  python scripts/crawl_week2.py
"""

import csv
import datetime
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
API = "https://en.wikipedia.org/w/api.php"
UA = ("SocialGraphs2026-Week2/1.0 (DTU 02805 student project; "
      "https://github.com/Oddvar112/Social-Graphs-and-Interactions)")
CATEGORY = "Category:Marvel Comics supervillains"

# [[Target]] or [[Target|label]]; anchors and File:/Category: are dropped later.
LINK_RE = re.compile(r"\[\[([^\[\]|]+?)(?:\|[^\[\]]*?)?\]\]")
YEAR_RE = re.compile(r"\b(19[3-9]\d|20[0-2]\d)\b")
REF_RE = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.S)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)

# Infoboxes that mean "this page is not one character".
NOT_A_CHARACTER = ("comics organization", "comics team", "comics series", "comic book title",
                   "graphic novel", "comics publisher", "comics story", "comics elements",
                   "comics species", "comics location", "comics set index", "comics race")


def api(params, retries=4):
    params = dict(params, format="json", formatversion="2")
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as exc:                     # noqa: BLE001, we retry everything once
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
            print("   retry after", exc)


def read_tsv(path, fieldnames=None):
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t", fieldnames=fieldnames))


def category_members(category):
    """Every article (namespace 0) directly in the category, with redirect flag."""
    members, cont = [], {}
    while True:
        d = api({"action": "query", "generator": "categorymembers", "gcmtitle": category,
                 "gcmlimit": "500", "gcmnamespace": "0", "prop": "info", **cont})
        members += d["query"]["pages"]
        if "continue" not in d:
            break
        cont = d["continue"]
        time.sleep(0.2)
    return members


def fetch_wikitext(titles, batch=20):
    """Raw wiki source, plus the title Wikipedia normalises each request to."""
    out, normalised = {}, {}
    for i in range(0, len(titles), batch):
        chunk = titles[i:i + batch]
        d = api({"action": "query", "prop": "revisions", "rvprop": "content",
                 "rvslots": "main", "titles": "|".join(chunk)})
        q = d.get("query", {})
        for n in q.get("normalized", []):
            normalised[n["from"]] = n["to"]
        for page in q.get("pages", []):
            if "revisions" in page:
                out[page["title"]] = page["revisions"][0]["slots"]["main"]["content"]
        time.sleep(0.2)
        if (i // batch) % 10 == 0:
            print(f"   wikitext {min(i + batch, len(titles))}/{len(titles)}")
    return out, normalised


def redirects_into(titles, batch=50):
    """Every redirect title that lands on one of `titles`: {redirect -> target}.

    Asking each page for its own redirects is 15 requests for 750 pages.
    Resolving every raw link target we meet instead would be ten thousand
    titles, so we go this way round.
    """
    mapping = {}
    for i in range(0, len(titles), batch):
        chunk = titles[i:i + batch]
        cont = {}
        while True:
            d = api({"action": "query", "prop": "redirects", "rdlimit": "max",
                     "rdnamespace": "0", "titles": "|".join(chunk), **cont})
            for page in d.get("query", {}).get("pages", []):
                for r in page.get("redirects", []):
                    mapping[r["title"]] = page["title"]
            if "continue" not in d:
                break
            cont = d["continue"]
            time.sleep(0.2)
        time.sleep(0.2)
    return mapping


def normalise_title(raw):
    """What MediaWiki does to a link target before looking it up."""
    t = raw.split("#", 1)[0].replace("_", " ").strip()
    t = re.sub(r"\s+", " ", t)
    if not t:
        return ""
    return t[0].upper() + t[1:]


def infobox_kind(text):
    m = re.search(r"\{\{\s*Infobox\s+([^\n|{}]+)", text)
    return m.group(1).strip().lower() if m else None


def infobox_field(text, name):
    """The value of one infobox parameter, however many lines it spans."""
    m = re.search(r"\n\s*\|\s*" + name + r"\s*=", text)
    if not m:
        return None
    start = m.end()
    depth = 0
    i = start
    while i < len(text):
        two = text[i:i + 2]
        if two == "{{":
            depth += 1
            i += 2
            continue
        if two == "}}":
            if depth == 0:
                break
            depth -= 1
            i += 2
            continue
        if text[i] == "\n" and depth == 0 and re.match(r"\s*\|", text[i + 1:i + 40]):
            break
        i += 1
    return text[start:i]


def debut_year(text):
    field = infobox_field(text, "debut")
    if field is None:
        field = infobox_field(text, "first")
    if not field:
        return None
    field = COMMENT_RE.sub("", REF_RE.sub("", field))
    years = [int(y) for y in YEAR_RE.findall(field)]
    return min(years) if years else None


def main():
    started = datetime.datetime.now(datetime.timezone.utc)
    roster = read_tsv(ROOT / "week1_nodes.tsv")
    hero_ids = [r["node_id"] for r in roster]
    hero_titles = {r["node_id"].replace("_", " "): r["node_id"] for r in roster}

    print(f"1. {CATEGORY}")
    members = category_members(CATEGORY)
    redirects = [m["title"] for m in members if "redirect" in m]
    lists = [m["title"] for m in members if m["title"].startswith("List of")]
    candidates = [m["title"] for m in members if "redirect" not in m and not m["title"].startswith("List of")]
    print(f"   {len(members)} members, {len(redirects)} redirects, {len(lists)} lists, {len(candidates)} candidates")

    # One union of titles to fetch. Hero titles come from the roster, so a hero
    # who is also in the villain category is fetched once under the roster title.
    union_titles = list(dict.fromkeys(list(hero_titles) + candidates))
    print(f"2. wikitext for {len(union_titles)} pages")
    pages, normalised = fetch_wikitext(union_titles)
    missing = [t for t in union_titles if normalised.get(t, t) not in pages]
    print(f"   got {len(pages)}, missing {len(missing)}: {missing[:8]}")

    # Filter the villain candidates the way the roster is filtered.
    dropped = {}
    villain_titles = []
    for t in candidates:
        text = pages.get(normalised.get(t, t), "")
        kind = infobox_kind(text) or ""
        if "{{disambiguation" in text.lower() or "{{disambig" in text.lower() or "{{dab" in text.lower():
            dropped[t] = "disambiguation"
        elif any(kind.startswith(bad) for bad in NOT_A_CHARACTER):
            dropped[t] = f"infobox: {kind}"
        elif not text:
            dropped[t] = "no wikitext"
        else:
            villain_titles.append(t)
    print(f"3. villains kept {len(villain_titles)}, dropped {len(dropped)}")
    for t, why in list(dropped.items())[:12]:
        print(f"   - {t}: {why}")

    villain_set = set(villain_titles)
    nodes = {}                                   # title -> record
    for title, node_id in hero_titles.items():
        nodes[title] = {"node_id": node_id, "name": title,
                        "side": "both" if title in villain_set else "hero"}
    for title in villain_titles:
        if title in nodes:
            continue
        nodes[title] = {"node_id": title.replace(" ", "_"), "name": title, "side": "villain"}
    print(f"   union: {len(nodes)} nodes "
          f"({sum(n['side'] == 'hero' for n in nodes.values())} hero, "
          f"{sum(n['side'] == 'villain' for n in nodes.values())} villain, "
          f"{sum(n['side'] == 'both' for n in nodes.values())} both)")

    print("4. redirects into the union")
    redir = redirects_into(sorted(nodes))
    print(f"   {len(redir)} redirect titles land on a union page")

    def land(raw):
        t = normalise_title(raw)
        if not t or ":" in t.split(" ")[0]:
            return None
        t = redir.get(t, t)
        return t if t in nodes else None

    print("5. edges")
    edges = set()
    kinds = {}
    for title, rec in nodes.items():
        text = pages.get(normalised.get(title, title))
        if text is None:
            continue
        kinds[title] = infobox_kind(text)
        rec["debut_year"] = debut_year(text)
        for m in LINK_RE.finditer(text):
            target = land(m.group(1))
            if target and target != title:
                edges.add((rec["node_id"], nodes[target]["node_id"]))
    edges = sorted(edges)

    # How much of the frozen snapshot does the live hero-hero part reproduce?
    snap = {(r["source"], r["target"]) for r in read_tsv(ROOT / "week1_edges.tsv", ["source", "target"])}
    hero_id_set = set(hero_ids)
    live_hh = {(s, t) for s, t in edges if s in hero_id_set and t in hero_id_set}
    jacc = len(snap & live_hh) / len(snap | live_hh)
    print(f"   {len(edges)} edges in the union; hero-hero live {len(live_hh)} vs snapshot {len(snap)}, "
          f"shared {len(snap & live_hh)}, Jaccard {jacc:.3f}")

    with_year = sum(1 for n in nodes.values() if n.get("debut_year"))
    print(f"   debut year found for {with_year}/{len(nodes)}")

    DATA.mkdir(exist_ok=True)
    with open(DATA / "week2_nodes.tsv", "w", encoding="utf-8", newline="") as f:
        f.write(f"# Week 2 crawl of live Wikipedia, {started:%Y-%m-%d}. Heroes = the frozen week-1 roster, "
                f"villains = {CATEGORY} (no redirects, lists, teams or disambiguation pages).\n")
        f.write("# side: hero | villain | both. debut_year from the infobox, blank when not found.\n")
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["node_id", "name", "side", "debut_year", "url"])
        for title in sorted(nodes, key=lambda t: nodes[t]["node_id"]):
            rec = nodes[title]
            w.writerow([rec["node_id"], rec["name"], rec["side"], rec.get("debut_year") or "",
                        "https://en.wikipedia.org/wiki/" + urllib.parse.quote(rec["node_id"])])
    with open(DATA / "week2_edges.tsv", "w", encoding="utf-8", newline="") as f:
        f.write(f"# Week 2 crawl of live Wikipedia, {started:%Y-%m-%d}. Edge A -> B when the article of A links "
                f"to the article of B; both in data/week2_nodes.tsv. {len(nodes)} nodes, {len(edges)} edges.\n")
        f.write("source\ttarget\n")               # a real header, not the week-1 comment trap
        for s, t in edges:
            f.write(f"{s}\t{t}\n")

    meta = {
        "crawled": started.isoformat(timespec="seconds"),
        "category": CATEGORY,
        "category_members": len(members),
        "category_redirects": len(redirects),
        "category_lists": len(lists),
        "villain_candidates": len(candidates),
        "villains_kept": len(villain_titles),
        "dropped": dropped,
        "nodes": len(nodes),
        "sides": {s: sum(n["side"] == s for n in nodes.values()) for s in ("hero", "villain", "both")},
        "both": sorted(n["name"] for n in nodes.values() if n["side"] == "both"),
        "edges": len(edges),
        "redirect_titles": len(redir),
        "hero_hero_live": len(live_hh),
        "hero_hero_snapshot": len(snap),
        "hero_hero_shared": len(snap & live_hh),
        "hero_hero_jaccard": round(jacc, 4),
        "debut_years_found": with_year,
        "infobox_kinds": dict(sorted(((k or "none", sum(1 for v in kinds.values() if v == k))
                                      for k in set(kinds.values())), key=lambda x: -x[1])),
        "missing_wikitext": missing,
    }
    (DATA / "week2_crawl.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote data/week2_nodes.tsv, data/week2_edges.tsv, data/week2_crawl.json")


if __name__ == "__main__":
    main()
