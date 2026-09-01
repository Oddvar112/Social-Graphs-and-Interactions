# Capes &amp; Edges

Group site for **02805 Social Graphs and Interactions** (DTU, autumn 2026), by **Oddvar** and **David**.

Eight weekly posts about one dataset: the 303 characters in Wikipedia's
[Category:Marvel Comics superheroes](https://en.wikipedia.org/wiki/Category:Marvel_Comics_superheroes)
and the 1784 directed links between their pages.

**Live site:** https://oddvar112.github.io/Social-Graphs-and-Interactions/

## What is here

| Path | What it is |
| --- | --- |
| `index.html` | Front page: the group, the course, the week index |
| `weeks/week1/index.html` | Week 1 post: degree distributions, in vs out, components, islands |
| `weeks/week1/game/` | **WEB-CRAWLER**, a browser game played on the real graph |
| `scripts/` | Everything that produces the data and the figures |
| `assets/figures/` | Generated figures, all written by `scripts/analyse.py` |
| `data/` | Generated: graph for the game, stats, API verification output |
| `week1_edges.tsv`, `week1_nodes.tsv` | The frozen course snapshot, unmodified |

## The game

**WEB-CRAWLER** puts you on a Wikipedia article as Spider-Man, Wolverine or Doctor Strange. You get a target
somewhere out in the network and have to click your way there and back along real outgoing links, one page at a time.
The map is dark until you walk on it, and your score is compared against the shortest route that actually exists,
computed with BFS on the same 1784 edges.

It is a playable demonstration of three facts from the week 1 post: the graph is directed and only 39 percent of links
are reciprocated, 20 characters are dead ends with no outgoing link at all, and in-degree is concentrated enough that
aiming for a hub is almost always the right move.

Characters are drawn procedurally on canvas, so the game ships no image or audio files at all.

## Reproducing everything

Requires Python 3.9+ with `networkx`, `numpy` and `matplotlib`.

```bash
python scripts/build_graph.py        # snapshot -> data/marvel_week1.js (the game's graph)
python scripts/analyse.py            # every figure and number in the week 1 post
python scripts/verify_wikipedia.py   # re-derives part of the snapshot from the live Wikipedia API
python scripts/test_game_rules.py    # checks every mission the game can hand out is finishable
```

Then serve the folder and open it:

```bash
python -m http.server 8000
```

The site is plain static HTML, CSS and JavaScript with no build step. `data/marvel_week1.js` is written as a `.js`
file rather than JSON on purpose, so the game also runs when opened directly from the filesystem.

## Loading the snapshot correctly

The edge list only mentions characters that have at least one link. Add the nodes first or you silently drop the 17
isolates and end up with 286 nodes instead of 303:

```python
G = nx.DiGraph()
for row in read_tsv("week1_nodes.tsv"):    # all 303 nodes FIRST
    G.add_node(row["node_id"], **row)
for row in read_tsv("week1_edges.tsv"):    # then the 1784 edges
    G.add_edge(row["source"], row["target"])
```

One more trap: the header line of `week1_edges.tsv` is itself written as a `#` comment. If you strip comments before
parsing you also strip the header, and `csv.DictReader` will treat the first real edge as column names. Pass the field
names in explicitly.

## Data

Frozen week 1 snapshot from the [course data page](https://sunelehmann.com/socialgraphs2026-web/data/), taken
2026-08-26. Article text and link structure from Wikipedia, CC BY-SA 4.0. Marvel characters and names are trademarks
of Marvel Characters, Inc.; this is a non-commercial student project.
