# Capes &amp; Edges

Group site for **02805 Social Graphs and Interactions** (DTU, autumn 2026), by **Oddvar**, **David** and **Milena**.

Eight weekly posts about one dataset: the 303 characters in Wikipedia's
[Category:Marvel Comics superheroes](https://en.wikipedia.org/wiki/Category:Marvel_Comics_superheroes)
and the 1784 directed links between their pages. From week 2 the villains join in: 363 characters from
[Category:Marvel Comics supervillains](https://en.wikipedia.org/wiki/Category:Marvel_Comics_supervillains),
crawled by us, and every link between any two of the 602.

**Live site:** https://oddvar112.github.io/Social-Graphs-and-Interactions/

## What is here

| Path | What it is |
| --- | --- |
| `index.html` | Front page: the group, the course, the week index |
| `weeks/week1/index.html` | Week 1 post: degree distributions, in vs out, components, islands |
| `weeks/week1/game/` | **WEB-CRAWLER**, a browser game played on the real graph |
| `weeks/week2/index.html` | Week 2 post: power-law fit, friendship paradox, shuffle tests, a grown Marvel, the villains |
| `weeks/week2/grow/` | **ORIGIN STORY**, an explorable that replays the universe by debut year next to a growing model |
| `weeks/week2/hubdial/` | **THE HUB DIAL**, our exercise 2.10 explorable: non-linear preferential attachment, one file, embeddable |
| `weeks/week3/index.html` | Week 3 post: paths, four centralities, the broker shuffle test, cliques |
| `weeks/week3/game/` | **KINGPIN**, a fragmentation game scored against rival centrality strategies |
| `scripts/` | Everything that produces the data and the figures |
| `assets/figures/` | Generated figures: `analyse.py` writes the week 1 ones, `analyse_week2.py` the `week2_*` ones |
| `data/` | Generated: graph for the game, stats, API verification output, the week 2 crawl |
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

## The explorable

**ORIGIN STORY** (week 2) replays the frozen network in order of first appearance, 1939 to 2020, with every
character's debut year read from its Wikipedia infobox, and grows a preferential-attachment network of the same
size next to it (plus growth without preference, as a control). A second replay adds the 363 villains from our
crawl, red against the heroes' cyan. Nodes sit on a spiral by arrival order, so the model's first-mover prediction
is visible as a glowing centre. On Wikipedia the glow is a ring: the class of 1962. Each arrival pops up as a comic
panel: the lead image of the character's Wikipedia article, loaded from Wikipedia at view time (only the URLs are
in the repo, in `data/week2_images.js`; the art is Marvel's and is shown for identification). Vanilla JavaScript,
one canvas, `?mode=combined&t=150` opens it at a moment.

## The explorable for the course page (exercise 2.10)

**THE HUB DIAL** (`weeks/week2/hubdial/index.html`) grows a network in which newcomers attach with probability
proportional to k^α and lets you turn α from 0 to 2.5. Below 1 hubs never form, at 1 it is Barabási-Albert, above
1 one node keeps a fixed share of all links however large the network grows; a sweep draws the biggest hub's share
against α at n = 303 and n = 3000 so the size-independence is visible, with Spider-Man's 7.4 percent as the
reference line. It is one self-contained HTML file, no libraries, no data files, styled after the course's own
explorables, with `?theme=light|dark`. Embed it with an iframe of about 860 px height.
## The other game

**KINGPIN** (week 3) hands you a contract: 3, 5 or 8 hits on the undirected giant component (277 characters,
1421 links). Each hit removes a character; whoever loses their last route to the big cluster is cut off. When the
hits run out, three rival hitmen run the same contract on the same board: one shoots the highest degree, one the
highest betweenness (Brandes, recomputed after every kill), one at random, thirty times over. The score screen is
one long "compared to what", plus the best hit-list simulated annealing ever found, which beats both greedy bots
on every contract, always contains Black Widow (degree 25, the only door to her corner), and never needs her to
be famous.

The algorithms live in `weeks/week3/game/core.js` with no DOM in them, so `scripts/test_kingpin_rules.py` runs
the exact shipped file under node and diffs every number against networkx (betweenness to 1e-13, the bots' kill
orders move for move).

## Reproducing everything

Requires Python 3.9+ with `networkx`, `numpy`, `matplotlib` and (from week 2) `scipy`.

```bash
python scripts/build_graph.py        # snapshot -> data/marvel_week1.js (the game's graph)
python scripts/analyse.py            # every figure and number in the week 1 post
python scripts/verify_wikipedia.py   # re-derives part of the snapshot from the live Wikipedia API
python scripts/test_game_rules.py    # checks every mission the game can hand out is finishable

python scripts/crawl_week2.py        # villains + debut years from live Wikipedia -> data/week2_*.tsv
python scripts/analyse_week2.py      # every figure and number in the week 2 post (about ten minutes; --quick for a look)
python scripts/heavytail.py          # self-test of the Clauset-Shalizi-Newman fitter on synthetic data
python scripts/build_week2_data.py   # debut years + combined network for ORIGIN STORY -> data/week2_years.js, week2_combined.js
python scripts/fetch_images.py       # lead-image URLs for ORIGIN STORY's character cards -> data/week2_images.js

python scripts/analyse_week3.py      # every figure and number in the week 3 post (a few minutes; --quick for a look)
python scripts/test_kingpin_rules.py # runs KINGPIN's shipped core.js under node and diffs it against networkx
```

The week 2 crawl is not frozen by the course, so `data/week2_*.tsv` is committed as our own snapshot (8 September
2026) and `data/week2_crawl.json` records which category members were dropped and why. Re-running the crawl will
give slightly different numbers, because Wikipedia moves.

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
