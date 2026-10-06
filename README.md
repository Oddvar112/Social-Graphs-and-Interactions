# Capes &amp; Edges

Group site for **02805 Social Graphs and Interactions** (DTU, autumn 2026), by **Oddvar**, **David** and **Milena**.

Eight weekly posts about one dataset: the 303 characters in Wikipedia's
[Category:Marvel Comics superheroes](https://en.wikipedia.org/wiki/Category:Marvel_Comics_superheroes)
and the 1784 directed links between their pages. From week 2 the villains join in: 363 characters from
[Category:Marvel Comics supervillains](https://en.wikipedia.org/wiki/Category:Marvel_Comics_supervillains),
crawled by us, and every link between any two of the 602. In week 4 the guests change: the course's philosophers
network, every philosopher on Wikipedia born before 1900, 1444 of them with 9140 links.

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
| `weeks/week4/index.html` | Week 4 post, on the philosophers: communities as a seating plan, Louvain against two nulls, greedy and a historian, twenty seeds that disagree about Aristotle |
| `weeks/week4/game/` | **SYMPOSIUM**, a seating game scored by modularity against Louvain, with the engine in `core.js` |
| `weeks/week5/index.html` | Week 5 post: the network gets its text. Counts and Zipf, fame against page length, every link as a typed sentence tested against the communities, Wikipedia copying itself, a search engine and the heroes who are not in the roster |
| `weeks/week5/game/` | **SECRET IDENTITY**, three kinds of bag-of-words round against a cosine machine, with the engine in `core.js` |
| `weeks/week6/index.html` | Week 6 post: names or meaning. Every TF-IDF cosine split exactly into names, habit words and other words, tested against 45 pairs read by hand; the lookalikes with and without their names; story twins that the network never linked |
| `weeks/week6/taboo/` | **TABOO-IDF**, a word game: describe a secret character one word at a time while a cosine machine ranks all 303 pages; names are taboo; the engine is in `core.js` |
| `weeks/week6/dial/` | **THE NAME DIAL**, an explorable: pick a page, turn how much names and habit words count, and watch its lookalikes re-rank, with the network's agreement measured live; the engine is in `core.js` |
| `scripts/` | Everything that produces the data and the figures |
| `assets/figures/` | Generated figures: `analyse.py` writes the week 1 ones, `analyse_week2.py` the `week2_*` ones |
| `data/` | Generated: graph for the game, stats, API verification output, the week 2 crawl |
| `week1_edges.tsv`, `week1_nodes.tsv` | The frozen course snapshot, unmodified |
| `week4_philosophers_nodes.tsv`, `week4_philosophers_edges.tsv` | The course's week 4 philosophers snapshot (15 September 2026), unmodified |
| `week4_edges_weighted.tsv` | The week 1 Marvel edges with a weight, from the course's week 4 release |
| `marvel_pages.zip` | The course's week 5 release: the full plain-text Wikipedia article of all 303 characters, unmodified |

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

## The seating game

**SYMPOSIUM** (week 4) is played on the course's philosophers network: every philosopher on Wikipedia born before
1900, undirected giant component of 1374 philosophers and 9139 links. Two modes:

* **One table.** You get a philosopher at the head of an empty table and 3, 5 or 9 seats. The score is the table's
  share of modularity times m, i.e. links inside minus (degree sum)² / 4m, shown as "links above chance", with each
  chair showing what that guest added. Serving dinner compares you with a greedy host (best next guest each time), the
  best table a deterministic swap search finds, and a random pick from the host's friends, and says how many of your
  guests Louvain puts in the host's community. Greedy, the swap search and the score live in `weeks/week4/game/core.js`.
* **The whole room.** 6, 12 or 20 place cards and up to nine tables. Each card seats one philosopher by hand; everyone
  else joins the table where most of the people they link to sit, ring by ring, and then guests keep moving to their
  neighbours' majority table until nobody wants to move (label propagation seeded by the player, deterministic). The
  score is modularity of the full 1374-guest seating. Serving dinner runs Louvain (best of 20 seeds), greedy modularity,
  seating by century and a random host, and shows where you and Louvain disagree, with how many of Louvain's own 19
  other runs move each of those guests.

Portraits are Wikipedia's page images, loaded at view time and credited; only URLs are stored (`data/week4_images.js`).

`scripts/build_week4_data.py` computes everything the game compares you with and writes `data/week4_symposium.js`;
`scripts/test_symposium_rules.py` runs the shipped `core.js` under node and diffs modularity, NMI, the seating rule,
the one-table score, the greedy host and the swap search against networkx, scikit-learn and plain Python
re-implementations, guest for guest.

## The word game

**SECRET IDENTITY** (week 5) is played on the text of the 303 pages. A case file deals three kinds of round:

* **Unmask.** One character's page as a bag of words, one clue at a time (with its count on the page and the number
  of pages it occurs in), four suspects; fewer clues, more points. Clue order is raw counts, NLTK stopwords removed,
  or distinctive (count × ln(303 / document frequency), next week's TF-IDF). Decoys are random or the three pages
  whose vectors point most in the same direction. A character's own names are never clues.
* **Impostor.** A real sentence from a page or a sentence from a trigram model trained on one Louvain community; the
  generator refuses anything that shares a nine-word run with the corpus, and both kinds go through the same
  detokenizer with matched lengths.
* **Redacted.** A concordance line with the key word blacked out; two of the three decoys are NLTK `similar()` words.

A bag-of-words machine plays every unmask round against you: the cosine between the revealed clues and each
suspect's stopword-free page vector. It never sees the answer; it reads at least two clues and commits as soon as one
suspect scores twice the runner-up (or is forced to guess on the tenth), and a wrong commitment scores zero. You see
when it commits, not to whom, until you commit yourself; the reveal shows the four cosines at that moment. Suspects
come from the 80 best-known characters or the full roster. A case of the day (seeded by the date) gives everyone the
same twelve rounds and a result to copy. The engine lives in `weeks/week5/game/core.js`; `scripts/build_week5_data.py`
writes `data/week5_secret.js`; `scripts/test_secret_identity.py` runs the shipped engine under node and diffs its
scores, picks and commitments against Python, checks that no generated sentence copies the corpus and that no clue
leaks a name, and reports how the machine fares over 300 simulated rounds per setting (5.6 points a round on the
default settings, 0.7 on raw counts).

The shared text machinery (loading, spaCy tokenization with a cache in `.cache/`, the trigram model, the
dependency-parsed relationship labeller, Louvain) is `scripts/week5_text.py`; `scripts/analyse_week5.py` produces every
figure and number in the post, and `data/week5_label_audit.tsv` holds the sixty labelled sentences we read by hand.

## The name dial

**THE NAME DIAL** (week 6) is the course's last 6.11 opener as an instrument: *find a way to tell name-driven matches from
story-driven ones.* The way is arithmetic. With TF-IDF as the course page defines it (tf = count / words, idf = ln(303 / df)) and
unit-length rows, the cosine of two pages is a sum with one term per word, so it splits exactly into three parts that add up
to it: the part from **names** (capitalised in the middle of a sentence, or in a page title), the part from **habit words**
(NLTK stopwords, and any word on at least a third of the 303 pages: *she*, *her*, *voiced*, *playable*, *series*) and the part
from every other word. The verdict on a pair is a rule on the three numbers: NAME if names are at least half of the cosine,
otherwise HABIT if habit words are at least as big as the rest, otherwise STORY. Nothing is learned.

The dial turns that split into a knob. Multiply the names part of every vector by a setting between 0 and 1, the habit part by another,
renormalise, and the cosine at that setting is `(wn^2 Pn + wh^2 Ph + Po) / (norm norm)` where Pn, Ph, Po are the three 303 x 303 matrices
of the split. So the whole thing runs in the browser from `data/week6_dial.js` (three matrices in millionths plus the links, 1.4 MB), no server:
choose a page, drag the two sliders, and its twelve nearest pages re-rank live as bars split into names, habit words and the rest; a meter
shows how many of its ten nearest are linked in the network, another the average over all 303 pages, and a curve shows that average as the
names dial goes from 0 to 100 (4.0 at 100, 2.0 at 0). The state is in the address, so a link keeps the dials.

`scripts/week6_text.py` is the shared pipeline (corpus, name rule, TF-IDF, decomposition, verdict, network distances, Louvain);
`scripts/analyse_week6.py` produces every figure and number in the post and reads `data/week6_pair_audit.tsv`, 45 pairs read by hand
and labelled before looking at the rule's verdict, back in to report how often the rule agrees (36 of 41 clear cases);
`scripts/build_week6_dial.py` writes the dial's data; `scripts/test_name_dial.py` recomputes the matrices from scratch in plain Python,
checks the shipped `core.js` against numpy and against the analysis' agreement numbers, and drives the shipped `dial.js` against a stub page,
under node or, if node is missing, macOS's built-in JavaScript engine.

## The Taboo game

**TABOO-IDF** (week 6) turns the machine round: a secret character is on screen and you make a cosine-similarity machine find them by
typing words. After every word all 303 pages are ranked by the cosine between your word list and their TF-IDF vectors (the sum of each
page's weight for your words over the square root of how many you typed); you win when the target is first. A word the name rule calls a
name is a strike when names are banned (it costs a word and the machine ignores it); a word fewer than two pages use is unknown and free;
you have twelve words, strikes and hints included, and score 11 minus the words you used. A hint spells the first letter, then three
letters, of one of the page's strongest plain words, for three words each. `data/week6_taboo.js` holds 14,860 words (every word on at least two
pages), their postings (217,026 page weights) and, for the 216 pages with 800 or more words, the oracle: the one word that puts the page ahead of every
other by the widest margin, with names banned and with names allowed, and its twelve strongest plain words. An oracle wins in one word whatever the
rule; with names allowed its word is a name for 212 of 216 pages and wins by a median margin of 0.58, with names banned 0.04. The game is the distance between
what the machine can do and what you can guess about a page's vocabulary. `scripts/build_week6_taboo.py` writes the data; `scripts/test_taboo.py`
recomputes every posting from the corpus, runs the shipped `core.js` against numpy and plays whole games through the shipped `game.js` against a stub page.

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

python scripts/build_week4_data.py --png assets/figures  # Louvain x20, greedy, century seating, both nulls, layout, figure -> data/week4_symposium.js (a few minutes)
python scripts/fetch_week4_images.py                     # portrait URLs from Wikipedia -> data/week4_images.js
python scripts/test_symposium_rules.py                   # runs SYMPOSIUM's shipped core.js under node and diffs it against networkx

python scripts/analyse_week5.py         # every figure and number in the week 5 post (3-4 minutes; the first run tokenizes and caches)
python scripts/build_week5_data.py      # bags of words, generated sentences, blanks -> data/week5_secret.js
python scripts/test_secret_identity.py  # runs SECRET IDENTITY's shipped core.js under node and diffs it against Python

python scripts/analyse_week6.py         # every figure and number in the week 6 post (a few seconds)
python scripts/build_week6_dial.py      # the three matrices behind the dial -> data/week6_dial.js
python scripts/test_name_dial.py        # recomputes the data from scratch, runs THE NAME DIAL's shipped core.js and dial.js
python scripts/build_week6_taboo.py     # postings, kinds and oracles for the game -> data/week6_taboo.js
python scripts/test_taboo.py            # recomputes every posting, runs TABOO-IDF's shipped core.js and game.js
```

Week 5 additionally needs spaCy with `en_core_web_sm`, NLTK with its stopword list, and scikit-learn.

Week 6 needs no spaCy or NLTK (the stopword list is inlined in `week5_text.py`), only `numpy`, `networkx` and `matplotlib`. Louvain is
networkx's, and its partition depends on the version: networkx 3.7 gives eight communities of the 277-node giant component where the
week 5 run had seven, so community numbers in the week 6 post are from the partition it recomputes.

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
