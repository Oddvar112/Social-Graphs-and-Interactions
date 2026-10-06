"""
Shared machinery for week 6: the Marvel pages as TF-IDF vectors under the course's
definitions, a stated rule for which words are names, and the exact split of every
cosine similarity into a names part and a rest part.

  load_corpus()            pages, ids, per-page token lists, and capitalisation counts
  name_words(corpus, G)    the set of words we call names, with the reason for each
  tfidf_matrix(corpus)     the page's own definitions: tf = count / length, idf = ln(N / df)
  cosine_matrix(X)         all-pairs cosine, diagonal set to -1
  split_pair(...)          cosine = names part + rest part, term by term
  decompose(...)           cosine = names + habit words + other words, and the verdict that follows
  distances(G)             undirected shortest paths inside the giant component

The decomposition is the whole method. With u = a / |a| and v = b / |b|, the cosine is
the sum over words of u_w * v_w, and each word is either a name or not, so

    cos(a, b) = sum over names of u_w v_w  +  sum over other words of u_w v_w

and the name share is the first sum divided by the cosine. No model, and the parts add up
to the number the lookalikes explorable already shows. The rest is then split once more,
into habit words and every other word. Habit words are the NLTK stopwords (she, her, the) and
every word found on at least a third of the 303 pages (voiced, playable, series, appears, issue):
the function words of English and the function words of Wikipedia. After the names are gone it
is these that hold a surprising number of pages together:

    NAME   if the names part is at least half of the cosine
    HABIT  otherwise, if the habit part is at least as big as the other words part
    STORY  otherwise: the match is made of words that are neither names nor habits

Everything is deterministic given the frozen data.
"""

import collections
import pathlib
import re
import sys

import numpy as np
import networkx as nx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import week5_text as W                                   # noqa: E402
from analyse import load_graph, pretty                   # noqa: E402,F401

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Characters that may precede a word without making it "mid-sentence".
_OPENERS = " \t\"'(“”‘’["
_STARTS = ".!?\n:"


class Corpus:
    pass


def load_corpus():
    """Letters-only lowercased tokens per page (week 6 page, section 3), plus how often each
    word is capitalised in the middle of a sentence, which is how we tell names from words."""
    c = Corpus()
    c.pages = W.load_pages()
    c.ids = sorted(c.pages)
    c.N = len(c.ids)
    c.tokens = {}
    c.seen = collections.Counter()
    c.cap = collections.Counter()
    c.mid_seen = collections.Counter()
    c.mid_cap = collections.Counter()
    for n in c.ids:
        text, out = c.pages[n], []
        for m in re.finditer(r"[A-Za-z]+", text):
            w = m.group(0)
            lw = w.lower()
            j = m.start() - 1
            while j >= 0 and text[j] in _OPENERS:
                j -= 1
            at_start = j < 0 or text[j] in _STARTS
            c.seen[lw] += 1
            c.cap[lw] += w[0].isupper()
            if not at_start:
                c.mid_seen[lw] += 1
                c.mid_cap[lw] += w[0].isupper()
            out.append(lw)
        c.tokens[n] = out
    c.vocab = sorted(c.seen)
    c.vi = {w: i for i, w in enumerate(c.vocab)}
    return c


def title_words(G):
    """Every word in the title of any of the 303 pages, with the parenthesised disambiguator
    ('(character)', '(Marvel Comics)', '(Johnny Blaze)') left off: Wikipedia adds those, the
    character does not carry them."""
    words = set()
    for n in G:
        words |= set(re.findall(r"[a-z]+", W.base_title(n).lower()))
    return words


def name_words(c, G=None):
    """Words we call names, as a dict word -> reason.

    capitalised   written with a capital in more than half of its mid-sentence uses (the course
                  page's rule, with sentence starts left out of the count: 'During', 'However' and
                  'Fictional' open sentences and are not names)
    title         appears in the title of one of the 303 pages and is capitalised in at least a fifth of
                  its mid-sentence uses (or is too rare to tell), so it is somebody's name even when it
                  is also a common word (Brain Drain, Ruby Summers, Storm). 'Time', 'two' and 'father'
                  are in titles too, but nobody capitalises them on sight, so they stay words.

    Stopwords are never names, whatever their capitalisation.
    """
    G = G if G is not None else load_graph()
    reasons = {}
    for w in c.vocab:
        if w in W.STOPWORDS:
            continue
        if c.mid_seen[w] and c.mid_cap[w] > c.mid_seen[w] / 2:
            reasons[w] = "capitalised"
    for w in title_words(G):
        if w in c.vi and w not in W.STOPWORDS:
            ms = c.mid_seen[w]
            if ms < 20 or c.mid_cap[w] >= ms / 5:
                reasons[w] = "both" if w in reasons else "title"
    return reasons


def course_rule_names(c):
    """The rule exactly as the course page states it (capitalised in more than half of all uses),
    for the comparison in the post."""
    return {w for w in c.vocab if c.cap[w] > c.seen[w] / 2}


def count_matrix(c):
    X = np.zeros((c.N, len(c.vocab)))
    for i, n in enumerate(c.ids):
        for w, k in collections.Counter(c.tokens[n]).items():
            X[i, c.vi[w]] = k
    return X


def tfidf_from_counts(C):
    """The course page's definitions: tf = count / |d|, idf = ln(N / df). Natural log."""
    N = C.shape[0]
    tf = C / C.sum(axis=1, keepdims=True)
    df = (C > 0).sum(axis=0)
    idf = np.log(N / df)
    return tf * idf, idf, df


def unit(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


def cosine_matrix(X):
    U = unit(X)
    S = U @ U.T
    np.fill_diagonal(S, -1)
    return S


def split_pair(U, a, b, is_name):
    """(cosine, name part, rest part, contribution vector) for pages a and b; U has unit rows."""
    contrib = U[a] * U[b]
    name_part = float(contrib[is_name].sum())
    rest = float(contrib[~is_name].sum())
    return name_part + rest, name_part, rest, contrib


def distances(G):
    """Shortest-path lengths on the undirected giant component, as dict of dicts keyed by node id."""
    U = G.to_undirected()
    giant = U.subgraph(max(nx.connected_components(U), key=len))
    return dict(nx.all_pairs_shortest_path_length(giant)), U


def topk(S, k):
    """Indices of the k most similar pages for every row of an all-pairs similarity matrix."""
    return np.argsort(-S, axis=1)[:, :k]


def classify(name, habit, other):
    """NAME / HABIT / STORY from the three parts of a cosine (see the module docstring)."""
    total = name + habit + other
    if total <= 0:
        return "story"
    if name >= 0.5 * total:
        return "name"
    return "habit" if habit >= other else "story"


def decompose(U, a, b, is_name, is_habit):
    """Split cos(a, b) into names, habit words and other words; U has unit rows.
    Returns a dict with the cosine, the three parts, the name share and the verdict."""
    contrib = U[a] * U[b]
    name = float(contrib[is_name].sum())
    habit = float(contrib[~is_name & is_habit].sum())
    other = float(contrib[~is_name & ~is_habit].sum())
    cos = name + habit + other
    return {"cos": cos, "name": name, "habit": habit, "other": other,
            "share": name / cos if cos > 0 else 0.0,
            "klass": classify(name, habit, other), "contrib": contrib}


GENERIC_TITLE = {"character", "characters", "comics", "comic", "marvel", "film", "films", "series", "universe",
                 "earth", "ultimate", "version", "mcu", "team", "fictional"}


def own_words(nid):
    """Words that are part of how a character is called: every word in its Wikipedia title,
    parenthesised real name included, minus the generic ones Wikipedia adds."""
    return set(re.findall(r"[a-z]+", nid.replace("_", " ").lower())) - GENERIC_TITLE


def name_kind(term, a, b, ids, base_words):
    """Why a shared name ties two pages: it is part of one of the pages' own names ('own'), it is
    somebody else's name on the roster, such as Cyclops on a page about Jean Grey ('roster'), or it is
    a team, a publisher, a creator, a place ('other')."""
    if term in own_words(ids[a]) or term in own_words(ids[b]):
        return "own"
    if term in base_words:
        return "roster"
    return "other"


class Model:
    pass


def build_model():
    """Everything both the analysis and the dial's data builder need, built once and the same way:
    the corpus, the name rule, TF-IDF with unit rows, the similarity matrices (names kept; names and
    stopwords out; names and habit words out), network distances and the Louvain partition of the giant component."""
    m = Model()
    m.c = c = load_corpus()
    m.G = load_graph()
    m.ids, m.N = c.ids, c.N
    m.ix = {n: i for i, n in enumerate(c.ids)}
    m.names = name_words(c, m.G)
    m.isname = np.array([w in m.names for w in c.vocab])
    m.isstop = np.array([w in W.STOPWORDS for w in c.vocab])
    m.C = count_matrix(c)
    m.X, m.idf, m.df = tfidf_from_counts(m.C)
    m.ishabit = m.isstop | ((m.df >= -(-m.N // 3)) & ~m.isname)       # stopwords, and words on a third of the pages
    m.Un = unit(m.X)
    m.length = m.C.sum(axis=1)
    m.dist, m.U = distances(m.G)
    giant = m.U.subgraph(max(nx.connected_components(m.U), key=len))
    m.Q, m.comms, _ = W.louvain_best(giant)
    m.comm_of = {n: k for k, cm in enumerate(m.comms) for n in cm}
    m.base_words = title_words(m.G)
    m.S_kept = cosine_matrix(m.X)
    m.S_nostop = cosine_matrix(m.X * ~m.isname * ~m.isstop)         # names and stopwords out
    m.S_clean = cosine_matrix(m.X * ~m.isname * ~m.ishabit)         # names and habit words out
    return m


def pair_set(top):
    """Unordered pairs (i < j) that appear in anyone's top-k list."""
    return sorted({(int(min(i, j)), int(max(i, j))) for i in range(top.shape[0]) for j in top[i]})
