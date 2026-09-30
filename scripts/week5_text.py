"""
Shared text machinery for week 5: the Marvel pages, tokenized once and cached,
plus the pieces both the post's analysis and the game's data builder need.

  load_pages()            node_id -> full plain-text Wikipedia article (marvel_pages.zip)
  load_tokens()           node_id -> {"tokens": [...], "sents": [(start, end), ...]}, spaCy
                          tokens with lowercase form, lemma, is_stop/is_punct/is_alpha,
                          cached in .cache/ (gitignored) because the pipeline takes a minute
  words(...)              the alphabetic, lowercased token stream of a page, with or
                          without stopwords (NLTK's English list, like the course explorable)
  clean_sentences(...)    prose sentences as surface-token lists, for n-gram work
  detok(...)              tokens back to a sentence (the same rule for real and generated text)
  TrigramModel            a trigram generator with bigram back-off and a no-copy rule
  character_names(...)    the strings that mean a character: title, unique real name
  RelationLabeller        for every link A -> B, the sentences on A's page that mention B,
                          dependency-parsed, and a foe / ally / family verdict
  louvain_best(...)       communities of the undirected giant component, best of 20 seeds

Everything is deterministic given the frozen data.
"""

import collections
import pathlib
import pickle
import re
import urllib.parse
import zipfile

import networkx as nx

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
ZIP = ROOT / "marvel_pages.zip"

# NLTK's English stopword list, inlined so the module has one dependency fewer at
# import time; nltk.corpus.stopwords.words("english") is the same 198 words.
STOPWORDS = set("""i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself yourselves he him
his himself she she's her hers herself it it's its itself they them their theirs themselves what which who whom this that
that'll these those am is are was were be been being have has had having do does did doing a an the and but if or because
as until while of at by for with about against between into through during before after above below to from up down in out
on off over under again further then once here there when where why how all any both each few more most other some such no
nor not only own same so than too very s t can will just don don't should should've now d ll m o re ve y ain aren aren't
couldn couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't
needn needn't shan shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't""".split())


# ----------------------------------------------------------------------------- loading
def load_pages():
    with zipfile.ZipFile(ZIP) as z:
        return {urllib.parse.unquote(n.split("/")[-1][:-4]): z.read(n).decode("utf-8")
                for n in z.namelist() if n.endswith(".txt") and "README" not in n}


def load_tokens(pages=None, verbose=True):
    """spaCy tokens per page (tagger + lemmatizer + sentencizer, no parser), cached."""
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / "week5_tokens.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    import spacy
    pages = pages or load_pages()
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.add_pipe("sentencizer")
    nlp.max_length = 200_000
    ids = sorted(pages)
    docs = {}
    for i, (nid, doc) in enumerate(zip(ids, nlp.pipe((pages[n] for n in ids), batch_size=8))):
        docs[nid] = {
            "tokens": [(t.text, t.lower_, t.lemma_.lower(), bool(t.is_stop), bool(t.is_punct),
                        bool(t.is_alpha), bool(t.is_space), bool(t.is_sent_start), t.idx) for t in doc],
            "sents": [(s.start_char, s.end_char) for s in doc.sents],
        }
        if verbose and i % 100 == 0:
            print(f"   tokenizing {i}/{len(ids)}", flush=True)
    pickle.dump(docs, open(cache, "wb"))
    return docs


# Token tuple fields, for readability.
TEXT, LOWER, LEMMA, IS_STOP, IS_PUNCT, IS_ALPHA, IS_SPACE, SENT_START, IDX = range(9)


def words(doc, stop=False, lemma=False):
    """Alphabetic, lowercased tokens of one tokenized page; stop=True drops NLTK stopwords."""
    out = []
    for t in doc["tokens"]:
        if not t[IS_ALPHA]:
            continue
        w = t[LEMMA] if lemma else t[LOWER]
        if stop and t[LOWER] in STOPWORDS:
            continue
        out.append(w)
    return out


def raw_tokens(doc):
    """Every non-whitespace spaCy token, surface form: what 'tokens' means before any choice."""
    return [t[TEXT] for t in doc["tokens"] if not t[IS_SPACE]]


# ----------------------------------------------------------------------------- sentences
BAD_TOKENS = {"#", "vol.", "Vol.", "ISBN", "pp.", "Retrieved", "Archived", "archived", "http", "https", "www"}


def is_clean(s, lo=8, hi=30):
    """A prose sentence the game can show: complete, mostly words, no citation debris."""
    n = len(s)
    if not (lo <= n <= hi) or s[-1] not in ".!?" or not s[0][0].isupper():
        return False
    if sum(1 for w in s if w.isalpha()) / n < 0.72:
        return False
    if s.count("(") != s.count(")") or s.count('"') % 2 or s.count("[") != s.count("]"):
        return False
    if any(w in BAD_TOKENS for w in s):
        return False
    if "\n" in " ".join(s):
        return False
    return True


def clean_sentences(doc, lo=8, hi=30):
    """Sentences of a page as surface-token lists, filtered by is_clean. The first
    sentence (the definition line every page shares) is dropped."""
    out, cur = [], []
    for t in doc["tokens"]:
        if t[SENT_START] and cur:
            out.append(cur)
            cur = []
        if not t[IS_SPACE]:
            cur.append(t[TEXT])
    if cur:
        out.append(cur)
    return [s for s in out[1:] if is_clean(s, lo, hi)]


_CLOSE = re.compile(r"^(n't|'s|'re|'m|'ve|'ll|'d|'|[,.;:!?%)\]])$")


def detok(toks):
    """Tokens back to text. One rule for real and generated sentences, so the game
    cannot be won by spotting a formatting difference between the two."""
    out, open_q = "", False
    for i, t in enumerate(toks):
        prev = toks[i - 1] if i else ""
        if not out:
            out = t
            continue
        if t == '"':
            out += (t if open_q else " " + t)
            open_q = not open_q
            continue
        if _CLOSE.match(t) or prev in ("(", "[", "$") or (prev == '"' and open_q) or t == "-" or prev == "-":
            out += t
        else:
            out += " " + t
    return out


class TrigramModel:
    """P(w_t | w_{t-2}, w_{t-1}) from counts, bigram back-off when a context is new.
    generate() refuses any sentence that shares a 9-token run with the training
    corpus, so nothing it returns is a copied sentence wearing a disguise."""

    def __init__(self, sentences, ngram_guard=9):
        self.tri = collections.defaultdict(collections.Counter)
        self.bi = collections.defaultdict(collections.Counter)
        self.guard = ngram_guard
        self.seen = set()
        self.real = set()
        for s in sentences:
            w = ["<s>", "<s>"] + list(s) + ["</s>"]
            for i in range(2, len(w)):
                self.tri[(w[i - 2], w[i - 1])][w[i]] += 1
                self.bi[w[i - 1]][w[i]] += 1
            self.real.add(tuple(s))
            for i in range(len(s) - ngram_guard + 1):
                self.seen.add(tuple(s[i:i + ngram_guard]))

    def add_guard(self, sentences):
        """Also refuse runs that appear in these sentences (the whole corpus, say)."""
        for s in sentences:
            self.real.add(tuple(s))
            for i in range(len(s) - self.guard + 1):
                self.seen.add(tuple(s[i:i + self.guard]))

    def sample(self, rng, maxlen=40):
        w = ["<s>", "<s>"]
        while len(w) < maxlen + 2:
            c = self.tri.get((w[-2], w[-1])) or self.bi.get(w[-1])
            if not c:
                break
            items = list(c.items())
            nxt = rng.choices([k for k, _ in items], [v for _, v in items])[0]
            if nxt == "</s>":
                break
            w.append(nxt)
        return w[2:]

    def copies(self, s):
        g = self.guard
        return tuple(s) in self.real or any(tuple(s[i:i + g]) in self.seen for i in range(len(s) - g + 1))

    def generate(self, rng, n, lo=8, hi=30, max_tries=5000):
        out, tries = [], 0
        while len(out) < n and tries < max_tries:
            tries += 1
            s = self.sample(rng)
            if is_clean(s, lo, hi) and not self.copies(s):
                out.append(s)
        return out


# ----------------------------------------------------------------------------- names
def base_title(nid):
    return re.sub(r"\s*\([^)]*\)\s*$", "", nid.replace("_", " ")).strip()


def alias_from_description(desc):
    """'Abomination (Emil Blonsky) is a character ...' -> 'Emil Blonsky'."""
    m = re.match(r"^(.*?)\s*\(([^)]+)\)", desc[:140])
    if not m:
        return None
    alias = re.sub(r"\s*,.*$", "", m.group(2)).strip()
    if 2 <= len(alias.split()) <= 3 and alias[0].isupper() and \
            alias.replace(" ", "").replace(".", "").replace("'", "").replace("-", "").isalpha():
        return alias
    return None


def character_names(G):
    """node_id -> set of strings that name that character: the title without its
    disambiguator, plus the real name from the description when no other character
    shares it (Peter Parker names five Spider-people, so it names none of them here)."""
    titles = {n: base_title(n) for n in G}
    aliases = {n: alias_from_description(G.nodes[n].get("description", "")) for n in G}
    shared = collections.Counter(a for a in aliases.values() if a)
    shared_titles = collections.Counter(titles.values())
    out = {}
    for n in G:
        names = set()
        if shared_titles[titles[n]] == 1:            # 'Captain Marvel' names two pages: usable for neither
            names.add(titles[n])
        if aliases[n] and shared[aliases[n]] == 1:
            names.add(aliases[n])
        out[n] = {x for x in names if len(x) >= 4}
    return out


def mask_tokens(G, nid, doc):
    """Lowercase tokens that would give the character away as a clue: the title, the
    real name, and the subject of the page's first sentence (everything before the
    first 'is' / 'was' / 'are', where Wikipedia lists a character's other names)."""
    m = set(re.findall(r"[a-z]+", nid.replace("_", " ").lower()))
    a = alias_from_description(G.nodes[nid].get("description", ""))
    if a:
        m |= set(re.findall(r"[a-z]+", a.lower()))
    first = doc["tokens"]
    end = doc["sents"][0][1] if doc["sents"] else 0
    subj = []
    for t in first:
        if t[IDX] >= end or t[LOWER] in ("is", "was", "are", "were"):
            break
        if t[IS_ALPHA]:
            subj.append(t[LOWER])
    m |= set(subj[:14])
    return {w for w in m if len(w) >= 2}


# ----------------------------------------------------------------------------- relations
FOE_N = set("enemy nemesis foe rival archenemy arch-enemy adversary antagonist villain opponent".split())
FOE_V = set("fight battle defeat kill attack confront oppose betray".split())
ALLY_N = set("ally friend friendship teammate partner sidekick mentor protégé apprentice pupil comrade".split())
FAM_N = set("wife husband spouse son daughter father mother brother sister sibling twin cousin uncle aunt nephew niece "
            "girlfriend boyfriend lover fiancé fiancée grandfather grandmother granddaughter grandson grandchild parent "
            "child widow ex-wife ex-husband".split())
FAM_V = set("marry date divorce wed".split())
PRON = {"he", "she", "they", "who", "it", "his", "her", "their", "him"}


class RelationLabeller:
    """For a link A -> B: find B's name on A's page (not in the definition line),
    dependency-parse those sentences, and count relationship words that are
    syntactically attached to the mention. Bag-of-words versions of this (any
    relationship word within ten tokens) have low precision because in comics
    everybody fights together: 'attack' near a name usually means side by side.

    Accepted attachments, with the mention M and the relationship word R:
      R of/to/with M            an enemy of Iron Fist, the daughter of Ulysses
      M appos R / R appos M     his wife, Jean Grey; a prominent new enemy, Venom
      compound                  teammate Tarantula, the vampire Morbius
      copula                    His main opponent was Misty Knight
      alongside M               fought alongside Nova
      verb + object             she fought Morbius / marries Jessica Jones, when the
                                verb's subject is the page's own character or a pronoun
    Rejected: M is the possessor of R (Nova's old enemy Sphinx) or R belongs to a
    third party (Franklin Richards, son of Mister Fantastic; Cyclops' girlfriend Jean).
    """

    def __init__(self, G, pages, docs, names=None):
        import spacy
        self.G, self.pages, self.docs = G, pages, docs
        self.names = names or character_names(G)
        self.nlp = spacy.load("en_core_web_sm")
        self._parsed = {}

    def sentences_with(self, u, v):
        own = {n.lower() for n in self.names[u]}
        pats = [n for n in self.names[v] if not any(n.lower() in o or o in n.lower() for o in own)]
        if not pats:
            return None, []
        # (?<![\w-]) ... (?![\w-]): 'Hulk' must not match inside 'She-Hulk' or 'Hulkling'
        rx = re.compile(r"(?<![\w-])(" + "|".join(re.escape(n) for n in sorted(pats, key=len, reverse=True)) + r")(?![\w-])")
        out = []
        for a, b in self.docs[u]["sents"][1:]:
            s = self.pages[u][a:b].strip()
            if rx.search(s) and len(s) < 600:
                out.append(s)
        return rx, out

    def parse(self, s):
        d = self._parsed.get(s)
        if d is None:
            d = self._parsed[s] = self.nlp(s)
        return d

    @staticmethod
    def _span_head(span):
        for t in span:
            if t.head not in span or t.head == t:
                return t
        return span[-1]

    @staticmethod
    def _subjects(V, depth=0):
        """The subject of a verb, following conjunctions and infinitive complements:
        'the Sinister Six reform and defeat the Hulk', 'Hydra plotted to kill Iron Fist'."""
        subs = [c for c in V.children if c.dep_ in ("nsubj", "nsubjpass")]
        if subs or depth > 2:
            return subs
        if V.dep_ in ("conj", "xcomp", "advcl", "ccomp", "pcomp", "acl") and V.head is not V:
            return RelationLabeller._subjects(V.head, depth + 1)
        if V.dep_ == "prep" or V.pos_ == "VERB" and V.head.pos_ in ("VERB", "AUX") and V.dep_ != "ROOT":
            return RelationLabeller._subjects(V.head, depth + 1)
        return subs

    def _third_party(self, R, own):
        """R's relation belongs to somebody else: 'X's enemy', 'son of X'."""
        for c in R.children:
            if c.dep_ in ("poss", "nmod") and c.text.lower() not in own and c.text.lower() not in PRON and \
                    (c.dep_ == "poss" or any(g.text in ("'s", "'") for g in c.children)):
                return True
            if c.dep_ == "prep" and c.text.lower() in ("of", "to"):
                for g in c.children:
                    if g.dep_ == "pobj" and g.text.lower() not in own and g.text.lower() not in PRON:
                        return True
        return False

    def relations(self, doc, rx, own):
        found = []
        for m in rx.finditer(doc.text):
            span = doc.char_span(m.start(), m.end(), alignment_mode="expand")
            if span is None:
                continue
            M, Mset = self._span_head(span), set(span)
            for R in doc:
                if R in Mset or R.text.lower() in own:
                    continue
                lem, txt = R.lemma_.lower(), R.text.lower()
                kind = None
                if R.pos_ in ("NOUN", "PROPN"):
                    kind = "foe" if lem in FOE_N else "ally" if lem in ALLY_N else "family" if lem in FAM_N else None
                elif txt == "alongside" and R.pos_ == "ADP":
                    kind = "ally"
                elif R.pos_ == "VERB" and lem in FOE_V:
                    kind = "foe_v"
                elif R.pos_ == "VERB" and lem in FAM_V:
                    kind = "family_v"
                if not kind:
                    continue
                ok, why = False, ""
                if kind.endswith("_v"):
                    preps = ("with", "against") if kind == "foe_v" else ("with", "to")
                    objs = [c for c in R.children if c.dep_ in ("dobj", "obj")]
                    objs += [g for c in R.children if c.dep_ == "prep" and c.text.lower() in preps
                             for g in c.children if g.dep_ == "pobj"]
                    subs = self._subjects(R)
                    if any(o == M or M in o.subtree for o in objs) and \
                            (not subs or any(s.text.lower() in PRON or s.text.lower() in own for s in subs)):
                        ok, why, kind = True, "verb+object", kind[:-2]
                elif M.dep_ == "poss" and M.head == R:
                    ok, why = False, "possessive"
                elif txt == "alongside":
                    ok = any(g == M or M in g.subtree for g in R.children if g.dep_ == "pobj")
                    why = "alongside"
                else:
                    of_M = any(c.dep_ == "prep" and c.text.lower() in ("of", "to", "with") and
                               any(g == M or (M in g.subtree and g.dep_ == "pobj") for g in c.children)
                               for c in R.children)
                    appos = (M.dep_ == "appos" and M.head == R) or (R.dep_ == "appos" and R.head == M)
                    compound = (M.head == R and M.dep_ in ("compound", "nmod", "flat", "npadvmod")) or \
                               (R.head == M and R.dep_ in ("compound", "nmod"))
                    copula = R.head.lemma_ == "be" and M.head == R.head and \
                        {R.dep_, M.dep_} <= {"attr", "nsubj"} and R.dep_ != M.dep_
                    if of_M:
                        # 'a partner of Wolverine's' hanging off a third party (Yukio, a partner of ...)
                        ok, why = not (R.dep_ == "appos" and R.head not in Mset and
                                       R.head.text.lower() not in own and R.head.text.lower() not in PRON), "R of M"
                    elif appos or compound or copula:
                        ok, why = not (self._third_party(R, own) or self._third_party(M, own)), "appos/compound/copula"
                if ok:
                    found.append((kind, R.text, why))
        return found

    def label(self, u, v):
        """-> (label, evidence) with label in foe / ally / family / mixed / unlabelled / none / ambiguous."""
        rx, hits = self.sentences_with(u, v)
        if rx is None:
            return "ambiguous", []
        if not hits:
            return "none", []
        own = {w for n in self.names[u] | self.names[v] for w in re.findall(r"[a-z]+", n.lower())} | \
              {w for w in re.findall(r"[a-z]+", u.replace("_", " ").lower())}
        counts, evidence = collections.Counter(), []
        for s in hits:
            rel = self.relations(self.parse(s), rx, own)
            for kind, w, why in rel:
                counts[kind] += 1
            if rel:
                evidence.append((s, rel))
        if not counts:
            return "unlabelled", []
        top = counts.most_common()
        if len(top) > 1 and top[1][1] == top[0][1]:
            return "mixed", evidence
        return top[0][0], evidence

    def label_all(self, verbose=True):
        out = {}
        for i, (u, v) in enumerate(self.G.edges()):
            out[(u, v)] = self.label(u, v)
            if verbose and i % 400 == 0:
                print(f"   labelling {i}/{self.G.number_of_edges()}", flush=True)
        return out


# ----------------------------------------------------------------------------- communities
def louvain_best(giant, seeds=range(20)):
    """Louvain on the undirected giant component, best modularity over 20 seeds,
    the way week 4 picked its partition. Returns (Q, communities sorted by size, seed)."""
    best = None
    for seed in seeds:
        comms = nx.community.louvain_communities(giant, seed=seed)
        q = nx.community.modularity(giant, comms)
        if best is None or q > best[0]:
            best = (q, comms, seed)
    q, comms, seed = best
    return q, sorted((sorted(c) for c in comms), key=len, reverse=True), seed
