"""
Shared machinery for week 7: the search engines that IN OTHER WORDS pits against you.

  load_clues()               the hand-written clues in data/week7_clues.tsv
  passages(text, words)      sentences packed greedily into passages of at most `words` words (the course's rule)
  SPECS                      the sentence encoders we compare, each used the way its model card says: pooling,
                             query and passage prefixes, input length
  Encoder(model)             one of them, run with plain `transformers` (safetensors weights only, no remote code)
  Dense(pages, ids, words, model)   every page's passages and their vectors (cached in .cache/), scored by the best
                             passage or by the mean of a page's passages
  keyword_scores(m, query)   sparse retrieval: cosine between the query's TF-IDF vector and every page's, with the
                             week 6 definitions (tf = count / words, idf = ln(303 / df), unit length)
  standout(scores)           how far an engine's number one stands out from all 303 pages
  fit_nobody(...)            the NOBODY HERE rule: a threshold on a standout measure, fitted to the game's scoring
  rank_of(scores, target)    1-based rank of the target page, ties broken against it (the worse rank)

MODEL is the course's encoder, all-MiniLM-L6-v2: the post's analysis is about it. GAME_MODEL is the stronger encoder
the game plays as MEANING, mxbai-embed-large-v1, chosen by scripts/compare_week7_encoders.py.

Sparse: a query is a bag of words, weighted like a page (tf = count / query length, idf as the corpus), unit length.
Its cosine with page p is then sum over the query's words of q_w * U[p, w]. A word on no page has idf undefined and
is dropped, which is exactly the vocabulary mismatch the week is about: the engine cannot see a word it has never read.

Dense: MiniLM reads at most 256 word pieces and 302 of 303 pages are longer, so pages are split into passages of at
most 180 words at sentence boundaries (4,479 passages; the course page has 4,473), each passage is encoded, and a page
is scored either by the mean of its passage vectors (renormalised) or by its best single passage. The query is encoded
with the model's query prefix. Both scores are dot products of unit vectors, so ranking by them is ranking by cosine.

Nobody here: a search engine always has a number one. To let it say "none of these", we look at how far that number
one stands out from the 303 pages, z = (best - mean) / standard deviation (or gap = (best - runner-up) / standard
deviation), and say nobody when it is below a threshold. The threshold is the one that scores best in the game:
an episode of 4 clues in the page's own words, 5 in other words and 1 about a missing character.
"""

import csv
import hashlib
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import week5_text as W                                   # noqa: E402,F401

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / ".cache"
CLUES = DATA / "week7_clues.tsv"
PASSAGE_WORDS = 180
MAX_PIECES = 256

BGE_QUERY = "Represent this sentence for searching relevant passages: "
SPECS = {
    "sentence-transformers/all-MiniLM-L6-v2": dict(short="MiniLM", pooling="mean", query="", passage="", max=256, batch=64),
    "BAAI/bge-small-en-v1.5": dict(short="bge-small", pooling="cls", query=BGE_QUERY, passage="", max=512, batch=64),
    "sentence-transformers/all-mpnet-base-v2": dict(short="all-mpnet-base", pooling="mean", query="", passage="", max=384, batch=32),
    "BAAI/bge-base-en-v1.5": dict(short="bge-base", pooling="cls", query=BGE_QUERY, passage="", max=512, batch=32),
    "intfloat/e5-base-v2": dict(short="e5-base", pooling="mean", query="query: ", passage="passage: ", max=512, batch=32),
    "mixedbread-ai/mxbai-embed-large-v1": dict(short="mxbai-embed-large", pooling="cls", query=BGE_QUERY, passage="", max=512, batch=16),
}
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GAME_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
EPISODE_MIX = {"page": 4, "other": 5, "absent": 1}


# ----------------------------------------------------------------------------- clues
def load_clues():
    """The clues, in file order: dicts with id, style (page / other / absent), target, clue."""
    out = []
    with open(CLUES, encoding="utf-8", newline="") as f:
        for row in csv.DictReader((ln for ln in f if not ln.startswith("#")), delimiter="\t"):
            if row.get("id") and row.get("clue"):
                out.append({k: v.strip() for k, v in row.items()})
    return out


# ----------------------------------------------------------------------------- passages
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def sentences(text):
    """Sentences: a split at sentence-ending punctuation followed by whitespace and a capital (or a quote)."""
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in _SENT.split(text) if s]


def passages(text, words=PASSAGE_WORDS):
    """Greedy packing of sentences into passages of at most `words` words; a lone sentence longer than that stays whole."""
    out, cur, n = [], [], 0
    for s in sentences(text):
        k = len(s.split())
        if cur and n + k > words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += k
    if cur:
        out.append(" ".join(cur))
    return out


# ----------------------------------------------------------------------------- the encoders
class Encoder:
    """A sentence encoder as its model card defines it: mean or CLS pooling, unit length, the query and passage
    prefixes it was trained with, and its input length. Plain `transformers`, safetensors weights only."""

    def __init__(self, model=MODEL, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.name = model
        self.spec = SPECS[model]
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, use_safetensors=True).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode(self, texts, query=False, batch=None):
        torch, sp = self.torch, self.spec
        prefix = sp["query"] if query else sp["passage"]
        texts = [prefix + t for t in texts]
        b = batch or sp["batch"]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), b):
                enc = self.tok(texts[i:i + b], padding=True, truncation=True, max_length=sp["max"], return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                if sp["pooling"] == "cls":
                    v = h[:, 0]
                else:
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(torch.nn.functional.normalize(v, p=2, dim=1).float().cpu().numpy())
        return np.vstack(out).astype(np.float32)

    def pieces(self, text):
        """How many word pieces the tokenizer makes of a text (without truncation)."""
        return len(self.tok(text, truncation=False)["input_ids"])


_ENC = {}


def encoder(model=MODEL):
    if model not in _ENC:
        _ENC[model] = Encoder(model)
    return _ENC[model]


def release(model):
    """Free a loaded encoder (the comparison loads six in turn)."""
    enc = _ENC.pop(model, None)
    if enc is not None:
        torch = enc.torch
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def encode_cached(texts, tag, model=MODEL):
    """Passage-side vectors for a list of texts, cached under .cache/ by a hash of the model name and the texts as
    the model reads them (with its passage prefix)."""
    CACHE.mkdir(exist_ok=True)
    pp = SPECS[model]["passage"]
    h = hashlib.sha1((model + "\n" + "\n".join(pp + t for t in texts)).encode("utf-8")).hexdigest()[:16]
    path = CACHE / f"week7_{tag}_{h}.npy"
    if path.exists():
        return np.load(path)
    E = encoder(model).encode(texts)
    np.save(path, E)
    return E


class Dense:
    """Every page's passages and their vectors, at one passage length and one encoder. page_mean[p] and best-passage scoring."""

    def __init__(self, pages, ids, words=PASSAGE_WORDS, model=MODEL):
        self.ids, self.words, self.model = ids, words, model
        self.texts, self.owner = [], []
        for i, n in enumerate(ids):
            ps = passages(pages[n], words)
            self.texts.extend(ps)
            self.owner.extend([i] * len(ps))
        self.owner = np.array(self.owner)
        self.E = encode_cached(self.texts, f"p{words}", model)
        N = len(ids)
        M = np.zeros((N, self.E.shape[1]), np.float32)
        np.add.at(M, self.owner, self.E)
        M /= np.linalg.norm(M, axis=1, keepdims=True)
        self.mean = M
        self.starts = np.searchsorted(self.owner, np.arange(N))
        self.ends = np.searchsorted(self.owner, np.arange(N), side="right")

    def scores(self, q):
        """(page-mean scores, best-passage scores, index of the best passage per page) for a unit query vector."""
        ps = self.E @ q
        N = len(self.ids)
        best = np.full(N, -1.0, np.float32)
        arg = np.zeros(N, int)
        np.maximum.at(best, self.owner, ps)
        for p in range(N):
            a, b = self.starts[p], self.ends[p]
            arg[p] = a + int(np.argmax(ps[a:b]))
        return self.mean @ q, best, arg


# ----------------------------------------------------------------------------- sparse retrieval
def query_tokens(text):
    return re.findall(r"[a-z]+", text.lower())


def keyword_scores(m, query):
    """Cosine between the query's TF-IDF vector and every page's unit TF-IDF vector (week6_text's model `m`).
    Returns (scores, the per-word contribution to each page as {word: array}, words the corpus has never seen)."""
    toks = query_tokens(query)
    known = [w for w in toks if w in m.c.vi and m.df[m.c.vi[w]] > 0]
    unknown = sorted({w for w in toks if w not in m.c.vi})
    if not known:
        return np.zeros(m.N), {}, unknown
    counts = {}
    for w in known:
        counts[w] = counts.get(w, 0) + 1
    q = {w: (k / len(toks)) * m.idf[m.c.vi[w]] for w, k in counts.items()}
    norm = np.sqrt(sum(v * v for v in q.values()))
    parts = {w: (v / norm) * m.Un[:, m.c.vi[w]] for w, v in q.items() if norm > 0}
    scores = sum(parts.values()) if parts else np.zeros(m.N)
    return scores, parts, unknown


def rank_of(scores, target):
    """1-based rank of `target`; a tie counts against it."""
    return int((scores >= scores[target]).sum())


def topk(scores, k):
    order = np.argsort(-scores, kind="stable")
    return [int(i) for i in order[:k]]


def distinctive_words(m, text, target):
    """Which of a clue's words are on the target page and are not habit words: the words a keyword search can use."""
    hits = []
    for w in dict.fromkeys(query_tokens(text)):
        j = m.c.vi.get(w)
        if j is None or m.ishabit[j]:
            continue
        if m.C[target, j] > 0:
            hits.append(w)
    return hits


# ----------------------------------------------------------------------------- nobody here
def standout(scores):
    """How far an engine's number one stands out from all the pages it ranked:
    z = (best - mean) / standard deviation, gap = (best - runner-up) / standard deviation, and the best score itself."""
    s = np.asarray(scores, dtype=np.float64)
    o = np.sort(s)[::-1]
    sd = float(s.std()) or 1.0
    return {"top": float(o[0]), "z": (float(o[0]) - float(s.mean())) / sd, "gap": (float(o[0]) - float(o[1])) / sd}


def episode_score(styles, right, nobody):
    """Expected points in a ten-clue episode (EPISODE_MIX) for one engine, from per-clue arrays: the clue's style,
    whether the engine's number one is the right page, and whether it says nobody. A present character scores only
    if the engine answers and is right; a missing one only if it says nobody."""
    styles, right, nobody = np.asarray(styles), np.asarray(right, bool), np.asarray(nobody, bool)
    total = 0.0
    for st, w in EPISODE_MIX.items():
        m = styles == st
        if m.any():
            ok = nobody[m] if st == "absent" else (~nobody[m] & right[m])
            total += w * ok.mean()
    return total


def fit_nobody(values, styles, right):
    """The threshold t ("say nobody when value < t") that maximises episode_score; on a tie the smallest t, the
    engine that abstains least. Returns (t, score)."""
    values = np.asarray(values, dtype=np.float64)
    styles, right = np.asarray(styles), np.asarray(right, bool)
    vals = np.unique(values)
    cands = np.concatenate([[vals[0] - 1], (vals[:-1] + vals[1:]) / 2, [vals[-1] + 1]])
    nobody = values[None, :] < cands[:, None]
    sc = np.zeros(len(cands))
    for st, w in EPISODE_MIX.items():
        m = styles == st
        if m.any():
            ok = nobody[:, m] if st == "absent" else (~nobody[:, m] & right[m][None, :])
            sc += w * ok.mean(axis=1)
    i = int(np.argmax(sc))
    return float(cands[i]), float(sc[i])


def loo_nobody(values, styles, right):
    """Leave one out: every clue's nobody decision from a threshold fitted without that clue."""
    values = np.asarray(values, dtype=np.float64)
    styles, right = np.asarray(styles), np.asarray(right, bool)
    n = len(values)
    out = np.zeros(n, bool)
    for i in range(n):
        keep = np.arange(n) != i
        t, _ = fit_nobody(values[keep], styles[keep], right[keep])
        out[i] = values[i] < t
    return out
