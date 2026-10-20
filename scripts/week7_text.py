"""
Shared machinery for week 7: the two search engines that IN OTHER WORDS pits against you.

  load_clues()               the hand-written clues in data/week7_clues.tsv
  passages(text, words)      sentences packed greedily into passages of at most `words` words (the course's rule)
  Encoder                    all-MiniLM-L6-v2 (mean pooling, unit length), the same vectors sentence-transformers gives,
                             computed with plain `transformers` and cached in .cache/
  keyword_scores(m, query)   sparse retrieval: cosine between the query's TF-IDF vector and every page's, with the
                             week 6 definitions (tf = count / words, idf = ln(303 / df), unit length)
  dense_scores(E, q)         dense retrieval: dot product of unit vectors, page-mean and best-passage
  rank_of(scores, target)    1-based rank of the target page, ties broken against it (the worse rank)

Sparse: a query is a bag of words, weighted like a page (tf = count / query length, idf as the corpus), unit length.
Its cosine with page p is then sum over the query's words of q_w * U[p, w]. A word on no page has idf undefined and
is dropped, which is exactly the vocabulary mismatch the week is about: the engine cannot see a word it has never read.

Dense: the encoder reads at most 256 word pieces and 302 of 303 pages are longer, so pages are split into passages
of at most 180 words at sentence boundaries (4,473 passages, the course's number), each passage is encoded, and a page
is scored either by the mean of its passage vectors (renormalised) or by its best single passage. The query is
encoded the same way. Both scores are dot products of unit vectors, so ranking by them is ranking by cosine.
"""

import csv
import hashlib
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import week5_text as W                                   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / ".cache"
CLUES = DATA / "week7_clues.tsv"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PASSAGE_WORDS = 180
MAX_PIECES = 256


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


# ----------------------------------------------------------------------------- the encoder
class Encoder:
    """all-MiniLM-L6-v2 as sentence-transformers defines it: mean pooling over the attention mask, unit length,
    input truncated to 256 word pieces. Plain `transformers` so there is one dependency fewer."""

    def __init__(self, model=MODEL, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.name = model
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode(self, texts, batch=64):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=MAX_PIECES, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                v = torch.nn.functional.normalize(v, p=2, dim=1)
                out.append(v.cpu().numpy())
        return np.vstack(out).astype(np.float32)

    def pieces(self, text):
        """How many word pieces the tokenizer makes of a text (without truncation)."""
        return len(self.tok(text, truncation=False)["input_ids"])


_ENC = None


def encoder():
    global _ENC
    if _ENC is None:
        _ENC = Encoder()
    return _ENC


def encode_cached(texts, tag):
    """Encode a list of texts, cached under .cache/ by a hash of the texts and the model name."""
    CACHE.mkdir(exist_ok=True)
    h = hashlib.sha1((MODEL + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
    path = CACHE / f"week7_{tag}_{h}.npy"
    if path.exists():
        return np.load(path)
    E = encoder().encode(texts)
    np.save(path, E)
    return E


class Dense:
    """Every page's passages and their vectors, at one passage length. page_mean[p] and best-passage scoring."""

    def __init__(self, pages, ids, words=PASSAGE_WORDS):
        self.ids, self.words = ids, words
        self.texts, self.owner = [], []
        for i, n in enumerate(ids):
            ps = passages(pages[n], words)
            self.texts.extend(ps)
            self.owner.extend([i] * len(ps))
        self.owner = np.array(self.owner)
        self.E = encode_cached(self.texts, f"p{words}")
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
