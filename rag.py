import re
import math
from collections import Counter

_STOPWORDS = set("""
a an and are as at be by for from has he in is it its of on
that the to was were will with this these those i you your we they
them their or if not but can could should would do does did have had
what when where which who whom why how about me my tell give show please there here also
""".split())


def _tokenize(text):
    return [t for t in re.findall(r"[a-zA-Z']+", text.lower())
            if t not in _STOPWORDS and len(t) > 1]


class DocumentIndex:
    """In-memory index of document chunks with TF-IDF cosine search."""

    def __init__(self):
        self.chunks = []       # list of dicts: doc, chunk_id, text, tf
        self._df = Counter()   # document frequency per term
        self._n_chunks = 0

    def _add_chunk(self, doc_name, text, title=None):
        tokens = _tokenize(text)
        if not tokens:
            return False
        chunk = {
            "doc": doc_name,
            "chunk_id": len(self.chunks),
            "text": text,
            "tf": Counter(tokens),
        }
        if title:
            chunk["title"] = title
        self.chunks.append(chunk)
        for term in set(tokens):
            self._df[term] += 1
        self._n_chunks += 1
        return True

    def add_sections(self, doc_name, text):
        """Index one chunk per '## ' section (each stays a self-contained fact block)."""
        added = 0
        for block in re.split(r"(?m)^(?=## )", text):
            block = block.strip()
            if not block:
                continue
            first = block.splitlines()[0]
            title = first.lstrip("# ").strip() or "Overview"
            if self._add_chunk(doc_name, block, title=title):
                added += 1
        return added

    def add_document(self, doc_name, text, chunk_words=180, overlap_words=30):
        """Split text into overlapping word chunks and index them."""
        words = text.split()
        if not words:
            return 0

        step = max(chunk_words - overlap_words, 1)
        added = 0

        for start in range(0, len(words), step):
            piece_words = words[start:start + chunk_words]
            if not piece_words:
                continue

            if self._add_chunk(doc_name, " ".join(piece_words)):
                added += 1

            if start + chunk_words >= len(words):
                break

        return added

    def _idf(self, term):
        df = self._df.get(term, 0)
        return math.log((self._n_chunks + 1) / (df + 1)) + 1

    def _vector(self, tf_counter):
        return {term: count * self._idf(term) for term, count in tf_counter.items()}

    @staticmethod
    def _cosine(v1, v2):
        common = set(v1) & set(v2)
        if not common:
            return 0.0
        dot = sum(v1[t] * v2[t] for t in common)
        norm1 = math.sqrt(sum(v * v for v in v1.values())) or 1e-9
        norm2 = math.sqrt(sum(v * v for v in v2.values())) or 1e-9
        return dot / (norm1 * norm2)

    def search(self, query, top_k=3, min_score=0.05):
        """Return up to top_k (score, chunk) pairs relevant to the query."""
        if not self.chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        query_vec = self._vector(Counter(query_tokens))

        scored = []
        for chunk in self.chunks:
            chunk_vec = self._vector(chunk["tf"])
            score = self._cosine(query_vec, chunk_vec)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [(score, chunk) for score, chunk in scored[:top_k] if score >= min_score]

    @property
    def documents(self):
        return sorted(set(c["doc"] for c in self.chunks))

    def has_documents(self):
        return len(self.chunks) > 0

    def clear(self):
        self.chunks = []
        self._df = Counter()
        self._n_chunks = 0
