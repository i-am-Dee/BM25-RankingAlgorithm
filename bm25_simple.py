# bm25_simple.py
# Minimal BM25 implementation from scratch (no external libraries)

import math
from collections import Counter

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        corpus: list[list[str]] - tokenized documents (each doc is a list of terms)
        k1, b: BM25 hyperparameters
        """
        self.corpus = corpus
        self.N = len(corpus)
        self.k1 = k1
        self.b = b

        # Precompute document lengths and average length
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 0.0

        # Document frequency for each term
        self.df = Counter()
        for doc in corpus:
            for term in set(doc):
                self.df[term] += 1

        # Precompute IDF (with 0.5-smoothing)
        self.idf = {}
        for term, df in self.df.items():
            # Classic BM25 uses ln((N - df + 0.5)/(df + 0.5))
            # Note: This can be negative for very common terms (that’s OK and expected).
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1e-12)

    @staticmethod
    def simple_tokenize(text):
        """A very simple lowercase tokenizer that keeps alphanumerics and underscores."""
        out, buf = [], []
        for ch in text.lower():
            if ch.isalnum() or ch == '_':
                buf.append(ch)
            else:
                if buf:
                    out.append(''.join(buf))
                    buf = []
        if buf:
            out.append(''.join(buf))
        return out

    def score(self, query_tokens, doc_index):
        doc = self.corpus[doc_index]
        dl = self.doc_lens[doc_index]
        tf = Counter(doc)

        score = 0.0
        for q in query_tokens:
            if q not in self.idf:
                continue
            f = tf.get(q, 0)
            if f == 0:
                continue
            idf = self.idf[q]
            denom = f + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
            score += idf * (f * (self.k1 + 1)) / (denom if denom != 0 else 1.0)
        return score

    def rank(self, raw_query, topk=None):
        q = self.simple_tokenize(raw_query)
        scores = [(i, self.score(q, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk] if topk is not None else scores


def demo():
    print("=== BM25 Demo (from scratch) ===")

    # Tiny example corpus
    raw_corpus = [
        "BM25 is a ranking function used by search engines.",
        "We explain the BM25 algorithm for information retrieval.",
        "Neural embeddings capture semantic similarity.",
        "Hybrid search combines BM25 with embeddings for better retrieval.",
    ]

    # Tokenize the corpus
    tokenized = [BM25.simple_tokenize(doc) for doc in raw_corpus]

    # Build the BM25 index
    bm25 = BM25(tokenized, k1=1.5, b=0.75)

    # Try a few sample queries
    sample_queries = [
        "bm25 ranking function",
        "hybrid search with embeddings",
        "semantic similarity",
    ]

    for q in sample_queries:
        ranked = bm25.rank(q)
        print(f"\nQuery: {q}")
        for idx, score in ranked:
            print(f"  Score={score: .4f} | Doc[{idx}]: {raw_corpus[idx]}")

if __name__ == "__main__":
    demo()
    # For interactive CLI, you can add:
    # user_q = input("\nType your own query (or press Enter to exit): ").strip()
    # if user_q:
    #     ... rank and print results ...
