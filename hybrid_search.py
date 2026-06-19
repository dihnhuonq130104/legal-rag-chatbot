"""Small dependency-free lexical index used by hybrid retrieval and evaluation."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict


class BM25Index:
    """BM25 over corpus chunks. Tokenization intentionally keeps Vietnamese words."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokens = [self.tokenize(document) for document in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()

        for document_id, tokens in enumerate(self.tokens):
            frequencies = Counter(tokens)
            document_frequency.update(frequencies.keys())
            for term, frequency in frequencies.items():
                self.postings[term].append((document_id, frequency))

        total = max(len(documents), 1)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores: defaultdict[int, float] = defaultdict(float)
        for term in set(self.tokenize(query)):
            for document_id, frequency in self.postings.get(term, []):
                length = self.lengths[document_id]
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self.average_length, 1)
                )
                scores[document_id] += self.idf[term] * frequency * (self.k1 + 1) / denominator
        return sorted(scores.items(), key=lambda result: result[1], reverse=True)[:top_k]
