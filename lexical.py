"""Лексический матчер: BM25 + character n-gram TF-IDF.

Используется как второй сигнал retrieval, объединяется с эмбеддингами в `matcher.py`
через Reciprocal Rank Fusion.
"""
from __future__ import annotations

import re
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN_RE = re.compile(r"[А-Яа-яA-Za-z0-9]+", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class LexicalIndex:
    """Объединённый ранкер: средне-нормированный rank по BM25 и char-n-gram."""

    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self._bm25 = BM25Okapi([tokenize(t) for t in corpus])
        self._char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )
        self._char_mat = self._char_vec.fit_transform(corpus)

    def score(self, query: str) -> np.ndarray:
        """Возвращает массив [len(corpus)] совокупных score (выше — релевантнее)."""
        bm = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)
        if bm.max() > 0:
            bm = bm / bm.max()
        qv = self._char_vec.transform([query])
        ch = (self._char_mat @ qv.T).toarray().ravel().astype(np.float32)
        if ch.max() > 0:
            ch = ch / ch.max()
        return 0.6 * bm + 0.4 * ch

    def topk(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        s = self.score(query)
        idx = np.argpartition(-s, min(k, len(s) - 1))[:k]
        idx = idx[np.argsort(-s[idx])]
        return [(int(i), float(s[i])) for i in idx]
