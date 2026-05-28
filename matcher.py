"""Retrieval-этап: для каждого реального запроса — top-K сценариев.

Каскад:
- Stage A (exact / button) обрабатывается в `dialog_context.py` и `data_loader.py`.
- Stage B/C здесь: BM25 + эмбеддинги, объединение через Reciprocal Rank Fusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from lexical import LexicalIndex


@dataclass
class Candidate:
    scenario_id: str
    rank_embed: int | None
    rank_lex: int | None
    score_embed: float | None
    score_lex: float | None
    rrf: float


def _argsort_topn(scores: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(scores))
    idx = np.argpartition(-scores, n - 1)[:n]
    return idx[np.argsort(-scores[idx])]


def retrieve_topk(
    real_norm_texts: List[str],
    real_embeds: np.ndarray,
    scen_df: pd.DataFrame,
    scen_embeds: np.ndarray,
    k: int = config.JUDGE_TOP_K,
    topn_per_source: int = 30,
) -> List[List[Candidate]]:
    """Возвращает для каждого реального запроса список Candidate (длина k).

    Параметры
    ---------
    real_norm_texts : нормализованные строки реальных запросов (для BM25)
    real_embeds     : [N_real, d] нормированные эмбеддинги реальных запросов
    scen_df         : DataFrame сценариев со столбцом `scenario_id` и `embed_text`
    scen_embeds     : [N_scen, d] нормированные эмбеддинги сценариев
    """
    lex = LexicalIndex(scen_df["embed_text"].tolist())
    scen_ids = scen_df["scenario_id"].to_numpy()
    sim_matrix = real_embeds @ scen_embeds.T  # [N_real, N_scen], cosine

    results: List[List[Candidate]] = []
    rrf_k = config.RRF_K

    for i, q in enumerate(tqdm(real_norm_texts, desc="retrieve", unit="q")):
        sim_row = sim_matrix[i]
        top_embed = _argsort_topn(sim_row, topn_per_source)
        lex_scores = lex.score(q)
        top_lex = _argsort_topn(lex_scores, topn_per_source)

        rrf_map: dict[int, dict] = {}
        for rank, j in enumerate(top_embed):
            rrf_map.setdefault(int(j), {"re": None, "rl": None, "se": None, "sl": None})
            rrf_map[int(j)]["re"] = rank
            rrf_map[int(j)]["se"] = float(sim_row[j])
        for rank, j in enumerate(top_lex):
            rrf_map.setdefault(int(j), {"re": None, "rl": None, "se": None, "sl": None})
            rrf_map[int(j)]["rl"] = rank
            rrf_map[int(j)]["sl"] = float(lex_scores[j])

        scored = []
        for j, d in rrf_map.items():
            score = 0.0
            if d["re"] is not None:
                score += 1.0 / (rrf_k + d["re"] + 1)
            if d["rl"] is not None:
                score += 1.0 / (rrf_k + d["rl"] + 1)
            scored.append((j, d, score))
        scored.sort(key=lambda x: -x[2])

        cands: List[Candidate] = []
        for j, d, rrf_score in scored[:k]:
            cands.append(
                Candidate(
                    scenario_id=str(scen_ids[j]),
                    rank_embed=d["re"],
                    rank_lex=d["rl"],
                    score_embed=d["se"],
                    score_lex=d["sl"],
                    rrf=rrf_score,
                )
            )
        results.append(cands)
    return results
