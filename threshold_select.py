"""Подбор бинарного порога по score_embed (cosine GigaChat) через LLM-судью.

Пайплайн этапа:
1. Выбираем N (по умолчанию 100) случайных free-text запросов с их top-1 сценарием
   и значением score_embed (cosine, непрерывный в [-1, 1]).
2. Прогоняем 100 пар через LLM-судью (`llm_judge.judge_pair`): 1 = покрывает, 0 = нет.
3. Сканируем все возможные пороги по cosine, считаем precision/recall/F-beta
   с positive=oos (запрос НЕ покрыт). Возвращаем порог, максимизирующий F-beta.

Замечание про score: top-K-ранжирование внутри matcher.py использует RRF
(BM25 + cosine), но как одномерная шкала уверенности этого top-1 берётся
непрерывный cosine — он на порядки granular-нее, чем дискретный RRF.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import config
import llm_judge

log = logging.getLogger(__name__)


SCORE_COL = "score_embed"


@dataclass
class ThresholdResult:
    threshold: float          # выбранный порог по score_embed: cosine >= T → covered
    fbeta: float              # значение F-beta в точке порога (positive=oos)
    beta: float
    precision_oos: float
    recall_oos: float
    n_sample: int
    n_pos_oos: int            # сколько из выборки на самом деле oos (по судье)
    n_neg_oos: int            # сколько покрыто
    n_errors: int             # сколько судья не смог разметить
    sampling_strategy: str = "random"
    # Альтернативный порог по F1 — для прозрачности сравнения
    threshold_f1: float | None = None
    f1: float | None = None
    precision_oos_f1: float | None = None
    recall_oos_f1: float | None = None
    audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    score_grid: pd.DataFrame = field(default_factory=pd.DataFrame)


def _build_pool(uniq: pd.DataFrame, candidates_top1: Sequence[dict]) -> pd.DataFrame:
    mask = (uniq["input_kind"] == "free")
    rows = []
    for i, row in uniq.loc[mask].iterrows():
        c = candidates_top1[i]
        if c is None or c.get("score_embed") is None:
            continue
        rows.append({
            "query_idx": i,
            "sample_text": row["sample_text"],
            "question_norm": row["question_norm"],
            "score_embed": float(c["score_embed"]),
            "score_rrf": float(c["score_rrf"]),
            "scenario_id": c["scenario_id"],
            "scenario_query": c["scenario_query"],
            "category": c["category"],
            "action_type": c["action_type"],
            "agent": c["agent"],
        })
    return pd.DataFrame(rows)


def sample_free_text_pairs(
    uniq: pd.DataFrame,
    candidates_top1: Sequence[dict],
    n: int = config.JUDGE_SAMPLE_N,
    seed: int = config.JUDGE_SAMPLE_SEED,
    strategy: str = config.JUDGE_SAMPLE_STRATEGY,
    n_bins: int = config.JUDGE_SAMPLE_BINS,
) -> pd.DataFrame:
    """Выборка из free-text запросов с непустым top-1.

    strategy='random':     n чистых случайных строк.
    strategy='stratified': N / n_bins из каждой децили score_rrf.
        Гарантирует наличие в выборке и низко-, и высоко-score запросов
        (борьба с классовым дисбалансом при оптимизации F-beta).
    """
    pool = _build_pool(uniq, candidates_top1)
    if pool.empty:
        log.warning("Empty free-text pool")
        return pool
    if len(pool) <= n:
        log.warning("Sample pool (%d) <= requested n (%d); using full pool", len(pool), n)
        return pool.reset_index(drop=True)

    strategy = (strategy or "random").lower()
    if strategy == "random":
        return pool.sample(n=n, random_state=seed).reset_index(drop=True)

    if strategy != "stratified":
        log.warning("Unknown sampling strategy %r; falling back to random", strategy)
        return pool.sample(n=n, random_state=seed).reset_index(drop=True)

    # Стратифицированный отбор по децилям score_embed
    try:
        pool["bin"] = pd.qcut(pool[SCORE_COL], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        log.warning("qcut failed (likely too many duplicate scores); using random sampling")
        return pool.sample(n=n, random_state=seed).reset_index(drop=True)

    actual_bins = pool["bin"].nunique()
    per_bin = max(1, n // max(actual_bins, 1))
    parts = []
    rng = np.random.default_rng(seed)
    for b, g in pool.groupby("bin"):
        take = min(per_bin, len(g))
        parts.append(g.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    sample = pd.concat(parts).drop(columns=["bin"]).reset_index(drop=True)

    # Добор до n, если из-за округления не хватает
    if len(sample) < n:
        remaining = pool.drop(sample.index, errors="ignore")
        if len(remaining) > 0:
            extra = remaining.sample(n=min(n - len(sample), len(remaining)), random_state=seed + 1)
            sample = pd.concat([sample, extra.drop(columns=["bin"], errors="ignore")])
    log.info("Stratified sampling: %d bins × ~%d = %d rows", actual_bins, per_bin, len(sample))
    return sample.head(n).reset_index(drop=True)


def judge_sample(sample: pd.DataFrame) -> pd.DataFrame:
    """Прогоняет LLM-судью по выборке. Добавляет колонки [label, reason] (1/0/-1)."""
    pairs = [
        (
            row["sample_text"],
            {
                "scenario_query": row["scenario_query"],
                "category": row["category"],
                "action_type": row["action_type"],
            },
        )
        for _, row in sample.iterrows()
    ]
    results = llm_judge.judge_pairs(pairs)
    out = sample.copy()
    out["label"] = [r.match for r in results]
    out["reason"] = [r.reason for r in results]
    return out


def select_threshold(
    scores: np.ndarray,
    labels_covered: np.ndarray,
    beta: float = config.JUDGE_FBETA,
) -> tuple[float, dict, pd.DataFrame]:
    """Порог T, максимизирующий F-beta с positive = oos.

    `labels_covered`: 1 (covered) / 0 (oos). Строки с -1 (ошибки) должны быть отфильтрованы
    ВЫЗЫВАЮЩИМ КОДОМ перед передачей сюда.

    Правило бакетизации: score >= T → covered, score < T → oos.
    """
    scores = np.asarray(scores, dtype=float)
    y_cov = np.asarray(labels_covered, dtype=int)
    y_oos = 1 - y_cov  # positive = oos

    if len(scores) == 0 or y_oos.sum() == 0 or (1 - y_oos).sum() == 0:
        log.warning("Degenerate sample (n=%d, n_oos=%d, n_cov=%d) — returning median",
                    len(scores), int(y_oos.sum()), int((1 - y_oos).sum()))
        return (float(np.median(scores)) if len(scores) else 0.0,
                {"fbeta": 0.0, "precision_oos": 0.0, "recall_oos": 0.0, "degenerate": True},
                pd.DataFrame())

    # Кандидаты на порог: уникальные значения score + точки между ними.
    uniq = np.unique(scores)
    midpoints = (uniq[:-1] + uniq[1:]) / 2.0
    grid = np.concatenate([[uniq.min() - 1e-9], midpoints, [uniq.max() + 1e-9]])

    b2 = beta * beta
    rows = []
    for t in grid:
        pred_oos = (scores < t).astype(int)
        tp = int(((pred_oos == 1) & (y_oos == 1)).sum())
        fp = int(((pred_oos == 1) & (y_oos == 0)).sum())
        fn = int(((pred_oos == 0) & (y_oos == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        fb_denom = b2 * prec + rec
        f1_denom = prec + rec
        fb = (1 + b2) * prec * rec / fb_denom if fb_denom else 0.0
        f1 = 2 * prec * rec / f1_denom if f1_denom else 0.0
        rows.append({
            "threshold": float(t),
            "precision_oos": prec,
            "recall_oos": rec,
            "f1": f1,
            "fbeta": fb,
        })

    grid_df = pd.DataFrame(rows)
    best_fb_idx = int(grid_df["fbeta"].idxmax())
    best_f1_idx = int(grid_df["f1"].idxmax())
    best_fb = grid_df.iloc[best_fb_idx]
    best_f1 = grid_df.iloc[best_f1_idx]
    return (
        float(best_fb["threshold"]),
        {
            "fbeta": float(best_fb["fbeta"]),
            "precision_oos": float(best_fb["precision_oos"]),
            "recall_oos": float(best_fb["recall_oos"]),
            "threshold_f1": float(best_f1["threshold"]),
            "f1": float(best_f1["f1"]),
            "precision_oos_f1": float(best_f1["precision_oos"]),
            "recall_oos_f1": float(best_f1["recall_oos"]),
            "degenerate": False,
        },
        grid_df,
    )


def calibrate(
    uniq: pd.DataFrame,
    candidates_top1: Sequence[dict],
) -> ThresholdResult:
    """Полный цикл: выборка → судья → подбор порога."""
    sample = sample_free_text_pairs(uniq, candidates_top1)
    if sample.empty:
        log.warning("Empty sample, using fallback threshold %.4f", config.FALLBACK_THRESHOLD_EMBED)
        return ThresholdResult(
            threshold=config.FALLBACK_THRESHOLD_EMBED,
            fbeta=0.0, beta=config.JUDGE_FBETA,
            precision_oos=0.0, recall_oos=0.0,
            n_sample=0, n_pos_oos=0, n_neg_oos=0, n_errors=0,
        )
    judged = judge_sample(sample)

    valid = judged[judged["label"].isin([0, 1])].copy()
    n_errors = int((judged["label"] == -1).sum())
    if len(valid) < 10:
        log.warning("Too few valid judgments (%d); falling back to %.4f",
                    len(valid), config.FALLBACK_THRESHOLD_EMBED)
        return ThresholdResult(
            threshold=config.FALLBACK_THRESHOLD_EMBED,
            fbeta=0.0, beta=config.JUDGE_FBETA,
            precision_oos=0.0, recall_oos=0.0,
            n_sample=len(judged), n_pos_oos=0, n_neg_oos=0, n_errors=n_errors,
            audit=judged,
        )

    threshold, metrics, grid = select_threshold(
        valid[SCORE_COL].to_numpy(),
        valid["label"].to_numpy(),
        beta=config.JUDGE_FBETA,
    )
    return ThresholdResult(
        threshold=threshold,
        fbeta=metrics["fbeta"],
        beta=config.JUDGE_FBETA,
        precision_oos=metrics["precision_oos"],
        recall_oos=metrics["recall_oos"],
        n_sample=len(judged),
        n_pos_oos=int((valid["label"] == 0).sum()),
        n_neg_oos=int((valid["label"] == 1).sum()),
        n_errors=n_errors,
        sampling_strategy=config.JUDGE_SAMPLE_STRATEGY,
        threshold_f1=metrics.get("threshold_f1"),
        f1=metrics.get("f1"),
        precision_oos_f1=metrics.get("precision_oos_f1"),
        recall_oos_f1=metrics.get("recall_oos_f1"),
        audit=judged,
        score_grid=grid,
    )
