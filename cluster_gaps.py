"""Кластеризация непокрытых запросов: UMAP → HDBSCAN, опц. именование тем через LLM."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
import gigachat_client

log = logging.getLogger(__name__)


def cluster(
    embeds: np.ndarray,
    min_cluster_size: int = config.CLUSTER_MIN_SIZE,
    umap_dim: int = config.CLUSTER_UMAP_DIM,
    random_state: int = 42,
) -> np.ndarray:
    """Возвращает массив cluster_id (длина = len(embeds)); -1 = noise."""
    if len(embeds) < min_cluster_size * 2:
        return np.full(len(embeds), -1, dtype=int)

    import hdbscan
    import umap

    reducer = umap.UMAP(
        n_components=umap_dim,
        metric="cosine",
        n_neighbors=15,
        random_state=random_state,
    )
    reduced = reducer.fit_transform(embeds)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(reduced).astype(int)


def name_cluster(samples: list[str]) -> str:
    """Просим GigaChat назвать тему кластера одним коротким предложением."""
    sample_block = "\n".join(f"- {s}" for s in samples[:10])
    messages = [
        {
            "role": "system",
            "content": (
                "Ты помогаешь HR-аналитику. Дан список похожих запросов сотрудников. "
                "Сформулируй одним коротким предложением (3-7 слов) общую тему этих запросов "
                "на русском языке. Без вступлений, только тема."
            ),
        },
        {"role": "user", "content": sample_block},
    ]
    try:
        return gigachat_client.chat_completion(messages, max_tokens=50).strip().strip('"«»')
    except Exception as e:
        log.warning("Cluster naming failed: %s", e)
        return samples[0] if samples else "<нет примеров>"


def summarize_clusters(
    queries_df: pd.DataFrame,
    cluster_ids: np.ndarray,
    name_topics: bool = True,
) -> pd.DataFrame:
    """Сводка по кластерам: id, размер, объём событий, примеры, suggested theme."""
    df = queries_df.copy()
    df["cluster_id"] = cluster_ids
    rows = []
    for cid, g in df[df["cluster_id"] >= 0].groupby("cluster_id"):
        top = g.sort_values("count", ascending=False)
        samples = top["sample_text"].head(10).tolist()
        rows.append(
            {
                "cluster_id": int(cid),
                "n_unique": int(len(g)),
                "n_events": int(g["count"].sum()),
                "top_examples": " | ".join(samples[:5]),
                "all_examples": samples,
                "theme": name_cluster(samples) if name_topics else samples[0] if samples else "",
            }
        )
    cols = ["cluster_id", "n_unique", "n_events", "top_examples", "all_examples", "theme"]
    out = pd.DataFrame(rows, columns=cols)
    if len(out):
        out = out.sort_values("n_events", ascending=False).reset_index(drop=True)
    return out
