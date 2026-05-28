"""Parquet-кеш эмбеддингов: ключ = sha1(text + model)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
import gigachat_client


def _key(text: str, model: str) -> str:
    return hashlib.sha1(f"{model}::{text}".encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {row.key: np.asarray(row.vec, dtype=np.float32) for row in df.itertuples()}


def _save(cache: dict[str, np.ndarray], path: Path) -> None:
    if not cache:
        return
    df = pd.DataFrame(
        {
            "key": list(cache.keys()),
            "vec": [v.tolist() for v in cache.values()],
        }
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def embed_with_cache(
    texts: List[str],
    model: str | None = None,
    cache_path: Path | None = None,
    desc: str = "embed",
) -> np.ndarray:
    """Возвращает матрицу эмбеддингов [len(texts), d] (L2-нормированные).

    Все промежуточные значения кешируются на диске; повторные вызовы
    для тех же текстов API не дёргают.
    """
    model = model or config.GIGACHAT_EMBED_MODEL
    cache_path = cache_path or config.EMBED_CACHE_PATH

    cache = _load(cache_path)
    keys = [_key(t, model) for t in texts]
    missing_idx = [i for i, k in enumerate(keys) if k not in cache]

    if missing_idx:
        miss_texts = [texts[i] for i in missing_idx]
        miss_keys = [keys[i] for i in missing_idx]
        bs = config.EMBED_BATCH_SIZE
        new_vecs: list[np.ndarray] = []
        for s in tqdm(range(0, len(miss_texts), bs), desc=desc, unit="batch"):
            chunk = miss_texts[s : s + bs]
            v = gigachat_client.embed_batch(chunk, model=model, batch_size=bs)
            new_vecs.append(v)
        all_new = np.vstack(new_vecs) if new_vecs else np.zeros((0, 0), np.float32)
        for k, v in zip(miss_keys, all_new):
            cache[k] = v.astype(np.float32)
        _save(cache, cache_path)

    dim = next(iter(cache.values())).shape[0]
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, k in enumerate(keys):
        out[i] = cache[k]
    return out
