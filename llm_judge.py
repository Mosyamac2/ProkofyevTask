"""Stage D — LLM cross-encoder.

Для каждого реального запроса даём LLM сам запрос и top-K сценариев-кандидатов.
LLM возвращает JSON: какой кандидат покрывает запрос (или ни один), уверенность,
короткое обоснование. Все ответы кешируются на диске.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

import config
import gigachat_client

log = logging.getLogger(__name__)

JUDGE_PARALLELISM = 6
_cache_lock = threading.Lock()

SYSTEM_PROMPT = (
    "Ты помогаешь HR-аналитику проверить, покрывает ли каталог сценариев реальные "
    "запросы пользователей внутренней HR-системы Сбера «Пульс». Тебе дают один "
    "реальный запрос сотрудника и список из {k} сценариев-кандидатов из каталога. "
    "Для каждого сценария указаны его категория и тип действия. "
    "Твоя задача — определить, есть ли среди кандидатов сценарий, который "
    "ПОКРЫВАЕТ намерение пользователя.\n\n"
    "ПОКРЫВАЕТ = одинаковое намерение (что хочет сделать) и одинаковый тип действия "
    "(поиск информации / выполнение / помощь). Просто общая тема — этого мало.\n\n"
    "ВЕРДИКТЫ:\n"
    "- covered: один из кандидатов точно покрывает запрос — назови его номер.\n"
    "- partial: кандидат близок по теме, но намерение или scope другие.\n"
    "- oos:    ни один кандидат не покрывает запрос (out-of-scope для каталога).\n\n"
    "ОТВЕТ — строго JSON без обёртки:\n"
    "{{\"verdict\": \"covered|partial|oos\", \"chosen\": <int|null>, "
    "\"confidence\": <0..1>, \"reason\": \"...\"}}"
)


@dataclass
class JudgeResult:
    verdict: str           # 'covered' | 'partial' | 'oos' | 'error'
    chosen: int | None     # 1-based индекс кандидата
    confidence: float
    reason: str


def _key(query: str, candidates_payload: str, model: str) -> str:
    blob = f"{model}::{query}\n---\n{candidates_payload}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {row.key: json.loads(row.payload) for row in df.itertuples()}


def _save_cache(cache: dict[str, dict], path: Path) -> None:
    df = pd.DataFrame(
        {"key": list(cache.keys()), "payload": [json.dumps(v, ensure_ascii=False) for v in cache.values()]}
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


_JSON_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def _parse_response(raw: str) -> JudgeResult:
    if not raw:
        return JudgeResult("error", None, 0.0, "empty response")
    m = _JSON_RE.search(raw)
    if not m:
        return JudgeResult("error", None, 0.0, f"no json in: {raw[:200]}")
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return JudgeResult("error", None, 0.0, f"json parse error: {e}; raw={raw[:200]}")
    verdict = str(d.get("verdict", "")).lower().strip()
    if verdict not in {"covered", "partial", "oos"}:
        verdict = "error"
    chosen = d.get("chosen")
    if chosen is not None:
        try:
            chosen = int(chosen)
        except (TypeError, ValueError):
            chosen = None
    try:
        conf = float(d.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    reason = str(d.get("reason", ""))[:500]
    return JudgeResult(verdict, chosen, conf, reason)


def _format_candidates(scen_rows: list[dict]) -> str:
    lines = []
    for i, s in enumerate(scen_rows, start=1):
        cat = s.get("category", "")
        atype = s.get("action_type", "")
        text = s.get("scenario_query", "")
        lines.append(f"{i}. [{cat}] [{atype}] {text}")
    return "\n".join(lines)


def _build_messages(query_text: str, scen_rows: list[dict]) -> tuple[str, list[dict]]:
    cand_block = _format_candidates(scen_rows)
    key = _key(query_text, cand_block, config.GIGACHAT_CHAT_MODEL)
    system = SYSTEM_PROMPT.format(k=len(scen_rows))
    user = (
        f"РЕАЛЬНЫЙ ЗАПРОС:\n{query_text}\n\n"
        f"КАНДИДАТЫ:\n{cand_block}\n\n"
        "Ответь JSON по схеме."
    )
    return key, [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def judge_single(query_text: str, scen_rows: list[dict], cache: dict[str, dict]) -> JudgeResult:
    key, messages = _build_messages(query_text, scen_rows)
    with _cache_lock:
        cached = cache.get(key)
    if cached is not None:
        return JudgeResult(**cached)
    raw = gigachat_client.chat_completion(messages)
    result = _parse_response(raw)
    with _cache_lock:
        cache[key] = asdict(result)
    return result


def judge_all(
    queries: list[str],
    candidates_per_query: list[list[dict]],
    cache_path: Path | None = None,
    flush_every: int = 100,
    parallelism: int = JUDGE_PARALLELISM,
) -> list[JudgeResult]:
    """Прогоняет LLM-судью по всем запросам, инкрементально сохраняет кеш.

    Параллелит запросы пулом потоков (ThreadPool — IO-bound, GIL не мешает).
    """
    cache_path = cache_path or config.JUDGE_CACHE_PATH
    cache = _load_cache(cache_path)

    items = list(zip(queries, candidates_per_query))
    results: list[JudgeResult | None] = [None] * len(items)

    misses_since_flush = 0

    def _work(idx_q_cands):
        i, q, cands = idx_q_cands
        return i, judge_single(q, cands, cache)

    tasks = [(i, q, c) for i, (q, c) in enumerate(items)]

    with ThreadPoolExecutor(max_workers=parallelism) as ex:
        futs = [ex.submit(_work, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc="LLM judge", unit="q"):
            i, res = f.result()
            results[i] = res
            misses_since_flush += 1
            if misses_since_flush >= flush_every:
                with _cache_lock:
                    _save_cache(cache, cache_path)
                misses_since_flush = 0

    if misses_since_flush:
        with _cache_lock:
            _save_cache(cache, cache_path)

    return [r if r is not None else JudgeResult("error", None, 0.0, "missing") for r in results]
