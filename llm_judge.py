"""LLM-судья (одна пара за раз, бинарный вердикт).

Используется только для разметки случайной выборки пар (real_query → top-1 scenario),
по которой подбирается порог по score_rrf. Никаких top-K, никаких grey-zone циклов.

Кеш на диске — sha1(model::query::scenario). Повторные прогоны не дёргают API.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

import config
import gigachat_client

log = logging.getLogger(__name__)

_cache_lock = threading.Lock()

SYSTEM_PROMPT = (
    "Ты помогаешь HR-аналитику проверить, покрывает ли каталог сценариев реальные "
    "запросы сотрудников внутренней HR-системы Сбера «Пульс». Тебе дают ОДИН "
    "реальный запрос сотрудника и ОДИН сценарий из каталога (с его категорией и "
    "типом действия).\n\n"
    "Реши: покрывает ли сценарий реальный запрос?\n"
    "ПОКРЫВАЕТ = одинаковое НАМЕРЕНИЕ (что хочет сделать пользователь) и одинаковый "
    "ОБЪЕКТ запроса. Просто общая тема (например, оба про отпуска) — этого мало.\n\n"
    "Ответ — СТРОГО JSON без обёртки, одна строка:\n"
    "{\"match\": 0|1, \"reason\": \"кратко, до 100 символов\"}\n"
    "1 = сценарий покрывает запрос, 0 = не покрывает."
)


@dataclass
class JudgeResult:
    match: int        # 1 covered, 0 not covered, -1 error
    reason: str

    @property
    def ok(self) -> bool:
        return self.match in (0, 1)


def _key(query: str, scenario_payload: str, model: str) -> str:
    blob = f"{model}::{query}\n---\n{scenario_payload}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {row.key: json.loads(row.payload) for row in df.itertuples()}


def _save_cache(cache: dict[str, dict], path: Path) -> None:
    df = pd.DataFrame(
        {
            "key": list(cache.keys()),
            "payload": [json.dumps(v, ensure_ascii=False) for v in cache.values()],
        }
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


_JSON_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def _parse_response(raw: str) -> JudgeResult:
    if not raw:
        return JudgeResult(-1, "empty response")
    m = _JSON_RE.search(raw)
    if not m:
        return JudgeResult(-1, f"no json: {raw[:200]}")
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return JudgeResult(-1, f"json parse: {e}; raw={raw[:200]}")
    try:
        match = int(d.get("match", -1))
    except (TypeError, ValueError):
        match = -1
    if match not in (0, 1):
        match = -1
    reason = str(d.get("reason", ""))[:200]
    return JudgeResult(match, reason)


def _format_scenario(scen: dict) -> str:
    cat = scen.get("category", "")
    atype = scen.get("action_type", "")
    text = scen.get("scenario_query", "")
    return f"[{cat}] [{atype}] {text}"


def judge_pair(query_text: str, scenario: dict, cache: dict[str, dict] | None = None) -> JudgeResult:
    """Один LLM-вызов: пара (запрос, сценарий) → 0/1."""
    if cache is None:
        cache = _load_cache(config.JUDGE_CACHE_PATH)
    scen_str = _format_scenario(scenario)
    key = _key(query_text, scen_str, config.GIGACHAT_CHAT_MODEL)
    with _cache_lock:
        cached = cache.get(key)
    if cached is not None:
        return JudgeResult(**cached)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"РЕАЛЬНЫЙ ЗАПРОС:\n{query_text}\n\nСЦЕНАРИЙ:\n{scen_str}\n\nОтветь JSON."},
    ]
    try:
        raw = gigachat_client.chat_completion(messages, max_tokens=200)
        result = _parse_response(raw)
    except Exception as exc:  # noqa: BLE001
        result = JudgeResult(-1, f"{type(exc).__name__}: {exc}"[:200])

    with _cache_lock:
        cache[key] = asdict(result)
    return result


def judge_pairs(
    pairs: Iterable[tuple[str, dict]],
    cache_path: Path | None = None,
    flush_every: int | None = None,
) -> list[JudgeResult]:
    """Прогоняет последовательность пар через судью, кеш сохраняет каждые
    `flush_every` запросов и в finally."""
    cache_path = cache_path or config.JUDGE_CACHE_PATH
    flush_every = flush_every if flush_every is not None else config.JUDGE_FLUSH_EVERY

    cache = _load_cache(cache_path)
    pairs_list = list(pairs)
    results: list[JudgeResult] = []

    def _flush() -> None:
        with _cache_lock:
            _save_cache(cache, cache_path)

    try:
        for i, (q, scen) in enumerate(tqdm(pairs_list, desc="LLM judge", unit="q")):
            results.append(judge_pair(q, scen, cache=cache))
            if (i + 1) % flush_every == 0:
                _flush()
    finally:
        _flush()

    return results
