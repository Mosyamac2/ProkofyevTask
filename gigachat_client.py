"""Тонкая обёртка над официальным SDK `gigachat`.

- Один singleton-клиент на процесс.
- Глобальный min-interval throttle на chat и embeddings (защита от 429).
- Retry поверх временных ошибок, в т.ч. RateLimitError, с длинным backoff.
- Удобный `embed_batch()` для эмбеддингов и `chat_completion()` для генерации.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Iterable, List, Optional

import numpy as np
from gigachat import GigaChat
from gigachat.exceptions import RateLimitError, ResponseError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config

log = logging.getLogger(__name__)

_client: Optional[GigaChat] = None
_client_lock = threading.Lock()


def _ensure_client() -> GigaChat:
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        if not config.GIGACHAT_AUTH_KEY:
            raise RuntimeError(
                "GIGACHAT_AUTH_KEY не задан. Скопируйте .env.example в .env "
                "и заполните ключ авторизации."
            )
        _client = GigaChat(
            credentials=config.GIGACHAT_AUTH_KEY,
            scope=config.GIGACHAT_SCOPE,
            verify_ssl_certs=config.GIGACHAT_VERIFY_SSL,
            timeout=config.EMBED_REQ_TIMEOUT,
        )
        return _client


def close():
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


class _Throttle:
    """Простой потокобезопасный min-interval throttle.

    Если несколько потоков обращаются параллельно — гарантирует, что
    между фактическими отправками будет ≥ min_interval секунд.
    """

    def __init__(self, min_interval_sec: float):
        self.min_interval = max(0.0, float(min_interval_sec))
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next_at - now
            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()
            self._next_at = now + self.min_interval

    def penalize(self, retry_after_sec: float) -> None:
        """Подвинуть «следующий слот» вперёд после 429."""
        if retry_after_sec <= 0:
            return
        with self._lock:
            target = time.monotonic() + float(retry_after_sec)
            if target > self._next_at:
                self._next_at = target


_CHAT_THROTTLE = _Throttle(config.JUDGE_MIN_INTERVAL_SEC)
_EMBED_THROTTLE = _Throttle(config.EMBED_MIN_INTERVAL_SEC)


def _retry_after_seconds(exc: BaseException) -> float:
    """Достаёт `Retry-After` из заголовков 429, если SDK его положил."""
    headers = getattr(exc, "headers", None)
    if not headers:
        return 0.0
    try:
        val = headers.get("Retry-After") or headers.get("retry-after")
    except Exception:
        return 0.0
    if not val:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _retry_wait(retry_state) -> float:
    """tenacity wait-функция: exp backoff + уважение Retry-After на 429."""
    base = min(config.RETRY_MAX_WAIT_SEC, 2.0 * (2 ** (retry_state.attempt_number - 1)))
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, RateLimitError):
        ra = _retry_after_seconds(exc)
        if ra > 0:
            base = max(base, ra)
        # 429 — подвинем все будущие вызовы в очереди
        _CHAT_THROTTLE.penalize(base)
        log.warning(
            "429 from GigaChat (attempt %d) — backing off %.1fs",
            retry_state.attempt_number, base,
        )
    return base


_RETRYABLE = (RateLimitError, ResponseError, ConnectionError, TimeoutError)


def _chat_retry():
    return retry(
        wait=_retry_wait,
        stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )


def _embed_retry():
    return retry(
        wait=wait_exponential(multiplier=1, min=2, max=config.RETRY_MAX_WAIT_SEC),
        stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )


@_embed_retry()
def _embed_call(texts: List[str], model: str) -> List[List[float]]:
    _EMBED_THROTTLE.wait()
    cl = _ensure_client()
    resp = cl.embeddings(texts=texts, model=model)
    return [d.embedding for d in resp.data]


def embed_batch(
    texts: Iterable[str],
    model: Optional[str] = None,
    batch_size: int = config.EMBED_BATCH_SIZE,
) -> np.ndarray:
    """Возвращает float32-матрицу [N, d] эмбеддингов в том же порядке, что texts."""
    model = model or config.GIGACHAT_EMBED_MODEL
    texts = list(texts)
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        vectors.extend(_embed_call(chunk, model))
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


@_chat_retry()
def chat_completion(
    messages: List[dict],
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> str:
    _CHAT_THROTTLE.wait()
    model = model or config.GIGACHAT_CHAT_MODEL
    cl = _ensure_client()
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = cl.chat(payload)
    return resp.choices[0].message.content
