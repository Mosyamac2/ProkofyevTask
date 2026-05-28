"""Тонкая обёртка над официальным SDK `gigachat`.

- Один singleton-клиент на процесс.
- Retry поверх временных ошибок.
- Удобный `embed_batch()` для эмбеддингов и `chat()` для генерации.
"""
from __future__ import annotations

import logging
import threading
from typing import Iterable, List, Optional

import numpy as np
from gigachat import GigaChat
from gigachat.exceptions import ResponseError
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


_RETRY = dict(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((ResponseError, ConnectionError, TimeoutError)),
    reraise=True,
)


@retry(**_RETRY)
def _embed_call(texts: List[str], model: str) -> List[List[float]]:
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


@retry(**_RETRY)
def chat_completion(
    messages: List[dict],
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> str:
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
