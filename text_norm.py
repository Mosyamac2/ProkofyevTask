"""Нормализация текста — для дедупликации и матчинга."""
from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")
_PUNCT_EDGES_RE = re.compile(r"^[\s\W_]+|[\s\W_]+$", flags=re.UNICODE)
_REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1+")
_ONLY_NON_LETTERS_RE = re.compile(r"^[\W\d_]+$", flags=re.UNICODE)


def normalize(text: str | None) -> str:
    """Приведение к каноничному виду: NFKC, lower, схлоп пробелов, обрезка пунктуации по краям."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.replace("ё", "е").replace("Ё", "Е")
    s = s.lower()
    s = _REPEAT_PUNCT_RE.sub(r"\1", s)
    s = _WS_RE.sub(" ", s).strip()
    s = _PUNCT_EDGES_RE.sub("", s).strip()
    return s


def is_junk(norm_text: str, min_len: int = 3) -> bool:
    """Реплика-мусор: пустая, слишком короткая, без букв, служебная команда."""
    if not norm_text or len(norm_text) < min_len:
        return True
    if _ONLY_NON_LETTERS_RE.match(norm_text):
        return True
    if norm_text.startswith("/") and len(norm_text.split()) == 1:
        return True
    return False


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
