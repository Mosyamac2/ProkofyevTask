"""Конфигурация пайплайна. Все настройки читаются из .env при наличии."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.resolve()
DATA_SCENARIOS = ROOT / "Сценарии B2E.xlsx"
DATA_LOGS = ROOT / "orch_logs_to_2026_05_27.xlsx"

CACHE_DIR = ROOT / "cache"
OUT_DIR = ROOT / "out"
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

EMBED_CACHE_PATH = CACHE_DIR / "embeddings.parquet"
JUDGE_CACHE_PATH = CACHE_DIR / "judge_cache.parquet"
INTERMEDIATE_DIR = CACHE_DIR / "intermediate"
INTERMEDIATE_DIR.mkdir(exist_ok=True)

# GigaChat
GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY", "").strip()
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip()
GIGACHAT_EMBED_MODEL = os.getenv("GIGACHAT_EMBED_MODEL", "EmbeddingsGigaR").strip()
GIGACHAT_CHAT_MODEL = os.getenv("GIGACHAT_CHAT_MODEL", "GigaChat-2-Max").strip()
GIGACHAT_VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true"

# Параметры пайплайна
EMBED_BATCH_SIZE = 50            # сколько строк за один embeddings-запрос
JUDGE_TOP_K = 5                  # сколько кандидатов отдавать LLM-судье
JUDGE_REQ_TIMEOUT = 120          # таймаут одного chat-запроса (сек)
EMBED_REQ_TIMEOUT = 60

# Retrieval-объединение
RRF_K = 60                        # параметр Reciprocal Rank Fusion

# Stage A
FUZZY_THRESHOLD = 92              # rapidfuzz partial_ratio для exact-варианта
BUTTON_N_QUESTION_RATIO = 0.80    # доля n_question>1 чтобы счесть текст «кнопкой»
BUTTON_MIN_COUNT = 5              # мин. частота, чтобы вообще считать button-кандидатом

# Фильтрация мусора
MIN_QUERY_LEN_CHARS = 3
MAX_QUERY_LEN_CHARS = 1000

# Калибровка / бакетизация (используется как fallback; основной вердикт — у Stage D)
COVERED_PROB_THRESHOLD = 0.70     # P(match) >= → точно покрыт
OOS_PROB_THRESHOLD = 0.30         # P(match) <  → совсем не похоже

# Кластеризация gap-ов
CLUSTER_MIN_SIZE = 6
CLUSTER_UMAP_DIM = 10
CLUSTER_NAME_BATCH = 1            # сколько кластеров именуем за один chat-запрос
