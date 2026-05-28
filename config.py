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
JUDGE_CACHE_PATH = CACHE_DIR / "judge_binary_cache.parquet"
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
JUDGE_TOP_K = 5                  # сколько кандидатов хранить в матчах (для прозрачности),
                                 # бинарный судья смотрит только на top-1
JUDGE_REQ_TIMEOUT = 120          # таймаут одного chat-запроса (сек)
EMBED_REQ_TIMEOUT = 60

# ─── Выбор порога через бинарного судью на случайной выборке ─────────────
# Берём N случайных пар (real_query → top-1 scenario), судья даёт 1/0,
# далее ищем порог по score_rrf, максимизирующий F-beta (positive = oos).
JUDGE_SAMPLE_N = int(os.getenv("JUDGE_SAMPLE_N", "100"))
JUDGE_FBETA = float(os.getenv("JUDGE_FBETA", "0.5"))   # F0.5: precision(oos) важнее recall
                                                       # (на нашем 70/30 imbalance F1/F2 деградируют)
JUDGE_SAMPLE_SEED = int(os.getenv("JUDGE_SAMPLE_SEED", "42"))
# random — чистый случайный выбор; stratified — по децилям score_rrf, чтобы
# выборка содержала запросы и из «вероятно покрыт», и из «вероятно oos» хвостов.
JUDGE_SAMPLE_STRATEGY = os.getenv("JUDGE_SAMPLE_STRATEGY", "stratified").lower()
JUDGE_SAMPLE_BINS = int(os.getenv("JUDGE_SAMPLE_BINS", "10"))

# Запасной порог по score_embed (cosine), если LLM-судья выключен (--no-llm).
# Для GigaChat EmbeddingsGigaR на русском типичный диапазон [0.3, 0.85].
# 0.60 ≈ умеренно похоже (для калибровки берётся реальное значение из выборки).
FALLBACK_THRESHOLD_EMBED = float(os.getenv("FALLBACK_THRESHOLD_EMBED", "0.60"))

# ─── Rate-limiting GigaChat ──────────────────────────────────────────────
# GigaChat-2-Max в свободном тарифе режет нас по RPS на chat/completions.
# Все настройки переопределяются переменными окружения.
JUDGE_PARALLELISM = int(os.getenv("JUDGE_PARALLELISM", "1"))
JUDGE_MIN_INTERVAL_SEC = float(os.getenv("JUDGE_MIN_INTERVAL_SEC", "1.2"))
EMBED_MIN_INTERVAL_SEC = float(os.getenv("EMBED_MIN_INTERVAL_SEC", "0.0"))
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "8"))
RETRY_MAX_WAIT_SEC = float(os.getenv("RETRY_MAX_WAIT_SEC", "60"))
JUDGE_FLUSH_EVERY = int(os.getenv("JUDGE_FLUSH_EVERY", "25"))

# Retrieval-объединение
RRF_K = 60                        # параметр Reciprocal Rank Fusion

# Stage A
FUZZY_THRESHOLD = 92              # rapidfuzz partial_ratio для exact-варианта
BUTTON_N_QUESTION_RATIO = 0.80    # доля n_question>1 чтобы счесть текст «кнопкой»
BUTTON_MIN_COUNT = 5              # мин. частота, чтобы вообще считать button-кандидатом

# Фильтрация мусора
MIN_QUERY_LEN_CHARS = 3
MAX_QUERY_LEN_CHARS = 1000

# Кластеризация gap-ов
CLUSTER_MIN_SIZE = 6
CLUSTER_UMAP_DIM = 10
CLUSTER_NAME_BATCH = 1            # сколько кластеров именуем за один chat-запрос
