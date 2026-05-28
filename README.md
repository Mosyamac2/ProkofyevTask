# Coverage check: HR-сценарии vs реальные запросы

Реализация плана из [`PLAN_v2.md`](PLAN_v2.md): каскад A→B→C→D для проверки, насколько каталог сценариев B2E покрывает реальные запросы пользователей в системе «Пульс».

## Быстрый старт

```bash
# 1. Зависимости
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Сертификат МинЦифры (если он не установлен системно)
#    можно оставить GIGACHAT_VERIFY_SSL=false для разовой аналитики.

# 3. Ключ GigaChat
cp .env.example .env
# отредактируйте .env: вставьте GIGACHAT_AUTH_KEY

# 4. Запуск всего пайплайна
python main.py
```

Артефакты появятся в `out/`:
- `matches.xlsx` — основное: для каждого уникального запроса лучший сценарий, вердикт, top-K альтернатив;
- `gaps.xlsx` — непокрытые запросы, отсортированные по частоте;
- `gap_themes.xlsx` — кластеры непокрытых запросов;
- `zombie_scenarios.xlsx` — сценарии без единого матча (forward утилизация);
- `coverage_segments.xlsx` — coverage по `pd.block` / времени / длине / `n_question`;
- `cover_letter.md` — краткая сводка для HR;
- `score_distribution.png` — гистограмма вероятностей.

Все промежуточные данные (эмбеддинги, ответы LLM-судьи) кешируются в `cache/` и переиспользуются при повторном запуске.

## Структура модулей

| Модуль                   | Назначение                                                 |
| ------------------------ | ---------------------------------------------------------- |
| `config.py`              | Все настройки + пути                                       |
| `text_norm.py`           | Нормализация текста                                        |
| `data_loader.py`         | Загрузка обоих xlsx, ffill, дедуп, фильтрация              |
| `dialog_context.py`      | Детекция button-реплик, отделение свободного ввода         |
| `gigachat_client.py`     | OAuth + embeddings + chat (с retry)                        |
| `embed_cache.py`         | Parquet-кеш эмбеддингов                                    |
| `lexical.py`             | BM25 + char n-gram TF-IDF                                  |
| `matcher.py`             | Cosine, RRF top-K, Stage A bypass                          |
| `llm_judge.py`           | Stage D — LLM cross-encoder с JSON-выходом и кешем         |
| `calibration.py`         | Platt scaling на agent-consistency, метрики калибровки     |
| `reverse_analysis.py`    | Утилизация сценариев, зомби-каталог                        |
| `cluster_gaps.py`        | UMAP + HDBSCAN + LLM-именование тем                        |
| `segments.py`            | Coverage по `pd.block` / времени / длине / `n_question`    |
| `report.py`              | Запись Excel, гистограммы, cover letter                    |
| `main.py`                | Каскад A→B→C→D                                             |
