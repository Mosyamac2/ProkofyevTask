# Handoff — ProkofyevTask «Покрытие сценариев HR vs Пульс»

## Что это

Pipeline, который для каждого реального запроса сотрудника в логах системы «Пульс»
(`orch_logs_to_2026_05_27.xlsx`, 76 939 строк, ~15 225 уникальных) находит ближайший
сценарий из HR-каталога B2E (`Сценарии B2E.xlsx`, 1518 уникальных), решает covered/oos
и собирает артефакты для HR-аналитика.

## Текущее состояние

- **Master**: коммит `e3df4ea` запушен на `github.com/Mosyamac2/ProkofyevTask`.
- **venv**: `/home/mosyamac/ProkofyevTask/.venv` (Python 3.12, все зависимости из
  `requirements.txt` установлены).
- **Финальный прогон**: пайплайн отработал полностью за 6 мин 13 с
  (`out/run_final.log`).
- **Финальный артефакт для заказчика**:
  `pulse_coverage_report_2026-05-29.zip` (5.4 МБ, 8 файлов).

## Архитектура (для нового разработчика за 30 секунд)

1. **`data_loader.py`** — читает оба xlsx, нормализует, фильтрует.
2. **`dialog_context.py`** — детектит UI-button реплики (не свободный текст).
3. **`embed_cache.py`** + **`gigachat_client.py`** — GigaChat embeddings с дисковым
   кешем и throttle от 429.
4. **`matcher.py`** — retrieval: BM25 + cosine, объединение через RRF, top-K.
5. **`llm_judge.py`** — single-pair бинарный судья (1 = покрывает, 0 = нет), кеш на
   диске. Используется только для калибровки выборки в 100 пар.
6. **`threshold_select.py`** — стратифицированная выборка 100 пар, прогон судьи,
   подбор порога по cosine максимизацией F-beta (positive=oos).
7. **`main.py`** — оркестратор: грузим → embed → retrieve → calibrate threshold → бакетизация
   → reverse utilization → clustering → segments → report.
8. **`cluster_gaps.py`** — UMAP+HDBSCAN на oos-эмбеддингах, темы именует LLM.
9. **`segments.py`**, **`reverse_analysis.py`**, **`report.py`** — финальная сборка.

## Финальный результат

- **Порог** cosine = **0.7162** (F0.5-оптимум, `JUDGE_FBETA=0.5`)
- **Coverage**: 52.9% уникальных / 48.7% событий
- **Precision(oos)** на выборке 100: **89.4%**, recall(oos) = 57.5%
- **320 зомби-сценариев** из 1518 (21%)
- **231 кластер тем gap-ов**, top-1 — «Самопрезентация» (22 957 событий — запросы вида
  «расскажи о себе», каталог это не покрывает)

## Как запустить

```bash
cd /home/mosyamac/ProkofyevTask
.venv/bin/python main.py            # полный прогон (~6-13 мин с кешем)
.venv/bin/python main.py --no-llm   # без LLM-судьи и без именования тем (~3 мин)
```

Конфиг через переменные окружения (см. `config.py`):
- `JUDGE_FBETA=0.5` — приоритет precision на oos (default)
- `JUDGE_PARALLELISM=1` — серийные вызовы GigaChat (защита от 429)
- `JUDGE_MIN_INTERVAL_SEC=1.2` — пауза между chat-вызовами
- `JUDGE_SAMPLE_STRATEGY=stratified` — sampling 100 пар по децилям cosine

## Ключевые решения и почему

- **Почему cosine, а не RRF**: RRF — ранг-based и дискретный (всего ~373 уникальных
  значений на 14 734 запросов). Cosine — непрерывный, тысячи градаций → пригоден
  как одномерная шкала для threshold.
- **Почему F0.5, а не F1/F2**: класс oos в выборке доминирует (73/26). F1 и F2
  деградируют в «всё помечаем oos», precision=majority_freq, recall=1.0.
  F0.5 штрафует за false alarm → выбирает балансированный порог.
- **Почему сняли Platt scaling**: разметка через `lt.choisen_agent`
  (noisy supervisor по агенту) оказалась нерепрезентативной — оркестратор сам по себе
  шумит, и Platt давал неинтерпретируемую вероятность.
- **Stage D переосмыслен**: вместо ~15k LLM-вызовов (по одному на grey-zone запрос)
  теперь 100 вызовов для калибровки порога. Все остальные запросы бакетятся
  детерминированно по cosine.

## Открытые вопросы и долги

1. **`.env` в истории git** (из initial commit) — `GIGACHAT_AUTH_KEY` утёк.
   Чтобы зачистить: `git filter-repo --path .env --invert-paths`. И ротировать ключ.
2. **`orch_logs_to_2026_05_27.xlsx` (11.6 МБ внутренних данных Сбера)** тоже в репо.
   Если репо публичный — закрыть его или удалить файл из истории.
3. **GitHub PAT был в plaintext в чате** (значение удалено из этого документа,
   осталось в истории диалога). Отозвать в https://github.com/settings/tokens.
4. **Recall(oos) = 0.575** — почти половина настоящих gap-ов «прошла» как covered.
   Если HR это критично, можно перейти на трёхклассовую систему (covered / uncertain
   / oos) с двумя порогами — pipeline это поддержит минимальной правкой `main.py`.
5. **Кластеризация даёт 231 тему** — много для ручного обзора. Возможно стоит
   агрегировать по топ-10/20 крупнейших и остальное складывать в «прочее».
6. **Размер калибровочной выборки (100)** — на нижней границе. Confidence на
   precision/recall = ± 8-10 п.п. Если бюджет позволяет, увеличить до 300-500.

## Где что лежит

```
/home/mosyamac/ProkofyevTask/
├── *.py                      # код пайплайна (под git)
├── .env                      # ключи GigaChat (в git ИСТОРИИ — проблема)
├── .venv/                    # venv (gitignored)
├── cache/
│   ├── embeddings.parquet    # эмбеддинги 1518 сценариев + 15225 запросов
│   └── judge_binary_cache.parquet  # 100 LLM-разметок
├── out/
│   ├── cover_letter.md       # сводка для HR
│   ├── matches.xlsx          # per-query вердикты
│   ├── gaps.xlsx             # только oos
│   ├── zombie_scenarios.xlsx # сценарии без матчей
│   ├── gap_themes.xlsx       # 231 кластер тем
│   ├── coverage_segments.xlsx
│   ├── score_distribution.png
│   ├── threshold_audit.xlsx  # 100 размеченных пар + sweep
│   ├── _cover_letter_f2_*.md # бэкапы предыдущих итераций
│   └── run_*.log
└── pulse_coverage_report_2026-05-29.zip  # deliverable для заказчика
```

## Полезные команды

```bash
# Пересчитать порог из кеша без новых LLM-вызовов
.venv/bin/python -c "
import threshold_select, ... # см. оригинальный скрипт из истории чата
"

# Глянуть распределение score_embed
.venv/bin/python -c "
import pandas as pd
m = pd.read_excel('out/matches.xlsx')
print(m['best_score_embed'].describe())
"

# Пересобрать архив для заказчика
DATE=$(date +%Y-%m-%d)
mkdir -p "pulse_coverage_report_${DATE}"
cp out/{cover_letter.md,matches.xlsx,gaps.xlsx,zombie_scenarios.xlsx,gap_themes.xlsx,coverage_segments.xlsx,score_distribution.png,threshold_audit.xlsx} "pulse_coverage_report_${DATE}/"
zip -r "pulse_coverage_report_${DATE}.zip" "pulse_coverage_report_${DATE}"
```
