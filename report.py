"""Сборка Excel-артефактов и cover letter."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config


MATCHES_COLS = [
    "query_id",
    "sample_text",
    "question_norm",
    "count",
    "n_users",
    "top_block",
    "top_agent",
    "input_kind",
    "match_stage",
    "final_verdict",
    "best_scenario_id",
    "best_scenario_text",
    "best_category",
    "best_agent",
    "best_score_rrf",
    "best_score_embed",
    "best_score_lex",
    "top2_scenario_text",
    "top2_score_rrf",
    "top3_scenario_text",
    "top3_score_rrf",
    "first_seen",
    "last_seen",
]


def save_matches(matches: pd.DataFrame, path: Optional[Path] = None) -> Path:
    path = path or (config.OUT_DIR / "matches.xlsx")
    cols = [c for c in MATCHES_COLS if c in matches.columns]
    matches[cols].to_excel(path, index=False)
    return path


def save_gaps(matches: pd.DataFrame, path: Optional[Path] = None) -> Path:
    path = path or (config.OUT_DIR / "gaps.xlsx")
    gaps = matches[matches["final_verdict"] == "oos"].copy()
    gaps = gaps.sort_values("count", ascending=False)
    cols = [
        "query_id", "sample_text", "count", "n_users",
        "top_block", "top_agent", "input_kind",
        "best_scenario_text", "best_category", "best_agent",
        "best_score_embed", "best_score_rrf",
        "gap_cluster_id", "gap_theme",
    ]
    cols = [c for c in cols if c in gaps.columns]
    gaps[cols].to_excel(path, index=False)
    return path


def save_zombies(scen_util: pd.DataFrame, path: Optional[Path] = None) -> Path:
    path = path or (config.OUT_DIR / "zombie_scenarios.xlsx")
    cols = [
        "scenario_id", "agent", "category", "scenario_query",
        "action_type", "status",
        "n_matches_unique", "n_matches_events",
        "n_covered_unique", "n_covered_events",
        "is_zombie",
    ]
    cols = [c for c in cols if c in scen_util.columns]
    scen_util[cols].to_excel(path, index=False)
    return path


def save_clusters(clusters_df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    path = path or (config.OUT_DIR / "gap_themes.xlsx")
    df = clusters_df.copy()
    if "all_examples" in df.columns:
        df["all_examples"] = df["all_examples"].map(lambda xs: " | ".join(xs))
    df.to_excel(path, index=False)
    return path


def save_segments(segments: dict[str, pd.DataFrame], path: Optional[Path] = None) -> Path:
    path = path or (config.OUT_DIR / "coverage_segments.xlsx")
    with pd.ExcelWriter(path) as w:
        for name, df in segments.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
    return path


def save_score_histogram(
    matches: pd.DataFrame,
    threshold: float | None = None,
    path: Optional[Path] = None,
) -> Path:
    path = path or (config.OUT_DIR / "score_distribution.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))

    s = matches["best_score_embed"].dropna()
    ax.hist(s, bins=60, color="#3a6ea5", edgecolor="white")
    if threshold is not None:
        ax.axvline(threshold, color="red", ls="--", lw=1.5, label=f"threshold = {threshold:.4f}")
        ax.legend()
    ax.set_title("Распределение cosine top-1 (score_embed) по уникальным запросам")
    ax.set_xlabel("cosine")
    ax.set_ylabel("кол-во уникальных запросов")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_threshold_audit(thr_result, path: Optional[Path] = None) -> Path:
    """Сохраняет 100 пар, на которых судья сделал разметку (для аудита HR)."""
    path = path or (config.OUT_DIR / "threshold_audit.xlsx")
    with pd.ExcelWriter(path) as w:
        if not thr_result.audit.empty:
            cols = [
                "sample_text", "scenario_query", "category", "action_type",
                "score_embed", "score_rrf", "label", "reason",
            ]
            cols = [c for c in cols if c in thr_result.audit.columns]
            thr_result.audit[cols].to_excel(w, sheet_name="sample", index=False)
        if not thr_result.score_grid.empty:
            thr_result.score_grid.to_excel(w, sheet_name="threshold_sweep", index=False)
    return path


def cover_letter(
    matches: pd.DataFrame,
    scen_util: pd.DataFrame,
    segments: dict[str, pd.DataFrame],
    clusters_df: pd.DataFrame,
    thr_result,
    path: Optional[Path] = None,
) -> Path:
    path = path or (config.OUT_DIR / "cover_letter.md")

    n_unique = len(matches)
    n_events = int(matches["count"].sum())
    covered = matches[matches["final_verdict"] == "covered"]
    oos = matches[matches["final_verdict"] == "oos"]

    cov_u = len(covered) / max(n_unique, 1)
    cov_e = int(covered["count"].sum()) / max(n_events, 1)
    oos_u = len(oos) / max(n_unique, 1)
    oos_e = int(oos["count"].sum()) / max(n_events, 1)

    n_scen = len(scen_util)
    n_zombies = int(scen_util["is_zombie"].sum())

    top_gaps = oos.sort_values("count", ascending=False).head(15)[["sample_text", "count"]]
    top_zombies = scen_util[scen_util["is_zombie"]].head(15)[
        ["agent", "category", "scenario_query", "status"]
    ]
    top_themes = clusters_df.head(10)[["theme", "n_unique", "n_events", "top_examples"]] if len(clusters_df) else None

    by_block = segments.get("by_block")

    lines = []
    lines.append("# Покрытие сценариев HR vs реальные запросы — сводка\n")
    lines.append("## TL;DR\n")
    lines.append(f"- Уникальных запросов в логах (после очистки): **{n_unique:,}**")
    lines.append(f"- Всего реплик пользователей: **{n_events:,}**")
    lines.append(f"- **Покрыто** (covered): {cov_u:.1%} уникальных / **{cov_e:.1%}** событий")
    lines.append(f"- **Не покрыто** (oos): **{oos_u:.1%}** уникальных / {oos_e:.1%} событий")
    lines.append(f"- Каталог сценариев: {n_scen} штук, из них **зомби** (без матчей): **{n_zombies}** "
                 f"({n_zombies / max(n_scen, 1):.1%})")
    lines.append("")

    lines.append("## Подбор порога\n")
    lines.append(f"- Score: cosine эмбеддингов GigaChat `{config.GIGACHAT_EMBED_MODEL}` "
                 f"(непрерывный, top-1 кандидат отбирается через RRF-фьюжн с BM25).")
    lines.append(f"- Бакетизация: `cosine >= {thr_result.threshold:.4f}` → covered, иначе oos.")
    if thr_result.n_sample > 0:
        lines.append(f"- Калибровочная выборка: **{thr_result.n_sample}** запросов "
                     f"({thr_result.sampling_strategy} sampling), размечены "
                     f"`{config.GIGACHAT_CHAT_MODEL}` бинарно (1/0).")
        lines.append(f"- В выборке: oos={thr_result.n_pos_oos}, covered={thr_result.n_neg_oos}, "
                     f"ошибок разметки={thr_result.n_errors}.")
        lines.append(f"- Выбранный порог по **F{thr_result.beta:g}** (positive=oos): "
                     f"cosine >= **{thr_result.threshold:.4f}** → covered; "
                     f"F={thr_result.fbeta:.3f}, precision(oos)={thr_result.precision_oos:.3f}, "
                     f"recall(oos)={thr_result.recall_oos:.3f}.")
        if thr_result.threshold_f1 is not None:
            lines.append(f"- Для сравнения, **F1**-оптимальный порог: "
                         f"cosine >= **{thr_result.threshold_f1:.4f}**; "
                         f"F1={thr_result.f1:.3f}, precision(oos)={thr_result.precision_oos_f1:.3f}, "
                         f"recall(oos)={thr_result.recall_oos_f1:.3f}.")
    else:
        lines.append(f"- LLM-судья был выключен (--no-llm). Использован запасной порог "
                     f"cosine {config.FALLBACK_THRESHOLD_EMBED:.4f} из config — метрики недоступны.")
    lines.append("")

    if by_block is not None and len(by_block):
        lines.append("## Coverage по блокам\n")
        lines.append("| Блок | Уник. запросов | События | Coverage (uniq) | Coverage (events) |")
        lines.append("|------|---------------:|--------:|----------------:|------------------:|")
        for _, r in by_block.sort_values("n_events", ascending=False).head(10).iterrows():
            blk = r.get("top_block") or r.get("level_0") or "—"
            lines.append(
                f"| {blk} | {int(r['n_unique'])} | {int(r['n_events'])} | "
                f"{r['coverage_unique']:.1%} | {r['coverage_events']:.1%} |"
            )
        lines.append("")

    lines.append("## Топ-15 самых частых непокрытых запросов\n")
    lines.append("| Запрос | Частота |")
    lines.append("|--------|--------:|")
    for _, r in top_gaps.iterrows():
        txt = str(r["sample_text"]).replace("|", "\\|")[:120]
        lines.append(f"| {txt} | {int(r['count'])} |")
    lines.append("")

    if top_themes is not None and len(top_themes):
        lines.append("## Топ-10 тем непокрытых запросов (кластеризация)\n")
        lines.append("| Тема | Уник. | События | Примеры |")
        lines.append("|------|------:|--------:|---------|")
        for _, r in top_themes.iterrows():
            ex = str(r["top_examples"]).replace("|", "\\|")[:200]
            theme = str(r["theme"]).replace("|", "\\|")[:80]
            lines.append(f"| {theme} | {int(r['n_unique'])} | {int(r['n_events'])} | {ex} |")
        lines.append("")

    lines.append("## Топ-15 зомби-сценариев (придуманы HR, не сработали ни разу)\n")
    lines.append("| Агент | Категория | Сценарий | Статус |")
    lines.append("|-------|-----------|----------|--------|")
    for _, r in top_zombies.iterrows():
        sc = str(r["scenario_query"]).replace("|", "\\|")[:120]
        lines.append(f"| {r['agent']} | {r['category']} | {sc} | {r['status']} |")
    lines.append("")

    lines.append("## Методология\n")
    lines.append(
        "1. **Stage A (exact/button-match)**: реплики, точно совпадающие со сценарием каталога или "
        "помеченные как UI-кнопки, считаются покрытыми сразу.\n"
        "2. **Stage B/C (BM25 + cosine RRF)**: для каждого свободного запроса находим top-K сценариев "
        "через RRF-фьюжн. Top-1 — лучший кандидат.\n"
        "3. Финальная мера похожести этой пары — `cosine` (score_embed) top-1: "
        "непрерывная, тысячи градаций.\n"
        f"4. **Подбор порога**: {config.JUDGE_SAMPLE_STRATEGY} выборка {config.JUDGE_SAMPLE_N} пар → "
        f"бинарная разметка `{config.GIGACHAT_CHAT_MODEL}` → порог по cosine, максимизирующий "
        f"F{config.JUDGE_FBETA:g} с positive=oos.\n"
        "5. Запросы с cosine выше порога — **covered**, ниже — **oos**.\n"
        "6. **Reverse-анализ**: сценарии без единого матча — зомби-сценарии.\n"
        "7. **Кластеризация gap-ов**: UMAP+HDBSCAN на эмбеддингах oos-запросов, темы именует LLM."
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
