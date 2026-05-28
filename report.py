"""Сборка Excel-артефактов и cover letter."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    "confidence",
    "best_scenario_id",
    "best_scenario_text",
    "best_category",
    "best_agent",
    "agent_consistency",
    "best_score_embed",
    "best_score_lex",
    "best_prob_match",
    "judge_reason",
    "top2_scenario_text",
    "top2_score_embed",
    "top3_scenario_text",
    "top3_score_embed",
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
        "best_score_embed", "best_prob_match", "judge_reason",
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


def save_score_histogram(matches: pd.DataFrame, path: Optional[Path] = None) -> Path:
    path = path or (config.OUT_DIR / "score_distribution.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    s = matches["best_score_embed"].dropna()
    axes[0].hist(s, bins=40, color="#3a6ea5", edgecolor="white")
    axes[0].set_title("Cosine similarity (top-1) по уникальным запросам")
    axes[0].set_xlabel("cosine")
    axes[0].set_ylabel("кол-во")

    if "best_prob_match" in matches.columns:
        p = matches["best_prob_match"].dropna()
        axes[1].hist(p, bins=40, color="#aa6c3a", edgecolor="white")
        axes[1].axvline(config.OOS_PROB_THRESHOLD, color="red", ls="--", lw=1, label="oos cutoff")
        axes[1].axvline(config.COVERED_PROB_THRESHOLD, color="green", ls="--", lw=1, label="covered cutoff")
        axes[1].set_title("Калиброванная P(match)")
        axes[1].set_xlabel("P(match)")
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def cover_letter(
    matches: pd.DataFrame,
    scen_util: pd.DataFrame,
    segments: dict[str, pd.DataFrame],
    clusters_df: pd.DataFrame,
    calibration_info: dict,
    path: Optional[Path] = None,
) -> Path:
    path = path or (config.OUT_DIR / "cover_letter.md")

    n_unique = len(matches)
    n_events = int(matches["count"].sum())
    covered = matches[matches["final_verdict"] == "covered"]
    partial = matches[matches["final_verdict"] == "partial"]
    oos = matches[matches["final_verdict"] == "oos"]

    cov_u = len(covered) / max(n_unique, 1)
    cov_e = int(covered["count"].sum()) / max(n_events, 1)
    par_u = len(partial) / max(n_unique, 1)
    par_e = int(partial["count"].sum()) / max(n_events, 1)
    oos_u = len(oos) / max(n_unique, 1)
    oos_e = int(oos["count"].sum()) / max(n_events, 1)

    n_scen = len(scen_util)
    n_zombies = int(scen_util["is_zombie"].sum())

    top_gaps = oos.sort_values("count", ascending=False).head(15)[
        ["sample_text", "count"]
    ]
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
    lines.append(f"- **Частично** (partial): {par_u:.1%} уникальных / {par_e:.1%} событий")
    lines.append(f"- **Не покрыто** (oos): **{oos_u:.1%}** уникальных / {oos_e:.1%} событий")
    lines.append(f"- Каталог сценариев: {n_scen} штук, из них **зомби** (без матчей): **{n_zombies}** "
                 f"({n_zombies/max(n_scen,1):.1%})")
    lines.append("")

    lines.append("## Калибровка\n")
    lines.append(f"- Platt scaling: P(match) = σ({calibration_info.get('a', 0):.2f}·cosine + {calibration_info.get('b', 0):.2f})")
    lines.append(f"- Agent-consistency rate (топ-1 сценарий принадлежит тому же агенту, что и оркестратор): "
                 f"**{calibration_info.get('consistency_overall', 0):.1%}**")
    by_bucket = calibration_info.get("consistency_by_bucket", {})
    if by_bucket:
        lines.append("- По бакетам вердикта Stage D:")
        for b, v in by_bucket.items():
            lines.append(f"    - `{b}`: {v:.1%}")
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
        "Каскад A→B→C→D (см. `PLAN_v2.md`): exact/button-match → BM25 + char-n-gram → "
        "эмбеддинги GigaChat (`{model}`) → LLM-судья (`{chat}`) для финального вердикта. "
        "Кросс-чек качества — через колонку `lt.choisen_agent` оркестратора как noisy supervisor.".format(
            model=config.GIGACHAT_EMBED_MODEL, chat=config.GIGACHAT_CHAT_MODEL,
        )
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
