"""Reverse-анализ: для каждого сценария считаем, сколько реальных запросов к нему привязалось.

Главный артефакт — `zombie_scenarios`: сценарии, к которым ни один реальный запрос
не привязался ни как `exact_scenario`, ни как `covered` через LLM-судью.
"""
from __future__ import annotations

import pandas as pd


def utilization(matches: pd.DataFrame, scen_df: pd.DataFrame) -> pd.DataFrame:
    """matches — итоговая таблица per-query с колонкой best_scenario_id.

    Возвращает per-scenario DataFrame:
        scenario_id, agent, category, scenario_query, action_type, status,
        n_matches_unique, n_matches_events, n_covered_unique, n_covered_events
    """
    used = matches.dropna(subset=["best_scenario_id"]).copy()
    used["is_covered"] = used["final_verdict"] == "covered"

    grp = used.groupby("best_scenario_id").agg(
        n_matches_unique=("query_id", "nunique"),
        n_matches_events=("count", "sum"),
        n_covered_unique=("is_covered", "sum"),
        n_covered_events=("count", lambda s: int(s[used.loc[s.index, "is_covered"]].sum())),
    )

    out = scen_df.merge(grp, left_on="scenario_id", right_index=True, how="left")
    for c in ("n_matches_unique", "n_matches_events", "n_covered_unique", "n_covered_events"):
        out[c] = out[c].fillna(0).astype(int)
    out["is_zombie"] = (out["n_matches_unique"] == 0)
    return out.sort_values(
        ["is_zombie", "n_covered_events", "n_matches_events"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
