"""Детекция button-реплик и обогащение запросов признаком input_kind.

Сигналы button-реплики:
1. Текст никогда (или почти никогда) не появляется как первая реплика диалога —
   `share_first_turn` мал, а `count` большой. Кнопки UI кликаются внутри диалога.
2. Текст следует после однотипного ответа агента (`agent_response` повторяется).

Сигналы свободного ввода:
1. Текст часто открывает диалог (share_first_turn → 1).
2. После запроса агент даёт разнообразные ответы.

Также особняком — точное совпадение нормализованного текста с одним из
сценарных запросов: такие запросы помечаем `input_kind='exact_scenario'` —
это «золотой» сигнал покрытия независимо от того, кнопка это или ввод.
"""
from __future__ import annotations

import pandas as pd

import config


def _agent_response_entropy(logs: pd.DataFrame) -> pd.Series:
    """Для каждого question_norm — кол-во уникальных ответов агента, нормированное на частоту."""
    grp = (
        logs.dropna(subset=["agent_response"])
        .groupby("question_norm")
        .agg(n_distinct_resp=("agent_response", "nunique"), n_resp=("agent_response", "size"))
    )
    grp["resp_diversity"] = (grp["n_distinct_resp"] / grp["n_resp"].clip(lower=1)).clip(0, 1)
    return grp["resp_diversity"]


def classify_input_kind(
    unique_q: pd.DataFrame,
    logs: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> pd.DataFrame:
    """Добавляет в unique_q колонки:
        input_kind        : 'exact_scenario' | 'button' | 'free'
        button_score      : float [0..1] — насколько уверенно текст похож на кнопку
        matched_scenario_id: str | None — для exact_scenario
        matched_scenario_text: str | None
    """
    df = unique_q.copy()

    scen_lookup = scenarios.set_index("scenario_query_norm")[["scenario_id", "scenario_query"]]
    scen_lookup = scen_lookup[~scen_lookup.index.duplicated(keep="first")]

    df["matched_scenario_id"] = df["question_norm"].map(scen_lookup["scenario_id"])
    df["matched_scenario_text"] = df["question_norm"].map(scen_lookup["scenario_query"])

    resp_div = _agent_response_entropy(logs)
    df["resp_diversity"] = df["question_norm"].map(resp_div)

    share_score = 1.0 - df["share_first_turn"].clip(0, 1)
    df["button_score"] = share_score.where(df["count"] >= config.BUTTON_MIN_COUNT, 0.0)

    is_button = (df["button_score"] >= 0.7) & (df["count"] >= config.BUTTON_MIN_COUNT)

    df["input_kind"] = "free"
    df.loc[is_button, "input_kind"] = "button"
    df.loc[df["matched_scenario_id"].notna(), "input_kind"] = "exact_scenario"

    return df
