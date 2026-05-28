"""Загрузка обоих xlsx и подготовка таблиц."""
from __future__ import annotations

import pandas as pd

import config
from text_norm import is_junk, normalize, truncate


SCENARIO_COLS = ["idx", "agent", "category", "scenario_query", "action_type", "status"]


def load_scenarios() -> pd.DataFrame:
    """Возвращает DataFrame со сценариями: ffill, нормализация, дедуп.

    Колонки результата:
        scenario_id : str  (стабильный ключ scen_{i})
        agent       : str
        category    : str
        scenario_query     : str  (оригинал)
        scenario_query_norm: str
        action_type : str (нормализованный, lower)
        status      : str
        embed_text  : str  (что подаётся в эмбеддинг: «{category}. {scenario_query}»)
    """
    df = pd.read_excel(config.DATA_SCENARIOS, header=None)
    df.columns = SCENARIO_COLS[: df.shape[1]]
    df[["agent", "category", "action_type", "status"]] = df[
        ["agent", "category", "action_type", "status"]
    ].ffill()
    df = df.dropna(subset=["scenario_query"]).reset_index(drop=True)

    df["scenario_query"] = df["scenario_query"].astype(str)
    df["scenario_query_norm"] = df["scenario_query"].map(normalize)
    df["category"] = df["category"].fillna("").astype(str).str.strip()
    df["category_norm"] = df["category"].map(normalize)
    df["agent"] = df["agent"].fillna("").astype(str).str.strip()
    df["action_type"] = df["action_type"].fillna("").astype(str).str.strip().str.lower()
    df["status"] = df["status"].fillna("").astype(str).str.strip()

    df["embed_text"] = (df["category"] + ". " + df["scenario_query"]).str.strip(". ").str.strip()

    df = df[df["scenario_query_norm"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["embed_text"]).reset_index(drop=True)
    df["scenario_id"] = [f"scen_{i:05d}" for i in range(len(df))]
    keep = ["scenario_id", "agent", "category", "category_norm",
            "scenario_query", "scenario_query_norm",
            "action_type", "status", "embed_text"]
    return df[keep]


def load_raw_logs() -> pd.DataFrame:
    """Сырой DataFrame логов с минимальной обработкой (нужен для dialog_context.py)."""
    df = pd.read_excel(config.DATA_LOGS)
    df = df.rename(columns={
        "lt.dialog_id": "dialog_id",
        "lt.n_question": "n_question",
        "lt.created_at": "created_at",
        "lt.user_id": "user_id",
        "pd.full_name": "full_name",
        "pd.block": "block",
        "last_user_question": "user_question",
        "last_agent_responce": "agent_response",
        "lt.choisen_agent": "chosen_agent",
    })
    df["user_question"] = df["user_question"].astype(str).map(
        lambda s: truncate(s, config.MAX_QUERY_LEN_CHARS)
    )
    df["question_norm"] = df["user_question"].map(normalize)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


def aggregate_unique_queries(logs: pd.DataFrame) -> pd.DataFrame:
    """Дедуплицируем по question_norm и собираем агрегаты на уникальный текст.

    Колонки результата:
        question_norm, sample_text, count, n_users, top_agent, top_block,
        first_seen, last_seen, n_first_turn, share_first_turn
    """
    junk_mask = logs["question_norm"].map(lambda s: is_junk(s, config.MIN_QUERY_LEN_CHARS))
    clean = logs.loc[~junk_mask].copy()

    def _mode(s: pd.Series):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) else None

    agg = (
        clean.groupby("question_norm", sort=False)
        .agg(
            sample_text=("user_question", "first"),
            count=("user_question", "size"),
            n_users=("user_id", "nunique"),
            top_agent=("chosen_agent", _mode),
            top_block=("block", _mode),
            first_seen=("created_at", "min"),
            last_seen=("created_at", "max"),
            n_first_turn=("n_question", lambda s: int((s == 1).sum())),
        )
        .reset_index()
    )
    agg["share_first_turn"] = agg["n_first_turn"] / agg["count"].clip(lower=1)
    agg = agg.sort_values("count", ascending=False).reset_index(drop=True)
    agg["query_id"] = [f"q_{i:06d}" for i in range(len(agg))]
    return agg
