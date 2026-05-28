"""Сегментный анализ покрытия: по `pd.block`, по времени, по длине запроса, по `n_question`."""
from __future__ import annotations

import pandas as pd


def _coverage_row(group: pd.DataFrame) -> pd.Series:
    n_unique = len(group)
    n_events = int(group["count"].sum())
    n_covered_unique = int((group["final_verdict"] == "covered").sum())
    n_covered_events = int(
        group.loc[group["final_verdict"] == "covered", "count"].sum()
    )
    return pd.Series(
        {
            "n_unique": n_unique,
            "n_events": n_events,
            "coverage_unique": n_covered_unique / max(n_unique, 1),
            "coverage_events": n_covered_events / max(n_events, 1),
            "n_covered_unique": n_covered_unique,
            "n_covered_events": n_covered_events,
        }
    )


def by_block(matches: pd.DataFrame) -> pd.DataFrame:
    return matches.groupby("top_block", dropna=False).apply(_coverage_row).reset_index()


def by_month(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()
    df["month"] = pd.to_datetime(df["last_seen"], errors="coerce").dt.to_period("M").astype(str)
    return df.groupby("month").apply(_coverage_row).reset_index()


def by_length_bucket(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()
    df["len_bucket"] = pd.cut(
        df["question_norm"].str.len(),
        bins=[0, 10, 20, 40, 80, 10_000],
        labels=["≤10", "11-20", "21-40", "41-80", "81+"],
    )
    return df.groupby("len_bucket").apply(_coverage_row).reset_index()


def by_input_kind(matches: pd.DataFrame) -> pd.DataFrame:
    return matches.groupby("input_kind").apply(_coverage_row).reset_index()


def collect(matches: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "by_block": by_block(matches),
        "by_month": by_month(matches),
        "by_length": by_length_bucket(matches),
        "by_input_kind": by_input_kind(matches),
    }
