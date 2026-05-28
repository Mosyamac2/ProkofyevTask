"""Калибровка: оценка качества retrieval через `chosen_agent` как noisy supervisor.

Подход (см. PLAN_v2 §6):
- Каждый реальный запрос → top-K сценариев из retrieval.
- Положительные пары: запрос → сценарий ТОГО ЖЕ агента, что и `chosen_agent`.
- Отрицательные: запрос → случайный сценарий ДРУГОГО агента.
- Обучаем Platt scaling: P(match | cosine) = σ(a·cosine + b).
- После калибровки рассчитываем agent-consistency rate по бакетам вердикта Stage D.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

log = logging.getLogger(__name__)


@dataclass
class PlattCalibrator:
    a: float
    b: float

    def predict_proba(self, cosine: np.ndarray) -> np.ndarray:
        x = self.a * cosine + self.b
        return 1.0 / (1.0 + np.exp(-x))


# Маппинг от названия агента в каталоге сценариев (B2E) к chosen_agent оркестратора.
# Карту специально делаем явной, чтобы не было ложных «согласованностей» по случайному совпадению строк.
AGENT_MAP_CATALOG_TO_ORCH: dict[str, set[str]] = {
    "Агент Аналитик": {"agent-analytics", "agent-py-reports", "agent-reportsemployee", "agent-team"},
    "Подбор": {"agent-recsourcing", "agent-recruiter", "agent-app-perftracker-issue-creator"},
    "Развитие (Тесты и опросы)": {"agent-assessment"},
    "Развитие (inTra)": {"agent-app-learning-helper"},
    "Профиль Сотрудника": {"agent-smart-profile"},
    "Профиль": {"agent-smart-profile"},
    "Льготы и компенсации": {"app-benefits-agent"},
    "Льготы": {"app-benefits-agent"},
    "Карьера": {"agent-da-hr-orchestrator", "agent-self-drm"},
    "Отсутствия": {"absences:vacations"},
    "Отпуска": {"absences:vacations"},
    "Менторинг": {"agent-p2p"},
    "Коучинг": {"agent-p2p"},
    "Взаимное развитие": {"agent-p2p"},
    "Напоминания": {"ai-reminder-agent"},
    "Календарь": {"spine-ui-dates:events"},
}


def map_catalog_agent(name: str) -> set[str]:
    if not name:
        return set()
    if name in AGENT_MAP_CATALOG_TO_ORCH:
        return AGENT_MAP_CATALOG_TO_ORCH[name]
    # Поддержка случаев типа «Подбор / Внутренние»
    for k, v in AGENT_MAP_CATALOG_TO_ORCH.items():
        if name.startswith(k) or k in name:
            return v
    return set()


def build_supervision_pairs(
    real_df: pd.DataFrame,
    real_embeds: np.ndarray,
    scen_df: pd.DataFrame,
    scen_embeds: np.ndarray,
    max_pairs: int = 20_000,
    rng_seed: int = 7,
) -> pd.DataFrame:
    """Собираем positive/negative пары на основе chosen_agent.

    Positive: (запрос, случайный сценарий агента, который ОБСЛУЖИЛ этот запрос).
    Negative: (запрос, случайный сценарий ДРУГОГО агента).
    Возвращает DataFrame с колонками [cosine, label].
    """
    rng = np.random.default_rng(rng_seed)
    scen_agents = scen_df["agent"].tolist()
    scen_idx_by_orch_agent: dict[str, list[int]] = {}
    for i, a in enumerate(scen_agents):
        for orch in map_catalog_agent(a):
            scen_idx_by_orch_agent.setdefault(orch, []).append(i)

    rows = []
    for i, row in enumerate(real_df.itertuples()):
        orch = row.top_agent
        if not orch or orch not in scen_idx_by_orch_agent:
            continue
        pos_pool = scen_idx_by_orch_agent[orch]
        neg_pool = [j for orch_other, lst in scen_idx_by_orch_agent.items()
                    for j in lst if orch_other != orch]
        if not pos_pool or not neg_pool:
            continue
        pos = rng.choice(pos_pool)
        neg = rng.choice(neg_pool)
        rows.append((float(real_embeds[i] @ scen_embeds[pos]), 1))
        rows.append((float(real_embeds[i] @ scen_embeds[neg]), 0))
        if len(rows) >= max_pairs:
            break
    return pd.DataFrame(rows, columns=["cosine", "label"])


def fit_platt(pairs: pd.DataFrame) -> PlattCalibrator:
    if len(pairs) < 50 or pairs["label"].nunique() < 2:
        # fallback: identity mapping
        return PlattCalibrator(a=8.0, b=-4.0)
    X = pairs["cosine"].to_numpy().reshape(-1, 1)
    y = pairs["label"].to_numpy()
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X, y)
    return PlattCalibrator(a=float(clf.coef_[0, 0]), b=float(clf.intercept_[0]))


def agent_consistency_rate(
    real_df: pd.DataFrame,
    best_scenario_agent: pd.Series,
) -> dict:
    """Доля случаев, где предсказанный сценарий принадлежит тому же агенту,
    что и `chosen_agent` оркестратора. Считаем суммарно и по бакетам.
    """
    df = real_df.copy()
    df["best_agent"] = best_scenario_agent.values
    df["orch_match"] = df.apply(
        lambda r: bool(r["top_agent"]) and r["top_agent"] in map_catalog_agent(r["best_agent"]),
        axis=1,
    )
    overall = float(df["orch_match"].mean())
    by_bucket = df.groupby("final_verdict")["orch_match"].mean().to_dict() if "final_verdict" in df.columns else {}
    return {"overall": overall, "by_bucket": by_bucket}
