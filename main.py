"""Полный каскад A→B→C→D + сборка артефактов.

Запуск:  python main.py
Параметры (env / CLI):
    --no-llm    : пропустить Stage D и кластерное именование (только эмбеддинги/BM25)
    --no-cluster: пропустить кластеризацию gap-ов
    --limit N   : ограничить количество уникальных запросов (для smoke-проверки)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
import calibration
import cluster_gaps
import data_loader
import dialog_context
import embed_cache
import llm_judge
import matcher
import report
import reverse_analysis
import segments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s :: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("main")


def run(no_llm: bool = False, no_cluster: bool = False, limit: int | None = None):
    t0 = time.time()

    # ── Load + clean ───────────────────────────────────────────────────
    log.info("Loading scenarios …")
    scen = data_loader.load_scenarios()
    log.info("Scenarios: %d unique", len(scen))

    log.info("Loading logs …")
    logs = data_loader.load_raw_logs()
    log.info("Raw logs: %d rows", len(logs))

    uniq = data_loader.aggregate_unique_queries(logs)
    log.info("Unique clean queries: %d (events=%d)", len(uniq), int(uniq["count"].sum()))

    uniq = dialog_context.classify_input_kind(uniq, logs, scen)
    log.info("Input kinds: %s", uniq["input_kind"].value_counts().to_dict())

    if limit:
        uniq = uniq.head(limit).reset_index(drop=True)
        log.info("Truncated to %d queries (--limit)", len(uniq))

    # ── Embeddings ─────────────────────────────────────────────────────
    log.info("Embedding scenarios (cached) …")
    scen_emb = embed_cache.embed_with_cache(
        scen["embed_text"].tolist(), desc="embed-scen"
    )
    log.info("Scenario embeddings: %s", scen_emb.shape)

    real_texts_for_embed = uniq["question_norm"].tolist()
    log.info("Embedding real queries (cached) …")
    real_emb = embed_cache.embed_with_cache(real_texts_for_embed, desc="embed-real")
    log.info("Real embeddings: %s", real_emb.shape)

    # ── Stage B + C retrieval (RRF top-K) ───────────────────────────────
    log.info("Retrieving top-K candidates (BM25 + cosine RRF) …")
    candidates = matcher.retrieve_topk(
        uniq["question_norm"].tolist(),
        real_emb,
        scen,
        scen_emb,
        k=config.JUDGE_TOP_K,
    )

    # ── Calibration via agent-consistency ──────────────────────────────
    log.info("Building supervision pairs and fitting Platt …")
    pairs = calibration.build_supervision_pairs(uniq, real_emb, scen, scen_emb)
    log.info("Supervision pairs: %d (positives=%d)", len(pairs), int(pairs["label"].sum()) if len(pairs) else 0)
    platt = calibration.fit_platt(pairs)
    log.info("Platt: a=%.3f b=%.3f", platt.a, platt.b)

    # ── Stage D: LLM judge ─────────────────────────────────────────────
    scen_by_id = scen.set_index("scenario_id")
    judge_inputs_q: list[str] = []
    judge_inputs_c: list[list[dict]] = []
    needs_judge_mask = []
    for i, row in uniq.iterrows():
        if row["input_kind"] == "exact_scenario":
            needs_judge_mask.append(False)
            continue
        needs_judge_mask.append(True)
        cands = candidates[i]
        scen_rows = []
        for c in cands:
            s = scen_by_id.loc[c.scenario_id]
            scen_rows.append({
                "scenario_id": c.scenario_id,
                "scenario_query": s["scenario_query"],
                "category": s["category"],
                "action_type": s["action_type"],
            })
        judge_inputs_q.append(row["sample_text"])
        judge_inputs_c.append(scen_rows)

    if no_llm:
        judge_results = [llm_judge.JudgeResult("partial", None, 0.5, "skip-llm flag")] * len(judge_inputs_q)
    else:
        log.info("Stage D — LLM judge on %d queries", len(judge_inputs_q))
        judge_results = llm_judge.judge_all(judge_inputs_q, judge_inputs_c)

    # ── Assemble final matches table ───────────────────────────────────
    rows = []
    judge_iter = iter(judge_results)
    for i, row in uniq.iterrows():
        cands = candidates[i]
        best_c = cands[0] if cands else None

        if row["input_kind"] == "exact_scenario":
            sid = row["matched_scenario_id"]
            s = scen_by_id.loc[sid] if sid in scen_by_id.index else None
            row_out = dict(row)
            row_out.update({
                "match_stage": "exact",
                "final_verdict": "covered",
                "confidence": 1.0,
                "best_scenario_id": sid,
                "best_scenario_text": s["scenario_query"] if s is not None else None,
                "best_category": s["category"] if s is not None else None,
                "best_agent": s["agent"] if s is not None else None,
                "best_score_embed": float(best_c.score_embed) if best_c and best_c.score_embed is not None else None,
                "best_score_lex": float(best_c.score_lex) if best_c and best_c.score_lex is not None else None,
                "judge_reason": "exact match с каталогом",
            })
        else:
            jr = next(judge_iter)
            chosen_idx = jr.chosen
            if jr.verdict == "covered" and chosen_idx and 1 <= chosen_idx <= len(cands):
                sel = cands[chosen_idx - 1]
            elif best_c is not None:
                sel = best_c
            else:
                sel = None
            sid = sel.scenario_id if sel else None
            s = scen_by_id.loc[sid] if sid in scen_by_id.index else None
            row_out = dict(row)
            row_out.update({
                "match_stage": "llm" if not no_llm else "retrieval",
                "final_verdict": jr.verdict,
                "confidence": jr.confidence,
                "best_scenario_id": sid,
                "best_scenario_text": s["scenario_query"] if s is not None else None,
                "best_category": s["category"] if s is not None else None,
                "best_agent": s["agent"] if s is not None else None,
                "best_score_embed": float(sel.score_embed) if sel and sel.score_embed is not None else None,
                "best_score_lex": float(sel.score_lex) if sel and sel.score_lex is not None else None,
                "judge_reason": jr.reason,
            })

        for k_alt in (2, 3):
            if best_c is not None and len(cands) >= k_alt:
                alt = cands[k_alt - 1]
                alt_s = scen_by_id.loc[alt.scenario_id] if alt.scenario_id in scen_by_id.index else None
                row_out[f"top{k_alt}_scenario_text"] = alt_s["scenario_query"] if alt_s is not None else None
                row_out[f"top{k_alt}_score_embed"] = float(alt.score_embed) if alt.score_embed is not None else None
            else:
                row_out[f"top{k_alt}_scenario_text"] = None
                row_out[f"top{k_alt}_score_embed"] = None

        rows.append(row_out)

    matches = pd.DataFrame(rows)

    # Калиброванная вероятность для best top-1
    if "best_score_embed" in matches.columns:
        cos = matches["best_score_embed"].fillna(0.0).to_numpy()
        matches["best_prob_match"] = platt.predict_proba(cos)

    # Agent consistency (по факту согласован ли best_agent с top_agent оркестратора)
    def _consistent(row):
        ta = row.get("top_agent")
        ba = row.get("best_agent")
        if not ta or not ba:
            return False
        return ta in calibration.map_catalog_agent(ba)
    matches["agent_consistency"] = matches.apply(_consistent, axis=1)

    cal_info = calibration.agent_consistency_rate(matches, matches["best_agent"])
    cal_info.update({"a": platt.a, "b": platt.b})
    cal_info["consistency_overall"] = cal_info.pop("overall")
    cal_info["consistency_by_bucket"] = cal_info.pop("by_bucket")

    # ── Reverse utilization ────────────────────────────────────────────
    log.info("Reverse utilization …")
    scen_util = reverse_analysis.utilization(matches, scen)
    log.info("Zombie scenarios: %d / %d", int(scen_util["is_zombie"].sum()), len(scen_util))

    # ── Gap clustering ─────────────────────────────────────────────────
    if not no_cluster:
        oos_mask = matches["final_verdict"] == "oos"
        if oos_mask.sum() >= config.CLUSTER_MIN_SIZE * 2:
            oos_idx = np.where(oos_mask.to_numpy())[0]
            oos_emb = real_emb[oos_idx]
            cids = cluster_gaps.cluster(oos_emb)
            log.info("Clusters in gaps: %d (noise=%d)",
                     len(set(cids)) - (1 if -1 in cids else 0), int((cids == -1).sum()))
            clusters_df = cluster_gaps.summarize_clusters(
                matches.iloc[oos_idx].reset_index(drop=True),
                cids,
                name_topics=not no_llm,
            )
            theme_lookup = {int(c): t for c, t in zip(clusters_df["cluster_id"], clusters_df["theme"])}
            matches["gap_cluster_id"] = None
            matches["gap_theme"] = None
            for off, cid in enumerate(cids):
                if cid < 0:
                    continue
                gi = oos_idx[off]
                matches.at[gi, "gap_cluster_id"] = int(cid)
                matches.at[gi, "gap_theme"] = theme_lookup.get(int(cid), "")
        else:
            log.info("Too few gaps for clustering")
            clusters_df = pd.DataFrame(columns=["cluster_id", "n_unique", "n_events", "top_examples", "theme"])
    else:
        clusters_df = pd.DataFrame(columns=["cluster_id", "n_unique", "n_events", "top_examples", "theme"])

    # ── Segments ───────────────────────────────────────────────────────
    log.info("Segment analysis …")
    seg = segments.collect(matches)

    # ── Reporting ──────────────────────────────────────────────────────
    log.info("Saving artifacts …")
    p_matches = report.save_matches(matches)
    p_gaps = report.save_gaps(matches)
    p_zomb = report.save_zombies(scen_util)
    p_clust = report.save_clusters(clusters_df) if len(clusters_df) else None
    p_seg = report.save_segments(seg)
    p_hist = report.save_score_histogram(matches)
    p_cover = report.cover_letter(matches, scen_util, seg, clusters_df, cal_info)

    log.info("Done in %.1fs. Artifacts:", time.time() - t0)
    for p in (p_matches, p_gaps, p_zomb, p_clust, p_seg, p_hist, p_cover):
        if p:
            log.info("   %s", p)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="не вызывать LLM-судью и не именовать кластеры")
    ap.add_argument("--no-cluster", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(no_llm=args.no_llm, no_cluster=args.no_cluster, limit=args.limit)
    finally:
        import gigachat_client
        gigachat_client.close()
