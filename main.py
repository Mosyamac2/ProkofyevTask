"""Полный пайплайн A→B→C + подбор порога через LLM-судью + сборка артефактов.

Запуск:  python main.py
Параметры (CLI):
    --no-llm    : пропустить LLM-судью, использовать config.FALLBACK_THRESHOLD_RRF
    --no-cluster: пропустить кластеризацию gap-ов
    --limit N   : ограничить количество уникальных запросов (для smoke-проверки)

Логика бакетизации (двухклассовая):
    score_embed (cosine) >= threshold → covered
    score_embed (cosine) <  threshold → oos
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

import cluster_gaps
import config
import data_loader
import dialog_context
import embed_cache
import matcher
import report
import reverse_analysis
import segments
import threshold_select

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
    scen_emb = embed_cache.embed_with_cache(scen["embed_text"].tolist(), desc="embed-scen")
    log.info("Scenario embeddings: %s", scen_emb.shape)

    log.info("Embedding real queries (cached) …")
    real_emb = embed_cache.embed_with_cache(uniq["question_norm"].tolist(), desc="embed-real")
    log.info("Real embeddings: %s", real_emb.shape)

    # ── Stage B + C retrieval (RRF top-K) ───────────────────────────────
    log.info("Retrieving top-K candidates (BM25 + cosine RRF) …")
    candidates = matcher.retrieve_topk(
        uniq["question_norm"].tolist(), real_emb, scen, scen_emb,
        k=config.JUDGE_TOP_K,
    )

    # ── Подготовка top-1 для дальнейших шагов ──────────────────────────
    scen_by_id = scen.set_index("scenario_id")
    top1_view: list[dict | None] = []
    for cands in candidates:
        if not cands:
            top1_view.append(None)
            continue
        c = cands[0]
        s = scen_by_id.loc[c.scenario_id]
        top1_view.append({
            "scenario_id": c.scenario_id,
            "scenario_query": s["scenario_query"],
            "category": s["category"],
            "action_type": s["action_type"],
            "agent": s["agent"],
            "score_rrf": float(c.rrf),
            "score_embed": float(c.score_embed) if c.score_embed is not None else None,
            "score_lex": float(c.score_lex) if c.score_lex is not None else None,
        })

    # ── Подбор порога через LLM-судью на N=100 ─────────────────────────
    if no_llm:
        log.info("--no-llm: using FALLBACK_THRESHOLD_EMBED=%.4f", config.FALLBACK_THRESHOLD_EMBED)
        thr_result = threshold_select.ThresholdResult(
            threshold=config.FALLBACK_THRESHOLD_EMBED,
            fbeta=0.0, beta=config.JUDGE_FBETA,
            precision_oos=0.0, recall_oos=0.0,
            n_sample=0, n_pos_oos=0, n_neg_oos=0, n_errors=0,
        )
    else:
        log.info("Calibrating threshold via LLM judge on N=%d random pairs (F%g, positive=oos) …",
                 config.JUDGE_SAMPLE_N, config.JUDGE_FBETA)
        thr_result = threshold_select.calibrate(uniq, top1_view)
        log.info(
            "Selected threshold (cosine)=%.5f | F%.1f=%.3f | precision(oos)=%.3f | recall(oos)=%.3f"
            " | n_oos=%d / n_cov=%d / errors=%d (sample=%d)",
            thr_result.threshold, thr_result.beta, thr_result.fbeta,
            thr_result.precision_oos, thr_result.recall_oos,
            thr_result.n_pos_oos, thr_result.n_neg_oos, thr_result.n_errors, thr_result.n_sample,
        )
    threshold = thr_result.threshold

    # ── Финальная таблица матчей: бинарная бакетизация ─────────────────
    rows = []
    for i, row in uniq.iterrows():
        c1 = top1_view[i]
        if row["input_kind"] == "exact_scenario":
            sid = row["matched_scenario_id"]
            s = scen_by_id.loc[sid] if sid in scen_by_id.index else None
            row_out = dict(row)
            row_out.update({
                "match_stage": "exact",
                "final_verdict": "covered",
                "best_scenario_id": sid,
                "best_scenario_text": s["scenario_query"] if s is not None else None,
                "best_category": s["category"] if s is not None else None,
                "best_agent": s["agent"] if s is not None else None,
                "best_score_rrf": float(c1["score_rrf"]) if c1 else None,
                "best_score_embed": float(c1["score_embed"]) if c1 and c1["score_embed"] is not None else None,
                "best_score_lex": float(c1["score_lex"]) if c1 and c1["score_lex"] is not None else None,
            })
        elif c1 is None:
            row_out = dict(row)
            row_out.update({
                "match_stage": "no-candidate",
                "final_verdict": "oos",
                "best_scenario_id": None,
                "best_scenario_text": None,
                "best_category": None,
                "best_agent": None,
                "best_score_rrf": None,
                "best_score_embed": None,
                "best_score_lex": None,
            })
        else:
            # Если score_embed по какой-то причине None — считаем oos
            embed = c1.get("score_embed")
            verdict = "covered" if (embed is not None and embed >= threshold) else "oos"
            row_out = dict(row)
            row_out.update({
                "match_stage": "retrieval",
                "final_verdict": verdict,
                "best_scenario_id": c1["scenario_id"],
                "best_scenario_text": c1["scenario_query"],
                "best_category": c1["category"],
                "best_agent": c1["agent"],
                "best_score_rrf": c1["score_rrf"],
                "best_score_embed": c1["score_embed"],
                "best_score_lex": c1["score_lex"],
            })

        cands = candidates[i]
        for k_alt in (2, 3):
            if len(cands) >= k_alt:
                alt = cands[k_alt - 1]
                alt_s = scen_by_id.loc[alt.scenario_id] if alt.scenario_id in scen_by_id.index else None
                row_out[f"top{k_alt}_scenario_text"] = alt_s["scenario_query"] if alt_s is not None else None
                row_out[f"top{k_alt}_score_rrf"] = float(alt.rrf)
            else:
                row_out[f"top{k_alt}_scenario_text"] = None
                row_out[f"top{k_alt}_score_rrf"] = None

        rows.append(row_out)

    matches = pd.DataFrame(rows)

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
    p_hist = report.save_score_histogram(matches, threshold=threshold)
    p_audit = report.save_threshold_audit(thr_result) if not no_llm else None
    p_cover = report.cover_letter(matches, scen_util, seg, clusters_df, thr_result)

    log.info("Done in %.1fs. Artifacts:", time.time() - t0)
    for p in (p_matches, p_gaps, p_zomb, p_clust, p_seg, p_hist, p_audit, p_cover):
        if p:
            log.info("   %s", p)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true",
                    help="не вызывать LLM-судью; использовать FALLBACK_THRESHOLD_RRF и не именовать кластеры")
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
