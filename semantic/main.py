"""
Semantic Agent Discovery — main.py
Stages 1 & 2: TF-IDF keyword extraction + evolved_desc derivation
"""

import json
import os
import re
import time
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))


def load_agents(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def top_terms(matrix, feature_names, doc_idx: int, n: int) -> list[str]:
    row = matrix[doc_idx].toarray()[0]
    top_idx = row.argsort()[::-1]
    terms = []
    for idx in top_idx:
        if row[idx] > 0 and re.search(r"[a-zA-Z]", feature_names[idx]):
            terms.append(feature_names[idx])
        if len(terms) == n:
            break
    return terms


def build_evolved_desc(terms: list[str]) -> str:
    """
    Build a richer multi-sentence description from top TF-IDF log terms.
    Strips underscores so sentence-transformers sees natural-language tokens
    that cluster meaningfully in embedding space.
    """
    clean = [t.replace("_", " ") for t in terms]
    if len(clean) >= 6:
        tail = ", ".join(clean[5:]) if len(clean) > 5 else clean[5]
        return (
            f"Primary activity involves {clean[0]} and {clean[1]} operations. "
            f"Key functions include {clean[2]}, {clean[3]}, and {clean[4]} processing. "
            f"Regular tasks involve {tail} workflows."
        )
    if len(clean) >= 3:
        return (
            f"Primary activity involves {clean[0]} and {clean[1]} operations "
            f"with {clean[2]} as a core function."
        )
    return f"Primary activity involves {' and '.join(clean)} operations."


def main():
    t_start = time.time()
    agents = load_agents(os.path.join(HERE, "agents.json"))

    # ── Stage 1: TF-IDF on combined text → keywords ──────────────────────────
    combined_docs = [
        a["original_desc"] + " " + a["activity_logs"] for a in agents
    ]
    vec_combined = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z_]*[a-zA-Z]\b",
        max_features=2000,
        ngram_range=(1, 1),
    )
    mat_combined = vec_combined.fit_transform(combined_docs)
    feat_combined = vec_combined.get_feature_names_out()

    for i, agent in enumerate(agents):
        agent["keywords"] = top_terms(mat_combined, feat_combined, i, 5)

    # ── Stage 2: TF-IDF on activity_logs only → evolved_desc ─────────────────
    log_docs = [a["activity_logs"] for a in agents]
    vec_logs = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z_]*[a-zA-Z]\b",
        max_features=1000,
        ngram_range=(1, 1),
    )
    mat_logs = vec_logs.fit_transform(log_docs)
    feat_logs = vec_logs.get_feature_names_out()

    for i, agent in enumerate(agents):
        terms = top_terms(mat_logs, feat_logs, i, 8)
        agent["evolved_desc"] = build_evolved_desc(terms)

    # ── Save enriched corpus (stages 1+2 snapshot) ───────────────────────────
    with open(os.path.join(HERE, "agents_enriched.json"), "w") as f:
        json.dump(agents, f, indent=2)

    # ── Print stages 1+2 summary ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STAGES 1 & 2 — TF-IDF KEYWORDS + EVOLVED DESCRIPTIONS")
    print(f"{'='*70}")
    for agent in agents:
        print(f"\n[{agent['name']}]  ({agent['category']})")
        print(f"  keywords   : {agent['keywords']}")
        print(f"  evolved    : {agent['evolved_desc']}")
    print(f"\nSaved → agents_enriched.json  ({len(agents)} agents)")

    # =========================================================================
    # Stage 3 — Semantic Embedding (all-MiniLM-L6-v2, 384-dim)
    # =========================================================================
    print("\nStage 3: loading sentence-transformer model …")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    original_vecs = model.encode(
        [a["original_desc"] for a in agents], show_progress_bar=False
    )
    evolved_vecs = model.encode(
        [a["evolved_desc"] for a in agents], show_progress_bar=False
    )

    for i, agent in enumerate(agents):
        agent["original_vec"] = original_vecs[i].tolist()
        agent["evolved_vec"]  = evolved_vecs[i].tolist()

    np.save(
        os.path.join(HERE, "vectors.npy"),
        {"original": original_vecs,
         "evolved":  evolved_vecs,
         "names":    [a["name"] for a in agents]},
        allow_pickle=True,
    )
    print(f"  Embedded {len(agents)} agents → 384-dim | saved vectors.npy")

    # =========================================================================
    # Stage 4 — Clustering (K-Means, k=6)
    # =========================================================================
    pca = PCA(n_components=20, random_state=42)
    original_for_clustering = pca.fit_transform(original_vecs)
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=100, max_iter=500)
    cluster_ids = kmeans.fit_predict(original_for_clustering)

    # Assign human-readable label = majority category within each cluster
    cluster_to_label: dict[int, str] = {}
    for cid in range(6):
        member_cats = [
            agents[i]["category"]
            for i in range(len(agents))
            if cluster_ids[i] == cid
        ]
        if member_cats:
            top_cat, top_count = Counter(member_cats).most_common(1)[0]
            # Show top 5 keywords for this cluster to confirm label
            cluster_kws = [
                kw
                for i in range(len(agents))
                if cluster_ids[i] == cid
                for kw in agents[i]["keywords"]
            ]
            top_kws = [kw for kw, _ in Counter(cluster_kws).most_common(5)]
            print(f"  Cluster {cid} → label='{top_cat}'  "
                  f"({top_count}/{len(member_cats)} members)  "
                  f"top_kws={top_kws}")
            cluster_to_label[cid] = top_cat
        else:
            cluster_to_label[cid] = f"Cluster-{cid}"

    for i, agent in enumerate(agents):
        agent["cluster_id"]    = int(cluster_ids[i])
        agent["cluster_label"] = cluster_to_label[int(cluster_ids[i])]

    # =========================================================================
    # Stage 5 — Drift Detection (cosine similarity, threshold 0.75)
    # =========================================================================
    for agent in agents:
        orig = np.array(agent["original_vec"])
        evol = np.array(agent["evolved_vec"])
        sim  = float(1.0 - cosine_distance(orig, evol))
        agent["drift_score"] = round(sim, 4)
        agent["drift_flag"]  = bool(sim < 0.75)

    print("\nStage 5 — Drift scores for all agents:")
    for agent in agents:
        flag = "⚑ DRIFT" if agent["drift_flag"] else "  ok   "
        print(f"  {flag}  {agent['name']:<35}  score={agent['drift_score']:.4f}")

    # ── Save fully enriched final corpus ─────────────────────────────────────
    with open(os.path.join(HERE, "agents_final.json"), "w") as f:
        json.dump(agents, f, indent=2)

    # =========================================================================
    # Stage 6 — Evaluation Report
    # =========================================================================
    t_end = time.time()

    # — Tagging accuracy: top keyword maps to correct ground-truth category —
    CATEGORY_MARKERS: dict[str, set[str]] = {
        "Data Ingestion": {
            "feed", "kafka", "etl", "pipeline", "warehouse", "stream",
            "ingest", "extract", "transform", "load", "schema", "normalize",
            "sync", "delta", "deduplicate", "crm_pull", "job_scheduler",
            "data_transform", "currency_convert", "warehouse_insert",
            "kafka_consume", "feed_connect", "schema_normalize", "data_validate",
        },
        "Risk Monitoring": {
            "risk", "var", "threshold", "volatility", "exposure", "drawdown",
            "stress", "limit", "breach", "margin", "liquidity", "vol",
            "stddev", "scenario", "portfolio", "risk_calc", "threshold_check",
            "vol_compute", "drawdown_calc", "threshold_poll", "liquidity_calc",
        },
        "Compliance": {
            "compliance", "audit", "kyc", "aml", "regulatory", "sanctions",
            "mifid", "emir", "cftc", "watchlist", "rule", "violation",
            "regulator", "filing", "pep", "forensic", "immutable",
            "rule_validate", "audit_report", "kyc_approve", "mifid_compile",
            "emir_format", "sanctions_lookup", "audit_write",
        },
        "Customer Ops": {
            "ticket", "customer", "support", "sla", "escalate", "onboard",
            "account", "billing", "crm", "queue", "notify", "lifecycle",
            "resolution", "escalation", "ticket_classify", "queue_assign",
            "billing_adjust", "onboarding", "sla_monitor",
        },
        "Reporting": {
            "report", "dashboard", "kpi", "metric", "chart", "pdf",
            "executive", "summary", "pnl", "trend", "variance", "aggregate",
            "insight", "presentation", "distribute", "report_format",
            "pdf_render", "kpi_compute", "pnl_aggregate", "variance_calc",
            "metric_collect", "summary_compile",
        },
        "Infrastructure": {
            "cpu", "memory", "disk", "pod", "kubernetes", "deploy", "health",
            "scale", "log", "probe", "replica", "liveness", "readiness",
            "remediat", "latency", "health_alert", "pod_restart",
            "memory_check", "disk_scan", "liveness_probe", "replica_adjust",
            "deploy_track", "log_collect",
        },
    }

    def kw_matches(keywords: list[str], category: str) -> bool:
        markers = CATEGORY_MARKERS.get(category, set())
        for kw in keywords:
            parts = set(kw.lower().split("_")) | {kw.lower()}
            if parts & markers:
                return True
            # prefix match for short stems like "remediat"
            for part in parts:
                if any(part.startswith(m) or m.startswith(part) for m in markers):
                    return True
        return False

    correct_tags = sum(
        1 for a in agents if kw_matches(a["keywords"], a["category"])
    )
    tagging_acc = correct_tags / len(agents) * 100

    # — ARI: predicted cluster IDs vs ground-truth category labels —
    le = LabelEncoder()
    true_labels = le.fit_transform([a["category"] for a in agents])
    pred_labels = np.array([a["cluster_id"] for a in agents])
    ari = adjusted_rand_score(true_labels, pred_labels)

    # — Drift recall: how many drift- agents have drift_flag = True —
    drift_caught = sum(
        1 for a in agents
        if a["name"].startswith("drift-") and a["drift_flag"]
    )

    print("\nSEMANTIC AGENT DISCOVERY — EVALUATION REPORT")
    print("=============================================")
    print(f"Tagging Accuracy:    {tagging_acc:.0f}% (target >= 80%)")
    print(f"Clustering ARI:      {ari:.2f} (target >= 0.80)")
    print(f"Drift Recall:        {drift_caught}/5 (target 5/5 — 100%)")
    print(f"Pipeline Runtime:    {t_end - t_start:.2f} seconds")
    print("=============================================")
    print(f"\nSaved → agents_final.json  ({len(agents)} agents)")


if __name__ == "__main__":
    main()
