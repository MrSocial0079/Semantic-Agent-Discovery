# Semantic Agent Discovery — Phase 2

NLP pipeline that reads a fleet of 30 synthetic AI agents and produces
tagged, clustered, and drift-monitored views of agent behaviour.

---

## Pipeline Overview

| Stage | Script | Output |
|-------|--------|--------|
| 1 — TF-IDF Keywords | `main.py` | `keywords` field per agent |
| 2 — Evolved Description | `main.py` | `evolved_desc` field per agent |
| 3 — Semantic Embedding | `main.py` | `original_vec`, `evolved_vec` (384-dim), `vectors.npy` |
| 4 — K-Means Clustering | `main.py` | `cluster_id`, `cluster_label` per agent |
| 5 — Drift Detection | `main.py` | `drift_score`, `drift_flag` per agent |
| 6 — Evaluation Report | `main.py` | Console report (accuracy / ARI / drift recall) |
| Visualisation | `visualize.py` | `fleet_map.png` (UMAP 2D scatter) |
| Semantic Search | `search.py` | Top-5 agents matching a natural-language query |

---

## Installation

```bash
pip install -r semantic/requirements.txt
```

Dependencies: `sentence-transformers`, `scikit-learn`, `scipy`,
`umap-learn`, `matplotlib`, `numpy`, `mplcursors`.

---

## Running the Pipeline

> All commands are run from the **project root** (`invexsai_yash 3/`).

### 1. Generate the corpus (one-time)
```bash
python semantic/generate_corpus.py
# → semantic/agents.json  (30 agents, do not overwrite)
```

### 2. Run the full NLP pipeline
```bash
python semantic/main.py
# → semantic/agents_enriched.json  (stages 1-2)
# → semantic/vectors.npy           (stage 3)
# → semantic/agents_final.json     (stages 4-6, fully enriched)
```

Expected evaluation report:
```
SEMANTIC AGENT DISCOVERY — EVALUATION REPORT
=============================================
Tagging Accuracy:    87% (target >= 80%)
Clustering ARI:      0.83 (target >= 0.80)
Drift Recall:        5/5 (target 5/5 — 100%)
Pipeline Runtime:    ~3 seconds
=============================================
```

### 3. Visualise the fleet
```bash
python semantic/visualize.py
# → semantic/fleet_map.png
```

Opens a UMAP 2D scatter plot:
- Each dot = one agent, coloured by cluster label
- Red `✕` markers = drift-flagged agents
- Hover over any dot (interactive mode) to see name, category, drift score

### 4. Semantic search
```bash
python semantic/search.py "compliance monitoring"
python semantic/search.py "infrastructure health"
python semantic/search.py "risk exposure"
```

Encodes the query with `all-MiniLM-L6-v2`, ranks all agents by cosine
similarity against their `evolved_vec`, and prints the top 5 with
cluster labels and drift flags.

---

## Output Files

| File | Description |
|------|-------------|
| `semantic/agents.json` | Raw corpus — 30 agents (do not modify) |
| `semantic/agents_enriched.json` | After stages 1-2 (keywords + evolved_desc) |
| `semantic/agents_final.json` | Fully enriched (all 6 stages) |
| `semantic/vectors.npy` | NumPy dict — `original`, `evolved`, `names` |
| `semantic/fleet_map.png` | UMAP visualisation |

---

## Agent Schema (agents_final.json)

```json
{
  "name": "risk-monitor-01",
  "category": "Risk Monitoring",
  "original_desc": "...",
  "tools": ["..."],
  "activity_logs": "...",
  "evolved_desc": "Primary activity involves ...",
  "keywords": ["risk_calc", "threshold_check", "..."],
  "original_vec": [0.123, ...],
  "evolved_vec": [0.456, ...],
  "cluster_id": 3,
  "cluster_label": "Risk Monitoring",
  "drift_score": 0.5041,
  "drift_flag": true
}
```

---

## Design Notes

- **Drift detection** compares the cosine similarity between `original_vec`
  (embedding of the original description) and `evolved_vec` (embedding of
  the TF-IDF-derived evolved description from activity logs). Agents with
  `drift_score < 0.75` are flagged. The five deliberately seeded drift
  agents (`drift-agent-01` through `drift-agent-05`) have scores below
  0.25, well below the threshold.

- **Clustering** is performed on PCA-reduced `original_vecs` (natural
  language, 384-dim → 20-dim) rather than the formulaic `evolved_vecs`,
  which yields ARI 0.83 vs. the ground-truth category labels.

- **Embedding model**: `all-MiniLM-L6-v2` (384-dim) from
  [sentence-transformers](https://www.sbert.net/).
