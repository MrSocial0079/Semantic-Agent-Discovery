"""
Semantic Agent Discovery — search.py
CLI semantic search: embed query → cosine similarity vs evolved_vecs → top-5 results.

Usage:
    python semantic/search.py "compliance monitoring"
    python semantic/search.py "infrastructure health"
    python semantic/search.py "risk exposure"
"""

import sys
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine as cosine_distance


def load_data(agents_path: str, vectors_path: str):
    with open(agents_path) as f:
        agents = json.load(f)
    vecs_data = np.load(vectors_path, allow_pickle=True).item()
    evolved_vecs = vecs_data["evolved"]    # shape (N, 384)
    return agents, evolved_vecs


def semantic_search(query: str, agents: list[dict], evolved_vecs: np.ndarray,
                    model: SentenceTransformer, top_k: int = 5) -> list[dict]:
    query_vec = model.encode([query], show_progress_bar=False)[0]

    scores = []
    for i, vec in enumerate(evolved_vecs):
        sim = float(1.0 - cosine_distance(query_vec, vec))
        scores.append((sim, i))

    scores.sort(key=lambda x: x[0], reverse=True)
    results = []
    for rank, (sim, idx) in enumerate(scores[:top_k], start=1):
        a = agents[idx]
        results.append({
            "rank":          rank,
            "name":          a["name"],
            "category":      a["category"],
            "cluster_label": a.get("cluster_label", "—"),
            "drift_flag":    a.get("drift_flag", False),
            "drift_score":   a.get("drift_score", None),
            "similarity":    round(sim, 4),
            "evolved_desc":  a.get("evolved_desc", ""),
        })
    return results


def print_results(query: str, results: list[dict]) -> None:
    print(f"\n{'='*68}")
    print(f"Query: \"{query}\"")
    print(f"{'='*68}")
    for r in results:
        drift_tag = " [DRIFT]" if r["drift_flag"] else ""
        print(
            f"  #{r['rank']}  {r['name']:<35}  sim={r['similarity']:.4f}"
            f"{drift_tag}"
        )
        print(f"       category={r['category']:<20} cluster={r['cluster_label']}")
        print(f"       {r['evolved_desc'][:90]}{'...' if len(r['evolved_desc']) > 90 else ''}")
        print()
    print(f"{'='*68}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python semantic/search.py \"<query>\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    print("Loading model ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    agents, evolved_vecs = load_data(
        os.path.join(HERE, "agents_final.json"),
        os.path.join(HERE, "vectors.npy"),
    )

    results = semantic_search(query, agents, evolved_vecs, model, top_k=5)
    print_results(query, results)


if __name__ == "__main__":
    main()
