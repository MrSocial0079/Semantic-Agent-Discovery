"""
Semantic Agent Discovery — visualize.py
UMAP 2D scatter: color by cluster_label, red X for drift agents.
Saves fleet_map.png.  Run after main.py.
"""

import json
import os
import numpy as np
import matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap

try:
    import mplcursors
    HAS_MPLCURSORS = True
except ImportError:
    HAS_MPLCURSORS = False


def main():
    # ── Load final enriched corpus ────────────────────────────────────────────
    with open(os.path.join(HERE, "agents_final.json")) as f:
        agents = json.load(f)

    # ── Load pre-saved embeddings (evolved_vecs used for visual layout) ───────
    vecs_data  = np.load(os.path.join(HERE, "vectors.npy"), allow_pickle=True).item()
    evolved_vecs   = vecs_data["evolved"]    # shape (30, 384)
    original_vecs  = vecs_data["original"]   # shape (30, 384)
    # Use average of original + evolved for a richer 2-D layout
    blend_vecs = (original_vecs + evolved_vecs) / 2.0

    # ── UMAP projection → 2D ─────────────────────────────────────────────────
    reducer = umap.UMAP(
        n_neighbors=8,
        min_dist=0.3,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(blend_vecs)   # (30, 2)

    # ── Assign colour per cluster_label ──────────────────────────────────────
    labels        = [a["cluster_label"] for a in agents]
    unique_labels = sorted(set(labels))
    palette = [
        "#4C72B0", "#DD8452", "#55A868",
        "#C44E52", "#8172B2", "#937860",
    ]
    colour_map = {lbl: palette[i % len(palette)]
                  for i, lbl in enumerate(unique_labels)}

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))
    scatter_handles = []

    for agent, (x, y) in zip(agents, coords):
        lbl   = agent["cluster_label"]
        col   = colour_map[lbl]
        drift = agent.get("drift_flag", False)

        if drift:
            sc = ax.scatter(x, y, c="red", marker="x", s=120,
                            linewidths=2.5, zorder=5)
        else:
            sc = ax.scatter(x, y, c=col, marker="o", s=80,
                            edgecolors="white", linewidths=0.6, zorder=4)

    # mplcursors hover labels (agent names)
    all_dots = ax.collections
    if HAS_MPLCURSORS and all_dots:
        names = [a["name"] for a in agents]
        cursor = mplcursors.cursor(all_dots, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            idx = sel.index
            a   = agents[idx]
            sel.annotation.set_text(
                f"{a['name']}\n"
                f"cat: {a['category']}\n"
                f"drift: {a['drift_score']:.3f}"
            )

    # ── Legend: clusters ──────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=colour_map[lbl], label=lbl)
        for lbl in unique_labels
    ]
    legend_patches.append(
        plt.Line2D([0], [0], marker="x", color="red", linestyle="None",
                   markersize=9, markeredgewidth=2.5, label="Drift Agent")
    )
    ax.legend(
        handles=legend_patches,
        title="Cluster / Drift",
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
        framealpha=0.85,
    )

    # ── Annotate agent names (small font) ─────────────────────────────────────
    for agent, (x, y) in zip(agents, coords):
        ax.annotate(
            agent["name"],
            (x, y),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=5.5,
            color="#333333",
        )

    ax.set_title(
        "InvexsAI — Semantic Agent Fleet Map (UMAP 2D)\n"
        "Color = cluster label  |  ✕ = drift-flagged agent",
        fontsize=12,
        pad=14,
    )
    ax.set_xlabel("UMAP-1", fontsize=9)
    ax.set_ylabel("UMAP-2", fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(HERE, "fleet_map.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
