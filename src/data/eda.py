"""
PHASE 2 — Étape 3 : analyse exploratoire (EDA).

Usage :
    python -m src.data.eda

Produit 3 figures dans reports/figures/ :
  - distribution_classes.png : nombre de commentaires par sentiment
  - longueur_textes.png      : distribution de la longueur (mots) par classe
  - top_mots.png             : 15 mots les plus discriminants par classe

Et un résumé chiffré dans reports/eda_summary.json.
"""

import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")                     # backend sans écran (serveur/CI)
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import (CLEAN_CSV, FIGURES_DIR, LABELS, REPORTS_DIR,
                              ensure_dirs)
from src.utils.logger import get_logger

log = get_logger("data.eda")

COLORS = {-1: "#d1495b", 0: "#8d99ae", 1: "#2a9d8f"}

# Mots vides anglais : ils sont fréquents partout, donc non discriminants.
STOPWORDS = set("""
the and for that with you not this have are was but they his her she him from
all can out has had its our who will what when which been would there their
one more about your some them than then него into just like only also were
i me my we us it is of to in on a an be do does did so no if or as at by
""".split())


def run() -> dict:
    ensure_dirs()
    df = pd.read_csv(CLEAN_CSV)
    df["n_words"] = df["text"].astype(str).str.split().str.len()

    # ---------------------------------------------------------------- 1
    counts = df["label"].value_counts().sort_index()
    ratio = counts.max() / counts.min()
    log.info("Distribution : %s | ratio max/min = %.2f", dict(counts), ratio)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([LABELS[i] for i in counts.index], counts.values,
           color=[COLORS[i] for i in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}".replace(",", " "), ha="center", va="bottom")
    ax.set_title("Distribution des classes de sentiment")
    ax.set_ylabel("Nombre de commentaires")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distribution_classes.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- 2
    fig, ax = plt.subplots(figsize=(7, 4))
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]["n_words"]
        ax.hist(sub[sub < 120], bins=40, alpha=0.55,
                label=LABELS[label], color=COLORS[label])
    ax.set_title("Longueur des commentaires (mots) par sentiment")
    ax.set_xlabel("Nombre de mots")
    ax.set_ylabel("Fréquence")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "longueur_textes.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- 3
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    top_words = {}
    for ax, label in zip(axes, sorted(df["label"].unique())):
        words = Counter()
        for txt in df[df["label"] == label]["text"].astype(str):
            words.update(w for w in txt.split()
                         if w not in STOPWORDS and len(w) > 2)
        common = words.most_common(15)[::-1]
        top_words[LABELS[label]] = dict(words.most_common(15))
        ax.barh([w for w, _ in common], [c for _, c in common],
                color=COLORS[label])
        ax.set_title(f"Top mots — {LABELS[label]}")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top_mots.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_comments": int(len(df)),
        "distribution": {LABELS[int(k)]: int(v) for k, v in counts.items()},
        "imbalance_ratio": round(float(ratio), 3),
        "longueur_mots": {
            "moyenne": round(float(df["n_words"].mean()), 1),
            "mediane": int(df["n_words"].median()),
            "p95": int(df["n_words"].quantile(0.95)),
            "max": int(df["n_words"].max()),
        },
        "top_mots": top_words,
    }
    (REPORTS_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("3 figures + eda_summary.json générés dans reports/.")
    return summary


if __name__ == "__main__":
    run()
