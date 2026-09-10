"""Rebuild the README figures from the distributed manifest and paper table."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
COLORS = ["#31456a", "#167d8d", "#d89535"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "#fafbfc",
                         "axes.facecolor": "#fafbfc", "savefig.facecolor": "#fafbfc"})
    manifest = json.loads((ROOT / "datasets/perpdscd/manifest.json").read_text())
    splits = manifest["splits"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    names = ["Train", "Validation", "Test"]
    values = [splits[key]["utterances"] for key in ("train", "validation", "test")]
    bars = axes[0].bar(names, values, color=COLORS, width=0.6)
    axes[0].bar_label(bars, labels=[f"{v:,}" for v in values], padding=5, fontsize=11)
    axes[0].set_ylim(0, max(values) * 1.18)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    axes[0].set_title("Utterances by split", loc="left", pad=16)
    axes[0].set_ylabel("Utterances")
    doctors = sum(s["doctor_responses"] for s in splits.values())
    patients = sum(s["patient_utterances"] for s in splits.values())
    axes[1].pie([patients, doctors], labels=[f"Patient\n{patients:,}", f"Doctor\n{doctors:,}"],
                colors=COLORS[:2], startangle=90, wedgeprops={"width": 0.28, "edgecolor": "#fafbfc"},
                textprops={"fontsize": 11})
    conversations = sum(s["conversations"] for s in splits.values())
    axes[1].text(0, 0, f"{conversations:,}\nconversations", ha="center", va="center", fontsize=14, fontweight="bold")
    axes[1].set_title("Speaker coverage", loc="left", pad=16)
    fig.savefig(OUT / "dataset-overview.png", dpi=180)
    plt.close(fig)

    rows = list(csv.DictReader((ROOT / "docs/paper-results.csv").open()))
    chosen = [r for r in rows if r["model"] in ("PDSS", "ABLE-TR", "ABLE-GR", "ABLE")]
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    palette = ["#b0bac9", "#7599b0", "#439699", "#175b70"]
    for i, row in enumerate(chosen):
        positions = [x + (i - 1.5) * 0.19 for x in range(4)]
        bars = ax.bar(positions, [float(row[k]) for k in ("PCA", "GAA", "PA", "EA")],
                      width=0.18, label=row["model"], color=palette[i])
        ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
    ax.set_xticks(range(4), ["Persona", "Gender–age", "Politeness", "Empathy"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Classifier agreement (%)")
    ax.set_title("Published automatic evaluation · EMNLP 2024, Table 2", loc="left", pad=18)
    ax.legend(ncol=4, loc="upper left", frameon=False)
    fig.savefig(OUT / "paper-results.png", dpi=180)
    plt.close(fig)

    if manifest.get("require_style_labels"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
        for ax, key, names, title in zip(axes, ("politeness_label", "empathy_label"),
                (["Impolite", "Neutral", "Polite"], ["Non-empathetic", "Neutral", "Empathetic"]),
                ("Politeness labels", "Empathy labels")):
            counts = [sum(s["labels"][key].get(str(i), 0) for s in splits.values()) for i in range(3)]
            bars = ax.bar(names, counts, color=COLORS, width=0.6)
            ax.bar_label(bars, labels=[f"{v:,}" for v in counts], padding=4)
            ax.set_ylim(0, max(counts) * 1.18)
            ax.set_title(title, loc="left", pad=16)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
            ax.set_ylabel("Utterances")
        fig.savefig(OUT / "label-distribution.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
