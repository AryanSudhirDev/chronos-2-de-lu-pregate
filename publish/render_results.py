"""Render recorded summaries only; never fits, forecasts or scores raw targets."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    holdout = json.loads((ROOT / "runs/holdout_summary.json").read_text())
    rows = {m["name"]: m for m in holdout["models"]}
    paper = json.loads((ROOT / "runs/paper/summary.json").read_text())["models"]
    readiness = json.loads((ROOT / "runs/readiness/summary.json").read_text())
    dev = {m["name"]: m["mae"] for m in readiness["models"]}
    panels = [
        ("Original holdout · 372 common days", ["Blend (post-gate)", "LoRA (post-gate)",
          "Zero-shot (post-gate)", "LEAR (post-gate)", "Published LoRA (pre-gate)"],
         [rows[k]["mae"] for k in ["blend_pool3", "chronos2_ft_lora", "chronos2_zs", "lear", "chronos2_ft_lora_pregate"]]),
        ("Later dev-only study · 123 days", ["LoRA (pre-gate)", "LightGBM (pre-gate)", "LEAR (pre-gate)"],
         [dev[k] for k in ["pregate_lora", "pregate_lgbm", "pregate_lear"]]),
        ("Prospective paper · only 5 scored days", ["LEAR (pre-gate)", "LoRA (pre-gate)", "LightGBM (pre-gate)"],
         [paper[k]["mae"] for k in ["pregate_lear", "pregate_lora", "pregate_lgbm"]]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    for ax, (title, names, values) in zip(axes, panels):
        colors = ["#df8346" if "LoRA" in n and "pre-gate" in n else "#426a8a" for n in names]
        bars = ax.barh(names, values, color=colors)
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.2f", padding=4)
        ax.set_xlim(0, max(values) * 1.23)
        ax.set_xlabel("MAE (EUR/MWh); lower is better")
        ax.set_title(title, fontsize=11, pad=18)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Chronos-2 electricity forecasting: different windows and information sets", fontsize=15)
    out = ROOT / "docs/results.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
