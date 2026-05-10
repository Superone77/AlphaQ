from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
CSV = BASE / "results" / "processed" / "dual_sensitivity_quantization_noise.csv"
OUT_EXP = BASE / "results" / "figures" / "dual_sensitivity_quantization_noise.pdf"
OUT_TMP = BASE.parents[1] / "tmp" / "dual_sensitivity_map_v1.pdf"


def main() -> None:
    df = pd.read_csv(CSV)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(df["delta_ppl_proxy"].min(), df["delta_ppl_proxy"].max())

    panel_specs = [
        ("Llama3.2-3B", [2.0e-5, 2.5e-5, 3.0e-5], ["2e-5", "2.5e-5", "3e-5"]),
        ("OLMoE-1B-7B", [1.5e-5, 2.0e-5], ["1.5e-5", "2e-5"]),
    ]

    for ax, (model, xticks, xticklabels) in zip(axes, panel_specs):
        sub = df[df["model"] == model].copy()
        ax.scatter(
            sub["quantization_noise"],
            sub["pl_alpha_hill"],
            s=52,
            c=sub["delta_ppl_proxy"],
            cmap=cmap,
            norm=norm,
            alpha=0.9,
            edgecolor="#5b6068",
            linewidth=0.5,
        )
        ax.set_title(model, fontsize=16, fontweight="bold")
        ax.set_xlabel("Quantization Noise", fontsize=16)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(alpha=0.18, linewidth=0.6, linestyle="--")

    axes[0].set_ylabel("PL_Alpha_Hill", fontsize=17)
    fig.tight_layout()
    OUT_EXP.parent.mkdir(parents=True, exist_ok=True)
    OUT_TMP.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_EXP, bbox_inches="tight")
    fig.savefig(OUT_TMP, bbox_inches="tight")


if __name__ == "__main__":
    main()
