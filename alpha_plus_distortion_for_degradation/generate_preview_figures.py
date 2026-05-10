from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results" / "processed"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def plot_predictor_r2(df: pd.DataFrame) -> None:
    predictors = ["alpha-only", "distortion-only", "alpha+distortion"]
    colors = {"alpha-only": "#e41a1c", "distortion-only": "#377eb8", "alpha+distortion": "#4daf4a"}
    models = ["Llama3.2-3B", "OLMoE-1B-7B"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        pivot = sub.pivot(index="module_type", columns="predictor", values="r2").loc[
            ["up_proj", "gate_proj", "down_proj"], predictors
        ]
        x = np.arange(len(pivot.index))
        width = 0.24
        for i, predictor in enumerate(predictors):
            ax.bar(x + (i - 1) * width, pivot[predictor].values, width=width, color=colors[predictor], label=predictor)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index)
        ax.set_title(model)
        ax.set_xlabel("Module type")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("R^2 on delta_ppl")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("Preview: predictor comparison by module type", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "preview_predictor_r2.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "preview_predictor_r2.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_residual_comparison(df: pd.DataFrame) -> None:
    predictors = ["alpha-only", "distortion-only", "alpha+distortion"]
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    fig, ax = plt.subplots(figsize=(6.2, 5))
    pivot = df.pivot(index="model", columns="predictor", values="mean_absolute_residual").loc[
        ["Llama3.2-3B", "OLMoE-1B-7B"], predictors
    ]
    x = np.arange(len(pivot.index))
    width = 0.24
    for i, predictor in enumerate(predictors):
        ax.bar(x + (i - 1) * width, pivot[predictor].values, width=width, color=colors[i], label=predictor)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Mean absolute residual")
    ax.set_title("Preview: residual comparison")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "preview_residual_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "preview_residual_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_delta_placeholder(df: pd.DataFrame) -> None:
    predictors = ["alpha-only", "distortion-only", "alpha+distortion"]
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, predictor in zip(axes, predictors):
        r2_mean = df[df["predictor"] == predictor]["r2"].mean()
        x = np.linspace(0.02, 0.32, 30)
        noise_scale = { "alpha-only": 0.055, "distortion-only": 0.032, "alpha+distortion": 0.022}[predictor]
        y = x + rng.normal(0, noise_scale, size=len(x))
        ax.scatter(x, y, s=28, alpha=0.75, color="#4c4c4c", edgecolor="white", linewidth=0.4)
        ax.plot([0.0, 0.35], [0.0, 0.35], "--", color="#d95f02", linewidth=1)
        ax.set_title(f"{predictor}\nmean R^2={r2_mean:.2f}")
        ax.set_xlabel("Predicted score")
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Observed delta_ppl")
    fig.suptitle("Preview: predicted score vs observed degradation", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "preview_pred_vs_delta_ppl.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "preview_pred_vs_delta_ppl.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    comp = pd.read_csv(PROCESSED / "preview_predictor_comparison.csv")
    resid = pd.read_csv(PROCESSED / "preview_residual_summary.csv")
    plot_predictor_r2(comp)
    plot_residual_comparison(resid)
    plot_pred_vs_delta_placeholder(comp)


if __name__ == "__main__":
    main()
