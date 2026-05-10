from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results" / "processed"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def plot_model_metric_curves(df: pd.DataFrame) -> None:
    colors = {"alpha": "#1b9e77", "quant_mse": "#d95f02", "alpha_x_mse": "#7570b3"}
    models = ["llama3_2_3b", "qwen3_8b"]
    titles = {"llama3_2_3b": "Llama3.2-3B", "qwen3_8b": "Qwen3-8B"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, model in zip(axes, models):
        sub = df[df["model_id"] == model]
        for metric, g in sub.groupby("group_metric"):
            ax.plot(g["group_id"], g["abs_drop"].abs(), marker="o", linewidth=2, label=metric, color=colors[metric])
        ax.set_title(titles[model])
        ax.set_xlabel("Group ID")
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Absolute MMLU drop")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Grouped MMLU degradation by metric and group", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "full_grouped_mmlu_curves.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "full_grouped_mmlu_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby(["model_id", "group_metric"])
        .agg(mean_abs_drop=("abs_drop", lambda s: s.abs().mean()), worst_abs_drop=("abs_drop", lambda s: s.abs().max()))
        .reset_index()
    )
    colors = {"alpha": "#1b9e77", "quant_mse": "#d95f02", "alpha_x_mse": "#7570b3"}
    models = ["llama3_2_3b", "qwen3_8b"]
    labels = {"llama3_2_3b": "Llama3.2-3B", "qwen3_8b": "Qwen3-8B"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=False)
    for ax, model in zip(axes, models):
        sub = summary[summary["model_id"] == model]
        ax.bar(sub["group_metric"], sub["mean_abs_drop"], color=[colors[m] for m in sub["group_metric"]], alpha=0.85)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(i, row["mean_abs_drop"] + 0.004, f"worst={row['worst_abs_drop']:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(labels[model])
        ax.set_ylabel("Mean absolute MMLU drop")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.suptitle("Grouped MMLU sensitivity summary by metric", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "full_metric_summary.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "full_metric_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_reference_protocol(df_ref: pd.DataFrame) -> None:
    dense = df_ref[(df_ref["run_type"] == "group_quant") & (df_ref["eval_task"] == "wikitext")].copy()
    dense["group_id"] = dense["group_id"].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    colors = {"llama3_2_3b": "#1b9e77", "qwen3_8b": "#d95f02"}
    titles = {"llama3_2_3b": "Llama3.2-3B", "qwen3_8b": "Qwen3-8B"}
    for ax, model in zip(axes, ["llama3_2_3b", "qwen3_8b"]):
        sub = dense[dense["model_id"] == model].sort_values("group_id")
        ax.plot(sub["group_id"], sub["abs_drop"], marker="o", linewidth=2, color=colors[model])
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(titles[model])
        ax.set_xlabel("Group ID")
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Absolute PPL change")
    fig.suptitle("Reference V1 grouped results", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "reference_v1_grouped_results.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "reference_v1_grouped_results.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(PROCESSED / "full_grouped_mmlu_results.csv")
    df_ref = pd.read_csv(PROCESSED / "legacy_reference_results.csv")
    plot_model_metric_curves(df)
    plot_metric_summary(df)
    plot_reference_protocol(df_ref)


if __name__ == "__main__":
    main()
