from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results" / "processed"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def plot_dense_group_curves(df: pd.DataFrame) -> None:
    dense = df[(df["task"] == "wikitext") & (df["group_id"].notna())].copy()
    dense["group_id"] = dense["group_id"].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    colors = {"Llama3.2-3B": "#1b9e77", "Qwen3-8B": "#d95f02"}
    for ax, model in zip(axes, ["Llama3.2-3B", "Qwen3-8B"]):
        sub = dense[dense["model"] == model].sort_values("group_id")
        ax.plot(sub["group_id"], sub["abs_drop"], marker="o", color=colors[model], linewidth=2)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(model)
        ax.set_xlabel("Group ID")
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Absolute PPL drop")
    fig.suptitle("Current visible grouped alpha results on WikiText", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "current_visible_dense_group_curves.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "current_visible_dense_group_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_dense_relative_drop(df: pd.DataFrame) -> None:
    dense = df[(df["task"] == "wikitext") & (df["group_id"].notna())].copy()
    dense["group_id"] = dense["group_id"].astype(int)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for model, color in [("Llama3.2-3B", "#1b9e77"), ("Qwen3-8B", "#d95f02")]:
        sub = dense[dense["model"] == model].sort_values("group_id")
        ax.plot(sub["group_id"], 100 * sub["rel_drop"], marker="o", linewidth=2, label=model, color=color)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Group ID")
    ax.set_ylabel("Relative drop (%)")
    ax.set_title("Current visible relative degradation by group")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "current_visible_dense_relative_drop.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "current_visible_dense_relative_drop.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_result_availability(df: pd.DataFrame) -> None:
    counts = (
        df.assign(is_group_run=df["run_type"].eq("single_group_quant"))
        .groupby(["model", "task"])["is_group_run"]
        .sum()
        .reset_index(name="visible_group_runs")
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    labels = [f"{m}\n{t}" for m, t in zip(counts["model"], counts["task"])]
    ax.bar(labels, counts["visible_group_runs"], color="#4c72b0")
    ax.set_ylabel("Visible grouped runs")
    ax.set_title("Current visible result coverage from Notion")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "current_visible_result_coverage.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "current_visible_result_coverage.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(PROCESSED / "current_visible_results.csv")
    plot_dense_group_curves(df)
    plot_dense_relative_drop(df)
    plot_result_availability(df)


if __name__ == "__main__":
    main()
