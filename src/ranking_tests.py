from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
import warnings

# Paths
ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "processed" / "survey_rankings.csv"
ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

# --- 1. Helper Function ---


def holm_bonferroni_adjust(pvalues):
    """Return Holm–Bonferroni adjusted p-values in the original order."""
    m = len(pvalues)
    # Handle NaNs by treating them as 1.0 for ordering; we'll restore NaN later
    pvalues_list = [p if not (p is None or np.isnan(p)) else 1.0 for p in pvalues]
    indexed = sorted(enumerate(pvalues_list), key=lambda x: x[1])

    adjusted = [0.0] * m
    last_adj = 0.0  # enforce monotonicity (non-decreasing adjusted p-values)
    for i, (orig_idx, p) in enumerate(indexed):
        adj = min((m - i) * p, 1.0)
        adj = max(adj, last_adj)
        adjusted[orig_idx] = adj
        last_adj = adj

    # Restore NaNs where original p was NaN/None
    for i, p in enumerate(pvalues):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            adjusted[i] = np.nan
    return adjusted


# --- 2. Main Analysis Function ---


def run_tests_and_save():
    # Load data
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find data file at {DATA_FILE}")
        return

    rank_cols = [
        "rank_google/gemini-2.5-flash-lite",
        "rank_meta-llama/llama-4-maverick",
        "rank_openai/gpt-4o-mini",
        "rank_qwen/qwen3-vl-8b-instruct",
    ]

    # Clean model names for plots (display)
    model_names_clean = [
        c.replace("rank_", "")
        .replace("google/gemini-2.5-flash-lite", "Google")
        .replace("meta-llama/llama-4-maverick", "Meta")
        .replace("openai/gpt-4o-mini", "OpenAI")
        .replace("qwen/qwen3-vl-8b-instruct", "Qwen")
        for c in rank_cols
    ]

    # --- Statistical Tests (PER TASK): aggregate across raters within each objectid ---

    # One consensus vector of ranks per task (median across raters)
    df_task = df.groupby("objectid")[rank_cols].median().reset_index()

    # Friedman test across tasks
    try:
        friedman_stat, friedman_p = friedmanchisquare(*[df_task[c] for c in rank_cols])
    except ValueError as e:
        print(f"Error during Friedman test: {e}")
        friedman_stat, friedman_p = np.nan, np.nan

    # Pairwise Wilcoxon across tasks (each task contributes one paired difference)
    pairs = list(combinations(rank_cols, 2))
    pair_results = []
    pvals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a, b in pairs:
            try:
                stat, p = wilcoxon(
                    df_task[a], df_task[b], zero_method="zsplit", correction=True
                )
            except ValueError:
                stat, p = np.nan, np.nan
            pair_results.append(
                {
                    "model_a": a.replace("rank_", ""),
                    "model_b": b.replace("rank_", ""),
                    "statistic": stat,
                    "pvalue": p,
                }
            )
            pvals.append(p)

    # Holm–Bonferroni adjustment
    adjusted = holm_bonferroni_adjust(pvals)
    for i, adj in enumerate(adjusted):
        pair_results[i]["p_adjusted_holm"] = adj

    # Kendall's W across tasks (agreement of model ranks over tasks)
    ranks = df_task[rank_cols].to_numpy()
    n, k = ranks.shape  # n = number of tasks, k = number of models
    R = np.sum(ranks, axis=0)  # sum of ranks per model across tasks
    R_bar = np.mean(R)
    S = np.sum((R - R_bar) ** 2)
    kendall_W = 12 * S / (k**2 * (n**3 - n)) if n > 1 else np.nan

    # Also compute from Friedman statistic (when available)
    kendall_W_from_friedman = (
        (friedman_stat / (n * (k - 1)))
        if (n > 0 and not np.isnan(friedman_stat))
        else np.nan
    )

    # --- Descriptive Statistics (on full dataset; label clearly as "All Ratings") ---

    # Long form for counts
    df_long = df.melt(
        id_vars=["objectid", "submission_seed"],
        value_vars=rank_cols,
        var_name="model",
        value_name="rank",
    )
    df_long["model"] = df_long["model"].str.replace("rank_", "", regex=False)

    # Rank counts per model
    rank_counts = (
        df_long.groupby(["model", "rank"]).size().unstack(fill_value=0).reset_index()
    )
    rank_counts = rank_counts.rename(
        columns={
            1: "count_rank_1",
            2: "count_rank_2",
            3: "count_rank_3",
            4: "count_rank_4",
        }
    )

    # Rank counts per object
    obj_counts = (
        df_long.groupby(["objectid", "model", "rank"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    obj_counts = obj_counts.rename(
        columns={
            1: "count_rank_1",
            2: "count_rank_2",
            3: "count_rank_3",
            4: "count_rank_4",
        }
    )

    # --- Save CSV outputs ---

    summary = {
        "friedman_stat": [friedman_stat],
        "friedman_p": [friedman_p],
        "kendall_W": [kendall_W],
        "kendall_W_from_friedman": [kendall_W_from_friedman],
        "n_tasks": [df_task.shape[0]],
        "n_unique_raters": [df["submission_seed"].nunique()],
        "n_submissions": [df.shape[0]],
    }
    pd.DataFrame(summary).to_csv(
        ANALYSIS_DIR / "ranking_tests_summary.csv", index=False
    )
    pd.DataFrame(pair_results).to_csv(ANALYSIS_DIR / "pairwise_tests.csv", index=False)
    rank_counts.to_csv(ANALYSIS_DIR / "rank_counts_per_model.csv", index=False)
    obj_counts.to_csv(ANALYSIS_DIR / "rank_counts_per_object.csv", index=False)
    print("Saved CSV results.")

    # --- Plots ---

    # Boxplot of task-level rank distributions per model
    plt.figure(figsize=(8, 6))
    ax = df_task[rank_cols].boxplot()
    ax.set_xticklabels(model_names_clean, rotation=45, ha="right")
    plt.title("Rank Distributions per Model (Task-Level Medians; Lower = Better)")
    plt.ylabel("Rank")
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "rank_distributions_boxplot.png", dpi=200)
    plt.close()
    print("Saved boxplot (task-level medians).")

    # Bar chart: counts of rank1..rank4 per model (All Ratings)
    rank_counts_plot = rank_counts.set_index("model")
    # Map index to friendly names for plotting
    friendly_index = [
        idx.replace("google/gemini-2.5-flash-lite", "Google")
        .replace("meta-llama/llama-4-maverick", "Meta")
        .replace("openai/gpt-4o-mini", "OpenAI")
        .replace("qwen/qwen3-vl-8b-instruct", "Qwen")
        for idx in rank_counts_plot.index
    ]
    rank_counts_plot.index = friendly_index
    # Ensure order matches model_names_clean
    rank_counts_plot = rank_counts_plot.loc[model_names_clean]
    rank_counts_plot[
        ["count_rank_1", "count_rank_2", "count_rank_3", "count_rank_4"]
    ].plot(kind="bar", stacked=False, figsize=(10, 6))
    plt.title("Counts of Ranks per Model (All Ratings)")
    plt.ylabel("Count")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Rank")
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "rank_counts_per_model.png", dpi=200)
    plt.close()
    print("Saved rank counts bar chart (all ratings).")

    # Heatmap of pairwise adjusted p-values (Holm), ordered as rank_cols
    clean_order = [c.replace("rank_", "") for c in rank_cols]
    name_to_idx = {name: i for i, name in enumerate(clean_order)}
    pmat = np.ones((len(clean_order), len(clean_order))) * np.nan
    for r in pair_results:
        a = name_to_idx[r["model_a"]]
        b = name_to_idx[r["model_b"]]
        p = r.get("p_adjusted_holm", np.nan)
        pmat[a, b] = p
        pmat[b, a] = p
    np.fill_diagonal(pmat, 0.0)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(pmat, cmap="viridis_r", vmin=0, vmax=1)
    plt.colorbar(im, label="Adjusted p-value (Holm)")
    # Use friendly display names matching rank_cols order
    plt.xticks(range(len(clean_order)), model_names_clean, rotation=45, ha="right")
    plt.yticks(range(len(clean_order)), model_names_clean)

    for i in range(len(clean_order)):
        for j in range(len(clean_order)):
            if i == j or np.isnan(pmat[i, j]):
                continue
            p_val = pmat[i, j]
            txt = f"{p_val:.3f}"
            color = "white" if p_val < 0.5 else "black"
            plt.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)

    plt.title("Pairwise Adjusted p-values (Holm) — Task-Level Inference")
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "pairwise_pvalues_heatmap.png", dpi=200)
    plt.close()
    print("Saved p-value heatmap.")

    print(f"\nSuccessfully saved summary, CSVs, and plots to: {ANALYSIS_DIR.resolve()}")


if __name__ == "__main__":
    run_tests_and_save()
