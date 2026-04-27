"""Statistical analysis required by the project rubric."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def run_statistical_analysis(train_df: pd.DataFrame, out_dir: str | Path, random_state: int = 42) -> dict:
    """Run CLT, hypothesis testing, and bootstrap analysis on src_bytes."""
    rng = np.random.default_rng(random_state)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normal = np.log1p(train_df.loc[train_df["label"] == "normal", "src_bytes"].astype(float).clip(lower=0))
    attack = np.log1p(train_df.loc[train_df["label"] != "normal", "src_bytes"].astype(float).clip(lower=0))

    # CLT demonstration: repeated sample means of normal traffic bytes.
    sample_size = min(40, len(normal))
    sample_means = np.array([
        rng.choice(normal.to_numpy(), size=sample_size, replace=True).mean()
        for _ in range(1000)
    ])

    shapiro_stat, shapiro_p = stats.shapiro(sample_means[:5000])
    mw_stat, mw_p = stats.mannwhitneyu(normal, attack, alternative="two-sided")
    t_stat, t_p = stats.ttest_ind(normal, attack, equal_var=False)

    # Bootstrap CI for the difference in mean log(src_bytes): attack - normal.
    boot_diffs = np.array([
        rng.choice(attack.to_numpy(), size=min(1000, len(attack)), replace=True).mean()
        - rng.choice(normal.to_numpy(), size=min(1000, len(normal)), replace=True).mean()
        for _ in range(2000)
    ])
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    summary = {
        "normal_count": int(len(normal)),
        "attack_count": int(len(attack)),
        "mean_log_src_bytes_normal": float(normal.mean()),
        "mean_log_src_bytes_attack": float(attack.mean()),
        "clt_shapiro_w": float(shapiro_stat),
        "clt_shapiro_p": float(shapiro_p),
        "mann_whitney_u": float(mw_stat),
        "mann_whitney_p": float(mw_p),
        "welch_t": float(t_stat),
        "welch_p": float(t_p),
        "bootstrap_mean_diff_attack_minus_normal_low": float(ci_low),
        "bootstrap_mean_diff_attack_minus_normal_high": float(ci_high),
    }

    pd.DataFrame([summary]).to_csv(out_dir / "statistical_summary.csv", index=False)

    fig = plt.figure(figsize=(8, 5))
    plt.hist(sample_means, bins=35)
    plt.title("CLT Demo: Sampling Distribution of Mean log(src_bytes)")
    plt.xlabel("Sample mean")
    plt.ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(out_dir / "clt_sample_means.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.hist(boot_diffs, bins=35)
    plt.axvline(ci_low, linestyle="--")
    plt.axvline(ci_high, linestyle="--")
    plt.title("Bootstrap 95% CI: Difference in Mean log(src_bytes)")
    plt.xlabel("Attack mean - Normal mean")
    plt.ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(out_dir / "bootstrap_ci.png", dpi=160)
    plt.close(fig)

    return summary
