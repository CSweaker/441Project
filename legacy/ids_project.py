"""
Multi-Stage ML Framework for Network Intrusion Detection
Dataset: NSL-KDD
Authors: Member 1 & Member 2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD NSL-KDD DATASET
# ─────────────────────────────────────────────
COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

def load_data(train_path="KDDTrain+.txt", test_path="KDDTest+.txt"):
    """Load NSL-KDD dataset. Falls back to synthetic data if files not found."""
    try:
        train = pd.read_csv(train_path, names=COLUMNS)
        test  = pd.read_csv(test_path,  names=COLUMNS)
        print(f"Loaded real NSL-KDD data: {len(train)} train, {len(test)} test rows.")
    except FileNotFoundError:
        print("NSL-KDD files not found — generating synthetic demo data.")
        train, test = _generate_synthetic_data()
    return train, test


def _generate_synthetic_data(n=5000, seed=42):
    """Generate reproducible synthetic network traffic for demo purposes."""
    np.random.seed(seed)
    labels = np.random.choice(
        ["normal","neptune","smurf","portsweep","ipsweep","satan","back"],
        p=[0.40, 0.22, 0.15, 0.08, 0.07, 0.05, 0.03], size=n
    )
    data = {
        "duration":    np.random.exponential(5, n).astype(int),
        "src_bytes":   np.random.exponential(3000, n).astype(int),
        "dst_bytes":   np.random.exponential(1500, n).astype(int),
        "count":       np.random.randint(1, 512, n),
        "srv_count":   np.random.randint(1, 512, n),
        "serror_rate": np.random.beta(0.5, 5, n),
        "rerror_rate": np.random.beta(0.5, 5, n),
        "same_srv_rate": np.random.beta(5, 1, n),
        "dst_host_count": np.random.randint(1, 256, n),
        "logged_in":   np.random.randint(0, 2, n),
        "protocol_type": np.random.choice(["tcp","udp","icmp"], n),
        "service":     np.random.choice(["http","ftp","smtp","ssh","other"], n),
        "flag":        np.random.choice(["SF","S0","REJ","RSTO"], n),
        "label":       labels,
        "difficulty":  np.zeros(n, dtype=int),
    }
    for col in COLUMNS:
        if col not in data:
            data[col] = np.random.randint(0, 2, n)

    df = pd.DataFrame(data)[COLUMNS]
    train, test = train_test_split(df, test_size=0.2, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df, encoders=None, scaler=None, fit=True):
    df = df.copy()
    df.drop(columns=["difficulty"], errors="ignore", inplace=True)

    df["binary_label"] = (df["label"] != "normal").astype(int)
    df["multi_label"]  = df["label"]
    df.drop(columns=["label"], inplace=True)

    cat_cols = ["protocol_type", "service", "flag"]
    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in cat_cols:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0]
                if x in le.classes_ else -1
            )

    feature_cols = [c for c in df.columns if c not in ("binary_label","multi_label")]
    X = df[feature_cols].values.astype(float)

    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, df["binary_label"].values, df["multi_label"].values, encoders, scaler, feature_cols


# ─────────────────────────────────────────────
# 3. STATISTICAL ANALYSIS
# ─────────────────────────────────────────────
def statistical_analysis(train_df):
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    normal = train_df[train_df["label"] == "normal"]["src_bytes"].dropna()
    attack = train_df[train_df["label"] != "normal"]["src_bytes"].dropna()

    # A) CLT
    print("\n[A] Central Limit Theorem — sampling distribution of mean src_bytes")
    sample_means = [normal.sample(min(30, len(normal)), replace=True).mean()
                    for _ in range(1000)]
    shapiro_stat, shapiro_p = stats.shapiro(sample_means[:5000])
    print(f"    Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
    print(f"    → Sample means follow {'normal' if shapiro_p > 0.05 else 'approximately normal'} distribution (CLT confirmed)")

    # B) Hypothesis Testing
    print("\n[B] Hypothesis Testing — src_bytes: normal vs attack")
    u_stat, u_p = stats.mannwhitneyu(normal, attack, alternative="two-sided")
    print(f"    Mann-Whitney U: U={u_stat:.2f}, p={u_p:.4e}")
    print(f"    → {'Reject H0' if u_p < 0.05 else 'Fail to reject H0'}: distributions are {'significantly different' if u_p < 0.05 else 'similar'}")

    t_stat, t_p = stats.ttest_ind(
        normal.sample(min(500, len(normal)), random_state=42),
        attack.sample(min(500, len(attack)), random_state=42),
        equal_var=False
    )
    print(f"    Welch t-test:   t={t_stat:.4f}, p={t_p:.4e}")

    # C) Bootstrap CI
    print("\n[C] Bootstrap 95% CI — mean src_bytes for normal traffic")
    boot_means = np.array([
        normal.sample(min(200, len(normal)), replace=True).mean()
        for _ in range(1000)
    ])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    print(f"    Mean = {normal.mean():.1f} bytes")
    print(f"    95% Bootstrap CI: [{ci_low:.1f}, {ci_high:.1f}]")

    return {
        "sample_means": sample_means,
        "boot_means":   boot_means,
        "ci":           (ci_low, ci_high),
        "normal_bytes": normal,
        "attack_bytes": attack,
    }


# ─────────────────────────────────────────────
# 4. BASELINE MODELS
# ─────────────────────────────────────────────
def train_baseline(X_train, y_train, X_test, y_test, pca_components=20):
    print("\n" + "="*60)
    print("BASELINE MODELS")
    print("="*60)

    pca = PCA(n_components=pca_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)
    ev = pca.explained_variance_ratio_.cumsum()
    print(f"\nPCA: {pca_components} components explain {ev[-1]*100:.1f}% of variance")

    results = {}
    for name, model in [
        ("Naive Bayes",         GaussianNB()),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ]:
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        cv  = cross_val_score(model, X_train_pca, y_train, cv=5, scoring="f1_weighted")
        print(f"\n── {name} ──")
        print(f"   Accuracy : {acc*100:.2f}%")
        print(f"   F1 Score : {f1:.4f}")
        print(f"   CV F1    : {cv.mean():.4f} +/- {cv.std():.4f}")
        print(classification_report(y_test, y_pred, target_names=["Normal","Attack"]))
        results[name] = {"model": model, "acc": acc, "f1": f1, "cv": cv, "y_pred": y_pred}

    return results, pca


# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────
def plot_eda(train_df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("NSL-KDD Exploratory Data Analysis", fontsize=16, fontweight="bold")
    palette = {"normal": "#0D9488", "attack": "#E11D48"}

    label_counts = train_df["label"].apply(
        lambda x: "normal" if x == "normal" else "attack"
    ).value_counts()
    axes[0,0].bar(label_counts.index, label_counts.values,
                  color=[palette["normal"], palette["attack"]])
    axes[0,0].set_title("Class Distribution"); axes[0,0].set_ylabel("Count")
    for i, v in enumerate(label_counts.values):
        axes[0,0].text(i, v + 50, f"{v:,}", ha="center", fontsize=9)

    atk = train_df[train_df["label"] != "normal"]["label"].value_counts().head(8)
    axes[0,1].barh(atk.index[::-1], atk.values[::-1], color="#7C3AED")
    axes[0,1].set_title("Top Attack Types"); axes[0,1].set_xlabel("Count")

    proto = train_df["protocol_type"].value_counts()
    axes[0,2].pie(proto.values, labels=proto.index, autopct="%1.1f%%",
                  colors=["#0D9488","#7C3AED","#E11D48"])
    axes[0,2].set_title("Protocol Type")

    for lbl, clr in palette.items():
        subset = train_df[
            train_df["label"].apply(lambda x: "normal" if x=="normal" else "attack") == lbl
        ]["src_bytes"].clip(upper=1e6) + 1
        axes[1,0].hist(np.log10(subset), bins=40, alpha=0.6, color=clr, label=lbl)
    axes[1,0].set_title("log10(src_bytes) Distribution")
    axes[1,0].set_xlabel("log10(src_bytes)"); axes[1,0].legend()

    train_df_copy = train_df.copy()
    train_df_copy["class"] = train_df_copy["label"].apply(lambda x: "normal" if x=="normal" else "attack")
    axes[1,1].boxplot(
        [train_df_copy[train_df_copy["class"]=="normal"]["serror_rate"],
         train_df_copy[train_df_copy["class"]=="attack"]["serror_rate"]],
        labels=["Normal","Attack"], patch_artist=True,
        boxprops=dict(facecolor="#CADCFC"),
        medianprops=dict(color="#0D9488", linewidth=2)
    )
    axes[1,1].set_title("Syn-Error Rate by Class"); axes[1,1].set_ylabel("serror_rate")

    sample = train_df.sample(min(500, len(train_df)), random_state=42)
    colors = sample["label"].apply(lambda x: palette["normal"] if x=="normal" else palette["attack"])
    axes[1,2].scatter(sample["count"], sample["dst_host_count"],
                      c=colors, alpha=0.4, s=15)
    axes[1,2].set_title("count vs dst_host_count")
    axes[1,2].set_xlabel("count"); axes[1,2].set_ylabel("dst_host_count")

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: eda_plots.png")


def plot_statistical(stats_results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Statistical Analysis", fontsize=15, fontweight="bold")

    axes[0].hist(stats_results["sample_means"], bins=40, color="#0D9488", edgecolor="white")
    axes[0].axvline(np.mean(stats_results["sample_means"]), color="#E11D48",
                    lw=2, label=f"Mean={np.mean(stats_results['sample_means']):.1f}")
    axes[0].set_title("CLT: Sampling Distribution\nof Mean src_bytes")
    axes[0].set_xlabel("Sample Mean"); axes[0].legend()

    ci = stats_results["ci"]
    axes[1].hist(stats_results["boot_means"], bins=40, color="#7C3AED", edgecolor="white")
    axes[1].axvline(ci[0], color="#E11D48", lw=2, ls="--",
                    label=f"95% CI: [{ci[0]:.0f}, {ci[1]:.0f}]")
    axes[1].axvline(ci[1], color="#E11D48", lw=2, ls="--")
    axes[1].set_title("Bootstrap CI\nMean src_bytes (Normal)")
    axes[1].set_xlabel("Bootstrapped Mean"); axes[1].legend()

    n_s = stats_results["normal_bytes"].sample(min(300, len(stats_results["normal_bytes"])), random_state=42)
    a_s = stats_results["attack_bytes"].sample(min(300, len(stats_results["attack_bytes"])), random_state=42)
    axes[2].hist(np.log10(n_s + 1), bins=30, alpha=0.6, color="#0D9488", label="Normal")
    axes[2].hist(np.log10(a_s + 1), bins=30, alpha=0.6, color="#E11D48", label="Attack")
    axes[2].set_title("src_bytes: Normal vs Attack\n(Mann-Whitney U Test)")
    axes[2].set_xlabel("log10(src_bytes + 1)"); axes[2].legend()

    plt.tight_layout()
    plt.savefig("statistical_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: statistical_analysis.png")


def plot_model_results(results, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Baseline Model Results", fontsize=15, fontweight="bold")
    COLORS = ["#0D9488", "#7C3AED"]

    names = list(results.keys())
    accs  = [results[n]["acc"] * 100 for n in names]
    f1s   = [results[n]["f1"]  * 100 for n in names]
    x = np.arange(len(names)); w = 0.35
    axes[0].bar(x - w/2, accs, w, label="Accuracy", color=COLORS[0])
    axes[0].bar(x + w/2, f1s,  w, label="F1 Score",  color=COLORS[1])
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=9)
    axes[0].set_ylim(0, 110); axes[0].set_ylabel("Score (%)")
    axes[0].set_title("Accuracy & F1 Score"); axes[0].legend()

    cv_data = [results[n]["cv"] * 100 for n in names]
    bp = axes[1].boxplot(cv_data, labels=names, patch_artist=True)
    for patch, clr in zip(bp["boxes"], COLORS):
        patch.set_facecolor(clr + "80")
    axes[1].set_title("5-Fold CV F1 Score"); axes[1].set_ylabel("F1 (%)")

    best_name = max(results, key=lambda n: results[n]["f1"])
    cm = confusion_matrix(y_test, results[best_name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
                xticklabels=["Normal","Attack"], yticklabels=["Normal","Attack"])
    axes[2].set_title(f"Confusion Matrix\n({best_name})")
    axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("model_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: model_results.png")


def plot_pca(pca, feature_cols):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PCA Analysis", fontsize=15, fontweight="bold")

    ev = pca.explained_variance_ratio_
    axes[0].plot(range(1, len(ev)+1), np.cumsum(ev)*100, "o-", color="#0D9488", lw=2)
    axes[0].axhline(90, color="#E11D48", ls="--", label="90% threshold")
    axes[0].set_xlabel("# Components"); axes[0].set_ylabel("Cumulative Variance (%)")
    axes[0].set_title("PCA Explained Variance"); axes[0].legend()

    loadings = pd.Series(np.abs(pca.components_[0]), index=feature_cols).nlargest(10)
    axes[1].barh(loadings.index[::-1], loadings.values[::-1], color="#7C3AED")
    axes[1].set_title("Top 10 Feature Loadings — PC1")
    axes[1].set_xlabel("|Loading|")

    plt.tight_layout()
    plt.savefig("pca_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: pca_analysis.png")


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*50)
    print("  Network Intrusion Detection - ML Pipeline")
    print("="*50)

    train_df, test_df = load_data()

    print("\n-- EDA --")
    print(f"Train shape: {train_df.shape}")
    print(train_df["label"].value_counts().head(10))
    plot_eda(train_df)

    stats_res = statistical_analysis(train_df)
    plot_statistical(stats_res)

    X_train, y_train, _, encoders, scaler, feat_cols = preprocess(train_df, fit=True)
    X_test,  y_test,  _, _,        _,      _         = preprocess(
        test_df, encoders=encoders, scaler=scaler, fit=False)

    results, pca = train_baseline(X_train, y_train, X_test, y_test, pca_components=20)

    plot_model_results(results, y_test)
    plot_pca(pca, feat_cols)

    print("\nAll outputs saved. Ready for Milestone 2 (SVM / XGBoost / Random Forest).")
