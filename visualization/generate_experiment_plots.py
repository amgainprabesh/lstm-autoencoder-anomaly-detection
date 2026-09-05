"""
Generate publication-quality diagnostic plots for Experiments 1 through 4.
Saves PNG charts into results/plots/ for embedding in the project report.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 13

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def plot_experiment_1():
    df = pd.read_csv(RESULTS_DIR / "experiment_1_window_size.csv")
    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=300)

    x = np.arange(len(df))
    width = 0.22

    rects1 = ax1.bar(x - width, df["precision"], width, label="Precision", color="#2563eb", alpha=0.85)
    rects2 = ax1.bar(x, df["recall"], width, label="Recall", color="#10b981", alpha=0.85)
    rects3 = ax1.bar(x + width, df["f1_score"], width, label="F1-Score", color="#f59e0b", alpha=0.85)

    ax1.set_xlabel("Sliding Window Size (Steps / Time Span)")
    ax1.set_ylabel("Metric Score (0.0 – 1.0)")
    ax1.set_title("Experiment 1: Impact of Sliding Window Size on Detection Performance", pad=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"w={int(w)}\n({int(w)//2} Hours)" for w in df["window_size"]])
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.6)

    for r in rects3:
        h = r.get_height()
        ax1.annotate(f"{h:.3f}",
                     xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "exp1_window_size.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved {out_path}")


def plot_experiment_2():
    df = pd.read_csv(RESULTS_DIR / "experiment_2_hidden_size.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    arch_labels = ["Small\n[32, 16]\n(8-dim latent)", "Medium (Ours)\n[64, 32]\n(16-dim latent)", "Large\n[128, 64]\n(32-dim latent)"]
    x = np.arange(len(df))

    color = "#dc2626"
    ax1.plot(x, df["val_loss"], marker="o", linewidth=2.5, markersize=8, color=color, label="Validation MSE Loss")
    ax1.set_ylabel("Validation Loss (MSE)", color=color, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(arch_labels)
    ax1.set_title("Validation Reconstruction Loss", pad=10, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    for i, txt in enumerate(df["val_loss"]):
        ax1.annotate(f"{txt:.6f}", (x[i], txt), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")

    width = 0.35
    ax2.bar(x - width/2, df["f1_score"], width, label="F1-Score", color="#2563eb", alpha=0.85)
    ax2.bar(x + width/2, df["precision"], width, label="Precision", color="#10b981", alpha=0.85)
    ax2.set_ylabel("Metric Score", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(arch_labels)
    ax2.set_ylim(0, 0.9)
    ax2.set_title("Out-of-Sample Detection Metrics", pad=10, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)
    for i, txt in enumerate(df["f1_score"]):
        ax2.annotate(f"F1={txt:.3f}", (x[i] - width/2, txt), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5, fontweight="bold")

    fig.suptitle("Experiment 2: Hidden Dimension Depth & Model Capacity vs Performance", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "exp2_model_capacity.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved {out_path}")


def plot_experiment_3():
    df = pd.read_csv(RESULTS_DIR / "experiment_3_threshold.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    ax1.plot(df["recall"], df["precision"], marker="s", linewidth=2.5, markersize=8, color="#7c3aed")
    for i, row in df.iterrows():
        ax1.annotate(f"p={row['percentile']}%\n(τ={row['threshold_value']:.4f})",
                     (row["recall"], row["precision"]),
                     textcoords="offset points", xytext=(10, -5), fontsize=8.5, fontweight="bold")
    ax1.set_xlabel("Recall", fontweight="bold")
    ax1.set_ylabel("Precision", fontweight="bold")
    ax1.set_title("Precision-Recall Trade-off Curve", pad=10, fontweight="bold")
    ax1.set_xlim(0.08, 0.30)
    ax1.set_ylim(0.55, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.6)

    x = np.arange(len(df))
    ax2.plot(x, df["precision"], marker="o", label="Precision", color="#2563eb", linewidth=2)
    ax2.plot(x, df["recall"], marker="^", label="Recall", color="#10b981", linewidth=2)
    ax2.plot(x, df["f1_score"], marker="d", label="F1-Score", color="#f59e0b", linewidth=2)
    ax2.plot(x, df["accuracy"], marker="x", label="Accuracy", color="#64748b", linewidth=2, linestyle=":")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{p}%" for p in df["percentile"]])
    ax2.set_xlabel("Validation Error Threshold Percentile", fontweight="bold")
    ax2.set_ylabel("Score", fontweight="bold")
    ax2.set_title("Metrics across Threshold Percentiles", pad=10, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Experiment 3: Threshold Percentile Sensitivity & Trade-Offs", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "exp3_threshold_sensitivity.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved {out_path}")


def plot_experiment_4():
    df = pd.read_csv(RESULTS_DIR / "experiment_4_baseline.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

    models = ["Statistical Filter\n(Rolling Z-Score)", "Dense Autoencoder\n(Non-Recurrent MLP)", "LSTM Autoencoder\n(Recurrent Ours)"]
    x = np.arange(len(models))
    width = 0.20

    r1 = ax.bar(x - 1.5*width, df["precision"], width, label="Precision", color="#2563eb", alpha=0.9)
    r2 = ax.bar(x - 0.5*width, df["recall"], width, label="Recall", color="#10b981", alpha=0.9)
    r3 = ax.bar(x + 0.5*width, df["f1_score"], width, label="F1-Score", color="#f59e0b", alpha=0.9)
    r4 = ax.bar(x + 1.5*width, df["accuracy"], width, label="Accuracy", color="#6366f1", alpha=0.9)

    ax.set_ylabel("Performance Metric Score (0.0 – 1.0)", fontweight="bold")
    ax.set_title("Experiment 4: Comparative Evaluation against Baseline Architectures", pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", ncol=4)
    ax.grid(True, linestyle="--", alpha=0.6)

    for r in [r1[-1], r2[-1], r3[-1], r4[-1]]:
        h = r.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1e3a8a")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "exp4_baseline_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved {out_path}")


def main():
    print("[*] Generating publication plots for Experiments 1-4...")
    plot_experiment_1()
    plot_experiment_2()
    plot_experiment_3()
    plot_experiment_4()
    print("[+] All experiment plots successfully generated in results/plots/!")


if __name__ == "__main__":
    main()
